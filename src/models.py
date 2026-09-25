"""Predictors, all behind one fit/predict interface so the harness treats them alike.

Baselines are not padding. Each one is here to absorb a specific alternative explanation
for any performance the supervised models show:

  GlobalMean        the null. Fixes the floor at Spearman 0.
  Blosum62          generic amino-acid exchangeability, untrained. If a trained model does
                    not beat this, it has learned nothing dataset-specific.
  PositionMean      the positional-effect model, fitted on training folds only. Under the
                    random split it can see siblings of every test variant; under the
                    position/region splits it cannot and must fall back. The gap between
                    those two numbers IS the leakage measurement.
  DomainMean        coarse regional tolerance - the resolution at which Spencer & Zhang
                    actually drew conclusions. The bar any residue-level model must clear
                    to claim it has added anything.
  EsmZeroShot       ESM-2 650M log-likelihood ratio, untrained. This is the direct,
                    executed test of the "pretrained protein sequence model" half of the
                    proposal under evaluation.
  Ridge / HGB       supervised models over the feature groups.

CodonOnly is a confound probe rather than a baseline: it uses only features of the
error-prone-PCR accessibility of each substitution and contains no protein biology at all.
Whatever it scores is an upper bound on how much of any other model's performance could be
explained by library-construction artefacts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import blosum62


class Predictor:
    name = "base"
    needs_features = True

    def fit(self, df_tr: pd.DataFrame, X_tr: np.ndarray, y_tr: np.ndarray) -> "Predictor":
        return self

    def predict(self, df_te: pd.DataFrame, X_te: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GlobalMean(Predictor):
    name = "global_mean"
    needs_features = False

    def fit(self, df_tr, X_tr, y_tr):
        self.mu = float(np.mean(y_tr))
        return self

    def predict(self, df_te, X_te):
        return np.full(len(df_te), self.mu)


class Blosum62(Predictor):
    """Untrained: predicted activity = BLOSUM62 score of the substitution."""

    name = "blosum62"
    needs_features = False

    def __init__(self):
        self.b62 = blosum62()

    def predict(self, df_te, X_te):
        return np.array(
            [self.b62[(w, m)] for w, m in zip(df_te["wt_aa"], df_te["mt_aa"])], dtype=float
        )


class GroupMean(Predictor):
    """Mean training score of a grouping column, with an explicit fallback chain.

    The fallback is the whole point for the position-grouped and region-blocked splits:
    a held-out position has no training mean, so the model degrades to the next coarser
    level. Making that degradation explicit is what turns this baseline into a diagnostic.
    """

    needs_features = False

    def __init__(self, key: str, fallback: str | None = None):
        self.key = key
        self.fallback = fallback
        self.name = f"{key}_mean" + (f"_fb_{fallback}" if fallback else "")

    def fit(self, df_tr, X_tr, y_tr):
        s = pd.Series(y_tr, index=df_tr.index)
        self.mu = float(s.mean())
        self.primary = s.groupby(df_tr[self.key]).mean().to_dict()
        self.secondary = (
            s.groupby(df_tr[self.fallback]).mean().to_dict() if self.fallback else {}
        )
        return self

    def predict(self, df_te, X_te):
        out = []
        for _, r in df_te.iterrows():
            v = self.primary.get(r[self.key])
            if v is None and self.fallback:
                v = self.secondary.get(r[self.fallback])
            out.append(self.mu if v is None else v)
        return np.asarray(out, dtype=float)


class ColumnScore(Predictor):
    """Untrained predictor that reads a single precomputed feature column.

    Used for the ESM-2 zero-shot log-likelihood ratio, so the zero-shot number is produced
    by exactly the same evaluation path as every trained model.
    """

    needs_features = True

    def __init__(self, col_index: int, name: str, sign: float = 1.0):
        self.col_index = col_index
        self.name = name
        self.sign = sign

    def predict(self, df_te, X_te):
        return self.sign * X_te[:, self.col_index].astype(float)


class Ridge(Predictor):
    """Standardised ridge with the penalty chosen by an inner CV *grouped by position*.

    Using plain KFold for the inner loop would tune alpha against position-level leakage
    and systematically under-regularise, so the inner loop uses the same grouping as the
    outer split.
    """

    def __init__(self, cols: np.ndarray | None = None, name: str = "ridge", alphas=None):
        self.cols = cols
        self.name = name
        self.alphas = alphas if alphas is not None else np.logspace(-1, 5, 25)

    def fit(self, df_tr, X_tr, y_tr):
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = X_tr if self.cols is None else X_tr[:, self.cols]
        cv = GroupKFold(n_splits=5)
        self.model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=self.alphas, cv=cv.split(X, y_tr, df_tr["pos"]))
        )
        self.model.fit(X, y_tr)
        return self

    def predict(self, df_te, X_te):
        X = X_te if self.cols is None else X_te[:, self.cols]
        return self.model.predict(X)


class HGB(Predictor):
    """Histogram gradient boosting - captures interactions the ridge cannot.

    Relevant here because the substantive extension needs position-level ESM embedding
    components to interact with substitution identity: the wild-type embedding is constant
    across the substitutions at a position, so only an interaction with the mutant-residue
    features can produce variant-specific predictions from it.

    Capacity is deliberately low (depth 3, 300 trees, strong leaf regularisation): with
    ~7300 training rows, a position-level ICC of 0.06 and no variant-level replicates,
    anything larger fits noise. Chosen by inner grouped CV, not by test performance.
    """

    def __init__(self, cols: np.ndarray | None = None, name: str = "hgb", seed: int = 0, **kw):
        self.cols = cols
        self.name = name
        self.seed = seed
        self.kw = dict(
            max_depth=3,
            max_iter=300,
            learning_rate=0.05,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
        )
        self.kw.update(kw)

    def fit(self, df_tr, X_tr, y_tr):
        from sklearn.ensemble import HistGradientBoostingRegressor

        X = X_tr if self.cols is None else X_tr[:, self.cols]
        self.model = HistGradientBoostingRegressor(random_state=self.seed, **self.kw)
        self.model.fit(X, y_tr)
        return self

    def predict(self, df_te, X_te):
        X = X_te if self.cols is None else X_te[:, self.cols]
        return self.model.predict(X)


def run_cv(
    model_factory,
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    shuffle_y_within: str | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold predictions. `model_factory` is called fresh for every fold.

    `shuffle_y_within='global'` permutes the training labels, giving the y-randomization
    control: any scheme that still scores above zero after this has a leak in the harness
    itself, not signal in the data.
    """
    from .splits import iter_folds

    rng = np.random.default_rng(seed)
    oof = np.full(len(df), np.nan)
    for _f, tr, te in iter_folds(folds):
        y_tr = y[tr].copy()
        if shuffle_y_within == "global":
            rng.shuffle(y_tr)
        m = model_factory()
        m.fit(df.iloc[tr], X[tr], y_tr)
        oof[te] = m.predict(df.iloc[te], X[te])
    if np.isnan(oof).any():
        raise RuntimeError("some rows never appeared in a test fold")
    return oof
