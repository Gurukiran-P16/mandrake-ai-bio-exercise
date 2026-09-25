"""EXPERIMENT 5 - does the training objective matter on this label?

Requirement 4 asks for a justified training objective, so the justification should be
measured rather than asserted.

The primary metric is Spearman, which argues for a ranking objective. But the label is a
log2 count ratio with excess kurtosis 10.2 and a minimum of -6.45 driven by low-count
variants, which argues for anything that de-emphasises the tails. Both arguments point
away from plain squared error on the raw score, so three targets are compared with the
model and split held fixed:

    raw          squared error on DMS_score as supplied
    rank         squared error on the within-training-fold rank (a cheap stand-in for a
                 ranking loss; monotone in the target, so Spearman against the true score
                 is unaffected by the transform itself)
    winsorized   squared error on DMS_score clipped to its 1st/99th training percentiles

Target transforms are fitted on TRAINING folds only, so no test information is used.

WHICH PART OF THE PROPOSAL THIS TESTS
    None of the architecture. This isolates the objective/target choice so the
    architecture specification can cite evidence for it.

Output
    results/objective.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RESULTS  # noqa: E402
from src.evaluate import core_metrics, paired_cluster_bootstrap_delta  # noqa: E402
from src.models import HGB  # noqa: E402
from src.splits import (  # noqa: E402
    SPLIT_BOOTSTRAP_UNIT,
    SPLIT_POSITION,
    SPLIT_REGION,
    assign_folds,
    iter_folds,
)

from experiments.exp1_protocol_and_baselines import (  # noqa: E402
    N_FOLDS,
    SEED,
    build_design_matrix,
)


def transform_train(y_tr: np.ndarray, kind: str):
    if kind == "raw":
        return y_tr
    if kind == "rank":
        from scipy.stats import rankdata

        return rankdata(y_tr) / len(y_tr)
    if kind == "winsorized":
        lo, hi = np.percentile(y_tr, [1, 99])
        return np.clip(y_tr, lo, hi)
    raise ValueError(kind)


def main() -> None:
    df, wt, X, names, groups, _ = build_design_matrix()
    y = df["DMS_score"].to_numpy(float)
    all_cols = np.arange(len(names))
    rows = []
    preds: dict[str, np.ndarray] = {}

    for split in [SPLIT_POSITION, SPLIT_REGION]:
        folds = assign_folds(df, split, N_FOLDS, SEED)
        clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[split] == "pos" else folds
        print(f"\n=== {split} ===")
        for kind in ["raw", "rank", "winsorized"]:
            oof = np.full(len(df), np.nan)
            for _f, tr, te in iter_folds(folds):
                m = HGB(cols=all_cols, name=kind)
                m.fit(df.iloc[tr], X[tr], transform_train(y[tr], kind))
                oof[te] = m.predict(df.iloc[te], X[te])
            preds[f"{split}__{kind}"] = oof
            # always scored against the ORIGINAL score, so the transforms are comparable
            met = core_metrics(y, oof, df["DMS_score_bin"].to_numpy())
            rows.append({"split": split, "target": kind, **met})
            print(f"  {kind:11s} rho {met['spearman']:+.3f}  r {met['pearson']:+.3f}  "
                  f"AUROC {met['auroc']:.3f}  P@5% {met['precision_at_5pct']:.3f}")
        for kind in ["rank", "winsorized"]:
            d = paired_cluster_bootstrap_delta(
                y, preds[f"{split}__{kind}"], preds[f"{split}__raw"],
                clusters, "spearman", 500, SEED,
            )
            print(f"  {kind} vs raw: delta {d['delta']:+.3f} "
                  f"[{d['lo']:+.3f},{d['hi']:+.3f}] p={d['p_two_sided']:.3f}")
            rows.append({"split": split, "target": f"{kind}_vs_raw_delta",
                         "spearman": d["delta"], "pearson": np.nan,
                         "precision_at_5pct": np.nan, "auroc": np.nan,
                         "n": len(y), "delta_lo": d["lo"], "delta_hi": d["hi"],
                         "delta_p": d["p_two_sided"]})

    pd.DataFrame(rows).to_csv(RESULTS / "objective.csv", index=False)
    print(f"\nwrote {RESULTS / 'objective.csv'}")


if __name__ == "__main__":
    main()
