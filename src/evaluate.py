"""Metrics and uncertainty.

Metric choices, justified against this specific dataset:

  spearman  PRIMARY. The score is a log2 fold change computed from counts with a +1
            pseudocount, giving excess kurtosis of 10.2 and a minimum of -6.45 driven by
            low-count variants. A rank statistic is the only correlation that is not
            dominated by those few variants, and it is the ProteinGym convention, so
            numbers here are comparable to published zero-shot results.
  pearson   Reported because the sqrt(ICC) ceiling is a Pearson bound; it is the metric
            the theoretical limit applies to directly.
  precision_at_5pct  The realistic downstream use is "pick residues worth testing".
            Rank-correlation over 8000 mostly-uninformative variants can look acceptable
            while the top of the list is useless, so the top of the list is scored too.
  auroc     On DMS_score_bin. Included only for comparability with ProteinGym leaderboards.
            It is *redundant* here: the supplied bin is exactly a median split of
            DMS_score, so it adds no information and inherits the same noise.

Uncertainty: a plain bootstrap over variants would ignore that variants at the same
position are correlated and that region-blocked folds are spatially contiguous. Every
confidence interval below resamples whole clusters (positions, or folds) with replacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _safe(fn, *a):
    try:
        v = fn(*a)
        v = v[0] if isinstance(v, tuple) else v
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.05) -> float:
    """Fraction of the predicted top-`frac` that is genuinely in the true top-`frac`."""
    k = max(int(round(frac * len(y_true))), 1)
    if k >= len(y_true):
        return np.nan
    top_true = set(np.argsort(-y_true)[:k].tolist())
    top_pred = np.argsort(-y_pred)[:k]
    return float(np.mean([i in top_true for i in top_pred]))


def auroc(y_bin: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_bin)) < 2:
        return np.nan
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_bin, y_pred))


def core_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_bin: np.ndarray | None = None
) -> dict[str, float]:
    if np.allclose(np.std(y_pred), 0):
        return {
            "spearman": 0.0,
            "pearson": 0.0,
            "precision_at_5pct": np.nan,
            "auroc": 0.5 if y_bin is not None else np.nan,
            "n": len(y_true),
        }
    return {
        "spearman": _safe(spearmanr, y_true, y_pred),
        "pearson": _safe(pearsonr, y_true, y_pred),
        "precision_at_5pct": precision_at_k(y_true, y_pred),
        "auroc": auroc(y_bin, y_pred) if y_bin is not None else np.nan,
        "n": len(y_true),
    }


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
    stat: str = "spearman",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile CI for a rank/linear correlation, resampling whole clusters.

    Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    fn = spearmanr if stat == "spearman" else pearsonr
    point = _safe(fn, y_true, y_pred)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_cluster[c] for c in picked])
        v = _safe(fn, y_true[sel], y_pred[sel])
        if np.isfinite(v):
            draws.append(v)
    if not draws:
        return point, np.nan, np.nan
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_cluster_bootstrap_delta(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    clusters: np.ndarray,
    stat: str = "spearman",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """CI for (metric of A) - (metric of B) on the same rows, resampling clusters.

    Used for every model-vs-baseline comparison, because with per-fold Spearman values
    around 0.1 the differences of interest are the same size as the sampling noise.
    """
    rng = np.random.default_rng(seed)
    fn = spearmanr if stat == "spearman" else pearsonr
    delta = _safe(fn, y_true, pred_a) - _safe(fn, y_true, pred_b)
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_cluster[c] for c in picked])
        d = _safe(fn, y_true[sel], pred_a[sel]) - _safe(fn, y_true[sel], pred_b[sel])
        if np.isfinite(d):
            draws.append(d)
    draws = np.asarray(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "delta": float(delta),
        "lo": float(lo),
        "hi": float(hi),
        "p_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
    }


def per_fold_metrics(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, folds: np.ndarray
) -> pd.DataFrame:
    """Metrics computed inside each fold, then reported as mean +/- sd.

    This removes any between-fold mean differences from the correlation, which matters for
    the region-blocked and domain-holdout splits: a model can score well on pooled
    predictions purely by ordering the domains correctly, without resolving any variant
    within a domain. Pooled and per-fold numbers are therefore both reported.
    """
    rows = []
    for f in np.unique(folds):
        m = folds == f
        r = core_metrics(y_true[m], y_pred[m], df["DMS_score_bin"].to_numpy()[m])
        r["fold"] = int(f)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    folds: np.ndarray,
    clusters: np.ndarray,
    model_name: str,
    split_name: str,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    pooled = core_metrics(y_true, y_pred, df["DMS_score_bin"].to_numpy())
    _, lo, hi = cluster_bootstrap_ci(y_true, y_pred, clusters, "spearman", n_boot, seed)
    pf = per_fold_metrics(df, y_true, y_pred, folds)
    return {
        "model": model_name,
        "split": split_name,
        "pooled_spearman": pooled["spearman"],
        "spearman_ci_lo": lo,
        "spearman_ci_hi": hi,
        "pooled_pearson": pooled["pearson"],
        "pooled_auroc": pooled["auroc"],
        "precision_at_5pct": pooled["precision_at_5pct"],
        "per_fold_spearman_mean": float(pf["spearman"].mean()),
        "per_fold_spearman_sd": float(pf["spearman"].std(ddof=1)),
        "n": int(pooled["n"]),
        "n_folds": int(pf.shape[0]),
    }
