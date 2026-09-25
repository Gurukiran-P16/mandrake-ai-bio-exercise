"""EXPERIMENT 3 - an attempt to refute my own preferred interpretation.

THE INTERPRETATION UNDER ATTACK
    "The binding constraint on predicting this label is the reliability of the label
     itself, not model capacity and not the absence of structural features. Therefore a
     structure/diffusion trunk cannot be validated on this dataset."

    It is a convenient conclusion: it lets me decline to build the expensive half of the
    proposal. So it needs a test that can go against me, not one that confirms it.

THREE TESTS, EACH WITH A PRE-COMMITTED PREDICTION

 A. NOISE-REDUCTION LADDER
    Hold the feature set fixed (position-level features only) and vary how much label
    noise is averaged away: per-variant, per-position, then per-window means over windows
    of 3, 5, 11, 25, 51 and 137 residues.
      supports me  -> Spearman climbs steeply as labels are averaged
      refutes me   -> Spearman stays flat, meaning the features never encoded the biology
                      and noise was never the binding constraint

 B. RELIABILITY-MATCHED HEADROOM
    At each aggregation scale, measure the target's split-half reliability empirically and
    compare achieved Spearman against the ceiling sqrt(reliability) that it implies.
      supports me  -> achieved sits near its ceiling at every scale
      refutes me   -> a large gap opens at the aggregated scales, i.e. real headroom that
                      better features (plausibly structural ones) could fill. THIS IS THE
                      OUTCOME THAT WOULD ARGUE FOR PURSUING THE STRUCTURE ARM.

 C. CAPACITY LADDER
    Vary model capacity (depth 1-10, 50-1500 trees) under the position-grouped split.
      supports me  -> flat or degrading with capacity
      refutes me   -> performance rises with capacity, so the model, not the label, was
                      the limitation

WHICH PART OF THE PROPOSAL THIS TESTS
    Test B is the one that bears on the structure/diffusion half, and it does so
    indirectly and by bounding, not by using structural features. A feature computed from
    the wild-type structure is constant across the substitutions measured at a position,
    so it lives in the position-only information channel that this experiment measures the
    capacity of. That bounds static-structure features. It does NOT bound per-variant
    structure prediction (folding each mutant separately), which is untested here.

Outputs
    results/challenge_aggregation.csv
    results/challenge_capacity.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RESULTS  # noqa: E402
from src.evaluate import core_metrics  # noqa: E402
from src.features import columns_only  # noqa: E402
from src.models import HGB, run_cv  # noqa: E402
from src.splits import SPLIT_POSITION, assign_folds  # noqa: E402

from experiments.exp1_protocol_and_baselines import (  # noqa: E402
    N_FOLDS,
    SEED,
    build_design_matrix,
)

WINDOWS = [1, 3, 5, 11, 25, 51, 137]


def split_half_reliability(values: list[np.ndarray], n_repeats: int = 200, seed: int = 0):
    """Spearman-Brown-corrected reliability of the mean of each group in `values`."""
    rng = np.random.default_rng(seed)
    usable = [v for v in values if len(v) >= 4]
    if len(usable) < 8:
        return np.nan, np.nan
    rs = []
    for _ in range(n_repeats):
        a, b = [], []
        for v in usable:
            idx = rng.permutation(len(v))
            h = len(v) // 2
            a.append(v[idx[:h]].mean())
            b.append(v[idx[h : 2 * h]].mean())
        rs.append(pearsonr(a, b)[0])
    r = float(np.mean(rs))
    return r, float(2 * r / (1 + r)) if r > -1 else np.nan


def blocked_folds(units: np.ndarray, n_folds: int) -> np.ndarray:
    """Contiguous-block folds over an ordered unit index (windows or positions)."""
    edges = np.linspace(units.min() - 0.5, units.max() + 0.5, n_folds + 1)
    return (np.digitize(units, edges) - 1).clip(0, n_folds - 1)


def _cv_predict(Xu: np.ndarray, target: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions for the aggregated regression."""
    import pandas as pd

    oof = np.full(len(target), np.nan)
    dummy = pd.DataFrame({"pos": np.arange(len(target)), "DMS_score_bin": 0})
    for fid in np.unique(folds):
        te = folds == fid
        tr = ~te
        if tr.sum() < 5 or te.sum() < 1:
            continue
        mdl = HGB(cols=np.arange(Xu.shape[1]), name="w")
        mdl.fit(dummy[tr], Xu[tr], target[tr])
        oof[te] = mdl.predict(dummy[te], Xu[te])
    return oof


def _permutation_null(
    Xu: np.ndarray, target: np.ndarray, folds: np.ndarray, n_perm: int = 20, seed: int = 0
) -> tuple[float, float]:
    """Empirical null for pooled out-of-fold Spearman under this exact CV design.

    Needed because pooled out-of-fold correlation is NOT centred on zero under blocked
    CV: each fold's model can only regress towards its own training mean, so when the
    number of units is small and spatially structured the pooled statistic acquires a
    large negative bias. Measuring that bias directly is the only way to read the
    aggregated scales honestly. Returns (null_mean, null_sd).
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_perm):
        t = rng.permutation(target)
        oof = _cv_predict(Xu, t, folds)
        m = np.isfinite(oof)
        if m.sum() > 10 and np.std(oof[m]) > 0:
            draws.append(spearmanr(t[m], oof[m])[0])
    if not draws:
        return np.nan, np.nan
    return float(np.mean(draws)), float(np.std(draws, ddof=1))


def test_a_and_b(df, X, names, groups) -> pd.DataFrame:
    """Noise-reduction ladder + reliability-matched headroom."""
    y = df["DMS_score"].to_numpy(float)
    pos = df["pos"].to_numpy()

    # Position-level features only: these are exactly the features that are constant
    # within a position, which is the channel a wild-type-structure feature would occupy.
    cols = columns_only(groups, ["position", "esm_emb"])
    extra = [i for i, n in enumerate(names)
             if n in ("esm_maskmarg_entropy", "esm_maskmarg_logp_wt",
                      "esm_wtmarg_entropy", "esm_wtmarg_logp_wt")]
    cols = np.array(sorted(set(cols.tolist()) | set(extra)), dtype=int)
    print(f"position-level feature channel: {len(cols)} features")

    rows = []

    # --- per-variant reference point (no aggregation) --------------------
    folds = assign_folds(df, SPLIT_POSITION, N_FOLDS, SEED)
    oof = run_cv(lambda: HGB(cols=cols, name="pos_only"), df, X, y, folds, seed=SEED)
    m = core_metrics(y, oof)
    # No split-half reliability is defined for a single unaggregated measurement, so the
    # variant row is the anchor of the ladder, not a ceiling comparison.
    rows.append({
        "scale": "variant", "window": 0, "n_units": len(y),
        "mean_labels_per_unit": 1.0,
        "spearman": m["spearman"], "pearson": m["pearson"],
        "reliability_split_half": np.nan, "reliability_full": np.nan,
        "ceiling_from_reliability": np.nan, "headroom": np.nan,
    })
    print(f"  variant-level (n={len(y)}): rho {m['spearman']:+.3f}")

    # --- aggregated scales ----------------------------------------------
    for w in WINDOWS:
        unit = (pos - 1) // w
        agg = pd.DataFrame({"unit": unit, "y": y})
        gy = agg.groupby("unit")["y"]
        unit_ids = np.sort(agg["unit"].unique())
        target = gy.mean().reindex(unit_ids).to_numpy()
        counts = gy.size().reindex(unit_ids).to_numpy()

        # features: mean of the position-level features over the window
        Xu = (
            pd.DataFrame(X[:, cols])
            .groupby(unit)
            .mean()
            .reindex(unit_ids)
            .to_numpy()
        )
        n_folds = min(N_FOLDS, max(2, len(unit_ids) // 3))
        f_blocked = blocked_folds(unit_ids, n_folds)
        oof = _cv_predict(Xu, target, f_blocked)
        rho = float(spearmanr(target, oof)[0])
        r_p = float(pearsonr(target, oof)[0])

        # Empirical null for this exact design, and a non-blocked sensitivity check.
        null_mu, null_sd = _permutation_null(Xu, target, f_blocked, 20, SEED)
        rng = np.random.default_rng(SEED)
        f_random = rng.permutation(np.arange(len(unit_ids)) % n_folds)
        oof_r = _cv_predict(Xu, target, f_random)
        rho_random = float(spearmanr(target, oof_r)[0])

        r_half, r_full = split_half_reliability(
            [v.to_numpy() for _, v in agg.groupby("unit")["y"]], 200, SEED
        )
        ceiling = float(np.sqrt(r_full)) if np.isfinite(r_full) and r_full > 0 else np.nan
        # Blocked CV is only interpretable while enough independent units remain; below
        # this the pooled statistic is dominated by the between-fold-mean artefact.
        reliable = len(unit_ids) >= 100
        rows.append({
            "scale": f"window_{w}", "window": w, "n_units": len(unit_ids),
            "mean_labels_per_unit": float(counts.mean()),
            "spearman": rho, "pearson": r_p,
            "spearman_perm_null_mean": null_mu, "spearman_perm_null_sd": null_sd,
            "spearman_above_null": rho - null_mu if np.isfinite(null_mu) else np.nan,
            "spearman_random_unit_cv": rho_random,
            "reliability_split_half": r_half, "reliability_full": r_full,
            "ceiling_from_reliability": ceiling,
            "headroom": (ceiling - r_p) if np.isfinite(ceiling) else np.nan,
            "blocked_cv_interpretable": reliable,
        })
        flag = "" if reliable else "   <-- too few units, blocked CV not interpretable"
        print(f"  window {w:3d} aa (n={len(unit_ids):4d} units, "
              f"{counts.mean():5.1f} labels/unit): rho {rho:+.3f} "
              f"(null {null_mu:+.3f}+-{null_sd:.3f}, above null {rho - null_mu:+.3f}; "
              f"random-unit CV {rho_random:+.3f})  "
              f"reliability {r_full:.3f} -> ceiling {ceiling:.3f}  "
              f"headroom {ceiling - r_p:+.3f}{flag}")
    return pd.DataFrame(rows)


def test_c(df, X, names, groups) -> pd.DataFrame:
    """Capacity ladder under the position-grouped split."""
    y = df["DMS_score"].to_numpy(float)
    folds = assign_folds(df, SPLIT_POSITION, N_FOLDS, SEED)
    all_cols = np.arange(len(names))
    rows = []
    for depth in [1, 2, 3, 6, 10]:
        for n_iter in [50, 300, 1500]:
            oof = run_cv(
                lambda d=depth, it=n_iter: HGB(
                    cols=all_cols, name="cap", max_depth=d, max_iter=it,
                    min_samples_leaf=20, l2_regularization=1.0,
                ),
                df, X, y, folds, seed=SEED,
            )
            m = core_metrics(y, oof, df["DMS_score_bin"].to_numpy())
            rows.append({"max_depth": depth, "max_iter": n_iter,
                         "spearman": m["spearman"], "pearson": m["pearson"],
                         "auroc": m["auroc"]})
            print(f"  depth {depth:2d} iter {n_iter:4d}: rho {m['spearman']:+.3f}")
    return pd.DataFrame(rows)


def main() -> None:
    df, wt, X, names, groups, have_esm = build_design_matrix()
    print("=" * 78)
    print("TEST A/B  noise-reduction ladder and reliability-matched headroom")
    print("=" * 78)
    ab = test_a_and_b(df, X, names, groups)
    ab.to_csv(RESULTS / "challenge_aggregation.csv", index=False)

    print("\n" + "=" * 78)
    print("TEST C  capacity ladder (position-grouped split)")
    print("=" * 78)
    c = test_c(df, X, names, groups)
    c.to_csv(RESULTS / "challenge_capacity.csv", index=False)

    print("\n" + "=" * 78)
    print("VERDICT INPUTS")
    print("=" * 78)
    cols = ["scale", "n_units", "mean_labels_per_unit", "spearman",
            "spearman_perm_null_mean", "spearman_above_null", "spearman_random_unit_cv",
            "pearson", "reliability_full", "ceiling_from_reliability", "headroom",
            "blocked_cv_interpretable"]
    print(ab[[c for c in cols if c in ab.columns]].round(3).to_string(index=False))
    ok = ab[ab.get("blocked_cv_interpretable", True) == True]  # noqa: E712
    if len(ok):
        print("\nInterpretable scales only (n_units >= 100):")
        print(f"  Spearman rises {ok['spearman'].iloc[0]:+.3f} -> "
              f"{ok['spearman'].iloc[-1]:+.3f} as labels per unit go "
              f"{ok['mean_labels_per_unit'].iloc[0]:.1f} -> "
              f"{ok['mean_labels_per_unit'].iloc[-1]:.1f}")
        print(f"  Pearson stays {ok['headroom'].min():.2f}-{ok['headroom'].max():.2f} "
              f"BELOW the reliability ceiling at every scale")
    best = c.loc[c["spearman"].idxmax()]
    base = c[(c["max_depth"] == 3) & (c["max_iter"] == 300)]["spearman"]
    print(f"\ncapacity ladder: best rho {best['spearman']:+.3f} at depth "
          f"{int(best['max_depth'])}/iter {int(best['max_iter'])}; "
          f"default depth3/iter300 rho {float(base.iloc[0]):+.3f}")
    print(f"capacity ladder spread: {c['spearman'].min():+.3f} to {c['spearman'].max():+.3f}")


if __name__ == "__main__":
    main()
