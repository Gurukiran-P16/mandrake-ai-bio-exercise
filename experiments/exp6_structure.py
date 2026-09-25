"""EXPERIMENT 6 - do static structural features close the regional headroom?

This is the experiment the report named as the one most likely to change its
recommendation, run to settle it.

THE CLAIM UNDER TEST
    exp3 test B found achieved correlation sitting 0.22-0.35 below the split-half
    reliability ceiling at every interpretable aggregation scale, with the gap widening as
    labels were averaged. The report's recommendation rested on the hypothesis that this
    unexplained regional signal is STRUCTURAL in origin. If that is right, adding features
    from one experimental structure should close a visible part of the gap.

    Pre-committed readings:
      headroom closes substantially -> the structure arm is worth real investment
      headroom does not move        -> the missing signal is not static WT structure, and
                                       the case for the structure arm weakens rather than
                                       strengthens

WHICH PART OF THE PROPOSAL THIS TESTS
    The structural half - but with STATIC features from a solved crystal structure, which
    is what the MODIFY recommendation proposed *instead of* a diffusion trunk. It still
    does NOT test any diffusion model, and it does NOT test per-variant structure
    prediction (folding each mutant), which remains untested in this project.

STRUCTURE: PDB 4UN3, chain B. See src/struct_feats.py for the two properties of this
structure that are recorded rather than ignored (it is the H840A nickase; 4.5% of
positions have no coordinates).

Paired confidence intervals on the structural gain are computed here too (they were
briefly a separate script; folded in because they are one bootstrap on this experiment's
own output, not an independent experiment).

Outputs
    results/struct_position_features.csv
    results/struct_reference_check.json
    results/struct_variant_level.csv
    results/struct_aggregation_ladder.csv
    results/struct_feature_correlations.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RESULTS, load_assay, read_wt_sequence  # noqa: E402
from src.evaluate import paired_cluster_bootstrap_delta, summarize  # noqa: E402
from src.features import (  # noqa: E402
    add_struct_features,
    columns_only,
)
from src.models import HGB, run_cv  # noqa: E402
from src.splits import (  # noqa: E402
    SPLIT_BOOTSTRAP_UNIT,
    SPLIT_POSITION,
    SPLIT_REGION,
    assign_folds,
)
from src.struct_feats import (  # noqa: E402
    STRUCT_FEATURES,
    build_struct_table,
    check_against_reference,
)

from experiments.exp1_protocol_and_baselines import (  # noqa: E402
    N_FOLDS,
    SEED,
    build_design_matrix,
)
from experiments.exp3_challenge import (  # noqa: E402
    WINDOWS,
    _cv_predict,
    _permutation_null,
    blocked_folds,
    split_half_reliability,
)


def paired_unit_bootstrap(
    target: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    n_boot: int = 2000, seed: int = 0,
) -> dict:
    """CI for r(A) - r(B) on the same aggregation units, resampling units.

    The point estimates alone are not enough: the structural gains are small and
    non-monotone across scales, which is what a weak effect plus noise looks like. This
    puts an interval on each one so the write-up cannot overclaim.
    """
    rng = np.random.default_rng(seed)
    n = len(target)
    delta = pearsonr(target, pred_a)[0] - pearsonr(target, pred_b)[0]
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if min(np.std(target[idx]), np.std(pred_a[idx]), np.std(pred_b[idx])) == 0:
            continue
        draws.append(pearsonr(target[idx], pred_a[idx])[0]
                     - pearsonr(target[idx], pred_b[idx])[0])
    draws = np.asarray(draws)
    return {
        "delta": float(delta),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "p_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
    }


def position_level_columns(names: list[str], groups: dict[str, list[int]],
                           with_struct: bool) -> np.ndarray:
    """The position-only information channel, optionally including structure.

    Identical to the channel used in exp3 test A/B, so the with/without comparison is
    apples to apples: same model, same splits, same target, only the feature set differs.
    """
    keep = ["position", "esm_emb"] + (["struct", "struct_win"] if with_struct else [])
    cols = set(columns_only(groups, keep).tolist())
    cols |= {i for i, n in enumerate(names)
             if n in ("esm_maskmarg_entropy", "esm_maskmarg_logp_wt",
                      "esm_wtmarg_entropy", "esm_wtmarg_logp_wt")}
    return np.array(sorted(cols), dtype=int)


def ladder(df: pd.DataFrame, X: np.ndarray, cols: np.ndarray, label: str,
           baseline_oof: dict[int, np.ndarray] | None = None
           ) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """The exp3 aggregation ladder, restricted to a given feature channel.

    If `baseline_oof` is supplied (scale -> out-of-fold predictions from another channel),
    each scale also gets a paired bootstrap of the difference against it.

    Returns (frame, oof_by_scale) as a plain tuple. An earlier version returned the arrays
    inside `DataFrame.attrs`, which pd.concat then tried to compare for equality and raised
    on the numpy arrays - the compute all succeeded and only the CSV write failed, which is
    a good argument for not smuggling state through metadata.
    """
    y = df["DMS_score"].to_numpy(float)
    pos = df["pos"].to_numpy()
    rows = []
    kept_oof: dict[int, np.ndarray] = {}
    print(f"\n  [{label}] {len(cols)} features")
    for w in WINDOWS:
        unit = (pos - 1) // w
        agg = pd.DataFrame({"unit": unit, "y": y})
        gy = agg.groupby("unit")["y"]
        unit_ids = np.sort(agg["unit"].unique())
        target = gy.mean().reindex(unit_ids).to_numpy()
        counts = gy.size().reindex(unit_ids).to_numpy()
        Xu = pd.DataFrame(X[:, cols]).groupby(unit).mean().reindex(unit_ids).to_numpy()

        n_folds = min(N_FOLDS, max(2, len(unit_ids) // 3))
        folds = blocked_folds(unit_ids, n_folds)
        oof = _cv_predict(Xu, target, folds)
        rho = float(spearmanr(target, oof)[0])
        r_p = float(pearsonr(target, oof)[0])
        null_mu, null_sd = _permutation_null(Xu, target, folds, 20, SEED)
        _, r_full = split_half_reliability(
            [v.to_numpy() for _, v in agg.groupby("unit")["y"]], 200, SEED
        )
        ceiling = float(np.sqrt(r_full)) if np.isfinite(r_full) and r_full > 0 else np.nan
        interpretable = len(unit_ids) >= 100
        kept_oof[w] = oof
        row = {
            "channel": label, "window": w, "n_units": len(unit_ids),
            "mean_labels_per_unit": float(counts.mean()),
            "spearman": rho, "pearson": r_p,
            "perm_null_mean": null_mu,
            "spearman_above_null": rho - null_mu if np.isfinite(null_mu) else np.nan,
            "reliability_full": r_full, "ceiling": ceiling,
            "headroom": (ceiling - r_p) if np.isfinite(ceiling) else np.nan,
            "interpretable": interpretable,
        }
        # Paired significance against the reference channel, on the same units.
        if baseline_oof is not None and w in baseline_oof and interpretable:
            d = paired_unit_bootstrap(target, oof, baseline_oof[w], 2000, SEED)
            row.update(delta_vs_baseline=d["delta"], delta_lo=d["lo"],
                       delta_hi=d["hi"], delta_p=d["p_two_sided"])
        if interpretable:
            extra = ""
            if "delta_p" in row:
                extra = (f"   delta {row['delta_vs_baseline']:+.3f} "
                         f"[{row['delta_lo']:+.3f},{row['delta_hi']:+.3f}] "
                         f"p={row['delta_p']:.3f}")
            print(f"    w={w:3d}  n={len(unit_ids):4d}  rho {rho:+.3f} (above null "
                  f"{rho - null_mu:+.3f})  r {r_p:+.3f}  ceiling {ceiling:.3f}  "
                  f"headroom {ceiling - r_p:+.3f}{extra}")
        rows.append(row)
    return pd.DataFrame(rows), kept_oof


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    from src.struct_feats import STRUCTURE_PDB

    if not STRUCTURE_PDB.exists():
        print("structure not found:", STRUCTURE_PDB)
        print("fetch it with:\n  mkdir -p data/structure\n"
              "  curl -o data/structure/4un3.pdb https://files.rcsb.org/download/4UN3.pdb")
        print("skipping exp6.")
        return
    wt = read_wt_sequence()

    print("=" * 78)
    print("STRUCTURE: PDB 4UN3 chain B, checked against the assay reference sequence")
    print("=" * 78)
    chk = check_against_reference(wt)
    (RESULTS / "struct_reference_check.json").write_text(json.dumps(chk, indent=2))
    print(f"  modelled {chk['n_modelled']}/{chk['n_reference']} positions "
          f"({100 * chk['coverage']:.1f}%)")
    print(f"  sequence mismatches vs reference: {chk['n_mismatches']} -> {chk['mismatches']}")
    print(f"  disordered runs with no coordinates: {chk['missing_runs']}")

    st = build_struct_table(wt)
    st.to_csv(RESULTS / "struct_position_features.csv")
    print(f"  structural feature table: {st.shape[0]} positions x {st.shape[1]} features")

    df, _wt, X0, names0, groups0, have_esm = build_design_matrix()
    X, names, groups = add_struct_features(df, X0, names0, groups0, st)
    y = df["DMS_score"].to_numpy(float)
    print(f"\n  design matrix {X0.shape} -> {X.shape}  "
          f"(struct {len(groups['struct'])}, struct_win {len(groups['struct_win'])})")

    # ---------------------------------------------------------------- test E
    print("\n" + "=" * 78)
    print("E  raw association of each structural feature with the POSITION-MEAN score")
    print("=" * 78)
    pm = df.groupby("pos")["DMS_score"].mean()
    rows = []
    for f in STRUCT_FEATURES:
        for w in ["", "_w11", "_w25"]:
            col = f + w
            if col not in st.columns:
                continue
            v = st[col].reindex(pm.index)
            m = v.notna().to_numpy() & np.isfinite(pm.to_numpy())
            if m.sum() < 50 or np.nanstd(v.to_numpy()[m]) == 0:
                continue
            rows.append({
                "feature": col,
                "spearman_vs_position_mean": float(spearmanr(v.to_numpy()[m], pm.to_numpy()[m])[0]),
                "n": int(m.sum()),
            })
    fc = pd.DataFrame(rows).sort_values("spearman_vs_position_mean")
    fc.to_csv(RESULTS / "struct_feature_correlations.csv", index=False)
    print(fc.head(8).round(3).to_string(index=False))
    print("  ...")
    print(fc.tail(5).round(3).to_string(index=False))

    # ---------------------------------------------------------------- test A
    print("\n" + "=" * 78)
    print("A  VARIANT level: does adding structure help the full model?")
    print("=" * 78)
    all_no_struct = np.arange(len(names0))
    all_with_struct = np.arange(len(names))
    struct_only = columns_only(groups, ["struct", "struct_win"])
    rows = []
    for split in [SPLIT_POSITION, SPLIT_REGION]:
        folds = assign_folds(df, split, N_FOLDS, SEED)
        clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[split] == "pos" else folds
        print(f"\n  --- {split} ---")
        preds = {}
        for tag, cols in [("no_struct", all_no_struct),
                          ("with_struct", all_with_struct),
                          ("struct_only", struct_only)]:
            oof = run_cv(lambda c=cols: HGB(cols=c, name=tag), df, X, y, folds, seed=SEED)
            preds[tag] = oof
            s = summarize(df, y, oof, folds, clusters, tag, split, 500, SEED)
            s["n_features"] = len(cols)
            rows.append(s)
            print(f"    {tag:12s} ({len(cols):3d} feats)  pooled rho {s['pooled_spearman']:+.3f}"
                  f"  per-fold {s['per_fold_spearman_mean']:+.3f}"
                  f"+-{s['per_fold_spearman_sd']:.3f}")
        d = paired_cluster_bootstrap_delta(y, preds["with_struct"], preds["no_struct"],
                                           clusters, "spearman", 500, SEED)
        print(f"    with_struct - no_struct: delta {d['delta']:+.4f} "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}] p={d['p_two_sided']:.3f}")
        rows.append({"model": "delta_with_minus_no_struct", "split": split,
                     "pooled_spearman": d["delta"], "spearman_ci_lo": d["lo"],
                     "spearman_ci_hi": d["hi"], "n": len(y),
                     "per_fold_spearman_mean": np.nan, "per_fold_spearman_sd": np.nan,
                     "delta_p": d["p_two_sided"]})
    pd.DataFrame(rows).to_csv(RESULTS / "struct_variant_level.csv", index=False)

    # ------------------------------------------------------------ tests B/C
    print("\n" + "=" * 78)
    print("B/C  REGIONAL level: does structure close the headroom? (the decisive test)")
    print("=" * 78)
    base, base_oof = ladder(df, X, position_level_columns(names, groups, False),
                            "without_structure")
    with_s, _ = ladder(df, X, position_level_columns(names, groups, True),
                       "with_structure", baseline_oof=base_oof)
    only_s, _ = ladder(df, X, struct_only, "structure_only", baseline_oof=base_oof)
    lad = pd.concat([base, with_s, only_s], ignore_index=True)
    lad.to_csv(RESULTS / "struct_aggregation_ladder.csv", index=False)

    print("\n" + "=" * 78)
    print("VERDICT INPUTS: headroom by channel (interpretable scales only)")
    print("=" * 78)
    ok = lad[lad["interpretable"]]
    piv = ok.pivot(index="window", columns="channel", values="headroom")
    order = [c for c in ["without_structure", "with_structure", "structure_only"]
             if c in piv.columns]
    piv = piv[order]
    piv["headroom_closed"] = piv["without_structure"] - piv["with_structure"]
    print(piv.round(3).to_string())
    rp = ok.pivot(index="window", columns="channel", values="pearson")[order]
    print("\nachieved Pearson r by channel:")
    print(rp.round(3).to_string())
    closed = float(piv["headroom_closed"].mean())
    print(f"\nmean headroom closed by adding structure: {closed:+.3f}")
    print(f"mean headroom remaining with structure:   "
          f"{float(piv['with_structure'].mean()):+.3f}")

    sig = ok[(ok["channel"] == "with_structure") & ok["delta_p"].notna()] \
        if "delta_p" in ok.columns else ok.iloc[0:0]
    if len(sig):
        print("\npaired test, with_structure vs sequence-only, per scale:")
        print(sig[["window", "n_units", "delta_vs_baseline", "delta_lo", "delta_hi",
                   "delta_p"]].round(4).to_string(index=False))
        n_sig = int((sig["delta_p"] < 0.05).sum())
        bonf = 0.05 / len(sig)
        print(f"\n  significant at p<0.05: {n_sig} of {len(sig)} scales")
        print(f"  Bonferroni threshold for {len(sig)} tests: {bonf:.4f}  -> "
              f"{int((sig['delta_p'] < bonf).sum())} of {len(sig)} survive correction")


if __name__ == "__main__":
    main()
