"""EXPERIMENT 1 - evaluation protocol, baselines, and the substantive extension.

Answers requirements 2 and 3 (the main predictor and its held-out predictions).

WHICH PART OF THE PROPOSAL THIS TESTS
    The proposal is: take components of a pretrained protein sequence model AND a
    diffusion-based structure predictor, replace the structure-prediction trunk with an
    activity head, predict gene-editor activity.

    This experiment tests the FIRST half only - whether a pretrained protein sequence
    model (ESM-2 650M) carries information about this activity label, zero-shot and as
    features for a supervised head. It does NOT test the structure/diffusion half. No
    structural coordinates, no predicted structures and no diffusion model are used
    anywhere in this repository.

Outputs
    results/split_assignments.csv       fold id per variant under every scheme
    results/split_leakage.csv           what each scheme actually removes
    results/heldout_predictions.csv     out-of-fold prediction per variant per model
    results/main_results.csv            metrics for every model x split
    results/model_vs_baseline.csv       paired cluster-bootstrap comparisons
    results/y_randomization.csv         negative control for the harness itself
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RESULTS, load_assay  # noqa: E402
from src.esm_feats import CACHE  # noqa: E402
from src.evaluate import paired_cluster_bootstrap_delta, summarize  # noqa: E402
from src.features import (  # noqa: E402
    add_esm_features,
    build_handcrafted,
    columns_only,
)
from src.models import (  # noqa: E402
    HGB,
    Blosum62,
    ColumnScore,
    GlobalMean,
    GroupMean,
    Ridge,
    run_cv,
)
from src.splits import (  # noqa: E402
    ALL_SPLITS,
    SPLIT_BOOTSTRAP_UNIT,
    SPLIT_CLAIMS,
    SPLIT_POSITION,
    assign_folds,
    build_split_table,
    leakage_report,
)

N_FOLDS = 10
SEED = 20260924


def load_esm_blocks() -> dict[str, np.ndarray | None]:
    """Load the cached ESM-2 blocks, or all-None if they are absent or disabled.

    Setting CAS9_SKIP_ESM=1 forces the ESM-free configuration even when the cache exists,
    so `run_all.py --skip-esm` means the same thing whether or not the cache was built
    on an earlier run.
    """
    import os

    if os.environ.get("CAS9_SKIP_ESM") == "1":
        print("[CAS9_SKIP_ESM=1] running without ESM-2 features")
        return {"wtmarg_logprobs": None, "maskmarg_logprobs": None, "embeddings": None}

    def maybe(p: Path):
        return np.load(p) if p.exists() else None

    return {
        "wtmarg_logprobs": maybe(CACHE / "esm2_650M_wtmarg_logprobs.npy"),
        "maskmarg_logprobs": maybe(CACHE / "esm2_650M_maskmarg_stride20_logprobs.npy"),
        "embeddings": maybe(CACHE / "esm2_650M_embeddings.npy"),
    }


def build_design_matrix():
    df, wt, _ = load_assay()
    X, names, groups = build_handcrafted(df)
    esm = load_esm_blocks()
    have_esm = any(v is not None for v in esm.values())
    X, names, groups = add_esm_features(df, X, names, groups, n_emb_pcs=32, **esm)
    return df, wt, X, names, groups, have_esm


def make_model_zoo(names: list[str], groups: dict[str, list[int]], have_esm: bool):
    """Model name -> zero-argument factory. Order is the reporting order."""
    idx = {n: i for i, n in enumerate(names)}
    handcrafted = columns_only(groups, ["subst", "codon", "position"])
    codon_only = columns_only(groups, ["codon"])
    all_cols = np.arange(len(names))

    zoo: dict[str, callable] = {
        "global_mean": GlobalMean,
        "blosum62": Blosum62,
        "domain_mean": lambda: GroupMean("domain"),
        "position_mean": lambda: GroupMean("pos", fallback="domain"),
        # confound probe: library-construction features only, no protein biology
        "codon_only_hgb": lambda: HGB(cols=codon_only, name="codon_only_hgb"),
        "ridge_handcrafted": lambda: Ridge(cols=handcrafted, name="ridge_handcrafted"),
        "hgb_handcrafted": lambda: HGB(cols=handcrafted, name="hgb_handcrafted"),
    }

    if have_esm:
        # Untrained zero-shot ESM-2 scores, run through the identical evaluation path.
        for tag in ("wtmarg", "maskmarg"):
            col = f"esm_{tag}_llr"
            if col in idx:
                zoo[f"esm650M_{tag}_zeroshot"] = (
                    lambda c=idx[col], t=tag: ColumnScore(c, f"esm650M_{t}_zeroshot")
                )
        esm_lm_only = columns_only(groups, ["esm_lm"])
        zoo["ridge_esm_lm"] = lambda: Ridge(cols=esm_lm_only, name="ridge_esm_lm")
        # ---- the substantive extension --------------------------------
        # every feature group: handcrafted biophysics + codon probe + position/domain
        # + ESM-2 language-model scores + PCA of ESM-2 wild-type embeddings.
        zoo["hgb_full_extension"] = lambda: HGB(cols=all_cols, name="hgb_full_extension")
        zoo["ridge_full_extension"] = lambda: Ridge(cols=all_cols, name="ridge_full_extension")
    return zoo


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df, wt, X, names, groups, have_esm = build_design_matrix()
    y = df["DMS_score"].to_numpy(float)
    print(f"design matrix {X.shape}  ESM features present: {have_esm}")
    print("feature group sizes: "
          + ", ".join(f"{g}={len(c)}" for g, c in groups.items() if c))

    # ---- split assignments + leakage diagnostics ------------------------
    split_table = build_split_table(df, N_FOLDS, SEED)
    split_table.to_csv(RESULTS / "split_assignments.csv", index=False)
    folds_by_split = {s: assign_folds(df, s, N_FOLDS, SEED) for s in ALL_SPLITS}

    leak = pd.DataFrame([leakage_report(df, folds_by_split[s], s) for s in ALL_SPLITS])
    leak.to_csv(RESULTS / "split_leakage.csv", index=False)
    print("\n--- what each split actually removes ---")
    print(leak[["scheme", "n_folds", "shared_position_rate",
                "neighbour_within_5_rate", "same_domain_rate"]].round(3).to_string(index=False))

    # ---- run every model under every split ------------------------------
    zoo = make_model_zoo(names, groups, have_esm)
    preds: dict[str, np.ndarray] = {}
    rows = []
    for split in ALL_SPLITS:
        folds = folds_by_split[split]
        clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[split] == "pos" else folds
        print(f"\n=== split {split} ({SPLIT_CLAIMS[split]}) ===")
        for mname, factory in zoo.items():
            oof = run_cv(factory, df, X, y, folds, seed=SEED)
            preds[f"{split}__{mname}"] = oof
            s = summarize(df, y, oof, folds, clusters, mname, split, n_boot=500, seed=SEED)
            s["claim"] = SPLIT_CLAIMS[split]
            rows.append(s)
            print(f"  {mname:24s} pooled rho {s['pooled_spearman']:+.3f} "
                  f"[{s['spearman_ci_lo']:+.3f},{s['spearman_ci_hi']:+.3f}]  "
                  f"per-fold {s['per_fold_spearman_mean']:+.3f}"
                  f"+-{s['per_fold_spearman_sd']:.3f}  "
                  f"AUROC {s['pooled_auroc']:.3f}  P@5% {s['precision_at_5pct']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "main_results.csv", index=False)

    # ---- held-out predictions (deliverable) -----------------------------
    hp = df[["mutant", "pos", "wt_aa", "mt_aa", "domain", "DMS_score", "DMS_score_bin"]].copy()
    for s in ALL_SPLITS:
        hp[f"fold_{s}"] = folds_by_split[s]
    for key, v in preds.items():
        hp[f"pred__{key}"] = v
    hp.to_csv(RESULTS / "heldout_predictions.csv", index=False)
    print(f"\nwrote {RESULTS / 'heldout_predictions.csv'}  "
          f"({hp.shape[0]} rows x {hp.shape[1]} cols)")

    # ---- paired comparisons against the baselines that matter -----------
    comp_rows = []
    challengers = [m for m in ("hgb_full_extension", "hgb_handcrafted",
                               "esm650M_maskmarg_zeroshot", "ridge_full_extension") if m in zoo]
    references = [m for m in ("domain_mean", "blosum62", "hgb_handcrafted",
                              "codon_only_hgb") if m in zoo]
    for split in ALL_SPLITS:
        folds = folds_by_split[split]
        clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[split] == "pos" else folds
        for a in challengers:
            for b in references:
                if a == b:
                    continue
                d = paired_cluster_bootstrap_delta(
                    y, preds[f"{split}__{a}"], preds[f"{split}__{b}"],
                    clusters, "spearman", n_boot=500, seed=SEED,
                )
                comp_rows.append({"split": split, "model": a, "reference": b, **d})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(RESULTS / "model_vs_baseline.csv", index=False)
    print("\n--- key paired comparisons (delta pooled Spearman, 95% cluster bootstrap) ---")
    show = comp[comp["model"] == challengers[0]] if challengers else comp
    print(show[["split", "model", "reference", "delta", "lo", "hi", "p_two_sided"]]
          .round(4).to_string(index=False))

    # ---- y-randomization: does the harness manufacture signal? ----------
    yr = []
    probe = "hgb_full_extension" if "hgb_full_extension" in zoo else "hgb_handcrafted"
    for split in ALL_SPLITS:
        folds = folds_by_split[split]
        clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[split] == "pos" else folds
        oof = run_cv(zoo[probe], df, X, y, folds, shuffle_y_within="global", seed=SEED)
        s = summarize(df, y, oof, folds, clusters, f"{probe}_Y_SHUFFLED", split, 300, SEED)
        yr.append(s)
        print(f"  y-shuffled {split:16s} pooled rho {s['pooled_spearman']:+.3f} "
              f"[{s['spearman_ci_lo']:+.3f},{s['spearman_ci_hi']:+.3f}]")
    pd.DataFrame(yr).to_csv(RESULTS / "y_randomization.csv", index=False)

    # ---- headline table -------------------------------------------------
    print("\n" + "=" * 78)
    print("POOLED SPEARMAN, models x splits")
    print("=" * 78)
    piv = res.pivot(index="model", columns="split", values="pooled_spearman")
    print(piv[[s for s in ALL_SPLITS if s in piv.columns]].round(3).to_string())


if __name__ == "__main__":
    main()
