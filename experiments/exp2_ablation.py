"""EXPERIMENT 2 - feature-group ablation.

Answers the ablation part of requirement 3.

WHICH PART OF THE PROPOSAL THIS TESTS
    Whether the pretrained-protein-language-model contribution (ESM-2 650M scores and
    embeddings) is doing work that handcrafted biophysics, position/domain annotation and
    error-prone-PCR accessibility features do not already do. That is the decisive question
    for the sequence-model half of the proposal: if the ESM blocks are removable at no
    cost, there is no case for building a bigger pretrained-model-based stack on this label.

    It says nothing about structural or diffusion features, which are not computed anywhere.

Run under two splits:
    position_grouped - generalization to unmeasured residues
    region_blocked   - generalization to unmeasured regions (the stricter claim)

Both leave-one-group-out and only-one-group are reported, because with correlated feature
blocks a group can look useless in LOGO (its information is duplicated elsewhere) while
still being sufficient on its own.

Outputs
    results/ablation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import RESULTS  # noqa: E402
from src.evaluate import paired_cluster_bootstrap_delta, summarize  # noqa: E402
from src.features import FEATURE_GROUPS, columns_excluding, columns_only  # noqa: E402
from src.models import HGB, run_cv  # noqa: E402
from src.splits import (  # noqa: E402
    SPLIT_BOOTSTRAP_UNIT,
    SPLIT_POSITION,
    SPLIT_REGION,
    assign_folds,
)
from experiments.exp1_protocol_and_baselines import (  # noqa: E402
    N_FOLDS,
    SEED,
    build_design_matrix,
)

SPLITS = [SPLIT_POSITION, SPLIT_REGION]


def main() -> None:
    df, wt, X, names, groups, have_esm = build_design_matrix()
    y = df["DMS_score"].to_numpy(float)
    present = [g for g in FEATURE_GROUPS if groups.get(g)]
    print(f"design matrix {X.shape}; groups present: {present}")

    rows = []
    for split in SPLITS:
        folds = assign_folds(df, split, N_FOLDS, SEED)
        clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[split] == "pos" else folds

        full_cols = np.arange(len(names))
        full_oof = run_cv(lambda: HGB(cols=full_cols, name="full"), df, X, y, folds, seed=SEED)
        s = summarize(df, y, full_oof, folds, clusters, "FULL", split, 500, SEED)
        s.update(variant="full", group="-", n_features=len(full_cols))
        rows.append(s)
        print(f"\n=== {split} ===\n  FULL ({len(full_cols)} feats) "
              f"rho {s['pooled_spearman']:+.3f}")

        for g in present:
            # leave one group out
            cols = columns_excluding(groups, [g], len(names))
            oof = run_cv(lambda c=cols: HGB(cols=c, name=f"minus_{g}"), df, X, y, folds, seed=SEED)
            s = summarize(df, y, oof, folds, clusters, f"minus_{g}", split, 500, SEED)
            d = paired_cluster_bootstrap_delta(y, oof, full_oof, clusters, "spearman", 500, SEED)
            s.update(variant="leave_one_group_out", group=g, n_features=len(cols),
                     delta_vs_full=d["delta"], delta_lo=d["lo"], delta_hi=d["hi"],
                     delta_p=d["p_two_sided"])
            rows.append(s)
            print(f"  -{g:10s} ({len(cols):3d} feats) rho {s['pooled_spearman']:+.3f}  "
                  f"delta {d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] p={d['p_two_sided']:.3f}")

            # only that group
            cols1 = columns_only(groups, [g])
            oof1 = run_cv(lambda c=cols1: HGB(cols=c, name=f"only_{g}"), df, X, y, folds, seed=SEED)
            s1 = summarize(df, y, oof1, folds, clusters, f"only_{g}", split, 500, SEED)
            s1.update(variant="only_one_group", group=g, n_features=len(cols1))
            rows.append(s1)
            print(f"   only {g:10s} ({len(cols1):3d} feats) rho {s1['pooled_spearman']:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ablation.csv", index=False)
    print(f"\nwrote {RESULTS / 'ablation.csv'}")
    print("\n--- leave-one-group-out, pooled Spearman ---")
    logo = out[out["variant"].isin(["full", "leave_one_group_out"])]
    print(logo.pivot(index="group", columns="split", values="pooled_spearman")
          .round(3).to_string())
    print("\n--- only-one-group, pooled Spearman ---")
    only = out[out["variant"] == "only_one_group"]
    print(only.pivot(index="group", columns="split", values="pooled_spearman")
          .round(3).to_string())


if __name__ == "__main__":
    main()
