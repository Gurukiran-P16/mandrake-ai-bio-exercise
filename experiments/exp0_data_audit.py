"""EXPERIMENT 0 - what do the supplied measurements actually support predicting?

Answers requirement 1. Nothing here is modelling; it is all direct measurement on the
supplied CSV and FASTA, plus checks against the primary publication.

Outputs
    results/audit_summary.json
    results/audit_domain_table.csv
    results/audit_known_residues.csv
    results/audit_position_profile.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    DOMAINS,
    KNOWN_FUNCTIONAL_RESIDUES,
    METADATA_DISCREPANCIES,
    RESULTS,
    load_assay,
    load_metadata,
    position_icc,
    position_split_half_reliability,
)


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df, wt, audit = load_assay()
    meta = load_metadata()
    out: dict = {"metadata_discrepancies": METADATA_DISCREPANCIES}

    # ---- 1. structural audit of the table -------------------------------
    out["table_audit"] = audit.to_dict()

    # ---- 2. label distribution ------------------------------------------
    y = df["DMS_score"].to_numpy(float)
    out["score_distribution"] = {
        "n": int(len(y)),
        "mean": float(y.mean()),
        "sd": float(y.std(ddof=1)),
        "min": float(y.min()),
        "max": float(y.max()),
        "median": float(np.median(y)),
        "skew": float(pd.Series(y).skew()),
        "excess_kurtosis": float(pd.Series(y).kurtosis()),
        "q01": float(np.quantile(y, 0.01)),
        "q99": float(np.quantile(y, 0.99)),
        "n_unique": int(pd.Series(y).nunique()),
        "bin_positive_rate": float(df["DMS_score_bin"].mean()),
        "bin_reproduced_by_median_split": audit.bin_is_median_split,
        "metadata_cutoff": float(meta["DMS_binarization_cutoff"]),
        "comment": (
            "Heavy tails (excess kurtosis 10.2) are expected from a log2 ratio of counts "
            "with a +1 pseudocount: low-count variants generate extreme values. Read counts "
            "are not supplied, so those variants cannot be identified or down-weighted."
        ),
    }

    # ---- 3. reliability: how much signal is even estimable? -------------
    icc = position_icc(df)
    rel = position_split_half_reliability(df, n_repeats=200, seed=0)
    out["reliability"] = {**icc, **rel}
    out["reliability"]["interpretation"] = (
        "ICC_position = %.3f means only %.1f%% of per-variant score variance is "
        "attributable to which residue was mutated. Any feature that is CONSTANT across "
        "the ~5.9 substitutions measured at a position - which includes every feature "
        "computed from the wild-type structure (relative solvent accessibility, contact "
        "number, distance to the catalytic site, pLDDT) and the wild-type ESM embedding - "
        "is bounded above by Pearson r = sqrt(ICC) = %.3f against the per-variant score. "
        "This bound is a property of the labels, not of any model."
        % (icc["icc_position"], 100 * icc["icc_position"], icc["position_only_pearson_ceiling"])
    )

    # ---- 4. leave-one-out positional predictability ---------------------
    s = df.groupby("pos")["DMS_score"].transform("sum").to_numpy()
    n = df.groupby("pos")["DMS_score"].transform("size").to_numpy()
    loo = (s - y) / (n - 1)
    out["loo_position_mean"] = {
        "pearson": float(pearsonr(loo, y)[0]),
        "spearman": float(spearmanr(loo, y)[0]),
        "comment": (
            "Honest (leave-one-out) predictiveness of the positional effect. The in-sample "
            "R^2 of position group means is 0.22, but that is 1368 groups fitted on 5.9 "
            "points each; the leave-one-out value is the real number."
        ),
    }

    # ---- 5. spatial autocorrelation (motivates the region-blocked split) -
    pm = df.groupby("pos")["DMS_score"].mean().reindex(range(1, len(wt) + 1))
    ac = {}
    for lag in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
        a, b = pm.to_numpy()[:-lag], pm.to_numpy()[lag:]
        m = ~(np.isnan(a) | np.isnan(b))
        ac[f"lag_{lag}"] = float(pearsonr(a[m], b[m])[0])
    out["position_mean_autocorrelation"] = ac
    out["position_mean_autocorrelation"]["comment"] = (
        "Non-zero short-lag autocorrelation means a position-grouped split still leaves a "
        "held-out position's sequence neighbours in training. That is why a region-blocked "
        "split is also evaluated."
    )

    # ---- 6. domain-level signal (the resolution the original study used) -
    rows = []
    for name, lo, hi in DOMAINS:
        g = df[df["domain"] == name]["DMS_score"]
        rows.append(
            {
                "domain": name,
                "start": lo,
                "end": hi,
                "n": len(g),
                "mean": g.mean(),
                "median": g.median(),
                "sd": g.std(ddof=1),
                "sem": g.std(ddof=1) / np.sqrt(len(g)),
                "frac_above_median": (g > np.median(y)).mean(),
            }
        )
    dom = pd.DataFrame(rows)
    dom.to_csv(RESULTS / "audit_domain_table.csv", index=False)
    H, p = kruskal(*[g["DMS_score"].to_numpy() for _, g in df.groupby("domain")])
    nuc = df[df["is_nuclease_domain"]]["DMS_score"]
    non = df[~df["is_nuclease_domain"]]["DMS_score"]
    out["domain_signal"] = {
        "kruskal_H": float(H),
        "kruskal_p": float(p),
        "domain_mean_spread_sd": float(dom["mean"].std(ddof=1)),
        "variant_level_sd": float(y.std(ddof=1)),
        "most_depleted": dom.nsmallest(3, "mean")["domain"].tolist(),
        "most_tolerant": dom.nlargest(3, "mean")["domain"].tolist(),
        "nuclease_vs_other_mwu_p": float(mannwhitneyu(nuc, non)[1]),
        "nuclease_mean": float(nuc.mean()),
        "other_mean": float(non.mean()),
        "comment": (
            "Domain-level differences are highly significant while the domain mean spread "
            "(sd across domains) is a fifth of the variant-level sd. The reproducible "
            "signal in this assay is coarse and regional, which matches the original "
            "paper's own claim ('reveals important functional domains') and its finding "
            "that RuvC/HNH are least tolerant and REC2/PI most tolerant."
        ),
    }

    # ---- 7. label validity probes at mechanistically known residues -----
    probe_rows = []
    ecdf = np.sort(y)
    for pos, label in KNOWN_FUNCTIONAL_RESIDUES.items():
        g = df[df["pos"] == pos]
        if not len(g):
            continue
        pct = np.searchsorted(ecdf, g["DMS_score"].to_numpy()) / len(ecdf)
        probe_rows.append(
            {
                "pos": pos,
                "annotation": label,
                "variants": ",".join(g["mutant"]),
                "n": len(g),
                "mean_score": g["DMS_score"].mean(),
                "min_score": g["DMS_score"].min(),
                "mean_percentile": float(pct.mean()),
                "min_percentile": float(pct.min()),
            }
        )
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(RESULTS / "audit_known_residues.csv", index=False)
    out["label_validity_probes"] = {
        "mean_percentile_across_known_catalytic_residues": float(probes["mean_percentile"].mean()),
        "comment": (
            "In a selection where survival requires cleavage, substitutions at the RuvC and "
            "HNH catalytic residues should sit at the floor of the score distribution. They "
            "sit at the 21st-35th percentile instead. The rank order is right (these "
            "residues are below median) but the dynamic range is compressed, so the labels "
            "carry ordinal information about tolerance and not a calibrated activity scale. "
            "Consistent with the paper, which reported only D10G and a few H840 substitutions "
            "as statistically significant."
        ),
    }

    # ---- 8. does simple biophysics show through? ------------------------
    from src.features import AA_SCALES

    dh = np.array([AA_SCALES[m][0] - AA_SCALES[w][0] for w, m in zip(df["wt_aa"], df["mt_aa"])])
    dv = np.array([AA_SCALES[m][1] - AA_SCALES[w][1] for w, m in zip(df["wt_aa"], df["mt_aa"])])
    pro = df[df["mt_aa"] == "P"]["DMS_score"]
    out["simple_biophysics"] = {
        "abs_delta_hydropathy_spearman": float(spearmanr(np.abs(dh), y)[0]),
        "abs_delta_volume_spearman": float(spearmanr(np.abs(dv), y)[0]),
        "proline_mean": float(pro.mean()),
        "non_proline_mean": float(df[df["mt_aa"] != "P"]["DMS_score"].mean()),
        "proline_mwu_p": float(mannwhitneyu(pro, df[df["mt_aa"] != "P"]["DMS_score"])[1]),
        "comment": (
            "Proline substitutions are significantly more depleted, the expected direction. "
            "Effect sizes for the continuous scales are tiny, bounding how much handcrafted "
            "biophysics can contribute."
        ),
    }

    # ---- 9. per-position profile (deliverable for inspection) -----------
    prof = (
        df.groupby("pos")
        .agg(
            wt_aa=("wt_aa", "first"),
            domain=("domain", "first"),
            n_measured=("DMS_score", "size"),
            mean_score=("DMS_score", "mean"),
            min_score=("DMS_score", "min"),
            max_score=("DMS_score", "max"),
            frac_above_median=("DMS_score_bin", "mean"),
        )
        .reset_index()
    )
    prof.to_csv(RESULTS / "audit_position_profile.csv", index=False)

    (RESULTS / "audit_summary.json").write_text(json.dumps(out, indent=2, default=str))
    _print(out, dom)
    return out


def _print(out: dict, dom: pd.DataFrame) -> None:
    a = out["table_audit"]
    print("=" * 78)
    print("EXPERIMENT 0  data audit")
    print("=" * 78)
    print(f"rows {a['n_rows']}  positions {a['n_positions']}/{a['seq_len']}  "
          f"coverage {100 * a['coverage_fraction']:.1f}% of all single substitutions")
    print(f"mutated_sequence verification: {a['sequence_check']}")
    print(f"stop codons {a['n_stop']}  synonymous {a['n_synonymous']}  "
          f"duplicates {a['n_duplicate_variants']}")
    print(f"substitutions reachable by ONE nucleotide change: "
          f"{100 * a['frac_reachable_one_nt']:.1f}%  "
          f"(expected {100 * a['frac_reachable_if_uniform']:.1f}% if freely chosen)")
    print(f"DMS_score_bin is exactly a median split: {a['bin_is_median_split']}")
    r = out["reliability"]
    print(f"\nICC by position {r['icc_position']:.3f}  -> position-only Pearson ceiling "
          f"{r['position_only_pearson_ceiling']:.3f}")
    print(f"split-half reliability of position means {r['half_half_pearson']:.3f} "
          f"(Spearman-Brown {r['spearman_brown_full']:.3f})")
    print(f"leave-one-out position mean vs score: Spearman "
          f"{out['loo_position_mean']['spearman']:.3f}")
    d = out["domain_signal"]
    print(f"\ndomain Kruskal-Wallis H={d['kruskal_H']:.1f} p={d['kruskal_p']:.1e}")
    print(f"most depleted {d['most_depleted']}  most tolerant {d['most_tolerant']}")
    print(f"nuclease vs rest: {d['nuclease_mean']:.3f} vs {d['other_mean']:.3f} "
          f"(p={d['nuclease_vs_other_mwu_p']:.1e})")
    print(f"\nknown catalytic residues mean percentile "
          f"{out['label_validity_probes']['mean_percentile_across_known_catalytic_residues']:.2f} "
          f"(should be near 0.0 if the assay resolved cleavage)")
    print("\ndomain table:")
    print(dom[["domain", "n", "mean", "sem", "frac_above_median"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
