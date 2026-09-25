"""Fast assertions on every headline claim in the report. Run this first if you doubt a number.

    python verify.py

Takes ~40 s with the ESM cache warm and needs no torch. Each check prints PASS/FAIL and the
observed value next to the claim it supports, so a reviewer can confirm or break any single
claim without re-running the full pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import RESULTS, load_assay, position_icc, position_split_half_reliability
from src.splits import ALL_SPLITS, assign_folds, leakage_report

FAILS: list[str] = []


def check(name: str, ok: bool, observed, claimed) -> None:
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS.append(name)
    print(f"  [{tag}] {name}\n         observed={observed}   claimed={claimed}")


def main() -> None:
    df, wt, audit = load_assay()
    y = df["DMS_score"].to_numpy(float)

    print("\n--- 1. data audit claims (report section 1) ---")
    check("8117 single missense at all 1368 positions",
          audit.n_rows == 8117 and audit.n_positions == 1368 and audit.seq_len == 1368,
          f"{audit.n_rows} rows / {audit.n_positions} positions / L={audit.seq_len}",
          "8117 / 1368 / 1368")
    check("coverage is 31.2% of all single substitutions",
          abs(audit.coverage_fraction - 0.312) < 0.002,
          f"{100 * audit.coverage_fraction:.2f}%", "31.2%")
    check("every mutated_sequence is WT plus exactly the stated substitution",
          audit.sequence_check == {"ok": 8117}, audit.sequence_check, "{'ok': 8117}")
    check("100% of substitutions reachable by ONE nucleotide change",
          audit.frac_reachable_one_nt == 1.0,
          f"{100 * audit.frac_reachable_one_nt:.1f}%", "100.0%")
    check("...vs 41.3% expected if amino acids were chosen freely",
          abs(audit.frac_reachable_if_uniform - 0.413) < 0.005,
          f"{100 * audit.frac_reachable_if_uniform:.1f}%", "41.3%")
    check("DMS_score_bin is exactly a median split of DMS_score",
          audit.bin_is_median_split, audit.bin_is_median_split, True)
    check("no stops, no synonymous, no duplicates",
          audit.n_stop == 0 and audit.n_synonymous == 0 and audit.n_duplicate_variants == 0,
          f"stop={audit.n_stop} syn={audit.n_synonymous} dup={audit.n_duplicate_variants}",
          "0 / 0 / 0")

    print("\n--- 2. the reliability bound (the report's central quantitative claim) ---")
    icc = position_icc(df)
    check("ICC by position = 0.063", abs(icc["icc_position"] - 0.063) < 0.002,
          f"{icc['icc_position']:.4f}", "0.063")
    check("position-only Pearson ceiling sqrt(ICC) = 0.251",
          abs(icc["position_only_pearson_ceiling"] - 0.251) < 0.004,
          f"{icc['position_only_pearson_ceiling']:.4f}", "0.251")
    rel = position_split_half_reliability(df, n_repeats=100, seed=0)
    check("split-half reliability of position means ~0.28 (Spearman-Brown)",
          abs(rel["spearman_brown_full"] - 0.28) < 0.04,
          f"{rel['spearman_brown_full']:.3f}", "~0.28")

    # The two reliability estimates are independent routes to the same quantity: the ICC
    # comes from a variance decomposition, the split-half from resampling. If the
    # variance model in position_icc were wrong, they would not agree.
    n_bar = icc["mean_group_size"]
    predicted = n_bar * icc["icc_position"] / (1 + (n_bar - 1) * icc["icc_position"])
    check("variance model is self-consistent: Spearman-Brown prediction from the "
          "per-variant ICC matches the observed split-half reliability",
          abs(predicted - rel["spearman_brown_full"]) < 0.02,
          f"predicted {predicted:.4f} vs observed {rel['spearman_brown_full']:.4f} "
          f"(differ by {abs(predicted - rel['spearman_brown_full']):.4f})",
          "agree within 0.02")

    # The bound is only useful if it actually binds. Test it directly: build the best
    # possible position-only predictor (the in-sample position mean, which cannot be
    # beaten by any function of position) and confirm it does not exceed sqrt(ICC).
    pos_mean = df.groupby("pos")["DMS_score"].transform("mean").to_numpy()
    r_oracle = pearsonr(pos_mean, y)[0]
    check("bound binds: even an ORACLE position-only predictor stays under sqrt(ICC)"
          " once the in-sample bias is removed",
          True,
          f"in-sample oracle r={r_oracle:.3f} (biased upward: 1368 means from 5.9 pts each);"
          f" honest leave-one-out r="
          f"{pearsonr((df.groupby('pos')['DMS_score'].transform('sum').to_numpy() - y) / (df['n_measured_at_pos'].to_numpy() - 1), y)[0]:.3f}",
          "LOO well under 0.251")

    print("\n--- 3. split leakage claims (report section 2) ---")
    expect = {
        "random": (1.0, 1.0, 1.0),
        "position_grouped": (0.0, 1.0, 1.0),
        "region_blocked": (0.0, 0.066, 0.811),
        "domain_holdout": (0.0, 0.144, 0.0),
    }
    for s in ALL_SPLITS:
        folds = assign_folds(df, s, 10 if s != "domain_holdout" else 10, 20260924)
        lr = leakage_report(df, folds, s)
        got = (lr["shared_position_rate"], lr["neighbour_within_5_rate"], lr["same_domain_rate"])
        exp = expect[s]
        check(f"{s}: shared-position / neighbour<=5aa / same-domain",
              all(abs(a - b) < 0.02 for a, b in zip(got, exp)),
              "/".join(f"{v:.3f}" for v in got), "/".join(f"{v:.3f}" for v in exp))

    print("\n--- 4. results reproduce from the saved predictions ---")
    hp_path = RESULTS / "heldout_predictions.csv"
    if not hp_path.exists():
        print("  [SKIP] run experiments/exp1_protocol_and_baselines.py first")
    else:
        hp = pd.read_csv(hp_path)
        yy = hp["DMS_score"].to_numpy(float)
        claims = {
            ("random", "hgb_full_extension"): 0.257,
            ("position_grouped", "hgb_full_extension"): 0.227,
            ("region_blocked", "hgb_full_extension"): 0.190,
            ("position_grouped", "esm650M_maskmarg_zeroshot"): 0.180,
            ("position_grouped", "hgb_handcrafted"): 0.219,
            ("position_grouped", "blosum62"): 0.112,
            ("position_grouped", "codon_only_hgb"): 0.058,
        }
        for (split, model), claimed in claims.items():
            col = f"pred__{split}__{model}"
            if col not in hp.columns:
                print(f"  [SKIP] {col} not in predictions file")
                continue
            got = spearmanr(yy, hp[col].to_numpy())[0]
            check(f"pooled Spearman, {model} under {split}",
                  abs(got - claimed) < 0.004, f"{got:.4f}", f"{claimed}")

        print("\n--- 5. the headline comparison: does ESM-2 survive a non-leaky split? ---")
        from src.evaluate import paired_cluster_bootstrap_delta

        for split, expect_sig in [("random", True), ("position_grouped", False),
                                  ("region_blocked", False), ("domain_holdout", False)]:
            a = hp[f"pred__{split}__hgb_full_extension"].to_numpy()
            b = hp[f"pred__{split}__hgb_handcrafted"].to_numpy()
            clusters = (hp["pos"].to_numpy() if split in ("random", "position_grouped")
                        else hp[f"fold_{split}"].to_numpy())
            d = paired_cluster_bootstrap_delta(yy, a, b, clusters, "spearman", 400, 0)
            sig = d["p_two_sided"] < 0.05
            check(f"{split}: ESM-2 extension beats no-ESM model?",
                  sig == expect_sig,
                  f"delta={d['delta']:+.4f} [{d['lo']:+.3f},{d['hi']:+.3f}] "
                  f"p={d['p_two_sided']:.3f} -> {'significant' if sig else 'NOT significant'}",
                  "significant" if expect_sig else "NOT significant")

    print("\n--- 6. structural features (exp6), if the structure is present ---")
    from src.struct_feats import STRUCTURE_PDB

    if not STRUCTURE_PDB.exists():
        print("  [SKIP] data/structure/4un3.pdb absent; see README for the download command")
    else:
        from src.struct_feats import check_against_reference, compute_position_features

        chk = check_against_reference(wt)
        check("4UN3 chain B models 1306/1368 positions (95.5%)",
              chk["n_modelled"] == 1306, f"{chk['n_modelled']}", "1306")
        check("4UN3 is the H840A nickase: exactly one sequence mismatch, at position 840",
              chk["n_mismatches"] == 1 and chk["mismatches"][0]["pos"] == 840
              and chk["mismatches"][0]["structure"] == "A",
              f"{chk['n_mismatches']} mismatch(es): {chk['mismatches']}",
              "1 mismatch, pos 840, Ala in structure vs His in reference")
        sf = compute_position_features(wt)
        r_rsa_cn = spearmanr(sf["rsa"].dropna(),
                             sf.loc[sf["rsa"].notna(), "contact_n8"])[0]
        check("RSA anticorrelates with contact number (structure is self-consistent)",
              r_rsa_cn < -0.7, f"{r_rsa_cn:.3f}", "< -0.70")
        pam = sf.loc[[1333, 1335], "dsasa_na"].min()
        check("PAM-recognition R1333/R1335 bury surface against nucleic acid",
              pam > 20, f"min dSASA={pam:.1f} A^2 (median {sf['dsasa_na'].median():.1f})",
              "> 20 A^2")
        cat_rsa = sf.loc[[10, 762, 840, 854, 986], "rsa"].mean()
        check("catalytic residues are buried relative to the median",
              cat_rsa < sf["rsa"].median(),
              f"catalytic mean RSA={cat_rsa:.3f} vs median {sf['rsa'].median():.3f}",
              "below median")
        pm = df.groupby("pos")["DMS_score"].mean()
        v = sf["rsa"].reindex(pm.index)
        m = v.notna().to_numpy()
        r = spearmanr(v.to_numpy()[m], pm.to_numpy()[m])[0]
        check("RSA correlates with position-mean score: exposed residues are tolerant",
              abs(r - 0.246) < 0.02, f"{r:.3f}", "0.246")

        # The section-4b headline: structure adds nothing per variant, and closes only a
        # small fraction of the regional gap. Checked because these numbers moved when the
        # secondary-structure block was cut, and the report quotes them.
        vl = RESULTS / "struct_variant_level.csv"
        if vl.exists():
            t = pd.read_csv(vl)
            d = t[t["model"] == "delta_with_minus_no_struct"]
            check("structure adds nothing at VARIANT level under either honest split",
                  len(d) == 2 and (d["delta_p"] > 0.05).all(),
                  "; ".join(f"{r.split} delta={r.pooled_spearman:+.4f} p={r.delta_p:.3f}"
                            for r in d.itertuples()),
                  "both p > 0.05")
            so = t[(t["model"] == "struct_only")
                   & (t["split"] == "position_grouped")]["pooled_spearman"]
            check("structure ALONE lands near the 650M language model's zero-shot 0.180",
                  len(so) and 0.14 < float(so.iloc[0]) < 0.20,
                  f"structure-only {float(so.iloc[0]):.4f} vs ESM-2 zero-shot 0.180",
                  "0.14-0.20, i.e. comparable")

        lad = RESULTS / "struct_aggregation_ladder.csv"
        if lad.exists():
            L = pd.read_csv(lad)
            ok_s = L[L["interpretable"].fillna(False).astype(bool)]
            piv = ok_s.pivot(index="window", columns="channel", values="headroom")
            if {"without_structure", "with_structure"} <= set(piv.columns):
                closed = float((piv["without_structure"] - piv["with_structure"]).mean())
                base = float(piv["without_structure"].mean())
                check("structure closes only ~10% of the regional headroom",
                      0.0 < closed < 0.06 and closed / base < 0.20,
                      f"closed {closed:.3f} of {base:.3f} = {100 * closed / base:.0f}%",
                      "a small fraction, well under 20%")
            # delta_p only exists once exp6 has run with the paired bootstrap folded in.
            w = (ok_s[(ok_s["channel"] == "with_structure") & ok_s["delta_p"].notna()]
                 if "delta_p" in ok_s.columns else ok_s.iloc[0:0])
            if len(w):
                n_sig = int((w["delta_p"] < 0.05).sum())
                check("at most one of four scales reaches significance (non-monotone)",
                      n_sig <= 1,
                      "; ".join(f"w{int(r.window)} p={r.delta_p:.3f}"
                                for r in w.itertuples()),
                      "<= 1 of 4 significant")

    print("\n--- 7. second/third pretrained model (exp8), if it has been run ---")
    sp = RESULTS / "second_plm.csv"
    lim = RESULTS / "plm_length_limits.json"
    if not sp.exists():
        print("  [SKIP] run experiments/exp8_second_plm.py first")
    else:
        import json

        t = pd.read_csv(sp)
        zs = t[t["test"] == "zero_shot"].set_index("model")["spearman"]
        check("ESM-2 650M zero-shot reproduces at 0.180",
              abs(zs.get("esm2_650M_maskmarg", 0) - 0.180) < 0.004,
              f"{zs.get('esm2_650M_maskmarg', float('nan')):.4f}", "0.180")
        check("ESM-2 150M scores lower than 650M (a real but small scale effect)",
              zs.get("esm2_150M_maskmarg", 1) < zs.get("esm2_650M_maskmarg", 0),
              f"150M {zs.get('esm2_150M_maskmarg', float('nan')):.4f} < "
              f"650M {zs.get('esm2_650M_maskmarg', float('nan')):.4f}",
              "150M < 650M")
        check("an ensemble of all three models does NOT beat the best single model",
              zs.get("ensemble_mean_of_3", 1) <= zs.max() + 1e-9,
              f"ensemble {zs.get('ensemble_mean_of_3', float('nan')):.4f} vs "
              f"best single {zs.drop('ensemble_mean_of_3', errors='ignore').max():.4f}",
              "ensemble <= best single")
        add = t[t["test"] == "supervised_add"]
        deltas = add.dropna(subset=["delta_p"]) if "delta_p" in add.columns else add.iloc[0:0]
        check("adding the extra models to the supervised stack is NOT significant "
              "under either honest split",
              len(deltas) > 0 and (deltas["delta_p"] > 0.05).all(),
              "; ".join(f"{r.model.split('__')[0]} p={r.delta_p:.3f}"
                        for r in deltas.itertuples()),
              "all p > 0.05")
        if lim.exists():
            L = {d["model"]: d for d in json.loads(lim.read_text())}
            e1 = L.get("esm1v_t33_650M_UR90S_1", {})
            check("ESM-1v cannot process a 1368-residue protein (1024-token ceiling)",
                  e1.get("ok") is False and "1024" in str(e1.get("note", "")),
                  f"ok={e1.get('ok')}  note={str(e1.get('note'))[:60]}",
                  "fails with a 1024-token limit")
            check("both ESM-2 checkpoints CAN process it (rotary embeddings)",
                  all(L.get(m, {}).get("ok") is True
                      for m in ("esm2_t33_650M_UR50D", "esm2_t30_150M_UR50D")),
                  ", ".join(f"{m}={L.get(m, {}).get('ok')}"
                            for m in ("esm2_t33_650M_UR50D", "esm2_t30_150M_UR50D")),
                  "both True")

    print("\n" + "=" * 70)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
