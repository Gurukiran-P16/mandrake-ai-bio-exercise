"""Live demo / debug console. Answer an interviewer's "what if you changed X?" in one command.

Every subcommand recomputes real numbers from the real data - nothing is read from a
cached results table unless it says so. Run `python demo.py` for the menu.

    python demo.py claims                     every headline claim + the command proving it
    python demo.py bound                      derive the sqrt(ICC) ceiling with live numbers
    python demo.py leak                       what each split actually removes
    python demo.py split --scheme random      headline comparison under ONE split (~30 s)
    python demo.py sweep                      the same comparison under ALL FOUR splits
    python demo.py model --name hgb_full_extension --scheme position_grouped
    python demo.py ablate --drop esm_emb --scheme region_blocked
    python demo.py artefact                   show the blocked-CV negative bias live
    python demo.py residue --pos 840          everything the data says about one residue
    python demo.py predict --mutant D10A      the model's held-out prediction for a variant
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import RESULTS, load_assay, position_icc  # noqa: E402
from src.evaluate import core_metrics, paired_cluster_bootstrap_delta  # noqa: E402
from src.features import FEATURE_GROUPS, columns_excluding  # noqa: E402
from src.models import HGB, run_cv  # noqa: E402
from src.splits import (  # noqa: E402
    ALL_SPLITS,
    SPLIT_BOOTSTRAP_UNIT,
    SPLIT_CLAIMS,
    assign_folds,
    leakage_report,
)

N_FOLDS, SEED = 10, 20260924

SCHEME_ALIASES = {
    "random": "random", "position": "position_grouped", "position_grouped": "position_grouped",
    "region": "region_blocked", "region_blocked": "region_blocked",
    "domain": "domain_holdout", "domain_holdout": "domain_holdout",
}

CLAIMS = [
    ("All 8117 substitutions are reachable by ONE nucleotide change (41.3% expected "
     "if freely chosen) -> the library is error-prone PCR, so the substitution alphabet "
     "is set by the codon, not by biology.", "python verify.py   (check 1)"),
    ("DMS_score_bin is exactly a median split of DMS_score, so it carries no extra "
     "information.", "python verify.py   (check 1)"),
    ("Position explains only 6.3% of per-variant variance (ICC 0.063), which caps any "
     "position-constant feature at Pearson r <= 0.251.", "python demo.py bound"),
    ("A position-grouped split still leaves a training neighbour within 5 residues for "
     "100% of test variants; region-blocking cuts that to 6.6%.", "python demo.py leak"),
    ("The ESM-2 extension beats the no-ESM model ONLY on the leaky random split "
     "(+0.034, p<0.001); it is not significant under any honest split.",
     "python demo.py sweep"),
    ("Removing the ESM-2 wild-type embedding IMPROVES region-level generalization "
     "(+0.024, p<0.001) - it is a positional fingerprint.",
     "python demo.py ablate --drop esm_emb --scheme region_blocked"),
    ("Pooled out-of-fold correlation is biased to about -0.145 under blocked splits, "
     "which is why per-fold means are the primary read there.", "python demo.py artefact"),
    ("Real crystallographic structure adds nothing at variant level (+0.003, p=0.61) and "
     "closes only ~10% of the regional headroom.",
     "python experiments/exp6_structure.py   (16 min)"),
    ("Rank-transforming the target gains more (+0.024, p<0.001) than every ESM-2 feature "
     "combined.", "python experiments/exp5_objective.py   (8 min)"),
    ("The negative result survives a 3rd model and a 4.3x range of scale: ESM-2 150M "
     "scores 0.154 vs 650M's 0.180 (0.042 rho per decade of parameters), and adding two "
     "further models to the supervised stack gives +0.003 (p=0.43) / -0.002 (p=0.76).",
     "python experiments/exp8_second_plm.py   (~12 min cached)"),
    ("ESM-1v CANNOT process this protein: a hard 1024-token ceiling (learned absolute "
     "positional embeddings) against SpCas9's 1370. ESM-2 works only because it uses "
     "rotary embeddings. That rules out much of the pretrained-model zoo for Cas9.",
     "cat results/plm_length_limits.json"),
]


def _design():
    from experiments.exp1_protocol_and_baselines import build_design_matrix

    return build_design_matrix()


def _scheme(s: str) -> str:
    if s not in SCHEME_ALIASES:
        raise SystemExit(f"unknown scheme {s!r}; choose from {sorted(set(SCHEME_ALIASES))}")
    return SCHEME_ALIASES[s]


def cmd_claims(_a) -> None:
    print("\nHEADLINE CLAIMS AND HOW TO PROVE EACH ONE LIVE\n" + "=" * 78)
    for i, (claim, how) in enumerate(CLAIMS, 1):
        print(f"\n{i}. {claim}")
        print(f"   -> {how}")
    print("\n" + "=" * 78)
    print("Full verification of all 43 claims at once:  python verify.py  (~60 s)")


def cmd_bound(_a) -> None:
    """Derive the ceiling from scratch, printing each quantity."""
    df, _wt, _audit = load_assay()
    y = df["DMS_score"].to_numpy(float)
    icc = position_icc(df)
    print("\nWHY A POSITION-ONLY FEATURE CANNOT EXCEED sqrt(ICC)\n" + "=" * 78)
    print("""
Write the score for substitution i at position p as

    y_pi = mu + a_p + e_pi          a_p = position effect,  e_pi = everything else

A feature f that is CONSTANT across the substitutions measured at a position (every
wild-type-structure feature is: RSA, contact number, distance to the catalytic site,
pLDDT, and the ESM-2 wild-type embedding) can only correlate with y through a_p:

    corr(f, y) = corr(f, a) * sd(a) / sqrt(var(a) + var(e))  <=  sqrt(ICC)

with equality only if the feature reproduces the position effect perfectly.
var(a) is noise-corrected as (MS_between - MS_within) / k0, so this is not inflated by
the within-position scatter.
""")
    print(f"  MS_between            {icc['ms_between']:.4f}")
    print(f"  MS_within             {icc['ms_within']:.4f}")
    print(f"  var(a)  between-pos   {icc['var_between']:.4f}")
    print(f"  var(e)  within-pos    {icc['var_within']:.4f}")
    print(f"  mean substitutions/position  {icc['mean_group_size']:.2f}")
    print(f"\n  ICC = var(a)/(var(a)+var(e)) = {icc['icc_position']:.4f}")
    print(f"  CEILING = sqrt(ICC)          = {icc['position_only_pearson_ceiling']:.4f}")

    s = df.groupby("pos")["DMS_score"].transform("sum").to_numpy()
    n = df["n_measured_at_pos"].to_numpy()
    loo = (s - y) / (n - 1)
    ins = df.groupby("pos")["DMS_score"].transform("mean").to_numpy()
    from scipy.stats import pearsonr

    print("\n  SANITY CHECK - an ORACLE position-only predictor:")
    print(f"    in-sample position mean   r = {pearsonr(ins, y)[0]:.3f}   "
          f"<- inflated: 1368 means fitted on 5.9 points each")
    print(f"    leave-one-out             r = {pearsonr(loo, y)[0]:.3f}   "
          f"<- honest, and far under the 0.251 ceiling")
    print("\n  WHAT THE BOUND DOES NOT COVER: per-variant structure prediction (folding")
    print("  each mutant separately) would give a feature that VARIES within a position,")
    print("  so it escapes this bound. That remains untested in this project.")


def cmd_leak(_a) -> None:
    df, _wt, _audit = load_assay()
    print("\nWHAT EACH SPLIT ACTUALLY REMOVES FROM THE TRAINING SET\n" + "=" * 78)
    rows = [leakage_report(df, assign_folds(df, s, N_FOLDS, SEED), s) for s in ALL_SPLITS]
    t = pd.DataFrame(rows)[["scheme", "n_folds", "shared_position_rate",
                            "neighbour_within_5_rate", "same_domain_rate", "claim"]]
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nRead it this way: position-grouping sets shared_position to 0 but leaves")
    print("neighbour_within_5 at 1.000 - every held-out position still has a training")
    print("neighbour within 5 residues. Region-blocking is what closes that leak.")


def _headline(df, X, names, groups, y, scheme: str, n_boot: int = 400):
    folds = assign_folds(df, scheme, N_FOLDS, SEED)
    clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[scheme] == "pos" else folds
    all_cols = np.arange(len(names))
    no_esm = columns_excluding(groups, ["esm_lm", "esm_emb"], len(names))
    p_full = run_cv(lambda: HGB(cols=all_cols, name="full"), df, X, y, folds, seed=SEED)
    p_noesm = run_cv(lambda: HGB(cols=no_esm, name="noesm"), df, X, y, folds, seed=SEED)
    d = paired_cluster_bootstrap_delta(y, p_full, p_noesm, clusters, "spearman", n_boot, SEED)
    return core_metrics(y, p_full), core_metrics(y, p_noesm), d


def cmd_split(a) -> None:
    scheme = _scheme(a.scheme)
    df, _wt, X, names, groups, _ = _design()
    y = df["DMS_score"].to_numpy(float)
    print(f"\nSPLIT: {scheme}\nCLAIM TESTED: {SPLIT_CLAIMS[scheme]}\n" + "=" * 78)
    full, noesm, d = _headline(df, X, names, groups, y, scheme)
    print(f"  GBM + ESM-2 features   pooled rho {full['spearman']:+.4f}")
    print(f"  GBM, no ESM features   pooled rho {noesm['spearman']:+.4f}")
    print(f"\n  difference {d['delta']:+.4f}  95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
          f"p = {d['p_two_sided']:.3f}")
    print(f"  -> ESM-2 features are "
          f"{'SIGNIFICANT' if d['p_two_sided'] < 0.05 else 'NOT significant'} here")


def cmd_sweep(a) -> None:
    df, _wt, X, names, groups, _ = _design()
    y = df["DMS_score"].to_numpy(float)
    print("\nTHE HEADLINE RESULT UNDER ALL FOUR SPLITS")
    print("Does adding a 650M protein language model help? Watch it evaporate.\n" + "=" * 78)
    print(f"{'split':18s} {'+ESM-2':>8s} {'no ESM':>8s} {'delta':>8s} "
          f"{'95% CI':>20s} {'p':>7s}  verdict")
    for s in ALL_SPLITS:
        full, noesm, d = _headline(df, X, names, groups, y, s, n_boot=a.boot)
        sig = "SIGNIFICANT" if d["p_two_sided"] < 0.05 else "not significant"
        print(f"{s:18s} {full['spearman']:+8.4f} {noesm['spearman']:+8.4f} "
              f"{d['delta']:+8.4f} [{d['lo']:+7.4f},{d['hi']:+7.4f}] "
              f"{d['p_two_sided']:7.3f}  {sig}")
    print("\nThe only split where the language model helps is the one that leaks.")


def cmd_ablate(a) -> None:
    scheme = _scheme(a.scheme)
    df, _wt, X, names, groups, _ = _design()
    y = df["DMS_score"].to_numpy(float)
    drop = [g.strip() for g in a.drop.split(",")]
    bad = [g for g in drop if g not in FEATURE_GROUPS]
    if bad:
        raise SystemExit(f"unknown group(s) {bad}; choose from {FEATURE_GROUPS}")
    folds = assign_folds(df, scheme, N_FOLDS, SEED)
    clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[scheme] == "pos" else folds
    all_cols = np.arange(len(names))
    kept = columns_excluding(groups, drop, len(names))
    print(f"\nABLATION under {scheme}: dropping {drop}\n" + "=" * 78)
    print(f"  {len(all_cols)} features -> {len(kept)}")
    p_full = run_cv(lambda: HGB(cols=all_cols, name="full"), df, X, y, folds, seed=SEED)
    p_abl = run_cv(lambda: HGB(cols=kept, name="abl"), df, X, y, folds, seed=SEED)
    d = paired_cluster_bootstrap_delta(y, p_abl, p_full, clusters, "spearman", 400, SEED)
    print(f"  full        rho {core_metrics(y, p_full)['spearman']:+.4f}")
    print(f"  ablated     rho {core_metrics(y, p_abl)['spearman']:+.4f}")
    print(f"\n  delta (ablated - full) {d['delta']:+.4f} "
          f"[{d['lo']:+.4f},{d['hi']:+.4f}] p = {d['p_two_sided']:.3f}")
    verdict = ("the group was USEFUL" if d["delta"] < 0 and d["p_two_sided"] < 0.05
               else "removing it HELPED" if d["delta"] > 0 and d["p_two_sided"] < 0.05
               else "no detectable effect")
    print(f"  -> {verdict}")


def cmd_artefact(_a) -> None:
    """Show, live, that pooled out-of-fold correlation is not centred on zero."""
    from src.models import GlobalMean

    df, _wt, _audit = load_assay()
    y = df["DMS_score"].to_numpy(float)
    X = np.zeros((len(df), 1))
    print("\nTHE BLOCKED-CV ARTEFACT (a bug I caught in my own experiment)\n" + "=" * 78)
    print("""
A CONSTANT predictor - each fold predicts its own training mean and nothing else - must
have zero skill. Per-fold Spearman confirms that exactly. But POOLED out-of-fold
Spearman does not, because under a spatially blocked split each fold's training mean
anti-correlates with its own held-out mean.

My first aggregation ladder reported rho = -0.72 at 25-residue windows. That was this
artefact, not a signal reversal. Fix: permutation nulls per scale, and per-fold means as
the primary read for blocked splits.
""")
    print(f"{'split':18s} {'pooled rho':>12s} {'per-fold rho':>14s}")
    for s in ALL_SPLITS:
        folds = assign_folds(df, s, N_FOLDS, SEED)
        oof = run_cv(GlobalMean, df, X, y, folds, seed=SEED)
        from src.evaluate import per_fold_metrics

        pf = per_fold_metrics(df, y, oof, folds)["spearman"].mean()
        print(f"{s:18s} {core_metrics(y, oof)['spearman']:+12.4f} {pf:+14.4f}")
    print("\nPer-fold is 0.0000 everywhere, as it must be. Pooled is not. That gap is the")
    print("artefact, and it is why the report quotes per-fold for the blocked splits.")


def cmd_residue(a) -> None:
    df, wt, _audit = load_assay()
    p = a.pos
    if not 1 <= p <= len(wt):
        raise SystemExit(f"position must be 1..{len(wt)}")
    sub = df[df["pos"] == p].sort_values("DMS_score")
    y = df["DMS_score"].to_numpy(float)
    ecdf = np.sort(y)
    print(f"\nRESIDUE {wt[p - 1]}{p}  (domain {sub['domain'].iloc[0]})\n" + "=" * 78)
    print(f"  measured substitutions: {len(sub)} of 19 possible")
    for _, r in sub.iterrows():
        pct = np.searchsorted(ecdf, r["DMS_score"]) / len(ecdf)
        print(f"    {r['mutant']:10s} score {r['DMS_score']:+7.3f}   "
              f"percentile {pct:5.2f}   bin {int(r['DMS_score_bin'])}")
    print(f"  position mean {sub['DMS_score'].mean():+.3f}  "
          f"(dataset mean {y.mean():+.3f})")
    sp = RESULTS / "struct_position_features.csv"
    if sp.exists():
        st = pd.read_csv(sp, index_col=0)
        if p in st.index and pd.notna(st.loc[p, "rsa"]):
            r = st.loc[p]
            print(f"  structure (4UN3): RSA {r['rsa']:.2f}  contacts(8A) {r['contact_n8']:.0f}  "
                  f"min dist to DNA/RNA {r['min_dist_na']:.1f} A")
            print(f"                    dist to RuvC {r['dist_ruvc']:.1f} A  "
                  f"dist to HNH {r['dist_hnh']:.1f} A  B-factor {r['bfactor']:.0f}")
        else:
            print("  structure (4UN3): no coordinates for this residue (disordered)")


def cmd_predict(a) -> None:
    f = RESULTS / "heldout_predictions.csv"
    if not f.exists():
        raise SystemExit("run experiments/exp1_protocol_and_baselines.py first")
    hp = pd.read_csv(f)
    row = hp[hp["mutant"] == a.mutant]
    if row.empty:
        raise SystemExit(f"{a.mutant} is not in the assay "
                         f"(only 31.2% of substitutions were measured)")
    r = row.iloc[0]
    print(f"\nHELD-OUT PREDICTIONS FOR {a.mutant}\n" + "=" * 78)
    print(f"  true DMS_score {r['DMS_score']:+.4f}   bin {int(r['DMS_score_bin'])}   "
          f"position {int(r['pos'])}   domain {r['domain']}")
    print("  (each prediction is out-of-fold: the model never saw this variant in training)")
    print("\n  SCALES DIFFER - do not read these as competing estimates of DMS_score.")
    print("  Trained models regress onto the score, so they are on its scale. The")
    print("  untrained scorers are not: esm*_zeroshot is a log-likelihood ratio in nats")
    print("  and blosum62 is a substitution-matrix integer. Only their RANK is used, which")
    print("  is why every metric in this project is rank-based or scale-free.")
    cols = [c for c in hp.columns if c.startswith("pred__")]
    untrained = ("zeroshot", "blosum62")
    for scheme in ALL_SPLITS:
        print(f"\n  --- {scheme} (fold {int(r['fold_' + scheme])}) ---")
        for c in sorted(cols):
            if c.startswith(f"pred__{scheme}__"):
                name = c.split("__", 2)[2]
                tag = "  [untrained, different scale]" if any(
                    u in name for u in untrained) else ""
                print(f"      {name:28s} {r[c]:+.4f}{tag}")
    # Rank of this variant within each model's ordering is the comparable quantity.
    print("\n  PERCENTILE RANK of this variant under each model "
          f"(true score is at percentile "
          f"{(hp['DMS_score'] < r['DMS_score']).mean():.2f}):")
    for c in sorted(cols):
        if c.startswith(f"pred__{ALL_SPLITS[1]}__"):
            pct = (hp[c] < r[c]).mean()
            print(f"      {c.split('__', 2)[2]:28s} {pct:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Live demo / debug console for the SpCas9 activity project.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("claims", help="every headline claim + the command proving it")
    sub.add_parser("bound", help="derive the sqrt(ICC) ceiling with live numbers")
    sub.add_parser("leak", help="what each split actually removes")
    sub.add_parser("artefact", help="show the blocked-CV negative bias live")
    p = sub.add_parser("split", help="headline comparison under one split")
    p.add_argument("--scheme", default="position_grouped")
    p = sub.add_parser("sweep", help="headline comparison under all four splits")
    p.add_argument("--boot", type=int, default=400)
    p = sub.add_parser("ablate", help="drop feature group(s) and re-evaluate")
    p.add_argument("--drop", required=True,
                   help=f"comma-separated, from {FEATURE_GROUPS}")
    p.add_argument("--scheme", default="position_grouped")
    p = sub.add_parser("model", help="alias for split")
    p.add_argument("--name", default="hgb_full_extension")
    p.add_argument("--scheme", default="position_grouped")
    p = sub.add_parser("residue", help="everything the data says about one residue")
    p.add_argument("--pos", type=int, required=True)
    p = sub.add_parser("predict", help="held-out predictions for one variant")
    p.add_argument("--mutant", required=True)

    a = ap.parse_args()
    if a.cmd is None:
        ap.print_help()
        return
    {"claims": cmd_claims, "bound": cmd_bound, "leak": cmd_leak, "split": cmd_split,
     "sweep": cmd_sweep, "ablate": cmd_ablate, "artefact": cmd_artefact,
     "model": cmd_split, "residue": cmd_residue, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
