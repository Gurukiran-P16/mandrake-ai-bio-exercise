"""Figures for the report. Reads only the CSVs written by experiments 0-3."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import RESULTS  # noqa: E402

PLOT_ORDER = [
    "global_mean",
    "codon_only_hgb",
    "blosum62",
    "domain_mean",
    "position_mean",
    "esm650M_maskmarg_zeroshot",
    "hgb_handcrafted",
    "hgb_full_extension",
]
NICE = {
    "global_mean": "global mean (null)",
    "codon_only_hgb": "ep-PCR codon features only",
    "blosum62": "BLOSUM62 (untrained)",
    "domain_mean": "domain mean",
    "position_mean": "position mean",
    "esm650M_maskmarg_zeroshot": "ESM-2 650M zero-shot LLR",
    "hgb_handcrafted": "GBM, no ESM features",
    "hgb_full_extension": "GBM + ESM-2 (extension)",
}
SPLITS = ["random", "position_grouped", "region_blocked", "domain_holdout"]
SPLIT_NICE = {
    "random": "random\n(leaky)",
    "position_grouped": "new\nposition",
    "region_blocked": "new\nregion",
    "domain_holdout": "new\ndomain",
}


def fig_main() -> None:
    res = pd.read_csv(RESULTS / "main_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), width_ratios=[1.45, 1])

    # --- panel A: per-fold Spearman, models x splits ---------------------
    ax = axes[0]
    models = [m for m in PLOT_ORDER if m in set(res["model"])]
    x = np.arange(len(models))
    w = 0.2
    colors = ["#b8b8b8", "#7fa8c9", "#3d6f96", "#16324a"]
    for i, sp in enumerate(SPLITS):
        sub = res[res["split"] == sp].set_index("model")
        vals = [sub.loc[m, "per_fold_spearman_mean"] if m in sub.index else np.nan
                for m in models]
        errs = [sub.loc[m, "per_fold_spearman_sd"] if m in sub.index else np.nan
                for m in models]
        ax.bar(
            x + (i - 1.5) * w, vals, w, yerr=errs, capsize=1.6,
            label=SPLIT_NICE[sp].replace("\n", " "), color=colors[i],
            error_kw={"lw": 0.7, "alpha": 0.6},
        )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE.get(m, m) for m in models], rotation=32,
                       ha="right", fontsize=8)
    ax.set_ylabel("Spearman $\\rho$ (mean over folds)")
    ax.set_title("A  Held-out performance by generalization claim", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, title="held-out unit", title_fontsize=7.5, frameon=False,
              loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    # --- panel B: aggregation ladder vs reliability ceiling --------------
    ax = axes[1]
    agg = pd.read_csv(RESULTS / "challenge_aggregation.csv")
    ok = agg[agg["blocked_cv_interpretable"].fillna(False).astype(bool)] \
        if "blocked_cv_interpretable" in agg.columns else agg
    lab = ok["mean_labels_per_unit"].to_numpy()
    ax.plot(lab, ok["pearson"], "o-", color="#16324a", lw=1.6, ms=5,
            label="achieved (position-level\nfeatures, blocked CV)")
    ax.plot(lab, ok["ceiling_from_reliability"], "s--", color="#c0504d", lw=1.4, ms=4.5,
            label="ceiling $\\sqrt{\\mathrm{reliability}}$")
    ax.fill_between(lab, ok["pearson"], ok["ceiling_from_reliability"],
                    color="#c0504d", alpha=0.12)
    var = agg[agg["scale"] == "variant"]
    if len(var):
        ax.plot([1], var["pearson"].to_numpy()[:1], "D", color="#16324a", ms=5)
        ax.annotate("per-variant", (1, float(var["pearson"].iloc[0])),
                    textcoords="offset points", xytext=(6, -11), fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("measurements averaged per predicted unit")
    ax.set_ylabel("Pearson $r$")
    ax.set_title("B  Label averaging raises the ceiling and\n"
                 "     leaves persistent unexplained headroom",
                 fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(RESULTS / "fig_main.png", dpi=200, bbox_inches="tight")
    print("wrote", RESULTS / "fig_main.png")


def fig_ablation() -> None:
    ab = pd.read_csv(RESULTS / "ablation.csv")
    logo = ab[ab["variant"] == "leave_one_group_out"]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    groups = ["subst", "position", "esm_lm", "esm_emb", "codon"]
    gn = {"subst": "substitution\nbiophysics", "position": "position /\ndomain",
          "esm_lm": "ESM-2 LM\nscores", "esm_emb": "ESM-2 WT\nembedding",
          "codon": "ep-PCR codon\nfeatures"}
    x = np.arange(len(groups))
    for i, sp in enumerate(["position_grouped", "region_blocked"]):
        sub = logo[logo["split"] == sp].set_index("group")
        d = [sub.loc[g, "delta_vs_full"] if g in sub.index else np.nan for g in groups]
        lo = [sub.loc[g, "delta_lo"] if g in sub.index else np.nan for g in groups]
        hi = [sub.loc[g, "delta_hi"] if g in sub.index else np.nan for g in groups]
        err = np.array([np.array(d) - np.array(lo), np.array(hi) - np.array(d)])
        ax.bar(x + (i - 0.5) * 0.36, d, 0.36, yerr=np.abs(err), capsize=2.2,
               label=sp.replace("_", " "), color=["#3d6f96", "#16324a"][i],
               error_kw={"lw": 0.8})
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([gn[g] for g in groups], fontsize=7.5)
    ax.set_ylabel("$\\Delta\\rho$ when group removed")
    ax.set_title("Feature-group ablation (negative = group was useful)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_ablation.png", dpi=200, bbox_inches="tight")
    print("wrote", RESULTS / "fig_ablation.png")


def fig_structure() -> None:
    """Does static structure close the regional headroom? (exp6)"""
    path = RESULTS / "struct_aggregation_ladder.csv"
    if not path.exists():
        print("skipping fig_structure: run experiments/exp6_structure.py first")
        return
    lad = pd.read_csv(path)
    ok = lad[lad["interpretable"].fillna(False).astype(bool)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

    style = {
        "without_structure": ("#16324a", "o-", "sequence only"),
        "with_structure": ("#2e8b57", "^-", "sequence + 4UN3 structure"),
        "structure_only": ("#c9852b", "v--", "structure only"),
    }
    ax = axes[0]
    for ch, (c, m, lab) in style.items():
        s = ok[ok["channel"] == ch]
        if len(s):
            ax.plot(s["mean_labels_per_unit"], s["pearson"], m, color=c, lw=1.6, ms=5,
                    label=lab)
    ref = ok[ok["channel"] == "without_structure"]
    if len(ref):
        ax.plot(ref["mean_labels_per_unit"], ref["ceiling"], "s--", color="#c0504d",
                lw=1.4, ms=4.5, label="ceiling $\\sqrt{\\mathrm{reliability}}$")
    ax.set_xscale("log")
    if len(ref):
        ticks = ref["mean_labels_per_unit"].to_numpy()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.0f}" for t in ticks], fontsize=8)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("measurements averaged per predicted unit")
    ax.set_ylabel("Pearson $r$")
    ax.set_title("A  Achieved vs ceiling, with and without structure",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1)

    ax = axes[1]
    piv = ok.pivot(index="window", columns="channel", values="headroom")
    if {"without_structure", "with_structure"} <= set(piv.columns):
        x = np.arange(len(piv))
        ax.bar(x - 0.2, piv["without_structure"], 0.4, color="#16324a",
               label="sequence only")
        ax.bar(x + 0.2, piv["with_structure"], 0.4, color="#2e8b57",
               label="+ structure")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(w)} aa" for w in piv.index], fontsize=8)
        ax.set_xlabel("aggregation window")
        ax.set_ylabel("unexplained headroom (ceiling $-$ $r$)")
        ax.set_title("B  Headroom remaining after adding structure",
                     fontsize=9.5, loc="left")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(RESULTS / "fig_structure.png", dpi=200, bbox_inches="tight")
    print("wrote", RESULTS / "fig_structure.png")


def main() -> None:
    fig_main()
    fig_ablation()
    fig_structure()


if __name__ == "__main__":
    main()
