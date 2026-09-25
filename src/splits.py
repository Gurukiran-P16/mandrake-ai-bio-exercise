"""Cross-validation schemes, each tied to an explicit generalization claim.

The dataset has a nested structure that makes the choice of split the single most
consequential methodological decision in this exercise:

  * 8117 variants sit at 1368 positions, 4-7 variants per position (mean 5.93).
    Variants at the same position share a positional effect, so a variant-level
    random split puts siblings of every test variant in the training set.
  * Position-level mean scores are autocorrelated along the sequence
    (Pearson r ~ 0.14 at lag 1, ~0.13 at lag 5, ~0.00 by lag 100), and the domain
    annotation is contiguous. So even a position-grouped split leaves a held-out
    position's sequence neighbours - and its whole domain - in the training set.

Each scheme below answers a different question. Reporting only the first would
overstate what the model can do.

  SPLIT_RANDOM   "given other substitutions measured at this position, predict a new
                 substitution at the same position"        (interpolation)
  SPLIT_POSITION "predict substitutions at positions with no measurements at all"
                                                            (new residues)
  SPLIT_REGION   "predict substitutions in a contiguous stretch of the protein that
                 was never measured"                        (new regions)
  SPLIT_DOMAIN   leave-one-domain-out; the strictest and most interpretable version
                 of the region claim, and the resolution at which the original study
                 actually drew its conclusions.

The difference between SPLIT_RANDOM and SPLIT_POSITION is a direct measurement of how
much apparent performance is positional memorization rather than transferable signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DOMAINS

SPLIT_RANDOM = "random"
SPLIT_POSITION = "position_grouped"
SPLIT_REGION = "region_blocked"
SPLIT_DOMAIN = "domain_holdout"

ALL_SPLITS = [SPLIT_RANDOM, SPLIT_POSITION, SPLIT_REGION, SPLIT_DOMAIN]

# The claim each split licenses, carried through to the results tables.
SPLIT_CLAIMS = {
    SPLIT_RANDOM: "new substitution at an already-measured position (interpolation)",
    SPLIT_POSITION: "new substitution at a position with no measurements",
    SPLIT_REGION: "new substitution in a contiguous unmeasured region (~137 aa)",
    SPLIT_DOMAIN: "new substitution in an entirely unmeasured structural domain",
}

# Grouping unit that the cluster bootstrap must resample to respect dependence.
SPLIT_BOOTSTRAP_UNIT = {
    SPLIT_RANDOM: "pos",
    SPLIT_POSITION: "pos",
    SPLIT_REGION: "fold",
    SPLIT_DOMAIN: "fold",
}


def _random_folds(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.tile(np.arange(k), int(np.ceil(n / k)))[:n]
    rng.shuffle(folds)
    return folds


def assign_folds(
    df: pd.DataFrame, scheme: str, n_folds: int = 10, seed: int = 20260924
) -> np.ndarray:
    """Return an integer fold label per row of `df` for the requested scheme."""
    if scheme == SPLIT_RANDOM:
        return _random_folds(len(df), n_folds, seed)

    if scheme == SPLIT_POSITION:
        positions = np.sort(df["pos"].unique())
        pf = dict(zip(positions, _random_folds(len(positions), n_folds, seed)))
        return df["pos"].map(pf).to_numpy()

    if scheme == SPLIT_REGION:
        # Contiguous equal-width blocks of sequence, so each fold holds out a whole
        # stretch and no test position has a training neighbour within the block.
        lo, hi = int(df["pos"].min()), int(df["pos"].max())
        edges = np.linspace(lo - 0.5, hi + 0.5, n_folds + 1)
        return (np.digitize(df["pos"].to_numpy(), edges) - 1).clip(0, n_folds - 1)

    if scheme == SPLIT_DOMAIN:
        order = {name: i for i, (name, _, _) in enumerate(DOMAINS)}
        return df["domain"].map(order).fillna(len(DOMAINS)).astype(int).to_numpy()

    raise ValueError(f"unknown split scheme: {scheme}")


def build_split_table(df: pd.DataFrame, n_folds: int = 10, seed: int = 20260924) -> pd.DataFrame:
    """Fold assignment for every variant under every scheme (a required deliverable)."""
    out = df[["mutant", "pos", "wt_aa", "mt_aa", "domain", "DMS_score", "DMS_score_bin"]].copy()
    for scheme in ALL_SPLITS:
        out[f"fold_{scheme}"] = assign_folds(df, scheme, n_folds, seed)
    return out


def iter_folds(folds: np.ndarray):
    """Yield (fold_id, train_index, test_index) for each fold present in `folds`."""
    for f in np.unique(folds):
        test = np.flatnonzero(folds == f)
        train = np.flatnonzero(folds != f)
        yield int(f), train, test


def leakage_report(df: pd.DataFrame, folds: np.ndarray, scheme: str) -> dict:
    """Quantify what a scheme actually removes from the training set.

    `shared_position_rate` - fraction of test variants whose position also appears in
        training. Non-zero only for the random split, by construction.
    `neighbour_within_5_rate` - fraction of test variants with a training variant within
        +/-5 residues. This is the leak that the position-grouped split does *not* close
        and that the region-blocked split does.
    `same_domain_rate` - fraction of test variants whose domain also appears in training.
    """
    rows = []
    for f, tr, te in iter_folds(folds):
        tr_pos = set(df["pos"].to_numpy()[tr])
        tr_dom = set(df["domain"].to_numpy()[tr])
        te_pos = df["pos"].to_numpy()[te]
        te_dom = df["domain"].to_numpy()[te]
        tr_sorted = np.sort(np.fromiter(tr_pos, dtype=int))
        near = np.array(
            [
                np.any(np.abs(tr_sorted - p) <= 5)
                for p in te_pos
            ]
        )
        rows.append(
            {
                "fold": f,
                "n_test": len(te),
                "shared_position_rate": float(np.mean([p in tr_pos for p in te_pos])),
                "neighbour_within_5_rate": float(near.mean()),
                "same_domain_rate": float(np.mean([d in tr_dom for d in te_dom])),
            }
        )
    t = pd.DataFrame(rows)
    return {
        "scheme": scheme,
        "claim": SPLIT_CLAIMS[scheme],
        "n_folds": len(t),
        "n_test_min": int(t["n_test"].min()),
        "n_test_max": int(t["n_test"].max()),
        "shared_position_rate": float(t["shared_position_rate"].mean()),
        "neighbour_within_5_rate": float(t["neighbour_within_5_rate"].mean()),
        "same_domain_rate": float(t["same_domain_rate"].mean()),
    }
