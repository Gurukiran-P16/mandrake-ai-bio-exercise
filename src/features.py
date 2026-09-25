"""Feature construction, organised into ablatable groups.

Groups, and what each one is for:

  subst      substitution identity and physicochemical change. Varies across the
             variants at a position, so it is not bounded by the positional ICC.
  codon      properties of the error-prone-PCR accessibility of the substitution.
             Included deliberately as a *confound probe*: the library was built by
             ep-PCR at 0.18% nucleotide mutation rate, so which amino acids appear at
             a position is set by the wild-type codon, not by biology. If codon
             features predict the score, part of any model's apparent skill is
             tracking library construction rather than protein function.
  position   sequence position and domain annotation. Constant within a position.
  esm_lm     ESM-2 650M language-model scores at the mutated position. The LLR varies
             across substitutions; entropy and wild-type log-probability do not.
  esm_emb    PCA of the ESM-2 650M final-layer wild-type representation at the
             mutated position. Constant within a position - see the ICC bound.

The position/esm_emb groups are exactly the groups subject to the sqrt(ICC) <= 0.25
ceiling derived in data.position_icc; subst/codon/esm_lm(LLR) are not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AAS, DOMAINS, one_nt_neighbourhood, genetic_code

FEATURE_GROUPS = ["subst", "codon", "position", "esm_lm", "esm_emb", "struct", "struct_win"]

# --- amino-acid scales -----------------------------------------------------
# Kyte-Doolittle hydropathy; residue volume (A^3, Zamyatnin); formal charge at pH 7;
# Grantham polarity; molecular weight; Pace-Scholtz helix propensity (kcal/mol,
# lower = more helix-favourable); Levitt beta-sheet propensity.
AA_SCALES = {
    #      hyd    vol  chg  pol     mw   helix  sheet
    "A": (1.8, 88.6, 0, 8.1, 89.1, 0.0, 0.83),
    "R": (-4.5, 173.4, 1, 10.5, 174.2, 0.21, 0.93),
    "N": (-3.5, 114.1, 0, 11.6, 132.1, 0.65, 0.89),
    "D": (-3.5, 111.1, -1, 13.0, 133.1, 0.69, 0.54),
    "C": (2.5, 108.5, 0, 5.5, 121.2, 0.68, 1.19),
    "Q": (-3.5, 143.8, 0, 10.5, 146.2, 0.39, 1.10),
    "E": (-3.5, 138.4, -1, 12.3, 147.1, 0.40, 0.37),
    "G": (-0.4, 60.1, 0, 9.0, 75.1, 1.00, 0.75),
    "H": (-3.2, 153.2, 0, 10.4, 155.2, 0.61, 0.87),
    "I": (4.5, 166.7, 0, 5.2, 131.2, 0.41, 1.60),
    "L": (3.8, 166.7, 0, 4.9, 131.2, 0.21, 1.30),
    "K": (-3.9, 168.6, 1, 11.3, 146.2, 0.26, 0.74),
    "M": (1.9, 162.9, 0, 5.7, 149.2, 0.24, 1.05),
    "F": (2.8, 189.9, 0, 5.2, 165.2, 0.54, 1.38),
    "P": (-1.6, 112.7, 0, 8.0, 115.1, 3.16, 0.55),
    "S": (-0.8, 89.0, 0, 9.2, 105.1, 0.50, 0.75),
    "T": (-0.7, 116.1, 0, 8.6, 119.1, 0.66, 1.19),
    "W": (-0.9, 227.8, 0, 5.4, 204.2, 0.49, 1.37),
    "Y": (-1.3, 193.6, 0, 6.2, 181.2, 0.53, 1.47),
    "V": (4.2, 140.0, 0, 5.9, 117.1, 0.61, 1.70),
}
SCALE_NAMES = ["hyd", "vol", "chg", "pol", "mw", "helix", "sheet"]

_B62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_B62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def blosum62() -> dict[tuple[str, str], int]:
    rows = [r.split() for r in _B62_ROWS.strip().splitlines()]
    return {
        (a, b): int(rows[i][j])
        for i, a in enumerate(_B62_ORDER)
        for j, b in enumerate(_B62_ORDER)
    }


def _codon_tables():
    code = genetic_code()
    n_codons: dict[str, int] = {a: 0 for a in AAS}
    for aa in code.values():
        if aa in n_codons:
            n_codons[aa] += 1
    nbr = one_nt_neighbourhood()
    # minimum number of nucleotide substitutions to convert any codon of a into one of b
    by_aa: dict[str, list[str]] = {a: [] for a in AAS}
    for c, a in code.items():
        if a in by_aa:
            by_aa[a].append(c)
    min_nt: dict[tuple[str, str], int] = {}
    for a in AAS:
        for b in AAS:
            best = 3
            for ca in by_aa[a]:
                for cb in by_aa[b]:
                    d = sum(x != y for x, y in zip(ca, cb))
                    best = min(best, d)
            min_nt[(a, b)] = best
    return n_codons, nbr, min_nt


def build_handcrafted(df: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Build the non-ESM feature blocks. Returns (X, names, group -> column indices)."""
    b62 = blosum62()
    n_codons, nbr, min_nt = _codon_tables()
    wt_aa = df["wt_aa"].to_numpy()
    mt_aa = df["mt_aa"].to_numpy()

    cols: list[np.ndarray] = []
    names: list[str] = []
    groups: dict[str, list[int]] = {g: [] for g in FEATURE_GROUPS}

    def add(vec: np.ndarray, name: str, group: str) -> None:
        groups[group].append(len(cols))
        cols.append(np.asarray(vec, dtype=np.float32))
        names.append(name)

    # --- subst -----------------------------------------------------------
    for i, s in enumerate(SCALE_NAMES):
        w = np.array([AA_SCALES[a][i] for a in wt_aa], dtype=np.float32)
        m = np.array([AA_SCALES[a][i] for a in mt_aa], dtype=np.float32)
        add(w, f"wt_{s}", "subst")
        add(m, f"mt_{s}", "subst")
        add(m - w, f"d_{s}", "subst")
        add(np.abs(m - w), f"absd_{s}", "subst")
    add([b62[(w, m)] for w, m in zip(wt_aa, mt_aa)], "blosum62", "subst")
    add([b62[(w, w)] for w in wt_aa], "blosum62_self", "subst")
    for a in AAS:
        add((wt_aa == a).astype(np.float32), f"wt_is_{a}", "subst")
        add((mt_aa == a).astype(np.float32), f"mt_is_{a}", "subst")
    add(((wt_aa == "G") | (wt_aa == "P")).astype(np.float32), "wt_is_GP", "subst")
    add((mt_aa == "P").astype(np.float32), "mt_is_Pro", "subst")
    add((mt_aa == "G").astype(np.float32), "mt_is_Gly", "subst")
    add((mt_aa == "C").astype(np.float32), "mt_is_Cys", "subst")

    # --- codon (library-construction confound probe) ---------------------
    add([n_codons[a] for a in wt_aa], "n_codons_wt", "codon")
    add([n_codons[a] for a in mt_aa], "n_codons_mt", "codon")
    add([len(nbr[a]) for a in wt_aa], "nbrhood_size_wt", "codon")
    add([min_nt[(w, m)] for w, m in zip(wt_aa, mt_aa)], "min_nt_changes", "codon")
    add(df["n_measured_at_pos"].to_numpy(), "n_measured_at_pos", "codon")

    # --- position --------------------------------------------------------
    add(df["rel_pos"].to_numpy(), "rel_pos", "position")
    add(df["pos"].to_numpy(), "pos_abs", "position")
    add(df["is_nuclease_domain"].to_numpy().astype(np.float32), "is_nuclease_domain", "position")
    for name, _, _ in DOMAINS:
        add((df["domain"].to_numpy() == name).astype(np.float32), f"dom_{name}", "position")

    X = np.column_stack(cols)
    return X, names, groups


def add_esm_features(
    df: pd.DataFrame,
    X: np.ndarray,
    names: list[str],
    groups: dict[str, list[int]],
    wtmarg_logprobs: np.ndarray | None,
    maskmarg_logprobs: np.ndarray | None,
    embeddings: np.ndarray | None,
    n_emb_pcs: int = 32,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Append the ESM-2 feature blocks. PCA on embeddings is fit on wild-type positions
    only (1368 rows, unsupervised, no label involvement), so it is not a leakage path."""
    from .esm_feats import positional_entropy, score_variants

    pos = df["pos"].to_numpy()
    wt_aa = df["wt_aa"].to_numpy()
    mt_aa = df["mt_aa"].to_numpy()
    ai = {a: i for i, a in enumerate(AAS)}
    new_cols, new_names, new_groups = [], [], {g: [] for g in FEATURE_GROUPS}

    def add(vec, name, group):
        new_groups[group].append(X.shape[1] + len(new_cols))
        new_cols.append(np.asarray(vec, dtype=np.float32))
        new_names.append(name)

    for tag, lp in (("wtmarg", wtmarg_logprobs), ("maskmarg", maskmarg_logprobs)):
        if lp is None:
            continue
        add(score_variants(lp, pos, wt_aa, mt_aa), f"esm_{tag}_llr", "esm_lm")
        add([lp[p - 1, ai[m]] for p, m in zip(pos, mt_aa)], f"esm_{tag}_logp_mt", "esm_lm")
        add([lp[p - 1, ai[w]] for p, w in zip(pos, wt_aa)], f"esm_{tag}_logp_wt", "esm_lm")
        ent = positional_entropy(lp)
        add([ent[p - 1] for p in pos], f"esm_{tag}_entropy", "esm_lm")
        # rank of the mutant AA within the position's distribution (1 = most likely)
        order = np.argsort(-lp, axis=1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(20)[None, :].repeat(lp.shape[0], 0), axis=1)
        add([rank[p - 1, ai[m]] for p, m in zip(pos, mt_aa)], f"esm_{tag}_rank_mt", "esm_lm")
        add([lp[p - 1].max() for p in pos], f"esm_{tag}_logp_argmax", "esm_lm")

    if embeddings is not None:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_emb_pcs, random_state=0)
        emb_pcs = pca.fit_transform(embeddings)  # (L, n_pcs), fit on the 1368 WT positions
        for j in range(n_emb_pcs):
            add(emb_pcs[pos - 1, j], f"esm_emb_pc{j:02d}", "esm_emb")

    if not new_cols:
        return X, names, groups
    X2 = np.column_stack([X] + new_cols)
    merged = {g: list(groups[g]) + new_groups[g] for g in FEATURE_GROUPS}
    return X2, names + new_names, merged


def add_struct_features(
    df: pd.DataFrame,
    X: np.ndarray,
    names: list[str],
    groups: dict[str, list[int]],
    struct_table: pd.DataFrame,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Append static structural feature blocks, looked up by residue position.

    Two groups: `struct` (per-residue) and `struct_win` (the same quantities pooled over
    5/11/25-residue windows). Both are constant across the substitutions measured at a
    position, so at variant level they sit inside the sqrt(ICC) <= 0.251 channel; the
    window block exists because the executed aggregation ladder located the unexplained
    signal at regional scale.

    Positions with no crystallographic coordinates stay NaN rather than being imputed.
    """
    pos = df["pos"].to_numpy()
    aligned = struct_table.reindex(range(1, struct_table.index.max() + 1))
    new_cols, new_names = [], []
    new_groups: dict[str, list[int]] = {g: [] for g in FEATURE_GROUPS}

    for col in aligned.columns:
        grp = "struct_win" if ("_w5" in col or "_w11" in col or "_w25" in col) else "struct"
        new_groups[grp].append(X.shape[1] + len(new_cols))
        new_cols.append(aligned[col].to_numpy(dtype=np.float32)[pos - 1])
        new_names.append(f"st_{col}")

    X2 = np.column_stack([X] + new_cols)
    merged = {g: list(groups.get(g, [])) + new_groups[g] for g in FEATURE_GROUPS}
    return X2, names + new_names, merged


def columns_excluding(groups: dict[str, list[int]], drop: list[str], n_cols: int) -> np.ndarray:
    """Column index array with the named feature groups removed (used by the ablation)."""
    dropped = set()
    for g in drop:
        dropped.update(groups.get(g, []))
    return np.array([i for i in range(n_cols) if i not in dropped], dtype=int)


def columns_only(groups: dict[str, list[int]], keep: list[str]) -> np.ndarray:
    idx: set[int] = set()
    for g in keep:
        idx.update(groups.get(g, []))
    return np.array(sorted(idx), dtype=int)
