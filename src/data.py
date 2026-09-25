"""Loading, validation and annotation of the CAS9_STRP1_Spencer_2017_positive assay table.

Everything here is derived from direct inspection of the supplied CSV/FASTA, not from
the filename or the metadata JSON. The metadata JSON is *not* trusted: see
`METADATA_DISCREPANCIES` below for fields that contradict the primary publication.
"""
from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RESULTS = REPO / "results"

ASSAY_CSV = DATA / "CAS9_STRP1_Spencer_2017_positive.csv"
WT_FASTA = DATA / "CAS9_STRP1_WT.fasta"
METADATA_JSON = DATA / "CAS9_STRP1_metadata.json"

AAS = "ACDEFGHIKLMNPQRSTVWY"

# ---------------------------------------------------------------------------
# Things the supplied metadata gets wrong, verified against Spencer & Zhang 2017
# (Sci Rep 7:16836, PMC5715146). Recorded here because they change the modelling.
# ---------------------------------------------------------------------------
METADATA_DISCREPANCIES = {
    "selection_type": (
        "metadata says 'Flow cytometry'; the positive-selection assay in the paper is a "
        "plate-based ccdB toxin survival selection in E. coli US0 hisB- pyrF-. No flow "
        "cytometry is involved in the positive selection."
    ),
    "DMS_total_number_mutants": (
        "metadata says 8117 and the CSV has 8117 rows; the paper reports 8549 accessible "
        "non-synonymous mutations, so ~5% of the library is absent from the ProteinGym table."
    ),
    "per_variant_reliability": (
        "the paper interprets variants only after a Fisher exact test with Benjamini-Hochberg "
        "correction, and aggregates to a per-residue 'mutability score'. ProteinGym supplies "
        "the raw unfiltered per-variant log2 fold change, which the authors never claimed was "
        "individually reliable."
    ),
}

# Standard SpCas9 domain boundaries (Nishimasu et al. 2014; Jiang & Doudna 2017).
# Used for the region-blocked split and the domain-mean baseline.
DOMAINS: list[tuple[str, int, int]] = [
    ("RuvC-I", 1, 59),
    ("BridgeHelix", 60, 93),
    ("REC1a", 94, 179),
    ("REC2", 180, 307),
    ("REC1b", 308, 717),
    ("RuvC-II", 718, 765),
    ("HNH", 766, 909),
    ("RuvC-III", 910, 1098),
    ("WED", 1099, 1200),
    ("TOPO", 1201, 1218),
    ("CTD-PI", 1219, 1368),
]

# Nuclease (catalytic) vs non-nuclease domains - used only for reporting.
NUCLEASE_DOMAINS = {"RuvC-I", "RuvC-II", "RuvC-III", "HNH"}

# Residues with established mechanistic roles, used as label-validity probes.
KNOWN_FUNCTIONAL_RESIDUES = {
    10: "D10 RuvC catalytic (D10A = nickase)",
    762: "E762 RuvC catalytic",
    840: "H840 HNH catalytic (H840A = nickase)",
    854: "N854 HNH",
    863: "N863 HNH",
    986: "D986 RuvC catalytic",
    1333: "R1333 PAM guanine contact",
    1335: "R1335 PAM guanine contact",
}

_BASES = "TCAG"
_AA_BY_CODON_ORDER = (
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
)


def genetic_code() -> dict[str, str]:
    """Standard genetic code as codon -> single-letter amino acid ('*' = stop)."""
    code, i = {}, 0
    for b1 in _BASES:
        for b2 in _BASES:
            for b3 in _BASES:
                code[b1 + b2 + b3] = _AA_BY_CODON_ORDER[i]
                i += 1
    return code


def one_nt_neighbourhood() -> dict[str, set[str]]:
    """Amino acids reachable from each amino acid by exactly one nucleotide substitution.

    Union over all synonymous codons of the source amino acid, because the library was
    made by error-prone PCR on a codon-optimised gene whose codon usage we do not have.
    This is therefore the *permissive* version of the accessibility set: if a substitution
    fails this test it is definitely not reachable by a single nucleotide change.
    """
    code = genetic_code()
    by_aa: dict[str, list[str]] = collections.defaultdict(list)
    for codon, aa in code.items():
        by_aa[aa].append(codon)

    nbr: dict[str, set[str]] = {}
    for aa in AAS:
        reach: set[str] = set()
        for codon in by_aa[aa]:
            for pos in range(3):
                for base in _BASES:
                    if base == codon[pos]:
                        continue
                    target = code[codon[:pos] + base + codon[pos + 1 :]]
                    if target not in ("*", aa):
                        reach.add(target)
        nbr[aa] = reach
    return nbr


def read_wt_sequence(path: Path = WT_FASTA) -> str:
    """Read the single-record reference FASTA and return the bare sequence."""
    seq_parts, n_records = [], 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            n_records += 1
        elif line:
            seq_parts.append(line)
    if n_records != 1:
        raise ValueError(f"expected exactly 1 FASTA record, found {n_records}")
    seq = "".join(seq_parts)
    unknown = set(seq) - set(AAS)
    if unknown:
        raise ValueError(f"non-standard residues in reference sequence: {sorted(unknown)}")
    return seq


def domain_of(pos: int) -> str:
    for name, lo, hi in DOMAINS:
        if lo <= pos <= hi:
            return name
    return "UNANNOTATED"


@dataclass
class AuditReport:
    """Result of validating the assay table against the reference sequence."""

    n_rows: int
    n_positions: int
    seq_len: int
    coverage_fraction: float
    muts_per_position: dict[int, int]
    frac_reachable_one_nt: float
    frac_reachable_if_uniform: float
    sequence_check: dict[str, int]
    bin_is_median_split: bool
    binarization_cutoff: float
    n_duplicate_variants: int
    n_stop: int
    n_synonymous: int

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["muts_per_position"] = {int(k): int(v) for k, v in self.muts_per_position.items()}
        return d


def load_assay(
    csv_path: Path = ASSAY_CSV,
    fasta_path: Path = WT_FASTA,
    validate: bool = True,
) -> tuple[pd.DataFrame, str, AuditReport]:
    """Load the assay table, parse variants, annotate, and validate against the reference.

    Returns (dataframe, wt_sequence, audit_report).

    The returned frame adds: wt_aa, pos, mt_aa, domain, is_nuclease_domain,
    rel_pos, reachable_1nt, n_measured_at_pos.
    """
    wt = read_wt_sequence(fasta_path)
    df = pd.read_csv(csv_path)

    expected = {"mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"assay CSV missing columns: {sorted(missing)}")

    # --- parse the variant identifier -------------------------------------
    parsed = df["mutant"].str.extract(r"^([A-Z])(\d+)([A-Z*])$")
    if parsed.isna().any().any():
        bad = df.loc[parsed.isna().any(axis=1), "mutant"].tolist()[:10]
        raise ValueError(f"unparseable mutant identifiers, e.g. {bad}")
    df["wt_aa"] = parsed[0]
    df["pos"] = parsed[1].astype(int)
    df["mt_aa"] = parsed[2]

    if validate:
        _check_positions(df, wt)

    # --- annotate ---------------------------------------------------------
    df["domain"] = [domain_of(p) for p in df["pos"]]
    df["is_nuclease_domain"] = df["domain"].isin(NUCLEASE_DOMAINS)
    df["rel_pos"] = (df["pos"] - 1) / (len(wt) - 1)
    nbr = one_nt_neighbourhood()
    df["reachable_1nt"] = [m in nbr[w] for w, m in zip(df["wt_aa"], df["mt_aa"])]
    df["n_measured_at_pos"] = df.groupby("pos")["pos"].transform("size")

    audit = _audit(df, wt, nbr)
    return df, wt, audit


def _check_positions(df: pd.DataFrame, wt: str) -> None:
    if df["pos"].min() < 1 or df["pos"].max() > len(wt):
        raise ValueError("variant positions fall outside the reference sequence")
    stated = df["wt_aa"].to_numpy()
    actual = np.array([wt[p - 1] for p in df["pos"]])
    n_bad = int((stated != actual).sum())
    if n_bad:
        raise ValueError(f"{n_bad} variants disagree with the reference residue")


def _verify_mutated_sequences(df: pd.DataFrame, wt: str) -> dict[str, int]:
    """Confirm mutated_sequence really is WT with exactly the stated single substitution."""
    counts: collections.Counter = collections.Counter()
    L = len(wt)
    for seq, pos, wt_aa, mt_aa in zip(
        df["mutated_sequence"], df["pos"], df["wt_aa"], df["mt_aa"]
    ):
        if len(seq) != L:
            counts["wrong_length"] += 1
            continue
        diffs = [i for i in range(L) if seq[i] != wt[i]]
        if len(diffs) == 0:
            counts["identical_to_wt"] += 1
        elif len(diffs) > 1:
            counts["multiple_substitutions"] += 1
        else:
            i = diffs[0]
            ok = (i + 1 == pos) and seq[i] == mt_aa and wt[i] == wt_aa
            counts["ok" if ok else "inconsistent_with_id"] += 1
    return dict(counts)


def _audit(df: pd.DataFrame, wt: str, nbr: dict[str, set[str]]) -> AuditReport:
    per_pos = df.groupby("pos").size()
    cutoff = float(df["DMS_score"].median())
    bin_from_cutoff = (df["DMS_score"] > cutoff).astype(int)
    frac_uniform = float(np.mean([len(nbr[w]) / 19 for w in df["wt_aa"]]))
    return AuditReport(
        n_rows=len(df),
        n_positions=int(df["pos"].nunique()),
        seq_len=len(wt),
        coverage_fraction=len(df) / (len(wt) * 19),
        muts_per_position=dict(collections.Counter(per_pos.values)),
        frac_reachable_one_nt=float(df["reachable_1nt"].mean()),
        frac_reachable_if_uniform=frac_uniform,
        sequence_check=_verify_mutated_sequences(df, wt),
        bin_is_median_split=bool((bin_from_cutoff == df["DMS_score_bin"]).all()),
        binarization_cutoff=cutoff,
        n_duplicate_variants=int(df["mutant"].duplicated().sum()),
        n_stop=int((df["mt_aa"] == "*").sum()),
        n_synonymous=int((df["mt_aa"] == df["wt_aa"]).sum()),
    )


def load_metadata(path: Path = METADATA_JSON) -> dict:
    return json.loads(path.read_text())


def position_icc(df: pd.DataFrame, value_col: str = "DMS_score") -> dict[str, float]:
    """One-way random-effects ICC of the score grouped by residue position.

    Interpretation used throughout this project: ICC is the fraction of per-variant
    score variance attributable to the residue position. Any feature that is constant
    across the substitutions measured at a position (which includes every feature
    computed from the wild-type structure) is bounded by sqrt(ICC) in Pearson
    correlation against the per-variant score.
    """
    y = df[value_col].to_numpy(float)
    n = len(y)
    groups = df.groupby("pos")[value_col]
    k = groups.size().to_numpy(float)
    a = len(k)
    grand = y.mean()
    ms_between = float((k * (groups.mean().to_numpy() - grand) ** 2).sum() / (a - 1))
    within = y - df.groupby("pos")[value_col].transform("mean").to_numpy()
    ms_within = float((within**2).sum() / (n - a))
    k0 = (n - (k**2).sum() / n) / (a - 1)
    var_between = max((ms_between - ms_within) / k0, 0.0)
    icc = var_between / (var_between + ms_within)
    return {
        "ms_between": ms_between,
        "ms_within": ms_within,
        "var_between": var_between,
        "var_within": ms_within,
        "icc_position": icc,
        "position_only_pearson_ceiling": float(np.sqrt(icc)),
        "mean_group_size": float(k.mean()),
    }


def position_split_half_reliability(
    df: pd.DataFrame, value_col: str = "DMS_score", n_repeats: int = 200, seed: int = 0
) -> dict[str, float]:
    """Empirical reliability of the *position-level* mean score.

    Splits the variants at each position into two halves, correlates the two sets of
    half-means across positions, and applies the Spearman-Brown correction. This is a
    model-free estimate of how much positional signal is even estimable from this table,
    and therefore of the best achievable performance for any position-only feature.
    """
    rng = np.random.default_rng(seed)
    from scipy.stats import pearsonr, spearmanr

    by_pos = {p: g[value_col].to_numpy(float) for p, g in df.groupby("pos")}
    usable = {p: v for p, v in by_pos.items() if len(v) >= 4}
    pear, spear = [], []
    for _ in range(n_repeats):
        a_means, b_means = [], []
        for v in usable.values():
            idx = rng.permutation(len(v))
            half = len(v) // 2
            a_means.append(v[idx[:half]].mean())
            b_means.append(v[idx[half : 2 * half]].mean())
        pear.append(pearsonr(a_means, b_means)[0])
        spear.append(spearmanr(a_means, b_means)[0])
    r_half = float(np.mean(pear))
    return {
        "n_positions_used": len(usable),
        "half_half_pearson": r_half,
        "half_half_spearman": float(np.mean(spear)),
        # Spearman-Brown: reliability of the full-length (all variants) position mean
        "spearman_brown_full": float(2 * r_half / (1 + r_half)) if r_half > -1 else float("nan"),
    }
