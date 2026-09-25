"""Static structural features for SpCas9 from one experimental structure.

This is the experiment the report named as the one most likely to change its
recommendation: the claim under test is that the 0.22-0.35 unexplained headroom at
regional resolution is structural in origin.

WHAT THIS IS AND IS NOT
    These are features of ONE experimental wild-type-backbone structure. They are a
    function of residue position only - identical for all ~5.9 substitutions measured at
    a position - so at variant level they fall under the sqrt(ICC) <= 0.251 bound derived
    in `data.position_icc`. That is precisely why the interesting test is at regional
    scale, where the bound does not apply.

    This is NOT a diffusion model, NOT a structure predictor, and NOT per-variant
    structure prediction. Nothing here folds a mutant.

STRUCTURE: PDB 4UN3 (Anders, Niewoehner, Duerst & Jinek, Nature 513:569, 2014),
SpCas9-sgRNA-DNA with a PAM-containing target, 2.5 A.

    chain B  Cas9 protein, residues 4-1364 modelled (1306 of 1368 positions)
    chain A  sgRNA, 81 nt
    chain C  target DNA strand, 28 nt
    chain D  non-target / PAM strand, 11 nt

TWO PROPERTIES OF THIS STRUCTURE THAT ARE RECORDED RATHER THAN IGNORED
    1. It is the **H840A** catalytically-inactive nickase, used to trap the substrate
       complex. Residue 840 is ALA in the coordinates and HIS in the assay's reference
       sequence. Every other modelled residue matches the reference exactly (1 mismatch
       in 1306). The HNH active site is therefore a dead one; backbone geometry is
       unaffected but the 840 side-chain environment is not the wild-type one.
    2. 62 of 1368 positions have no coordinates (4.5%), in 8 disordered runs:
       1-3, 385, 711-718, 766-775, 1013-1029, 1051-1058, 1242-1252, 1365-1368.
       These become NaN, which HistGradientBoosting consumes natively. They are NOT
       imputed, because imputing them would invent structure where the crystal has none.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import REPO

STRUCTURE_PDB = REPO / "data" / "structure" / "4un3.pdb"
PROTEIN_CHAIN = "B"
NA_CHAINS = ("A", "C", "D")

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Theoretical maximum solvent accessibility, Tien et al. PLoS ONE 8:e80635 (2013).
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "Q": 225.0, "E": 223.0,
    "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0,
    "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

# Catalytic centres. RuvC triad D10/E762/D986 (+H983); HNH D839/H840/N863.
RUVC_RESIDUES = (10, 762, 983, 986)
HNH_RESIDUES = (839, 840, 863)
PAM_RESIDUES = (1333, 1335)

# Secondary structure is deliberately NOT included. A proper assignment needs DSSP, which
# is not available in this environment, and the Ca-geometry substitute I first wrote was
# extra surface area on a feature block that contributed nothing to the result. Dropped
# rather than defended.
STRUCT_FEATURES = [
    "rsa", "sasa_protein", "dsasa_na", "is_na_interface", "contact_n8", "contact_n12",
    "min_dist_na", "dist_ruvc", "dist_hnh", "dist_pam", "dist_centroid",
    "bfactor", "has_coords",
]
WINDOW_SIZES = (5, 11, 25)


def _load_model(pdb_path: Path = STRUCTURE_PDB):
    from Bio.PDB import PDBParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PDBParser(QUIET=True).get_structure("cas9", str(pdb_path))[0]


def _protein_residues(model) -> dict[int, object]:
    """Standard amino-acid residues of the protein chain, keyed by author residue number."""
    out = {}
    for r in model[PROTEIN_CHAIN]:
        if r.id[0] == " " and r.get_resname() in THREE2ONE:
            out[r.id[1]] = r
    return out


def check_against_reference(wt: str, pdb_path: Path = STRUCTURE_PDB) -> dict:
    """Confirm the structure's numbering and sequence line up with the assay reference."""
    res = _protein_residues(_load_model(pdb_path))
    present = sorted(i for i in res if 1 <= i <= len(wt))
    mismatches = [
        {"pos": i, "structure": THREE2ONE[res[i].get_resname()], "reference": wt[i - 1]}
        for i in present
        if THREE2ONE[res[i].get_resname()] != wt[i - 1]
    ]
    missing = [i for i in range(1, len(wt) + 1) if i not in res]
    runs, out = [], []
    for i in missing:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    for a, b in runs:
        out.append(f"{a}-{b}" if a != b else str(a))
    return {
        "n_modelled": len(present),
        "n_reference": len(wt),
        "coverage": len(present) / len(wt),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "n_missing": len(missing),
        "missing_runs": out,
    }


def compute_position_features(
    wt: str, pdb_path: Path = STRUCTURE_PDB, probe_points: int = 100
) -> pd.DataFrame:
    """Per-residue structural features, indexed 1..len(wt). Missing residues are NaN."""
    from Bio.PDB.SASA import ShrakeRupley

    model = _load_model(pdb_path)
    res = _protein_residues(model)

    # --- SASA twice: protein alone, then in the nucleoprotein complex ---------
    # The difference isolates burial caused by bound sgRNA/DNA, i.e. the nucleic-acid
    # interface, which is the mechanistically interesting surface for a nuclease.
    import copy

    sr = ShrakeRupley(n_points=probe_points)

    protein_only = copy.deepcopy(model)
    for cid in [c.id for c in protein_only if c.id != PROTEIN_CHAIN]:
        protein_only.detach_child(cid)
    for r in list(protein_only[PROTEIN_CHAIN]):
        if r.id[0] != " " or r.get_resname() not in THREE2ONE:
            protein_only[PROTEIN_CHAIN].detach_child(r.id)
    sr.compute(protein_only, level="R")
    sasa_alone = {
        r.id[1]: float(r.sasa) for r in protein_only[PROTEIN_CHAIN] if hasattr(r, "sasa")
    }

    complexed = copy.deepcopy(model)
    keep = {PROTEIN_CHAIN, *NA_CHAINS}
    for cid in [c.id for c in complexed if c.id not in keep]:
        complexed.detach_child(cid)
    for ch in complexed:  # strip waters and ions from every kept chain
        for r in list(ch):
            if r.id[0] != " ":
                ch.detach_child(r.id)
    sr.compute(complexed, level="R")
    sasa_complex = {
        r.id[1]: float(r.sasa)
        for r in complexed[PROTEIN_CHAIN]
        if hasattr(r, "sasa") and r.get_resname() in THREE2ONE
    }

    # --- geometry ------------------------------------------------------------
    na_atoms = np.array(
        [a.get_coord() for cid in NA_CHAINS for r in model[cid] if r.id[0] == " " for a in r]
    )
    cb = {}
    for i, r in res.items():
        atom = "CB" if "CB" in r else ("CA" if "CA" in r else None)
        if atom:
            cb[i] = r[atom].get_coord()
    cb_ids = np.array(sorted(cb))
    cb_xyz = np.array([cb[i] for i in cb_ids])
    centroid = cb_xyz.mean(axis=0)

    def centre_of(positions) -> np.ndarray | None:
        pts = [res[p]["CA"].get_coord() for p in positions if p in res and "CA" in res[p]]
        return np.mean(pts, axis=0) if pts else None

    ruvc_c, hnh_c, pam_c = centre_of(RUVC_RESIDUES), centre_of(HNH_RESIDUES), centre_of(PAM_RESIDUES)

    rows = []
    for pos in range(1, len(wt) + 1):
        if pos not in res:
            rows.append({"pos": pos, **{f: np.nan for f in STRUCT_FEATURES}, "has_coords": 0})
            continue
        r = res[pos]
        aa = THREE2ONE[r.get_resname()]
        s_alone = sasa_alone.get(pos, np.nan)
        s_cplx = sasa_complex.get(pos, np.nan)
        atoms = np.array([a.get_coord() for a in r])
        d_na = float(np.min(np.linalg.norm(na_atoms[None, :, :] - atoms[:, None, :], axis=2)))
        p = cb.get(pos)
        if p is not None:
            d = np.linalg.norm(cb_xyz - p, axis=1)
            n8 = int(((d < 8.0) & (d > 0)).sum())
            n12 = int(((d < 12.0) & (d > 0)).sum())
            d_cent = float(np.linalg.norm(p - centroid))
        else:
            n8 = n12 = d_cent = np.nan
        ca = r["CA"].get_coord() if "CA" in r else None
        rows.append({
            "pos": pos,
            # RSA is normalised by the REFERENCE residue's max ASA, not the structure's,
            # so the H840A substitution in the crystal does not distort position 840.
            "rsa": s_alone / MAX_ASA[wt[pos - 1]] if np.isfinite(s_alone) else np.nan,
            "sasa_protein": s_alone,
            "dsasa_na": (s_alone - s_cplx) if np.isfinite(s_alone) and np.isfinite(s_cplx) else np.nan,
            "is_na_interface": float((s_alone - s_cplx) > 1.0)
            if np.isfinite(s_alone) and np.isfinite(s_cplx) else np.nan,
            "contact_n8": n8,
            "contact_n12": n12,
            "min_dist_na": d_na,
            "dist_ruvc": float(np.linalg.norm(ca - ruvc_c)) if ca is not None and ruvc_c is not None else np.nan,
            "dist_hnh": float(np.linalg.norm(ca - hnh_c)) if ca is not None and hnh_c is not None else np.nan,
            "dist_pam": float(np.linalg.norm(ca - pam_c)) if ca is not None and pam_c is not None else np.nan,
            "dist_centroid": d_cent,
            "bfactor": float(np.mean([a.get_bfactor() for a in r])),
            "has_coords": 1,
        })
    return pd.DataFrame(rows).set_index("pos")


def add_window_features(feat: pd.DataFrame, windows=WINDOW_SIZES) -> pd.DataFrame:
    """Centred rolling means of each structural feature.

    Motivated directly by the executed aggregation ladder: the unexplained signal sits at
    regional scale, so the structural input is supplied at that scale as well as per-residue.
    `min_periods=1` means a window straddling a disordered run averages whatever is modelled.
    """
    out = [feat]
    base = [c for c in feat.columns if c not in ("has_coords",)]
    for w in windows:
        roll = feat[base].rolling(w, center=True, min_periods=1).mean()
        roll.columns = [f"{c}_w{w}" for c in base]
        out.append(roll)
    return pd.concat(out, axis=1)


def build_struct_table(wt: str, pdb_path: Path = STRUCTURE_PDB) -> pd.DataFrame:
    return add_window_features(compute_position_features(wt, pdb_path))
