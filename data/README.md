# Data provenance and licensing

Everything in this directory is third-party data redistributed with attribution. None of
it is my work.

## `CAS9_STRP1_Spencer_2017_positive.csv`, `CAS9_STRP1_WT.fasta`, `CAS9_STRP1_metadata.json`

The ProteinGym v1.3 assay table for `CAS9_STRP1_Spencer_2017_positive`, unchanged, plus the
reference sequence derived from its `target_seq` metadata field. `provenance.json` carries
the source URLs and the SHA-256 of the assay CSV as supplied.

- **Underlying experiment:** Spencer JM & Zhang X, *Deep mutational scanning of S. pyogenes
  Cas9 reveals important functional domains*, Scientific Reports **7**:16836 (2017).
  Published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Benchmark packaging:** Notin et al., *ProteinGym: Large-Scale Benchmarks for Protein
  Fitness Prediction and Design*. Repository:
  <https://github.com/OATML-Markslab/ProteinGym>. MIT licensed — see `ProteinGym_LICENSE`.

## `structure/4un3.pdb`

PDB entry **4UN3**, SpCas9 in complex with sgRNA and a PAM-containing target DNA, 2.5 Å.
Anders C, Niewoehner O, Duerst A & Jinek M, *Structural basis of PAM-dependent target DNA
recognition by the Cas9 endonuclease*, Nature **513**:569 (2014). Downloaded from RCSB PDB
(<https://files.rcsb.org/download/4UN3.pdb>); PDB data are released without restriction.

Note that 4UN3 is the **H840A** catalytically-inactive nickase, used to trap the substrate
complex — one residue differs from the assay reference sequence. This is handled explicitly
in `src/struct_feats.py` rather than ignored.

## Not redistributed here

ESM-2 and ESM-1v weights are downloaded on demand from the public `fair-esm` bucket and are
not committed (see `.gitignore`). They are MIT licensed by Meta AI:
Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language
model*, Science **379**:1123 (2023).
