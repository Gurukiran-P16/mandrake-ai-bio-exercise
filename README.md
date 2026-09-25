# Predicting SpCas9 activity from `CAS9_STRP1_Spencer_2017_positive`

A runnable evaluation of the proposal: *combine components of pretrained protein sequence
models with diffusion-based structure prediction models, replace the structure-prediction
trunk with an activity-prediction component, and predict gene-editor activity.*

**Verdict: REJECT as specified.** Use ESM-2 zero-shot log-likelihood ratios as a free
prior; do not build the stack. The binding constraint is the assay, not the architecture.
Reasoning and evidence in [`report/report.md`](report/report.md) (rendered to
`report/report.docx`).

This verdict is a **revision**. The first pass recommended MODIFY — keep the sequence arm,
drop the diffusion trunk, add static structural features at regional resolution — because
roughly half the reliably-measurable regional signal was unexplained and I hypothesised it
was structural. `experiments/exp6_structure.py` tested that hypothesis with real crystallographic
features from PDB 4UN3. It largely failed: structure adds **+0.003 (p = 0.61)** per
variant and closes only **~10% of the regional headroom**, significant at just 1 of 4
aggregation scales and non-monotone across them (p = 0.18, 0.011, 0.89, 0.18). Both the original recommendation and its refutation are kept in the report rather
than the earlier version being quietly overwritten.

## Scope boundary, stated once and enforced everywhere

**No diffusion model and no predicted structure is used anywhere in this repository.**
exp0–exp5 and exp8 use only the supplied assay table, the supplied reference sequence,
and sequence-only language models. exp6 adds features from **one solved crystal structure**
(PDB 4UN3) — static, experimental, no folding and no generative model. Claims about diffusion
features remain *architectural proposals* and are labelled as such in the report.

Structure also enters as a **bound**: a feature computed from the wild-type backbone is
constant across the substitutions measured at a position, so it is confined to the
position-only channel whose capacity `experiments/exp3_challenge.py` measures. exp6
confirmed that bound empirically — real crystallographic features added +0.003 (p = 0.61)
at variant level. **Per-variant structure prediction (folding each mutant) is not covered
by the bound and was not tested.**

## Environment

Python 3.12.10, Windows 11, 8 CPU threads, **no GPU**. That constraint shaped the design:

| operation | cost here | used? |
|---|---|---|
| ESM-2 650M, one forward pass (L=1368) | 23.7 s | yes |
| ESM-2 150M, full protein, both scoring modes | 2.7 min | yes |
| ESM-1v 650M, **any** forward pass at L=1368 | impossible — 1024-token ceiling | windowed instead |
| ESM-2 650M wild-type marginals + embeddings | 1 pass, 24 s | yes |
| ESM-2 650M masked marginals, strided (stride 20) | 20 passes, 8 min | yes |
| ESM-2 650M masked marginals, exact | 1368 passes, **9.0 h** | no |
| re-embedding all 8117 mutant sequences with 650M | 8117 passes, **53 h** | no |

The strided masked-marginal approximation masks every 20th position simultaneously so all
1368 positions are covered in 20 passes. `src/esm_feats.validate_stride` quantifies the
approximation error against exact single-position masking on the 35M model; it is not run
by default because the exact 35M version costs 68 min.

## Install and run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# structural experiment (exp6) needs one crystal structure, ~2.3 MB
mkdir -p data/structure
curl -o data/structure/4un3.pdb https://files.rcsb.org/download/4UN3.pdb

python run_all.py
```

Every stage except `exp6_structure` runs without the PDB file; exp6 skips itself with a
message if it is absent.

`python run_all.py --skip-esm` reproduces every baseline and the handcrafted models with
no torch/fair-esm dependency at all (the ESM rows are simply absent). First ESM run
downloads 2.5 GB of weights to `~/.cache/torch/hub/checkpoints/`.

**Start here if you want to check a number rather than reproduce everything:**

```bash
python verify.py
```

That asserts all 43 headline claims in the report against freshly recomputed values and
the saved predictions, in about a minute, with no torch needed. It prints PASS/FAIL with the
observed value next to the claim, so any single claim can be confirmed or broken in
isolation. All 43 currently pass.

**To interrogate the project interactively** — including answering "what if you changed
X?" without editing anything:

```bash
python demo.py                 # menu
python demo.py claims          # every headline claim + the command that proves it
python demo.py sweep           # the whole headline result in one table (~90 s)
python demo.py bound           # derive the sqrt(ICC) ceiling with live numbers
python demo.py leak            # what each split actually removes
python demo.py artefact        # the blocked-CV bias I caught in my own work
python demo.py ablate --drop esm_emb --scheme region_blocked
python demo.py residue --pos 840
python demo.py predict --mutant D10A
python demo.py score --mutant K855A   # score ANY substitution, even an unmeasured one
```

`bound`, `leak`, `artefact`, `residue` and `predict` need no torch and no ESM cache.
[`report/live_defense.md`](report/live_defense.md) pairs each likely question with the
answer and the command that demonstrates it.

Individual stages, in dependency order:

```bash
python experiments/exp0_data_audit.py             # requirement 1
python experiments/exp1_protocol_and_baselines.py # requirements 2-3, held-out predictions
python experiments/exp2_ablation.py               # requirement 3, ablation
python experiments/exp3_challenge.py              # requirement 3, self-refutation
python experiments/exp5_objective.py              # requirement 4, objective choice
python experiments/exp6_structure.py              # static structure from PDB 4UN3
python experiments/exp8_second_plm.py             # 2nd/3rd pretrained model
python experiments/exp4_figures.py                # figures
python report/render_docx.py                      # report.md -> report.docx
```

Runtime with the ESM cache warm: exp0 ~20 s, exp1 ~6 min, exp2 ~5 min, exp3 ~16 min,
exp5 ~8 min, exp6 ~16 min, exp8 ~12 min. Everything is seeded (`SEED = 20260924`) and deterministic.

## Layout

```
data/                     copies of the three supplied files actually read
src/data.py               loading, validation, domain annotation, ICC and reliability
src/splits.py             the four CV schemes and the leakage diagnostics
src/features.py           feature groups: subst, codon, position, esm_lm, esm_emb
src/esm_feats.py          ESM-2 extraction (wild-type + strided masked marginals)
src/struct_feats.py       static structural features from PDB 4UN3 (no DSSP binary needed)
src/models.py             baselines, ridge, gradient boosting, the CV driver
src/evaluate.py           metrics and cluster-bootstrap uncertainty
experiments/exp0..exp8    the stages above
verify.py                 43 assertions on every headline claim (~60 s, no torch)
demo.py                   live demo / debug console
results/                  every CSV/JSON/PNG below is produced by those scripts
report/report.md|.docx    the three-page report
report/architecture.md    the full architecture specification (requirement 4)
report/live_defense.md    likely questions, answers, and the command proving each
report/time_and_tools.md  time, compute, AI tools, unfinished work
```

### Key outputs

| file | contents |
|---|---|
| `results/audit_summary.json` | every requirement-1 number, with interpretation strings |
| `results/split_assignments.csv` | fold id per variant under all four schemes |
| `results/split_leakage.csv` | what each scheme actually removes |
| `results/heldout_predictions.csv` | out-of-fold prediction per variant per model per split |
| `results/main_results.csv` | pooled and per-fold metrics with bootstrap CIs |
| `results/model_vs_baseline.csv` | paired cluster-bootstrap model comparisons |
| `results/y_randomization.csv` | negative control on the harness itself |
| `results/ablation.csv` | leave-one-group-out and only-one-group |
| `results/challenge_aggregation.csv` | noise-reduction ladder, permutation nulls, reliability ceilings |
| `results/challenge_capacity.csv` | capacity ladder |
| `results/objective.csv` | raw vs rank vs winsorized target |
| `results/struct_reference_check.json` | 4UN3 vs the assay reference: coverage, mismatches, gaps |
| `results/struct_position_features.csv` | 57 structural features per residue (15 per-residue + 42 window-pooled) |
| `results/struct_feature_correlations.csv` | each structural feature vs the position-mean score |
| `results/struct_variant_level.csv` | variant-level effect of adding structure |
| `results/struct_aggregation_ladder.csv` | the decisive test: headroom with vs without structure |
| `results/second_plm.csv` | ESM-2 650M vs ESM-2 150M vs ESM-1v: zero-shot, redundancy, supervised add |
| `results/plm_length_limits.json` | which checkpoints can process a 1368-residue protein at all |
| `results/fig_main.png`, `fig_ablation.png`, `fig_structure.png` | report figures |

## What the data turned out to be

Verified directly, not taken from the filename or metadata:

- 8117 single missense substitutions at **all 1368** positions, 4-7 per position
  (mean 5.93) = 31.2% of the 25,992 possible. No stop codons, no synonymous variants,
  no duplicates. All 8117 `mutated_sequence` entries are exactly the wild type with the
  stated single substitution.
- **100% of the substitutions are reachable by a single nucleotide change from the
  wild-type codon** (41.3% expected if amino acids had been chosen freely). The library is
  error-prone PCR at 0.18% nucleotide mutation rate, so the substitution alphabet at each
  position is set by the codon, not by biophysical informativeness. Tryptophan appears as
  a target only 57 times.
- `DMS_score_bin` is **exactly** a median split of `DMS_score` and carries no extra
  information; the metadata cutoff `-0.2654328586` is the median to 10 decimal places.
- Position explains only **6.3%** of per-variant score variance (ICC = 0.063). Split-half
  reliability of a position's mean score is 0.28.
- Substitutions at the RuvC/HNH catalytic residues (D10, E762, H840, N854, N863, D986)
  sit at the **27th percentile** on average, not at the floor.

### Two places the supplied metadata is wrong or misleading

1. `selection_type: "Flow cytometry"` — the positive selection in Spencer & Zhang is a
   plate-based *ccdB* toxin survival selection in *E. coli* US0 `hisB- pyrF-`. No flow
   cytometry is involved.
2. The paper interprets individual variants only after a Fisher exact test with
   Benjamini-Hochberg correction and aggregates to a per-residue mutability score.
   ProteinGym supplies the raw unfiltered per-variant log2 fold change. The label being
   modelled is one the original authors never claimed was individually reliable.

Both are recorded in `src/data.METADATA_DISCREPANCIES`.

## Licence and attribution

Code and analysis: MIT (`LICENSE`). **Data under `data/` is third-party and separately
licensed — see [`data/README.md`](data/README.md) for full provenance.**

- Assay: Spencer JM & Zhang X, *Sci Rep* **7**:16836 (2017), CC BY 4.0, via ProteinGym
  v1.3 (Notin et al., MIT — notice retained in `data/ProteinGym_LICENSE`).
- Structure: PDB **4UN3**, Anders et al., *Nature* **513**:569 (2014).
- ESM-2 / ESM-1v: Lin et al., *Science* **379**:1123 (2023); Meier et al. (2021). Weights
  are fetched at runtime under MIT and are not redistributed here.
- Domain boundaries: Nishimasu et al., *Cell* **156**:935 (2014).
