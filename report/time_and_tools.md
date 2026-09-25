# Time, compute, AI tools, and what is unfinished

An honest record. Where a number is measured it says so; where it is an estimate it says
that too.

## How this was produced

The assignment permits AI tools. **This file is the disclosure.**

The code in `src/` and `experiments/`, and the first drafts of `report.md` and
`architecture.md`, were written by Claude (Opus 5) in Claude Code, working from the
assignment pack and my direction. It ran every experiment. I set the research question and
the scope, decided what to test next, reviewed the output, and made the calls recorded
under *What I reviewed and questioned* below.

I have not tried to disguise that split, and I can explain, debug or modify any part of the
repository. `python demo.py claims` lists every headline claim with the command that
reproduces it; `python verify.py` re-checks all of them in about 40 seconds.

## Time

Two numbers, because they differ and only one of them is the five-hour figure.

| | |
|---|---|
| **Elapsed, start to finish** | **≈ 6 h 10 m** |
| Unattended compute inside that (model downloads, CPU runs) | ≈ 1 h 35 m |
| **Attended effort** | **≈ 4 h 35 m** |

So the exercise fits the five-hour budget on attended effort and overruns it on elapsed
time. Both are stated rather than picking the flattering one. A single ESM-1v checkpoint
was a 7.8 GB download; that alone is 25 minutes in the elapsed column and nothing in the
attended one.

Most of my own reading happened while compute was running, so my hours and the elapsed
total overlap rather than add.

| workstream | attended | notes |
|---|---|---|
| read the pack; inspect the CSV, FASTA and metadata directly | 25 m | where the single-nucleotide finding came from |
| retrieve and read the original paper (PMC5715146) | 10 m | the Nature URL redirects to a login; the PMC copy works |
| environment: CPU PyTorch, fair-esm, Biopython, matplotlib | 12 m | plus 2.5 GB of ESM-2 weights, unattended |
| write `src/` (data, splits, features, models, evaluation, ESM) | 55 m | |
| data audit (exp0) | 10 m | requirement 1 |
| protocol, baselines and the extension (exp1) | 25 m | requirements 2–3 |
| feature-group ablation (exp2) | 15 m | |
| self-refutation experiment (exp3) | 35 m | includes rewriting it after catching the cross-validation artefact below |
| training-objective comparison (exp5) | 8 m | ran on spare cores alongside exp3 |
| structural features from PDB 4UN3 (exp6) | 40 m | and the report revision it forced |
| second and third pretrained model (exp8) | 25 m | |
| report, architecture spec, README, DOCX renderer | 55 m | |

`demo.py` and `report/live_defense.md` are interview preparation rather than assignment
deliverables and are excluded from the totals above.

## Compute

One laptop: Windows 11, Python 3.12.10, 8 CPU threads, **no GPU**. Measured costs:

| operation | measured | used? |
|---|---|---|
| ESM-2 650M, one forward pass (L=1368) | 23.7 s | yes |
| ESM-2 650M wild-type marginals + embeddings | 1 pass, 23 s | yes |
| ESM-2 650M strided masked marginals (stride 20) | 20 passes, 7.6 min | yes |
| ESM-2 150M, full protein, both scoring modes | 2.7 min | yes |
| Structural features from 4UN3 (SASA + geometry) | 7 s | yes |
| ESM-2 650M **exact** masked marginals | 1368 passes, **9.0 h** | no — out of budget |
| Re-embedding all 8117 mutant sequences | 8117 passes, **53 h** | no — out of budget |
| ESM-1v at full length | impossible — 1024-token ceiling | windowed instead |

No cloud compute and no paid API calls beyond the Claude session. Downloaded weights:
ESM-2 650M (2.5 GB), ESM-2 150M (0.6 GB), ESM-1v member 1 (7.8 GB), all from the public
`fair-esm` bucket. One crystal structure (PDB 4UN3, 2.3 MB) from RCSB.

## AI tools used

- **Claude Code (Opus 5)** — the code, the experiments, and the drafting described above.
- **ESM-2 650M / 150M and ESM-1v** via `fair-esm` 2.0.0 — as *subjects of study*, not as
  authoring tools.
- **Web retrieval** — one search and one page fetch, to read Spencer & Zhang 2017 for the
  assay design. That retrieval changed two modelling-relevant facts: the supplied metadata
  calls the selection "Flow cytometry" when the positive selection is a plate-based *ccdB*
  survival assay, and the authors' own analysis was significance-filtered and aggregated
  to per-residue scores, whereas ProteinGym supplies the raw per-variant value.
- Nothing else.

## What I reviewed and questioned

Recorded because the judgement is the part being assessed, and because it is checkable.

**Where I slowed down.** The reliability ceiling. The report claims that any feature
constant across the substitutions at a position cannot exceed a correlation of about 0.25,
and derives it from a variance decomposition. I went back over that argument and read up
on what an intraclass correlation actually is before I was willing to rely on it, because
the entire recommendation leans on it. Having checked it, I asked for it to be stated in
plain terms in the report rather than left as a formula — a feature that gives the same
value to all six mutants at a position cannot tell them apart, so it is limited to the
share of variation that lies between positions, which is 6%.

**What I did not believe.** The first version's explanation was that the labels are too
noisy for anything else to matter. I did not think that was complete. The report's own
challenge experiment had found that roughly half the signal recoverable at a regional
level was unexplained by any feature in the model, and "it's all noise" does not account
for a gap that size — a gap that *widens* as you average the noise down.

**What I did about it.** I asked for the structural experiment specifically to attack the
negative result rather than to take it on trust. If burial and active-site distance were
the missing piece, that was the cheapest way to find out, and if they were not, the
recommendation needed to change. They were not: real crystallographic features added 0.003
per variant (p = 0.61) and closed roughly a tenth of the regional gap, significant at just
one of four aggregation scales and non-monotone across them. The recommendation moved from
*modify* to *reject*, and my original doubt turned out to be justified in a way I had not
expected — the gap is real, it is not noise, and it is not static structure either. I do
not know what it is, and the report says so.

**What I checked directly.** About two hours on the report, the architecture specification
and a good part of the code, running commands myself rather than reading numbers off the
page.

**What I asked to be cut.** A homemade secondary-structure assignment (a Cα-geometry
substitute for DSSP, which was unavailable), a separate significance script folded back
into the experiment it belonged to, and a low-rank interaction module in the architecture
specification that was elegant and unexecuted. All three were surface area I could not
defend as load-bearing, on a project whose conclusions did not depend on them.

## What went wrong, and what was fixed

1. **A NumPy type error** combining string arrays with a bitwise operator in
   `features.build_handcrafted`. Caught by the first smoke test.
2. **A pandas/NumPy mismatch** where a reliability function was handed Series instead of
   arrays. The call was redundant and was removed.
3. **A cross-validation artefact — the one that mattered.** The first aggregation ladder
   reported correlations of −0.72, −0.76 and −0.35 at the wider window sizes. Those are not
   signal reversals. Pooled out-of-fold correlation is not centred on zero under a spatially
   blocked split: each fold can only regress towards its own training mean, and with 55, 27
   and 10 units the statistic is dominated by differences between fold means. The same bias
   showed up independently in a constant predictor — pooled −0.145, per-fold exactly 0.000.
   Fixed by adding a permutation null at every scale, marking scales with fewer than 100
   units as uninterpretable, and switching the blocked splits to per-fold means. Reporting
   the original number would have been reporting an effect that does not exist.
   Reproduce it: `python demo.py artefact`.
4. **The structure is not wild type.** PDB 4UN3 is the H840A catalytically-dead nickase,
   used to trap the substrate complex. Caught by diffing the coordinate sequence against
   the assay reference — one mismatch in 1,306 residues, at position 840. Relative solvent
   accessibility is therefore normalised by the *reference* residue rather than the
   crystal's. I had also assumed the protein was chain A; it is chain B, and chain A is the
   guide RNA. An inspection pass caught that before any feature code was written.
5. **ESM-1v cannot read this protein.** It failed with
   `Sequence length 1370 above maximum sequence length of 1024`. ESM-1v and ESM-1b use
   learned absolute positional embeddings capped at 1024 tokens; SpCas9 needs 1,370. ESM-2
   works only because it uses rotary embeddings. Recorded as an architecture finding in
   `results/plm_length_limits.json` rather than worked around silently.

## Unfinished

Ranked by how much each would change the conclusions.

1. **Per-variant structure prediction** — folding each mutant separately. The 0.25 ceiling
   covers features of the *wild-type* structure only, and exp6 confirmed that bound
   empirically; neither says anything about a structure predicted per mutant. This is now
   the largest untested claim in the original proposal. It needs a GPU.
2. **The residual regional signal is unexplained.** Two independent descriptions of
   position — a 650M language model's embeddings and a 2.5 Å crystal structure — plateau in
   the same place, about 0.25 below the label's own reliability, and a library/sequencing
   artefact is ruled out (|ρ| ≤ 0.10 against coverage proxies). I cannot separate regional
   expression and folding effects, which the assay conflates with cleavage, from structural
   context beyond the features I computed.
3. **No per-variant reliability estimate exists**, because the supplied table has no
   replicate read counts. Everything rests on a positional intraclass correlation. This is
   also why the recommended next experiment is a replicated assay rather than a better model.
4. **The strided masked-marginal approximation is unvalidated on 650M.** Exact masking
   costs 9 h. Indirect reassurance only: wild-type and strided masked marginals agree to
   0.001 Spearman on the downstream task. `src/esm_feats.validate_stride` would measure it
   properly on the 35M model (68 min) and was not run.
5. **No ESM-2 fine-tuning**, not even the top two layers. Argued against on label-budget
   grounds and gated in the architecture specification, but not attempted — 8,117
   forward-and-backward passes through 650M on 8 CPU threads is not feasible here.
6. **ESM-1v was not evaluated fairly** because of its length ceiling, so the
   model-family axis is tested only through windowing. ESM-2 3B and the five-member ESM-1v
   ensemble were not run. The scale axis (150M against 650M) is clean.
7. **No multi-mutant or epistasis analysis** — impossible; the dataset is single mutants only.
8. **No secondary-structure features.** DSSP is unavailable in this environment and the
   geometric substitute was cut rather than defended.
9. **The three-page limit is a calculated estimate, not a measured render** — no Word or
   LibreOffice was available to paginate. Calculated at 2.9–3.0 pages.
