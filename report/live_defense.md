# Live defence pack

The assignment says: *"Be prepared to explain, debug or modify your implementation live."*
This is that preparation. Every answer below is a number I actually computed, with the
command that reproduces it in front of the interviewer.

**Before you walk in, run these three and read the output once:**

```bash
python verify.py            # 43 checks, ~60 s, all pass
python demo.py claims       # every headline claim + the command that proves it
python demo.py sweep        # the whole headline result in one table, ~90 s
```

If you only remember one command, it is `python demo.py sweep`. It shows the central
finding — a 650M protein language model helps only on the split that leaks — as a live
five-row table.

---

## The 60-second version

> We were asked to evaluate combining a pretrained protein sequence model with a
> diffusion structure model, swapping the structure trunk for an activity head. I said
> **reject as specified**, and the reason is the label, not the architecture.
>
> The assay is a bacterial survival selection on one guide RNA. Only 6.3% of the
> per-variant score variance is attributable to *which residue* was mutated, which puts a
> hard ceiling of r ≤ 0.25 on any feature that is constant across the substitutions at a
> position — and every wild-type-structure feature is such a feature.
>
> ESM-2 650M zero-shot does work, modestly, ρ ≈ 0.18 against 0.11 for a substitution
> matrix. But supervising on all 8,117 labels adds nothing that survives a split which
> actually tests generalization, and the model's 1280-dimensional embedding *hurts*
> region-level transfer because it acts as a positional fingerprint.
>
> I then ran the follow-up my own report called for — real crystallographic features from
> PDB 4UN3 — because I had found half the reliable regional signal unexplained and
> hypothesised it was structural. It wasn't: structure adds +0.003 (p = 0.61) per variant
> and closes about 10% of the regional gap. So I changed my recommendation from modify to
> reject, and the next investment should be a better assay, not a bigger model.

---

## A. The bound — the single most likely line of questioning

**Q1. Why is √ICC a legitimate ceiling and not a rule of thumb?**
Write the score as `y_pi = μ + a_p + e_pi`, where `a_p` is the position effect and `e_pi`
is everything else. A feature `f` that is constant within a position can only correlate
with `y` through `a_p`, so
`corr(f,y) = corr(f,a) · sd(a)/√(var(a)+var(e)) ≤ √ICC`, with equality only if `f`
reproduces the position effect exactly. `var(a)` is estimated noise-corrected as
`(MS_between − MS_within)/k₀`, so it is not inflated by within-position scatter.
→ `python demo.py bound` prints every term: MS_between 0.4971, MS_within 0.3555,
var(a) 0.0239, ICC 0.0629, ceiling 0.2508. Code: [src/data.py](../src/data.py) `position_icc`.

**Q2. Isn't the within-position variance partly real biology, not noise? Then your ICC
under-states the signal.**
Correct, and I say so in the report. `var(e)` mixes genuine substitution-specific effects
with measurement noise, and I cannot separate them without replicates. But that does not
weaken the bound — it *is* the bound. The claim is specifically about features that cannot
distinguish substitutions at the same residue. Real substitution-specific biology in
`var(e)` is exactly what such a feature is unable to reach.

**Q3. How do you know the bound actually binds?**
Build the best possible position-only predictor — the position mean itself, which no
function of position can beat — and check it. In-sample it gives r = 0.470, but that is
1,368 means fitted on 5.9 points each. The honest leave-one-out value is **r = 0.130**,
well under 0.251. Then the empirical test: exp6 added 49 real crystallographic features and
gained **+0.003, p = 0.61**. The bound predicted that.

**Q4. What escapes the bound?**
Any feature that *varies* across substitutions at one position: BLOSUM62, amino-acid
property deltas, the ESM log-likelihood ratio — and **per-variant structure prediction**,
i.e. folding each mutant separately. That last one I did not test and cannot rule out.
Say this unprompted; it is the honest limit of the argument.

**Q5. Is 0.28 split-half reliability of the position means consistent with ICC 0.063?**
Yes, and they are independent estimates of the same thing. Spearman–Brown corrected
split-half reliability of a position's mean is 0.284; ICC is 0.0629 per variant. With
~5.9 measurements per position, averaging lifts reliability roughly as
`n·ICC/(1+(n−1)·ICC)` = 5.93×0.063/(1+4.93×0.063) ≈ 0.28. They agree.

---

## B. The splits

**Q6. ProteinGym uses random CV. Why did you deviate?**
Because random CV answers a question nobody is asking. There are 4–7 variants at every
position, so a random split puts siblings of every test variant in training. I report the
random split for comparability and then show it is the *only* split where the language
model helps. → `python demo.py sweep`

**Q7. Position-grouping fixes that, so why two more splits?**
Because position-grouping does not close the leak. Position means are autocorrelated along
the sequence (r = 0.14 at lag 1, ~0 by lag 100), and **100% of test variants still have a
training neighbour within 5 residues** under position-grouping. Region-blocking cuts that
to 6.6%; domain-holdout removes the domain entirely.
→ `python demo.py leak`

**Q8. Why did `domain_mean` score −0.150 under domain-holdout? A negative correlation
looks like a bug.**
It is a real artefact and I caught it in my own work. When a whole domain is held out there
is no training mean for it, so the baseline falls back to the global mean — a constant per
fold. Pooled out-of-fold correlation is *not* centred on zero for a constant-per-fold
predictor under a spatially blocked split, because each fold's training mean
anti-correlates with its own held-out mean. Per-fold Spearman is exactly 0.0000, as it must
be. → `python demo.py artefact`

**Q9. So which number should I trust?**
Per-fold means for the blocked splits; pooled is fine for random and position-grouped.
The report states this and quotes per-fold in the headline table. This is also why my
first aggregation ladder reported ρ = −0.72 at 25-residue windows — the same artefact,
magnified by having only 55 units. I added permutation nulls per scale and excluded
scales with fewer than 100 units rather than reporting a fabricated effect.

---

## C. Statistics

**Q10. Why a cluster bootstrap instead of the ordinary one?**
Variants at the same position are correlated, and region-blocked folds are spatially
contiguous, so resampling individual variants would treat dependent observations as
independent and give intervals that are too narrow. Every interval resamples whole
positions (random, position-grouped) or whole folds (region, domain).
Code: [src/evaluate.py](../src/evaluate.py) `cluster_bootstrap_ci`.

**Q11. You ran a lot of comparisons. Multiple testing?**
Handled explicitly, and the honest answer moved when I re-ran it. The structural gain is
significant at 1 of 4 aggregation scales (3 aa, p = 0.011). A Bonferroni threshold for four
tests is 0.0125, so that scale now *marginally clears* correction — in an earlier run it was
p = 0.017 and did not. I did not keep the more convenient sentence. What the argument rests
on instead is that the effect is **non-monotone**: p = 0.18, 0.011, 0.89, 0.18 across the
four scales. A gain that appears, vanishes at the adjacent scale and returns is not the
signature of a real effect, and it closes only ~10% of the gap either way. Elsewhere the
headline comparisons are pre-specified (extension vs no-ESM under four splits), not
selected after the fact.

**Q12. How do you know your CV harness isn't leaking?**
Two negative controls. Label permutation gives −0.031 to +0.008 across all four splits, and
a constant predictor gives per-fold exactly 0.0000. If the harness manufactured signal,
both would be non-zero. → `results/y_randomization.csv`, `python demo.py artefact`

**Q13. Why Spearman as the primary metric?**
The score is a log2 count ratio with a +1 pseudocount, giving excess kurtosis 10.2 and a
minimum of −6.45 driven by low-count variants. Pearson would be hostage to a handful of
them. Spearman is also the ProteinGym convention, so the numbers are comparable. I report
Pearson too because the √ICC bound is a Pearson bound, and Precision@5% because the real
use is shortlisting residues to test.

---

## D. The ESM-2 work

**Q14. Strided masked marginals are an approximation. Doesn't that undercut your negative
result?**
It could, so I checked the direction it would bias. Exact masked marginals on 650M cost
1,368 forward passes = 9.0 h on this machine; masking every 20th position covers all
positions in 20 passes = 8 min. Reassurance: exact *wild-type* marginals (no approximation
at all, one forward pass) and the strided masked marginals give ρ = 0.179 and 0.180 — a
difference of 0.001. If the approximation were damaging, those two would diverge.
`src/esm_feats.validate_stride` quantifies it properly against exact masking on the 35M
model; I did not run it (68 min) and say so.

**Q15. Why is the ESM embedding "position-only"? It's a 1280-dimensional learned
representation.**
Because I embed the **wild-type** sequence once and look up row `i` for a variant at
position `i`. All ~6 substitutions at a position get the identical 1280-vector. Embedding
each mutant separately would be variant-specific, but that is 8,117 forward passes = 53 h
on this hardware. Stated as a compute limit, not hidden.

**Q16. Then your extension is crippled by design — of course it didn't help.**
Partly fair, and it is why the gradient-boosted model is there: it can interact the
position embedding with the mutant-residue features, which is the only route to
variant-specific predictions from a positional embedding. The capacity ladder then shows
`max_depth=1` is optimal and depth 10 is worse, meaning there were no useful interactions
to find. The architecture spec's `sub_proj` module is my proposed fix — a low-rank version
of the embedding × substitution interaction that gets variant specificity without
re-embedding.

**Q17. Why didn't you fine-tune ESM-2?**
Two reasons, one principled and one practical. Principled: 8,117 labels at ICC 0.063
against 650M parameters, and my capacity ladder already shows *degradation* with added
capacity on 131 features. Practical: backpropagating through 650M over 8,117 examples on
8 CPU threads with no GPU is not feasible here. The architecture spec gates it — unfreeze
layers 31–32 only if the frozen head first beats zero-shot by more than the bootstrap CI
width. On my evidence that gate does not open.

**Q17b. You only tested one language model. How do you know this isn't just ESM-2 being
bad?**
Fair — it was the softest claim in the report, so I tested it (exp8) on two axes: a
different family (ESM-1v, UniRef90, built for zero-shot variant effects) and a different
scale (ESM-2 150M).

| model | zero-shot ρ |
|---|---|
| ESM-2 650M, masked marginals | 0.180 |
| ESM-2 150M, masked marginals | 0.154 |
| ESM-1v 650M, windowed (see Q17c) | 0.073 |
| unweighted mean of all three | 0.179 |

Adding both extra models to the supervised stack: **+0.003 (p = 0.43)** for new positions,
**−0.002 (p = 0.76)** for new regions. The ensemble does not beat the best single model.
The negative result holds.

**Q17d. Scale gave +0.026 from 150M to 650M. Wouldn't a bigger model close the gap?**
No, and the arithmetic is worth having ready. Those two points give **0.042 Spearman per
decade of parameters**. Closing even a +0.10 gap needs 2.4 decades — about **238× more
parameters, ≈150 billion** — and closing the gap to the reliability ceiling needs a
model that does not and will not exist. Scale is a real effect here and a practically
irrelevant one. The two ESM-2 sizes also agree with each other at ρ = 0.71, so they are
largely reading the same signal.

**Q17c. And you found something unexpected while doing it.**
Yes — ESM-1v **cannot process this protein at all**:

```
ValueError: Sequence length 1370 above maximum sequence length of 1024
```

ESM-1v and ESM-1b inherit RoBERTa-style *learned absolute* positional embeddings, capped
at 1024 tokens. SpCas9 is 1,368 residues = 1,370 tokens with BOS/EOS. ESM-2 replaced those
with **rotary** embeddings and has no length ceiling, which is the only reason my main
experiments ran at all.

This is worth raising unprompted because it is an architecture constraint, not a nuisance:
**a 1,368-residue target silently rules out much of the pretrained-protein-model zoo.**
Anything with learned absolute positions — ESM-1b, ESM-1v, original ProtBert-BFD — needs
windowing or truncation on Cas9. I handled it with five overlapping 1,022-residue windows,
scoring each position from the window where it sits most centrally.
→ `results/plm_length_limits.json` records the probe as a machine-checkable fact.

**Q17c-follow-up. ESM-1v scored 0.073, much worse than ESM-2. So ESM-1v is worse?**
**I do not claim that, and you should not let me.** Two explanations are confounded: ESM-1v
may genuinely be worse here, or the windowing may have broken it. The evidence favours the
windowing. The worst-placed residue sits **511 residues from its window centre**, every
window is a *truncated* protein the model has no way to know is truncated, and ESM-1v's
LLRs agree with ESM-2's at only **ρ = 0.36** — against **ρ = 0.71** between the two ESM-2
sizes. Two models of one family at different scales agree far better than one model with
full context versus another with truncated context. The honest statement is that **ESM-1v
could not be evaluated fairly on this protein**, and that is a consequence of its
architecture, not a measurement of its quality.

**Q18. Removing the embedding *improved* region-level generalization. Explain that.**
+0.024 [+0.015, +0.034], p < 0.001. The embedding encodes where in the sequence a residue
sits, so the model can partly identify the region. That is useful when neighbouring
regions are in training (position-grouped) and useless-to-harmful when a contiguous block
is held out, because the fingerprint points at training regions that no longer apply.
→ `python demo.py ablate --drop esm_emb --scheme region_blocked`

---

## E. The structure work

**Q19. Why PDB 4UN3 and not an AlphaFold model?**
Because a predicted structure would let a reviewer ask whether I had smuggled a structure
*predictor* into a project whose scope excludes them. 4UN3 is experimental, 2.5 Å, and
crucially it is the **substrate-bound** complex — Cas9 with sgRNA and PAM-containing target
DNA — so I can compute distance-to-nucleic-acid and interface burial, which is where a
nuclease's mechanism lives.

**Q20. You said 4UN3 is H840A. Doesn't that invalidate the features?**
It is worth flagging and it is why I diffed the coordinate sequence against the assay
reference: exactly 1 mismatch in 1,306 modelled residues, at position 840. 4UN3 is the
catalytically-dead nickase used to trap the substrate complex. Backbone geometry is
unaffected; the 840 side-chain environment is not wild type. I normalise RSA by the
**reference** residue's maximum accessibility rather than the crystal's, so position 840 is
not distorted by an Ala/His swap.
→ `results/struct_reference_check.json`

**Q21. No secondary structure in your feature set. Why not?**
A proper assignment needs DSSP, which was not available in this environment. I first wrote
a Ca-geometry substitute, validated it (the bridge helix at residues 60-93 came out 85%
helical, which is right), and then **cut it** — it contributed nothing to a result that was
negative anyway, and it was surface area I could not defend as load-bearing. I would rather
say "no secondary structure, here is why" than defend a homemade version of a standard tool
on a feature block that did not matter.

**Q22. How do you know the structural features are right at all?**
Three independent checks, all in `verify.py`. RSA vs contact number ρ = −0.853 (buried
residues have more neighbours). The PAM-recognition arginines R1333/R1335 bury 72 and
82 Å² against nucleic acid against a dataset median of 0. Catalytic residues average
RSA 0.067 against a median of 0.225. And the features recover textbook tolerance
determinants against the position-mean score: RSA +0.246, contact number −0.216,
distance-to-RuvC +0.168, B-factor +0.154, all with the correct sign.

**Q23. Structure-only collapsed from r = 0.277 at 1 aa to 0.148 at 11 aa. Isn't that
evidence structure is useless at regional scale?**
No, and I flag it as a small-sample result rather than an information claim. At 11-residue
windows there are only 125 units and 49 features, with contiguous blocked folds — it
overfits, and there it is *significantly worse* than sequence-only (−0.321, p = 0.002),
which is what overfitting looks like, not an absence of information. The comparison that
matters holds the sample fixed: sequence-only versus sequence-plus-structure on the *same*
units, which is +0.035 at that scale (p = 0.18).

---

## F. The negative result

**Q24. How do you know this is the label's fault and not your modelling?**
Three converging lines. (i) Capacity: performance peaks at `max_depth=1` and declines
monotonically to depth 10 — more model makes it worse. (ii) Label averaging: holding
features fixed and averaging labels over wider windows lifts above-null ρ from 0.285 to
0.487, so the same features do much better once noise is averaged away. (iii) Two
physically independent positional descriptions — a 650M language model's embeddings and a
2.5 Å crystal structure — plateau in the *same* place, far below the label's own
reliability ceiling. That is what a label limit looks like, not a modelling limit.

**Q25. But half the reliable regional signal is still unexplained. So something *is*
missing.**
Yes, and I say so rather than claiming the noise story explains everything — this is the
part of my own preferred interpretation that the evidence refuted. I ruled out the cheap
artefactual explanation (window-mean score is essentially uncorrelated with coverage
proxies: |ρ| ≤ 0.10) and I ruled out static structure. What remains: per-variant
structural or dynamic effects, or the expression/translation/folding effects the selection
conflates with cleavage — which the original authors flagged themselves. I cannot separate
those with this data.

**Q25b. Give me one concrete variant where I can see this.**
Run `python demo.py predict --mutant D10A`. D10A is the canonical RuvC nickase mutation —
it abolishes cleavage of the non-target strand and is one of the two best-characterised
loss-of-function substitutions in all of Cas9. The assay puts it at the **26th
percentile**: below median, but unremarkable. Every trained model, and ESM-2 zero-shot,
ranks it at the **1st to 14th percentile** — i.e. near-dead.

Say this out loud: *the models are "wrong" in the direction biology says is right.* They
rank D10A as severely deleterious because it is a deeply conserved catalytic aspartate;
the label disagrees. When your model and 20 years of structural biology agree against the
label, the label is the thing to doubt. That single command is the most persuasive
40 seconds in the whole project.

**Q26. ρ = 0.18 sounds like a bad model. Published ProteinGym numbers are higher.**
They are, on other assays — the benchmark average is far above this. That is the point:
this particular assay is unusually low-signal. The original paper's own title claims
functional *domains*, not residues, and the authors interpreted individual variants only
after a Fisher exact test with Benjamini–Hochberg correction, aggregating to a per-residue
mutability score. ProteinGym hands you the raw unfiltered per-variant log2 fold change —
a label the original authors never claimed was individually reliable.

---

## G. The data forensics

**Q27. What is the significance of the single-nucleotide finding?**
All 8,117 substitutions are reachable by one nucleotide change from the wild-type codon,
against 41.3% expected if amino acids had been chosen freely. That is a signature, not a
coincidence: the library is error-prone PCR at a 0.18% nucleotide rate. Three consequences.
The substitution alphabet at each position is set by the codon, so radical substitutions
are systematically missing — the authors note alanine was not generated at critical sites.
Tryptophan appears as a target only 57 times because it has one codon. And `mt_aa` is not
independent of `wt_aa`, so a model using substitution identity is partly learning library
construction: features describing *only* ep-PCR accessibility score ρ = 0.058 on their own,
with no protein information at all.

**Q28. You said the metadata is wrong. Which parts, and how do you know?**
Two things, both checked against the primary paper (PMC5715146). `selection_type` is given
as "Flow cytometry"; the positive selection is actually a plate-based *ccdB* toxin survival
selection in *E. coli* US0 hisB⁻ pyrF⁻ — no flow cytometry involved. And the supplied label
is the raw per-variant log2 fold change, whereas the authors' own analysis was
significance-filtered and aggregated. Recorded in
[src/data.py](../src/data.py) `METADATA_DISCREPANCIES`.

**Q29. `DMS_score_bin` — why do you say it's useless?**
It is exactly `DMS_score > median`, reproduced to 10 decimal places (the metadata cutoff
−0.2654328586 *is* the median), and it is 4059/4058 by construction. It carries no
information beyond `DMS_score` and it is not a biologically calibrated
functional/non-functional threshold. I report AUROC on it only for leaderboard
comparability and flag it as redundant.

---

## H. If they ask you to change something live

| "What if you…" | Command | Roughly |
|---|---|---|
| used a random split instead | `python demo.py split --scheme random` | 30 s |
| showed all four splits | `python demo.py sweep` | 90 s |
| dropped the ESM embedding | `python demo.py ablate --drop esm_emb --scheme region_blocked` | 40 s |
| dropped ESM entirely | `python demo.py ablate --drop esm_lm,esm_emb` | 40 s |
| dropped the codon features | `python demo.py ablate --drop codon` | 40 s |
| proved the ceiling | `python demo.py bound` | 5 s |
| showed the artefact was real | `python demo.py artefact` | 20 s |
| looked at one residue | `python demo.py residue --pos 840` | 5 s |
| looked at one variant's prediction | `python demo.py predict --mutant D10A` | 3 s |
| re-checked every claim | `python verify.py` | 40 s |
| changed model capacity | edit `HGB(...)` in [src/models.py](../src/models.py), rerun `demo.py split` | 1 min |
| changed the number of embedding PCs | `n_emb_pcs=` in [src/features.py](../src/features.py) `add_esm_features` | 1 min |
| added a feature group | add to `FEATURE_GROUPS`, emit it in `build_handcrafted` | — |
| reran everything | `python run_all.py` | ~50 min |

**If a command fails in the room:** `verify.py` and `demo.py bound/leak/artefact/residue/
predict` need **no torch and no ESM cache** — they run on numpy/pandas/scipy/sklearn alone.
The subcommands that need the ESM cache are `split`, `sweep` and `ablate`. If the cache is
missing, `python run_all.py --skip-esm` reproduces every non-ESM result.

---

## I. Weaknesses to concede before you're asked

Volunteering these reads as confidence, not doubt. Each one is already in the report.

1. **ESM-1v could not be evaluated fairly** because of its 1,024-token ceiling, so the
   family axis is tested only imperfectly (windowed). ESM-2 3B and the 5-member ESM-1v
   ensemble were not run. The scale axis (150M vs 650M) is clean.
2. **Per-variant structure prediction is untested**, and it is the one structural avenue my
   bound does not cover.
3. **No per-variant reliability estimate exists**, because no replicates were supplied.
   Everything rests on a positional ICC.
4. **The strided masking approximation is unvalidated on 650M** — only indirectly
   reassured.
5. **The residual regional signal is unexplained.** I ruled out two candidates and named
   the rest; I do not know what it is.
6. **The three-page limit is my own line-height calculation**, not a measured render — no
   Word or LibreOffice was available to count pages.
7. **No secondary structure in the features at all** — DSSP was unavailable and I cut the
   geometric substitute rather than defend it.
8. **I went 25 minutes over the five-hour budget** (the structural follow-up), and the log
   says so rather than folding it into the original total.

---

## J. Biology you should be able to answer cold

- **What is SpCas9?** A 1,368-residue RNA-guided DNA endonuclease from *Streptococcus
  pyogenes*. A guide RNA directs it to a ~20 nt target next to a PAM (5′-NGG-3′); it makes
  a double-strand break. Two nuclease domains: **RuvC** (three segments, catalytic D10,
  E762, H983, D986) cuts the non-target strand; **HNH** (766–909, catalytic H840) cuts the
  target strand. D10A or H840A alone makes a **nickase**; both makes dead Cas9 (dCas9).
- **What did this assay measure?** Survival of *E. coli* whose growth requires Cas9 to
  cleave a plasmid carrying the *ccdB* toxin gene. Higher log2 fold change = the variant
  retained enough function to remove the toxin. It therefore conflates expression,
  translation, folding and cleavage — the authors say so explicitly.
- **Why is that a weak proxy for a gene editor?** One guide, one target (T2), one PAM,
  prokaryotic chromatin-free DNA. The authors attribute discrepancies with human-cell
  activity to DNA accessibility. It measures nothing about specificity or off-target
  behaviour, which is what actually matters for an editor.
- **Does the data recover known biology?** At domain resolution, yes, and that is
  reassuring: RuvC and HNH are the least tolerant domains, REC2 and the PAM-interacting
  domain the most (Kruskal–Wallis H = 234, p = 1.3 × 10⁻⁴⁴), which reproduces the original
  paper's conclusion. REC2 being tolerant is independently sensible — it can be deleted
  with activity retained.
- **And at residue resolution?** No, and that is the core problem. The catalytic residues
  average the **27th percentile** rather than sitting at the floor. Run
  `python demo.py residue --pos 840`: six of seven H840 substitutions are below median, but
  H840P sits at the **72nd percentile** — proline at the catalytic histidine of the HNH
  nuclease, scoring above average. That single line is the clearest evidence in the whole
  project that the per-variant labels are not reliable.
