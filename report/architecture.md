# Architecture specification

Requirement 4. **Everything in section B is a proposal and has not been executed.**
Section A is what actually ran and produced every number in the report.

Verified module names and shapes come from `fair-esm` 2.0.0 (`esm.pretrained`), inspected
directly rather than quoted from documentation.

---

## A. Executed architecture (in this repository)

### A.1 Frozen sequence trunk

| item | value |
|---|---|
| checkpoint | `esm2_t33_650M_UR50D` (Lin et al. 2023), `fair-esm==2.0.0` |
| loaded via | `esm.pretrained.esm2_t33_650M_UR50D()` |
| layers / d_model / heads | 33 / 1280 / 20 |
| vocabulary | 33 tokens; `cls_idx=0`, `padding_idx=1`, `eos_idx=2`, `mask_idx=32`; `prepend_bos=True`, `append_eos=True` |
| modules used | `embed_tokens` (Embedding 33x1280), `layers[0..32]` (`TransformerLayer`: `self_attn`, `self_attn_layer_norm`, `fc1`, `fc2`, `final_layer_norm`), `emb_layer_norm_after`, `lm_head` (`RobertaLMHead`: `dense`, `layer_norm`, weight tied to `embed_tokens`) |
| modules NOT used | `contact_head` (`ContactPredictionHead`) - it produces an L x L contact map, i.e. a position-pair feature, and is therefore subject to the same positional bound as any other wild-type-derived feature |
| gradients | none. Pure inference under `torch.no_grad()`, `model.eval()`, `torch.set_num_threads(8)` |

### A.2 Interfaces and exact shapes

```
wild-type sequence (str, L=1368)
  -> alphabet.get_batch_converter()            -> tokens          (1, 1370) int64
  -> model(tokens, repr_layers=[33])
       ['logits']                              -> (1, 1370, 33)
       ['representations'][33]                 -> (1, 1370, 1280)

slice [0, 1:L+1] to drop BOS/EOS, then select the 20 standard-AA token ids:
  log_softmax(logits)[..., aa_ids]             -> P  (1368, 20)  float32
  representations                              -> H  (1368, 1280) float32
```

`P` and `H` are cached to `results/esm_cache/*.npy` so the whole downstream pipeline
re-runs in minutes without touching torch.

Two scoring modes are computed from `P`:

- **wild-type marginals** - one forward pass of the unmasked sequence. Exact, 23.3 s.
- **masked marginals, strided** - positions `{k, k+20, k+40, ...}` are replaced with
  `mask_idx` simultaneously, so all 1368 positions are covered in 20 passes (8 min)
  rather than 1368 passes (9.0 h). Masked positions are >=20 residues apart in sequence.
  This is an approximation; `src/esm_feats.validate_stride` measures the error against
  exact single-position masking on the 35M model and is not run by default (68 min).

Variant score: `LLR(i, wt->mt) = log P[i, mt] - log P[i, wt]`.

### A.3 Feature assembly -> (8117, 131)

| group | dim | contents | constant within a position? |
|---|---|---|---|
| `subst` | 74 | 7 AA scales as wt/mt/delta/|delta|, BLOSUM62, wt and mt one-hot, Pro/Gly/Cys flags | no |
| `codon` | 5 | codon count, 1-nt neighbourhood size, minimum nt changes, number measured at position | no |
| `position` | 14 | absolute and relative position, 11 domain indicators, nuclease-domain flag | **yes** |
| `esm_lm` | 12 | per mode: LLR, log P(mt), log P(wt), positional entropy, rank of mt, max log P | LLR and rank: no; others: **yes** |
| `esm_emb` | 32 | PCA of `H` (1280 -> 32), fitted on the 1368 wild-type positions only, unsupervised | **yes** |

PCA is fitted on wild-type positions with no label involvement, so it is not a leakage
path; it is refitted identically for every fold.

### A.4 Head and objective

```
HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                              min_samples_leaf=40, l2_regularization=1.0)
RidgeCV(alphas=logspace(-1, 5, 25), cv=GroupKFold(5) grouped by position)
```

Objective: squared error on `DMS_score`. The inner ridge CV is **grouped by position**;
plain KFold there would tune the penalty against position-level leakage and
systematically under-regularise. Capacity is deliberately small - the capacity ladder in
`exp3` shows performance peaking at `max_depth=1` and falling monotonically to depth 10,
so there is nothing for a larger head to learn.

`exp5_objective.py` compares squared error on the raw score against rank-transformed and
winsorized targets, with the transform fitted on training folds only.

---

## B. Proposed architecture (NOT EXECUTED)

**Read this as a conditional, not as the recommendation.** It is the most defensible thing
to build *if the label were good enough to justify building anything* — it keeps the
sequence-model half, removes the diffusion trunk, and adds static structural features at
regional resolution.

When this section was first written that was the recommendation (MODIFY). It is no longer.
`exp6` tested the structural block and it added +0.003 (p = 0.61) per variant and closed
~10% of the regional headroom, significant at just 1 of 4 aggregation scales and
non-monotone across them; `exp8` tested whether the sequence-model finding was an artefact of one
checkpoint. The report's verdict is now **reject as specified** — ship the ESM-2 zero-shot
LLR and build none of the below against this assay. The specification is kept because the
assignment asks for a concrete architecture and because it is what I would build on a
replicated assay, and because a reader should be able to see what the evidence overturned.

### B.1 Why the diffusion structure trunk is dropped

Three independent reasons, the first two specific to this protein and label:

1. **The proposal's swap is not well posed.** In AlphaFold2/3 and the diffusion folding
   models built on them, the *trunk* (Evoformer / Pairformer) **is** the representation
   and the structure module or diffusion head is the output head. "Replacing the
   structure-prediction trunk with an activity-prediction component" removes the
   representation and keeps nothing worth transferring. What is presumably meant -
   replacing the *head* - is coherent, but its input is a pair representation of shape
   `(L, L, c_z)`. At `L=1368` and `c_z=128` that is 1368^2 x 128 x 4 bytes = **0.96 GB per
   example in fp32**, before activations or recycling. That is the concrete blocker on
   modest compute, and it does not go away with a smaller head.
2. **For single substitutions the features would collapse into the bounded channel.**
   A predicted backbone for a single point mutant is essentially the wild-type backbone.
   Any feature read off it is therefore close to constant across the ~5.9 substitutions
   measured at a position, which puts it in the position-only channel that the executed
   experiments bound at Pearson `r <= sqrt(ICC) = 0.251` against the per-variant label.
   **This bound applies to static wild-type-derived structural features. It does not
   apply to genuinely per-variant structure prediction**, which is untested here and is
   the honest gap in this argument.
3. **This label cannot referee it.** With per-variant ICC 0.063, the difference between a
   good and a bad structural representation is smaller than the label noise.

### B.2 Components, with retain / replace / freeze / train

| # | component | source | action | justification |
|---|---|---|---|---|
| 1 | `esm2_t33_650M_UR50D`, `embed_tokens` + `layers[0..30]` | pretrained | **retain, freeze** | 8117 labels at ICC 0.063 cannot supervise 650M parameters; frozen also means one cacheable forward pass |
| 2 | `layers[31..32]` | pretrained | **freeze in stage 1; unfreeze only if gated** (see B.4) | fine-tuning is the first thing to try if and only if the frozen head plateaus above baseline; current evidence says it will not |
| 3 | `lm_head` | pretrained | **retain, inference only** | supplies the LLR, the single most useful ESM feature measured |
| 4 | `contact_head` | pretrained | **drop** | position-pair feature, same bound as B.1.2 |
| 5 | Evoformer/Pairformer trunk + diffusion head | folding model | **reject** | B.1 |
| 6 | `res_proj`: Linear(1280 -> 128) + LayerNorm + GELU | new | **train** | compresses `H[i]` to a size the label budget supports |
| 7 | `sub_encoder`: the substitution's own features (which amino acid, to which, BLOSUM62, the property deltas, the ESM log-likelihood ratio) -> Linear(30 -> 32) | new | **train** | the frozen embedding is identical for all ~6 substitutions at a position, so *something* must carry which substitution it is. Re-embedding each mutant would do it properly and costs 53 h here, so the substitution is encoded directly instead. Deliberately plain: an earlier draft used a low-rank interaction between the residue embedding and the token-embedding difference, which is more elegant and which I could not justify as load-bearing |
| 8 | `struct_feats`: static per-residue features | PDB **4UN3** (SpCas9-sgRNA-DNA, 2.5 A); Shrake-Rupley SASA via Biopython, no DSSP binary required | **retain as input, no learning** | replaces the diffusion trunk at ~7 CPU-seconds instead of GPU-days. Now **executed** - see `exp6` and report section 4b |
| 9 | `window_pool`: mean-pool `struct_feats` over w in {5, 11, 25} | new | fixed, no parameters | the executed aggregation ladder puts the unexplained headroom at regional scale, so the structural signal is injected at that scale |
| 10 | `head`: MLP [209 -> 128 -> 64 -> 1], GELU, dropout 0.2 | new | **train** | ~35k parameters |
| 11 | `window_head`: Linear(128 -> 1) on pooled hidden states | new | **train** | auxiliary target at the resolution where signal is reliable |

Trainable parameter count: `res_proj` 1280x128 + `sub_encoder` 30x32 + head + window head
= **~210k**, against 8117 labels. That ratio is the point; it is roughly 1700x smaller
than fine-tuning the trunk.

### B.3 Representation shapes end to end

```
H[i]                      (1280,)   frozen ESM-2 residue representation, position i
substitution features     (30,)     which AA to which, BLOSUM62, property deltas, LLR
P[i]                      (20,)     frozen ESM-2 log-probabilities at position i
struct_feats[i]           (13,)     as built in exp6: RSA, SASA, dSASA against nucleic
                                    acid, interface flag, Cbeta contact number at 8 and
                                    12 A, min distance to nucleic acid, distances to the
                                    RuvC / HNH / PAM centres, distance to centroid,
                                    B-factor, coordinate flag (DSSP unavailable, so no
                                    secondary structure)
window_pool(struct)[i]    (36,)     12 of the above, mean-pooled at w = 5, 11, 25
z_i = [ res_proj(H[i]) ; sub_encoder(substitution features) ; struct_feats ; window_pool ]
                          (128 + 32 + 13 + 36) = (209,)
y_hat            = head(z_i)                      scalar, per-variant score
y_hat_window     = window_head(mean_{i in w} z_i) scalar, per-11-residue-window mean
```

### B.4 Training objective

```
L = L_rank(y_hat, y)  +  lambda_w * MSE(y_hat_window, window_mean(y))  +  1e-4 * ||theta||^2
lambda_w = 1.0
```

`L_rank` is a pairwise logistic (RankNet) loss over sampled variant pairs. Chosen over
squared error because the reported metric is Spearman and because the label's excess
kurtosis of 10.2 means squared error spends most of its gradient on pseudocount-driven
extremes that carry the least information.

**This is the one design choice with executed evidence behind it.** `exp5_objective.py`
swaps only the target, holding model and split fixed:

| target | new position | new region |
|---|---|---|
| squared error on raw `DMS_score` | 0.227 | 0.190 |
| squared error on training-fold **rank** | **0.252** | **0.205** |
| squared error on winsorized (1st/99th pct) | 0.235 | 0.189 |
| rank vs raw, paired cluster bootstrap | **+0.024 [+0.015, +0.035]**, p < 0.001 | **+0.015 [+0.006, +0.027]**, p < 0.001 |

The rank target gained more than every ESM-2 feature combined, which is why the objective
is specified as a ranking loss rather than regression on the raw score. The rank transform
is a monotone map fitted on training folds only, so Spearman against the true score is not
affected by the transform itself.

The window term is the part the executed evidence most directly motivates: test B of the
challenge experiment shows achieved correlation sitting 0.22-0.35 below the reliability
ceiling at every aggregation scale, i.e. the recoverable signal that is being missed lives
at regional resolution.

Optimiser AdamW, lr 3e-4 (1e-5 discriminative for any unfrozen ESM layer), weight decay
1e-2, batch 256 variants, cosine schedule, 60 epochs, early stopping on
**position-grouped** validation Spearman.

**Stage-2 gate.** Unfreeze `layers[31..32]` only if stage 1 exceeds the ESM-2 zero-shot
baseline by more than the cluster-bootstrap CI width (~0.03 Spearman) under the
region-blocked split. On the evidence in this report, that gate does not currently open.
