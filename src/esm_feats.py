"""ESM-2 feature extraction for single substitutions in one fixed background sequence.

Compute budget drove every choice here, and the choices are stated rather than hidden.

Measured on this machine (8 CPU threads, no GPU, L=1368):
    esm2_t12_35M_UR50D   2.97 s / forward pass
    esm2_t33_650M_UR50D  23.66 s / forward pass

That rules out two things a GPU would make routine:
  * exact masked-marginals on 650M  = 1368 passes = 9.0 h
  * re-embedding every mutant       = 8117 passes = 53 h

What is computed instead:

1. wt_marginals     - one forward pass of the unmasked wild type. Exact, 24 s.
                      Scores a variant as log p(mt | WT context) - log p(wt | WT context).
2. masked_marginals - the ProteinGym-preferred scoring mode, approximated by masking
                      every `stride`-th position in a single pass, so all L positions
                      are covered in `stride` passes instead of L. With stride 20 this
                      is 20 passes = 8 min on 650M. Masked positions are >= 20 residues
                      apart in sequence. This is an approximation: `validate_stride`
                      quantifies the error against exact single-position masking on 35M.
3. embeddings       - final-layer per-residue representation of the wild type,
                      shape (L, d). NOTE: this is a function of position only. It is
                      identical for all ~6 substitutions measured at a position, so it
                      cannot separate them. That property is what the ICC bound in
                      `data.position_icc` applies to.

Nothing in this file touches protein structure or any diffusion model.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .data import AAS, REPO

CACHE = REPO / "results" / "esm_cache"

MODELS = {
    "esm2_650M": "esm2_t33_650M_UR50D",
    "esm2_35M": "esm2_t12_35M_UR50D",
}


def _load(model_key: str, n_threads: int = 8):
    import esm
    import torch

    torch.set_num_threads(n_threads)
    model, alphabet = getattr(esm.pretrained, MODELS[model_key])()
    model.eval()
    return model, alphabet


def _aa_token_ids(alphabet) -> np.ndarray:
    """Token ids of the 20 standard amino acids, in the order of `data.AAS`."""
    return np.array([alphabet.get_idx(a) for a in AAS], dtype=int)


def compute_wt_marginals(
    wt: str, model_key: str = "esm2_650M", n_threads: int = 8
) -> dict[str, np.ndarray]:
    """Single forward pass of the unmasked wild type.

    Returns log_probs of shape (L, 20) over `data.AAS` and the final-layer
    representation of shape (L, d).
    """
    import torch

    model, alphabet = _load(model_key, n_threads)
    _, _, toks = alphabet.get_batch_converter()([("wt", wt)])
    layer = model.num_layers
    t0 = time.time()
    with torch.no_grad():
        out = model(toks, repr_layers=[layer])
    elapsed = time.time() - t0

    # strip the prepended BOS token; ESM-2 also appends EOS
    logits = out["logits"][0, 1 : len(wt) + 1, :]
    log_probs = torch.log_softmax(logits, dim=-1).numpy()[:, _aa_token_ids(alphabet)]
    reps = out["representations"][layer][0, 1 : len(wt) + 1, :].numpy()
    return {
        "log_probs": log_probs.astype(np.float32),
        "embeddings": reps.astype(np.float32),
        "seconds": np.array([elapsed]),
    }


def compute_masked_marginals(
    wt: str,
    model_key: str = "esm2_650M",
    stride: int = 20,
    n_threads: int = 8,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Strided multi-mask approximation to masked-marginals.

    Pass `k` masks positions {k, k+stride, k+2*stride, ...} simultaneously and reads the
    predictive distribution at each masked position. `stride` passes cover every position.
    Set stride=1 to recover exact single-position masking (L passes).
    """
    import torch

    model, alphabet = _load(model_key, n_threads)
    _, _, toks = alphabet.get_batch_converter()([("wt", wt)])
    L = len(wt)
    aa_ids = _aa_token_ids(alphabet)
    out_lp = np.full((L, 20), np.nan, dtype=np.float32)

    t0 = time.time()
    for k in range(stride):
        idx = np.arange(k, L, stride)
        if idx.size == 0:
            continue
        masked = toks.clone()
        masked[0, idx + 1] = alphabet.mask_idx  # +1 for BOS
        with torch.no_grad():
            logits = model(masked)["logits"]
        lp = torch.log_softmax(logits[0, 1 : L + 1, :], dim=-1).numpy()[:, aa_ids]
        out_lp[idx] = lp[idx]
        if verbose:
            print(
                f"  [{model_key} stride={stride}] pass {k + 1}/{stride} "
                f"({idx.size} positions) {time.time() - t0:.0f}s",
                flush=True,
            )
    if np.isnan(out_lp).any():
        raise RuntimeError("some positions were never unmasked")
    return {"log_probs": out_lp, "seconds": np.array([time.time() - t0])}


def score_variants(
    log_probs: np.ndarray, positions: np.ndarray, wt_aas: np.ndarray, mt_aas: np.ndarray
) -> np.ndarray:
    """Log-likelihood ratio score log p(mt) - log p(wt) at the mutated position."""
    ai = {a: i for i, a in enumerate(AAS)}
    rows = positions - 1
    return np.array(
        [
            log_probs[r, ai[m]] - log_probs[r, ai[w]]
            for r, w, m in zip(rows, wt_aas, mt_aas)
        ],
        dtype=np.float32,
    )


def positional_entropy(log_probs: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of the model's per-position distribution over the 20 AAs.

    A conservation proxy that is a function of position only.
    """
    lp = log_probs - log_probs.max(axis=1, keepdims=True)
    p = np.exp(lp)
    p /= p.sum(axis=1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1).astype(np.float32)


def build_cache(wt: str, stride: int = 20, n_threads: int = 8) -> dict[str, Path]:
    """Compute and cache every ESM feature block used downstream."""
    CACHE.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    print("[1/2] esm2_650M wild-type marginals + embeddings (1 forward pass)", flush=True)
    wtm = compute_wt_marginals(wt, "esm2_650M", n_threads)
    np.save(CACHE / "esm2_650M_wtmarg_logprobs.npy", wtm["log_probs"])
    np.save(CACHE / "esm2_650M_embeddings.npy", wtm["embeddings"])
    print(f"      done in {wtm['seconds'][0]:.1f}s  "
          f"log_probs {wtm['log_probs'].shape}  emb {wtm['embeddings'].shape}", flush=True)
    written["wtmarg"] = CACHE / "esm2_650M_wtmarg_logprobs.npy"
    written["emb"] = CACHE / "esm2_650M_embeddings.npy"

    print(f"[2/2] esm2_650M strided masked marginals (stride={stride})", flush=True)
    mm = compute_masked_marginals(wt, "esm2_650M", stride, n_threads)
    np.save(CACHE / f"esm2_650M_maskmarg_stride{stride}_logprobs.npy", mm["log_probs"])
    print(f"      done in {mm['seconds'][0]:.0f}s", flush=True)
    written["maskmarg"] = CACHE / f"esm2_650M_maskmarg_stride{stride}_logprobs.npy"
    return written


def validate_stride(wt: str, stride: int = 20, n_threads: int = 8) -> dict[str, float]:
    """Quantify the strided-masking approximation error on the 35M model.

    Exact single-position masking on 35M costs 1368 x 2.97 s = 68 min, which is affordable
    once; the same comparison on 650M would cost 9 h and is not run. The reported
    agreement is therefore evidence about the approximation itself, measured on a smaller
    model of the same family.
    """
    from scipy.stats import pearsonr, spearmanr

    CACHE.mkdir(parents=True, exist_ok=True)
    approx = compute_masked_marginals(wt, "esm2_35M", stride, n_threads, verbose=False)
    np.save(CACHE / f"esm2_35M_maskmarg_stride{stride}_logprobs.npy", approx["log_probs"])
    exact = compute_masked_marginals(wt, "esm2_35M", 1, n_threads, verbose=True)
    np.save(CACHE / "esm2_35M_maskmarg_stride1_logprobs.npy", exact["log_probs"])
    a, e = approx["log_probs"].ravel(), exact["log_probs"].ravel()
    return {
        "stride": stride,
        "pearson_logprob": float(pearsonr(a, e)[0]),
        "spearman_logprob": float(spearmanr(a, e)[0]),
        "mean_abs_diff_nats": float(np.abs(a - e).mean()),
        "approx_seconds": float(approx["seconds"][0]),
        "exact_seconds": float(exact["seconds"][0]),
    }
