"""EXPERIMENT 8 - does the negative result survive a SECOND pretrained model?

THE SOFT CLAIM THIS ATTACKS
    The report's central negative finding is that a pretrained protein sequence model adds
    nothing to this label that survives a non-leaky split. That claim rested on ONE
    checkpoint, ESM-2 650M. A reviewer is entitled to say: you tested one model and drew a
    conclusion about a model class.

AN ARCHITECTURAL OBSTACLE, FOUND BY RUNNING INTO IT
    The obvious second test is ESM-1v (Meier et al. 2021): same size as ESM-2 650M, but
    trained on UniRef90 and built specifically for zero-shot variant-effect prediction.
    It cannot process this protein at all.

        ValueError: Sequence length 1370 above maximum sequence length of 1024

    ESM-1v and ESM-1b inherit RoBERTa-style *learned absolute* positional embeddings,
    capped at 1024 tokens. SpCas9 is 1368 residues, so 1370 tokens with BOS/EOS. ESM-2
    replaced those with rotary embeddings and therefore has no length ceiling, which is
    the only reason the main experiments ran at all.

    This is a real constraint on the proposal, not a nuisance: **a 1368-residue target
    silently rules out a large part of the pretrained-protein-model zoo.** Anything with
    learned absolute positions - ESM-1b, ESM-1v, the original ProtBert-BFD - needs
    windowing or truncation on Cas9. It belongs in an architecture decision, so it is
    recorded here rather than quietly worked around.

WHAT IS RUN INSTEAD - two axes, both cheap
    1. ESM-1v via WINDOWING. Five overlapping 1022-residue windows cover the protein;
       every position is scored from the window in which it sits most centrally, so no
       position is scored near a truncation edge. This changes the context each residue
       sees relative to ESM-2's full-length pass, and that caveat is reported with the
       number rather than buried. Wild-type marginals only (5 passes): ESM-2 showed
       wild-type and masked marginals differ by 0.001 Spearman on this task, so
       wild-type marginals are an adequate and evidenced proxy.
    2. ESM-2 150M, full length, both scoring modes - the SCALE axis. If a 150M model
       matches a 650M model, the bottleneck is not model capacity.

    Together these test family/training-data (axis 1) and scale (axis 2).
    Not run: the 5-member ESM-1v ensemble average, and ESM-2 3B (11 GB, ~35 min).

FOUR TESTS
    A. zero-shot for every model, identical evaluation path
    B. redundancy: how correlated are the models' LLRs with each other?
    C. does ADDING a second model to the supervised stack buy anything?
    D. the length-limit finding itself, recorded as a machine-checkable fact

Outputs
    results/esm_cache/esm1v_*.npy, esm2_150M_*.npy
    results/second_plm.csv
    results/plm_length_limits.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import AAS, RESULTS, read_wt_sequence  # noqa: E402
from src.esm_feats import (  # noqa: E402
    CACHE,
    positional_entropy,
    score_variants,
)
from src.evaluate import core_metrics, paired_cluster_bootstrap_delta  # noqa: E402
from src.models import HGB, ColumnScore, run_cv  # noqa: E402
from src.splits import (  # noqa: E402
    SPLIT_BOOTSTRAP_UNIT,
    SPLIT_POSITION,
    SPLIT_REGION,
    assign_folds,
)

from experiments.exp1_protocol_and_baselines import (  # noqa: E402
    N_FOLDS,
    SEED,
    build_design_matrix,
)

ESM1V = "esm1v_t33_650M_UR90S_1"
ESM2_SMALL = "esm2_t30_150M_UR50D"
STRIDE = 20
N_WINDOWS = 5
WINDOW_LEN = 1022  # + BOS + EOS = 1024, the ESM-1v ceiling


def probe_length_limit(model_name: str, L: int = 1368, n_threads: int = 8) -> dict:
    """Record whether a checkpoint can process a sequence of this length at all.

    Run as a first-class test rather than discovered by a crash, because "this model
    cannot see your protein" is an architecture decision, not a bug.
    """
    import esm
    import torch

    torch.set_num_threads(n_threads)
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    dummy = "A" * L
    _, _, toks = alphabet.get_batch_converter()([("x", dummy)])

    # ESM-2 exposes embed_dim/num_layers on the module; the ESM-1 family
    # (ProteinBertModel, used by ESM-1b and ESM-1v) keeps them on model.args.
    def attr(*names, default=None):
        for n in names:
            if hasattr(model, n):
                return getattr(model, n)
            if hasattr(model, "args") and hasattr(model.args, n):
                return getattr(model.args, n)
        return default

    info = {
        "model": model_name,
        "tokens_required": int(toks.shape[1]),
        "num_layers": int(attr("num_layers", "layers", default=-1)),
        "embed_dim": int(attr("embed_dim", default=-1)),
        "arch_class": type(model).__name__,
    }
    try:
        with torch.no_grad():
            model(toks)
        info.update(ok=True, limit=None, note="no length ceiling (rotary embeddings)")
    except Exception as e:  # noqa: BLE001 - the failure mode IS the measurement here
        info.update(ok=False, limit=1024, note=f"{type(e).__name__}: {e}")
    del model
    return info


def _window_starts(L: int, n: int = N_WINDOWS, w: int = WINDOW_LEN) -> list[int]:
    if L <= w:
        return [0]
    return [int(round(s)) for s in np.linspace(0, L - w, n)]


def compute_windowed_wtmarg(
    model_name: str, wt: str, n_threads: int = 8
) -> tuple[np.ndarray, dict]:
    """Wild-type marginals for a length-limited model, via overlapping windows.

    Every position is taken from the window in which it is most CENTRAL, so no position is
    scored from near a truncation edge. Returns (log_probs (L,20), provenance dict).
    """
    import esm
    import torch

    torch.set_num_threads(n_threads)
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    aa_ids = np.array([alphabet.get_idx(a) for a in AAS], dtype=int)
    L = len(wt)
    starts = _window_starts(L)
    out = np.full((L, 20), np.nan, dtype=np.float32)
    best = np.full(L, np.inf)
    assigned = np.full(L, -1)

    t0 = time.time()
    for wi, s in enumerate(starts):
        seg = wt[s : s + WINDOW_LEN]
        _, _, toks = alphabet.get_batch_converter()([("w", seg)])
        with torch.no_grad():
            logits = model(toks)["logits"]
        lp = torch.log_softmax(logits[0, 1 : len(seg) + 1, :], dim=-1).numpy()[:, aa_ids]
        centre = s + len(seg) / 2
        for j in range(len(seg)):
            p = s + j
            d = abs(p - centre)
            if d < best[p]:
                best[p] = d
                out[p] = lp[j]
                assigned[p] = wi
        print(f"  [{model_name} window {wi + 1}/{len(starts)}] residues "
              f"{s + 1}-{s + len(seg)}  {time.time() - t0:.0f}s", flush=True)
    del model
    if np.isnan(out).any():
        raise RuntimeError("windows did not cover every position")
    prov = {"model": model_name, "n_windows": len(starts), "window_len": WINDOW_LEN,
            "starts_1based": [s + 1 for s in starts],
            "max_distance_from_window_centre": float(best.max()),
            "seconds": float(time.time() - t0)}
    return out, prov


def compute_full(model_name: str, wt: str, stride: int = STRIDE, n_threads: int = 8):
    """Wild-type + strided masked marginals + embeddings for a model with no length limit."""
    import esm
    import torch

    torch.set_num_threads(n_threads)
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    _, _, toks = alphabet.get_batch_converter()([("wt", wt)])
    L, layer = len(wt), model.num_layers
    aa_ids = np.array([alphabet.get_idx(a) for a in AAS], dtype=int)

    t0 = time.time()
    with torch.no_grad():
        o = model(toks, repr_layers=[layer])
    wtm = torch.log_softmax(o["logits"][0, 1 : L + 1, :], dim=-1).numpy()[:, aa_ids]
    emb = o["representations"][layer][0, 1 : L + 1, :].numpy()
    print(f"  [{model_name}] wt-marginals + embeddings in {time.time() - t0:.0f}s",
          flush=True)

    mm = np.full((L, 20), np.nan, dtype=np.float32)
    t0 = time.time()
    for k in range(stride):
        idx = np.arange(k, L, stride)
        masked = toks.clone()
        masked[0, idx + 1] = alphabet.mask_idx
        with torch.no_grad():
            lg = model(masked)["logits"]
        mm[idx] = torch.log_softmax(lg[0, 1 : L + 1, :], dim=-1).numpy()[:, aa_ids][idx]
    print(f"  [{model_name}] strided masked marginals in {time.time() - t0:.0f}s",
          flush=True)
    del model
    return wtm.astype(np.float32), mm, emb.astype(np.float32)


def main() -> None:
    import json

    wt = read_wt_sequence()
    df, _wt, X, names, groups, have_esm = build_design_matrix()
    if not have_esm:
        print("ESM-2 cache absent; run exp1 first so the models can be compared")
        return
    y = df["DMS_score"].to_numpy(float)
    pos = df["pos"].to_numpy()
    wt_aa = df["wt_aa"].to_numpy()
    mt_aa = df["mt_aa"].to_numpy()
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- test D
    print("=" * 78)
    print("D  can each checkpoint even process a 1368-residue protein?")
    print("=" * 78)
    limits = []
    for m in ["esm2_t33_650M_UR50D", ESM2_SMALL, ESM1V]:
        try:
            info = probe_length_limit(m, len(wt))
        except Exception as e:  # noqa: BLE001 - never let the probe kill the experiment
            info = {"model": m, "ok": None, "note": f"probe itself failed: "
                    f"{type(e).__name__}: {e}"}
        limits.append(info)
        print(f"  {m:24s} {info.get('tokens_required', '?')} tokens  "
              f"{'OK' if info.get('ok') else 'FAILS'}  {info['note'][:60]}")
    (RESULTS / "plm_length_limits.json").write_text(json.dumps(limits, indent=2))

    # ---------------------------------------------------------- features
    print("\n=== feature extraction ===")
    p_1v = CACHE / "esm1v_m1_windowed_wtmarg_logprobs.npy"
    if p_1v.exists():
        print(f"  using cached {ESM1V} windowed features")
        lp_1v = np.load(p_1v)
        prov = json.loads((CACHE / "esm1v_m1_windowed_provenance.json").read_text())
    else:
        lp_1v, prov = compute_windowed_wtmarg(ESM1V, wt)
        np.save(p_1v, lp_1v)
        (CACHE / "esm1v_m1_windowed_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"  ESM-1v windowing: {prov['n_windows']} windows of {prov['window_len']}, "
          f"worst position {prov['max_distance_from_window_centre']:.0f} residues "
          f"from a window centre")

    p_sm = CACHE / f"esm2_150M_maskmarg_stride{STRIDE}_logprobs.npy"
    if p_sm.exists():
        print(f"  using cached {ESM2_SMALL} features")
        sm_wt = np.load(CACHE / "esm2_150M_wtmarg_logprobs.npy")
        sm_mm = np.load(p_sm)
    else:
        sm_wt, sm_mm, sm_emb = compute_full(ESM2_SMALL, wt)
        np.save(CACHE / "esm2_150M_wtmarg_logprobs.npy", sm_wt)
        np.save(p_sm, sm_mm)
        np.save(CACHE / "esm2_150M_embeddings.npy", sm_emb)

    # ---- variant scores, identical protocol for every model -------------
    llr = {
        "esm2_650M_wtmarg": score_variants(
            np.load(CACHE / "esm2_650M_wtmarg_logprobs.npy"), pos, wt_aa, mt_aa),
        "esm2_650M_maskmarg": score_variants(
            np.load(CACHE / f"esm2_650M_maskmarg_stride{STRIDE}_logprobs.npy"),
            pos, wt_aa, mt_aa),
        "esm2_150M_wtmarg": score_variants(sm_wt, pos, wt_aa, mt_aa),
        "esm2_150M_maskmarg": score_variants(sm_mm, pos, wt_aa, mt_aa),
        "esm1v_650M_wtmarg_windowed": score_variants(lp_1v, pos, wt_aa, mt_aa),
    }

    rows = []
    print("\n" + "=" * 78)
    print("A  zero-shot, every model, identical evaluation path")
    print("=" * 78)
    for k, v in llr.items():
        m = core_metrics(y, v, df["DMS_score_bin"].to_numpy())
        rows.append({"test": "zero_shot", "model": k, **m})
        print(f"  {k:28s} rho {m['spearman']:+.4f}  r {m['pearson']:+.4f}  "
              f"AUROC {m['auroc']:.3f}  P@5% {m['precision_at_5pct']:.3f}")

    print("\n" + "=" * 78)
    print("B  redundancy: are these models saying the same thing?")
    print("=" * 78)
    ref = llr["esm2_650M_maskmarg"]
    for k in ["esm2_150M_maskmarg", "esm1v_650M_wtmarg_windowed"]:
        rho = float(spearmanr(ref, llr[k])[0])
        print(f"  ESM-2 650M vs {k:28s} LLR agreement: Spearman {rho:+.3f}  "
              f"Pearson {pearsonr(ref, llr[k])[0]:+.3f}")
        rows.append({"test": "redundancy", "model": f"esm2_650M_vs_{k}",
                     "spearman": rho, "pearson": float(pearsonr(ref, llr[k])[0]),
                     "n": len(y), "precision_at_5pct": np.nan, "auroc": np.nan})
    ens = np.mean([llr["esm2_650M_maskmarg"], llr["esm2_150M_maskmarg"],
                   llr["esm1v_650M_wtmarg_windowed"]], axis=0)
    m_ens = core_metrics(y, ens, df["DMS_score_bin"].to_numpy())
    rows.append({"test": "zero_shot", "model": "ensemble_mean_of_3", **m_ens})
    print(f"\n  mean LLR of all three models: rho {m_ens['spearman']:+.4f}  "
          f"(best single {max(core_metrics(y, v)['spearman'] for v in llr.values()):+.4f})")
    ent1 = positional_entropy(lp_1v)
    ent2 = positional_entropy(
        np.load(CACHE / f"esm2_650M_maskmarg_stride{STRIDE}_logprobs.npy"))
    print(f"  positional entropy agreement (650M vs ESM-1v): "
          f"Spearman {spearmanr(ent1, ent2)[0]:+.3f}")

    print("\n" + "=" * 78)
    print("C  does adding the other two models to the supervised stack buy anything?")
    print("=" * 78)
    extra = [llr["esm1v_650M_wtmarg_windowed"], llr["esm2_150M_wtmarg"],
             llr["esm2_150M_maskmarg"], ent1[pos - 1],
             np.array([lp_1v[p - 1].max() for p in pos]),
             np.array([sm_mm[p - 1].max() for p in pos])]
    X2 = np.column_stack([X] + [np.asarray(c, dtype=np.float32) for c in extra])
    print(f"  design matrix {X.shape} -> {X2.shape}")

    base_cols = np.arange(X.shape[1])
    both_cols = np.arange(X2.shape[1])
    for split in [SPLIT_POSITION, SPLIT_REGION]:
        folds = assign_folds(df, split, N_FOLDS, SEED)
        clusters = df["pos"].to_numpy() if SPLIT_BOOTSTRAP_UNIT[split] == "pos" else folds
        p_base = run_cv(lambda: HGB(cols=base_cols, name="esm2_only"),
                        df, X2, y, folds, seed=SEED)
        p_both = run_cv(lambda: HGB(cols=both_cols, name="three_plms"),
                        df, X2, y, folds, seed=SEED)
        m_base, m_both = core_metrics(y, p_base), core_metrics(y, p_both)
        d = paired_cluster_bootstrap_delta(y, p_both, p_base, clusters, "spearman", 500, SEED)
        print(f"  --- {split} ---")
        print(f"    ESM-2 650M only        rho {m_base['spearman']:+.4f}")
        print(f"    all three models       rho {m_both['spearman']:+.4f}")
        print(f"    delta {d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] "
              f"p={d['p_two_sided']:.3f}")
        rows.append({"test": "supervised_add", "model": f"{split}__esm2_650M_only",
                     **m_base})
        rows.append({"test": "supervised_add", "model": f"{split}__three_plms",
                     **m_both, "delta_vs_esm2": d["delta"], "delta_lo": d["lo"],
                     "delta_hi": d["hi"], "delta_p": d["p_two_sided"]})

    pd.DataFrame(rows).to_csv(RESULTS / "second_plm.csv", index=False)
    print(f"\nwrote {RESULTS / 'second_plm.csv'}")


if __name__ == "__main__":
    main()
