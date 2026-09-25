"""Reproduce everything, in order. Usage: python run_all.py [--skip-esm]

With the ESM cache already present this takes ~25 min on 8 CPU threads.
Building the ESM cache from scratch adds ~9 min plus a 2.5 GB weight download.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import RESULTS, read_wt_sequence  # noqa: E402
from src.esm_feats import CACHE  # noqa: E402

STEPS = [
    ("exp0_data_audit", "data audit (requirement 1)"),
    ("exp1_protocol_and_baselines", "protocol, baselines, extension (requirements 2-3)"),
    ("exp2_ablation", "feature-group ablation (requirement 3)"),
    ("exp3_challenge", "self-refutation experiment (requirement 3)"),
    ("exp5_objective", "training-objective comparison (requirement 4)"),
    ("exp6_structure", "static structural features from PDB 4UN3 (the follow-up test)"),
    ("exp8_second_plm", "second pretrained model (ESM-1v) - does the finding generalise?"),
    ("exp4_figures", "report figures"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-esm", action="store_true",
                    help="run without ESM features (baselines and handcrafted models only)")
    ap.add_argument("--stride", type=int, default=20,
                    help="masked-marginal stride; 1 = exact but 1368 forward passes")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.skip_esm:
        # Honoured by experiments/exp1.load_esm_blocks, so --skip-esm behaves the same way
        # whether or not a cache from an earlier run is sitting on disk.
        import os

        os.environ["CAS9_SKIP_ESM"] = "1"
        print("--skip-esm: ESM-2 features disabled for this run")

    need = CACHE / f"esm2_650M_maskmarg_stride{args.stride}_logprobs.npy"
    if not args.skip_esm and not need.exists():
        print("=" * 78)
        print("building ESM-2 650M feature cache")
        print("=" * 78)
        from src.esm_feats import build_cache

        build_cache(read_wt_sequence(), stride=args.stride, n_threads=8)

    import importlib

    for mod, desc in STEPS:
        t0 = time.time()
        print("\n" + "=" * 78)
        print(f"RUNNING {mod}  -  {desc}")
        print("=" * 78, flush=True)
        m = importlib.import_module(f"experiments.{mod}")
        m.main()
        print(f"[{mod} finished in {time.time() - t0:.0f}s]")

    print("\nall outputs in", RESULTS)


if __name__ == "__main__":
    main()
