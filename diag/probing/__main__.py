"""
Unified probing CLI — run all (or selected) analyses from a token cache.

Usage:
    python -m diag.probing --cache diag/seq_all_cache.pt --analyses all
    python -m diag.probing --cache diag/seq_all_cache.pt --analyses actor,svd
    python -m diag.probing --cache diag/seq_all_cache.pt --analyses critic --out diag/probes_out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from diag.probing._common import load_cache, numpy_from_cache, get_labels_from_cache, check_labels
from diag.probing import actor_probes, critic_probes, detr_probes, svd_probes

ALL_MODULES = {
    "actor":  actor_probes,
    "critic": critic_probes,
    "detr":   detr_probes,
    "svd":    svd_probes,
}


def _build_cache(data: dict) -> dict:
    """
    Convert loaded .pt dict into the unified cache dict expected by modules.

    Each module declares REQUIRED_KEYS and checks via can_run(cache).
    """
    import torch

    cache = {}

    # Tensors that modules may need (keep as torch tensors for SVD module)
    for key in ("seq_all", "detr_queries", "critic_cls", "critic_seq", "critic_values",
                "backbone_cls", "backbone_values"):
        if key in data:
            v = data[key]
            cache[key] = v if isinstance(v, torch.Tensor) else torch.tensor(v)

    # Numpy arrays
    for key in ("actions", "cur_hp_own", "cur_hp_opp",
                "next_hp_own", "next_hp_opp",
                "next_valid", "no_switch_valid", "opp_no_switch_valid"):
        v = numpy_from_cache(data, key)
        if v is not None:
            cache[key] = v

    # Labels (flat keys — modules expect y_return, y_win, etc. directly in cache)
    labels = get_labels_from_cache(data)
    if check_labels(labels):
        cache.update(labels)

    cache["n_transitions"] = int(data.get("n_transitions", cache["seq_all"].shape[0]))

    return cache


def main():
    parser = argparse.ArgumentParser(
        description="Run probing analyses on a token cache.",
    )
    parser.add_argument("--cache", required=True,
                        help="Path to token cache (.pt file from make_token_cache.py)")
    parser.add_argument("--analyses", default="all",
                        help="Comma-separated list of analyses to run: "
                             "all, actor, critic, detr, svd")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: same dir as cache)")
    parser.add_argument("--val_frac", type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    args = parser.parse_args()

    # Parse analysis list
    if args.analyses.strip().lower() == "all":
        requested = list(ALL_MODULES.keys())
    else:
        requested = [s.strip().lower() for s in args.analyses.split(",")]
        unknown = [r for r in requested if r not in ALL_MODULES]
        if unknown:
            parser.error(f"Unknown analyses: {unknown}. Choose from: {list(ALL_MODULES.keys())}")

    # Load cache
    cache_path = Path(args.cache)
    if not cache_path.exists():
        parser.error(f"Cache file not found: {cache_path}")

    data = load_cache(cache_path)
    cache = _build_cache(data)
    N = cache["n_transitions"]

    # Output directory
    out_dir = Path(args.out) if args.out else cache_path.parent / "probes_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Train / val split
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(N)
    n_val = max(1, int(N * args.val_frac))
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())
    print(f"Split: train={len(train_idx)}  val={len(val_idx)}  (seed={args.seed})\n")

    # Run each analysis
    results = {}
    for name in requested:
        module = ALL_MODULES[name]
        if module.can_run(cache):
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
            result = module.run(cache, train_idx, val_idx, out_dir)
            results[name] = result
        else:
            missing = module.REQUIRED_KEYS - set(cache.keys())
            print(f"[SKIP] {name}: missing keys {missing}")

    print(f"\n{'='*60}")
    print(f"Done. {len(results)}/{len(requested)} analyses completed.")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
