"""
Mine hard negatives from a raw pool using a trained Viola-Jones cascade.

The cascade's `classify(patch) == 1` flags patches that escape ALL stages —
precisely the "hard" negatives that further cascade training needs to learn
to reject. We do a single pass over the raw pool, keep every patch the
cascade misclassifies, and save them as a (N, R, R) uint8 .npy.

Why a separate tool rather than baking this into prepare_data.py:
  - A hard-neg pool is model-derived: it changes every time the cascade
    that mined it changes. prepare_data.py outputs are meant to be
    deterministic from raw HF data + flags.
  - You'll typically mine once and reuse the pool across many training
    runs (ablations, seed sweeps). Coupling to prepare_data would force
    a full re-bundle every time the reference model changes.
  - The lineage is encoded in the default output filename
    (`weights/<res>/<weights-stem>__hardneg.npy`) — you can tell which
    cascade mined a given pool just by looking at it.

Usage:
    python tools/mine_hard_negatives.py \\
        --weights weights/19/cvj_weights_<ts>.pkl \\
        --data-dir data/19

Wire the output into a training run with:
    python main.py train --data-dir data/19 \\
        --hard-neg-pool weights/19/cvj_weights_<ts>__hardneg.npy ...

Important: budget > pool_size is wasted. `classify(patch)` is deterministic,
so re-sampling the same patch returns the same result; once you've checked
every unique patch you're done. If the output is smaller than you'd like,
the fix is to regenerate `caltech_pool.npy` with a larger `--pool-size`,
NOT to crank up a budget knob (there isn't one).
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_pretty_time  # noqa: E402
from violajones import ViolaJones  # noqa: E402


def mine(clf: ViolaJones, pool: np.ndarray, batch_size: int = 10000):
    """Single pass over `pool` (uint8, shape (N, R, R)); keep patches the
    cascade misclassifies as faces.

    Batching is for memory locality when `pool` is memory-mapped — `classify`
    is per-patch internally, so batching doesn't change throughput meaningfully.
    """
    n = len(pool)
    r = pool.shape[1]
    assert clf.base_width == r and clf.base_height == r, (
        f"Cascade trained at {clf.base_width}×{clf.base_height}, "
        f"pool is {r}×{r}. Resolution mismatch.")
    keeps = []
    pbar = tqdm(total=n, desc="Mining", unit="patch")
    for i0 in range(0, n, batch_size):
        i1 = min(i0 + batch_size, n)
        # `.copy()` detaches from the memmap so we can close the file at the
        # end without dangling references; per-patch cost is ~360 bytes.
        for patch in pool[i0:i1]:
            if clf.classify(patch) == 1:
                keeps.append(patch.copy())
        pbar.update(i1 - i0)
        pbar.set_postfix(
            kept=f"{len(keeps):,}",
            fpr=f"{100.0 * len(keeps) / max(i1, 1):.3f}%",
        )
    pbar.close()
    if not keeps:
        return np.empty((0, r, r), dtype=pool.dtype)
    return np.stack(keeps)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="Path to a trained cascade .pkl. Its resolution "
                         "must match the raw pool.")
    ap.add_argument("--data-dir", required=True,
                    help="Directory containing caltech_pool.npy (used as "
                         "the default raw pool source).")
    ap.add_argument("--raw-pool", default=None,
                    help="Override path to the raw negative pool .npy "
                         "(default: <data-dir>/caltech_pool.npy).")
    ap.add_argument("--out", default=None,
                    help="Output .npy path. Default: "
                         "weights/<res>/<weights-stem>__hardneg.npy")
    ap.add_argument("--batch-size", type=int, default=10000,
                    help="Patches loaded per inner loop (default: 10,000). "
                         "Only affects memmap locality, not throughput.")
    args = ap.parse_args()

    print(f"Loading cascade: {args.weights}")
    clf = ViolaJones.load(args.weights)
    print(f"\t- {len(clf.clfs)} stages, base {clf.base_width}×{clf.base_height}")

    raw_path = args.raw_pool or os.path.join(args.data_dir, "caltech_pool.npy")
    if not os.path.exists(raw_path):
        ap.error(f"Raw pool not found: {raw_path}")
    print(f"Loading raw pool: {raw_path}")
    pool = np.load(raw_path, mmap_mode="r")
    print(f"\t- {len(pool):,} patches @ {pool.shape[1]}×{pool.shape[2]}")

    print("\nMining (single pass over pool)...")
    start = time.time()
    hard = mine(clf, pool, batch_size=args.batch_size)
    elapsed = get_pretty_time(start)

    fpr = len(hard) / max(len(pool), 1)
    print(f"\nKept {len(hard):,} hard negatives from {len(pool):,} "
          f"({100*fpr:.3f}% pool-FPR) in {elapsed}")

    if args.out:
        out_path = args.out
    else:
        res = clf.base_width
        stem = os.path.splitext(os.path.basename(args.weights))[0]
        out_dir = os.path.join("weights", str(res))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}__hardneg.npy")

    np.save(out_path, hard)
    print(f"Saved -> {out_path}")
    print(f"\nNext: python main.py train --data-dir {args.data_dir} "
          f"--hard-neg-pool {out_path} ...")

    # Sanity warnings — these are signals, not errors.
    if len(hard) == 0:
        print("\nWarning: zero hard negatives. The cascade rejects every "
              "patch in this pool — either the pool is too easy or the "
              "cascade is over-fit. Try a fresh / bigger raw pool.")
    elif fpr < 0.001:
        print(f"\nWarning: pool-FPR is very low ({100*fpr:.3f}%). The "
              f"cascade already rejects most of this pool. To get a bigger "
              f"hard pool, regenerate caltech_pool.npy with a larger "
              f"--pool-size in tools/prepare_data.py.")


if __name__ == "__main__":
    main()
