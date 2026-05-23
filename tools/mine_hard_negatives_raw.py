"""
Stream hard negatives directly from the raw HF Caltech images.

Unlike `tools/mine_hard_negatives.py` (which mines from the pre-built
`caltech_pool.npy`, a finite ~50M-patch snapshot), this tool taps the raw
Caltech image set inside the HF dataset and keeps drawing fresh random
patches until the target count of very-hard negatives is reached. There is
no intermediate 18-30 GB pool to materialize.

Why this matters for late-stage cascades: at stages 12+, the v2-class cbcl
cascade has cumulative FPR ~0.0001-0.001. A single pass over the 50M pool
yields only ~5-50K candidates that fool the full cascade, and even fewer
that the new training run actually finds hard. Streaming lets us scan
arbitrarily many fresh patches (10× pool size is feasible in one run)
without disk overhead.

Use this output as the `--very-hard-neg-pool` reservoir for a fresh
training run; the cascade trainer will use it ONLY as a last-resort top-up
when normal per-stage mining (seed + caltech_pool) returns fewer patches
than `--target-neg-per-stage`.

Usage:
    python tools/mine_hard_negatives_raw.py \\
        --weights weights/19/cbcl__19_v2.pkl \\
        --target 15000 \\
        --out weights/19/cbcl__19_v2__vhardneg_raw.npy
"""

import argparse
import os
import sys
import time

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_pretty_time  # noqa: E402
from violajones import ViolaJones  # noqa: E402


def mine_from_raw(clf: ViolaJones, ds_caltech, resolution: int,
                  target: int, budget: int,
                  patches_per_image: int, rng: np.random.Generator):
    """Multi-pass over the Caltech image set with random crops per image.

    For each image: load → grayscale → take `patches_per_image` random
    crops at `resolution`×`resolution` → classify → keep where classify==1.

    Stops when `target` patches have been kept OR `budget` patches have
    been classified, whichever comes first. The budget bounds wall-clock
    time even when the pool is too hard to yield `target` patches.
    """
    n_imgs = len(ds_caltech)
    print(f"Caltech source pool: {n_imgs:,} images")
    kept = []
    sampled = 0
    pass_idx = 0
    pbar = tqdm(total=target, desc="Mining (raw)", unit="kept")
    start = time.time()
    while len(kept) < target and sampled < budget:
        pass_idx += 1
        order = rng.permutation(n_imgs)
        for idx in order:
            if len(kept) >= target or sampled >= budget:
                break
            pil = ds_caltech[int(idx)]["image"]
            if pil.mode != "L":
                pil = pil.convert("L")
            iw, ih = pil.size
            if iw < resolution or ih < resolution:
                continue
            arr = np.asarray(pil, dtype=np.uint8)
            n = min(patches_per_image, budget - sampled)
            xs = rng.integers(0, iw - resolution + 1, size=n)
            ys = rng.integers(0, ih - resolution + 1, size=n)
            for x, y in zip(xs, ys):
                patch = arr[int(y):int(y) + resolution,
                            int(x):int(x) + resolution]
                sampled += 1
                if clf.classify(patch) == 1:
                    kept.append(patch.copy())
                    pbar.update(1)
                    if len(kept) >= target:
                        break
            pbar.set_postfix(
                sampled=f"{sampled:,}",
                fpr=f"{100.0 * len(kept) / max(sampled, 1):.4f}%",
                pass_=pass_idx,
            )
    pbar.close()
    elapsed = get_pretty_time(start)
    print(f"\nKept {len(kept):,} hard negatives from {sampled:,} sampled "
          f"({100.0 * len(kept) / max(sampled, 1):.4f}% FPR) "
          f"in {elapsed}, after {pass_idx} pass(es).")
    if not kept:
        return np.empty((0, resolution, resolution), dtype=np.uint8)
    return np.stack(kept)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="Path to the oracle cascade .pkl. Defines the "
                         "target resolution; raw images smaller than that "
                         "are skipped.")
    ap.add_argument("--repo-id", default="salvacarrion/face-detection",
                    help="HF dataset repo id (default: %(default)s).")
    ap.add_argument("--target", type=int, default=15000,
                    help="Target number of hard negatives to collect "
                         "(default: 15000).")
    ap.add_argument("--budget", type=int, default=500_000_000,
                    help="Hard cap on patches inspected — protects "
                         "wall-clock when the cascade is very strong and "
                         "yield is low (default: 500M).")
    ap.add_argument("--patches-per-image", type=int, default=200,
                    help="Random crops sampled per Caltech image per "
                         "pass (default: 200). Higher reduces HF/PIL "
                         "decode overhead per patch.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for crop offsets (default: 42).")
    ap.add_argument("--out", default=None,
                    help="Output .npy path. Default: "
                         "weights/<res>/<weights-stem>__vhardneg_raw.npy")
    args = ap.parse_args()

    print(f"Loading cascade: {args.weights}")
    clf = ViolaJones.load(args.weights)
    res = clf.base_width
    print(f"\t- {len(clf.clfs)} stages, base {res}×{res}")

    print(f"\nLoading HF dataset: {args.repo_id}")
    ds = load_dataset(args.repo_id)
    ds_caltech = ds["negatives"].filter(lambda x: x["source"] == "caltech")

    rng = np.random.default_rng(args.seed)
    hard = mine_from_raw(
        clf, ds_caltech, res,
        target=args.target, budget=args.budget,
        patches_per_image=args.patches_per_image, rng=rng,
    )

    if args.out:
        out_path = args.out
    else:
        stem = os.path.splitext(os.path.basename(args.weights))[0]
        out_dir = os.path.join("weights", str(res))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}__vhardneg_raw.npy")

    np.save(out_path, hard)
    print(f"\nSaved -> {out_path}")
    print(f"\nWire into training (top-up reservoir):\n"
          f"  python main.py train --data-dir <dir> "
          f"--very-hard-neg-pool {out_path} ...")

    if len(hard) < args.target:
        print(f"\nWarning: only {len(hard):,}/{args.target:,} reached. "
              f"Either raise --budget or accept a smaller pool (the "
              f"trainer uses it as a top-up — it doesn't need to be full).")


if __name__ == "__main__":
    main()
