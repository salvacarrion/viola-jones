"""Build training-ready NPY bundles from the HF face-detection dataset.

Downloads `salvacarrion/face-detection` from the Hugging Face Hub (cached on
first run) and writes NPY bundles consumed directly by `main.py`. The user
picks ONE face source (celeba or fddb) and a target resolution; this script
samples from it, splits 80/10/10 train/val/test, resizes to the target
resolution, and extracts a Caltech negative pool for hard-negative mining.

CBCL test set is bundled separately (the academic benchmark) — training never
sees CBCL.

Usage (defaults give a good first run):
    python tools/prepare_data.py
        # = --face-source celeba --n-faces 10000 --resolution 24

Or explicit:
    python tools/prepare_data.py \\
        --face-source fddb \\
        --n-faces 10000 \\
        --resolution 24 \\
        --augment

Output layout:
    data/<res>/
        train_pos.npy, val_pos.npy, test_pos.npy
        caltech_pool.npy        # Hard-neg mining pool
        cbcl_test_pos.npy, cbcl_test_neg.npy   # Academic benchmark
        manifest.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_BASE = REPO_ROOT / "data"
DEFAULT_HF_REPO = "salvacarrion/face-detection"

DEFAULT_N_FACES = 10000
DEFAULT_RESOLUTION = 24
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_VAL_FRAC = 0.1
DEFAULT_POOL_SIZE = 1_000_000
DEFAULT_PATCHES_PER_IMAGE = 40


def _to_gray(pil_img: Image.Image, resolution: int) -> np.ndarray:
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    if pil_img.size != (resolution, resolution):
        pil_img = pil_img.resize((resolution, resolution), Image.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


def gather_faces(ds_train, source: str, resolution: int, n_faces: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Filter train split by source, sample n_faces, and return as (N, res, res) uint8."""
    print(f"Filtering train split by source='{source}'...")
    subset = ds_train.filter(lambda x: x["source"] == source)
    n_avail = len(subset)
    print(f"\t- {n_avail:,} {source} faces available")

    if n_faces > n_avail:
        print(f"\t! requested {n_faces:,} but only {n_avail:,} available; capping")
        n_faces = n_avail

    pick = rng.choice(n_avail, size=n_faces, replace=False)
    pick.sort()  # sequential reads are faster than random
    arr = np.empty((n_faces, resolution, resolution), dtype=np.uint8)
    for out_idx, in_idx in enumerate(tqdm(pick, desc=f"{source} crops", unit="img")):
        arr[out_idx] = _to_gray(subset[int(in_idx)]["image"], resolution)
    return arr


def split_three_way(n: int, train_frac: float, val_frac: float,
                    rng: np.random.Generator):
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def build_caltech_pool(ds_negatives, resolution: int, n_patches: int,
                       patches_per_image: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Sample random patches from Caltech color JPGs at the target resolution."""
    print(f"Building Caltech pool: {n_patches:,} patches @ {resolution}×{resolution}")
    n_imgs = len(ds_negatives)
    order = rng.permutation(n_imgs)

    pool = np.empty((n_patches, resolution, resolution), dtype=np.uint8)
    filled = 0
    skipped = 0
    pbar = tqdm(total=n_patches, desc="Caltech pool", unit="patch")

    for idx in order:
        if filled >= n_patches:
            break
        pil = ds_negatives[int(idx)]["image"]
        if pil.mode != "L":
            pil = pil.convert("L")
        iw, ih = pil.size
        if iw < resolution or ih < resolution:
            skipped += 1
            continue
        arr = np.asarray(pil, dtype=np.uint8)
        n = min(patches_per_image, n_patches - filled)
        for _ in range(n):
            x = int(rng.integers(0, iw - resolution + 1))
            y = int(rng.integers(0, ih - resolution + 1))
            pool[filled] = arr[y:y + resolution, x:x + resolution]
            filled += 1
            pbar.update(1)

    pbar.close()
    if skipped:
        print(f"\t- skipped {skipped} images smaller than {resolution}px")
    if filled < n_patches:
        print(f"\t! only filled {filled:,}/{n_patches:,} patches")
        pool = pool[:filled]
    return pool


def gather_cbcl_test(ds_test, resolution: int):
    """Return (cbcl_test_pos, cbcl_test_neg) uint8 arrays at the target resolution."""
    print("Loading CBCL test split (academic benchmark)...")
    pos_imgs, neg_imgs = [], []
    for row in tqdm(ds_test, desc="CBCL test", unit="img"):
        target = pos_imgs if row["label"] == 1 else neg_imgs
        target.append(_to_gray(row["image"], resolution))
    pos = np.stack(pos_imgs, axis=0) if pos_imgs else np.empty((0, resolution, resolution), dtype=np.uint8)
    neg = np.stack(neg_imgs, axis=0) if neg_imgs else np.empty((0, resolution, resolution), dtype=np.uint8)
    return pos, neg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face-source", choices=["celeba", "fddb"], default="celeba",
                    help="Face dataset to train on (default: celeba).")
    ap.add_argument("--n-faces", type=int, default=DEFAULT_N_FACES,
                    help=f"Faces to sample from source (default: {DEFAULT_N_FACES:,}).")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                    help=f"Target square resolution in px (default: {DEFAULT_RESOLUTION}).")
    ap.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC,
                    help=f"Train fraction (default: {DEFAULT_TRAIN_FRAC}).")
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                    help=f"Val fraction; test = 1 - train - val (default: {DEFAULT_VAL_FRAC}).")
    ap.add_argument("--augment", action="store_true",
                    help="Add horizontal-flip mirrors to training positives only.")
    ap.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                    help=f"Caltech patches to extract (default: {DEFAULT_POOL_SIZE:,}).")
    ap.add_argument("--patches-per-image", type=int, default=DEFAULT_PATCHES_PER_IMAGE,
                    help=f"Random crops per Caltech image (default: {DEFAULT_PATCHES_PER_IMAGE}).")
    ap.add_argument("--include-cbcl-test", action="store_true", default=True,
                    help="Bundle CBCL test as cbcl_test_*.npy (default: on).")
    ap.add_argument("--no-include-cbcl-test", dest="include_cbcl_test",
                    action="store_false")
    ap.add_argument("--repo-id", default=DEFAULT_HF_REPO,
                    help=f"HF dataset repo id (default: {DEFAULT_HF_REPO}).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir (default: data/<resolution>/).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac <= 0:
        ap.error(f"train_frac + val_frac must be < 1 "
                 f"(got {args.train_frac + args.val_frac}).")

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir or DEFAULT_OUT_BASE / str(args.resolution)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Preparing data @ {args.resolution}×{args.resolution} → {out_dir}/")
    print(f"   source={args.face_source}  n_faces={args.n_faces:,}  "
          f"split={args.train_frac:.2f}/{args.val_frac:.2f}/{test_frac:.2f}  "
          f"augment={args.augment}")
    print(f"   HF repo: {args.repo_id}")

    # ---- Load HF dataset (downloads on first call, cached afterwards) ----
    ds = load_dataset(args.repo_id)
    print(f"\nLoaded HF dataset:  {dict({s: len(ds[s]) for s in ds})}")

    # ---- Faces: filter, sample, resize, split ----
    faces = gather_faces(ds["train"], args.face_source, args.resolution,
                         args.n_faces, rng)
    n = len(faces)
    train_idx, val_idx, test_idx = split_three_way(
        n, args.train_frac, args.val_frac, rng)
    train_pos = faces[train_idx]
    val_pos   = faces[val_idx]
    test_pos  = faces[test_idx]

    if args.augment:
        train_pos = np.concatenate([train_pos, train_pos[:, :, ::-1]], axis=0)
        print(f"   Augmented train_pos with h-flips → {len(train_pos):,}")

    # ---- Caltech mining pool ----
    caltech_pool = build_caltech_pool(
        ds["negatives"], args.resolution, args.pool_size,
        args.patches_per_image, rng)

    # ---- CBCL benchmark (optional) ----
    cbcl_pos = cbcl_neg = None
    if args.include_cbcl_test:
        cbcl_pos, cbcl_neg = gather_cbcl_test(ds["test"], args.resolution)

    # ---- Save ----
    print("\nSaving bundles...")
    np.save(out_dir / "train_pos.npy",    train_pos)
    np.save(out_dir / "val_pos.npy",      val_pos)
    np.save(out_dir / "test_pos.npy",     test_pos)
    np.save(out_dir / "caltech_pool.npy", caltech_pool)
    if cbcl_pos is not None:
        np.save(out_dir / "cbcl_test_pos.npy", cbcl_pos)
        np.save(out_dir / "cbcl_test_neg.npy", cbcl_neg)

    manifest = {
        "resolution": args.resolution,
        "seed": args.seed,
        "face_source": args.face_source,
        "n_faces_sampled": int(n),
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": test_frac,
        "augment": bool(args.augment),
        "repo_id": args.repo_id,
        "counts": {
            "train_pos":     int(len(train_pos)),
            "val_pos":       int(len(val_pos)),
            "test_pos":      int(len(test_pos)),
            "caltech_pool":  int(len(caltech_pool)),
            "cbcl_test_pos": int(len(cbcl_pos)) if cbcl_pos is not None else 0,
            "cbcl_test_neg": int(len(cbcl_neg)) if cbcl_neg is not None else 0,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n== Done → {out_dir}/")
    for k, v in manifest["counts"].items():
        if v:
            print(f"   {k:18s}: {v:,}")
    print(f"\nNext: python main.py train --data-dir {out_dir}")


if __name__ == "__main__":
    main()
