"""Build training-ready NPY bundles from the HF face-detection dataset.

Downloads `salvacarrion/face-detection` from the Hugging Face Hub (cached on
first run) and writes NPY bundles consumed directly by `main.py`.

Data layout:
    `--face-source <X>` drives the TRAINING positives. May be augmented and
        jittered. Single or '+'-combined (e.g. 'celeba_aligned+cbcl').
    `--benchmark <Y>` (default: cbcl, or 'none' to skip eval) drives BOTH
        the validation set and the test set:
          - val_pos: `--val-size` faces sampled from the benchmark's HF train
            split, ALWAYS un-augmented (center crop only). Used at training
            time for (a) per-stage threshold calibration and (b) the
            cumulative-recall stop criterion. Anchoring val to the benchmark's
            distribution eliminates the val→test gap that augmented val sets
            introduce.
          - test_pos / test_neg: from the benchmark's HF test split,
            untouched. Used only by `main.py test`.
    `--neg-source <Z>` controls the negative pool for hard-negative mining:
          - caltech: random patches from Caltech-256 (diverse, easy)
          - benchmark: benchmark non-faces (matched, small, will exhaust)
          - mixed (default+recommended): benchmark non-faces as stage-1 seed
            + Caltech patches as the deeper mining pool

When `--face-source` overlaps with `--benchmark` (e.g. cbcl in both), val
indices are reserved BEFORE training samples are drawn — no leakage.

Output layout:
    data/<resolution>_<face_source>/
        train_pos.npy      # training positives
        val_pos.npy        # benchmark-anchored val (calibration + stop)
        test_pos.npy       # benchmark test positives (or absent if --benchmark none)
        test_neg.npy       # benchmark test negatives
        caltech_pool.npy   # mining pool for hard negatives
        neg_seed.npy       # matched-domain stage-1 seed (when neg-source != caltech)
        manifest.json

Usage:
    python tools/prepare_data.py
        # = --face-source celeba_aligned --benchmark cbcl --resolution 24 ...

    python tools/prepare_data.py \\
        --face-source celeba_aligned+cbcl --resolution 19 \\
        --augment --jitter 2 \\
        --benchmark cbcl --val-size 500 \\
        --neg-source mixed --pool-size 50000000

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 to skip the Hub freshness check on
cached runs.
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
DEFAULT_VAL_SIZE = 500
DEFAULT_POOL_SIZE = 1_000_000
DEFAULT_PATCHES_PER_IMAGE = 40


def _to_gray(pil_img: Image.Image, resolution: int) -> np.ndarray:
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")
    if pil_img.size != (resolution, resolution):
        pil_img = pil_img.resize((resolution, resolution), Image.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


def jitter_crops(faces_big: np.ndarray, target_res: int, max_shift: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Generate `2 * N` crops at `target_res` from `faces_big` (gathered at
    `target_res + 2*max_shift`). The first N crops are the centered crop;
    the next N are random shifts in `[0, 2*max_shift]` per axis, per face.

    Doubles unique-face count while preserving pixel alignment — a 1-2 px
    shift on a 19×19 face is geometrically plausible (camera jitter,
    alignment slack) and matches the `--detect-shift` of inference. Without
    jitter the cascade is brittle to sub-pixel offsets.
    """
    n, big, _ = faces_big.shape
    assert big == target_res + 2 * max_shift, (
        f"jitter_crops expects faces of size {target_res + 2*max_shift}, "
        f"got {big}")
    out = np.empty((2 * n, target_res, target_res), dtype=faces_big.dtype)
    c = max_shift
    out[:n] = faces_big[:, c:c + target_res, c:c + target_res]
    dxs = rng.integers(0, 2 * max_shift + 1, size=n)
    dys = rng.integers(0, 2 * max_shift + 1, size=n)
    for i in range(n):
        dx, dy = int(dxs[i]), int(dys[i])
        out[n + i] = faces_big[i, dy:dy + target_res, dx:dx + target_res]
    return out


def gather_faces(ds_train, source: str, resolution: int, n_faces: int,
                 rng: np.random.Generator,
                 exclude_indices=None,
                 return_indices: bool = False):
    """Sample `n_faces` from `ds_train` where source==`source` AND label==1.

    The label filter matters for sources that ship both faces and non-faces
    under the same `source` tag (MIT CBCL has 2429 faces + 4548 non-faces).

    `exclude_indices`: optional set of indices (into the filtered subset) to
    skip. Used to prevent leakage when the same source appears in both
    `--face-source` (training) and `--benchmark` (val/test) — the val draw
    happens first, marks its indices, and training excludes them.

    Returns `arr` (or `(arr, picked_indices)` if `return_indices=True`).
    Indices are into the *filtered* `source AND label==1` subset.
    """
    print(f"Filtering train split by source='{source}' AND label==1...")
    subset = ds_train.filter(lambda x: x["source"] == source and x["label"] == 1)
    n_avail = len(subset)
    print(f"\t- {n_avail:,} {source} faces available")

    if exclude_indices is not None and len(exclude_indices) > 0:
        excl_set = set(int(i) for i in exclude_indices)
        available = np.array([i for i in range(n_avail) if i not in excl_set],
                             dtype=np.int64)
        print(f"\t- excluding {len(excl_set):,} indices reserved for val "
              f"(usable: {len(available):,})")
    else:
        available = np.arange(n_avail, dtype=np.int64)

    n_pick = min(n_faces, len(available))
    if n_pick < n_faces:
        print(f"\t! requested {n_faces:,} but only {len(available):,} "
              f"usable; capping to {n_pick:,}")

    pick = rng.choice(available, size=n_pick, replace=False)
    pick.sort()  # sequential reads on the HF dataset are faster than random

    arr = np.empty((n_pick, resolution, resolution), dtype=np.uint8)
    for out_idx, in_idx in enumerate(tqdm(pick, desc=f"{source} crops",
                                          unit="img")):
        arr[out_idx] = _to_gray(subset[int(in_idx)]["image"], resolution)

    if return_indices:
        return arr, pick
    return arr


def gather_benchmark_negatives(ds_train, benchmark: str, resolution: int,
                               augment: bool = False) -> np.ndarray:
    """Non-face crops from the benchmark source's train split.

    These are the matched-domain stage-1 seed pool. For CBCL, ~4548 crops
    curated to look face-like — the cascade learns to reject benchmark-style
    non-faces from the first stage instead of waiting until hard-neg mining
    eventually finds them in random Caltech patches.
    """
    print(f"Filtering train split by source='{benchmark}' AND label==0 (non-faces)...")
    subset = ds_train.filter(lambda x: x["source"] == benchmark and x["label"] == 0)
    n = len(subset)
    print(f"\t- {n:,} {benchmark} non-faces available")
    arr = np.empty((n, resolution, resolution), dtype=np.uint8)
    for i, row in enumerate(tqdm(subset, desc=f"{benchmark} non-faces", unit="img")):
        arr[i] = _to_gray(row["image"], resolution)
    if augment:
        arr = np.concatenate([arr, arr[:, :, ::-1]], axis=0)
        print(f"\t- augmented with h-flips → {len(arr):,}")
    return arr


def _shrink_npy_first_dim(path: Path, new_n: int, chunk: int = 100_000) -> None:
    """Shrink an .npy file's first dim from N to `new_n` (`new_n < N`) by
    streaming chunks into a temp file, then renaming over the original.

    Used only when `build_caltech_pool` ends a pass with no progress before
    hitting the target — never the hot path. Avoids loading the full memmap
    into RAM (the whole point of memmapping it in the first place).
    """
    src = np.load(path, mmap_mode='r')
    if new_n >= len(src):
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    dst = np.lib.format.open_memmap(
        tmp, mode='w+', dtype=src.dtype,
        shape=(new_n,) + src.shape[1:])
    for i in range(0, new_n, chunk):
        j = min(i + chunk, new_n)
        dst[i:j] = src[i:j]
    dst.flush()
    del dst
    del src
    tmp.replace(path)


def build_caltech_pool(ds_negatives, resolution: int, n_patches: int,
                       patches_per_image: int,
                       rng: np.random.Generator,
                       out_path: Path) -> np.ndarray:
    """Sample random patches from Caltech-256 grayscale at the target resolution.

    Writes directly to `out_path` as an .npy memmap so peak RAM stays bounded
    regardless of `n_patches`. A 50M-patch pool at 24×24 is 28.8 GB and an
    in-RAM `np.empty(...)` would get OOM-killed on most laptops; the memmap
    lets the OS stream dirty pages to disk as it fills.

    Multi-pass: walks the (shuffled) image set repeatedly and pulls
    `patches_per_image` fresh random crops per image per pass. Each pass
    re-permutes and re-randomizes offsets, so duplicate patches across
    passes are unlikely (a 300×300 image has ~80k unique 19×19 positions —
    far more than we sample). Stops when `n_patches` is reached or no usable
    image remains (every image smaller than `resolution`).
    """
    size_gb = n_patches * resolution * resolution / 1e9
    print(f"Building Caltech pool: {n_patches:,} patches @ {resolution}×{resolution}")
    print(f"\t- memmap → {out_path} (~{size_gb:.1f} GB on disk)")
    ds_caltech = ds_negatives.filter(lambda x: x["source"] == "caltech")
    n_imgs = len(ds_caltech)
    print(f"\t- {n_imgs:,} Caltech source images available "
          f"(~{n_imgs * patches_per_image:,} patches/pass)")

    pool = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=np.uint8,
        shape=(n_patches, resolution, resolution))
    filled = 0
    skipped_total = 0
    pbar = tqdm(total=n_patches, desc="Caltech pool", unit="patch")

    pass_idx = 0
    while filled < n_patches:
        progress_start = filled
        order = rng.permutation(n_imgs)
        pass_idx += 1
        for idx in order:
            if filled >= n_patches:
                break
            pil = ds_caltech[int(idx)]["image"]
            if pil.mode != "L":
                pil = pil.convert("L")
            iw, ih = pil.size
            if iw < resolution or ih < resolution:
                if pass_idx == 1:
                    skipped_total += 1
                continue
            arr = np.asarray(pil, dtype=np.uint8)
            n = min(patches_per_image, n_patches - filled)
            for _ in range(n):
                x = int(rng.integers(0, iw - resolution + 1))
                y = int(rng.integers(0, ih - resolution + 1))
                pool[filled] = arr[y:y + resolution, x:x + resolution]
                filled += 1
                pbar.update(1)
        if filled == progress_start:
            print(f"\t! pass {pass_idx}: no usable images, stopping early")
            break

    pbar.close()
    pool.flush()
    if pass_idx > 1:
        print(f"\t- completed {pass_idx} passes over the image set")
    if skipped_total:
        print(f"\t- skipped {skipped_total} images smaller than {resolution}px")
    if filled < n_patches:
        print(f"\t! only filled {filled:,}/{n_patches:,} patches — shrinking file")
        del pool
        _shrink_npy_first_dim(out_path, filled)
        pool = np.load(out_path, mmap_mode='r')
    return pool


def gather_benchmark_test(ds_test, benchmark: str, resolution: int):
    """Load (test_pos, test_neg) from the benchmark's HF test split.

    The current HF dataset's "test" split is CBCL-only, so for `benchmark`
    other than 'cbcl' this may return empty arrays. When we add other
    benchmarks (FDDB, WIDER), this is where their test split would plug in.
    """
    print(f"Loading benchmark test split ('{benchmark}')...")
    pos_imgs, neg_imgs = [], []
    for row in tqdm(ds_test, desc=f"{benchmark} test", unit="img"):
        # Filter by source when the test split has one (defensive — when only
        # one benchmark lives in test, the field may be absent).
        src = row.get("source", benchmark)
        if src != benchmark:
            continue
        target = pos_imgs if row["label"] == 1 else neg_imgs
        target.append(_to_gray(row["image"], resolution))
    pos = (np.stack(pos_imgs) if pos_imgs
           else np.empty((0, resolution, resolution), dtype=np.uint8))
    neg = (np.stack(neg_imgs) if neg_imgs
           else np.empty((0, resolution, resolution), dtype=np.uint8))
    return pos, neg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- Training positives ---
    ap.add_argument("--face-source", default="celeba_aligned",
                    help="Source(s) for TRAINING positives. Single source: "
                         "'celeba', 'celeba_aligned', or 'cbcl'. Multi-source: "
                         "'+'-separated (e.g. 'celeba_aligned+cbcl'). "
                         "`--n-faces` is the per-source cap.")
    ap.add_argument("--n-faces", type=int, default=DEFAULT_N_FACES,
                    help=f"Per-source faces to sample for training "
                         f"(default: {DEFAULT_N_FACES:,}). Auto-caps to source size.")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                    help=f"Square crop side in px (default: {DEFAULT_RESOLUTION}).")
    ap.add_argument("--augment", action="store_true",
                    help="H-flip augmentation on training positives only "
                         "(val and test stay un-augmented).")
    ap.add_argument("--jitter", type=int, default=0,
                    help="Shift-jitter on training positives: max pixel shift "
                         "(default: 0). Faces gathered at (resolution + 2*jitter), "
                         "yielding TWO crops per face — center + random offset. "
                         "Critical at 19×19 to teach robustness to ±jitter px "
                         "alignment slack at inference time (sliding-window step). "
                         "Val/test never get jitter — they must match the "
                         "un-augmented benchmark distribution.")
    # --- Benchmark (val + test) ---
    ap.add_argument("--benchmark", default="cbcl",
                    help="Benchmark used for BOTH val and test (default: cbcl). "
                         "Val: `--val-size` faces sampled from the benchmark's "
                         "HF train split, un-augmented — used for per-stage "
                         "threshold calibration and cumulative-recall stopping. "
                         "Test: from the benchmark's HF test split, untouched. "
                         "Set to 'none' to skip val/test entirely (train-only).")
    ap.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE,
                    help=f"Number of validation faces sampled from the benchmark "
                         f"(default: {DEFAULT_VAL_SIZE}). Used un-augmented for "
                         f"per-stage calibration. When face-source overlaps with "
                         f"benchmark, these indices are excluded from training "
                         f"to prevent leakage.")
    # --- Negatives ---
    ap.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                    help=f"Caltech patches to extract (default: {DEFAULT_POOL_SIZE:,}).")
    ap.add_argument("--patches-per-image", type=int, default=DEFAULT_PATCHES_PER_IMAGE,
                    help=f"Random crops per Caltech image per pass "
                         f"(default: {DEFAULT_PATCHES_PER_IMAGE}). Pool builder "
                         f"loops over the image set as many times as needed to "
                         f"reach --pool-size, re-randomizing offsets each pass.")
    ap.add_argument("--neg-source", choices=["caltech", "benchmark", "mixed"],
                    default="mixed",
                    help="Negative pool source. "
                         "'caltech': random patches from Caltech-256 — diverse but easy. "
                         "'benchmark': benchmark non-faces only (matched, small, "
                         "will exhaust after a few stages). "
                         "'mixed' (recommended, default): benchmark non-faces as the "
                         "stage-1 seed (matched-domain) + Caltech patches for deeper "
                         "hard-negative mining (diverse).")
    # --- Misc ---
    ap.add_argument("--repo-id", default=DEFAULT_HF_REPO,
                    help=f"HF dataset repo id (default: {DEFAULT_HF_REPO}).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir. Default: data/<resolution>_<face-source>/ "
                         "(e.g. data/19_cbcl/, data/19_celeba_aligned+cbcl/). "
                         "Override with this flag for shorter aliases like "
                         "data/19_celeba/.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --- Validate flags ---
    valid_sources = {"celeba", "celeba_aligned", "cbcl"}
    face_sources = [s.strip() for s in args.face_source.split("+") if s.strip()]
    bad = [s for s in face_sources if s not in valid_sources]
    if bad:
        ap.error(f"unknown face source(s): {bad}. valid: {sorted(valid_sources)}")
    if not face_sources:
        ap.error("--face-source must list at least one source")

    benchmark = args.benchmark.lower() if args.benchmark else "none"
    if benchmark == "none":
        benchmark = None
    elif benchmark not in valid_sources:
        ap.error(f"unknown benchmark: {benchmark!r}. "
                 f"valid: {sorted(valid_sources)} or 'none'")

    if args.neg_source in ("benchmark", "mixed") and benchmark is None:
        ap.error(f"--neg-source {args.neg_source} requires --benchmark != none "
                 f"(needs benchmark non-faces as seed)")

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir or DEFAULT_OUT_BASE / f"{args.resolution}_{args.face_source}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Preparing data @ {args.resolution}×{args.resolution} → {out_dir}/")
    print(f"   face_source={args.face_source}  benchmark={benchmark or 'none'}  "
          f"val_size={args.val_size}  neg_source={args.neg_source}  "
          f"augment={args.augment}  jitter={args.jitter}")
    print(f"   HF repo: {args.repo_id}")

    # --- Load HF dataset (may stall briefly on Hub freshness check) ---
    print("\nLoading HF dataset...")
    ds = load_dataset(args.repo_id)
    print(f"\nLoaded HF dataset:  {dict({s: len(ds[s]) for s in ds})}")

    # =====================================================================
    # 1) VAL — sampled from benchmark FIRST so we can exclude its indices
    #          when gathering training positives (no leakage).
    # =====================================================================
    val_pos = None
    val_indices_used = None
    if benchmark is not None:
        print(f"\n[1/5] Sampling val set: {args.val_size} {benchmark} faces "
              f"(un-augmented, center crop)...")
        val_pos, val_indices_used = gather_faces(
            ds["train"], benchmark, args.resolution, args.val_size, rng,
            return_indices=True)
        print(f"   val_pos: {len(val_pos):,} faces "
              f"(used as calibration + stop-criterion anchor)")

    # =====================================================================
    # 2) TRAIN positives — from face_source(s), with augment+jitter
    #    If a face source matches `benchmark`, exclude val indices.
    # =====================================================================
    print(f"\n[2/5] Gathering training positives from {face_sources}...")
    if args.jitter > 0:
        gather_res = args.resolution + 2 * args.jitter
        print(f"   Jitter={args.jitter}: gathering at {gather_res}×{gather_res} "
              f"to enable center+shift expansion.")
    else:
        gather_res = args.resolution

    parts = []
    for src in face_sources:
        excl = val_indices_used if (src == benchmark and val_indices_used is not None) else None
        arr = gather_faces(ds["train"], src, gather_res, args.n_faces, rng,
                           exclude_indices=excl)
        print(f"   {src}: {len(arr):,} faces")
        parts.append(arr)
    train_pos = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    if len(face_sources) > 1:
        perm = rng.permutation(len(train_pos))
        train_pos = train_pos[perm]
        print(f"   combined+shuffled: {len(train_pos):,} faces across "
              f"{len(face_sources)} sources")

    if args.jitter > 0:
        train_pos = jitter_crops(train_pos, args.resolution, args.jitter, rng)
        print(f"   Jitter expansion: → {len(train_pos):,} (center + shifted)")
    if args.augment:
        train_pos = np.concatenate([train_pos, train_pos[:, :, ::-1]], axis=0)
        print(f"   H-flip augment: → {len(train_pos):,}")

    # =====================================================================
    # 3) NEGATIVE POOLS — Caltech for mining + optional benchmark seed
    # =====================================================================
    print(f"\n[3/5] Building negative pools (neg-source={args.neg_source})...")
    neg_seed = None
    caltech_pool = None
    caltech_pool_path = out_dir / "caltech_pool.npy"
    if args.neg_source == "caltech":
        caltech_pool = build_caltech_pool(
            ds["negatives"], args.resolution, args.pool_size,
            args.patches_per_image, rng, out_path=caltech_pool_path)
    elif args.neg_source == "benchmark":
        bench_neg = gather_benchmark_negatives(
            ds["train"], benchmark, args.resolution, augment=args.augment)
        # Use the same array as both seed and the main mining pool. It'll
        # exhaust after a few stages — expected for `benchmark` mode.
        neg_seed = bench_neg
        caltech_pool = bench_neg
    elif args.neg_source == "mixed":
        neg_seed = gather_benchmark_negatives(
            ds["train"], benchmark, args.resolution, augment=args.augment)
        caltech_pool = build_caltech_pool(
            ds["negatives"], args.resolution, args.pool_size,
            args.patches_per_image, rng, out_path=caltech_pool_path)

    # =====================================================================
    # 4) TEST — from benchmark test split (untouched)
    # =====================================================================
    test_pos = test_neg = None
    if benchmark is not None:
        print(f"\n[4/5] Loading test set ({benchmark} test split)...")
        test_pos, test_neg = gather_benchmark_test(
            ds["test"], benchmark, args.resolution)
        print(f"   test_pos: {len(test_pos):,} faces, "
              f"test_neg: {len(test_neg):,} non-faces")
    else:
        print("\n[4/5] Skipping test set (--benchmark none)")

    # =====================================================================
    # 5) SAVE bundles + manifest
    # =====================================================================
    print(f"\n[5/5] Saving bundles to {out_dir}/...")
    np.save(out_dir / "train_pos.npy", train_pos)
    if val_pos is not None:
        np.save(out_dir / "val_pos.npy", val_pos)
    elif (out_dir / "val_pos.npy").exists():
        (out_dir / "val_pos.npy").unlink()  # stale from previous run
    # build_caltech_pool memmaps directly to disk; only save here when the
    # pool is an in-RAM array (neg-source=benchmark, where pool == bench_neg).
    if not isinstance(caltech_pool, np.memmap):
        np.save(out_dir / "caltech_pool.npy", caltech_pool)
    neg_seed_path = out_dir / "neg_seed.npy"
    if neg_seed is not None:
        np.save(neg_seed_path, neg_seed)
    elif neg_seed_path.exists():
        neg_seed_path.unlink()
    if test_pos is not None and len(test_pos) > 0:
        np.save(out_dir / "test_pos.npy", test_pos)
        np.save(out_dir / "test_neg.npy", test_neg)
    else:
        for p in (out_dir / "test_pos.npy", out_dir / "test_neg.npy"):
            if p.exists():
                p.unlink()

    manifest = {
        "resolution": args.resolution,
        "seed": args.seed,
        "face_source": args.face_source,
        "face_sources_parsed": face_sources,
        "benchmark": benchmark,
        "val_size": int(args.val_size) if benchmark else 0,
        "neg_source": args.neg_source,
        "n_faces_per_source": args.n_faces,
        "augment": bool(args.augment),
        "jitter": int(args.jitter),
        "repo_id": args.repo_id,
        "counts": {
            "train_pos":    int(len(train_pos)),
            "val_pos":      int(len(val_pos)) if val_pos is not None else 0,
            "test_pos":     int(len(test_pos)) if test_pos is not None else 0,
            "test_neg":     int(len(test_neg)) if test_neg is not None else 0,
            "caltech_pool": int(len(caltech_pool)),
            "neg_seed":     int(len(neg_seed)) if neg_seed is not None else 0,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n== Done → {out_dir}/")
    for k, v in manifest["counts"].items():
        if v:
            print(f"   {k:14s}: {v:,}")
    print(f"\nNext: python main.py train --data-dir {out_dir}")


if __name__ == "__main__":
    main()
