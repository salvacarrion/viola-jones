"""Curate face crops from raw datasets for Viola-Jones training.

Reads from raw datasets on disk and writes grayscale PNG crops grouped by
source dataset. The training pipeline turns these into NPY bundles later.

Sources:
  FDDB           : Roboflow re-pack (COCO format), bbox-based tight crop +
                   10% margin. min(w, h) >= 24 px to avoid upsampling artifacts.
                   → 48x48
  CelebA         : Aligned set with 5-point landmarks. Face bbox is computed
                   from eye/mouth midpoints (eye-to-mouth distance x 2.63 ~=
                   face height). → 48x48
  CBCL           : MIT CBCL Face Database #1. Kept at native 19x19. Faces and
                   non-faces written to separate subdirs per split. → 19x19
  caltech-source : Filters raw Caltech-256 by removing face/people/human
                   categories and copies the rest into datasets/caltech/source/
                   so it can be distributed alongside the curated faces.
  caltech        : Builds a bootstrap negative pool NPY from the bundled
                   `datasets/caltech/source/` (or a custom root). Size is
                   user-configurable; larger = better hard-negative mining at
                   later cascade stages.

Usage:
  python tools/build_dataset.py fddb
  python tools/build_dataset.py celeba --n-samples 50000
  python tools/build_dataset.py cbcl
  python tools/build_dataset.py caltech-source
  python tools/build_dataset.py caltech --n-patches 1000000
  python tools/build_dataset.py stats
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED_DIR = REPO_ROOT / "datasets"
OUT_RES = 48
MARGIN = 0.10
FDDB_MIN_BBOX = 24
EYE_TO_MOUTH_TO_FACE_H = 2.63  # face_height = eye_to_mouth_dist * 2.63
FACE_CENTER_OFFSET = 0.37      # face center is 0.37 * eye_to_mouth ABOVE eye-mouth midpoint

DEFAULT_FDDB_ROOT = Path("/Users/salvacarrion/Desktop/datasets/Face detection.v1-fddb.coco")
DEFAULT_CELEBA_ROOT = Path("/Users/salvacarrion/Desktop/datasets/img_align_celeba")
DEFAULT_LANDMARKS = Path("/Users/salvacarrion/Desktop/datasets/list_landmarks_align_celeba.txt")
DEFAULT_CBCL_ROOT = Path("/Users/salvacarrion/Desktop/datasets/mitcbcl")
# Maintainer-only path: external Caltech-256 raw, used by `caltech-source` to
# build the bundled, filtered copy in datasets/caltech/source/.
DEFAULT_CALTECH_RAW_EXTERNAL = Path("/Users/salvacarrion/Desktop/datasets/256_ObjectCategories")
# Bundled, filtered Caltech root that ships with the dataset. Pool generation
# (`caltech` subcommand) and prepare_data.py both default here.
DEFAULT_CALTECH_SOURCE = CURATED_DIR / "caltech" / "source"
DEFAULT_POOL_DIR = REPO_ROOT / "bootstrap"
CBCL_RES = 19

# Caltech-256 category dirs whose names contain any of these substrings are
# skipped to avoid mining faces/people as negatives. Hits 3 categories in the
# standard 256 release: 253.faces-easy-101, 159.people, 112.human-skeleton.
CALTECH_FACE_KEYWORDS = ("face", "people", "human")


def square_crop(img_pil, cx, cy, side):
    """Square crop of `side` px centered on (cx, cy). Reflect-pads if outside."""
    half = side / 2.0
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = int(round(cx + half))
    bottom = int(round(cy + half))

    arr = np.asarray(img_pil)
    if arr.ndim == 2:
        h, w = arr.shape
        pad_spec = ((max(0, -top), max(0, bottom - h)),
                    (max(0, -left), max(0, right - w)))
    else:
        h, w = arr.shape[:2]
        pad_spec = ((max(0, -top), max(0, bottom - h)),
                    (max(0, -left), max(0, right - w)),
                    (0, 0))

    if any(v > 0 for pair in pad_spec[:2] for v in pair):
        arr = np.pad(arr, pad_spec, mode="reflect")
        left += pad_spec[1][0]
        right += pad_spec[1][0]
        top += pad_spec[0][0]
        bottom += pad_spec[0][0]

    return Image.fromarray(arr[top:bottom, left:right])


def to_gray_48(crop_pil):
    return crop_pil.convert("L").resize((OUT_RES, OUT_RES), Image.BILINEAR)


def _wipe_pngs(out_dir):
    for p in out_dir.glob("*.png"):
        p.unlink()


def extract_fddb(root):
    """Extract FDDB faces into a single `fddb/train/` directory.

    The Roboflow re-pack ships train/ and valid/ subfolders with their own
    image-id spaces. We merge both into one dataset because the train/valid
    split is arbitrary and downstream tooling (`prepare_data.py`) creates its
    own homogeneous splits. Crops from the valid/ subfolder are prefixed
    `v_` to avoid filename collisions with train/.
    """
    out_dir = CURATED_DIR / "fddb" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    _wipe_pngs(out_dir)

    items, kept, skipped_small, skipped_bad = {}, 0, 0, 0

    for src_split, fname_prefix in [("train", ""), ("valid", "v_")]:
        src_dir = root / src_split
        coco_path = src_dir / "_annotations.coco.json"
        if not coco_path.exists():
            print(f"  ! {coco_path} not found, skipping")
            continue

        with open(coco_path) as f:
            d = json.load(f)

        img_meta_by_id = {im["id"]: im for im in d["images"]}
        by_image = {}
        for ann in d["annotations"]:
            if ann["category_id"] != 1:
                continue
            by_image.setdefault(ann["image_id"], []).append(ann)

        for img_id, anns in tqdm(by_image.items(),
                                 desc=f"FDDB {src_split}", unit="img"):
            img_meta = img_meta_by_id[img_id]
            img_pil = None
            seq = 0
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if any(v in (None, 0) for v in (x, y, w, h)):
                    skipped_bad += 1
                    continue
                if min(w, h) < FDDB_MIN_BBOX:
                    skipped_small += 1
                    continue
                if img_pil is None:
                    try:
                        img_pil = Image.open(src_dir / img_meta["file_name"])
                    except (FileNotFoundError, OSError) as e:
                        skipped_bad += len(anns)
                        print(f"  ! cannot open {img_meta['file_name']}: {e}")
                        break

                cx, cy = x + w / 2.0, y + h / 2.0
                side = max(w, h) * (1.0 + MARGIN)
                try:
                    gray = to_gray_48(square_crop(img_pil, cx, cy, side))
                except (ValueError, OSError) as e:
                    skipped_bad += 1
                    print(f"  ! crop failed {img_id}/{ann['id']}: {e}")
                    continue
                fname = f"{fname_prefix}img{img_id:04d}_{seq:03d}.png"
                gray.save(out_dir / fname, optimize=True)
                items[fname] = {
                    "orig_image": img_meta["file_name"],
                    "ann_id": ann["id"],
                    "bbox_xywh": [x, y, w, h],
                    "_from_split": src_split,
                }
                kept += 1
                seq += 1

    meta = {
        "source": "FDDB-roboflow v1 (train+valid merged)",
        "params": {"resolution": OUT_RES, "min_bbox_px": FDDB_MIN_BBOX,
                   "margin": MARGIN, "color_mode": "L"},
        "n_crops": kept,
        "n_skipped_small_bbox": skipped_small,
        "n_skipped_invalid": skipped_bad,
        "items": items,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"FDDB merged: kept {kept:,}, skipped (small) {skipped_small:,}, "
          f"skipped (bad) {skipped_bad:,}")


def parse_landmarks(path):
    out = {}
    with open(path) as f:
        next(f)  # count line
        next(f)  # header
        for line in f:
            parts = line.split()
            coords = [int(v) for v in parts[1:]]
            out[parts[0]] = {
                "left_eye":    (coords[0], coords[1]),
                "right_eye":   (coords[2], coords[3]),
                "nose":        (coords[4], coords[5]),
                "left_mouth":  (coords[6], coords[7]),
                "right_mouth": (coords[8], coords[9]),
            }
    return out


def _celeba_face_bbox(lm):
    """Return (cx, cy, side) computed from 5-point landmarks."""
    le, re = lm["left_eye"], lm["right_eye"]
    lmm, rmm = lm["left_mouth"], lm["right_mouth"]
    eye_x = (le[0] + re[0]) / 2.0
    eye_y = (le[1] + re[1]) / 2.0
    mouth_x = (lmm[0] + rmm[0]) / 2.0
    mouth_y = (lmm[1] + rmm[1]) / 2.0

    eye_to_mouth = ((mouth_x - eye_x) ** 2 + (mouth_y - eye_y) ** 2) ** 0.5
    if eye_to_mouth < 5:
        return None  # degenerate landmarks (face turned 90deg or label error)

    mid_x = (eye_x + mouth_x) / 2.0
    mid_y = (eye_y + mouth_y) / 2.0
    # Vector mouth->eye points "up" along the face axis; shift midpoint that way.
    cx = mid_x + (eye_x - mouth_x) * FACE_CENTER_OFFSET
    cy = mid_y + (eye_y - mouth_y) * FACE_CENTER_OFFSET
    side = eye_to_mouth * EYE_TO_MOUTH_TO_FACE_H * (1.0 + MARGIN)
    return cx, cy, side


def extract_celeba(n_samples, root, landmarks_path, seed):
    out_dir = CURATED_DIR / "celeba" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    _wipe_pngs(out_dir)

    landmarks = parse_landmarks(landmarks_path)
    all_files = sorted(landmarks.keys())
    rng = random.Random(seed)
    sample = rng.sample(all_files, min(n_samples, len(all_files)))

    items, kept, skipped = {}, 0, 0
    for fname in tqdm(sample, desc="CelebA", unit="img"):
        lm = landmarks[fname]
        bbox = _celeba_face_bbox(lm)
        if bbox is None:
            skipped += 1
            continue
        cx, cy, side = bbox
        try:
            img_pil = Image.open(root / fname)
            gray = to_gray_48(square_crop(img_pil, cx, cy, side))
        except (FileNotFoundError, OSError, ValueError) as e:
            skipped += 1
            print(f"  ! {fname}: {e}")
            continue
        out_name = fname.replace(".jpg", ".png")
        gray.save(out_dir / out_name, optimize=True)
        items[out_name] = {
            "orig_image": fname,
            "computed_bbox": {"cx": round(cx, 1), "cy": round(cy, 1),
                              "side": round(side, 1)},
        }
        kept += 1

    meta = {
        "source": "CelebA aligned (landmark-derived bbox)",
        "params": {"resolution": OUT_RES, "margin": MARGIN,
                   "eye_to_mouth_to_face_height": EYE_TO_MOUTH_TO_FACE_H,
                   "face_center_offset": FACE_CENTER_OFFSET,
                   "color_mode": "L", "seed": seed, "n_sampled": len(sample)},
        "n_crops": kept,
        "n_skipped": skipped,
        "items": items,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"CelebA: kept {kept:,}, skipped {skipped:,}")


def stats_and_mosaic():
    """Compute per-dataset stats and 10x10 mosaics. Writes STATS.md."""
    md = ["# Curated dataset stats\n",
          "Auto-generated by `tools/build_dataset.py stats`.\n\n"]

    # Glob up to 3 levels deep to catch cbcl/train/faces, cbcl/test/nofaces, etc.
    all_meta = sorted(CURATED_DIR.glob("*/*/meta.json")) + \
               sorted(CURATED_DIR.glob("*/*/*/meta.json"))
    for meta_path in all_meta:
        ds_dir = meta_path.parent
        rel = ds_dir.relative_to(CURATED_DIR)
        with open(meta_path) as f:
            meta = json.load(f)
        pngs = sorted(ds_dir.glob("*.png"))
        pngs = [p for p in pngs if not p.name.startswith("_")]
        if not pngs:
            continue

        rng = random.Random(0)
        sample_paths = rng.sample(pngs, min(500, len(pngs)))
        s, sq, n = 0.0, 0.0, 0
        for p in sample_paths:
            arr = np.asarray(Image.open(p), dtype=np.float64)
            s += arr.sum()
            sq += (arr ** 2).sum()
            n += arr.size
        mean = s / n
        std = (sq / n - mean ** 2) ** 0.5

        # Mosaic: tile at native resolution (CBCL is 19px, others 48px)
        sample_img = Image.open(pngs[0])
        tile_res = sample_img.size[0]
        rng2 = random.Random(0)
        mosaic_n = min(100, len(pngs))
        rows = cols = int(mosaic_n ** 0.5)
        mosaic_pngs = rng2.sample(pngs, rows * cols)
        mosaic = Image.new("L", (tile_res * cols, tile_res * rows))
        for i, p in enumerate(mosaic_pngs):
            r, c = divmod(i, cols)
            mosaic.paste(Image.open(p), (c * tile_res, r * tile_res))
        mosaic_path = ds_dir / "_mosaic.png"
        mosaic.save(mosaic_path)

        md.append(f"## {rel}\n")
        md.append(f"- N crops: **{len(pngs):,}**\n")
        md.append(f"- Pixel mean / std: {mean:.1f} / {std:.1f}\n")
        md.append(f"- Source: `{meta['source']}`\n")
        md.append(f"\n![{rel}](./{rel.as_posix()}/_mosaic.png)\n\n")

    (CURATED_DIR / "STATS.md").write_text("".join(md))
    print(f"Wrote {CURATED_DIR / 'STATS.md'}")


def extract_cbcl(root):
    """Save CBCL 19x19 PNGs to datasets/cbcl/{train,test}/{faces,nofaces}/.

    Prefers the pre-bundled NPY files (faster); falls back to PGM dirs.
    Native 19x19 resolution is preserved — no upsampling.
    """
    SPLITS = ["train", "test"]
    CLASSES = {"faces": 1, "nofaces": 0}

    # Prefer NPY bundles; fall back to PGM files.
    npy_map = {
        ("train", "faces"):   root / "cbcl_train_faces_19x19g.npy",
        ("train", "nofaces"): root / "cbcl_train_nofaces_19x19g.npy",
        ("test",  "faces"):   root / "cbcl_test_faces_19x19g.npy",
        ("test",  "nofaces"): root / "cbcl_test_nofaces_19x19g.npy",
    }
    pgm_map = {
        ("train", "faces"):   root / "train" / "face",
        ("train", "nofaces"): root / "train" / "non-face",
        ("test",  "faces"):   root / "test"  / "face",
        ("test",  "nofaces"): root / "test"  / "non-face",
    }

    summary = {}
    for split in SPLITS:
        for cls in CLASSES:
            out_dir = CURATED_DIR / "cbcl" / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            _wipe_pngs(out_dir)

            npy_path = npy_map[(split, cls)]
            if npy_path.exists():
                arrays = np.load(npy_path)  # (N, 19, 19) uint8
                items = {}
                for i, arr in enumerate(tqdm(arrays, desc=f"CBCL {split}/{cls}", unit="img")):
                    img = Image.fromarray(arr.astype(np.uint8), mode="L")
                    fname = f"{i:05d}.png"
                    img.save(out_dir / fname, optimize=True)
                    items[fname] = {"source": "npy", "index": i}
                source_desc = str(npy_path.name)
            else:
                pgm_dir = pgm_map[(split, cls)]
                pgm_files = sorted(pgm_dir.glob("*.pgm")) if pgm_dir.exists() else []
                items = {}
                for pgm in tqdm(pgm_files, desc=f"CBCL {split}/{cls} (PGM)", unit="img"):
                    try:
                        img = Image.open(pgm).convert("L")
                    except OSError as e:
                        print(f"  ! {pgm.name}: {e}")
                        continue
                    fname = pgm.stem + ".png"
                    img.save(out_dir / fname, optimize=True)
                    items[fname] = {"source": pgm.name}
                source_desc = str(pgm_dir)

            meta = {
                "source": f"MIT CBCL Face DB #1 ({split}/{cls})",
                "params": {"resolution": CBCL_RES, "color_mode": "L",
                           "source_path": source_desc},
                "n_crops": len(items),
                "items": items,
            }
            with open(out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            summary[(split, cls)] = len(items)
            print(f"CBCL {split}/{cls}: {len(items):,} images")

    print("\nCBCL summary:")
    for (split, cls), n in summary.items():
        print(f"  {split:5s} / {cls:8s}: {n:,}")


def extract_caltech_source(raw_root, out_dir):
    """Copy Caltech-256 categories into `out_dir`, skipping face/people/human dirs.

    Source for the filtered, distributable Caltech bundle. Run once by the
    dataset maintainer before uploading to Hugging Face. The resulting
    directory ships alongside the curated face crops and is what
    `prepare_data.py` reads when generating the hard-negative mining pool.
    """
    import shutil

    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Raw Caltech-256 not found at {raw_root}. Pass --raw-root or "
            f"download from https://data.caltech.edu/records/nyy15-4j048")

    if out_dir.exists():
        print(f"Wiping existing {out_dir}/ for clean re-run...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    kept_cats, skipped_cats = [], []
    for cat_dir in tqdm(sorted(raw_root.iterdir()), desc="Categories", unit="cat"):
        if not cat_dir.is_dir():
            continue
        name_lower = cat_dir.name.lower()
        if any(kw in name_lower for kw in CALTECH_FACE_KEYWORDS):
            skipped_cats.append(cat_dir.name)
            continue
        shutil.copytree(cat_dir, out_dir / cat_dir.name,
                        ignore=shutil.ignore_patterns(".DS_Store", "Thumbs.db"))
        kept_cats.append(cat_dir.name)

    n_imgs = sum(1 for _ in out_dir.rglob("*")
                 if _.is_file() and _.suffix.lower() in
                 (".jpg", ".jpeg", ".png"))

    meta = {
        "source": "Caltech-256 (filtered for face-detection negatives)",
        "params": {
            "raw_root": str(raw_root),
            "excluded_keywords": list(CALTECH_FACE_KEYWORDS),
            "n_categories_kept": len(kept_cats),
            "n_categories_excluded": len(skipped_cats),
            "excluded_categories": skipped_cats,
        },
        "n_images": n_imgs,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCaltech-source: kept {len(kept_cats)} categories, "
          f"{n_imgs:,} images.")
    print(f"Excluded ({len(skipped_cats)}): {skipped_cats}")
    print(f"Written to {out_dir}/")


def build_caltech_pool(root, n_patches, resolution, output, seed):
    """Extract random non-face patches from Caltech-256 and save as a single NPY.

    The output file is NOT part of the distributable dataset zip — it can be
    several GB depending on n_patches.  Larger pools give hard-negative mining
    more room to find difficult examples at later cascade stages.

    Face-category dirs are skipped automatically (any dir whose name contains
    a word in CALTECH_FACE_KEYWORDS).
    """
    # Collect all image files, skipping face categories
    all_images = []
    skipped_cats = []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        name_lower = cat_dir.name.lower()
        if any(kw in name_lower for kw in CALTECH_FACE_KEYWORDS):
            skipped_cats.append(cat_dir.name)
            continue
        all_images.extend(cat_dir.glob("*.jpg"))
        all_images.extend(cat_dir.glob("*.png"))
        all_images.extend(cat_dir.glob("*.JPEG"))
        all_images.extend(cat_dir.glob("*.JPG"))

    print(f"Caltech-256: {len(all_images):,} images across "
          f"{len(all_images)//len(all_images)*0 + len(set(p.parent for p in all_images)):,} categories")
    print(f"Skipped face categories: {skipped_cats}")

    rng = random.Random(seed)
    rng.shuffle(all_images)

    patches_per_image = max(1, -(-n_patches // len(all_images)))  # ceil div
    print(f"Target: {n_patches:,} patches  |  ~{patches_per_image} crop(s)/image  "
          f"|  resolution: {resolution}×{resolution}")

    output.parent.mkdir(parents=True, exist_ok=True)

    pool = []
    failed = 0
    pbar = tqdm(total=n_patches, desc="Caltech pool", unit="patch")
    for img_path in all_images:
        if len(pool) >= n_patches:
            break
        try:
            img = Image.open(img_path).convert("L")
        except (OSError, Exception):
            failed += 1
            continue

        iw, ih = img.size
        if iw < resolution or ih < resolution:
            continue

        img_arr = np.asarray(img, dtype=np.uint8)
        n = min(patches_per_image, n_patches - len(pool))
        for _ in range(n):
            x = rng.randint(0, iw - resolution)
            y = rng.randint(0, ih - resolution)
            patch = img_arr[y:y + resolution, x:x + resolution]
            pool.append(patch)
            pbar.update(1)

    pbar.close()

    pool_arr = np.array(pool, dtype=np.uint8)
    np.save(output, pool_arr)
    print(f"Saved {len(pool_arr):,} patches → {output}  "
          f"({output.stat().st_size / 1e6:.0f} MB)")
    if failed:
        print(f"  ! {failed} images could not be opened")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_fddb = sub.add_parser("fddb",
        help="Extract FDDB faces (merges raw train+valid into datasets/fddb/train/)")
    p_fddb.add_argument("--root", type=Path, default=DEFAULT_FDDB_ROOT)

    p_cel = sub.add_parser("celeba")
    p_cel.add_argument("--n-samples", type=int, default=50000)
    p_cel.add_argument("--root", type=Path, default=DEFAULT_CELEBA_ROOT)
    p_cel.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    p_cel.add_argument("--seed", type=int, default=42)

    p_cbcl = sub.add_parser("cbcl")
    p_cbcl.add_argument("--root", type=Path, default=DEFAULT_CBCL_ROOT)

    p_cs = sub.add_parser("caltech-source",
        help="Copy filtered Caltech-256 raw (no faces/people/humans) into "
             "datasets/caltech/source/ for bundling with the curated dataset.")
    p_cs.add_argument("--raw-root", type=Path, default=DEFAULT_CALTECH_RAW_EXTERNAL,
                      help="External Caltech-256 root with NNN.<category>/ dirs.")
    p_cs.add_argument("--out-dir", type=Path, default=DEFAULT_CALTECH_SOURCE,
                      help="Where to write the filtered copy.")

    p_cal = sub.add_parser("caltech",
        help="Build bootstrap negative pool NPY from a Caltech root "
             "(default: datasets/caltech/source/).")
    p_cal.add_argument("--root", type=Path, default=DEFAULT_CALTECH_SOURCE,
                       help="Caltech root with NNN.<category>/ subdirs "
                            "(default: bundled datasets/caltech/source/).")
    p_cal.add_argument("--n-patches", type=int, default=1_000_000,
                       help="Total patches to extract (default: 1 000 000)")
    p_cal.add_argument("--resolution", type=int, default=48,
                       help="Crop size in pixels (default: 48)")
    p_cal.add_argument("--output", type=Path, default=None,
                       help="Output .npy path (default: bootstrap/caltech_pool_RESxRES.npy)")
    p_cal.add_argument("--seed", type=int, default=42)

    sub.add_parser("stats")

    args = p.parse_args()
    if args.cmd == "fddb":
        extract_fddb(args.root)
    elif args.cmd == "celeba":
        extract_celeba(args.n_samples, args.root, args.landmarks, args.seed)
    elif args.cmd == "cbcl":
        extract_cbcl(args.root)
    elif args.cmd == "caltech-source":
        extract_caltech_source(args.raw_root, args.out_dir)
    elif args.cmd == "caltech":
        out = args.output or DEFAULT_POOL_DIR / f"caltech_pool_{args.resolution}x{args.resolution}.npy"
        build_caltech_pool(args.root, args.n_patches, args.resolution, out, args.seed)
    elif args.cmd == "stats":
        stats_and_mosaic()


if __name__ == "__main__":
    main()
