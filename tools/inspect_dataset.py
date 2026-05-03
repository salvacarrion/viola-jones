"""
Dump a visual sample of the HF face-detection dataset to ~/Desktop for
manual inspection.

Layout:
    ~/Desktop/face-detection-inspection/
        README.md
        cbcl/train/faces/        + _grid.png
        cbcl/train/non_faces/    + _grid.png
        cbcl/test/faces/         + _grid.png
        cbcl/test/non_faces/     + _grid.png
        celeba/train/faces/      + _grid.png
        fddb/train/faces/        + _grid.png
        caltech/negatives/       + _grid.png    (full photos for hard-neg mining)

Each leaf folder has up to N samples (default 200) and a `_grid.png`
mosaic for at-a-glance review.
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


GRID_COLS = 10
GRID_ROWS = 10
GRID_CELL_PX = 96
GRID_PAD_PX = 2


def make_grid(images, cols=GRID_COLS, max_imgs=GRID_COLS * GRID_ROWS,
              cell=GRID_CELL_PX, pad=GRID_PAD_PX):
    images = images[:max_imgs]
    rows = (len(images) + cols - 1) // cols
    w = cols * cell + (cols + 1) * pad
    h = rows * cell + (rows + 1) * pad
    canvas = Image.new("RGB", (w, h), (32, 32, 32))
    for i, img in enumerate(images):
        r, c = i // cols, i % cols
        x = pad + c * (cell + pad)
        y = pad + r * (cell + pad)
        thumb = img.convert("RGB").copy()
        thumb.thumbnail((cell, cell), Image.BILINEAR)
        ox = x + (cell - thumb.width) // 2
        oy = y + (cell - thumb.height) // 2
        canvas.paste(thumb, (ox, oy))
    return canvas


def dump_bucket(ds_split, indices, out_dir: Path, label: str,
                n_samples: int, rng: np.random.Generator):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_avail = len(indices)
    take = min(n_samples, n_avail)
    if take == 0:
        return 0
    pick = rng.choice(n_avail, size=take, replace=False)
    pick.sort()  # sequential reads are faster for arrow
    chosen_idxs = [int(indices[int(i)]) for i in pick]

    images = []
    for i, idx in enumerate(tqdm(chosen_idxs, desc=label, unit="img", leave=False)):
        img = ds_split[idx]["image"]
        img.save(out_dir / f"{i:04d}.png")
        images.append(img)

    if images:
        make_grid(images).save(out_dir / "_grid.png")
    return n_avail


def bucket_indices(ds_split):
    """Return dict: (source, label) -> list of row indices in this split."""
    sources = ds_split["source"]
    labels = ds_split["label"]
    out = {}
    for i, (s, l) in enumerate(zip(sources, labels)):
        out.setdefault((s, l), []).append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=200,
                    help="Samples per leaf folder (default: 200).")
    ap.add_argument("--out", type=Path,
                    default=Path.home() / "Desktop" / "face-detection-inspection")
    ap.add_argument("--repo-id", default="salvacarrion/face-detection")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.repo_id} ...")
    ds = load_dataset(args.repo_id)
    train_idx = bucket_indices(ds["train"])
    test_idx = bucket_indices(ds["test"])
    neg_idx = bucket_indices(ds["negatives"])

    counts = {
        "train": {f"{s} (label={l})": len(v) for (s, l), v in sorted(train_idx.items())},
        "test":  {f"{s} (label={l})": len(v) for (s, l), v in sorted(test_idx.items())},
        "negatives": {f"{s} (label={l})": len(v) for (s, l), v in sorted(neg_idx.items())},
    }
    print("\nDataset breakdown:")
    for split, d in counts.items():
        print(f"  {split}:")
        for k, v in d.items():
            print(f"    {k:30s} {v:>10,}")

    plan = [
        # (split_obj, indices_for_bucket, out_path, label)
        (ds["train"], train_idx.get(("celeba", 1), []),
         out / "celeba" / "train" / "faces", "celeba/train/faces"),
        (ds["train"], train_idx.get(("fddb",   1), []),
         out / "fddb" / "train" / "faces", "fddb/train/faces"),
        (ds["train"], train_idx.get(("cbcl",   1), []),
         out / "cbcl" / "train" / "faces", "cbcl/train/faces"),
        (ds["train"], train_idx.get(("cbcl",   0), []),
         out / "cbcl" / "train" / "non_faces", "cbcl/train/non_faces"),
        (ds["test"],  test_idx.get(("cbcl",    1), []),
         out / "cbcl" / "test" / "faces", "cbcl/test/faces"),
        (ds["test"],  test_idx.get(("cbcl",    0), []),
         out / "cbcl" / "test" / "non_faces", "cbcl/test/non_faces"),
        (ds["negatives"], neg_idx.get(("caltech", 0), []),
         out / "caltech" / "negatives", "caltech/negatives"),
    ]

    print(f"\nDumping samples to {out}/")
    for ds_split, idxs, path, label in plan:
        if not idxs:
            print(f"  - {label}: empty, skipped")
            continue
        n_avail = dump_bucket(ds_split, idxs, path, label, args.n_samples, rng)
        n_taken = min(args.n_samples, n_avail)
        print(f"  - {label}: {n_taken}/{n_avail:,} dumped")

    # ---- README ----
    readme = out / "README.md"
    lines = [
        "# face-detection — inspection dump",
        "",
        f"Source: `{args.repo_id}` (HF Hub).",
        f"Sampled `{args.n_samples}` images per leaf folder. Full counts below.",
        "",
        "## What this folder is",
        "",
        "A visual sample of the HF dataset organized by `dataset/split/category` so",
        "you can eyeball whether the crops, labels and resolutions look right.",
        "Each leaf folder has up to N PNGs plus a `_grid.png` mosaic.",
        "",
        "## HF dataset structure (the source we sampled from)",
        "",
        "The HF dataset has 3 splits — note these are **NOT** train/val/test in the",
        "ML sense. They are *roles* the data plays in Viola-Jones training:",
        "",
        "| HF split    | What it is                                | Used for                                         |",
        "|-------------|-------------------------------------------|--------------------------------------------------|",
        "| `train`     | Pool of face crops (some non-faces too)   | Training positives (and CBCL non-faces optional) |",
        "| `test`      | CBCL benchmark (faces + non-faces)        | Final evaluation only                            |",
        "| `negatives` | Caltech-256 full photos (RGB)             | Pool to mine random patches as hard negatives    |",
        "",
        "Inside `train`, content is mixed by `source`. Only CBCL ships both",
        "labels (faces AND non-faces). CelebA and FDDB are face-only.",
        "",
        "## Counts (full dataset, not just sampled)",
        "",
    ]
    for split, d in counts.items():
        lines.append(f"### `{split}` split")
        lines.append("")
        lines.append("| source — label | count |")
        lines.append("|---|---:|")
        for k, v in d.items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")

    lines += [
        "## Folder layout in this dump",
        "",
        "```",
        "face-detection-inspection/",
        "├── cbcl/",
        "│   ├── train/",
        "│   │   ├── faces/        ← positives, source for matched-domain training",
        "│   │   └── non_faces/    ← curated face-like patches (Option B seed)",
        "│   └── test/",
        "│       ├── faces/        ← benchmark positives (472 total)",
        "│       └── non_faces/    ← benchmark negatives (23,573 total)",
        "├── celeba/train/faces/   ← face-only source",
        "├── fddb/train/faces/     ← face-only source",
        "└── caltech/negatives/    ← full RGB photos for hard-neg mining",
        "```",
        "",
        "## Native resolutions",
        "",
        "- `train` crops: 48×48 grayscale (CelebA/FDDB/CBCL all resized here)",
        "- `test` crops: 19×19 grayscale (CBCL benchmark native size)",
        "- `negatives`: variable RGB photos (Caltech-256 originals)",
        "",
        "## How the V-J pipeline uses each chunk",
        "",
        "- **Positives**: `train/<source>/faces/` filtered by `--face-source`, split",
        "  80/10/10 by `prepare_data.py` into train/val/test internal splits.",
        "- **Negatives (Option A, current default)**: random patches cropped from",
        "  `caltech/negatives/`. Diverse but easy — never sees face-like patterns.",
        "- **Negatives (Option B, matched)**: `cbcl/train/non_faces/` as the seed pool",
        "  for stage 1 (matched-domain), Caltech for hard-neg mining in deeper stages.",
        "- **Evaluation**: `cbcl/test/faces` + `cbcl/test/non_faces` (the benchmark).",
        "",
    ]
    readme.write_text("\n".join(lines))
    print(f"\nREADME -> {readme}")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
