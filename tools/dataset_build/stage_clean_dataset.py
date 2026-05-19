"""
Stage the clean + aligned face dataset into a local HF dataset clone.

Run AFTER `tools/build_clean_dataset.py`. This script:

1. Builds a new `train-00000-of-00001.parquet` from the *_clean.npy and
   *_aligned.npy bundles in --clean-dir. Sources written:
     - `celeba` (label=1) — loose portrait crop
     - `celeba_aligned` (label=1) — tight VJ crop
     - `cbcl` (label=1) — CBCL faces, already canonically aligned
     - `cbcl` (label=0) — CBCL non-faces (matched-domain hard-neg seed)
   No `cbcl_aligned` — CBCL is already aligned by construction; a duplicate
   variant would be bit-identical to `cbcl`.
2. Restores the negatives split to caltech-only (2 shards), undoing any
   previous reshard that had carved out a CBCL-negatives shard.
3. Rewrites README.md with the new structure and row counts auto-computed
   from the NPYs.

No git operations — you review with `git status` / `git diff` and commit/push
yourself. Parquet files are already covered by the LFS rule in .gitattributes,
so `git add` pushes them via LFS automatically.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image
from tqdm.auto import tqdm

# Use the `datasets` library to declare the schema. Writing the parquet via
# `Dataset.to_parquet` embeds the `huggingface` schema metadata that the HF
# Hub preview needs to decode the `image` struct as an actual image instead
# of showing the raw bytes dict.
FEATURES = Features({
    'image':    HFImage(),
    'label':    Value('int64'),
    'source':   Value('string'),
    'category': Value('string'),
})


def encode_png(face):
    buf = io.BytesIO()
    Image.fromarray(face, mode='L').save(buf, format='PNG')
    return buf.getvalue()


def build_block(arr, source_name, label, desc):
    """Encode an NPY of (N, H, W) uint8 faces into row-dict form for pa.table."""
    images = []
    for face in tqdm(arr, desc=desc, unit='img', leave=False):
        images.append({'bytes': encode_png(face), 'path': None})
    n = len(arr)
    return {
        'image':    images,
        'label':    [label] * n,
        'source':   [source_name] * n,
        'category': [None] * n,
    }


def stack_blocks(blocks):
    """Concatenate per-source row dicts into one dict-of-columns."""
    out = {k: [] for k in ('image', 'label', 'source', 'category')}
    for b in blocks:
        for k in out:
            out[k].extend(b[k])
    return out


def write_parquet(data_dict, path):
    """Write `data_dict` to parquet via Dataset.to_parquet so the resulting
    file carries the `huggingface` schema metadata (`image` declared as the
    `Image` feature). Without it, the Hub preview shows raw bytes dicts
    instead of decoded thumbnails."""
    ds = Dataset.from_dict(data_dict, features=FEATURES)
    ds.to_parquet(str(path))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  wrote {path.name} ({size_mb:.1f} MB, {len(ds):,} rows)")


def restore_negatives_layout(data_dir):
    """Restore the negatives split to 2 caltech-only shards.

    Reverses the earlier `-of-00003` reshard (which had added a CBCL shard):
      1. Delete `negatives-00002-of-00003.parquet` if present (cbcl negs
         now live in train).
      2. Rename `negatives-*-of-00003.parquet` back to `-of-00002`.

    Idempotent: noop if already on `-of-00002` layout.
    """
    cbcl_shard = data_dir / 'negatives-00002-of-00003.parquet'
    if cbcl_shard.exists():
        cbcl_shard.unlink()
        print(f"  deleted {cbcl_shard.name} (cbcl negatives back in train)")

    olds = sorted(data_dir.glob('negatives-*-of-00003.parquet'))
    if not olds:
        if any(data_dir.glob('negatives-*-of-00002.parquet')):
            print("  (negatives already on -of-00002 layout, skipping rename)")
        else:
            print("  WARNING: no negatives shards matched any expected pattern")
        return
    for old in olds:
        new = old.parent / old.name.replace('-of-00003', '-of-00002')
        print(f"  rename {old.name} → {new.name}")
        old.rename(new)


def build_readme(stats):
    return f"""---
license: other
license_name: research-only-mixed
pretty_name: Face Detection (CelebA + CBCL + Caltech-256)
task_categories:
  - object-detection
  - image-classification
tags:
  - face-detection
  - viola-jones
size_categories:
  - 100K<n<1M
---

# Face Detection Dataset

Small grayscale face crops + a large pool of natural-image negatives, packaged
for classical face detectors (Viola-Jones, Haar cascades, sliding-window
classifiers).

## At a glance

| Split | Rows | Faces / Non-faces |
|---|---|---|
| `train`     | {stats['train_total']:,} | {stats['train_total'] - stats['cbcl_neg']:,} / {stats['cbcl_neg']:,} |
| `test`      | 24,045 | 472 / 23,573 (CBCL benchmark) |
| `negatives` | {stats['neg_total']:,} | — / {stats['neg_total']:,} (Caltech-256) |

Same schema across all splits: `image`, `label` (0/1), `source`, `category`.

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("salvacarrion/face-detection")

# All faces for training
faces = ds["train"].filter(lambda x: x["label"] == 1)

# CBCL benchmark for evaluation
test = ds["test"]

# Hard-negative pool (raw color JPGs at native resolution)
negs = ds["negatives"]
```

## Face sources (train split)

CelebA ships in **two paired variants at the same index** — same person,
two crop styles:

- `celeba` — loose portrait framing (hair and jaw visible).
- `celeba_aligned` — tight Viola-Jones crop (eyes pinned to fixed pixel
  positions, face fills the 48×48 frame).

CBCL is already canonically aligned, so it has a single variant.

| `source` | Label | Count | Notes |
|---|---|---|---|
| `celeba`         | 1 | {stats['celeba']:,} | Loose portrait. Filtered for frontal pose using the manual CelebA landmarks. |
| `celeba_aligned` | 1 | {stats['celeba']:,} | Same {stats['celeba']:,} faces tightly warped to CBCL-matched geometry. |
| `cbcl`           | 1 | {stats['cbcl']:,} | MIT CBCL #1, upsampled from native 19×19. Already canonically aligned. |
| `cbcl`           | 0 | {stats['cbcl_neg']:,} | CBCL non-faces — matched-domain seed for stage 1 of cascade training. |

Picking `source == "cbcl"` from train returns **both faces and non-faces** —
filter by `label` to pick.

## Test split

The classic CBCL Viola-Jones benchmark, untouched. Faces and non-faces live
together in one split so accuracy is one pass.

## Negatives split

Caltech-256 source JPGs (native color, varying resolution). Categories
containing `face`, `people`, or `human` are excluded. Sample patches at
whatever resolution your detector needs.

## Aligned geometry (the `*_aligned` variants)

Two-point similarity transform — face shape preserved, no stretching:

- Left eye  → (12, 10) in a 48×48 frame
- Right eye → (36, 10)

The canonical positions were measured empirically from the mean of CBCL
training faces, so models trained on `celeba_aligned` generalize cleanly to
the CBCL test benchmark.

## License

Research / non-commercial use only. Cite the original sources:

- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [MIT CBCL Face Database #1](http://cbcl.mit.edu/software-datasets/FaceData2.html)
- [Caltech-256](https://data.caltech.edu/records/nyy15-4j048)
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clean-dir', default='data/clean',
                    help='Where the *_clean.npy / *_aligned.npy / '
                         'cbcl_negatives.npy live.')
    ap.add_argument('--repo-dir', default='/Users/salvacarrion/face-detection',
                    help='Local clone of the HF dataset repo.')
    args = ap.parse_args()

    clean_dir = Path(args.clean_dir)
    repo_dir  = Path(args.repo_dir)
    data_dir  = repo_dir / 'data'
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Expected HF clone at {repo_dir} with a data/ subdirectory.")

    print(f"Loading NPYs from {clean_dir} ...")
    celeba_clean   = np.load(clean_dir / 'celeba_clean.npy')
    celeba_aligned = np.load(clean_dir / 'celeba_aligned.npy')
    cbcl_faces     = np.load(clean_dir / 'cbcl_clean.npy')
    cbcl_negs      = np.load(clean_dir / 'cbcl_negatives.npy')
    # NB: we deliberately skip `cbcl_aligned.npy`. CBCL is already
    # canonically aligned (the whole point of the MIT-curated set), so the
    # "_aligned" variant would be a bit-identical copy of `cbcl` — pure
    # redundancy. Only celeba has the meaningful `<src>` / `<src>_aligned`
    # paired distinction.
    print(f"  celeba_clean   : {len(celeba_clean):,}")
    print(f"  celeba_aligned : {len(celeba_aligned):,}")
    print(f"  cbcl_faces     : {len(cbcl_faces):,}")
    print(f"  cbcl_negatives : {len(cbcl_negs):,}")

    # ---- 1) train parquet — face crops (label=1) + CBCL non-faces (label=0) ----
    print("\nBuilding train parquet (overwrites existing) ...")
    train_data = stack_blocks([
        build_block(celeba_clean,   'celeba',         1, 'encode celeba'),
        build_block(celeba_aligned, 'celeba_aligned', 1, 'encode celeba_aligned'),
        build_block(cbcl_faces,     'cbcl',           1, 'encode cbcl faces'),
        build_block(cbcl_negs,      'cbcl',           0, 'encode cbcl non-faces'),
    ])
    write_parquet(train_data, data_dir / 'train-00000-of-00001.parquet')

    # ---- 2) negatives — restore to caltech-only (2 shards) ----
    print("\nRestoring negatives split to caltech-only layout ...")
    restore_negatives_layout(data_dir)

    # ---- 3) README ----
    print("\nWriting README.md ...")
    # Caltech rows: try the existing shards, fall back to a hardcoded number
    # if the files are LFS pointers (clone without `git lfs pull`).
    CALTECH_FALLBACK = 29879
    caltech_total = 0
    for p in sorted(data_dir.glob('negatives-*-of-00002.parquet')):
        try:
            caltech_total += pq.ParquetFile(p).metadata.num_rows
        except (pa.ArrowInvalid, OSError) as e:
            print(f"  warning: {p.name} unreadable ({type(e).__name__}). "
                  f"Looks like an LFS pointer; using fallback "
                  f"caltech_total={CALTECH_FALLBACK:,}. Run `git lfs pull` in "
                  f"the repo to get exact counts.")
            caltech_total = CALTECH_FALLBACK
            break
    stats = {
        'celeba':       len(celeba_clean),
        'cbcl':         len(cbcl_faces),
        'cbcl_neg':     len(cbcl_negs),
        'train_total':  (len(celeba_clean) + len(celeba_aligned)
                         + len(cbcl_faces) + len(cbcl_negs)),
        'neg_total':    caltech_total,
    }
    (repo_dir / 'README.md').write_text(build_readme(stats))
    print(f"  wrote {repo_dir / 'README.md'}")

    print(f"\nDone. Review and commit:")
    print(f"  cd {repo_dir}")
    print(f"  git status")
    print(f"  git diff README.md")
    print(f"  git lfs ls-files            # confirm new parquets are LFS-tracked")
    print(f"  git add data/ README.md")
    print(f"  git commit -m 'Rebuild with clean + aligned face pairs; "
          f"move CBCL negs to negatives split'")
    print(f"  git push")


if __name__ == '__main__':
    main()
