"""
Surgical fix: drop the redundant `cbcl_aligned` source from the train parquet.

`cbcl_aligned` was a bit-identical copy of `cbcl` (CBCL is already canonically
aligned, so no warping was needed). Removing it saves ~5MB and clears up a
confusing source name. The CBCL faces remain available under `source='cbcl'`.

Operates on the LOCAL hf-clone parquet — no need for the original NPYs.
Idempotent: a second run finds no `cbcl_aligned` rows and is a noop.
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

# Pull the README template + counts logic from the main staging script so
# we keep the source of truth in one place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_clean_dataset import build_readme  # noqa: E402


REPO_DIR = Path('/Users/salvacarrion/face-detection')
TRAIN_PARQUET = REPO_DIR / 'data' / 'train-00000-of-00001.parquet'
README_PATH = REPO_DIR / 'README.md'
CALTECH_FALLBACK = 29_879


def main():
    if not TRAIN_PARQUET.exists():
        raise FileNotFoundError(TRAIN_PARQUET)

    print(f"Reading {TRAIN_PARQUET} ...")
    ds = Dataset.from_parquet(str(TRAIN_PARQUET))
    print(f"  {len(ds):,} rows before")
    before_sources = set(ds['source'])
    if 'cbcl_aligned' not in before_sources:
        print("  no cbcl_aligned rows present — noop")
    else:
        ds = ds.filter(lambda x: x['source'] != 'cbcl_aligned',
                       desc="drop cbcl_aligned")
        print(f"  {len(ds):,} rows after")

    # Compute the counts the README needs from the filtered dataset.
    sources = ds['source']
    labels  = ds['label']
    counts = {
        'celeba':   sum(1 for s, l in zip(sources, labels) if s == 'celeba' and l == 1),
        'cbcl_pos': sum(1 for s, l in zip(sources, labels) if s == 'cbcl' and l == 1),
        'cbcl_neg': sum(1 for s, l in zip(sources, labels) if s == 'cbcl' and l == 0),
    }
    print(f"  counts: {counts}")

    # Caltech total: read from the negatives shards if accessible, else
    # use the well-known fallback.
    caltech_total = 0
    for p in sorted((REPO_DIR / 'data').glob('negatives-*-of-00002.parquet')):
        try:
            caltech_total += pq.ParquetFile(p).metadata.num_rows
        except Exception:
            caltech_total = CALTECH_FALLBACK
            print(f"  caltech: using fallback {CALTECH_FALLBACK:,} "
                  f"(parquets look like LFS pointers)")
            break
    if caltech_total == 0:
        caltech_total = CALTECH_FALLBACK

    # Write the filtered parquet back. `to_parquet` re-emits the HF schema
    # metadata so the Hub preview keeps decoding `image` as a thumbnail.
    print(f"Writing {TRAIN_PARQUET} ...")
    ds.to_parquet(str(TRAIN_PARQUET))
    size_mb = TRAIN_PARQUET.stat().st_size / (1024 * 1024)
    print(f"  {size_mb:.1f} MB, {len(ds):,} rows")

    # Regenerate README with the new counts.
    stats = {
        'celeba':      counts['celeba'],
        'cbcl':        counts['cbcl_pos'],
        'cbcl_neg':    counts['cbcl_neg'],
        'train_total': len(ds),
        'neg_total':   caltech_total,
    }
    README_PATH.write_text(build_readme(stats))
    print(f"Wrote {README_PATH}")

    print("\nNext:")
    print(f"  cd {REPO_DIR}")
    print(f"  git diff README.md")
    print(f"  git add data/ README.md")
    print(f"  git commit -m 'Drop redundant cbcl_aligned source "
          f"(identical to cbcl by construction)'")
    print(f"  git push")


if __name__ == '__main__':
    main()
