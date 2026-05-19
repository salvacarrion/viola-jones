"""
Extract a local HF dataset clone to a flat folder of PNG/JPG files for
inspection. Designed for visual review, not for re-training.

Layout written to --out-dir:
    train/<source>/             one PNG per face crop (grayscale 48×48)
    test/cbcl_label0/           non-face crops (label==0)
    test/cbcl_label1/           face crops (label==1)
    negatives/caltech/<cat>/    Caltech color JPGs by category
    negatives/cbcl/             CBCL 48×48 non-faces

By default we cap each folder at --limit images so 50K-image folders don't
explode your Finder. Pass --limit 0 for "no cap" if you want every row.
"""

import argparse
import io
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm


def export_parquet(parquet_path, split_dir, limit, default_ext='png'):
    """Iterate a parquet and dump each row's image to a per-source folder."""
    pf = pq.ParquetFile(parquet_path)
    print(f"\n=== {parquet_path.name} ({pf.metadata.num_rows:,} rows) ===")
    counts = Counter()              # (source, label[, category]) → idx for naming
    written = Counter()             # same key → how many we've already kept

    for batch in pf.iter_batches(batch_size=1000):
        sources    = batch.column('source').to_pylist()
        labels     = batch.column('label').to_pylist()
        categories = batch.column('category').to_pylist()
        imgs       = batch.column('image').to_pylist()
        for src, lbl, cat, ib in tqdm(list(zip(sources, labels, categories, imgs)),
                                       desc=parquet_path.stem, unit='img', leave=False):
            # Folder per (source, label) — `test` has both labels under cbcl,
            # everything else collapses to one label per source.
            if src == 'cbcl' and lbl == 0 and 'test' in parquet_path.name:
                sub = split_dir / 'cbcl_label0'
            elif src == 'cbcl' and lbl == 1 and 'test' in parquet_path.name:
                sub = split_dir / 'cbcl_label1'
            elif src == 'caltech' and cat:
                sub = split_dir / 'caltech' / str(cat)
            else:
                sub = split_dir / str(src)

            key = str(sub)
            if limit > 0 and written[key] >= limit:
                continue
            sub.mkdir(parents=True, exist_ok=True)

            # Use the source bytes as-is (they're already encoded PNG/JPG in
            # the parquet) so we don't re-encode and degrade quality.
            data = ib['bytes']
            # Sniff format from the first few bytes; fall back to PNG.
            ext = default_ext
            if data[:3] == b'\xff\xd8\xff':
                ext = 'jpg'
            elif data[:8] == b'\x89PNG\r\n\x1a\n':
                ext = 'png'
            fname = sub / f"{counts[key]:05d}.{ext}"
            counts[key] += 1
            written[key] += 1
            fname.write_bytes(data)

    # Per-folder summary
    for key, n in sorted(written.items()):
        print(f"  {key}: {n:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-dir', default='/Users/salvacarrion/face-detection',
                    help='Local HF dataset clone (must have data/*.parquet '
                         'pulled via `git lfs pull`).')
    ap.add_argument('--out-dir', default='/Users/salvacarrion/Desktop/face-detection-export',
                    help='Where to write the extracted images.')
    ap.add_argument('--limit', type=int, default=300,
                    help='Cap per (source, label, category) folder. '
                         'Use 0 for no cap (warning: caltech will dump '
                         '~30K images).')
    ap.add_argument('--splits', nargs='+',
                    default=['train', 'test', 'negatives'])
    args = ap.parse_args()

    repo_dir = Path(args.repo_dir)
    out_dir  = Path(args.out_dir)
    data_dir = repo_dir / 'data'
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = out_dir / split
        shards = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not shards:
            print(f"\n(no shards found for split '{split}')")
            continue
        for sh in shards:
            # Skip LFS pointer files (they aren't actual parquet)
            if sh.stat().st_size < 1024:
                print(f"\n(skip {sh.name}: looks like an LFS pointer, "
                      f"run `git lfs pull` in {repo_dir})")
                continue
            export_parquet(sh, split_dir, limit=args.limit)

    print(f"\nDone. Open {out_dir}/ in Finder to browse.")


if __name__ == '__main__':
    main()
