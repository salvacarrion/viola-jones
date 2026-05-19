"""
Upload aligned face NPYs as new parquet shards on the HF dataset.

For each source in `data/aligned/aligned_<src>.npy`, build a parquet that
mirrors the existing dataset schema:

    image:    struct<bytes: binary, path: string>
    label:    int64        (always 1 — these are face crops)
    source:   string       (set to '<src>_aligned')
    category: string       (None)

The new files are uploaded with the train split prefix so the HF datasets
loader auto-merges them with the existing originals. Loading then yields
rows from both the original `celeba`/`fddb`/`cbcl` and the curated
`celeba_aligned`/`fddb_aligned`/`cbcl_aligned` sources.

Auth: requires a HuggingFace token with write access. Set HF_TOKEN in the
environment or run `huggingface-cli login` first.

Use --dry-run to build the parquets locally without pushing.
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from PIL import Image
from tqdm.auto import tqdm


# Match the existing dataset schema exactly so HF datasets can concatenate
# the new shards with the originals without a schema-merge error.
IMAGE_TYPE = pa.struct([
    pa.field('bytes', pa.binary()),
    pa.field('path', pa.string()),
])
SCHEMA = pa.schema([
    pa.field('image', IMAGE_TYPE),
    pa.field('label', pa.int64()),
    pa.field('source', pa.string()),
    pa.field('category', pa.string()),
])


def encode_png(face):
    """uint8 grayscale array → PNG bytes (matches how the originals are stored)."""
    buf = io.BytesIO()
    Image.fromarray(face, mode='L').save(buf, format='PNG')
    return buf.getvalue()


def build_parquet(npy_path, source_name, out_path):
    """NPY of (N, H, W) uint8 faces → parquet shard with the HF schema."""
    arr = np.load(npy_path)
    print(f"  loaded {len(arr):,} faces from {npy_path.name}")

    images = []
    for face in tqdm(arr, desc=f"encoding {source_name}", unit='img'):
        images.append({'bytes': encode_png(face), 'path': None})

    table = pa.table({
        'image':    images,
        'label':    [1] * len(arr),
        'source':   [source_name] * len(arr),
        'category': [None] * len(arr),
    }, schema=SCHEMA)
    pq.write_table(table, out_path, compression='snappy')
    size_kb = out_path.stat().st_size // 1024
    print(f"  wrote {out_path.name} ({size_kb:,} KB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--aligned-dir', default='data/aligned',
                    help='Where the aligned_<src>.npy files live.')
    ap.add_argument('--out-dir', default='data/aligned',
                    help='Where to write the new parquet shards.')
    ap.add_argument('--repo-id', default='salvacarrion/face-detection',
                    help='HuggingFace dataset repo to push to.')
    ap.add_argument('--sources', nargs='+',
                    default=['celeba', 'fddb', 'cbcl'],
                    help='Sources to package. Each looks for '
                         'aligned_<src>.npy and uploads as <src>_aligned.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Build parquets locally but skip the HF upload.')
    args = ap.parse_args()

    aligned_dir = Path(args.aligned_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = None if args.dry_run else HfApi()

    for source in args.sources:
        npy_path = aligned_dir / f"aligned_{source}.npy"
        if not npy_path.exists():
            print(f"\n=== {source} ===\n  skip: {npy_path} not found")
            continue

        new_source = f"{source}_aligned"
        # `train-*` prefix so HF datasets adds these to the train split.
        # Including `_aligned_` keeps the file name self-describing.
        out_path = out_dir / f"train-{new_source}-00000-of-00001.parquet"
        print(f"\n=== {new_source} ===")
        build_parquet(npy_path, new_source, out_path)

        if args.dry_run:
            continue

        path_in_repo = out_path.name
        print(f"  uploading -> {args.repo_id}:{path_in_repo}")
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type='dataset',
            commit_message=f"Add {new_source} ({len(np.load(npy_path)):,} faces)",
        )
        print(f"  uploaded.")

    if args.dry_run:
        print("\n(dry-run: nothing was uploaded. Drop --dry-run to push.)")


if __name__ == '__main__':
    main()
