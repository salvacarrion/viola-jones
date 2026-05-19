#!/usr/bin/env python3
"""
Upload the curated face-detection dataset to Hugging Face Hub as Parquet shards.

Splits pushed
  train     — CelebA (50 k) + FDDB (11 k) + CBCL train faces + CBCL train nofaces
  test      — CBCL test faces + CBCL test nofaces  (classic VJ benchmark)
  negatives — Caltech-256 filtered raw color JPGs  (bootstrap negative pool)

Columns
  image     : Image()          decoded image bytes
  label     : ClassLabel       0 = noface, 1 = face
  source    : string           "celeba" | "fddb" | "cbcl" | "caltech"
  category  : string | null    Caltech category folder name; null for all others

Usage
  python tools/push_to_hf.py
  python tools/push_to_hf.py --repo-id salvacarrion/face-detection --private
  python tools/push_to_hf.py --no-caltech
"""

import argparse
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi

LABEL_NOFACE = 0
LABEL_FACE = 1

FEATURES = Features({
    "image": Image(),
    "label": ClassLabel(names=["noface", "face"]),
    "source": Value("string"),
    "category": Value("string"),
})


def _png_rows(folder: Path, label: int, source: str) -> list[dict]:
    return [
        {"image": str(p), "label": label, "source": source, "category": None}
        for p in sorted(folder.glob("*.png"))
        if not p.name.startswith("_")
    ]


def _caltech_rows(source_dir: Path) -> list[dict]:
    rows = []
    for cat_dir in sorted(source_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        for img in sorted(cat_dir.glob("*.jpg")):
            rows.append({
                "image": str(img),
                "label": LABEL_NOFACE,
                "source": "caltech",
                "category": cat_dir.name,
            })
    return rows


def build_dataset(root: Path, include_caltech: bool) -> DatasetDict:
    train_rows: list[dict] = []
    test_rows: list[dict] = []

    sources = [
        (root / "celeba" / "train",        LABEL_FACE,   "celeba", "train"),
        (root / "fddb"   / "train",        LABEL_FACE,   "fddb",   "train"),
        (root / "cbcl"   / "train" / "faces",   LABEL_FACE,   "cbcl",   "train"),
        (root / "cbcl"   / "train" / "nofaces",  LABEL_NOFACE, "cbcl",   "train"),
        (root / "cbcl"   / "test"  / "faces",    LABEL_FACE,   "cbcl",   "test"),
        (root / "cbcl"   / "test"  / "nofaces",  LABEL_NOFACE, "cbcl",   "test"),
    ]
    for folder, label, source, split in sources:
        if not folder.exists():
            print(f"  [skip] {folder} not found")
            continue
        rows = _png_rows(folder, label, source)
        tag = "face" if label == LABEL_FACE else "noface"
        print(f"  {str(folder.relative_to(root)):<35} {len(rows):>7,}  ({tag})")
        (train_rows if split == "train" else test_rows).extend(rows)

    splits: dict[str, Dataset] = {
        "train": Dataset.from_list(train_rows, features=FEATURES),
        "test":  Dataset.from_list(test_rows,  features=FEATURES),
    }

    if include_caltech:
        caltech_src = root / "caltech" / "source"
        if caltech_src.exists():
            neg_rows = _caltech_rows(caltech_src)
            print(f"  {'caltech/source':<35} {len(neg_rows):>7,}  (noface)")
            splits["negatives"] = Dataset.from_list(neg_rows, features=FEATURES)
        else:
            print(f"  [skip] {caltech_src} not found")

    print()
    for split, ds in splits.items():
        print(f"  split={split:<12} {len(ds):>7,} rows")

    return DatasetDict(splits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Push face-detection dataset to HF Hub")
    parser.add_argument("--dataset-path", default="datasets",
                        help="Root of the local datasets/ folder (default: datasets)")
    parser.add_argument("--repo-id", default="salvacarrion/face-detection",
                        help="HF repo id, e.g. username/dataset-name")
    parser.add_argument("--no-caltech", action="store_true",
                        help="Skip the Caltech negatives split")
    parser.add_argument("--private", action="store_true",
                        help="Create/keep the repo private")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Parquet shards per split (default: auto ~500 MB each)")
    args = parser.parse_args()

    root = Path(args.dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    print("Collecting rows…")
    dd = build_dataset(root, include_caltech=not args.no_caltech)

    print(f"\nPushing parquet shards to {args.repo_id}…")
    push_kwargs: dict = dict(repo_id=args.repo_id, private=args.private)
    if args.num_shards is not None:
        push_kwargs["num_shards"] = {split: args.num_shards for split in dd}
    dd.push_to_hub(**push_kwargs)

    # Upload the dataset card (README.md) separately
    readme = root / "README.md"
    if readme.exists():
        print("Uploading dataset card (README.md)…")
        HfApi().upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(f"\nDone — https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
