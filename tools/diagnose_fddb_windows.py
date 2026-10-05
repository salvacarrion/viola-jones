"""
Window-level diagnostic on FDDB: how much real-world background gets through a
cascade, and which stages reject it.

For each model it runs `find_faces` (no NMS) on FDDB images and reports:
  - windows evaluated, windows surviving the whole cascade, and how many of
    those survivors are background (max IoU with every GT box < 0.2)
  - background false-positive rate per window and per image
  - per stage: number of weak classifiers and % of incoming windows rejected

This is where docs/RETRAINING.md's numbers come from (ours vs the native
OpenCV port: 4.5e-4 vs 2.5e-5 background windows per window on fold 1).

Usage:
    python tools/diagnose_fddb_windows.py --folds 1 \
        --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl weights/24/opencv_default.pkl
    # legacy feature-scaled pyramid, for the before/after comparison
    python tools/diagnose_fddb_windows.py --folds 1 --pyramid features \
        --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_fddb import iou, load_fddb, parse_folds  # noqa: E402
from violajones import ViolaJones  # noqa: E402

_W = {}


def _init(wpath, min_face, pyramid):
    _W.update(clf=ViolaJones.load(wpath), min_face=min_face, pyramid=pyramid)


def _one(sample):
    path, gts = sample
    stats = {}
    with Image.open(path) as img:
        regions = _W["clf"].find_faces(img, min_face_size=_W["min_face"],
                                       pyramid=_W["pyramid"], stats=stats)
    bg = sum(1 for r in regions
             if max((iou(r[:4], g) for g in gts), default=0.0) < 0.2)
    return stats.get("windows", 0), stats.get("stage_alive"), len(regions), bg


def stumps_per_stage(clf):
    if hasattr(clf, "_cstages"):                       # OpenCVCascade
        return [len(cs["thr"]) for cs in clf._cstages]
    return [len(stage.clfs) for stage in clf.clfs]     # ViolaJones


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--fddb-dir", default="data/fddb")
    ap.add_argument("--folds", default="1")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--min-face", type=int, default=40)
    ap.add_argument("--pyramid", choices=["image", "features"], default="image",
                    help="pyramid for ViolaJones models (OpenCV port is always 'image')")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    samples = load_fddb(args.fddb_dir, parse_folds(args.folds))
    if args.max_images > 0:
        samples = samples[:args.max_images]
    print(f"FDDB folds {args.folds}: {len(samples):,} images, "
          f"min-face {args.min_face}px")

    for wpath in args.weights:
        clf = ViolaJones.load(wpath)
        pyramid = "image" if hasattr(clf, "_cstages") else args.pyramid
        with Pool(args.workers, initializer=_init,
                  initargs=(wpath, args.min_face, pyramid)) as pool:
            rows = list(tqdm(pool.imap(_one, samples, chunksize=4),
                             total=len(samples), desc=os.path.basename(wpath)))
        windows = sum(r[0] for r in rows)
        alive = sum(r[1] for r in rows if r[1] is not None)
        survivors = sum(r[2] for r in rows)
        bg = sum(r[3] for r in rows)
        print(f"\n### {wpath}  (pyramid={pyramid})")
        print(f"windows evaluated     {windows:>12,}")
        print(f"survive the cascade   {survivors:>12,}")
        print(f"  of which background {bg:>12,}   (max IoU with GT < 0.2)")
        print(f"background FPR/window {bg / max(windows, 1):>12.2e}")
        print(f"background per image  {bg / len(samples):>12.1f}   (before NMS)")
        print("stage  stumps  rejected")
        prev = windows
        for k, (n, t) in enumerate(zip(alive, stumps_per_stage(clf))):
            print(f"{k + 1:>5}  {t:>6}  {100 * (1 - n / max(prev, 1)):>7.0f}%")
            prev = n


if __name__ == "__main__":
    main()
