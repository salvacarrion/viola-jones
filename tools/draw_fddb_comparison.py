"""
Before/after figures on FDDB images: ground truth (green), our cascade (red),
OpenCV (blue).

Each output PNG has two panels:
  - left  "before": our detector as it ran before the inference fix
            (pyramid="features", raw training-window boxes)
  - right "after":  the current default (pyramid="image", boxes mapped to
            face boxes with the model's stored box_transform)
OpenCV and ground truth are identical in both panels, so the only thing that
changes is our detector.

Usage:
    python tools/draw_fddb_comparison.py \
        --weights weights/24/celeba.pkl \
        --names 2002/08/02/big/img_1231 2003/01/17/big/img_610
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_fddb import load_fddb  # noqa: E402
from utils import apply_box_transform, non_maximum_supression  # noqa: E402
from violajones import ViolaJones  # noqa: E402

GREEN, RED, BLUE = (0, 200, 0), (230, 30, 30), (40, 90, 255)


def ours(clf, pil, pyramid, box_tf, min_face, nms_thr):
    regions = clf.find_faces(pil, min_face_size=min_face, pyramid=pyramid)
    if not regions:
        return []
    regions = non_maximum_supression(regions, threshold=nms_thr,
                                     mode="weighted", metric="hybrid")
    return apply_box_transform(regions, box_tf)


def panel(pil, gts, mine, cv, title):
    img = pil.convert("RGB")
    d = ImageDraw.Draw(img)
    for boxes, color in ((gts, GREEN), (cv, BLUE), (mine, RED)):
        for b in boxes:
            d.rectangle(tuple(float(v) for v in b[:4]), outline=color, width=2)
    d.rectangle((0, 0, img.width, 18), fill=(0, 0, 0))
    d.text((6, 3), title, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--names", nargs="+", required=True,
                    help="FDDB image ids, e.g. 2002/08/02/big/img_1231")
    ap.add_argument("--fddb-dir", default="data/fddb")
    ap.add_argument("--cascade", default="default")
    ap.add_argument("--min-face", type=int, default=40)
    ap.add_argument("--nms-threshold", type=float, default=0.3)
    ap.add_argument("--min-neighbors", type=int, default=2)
    ap.add_argument("--out", default="images/outputs/fddb_comparison")
    args = ap.parse_args()

    gt_by_path = dict(load_fddb(args.fddb_dir, range(1, 11)))
    clf = ViolaJones.load(args.weights)
    box_tf = getattr(clf, "box_transform", None)
    if box_tf is None:
        print("WARNING: model has no box_transform; 'after' panel uses raw boxes. "
              "Fit one with tools/eval_fddb.py --save-box-transform")
    cascade = cv2.CascadeClassifier(os.path.join(
        cv2.data.haarcascades, f"haarcascade_frontalface_{args.cascade}.xml"))
    os.makedirs(args.out, exist_ok=True)

    for name in args.names:
        path = next((p for p in gt_by_path if p.endswith(name + ".jpg")), None)
        if path is None:
            print(f"skip {name}: not in FDDB folds")
            continue
        pil = Image.open(path).convert("L")
        gts = gt_by_path[path]
        rects = cascade.detectMultiScale(
            np.asarray(pil), scaleFactor=1.1, minNeighbors=args.min_neighbors,
            minSize=(args.min_face, args.min_face))
        cv = [(x, y, x + w, y + h) for (x, y, w, h) in rects]
        before = ours(clf, pil, "features", None, args.min_face, args.nms_threshold)
        after = ours(clf, pil, "image", box_tf, args.min_face, args.nms_threshold)
        left = panel(pil, gts, before, cv, f"before: {len(before)} boxes")
        right = panel(pil, gts, after, cv, f"after: {len(after)} boxes")
        canvas = Image.new("RGB", (left.width * 2 + 8, left.height), (255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width + 8, 0))
        out = os.path.join(args.out, name.replace("/", "_") + ".png")
        canvas.save(out)
        print(f"{name}: GT {len(gts)} | before {len(before)} | after {len(after)} "
              f"| opencv {len(cv)} -> {out}")


if __name__ == "__main__":
    main()
