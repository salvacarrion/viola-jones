"""
OpenCV Haar-cascade baseline — reference numbers for our from-scratch detector.

OpenCV ships the canonical Viola-Jones frontal-face cascades inside the
`opencv-python` package itself (`cv2.data.haarcascades`), so nothing needs to
be downloaded. `haarcascade_frontalface_default.xml` is the classic VJ-style
model; `_alt` / `_alt2` were trained with GentleBoost and the extended
(Lienhart) rotated Haar set; `_alt_tree` is a 20×20 tree-structured variant.

Two sub-commands mirror our own two evaluation surfaces:

  benchmark  — same CBCL patch test set as `python main.py test`
               (test_pos.npy + test_neg.npy). Each patch is fed to
               detectMultiScale restricted to a single window, giving a
               per-patch face/non-face decision in the shape of our
               `clf.classify`. *** This does NOT transfer to OpenCV: it
               scores ~0% recall on the CBCL patches — see CAVEAT below. ***

  detect     — same full images as `python main.py detect`. Runs the native
               multi-scale sliding-window detector and writes annotated PNGs
               under images/outputs/opencv_<cascade>/ so they sit next to our
               own *_detected.png for a visual side-by-side. THIS is the valid
               comparison surface for OpenCV.

CAVEAT 1 — the patch benchmark does not transfer to OpenCV. Measured: the
default/alt/alt2 cascades return 0/472 detections on the CBCL positive
patches. This is *not* a bug (the same cascade finds 8 faces in
images/people.png). OpenCV's cascade was trained with the face occupying the
*centre* of the window with a context border (forehead/background) around it;
its Haar features key off that border contrast. The CBCL crops are extremely
tight (face to the edges, no margin), so those border features never fire.
Recall climbs monotonically with fabricated margin (0% tight → 8.7% at +40%
gray border → 22.7% at +60%), confirming the mechanism. Our own detector was
trained directly on these tight crops, so it is matched to the test set and
OpenCV is not. There is no honest patch-level number for OpenCV — use the
`detect` sub-command (full images) as the reference instead.

CAVEAT 2 — even for full images, the OpenCV cascades were trained on a
different (and much larger) positive set, with the extended Haar feature set
and GentleBoost. This is a "reference implementation everyone uses" baseline,
NOT a same-data ablation of our training pipeline.

Usage:
    # Part 1 — quantitative patch benchmark (default + alt + alt2)
    python tools/baseline_opencv.py benchmark --data-dir data/24_cbcl \
        --cascade default alt alt2

    # Part 2 — qualitative full-image detection (no ground truth → visual only)
    python tools/baseline_opencv.py detect \
        --images images/people.png images/clase.png images/physics.jpg \
                 images/i1.jpg images/judybats.jpg \
        --cascade default
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from tqdm.auto import tqdm

# Allow `python tools/baseline_opencv.py` from the repo root to import utils.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import draw_bounding_boxes, load_image, get_pretty_time  # noqa: E402


def cascade_path(name):
    """Map a short name to the bundled cascade XML and load it."""
    fname = f"haarcascade_frontalface_{name}.xml"
    path = os.path.join(cv2.data.haarcascades, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cascade '{name}' not found at {path}. "
            f"Available: default, alt, alt2, alt_tree.")
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"Failed to load cascade XML: {path}")
    return clf, path


def _metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "true_positive": tp, "true_negative": tn,
        "false_positive": fp, "false_negative": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": prec, "recall": rec, "specifity": spec, "f1": f1,
    }


def benchmark_cascade(clf, X, y, window=24, equalize=False):
    """Per-patch face/non-face decision via single-window detectMultiScale.

    Each patch is resized to `window`×`window` (the cascade's base window —
    24 for default/alt/alt2, which also covers the 20×20 alt_tree after one
    scale step) and detectMultiScale is pinned to that one window via
    minSize == maxSize. With only a single window position there is no
    grouping to do, so `minNeighbors=0` keeps the raw cascade verdict — the
    direct analogue of our `clf.classify(patch)`. A patch counts as a face
    iff at least one detection is returned.
    """
    tp = tn = fp = fn = 0
    win = (window, window)
    for i in tqdm(range(len(y)), desc="Evaluating", unit="patch", leave=False):
        img = np.ascontiguousarray(X[i], dtype=np.uint8)
        if img.shape[0] != window or img.shape[1] != window:
            img = cv2.resize(img, win, interpolation=cv2.INTER_LINEAR)
        if equalize:
            img = cv2.equalizeHist(img)
        faces = clf.detectMultiScale(img, scaleFactor=1.1, minNeighbors=0,
                                     minSize=win, maxSize=win)
        pred = 1 if len(faces) > 0 else 0
        if pred == 1 and y[i] == 1:
            tp += 1
        elif pred == 0 and y[i] == 0:
            tn += 1
        elif pred == 1 and y[i] == 0:
            fp += 1
        else:
            fn += 1
    return _metrics(tp, tn, fp, fn)


def cmd_benchmark(args):
    pos_path = os.path.join(args.data_dir, "test_pos.npy")
    neg_path = os.path.join(args.data_dir, "test_neg.npy")
    for p in (pos_path, neg_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Point --data-dir at a prepared bundle "
                f"(e.g. data/24_cbcl).")
    pos = np.load(pos_path)
    neg = np.load(neg_path)
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    res = pos.shape[1]
    window = args.window if args.window else max(res, 24)
    print(f"OpenCV {cv2.__version__}")
    print(f"Test set: {len(pos):,} faces + {len(neg):,} non-faces "
          f"@ {res}×{res}  (window fed to cascade: {window}×{window}"
          + ("  +equalizeHist" if args.equalize else "") + ")")

    rows = []
    for name in args.cascade:
        clf, path = cascade_path(name)
        print(f"\n[{name}] {path}")
        m = benchmark_cascade(clf, X, y, window=window, equalize=args.equalize)
        rows.append((name, m))
        for k, v in m.items():
            label = f"\t- {k}: "
            print(label + (f"{v:,}" if k.startswith(("true_", "false_"))
                           else f"{v:.3f}"))
        if m["recall"] < 0.05:
            print("\t  NOTE: ~0 recall is expected, not a bug. OpenCV's "
                  "cascade needs a context margin around the face; CBCL crops "
                  "are too tight for its border Haar features to fire. Use the "
                  "`detect` sub-command (full images) to compare OpenCV.")

    if len(rows) > 1:
        print("\n=== Comparison (same CBCL test patches) ===")
        hdr = f"{'cascade':<10} {'F1':>6} {'recall':>7} {'prec':>6} {'spec':>6} {'acc':>6}"
        print(hdr)
        print("-" * len(hdr))
        for name, m in rows:
            print(f"{name:<10} {m['f1']:>6.3f} {m['recall']:>7.3f} "
                  f"{m['precision']:>6.3f} {m['specifity']:>6.3f} "
                  f"{m['accuracy']:>6.3f}")


def cmd_detect(args):
    print(f"OpenCV {cv2.__version__}")
    for name in args.cascade:
        clf, path = cascade_path(name)
        out_dir = os.path.join(args.output, f"opencv_{name}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{name}] {path}  ->  {out_dir}/")
        for img_path in args.images:
            pil_img = load_image(img_path)
            gray = cv2.cvtColor(np.array(pil_img.convert("RGB")),
                                cv2.COLOR_RGB2GRAY)
            if args.equalize:
                gray = cv2.equalizeHist(gray)
            min_size = (args.min_size, args.min_size) if args.min_size else (0, 0)
            faces = clf.detectMultiScale(
                gray, scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors, minSize=min_size)
            # detectMultiScale returns (x, y, w, h) → (x1, y1, x2, y2).
            regions = [(int(x), int(y), int(x + w), int(y + h))
                       for (x, y, w, h) in faces]
            print(f"\t- {os.path.basename(img_path)}: {len(regions)} face(s)")
            drawn = draw_bounding_boxes(pil_img, regions, thickness=2)
            out_path = os.path.join(
                out_dir,
                os.path.splitext(os.path.basename(img_path))[0] + "_detected.png")
            drawn.convert("RGB").save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="OpenCV Haar-cascade baseline for the VJ detector")
    sub = parser.add_subparsers(dest="mode", required=True)

    pb = sub.add_parser("benchmark",
                        help="per-patch CBCL benchmark (comparable to main.py test)")
    pb.add_argument("--data-dir", default="data/24_cbcl",
                    help="bundle dir with test_pos.npy / test_neg.npy "
                         "(default: data/24_cbcl; the CBCL test set is the "
                         "same across all data/* dirs)")
    pb.add_argument("--cascade", nargs="+", default=["default"],
                    metavar="NAME",
                    help="one or more of: default alt alt2 alt_tree "
                         "(default: default)")
    pb.add_argument("--window", type=int, default=0,
                    help="window size fed to the cascade (default: max(res,24))")
    pb.add_argument("--equalize", action="store_true",
                    help="apply cv2.equalizeHist to each patch first")

    pd = sub.add_parser("detect",
                        help="full-image detection (visual side-by-side; no GT)")
    pd.add_argument("--images", nargs="+", required=True, metavar="IMG")
    pd.add_argument("--cascade", nargs="+", default=["default"], metavar="NAME")
    pd.add_argument("--output", default="images/outputs",
                    help="root dir; outputs land in <output>/opencv_<cascade>/")
    pd.add_argument("--scale-factor", type=float, default=1.1)
    pd.add_argument("--min-neighbors", type=int, default=5)
    pd.add_argument("--min-size", type=int, default=30,
                    help="smallest face in px (0 disables; default: 30)")
    pd.add_argument("--equalize", action="store_true")

    args = parser.parse_args()
    t0 = time.time()
    if args.mode == "benchmark":
        cmd_benchmark(args)
    elif args.mode == "detect":
        cmd_detect(args)
    print("\n" + get_pretty_time(t0, s="Total time: "))


if __name__ == "__main__":
    main()
