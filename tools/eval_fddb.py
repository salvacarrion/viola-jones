"""
FDDB comparison harness — the one comparable metric for our detector vs OpenCV.

The CBCL patch benchmark (`main.py test`) cannot score OpenCV: its cascade
needs a context margin around the face that CBCL's tight crops don't have, so
it returns ~0% there (see tools/baseline_opencv.py). The fair common ground is
FULL-IMAGE detection on a set with ground-truth boxes, scored by IoU. FDDB
(Face Detection Data Set and Benchmark, UMass) is the canonical Viola-Jones-era
benchmark: 2845 images, 5171 faces, annotated as ellipses.

This runs BOTH detectors in their native multi-scale mode on the same images
and reports, per detector:
  - AP (area under the precision-recall curve, all-points / VOC-style)
  - recall & precision at the detector's natural operating point
  - recall at fixed total-false-positive budgets (FDDB-ROC style)

Protocol notes / honest caveats:
  * GT ellipses are converted to their axis-aligned bounding boxes. FDDB
    ellipses include forehead+chin, so they run a bit taller than a typical
    detector box; IoU>=0.5 against the bbox is a standard, slightly strict
    simplification of the official ellipse eval. Tune with --iou.
  * OpenCV detections come from detectMultiScale3 (confidence = stage
    levelWeight). --min-neighbors prunes the low-confidence tail before we
    see it, so it truncates the high-recall end of OpenCV's curve — lower it
    for a fuller curve.
  * Different training data is the point of a baseline, not a flaw: both
    detectors are scored identically on the same images.

Data layout (produced by extracting the HF mirror tarballs):
    data/fddb/originalPics/<year>/...           # images
    data/fddb/FDDB-folds/FDDB-fold-NN-ellipseList.txt

Usage:
    # quick smoke on 40 images of fold 1
    python tools/eval_fddb.py --weights weights/24/<model>.pkl \
        --cascade default --folds 1 --max-images 40

    # full fold 1 (~284 images)
    python tools/eval_fddb.py --weights weights/24/<model>.pkl --cascade default --folds 1
"""

import argparse
import glob
import math
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import non_maximum_supression, get_pretty_time  # noqa: E402
from violajones import ViolaJones  # noqa: E402
from main import pick_weights  # noqa: E402


# ----------------------------------------------------------------------------
# FDDB parsing
# ----------------------------------------------------------------------------
def ellipse_to_bbox(major, minor, angle, cx, cy):
    """Axis-aligned bbox of a rotated ellipse (semi-axes major/minor, angle
    in radians). half-width = sqrt(a²cos²θ + b²sin²θ)."""
    c, s = math.cos(angle), math.sin(angle)
    hw = math.sqrt((major * c) ** 2 + (minor * s) ** 2)
    hh = math.sqrt((major * s) ** 2 + (minor * c) ** 2)
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def find_images_root(fddb_dir):
    """Locate the dir that directly contains the FDDB year folders (2002/2003)."""
    for cand in (os.path.join(fddb_dir, "originalPics"), fddb_dir):
        if os.path.isdir(os.path.join(cand, "2002")):
            return cand
    # fall back: search for a '2002' dir anywhere under fddb_dir
    for dirpath, dirnames, _ in os.walk(fddb_dir):
        if "2002" in dirnames:
            return dirpath
    raise FileNotFoundError(
        f"Could not find FDDB images (a '2002' folder) under {fddb_dir}. "
        f"Extract originalPics.tar.gz there.")


def load_fddb(fddb_dir, folds):
    """Return list of (image_path, [gt_bbox, ...]) for the requested folds."""
    folds_dir = None
    for cand in (os.path.join(fddb_dir, "FDDB-folds"), fddb_dir):
        if glob.glob(os.path.join(cand, "FDDB-fold-*-ellipseList.txt")):
            folds_dir = cand
            break
    if folds_dir is None:
        raise FileNotFoundError(
            f"No FDDB-fold-*-ellipseList.txt under {fddb_dir}. "
            f"Extract FDDB-folds.tgz there.")
    img_root = find_images_root(fddb_dir)

    samples = []
    for fold in folds:
        f = os.path.join(folds_dir, f"FDDB-fold-{fold:02d}-ellipseList.txt")
        if not os.path.exists(f):
            print(f"  warn: {f} missing, skipping")
            continue
        with open(f) as fh:
            lines = [ln.rstrip("\n") for ln in fh]
        i = 0
        while i < len(lines):
            rel = lines[i].strip()
            if not rel:
                i += 1
                continue
            n = int(lines[i + 1])
            boxes = []
            for k in range(n):
                parts = lines[i + 2 + k].split()
                maj, minr, ang, cx, cy = (float(parts[0]), float(parts[1]),
                                          float(parts[2]), float(parts[3]),
                                          float(parts[4]))
                boxes.append(ellipse_to_bbox(maj, minr, ang, cx, cy))
            img_path = os.path.join(img_root, rel + ".jpg")
            samples.append((img_path, boxes))
            i += 2 + n
    return samples


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_image(dets, gts, thr):
    """Greedy match detections (sorted desc by score) to GT, IoU>=thr,
    each GT once. Returns per-detection tp flags aligned to `dets`."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i][4])
    used = [False] * len(gts)
    tp = [False] * len(dets)
    for di in order:
        best_j, best_iou = -1, thr
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou(dets[di][:4], g)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            tp[di] = True
    return tp


def compute_metrics(per_image, n_gt, fp_budgets):
    """per_image: list of (scores[], tp_flags[]). Returns AP, operating-point
    recall/precision, and recall at each total-FP budget."""
    all_scores, all_tp = [], []
    n_det = 0
    for scores, tps in per_image:
        all_scores.extend(scores)
        all_tp.extend(tps)
        n_det += len(scores)
    if n_det == 0:
        return {"ap": 0.0, "recall": 0.0, "precision": 0.0,
                "n_det": 0, "tp": 0, "fp": 0,
                "recall_at_fp": {b: 0.0 for b in fp_budgets}}
    order = np.argsort(-np.asarray(all_scores))
    tp_sorted = np.asarray(all_tp, dtype=np.float64)[order]
    fp_sorted = 1.0 - tp_sorted
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)
    recall = cum_tp / max(n_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # All-points AP: area under the max-envelope precision-recall curve.
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    total_tp = int(cum_tp[-1])
    total_fp = int(cum_fp[-1])
    recall_at_fp = {}
    for b in fp_budgets:
        # recall once total FP across all images reaches budget b
        reached = np.searchsorted(cum_fp, b, side="right")
        reached = min(reached, len(recall) - 1)
        recall_at_fp[b] = float(recall[reached]) if len(recall) else 0.0

    return {"ap": ap, "recall": float(recall[-1]),
            "precision": float(precision[-1]),
            "n_det": n_det, "tp": total_tp, "fp": total_fp,
            "recall_at_fp": recall_at_fp}


# ----------------------------------------------------------------------------
# Detectors → {image_path: [(x1,y1,x2,y2,score), ...]}
# ----------------------------------------------------------------------------
def run_ours(clf, samples, min_face, max_face, nms_thr):
    out = []
    for img_path, _ in tqdm(samples, desc="ours", unit="img"):
        try:
            pil = Image.open(img_path)
        except FileNotFoundError:
            out.append([])
            continue
        regions = clf.find_faces(pil, min_face_size=min_face,
                                 max_face_size=max_face)
        if regions:
            regions = non_maximum_supression(regions, threshold=nms_thr,
                                              mode="weighted", metric="hybrid")
            regions = [tuple(float(v) for v in r) for r in regions]
        out.append(list(regions))
    return out


def run_opencv(cascade, samples, scale_factor, min_neighbors, min_face):
    ms = (min_face, min_face) if min_face else (0, 0)
    out = []
    for img_path, _ in tqdm(samples, desc="opencv", unit="img"):
        try:
            pil = Image.open(img_path)
        except FileNotFoundError:
            out.append([])
            continue
        gray = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
        rects, _lvl, weights = cascade.detectMultiScale3(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
            minSize=ms, outputRejectLevels=True)
        dets = []
        for (x, y, w, h), wt in zip(rects, np.ravel(weights)):
            dets.append((float(x), float(y), float(x + w), float(y + h),
                         float(wt)))
        out.append(dets)
    return out


def evaluate_detector(name, det_lists, samples, iou_thr, fp_budgets):
    per_image = []
    n_gt = 0
    for (img_path, gts), dets in zip(samples, det_lists):
        n_gt += len(gts)
        tp = match_image(dets, gts, iou_thr)
        per_image.append(([d[4] for d in dets], tp))
    m = compute_metrics(per_image, n_gt, fp_budgets)
    print(f"\n[{name}]")
    print(f"\t- detections: {m['n_det']:,}  (TP {m['tp']:,} / FP {m['fp']:,})")
    print(f"\t- AP:          {m['ap']:.3f}")
    print(f"\t- recall:      {m['recall']:.3f}   (operating point: all detections)")
    print(f"\t- precision:   {m['precision']:.3f}")
    print("\t- recall @ total-FP budget:")
    for b in fp_budgets:
        print(f"\t    {b:>5} FP: {m['recall_at_fp'][b]:.3f}")
    return m, n_gt


def main():
    ap = argparse.ArgumentParser(description="FDDB eval: our VJ vs OpenCV")
    ap.add_argument("--fddb-dir", default="data/fddb")
    ap.add_argument("--weights", default=None,
                    help="our cascade .pkl (default: most recent under weights/)")
    ap.add_argument("--cascade", default="default",
                    help="OpenCV cascade: default | alt | alt2 | alt_tree")
    ap.add_argument("--folds", default="1",
                    help="comma list, e.g. '1' or '1,2,3' (1..10; default: 1)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap total images (0 = all in the folds)")
    ap.add_argument("--iou", default="0.5",
                    help="IoU threshold(s) for a true positive; comma list "
                         "evaluates several in one detection pass, e.g. "
                         "'0.3,0.5' (default: 0.5)")
    ap.add_argument("--min-face", type=int, default=40,
                    help="smallest face in px for both detectors (default: 40)")
    ap.add_argument("--max-face", type=int, default=0,
                    help="largest face in px for ours (0 = unbounded)")
    ap.add_argument("--nms-threshold", type=float, default=0.3)
    ap.add_argument("--scale-factor", type=float, default=1.1,
                    help="OpenCV pyramid step (default: 1.1)")
    ap.add_argument("--min-neighbors", type=int, default=2,
                    help="OpenCV grouping; lower → fuller ROC tail (default: 2)")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--skip-opencv", action="store_true")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    iou_thrs = [float(x) for x in str(args.iou).split(",") if x.strip()]
    samples = load_fddb(args.fddb_dir, folds)
    if args.max_images > 0:
        samples = samples[:args.max_images]
    n_gt_total = sum(len(g) for _, g in samples)
    print(f"FDDB: {len(samples):,} images, {n_gt_total:,} GT faces "
          f"(folds {folds})  |  IoU>={iou_thrs}  min-face={args.min_face}px")

    fp_budgets = [50, 100, 284, 500, 1000]
    max_face = args.max_face if args.max_face > 0 else None

    # Detection (the expensive part) runs ONCE per detector; metrics are then
    # recomputed cheaply at every IoU threshold.
    det_lists = {}
    if not args.skip_ours:
        wpath = pick_weights(args.weights)
        print(f"\nOur cascade: {wpath}")
        clf = ViolaJones.load(wpath)
        det_lists["ours"] = run_ours(clf, samples, args.min_face, max_face,
                                     args.nms_threshold)
    if not args.skip_opencv:
        fname = f"haarcascade_frontalface_{args.cascade}.xml"
        cpath = os.path.join(cv2.data.haarcascades, fname)
        cascade = cv2.CascadeClassifier(cpath)
        print(f"\nOpenCV cascade ({cv2.__version__}): {cpath}")
        det_lists[f"opencv:{args.cascade}"] = run_opencv(
            cascade, samples, args.scale_factor, args.min_neighbors,
            args.min_face)

    for thr in iou_thrs:
        print(f"\n############ IoU >= {thr:.2f} ############")
        results = {}
        for name, det in det_lists.items():
            results[name] = evaluate_detector(name, det, samples, thr,
                                              fp_budgets)[0]
        if len(results) > 1:
            print(f"\n=== Comparison (same FDDB images, IoU>={thr:.2f}) ===")
            hdr = f"{'detector':<16} {'AP':>6} {'recall':>7} {'prec':>6}"
            print(hdr); print("-" * len(hdr))
            for name, m in results.items():
                print(f"{name:<16} {m['ap']:>6.3f} {m['recall']:>7.3f} "
                      f"{m['precision']:>6.3f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("\n" + get_pretty_time(t0, s="Total time: "))
