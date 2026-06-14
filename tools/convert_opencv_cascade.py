"""
Convert an OpenCV pretrained Haar cascade XML into a native OpenCVCascade
pickle under weights/, so it runs through our own pipeline (main.py
detect/test, tools/eval_fddb.py) with no cv2 dependency at inference.

Supports the new-format ("opencv-cascade-classifier"), axis-aligned, stump
cascades — haarcascade_frontalface_{default,alt,alt2}. Tilted features or
deep trees are rejected with a clear error.

After building, it VALIDATES the port against cv2 itself: every base-window
(24×24) patch is classified by both our evaluator and a single-window
cv2.detectMultiScale, and the agreement rate is reported. Parity should be
~100% — that is what makes the pickle a faithful, native reproduction.

Usage:
    python tools/convert_opencv_cascade.py --cascade default
    python tools/convert_opencv_cascade.py --cascade alt --out weights/24/opencv_alt.pkl
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opencv_cascade import OpenCVCascade  # noqa: E402


def _floats(text):
    return [float(t) for t in text.split()]


def parse_cascade(xml_path, base_scale=1.1, shift=1):
    root = ET.parse(xml_path).getroot()
    casc = root.find("cascade") if root.tag != "cascade" else root
    if casc is None:
        casc = root  # some files put cascade fields at the root

    width = int(casc.findtext("width"))
    height = int(casc.findtext("height"))

    # features: list of [(x, y, w, h, weight), ...]
    features = []
    for feat in casc.find("features"):
        rects = []
        for r in feat.find("rects"):
            vals = _floats(r.text)
            x, y, w, h, wt = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3]), vals[4]
            rects.append((x, y, w, h, wt))
            if len(vals) > 5 and vals[5] != 0:  # tilted flag
                raise ValueError("Tilted feature found — not supported "
                                 "(only axis-aligned cascades port natively).")
        features.append(rects)

    stages = []
    for st in casc.find("stages"):
        stage_threshold = float(st.findtext("stageThreshold"))
        stumps = []
        for wc in st.find("weakClassifiers"):
            inodes = _floats(wc.findtext("internalNodes"))
            leaves = _floats(wc.findtext("leafValues"))
            if len(inodes) != 4 or len(leaves) != 2:
                raise ValueError(
                    f"Weak classifier is not a depth-1 stump "
                    f"(internalNodes={inodes}, leafValues={leaves}). "
                    f"Deep trees are not supported.")
            feat_idx = int(inodes[2])
            thr = inodes[3]
            left, right = leaves[0], leaves[1]
            rects = features[feat_idx]
            coords = np.array([(x, y, w, h) for (x, y, w, h, _) in rects],
                              dtype=np.int64)
            weights = np.array([wt for (*_, wt) in rects], dtype=np.float64)
            stumps.append((coords, weights, thr, left, right))
        stages.append((stage_threshold, stumps))

    n_stumps = sum(len(s) for _, s in stages)
    print(f"Parsed {xml_path}")
    print(f"  window {width}x{height}, {len(stages)} stages, "
          f"{n_stumps} stumps, {len(features)} features")
    return OpenCVCascade(stages, width, height,
                         base_scale=base_scale, shift=shift,
                         source=os.path.basename(xml_path))


def validate(clf, cascade_name, n_neg=3000):
    """Compare our evaluator to cv2 single-window classification on patches."""
    import cv2
    xml = os.path.join(cv2.data.haarcascades,
                       f"haarcascade_frontalface_{cascade_name}.xml")
    cv = cv2.CascadeClassifier(xml)
    w = clf.base_width

    # gather patches: CBCL test faces + negatives if available, else random
    patches = []
    for rel in (f"data/24_cbcl/test_pos.npy", f"data/24_cbcl/test_neg.npy"):
        if os.path.exists(rel):
            a = np.load(rel)
            patches.append(a if "pos" in rel else a[:n_neg])
    if patches:
        X = np.concatenate(patches, axis=0)
    else:
        rng = np.random.default_rng(0)
        X = rng.integers(0, 256, size=(2000, w, w), dtype=np.uint8)

    agree = disagree = 0
    win = (w, w)
    for p in X:
        p = np.ascontiguousarray(p, dtype=np.uint8)
        if p.shape[0] != w:
            p = cv2.resize(p, win)
        ours = clf.classify(p)
        cvhit = 1 if len(cv.detectMultiScale(
            p, scaleFactor=1.1, minNeighbors=0, minSize=win, maxSize=win)) else 0
        if ours == cvhit:
            agree += 1
        else:
            disagree += 1
    total = agree + disagree
    print(f"  parity vs cv2: {agree}/{total} agree "
          f"({100*agree/total:.2f}%), {disagree} differ")
    return agree / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", default="default",
                    help="default | alt | alt2 (bundled OpenCV cascade name)")
    ap.add_argument("--xml", default=None, help="explicit XML path (overrides --cascade)")
    ap.add_argument("--out", default=None, help="output .pkl path")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--base-scale", type=float, default=1.1,
                    help="pyramid growth baked into the model (matches cv2 "
                         "scaleFactor; default 1.1)")
    ap.add_argument("--shift", type=int, default=2,
                    help="sliding-window step baked into the model (default 2)")
    args = ap.parse_args()

    if args.xml:
        xml_path = args.xml
    else:
        import cv2
        xml_path = os.path.join(cv2.data.haarcascades,
                                f"haarcascade_frontalface_{args.cascade}.xml")
    clf = parse_cascade(xml_path, base_scale=args.base_scale, shift=args.shift)

    if not args.no_validate:
        validate(clf, args.cascade)

    out = args.out or f"weights/{clf.base_width}/opencv_{args.cascade}.pkl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    clf.save(out)
    print(f"  saved native cascade -> {out}")


if __name__ == "__main__":
    main()
