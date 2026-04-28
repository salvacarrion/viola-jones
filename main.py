"""
Viola-Jones Algorithm — CLI entry point.

Workflow:
  1. python tools/prepare_data.py        # download HF dataset + preprocess
  2. python main.py train --data-dir data/24
  3. python main.py test  --data-dir data/24
  4. python main.py detect
"""

import argparse
import glob
import os
import time

import numpy as np

from utils import (
    draw_bounding_boxes, evaluate, get_pretty_time, load_image,
    non_maximum_supression,
)
from violajones import ViolaJones


def _load_data(data_dir):
    """Load NPY bundles produced by tools/prepare_data.py from `data_dir`."""
    data_dir = str(data_dir)
    paths = {k: os.path.join(data_dir, f"{k}.npy")
             for k in ("train_pos", "val_pos", "caltech_pool")}
    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {data_dir}. "
            f"Run `python tools/prepare_data.py` first.")
    return {k: np.load(p) for k, p in paths.items()}


def train(data_dir, layers, layer_recall=0.99,
          target_neg_per_stage=3000, neg_sample_budget=100000,
          weights_dir="weights", seed=42):
    bundles = _load_data(data_dir)
    train_pos = bundles["train_pos"]
    val_pos = bundles["val_pos"]
    neg_pool = bundles["caltech_pool"]
    res = train_pos.shape[1]
    print(f"Loaded data @ {res}×{res} from {data_dir}")
    print(f"\t- train_pos: {len(train_pos):,}")
    print(f"\t- val_pos:   {len(val_pos):,}")
    print(f"\t- neg_pool:  {len(neg_pool):,}")

    # Per-resolution feature cache (reused on subsequent runs at same res).
    cache_dir = os.path.join(str(data_dir), "_cache") + os.sep
    os.makedirs(cache_dir, exist_ok=True)

    print("\nTraining Viola-Jones...")
    clf = ViolaJones(layers=layers, features_path=cache_dir,
                     layer_recall=layer_recall, base_size=res)
    clf.train(train_pos, val_pos, neg_pool,
              target_neg_per_stage=target_neg_per_stage,
              neg_sample_budget=neg_sample_budget,
              seed=seed)
    print("Training finished!")

    weights_subdir = os.path.join(weights_dir, str(res))
    os.makedirs(weights_subdir, exist_ok=True)
    out_path = os.path.join(weights_subdir, f"cvj_weights_{int(time.time())}")
    clf.save(out_path)
    print(f"Weights saved -> {out_path}.pkl")
    return clf


def test(weights_path, data_dir):
    weights_path = pick_weights(weights_path)
    print(f"Using weights: {weights_path}")
    clf = ViolaJones.load(weights_path)

    pos_path = os.path.join(str(data_dir), "cbcl_test_pos.npy")
    neg_path = os.path.join(str(data_dir), "cbcl_test_neg.npy")
    if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
        raise FileNotFoundError(
            f"CBCL test bundles not found at {data_dir}. Re-run "
            f"`python tools/prepare_data.py` (default includes CBCL test).")

    pos = np.load(pos_path)
    neg = np.load(neg_path)
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    print(f"CBCL test set: {len(pos):,} faces + {len(neg):,} non-faces "
          f"@ {pos.shape[1]}×{pos.shape[2]}")

    print("\nEvaluating...")
    metrics = evaluate(clf, X, y)
    print("\nMetrics:")
    counter = 0
    for k, v in metrics.items():
        counter += 1
        if counter <= 4:
            print(f"\t- {k}: {v:,}")
        else:
            print(f"\t- {k}: {v:.3f}")


def pick_weights(path=None):
    """Return `path` if given, otherwise the most recent checkpoint under weights/."""
    if path is not None:
        return path
    candidates = sorted(glob.glob("weights/**/cvj_weights_*.pkl", recursive=True),
                        key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(
            "No trained weights under weights/. Run `python main.py train ...` first.")
    return candidates[-1]


def find_faces(weight_path=None, image_paths=None, output_dir="images/outputs",
               nms_threshold=0.3):
    weight_path = pick_weights(weight_path)
    print(f"Using weights: {weight_path}")

    if image_paths is None:
        image_paths = ["images/judybats.jpg", "images/people.png"]

    os.makedirs(output_dir, exist_ok=True)
    clf = ViolaJones.load(weight_path)

    for face_path in image_paths:
        print(f"Detecting on {face_path}")
        pil_img = load_image(face_path)
        regions = clf.find_faces(pil_img)
        print(f"\t- raw regions: {len(regions)}")

        if regions:
            regions = non_maximum_supression(regions, threshold=nms_threshold)
            print(f"\t- after NMS:  {len(regions)}")

        drawn_img = draw_bounding_boxes(pil_img, list(regions), thickness=2)
        out_path = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(face_path))[0] + "_detected.png")
        drawn_img.convert("RGB").save(out_path)
        print(f"\t- saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Viola-Jones face detector")
    parser.add_argument("mode", choices=["train", "test", "detect"],
                        help="train: fit cascade | "
                             "test: evaluate on CBCL benchmark | "
                             "detect: run inference on images")

    # Common
    parser.add_argument("--data-dir", default="data/24",
                        help="Directory with NPY bundles produced by "
                             "tools/prepare_data.py (default: data/24)")
    parser.add_argument("--weights-path", default=None,
                        help="Checkpoint to load; omit to auto-pick the most recent one")

    # Train
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 20, 50, 100],
                        metavar="T",
                        help="Weak learners per cascade stage (default: 5 20 50 100). "
                             "Stage 1 must be >=~3 — with T=1 the layer's score is "
                             "binary so 99%% recall calibration drops the threshold to 0.")
    parser.add_argument("--layer-recall", type=float, default=0.99,
                        help="Per-stage face-recall target (default: 0.99)")
    parser.add_argument("--target-neg-per-stage", type=int, default=3000,
                        help="Negatives mined from pool per cascade stage (default: 3000)")
    parser.add_argument("--neg-sample-budget", type=int, default=100000,
                        help="Max patches sampled per stage when mining (default: 100000)")
    parser.add_argument("--seed", type=int, default=42)

    # Detect
    parser.add_argument("--detect-images", nargs="+",
                        default=["images/people.png", "images/clase.png",
                                 "images/physics.jpg", "images/i1.jpg",
                                 "images/judybats.jpg"],
                        metavar="IMG",
                        help="Image paths to run face detection on")
    parser.add_argument("--detect-output", default="images/outputs",
                        help="Directory where annotated outputs are saved")
    parser.add_argument("--nms-threshold", type=float, default=0.3,
                        help="IoU threshold for NMS post-processing (default: 0.3)")

    args = parser.parse_args()

    start_time = time.time()
    print(f"Starting (mode={args.mode})...")

    if args.mode == "train":
        train(args.data_dir, args.layers,
              layer_recall=args.layer_recall,
              target_neg_per_stage=args.target_neg_per_stage,
              neg_sample_budget=args.neg_sample_budget,
              seed=args.seed)
    elif args.mode == "test":
        test(args.weights_path, args.data_dir)
    elif args.mode == "detect":
        find_faces(weight_path=args.weights_path,
                   image_paths=args.detect_images,
                   output_dir=args.detect_output,
                   nms_threshold=args.nms_threshold)

    print("\n" + get_pretty_time(start_time, s="Total time: "))
