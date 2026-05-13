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
    bundles = {k: np.load(p) for k, p in paths.items()}
    # Optional matched-domain seed (`prepare_data.py --neg-source cbcl|mixed`).
    seed_path = os.path.join(data_dir, "cbcl_neg_seed.npy")
    bundles["cbcl_neg_seed"] = np.load(seed_path) if os.path.exists(seed_path) else None
    val_cbcl_path = os.path.join(data_dir, "val_cbcl_pos.npy")
    bundles["val_cbcl_pos"] = np.load(val_cbcl_path) if os.path.exists(val_cbcl_path) else None
    return bundles


def train(data_dir, layer_recall=0.99,
          target_neg_per_stage=3000, neg_sample_budget=100000,
          weights_dir="weights", seed=42, target_stage_fpr=None,
          max_stages=30, max_wcs_per_stage=500, min_wcs_per_stage=10,
          min_cascade_recall=0.80, resume_from=None):
    bundles = _load_data(data_dir)
    train_pos = bundles["train_pos"]
    val_pos = bundles["val_pos"]
    neg_pool = bundles["caltech_pool"]
    seed_neg_pool = bundles["cbcl_neg_seed"]
    res = train_pos.shape[1]
    print(f"Loaded data @ {res}×{res} from {data_dir}")
    print(f"\t- train_pos:     {len(train_pos):,}")
    print(f"\t- val_pos:       {len(val_pos):,}")
    print(f"\t- neg_pool:      {len(neg_pool):,}")
    if seed_neg_pool is not None:
        print(f"\t- cbcl_neg_seed: {len(seed_neg_pool):,}  "
              f"(matched-domain stage-1 seed)")
    val_cal_pos = bundles.get("val_cbcl_pos")
    if val_cal_pos is not None:
        print(f"\t- val_cbcl_pos:  {len(val_cal_pos):,}  (CBCL-only calibration subset)")

    # Per-resolution feature cache (reused on subsequent runs at same res).
    cache_dir = os.path.join(str(data_dir), "_cache") + os.sep
    os.makedirs(cache_dir, exist_ok=True)

    # Allocate the output path BEFORE training so per-stage checkpoints
    # land at the same file the final cascade will be saved to. After
    # each stage `clf.train` overwrites this file with the partial cascade
    # — so a crash at hour 9 of a 12 h run leaves a usable 6-stage model
    # rather than nothing. `pick_weights` picks the most recently mtime'd
    # file, so the partial-then-final progression works transparently.
    weights_subdir = os.path.join(weights_dir, str(res))
    os.makedirs(weights_subdir, exist_ok=True)
    out_path = os.path.join(weights_subdir, f"cvj_weights_{int(time.time())}")

    print("\nTraining Viola-Jones...")
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        clf = ViolaJones.load(resume_from)
        assert clf.base_width == res and clf.base_height == res, (
            f"Checkpoint resolution {clf.base_width}×{clf.base_height} "
            f"does not match data resolution {res}×{res}.")
        print(f"\t- loaded {len(clf.clfs)} trained stage(s) — will continue from stage {len(clf.clfs) + 1}")
        # Patch training config with current CLI args so the resumed run
        # uses the same hyperparameters as if it had never been interrupted.
        clf.features_path      = cache_dir
        clf.layer_recall       = layer_recall
        clf.target_stage_fpr   = target_stage_fpr
        clf.max_stages         = max_stages
        clf.max_wcs_per_stage  = max_wcs_per_stage
        clf.min_wcs_per_stage  = min_wcs_per_stage
        clf.min_cascade_recall = min_cascade_recall
    else:
        clf = ViolaJones(features_path=cache_dir,
                         layer_recall=layer_recall, base_size=res,
                         target_stage_fpr=target_stage_fpr,
                         max_stages=max_stages,
                         max_wcs_per_stage=max_wcs_per_stage,
                         min_wcs_per_stage=min_wcs_per_stage,
                         min_cascade_recall=min_cascade_recall)
    clf.train(train_pos, val_pos, neg_pool,
              seed_neg_pool=seed_neg_pool,
              val_cal_pos=val_cal_pos,
              target_neg_per_stage=target_neg_per_stage,
              neg_sample_budget=neg_sample_budget,
              seed=seed,
              checkpoint_path=out_path)
    print("Training finished!")
    print(f"Final weights -> {out_path}.pkl")
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
               nms_threshold=0.3, nms_mode="weighted", nms_metric="hybrid",
               min_shift=None, scale_factor=None):
    weight_path = pick_weights(weight_path)
    print(f"Using weights: {weight_path}")

    if image_paths is None:
        image_paths = ["images/judybats.jpg", "images/people.png"]

    os.makedirs(output_dir, exist_ok=True)
    clf = ViolaJones.load(weight_path)

    for face_path in image_paths:
        print(f"Detecting on {face_path}")
        pil_img = load_image(face_path)
        regions = clf.find_faces(pil_img, growth=scale_factor, min_shift=min_shift)
        print(f"\t- raw regions: {len(regions)}")

        if regions:
            regions = non_maximum_supression(regions, threshold=nms_threshold,
                                             mode=nms_mode, metric=nms_metric)
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
    parser.add_argument("--max-stages", type=int, default=30,
                        help="Hard cap on cascade depth (default: 30). Training stops "
                             "earlier if cumulative val recall drops below "
                             "--min-cascade-recall or the negative pool is exhausted.")
    parser.add_argument("--max-wcs-per-stage", type=int, default=500,
                        help="Max weak classifiers per stage (default: 500). With "
                             "--target-stage-fpr, stages stop earlier when the FPR "
                             "target is met — this is only the hard upper bound.")
    parser.add_argument("--min-wcs-per-stage", type=int, default=10,
                        help="Min weak classifiers per stage before --target-stage-fpr "
                             "can early-stop (default: 10). Prevents 1-2-stump stages "
                             "where calibration can't pick a useful threshold and the "
                             "layer collapses to accepting only windows the single "
                             "stump fires on.")
    parser.add_argument("--min-cascade-recall", type=float, default=0.80,
                        help="Stop adding stages when cumulative val-pos recall drops "
                             "below this (default: 0.80). Prevents deep cascades from "
                             "trading too much recall for marginal specificity gains.")
    parser.add_argument("--layer-recall", type=float, default=0.99,
                        help="Per-stage face-recall target (default: 0.99)")
    parser.add_argument("--target-neg-per-stage", type=int, default=3000,
                        help="Negatives mined from pool per cascade stage (default: 3000)")
    parser.add_argument("--neg-sample-budget", type=int, default=100000,
                        help="Max patches sampled per stage when mining (default: 100000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", default=None, metavar="PKL",
                        help="Path to a .pkl checkpoint to resume training from. "
                             "Loads the partial cascade and continues from the next "
                             "stage. All other training args (--max-wcs-per-stage, "
                             "--target-stage-fpr, etc.) are applied to the resumed run.")
    parser.add_argument("--target-stage-fpr", type=float, default=0.5,
                        help="Per-stage FPR target for adaptive training (paper §3). "
                             "Each stage stops adding weak classifiers once training-"
                             "negative FPR drops to this value (default: 0.5). "
                             "Set to 0.0 to disable adaptive mode (fixed-T = --max-wcs-per-stage).")

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
    parser.add_argument("--nms-mode", choices=["weighted", "greedy"],
                        default="weighted",
                        help="NMS strategy. 'weighted' (default) fuses "
                             "overlapping boxes into a score-weighted average "
                             "— better for multi-scale duplicates of the same "
                             "face. 'greedy' is classic NMS (drop overlapping).")
    parser.add_argument("--nms-metric", choices=["hybrid", "iou", "iom"],
                        default="hybrid",
                        help="Overlap metric. 'hybrid' (default) fuses if "
                             "either IoU > threshold or IoM > 0.7 — handles "
                             "both equal-scale duplicates and nested "
                             "different-scale duplicates (the cluster of "
                             "tiny boxes inside a face box at scale 1× vs "
                             "2×). 'iou' is standard. 'iom' is "
                             "intersection-over-min-area (more aggressive).")
    parser.add_argument("--detect-shift", type=int, default=None,
                        help="Sliding-window step in pixels (default: model's "
                             "self.shift, typically 2). Increase to 3-4 for "
                             "faster detection at a small recall cost.")
    parser.add_argument("--detect-scale", type=float, default=None,
                        help="Scale-pyramid growth factor (default: model's "
                             "self.base_scale, typically 1.25). Increase to "
                             "1.3–1.5 for fewer pyramid levels and faster "
                             "detection.")

    args = parser.parse_args()

    start_time = time.time()
    print(f"Starting (mode={args.mode})...")

    if args.mode == "train":
        tgt_fpr = args.target_stage_fpr if args.target_stage_fpr > 0.0 else None
        train(args.data_dir,
              layer_recall=args.layer_recall,
              target_neg_per_stage=args.target_neg_per_stage,
              neg_sample_budget=args.neg_sample_budget,
              seed=args.seed,
              target_stage_fpr=tgt_fpr,
              max_stages=args.max_stages,
              max_wcs_per_stage=args.max_wcs_per_stage,
              min_wcs_per_stage=args.min_wcs_per_stage,
              min_cascade_recall=args.min_cascade_recall,
              resume_from=args.resume_from)
    elif args.mode == "test":
        test(args.weights_path, args.data_dir)
    elif args.mode == "detect":
        find_faces(weight_path=args.weights_path,
                   image_paths=args.detect_images,
                   output_dir=args.detect_output,
                   nms_threshold=args.nms_threshold,
                   nms_mode=args.nms_mode,
                   nms_metric=args.nms_metric,
                   min_shift=args.detect_shift,
                   scale_factor=args.detect_scale)

    print("\n" + get_pretty_time(start_time, s="Total time: "))
