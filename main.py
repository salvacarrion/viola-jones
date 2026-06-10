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
    """Load NPY bundles produced by tools/prepare_data.py from `data_dir`.

    Required: train_pos.npy, val_pos.npy, caltech_pool.npy
    Optional: neg_seed.npy (matched-domain stage-1 seed when
              `prepare_data.py --neg-source` is 'benchmark' or 'mixed')
    """
    data_dir = str(data_dir)
    paths = {k: os.path.join(data_dir, f"{k}.npy")
             for k in ("train_pos", "val_pos", "caltech_pool")}
    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {data_dir}. "
            f"Run `python tools/prepare_data.py` first.")
    # caltech_pool can be tens of GB at production scale (e.g. 100M patches
    # @ 24×24 ≈ 57 GB) — always memmap. The mining code in violajones.py
    # already reads it as random-ordered contiguous chunks for prefetch
    # friendliness, so memmap is bit-for-bit equivalent to a full load.
    # train_pos / val_pos are MB-sized and stay in RAM.
    bundles = {
        "train_pos": np.load(paths["train_pos"]),
        "val_pos": np.load(paths["val_pos"]),
        "caltech_pool": np.load(paths["caltech_pool"], mmap_mode="r"),
    }
    seed_path = os.path.join(data_dir, "neg_seed.npy")
    bundles["neg_seed"] = np.load(seed_path) if os.path.exists(seed_path) else None
    return bundles


def train(data_dir, layer_recall=0.99,
          target_neg_per_stage=3000, neg_sample_budget=100000,
          weights_dir="weights", seed=42, target_stage_fpr=None,
          max_stages=30, max_wcs_per_stage=200, min_wcs_per_stage=1,
          min_cascade_recall=0.80, min_stage_negatives=0, resume_from=None,
          hard_neg_pool=None, very_hard_neg_pool=None,
          drop_low_score_pos=0.0):
    bundles = _load_data(data_dir)
    train_pos = bundles["train_pos"]
    val_pos = bundles["val_pos"]
    neg_pool = bundles["caltech_pool"]
    neg_seed = bundles["neg_seed"]
    res = train_pos.shape[1]
    # Optional override: replace the raw mining pool with a pre-mined hard-neg
    # pool produced by tools/mine_hard_negatives.py. Skips the "easy" patches
    # an earlier cascade already rejects, so every new stage starts against
    # materially harder negatives. neg_seed stays as the stage-1 seed.
    if hard_neg_pool is not None:
        print(f"Override: loading hard-neg pool from {hard_neg_pool}")
        hard = np.load(hard_neg_pool, mmap_mode="r")
        assert hard.shape[1] == res and hard.shape[2] == res, (
            f"Hard-neg pool is {hard.shape[1]}×{hard.shape[2]}, "
            f"data is {res}×{res}. Resolution mismatch.")
        neg_pool = hard

    # Optional positive filtering by oracle score.
    # face_scores.npy is produced by tools/score_faces.py and ranks each
    # train_pos sample by cumulative AdaBoost margin against a frozen oracle
    # cascade. Dropping the bottom-fraction removes the crops the oracle is
    # most confident are NOT canonical aligned faces — typically alignment
    # failures the dataset still carries.
    pos_cache_suffix = ""
    if drop_low_score_pos > 0.0:
        scores_path = os.path.join(str(data_dir), "face_scores.npy")
        if not os.path.exists(scores_path):
            raise FileNotFoundError(
                f"--drop-low-score-pos {drop_low_score_pos} requires "
                f"{scores_path}. Run `python tools/score_faces.py --weights "
                f"<oracle.pkl> --data-dir {data_dir}` first.")
        scores = np.load(scores_path)
        if len(scores) != len(train_pos):
            raise ValueError(
                f"face_scores.npy has {len(scores)} entries but train_pos "
                f"has {len(train_pos)}. Re-run score_faces.py.")
        n_drop = int(round(drop_low_score_pos * len(train_pos)))
        if n_drop > 0:
            keep = np.argsort(scores)[n_drop:]  # bottom-N indices removed
            cutoff = float(scores[np.argsort(scores)[n_drop - 1]])
            print(f"Filter: dropping {n_drop:,}/{len(train_pos):,} "
                  f"({100*drop_low_score_pos:.1f}%) lowest-scoring positives "
                  f"(cutoff score ≤ {cutoff:+.4f})")
            train_pos = train_pos[keep]
            # Separate cache key so the cached features for the full set don't
            # leak into the filtered run and vice versa.
            pos_cache_suffix = f"__drop{drop_low_score_pos:.2f}"

    if very_hard_neg_pool is not None:
        vh = np.load(very_hard_neg_pool, mmap_mode="r")
        assert vh.shape[1] == res and vh.shape[2] == res, (
            f"Very-hard pool is {vh.shape[1]}×{vh.shape[2]}, "
            f"data is {res}×{res}. Resolution mismatch.")
    else:
        vh = None

    print(f"Loaded data @ {res}×{res} from {data_dir}")
    print(f"\t- train_pos: {len(train_pos):,}"
          + (f"  (filtered: -{drop_low_score_pos:.0%})"
             if drop_low_score_pos > 0 else ""))
    print(f"\t- val_pos:   {len(val_pos):,}  (benchmark anchor, un-augmented)")
    print(f"\t- neg_pool:  {len(neg_pool):,}"
          + ("  (hard-mined)" if hard_neg_pool is not None else ""))
    if neg_seed is not None:
        print(f"\t- neg_seed:  {len(neg_seed):,}  (matched-domain stage-1 seed)")
    if vh is not None:
        print(f"\t- vhard_pool: {len(vh):,}  (top-up reservoir for mining shortfalls)")

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
        clf.min_stage_negatives = min_stage_negatives
    else:
        clf = ViolaJones(features_path=cache_dir,
                         layer_recall=layer_recall, base_size=res,
                         target_stage_fpr=target_stage_fpr,
                         max_stages=max_stages,
                         max_wcs_per_stage=max_wcs_per_stage,
                         min_wcs_per_stage=min_wcs_per_stage,
                         min_cascade_recall=min_cascade_recall,
                         min_stage_negatives=min_stage_negatives)
    clf.train(train_pos, val_pos, neg_pool,
              neg_seed=neg_seed,
              very_hard_pool=vh,
              target_neg_per_stage=target_neg_per_stage,
              neg_sample_budget=neg_sample_budget,
              seed=seed,
              checkpoint_path=out_path,
              pos_cache_suffix=pos_cache_suffix)
    print("Training finished!")
    print(f"Final weights -> {out_path}.pkl")
    return clf


def test(weights_path, data_dir):
    weights_path = pick_weights(weights_path)
    print(f"Using weights: {weights_path}")
    clf = ViolaJones.load(weights_path)

    pos_path = os.path.join(str(data_dir), "test_pos.npy")
    neg_path = os.path.join(str(data_dir), "test_neg.npy")
    if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
        raise FileNotFoundError(
            f"Test bundles not found at {data_dir}. Re-run "
            f"`python tools/prepare_data.py` with --benchmark != none.")

    pos = np.load(pos_path)
    neg = np.load(neg_path)
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    print(f"Test set: {len(pos):,} faces + {len(neg):,} non-faces "
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
               min_shift=None, scale_factor=None,
               min_face_size=None, max_face_size=None,
               min_score=None):
    weight_path = pick_weights(weight_path)
    print(f"Using weights: {weight_path}")

    if image_paths is None:
        image_paths = ["images/judybats.jpg", "images/people.png"]

    os.makedirs(output_dir, exist_ok=True)
    clf = ViolaJones.load(weight_path)

    for face_path in image_paths:
        print(f"Detecting on {face_path}")
        pil_img = load_image(face_path)
        regions = clf.find_faces(pil_img, growth=scale_factor, min_shift=min_shift,
                                 min_face_size=min_face_size,
                                 max_face_size=max_face_size,
                                 min_score=min_score)
        if regions:
            scores = [r[4] for r in regions if len(r) >= 5]
            score_stats = (f"  scores: min={min(scores):.3f} "
                           f"median={sorted(scores)[len(scores)//2]:.3f} "
                           f"max={max(scores):.3f}") if scores else ""
        else:
            score_stats = ""
        print(f"\t- raw regions: {len(regions)}{score_stats}")

        if regions:
            regions = non_maximum_supression(regions, threshold=nms_threshold,
                                             mode=nms_mode, metric=nms_metric)
            print(f"\t- after NMS:  {len(regions)}")
            # for r in regions: print(f"\t  box: {tuple(int(v) for v in r[:4])} score={float(r[4]):.1f}" if len(r) >= 5 else f"\t  box: {tuple(int(v) for v in r[:4])}")

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
    parser.add_argument("--min-wcs-per-stage", type=int, default=1,
                        help="Min weak classifiers per stage before --target-stage-fpr "
                             "can early-stop (default: 1 = safeguard inert). Since the "
                             "FPR check now runs at the calibrated operating-point "
                             "threshold, single-stump stages no longer collapse, so this "
                             "floor is mostly a safety net. Raise it (e.g. to 5) if you "
                             "still see stages stopping suspiciously early.")
    parser.add_argument("--min-cascade-recall", type=float, default=0.80,
                        help="Stop adding stages when cumulative val-pos recall drops "
                             "below this (default: 0.80). Prevents deep cascades from "
                             "trading too much recall for marginal specificity gains.")
    parser.add_argument("--min-stage-negatives", type=int, default=0,
                        help="Stop adding stages if mining yields fewer than this many "
                             "hard negatives (default: 0). Prevents training "
                             "degenerated stages when the negative pool is exhausted.")
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
    parser.add_argument("--hard-neg-pool", default=None, metavar="NPY",
                        help="Path to a hard-negative pool .npy produced by "
                             "tools/mine_hard_negatives.py. If given, replaces "
                             "the data-dir's caltech_pool.npy as the mining "
                             "source — every stage's hard-neg mining draws from "
                             "patches the previous cascade already misclassifies, "
                             "so AdaBoost starts against materially harder "
                             "negatives. Resolution must match the data dir.")
    parser.add_argument("--very-hard-neg-pool", default=None, metavar="NPY",
                        help="Path to a pre-mined very-hard-negative pool "
                             "(produced by tools/mine_hard_negatives.py against "
                             "a strong oracle cascade). Used ONLY as a top-up "
                             "reservoir when normal per-stage mining "
                             "(seed + caltech_pool) returns fewer patches than "
                             "--target-neg-per-stage. Lets deep stages keep "
                             "training even when the regular pool is "
                             "depleted, which is what stops cbcl/celeba+cbcl "
                             "at stage 15-16. Resolution must match the data dir.")
    parser.add_argument("--drop-low-score-pos", type=float, default=0.0,
                        metavar="FRAC",
                        help="Fraction in [0, 1] of lowest-scoring positives "
                             "to drop before training. Requires "
                             "<data-dir>/face_scores.npy (produce with "
                             "tools/score_faces.py against a frozen oracle "
                             "cascade). Removes the crops least face-like in "
                             "the oracle's view — typically residual "
                             "alignment failures CelebA still carries. "
                             "Default: 0.0 (no filtering).")
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
    parser.add_argument("--detect-min-face", type=int, default=None,
                        metavar="PX",
                        help="Smallest face size in image pixels. Skips "
                             "pyramid scales below this — clamped to the "
                             "training resolution (model's base_width).")
    parser.add_argument("--detect-max-face", type=int, default=None,
                        metavar="PX",
                        help="Largest face size in image pixels. Stops the "
                             "pyramid once the window exceeds this.")
    parser.add_argument("--detect-min-score", type=float, default=None,
                        metavar="S",
                        help="Discard detections whose accumulated cascade "
                             "margin (sum of per-stage `vote - layer_thr` "
                             "over every stage the window passed) is below "
                             "this value. None disables filtering. Each "
                             "detect run prints min/median/max score of the "
                             "raw regions — use that to pick a value. "
                             "Typical useful range 0.05–0.5 for a 10-15 "
                             "stage cascade; the tuned cascades produce "
                             "more low-score borderline detections than the "
                             "raw ones, so this is the recommended knob to "
                             "clean up sliding-window output without "
                             "retraining.")

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
              min_stage_negatives=args.min_stage_negatives,
              resume_from=args.resume_from,
              hard_neg_pool=args.hard_neg_pool,
              very_hard_neg_pool=args.very_hard_neg_pool,
              drop_low_score_pos=args.drop_low_score_pos)
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
                   scale_factor=args.detect_scale,
                   min_face_size=args.detect_min_face,
                   max_face_size=args.detect_max_face,
                   min_score=args.detect_min_score)

    print("\n" + get_pretty_time(start_time, s="Total time: "))
