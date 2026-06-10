"""Extend the last stage of a Viola-Jones cascade past its original WC cap.

When a cascade stage hits --max-wcs-per-stage without reaching
--target-stage-fpr, the cascade stops with "capacity ceiling".  This tool
continues that stage's AdaBoost training with more rounds, reusing the same
hard-negative difficulty level (mines against the cascade WITHOUT the last
stage, so negatives match what the capped stage originally trained on).

Workflow:
  1. Load the checkpoint (e.g. 9 stages, last one capped at 800 WCs)
  2. Pop the last stage, keep its trained WCs and alphas
  3. Re-mine hard negatives using the truncated cascade (stages 1-8)
  4. Rebuild the training feature matrix
  5. Replay the saved 800 WC rounds to reconstruct AdaBoost weight state
  6. Continue boosting from round 801 to the new cap (e.g. 1200)
  7. Re-calibrate threshold, push extended stage back, and save

Usage:
    python tools/extend_stage.py \\
        --checkpoint weights/24/cvj_weights_1780281293.pkl \\
        --data-dir data/24_celeba_aligned \\
        --new-max-wcs 1200 \\
        --target-neg-per-stage 10000 \\
        --neg-sample-budget 100000000 \\
        --target-stage-fpr 0.5 \\
        --drop-low-score-pos 0.02
"""

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
from tqdm.auto import tqdm

# Repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaboost import AdaBoost
from utils import (
    build_features, apply_features, integral_image, get_pretty_time,
)
from violajones import ViolaJones
from weakclassifier import WeakClassifier


def _feature_key(haar):
    """Hashable key for a HaarFeature, based on its rectangle geometry."""
    pos = tuple((r.x, r.y, r.width, r.height) for r in haar.positive_regions)
    neg = tuple((r.x, r.y, r.width, r.height) for r in haar.negative_regions)
    return (pos, neg)


def _build_feature_index(features):
    """Map feature geometry → index in the features list."""
    idx_map = {}
    for i, f in enumerate(features):
        key = _feature_key(f)
        if key in idx_map:
            # Duplicate features are possible in theory; keep first occurrence.
            pass
        else:
            idx_map[key] = i
    return idx_map


def _sample_stds(samples):
    """Per-sample pixel std, floored at 1.0."""
    stds = samples.reshape(len(samples), -1).std(axis=1)
    return np.maximum(stds, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the .pkl checkpoint with the capped stage")
    ap.add_argument("--data-dir", required=True,
                    help="Data directory with NPY bundles (same as training)")
    ap.add_argument("--new-max-wcs", type=int, default=1200,
                    help="New max weak classifiers for the last stage "
                         "(default: 1200)")
    ap.add_argument("--target-neg-per-stage", type=int, default=10000,
                    help="Hard negatives to mine (default: 10000)")
    ap.add_argument("--neg-sample-budget", type=int, default=100_000_000,
                    help="Max patches sampled when mining (default: 100M)")
    ap.add_argument("--target-stage-fpr", type=float, default=0.5,
                    help="Per-stage FPR target (default: 0.5)")
    ap.add_argument("--layer-recall", type=float, default=0.99,
                    help="Per-stage face-recall target (default: 0.99)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None,
                    help="Output path (default: <checkpoint>_extN.pkl)")
    ap.add_argument("--drop-low-score-pos", type=float, default=0.0,
                    help="Fraction of lowest-scoring positives to drop "
                         "(same as training run)")
    ap.add_argument("--precompute-sort-index", action="store_true",
                    help="Precompute and cache sorting indices of training samples in RAM "
                         "(faster training, but uses significant extra RAM).")
    args = ap.parse_args()

    start_time = time.time()

    # ---- 1. Load checkpoint and pop last stage ----
    print(f"Loading checkpoint: {args.checkpoint}")
    clf = ViolaJones.load(args.checkpoint)
    n_stages = len(clf.clfs)
    print(f"  Cascade has {n_stages} stages, "
          f"T per stage = {[len(s.clfs) for s in clf.clfs]}")

    last_stage = clf.clfs[-1]
    saved_wcs = list(last_stage.clfs)
    saved_alphas = list(last_stage.alphas)
    n_existing = len(saved_wcs)
    print(f"  Last stage (stage {n_stages}): {n_existing} WCs, "
          f"threshold={last_stage.threshold:.4f}")
    if hasattr(last_stage, 'final_fpr') and last_stage.final_fpr is not None:
        print(f"  Last stage final FPR: {last_stage.final_fpr:.4f}")

    if n_existing >= args.new_max_wcs:
        print(f"Last stage already has {n_existing} WCs >= new-max-wcs "
              f"{args.new_max_wcs}. Nothing to do.")
        return

    # Pop the last stage to get the truncated cascade for mining
    clf.clfs = clf.clfs[:-1]
    print(f"\n  Truncated to {len(clf.clfs)} stages for hard-neg mining")

    # ---- 2. Load data ----
    data_dir = str(args.data_dir)
    train_pos = np.load(os.path.join(data_dir, "train_pos.npy"))
    val_pos = np.load(os.path.join(data_dir, "val_pos.npy"))
    neg_pool = np.load(os.path.join(data_dir, "caltech_pool.npy"),
                       mmap_mode="r")
    seed_path = os.path.join(data_dir, "neg_seed.npy")
    neg_seed = np.load(seed_path) if os.path.exists(seed_path) else None

    # Apply same positive filtering as training
    pos_cache_suffix = ""
    if args.drop_low_score_pos > 0.0:
        scores_path = os.path.join(data_dir, "face_scores.npy")
        scores = np.load(scores_path)
        n_drop = int(round(args.drop_low_score_pos * len(train_pos)))
        if n_drop > 0:
            keep = np.argsort(scores)[n_drop:]
            cutoff = float(scores[np.argsort(scores)[n_drop - 1]])
            print(f"Filter: dropping {n_drop:,}/{len(train_pos):,} "
                  f"({100*args.drop_low_score_pos:.1f}%) lowest-scoring "
                  f"positives (cutoff score <= {cutoff:+.4f})")
            train_pos = train_pos[keep]
            pos_cache_suffix = f"__drop{args.drop_low_score_pos:.2f}"

    res = train_pos.shape[1]
    print(f"\nData @ {res}×{res}: {len(train_pos):,} train_pos, "
          f"{len(val_pos):,} val_pos, {len(neg_pool):,} neg_pool")

    # ---- 3. Mine hard negatives with truncated cascade ----
    print(f"\nMining hard negatives with {len(clf.clfs)}-stage cascade "
          f"(target={args.target_neg_per_stage:,}, "
          f"budget={args.neg_sample_budget:,})...")
    rng = np.random.default_rng(args.seed)
    mined_negs = clf._mine_hard_negatives(
        neg_pool, args.target_neg_per_stage, args.neg_sample_budget, rng,
        neg_seed=neg_seed)
    print(f"Mined {len(mined_negs):,} hard negatives")

    if len(mined_negs) == 0:
        print("ERROR: no hard negatives mined. Cannot extend stage.")
        return

    # ---- 4. Build feature matrices ----
    print("\nBuilding features...")
    features = build_features(res, res)
    print(f"  {len(features):,} features")
    feat_idx_map = _build_feature_index(features)

    # Positive features (cached)
    cache_dir = os.path.join(data_dir, "_cache") + os.sep
    pos_cache = cache_dir + "xf_pos" + pos_cache_suffix + ".npy"
    val_cache = cache_dir + "xf_val.npy"

    print("Loading cached positive features (memmap)...")
    X_f_pos = np.load(pos_cache, mmap_mode='r')
    X_f_val = np.load(val_cache, mmap_mode='r')

    # Negative features (compute fresh from mined negatives)
    print("Computing integral images for negatives...")
    neg_ii = np.array(list(map(integral_image, mined_negs)), dtype=np.uint32)
    neg_std = _sample_stds(mined_negs)

    print("Applying features to negatives...")
    X_f_neg = apply_features(neg_ii, features).astype(np.float32)
    X_f_neg /= neg_std[np.newaxis, :]
    del neg_ii  # free memory

    # Concatenate into training matrix
    X_f_stage = np.concatenate([X_f_pos, X_f_neg], axis=1)
    y_stage = np.concatenate([
        np.ones(X_f_pos.shape[1], dtype=np.float32),
        np.zeros(X_f_neg.shape[1], dtype=np.float32),
    ])
    del X_f_neg
    n_features, n_samples = X_f_stage.shape
    pos_num = float(np.sum(y_stage))
    neg_num = float(n_samples - pos_num)
    neg_mask = (y_stage == 0)

    # Val pos std for post-training calibration
    X_val_ii = np.array(list(map(integral_image, val_pos)), dtype=np.uint32)
    X_val_std = _sample_stds(val_pos)

    print(f"\nTraining matrix: {n_features:,} features × {n_samples:,} "
          f"samples ({int(pos_num):,} pos + {int(neg_num):,} neg)")

    # ---- 5. Replay saved rounds to reconstruct AdaBoost state ----
    print(f"\nReplaying {n_existing} saved WC rounds to reconstruct "
          f"AdaBoost state...")

    # Resolve feature indices for all saved WCs
    wc_indices = []
    for i, wc in enumerate(saved_wcs):
        key = _feature_key(wc.haar_feature)
        j = feat_idx_map.get(key)
        if j is None:
            print(f"ERROR: WC {i} feature not found in feature list. "
                  f"Cannot replay.")
            return
        wc_indices.append(j)

    # Initialize weights exactly as AdaBoost.train does
    weights = np.where(y_stage == 1,
                       0.5 / pos_num,
                       0.5 / neg_num).astype(np.float32)

    # Running score accumulators for calibrated FPR mode
    running_neg_scores = np.zeros(int(neg_mask.sum()), dtype=np.float64)
    running_val_scores = np.zeros(X_f_val.shape[1], dtype=np.float64)
    sum_alpha = 0.0

    replay_start = time.time()
    for t in tqdm(range(n_existing), desc="Replaying rounds", unit="round"):
        # Normalize weights (exactly as train does)
        w_sum = float(weights.sum())
        if w_sum == 0.0:
            print(f"[Replay] weights collapsed at round {t}")
            break
        weights /= w_sum

        wc = saved_wcs[t]
        alpha = saved_alphas[t]
        j = wc_indices[t]

        # Compute predictions on training data using the feature matrix
        fv = X_f_stage[j]
        preds = (wc.polarity * fv < wc.polarity * wc.threshold).astype(
            np.float32)

        # Compute error under current weights (for beta)
        err = float((weights * np.abs(preds - y_stage)).sum())
        err_clipped = float(np.clip(err, 1e-10, 0.5 - 1e-10))
        beta = err_clipped / (1.0 - err_clipped)

        # Update weights
        incorrectness = np.abs(preds - y_stage)
        weights = weights * (beta ** (1.0 - incorrectness))

        # Update running scores
        running_neg_scores += alpha * preds[neg_mask]
        sum_alpha += alpha

        # Update val scores
        fv_val = X_f_val[j]
        preds_val = (wc.polarity * fv_val <
                     wc.polarity * wc.threshold).astype(np.float32)
        running_val_scores += alpha * preds_val

    replay_time = time.time() - replay_start
    print(f"Replay done in {replay_time:.1f}s")

    # Check current state after replay
    if sum_alpha > 0:
        val_scores = running_val_scores / sum_alpha
        neg_scores_norm = running_neg_scores / sum_alpha
        # Calibrate threshold at current state
        new_ab = AdaBoost(n_estimators=args.new_max_wcs)
        new_ab.clfs = list(saved_wcs)
        new_ab.alphas = list(saved_alphas)
        op_thr = new_ab._calibrated_threshold(val_scores, args.layer_recall)
        cur_fpr = float((neg_scores_norm >= op_thr).mean())
        cur_recall = float((val_scores >= op_thr).mean())
        print(f"\nState after replay of {n_existing} rounds:")
        print(f"  FPR={cur_fpr:.4f}  recall={cur_recall:.4f}  "
              f"threshold={op_thr:.4f}")

    # ---- 6. Continue boosting ----
    n_extra = args.new_max_wcs - n_existing
    print(f"\nContinuing boosting: rounds {n_existing+1}..{args.new_max_wcs} "
          f"({n_extra} new rounds)")

    # Precompute sorting indices if requested (disk-backed memmap).
    sort_idx = None
    sort_idx_path = None
    if args.precompute_sort_index:
        sort_idx_dir = os.path.join(data_dir, "_cache")
        os.makedirs(sort_idx_dir, exist_ok=True)
        sort_idx_path = os.path.join(sort_idx_dir, "_sort_idx_extend.npy")
        idx_dtype = np.uint16 if n_samples <= 65535 else np.int32
        print(f"Precomputing sort index ({np.dtype(idx_dtype).name}) → {sort_idx_path} ...")
        start_sort = time.time()
        _wmap = np.memmap(sort_idx_path, dtype=idx_dtype, mode='w+',
                          shape=(n_features, n_samples))
        for f0 in range(0, n_features, 500):
            f1 = min(f0 + 500, n_features)
            _wmap[f0:f1] = np.argsort(
                X_f_stage[f0:f1], axis=1, kind='quicksort').astype(idx_dtype)
        _wmap.flush()
        del _wmap
        sort_idx = np.memmap(sort_idx_path, dtype=idx_dtype, mode='r',
                             shape=(n_features, n_samples))
        _bytes = n_features * n_samples * np.dtype(idx_dtype).itemsize
        print("  Sort index precomputed in {:.1f}s  "
              "({:.1f} GB on disk, paged on demand)".format(
                  time.time() - start_sort, _bytes / 1e9))

    target_fpr = args.target_stage_fpr
    target_recall = args.layer_recall

    # Re-create AdaBoost with the extended WC list
    ab = AdaBoost(n_estimators=args.new_max_wcs,
                  min_estimators=n_existing + 1)
    ab.clfs = list(saved_wcs)
    ab.alphas = list(saved_alphas)

    boost_start = time.time()
    pbar = tqdm(range(n_existing, args.new_max_wcs),
                desc='Extending boost', unit='round', leave=True)
    for t in pbar:
        w_sum = float(weights.sum())
        if w_sum == 0.0:
            pbar.write(f"[Extend] weights collapsed at round {t}")
            break
        weights /= w_sum

        best_j, thr, pol, err, preds = ab._best_stump(
            X_f_stage, y_stage, weights, feature_chunk=500, sort_idx=sort_idx)

        if err >= 0.5:
            pbar.write(f"[Extend] best weak error={err:.4f} >= 0.5; "
                       f"stopping at round {t}/{args.new_max_wcs}")
            break

        err_clipped = float(np.clip(err, 1e-10, 0.5 - 1e-10))
        beta = err_clipped / (1.0 - err_clipped)
        alpha = math.log(1.0 / beta)
        incorrectness = np.abs(preds - y_stage)
        weights = weights * (beta ** (1.0 - incorrectness))

        wc = WeakClassifier(haar_feature=features[best_j],
                            threshold=float(thr), polarity=int(pol))
        ab.alphas.append(float(alpha))
        ab.clfs.append(wc)

        # Update running scores
        running_neg_scores += alpha * preds[neg_mask]
        sum_alpha += alpha

        fv_val = X_f_val[best_j]
        preds_val = (pol * fv_val < pol * thr).astype(np.float32)
        running_val_scores += alpha * preds_val

        val_scores = running_val_scores / sum_alpha
        op_thr = ab._calibrated_threshold(val_scores, target_recall)
        ab.threshold = op_thr
        recall_at_thr = float((val_scores >= op_thr).mean())
        fpr = float((running_neg_scores / sum_alpha >= op_thr).mean())

        ab.final_fpr = fpr
        ab.final_recall = recall_at_thr

        pbar.set_postfix(err=f'{err:.4f}', alpha=f'{alpha:.3f}',
                         thr=f'{op_thr:.3f}', rec=f'{recall_at_thr:.3f}',
                         fpr=f'{fpr:.3f}')

        if (fpr <= target_fpr and recall_at_thr >= target_recall
                and len(ab.clfs) >= ab.min_estimators):
            pbar.write(f"[Extend] FPR {fpr:.3f} <= {target_fpr:.3f} & "
                       f"recall {recall_at_thr:.3f} >= {target_recall:.3f}; "
                       f"early stop at round {t+1}/{args.new_max_wcs}")
            break

    print(f"\n  Extended stage: {len(ab.clfs)} total WCs "
          f"({len(ab.clfs) - n_existing} new) in "
          f"{get_pretty_time(boost_start)}")

    # Clean up memmap sort index file if we created one.
    if sort_idx_path is not None:
        del sort_idx
        try:
            os.remove(sort_idx_path)
        except OSError:
            pass

    # ---- 7. Final calibration and save ----
    ab.calibrate(X_val_ii, X_val_std, target_recall=args.layer_recall)
    print(f"  Final threshold (after calibrate): {ab.threshold:.4f}")
    if ab.final_fpr is not None:
        print(f"  Final FPR: {ab.final_fpr:.4f}")
    if ab.final_recall is not None:
        print(f"  Final recall: {ab.final_recall:.4f}")

    # Push extended stage back
    clf.clfs.append(ab)
    print(f"\n  Cascade: {len(clf.clfs)} stages, "
          f"T per stage = {[len(s.clfs) for s in clf.clfs]}")

    # Cumulative val recall check
    n_pass = sum(1 for ii, s in zip(X_val_ii, X_val_std)
                 if clf.classify_ii(ii, std=float(s)) == 1)
    cum_recall = n_pass / len(X_val_ii)
    print(f"  Cumulative val recall: {cum_recall:.4f}")

    # Save
    out = args.output
    if out is None:
        from pathlib import Path
        p = Path(args.checkpoint)
        stem = p.stem if not p.stem.endswith(".pkl") else p.stem[:-4]
        out = str(p.parent / f"{stem}_ext{args.new_max_wcs}.pkl")
    save_path = out[:-4] if out.endswith(".pkl") else out
    clf.save(save_path)
    print(f"\nSaved → {save_path}.pkl")
    print(get_pretty_time(start_time, s="Total time: "))


if __name__ == "__main__":
    main()
