"""
Per-stage cascade diagnostic on the CBCL test set.

Reports, for each stage:
  - layer_threshold (after calibration)
  - n_clfs (T)
  - pass-rate on CBCL faces  (should stay high)
  - pass-rate on CBCL non-faces (should drop fast)
  - per-stage rejection of non-faces

Plus the score-distribution overlap (positive vs negative scores) per stage,
which tells you whether each stage *could* reject more if calibrated tighter.
"""

import argparse
import os
import sys
import numpy as np

# Repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import integral_image  # noqa: E402
from violajones import ViolaJones  # noqa: E402


def per_stage_scores(clf: ViolaJones, X):
    """Score every sample at every stage. Returns (n_samples, n_stages) float32."""
    n = len(X)
    S = np.zeros((n, len(clf.clfs)), dtype=np.float32)
    for i, img in enumerate(X):
        ii = integral_image(img)
        std = max(float(np.std(img)), 1.0)
        for s, stage in enumerate(clf.clfs):
            S[i, s] = stage.score(ii, scale=1.0, std=std)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-neg", type=int, default=5000,
                    help="Cap negatives sampled for speed")
    args = ap.parse_args()

    clf = ViolaJones.load(args.weights)
    pos = np.load(os.path.join(args.data_dir, "cbcl_test_pos.npy"))
    neg = np.load(os.path.join(args.data_dir, "cbcl_test_neg.npy"))
    if len(neg) > args.max_neg:
        rng = np.random.default_rng(0)
        neg = neg[rng.choice(len(neg), size=args.max_neg, replace=False)]

    print(f"Cascade: {len(clf.clfs)} stages, T per stage = "
          f"{[len(s.clfs) for s in clf.clfs]}")
    print(f"Layer thresholds: {[round(s.threshold, 4) for s in clf.clfs]}")
    print(f"CBCL test eval set: {len(pos)} faces, {len(neg)} non-faces "
          f"@ {pos.shape[1]}×{pos.shape[2]}\n")

    Sp = per_stage_scores(clf, pos)   # (P, K)
    Sn = per_stage_scores(clf, neg)   # (N, K)

    print(f"{'stage':>5} {'T':>4} {'thr':>7} | "
          f"{'pos μ':>7} {'pos p1':>7} {'neg μ':>7} {'neg p99':>7} | "
          f"{'pass+':>7} {'pass-':>7} {'rej- this':>10}")
    print("-" * 88)

    pass_pos = np.ones(len(pos), dtype=bool)
    pass_neg = np.ones(len(neg), dtype=bool)
    for s in range(len(clf.clfs)):
        thr = clf.clfs[s].threshold
        p_passes = Sp[:, s] >= thr
        n_passes = Sn[:, s] >= thr
        pass_pos_new = pass_pos & p_passes
        pass_neg_new = pass_neg & n_passes
        # Rejection of negatives this stage adds (among those that passed prior stages)
        rej_this = 1.0 - pass_neg_new.sum() / max(pass_neg.sum(), 1)
        print(f"{s+1:>5} {len(clf.clfs[s].clfs):>4} {thr:>7.4f} | "
              f"{Sp[:, s].mean():>7.3f} {np.percentile(Sp[:, s], 1):>7.3f} "
              f"{Sn[:, s].mean():>7.3f} {np.percentile(Sn[:, s], 99):>7.3f} | "
              f"{pass_pos_new.mean():>7.3f} {pass_neg_new.mean():>7.3f} "
              f"{rej_this:>10.3f}")
        pass_pos, pass_neg = pass_pos_new, pass_neg_new

    print()
    print(f"FINAL: recall = {pass_pos.mean():.3f}   "
          f"FPR = {pass_neg.mean():.3f}   "
          f"specificity = {1 - pass_neg.mean():.3f}")


if __name__ == "__main__":
    main()
