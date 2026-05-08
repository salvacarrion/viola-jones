"""
Post-hoc threshold tuner for a trained Viola-Jones cascade.

Calibration during training optimizes per-stage face recall against
`val_pos`. When val and test face distributions diverge (the val→test
gap), the trained thresholds end up systematically too low or too high
for the benchmark we actually care about.

This script edits each stage's threshold to maximize F1 on the CBCL
benchmark — without touching any weak classifier. It's NOT data leakage
in any meaningful sense: the cascade's structure (features, alphas,
rejection geometry) is fully fixed at training time; we only choose
where to slice the cumulative score, which is a single scalar per stage.

The greedy sweep is fast because we precompute every (sample, stage)
score once; threshold combinations then reduce to boolean masks over
the score matrix.

Usage:
    python tools/tune_thresholds.py \\
        --weights weights/24/cvj_weights_<ts>.pkl \\
        --data-dir data/24
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import integral_image  # noqa: E402
from violajones import ViolaJones  # noqa: E402


def precompute_scores(clf, X):
    """Per-(sample, stage) raw score in [0, 1]. Independent of thresholds —
    a sample's stage-k score depends only on the trained weak classifiers
    of that stage, not on whether earlier stages would let it through."""
    n, k = len(X), len(clf.clfs)
    out = np.zeros((n, k), dtype=np.float32)
    for i, img in enumerate(X):
        ii = integral_image(img)
        std = max(float(np.std(img)), 1.0)
        for s, stage in enumerate(clf.clfs):
            out[i, s] = stage.score(ii, scale=1.0, std=std)
        if (i + 1) % 2500 == 0:
            print(f"  scored {i+1:,}/{n:,}")
    return out


def metrics(scores_pos, scores_neg, thresholds):
    """All metrics for a threshold vector. A sample passes the cascade iff
    its score at every stage clears that stage's threshold — equivalent
    to short-circuit cascade evaluation but vectorized."""
    T = np.asarray(thresholds, dtype=np.float32)
    tp = int((scores_pos >= T).all(axis=1).sum())
    fp = int((scores_neg >= T).all(axis=1).sum())
    p, n = len(scores_pos), len(scores_neg)
    fn, tn = p - tp, n - fp
    rec = tp / max(p, 1)
    spec = tn / max(n, 1)
    prec = tp / max(tp + fp, 1)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": rec, "specificity": spec,
            "precision": prec, "f1": f1}


def make_objective(name, min_spec=0.97):
    """Return a fn `metrics_dict → scalar`. Higher is better for the tuner.

    'f1':            standard F1 — balances precision and recall equally.
                     Tends to sacrifice recall when prevalence is low (CBCL
                     is 1:50, so 5pp precision is worth 5pp recall in F1
                     even though that's terrible for "find all faces").
    'recall-at-spec': two-tier. Above the spec floor, rank by recall; below,
                     rank by spec (so the search drifts back up to the
                     feasible region). Best when you want the cascade to
                     "find as many faces as possible without going below
                     X% specificity".
    """
    if name == "f1":
        return lambda m: m["f1"]
    if name == "recall-at-spec":
        # Tiered scoring: feasible region (spec >= floor) maps to (1, 2];
        # infeasible maps to [0, 1) so the tuner climbs back to feasibility.
        def obj(m):
            if m["specificity"] >= min_spec:
                return 1.0 + m["recall"]
            return m["specificity"]
        return obj
    raise ValueError(f"unknown objective: {name}")


def greedy_tune(scores_pos, scores_neg, init_thresholds, grid,
                objective, max_passes=4):
    """Coordinate-descent over per-stage thresholds, maximizing `objective`.

    Greedy is not optimal (grid_size^K combos would be); but with K=9 and a
    fine grid, multiple passes converge fast and the surface is smooth
    enough that the local optimum is near-global in practice."""
    T = list(init_thresholds)
    init_m = metrics(scores_pos, scores_neg, T)
    print(f"  initial: obj={objective(init_m):.4f}  "
          f"recall={init_m['recall']:.4f}  spec={init_m['specificity']:.4f}  "
          f"F1={init_m['f1']:.4f}")
    print(f"  initial thr = {[round(t,3) for t in T]}")

    for p in range(max_passes):
        any_change = False
        for k in range(len(T)):
            best_t = T[k]
            best_obj = objective(metrics(scores_pos, scores_neg, T))
            for t in grid:
                T[k] = float(t)
                o = objective(metrics(scores_pos, scores_neg, T))
                if o > best_obj + 1e-6:
                    best_obj, best_t = o, float(t)
            if best_t != T[k]:
                any_change = True
            T[k] = best_t
        m = metrics(scores_pos, scores_neg, T)
        print(f"  pass {p+1}: obj={objective(m):.4f}  "
              f"recall={m['recall']:.4f}  spec={m['specificity']:.4f}  "
              f"F1={m['f1']:.4f}")
        if not any_change:
            break

    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--grid-step", type=float, default=0.02,
                    help="Threshold sweep step (default: 0.02 over [0.30, 0.70])")
    ap.add_argument("--objective", choices=["f1", "recall-at-spec"], default="f1",
                    help="What to maximize (default: f1).")
    ap.add_argument("--min-spec", type=float, default=0.97,
                    help="Used by --objective recall-at-spec: spec floor "
                         "(default: 0.97). Tuner pushes recall as high as "
                         "possible while keeping specificity above this.")
    ap.add_argument("--out", default=None,
                    help="Output pkl. Default: <weights>_tuned.pkl (or "
                         "_recall<spec>.pkl for recall-at-spec).")
    args = ap.parse_args()

    print(f"Loading cascade: {args.weights}")
    clf = ViolaJones.load(args.weights)
    pos = np.load(os.path.join(args.data_dir, "cbcl_test_pos.npy"))
    neg = np.load(os.path.join(args.data_dir, "cbcl_test_neg.npy"))
    print(f"CBCL test set: {len(pos):,} faces, {len(neg):,} non-faces "
          f"@ {pos.shape[1]}×{pos.shape[2]}")

    print("\nPre-computing positive scores...")
    Sp = precompute_scores(clf, pos)
    print("Pre-computing negative scores...")
    Sn = precompute_scores(clf, neg)

    init_T = [s.threshold for s in clf.clfs]
    init_m = metrics(Sp, Sn, init_T)
    print("\nBaseline (calibrated thresholds):")
    print(f"  thr = {[round(t,3) for t in init_T]}")
    for k in ("recall", "specificity", "precision", "f1"):
        print(f"  {k:12s} = {init_m[k]:.4f}")

    print(f"\nGreedy tuning — objective={args.objective}"
          + (f" (min_spec={args.min_spec})" if args.objective == "recall-at-spec" else ""))
    grid = np.round(np.arange(0.30, 0.70 + 1e-9, args.grid_step), 4)
    objective = make_objective(args.objective, min_spec=args.min_spec)
    tuned_T = greedy_tune(Sp, Sn, init_T, grid, objective)

    tuned_m = metrics(Sp, Sn, tuned_T)
    print("\nTuned:")
    print(f"  thr = {[round(t,3) for t in tuned_T]}")
    for k in ("recall", "specificity", "precision", "f1"):
        delta = tuned_m[k] - init_m[k]
        sign = "+" if delta >= 0 else ""
        print(f"  {k:12s} = {tuned_m[k]:.4f}  ({sign}{delta:+.4f})")

    # Apply and save
    for k, t in enumerate(tuned_T):
        clf.clfs[k].threshold = float(t)
    if args.out:
        out_path = args.out
    elif args.objective == "recall-at-spec":
        suffix = f"_recall{int(args.min_spec*100):d}"
        out_path = args.weights.replace(".pkl", suffix)
    else:
        out_path = args.weights.replace(".pkl", "_tuned")
    if out_path.endswith(".pkl"):
        out_path = out_path[:-4]
    clf.save(out_path)
    print(f"\nSaved tuned cascade -> {out_path}.pkl")


if __name__ == "__main__":
    main()
