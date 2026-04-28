import math
import time
import numpy as np
from tqdm.auto import tqdm

from weakclassifier import WeakClassifier


class AdaBoost:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.alphas = []
        self.clfs = []
        # Acceptance threshold as a fraction of sum(alphas) in [0, 1].
        # 0.5 = plain weighted majority vote (default AdaBoost). After training,
        # `calibrate()` lowers this so the layer keeps ~all training faces — see
        # §4.2 of Viola & Jones (2001).
        self.threshold = 0.5

    def train(self, X, y, features, feature_chunk=2000):
        """
        Boost `n_estimators` decision-stump rounds.

        Args:
            X:          (n_features, n_samples) feature-value matrix (already
                        variance-normalized by the caller).
            y:          (n_samples,) labels in {0, 1}.
            features:   list of HaarFeature, len == n_features. The chosen
                        weak classifier's `haar_feature` is set from this list.
            feature_chunk: features processed per vectorized batch — caps
                        the per-batch matrices at ~`feature_chunk × n_samples
                        × 4 bytes` floats. Tune down on memory pressure.

        The expensive step (per-feature optimal threshold + weighted error)
        is fully vectorized: for each batch of features we sort row-wise,
        run an exclusive prefix sum of weighted positives/negatives, and
        pick the (threshold, polarity) that minimizes weighted error in a
        single `np.argmin`. ~30-50× faster than the Python triple loop.
        """
        y = y.astype(np.float32, copy=False)
        n_features, n_samples = X.shape
        assert len(features) == n_features
        assert y.shape == (n_samples,)

        pos_num = float(np.sum(y))
        neg_num = float(n_samples - pos_num)
        if pos_num == 0 or neg_num == 0:
            raise ValueError("AdaBoost.train requires both positive and negative samples.")

        # Initial weights: balanced between classes (Viola-Jones §3, eq. 1).
        weights = np.where(y == 1, 0.5 / pos_num, 0.5 / neg_num).astype(np.float32)

        start_time = time.time()
        pbar = tqdm(range(self.n_estimators), desc='Boosting rounds',
                    unit='round', leave=True)
        for t in pbar:
            w_sum = float(weights.sum())
            if w_sum == 0.0:
                print("[AdaBoost] weights collapsed to zero at round {}".format(t))
                break
            weights /= w_sum

            best_j, thr, pol, err, preds = self._best_stump(
                X, y, weights, feature_chunk=feature_chunk)

            if err >= 0.5:
                # Halting condition: no weak classifier beats random under
                # the current weights → adding it gives alpha ≤ 0.
                pbar.write("[AdaBoost] best weak error={:.4f} ≥ 0.5; stopping "
                           "stage at round {}/{}".format(err, t, self.n_estimators))
                break

            # Clip err away from {0, 1} for numerical stability:
            #  - err == 0 (perfect on weighted train) gives beta=0; the
            #    weight update `beta**1 = 0` collapses all surviving weights
            #    to zero on the next round, so we cap alpha at the value
            #    implied by 1e-10 rather than blowing up to ∞.
            #  - tiny float drift can put err slightly negative, which would
            #    make beta negative and `math.log` raise a domain error.
            err_clipped = float(np.clip(err, 1e-10, 0.5 - 1e-10))
            beta = err_clipped / (1.0 - err_clipped)
            alpha = math.log(1.0 / beta)
            incorrectness = np.abs(preds - y)         # 0 if correct, 1 if wrong
            weights = weights * (beta ** (1.0 - incorrectness))

            clf = WeakClassifier(haar_feature=features[best_j],
                                 threshold=float(thr), polarity=int(pol))
            self.alphas.append(float(alpha))
            self.clfs.append(clf)
            pbar.set_postfix(err='{:.4f}'.format(err), alpha='{:.3f}'.format(alpha))

        print("\t- AdaBoost stage: {} weak classifiers in {}".format(
            len(self.clfs), _fmt_time(start_time)))

    def _best_stump(self, X, y, weights, feature_chunk):
        """
        Vectorized pass over all features. For each feature, find the
        (threshold, polarity) minimizing weighted error; then pick the
        single feature with lowest error.

        Returns (best_j, threshold, polarity, error, predictions).
        """
        n_features, n_samples = X.shape

        total_pos_w = float((weights * y).sum())
        total_neg_w = float(weights.sum() - total_pos_w)

        # Per-sample weight contribution if positive / negative
        pos_w = (weights * y).astype(np.float32)
        neg_w = (weights - pos_w).astype(np.float32)

        global_best_err = np.inf
        global_best = (0, 0.0, 1)  # (j, threshold, polarity)
        global_best_preds = None

        for f0 in range(0, n_features, feature_chunk):
            f1 = min(f0 + feature_chunk, n_features)
            X_chunk = X[f0:f1]                                  # (cf, n_samples)

            # Sort feature values per row
            sort_idx = np.argsort(X_chunk, axis=1, kind='quicksort')

            # Take per-feature sorted weight contributions
            sorted_pos_w = pos_w[sort_idx]                       # (cf, n_samples)
            sorted_neg_w = neg_w[sort_idx]
            # Exclusive prefix sums = cumsum minus current
            cum_pos = np.cumsum(sorted_pos_w, axis=1) - sorted_pos_w
            cum_neg = np.cumsum(sorted_neg_w, axis=1) - sorted_neg_w

            # err_pos: polarity=+1 (face = below threshold).
            # err_neg: polarity=-1 (face = above threshold).
            err_pos = cum_neg + (total_pos_w - cum_pos)
            err_neg = cum_pos + (total_neg_w - cum_neg)
            err = np.minimum(err_pos, err_neg)                   # (cf, n_samples)

            # Best threshold position per feature (in sorted order)
            best_idx = np.argmin(err, axis=1)                    # (cf,)
            rows = np.arange(f1 - f0)
            chunk_best_err = err[rows, best_idx]                 # (cf,)

            # Single best feature in this chunk
            local_j = int(np.argmin(chunk_best_err))
            local_err = float(chunk_best_err[local_j])
            if local_err < global_best_err:
                global_best_err = local_err
                # Threshold = feature value at the best sorted position
                sorted_X = np.take_along_axis(X_chunk, sort_idx, axis=1)
                thr = float(sorted_X[local_j, best_idx[local_j]])
                pol = 1 if err_pos[local_j, best_idx[local_j]] < err_neg[local_j, best_idx[local_j]] else -1
                global_best = (f0 + local_j, thr, pol)
                # Predictions for the global best (so far): polarity * X < polarity * thr
                fv = X[f0 + local_j]
                global_best_preds = (pol * fv < pol * thr).astype(np.float32)

        j, thr, pol = global_best
        return j, thr, pol, global_best_err, global_best_preds

    def score(self, X, scale=1.0, std=1.0, ox=0, oy=0):
        """Weighted vote in [0, 1]; higher means more face-like."""
        denom = sum(self.alphas)
        if denom <= 0:
            return 0.0
        total = sum(a * c.classify(X, scale=scale, std=std, ox=ox, oy=oy)
                    for a, c in zip(self.alphas, self.clfs))
        return float(total) / float(denom)

    def classify(self, X, scale=1.0, std=1.0, ox=0, oy=0):
        return 1 if self.score(X, scale=scale, std=std, ox=ox, oy=oy) >= self.threshold else 0

    def calibrate(self, X_pos_ii, X_pos_std=None, target_recall=0.99):
        """
        Lower `self.threshold` so the layer accepts at least `target_recall`
        of the positive (face) integral images in `X_pos_ii`.

        Per-stage calibration from §4.2 of Viola-Jones: each layer commits to
        keeping ~all faces and contributes only rejection power against
        non-faces. Without it, deep cascades collapse face recall.

        `X_pos_std` is the per-sample pixel std used by the inference-time
        variance normalization. Pass the same stds you'll see at inference so
        the calibrated threshold matches deployment behavior.
        """
        if not self.clfs or len(X_pos_ii) == 0:
            return
        if X_pos_std is None:
            X_pos_std = np.ones(len(X_pos_ii), dtype=np.float64)
        scores = np.array(
            [self.score(ii, std=float(s)) for ii, s in zip(X_pos_ii, X_pos_std)],
            dtype=np.float64,
        )
        sorted_desc = np.sort(scores)[::-1]
        # We want fraction(scores >= threshold) >= target_recall, so we set
        # the threshold to the score at the ceil(target_recall * n)-th-
        # largest position.
        k = max(1, int(np.ceil(target_recall * len(sorted_desc))))
        # Don't go above the default 0.5 — calibration is allowed to *loosen*
        # the layer, never to tighten it past the un-calibrated AdaBoost rule.
        self.threshold = float(min(0.5, sorted_desc[k - 1]))


def _fmt_time(start):
    """Helper for short MM:SS.s prints."""
    elapsed = time.time() - start
    m, s = divmod(elapsed, 60)
    return "{:02d}:{:05.2f}".format(int(m), s)
