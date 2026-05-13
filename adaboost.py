import math
import time
import numpy as np
from tqdm.auto import tqdm

from weakclassifier import WeakClassifier


class AdaBoost:
    def __init__(self, n_estimators=10, min_estimators=1):
        self.n_estimators = n_estimators
        # Floor on the adaptive early-stop: a stage with only 1-2 stumps has
        # a 0/1-valued score, so the post-training `calibrate()` can't pick
        # a useful operating point and the layer threshold collapses to the
        # 0.95 cap, accepting only windows where that single stump fires.
        # Keep adding rounds until at least `min_estimators` are in before
        # the FPR check can fire.
        self.min_estimators = min_estimators
        self.alphas = []
        self.clfs = []
        # Acceptance threshold as a fraction of sum(alphas) in [0, 1].
        # 0.5 = plain weighted majority vote (default AdaBoost). After training,
        # `calibrate()` lowers this so the layer keeps ~all training faces — see
        # §4.2 of Viola & Jones (2001).
        self.threshold = 0.5

    # [OPT] feature_chunk=500 (was 2000): caps per-batch intermediates in
    # _best_stump at ~0.75 GB instead of ~3 GB. Bit-identical results;
    # slightly more loop iterations but much less memory pressure.
    def train(self, X, y, features, feature_chunk=500, target_stage_fpr=None,
              X_val=None, target_recall=None):
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
            X_val:      (n_features, n_val_pos) variance-normalized feature
                        matrix for the validation positives used to set the
                        per-round calibrated threshold. Required for the
                        "calibrated FPR" early-stop path described below.
            target_recall: face-recall target for the per-round threshold
                        calibration (e.g. 0.99). When `target_recall` and
                        `X_val` are both set, `self.threshold` is recomputed
                        every round so that at least this fraction of val_pos
                        passes, and the FPR early-stop is evaluated AT that
                        calibrated threshold — not at the un-calibrated 0.5.
                        This matches V&J §3 (d ≥ 0.997 AND f ≤ 0.5 at the
                        operating point). When omitted, the FPR check falls
                        back to threshold=0.5 (legacy behavior).

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

        calibrated_mode = (target_stage_fpr is not None
                           and X_val is not None
                           and target_recall is not None)
        if calibrated_mode:
            assert X_val.shape[0] == n_features, (
                "X_val must have the same feature axis as X "
                "({} vs {}).".format(X_val.shape[0], n_features))

        # Initial weights: balanced between classes (Viola-Jones §3, eq. 1).
        weights = np.where(y == 1, 0.5 / pos_num, 0.5 / neg_num).astype(np.float32)

        # Adaptive FPR mode: accumulate weighted scores for the stage
        # negatives (and, in calibrated mode, the validation positives) so
        # we can probe the operating point each round without re-scoring
        # from scratch.
        neg_mask = (y == 0)
        if target_stage_fpr is not None:
            running_neg_scores = np.zeros(int(neg_mask.sum()), dtype=np.float64)
            sum_alpha_fpr = 0.0
            if calibrated_mode:
                running_val_scores = np.zeros(X_val.shape[1], dtype=np.float64)

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
            if target_stage_fpr is not None:
                running_neg_scores += alpha * preds[neg_mask]
                sum_alpha_fpr += alpha
                recall_at_thr = 1.0  # legacy mode assumes recall is honored
                if calibrated_mode:
                    # Score the just-chosen WC on val_pos and update the
                    # running val score, then recalibrate `self.threshold`
                    # to keep `target_recall` of val_pos passing.
                    fv_val = X_val[best_j]
                    preds_val = (pol * fv_val < pol * thr).astype(np.float32)
                    running_val_scores += alpha * preds_val
                    val_scores = running_val_scores / sum_alpha_fpr
                    op_thr = self._calibrated_threshold(val_scores, target_recall)
                    self.threshold = op_thr
                    # Recall AT the calibrated threshold. With few weak
                    # classifiers the score is so coarse (2-3 distinct
                    # values) that the min_thr floor / 0.95 cap inside
                    # `_calibrated_threshold` can clamp the threshold above
                    # the target-recall quantile — meaning the operating
                    # point silently under-recalls. We measure recall
                    # explicitly so the early-stop refuses to fire until
                    # both d ≥ target_recall AND f ≤ target_stage_fpr can
                    # be honored simultaneously (V&J §3).
                    recall_at_thr = float((val_scores >= op_thr).mean())
                    # FPR at the actual operating point — the metric that
                    # matches the deployed cascade behavior.
                    fpr = float((running_neg_scores / sum_alpha_fpr >= op_thr).mean())
                    pbar.set_postfix(err='{:.4f}'.format(err),
                                     alpha='{:.3f}'.format(alpha),
                                     thr='{:.3f}'.format(op_thr),
                                     rec='{:.3f}'.format(recall_at_thr),
                                     fpr='{:.3f}'.format(fpr))
                else:
                    # Legacy: FPR at the un-calibrated 0.5 majority-vote
                    # threshold. Reported FPR will under-estimate the real
                    # operating-point FPR once `calibrate()` lowers the
                    # threshold to keep faces.
                    fpr = float((running_neg_scores / sum_alpha_fpr >= 0.5).mean())
                    pbar.set_postfix(err='{:.4f}'.format(err),
                                     alpha='{:.3f}'.format(alpha),
                                     fpr='{:.3f}'.format(fpr))
                recall_ok = (not calibrated_mode) or (recall_at_thr >= target_recall)
                if (fpr <= target_stage_fpr
                        and recall_ok
                        and len(self.clfs) >= self.min_estimators):
                    pbar.write("[AdaBoost] FPR {:.3f} ≤ {:.3f} & recall {:.3f} ≥ "
                               "{:.3f}; early stop at round {}/{}".format(
                                   fpr, target_stage_fpr,
                                   recall_at_thr if calibrated_mode else 1.0,
                                   target_recall if calibrated_mode else 0.0,
                                   t + 1, self.n_estimators))
                    break
            else:
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
        self.threshold = self._calibrated_threshold(scores, target_recall)

    def _calibrated_threshold(self, val_scores, target_recall):
        """
        Pick the largest threshold such that `target_recall` of `val_scores`
        is ≥ threshold. Shared by `calibrate()` (post-stage, integral-image
        path) and the per-round calibration inside `train()` (running
        feature-matrix path).

        Includes two safety bounds:
          - Floor at `min(alphas)/sum(alphas)`: the score when only the
            lowest-alpha weak classifier fires. With small T and high
            target_recall a few faces may genuinely score 0; without the
            floor the threshold collapses to 0 and the stage accepts every
            window — flat patches included — defeating the cascade.
          - Cap at 0.95: guards against the degenerate "threshold ≈ 1.0
            accepts nothing" failure mode when a single dominant WC pushes
            the recall-quantile to ~1.0.
        """
        sorted_desc = np.sort(np.asarray(val_scores, dtype=np.float64))[::-1]
        # We want fraction(scores >= threshold) >= target_recall, so we set
        # the threshold to the score at the ceil(target_recall * n)-th-
        # largest position.
        k = max(1, int(np.ceil(target_recall * len(sorted_desc))))
        sum_alpha = sum(self.alphas)
        min_thr = (min(self.alphas) / sum_alpha) if sum_alpha > 0 else 0.0
        return float(min(0.95, max(min_thr, sorted_desc[k - 1])))


def _fmt_time(start):
    """Helper for short MM:SS.s prints."""
    elapsed = time.time() - start
    m, s = divmod(elapsed, 60)
    return "{:02d}:{:05.2f}".format(int(m), s)
