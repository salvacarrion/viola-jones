import numpy as np


class WeakClassifier:

    def __init__(self, haar_feature=None, threshold=None, polarity=None):
        self.haar_feature = haar_feature
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, ii, scale=1.0, std=1.0, ox=0, oy=0):
        """
        Classifies an integral image (per-window or full-image with offset).

        `std` implements per-window variance normalization (Viola-Jones §5.1).
        Training feature values were divided by each sample's pixel std, so the
        learned threshold is in "std-normalized" units. At inference we
        multiply the threshold back by the window's std (rather than dividing
        every pixel up front, which would invalidate the integral image).

        `ox`, `oy` are the window-origin offset into a full-image II — used by
        sliding-window detection to query features at absolute coordinates
        without recomputing the II per window. Pass 0,0 for per-window II.
        """
        feature_value = self.haar_feature.compute_value(ii, scale, ox, oy)
        return 1 if self.polarity * feature_value < self.polarity * self.threshold * (scale**2) * std else 0

    def classify_f(self, feature_value):
        """
        Classifies an image given its feature vale or array
        """
        a = self.polarity * feature_value
        b = self.polarity * self.threshold
        return np.less(a, b).astype(int)

    def train(self, X, y, weights, total_pos_weights=None, total_neg_weights=None):
        # Compute total pos/neg weights if not given
        if not total_pos_weights:
            total_pos_weights = np.sum(weights[np.where(y == 1)])
        if not total_neg_weights:
            total_neg_weights = np.sum(weights[np.where(y == 0)])

        # Sort features according to their numeric value
        sorted_features = sorted(zip(weights, X, y), key=lambda a: a[1])

        pos_seen, neg_seen = 0, 0
        sum_pos_weights, sum_neg_weights = 0, 0

        min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
        for w, f, label in sorted_features:
            # err_pos: error if polarity=+1 (face = below threshold) -> negatives below + positives above are wrong
            # err_neg: error if polarity=-1 (face = above threshold) -> positives below + negatives above are wrong
            err_pos = sum_neg_weights + (total_pos_weights - sum_pos_weights)
            err_neg = sum_pos_weights + (total_neg_weights - sum_neg_weights)
            error = min(err_pos, err_neg)

            # Save best values. Polarity must come from which branch won the min,
            # using the (weighted) errors — not raw pos/neg counts, which assume balanced classes.
            if error < min_error:
                min_error = error
                self.threshold = f  # Best feature value
                self.polarity = 1 if err_pos < err_neg else -1

            # Keep counts
            if label == 1:
                pos_seen += 1
                sum_pos_weights += w
            else:
                neg_seen += 1
                sum_neg_weights += w
