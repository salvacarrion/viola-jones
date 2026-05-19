class WeakClassifier:
    """Single-Haar-feature decision stump. Constructed by `AdaBoost._best_stump`."""

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
