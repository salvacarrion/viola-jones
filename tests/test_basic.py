import unittest
import numpy as np

from utils import RectangleRegion, integral_image, integral_image_pow2

# Common variables. Numpy convention — img[row, col].
img = np.array([[5, 2, 5, 2],
                [3, 6, 3, 6],
                [5, 2, 5, 2],
                [3, 6, 3, 6]], dtype=np.uint8)


class TestAll(unittest.TestCase):

    def test_integral_image(self):
        # Padded II: shape (h+1, w+1), first row/col are zero so rectangle
        # sums reduce to four unconditional reads.
        ii = integral_image(img)
        ii_reference = np.array([[0,  0,  0,  0,  0],
                                 [0,  5,  7, 12, 14],
                                 [0,  8, 16, 24, 32],
                                 [0, 13, 23, 36, 46],
                                 [0, 16, 32, 48, 64]])
        self.assertTrue(np.array_equal(ii, ii_reference))

    def test_integral_image_pow2(self):
        ii = integral_image_pow2(img)
        ii_reference = np.array([[0,  0,   0,   0,   0],
                                 [0, 25,  29,  54,  58],
                                 [0, 34,  74, 108, 148],
                                 [0, 59, 103, 162, 206],
                                 [0, 68, 148, 216, 296]])
        self.assertTrue(np.array_equal(ii, ii_reference))

    def test_compute_region(self):
        # RectangleRegion(x=col, y=row, width=col-span, height=row-span)
        ii = integral_image(img)

        # 1×1 at (col=0, row=0) → img[0,0] = 5
        self.assertEqual(RectangleRegion(0, 0, 1, 1).compute_region(ii), 5)
        # 2×2 at top-left → sum img[0:2, 0:2] = 5+2+3+6 = 16
        self.assertEqual(RectangleRegion(0, 0, 2, 2).compute_region(ii), 16)
        # 3×3 at top-left → sum img[0:3, 0:3] = 36
        self.assertEqual(RectangleRegion(0, 0, 3, 3).compute_region(ii), 36)
        # 1×1 at (col=1, row=1) → img[1,1] = 6
        self.assertEqual(RectangleRegion(1, 1, 1, 1).compute_region(ii), 6)
        # 2×2 at (col=1, row=1) → img[1:3, 1:3] = 6+3+2+5 = 16
        self.assertEqual(RectangleRegion(1, 1, 2, 2).compute_region(ii), 16)
        # Non-square: 1×2 at (col=0, row=0) → img[0:2, 0:1] = 5+3 = 8
        self.assertEqual(RectangleRegion(0, 0, 1, 2).compute_region(ii), 8)
        # Non-square: 2×1 at (col=0, row=0) → img[0:1, 0:2] = 5+2 = 7
        self.assertEqual(RectangleRegion(0, 0, 2, 1).compute_region(ii), 7)
        # With offset (full-image II query): same 1×1 at (col=0,row=0) but
        # window starts at (ox=1, oy=1) → reads img[1, 1] = 6
        self.assertEqual(
            RectangleRegion(0, 0, 1, 1).compute_region(ii, ox=1, oy=1), 6)

    def test_adaboost_precomputed_sort_index(self):
        from adaboost import AdaBoost
        from utils import build_features, integral_image
        import numpy as np
        import tempfile
        import os

        # Generate some synthetic training data
        rng = np.random.default_rng(42)
        n_pos = 20
        n_neg = 30
        res = 24

        # Synthetic images
        pos_images = rng.integers(0, 256, (n_pos, res, res), dtype=np.uint8)
        neg_images = rng.integers(0, 256, (n_neg, res, res), dtype=np.uint8)

        pos_ii = np.array(list(map(integral_image, pos_images)), dtype=np.uint32)
        neg_ii = np.array(list(map(integral_image, neg_images)), dtype=np.uint32)

        pos_std = np.maximum(pos_images.reshape(n_pos, -1).std(axis=1), 1.0).astype(np.float32)
        neg_std = np.maximum(neg_images.reshape(n_neg, -1).std(axis=1), 1.0).astype(np.float32)

        # Build features
        features = build_features(res, res)[:100]  # Just use the first 100 features for speed

        from utils import apply_features
        X_pos = apply_features(pos_ii, features).astype(np.float32) / pos_std[np.newaxis, :]
        X_neg = apply_features(neg_ii, features).astype(np.float32) / neg_std[np.newaxis, :]

        X = np.concatenate([X_pos, X_neg], axis=1)
        y = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)])

        # Train without precompute
        ab_no_pre = AdaBoost(n_estimators=5)
        ab_no_pre.train(X, y, features, feature_chunk=20, precompute_sort_index=False)

        # Train with precompute (memmap-backed)
        with tempfile.TemporaryDirectory() as tmpdir:
            ab_pre = AdaBoost(n_estimators=5)
            ab_pre.train(X, y, features, feature_chunk=20,
                         precompute_sort_index=True, sort_index_dir=tmpdir)
            # Verify the memmap file was cleaned up
            leftover = [f for f in os.listdir(tmpdir) if f.endswith('.npy')]
            self.assertEqual(leftover, [],
                             "Sort index memmap should be cleaned up after training")

        # Verify classifiers are identical
        self.assertEqual(len(ab_no_pre.clfs), len(ab_pre.clfs))
        for clf1, clf2 in zip(ab_no_pre.clfs, ab_pre.clfs):
            self.assertEqual(clf1.threshold, clf2.threshold)
            self.assertEqual(clf1.polarity, clf2.polarity)
            # Compare feature geometry
            self.assertEqual(clf1.haar_feature.positive_regions, clf2.haar_feature.positive_regions)
            self.assertEqual(clf1.haar_feature.negative_regions, clf2.haar_feature.negative_regions)

        # Verify alphas are identical
        self.assertTrue(np.allclose(ab_no_pre.alphas, ab_pre.alphas))


if __name__ == "__main__":
    unittest.main()
