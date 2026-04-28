import unittest
import numpy as np

from weakclassifier import WeakClassifier
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

    def test_weak_classifier(self):
        X = np.array([1, 3, 4, 6, 9, 10, 12, 14])
        y = np.array([1, 0, 0, 0, 1, 1, 1, 1])
        w = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        clf = WeakClassifier()
        clf.train(X, y, w)

        self.assertTrue(clf.threshold == 9)
        self.assertTrue(clf.classify_f(6) == 0)
        self.assertTrue(clf.classify_f(11) == 1)


if __name__ == "__main__":
    unittest.main()
