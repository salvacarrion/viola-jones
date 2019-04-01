import unittest
import numpy as np

from weakclassifier import WeakClassifier
from utils import RectangleRegion, integral_image, integral_image_pow2

# Common variables
img = np.array([[5, 2, 5, 2],
                [3, 6, 3, 6],
                [5, 2, 5, 2],
                [3, 6, 3, 6]], dtype=np.uint8)


class TestAll(unittest.TestCase):

    def test_integral_image(self):
        # Create integral image
        ii = integral_image(img)
        ii_reference = np.array([[5,  7, 12, 14],
                                 [8, 16, 24, 32],
                                 [13, 23, 36, 46],
                                 [16, 32, 48, 64]])
        self.assertTrue(np.array_equal(ii, ii_reference))

    def test_integral_image_pow2(self):
        # Create integral image
        ii = integral_image_pow2(img)
        ii_reference = np.array([[25,  29,  54,  58],
                                 [34,  74, 108, 148],
                                 [59, 103, 162, 206],
                                 [68, 148, 216, 296]])
        self.assertTrue(np.array_equal(ii, ii_reference))

    def test_compute_region(self):

        # Create integral image
        ii = integral_image(img)

        # Set regions to test
        r1 = RectangleRegion(0, 0, 1, 1)
        r2 = RectangleRegion(0, 0, 2, 2)
        r3 = RectangleRegion(0, 0, 3, 3)
        r4 = RectangleRegion(1, 1, 1, 1)
        r5 = RectangleRegion(1, 1, 2, 2)

        self.assertEqual(r1.compute_region(ii), 5)
        self.assertEqual(r2.compute_region(ii), 16)
        self.assertEqual(r3.compute_region(ii), 36)
        self.assertEqual(r4.compute_region(ii), 6)
        self.assertEqual(r5.compute_region(ii), 16)

    def test_weak_classifier(self):
        X = np.array([1, 3, 4, 6, 9, 10, 12, 14])
        y = np.array([1, 0, 0, 0, 1, 1, 1, 1])
        w = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        # Train weak classifier
        clf = WeakClassifier()
        clf.train(X, y, w)

        self.assertTrue(clf.threshold == 9)
        self.assertTrue(clf.classify_f(6) == 0)
        self.assertTrue(clf.classify_f(11) == 1)


if __name__ == "__main__":
    unittest.main()
