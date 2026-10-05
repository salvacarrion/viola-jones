import os
import unittest

from PIL import Image

from utils import apply_box_transform
from violajones import ViolaJones

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO, "weights/24/celeba.pkl")
IMAGE = os.path.join(REPO, "images/people.png")


class TestBoxTransform(unittest.TestCase):

    def test_none_is_identity(self):
        regions = [(10, 20, 34, 44, 0.5)]
        self.assertEqual(apply_box_transform(regions, None), regions)

    def test_scale_and_shift_about_centre(self):
        # 24×24 box centred at (22, 32); widen ×1.5, heighten ×2, move the
        # centre up by a quarter of the height. Score is carried through.
        out = apply_box_transform([(10, 20, 34, 44, 0.5)], (1.5, 2.0, 0.0, -0.25))
        x1, y1, x2, y2, score = out[0]
        self.assertAlmostEqual(x2 - x1, 36.0)
        self.assertAlmostEqual(y2 - y1, 48.0)
        self.assertAlmostEqual((x1 + x2) / 2, 22.0)
        self.assertAlmostEqual((y1 + y2) / 2, 26.0)
        self.assertEqual(score, 0.5)


@unittest.skipUnless(os.path.exists(WEIGHTS) and os.path.exists(IMAGE),
                     "needs the tracked weights and sample image")
class TestPyramidModes(unittest.TestCase):

    def test_modes_agree_at_base_scale(self):
        # At scale 1 both pyramids evaluate the same windows on the same
        # integral image, so they must return identical detections. They only
        # diverge at s > 1, where "features" truncates scaled rectangles.
        clf = ViolaJones.load(WEIGHTS)
        with Image.open(IMAGE) as f:
            img = f.copy()
        kw = dict(min_face_size=clf.base_width, max_face_size=clf.base_width)
        by_image = clf.find_faces(img, pyramid="image", **kw)
        by_features = clf.find_faces(img, pyramid="features", **kw)
        self.assertGreater(len(by_image), 0)
        self.assertEqual(by_image, by_features)

    def test_rejects_unknown_mode(self):
        clf = ViolaJones.load(WEIGHTS)
        with Image.open(IMAGE) as img, self.assertRaises(ValueError):
            clf.find_faces(img, pyramid="nope")


if __name__ == "__main__":
    unittest.main()
