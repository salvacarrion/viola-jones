"""
Native (pure-NumPy) evaluator for an OpenCV Haar cascade.

This makes an OpenCV pretrained frontal-face cascade run inside OUR pipeline:
it duck-types the ViolaJones interface (`classify`, `find_faces`,
`base_width/base_height/base_scale/shift`, `save`) and is pickled to
weights/, so `main.py detect`, `main.py test` and `tools/eval_fddb.py` all
work against it with NO cv2 dependency at inference. It *reproduces* OpenCV's
cascade (their trained weights), it does not retrain anything.

Only axis-aligned (non-tilted) stump cascades are supported — which covers
haarcascade_frontalface_{default,alt,alt2}. The XML→object conversion lives
in tools/convert_opencv_cascade.py.

Evaluation follows OpenCV's HaarEvaluator exactly, in the "scale the image"
regime: every pyramid level downsizes the image and runs the cascade at the
base 24×24 window, so each window reduces to the unambiguous scale-1 math:

  per window:
    nf = sqrt(winArea * Σx² − (Σx)²);  nf = max(nf, 1)      # winArea·std
    for each stage:
      stage_sum = Σ_stumps  (Σ_rects w·rectsum < thr·nf ? leaf_left : leaf_right)
      reject window if stage_sum < stage_threshold
    accept if it clears every stage

`Σx`/`Σx²` are taken over the whole 24×24 window from the padded integral
images. The per-stump threshold compares the weighted rectangle sum against
`thr·nf` (the variance normalization of Viola-Jones §5.1, OpenCV's constants).
"""

import pickle

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from utils import integral_image, integral_image_pow2


class OpenCVCascade:
    def __init__(self, stages, base_width, base_height,
                 base_scale=1.25, shift=2, source="opencv"):
        """`stages`: list of (stage_threshold, stumps) where each stump is
        (rects, weights, threshold, left, right):
          rects   : (nrect, 4) int   [x, y, w, h] in base-window coords
          weights : (nrect,)   float rectangle weights
          threshold, left, right : floats (node threshold + leaf values)
        """
        self.base_width = base_width
        self.base_height = base_height
        self.base_scale = base_scale
        self.shift = shift
        self.source = source
        # keep raw stages (picklable, simple) ...
        self.stages = stages
        # ... and a flattened, vectorization-friendly view per stage.
        self._compile()

    def _compile(self):
        """Pack each stage into flat arrays for batched evaluation."""
        self._cstages = []
        for stage_threshold, stumps in self.stages:
            coords, weights, bounds = [], [], []
            thr, left, right = [], [], []
            cur = 0
            for rects, w, t, l, r in stumps:
                bounds.append(cur)
                for (rx, ry, rw, rh), wt in zip(rects, w):
                    coords.append((rx, ry, rx + rw, ry + rh))  # exclusive br
                    weights.append(wt)
                    cur += 1
                thr.append(t); left.append(l); right.append(r)
            self._cstages.append({
                "stage_threshold": float(stage_threshold),
                "coords": np.asarray(coords, dtype=np.int64),
                "weights": np.asarray(weights, dtype=np.float64),
                "bounds": np.asarray(bounds, dtype=np.int64),
                "thr": np.asarray(thr, dtype=np.float64),
                "left": np.asarray(left, dtype=np.float64),
                "right": np.asarray(right, dtype=np.float64),
            })

    # ---- core batched evaluation at the base window over a grid ----------
    def _eval_grid(self, ii, ii2, ox, oy):
        """Run the full cascade at the BASE window for every origin (ox,oy).

        ii/ii2 are padded integral images (int64) of the (already scaled)
        image. Returns (alive_index, score) — score is the accumulated
        per-stage margin (stage_sum − stage_threshold) over passed stages.
        """
        w, h = self.base_width, self.base_height
        area = float(w * h)
        # variance norm factor per window: nf = sqrt(area*Σx² − (Σx)²), ≥1
        x2 = ox + w
        y2 = oy + h
        sx = (ii[oy + h, ox + w] - ii[oy, ox + w]
              - ii[oy + h, ox] + ii[oy, ox]).astype(np.float64)
        sx2 = (ii2[oy + h, ox + w] - ii2[oy, ox + w]
               - ii2[oy + h, ox] + ii2[oy, ox]).astype(np.float64)
        nf = area * sx2 - sx * sx
        nf = np.sqrt(np.maximum(nf, 0.0))
        nf = np.where(nf >= 1.0, nf, 1.0)

        alive = np.arange(ox.size, dtype=np.int64)
        score = np.zeros(ox.size, dtype=np.float64)
        for cs in self._cstages:
            if alive.size == 0:
                break
            aox, aoy, anf = ox[alive], oy[alive], nf[alive]
            coords = cs["coords"]
            weights = cs["weights"]
            # rectangle sums for every rect of every stump, batched over alive
            cx1 = coords[:, 0]; cy1 = coords[:, 1]
            cx2 = coords[:, 2]; cy2 = coords[:, 3]
            # shape (n_alive, n_rects)
            A = ii[aoy[:, None] + cy2[None, :], aox[:, None] + cx2[None, :]]
            B = ii[aoy[:, None] + cy1[None, :], aox[:, None] + cx2[None, :]]
            C = ii[aoy[:, None] + cy2[None, :], aox[:, None] + cx1[None, :]]
            D = ii[aoy[:, None] + cy1[None, :], aox[:, None] + cx1[None, :]]
            rect_sums = (A - B - C + D).astype(np.float64) * weights[None, :]
            # sum rects per stump → (n_alive, n_stumps)
            fv = np.add.reduceat(rect_sums, cs["bounds"], axis=1)
            # node decision: fv < thr*nf ? left : right
            t = cs["thr"][None, :] * anf[:, None]
            leaf = np.where(fv < t, cs["left"][None, :], cs["right"][None, :])
            stage_sum = leaf.sum(axis=1)
            passed = stage_sum >= cs["stage_threshold"]
            score[alive[passed]] += stage_sum[passed] - cs["stage_threshold"]
            alive = alive[passed]
        return alive, score

    # ---- ViolaJones-compatible API --------------------------------------
    def classify(self, image):
        """Classify a single base-sized (e.g. 24×24) patch → 1/0."""
        arr = np.asarray(image)
        if arr.shape[0] != self.base_height or arr.shape[1] != self.base_width:
            pil = Image.fromarray(arr.astype(np.uint8))
            pil = pil.resize((self.base_width, self.base_height))
            arr = np.asarray(pil)
        ii = integral_image(arr).astype(np.int64)
        ii2 = integral_image_pow2(arr).astype(np.int64)
        ox = np.zeros(1, dtype=np.int64)
        oy = np.zeros(1, dtype=np.int64)
        alive, _ = self._eval_grid(ii, ii2, ox, oy)
        return 1 if alive.size > 0 else 0

    def find_faces(self, pil_image, growth=None, min_shift=None,
                   min_face_size=None, max_face_size=None, min_score=None):
        """Multi-scale detection. Returns (x1,y1,x2,y2,score) tuples in image
        pixels (caller applies NMS, exactly like ViolaJones.find_faces)."""
        w, h = self.base_width, self.base_height
        growth = self.base_scale if growth is None else growth
        step = self.shift if min_shift is None else min_shift

        gray = np.asarray(pil_image.convert("L"))
        img_h, img_w = gray.shape
        base = max(w, h)
        scale_min = max(1.0, (min_face_size / base) if min_face_size else 1.0)
        scale_max = (max_face_size / base) if max_face_size else None
        pil_gray = Image.fromarray(gray)

        regions = []
        scale = scale_min
        pbar = tqdm(desc="Pyramid", unit="scale", leave=False)
        while int(w * scale) <= img_w and int(h * scale) <= img_h:
            if scale_max is not None and scale > scale_max:
                break
            # "scale the image": shrink by `scale` so the base window matches.
            rw = max(w, int(round(img_w / scale)))
            rh = max(h, int(round(img_h / scale)))
            small = np.asarray(pil_gray.resize((rw, rh), Image.BILINEAR))
            ii = integral_image(small).astype(np.int64)
            ii2 = integral_image_pow2(small).astype(np.int64)

            ys = np.arange(0, rh - h + 1, step, dtype=np.int64)
            xs = np.arange(0, rw - w + 1, step, dtype=np.int64)
            if ys.size and xs.size:
                yy, xx = np.meshgrid(ys, xs, indexing="ij")
                ox = xx.ravel(); oy = yy.ravel()
                alive, score = self._eval_grid(ii, ii2, ox, oy)
                if alive.size:
                    sc = score[alive]
                    if min_score is not None:
                        keep = sc >= float(min_score)
                        alive = alive[keep]; sc = sc[keep]
                    sx = (img_w / rw); sy = (img_h / rh)
                    for i, a in enumerate(alive.tolist()):
                        x1 = ox[a] * sx; y1 = oy[a] * sy
                        regions.append((x1, y1, x1 + w * sx, y1 + h * sy,
                                        float(sc[i])))
            pbar.set_postfix(scale="{:.2f}".format(scale),
                             detections=len(regions))
            pbar.update(1)
            scale *= growth
        pbar.close()
        return regions

    def save(self, filename):
        path = filename if filename.endswith(".pkl") else filename + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
