import time

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from features import RectangleRegion, HaarFeature  # noqa: F401  (re-exported for tests)


def load_image(image_path, as_numpy=False):
    pil_img = Image.open(image_path)
    if as_numpy:
        return np.array(pil_img)
    return pil_img


def integral_image(img):
    """
    Padded summed-area table:  ii[r+1, c+1] = sum of img[:r+1, :c+1],
    with ii[0, :] = ii[:, 0] = 0.

    The zero pad lets `RectangleRegion.compute_region` reduce every rectangle
    sum to four unconditional reads — no `if x1>0` branches per lookup, which
    matters when we evaluate millions of feature lookups per image.
    """
    arr = np.asarray(img, dtype=np.uint32)
    h, w = arr.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.uint32)
    ii[1:, 1:] = arr.cumsum(axis=0).cumsum(axis=1)
    return ii


def integral_image_pow2(img):
    """Padded squared-image II — used for O(1) per-window pixel std."""
    arr = np.asarray(img, dtype=np.uint64) ** 2
    h, w = arr.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.uint64)
    ii[1:, 1:] = arr.cumsum(axis=0).cumsum(axis=1)
    return ii


def build_features(img_w, img_h, shift=1, min_w=4, min_h=4):
    """
    Generate values from Haar features.

    White rectangles subtract from black ones.
    """
    features = []  # [Tuple(positive regions, negative regions),...]

    # Scale feature window
    for w_width in range(min_w, img_w + 1):
        for w_height in range(min_h, img_h + 1):

            # Walk through all the image
            x = 0
            while x + w_width <= img_w:
                y = 0
                while y + w_height <= img_h:

                    # Possible Haar regions
                    immediate = RectangleRegion(x, y, w_width, w_height)
                    right = RectangleRegion(x + w_width, y, w_width, w_height)
                    right_2 = RectangleRegion(x + w_width * 2, y, w_width, w_height)
                    bottom = RectangleRegion(x, y + w_height, w_width, w_height)
                    bottom_2 = RectangleRegion(x, y + w_height * 2, w_width, w_height)
                    bottom_right = RectangleRegion(x + w_width, y + w_height, w_width, w_height)

                    # [Haar] 2 rectangles: horizontal (w-b)
                    if x + w_width * 2 <= img_w:
                        features.append(HaarFeature([immediate], [right]))
                    # [Haar] 2 rectangles: vertical (w-b)
                    if y + w_height * 2 <= img_h:
                        features.append(HaarFeature([bottom], [immediate]))

                    # [Haar] 3 rectangles: horizontal (w-b-w)
                    if x + w_width * 3 <= img_w:
                        features.append(HaarFeature([immediate, right_2], [right]))
                    # [Haar] 3 rectangles: vertical (w-b-w)
                    if y + w_height * 3 <= img_h:
                        features.append(HaarFeature([immediate, bottom_2], [bottom]))

                    # [Haar] 4 rectangles
                    if x + w_width * 2 <= img_w and y + w_height * 2 <= img_h:
                        features.append(HaarFeature([immediate, bottom_right], [bottom, right]))

                    y += shift
                x += shift
    return features


def features_to_arrays(features):
    """
    Pack a list of HaarFeature objects into flat arrays for vectorized
    evaluation. Rectangles are emitted in feature order so the per-feature
    sum becomes a single `np.add.reduceat`.

    Returns:
        coords: (n_rects, 4) int32 array of [x1, y1, x2, y2] in padded-II
            coordinates (exclusive bottom-right).
        signs:  (n_rects,) int8 — +1 for negative_regions (added),
                -1 for positive_regions (subtracted).
                Convention follows HaarFeature.compute_value =
                sum(neg) - sum(pos).
        boundaries: (n_features,) int32 — start index of each feature's
            rectangle slice within `coords`/`signs`. Empty features (no
            regions) are not allowed.
    """
    coords, signs, boundaries = [], [], []
    cursor = 0
    for f in features:
        boundaries.append(cursor)
        for r in f.positive_regions:
            coords.append((r.x, r.y, r.x + r.width, r.y + r.height))
            signs.append(-1)
            cursor += 1
        for r in f.negative_regions:
            coords.append((r.x, r.y, r.x + r.width, r.y + r.height))
            signs.append(1)
            cursor += 1
    return (np.asarray(coords, dtype=np.int32),
            np.asarray(signs, dtype=np.int8),
            np.asarray(boundaries, dtype=np.int32))


def apply_features(X_ii, features, chunk_size=200):
    """
    Vectorized batch evaluation of all Haar features on all integral images.

    Replaces a Python triple-loop (features × samples × rectangles) with
    numpy fancy indexing. For typical training batches at 24×24 (~50k
    features × ~13k samples) this is ~30-50× faster than the per-sample
    `compute_value` loop.

    Args:
        X_ii: (n_samples, H+1, W+1) padded integral images.
        features: list of HaarFeature objects.
        chunk_size: samples processed per batch — caps the (chunk_size ×
            n_rects) intermediate at a few hundred MB. Lower if you OOM.

    Returns:
        (n_features, n_samples) int32 feature-value matrix.
    """
    coords, signs, boundaries = features_to_arrays(features)
    n_features = len(features)
    n_samples = len(X_ii)
    x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]

    out = np.empty((n_features, n_samples), dtype=np.int32)
    pbar = tqdm(total=n_samples, desc='Applying features',
                unit='img', leave=False)
    for s0 in range(0, n_samples, chunk_size):
        s1 = min(s0 + chunk_size, n_samples)
        ii = X_ii[s0:s1]  # (b, H+1, W+1)
        # ii[:, y, x] with array (y, x) gives (b, n_rects)
        A = ii[:, y2, x2].astype(np.int64)
        B = ii[:, y1, x2].astype(np.int64)
        C = ii[:, y2, x1].astype(np.int64)
        D = ii[:, y1, x1].astype(np.int64)
        rect_sums = A - B - C + D                       # (b, n_rects)
        rect_vals = rect_sums * signs[np.newaxis, :]    # broadcast sign
        # Sum rectangles belonging to each feature → (b, n_features)
        feat_vals = np.add.reduceat(rect_vals, boundaries, axis=1)
        out[:, s0:s1] = feat_vals.T.astype(np.int32)
        pbar.update(s1 - s0)
    pbar.close()
    return out


def evaluate(clf, X, y):
    metrics = {}
    true_positive, true_negative = 0, 0
    false_positive, false_negative = 0, 0

    pbar = tqdm(range(len(y)), desc='Evaluating', unit='img')
    for i in pbar:
        prediction = clf.classify(X[i])
        if prediction == y[i]:
            if prediction == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if prediction == 1:
                false_positive += 1
            else:
                false_negative += 1

        # Live metrics every ~500 samples — cheap, helps spot crashes early
        if (i + 1) % 500 == 0:
            seen = true_positive + true_negative + false_positive + false_negative
            acc = (true_positive + true_negative) / max(seen, 1)
            tp_fn = true_positive + false_negative
            tp_fp = true_positive + false_positive
            rec = true_positive / tp_fn if tp_fn else 0.0
            prec = true_positive / tp_fp if tp_fp else 0.0
            pbar.set_postfix(acc='{:.3f}'.format(acc),
                             rec='{:.3f}'.format(rec),
                             prec='{:.3f}'.format(prec))

    metrics['true_positive'] = true_positive
    metrics['true_negative'] = true_negative
    metrics['false_positive'] = false_positive
    metrics['false_negative'] = false_negative

    total = true_positive + false_negative + true_negative + false_positive
    metrics['accuracy'] = (true_positive + true_negative) / total
    metrics['precision'] = true_positive / (true_positive + false_positive)
    metrics['recall'] = true_positive / (true_positive + false_negative)
    metrics['specifity'] = true_negative / (true_negative + false_positive)
    metrics['f1'] = (2.0 * metrics['precision'] * metrics['recall']) \
                    / (metrics['precision'] + metrics['recall'])
    return metrics


def get_pretty_time(start_time, end_time=None, s="", divisor=1.0):
    if not end_time:
        end_time = time.time()
    hours, rem = divmod((end_time - start_time) / divisor, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{}{:0>2}:{:0>2}:{:05.8f}".format(s, int(hours), int(minutes), seconds)


def draw_bounding_boxes(pil_image, regions, color="green", thickness=3):
    """Each region may be (x1,y1,x2,y2) or (x1,y1,x2,y2,score) — only the
    first 4 coords are drawn."""
    source_img = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(source_img)
    for rect in regions:
        draw.rectangle(tuple(rect[:4]), outline=color, width=thickness)
    return source_img


def non_maximum_supression(regions, threshold=0.5):
    """
    Greedy NMS. Accepts (x1,y1,x2,y2) or (x1,y1,x2,y2,score) tuples; if a
    score is present, suppression is score-ordered (highest kept first) so
    a deep-passing detection beats a shallow neighbour. Without a score we
    fall back to y2-ordering as before.

    Overlap metric is (intersection / smaller-box area) — the same form the
    original code used. Returns a (n, 4 or 5) ndarray of survivors, dtype
    int for coords / float for scores.
    """
    boxes = np.asarray(regions, dtype=np.float64)
    if len(boxes) == 0:
        return []
    has_score = boxes.shape[1] >= 5

    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # Highest score first (last in idxs). Fall back to y2 ordering when no
    # score is provided.
    idxs = np.argsort(boxes[:, 4]) if has_score else np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > threshold)[0])))

    out = boxes[pick]
    if has_score:
        # Coords as int, score as float — convert manually since astype('int')
        # would truncate the score column.
        return np.column_stack([out[:, :4].astype(int), out[:, 4]])
    return out.astype(int)
