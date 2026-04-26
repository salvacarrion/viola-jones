import os
import pickle
import time
import numpy as np
from utils import *

from adaboost import AdaBoost


class ViolaJones:

    def __init__(self, layers, features_path=None, layer_recall=0.99):
        assert isinstance(layers, list)
        self.layers = layers  # list with the number T of weak classifiers
        self.clfs = []
        self.base_width, self.base_height = 19, 19  # Size of the images from training dataset
        self.base_scale, self.shift = 1.25, 2
        self.features_path = features_path  # Path to save the features
        # Per-stage face-recall target used to calibrate each AdaBoost layer
        # after training. 0.99 ≈ paper-style; cumulative recall ≈ layer_recall^N.
        self.layer_recall = layer_recall

    def train(self, X, y, neg_pool=None, target_neg_per_stage=3000,
              neg_sample_budget=100000, val_split=0.15, seed=42):
        """
        Train a Viola-Jones cascade.

        If `neg_pool` (np.ndarray of shape (N, h, w) of face-free patches) is
        given, the cascade does paper-style hard-negative mining (§3.1):
        between stages it samples patches from the pool, runs them through
        the partial cascade, and keeps the false positives as the next
        stage's training negatives. Otherwise it falls back to the in-set
        FP loop over the CBCL non-faces in `y`.

        Per-window variance normalization (§5.1) is always on:
          - Per-sample pixel stds are computed for positives and negatives.
          - Feature values are divided by these stds before AdaBoost sees
            them, so weak-classifier thresholds are learned in
            std-normalized units.
          - At inference WeakClassifier.classify multiplies the threshold
            back by the window's std (see weakclassifier.py).

        Calibration is done on a held-out fraction of the positives
        (`val_split`) — fitting the threshold on the same positives the
        weak classifiers were optimized on overstates recall.
        """
        rng = np.random.default_rng(seed)
        pos_indices_all = np.where(y == 1)[0]
        cbcl_neg_indices = np.where(y == 0)[0]
        img_h, img_w = X[0].shape

        print("Summary input data:")
        print("\t- Positives: {:,}".format(len(pos_indices_all)))
        print("\t- CBCL negatives in y: {:,}".format(len(cbcl_neg_indices)))
        if neg_pool is not None:
            print("\t- Bootstrap pool: {:,}".format(len(neg_pool)))
        print("\t- Size (WxH): {}x{}".format(img_w, img_h))

        # ---- Held-out positives for per-stage calibration ----
        n_val = max(1, int(val_split * len(pos_indices_all)))
        shuffled = rng.permutation(pos_indices_all)
        val_pos_idxs = shuffled[:n_val]
        train_pos_idxs = shuffled[n_val:]

        X_pos = X[train_pos_idxs]
        X_val_pos = X[val_pos_idxs]
        print("Train positives: {:,}  |  Val positives (calibration): {:,}".format(
            len(X_pos), len(X_val_pos)))

        # ---- Positive-side precomputation (done once) ----
        print("Computing integral images for positives...")
        X_pos_ii = np.array(list(map(integral_image, X_pos)), dtype=np.uint32)
        X_val_pos_ii = np.array(list(map(integral_image, X_val_pos)), dtype=np.uint32)
        X_pos_std = self._sample_stds(X_pos)
        X_val_pos_std = self._sample_stds(X_val_pos)

        print("Building features...")
        start_time = time.time()
        features = build_features(img_w, img_h)
        print("\t- Num. features: {:,}".format(len(features)))
        print("\t- Total time: " + get_pretty_time(start_time))

        X_f_pos = self._apply_and_normalize(
            X_pos_ii, X_pos_std, features, cache_name="xf_pos")

        # ---- Stage 1 negatives: random sample from pool, or CBCL fallback ----
        if neg_pool is not None:
            n_take = min(target_neg_per_stage, len(neg_pool))
            idxs = rng.choice(len(neg_pool), size=n_take, replace=False)
            current_negs_X = neg_pool[idxs]
        else:
            current_negs_X = X[cbcl_neg_indices]

        # ---- Cascade training ----
        for stage_idx, T in enumerate(self.layers):
            print("\n[CascadeClassifier] Stage {}/{} (T={})".format(
                stage_idx + 1, len(self.layers), T))
            if len(current_negs_X) == 0:
                print("Cascade rejects all available negatives. Stopping early.")
                break

            print("Stage negatives: {:,}".format(len(current_negs_X)))
            stage_start = time.time()

            current_negs_ii = np.array(
                list(map(integral_image, current_negs_X)), dtype=np.uint32)
            current_negs_std = self._sample_stds(current_negs_X)

            print("Applying features to stage negatives...")
            X_f_neg = apply_features(current_negs_ii, features).astype(np.float32)
            X_f_neg /= current_negs_std[np.newaxis, :]

            X_f_stage = np.concatenate([X_f_pos, X_f_neg], axis=1)
            y_stage = np.concatenate([
                np.ones(X_f_pos.shape[1], dtype=np.float32),
                np.zeros(X_f_neg.shape[1], dtype=np.float32),
            ])
            ii_stage = np.concatenate([X_pos_ii, current_negs_ii], axis=0)

            perm = rng.permutation(len(y_stage))
            X_f_stage = X_f_stage[:, perm]
            y_stage = y_stage[perm]
            ii_stage = ii_stage[perm]

            clf = AdaBoost(n_estimators=T)
            clf.train(X_f_stage, y_stage, features, ii_stage)

            clf.calibrate(X_val_pos_ii, X_val_pos_std, target_recall=self.layer_recall)
            print("\t- layer threshold (after calibrate): {:.4f}".format(clf.threshold))
            self.clfs.append(clf)

            print("\t- stage time: " + get_pretty_time(stage_start))

            # ---- Mine negatives for the next stage ----
            if stage_idx + 1 < len(self.layers):
                if neg_pool is not None:
                    current_negs_X = self._mine_hard_negatives(
                        neg_pool, target_neg_per_stage, neg_sample_budget, rng)
                else:
                    kept = [neg for neg in current_negs_X if self.classify(neg) == 1]
                    current_negs_X = (np.array(kept) if kept
                                      else np.empty((0, img_h, img_w), dtype=X.dtype))
                print("\t- negatives carried to next layer: {}".format(len(current_negs_X)))

    @staticmethod
    def _sample_stds(samples):
        """Per-sample pixel std, floored at 1.0 to dodge division by zero."""
        stds = samples.reshape(len(samples), -1).std(axis=1)
        return np.maximum(stds, 1.0).astype(np.float32)

    def _apply_and_normalize(self, X_ii, X_std, features, cache_name=None):
        """apply_features + per-sample variance normalization, with optional disk cache."""
        cache_path = (self.features_path + cache_name + ".npy"
                      if (cache_name and self.features_path) else None)
        if cache_path and os.path.exists(cache_path):
            print("Loading cached normalized features: {}".format(cache_path))
            return np.load(cache_path)
        print("Applying features to {:,} samples...".format(len(X_ii)))
        start_time = time.time()
        X_f = apply_features(X_ii, features).astype(np.float32)
        X_f /= X_std[np.newaxis, :]
        print("\t- Total time: " + get_pretty_time(start_time))
        if cache_path:
            np.save(cache_path, X_f)
            print("Cached -> {}".format(cache_path))
        return X_f

    def _mine_hard_negatives(self, neg_pool, target, budget, rng):
        """Sample patches from `neg_pool`, keep those misclassified as faces."""
        found = []
        sampled = 0
        pool_size = len(neg_pool)
        print("Mining hard negatives (target={}, budget={})...".format(target, budget))
        start_time = time.time()
        while len(found) < target and sampled < budget:
            batch_size = min(2000, budget - sampled, pool_size)
            idxs = rng.choice(pool_size, size=batch_size, replace=False)
            for idx in idxs:
                patch = neg_pool[idx]
                if self.classify(patch) == 1:
                    found.append(patch)
                    if len(found) >= target:
                        break
            sampled += batch_size
        print("\t- mined {} from {} patches ({:.2f}% FPR) in {}".format(
            len(found), sampled, 100.0 * len(found) / max(sampled, 1),
            get_pretty_time(start_time)))
        if not found:
            return np.empty((0, *neg_pool.shape[1:]), dtype=neg_pool.dtype)
        return np.array(found)

    def classify(self, image, scale=1.0):
        """If a no-face is found, reject now. Else, keep looking."""
        ii = integral_image(image)
        std = max(float(np.std(image)), 1.0)
        return self.classify_ii(ii, scale=scale, std=std)

    def classify_ii(self, ii, scale=1.0, std=1.0):
        for clf in self.clfs:
            if clf.classify(ii, scale=scale, std=std) == 0:
                return 0
        return 1

    def find_faces(self, pil_image):
        """Receives a PIL image."""
        w, h = self.base_width, self.base_height
        growth = self.base_scale
        regions = []

        pil_image = pil_image.convert('L')
        image = np.array(pil_image)
        img_h, img_w = image.shape

        # Sliding window. Per-window integral image (slicing a precomputed
        # full-image ii is not a valid integral image of the crop) plus
        # per-window std for variance normalization (paper §5.1).
        counter = 0
        scale = 1.0
        while int(w * scale) <= img_w and int(h * scale) <= img_h:
            win_w = int(w * scale)
            win_h = int(h * scale)

            for y1 in np.arange(0, img_h - win_h + 1, self.shift):
                for x1 in np.arange(0, img_w - win_w + 1, self.shift):
                    y1, x1 = int(y1), int(x1)
                    y2, x2 = y1 + win_h, x1 + win_w
                    window = image[y1:y2, x1:x2]
                    window_ii = integral_image(window)
                    window_std = max(float(np.std(window)), 1.0)

                    if self.classify_ii(window_ii, scale=scale, std=window_std):
                        regions.append((x1, y1, x2, y2))
                    counter += 1

            scale *= growth

        return regions

    def save(self, filename):
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
