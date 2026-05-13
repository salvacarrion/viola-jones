import os
import pickle
import time
import numpy as np
from utils import *
from tqdm.auto import tqdm

from adaboost import AdaBoost


class ViolaJones:

    def __init__(self, features_path=None, layer_recall=0.99,
                 base_size=19, target_stage_fpr=None,
                 max_stages=30, max_wcs_per_stage=200,
                 min_wcs_per_stage=1,
                 min_cascade_recall=0.80):
        self.clfs = []
        # Native training-window size; sliding-window inference starts here
        # and grows by `base_scale`. Overwritten in `train()` based on the
        # actual training data shape so the saved checkpoint is self-describing.
        self.base_width = self.base_height = base_size
        self.base_scale, self.shift = 1.25, 2
        self.features_path = features_path
        # Per-stage face-recall target used to calibrate each AdaBoost layer
        # after training. 0.99 ≈ paper-style; cumulative recall ≈ layer_recall^N.
        self.layer_recall = layer_recall
        # Per-stage FPR target for adaptive training (paper §3). Each stage
        # stops adding weak classifiers once training-negative FPR drops here.
        # None = fixed-T mode (use exactly max_wcs_per_stage rounds per stage).
        self.target_stage_fpr = target_stage_fpr
        # Adaptive cascade depth: keep adding stages until cumulative val recall
        # drops below min_cascade_recall, negatives are exhausted, or max_stages
        # is reached. Number of stages emerges from training, not pre-specified.
        self.max_stages = max_stages
        self.max_wcs_per_stage = max_wcs_per_stage
        # Floor for adaptive early-stop. With < ~10 weak classifiers the
        # ensemble score is too coarse for `calibrate()` to find a useful
        # threshold, so we force every stage to add at least this many
        # rounds even if --target-stage-fpr would have stopped earlier.
        self.min_wcs_per_stage = min_wcs_per_stage
        self.min_cascade_recall = min_cascade_recall

    def train(self, train_pos, val_pos, neg_pool, seed_neg_pool=None,
              val_cal_pos=None,
              target_neg_per_stage=3000, neg_sample_budget=100000, seed=42,
              checkpoint_path=None):
        """
        Train a Viola-Jones cascade.

        Args:
            train_pos: (N, h, w) uint8 array of face crops used for AdaBoost.
            val_pos:   (M, h, w) uint8 array of held-out faces used to
                       calibrate each stage's threshold to `layer_recall`.
            neg_pool:  (P, h, w) uint8 array of face-free patches at the
                       same resolution. Used as the stage-1 seed (when
                       `seed_neg_pool` is None) and as the pool for
                       paper-style hard-negative mining (§3.1): between
                       stages, partial-cascade false positives are kept
                       as the next stage's training negatives.
            seed_neg_pool: optional (Q, h, w) uint8 array of "near-domain"
                       non-faces used ONLY for the stage-1 random sample.
                       When training for a benchmark whose non-faces look
                       face-like (e.g. CBCL), seeding stage 1 with matched
                       non-faces fixes specificity drops that pure-Caltech
                       seeding can't reach. From stage 2 on, hard-neg
                       mining still draws from `neg_pool`.

        Per-window variance normalization (§5.1) is always on:
          - Per-sample pixel stds are computed for positives and negatives.
          - Feature values are divided by these stds before AdaBoost sees
            them, so weak-classifier thresholds are learned in
            std-normalized units.
          - At inference WeakClassifier.classify multiplies the threshold
            back by the window's std (see weakclassifier.py).

        Calibration uses `val_pos` — fitting the threshold on the same
        positives the weak classifiers were optimized on overstates recall.
        """
        rng = np.random.default_rng(seed)
        if len(train_pos) == 0 or len(val_pos) == 0:
            raise ValueError("train_pos and val_pos must be non-empty.")
        if len(neg_pool) == 0:
            raise ValueError("neg_pool must be non-empty.")
        img_h, img_w = train_pos[0].shape
        self.base_width, self.base_height = img_w, img_h

        print("Summary input data:")
        print("\t- Train positives: {:,}".format(len(train_pos)))
        print("\t- Val   positives: {:,}".format(len(val_pos)))
        print("\t- Negative pool:   {:,}".format(len(neg_pool)))
        print("\t- Size (WxH): {}x{}".format(img_w, img_h))

        # ---- Positive-side precomputation (done once) ----
        print("Computing integral images for positives...")
        X_pos_ii = np.array(list(map(integral_image, train_pos)), dtype=np.uint32)
        X_val_pos_ii = np.array(list(map(integral_image, val_pos)), dtype=np.uint32)
        X_pos_std = self._sample_stds(train_pos)
        X_val_pos_std = self._sample_stds(val_pos)

        # Calibration positives: use CBCL-only val when training multi-source
        # so CelebA outliers don't drag per-stage thresholds toward zero.
        if val_cal_pos is not None and len(val_cal_pos) > 0:
            print("Computing integral images for calibration val (CBCL-only)...")
            print("\t- Calibration val: {:,} (CBCL-only, instead of mixed {:,})".format(
                len(val_cal_pos), len(val_pos)))
            X_cal_pos_ii  = np.array(list(map(integral_image, val_cal_pos)), dtype=np.uint32)
            X_cal_pos_std = self._sample_stds(val_cal_pos)
            cal_cache_name = "xf_val_cal"
        else:
            X_cal_pos_ii  = X_val_pos_ii
            X_cal_pos_std = X_val_pos_std
            cal_cache_name = "xf_val_pos"

        print("Building features...")
        start_time = time.time()
        features = build_features(img_w, img_h)
        print("\t- Num. features: {:,}".format(len(features)))
        print("\t- Total time: " + get_pretty_time(start_time))

        X_f_pos = self._apply_and_normalize(
            X_pos_ii, X_pos_std, features, cache_name="xf_pos")
        # Calibration-set features: needed by AdaBoost.train so it can
        # recalibrate the layer threshold every round and check FPR at the
        # actual operating point (paper §3 — d ≥ target_recall AND f ≤ target
        # measured at the same threshold). Same cache mechanism as X_f_pos.
        X_f_val_cal = self._apply_and_normalize(
            X_cal_pos_ii, X_cal_pos_std, features, cache_name=cal_cache_name)

        # ---- Determine starting stage (fresh vs. resume) ----
        # If self.clfs already has stages (loaded from a checkpoint), re-mine
        # hard negatives for the next unfinished stage and skip the stages that
        # are already trained. Otherwise, seed stage 1 from the matched-domain
        # pool as usual.
        start_stage = len(self.clfs)
        if start_stage == 0:
            seed_src = seed_neg_pool if (seed_neg_pool is not None and
                                         len(seed_neg_pool) > 0) else neg_pool
            seed_label = "seed_neg_pool" if seed_src is seed_neg_pool else "neg_pool"
            n_take = min(target_neg_per_stage, len(seed_src))
            idxs = rng.choice(len(seed_src), size=n_take, replace=False)
            current_negs_X = seed_src[idxs]
            print("Stage 1 seed negatives: {:,} sampled from {}".format(n_take, seed_label))
        else:
            print("Resuming training: {:,} stages already loaded — re-mining "
                  "hard negatives for stage {:,}...".format(start_stage, start_stage + 1))
            current_negs_X = self._mine_hard_negatives(
                neg_pool, target_neg_per_stage, neg_sample_budget, rng,
                seed_neg_pool=seed_neg_pool)
            print("\t- {:,} hard negatives ready for stage {:,}".format(
                len(current_negs_X), start_stage + 1))

        # ---- Cascade training ----
        max_stages = getattr(self, 'max_stages', 30)
        max_wcs    = getattr(self, 'max_wcs_per_stage', 200)
        min_recall = getattr(self, 'min_cascade_recall', 0.80)
        for stage_idx in range(start_stage, max_stages):
            print("\n[CascadeClassifier] Stage {}/≤{} (max_wcs={})".format(
                stage_idx + 1, max_stages, max_wcs))
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
            # [OPT] Free X_f_neg immediately — its data is already copied
            # into X_f_stage by np.concatenate; keeping both wastes ~2.4 GB.
            del X_f_neg

            # [OPT] Permutation commented out to avoid a full copy of
            # X_f_stage (~10 GB) via fancy indexing. AdaBoost._best_stump
            # does argsort per feature over ALL samples, so column order
            # is irrelevant to the result.
            # perm = rng.permutation(len(y_stage))
            # X_f_stage = X_f_stage[:, perm]
            # y_stage = y_stage[perm]

            clf = AdaBoost(n_estimators=max_wcs,
                           min_estimators=getattr(self, 'min_wcs_per_stage', 1))
            clf.train(X_f_stage, y_stage, features,
                      target_stage_fpr=getattr(self, 'target_stage_fpr', None),
                      X_val=X_f_val_cal,
                      target_recall=self.layer_recall)

            # Post-train calibration is a no-op when AdaBoost.train already
            # set the threshold via the same val set + target_recall; we keep
            # the call for the fixed-T (no target_stage_fpr) code path, where
            # the per-round calibration is skipped.
            clf.calibrate(X_cal_pos_ii, X_cal_pos_std, target_recall=self.layer_recall)
            print("\t- layer threshold (after calibrate): {:.4f}".format(clf.threshold))
            self.clfs.append(clf)

            print("\t- stage time: " + get_pretty_time(stage_start))

            # Per-stage checkpoint: lets long training runs survive a crash /
            # laptop sleep / Ctrl-C. The saved file IS a usable cascade with
            # however many stages have completed — `main.py test` and
            # `detect` work against partial cascades, just at the
            # in-progress operating point.
            if checkpoint_path is not None:
                self.save(checkpoint_path)
                print("\t- checkpoint -> {}.pkl  (stages {}/≤{})".format(
                    checkpoint_path, stage_idx + 1, max_stages))

            # ---- Global stopping: cumulative recall on val_pos ----
            # Count how many val faces pass ALL trained stages so far. If it
            # drops below min_cascade_recall, adding more stages would only
            # hurt recall further without proportional FPR gain.
            n_pass = sum(1 for ii, s in zip(X_val_pos_ii, X_val_pos_std)
                         if self.classify_ii(ii, std=float(s)) == 1)
            cum_recall = n_pass / len(val_pos)
            print("\t- cumulative val recall: {:.4f}  (stop if < {})".format(
                cum_recall, min_recall))
            if cum_recall < min_recall:
                print("\t! Recall {:.4f} < {}: stopping cascade.".format(
                    cum_recall, min_recall))
                break

            # ---- Mine negatives for the next stage ----
            current_negs_X = self._mine_hard_negatives(
                neg_pool, target_neg_per_stage, neg_sample_budget, rng,
                seed_neg_pool=seed_neg_pool)
            print("\t- negatives carried to next stage: {}".format(len(current_negs_X)))

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
            # [OPT] Memory-mapped load: X_f_pos is never modified during
            # training (only read for concat + calibration), so we let the
            # OS page it in/out from the .npy on demand instead of loading
            # ~7.7 GB into RAM. Numerically identical to np.load(cache_path).
            print("Loading cached normalized features (memmap): {}".format(cache_path))
            # return np.load(cache_path)  # original: full load into RAM
            return np.load(cache_path, mmap_mode='r')
        print("Applying features to {:,} samples...".format(len(X_ii)))
        start_time = time.time()
        X_f = apply_features(X_ii, features).astype(np.float32)
        X_f /= X_std[np.newaxis, :]
        print("\t- Total time: " + get_pretty_time(start_time))
        if cache_path:
            np.save(cache_path, X_f)
            print("Cached -> {}".format(cache_path))
        return X_f

    def _mine_from_pool(self, pool, target, budget, rng, label=""):
        """Sample patches from one pool, keep those misclassified as faces.

        Returns a list (not array) so the caller can concatenate across pools.
        """
        found = []
        sampled = 0
        pool_size = len(pool)
        if pool_size == 0 or target <= 0 or budget <= 0:
            return found
        prefix = f"[{label}] " if label else ""
        start_time = time.time()
        pbar = tqdm(total=target, desc=f"Mining{' '+label if label else ''}",
                    unit='neg', leave=False)
        while len(found) < target and sampled < budget:
            batch_size = min(2000, budget - sampled, pool_size)
            idxs = rng.choice(pool_size, size=batch_size, replace=False)
            for idx in idxs:
                patch = pool[idx]
                if self.classify(patch) == 1:
                    found.append(patch)
                    pbar.update(1)
                    if len(found) >= target:
                        break
            sampled += batch_size
            pbar.set_postfix(sampled='{:,}'.format(sampled),
                           fpr='{:.2f}%'.format(100.0 * len(found) / max(sampled, 1)))
        pbar.close()
        print("\t- {}mined {} from {} ({:.2f}% FPR) in {}".format(
            prefix, len(found), sampled,
            100.0 * len(found) / max(sampled, 1), get_pretty_time(start_time)))
        return found

    def _mine_hard_negatives(self, neg_pool, target, budget, rng,
                             seed_neg_pool=None):
        """Mine hard negatives, optionally stratified across two pools.

        When `seed_neg_pool` is given (e.g. CBCL non-faces), allocate half
        the target/budget to it and the other half to `neg_pool` (Caltech).
        This keeps matched-domain hard negatives in the training mix at
        EVERY stage, not just stage 1 — fixing the "stage 9 trained with
        only Caltech-flavored negatives" symptom we observed.
        """
        print("Mining hard negatives (target={}, budget={})...".format(target, budget))
        found = []
        if seed_neg_pool is not None and len(seed_neg_pool) > 0:
            t_seed = target // 2
            t_main = target - t_seed
            # Cap seed budget at "scan whole pool 2×" — past that, we're
            # just resampling the same patches and mining returns nothing
            # new. Any unspent budget is reallocated to the main pool below.
            b_seed_requested = budget // 2
            b_seed = min(b_seed_requested, len(seed_neg_pool) * 2)
            b_main = budget - b_seed
            found += self._mine_from_pool(seed_neg_pool, t_seed, b_seed, rng,
                                          label="seed")
            # If the seed pool didn't yield its share (small pool gets
            # exhausted in deep stages), backfill from the main pool.
            shortfall = t_seed - len(found)
            if shortfall > 0:
                t_main += shortfall
            found += self._mine_from_pool(neg_pool, t_main, b_main, rng,
                                          label="caltech")
        else:
            found += self._mine_from_pool(neg_pool, target, budget, rng)
        if not found:
            return np.empty((0, *neg_pool.shape[1:]), dtype=neg_pool.dtype)
        return np.array(found)

    def classify(self, image, scale=1.0):
        """If a no-face is found, reject now. Else, keep looking."""
        ii = integral_image(image)
        std = max(float(np.std(image)), 1.0)
        return self.classify_ii(ii, scale=scale, std=std)

    def classify_ii(self, ii, scale=1.0, std=1.0, ox=0, oy=0):
        for clf in self.clfs:
            if clf.classify(ii, scale=scale, std=std, ox=ox, oy=oy) == 0:
                return 0
        return 1

    def find_faces(self, pil_image, growth=None, min_shift=None):
        """
        Multi-scale sliding-window detection on a PIL image.

        Strategy (paper §5):
          - One padded integral image + one squared II for the WHOLE image;
            every per-window feature lookup and per-window std becomes O(1).
            Querying via (ox, oy) offsets means we never re-cumsum a cropped
            window (the old per-window II was correct but ~100× slower).
          - Window shift grows with scale: at scale s we step `max(1, s)`
            pixels. Stepping 2px at scale 8 just produces near-duplicate
            detections that NMS later collapses anyway.

        Returns:
            list of (x1, y1, x2, y2, score) tuples in image coordinates.
            `score` is the cascade depth (number of stages passed) — useful
            as a confidence signal for `non_maximum_supression`.
        """
        w, h = self.base_width, self.base_height
        if growth is None:
            growth = self.base_scale
        if min_shift is None:
            min_shift = self.shift

        pil_image = pil_image.convert('L')
        image = np.array(pil_image)
        img_h, img_w = image.shape
        if img_h < h or img_w < w:
            return []

        # Single-shot integral images for the entire image
        ii = integral_image(image)               # (img_h+1, img_w+1)
        ii2 = integral_image_pow2(image)

        # Cheap up-front pass to size the progress bar correctly
        total_windows = 0
        s = 1.0
        while int(w * s) <= img_w and int(h * s) <= img_h:
            wh, ww = int(h * s), int(w * s)
            shift = max(min_shift, int(s))
            total_windows += (1 + (img_h - wh) // shift) * (1 + (img_w - ww) // shift)
            s *= growth

        regions = []
        scale = 1.0
        pbar = tqdm(total=total_windows, desc='Sliding window',
                    unit='win', leave=False)
        while int(w * scale) <= img_w and int(h * scale) <= img_h:
            win_w = int(w * scale)
            win_h = int(h * scale)
            area = float(win_w * win_h)
            shift = max(min_shift, int(scale))

            for y1 in range(0, img_h - win_h + 1, shift):
                y2 = y1 + win_h
                for x1 in range(0, img_w - win_w + 1, shift):
                    x2 = x1 + win_w
                    # Per-window mean and std via the full-image IIs
                    sum_x = (int(ii[y2, x2]) - int(ii[y1, x2])
                             - int(ii[y2, x1]) + int(ii[y1, x1]))
                    sum_x2 = (int(ii2[y2, x2]) - int(ii2[y1, x2])
                              - int(ii2[y2, x1]) + int(ii2[y1, x1]))
                    mean = sum_x / area
                    var = sum_x2 / area - mean * mean
                    std = max(var, 0.0) ** 0.5
                    std = std if std >= 1.0 else 1.0

                    score = self._cascade_score(
                        ii, scale=scale, std=std, ox=x1, oy=y1)
                    if score > 0:
                        regions.append((x1, y1, x2, y2, score))
                    pbar.update(1)

            pbar.set_postfix(scale='{:.2f}'.format(scale),
                             detections=len(regions))
            scale *= growth
        pbar.close()
        return regions

    def _cascade_score(self, ii, scale=1.0, std=1.0, ox=0, oy=0):
        """
        Run the cascade. Return:
          0 if any stage rejects (early exit, paper §4),
          else the number of stages passed (= cascade depth) — used as a
          confidence proxy for downstream NMS so deeper-passing windows
          beat shallower neighbors.
        """
        for i, clf in enumerate(self.clfs):
            if clf.classify(ii, scale=scale, std=std, ox=ox, oy=oy) == 0:
                return 0
        return len(self.clfs)

    def save(self, filename):
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
