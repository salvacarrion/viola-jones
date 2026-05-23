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
                 min_cascade_recall=0.80,
                 min_stage_negatives=0):
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
        self.min_stage_negatives = min_stage_negatives

    def train(self, train_pos, val_pos, neg_pool, neg_seed=None,
              very_hard_pool=None,
              target_neg_per_stage=3000, neg_sample_budget=100000, seed=42,
              checkpoint_path=None, pos_cache_suffix=""):
        """
        Train a Viola-Jones cascade.

        Args:
            train_pos: (N, h, w) uint8 array of face crops used for AdaBoost.
                       May be augmented (jitter, h-flip) — diversity helps
                       weak classifiers generalize.
            val_pos:   (M, h, w) uint8 array of held-out faces sampled from
                       the benchmark distribution (un-augmented, center crop).
                       Used for BOTH:
                         - per-stage threshold calibration (keep ≥ layer_recall
                           of these passing each stage),
                         - cumulative-recall stop criterion across stages.
                       Anchoring val to the benchmark eliminates the val→test
                       gap that augmented val sets introduce.
            neg_pool:  (P, h, w) uint8 array of face-free patches at the
                       same resolution. Used as the stage-1 seed (when
                       `neg_seed` is None) and as the pool for paper-style
                       hard-negative mining (§3.1): between stages,
                       partial-cascade false positives are kept as the next
                       stage's training negatives.
            neg_seed:  optional (Q, h, w) uint8 array of "near-domain"
                       non-faces used ONLY for the stage-1 random sample
                       (and 50/50 alongside `neg_pool` in deeper mining).
                       Benchmark-style non-faces (e.g. CBCL) here fix the
                       specificity drops that pure-Caltech seeding can't
                       reach when the test distribution differs.

        Per-window variance normalization (§5.1) is always on:
          - Per-sample pixel stds are computed for positives and negatives.
          - Feature values are divided by these stds before AdaBoost sees
            them, so weak-classifier thresholds are learned in
            std-normalized units.
          - At inference WeakClassifier.classify multiplies the threshold
            back by the window's std (see weakclassifier.py).
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
        # val_pos plays both roles (calibration anchor + stop criterion).
        # We keep ONE set of val features cached for use by both AdaBoost's
        # per-round threshold recalibration AND the cumulative-recall check
        # at the end of each stage. Sharing the same set is what closes the
        # "calibrate against X, check recall against Y" inconsistency that
        # used to stop the cascade prematurely.
        print("Computing integral images for positives...")
        X_pos_ii = np.array(list(map(integral_image, train_pos)), dtype=np.uint32)
        X_val_ii = np.array(list(map(integral_image, val_pos)), dtype=np.uint32)
        X_pos_std = self._sample_stds(train_pos)
        X_val_std = self._sample_stds(val_pos)

        print("Building features...")
        start_time = time.time()
        features = build_features(img_w, img_h)
        print("\t- Num. features: {:,}".format(len(features)))
        print("\t- Total time: " + get_pretty_time(start_time))

        X_f_pos = self._apply_and_normalize(
            X_pos_ii, X_pos_std, features,
            cache_name="xf_pos" + pos_cache_suffix)
        # Validation-set features: needed by AdaBoost.train so it can
        # recalibrate the layer threshold every round and check FPR at the
        # actual operating point (paper §3 — d ≥ target_recall AND f ≤ target
        # measured at the same threshold).
        X_f_val = self._apply_and_normalize(
            X_val_ii, X_val_std, features, cache_name="xf_val")

        # ---- Determine starting stage (fresh vs. resume) ----
        # If self.clfs already has stages (loaded from a checkpoint), re-mine
        # hard negatives for the next unfinished stage and skip the stages that
        # are already trained. Otherwise, seed stage 1 from the matched-domain
        # pool as usual.
        start_stage = len(self.clfs)
        if start_stage == 0:
            seed_src = neg_seed if (neg_seed is not None and
                                    len(neg_seed) > 0) else neg_pool
            seed_label = "neg_seed" if seed_src is neg_seed else "neg_pool"
            n_take = min(target_neg_per_stage, len(seed_src))
            idxs = rng.choice(len(seed_src), size=n_take, replace=False)
            current_negs_X = seed_src[idxs]
            print("Stage 1 seed negatives: {:,} sampled from {}".format(n_take, seed_label))
        else:
            print("Resuming training: {:,} stages already loaded — re-mining "
                  "hard negatives for stage {:,}...".format(start_stage, start_stage + 1))
            current_negs_X = self._mine_hard_negatives(
                neg_pool, target_neg_per_stage, neg_sample_budget, rng,
                neg_seed=neg_seed, very_hard_pool=very_hard_pool)
            print("\t- {:,} hard negatives ready for stage {:,}".format(
                len(current_negs_X), start_stage + 1))

        # ---- Cascade training ----
        max_stages = getattr(self, 'max_stages', 30)
        max_wcs    = getattr(self, 'max_wcs_per_stage', 200)
        min_recall = getattr(self, 'min_cascade_recall', 0.80)
        for stage_idx in range(start_stage, max_stages):
            print("\n[CascadeClassifier] Stage {}/≤{} (max_wcs={})".format(
                stage_idx + 1, max_stages, max_wcs))
            if len(current_negs_X) == 0 or len(current_negs_X) < getattr(self, 'min_stage_negatives', 0):
                print(f"Cascade found {len(current_negs_X)} negatives (requires >= {getattr(self, 'min_stage_negatives', 0)}). Stopping early.")
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
                      X_val=X_f_val,
                      target_recall=self.layer_recall)

            # Post-train calibration is a no-op when AdaBoost.train already
            # set the threshold via the same val set + target_recall; we keep
            # the call for the fixed-T (no target_stage_fpr) code path, where
            # the per-round calibration is skipped.
            clf.calibrate(X_val_ii, X_val_std, target_recall=self.layer_recall)
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

            # ---- Capacity-ceiling check: stage capped without hitting target ----
            # AdaBoost.train sets clf.final_fpr to the FPR at the operating
            # point at end of training. If `target_stage_fpr` was active and
            # final_fpr > target, the stage ran to max_wcs_per_stage without
            # being able to satisfy the per-stage criterion — the cascade has
            # hit its representational ceiling for the (resolution × jitter ×
            # neg-budget) combination. Continuing would just stack more
            # degenerate stages (each with FPR > target). Stop and surface
            # the issue, but KEEP the capped stage in the cascade — it still
            # adds some rejection (it just doesn't hit 50% on its own), and
            # the work is already done. The user can:
            #   1) resume with --target-stage-fpr > final_fpr to extend with
            #      a relaxed criterion (the capped stage already satisfies it),
            #   2) drop the capped stage via tools/truncate_checkpoint.py for
            #      a "pristine" cascade where every stage satisfies the target.
            tgt_fpr = getattr(self, 'target_stage_fpr', None)
            if (tgt_fpr is not None
                    and clf.final_fpr is not None
                    and clf.final_fpr > tgt_fpr):
                print("\t! Stage {} capped without satisfying target_stage_fpr: "
                      "final FPR {:.4f} > target {:.3f}.".format(
                          stage_idx + 1, clf.final_fpr, tgt_fpr))
                print("\t  Cascade at capacity ceiling — stopping.")
                print("\t  Stage kept in checkpoint (still rejects ~{:.0%} of "
                      "remaining hard-negs).".format(1.0 - clf.final_fpr))
                print("\t  To extend: --resume-from with --target-stage-fpr "
                      "> {:.3f}.".format(clf.final_fpr))
                print("\t  Or drop this stage with "
                      "tools/truncate_checkpoint.py for a pristine cascade.")
                break

            # ---- Global stopping: cumulative recall on val_pos ----
            # Measured on X_val_ii — the SAME set the per-stage threshold
            # calibrates against. Consistency matters: thresholds are tuned
            # to keep `layer_recall` of val_pos passing, so cumulative recall
            # here tracks what we actually retain at the test distribution
            # (val_pos is sampled from the benchmark, un-augmented).
            n_pass = sum(1 for ii, s in zip(X_val_ii, X_val_std)
                         if self.classify_ii(ii, std=float(s)) == 1)
            cum_recall = n_pass / len(X_val_ii)
            print("\t- cumulative val recall: {:.4f}  (stop if < {})".format(
                cum_recall, min_recall))
            if cum_recall < min_recall:
                print("\t! Recall {:.4f} < {}: stopping cascade.".format(
                    cum_recall, min_recall))
                break

            # ---- Mine negatives for the next stage ----
            current_negs_X = self._mine_hard_negatives(
                neg_pool, target_neg_per_stage, neg_sample_budget, rng,
                neg_seed=neg_seed, very_hard_pool=very_hard_pool)
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

    def _mine_from_pool(self, pool, target, budget, rng, label="",
                        chunk_size=50000):
        """Sample patches from one pool, keep those misclassified as faces.

        Returns a list (not array) so the caller can concatenate across pools.

        Memory pattern: visits the pool in `chunk_size`-sized contiguous
        slices, with chunk order randomly permuted. Within a chunk, traversal
        is sequential. This is the key to making mining work on a pool larger
        than RAM:

          - With random per-patch indexing (the previous behavior), each
            `pool[idx]` on a memmap'd .npy triggers a page fault for a
            ~4 KB OS page that holds ~11 contiguous patches — of which we
            use 1. With a 50M-patch pool (~18 GB) and 16 GB RAM, the page
            cache thrashes: throughput drops from ~30 K patches/s to
            ~8 K patches/s, and swap fills up.

          - Sequential reads inside a chunk let OS prefetch do its job
            (11 patches per page, all consumed). Random chunk order
            preserves unbiasedness against budget cutoffs — we don't bias
            toward the start of the file when the cascade is strong and
            we stop early.

        Since `classify(patch)` is deterministic, traversal order does NOT
        change which patches end up in `found` — only the wall-clock cost
        of reaching them. Result is statistically equivalent to the prior
        random-index implementation, just memory-friendly.

        `.copy()` on found patches is load-bearing: without it, each kept
        patch is a numpy view into the chunk, which keeps the whole chunk
        alive — defeating the chunked-release pattern and leaking memory
        in proportion to (#chunks visited × chunk_size).
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

        n_chunks = (pool_size + chunk_size - 1) // chunk_size
        chunk_order = rng.permutation(n_chunks)
        done = False
        for chunk_idx in chunk_order:
            if done:
                break
            i0 = int(chunk_idx) * chunk_size
            i1 = min(i0 + chunk_size, pool_size)
            # Materialize the chunk in RAM (~18 MB at chunk_size=50000 @19×19).
            # For memmap pools this is a contiguous read — OS prefetch loves it.
            chunk = np.asarray(pool[i0:i1])
            for patch in chunk:
                if self.classify(patch) == 1:
                    # `.copy()`: detach from `chunk` so we can release it on
                    # next iteration without invalidating kept patches.
                    found.append(patch.copy())
                    pbar.update(1)
                sampled += 1
                if len(found) >= target or sampled >= budget:
                    done = True
                    break
            pbar.set_postfix(sampled='{:,}'.format(sampled),
                             fpr='{:.2f}%'.format(100.0 * len(found) / max(sampled, 1)))
            # Drop the local ref so the chunk can be GC'd before the next slice.
            del chunk
        pbar.close()
        print("\t- {}mined {} from {} ({:.2f}% FPR) in {}".format(
            prefix, len(found), sampled,
            100.0 * len(found) / max(sampled, 1), get_pretty_time(start_time)))
        return found

    def _mine_hard_negatives(self, neg_pool, target, budget, rng,
                             neg_seed=None, very_hard_pool=None):
        """Mine hard negatives, optionally stratified across two pools.

        When `neg_seed` is given (e.g. CBCL non-faces), allocate half the
        target/budget to it and the other half to `neg_pool` (Caltech).
        This keeps matched-domain hard negatives in the training mix at
        EVERY stage, not just stage 1 — fixing the "stage 9 trained with
        only Caltech-flavored negatives" symptom we observed.

        When `very_hard_pool` is given (pre-mined against a strong oracle
        cascade), it is used ONLY as a last-resort top-up after seed +
        Caltech mining returns fewer patches than `target`. Deep stages
        deplete the main pool's mineable patches (each pass takes longer
        and finds fewer "still-misclassified-as-face" patches); the
        reservoir absorbs that shortfall instead of letting `target` slip.
        """
        print("Mining hard negatives (target={}, budget={})...".format(target, budget))
        found = []
        if neg_seed is not None and len(neg_seed) > 0:
            t_seed = target // 2
            t_main = target - t_seed
            # Cap seed budget at "scan whole pool 2×" — past that, we're
            # just resampling the same patches and mining returns nothing
            # new. Any unspent budget is reallocated to the main pool below.
            b_seed_requested = budget // 2
            b_seed = min(b_seed_requested, len(neg_seed) * 2)
            b_main = budget - b_seed
            found += self._mine_from_pool(neg_seed, t_seed, b_seed, rng,
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
        # Top up from the very-hard reservoir only after the regular pools
        # have been exhausted. Order matters: regular mining never sees
        # these patches, so the cascade keeps learning against fresh
        # negatives while it can. When mining naturally falls short
        # (typical at stages 12+), the reservoir fills the gap.
        shortfall = target - len(found)
        if shortfall > 0 and very_hard_pool is not None and len(very_hard_pool) > 0:
            # Budget here is "scan the whole reservoir once" — these patches
            # are already model-verified hard, so we want all the ones the
            # current partial cascade still misclassifies. Reservoir is
            # small (10-100K), single pass is cheap.
            topup = self._mine_from_pool(
                very_hard_pool, shortfall, len(very_hard_pool), rng,
                label="vhard")
            found += topup
            print("\t- vhard top-up: requested {}, kept {}".format(
                shortfall, len(topup)))
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

    def find_faces(self, pil_image, growth=None, min_shift=None,
                   min_face_size=None, max_face_size=None,
                   min_score=None):
        """
        Multi-scale sliding-window detection on a PIL image.

        Strategy (paper §5) — fully vectorized cascade per scale:
          - One padded integral image + one squared II for the WHOLE image.
          - At each scale we form the grid of window origins (x1, y1) and
            evaluate every cascade stage as a batched NumPy reduction over
            *all* surviving windows. Stage k sees only windows that passed
            stage k-1, preserving the per-window early-exit cascade without
            any Python-level per-window dispatch. The inner work is 4
            fancy-indexed array reads per Haar rectangle, so we replace
            millions of scalar Python lookups with a handful of vectorized
            ones — ~20-50× faster than the old per-window loop.
          - Window shift grows with scale: at scale s we step `max(1, s)`
            pixels. Stepping 2px at scale 8 just produces near-duplicate
            detections that NMS later collapses anyway.

        Args:
            min_face_size: smallest detectable face in image pixels. Skips
                pyramid scales whose window is smaller than this. The
                training resolution (`base_width`) is the hard floor — a
                value below it is silently clamped.
            max_face_size: largest detectable face in image pixels. Stops
                the pyramid once the window exceeds this.
            min_score: discard regions whose accumulated cascade margin
                (the `score` field — see Returns) falls below this value.
                None disables filtering. Typical useful range is 0.05–0.5
                for a 10–15 stage cascade; higher values filter aggressively.

        Returns:
            list of (x1, y1, x2, y2, score) tuples in image coordinates.
            `score` is the sum of per-stage AdaBoost margins
            (`vote − layer_threshold`, where `vote = Σαᵢ·hᵢ(x)/Σαᵢ`)
            accumulated across every stage the window passed. A confident
            face clears every threshold with room to spare and accumulates
            a large margin; a borderline FP scrapes through and ends with
            a near-zero score. This continuous score is what
            `non_maximum_supression` uses to weight the cluster centroid
            and to pick the representative in greedy mode.
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

        # Single-shot integral images for the entire image. Cast to int64
        # up front so all the rect-sum subtractions stay in signed
        # arithmetic without per-window casts.
        ii = integral_image(image).astype(np.int64)
        ii2 = integral_image_pow2(image).astype(np.int64)

        # Resolve scale bounds from face-size limits. base_width is the
        # floor — smaller windows can't match the learned features.
        start_scale = 1.0
        if min_face_size is not None:
            start_scale = max(1.0, float(min_face_size) / max(w, h))
        max_scale = (float(max_face_size) / max(w, h)
                     if max_face_size is not None else None)

        # Snapshot per-stage data once: alphas vector, sum(alphas), the
        # calibrated layer threshold, and the WC list.
        stages = [(np.asarray(c.alphas, dtype=np.float64),
                   float(sum(c.alphas)),
                   float(c.threshold),
                   c.clfs)
                  for c in self.clfs]

        regions = []
        scale = start_scale
        pbar = tqdm(desc='Pyramid', unit='scale', leave=False)
        while int(w * scale) <= img_w and int(h * scale) <= img_h:
            if max_scale is not None and scale > max_scale:
                break
            win_w = int(w * scale)
            win_h = int(h * scale)
            shift = max(min_shift, int(scale))

            ys = np.arange(0, img_h - win_h + 1, shift, dtype=np.int64)
            xs = np.arange(0, img_w - win_w + 1, shift, dtype=np.int64)
            if ys.size == 0 or xs.size == 0:
                scale *= growth
                continue
            yy, xx = np.meshgrid(ys, xs, indexing='ij')
            x1 = xx.ravel()
            y1 = yy.ravel()
            x2 = x1 + win_w
            y2 = y1 + win_h

            # Per-window pixel std via the full-image IIs (vectorized).
            area = float(win_w * win_h)
            sum_x = ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]
            sum_x2 = ii2[y2, x2] - ii2[y1, x2] - ii2[y2, x1] + ii2[y1, x1]
            mean = sum_x / area
            var = sum_x2 / area - mean * mean
            std = np.sqrt(np.maximum(var, 0.0))
            std = np.where(std >= 1.0, std, 1.0)

            # Cascade: filter the alive index set stage by stage, and
            # accumulate the per-stage margin (vote − layer_thr) as a
            # continuous confidence score per surviving window.
            alive = np.arange(x1.size, dtype=np.int64)
            margin_sum = np.zeros(x1.size, dtype=np.float64)
            s2 = scale * scale
            for alphas, sum_alpha, layer_thr, wcs in stages:
                if alive.size == 0 or sum_alpha <= 0.0:
                    break
                ox = x1[alive]
                oy = y1[alive]
                std_a = std[alive]
                total = np.zeros(alive.size, dtype=np.float64)
                for alpha, wc in zip(alphas, wcs):
                    fv = self._batch_haar_value(wc.haar_feature, ii,
                                                scale, ox, oy)
                    thr_scaled = wc.threshold * s2 * std_a
                    pol = wc.polarity
                    pred = pol * fv < pol * thr_scaled
                    total += alpha * pred
                vote = total / sum_alpha
                passed = vote >= layer_thr
                margin_sum[alive[passed]] += vote[passed] - layer_thr
                alive = alive[passed]

            if alive.size:
                if min_score is not None:
                    keep = margin_sum[alive] >= float(min_score)
                    alive = alive[keep]
            if alive.size:
                xs1 = x1[alive].tolist()
                ys1 = y1[alive].tolist()
                xs2 = x2[alive].tolist()
                ys2 = y2[alive].tolist()
                scs = margin_sum[alive].tolist()
                regions.extend((xs1[i], ys1[i], xs2[i], ys2[i], scs[i])
                               for i in range(alive.size))

            pbar.set_postfix(scale='{:.2f}'.format(scale),
                             detections=len(regions))
            pbar.update(1)
            scale *= growth
        pbar.close()
        return regions

    @staticmethod
    def _batch_haar_value(haar, ii, scale, ox, oy):
        """Vectorized Haar feature value across a batch of window origins.

        `ox`, `oy` are int64 arrays. Returns an int64 array of
        `sum(negative_regions) - sum(positive_regions)`, element-wise
        identical to `HaarFeature.compute_value` for each (ox[i], oy[i]).
        """
        def rects_sum(regions):
            total = np.zeros(ox.shape, dtype=np.int64)
            for r in regions:
                dx1 = int(r.x * scale)
                dy1 = int(r.y * scale)
                dx2 = int((r.x + r.width) * scale)
                dy2 = int((r.y + r.height) * scale)
                total += (ii[oy + dy2, ox + dx2]
                          - ii[oy + dy1, ox + dx2]
                          - ii[oy + dy2, ox + dx1]
                          + ii[oy + dy1, ox + dx1])
            return total
        return rects_sum(haar.negative_regions) - rects_sum(haar.positive_regions)

    def save(self, filename):
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
