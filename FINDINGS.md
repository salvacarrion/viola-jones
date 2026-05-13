# Viola-Jones — Experimental Findings

Complete record of training runs, failure diagnoses, fixes, and open questions.
All benchmark numbers are on the **CBCL test set**: 472 faces / 23 573 non-faces.

---

## Quick summary

| Best model | Config | Recall | Spec | F1 |
|---|---|---|---|---|
| `weights/24/cvj_weights_1777843525_tuned.pkl` | CBCL-only, 24×24, 9 stages, post-hoc F1 tune | 0.597 | 0.995 | **0.653** |

Everything else — CelebA, multi-source, deeper stages on bad bases — performed worse.
The six fixes documented below are what got from F1=0.044 to F1=0.653.

---

## All runs, chronological

### Run 1 — Baseline (19×19, FDDB faces, Caltech negatives)

```
Config:  --face-source fddb  --neg-source caltech  --resolution 19  --layers 5 20 50 100
```

| Recall | Spec | Precision | F1 | Train time |
|-------:|-----:|----------:|---:|-----------:|
| 0.784  | 0.326 | 0.023 | 0.044 | ~1 h 47 m |

**What happened:** recall was fine (0.78) but specificity was 0.326 — the cascade called
67 % of CBCL non-faces "face". F1 collapsed entirely because the 1:50 face/non-face ratio
means even a 5 % FPR drowns true positives in false positives.

**Root cause:** Caltech-256 negatives (random crops of object photos — textures, edges,
uniform regions) are trivially different from CBCL non-faces (curated to be face-like:
face-pose backgrounds, near-symmetric features). The cascade learned "not flat, not
pure-edge" and thought it was done. CBCL near-faces crept through unchallenged.

---

### Run 2 — Matched seed, CBCL faces (19×19)

```
Config:  --face-source cbcl  --neg-source mixed  --resolution 19  --layers 5 20 50 100
```

| Recall | Spec | Precision | F1 | Train time |
|-------:|-----:|----------:|---:|-----------:|
| 0.502  | 0.979 | 0.292 | 0.369 | ~15 m |

**Fix applied:** `--neg-source mixed` seeds stage 1 with CBCL non-faces instead of
Caltech patches. Specificity jumped 0.33 → 0.98 in one shot. This single change is
the largest improvement of the whole project.

**Residual problem:** recall dropped to 0.50. Hard-negative mining in stages 2–9 still
draws only from Caltech. By stage 9 the log showed "Stage negatives: 947" — the
cascade had memorised Caltech so completely that only 947 of 1 M random patches still
passed. CBCL non-face patterns crept back in unchallenged in late stages.

**Run-to-run variance** at this config: ±0.025 F1 across retrains with identical
commands, driven by random seeds in mining and the val/train split.

---

### Run 3 — Shorter cascade, CBCL (19×19)

```
Config:  --face-source cbcl  --neg-source mixed  --resolution 19  --layers 5 10 20 40
```

| Recall | Spec | Precision | F1 |
|-------:|-----:|----------:|---:|
| 0.710  | 0.916 | 0.138 | 0.231 |

Fewer stages → higher recall but lower specificity. Worse F1 than the 4-stage run because
the precision drop from spec=0.916 more than offsets the recall gain at 1:50 class ratio.

---

### Run 4 — CelebA faces, matched seed (19×19)

```
Config:  --face-source celeba  --neg-source mixed  --resolution 19  --layers 5 20 50 100
```

| Recall | Spec | Precision | F1 |
|-------:|-----:|----------:|---:|
| 0.017  | 0.977 | 0.017 | 0.017 |

**What happened:** catastrophic recall collapse.

**Root cause (pixel alignment):** CBCL crops are tightly cropped 19×19 native captures —
eyes land at a fixed row, nose at another, mouth at another. CelebA crops are 48×48
native with looser framing (more forehead, hair, neck); bilinearly downsized to 19×19,
the same landmarks land at different pixel rows. A 1-pixel misalignment on a 19-pixel
face is 5 % of the face height. Haar features tile rectangle sums at fixed grid
positions, so the distribution of eye-region features shifts entirely. The cascade
learned "eyes at row 8"; CelebA faces have eyes at row 6 and look like non-faces in
feature space.

**Lesson:** resolution matters more than face source. At 19×19 only
tightly-cropped, aligned sources (CBCL) are usable.

---

### Run 5 — Smoke test 24×24 (CBCL+jitter, 4 stages)

```
Config:  --face-source cbcl  --neg-source mixed  --resolution 24  --jitter 2
         --layers 5 10 20 30
```

| Recall | Spec | Precision | F1 | Train time |
|-------:|-----:|----------:|---:|-----------:|
| 0.909  | 0.694 | 0.057 | 0.107 | ~50 m |

**Purpose:** confirm 24×24 doesn't break alignment. Recall 0.91 confirms it.
Specificity 0.69 is low because 4 shallow stages can't reject face-like non-faces.

**Key insight:** 24×24 has ~5 px alignment slack per edge. CelebA bilinear downsample
now lands within the same Haar grid bucket as CBCL. The fix (resolution) costs ~10×
more Haar features (60 153 features vs ~6 000 at 19×19) and proportionally longer
training, but enables multi-source training that 19×19 could not.

---

### Run 6 — Deep cascade 24×24, CBCL-only **(best model)**

```
Config:  --face-source cbcl  --neg-source mixed  --resolution 24  --jitter 2  --augment
         --layers 5 10 20 40 60 80 100 150 200 250  (855 WCs total, 9 stages)
         --target-neg-per-stage 5000  --neg-sample-budget 500000
Weights: weights/24/cvj_weights_1777843525.pkl
```

**Untuned:**

| Recall | Spec | Precision | F1 | Train time |
|-------:|-----:|----------:|---:|-----------:|
| 0.826  | 0.943 | 0.228 | 0.357 | ~10 h |

**After `--objective recall-at-spec --min-spec 0.97`:**

| Recall | Spec | Precision | F1 |
|-------:|-----:|----------:|---:|
| 0.636  | 0.971 | 0.309 | 0.416 |

**After `--objective f1` (best benchmark headline):**

| Recall | Spec | Precision | F1 |
|-------:|-----:|----------:|---:|
| 0.597  | 0.995 | 0.719 | **0.653** |

**Why threshold tuning gained so much (0.357 → 0.653):** `adaboost.py calibrate()` had
a hardcoded `min(0.5, ...)` cap. At deep stages, val-pos scores clustered well above 0.5
— calibration WANTED to push thresholds to 0.6+ but got clamped. The stages were
accepting far more non-faces than they needed to. Post-hoc tuning bypassed the cap and
found the correct cut points. **Fix:** raised cap to 0.95 in `calibrate()` so future
runs bake in correct thresholds during training without needing post-hoc correction.

**Cascade diagnose output (per-stage scores before tuning):**
```
Stage  thr   pos_μ  pos_p1  neg_μ  neg_p99  pass+   pass-   rej_this
  1   0.145  0.614  0.145   0.141  0.562    99.2%   28.6%   71.4%
  2   0.140  0.620  0.236   0.113  0.476    98.1%   30.8%   22.0%
  ...
  9   0.500  0.832  0.500   0.441  0.769    83.1%    6.7%    4.2%
```
Stage 9 threshold stuck at 0.500 (the old cap). After tuning, stage 9 moved to ~0.72,
cutting stage-9 FPR from 6.7 % to 0.5 % — the biggest single-stage gain.

**Visual detection quality:** finds most frontal faces in group shots, misses
profile/occluded faces, produces false positives on suits/ties (dark-light-dark vertical
pattern matches eye-nose-eye Haar template) and building windows.
Raw detections ~300 per image; after hybrid NMS, typically 5–30 final boxes.

---

### Run 7 — CelebA+CBCL multi-source (24×24, 9 stages)

```
Config:  --face-source celeba+cbcl  --n-faces 10000  --neg-source mixed  --resolution 24
         --jitter 2  --augment
         --layers 5 10 20 40 60 80 100 130 160
         ~50 K train positives (10K CelebA + 2.4K CBCL × augment × jitter)
Weights: weights/24/cvj_weights_1778107248.pkl (or cvj_weights_1778197566.pkl)
```

| Recall | Spec | Precision | F1 |
|-------:|-----:|----------:|---:|
| 0.814  | ~0.78 | ~0.06 | ~0.11 (very rough) |

Visual output: 106 254 raw detections on `people.png`. Cascade barely rejects anything.

**What happened:** val_pos contains both CBCL and CelebA crops. CelebA crops score lower
on the cascade than CBCL crops — they look less face-like to the features learned at
stage 1 (which were trained with CBCL-aligned faces). So `calibrate()` pushes thresholds
down to accommodate the weakest CelebA outliers in val, opening the gate for non-faces.
Effectively, the weakest source in a multi-source val drags every stage's threshold
toward zero.

**CelebA-only variant was even worse:**

| Recall | Spec | F1 | Notes |
|-------:|-----:|---:|-------|
| 0.034  | ~0.98 | ~0.06 | pos μ = 0.162 < neg μ = 0.198 — cascade inverted |

Stage-1 score diagnostic showed `pos μ < neg μ`: non-faces literally scored higher than
real faces. The cascade had been trained with CBCL-style aligned crops (from the
CBCL non-face seed) and encountered CelebA faces at evaluation time — same alignment
gap as Run 4 but at 24×24. At 24×24 the gap is smaller (5 px slack vs 1 px at 19×19)
but CelebA's loose framing still shifts the eye-region Haar features relative to the
tight CBCL training crops.

**Conclusion:** CelebA faces are not interchangeable with CBCL faces even at 24×24 when
the detection benchmark is CBCL. The domain gap persists. More stages make it worse,
not better (adding stages to a cascade with recall=0.034 pushes recall toward 0.000).

---

## Bugs found and fixed

### 1. `min_shift=1` in `find_faces` ignored `self.shift=2`

`ViolaJones.__init__` sets `self.shift = 2` but `find_faces(pil_image, growth=None, min_shift=1)`
defaulted to 1 and never read `self.shift`. At scale < 2.0, `int(scale)=1 < min_shift=1`
so the window stepped every pixel instead of every 2 pixels. **Quadrupled window count.**
For a 640×480 image with a 24×24 base window, this generated ~280 K windows at scale=1.0
instead of ~70 K. With 5 images and 9 cascade stages, detection took 12+ minutes.

**Fix:** `find_faces(pil_image, growth=None, min_shift=None)` with
`if min_shift is None: min_shift = self.shift` at the top of the method.

**Actual detection time after fix:** ~3 min for 5 images (4× improvement).

### 2. Calibration 0.5 cap suppressed deep-stage thresholds

`calibrate()` had `self.threshold = float(min(0.5, max(min_thr, sorted_desc[k-1])))`.
At deep stages, 95%+ of val faces score 0.65–0.90, so the correct threshold is above 0.5.
The cap clamped it at 0.5, leaving late stages accepting 5–7 % of non-faces unnecessarily.

**Fix:** raised to `min(0.95, ...)`. Now deep stages get calibrated to their correct
operating point during training, reducing (but not eliminating) the gap between training
calibration and post-hoc tuning.

### 3. Stale feature cache after data source change

`data/24/_cache/xf_pos.npy` is written after the first training run and reused on
subsequent runs. If `train_pos` changes (different source, resolution, or sample count)
but the cache exists, the old features are silently reused — producing a mismatch
between the feature matrix dimensions and the current `train_pos.shape`.

**Fix (workaround):** always `rm -rf data/24/_cache` before retraining with different
data. The cache is safe to reuse only when the exact same `prepare_data.py` command was
run.

### 4. Jitter leakage across train/val split

Original multi-source implementation called `jitter_crops()` on the full face array
then `split_three_way()`. The same underlying face could end up in train (center crop)
AND val (jittered crop) — breaking the independence assumption for calibration.

**Fix:** call `split_three_way()` on `faces_unique` (pre-jitter unique face indices),
then apply `jitter_crops()` per split separately.

### 5. Asymmetric NMS for nested boxes

Old code used `intersection / smaller-area` as the overlap metric, but the
"smaller area" comparison was order-dependent (which box was `idx` vs `remaining`
varied with iteration order). A 24×24 box inside a 48×48 box gave different merge
decisions depending on which box was processed first.

Standard IoU is symmetric but undershoots for deeply nested boxes: a 24×24 box
fully inside a 48×48 box has IoU = 576/(576+2304-576) ≈ 0.25, below the default 0.3
threshold — so they don't fuse, producing "boxes on boxes" stacking.

**Fix:** `metric='hybrid'` (default): fuse if `IoU > threshold OR IoM > 0.7`,
where `IoM = intersection / min(area1, area2)`. IoM is 1.0 for any nested box
regardless of scale ratio, so it catches what IoU misses.

### 6. Adaptive cascade — three iterations to a correct early-stop

This one is worth recording in detail because the bug-hunt taught more about the
algorithm than the fix itself. The goal was to replace the rigid
`--layers 5 10 20 40 ...` schedule with a per-stage FPR target (V&J §3) so cascade
depth and per-stage weak-classifier count emerge from training.

**Iteration 1 — FPR at threshold 0.5.** First version of `target_stage_fpr` in
`adaboost.py`:

```python
fpr = (running_neg_scores / sum_alpha_fpr >= 0.5).mean()
if fpr <= target_stage_fpr: break
```

Symptom: every stage stopped at round 1.

```
[AdaBoost] FPR 0.150 ≤ 0.500; early stop at round 1/200
   - layer threshold (after calibrate): 0.9500
   - cumulative val recall: 0.8017  (stop if < 0.9)
   ! Recall 0.8017 < 0.9: stopping cascade.
```

Root cause: the FPR check measures at `score ≥ 0.5` (un-calibrated majority vote),
but the deployed threshold after `calibrate()` is ~0.15-0.20 (the operating point
that maintains `--layer-recall=0.99` on val_pos). At threshold 0.5 a single decent
stump easily achieves FPR=0.15; at the actual deployed threshold the FPR is much
higher and recall collapses. The metric was lying.

**Iteration 2 — `--min-wcs-per-stage` floor.** Workaround: gate the early-stop on
`len(self.clfs) >= min_estimators`. Stage 1 now trained 5 WCs before the FPR check
could fire.

Symptom: every stage stopped at exactly `min_wcs`.

```
[AdaBoost] FPR 0.141 ≤ 0.300; early stop at round 5/200  (stage 1)
[AdaBoost] FPR 0.128 ≤ 0.300; early stop at round 5/200  (stage 2)
[AdaBoost] FPR 0.129 ≤ 0.300; early stop at round 5/200  (stage 3)
...
```

Cascade compounded as expected (cumulative val-recall 0.99 → 0.98), but hard-neg
mining FPR per stage dropped slowly (62 % → 42 % → 31 % → 18 % → 11 %). The 5-WC
floor was hiding the fact that the metric itself was wrong: still at threshold 0.5,
still lying about the operating point. The cascade was working but each stage was
weaker than its log suggested.

**Iteration 3 — calibrated FPR + recall, both checked.** The fix:

1. Pre-compute features for val_pos once (cached as `data/24/_cache/xf_val_pos.npy`
   or `xf_val_cal.npy` when CBCL-only calibration set exists).
2. Per round in `AdaBoost.train`: apply the newly chosen WC to val_pos features,
   maintain `running_val_scores`, recompute `self.threshold` via
   `_calibrated_threshold()` so `target_recall` of val_pos passes.
3. Compute `recall_at_thr = (val_scores >= threshold).mean()` and
   `fpr_at_thr = (neg_scores >= threshold).mean()` — both at the same calibrated
   threshold, both reflecting deployed behavior.
4. Early-stop only when `recall_at_thr ≥ target_recall AND fpr_at_thr ≤
   target_stage_fpr` — V&J §3 verbatim.

The recall check matters because of an edge case: with 1-2 WCs the ensemble score
takes only 2-3 distinct values; the recall-quantile lookup inside
`_calibrated_threshold()` gets clamped by the `min(alphas)/sum(alphas)` floor or
the 0.95 cap. At that clamped threshold, recall silently falls below target. The
explicit check `recall_at_thr >= target_recall` refuses to early-stop until the
threshold has settled into the regime where both metrics reflect reality.

The barplot in the boosting progress now shows `thr=` decreasing from 0.95 toward
the eventual operating point (~0.20-0.30), `rec=` climbing toward `target_recall`,
and `fpr=` initially rising (as `thr` drops) then falling (as the ensemble
strengthens at the lower threshold).

`--min-wcs-per-stage` survives as an inert safeguard with default 1.

**Lesson:** when introducing a "stop when metric X is good enough" rule, verify
that X is measured at the same operating point as deployment. A metric at the
wrong threshold is worse than no metric — it gives confident misleading numbers.

---

## What the other LLM suggested — assessment

### Adaptive per-stage FPR training
Instead of a fixed T rounds per stage, train until the stage's training FPR drops
below a target (e.g. 50 %). Early stages need 5–10 weak classifiers; later stages
may need 200+. This is actually closer to the original Viola & Jones paper.

**Assessment: valid and worth implementing.** The current fixed-T training sometimes
adds unnecessary weak classifiers to early stages (already at low FPR) and stops too
early in late stages. The main cost is implementation complexity (need a per-stage FPR
loop instead of `range(T)`). The payoff is a tighter cascade: fewer total weak
classifiers, faster inference, and better calibration at each stage boundary.

**UPDATE — implemented, after two false starts.** Naive version (FPR measured at the
un-calibrated threshold 0.5, no recall check) collapsed every stage to 1 weak
classifier. A `--min-wcs-per-stage` floor hid the symptom but not the cause. Working
version (Bug #6 below) recalibrates the stage threshold every round on val_pos and
requires BOTH `recall ≥ target_recall` AND `fpr ≤ target_stage_fpr` at that
calibrated threshold — V&J §3 verbatim. See README § 7 for the narrative.

### `min_w=1, min_h=1` in `build_features`
Adding 1–3 px Haar features would expand the feature set from ~60 K to potentially
500 K+ features at 24×24.

**Assessment: not worth it.** Tiny features are noise at this resolution — a 1×2 Haar
on a 24×24 face doesn't encode meaningful structure. Training time scales linearly in
feature count; 10× more features means 10× longer training per round. The original
paper used `min_size=1` but their 24×24 window was 400 dpi film scans, not bilinearly
downsampled 48×48 crops. Stick with `min_w=4, min_h=4`.

---

## What worked vs what didn't

| Change | Effect | Why |
|--------|--------|-----|
| Matched-domain CBCL seed (stage 1) | Spec 0.33 → 0.98 | Training negatives matched test distribution |
| 24×24 resolution | Enables CelebA, solves alignment | 5 px slack vs 1 px at 19×19 |
| Stratified mining (CBCL+Caltech, all stages) | Prevents specificity drift in deep stages | Keeps matched-domain negatives throughout |
| Post-hoc threshold tuning | F1 0.357 → 0.653 (same model) | Bypasses the 0.5 calibration cap |
| Raising calibration cap to 0.95 | Bakes correct thresholds during training | Deep stages can now calibrate above 0.5 |
| Hybrid NMS (IoU OR IoM) | Eliminates "boxes on boxes" | IoM handles nested multi-scale duplicates |
| Score-weighted average NMS | Tighter box localization | Fuses cluster into weighted centroid |
| Fix `min_shift` bug | 4× faster detection | Actually uses `self.shift=2` as intended |
| CelebA-only training | Cascade inverted (recall 0.034) | Domain gap persists even at 24×24 |
| CelebA+CBCL multi-source | 106 K raw detections, unusable | CelebA outliers in val drag thresholds to zero |
| Adding stages to a bad base | Makes recall worse | More rejection stages = lower recall |
| Jitter augmentation | Mild improvement in val/test generalization | Translation invariance within ±2 px |
| H-flip augmentation | Marginal | Already augmented implicitly by symmetric Haar |

---

## Current best configuration

**Data:**
```bash
python tools/prepare_data.py \
    --face-source cbcl \
    --neg-source mixed \
    --resolution 24 \
    --augment \
    --jitter 2
```

**Train:**
```bash
python main.py train \
    --data-dir data/24 \
    --layers 5 10 20 40 60 80 100 150 200 250 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 500000
```

**Tune:**
```bash
# For best benchmark F1:
python tools/tune_thresholds.py \
    --weights $(ls -t weights/24/*.pkl | head -1) \
    --data-dir data/24 \
    --objective f1

# For visual detection (recall-first):
python tools/tune_thresholds.py \
    --weights $(ls -t weights/24/*.pkl | head -1) \
    --data-dir data/24 \
    --objective recall-at-spec --min-spec 0.97
```

**Detect:**
```bash
python main.py detect \
    --weights-path weights/24/cvj_weights_1777843525_tuned.pkl \
    --detect-output images/outputs_best
```

---

## What would plausibly improve on F1=0.653

In rough priority order:

1. **Deeper cascade on CBCL-only data.** The 9-stage run saturated at 855 total weak
   classifiers. Going to 12–15 stages with the same budget-per-stage gives later stages
   more capacity to reject the residual near-faces that get through stage 9. Worth a
   12–15 h run. Keep `--face-source cbcl` only. Now easier to attempt — `--max-stages
   N` instead of hand-tuning `--layers` (see Bug #6 / README § 7).

2. **Re-measure with the adaptive cascade.** The F1=0.653 best was a fixed-`--layers`
   schedule. With the new adaptive trainer + calibrated FPR+recall check, the same
   data should produce a tighter cascade — early stages with fewer WCs (faster
   inference), late stages with more (better rejection), and per-stage thresholds
   already at the operating point so the post-hoc tuner has less to fix. Expect a
   modest gain or roughly equivalent F1 at lower compute.

3. **More CBCL faces.** CBCL HF train split has 2 429 unique faces. All augmentation
   and jitter permutations of that set are ~7 800 samples — still limited for 12+
   stages. There is no easy fix here without a different benchmark dataset.

4. **Larger CBCL non-face pool.** The CBCL non-face seed has 4 548 unique patches
   (×2 with h-flip = 9 096). In deep stages, stratified mining exhausts this pool
   quickly and falls back to Caltech. A larger curated pool would keep matched-domain
   negatives available longer.

5. **Do NOT try:** CelebA faces (domain gap), min_w=1 features (10× cost, marginal
   gain), adding stages to the existing CelebA models (they are inverted — more stages
   kill recall).
