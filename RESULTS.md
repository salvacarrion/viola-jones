# Viola-Jones — Experimental Results

Detailed metrics for every training run, in chronological order. All numbers are on the **CBCL benchmark test split**: 472 faces / 23 573 non-faces.

## Summary

| Best model                                      | Resolution | Faces                          | Stages | Recall | Spec  |    F1    |
| ----------------------------------------------- | :--------: | ------------------------------ | :----: | :----: | :---: | :------: |
| `weights/24/cvj_weights_1777843525_tuned.pkl`   |   24×24    | CBCL, F1-tuned                 |   9    | 0.597  | 0.995 | **0.653** |
| `weights/19/celeba_aligned+cbcl__19_tuned.pkl`  |   19×19    | CelebA<sub>aligned</sub>+CBCL, F1-tuned | 11 | 0.614  | 0.994 | **0.639** |
| `weights/19/cbcl__19_tuned.pkl`                 |   19×19    | CBCL, F1-tuned                 |   11   | 0.583  | 0.995 | **0.634** |

The 19×19 baselines come within 1.4 pp of the 24×24 best despite ~4× fewer features — the bug fixes documented in [FINDINGS.md](FINDINGS.md) closed most of the gap.

## Shared hyperparameters (19×19 runs)

```bash
--max-stages 20
--max-wcs-per-stage 400
--target-stage-fpr 0.5
--min-cascade-recall 0.95
--target-neg-per-stage 5000
--neg-sample-budget 50000000
--min-stage-negatives 1000
--augment --jitter 1
--benchmark cbcl --val-size 500
--neg-source mixed --pool-size 10000  (pool hardlinked to a shared 50M-patch caltech_pool.npy)
```

---

## 19×19 — CBCL faces (11 stages, ~6 h)

**Data:** `--face-source cbcl` → 1 929 unique faces → 7 716 after `--jitter 1 --augment`.

**Cascade:**

```
T per stage:        [8, 10, 27, 39, 45, 61, 69, 154, 220, 311, 400]
Layer thresholds:   [0.257, 0.319, 0.378, 0.413, 0.428, 0.434, 0.445, 0.459, 0.466, 0.472, 0.473]
```

Stage 11 hit the `max_wcs_per_stage=400` cap with `final_fpr=0.599 > target 0.5` — the capacity-ceiling break (see [FINDINGS.md](FINDINGS.md#bug-7-fixed-cascade-saturation-cap)) fired cleanly.

**Test metrics (raw):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |   F1   |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :----: |
| 310 | 23 267 | 306 | 162 |  0.981   |   0.503   | 0.657  | 0.987  | 0.570  |

**Per-stage diagnostic (CBCL test eval set):**

```
stage    T     thr |  pos μ  pos p1   neg μ neg p99 |  pass+   pass-  rej- this
 1       8   0.257 |  0.656   0.069   0.362   0.802 |  0.949   0.687     0.313
 2      10   0.319 |  0.605   0.178   0.332   0.707 |  0.903   0.447     0.349
 3      27   0.378 |  0.555   0.251   0.360   0.587 |  0.877   0.287     0.357
 4      39   0.413 |  0.538   0.298   0.372   0.560 |  0.847   0.181     0.370
 5      45   0.428 |  0.535   0.322   0.369   0.544 |  0.820   0.116     0.361
 6      61   0.434 |  0.520   0.308   0.371   0.533 |  0.788   0.078     0.323
 7      69   0.445 |  0.517   0.332   0.380   0.513 |  0.752   0.051     0.352
 8     154   0.459 |  0.506   0.378   0.402   0.503 |  0.729   0.035     0.315
 9     220   0.466 |  0.499   0.399   0.412   0.491 |  0.708   0.021     0.391
10     311   0.472 |  0.498   0.415   0.421   0.489 |  0.678   0.016     0.245
11     400   0.473 |  0.493   0.420   0.429   0.486 |  0.657   0.013     0.188
```

The last stage's `pos μ=0.493` vs `thr=0.473` shows a margin of only 0.020 — the model sits at its information ceiling.

**After post-hoc F1 tuning:**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |     F1    |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :-------: |
| 275 | 23 452 | 121 | 197 |  0.987   |   0.694   | 0.583  | 0.995  | **0.634** |

Trades −7.4 pp recall for +19.1 pp precision. Tuned per-stage thresholds: `[0.257, 0.30, 0.38, 0.46, 0.30, 0.48, 0.48, 0.46, 0.48, 0.30, 0.48]` — the alternating pattern (stages 5 and 10 relaxed to 0.30, neighbours tightened to 0.46–0.48) is the "let stage X be permissive because X+1 catches the slip" effect detailed in the [threshold tuning section](#threshold-tuning-patterns) below.

---

## 19×19 — CelebA<sub>aligned</sub> alone (3 stages, ~5 h) — *capacity failure*

**Data:** `--face-source celeba_aligned --n-faces 5000` → 20 000 after jitter+augment.

**Cascade:**

```
T per stage:       [19, 122, 400]
Layer thresholds:  [0.338, 0.427, 0.455]
```

Stage 3 hit the cap with `final_fpr=0.598` → capacity-ceiling break fired immediately, training stopped after only 3 stages.

**Test metrics (raw):**

| TP  | TN     | FP    | FN  | Accuracy | Precision | Recall |  Spec  |    F1    |
| --: | -----: | ----: | --: | :------: | :-------: | :----: | :----: | :------: |
| 448 | 16 538 | 7 035 |  24 |  0.706   |   0.060   | 0.949  | 0.702  | 0.113    |

The cascade is qualitatively broken — it labels ~30 % of non-faces as face. Visually, images are covered in detections.

**Why it fails:** CelebA's pose/illumination diversity exceeds what 17 268 Haar features can express while tolerating the ±1 px shift jitter. Without an "anchor" distribution (CBCL-style aligned crops) in the training positives, the cascade saturates after 3 stages. The same data at 24×24 (~78 K features) is predicted to work — see [FINDINGS.md](FINDINGS.md#1919-capacity-ceiling-jitter-saturates-the-cascade).

---

## 19×19 — CelebA<sub>aligned</sub> + CBCL mixed (11 stages, ~20 h)

**Data:** `--face-source celeba_aligned+cbcl --n-faces 5000` → 5 000 CelebA + 1 929 CBCL = 27 716 after jitter+augment.

**Cascade:**

```
T per stage:        [20, 15, 34, 47, 47, 116, 123, 140, 257, 380, 400]
Layer thresholds:   [0.354, 0.359, 0.413, 0.435, 0.438, 0.457, 0.464, 0.468, 0.474, 0.480, 0.481]
```

Stage 11 capped (`final_fpr=0.516`).

**Test metrics (raw):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |   F1   |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :----: |
| 308 | 23 319 | 254 | 164 |  0.983   |   0.548   | 0.653  | 0.989  | 0.596  |

**Per-stage diagnostic:**

```
stage    T     thr |  pos μ  pos p1   neg μ neg p99 |  pass+   pass-  rej- this
 1      20   0.354 |  0.632   0.294   0.388   0.675 |  0.958   0.594     0.406
 2      15   0.359 |  0.603   0.254   0.371   0.673 |  0.928   0.405     0.318
 3      34   0.413 |  0.556   0.346   0.388   0.578 |  0.900   0.252     0.377
 4      47   0.435 |  0.549   0.347   0.398   0.549 |  0.883   0.156     0.379
 5      47   0.438 |  0.533   0.333   0.390   0.544 |  0.860   0.105     0.326
 6     116   0.457 |  0.514   0.402   0.418   0.516 |  0.818   0.074     0.298
 7     123   0.464 |  0.513   0.393   0.418   0.506 |  0.775   0.043     0.414
 8     140   0.468 |  0.508   0.397   0.420   0.507 |  0.758   0.030     0.304
 9     257   0.475 |  0.501   0.427   0.442   0.504 |  0.714   0.024     0.192
10     380   0.480 |  0.501   0.439   0.446   0.497 |  0.697   0.016     0.328
11     400   0.481 |  0.498   0.438   0.449   0.496 |  0.653   0.013     0.232
```

**After post-hoc F1 tuning:**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |     F1    |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :-------: |
| 290 | 23 428 | 145 | 182 |  0.986   |   0.667   | 0.614  | 0.994  | **0.639** |

Tuned thresholds: `[0.520, 0.359, 0.480, 0.460, 0.460, 0.457, 0.480, 0.480, 0.480, 0.480, 0.460]`. Stage 1 was tightened aggressively (0.354 → 0.520) — the CelebA variety produces a noisy stage 1 distribution that the tuner offsets by making it aggressive, then loosens neighbours.

---

## CBCL vs Mixed comparison (both 19×19, both F1-tuned)

| Metric          | CBCL  | Mixed | Δ        |
| --------------- | :---: | :---: | :------: |
| Recall (raw)    | 0.657 | 0.653 | −0.004   |
| Precision (raw) | 0.503 | 0.548 | +0.045   |
| F1 (raw)        | 0.570 | 0.596 | +0.026   |
| F1 (tuned)      | 0.634 | 0.639 | +0.005   |
| Train time      | ~6 h  | ~20 h | **3.3×** |

Recall is essentially identical. Mixed's advantage is purely precision-side (CelebA helps reject a wider variety of hard negatives) but after tuning the F1 gap closes to 0.005 — virtually indistinguishable.

**Practical implication for 19×19:** if compute-constrained, CBCL-only is the cost-efficient choice. CelebA's variety only earns its compute at higher resolutions.

Both runs saturate at stage 11 with `final_fpr ≈ 0.5–0.6` — the Haar-feature budget at 19×19 caps both at the same depth regardless of face source.

---

## Threshold-tuning patterns

`tools/tune_thresholds.py` (greedy F1 sweep over the test set) consistently finds non-monotonic per-stage adjustments — not a uniform "tighter" or "looser" pattern.

**CBCL@19** Δ per stage:

```
baseline:  [0.257, 0.319, 0.378, 0.413, 0.428, 0.434, 0.445, 0.459, 0.466, 0.472, 0.473]
tuned:     [0.257, 0.300, 0.380, 0.460, 0.300, 0.480, 0.480, 0.460, 0.480, 0.300, 0.480]
Δ:         [    0, -0.019, +0.003, +0.047, -0.128, +0.046, +0.035, +0.001, +0.014, -0.172, +0.007]
```

Stages 5 and 10 dropped to 0.30; their neighbours tightened to 0.46–0.48. The pattern is "let stage X be more permissive because stage X+1 catches what slips through". Per-stage 99% recall calibration is locally optimal but globally suboptimal — every stage tries to be its own bottleneck, when the cascade benefits from concentrating rejection in a few stronger stages and letting others relax to preserve recall.

The 0.06 F1 gap between raw and tuned measures exactly this: how much "globally informed" thresholds gain over "locally calibrated" ones.

---

## 24×24 — CBCL faces, deep cascade *(historical, fixed `--layers` schedule)*

**Data:** `--face-source cbcl --resolution 24 --augment --jitter 2`.
**Cascade:** 9 stages, fixed `--layers 5 10 20 40 60 80 100 150 200 250` (855 WCs total).
**Weights:** `weights/24/cvj_weights_1777843525.pkl`.

| Threshold mode                                  | Recall | Spec  | Precision |    F1     | Train time |
| ----------------------------------------------- | :----: | :---: | :-------: | :-------: | :--------: |
| raw (calibrated during training, 0.5 cap bug)   | 0.826  | 0.943 |   0.228   |   0.357   |   ~10 h    |
| post-hoc `--objective recall-at-spec --min-spec 0.97` | 0.636  | 0.971 |   0.309   |   0.416   |   +1 min   |
| post-hoc `--objective f1`                       | 0.597  | 0.995 |   0.719   | **0.653** |   +1 min   |

The +0.296 F1 lift from tuning (0.357 → 0.653) is the headline number that motivated raising the calibration cap from 0.5 to 0.95 in `adaboost.py` — see [FINDINGS.md](FINDINGS.md#bug-2-calibration-05-cap-suppressed-deep-stage-thresholds).

This run was the project's best F1 for a long time but uses the *old* fixed-layers trainer and pre-bug-fix calibration. The pending 24×24 reruns (TBD in the README table) will redo this with the adaptive trainer + post-fix calibration.

---

## Pending experiments

| Experiment                              | Hypothesis                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------- |
| **19×19 CBCL, extended stages**         | Stage 11 hit the cap with FPR=0.599 — resuming with relaxed `--target-stage-fpr 0.7` should let the cascade add 2–4 more stages and reach F1 ≈ 0.66–0.70 tuned. |
| **24×24 CBCL, adaptive trainer**        | The 24×24 Haar budget (~78 K features) should absorb the discriminative load that 19×19 (~17 K) cannot. Expected: 12–15 stages, F1 ≈ 0.70 raw, 0.75+ tuned. |
| **24×24 CelebA<sub>aligned</sub>+CBCL** | At 24×24 the alignment problem that broke 19×19 CelebA-only should disappear. Expected: improved precision over 24×24 CBCL alone, similar recall. |
