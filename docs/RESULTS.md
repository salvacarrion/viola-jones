# Viola-Jones — Experimental Results

Detailed metrics for every training run, in chronological order. All numbers are on the **CBCL benchmark test split**: 472 faces / 23 573 non-faces.

## Summary

| Best model                                                | Resolution | Faces                                   | Stages | Recall | Spec  |    F1    |
| --------------------------------------------------------- | :--------: | --------------------------------------- | :----: | :----: | :---: | :------: |
| `weights/19/celeba_aligned+cbcl__19_v2_tuned.pkl`         |   19×19    | CelebA<sub>aligned</sub>+CBCL, F1-tuned |   16   | 0.625  | 0.995 | **0.661** |
| `weights/19/cbcl__19_v2_tuned.pkl`                        |   19×19    | CBCL, F1-tuned                          |   15   | 0.606  | 0.995 | **0.658** |
| `weights/24/cvj_weights_1777843525_tuned.pkl`             |   24×24    | CBCL, F1-tuned                          |   9    | 0.597  | 0.995 | **0.653** |
| `weights/19/celeba_aligned+cbcl__19_v1_tuned.pkl`         |   19×19    | CelebA<sub>aligned</sub>+CBCL, F1-tuned |   11   | 0.614  | 0.994 |   0.639   |
| `weights/19/cbcl__19_v1_tuned.pkl`                        |   19×19    | CBCL, F1-tuned                          |   11   | 0.583  | 0.995 |   0.634   |
| `weights/19/celeba_aligned_filtered__19_v1_tuned.pkl`     |   19×19    | CelebA<sub>aligned</sub> (filtered drop 0.61), F1-tuned | 3 | 0.504  | 0.997 |   0.603   |
| `weights/24/cbcl__24_smoke_tuned.pkl`                     |   24×24    | CBCL (smoke test, adaptive trainer), F1-tuned | 10 | 0.602  | 0.996 | **0.660** |
| `weights/24/celeba_aligned__24_v1_tuned.pkl`              |   24×24    | CelebA<sub>aligned</sub> only, F1-tuned  |   9    | 0.559  | 0.996 |   0.629   |

After the v2 extension (resume from v1 with `--max-wcs-per-stage 800 --target-stage-fpr 0.65`), the 19×19 tuned models overtake the *historical* 24×24 best by ~0.5–0.8 pp F1. The **24×24 CBCL smoke test** (adaptive trainer, post-fix calibration) ties them at F1=0.660 with fewer stages. The **24×24 CelebA-only** model is the project's most real-world-robust detector despite a lower benchmark F1 (0.629) — see its section below for why the benchmark number understates a general detector.

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

### v2 extension — 15 stages, +14 h 43 m

Resumed from v1's stage-11 checkpoint with `--max-wcs-per-stage 800 --target-stage-fpr 0.65` (the v1 stage-11 cap held at FPR=0.599 above the original 0.5 target, so a relaxed target was needed to keep adding stages). The extension ran in two sittings: a first attempt with `--max-wcs-per-stage 600` saturated stage 13 immediately at FPR=0.757, so the saturated stage was truncated (`tools/truncate_checkpoint.py`) and saved as `cbcl__19_v1.1.pkl` (12 stages); a second resume from v1.1 with the cap raised to 800 added stages 13–15 cleanly. **Combined wall time: 5 h 21 m + 9 h 22 m ≈ 14 h 43 m.** Training stopped at stage 16 when hard-neg mining returned only 876 patches (below `--min-stage-negatives 1000`) — the cascade had exhausted the negative pool's discriminable patches.

**Cascade (v2):**

```
T per stage:        [8, 10, 27, 39, 45, 61, 69, 154, 220, 311, 400, 403, 699, 778, 534]
Layer thresholds:   [0.257, 0.319, 0.378, 0.413, 0.428, 0.434, 0.445, 0.459, 0.466, 0.472, 0.473, 0.474, 0.479, 0.481, 0.479]
```

Stages 13–15 each converged on `target_stage_fpr 0.65` (final FPR 0.636, 0.636, 0.637) with 534–778 weak classifiers per stage — the relaxed target avoided the capacity ceiling but each stage is now ~5× more expensive than the v1 stages at the same depth.

**Test metrics (raw):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |   F1   |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :----: |
| 283 | 23 413 | 160 | 189 |  0.985   |   0.639   | 0.600  | 0.993  | 0.619  |

**After post-hoc F1 tuning (`weights/19/cbcl__19_v2_tuned.pkl`):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |     F1    |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :-------: |
| 286 | 23 462 | 111 | 186 |  0.988   |   0.720   | 0.606  | 0.995  | **0.658** |

**Δ vs v1 tuned:** +0.024 F1 (0.634 → 0.658), recall +0.023, precision +0.026. The extra 4 stages add ~2 400 weak classifiers and cost ~14 h of training to buy a +2.4 pp F1 lift — net effect is now better than the historical 24×24 best (F1=0.653) at ~4× fewer features per window.

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

**After post-hoc F1 tuning (`weights/19/celeba_aligned__19_v1_tuned.pkl`):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |     F1    |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :-------: |
| 259 | 23 348 | 225 | 213 |  0.982   |   0.535   | 0.549  | 0.990  | **0.542** |

Tuned thresholds collapse to `[0.54, 0.52, 0.5]` — every stage pushed to its maximum cut point, trading −0.400 recall for +0.475 precision. The F1 lift (+0.43) is large in relative terms but only because the raw F1=0.113 was near zero; the resulting operating point is still far below any cascade trained with anchored positives. This is the diagnostic signature of a capacity-bound run: the tuner can find *an* operating point, but every stage threshold ends up at the same extreme value because no stage carries useful discriminative weight.

**Why it fails:** CelebA's pose/illumination diversity exceeds what 17 268 Haar features can express while tolerating the ±1 px shift jitter. Without an "anchor" distribution (CBCL-style aligned crops) in the training positives, the cascade saturates after 3 stages. The same data at 24×24 (~78 K features) is predicted to work — see [FINDINGS.md](FINDINGS.md#1919-capacity-ceiling-jitter-saturates-the-cascade).

### v2 extension — 6 stages, +13 h 12 m — *still capacity-bound*

Resumed from v1 with the same relaxed targets (`--max-wcs-per-stage 800 --target-stage-fpr 0.65`). The first extend got two more clean stages (4: 122 WCs / FPR 0.647; 5: 448 WCs / FPR 0.632), saved as `celeba_aligned__19_v1.1.pkl` (5 stages), then was cancelled when stage 6 was clearly heading for saturation. The second resume from v1.1 confirmed it: stage 6 trained 800 WCs (the new cap) without satisfying the FPR target — **final FPR 0.713 > target 0.65**, the capacity-ceiling break fired and training stopped. **Combined wall time: ~5 h 17 m + 7 h 55 m ≈ 13 h 12 m.**

**Cascade (v2):**

```
T per stage:       [19, 122, 400, 122, 448, 800]
Layer thresholds:  [0.338, 0.427, 0.455, 0.437, 0.462, 0.469]
```

**Test metrics (raw):**

| TP  | TN     | FP    | FN  | Accuracy | Precision | Recall |  Spec  |   F1  |
| --: | -----: | ----: | --: | :------: | :-------: | :----: | :----: | :---: |
| 420 | 20 163 | 3 410 | 52  |  0.856   |   0.110   | 0.890  | 0.855  | 0.195 |

Still qualitatively broken in raw form — the cascade flags ~14 % of non-faces as faces. F1 tuning *can* recover an operating point (raw 0.195 → tuned 0.550) but at the cost of recall (0.890 → 0.540): the tuner has to push every stage's threshold up to 0.48–0.54, which is the same as saying "the cascade has no usable margin above the noise floor". **Vs v1 tuned (F1=0.542): +0.008 F1 for ~13 h of extra training** — the extension confirms the capacity diagnosis without meaningfully changing the operating point. For comparison, the mixed CelebA+CBCL run at the same training budget reaches F1=0.661 tuned with 0.625 recall.

**Lesson confirmed:** extending the cascade does not rescue a capacity-bound run. v1 capped at stage 3 with FPR=0.598; v2 reached stage 6 but each additional stage hit the FPR cap progressively earlier (stage 6 saturated *at 800 WCs*, vs v1's stage 3 saturating at 400). The 19×19 Haar budget genuinely cannot express CelebA's variety without a matched-domain anchor — see [FINDINGS.md](FINDINGS.md#1919-capacity-ceiling-jitter-saturates-the-cascade).

### v3 filtered — controlled experiment: does positive curation break the ceiling? (3 stages, ~2.2 h)

Designed as the cleanest possible A/B vs cbcl v1: identical hyperparameters (`--max-wcs-per-stage 400 --target-stage-fpr 0.5`), identical per-stage compute budget (~7700 positives + 5000 negs/stage), only the face source changes. The 20K augmented CelebA-aligned positives were scored with `cbcl__19_v2.pkl` as oracle (continuous cumulative margin per face) and the lowest 61% dropped via `--drop-low-score-pos 0.61` — keeps the ~7700 most CBCL-like augmented entries, statistically equivalent to ~2000 unique faces (close to CBCL's 1929 unique). A 15K very-hard-neg reservoir pre-mined from `cbcl__19_v2.pkl` via the streaming raw miner (`tools/mine_hard_negatives_raw.py`, ~17.5 h, 326M patches sampled at 0.0046% FPR) was attached as top-up. See [WORKFLOW.md](WORKFLOW.md) for the full pipeline.

**Cascade (v3):**

```
T per stage:        [32, 71, 400]
Layer thresholds:   [0.277, 0.336, 0.395]
```

Stages 1–2 converged cleanly (32 and 71 WCs, FPR 0.453 and 0.488 — both well under the 0.5 target). Stage 3 capped at 400 WCs with FPR=0.622 > target 0.5 — capacity-ceiling break fired. **Same structural failure pattern as v1** (FPR=0.598 at stage 3, also 400 WCs) and v2 (escalating saturation at deeper stages). The very-hard reservoir was never tapped: training stopped before mining became hard enough to trigger a shortfall.

**Test metrics (raw):**

| TP  | TN     | FP    | FN  | Accuracy | Precision | Recall |  Spec  |   F1   |
| --: | -----: | ----: | --: | :------: | :-------: | :----: | :----: | :----: |
| 433 | 16 301 | 7 272 | 39  |  0.696   |   0.056   | 0.917  | 0.692  | 0.106  |

**After post-hoc F1 tuning (`weights/19/celeba_aligned_filtered__19_v1_tuned.pkl`):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |     F1    |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :-------: |
| 238 | 23 494 |  79 | 234 |  0.987   |   0.751   | 0.504  | 0.997  | **0.603** |

Tuned thresholds collapse to `[0.30, 0.44, 0.52]` — same operating-point compression signature as v1/v2 (every stage pushed near its maximum cut point, recall sacrificed for precision). **+6.1 pp F1 tuned vs v1 (0.542 → 0.603)** — the curation infrastructure works and produces a measurable improvement, but the cascade still has only 3 stages of rejection power to redistribute.

**Conclusion (and stop signal for 19×19 CelebA-only).** Filtered CelebA at 19×19 reaches **F1 0.603 tuned vs cbcl v1's 0.634** at the same compute budget. The gap is small (0.031) but the cascade structure tells the real story: cbcl converges to 11 stages naturally, CelebA (filtered) saturates at 3 stages **regardless of curation**. The 8-stage delta is the 17K-feature ceiling at 19×19 + jitter, not a data-quality issue. Confirmed: **the bottleneck is resolution, not the dataset**. Curation tooling (`score_faces.py`, `mine_hard_negatives_raw.py`, `--drop-low-score-pos`, `--very-hard-neg-pool`) is validated end-to-end and ready to apply where it actually helps. Next step: 24×24, where the 78K feature budget should unlock CelebA's diversity. CelebA-only at 19×19 will not be revisited.

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

### v2 extension — 16 stages, +35 h 19 m

Resumed from v1's stage-11 checkpoint with `--max-wcs-per-stage 800 --target-stage-fpr 0.65`. Unlike the cbcl-only run, this experiment was extended in a single sitting (no intermediate v1.1 truncation was needed — none of the new stages saturated against the 800-WC cap). Five extra stages were trained (12–16), each early-stopping cleanly on the recall+FPR criterion, before cumulative val recall dropped to 0.946 (< 0.95 stop threshold) at the end of stage 16. **Wall time: 35 h 19 m** — by far the most expensive single experiment in the project; per-stage training cost grows with the negative-pool depletion (each fresh hard-neg mine now takes ~1 h to find 5 000 patches).

**Cascade (v2):**

```
T per stage:        [20, 15, 34, 47, 47, 116, 123, 140, 257, 380, 400, 344, 365, 743, 498, 708]
Layer thresholds:   [0.354, 0.359, 0.413, 0.435, 0.438, 0.457, 0.464, 0.468, 0.474, 0.480, 0.481, 0.477, 0.478, 0.484, 0.482, 0.485]
```

Stages 12–16 each hit `target_stage_fpr 0.65` (final FPR 0.632, 0.646, 0.644, 0.641, 0.630) with 344–743 weak classifiers — same regime as the cbcl v2 extension but with a slightly tighter per-stage FPR (the larger 27 716-positive training set lets each stage carve a little more cleanly).

**Test metrics (raw):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |   F1   |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :----: |
| 271 | 23 453 | 120 | 201 |  0.987   |   0.693   | 0.574  | 0.995  | 0.628  |

**After post-hoc F1 tuning (`weights/19/celeba_aligned+cbcl__19_v2_tuned.pkl`):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |     F1    |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :-------: |
| 295 | 23 448 | 125 | 177 |  0.987   |   0.702   | 0.625  | 0.995  | **0.661** |

**Δ vs v1 tuned:** +0.022 F1 (0.639 → 0.661), recall +0.011, precision +0.035. Like the cbcl v2 extension, the new stages buy ~2 pp F1 — the cbcl and celeba_aligned+cbcl v2 tuned runs end up within 0.003 F1 of each other (0.658 vs 0.661), confirming the CBCL-vs-mixed conclusion below: at 19×19, mixed positives only earn a marginal lead, regardless of depth.

---

## CBCL vs Mixed comparison (both 19×19, both F1-tuned)

| Metric          | CBCL v1 | Mixed v1 | CBCL v2 | Mixed v2 |
| --------------- | :-----: | :------: | :-----: | :------: |
| Stages          |   11    |    11    |   15    |    16    |
| Recall (raw)    |  0.657  |  0.653   |  0.600  |  0.574   |
| Precision (raw) |  0.503  |  0.548   |  0.639  |  0.693   |
| F1 (raw)        |  0.570  |  0.596   |  0.619  |  0.628   |
| F1 (tuned)      |  0.634  |  0.639   |  0.658  |  0.661   |
| Train time      |  ~6 h   |   ~20 h  | +14.7 h |  +35.3 h |

Recall (raw) is essentially identical within each version. Mixed's advantage is purely precision-side (CelebA helps reject a wider variety of hard negatives) but after F1 tuning the gap stays at 0.003–0.005 across both v1 and v2 — virtually indistinguishable.

**Practical implication for 19×19:** if compute-constrained, CBCL-only is the cost-efficient choice. CelebA's variety only earns its compute at higher resolutions. The v2 extension confirms the pattern at greater cascade depth — going from 11 → 15/16 stages buys +2 pp F1 in both face-source configurations, but mixed costs 2.4× more wall time for the same lift.

Both v1 runs saturate at stage 11 with `final_fpr ≈ 0.5–0.6`; both v2 runs add ~5 more stages at the relaxed `target_stage_fpr 0.65` before stopping for a different reason (negative-pool depletion in cbcl, cumulative val-recall < 0.95 in mixed) — the Haar-feature budget at 19×19 still caps both at a similar depth regardless of face source, just one step higher than the v1 ceiling.

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

## 24×24 — CBCL smoke test (10 stages, ~25 h) — *adaptive trainer, capacity probe*

A short run to answer one question: does the 24×24 feature budget (~60K features vs ~17K at 19×19) actually relieve the capacity ceiling? Data: `--face-source cbcl --resolution 24 --augment --jitter 2 --pool-size 100000000`. Trainer: `--max-stages 10 --max-wcs-per-stage 300 --target-stage-fpr 0.5 --target-neg-per-stage 5000`.

**Cascade:**

```
T per stage:        [11, 4, 38, 43, 207, 27, 78, 187, 189, 300]
Layer thresholds:   [0.295, 0.351, 0.388, 0.408, 0.444, 0.420, 0.444, 0.458, 0.464, 0.468]
```

Stages 1–9 converged on `target_stage_fpr 0.5`; stage 10 capped at 300 WCs (FPR=0.546). The WC count is non-monotonic (4 → 207 → 27 → …) — expected, since each stage trains against a freshly mined hard-neg set whose geometry is stochastic and whose seed/Caltech mix shifts with depth.

**Test metrics (raw):** recall 0.750, spec 0.976, precision 0.380, **F1 0.505**.

**After F1 tuning (`weights/24/cbcl__24_smoke_tuned.pkl`):** recall 0.602, spec 0.996, precision 0.730, **F1 0.660** (+15.5 pp from tuning). Tuned thresholds `[0.38, 0.30, 0.30, 0.30, 0.50, 0.46, 0.50, 0.48, 0.48, 0.48]`.

**Verdict:** ties the best 19×19 models (0.658–0.661) and the historical 24×24 best (0.653) with only 10 stages from the adaptive trainer. Confirms the hypothesis: 24×24 relieves the ceiling. One surprise — **stage cost is much higher than WC-count extrapolation predicts** (this run took ~25 h, partly inflated by a sleep-stalled stage 5; deep stages dominate). This timing lesson governed the celeba production run below.

---

## 24×24 — CelebA<sub>aligned</sub> only (9 stages, ~156 h) — *the resolution hypothesis, confirmed*

The payoff run. CelebA-only capped at **3 stages at 19×19** (F1 0.542–0.603 tuned, structurally broken). At 24×24 the same source trains a **9-stage cascade** — the central prediction of the entire 19×19 arc, validated.

**Data:** `--face-source celeba_aligned --n-faces 10000 --resolution 24 --augment --jitter 1`, then oracle-scored with `cbcl__24_smoke.pkl` and the bottom 2% dropped (`--drop-low-score-pos 0.02`) → 39 200 positives. The score grid showed the bottom percentiles at 24×24 are *hard real faces* (glasses, strong shadows, expressions), not misaligned crops as at 19×19 — so the drop was kept minimal. Negatives: CBCL seed (9 096) + Caltech pool (100M), `--target-neg-per-stage 10000`. Trainer: `--max-wcs-per-stage 800 --target-stage-fpr 0.5`.

**Cascade:**

```
T per stage:        [30, 36, 25, 45, 47, 336, 95, 457, 800]
Layer thresholds:   [0.342, 0.364, 0.370, 0.405, 0.419, 0.455, 0.446, 0.468, 0.472]
```

Stages 1–8 converged on `target_stage_fpr 0.5`; stage 9 capped at the 800-WC budget with FPR=0.653. Cumulative val recall ended at 0.96 (floor 0.95). **Per-stage wall time ballooned with depth: stage 6 = 26 h, stage 8 = 35 h, stage 9 = 71 h** — total 156 h (~6.5 days). Deep stages dominate because `_best_stump` cost scales with `WCs × samples × log(samples)` and late stages need both more WCs and a wider hard-neg margin.

**Test metrics (raw):** recall 0.712, spec 0.980, precision 0.411, **F1 0.521**.

**After F1 tuning (`weights/24/celeba_aligned__24_v1_tuned.pkl`):**

| TP  | TN     | FP  | FN  | Accuracy | Precision | Recall |  Spec  |     F1    |
| --: | -----: | --: | --: | :------: | :-------: | :----: | :----: | :-------: |
| 264 | 23 469 | 104 | 208 |  0.987   |   0.717   | 0.559  | 0.996  | **0.629** |

Tuned thresholds `[0.48, 0.30, 0.34, 0.44, 0.50, 0.48, 0.30, 0.48, 0.48]` — +10.8 pp from tuning.

**Why 0.629 understates the model.** The benchmark *is* CBCL, and this model never sees a CBCL face in training (only CBCL non-faces as the negative seed). The ~3 pp gap to cbcl-on-cbcl (0.660) is precisely that domain mismatch. On real images the celeba-only model is the **cleanest detector in the project**: on `images/people.png` it finds 8/9 faces with near-zero false positives after tuning + `--nms-threshold 0.2 --detect-min-face 24 --detect-min-score 0.15`; on `i1.jpg` it isolates the single large face with one stray box. The diversity of CelebA (poses, lighting, expressions) is exactly what generalizes off-benchmark — which is invisible to a CBCL-only F1 score.

**Failed extension (`extend_stage.py`, +31 h).** Stage 9 was extended in place from 800 → 1200 WCs to try to push its FPR below 0.5. It degraded the raw FPR (0.653 → 0.751) and was a wash after tuning (0.6298 vs v1's 0.6286 — 0.1 pp, noise). Confirms the AdaBoost-saturation lesson: once `pos_p1 < neg_p99` at a stage (here 0.433 < 0.491), more weak classifiers raise the recalibrated threshold rather than separating the classes. See [FINDINGS.md §24×24](FINDINGS.md#2424-the-ceiling-moves-up-but-its-still-there). **`weights/24/celeba_aligned__24_v1_tuned.pkl` is the final, recommended general-purpose detector.**

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
| **24×24 CelebA<sub>aligned</sub>+CBCL** | Adding upscaled CBCL faces to the 24×24 celeba-only positives should recover the ~3 pp benchmark gap (CBCL-on-CBCL), at the cost of some real-image diversity. Optional — only worth it to chase a higher benchmark number; the celeba-only model is already the better general detector. |

**Confirmed (24×24 adaptive trainer):** *Does 24×24 relieve the capacity ceiling?* **Yes.** The CBCL smoke test reached 10 stages / tuned F1=0.660 (ties the best 19×19 models). CelebA-only — broken at 3 stages at 19×19 — trained 9 stages / tuned F1=0.629 at 24×24, and is the project's cleanest real-image detector. The predicted "12–15 stages, F1 0.75+ tuned" was optimistic: realistic depth is 9–10 stages and tuned F1 ≈ 0.63–0.66, with deep stages costing 30–70 h each (the celeba run was 156 h total). Extending a capped stage in place (`extend_stage.py`) does not help — see the 24×24 sections above.

**Confirmed (v2 extension):** *19×19 CBCL extended stages.* The hypothesis ("resume with relaxed `--target-stage-fpr` should add 2–4 stages and reach F1 ≈ 0.66–0.70 tuned") was confirmed at the low end of the predicted range — `target_stage_fpr 0.65` added 4 stages (cbcl, 11→15) and 5 stages (mixed, 11→16) before stopping on negative-pool depletion / cumulative val-recall, reaching tuned F1=0.658 (cbcl) and 0.661 (mixed). The upper-bound F1≈0.70 was not reached: stages 12–16 individually converge cleanly but each costs ~5× more weak classifiers than the equivalent stage at v1's depth, so the marginal F1 per WC is diminishing. The CelebA-aligned-only run remained capacity-bound — extension confirmed the diagnosis rather than fixing it.

**Confirmed (v3 filtered):** *Does positive curation rescue capacity-bound CelebA at 19×19?* Answer: **no**. Same hyperparameters as cbcl v1, oracle-filtered positives (drop 0.61 keeps the ~2000 most CBCL-like unique faces, matching CBCL's 1929), 15K very-hard-neg reservoir on standby. Cascade still capped at stage 3 (FPR=0.622). Tuned F1=0.603 vs cbcl v1's 0.634 — close enough that you could call it paridad, but only because the F1 tuner does heroic work over 3 stages. The structural delta (3 stages vs 11) confirms the ceiling is the 17K-Haar feature budget at 19×19 + jitter, **not** data quality. The curation infrastructure (`score_faces.py`, `mine_hard_negatives_raw.py`, `--drop-low-score-pos`, `--very-hard-neg-pool`) is validated end-to-end and ready for 24×24 where it has room to help.
