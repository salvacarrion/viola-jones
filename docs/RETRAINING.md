# Retraining guide

The repo is archived with the models it has. This document is for whoever retrains: what limits the current models, what to change (ranked by expected impact), where in the code, and how to evaluate the result so the number means something. The measurements behind it are in [OPENCV_COMPARISON_FINDINGS.md](OPENCV_COMPARISON_FINDINGS.md).

## Where the current models stand

After the two inference/evaluation fixes (image pyramid and face-box mapping, see [OPENCV_COMPARISON_FINDINGS.md](OPENCV_COMPARISON_FINDINGS.md#what-was-wrong-three-causes-measured)), the best model finds faces almost as often as OpenCV on FDDB (recall at IoU 0.3 within a few points) but lets far more background through. That residual gap is a property of the trained cascade and needs retraining:

| FDDB fold 1, min-face 40, image pyramid | Ours (`celeba_aligned__24_v2_s11_tuned`) | OpenCV `default` (native port) |
| :-- | :-: | :-: |
| Windows evaluated | 30.3 M | 15.5 M |
| Background windows passing the cascade, per window | 4.5e-4 | 2.5e-5 |
| Background windows passing, per image (before NMS) | 47 | 1.3 |
| Stages / stumps | 11 / 3818 | 25 / 2913 |

The per-stage rejection on real FDDB windows shows where it goes wrong. Ours rejects 98% at stage 1 and then becomes erratic; the three deepest stages hold 72% of all stumps (2747) and reject only 4%, 14% and 9% of what reaches them. OpenCV keeps rejecting 30 to 50% per stage through stage 11 with fewer than 100 stumps per stage:

| Stage | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Ours: stumps | 30 | 36 | 25 | 45 | 47 | 336 | 95 | 457 | 522 | 1025 | 1200 |
| Ours: rejected | 98% | 64% | 8% | 32% | 15% | 54% | 11% | 30% | 4% | 14% | 9% |
| OpenCV: stumps | 9 | 16 | 27 | 32 | 52 | 53 | 62 | 72 | 83 | 91 | 99 |
| OpenCV: rejected | 58% | 51% | 47% | 43% | 43% | 38% | 32% | 33% | 30% | 39% | 29% |

Both tables come from `python tools/diagnose_fddb_windows.py --folds 1 --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl weights/24/opencv_default.pkl`. The deep stages learned to reject the hard negatives they were trained on (Caltech-256 object photos and CBCL non-faces) and those do not look like the background of photos with people in them (clothes, hands, hair, crowds, text). Fold 2 gives the same numbers to within a point.

## How much could it improve?

An estimate, not a measurement: nothing was retrained. Starting from the best model's detections on FDDB folds 2-10 (face boxes), we randomly removed a fraction of its false positives while keeping every true positive (mean of 5 seeds), which is what better negatives would do in the ideal case:

| Scenario (FDDB folds 2-10) | AP@0.3 | AP@0.5 | precision@0.5 |
| :-- | :-: | :-: | :-: |
| Today (best model, after the inference fixes) | 0.589 | 0.471 | 0.25 |
| False positives ÷ 2 | 0.638 | 0.520 | 0.40 |
| False positives ÷ 4 | 0.671 | 0.552 | 0.57 |
| False positives ÷ 8 (≈ OpenCV's precision) | 0.692 | 0.571 | 0.72 |
| OpenCV `default`, same protocol | 0.737 | 0.726 | 0.76 |

Two readings:

- **Better negatives (item 1) are worth up to ~+0.10 AP@0.3**, which brings the model within ~0.05 of OpenCV at IoU 0.3. Matching OpenCV's precision needs ~8× fewer false positives (we return 8334 at IoU 0.5 vs ~1100 for OpenCV).
- **At IoU 0.5 that is not enough** (0.57 vs 0.73), because our boxes are also less accurate: 72% of faces are found at IoU 0.3 but only 60% at IoU 0.5, while OpenCV loses one point (75% vs 74%). That is the box-geometry cost of training on tight crops; positives with context (item 2) target it. If both were fixed, AP@0.5 would approach the AP@0.3 column, i.e. roughly 0.65 to 0.70.

So a full retrain along items 1 to 3 could plausibly take the best model from AP@0.5 0.47 to the 0.6 to 0.7 range, close to OpenCV. OpenCV's cascade is a fair estimate of the ceiling for this family of detectors (Haar features + boosted cascade); going clearly beyond it requires a different kind of detector, not a better-trained Viola-Jones.

## What to change, ranked

The ranking is by expected impact on in-the-wild detection. Items 1 to 3 address problems measured in this repo; items 4 and 5 are standard differences with OpenCV's trainer. None of the gains below has been measured, because nothing was retrained after the diagnosis.

### 1. Mine negatives from real scenes, at every pyramid scale

This is the main lever. Today every stage mines 24×24 patches from pre-extracted pools (`caltech_pool.npy` plus the CBCL seed, see `ViolaJones._mine_hard_negatives` / `_mine_from_pool` in [violajones.py](../violajones.py)). Instead, run the partial cascade exactly as it runs at inference (`find_faces(..., pyramid="image")`) over full face-free images and keep every window that passes as a negative for the next stage. In image-pyramid mode a surviving window is already a 24×24 crop of the resized image, so it can be sliced out of `small` with no extra resampling.

- **Sources.** Images in the domain you will detect on, without faces: COCO images with no `person` annotation, Places365 / SUN scenes, and, to get the hardest negatives (bodies, clothes, hands), WIDER FACE train images with every window that overlaps an annotated face (IoU > 0.1) excluded. WIDER does not annotate every tiny or blurred face, so only take windows ≥ 40 px from it. Never use FDDB: it is the benchmark.
- **Code.** [tools/mine_hard_negatives_raw.py](../tools/mine_hard_negatives_raw.py) already streams raw images and keeps what a cascade misclassifies; the change is to scan whole images over the scale pyramid instead of sampling random patches, and to call it once per stage from the training loop.
- **Stop criterion.** Measure the per-window FPR on a held-out set of face-free scenes after each stage and stop when it plateaus, instead of relying only on the per-stage FPR over the mined set.

### 2. Train on positives with context around the face

All models here were trained on tight eyes-to-mouth crops (CBCL framing, which CelebA was aligned to). That throws away the head outline, the forehead/hair contrast and the cheek/background edge, which are among the most discriminative cues in real photos, and it is why the detector boxes need a ~1.4 × 2.1 rescale to match FDDB (up to 1.6 × 2.5 for the CBCL-only models). OpenCV's cascades were trained with the face plus a margin.

- **What.** Crop roughly 1.2 to 1.3× the face box (forehead to chin, ear to ear) before downsizing to 24×24. After training, `box_transform` should come out close to identity.
- **Code.** The crop geometry lives in [tools/dataset_build/build_dataset.py](../tools/dataset_build/build_dataset.py) (`EYE_TO_MOUTH_TO_FACE_H`, `FACE_CENTER_OFFSET`, `MARGIN`) and [tools/dataset_build/align_faces.py](../tools/dataset_build/align_faces.py). The HF dataset only stores the tight 48×48 crops, so this needs a rebuild from raw CelebA (`img_align_celeba` + `list_landmarks_align_celeba.txt`, re-downloadable from the official CelebA page).
- **Consequence.** CBCL faces are native 19×19 tight crops and cannot be re-cropped with margin, so they stop being usable as positives, and the CBCL patch benchmark stops being a fair test of the new models (OpenCV scores 0 on it for exactly this reason). Use FDDB for model selection (item 3).

### 3. Select and tune on a scene-level metric, not CBCL F1

Every decision in this project (stage calibration, early stopping, `tools/tune_thresholds.py`, choosing the final model) maximised F1 on CBCL patches. That metric never sees a full image, never sees the pyramid, and its negatives are curated face-like patches, so it rewarded models that flood real photos with false positives and it hid the pyramid bug for the whole project (patch-level `classify` only runs at scale 1).

- Tune per-stage thresholds against FDDB AP on the fit fold (fold 1), or against recall at a fixed per-window FPR on face-free scenes.
- Report the final number on FDDB folds 2 to 10 only, with the box transform fitted on fold 1 (the protocol in "Evaluate" below).

### 4. Gentle AdaBoost with real-valued stumps

Our stumps vote 0/1 and the stage sums `α·h(x)`; OpenCV's stumps output one real value per side of the split (for example +2.09 / −2.22 in its first stage). With the same feature, a real-valued stump carries more information, which is why OpenCV's first stage needs 9 stumps and ours 30, and why its stages stay small. The sort + prefix-sum search in `AdaBoost._best_stump` ([adaboost.py](../adaboost.py)) already computes the weighted label sums on each side of every split; the Gentle AdaBoost leaf values are those weighted means. [weakclassifier.py](../weakclassifier.py) and `ViolaJones._run_cascade` then add the leaf value instead of `α·[pred]`.

### 5. Make the three-rectangle features zero-sum

In `build_features` ([utils.py](../utils.py)) a three-rectangle feature is `centre − (left + right)` with three equal areas, so its weights sum to −1 and every value carries a `−area · mean` term: the stump partly thresholds the window's brightness-to-contrast ratio instead of its structure. OpenCV weights the centre ×3 against the full ×1, which sums to zero. The fix is to count the centre twice, `HaarFeature([immediate, right_2], [right, right])` (same for the vertical variant). This changes the feature values, so it only applies to new training runs; 343 of the 3818 stumps in the best model are three-rectangle features.

### 6. Keep inference geometry identical to training geometry

Already done, keep it that way: training patches are downsized with PIL bilinear (`prepare_data.py`) and the default `pyramid="image"` downsizes the image with the same filter and evaluates the native 24×24 window. The legacy `pyramid="features"` mode scales the rectangles instead and lets ~3.5× more background through, because the truncated rectangles of one feature end up with unequal areas. If you change the resize filter in one place, change it in both.

## What not to spend time on

These were tried and measured; they move the operating point, not the ceiling:

- More weak classifiers in a saturated stage (`tools/extend_stage.py`): FPR got worse, see [FINDINGS.md](FINDINGS.md#2424-the-ceiling-moves-up-but-its-still-there).
- Deeper cascades on the same negatives: stage 12 of the best model rejected 6% of its own training negatives with 1600 stumps.
- `--detect-min-score` and OpenCV-style `minNeighbors` grouping: they only thin the score-ranked list, trading recall for precision. A score cut cannot raise AP by construction, and requiring 2 to 4 fused windows per detection lowered FDDB AP by 0.005 to 0.015 in the post-mortem.
- `--detect-min-face 80`: it looked like a 4× AP win under the old raw-box evaluation, but only because bigger windows overlap FDDB's tall boxes more. With face boxes it lowers FDDB AP (folds 2-10: 0.59 → 0.43 at IoU 0.3, 0.47 → 0.32 at IoU 0.5) by dropping every face under ~110 px.
- Positive curation at 19×19: +6 pp CBCL F1, no structural change.

## Cost

The trainer is pure NumPy: the 24×24 CelebA model took ~156 h for 9 stages plus ~65 h for the deepening. Always use `--precompute-sort-index` (≈5× faster rounds). Item 1 makes mining more expensive (full-image scans per stage) and items 4 and 5 should make stages shorter. For a quick sanity check of a new data recipe, `opencv_traincascade` (C++, multi-threaded) typically trains a 20-stage cascade from the same positives and negatives in hours, and if trained with stumps (`-maxDepth 1`) and upright features (`-mode BASIC` or `CORE`) its XML loads here with `tools/convert_opencv_cascade.py --xml <file>`.

## Evaluate a new model

```bash
# 1. Patch-level sanity (only meaningful for tight-crop models, see item 2)
python main.py test --weights-path weights/24/<model>.pkl --data-dir data/24_<source>

# 2. Fit the face-box mapping on FDDB fold 1 and store it in the .pkl
python tools/eval_fddb.py --weights weights/24/<model>.pkl --skip-opencv \
    --folds 1 --box-fit-folds 1 --save-box-transform

# 3. Headline: held-out folds 2-10, against OpenCV under the same protocol
python tools/eval_fddb.py --weights weights/24/<model>.pkl --cascade default \
    --box-fit-folds 1 --folds 2,3,4,5,6,7,8,9,10 --iou 0.3,0.5

# 4. Visual check
python main.py detect --weights-path weights/24/<model>.pkl --detect-output images/outputs/<model>
```

Compare the `ours+box` rows against `opencv:default+box`. The raw `ours` rows only tell you how far the training crop is from FDDB's face box.
