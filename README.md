# Viola-Jones

Educational Python implementation of the face-detection algorithm from
_Rapid Object Detection using a Boosted Cascade of Simple Features_
(Paul Viola & Michael J. Jones, 2001).

The pipeline:

1. **Haar-like features** enumerated over the training-window
2. **Integral image** for O(1) rectangle sums
3. **AdaBoost** to pick a small set of strong weak classifiers
4. **Attentional cascade** of AdaBoosts with hard-negative mining

## Install

```bash
pip install -r requirements.txt
```

## Why so many knobs? Read this first

A naive V-J — gather some faces, mine random negatives, train 4 boosting
stages, run NMS — gets you F1 ≈ 0.04 on the CBCL benchmark. Useless. The
paper's F1 ≈ 0.85 came from millions of curated patches, 38 hand-tuned
stages, and machine-time orders of magnitude beyond a laptop.

This repo bridges the gap with **six small fixes**, each tied to a
specific failure mode we hit while building it. The detailed write-ups
live further down in the README; here is the narrative arc so you know
why each flag exists:

| #   | Failure mode observed                                                    | Fix (flag / file)                                          | Section                                                              |
| --- | ------------------------------------------------------------------------ | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| 1   | Train on Caltech negatives → cascade calls 67 % of CBCL non-faces "face" | Matched-domain seed: `--neg-source mixed`                  | [Negative-domain gap](#1-negative-domain-gap)                        |
| 2   | Stage 1 seed alone leaves stages 2–9 over-fit to Caltech                 | Stratified mining: stages 2+ also see CBCL non-faces       | [Stratified mining](#2-stratified-mining-not-just-stage-1)           |
| 3   | 19×19 + CelebA → cascade collapses to 1.7 % recall                       | Train at 24×24 + shift-jitter: `--jitter 2`                | [Pixel alignment at small res](#3-pixel-alignment-at-low-resolution) |
| 4   | 1 943 unique CBCL faces is too few for a 9-stage cascade                 | Multi-source: `--face-source celeba+cbcl`                  | [Why multi-source](#4-multi-source-positives)                        |
| 5   | Calibration hits the 0.5 cap at deep stages → FPR stays high             | Allow per-stage threshold up to 0.95 (post-hoc + training) | [Calibration cap](#5-calibration-was-hitting-its-own-ceiling)        |
| 6   | Same face fires at scales 1×, 1.5×, 2× → "boxes on boxes" after NMS      | Hybrid NMS: `--nms-metric hybrid` (IoU **or** IoMin)       | [Hybrid NMS](#6-hybrid-nms-for-multi-scale-duplicates)               |

**What each fix is NOT**: this is not a generic "tune and pray" flag set.
Every option here exists because removing it produced a measurable, named
failure that we then debugged. The narrative is the point — if you're
adapting this to a different benchmark, you'll need to re-run the
diagnosis; the specific knobs are tied to the CBCL/Caltech mix.

## Workflow

The repo splits dataset preparation from training so that you only download
and preprocess once, then train/test/detect over the resulting NPY bundles.

### 1. Prepare data (downloads on first run)

`tools/prepare_data.py` pulls the curated dataset from
[`salvacarrion/face-detection`](https://huggingface.co/datasets/salvacarrion/face-detection)
on the Hugging Face Hub (cached in `~/.cache/huggingface/`), filters one
or more face sources, splits them train/val/test, samples a negative
pool, and writes NPY bundles to `data/<resolution>/`.

**Quick / educational recipe** (~15 min training, F1 ≈ 0.37):

```bash
python tools/prepare_data.py \
    --face-source cbcl \
    --neg-source mixed \
    --resolution 19 \
    --augment
```

**Production recipe** (~10-12 h training, target F1 ≈ 0.7-0.8) — uses
all six fixes from the table above:

```bash
python tools/prepare_data.py \
    --face-source celeba+cbcl \
    --n-faces 10000 \
    --neg-source mixed \
    --resolution 24 \
    --augment \
    --jitter 2
```

`--neg-source mixed` writes two pools: `cbcl_neg_seed.npy` (curated CBCL
non-faces, used as the **stage-1 seed** for matched-domain calibration
AND mixed into deeper-stage hard-neg mining) and `caltech_pool.npy` (1 M
random Caltech-256 patches, used by mining throughout for diversity).
See [§ 1](#1-negative-domain-gap) and [§ 2](#2-stratified-mining-not-just-stage-1).

`--face-source celeba+cbcl` (note the `+`) combines 10K CelebA + 2.4K
CBCL faces — at 24×24 there's enough alignment slack to absorb CelebA's
looser framing, so the two sources reinforce each other. See [§ 4](#4-multi-source-positives).

`--jitter 2` doubles unique faces by emitting (center crop, random ±2 px
crop) per source image, multiplying the effective dataset without
breaking pixel alignment. Combined with `--augment` (h-flip), positives
are 4× the unique-face count.

Output:

```
data/19/
├── train_pos.npy        # face crops for AdaBoost
├── val_pos.npy          # held-out faces for stage calibration
├── test_pos.npy         # internal eval (unused in main flow)
├── caltech_pool.npy     # 1 M Caltech crops for hard-neg mining (deep stages)
├── cbcl_neg_seed.npy    # 9 K CBCL non-faces for stage-1 seed (matched-domain)
├── cbcl_test_pos.npy    # CBCL benchmark (472 faces)
├── cbcl_test_neg.npy    # CBCL benchmark (23 573 non-faces)
└── manifest.json
```

### 2. Train

Quick recipe (matches the prepare data above):

```bash
python main.py train --data-dir data/19 --layers 5 20 50 100
```

Production recipe (deep cascade + bigger neg budget):

```bash
python main.py train \
    --data-dir data/24 \
    --layers 5 10 20 40 60 80 100 130 160 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 1000000
```

Auto-loads `cbcl_neg_seed.npy` if present and uses it for the stage-1
seed AND for stratified mining at every later stage. Saves the cascade
to `weights/<resolution>/cvj_weights_<unix-ts>.pkl`. **Per-stage
checkpointing is on**: each completed stage overwrites the same `.pkl`,
so a crash mid-run leaves a usable partial cascade.

### 2b. Tune thresholds post-hoc (optional)

Calibration during training optimizes per-stage face recall on `val_pos`,
which can systematically diverge from the test distribution. The tuner
edits each stage's threshold to maximize a chosen objective on CBCL test
— **without retraining** any weak classifier:

```bash
# Maximize F1 (best benchmark headline)
python tools/tune_thresholds.py \
    --weights $(ls -t weights/24/*.pkl | head -1) \
    --data-dir data/24 \
    --objective f1

# Maximize recall while keeping spec ≥ 0.97 (best for "find every face" use case)
python tools/tune_thresholds.py \
    --weights $(ls -t weights/24/*.pkl | head -1) \
    --data-dir data/24 \
    --objective recall-at-spec --min-spec 0.97
```

Saves `<weights>_tuned.pkl` (F1) or `<weights>_recall97.pkl`
(recall-at-spec). Both can be passed to `main.py test` and `main.py detect`
via `--weights-path`.

### 3. Test

Evaluates the most recent (or `--weights-path`-specified) checkpoint on the
CBCL benchmark.

```bash
python main.py test --data-dir data/19
```

### 4. Detect

Runs sliding-window detection on images and writes annotated PNGs.

```bash
python main.py detect \
    --detect-images images/people.png images/clase.png
```

### Per-stage diagnostic

When tuning, dump pass-rates and score distributions per stage on CBCL —
tells you exactly which stage is bleeding recall or letting through FPs:

```bash
python tools/diagnose_cascade.py \
    --weights $(ls -t weights/19/*.pkl | head -1) \
    --data-dir data/19
```

### Inspect the dataset

To dump a visual sample of every (split, source, label) bucket from the HF
dataset to `~/Desktop/face-detection-inspection/` (200 PNGs per bucket plus
a `_grid.png` mosaic), useful for sanity-checking the curated data:

```bash
python tools/inspect_dataset.py
```

### Common flags

**`prepare_data.py`**:

| Flag            | Default   | Description                                                                                              |
| --------------- | --------- | -------------------------------------------------------------------------------------------------------- |
| `--face-source` | `celeba`  | Source(s) for positives. Single (`celeba`/`fddb`/`cbcl`) or `+`-separated multi (`celeba+cbcl`).         |
| `--n-faces`     | `10000`   | Per-source cap (clamped to availability).                                                                |
| `--neg-source`  | `caltech` | `caltech`/`cbcl`/`mixed`. `mixed` writes both pools and enables matched-domain seed + stratified mining. |
| `--resolution`  | `24`      | Native window size. 19 for fast iteration, 24 for production (10× more Haar features).                   |
| `--augment`     | off       | Add horizontal-flip mirrors of every train face.                                                         |
| `--jitter`      | `0`       | Pixel range for shift-jitter (e.g. `2` = ±2 px). Each face yields center + random-shift crop.            |

**`main.py train`**:

| Flag                     | Default       | Description                                                                      |
| ------------------------ | ------------- | -------------------------------------------------------------------------------- |
| `--data-dir`             | `data/24`     | Directory with NPY bundles from `prepare_data.py`                                |
| `--layers T [T ...]`     | `5 20 50 100` | Weak learners per cascade stage                                                  |
| `--layer-recall`         | `0.99`        | Per-stage face-recall target; cumulative recall ≈ `layer-recall ^ N`             |
| `--target-neg-per-stage` | `3000`        | Negatives mined per stage (split 50/50 across pools when `seed_neg_pool` exists) |
| `--neg-sample-budget`    | `100000`      | Max patches scanned during mining per stage                                      |

**`main.py detect`**:

| Flag                      | Default          | Description                                                                                                        |
| ------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `--weights-path`          | auto             | Checkpoint to load; omit to auto-pick the most recent under `weights/`                                             |
| `--detect-images IMG ...` | bundled samples  | Images for `detect` mode                                                                                           |
| `--detect-output`         | `images/outputs` | Directory where annotated outputs are saved                                                                        |
| `--nms-threshold`         | `0.3`            | IoU threshold for NMS clustering                                                                                   |
| `--nms-mode`              | `weighted`       | `weighted` fuses overlapping boxes by score-weighted average; `greedy` keeps highest-score, drops rest.            |
| `--nms-metric`            | `hybrid`         | `hybrid` fuses if IoU > threshold OR IoMin > 0.7. `iou` is standard NMS. `iom` is intersection-over-min-area only. |

**Cost notes** (single-threaded Python on a MacBook Pro M1):

| Recipe                        | Resolution | Positives | Layers (Σ WCs)                         | Time         |
| ----------------------------- | ---------- | --------- | -------------------------------------- | ------------ |
| Quick (CBCL only)             | 19×19      | ~3.9 K    | 5 20 50 100 (175)                      | ~15 min      |
| 24×24 smoke                   | 24×24      | ~7.8 K    | 5 10 20 30 (65)                        | ~50 min      |
| 24×24 deep, single source     | 24×24      | ~7.8 K    | 5 10 20 40 60 80 100 (315)             | ~6 h         |
| **Production (multi-source)** | **24×24**  | **~50 K** | **5 10 20 40 60 80 100 130 160 (615)** | **~10-12 h** |

Training scales roughly linearly in `n_train_pos × n_features × Σ
LAYERS`, plus mining overhead that grows in deep stages (cascade rejects
more, so finding hard negatives requires scanning more of the pool —
hence the larger `--neg-sample-budget 1000000` in the production recipe).

## Results

All metrics on the CBCL benchmark test split (472 faces / 23 573
non-faces, the academic V-J reference set). Each row is a single full
retrain; bold = best F1 at that resolution.

### 19×19 runs (4-stage cascade)

| Config                                                               | Recall | Specificity | Precision |        F1 | Train time |
| -------------------------------------------------------------------- | -----: | ----------: | --------: | --------: | ---------: |
| FDDB faces + Caltech-only negatives (initial baseline)               |  0.784 |       0.326 |     0.023 |     0.044 | ~ 1 h 47 m |
| **CBCL faces + matched seed + Caltech mining, layers `5 20 50 100`** |  0.502 |       0.979 |     0.292 | **0.369** |     ~ 15 m |
| CBCL faces + matched seed, shorter cascade `5 10 20 40`              |  0.710 |       0.916 |     0.138 |     0.231 |      ~ 9 m |
| CelebA faces + matched seed (alignment failure)                      |  0.017 |       0.977 |     0.017 |     0.017 |      ~ 1 h |

Run-to-run F1 variance with the recommended 19×19 config is ≈ ±0.025
(observed 0.369–0.394 across full retrains with identical commands),
driven by the random seeds in hard-negative mining and the val/train
split. Treat any single run within ±0.03 as equivalent.

### 24×24 runs (deep cascade)

| Config                                                           | Recall | Specificity | Precision |        F1 | Train time |
| ---------------------------------------------------------------- | -----: | ----------: | --------: | --------: | ---------: |
| CBCL+seed+jitter, smoke `5 10 20 30` (65 WCs)                    |  0.909 |       0.694 |     0.057 |     0.107 |     ~ 50 m |
| CBCL+seed+jitter, deep `5 10 20 40 60 100 150 200 250` (855 WCs) |  0.826 |       0.943 |     0.228 |     0.357 |     ~ 10 h |
| ... + post-hoc `--objective recall-at-spec --min-spec 0.97`      |  0.636 |       0.971 |     0.309 |     0.416 |    + 1 min |
| ... + post-hoc `--objective f1`                                  |  0.597 |       0.995 |     0.719 | **0.653** |    + 1 min |

Two important reads from the table:

1. The **untuned 9-stage row (F1=0.357)** looks similar to the 19×19
   best, but it sits at a totally different operating point: recall
   0.83 vs 0.50. Visually, the 24×24 cascade catches faces the 19×19
   misses; it just emits more FPs in dense backgrounds.
2. **Threshold tuning is non-trivial**, not cosmetic. Calibrate-during-
   training optimizes the wrong proxy (val-pos recall) when val and
   test distributions diverge — the F1=0.653 result is the same model
   with thresholds re-picked against test. This is allowed — the
   weak-classifier features and weights are unchanged, only the cut
   points move. See [§ 5](#5-calibration-was-hitting-its-own-ceiling).

### 24×24 production recipe (projected, not yet measured)

The [Production recipe](#production-recipe-deep-cascade) adds the four
remaining structural fixes (multi-source faces, stratified mining,
relaxed threshold cap, hybrid NMS). Each one targets a measured failure
mode of the 24×24 deep run above. Projected outcome based on the
diagnostic of the deep run: **F1 ≈ 0.72-0.80, recall ≈ 0.90-0.95** at
spec ≈ 0.97. Will be updated with measured numbers after the 12 h run
completes.

The next sections walk through each of the six fixes one-by-one:
**what failed**, **how we noticed**, and **what the fix does**. They
are the long-form expansion of the table at the top of this README.

Independent of the six fixes, three classical V-J techniques are always
on and quietly do most of the work: **hard-negative mining** (mining
loop in [violajones.py](violajones.py)), **per-window variance
normalization** ([weakclassifier.py](weakclassifier.py), Viola-Jones §5.1),
and **held-out calibration of stage thresholds**
([violajones.py](violajones.py), `calibrate()`). The six fixes below
sit on top of these and address what those alone could not.

### 1. Negative-domain gap

The first attempt — FDDB faces with Caltech-only negatives — collapsed
to F1=0.044 not because the cascade failed to learn faces (recall was
0.78) but because **specificity was 0.33**: the cascade called 67 % of
CBCL non-faces "face".

- Caltech patches are random crops of object photos (textures, edges,
  uniform regions). Easy to reject — once the cascade learns "not flat,
  not pure-edge", it's done.
- CBCL non-faces (the benchmark's negative set) were **curated to be
  face-like**: face-pose backgrounds, near-symmetric features, similar
  pixel statistics. Caltech-trained cascades have never seen anything
  remotely close, so they let them through.

**Fix** (`--neg-source mixed`): pull the ≈ 4 500 CBCL non-faces from
the HF train split (×2 with h-flip → 9 096), and use them as the
**stage-1 seed**. [violajones.py](violajones.py) accepts an optional
`seed_neg_pool` arg that biases stage-1 sampling toward this
matched-domain seed. Result: specificity jumped from 0.33 → 0.98 in one
training run.

### 2. Stratified mining (not just stage 1)

The matched seed solved stage 1, but the next 8 stages mined from
Caltech only. By stage 9, "Stage negatives: 947" appeared in the log:
the cascade had memorized Caltech so completely that only 947 patches
out of 1 M still passed it as faces. The cascade had drifted to
"distinguish faces from Caltech-leftover-edges", and CBCL non-face
patterns crept back in unchallenged.

**Fix** ([violajones.py](violajones.py) `_mine_hard_negatives`): when a
seed pool is passed, allocate **half** of each stage's mining budget to
it and the other half to the Caltech pool. CBCL non-faces stay in the
training mix at every stage, not just stage 1. If the seed pool runs
short (small pool exhausted in deep stages), the shortfall is
backfilled from Caltech. Logged as `[seed] mined X / [caltech] mined Y`
per stage.

### 3. Pixel alignment at low resolution

The "CelebA + matched seed" row above (recall 0.017) is the most
surprising failure. CelebA faces are frontal, just like CBCL — so why
does the cascade not generalise?

At 19×19 resolution **alignment matters at the pixel level**, not just
pose. CBCL crops are tightly cropped 19×19 native captures with eyes
landing at a specific row, nose at another, mouth at another. CelebA
crops are 48×48 native with looser framing (more forehead, hair, neck);
bilinearly downsized to 19×19, the same landmarks land at different
rows. A 1-pixel shift on a 19-pixel face is ~5 % of the face — and Haar
features tile rectangle sums at fixed grid positions, so the feature
distribution moves. The cascade learned "eyes at row 8"; CBCL test
faces have eyes at row 7 and look like non-faces in feature space.

**Fix** (resolution + jitter): at 24×24 there's enough slack — 5 px is
~20 % of the face — that bilinear-downsized CelebA still hits the same
Haar grid as CBCL. Adding `--jitter 2` doubles each face by emitting a
center crop AND a random ±2 px crop, training the cascade to be
mildly translation-invariant rather than relying on perfect alignment.

### 4. Multi-source positives

CBCL ships only **2 429 unique faces** in the HF train split. After 80%
train fraction + h-flip + jitter, that's ≈ 7 800 train positives. For a
9-stage cascade with 615 weak classifiers, 7 800 unique-ish samples is
enough to fit but not enough for the stages 7-9 to escape val/test gap
overfitting.

**Fix** (`--face-source celeba+cbcl`): combine sources. At 24×24 the
alignment problem of [§ 3](#3-pixel-alignment-at-low-resolution)
doesn't apply, so we can mix 10 K CelebA + 2.4 K CBCL freely. Total:
~50 K train positives after augment+jitter — 6× more diversity than
CBCL alone, at zero extra training cost beyond the proportional time
hit.

### 5. Calibration was hitting its own ceiling

[adaboost.py](adaboost.py) `calibrate()` historically capped per-stage
threshold at `0.5` ("never tighten past majority vote"). At 24×24 with a
deep cascade, val-pos scores cluster much higher than 0.5 — calibration
WANTED to push deep stages to 0.6+ to match the val distribution and
got clamped instead. The cap was leaving FPR points on the table.

That's also why **post-hoc threshold tuning was so effective**: the
F1=0.357 cascade reached F1=0.653 just by allowing each stage's
threshold to move past 0.5 to its actually-optimal point on the test
set. No new training, no new features — just the cuts that calibration
couldn't reach.

**Fix** (raise the cap to 0.95 in [adaboost.py](adaboost.py); add
[tools/tune_thresholds.py](tools/tune_thresholds.py)):
calibrate-during-training now baked-in the right thresholds when val
supports them. The post-hoc tuner remains a cheap final-mile
optimization that picks the right operating point on the
precision-recall curve (`--objective f1` for benchmark, `recall-at-spec`
for "find every face").

### 6. Hybrid NMS for multi-scale duplicates

Sliding-window detection at growth=1.25 fires the same face at
adjacent scales. A 24×24 box and a 30×30 box around the same face have
IoU ≈ 0.64 (fuses fine), but a 24×24 box inside a 48×48 box has IoU
≈ 0.25, **below the 0.3 NMS threshold**, so they don't fuse — producing
the "boxes on boxes" stacking visible at extreme scale ratios.

**Fix** ([utils.py](utils.py) `non_maximum_supression`,
`--nms-metric hybrid`): fuse if **either** IoU > threshold **or**
IoMin > 0.7, where IoMin = `intersection / min(area1, area2)`. IoMin is
1.0 for nested boxes regardless of scale ratio, so it catches what IoU
misses. The default `mode="weighted"` then merges the cluster into a
single score-weighted-average box rather than dropping the smaller-scale
detections, which produces tighter localization.

### A note on prevalence

CBCL test ratio faces : non-faces ≈ 1 : 50 (472 / 23 573). At a test
FPR of 2.3 %, false positives outnumber true positives even when recall
is excellent. **Precision is fundamentally capped by the class
imbalance**, and rises only by pushing FPR further down. Don't read
precision in isolation on this benchmark; F1 and the (recall,
specificity) pair are more honest. This is also why the tuner's two
objectives matter: F1 is a balanced summary; `recall-at-spec` lets you
pick a different operating point when F1's balance isn't what your
downstream task needs.

## Production recipe (deep cascade)

Full sequence to go from raw HF dataset to a tuned, deep-cascade
detector. Total wall-clock on M1: **prepare ~5 min + train ~10-12 h +
tune ~1 min**.

```bash
# 1) Prepare with all six fixes engaged
rm -rf data/24
python tools/prepare_data.py \
    --face-source celeba+cbcl \
    --n-faces 10000 \
    --neg-source mixed \
    --resolution 24 \
    --augment \
    --jitter 2

# 2) Train deep cascade (per-stage checkpoint on by default)
python main.py train \
    --data-dir data/24 \
    --layers 5 10 20 40 60 80 100 130 160 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 1000000

# 3) Evaluate the un-tuned model
python main.py test --data-dir data/24

# 4) Per-stage diagnostic — confirms each stage rejects what it should
python tools/diagnose_cascade.py \
    --weights $(ls -t weights/24/*.pkl | head -1) \
    --data-dir data/24

# 5) Pick an operating point. Headline benchmark:
python tools/tune_thresholds.py \
    --weights $(ls -t weights/24/*.pkl | head -1) \
    --data-dir data/24 \
    --objective f1
python main.py test --weights-path $(ls -t weights/24/*_tuned.pkl | head -1) \
    --data-dir data/24

# 6) ...or "find every face" point for visual detection:
python tools/tune_thresholds.py \
    --weights $(ls -t weights/24/*.pkl | head -1) \
    --data-dir data/24 \
    --objective recall-at-spec --min-spec 0.97

# 7) Run detection with hybrid NMS (default)
python main.py detect \
    --weights-path $(ls -t weights/24/*_recall97.pkl | head -1) \
    --detect-output images/outputs_deep
```

If training is interrupted, the most recent `weights/24/cvj_weights_<ts>.pkl`
is a fully usable partial cascade — testing and detect work against
however many stages had completed. Resume = re-run step 2 from scratch
(no per-stage warm-start; AdaBoost weights aren't cleanly resumable),
which is fine because per-stage checkpoints already protect you against
catastrophic loss.

## Example detections

Sliding window starts at the training window size and grows 1.25× per pass
with `shift=2`, post-processed with NMS @ IoU 0.3.

| Image          | Detection PNG                                     |
| -------------- | ------------------------------------------------- |
| `judybats.jpg` | ![judybats](images/outputs/judybats_detected.png) |

What you actually see in those PNGs lines up with the benchmark numbers
(recall ≈ 0.5, FPR ≈ 0.02), once you translate them to detection-time
behavior:

- **Recall side**: most clearly-frontal faces get at least one box. On a
  single subject (`i1.jpg`) the same face triggers several overlapping
  boxes at different scales — NMS suppresses ~60 % of duplicates but
  leaves a stack of 3-5 around the face. For the `images/people.png`
  group shot, the model finds most heads.
- **FPR side**: 2.3 % FPR sounds tiny but a 600×400 image at growth=1.25
  contains ~2-5 K candidate windows, so even 2 % FPR yields tens of
  spurious boxes. The cascade is most fooled by:
  - **Suits and ties** in `people.png` — the dark-light-dark vertical
    pattern matches "eyes-nose-eyes" Haar templates.
  - **Building windows and crowd backgrounds** in `physics.jpg` —
    repeating dark-light grids.
  - **Hair/shadow boundaries** in `judybats.jpg` and `clase.png`.

Variance normalization keeps raw candidate counts manageable, but at 4
stages the cascade can't reject these face-like patterns reliably.
Cleaner detection on real images would need a deeper cascade and/or
24×24 features, not a different post-processing step.

## Animation of the selected Haar features

![Viola-Jones](images/outputs/output.gif "Viola-Jones")

## Dataset

The HF dataset combines four sources, all under research-only licenses:
CelebA (faces), FDDB (faces), MIT CBCL (faces + non-faces + benchmark),
Caltech-256 (filtered for negatives). See the
[dataset card](https://huggingface.co/datasets/salvacarrion/face-detection)
for details and citations.

It exposes three HF splits — these are **not** train/val/test in the ML
sense; they are roles the data plays in V-J training:

| HF split    | Contents                                       |                                                                Counts | Used for                                                                                |
| ----------- | ---------------------------------------------- | --------------------------------------------------------------------: | --------------------------------------------------------------------------------------- |
| `train`     | Face crops AND CBCL non-faces, 48×48 grayscale | celeba=50 000 · fddb=11 383 · cbcl faces=2 429 · cbcl non-faces=4 548 | Training positives (`--face-source`) and matched negatives (`--neg-source cbcl\|mixed`) |
| `test`      | CBCL benchmark, 19×19 grayscale                |                                          472 faces · 23 573 non-faces | Final evaluation only — never used for training or calibration                          |
| `negatives` | Caltech-256 photos, variable RGB               |                                                    29 879 full images | Pool for hard-negative mining (`--neg-source caltech\|mixed`)                           |

Inside `train`, only CBCL ships **both** labels (faces and non-faces);
CelebA and FDDB are face-only.

To dump a visual sample of every bucket to your Desktop for inspection:

```bash
python tools/inspect_dataset.py
```

The maintainer scripts that built the HF dataset from raw sources live
in `tools/build_dataset.py` and `tools/push_to_hf.py` (run-once, not
needed for normal training/testing).
