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

## Workflow

The repo splits dataset preparation from training so that you only download
and preprocess once, then train/test/detect over the resulting NPY bundles.

### 1. Prepare data (downloads on first run)

`tools/prepare_data.py` pulls the curated dataset from
[`salvacarrion/face-detection`](https://huggingface.co/datasets/salvacarrion/face-detection)
on the Hugging Face Hub (cached in `~/.cache/huggingface/`), filters one face
source, splits it train/val/test, samples a negative pool, and writes
NPY bundles to `data/<resolution>/`.

The recommended config — what the [Results](#results) section was trained on —
matches positive AND negative domains to the CBCL benchmark:

```bash
python tools/prepare_data.py \
    --face-source cbcl \
    --neg-source mixed \
    --resolution 19 \
    --augment
```

`--neg-source mixed` writes two pools: `cbcl_neg_seed.npy` (curated CBCL
non-faces, used as the **stage-1 seed** for matched-domain calibration)
and `caltech_pool.npy` (1 M random Caltech-256 patches, used by
hard-negative mining from stage 2 on, for diversity in deep stages). See
[§ Why the matched seed matters](#why-the-matched-seed-matters) below.

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

```bash
python main.py train --data-dir data/19 --layers 5 20 50 100
```

Auto-loads `cbcl_neg_seed.npy` if present and uses it for the stage-1 seed.
Saves the cascade to `weights/<resolution>/cvj_weights_<unix-ts>.pkl`.

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

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data/24` | Directory with NPY bundles from `prepare_data.py` |
| `--weights-path` | auto | Checkpoint to load; omit to auto-pick the most recent under `weights/` |
| `--layers T [T ...]` | `5 20 50 100` | Weak learners per cascade stage |
| `--layer-recall` | `0.99` | Per-stage face-recall target; cumulative recall ≈ `layer-recall ^ N` |
| `--target-neg-per-stage` | `3000` | Negatives mined from pool per cascade stage |
| `--neg-sample-budget` | `100000` | Max patches sampled per stage when mining |
| `--detect-images IMG ...` | bundled samples | Images for `detect` mode |
| `--detect-output` | `images/outputs` | Directory where annotated outputs are saved |
| `--nms-threshold` | `0.3` | IoU threshold for NMS post-processing |

Cost notes: training is roughly linear in `n_train_pos × n_features × Σ
LAYERS`, with an extra 3 000-negative mining pass per stage. On a MacBook
Pro M1 (single-threaded Python), the recommended config above (~3 400
CBCL train positives at 19×19, layers `5 20 50 100`) finishes training in
≈ **15 minutes**. Replacing `--face-source cbcl` with `--face-source
celeba --n-faces 10000` raises training time to **~1 h** because there
are 5× more positives to compute features over.

## Results

All metrics on the CBCL benchmark test split (472 faces / 23 573
non-faces, the academic V-J reference set), at 19×19. Each row is a
single full retrain; bold = best F1.

| Config                                                                 | Recall | Specificity | Precision |   F1   | Train time |
|------------------------------------------------------------------------|-------:|------------:|----------:|-------:|-----------:|
| FDDB faces + Caltech-only negatives (initial baseline)                 |  0.784 |       0.326 |     0.023 |  0.044 | ~ 1 h 47 m |
| **CBCL faces + matched seed + Caltech mining, layers `5 20 50 100`**   |  0.502 |       0.979 |     0.292 | **0.369** | ~ 15 m |
| CBCL faces + matched seed, shorter cascade `5 10 20 40`                |  0.710 |       0.916 |     0.138 |  0.231 | ~ 9 m  |
| CelebA faces + matched seed (alignment failure)                        |  0.017 |       0.977 |     0.017 |  0.017 | ~ 1 h  |

Run-to-run F1 variance with the recommended config is ≈ ±0.025 (observed
0.369–0.394 across full retrains with identical commands), driven by the
random seeds in hard-negative mining and the val/train split. Treat any
single run within ±0.03 as equivalent.

Three complementary techniques drive the best-row numbers:

1. **Hard-negative mining** ([violajones.py](violajones.py)): each cascade
   stage trains on 3 000 fresh false positives mined from
   `caltech_pool.npy`, so the cascade isn't limited to a small fixed
   negative set.
2. **Per-window variance normalization** (Viola-Jones §5.1): each training
   sample's feature row is divided by its pixel std before AdaBoost sees
   it; at inference `WeakClassifier.classify` multiplies the threshold
   back by the window's std ([weakclassifier.py](weakclassifier.py)). This
   reduces false positives on high-contrast background regions.
3. **Held-out calibration**: `val_pos.npy` (10 % of the prepared faces) is
   reserved for calibrating `LAYER_RECALL=0.99` per stage
   ([violajones.py](violajones.py)). Fitting the threshold on the same
   positives the weak classifiers were optimized on overstates recall.

### Why the matched seed matters

The first attempt — FDDB faces with Caltech-only negatives — collapsed to
F1=0.044 not because the cascade failed to learn faces (recall was 0.78),
but because **specificity was 0.33**: the cascade called 67 % of CBCL
non-faces "face". The reason is a **negative-domain gap**:

- Caltech patches are random crops of object photos (textures, edges,
  uniform regions). Easy to reject — once the cascade learns "not flat,
  not pure-edge", it's done.
- CBCL non-faces (the benchmark's negative set) were curated to be
  face-like: face-pose backgrounds, near-symmetric features, similar
  pixel statistics. Caltech-trained cascades have never seen anything
  remotely close, so they let them through.

Fix (the `--neg-source mixed` flag in
[tools/prepare_data.py](tools/prepare_data.py)): pull the ≈ 4 500 CBCL
non-faces from the HF train split, augment with horizontal flip → 9 096
patches, and use them as the **stage-1 seed** for the cascade.
[violajones.py](violajones.py) accepts an optional `seed_neg_pool` arg
that biases stage-1 sampling toward this matched-domain seed; from stage
2 on, hard-neg mining still draws from the broader Caltech pool for
deep-stage diversity. Result: specificity jumped from 0.33 → 0.98 in one
training run.

### Why CelebA at 19×19 collapses

The "CelebA + matched seed" row above (recall 0.017) is the most
surprising failure. CelebA faces are frontal, just like CBCL — so why
does the cascade not generalise?

At 19×19 resolution **alignment matters at the pixel level**, not just
pose. CBCL crops are tightly cropped 19×19 native captures with eyes
landing at a specific row, nose at another, mouth at another. CelebA
crops are 48×48 native with looser framing (more forehead, hair, neck);
when bilinearly downsized to 19×19, the same landmarks land at different
rows. A 1-pixel shift on a 19-pixel face is ~5 % of the face — and Haar
features tile rectangle sums at fixed grid positions, so the feature
distribution moves. The cascade learned "eyes at row 8" (CelebA at 19);
CBCL test faces have eyes at row 7 and look like non-faces in feature
space.

Conclusion: at 19×19 the training and test crops must share alignment
conventions, not just pose category. CelebA→CBCL works at 24×24 in the
literature because the extra 5 pixels of slack absorb sub-pixel
misalignment; at 19×19 there is no slack. The matched-domain CBCL
training is the only configuration that gets above F1=0.1 here.

### What still bottlenecks the result

F1≈0.4 is the practical ceiling for this setup. The remaining levers, in
roughly decreasing impact:

- **Cascade depth.** The paper uses 38 stages and ≈ 6 000 weak
  classifiers; this implementation uses 4 stages and 175 total WCs. More
  stages would push FPR further down without hurting recall much.
- **Resolution.** 19×19 caps the discriminative power of Haar features
  (≈ 17 K candidates). Doubling to 24×24 raises that to ≈ 134 K and is
  what the original paper used.
- **Train-set size.** Only 1 943 unique CBCL training faces (3 886 with
  flip). Stronger augmentation — small shift jitter, brightness — could
  multiply the effective training set ~5× without breaking alignment.
- **Calibration generalisation gap.** Calibration on CBCL train faces
  overshoots recall on CBCL test faces by ~30 percentage points.
  Increasing `--val-frac` from 0.1 → 0.2 with augmented val_pos was tried
  and made no measurable difference (0.394 → 0.387 F1, within run-to-run
  noise) — the two CBCL splits are genuinely different photo pools, not
  just a sample-size issue. There is no fix without leaking test data
  into calibration.

### A note on prevalence

Test ratio faces : non-faces ≈ 1 : 50 (472 / 23 573). At a test FPR of
2.3 %, the 576 false positives still outnumber the 257 true positives —
**precision is fundamentally capped by the class imbalance**, and will
only rise by pushing FPR further down. Don't read precision in isolation
on this benchmark; F1 and the (recall, specificity) pair are more
honest.

## Example detections

Sliding window starts at the training window size and grows 1.25× per pass
with `shift=2`, post-processed with NMS @ IoU 0.3.

|  Image          | Detection PNG |
|-----------------|---------------|
| `people.png`    | ![people](images/outputs/people_detected.png)     |
| `clase.png`     | ![clase](images/outputs/clase_detected.png)       |
| `physics.jpg`   | ![physics](images/outputs/physics_detected.png)   |
| `i1.jpg`        | ![i1](images/outputs/i1_detected.png)             |
| `judybats.jpg`  | ![judybats](images/outputs/judybats_detected.png) |

Variance normalization suppresses high-contrast non-face regions before
they ever reach the cascade, so raw candidate counts stay manageable and
NMS resolves the rest. On real-image detection, the FPR=2.3 % from the
benchmark becomes much more tolerable — most spurious windows collapse
into a few duplicates that NMS removes.

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

| HF split    | Contents                                                    | Counts                                              | Used for                                                                |
|-------------|-------------------------------------------------------------|----------------------------------------------------:|-------------------------------------------------------------------------|
| `train`     | Face crops AND CBCL non-faces, 48×48 grayscale              | celeba=50 000 · fddb=11 383 · cbcl faces=2 429 · cbcl non-faces=4 548 | Training positives (`--face-source`) and matched negatives (`--neg-source cbcl\|mixed`) |
| `test`      | CBCL benchmark, 19×19 grayscale                             | 472 faces · 23 573 non-faces                       | Final evaluation only — never used for training or calibration          |
| `negatives` | Caltech-256 photos, variable RGB                            | 29 879 full images                                 | Pool for hard-negative mining (`--neg-source caltech\|mixed`)           |

Inside `train`, only CBCL ships **both** labels (faces and non-faces);
CelebA and FDDB are face-only.

To dump a visual sample of every bucket to your Desktop for inspection:

```bash
python tools/inspect_dataset.py
```

The maintainer scripts that built the HF dataset from raw sources live
in `tools/build_dataset.py` and `tools/push_to_hf.py` (run-once, not
needed for normal training/testing).
