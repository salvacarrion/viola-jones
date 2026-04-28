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
source, splits it 80/10/10, samples a Caltech negative pool, and writes
NPY bundles to `data/<resolution>/`.

```bash
# Defaults: --face-source celeba --n-faces 10000 --resolution 24
python tools/prepare_data.py

# Or explicit
python tools/prepare_data.py \
    --face-source fddb \
    --n-faces 10000 \
    --resolution 24 \
    --augment
```

Output:

```
data/24/
├── train_pos.npy        # face crops for AdaBoost
├── val_pos.npy          # held-out faces for stage calibration
├── test_pos.npy         # face crops for matched-distribution eval
├── caltech_pool.npy     # ~1 M Caltech crops for hard-neg mining
├── cbcl_test_pos.npy    # CBCL benchmark (faces)
├── cbcl_test_neg.npy    # CBCL benchmark (non-faces)
└── manifest.json
```

### 2. Train

```bash
python main.py train --data-dir data/24
```

Saves the cascade to `weights/<resolution>/cvj_weights_<unix-ts>.pkl`.

### 3. Test

Evaluates the most recent (or `--weights-path`-specified) checkpoint on the
CBCL benchmark.

```bash
python main.py test --data-dir data/24
```

### 4. Detect

Runs sliding-window detection on images and writes annotated PNGs.

```bash
python main.py detect \
    --detect-images images/people.png images/clase.png
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

Cost notes: training is roughly linear in samples × features × Σ`LAYERS`,
with an extra 3 000-negative mining pass per stage from the bootstrap pool.
On a MacBook Pro M1 (single-threaded Python), a paper-style
`[1, 10, 50, 100]` run on ~10 k CelebA + 1 M Caltech pool at 24×24 takes
roughly 2–3 hours; a smaller `[5, 10]` smoke run finishes in ~10 minutes.

## Results

Trained on 10 000 CelebA faces (8 000 train / 1 000 val / 1 000 test) with
`LAYERS=[1, 10, 50, 100]` and a 1 M Caltech bootstrap pool. Reported on the
CBCL test set (472 / 23 573 faces / non-faces) for direct comparison with
the literature.

| Setup                                                   |  Test F1 | Test acc. | Test precision | Test recall |
|---------------------------------------------------------|---------:|----------:|---------------:|------------:|
| `LAYERS=[5, 10]` (smoke)                                |    0.306 |     0.963 |          0.243 |       0.411 |
| `LAYERS=[2, 10, 30]`                                    |    0.097 |     0.694 |          0.052 |       0.839 |
| `LAYERS=[1, 10, 50, 100]` (paper-style + var-norm + cal.) | **0.375** | 0.958 | **0.265** | **0.642** |

Three complementary techniques drive the test-set numbers:

1. **Hard-negative mining** ([violajones.py](violajones.py)): each cascade
   stage trains on 3 000 fresh false positives mined from the ~1 M
   face-free Caltech patches in `caltech_pool.npy`, so the cascade isn't
   limited to a small fixed negative set.
2. **Per-window variance normalization** (Viola-Jones §5.1): each training
   sample's feature row is divided by its pixel std before AdaBoost sees
   it; at inference `WeakClassifier.classify` multiplies the threshold
   back by the window's std ([weakclassifier.py](weakclassifier.py)). This
   reduces false positives on high-contrast background regions.
3. **Held-out calibration**: `val_pos.npy` (10 % of the prepared faces) is
   reserved for calibrating `LAYER_RECALL=0.99` per stage
   ([violajones.py](violajones.py)). Fitting the threshold on the same
   positives the weak classifiers were optimized on overstates recall.

### Why test metrics look much worse than train

- **Prevalence shift.** Test ratio faces : non-faces ≈ 1 : 50 (472 / 23 573).
  At a test FPR of just 3.6 %, FPs (839) still outnumber TPs (303) —
  precision is fundamentally capped by the class imbalance and will only
  rise by reducing FPR further.
- **Distribution gap.** CelebA training faces are tightly aligned 48×48
  headshots; CBCL test faces are 19×19 with different illumination and
  cropping. Training and test distributions are not identical.
- **Limited cascade depth.** The paper uses 38 stages with thousands of
  weak classifiers; this implementation uses 4 stages and 161 total WCs.
  More stages trained on a richer negative pool would push FPR lower
  without hurting recall.

## Example detections

Sliding window starts at the training window size and grows 1.25× per pass
with `shift=2`, post-processed with NMS @ IoU 0.3.

| Image           | Raw windows ⇒ post-NMS | Detection PNG |
|-----------------|------------------------|---------------|
| `people.png`    |    146 ⇒ 26 | ![people](images/outputs/people_detected.png)     |
| `clase.png`     |    694 ⇒ 49 | ![clase](images/outputs/clase_detected.png)       |
| `physics.jpg`   |    490 ⇒ 44 | ![physics](images/outputs/physics_detected.png)   |
| `i1.jpg`        |    154 ⇒  5 | ![i1](images/outputs/i1_detected.png)             |
| `judybats.jpg`  |  1 495 ⇒ 45 | ![judybats](images/outputs/judybats_detected.png) |

Variance normalization suppresses high-contrast non-face regions before
they ever reach the cascade, so raw candidate counts stay manageable
(`people.png`: 146, `i1.jpg`: 154) and most "noisy" images (`clase.png`,
`physics.jpg`, `judybats.jpg`) still resolve to a few dozen boxes after
NMS — consistent with the cascade's 0.642 test recall.

## Animation of the selected Haar features

![Viola-Jones](images/outputs/output.gif "Viola-Jones")

## Dataset

The HF dataset combines four sources, all under research-only licenses:
CelebA (faces), FDDB (faces), MIT CBCL (faces + benchmark), Caltech-256
(filtered for negatives). See the
[dataset card](https://huggingface.co/datasets/salvacarrion/face-detection)
for details and citations.

The maintainer scripts that built it from raw sources live in
`tools/build_dataset.py` and `tools/push_to_hf.py` (run-once, not needed
for normal training/testing).
