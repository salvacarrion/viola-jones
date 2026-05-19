# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

The codebase is a from-scratch Viola-Jones face detector. End-to-end flow:

```
tools/prepare_data.py  →  data/<res>/*.npy
main.py train          →  weights/<res>/cvj_weights_<ts>.pkl   (per-stage checkpoint)
main.py test           →  metrics on CBCL benchmark
main.py detect         →  annotated PNGs under images/outputs/
```

The dataset (`salvacarrion/face-detection` on HF Hub) is downloaded once and cached in `~/.cache/huggingface/`. Preprocessed NPY bundles under `data/` and trained weights under `weights/` are gitignored and reproducible from the prepare/train commands.

## Common commands

Quick recipe (19×19, ~15 min training, F1 ≈ 0.37):

```bash
python tools/prepare_data.py --face-source cbcl --neg-source mixed --resolution 19 --augment
python main.py train  --data-dir data/19 --max-stages 6 --max-wcs-per-stage 100
python main.py test   --data-dir data/19
python main.py detect --detect-images images/people.png
```

Production recipe (24×24, ~10-12 h training):

```bash
python tools/prepare_data.py --face-source celeba+cbcl --n-faces 10000 --neg-source mixed --resolution 24 --augment --jitter 2
python main.py train --data-dir data/24 --max-stages 30 --max-wcs-per-stage 200 --target-stage-fpr 0.3 --min-cascade-recall 0.80 --target-neg-per-stage 5000 --neg-sample-budget 1000000
```

Resume an interrupted training run (the in-place `.pkl` is always a usable partial cascade):

```bash
python main.py train --data-dir data/24 --resume-from $(ls -t weights/24/cvj_weights_*.pkl | head -1) ...
```

Post-hoc threshold tuning (no retraining — only moves the per-stage cut points):

```bash
python tools/tune_thresholds.py --weights $(ls -t weights/24/*.pkl | head -1) --data-dir data/24 --objective f1
python tools/tune_thresholds.py --weights $(ls -t weights/24/*.pkl | head -1) --data-dir data/24 --objective recall-at-spec --min-spec 0.97
# Writes <weights>_tuned.pkl or <weights>_recall97.pkl — pass to test/detect via --weights-path
```

Per-stage diagnostic (pass rates + score distributions on CBCL):

```bash
python tools/diagnose_cascade.py --weights $(ls -t weights/19/*.pkl | head -1) --data-dir data/19
```

Run unit tests (integral image / Haar geometry):

```bash
python -m unittest discover -s tests
# Single test method:
python -m unittest discover -s tests -k test_integral_image
```

When `--weights-path` is omitted, `main.py` auto-picks the most recent `weights/**/cvj_weights_*.pkl` by mtime — this is what makes the per-stage checkpoint-then-final flow transparent.

## Architecture

### Files

- `main.py` — CLI entry (train / test / detect)
- `violajones.py` — cascade class: orchestrates AdaBoost stages, hard-neg mining, multi-scale sliding-window inference
- `adaboost.py` — boosting loop with per-round calibrated FPR + recall early-stop
- `weakclassifier.py` — single Haar-feature decision stump (built by `AdaBoost._best_stump`); `classify` applies inference-time variance normalization
- `features.py` — `RectangleRegion`, `HaarFeature` — operate on a *padded* (H+1, W+1) integral image so rectangle sums are 4 unconditional reads
- `utils.py` — `integral_image` (+ pow2), `apply_features` (vectorized), `non_maximum_supression`, `evaluate`
- `tools/prepare_data.py` — HF download → NPY bundles
- `tools/tune_thresholds.py` — post-hoc per-stage threshold optimizer
- `tools/diagnose_cascade.py` — per-stage debug dump on CBCL test
- `tools/inspect_dataset.py` — dump visual samples of every HF bucket for sanity-check
- `tools/mine_hard_negatives.py` — pre-mine a hard-neg pool from an existing cascade (optional `--hard-neg-pool` for training)
- `tools/truncate_checkpoint.py` — keep only the first N stages of a saved cascade
- `tools/dataset_build/` — maintainer-only scripts that built the HF dataset from raw sources (mediapipe alignment, parquet packaging, HF upload). Not needed for normal training/testing/detection — see `tools/dataset_build/requirements.txt` for its extra deps

### Three layers of the cascade

1. **WeakClassifier** (single Haar stump). `feature_value < polarity·threshold·scale²·std → 1`. The `scale²` and `std` factors implement Viola-Jones §5.1 *per-window variance normalization* at inference time: training feature values were divided by each sample's pixel std, so the learned threshold is in std-normalized units and is multiplied back at inference. The `scale²` accounts for area growth when the same feature is evaluated at a larger window in the pyramid.

2. **AdaBoost stage** (`adaboost.py`). Boosts decision stumps; `_best_stump` is fully vectorized (sort + prefix-sum + argmin per feature chunk). The early-stop in `train()` is **two-condition** (paper §3): at each round it recalibrates `self.threshold` so `target_recall` of `val_pos` passes, then halts only when both `recall ≥ target_recall` AND `fpr ≤ target_stage_fpr` hold *at that calibrated threshold*. Naive single-condition early-stops (FPR-at-0.5) collapse every stage to 1 stump — see `README.md` §7 for the full debug story.

3. **ViolaJones cascade** (`violajones.py`). Holds a list of trained `AdaBoost` stages plus `base_width/base_height/base_scale/shift` for inference. Training loop: mine hard negatives → train AdaBoost → calibrate threshold → check cumulative val recall → loop. Number of stages and weak classifiers per stage both emerge from training, not pre-specified.

### Inference fast path

`ViolaJones.find_faces` is fully vectorized. It builds one padded II (and one squared II for per-window std) for the whole image, then for each pyramid scale forms the grid of window origins and runs the cascade as a batched NumPy reduction over surviving windows. `_batch_haar_value` does fancy-indexed rectangle sums for all live windows at once — the inner work is 4 array reads per Haar rectangle, no per-window Python dispatch.

### Hard-negative mining (`_mine_hard_negatives`)

When a `neg_seed` pool exists (CBCL non-faces from `--neg-source mixed`), every stage's mining budget is **stratified 50/50** between the seed and the main Caltech pool — matched-domain negatives stay in the training mix at *every* stage, not just stage 1. Mining within a pool reads the (possibly memmap'd) `.npy` in random-ordered contiguous chunks for OS-prefetch friendliness; sequential per-patch reads are critical because the pools are larger than RAM in the production recipe.

### Calibration

`AdaBoost._calibrated_threshold` is used both during training (per round) and post-stage. It picks the largest threshold keeping `target_recall` of val scores, bounded below by `min(alphas)/sum(alphas)` (the "single-cheapest-WC-fires" floor — without this, the threshold collapses to 0 and the stage accepts everything) and above by `0.95` (guards against the "threshold ≈ 1.0 accepts nothing" degenerate case). The 0.95 cap was originally 0.5 and was raised because calibration was hitting its own ceiling at deep stages — see `README.md` §5.

### Feature cache

`ViolaJones._apply_and_normalize` writes `data/<res>/_cache/xf_{pos,val}.npy` on first run and **memory-maps** them on subsequent runs (`mmap_mode='r'`). The pos feature matrix is ~7.7 GB at 24×24 — full RAM load would OOM laptops. Memmap is read-only and the matrix is only consumed via concat + calibration, so this is bit-for-bit equivalent to a full load.

### NMS

`utils.non_maximum_supression` defaults to `mode="weighted"` (fuse overlapping cluster into score-weighted-average box) + `metric="hybrid"` (fuse if IoU > threshold **or** IoMin > 0.7). IoMin catches nested boxes at very different scales (24×24 inside 48×48 has IoU ≈ 0.25, below default threshold — so plain IoU would not fuse them). The algorithm is **connected-components**: it builds the full pairwise fuse matrix and groups boxes by transitive overlap, so order-of-processing can't leave a duplicate behind (the old greedy version had a chain bug where a partial-overlap box could "use up" a nested duplicate before its parent saw it). The detection score returned by `find_faces` is the **sum of per-stage AdaBoost margins** (`vote − layer_threshold` accumulated across every stage the window passed) — a continuous confidence signal that drives `weighted` centroid weighting and `greedy` representative selection.

## Conventions and gotchas

- **Resolution is the data axis**, not a config knob: weights live under `weights/<res>/`, NPY bundles under `data/<res>/`, and the trained `ViolaJones` stores `base_width/base_height` from the training data. `--resume-from` asserts the checkpoint resolution matches the data dir.
- **`val_pos` is the calibration AND stop-criterion anchor**. It's sampled from the benchmark's HF train split, un-augmented (no jitter, no flip). Using the same set for both eliminates the "calibrate against X, stop against Y" inconsistency that used to terminate the cascade after one stage.
- **Per-stage checkpoint is on by default**: `clf.save(checkpoint_path)` after every completed stage overwrites the same `.pkl`. A crashed run leaves a usable partial cascade — testing and detection work against it. RNG state is *not* preserved on resume, so a resumed run is close-but-not-identical to an uninterrupted one.
- **Coordinate conventions**: `RectangleRegion(x=col, y=row, width=col-span, height=row-span)`. Padded II is `(H+1, W+1)` with zero first row/col. Detection regions are `(x1, y1, x2, y2, score)` tuples in image pixels.
- **No `_pycache_` cleanup needed**: `*.pyc`, `data/`, `*.npy`, `bootstrap/`, `datasets/` are all in `.gitignore`.
- **README.md** is the canonical narrative of *why each flag exists* — each option is tied to a measured failure mode (negative-domain gap, alignment-at-low-res, calibration cap, etc.). When changing defaults or adding new flags, preserve the diagnosis→fix mapping in the README.
- **`FINDINGS.md`** keeps additional experimental notes from the build.
