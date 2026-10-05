# Viola-Jones: Workflow

Quick guide to the end-to-end loop as it was actually run: prepare data, train a baseline cascade, iterate with curation (score + filter + raw hard-negative reservoir), evaluate, diagnose, tune and detect. Reading time ~5 min; running time ~30 min minimum (cold start, quick 19×19 recipe) or ~12-40 h (full iteration recipe).

For the "why" behind each flag and the failures that motivated each fix, see [FINDINGS.md](FINDINGS.md). For per-experiment numbers, [RESULTS.md](RESULTS.md). If you are going to retrain, read [RETRAINING.md](RETRAINING.md) first: it summarises what limits the current models on real images and what to change.

---

## Pipeline at a glance

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ prepare_data.py │───▶│ data/<res>_<src>/   │───▶│ main.py train       │
│ (HF → NPY)      │    │   train_pos, val,   │    │ (cold start)        │
└─────────────────┘    │   neg_seed, pool    │    └──────────┬──────────┘
                       └─────────────────────┘               │
                                                             ▼
                                                  weights/<res>/baseline.pkl
                                                             │
            ┌────────────────────────────────────────────────┤
            ▼                                                ▼
┌─────────────────────┐  oracle  ┌─────────────────────┐    ┌──────────────────┐
│ score_faces.py      │◀────────│ mine_hard_negs_raw  │    │ test + diagnose  │
│ (rank positives)    │   (cbcl │ (stream from HF)    │    │ + tune_thresholds│
└──────────┬──────────┘   _v2)  └──────────┬──────────┘    └──────────────────┘
           │                                │
           ▼                                ▼
  data/<dir>/face_scores      weights/<res>/<stem>__vhardneg_raw.npy
  + score_samples.png
           │                                │
           └──────────────┬─────────────────┘
                          ▼
       ┌──────────────────────────────────────┐
       │ main.py train                        │
       │   --drop-low-score-pos FRAC          │
       │   --very-hard-neg-pool <vhard.npy>   │
       └──────────────────┬───────────────────┘
                          ▼
              weights/<res>/<run_v3>.pkl
                          │
                          ▼
              detect + tune + iterate
```

---

## 1. Prepare data

Once per `(resolution, positive source)` combination. Output goes to `data/<res>_<src>/` with `train_pos.npy`, `val_pos.npy`, `caltech_pool.npy`, `neg_seed.npy`, `test_pos.npy`, `test_neg.npy`, `manifest.json`.

**Quick 19×19 recipe (CBCL):**

```bash
python tools/prepare_data.py \
    --face-source cbcl \
    --neg-source mixed \
    --resolution 19 \
    --augment --jitter 1 \
    --benchmark cbcl --val-size 500 \
    --pool-size 50000000 \
    --out-dir data/19_cbcl
```

**Recipe with aligned CelebA (more faces, validated frontal alignment):**

```bash
python tools/prepare_data.py \
    --face-source celeba_aligned --n-faces 5000 \
    --neg-source mixed --resolution 19 \
    --augment --jitter 1 \
    --benchmark cbcl --val-size 500 \
    --pool-size 50000000 \
    --out-dir data/19_celeba_aligned
```

**Mixed recipe (aligned CelebA + CBCL):**

```bash
python tools/prepare_data.py \
    --face-source celeba_aligned+cbcl --n-faces 5000 \
    --neg-source mixed --resolution 19 \
    --augment --jitter 1 \
    --benchmark cbcl --val-size 500 \
    --pool-size 50000000 \
    --out-dir data/19_celeba_aligned+cbcl
```

Quick visual inspection of the dataset buckets (sanity check):

```bash
python tools/inspect_dataset.py --out-dir samples/19_celeba_aligned
```

---

## 2. Train a baseline (cold start)

No oracle yet: this is the first cascade on the prepared dataset. It is the one later used as the oracle to score positives and mine negatives.

```bash
python main.py train \
    --data-dir data/19_cbcl \
    --max-stages 20 \
    --max-wcs-per-stage 400 \
    --target-stage-fpr 0.5 \
    --min-cascade-recall 0.95 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 \
    --min-stage-negatives 1000
```

Hyperparameters are explained in [FINDINGS.md §7](FINDINGS.md#7-adaptive-cascade-calibrated-fpr--recall). The live checkpoint (`weights/19/cvj_weights_<ts>.pkl`) is overwritten after every stage, so if the run is interrupted at hour 5 of 15 the file is still a usable partial cascade.

When it finishes, give it a readable name:

```bash
mv weights/19/cvj_weights_<ts>.pkl weights/19/cbcl__19_v1.pkl
```

---

## 3. Test + diagnose + tune (always, after training)

Three cheap steps (<1 min each) to run on **every** new model. Without them you do not know whether the model is any good.

```bash
# Test on the CBCL benchmark (472 faces / 23K non-faces)
python main.py test \
    --data-dir data/19_cbcl \
    --weights-path weights/19/cbcl__19_v1.pkl

# Per-stage pass rates + score distributions
python tools/diagnose_cascade.py \
    --weights weights/19/cbcl__19_v1.pkl \
    --data-dir data/19_cbcl

# Greedy F1 sweep over the per-stage thresholds (no retraining)
python tools/tune_thresholds.py \
    --weights weights/19/cbcl__19_v1.pkl \
    --data-dir data/19_cbcl \
    --objective f1
# -> writes cbcl__19_v1_tuned.pkl

# Test after tuning
python main.py test \
    --data-dir data/19_cbcl \
    --weights-path weights/19/cbcl__19_v1_tuned.pkl
```

The tuned F1 is the one to report. The raw-vs-tuned gap (~5-7 pp at 19×19) measures how much the on-the-fly calibration leaves on the table compared with a global optimisation afterwards.

**Tuner objectives:**

- `--objective f1`: for the benchmark and comparisons (default).
- `--objective recall-at-spec --min-spec 0.97`: "find as many faces as possible" at a given minimum specificity.

For in-the-wild performance, also run the FDDB evaluation (§7): CBCL F1 says nothing about how the model behaves on full photos.

---

## 4. Positive curation (score + visualise + filter)

Use this when a dataset has alignment noise (typically CelebA, even after "alignment"). The CBCL baseline acts as a frozen oracle that ranks every face of a different dataset.

**4.1. Score the positives** (~3 min for 20K faces):

```bash
python tools/score_faces.py \
    --weights weights/19/cbcl__19_v1.pkl \
    --data-dir data/19_celeba_aligned \
    --save-samples 16
```

It produces:

- `data/19_celeba_aligned/face_scores.npy`: one float32 per face, the sum of per-stage margins (no short-circuit).
- `data/19_celeba_aligned/score_samples.png`: a grid with 8 percentile bands (p00-p05 worst, ..., p95-p100 best), green border if the cascade accepts the face and red if it rejects it, yellow score per face.

Look at the PNG and decide which bottom fraction to drop:

- If the bottom 10-20% are clearly non-canonical crops (off-centre eyes, too much forehead or neck, partial profiles): drop 0.20.
- If the bottom 20-30% still look "ugly": drop 0.30.
- If the first bands are reasonable faces: drop 0.10 or do not filter at all.

The command also prints:

- `passed/rejected`: exact % the cascade accepts/rejects with `classify()` (a deterministic criterion, different from the sign of the score).
- A percentile histogram of the continuous score (the ranking signal).

**4.2. Mine very-hard negatives from the raw HF images** (~2-6 h, depending on the oracle's strength):

```bash
python tools/mine_hard_negatives_raw.py \
    --weights weights/19/cbcl__19_v1.pkl \
    --target 15000 \
    --budget 500000000 \
    --patches-per-image 200 \
    --out weights/19/cbcl__19_v1__vhardneg_raw.npy
```

It streams random patches from `ds["negatives"]` (the raw Caltech images in the HF dataset) until it reaches `--target` or exhausts `--budget`. No intermediate pool on disk; multiple passes when needed. If the output is smaller than the target that is fine: the trainer uses it as a top-up reservoir.

Key difference from the legacy [tools/mine_hard_negatives.py](../tools/mine_hard_negatives.py): that one mines the finite `caltech_pool.npy` (~50M patches); this one streams raw images without that bound. Use the new one when the cascade is strong and the finite pool runs dry.

---

## 5. Retrain with curation

Same configuration as the baseline plus the two new flags:

```bash
python main.py train \
    --data-dir data/19_celeba_aligned \
    --max-stages 20 \
    --max-wcs-per-stage 400 \
    --target-stage-fpr 0.5 \
    --min-cascade-recall 0.95 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 \
    --min-stage-negatives 1000 \
    --drop-low-score-pos 0.20 \
    --very-hard-neg-pool weights/19/cbcl__19_v1__vhardneg_raw.npy
```

What changes internally:

- `--drop-low-score-pos 0.20` drops the bottom 20% of `train_pos` ranked by `face_scores.npy`. The feature cache is rotated to `xf_pos__drop0.20.npy` so it does not pollute the full-set cache.
- `--very-hard-neg-pool` loads the reservoir and uses it **only** as a top-up when seed + Caltech mining comes up short at a stage (typically stage 12+).

After training: rename, test, diagnose, tune, test the tuned model: the same loop as §3.

---

## 6. Extend stages (resume)

If a cascade capped stages with `final_fpr > target_stage_fpr` (capacity ceiling, see [FINDINGS.md](FINDINGS.md#1919-capacity-ceiling-jitter-saturates-the-cascade)), it can be resumed with a relaxed target:

```bash
python main.py train \
    --data-dir data/19_cbcl \
    --resume-from weights/19/cbcl__19_v1.pkl \
    --max-stages 20 \
    --max-wcs-per-stage 800 \
    --target-stage-fpr 0.65 \
    --min-cascade-recall 0.95 \
    --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 \
    --min-stage-negatives 1000
```

If a stage saturates during the resume (FPR above the new target) and you want to drop it before continuing:

```bash
python tools/truncate_checkpoint.py \
    --weights weights/19/cvj_weights_<ts>.pkl \
    --keep-stages 12 \
    --out weights/19/cbcl__19_v1.1.pkl
```

---

## 7. Detect on real images

Before detecting (or comparing on FDDB), fit the model's box transform. The cascade fires on its *training window*, which for every model here is a tight eyes-to-mouth crop; `box_transform` maps it to a face box (forehead to chin). It is fitted on FDDB fold 1 and stored inside the `.pkl` (every shipped model already has one):

```bash
python tools/eval_fddb.py --weights weights/24/<model>.pkl --skip-opencv \
    --folds 1 --box-fit-folds 1 --save-box-transform
```

Then:

```bash
python main.py detect \
    --detect-images images/people.png images/judybats.jpg \
    --detect-output images/outputs/celeba_aligned__24_v2_s11 \
    --weights-path weights/24/celeba_aligned__24_v2_s11_tuned.pkl
```

And the in-the-wild benchmark against OpenCV (boxes fitted on fold 1, metrics on the held-out folds 2-10):

```bash
python tools/eval_fddb.py --weights weights/24/<model>.pkl --cascade default \
    --box-fit-folds 1 --folds 2,3,4,5,6,7,8,9,10 --iou 0.3,0.5
```

**Recommendations from experience:**

- **Use the `_tuned.pkl`.** The raw model used to be recommended for detection, but with the fixed pyramid and face boxes the tuned one also wins on FDDB (24×24 CelebA v2, fold 1: AP@0.5 0.45 tuned vs 0.31 raw).
- **The pyramid scales the image** (`--detect-pyramid image`, the default), as OpenCV does, with the same bilinear filter that built the training patches. The old `--detect-pyramid features` mode (scaling the Haar rectangles) lets ~3.5× more background through; it exists only to reproduce old numbers. See [FINDINGS.md §B10](FINDINGS.md#b10-the-sliding-window-pyramid-scaled-the-features-instead-of-the-image).
- **`--raw-boxes`** draws the raw window (eyes to mouth) instead of the face box. Useful for debugging, not for evaluating against annotations.
- **`--nms-threshold 0.2-0.3`** + **`--nms-metric hybrid`** (default): fuses nested duplicates (the same face detected at scales 1×, 1.5×, 2×). See [FINDINGS.md §6](FINDINGS.md#6-hybrid-nms-for-multi-scale-duplicates).
- **`--detect-min-face`** only if you know there are no small faces (portraits): it removes pyramid levels and their false positives, but also the small faces. On FDDB, raising it to 80 *lowers* AP@0.5 from 0.47 to 0.32.
- **`--detect-scale 1.3`**: fewer pyramid levels than the default 1.25, ~30% faster with a marginal recall loss.
- **`--detect-min-score`**: drops detections with a low accumulated margin (likely false positives). Every run prints the min/median/max score to help pick the cut; `0.8` works well for the best 24×24 model.

---

## 8. Common gotchas

- **Stale feature cache**: changing `--face-source` without deleting `data/<res>_<src>/_cache/` reuses old features with the new `train_pos` (a silent mismatch). Fix: `rm -rf data/<res>_<src>/_cache/` before retraining on different data. The `--drop-low-score-pos` filter already rotates the cache automatically.
- **Stale `cvj_weights_*.pkl`**: the extension scripts (`scripts/run_19_extend.sh`) refuse to start if there is an un-renamed `cvj_weights_*.pkl` in `weights/<res>/`. Clean up or rename first.
- **Weights auto-pick**: if you omit `--weights-path` in test/detect, the most recent `cvj_weights_*.pkl` under `weights/` (by mtime) is used, falling back to the shipped best model on a fresh clone. Handy for iterating; dangerous with several runs in parallel.
- **`--resume-from` checks the resolution**: the checkpoint only resumes if `clf.base_width` matches the data dir's resolution; a mismatch fails cleanly.
- **`--min-stage-negatives 1000`**: if mining returns fewer, the cascade stops with a clear message. It is the typical symptom of an exhausted pool at late stages; the fix is to pre-mine a very-hard reservoir with [tools/mine_hard_negatives_raw.py](../tools/mine_hard_negatives_raw.py) and pass it via `--very-hard-neg-pool`.

---

## 9. Full end-to-end recipe (copy-paste)

For an "aligned CelebA improved with a CBCL oracle" experiment:

```bash
# 1. Prepare data (~10-15 min)
python tools/prepare_data.py --face-source cbcl --neg-source mixed \
    --resolution 19 --augment --jitter 1 --benchmark cbcl --val-size 500 \
    --pool-size 50000000 --out-dir data/19_cbcl
python tools/prepare_data.py --face-source celeba_aligned --n-faces 5000 \
    --neg-source mixed --resolution 19 --augment --jitter 1 \
    --benchmark cbcl --val-size 500 --pool-size 50000000 \
    --out-dir data/19_celeba_aligned

# 2. Train the CBCL baseline (oracle) (~6 h)
python main.py train --data-dir data/19_cbcl --max-stages 20 \
    --max-wcs-per-stage 400 --target-stage-fpr 0.5 \
    --min-cascade-recall 0.95 --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 --min-stage-negatives 1000
mv weights/19/cvj_weights_*.pkl weights/19/cbcl__19_v1.pkl

# 3. Score the CelebA positives + visualise
python tools/score_faces.py --weights weights/19/cbcl__19_v1.pkl \
    --data-dir data/19_celeba_aligned --save-samples 16
# -> look at data/19_celeba_aligned/score_samples.png before choosing FRAC

# 4. Pre-mine very-hard negatives from the raw HF images (~2-6 h)
python tools/mine_hard_negatives_raw.py \
    --weights weights/19/cbcl__19_v1.pkl --target 15000 \
    --out weights/19/cbcl__19_v1__vhardneg_raw.npy

# 5. Train CelebA with curation (~5-10 h)
python main.py train --data-dir data/19_celeba_aligned --max-stages 20 \
    --max-wcs-per-stage 400 --target-stage-fpr 0.5 \
    --min-cascade-recall 0.95 --target-neg-per-stage 5000 \
    --neg-sample-budget 50000000 --min-stage-negatives 1000 \
    --drop-low-score-pos 0.20 \
    --very-hard-neg-pool weights/19/cbcl__19_v1__vhardneg_raw.npy
mv weights/19/cvj_weights_*.pkl weights/19/celeba_aligned__19_v3.pkl

# 6. Evaluate + tune
python main.py test --data-dir data/19_celeba_aligned \
    --weights-path weights/19/celeba_aligned__19_v3.pkl
python tools/diagnose_cascade.py \
    --weights weights/19/celeba_aligned__19_v3.pkl \
    --data-dir data/19_celeba_aligned
python tools/tune_thresholds.py \
    --weights weights/19/celeba_aligned__19_v3.pkl \
    --data-dir data/19_celeba_aligned --objective f1
python main.py test --data-dir data/19_celeba_aligned \
    --weights-path weights/19/celeba_aligned__19_v3_tuned.pkl

# 7. Face boxes, in-the-wild benchmark, detect
python tools/eval_fddb.py --weights weights/19/celeba_aligned__19_v3_tuned.pkl \
    --skip-opencv --folds 1 --box-fit-folds 1 --save-box-transform
python tools/eval_fddb.py --weights weights/19/celeba_aligned__19_v3_tuned.pkl \
    --cascade default --box-fit-folds 1 --folds 2,3,4,5,6,7,8,9,10 --iou 0.3,0.5
python main.py detect \
    --weights-path weights/19/celeba_aligned__19_v3_tuned.pkl \
    --detect-images images/people.png images/judybats.jpg \
    --detect-output images/outputs/celeba_aligned__19_v3
```
