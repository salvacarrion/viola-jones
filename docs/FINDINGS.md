# Viola-Jones — Findings

Technical notes from building the cascade: design decisions, bugs found and fixed, and lessons that don't fit in the README. For per-run metrics see [RESULTS.md](RESULTS.md).

---

## Why so many knobs? The seven-fix narrative

A naive Viola-Jones — gather faces, mine random negatives, train 4 boosting stages, run NMS — gets F1 ≈ 0.04 on the CBCL benchmark. The paper's F1 ≈ 0.85 came from millions of curated patches, 38 hand-tuned stages, and machine-time orders of magnitude beyond a laptop. This repo bridges the gap with seven small fixes, each tied to a specific failure observed while building it:

| # | Failure observed                                                          | Fix                                                            |
| - | ------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 1 | Train on Caltech negatives → cascade calls 67 % of CBCL non-faces "face"  | Matched-domain seed: `--neg-source mixed`                      |
| 2 | Stage 1 seed alone leaves stages 2+ over-fit to Caltech                   | Stratified mining (50/50 seed/Caltech at every stage)          |
| 3 | 19×19 + CelebA → cascade collapses to 1.7 % recall                        | Either train at 24×24, or use CelebA<sub>aligned</sub> at 19×19 |
| 4 | 1 929 unique CBCL faces is too few for some 9+ stage cascades             | Multi-source: `--face-source celeba_aligned+cbcl`              |
| 5 | Calibration hits the 0.5 cap at deep stages → FPR stays high              | Allow per-stage threshold up to 0.95 (training + post-hoc)     |
| 6 | Same face fires at scales 1×, 1.5×, 2× → boxes on boxes after NMS         | Hybrid NMS: fuse if IoU > threshold **or** IoMin > 0.7         |
| 7 | Hand-tuned `--layers` schedule rigid; first adaptive attempts collapsed   | Calibrated-threshold FPR+recall early-stop (`--target-stage-fpr`) |

The sections below walk through each in detail. Independent of these, three classical V-J techniques are always on: **hard-negative mining**, **per-window variance normalization** ([weakclassifier.py](weakclassifier.py), §5.1 of the paper), and **held-out calibration of stage thresholds**.

### 1. Negative-domain gap

The first attempt — FDDB faces with Caltech-only negatives — collapsed to F1=0.044. Not because the cascade failed to learn faces (recall was 0.78) but because specificity was 0.33: the cascade called 67 % of CBCL non-faces "face".

- Caltech patches are random crops of object photos (textures, edges, uniform regions). Easy to reject — once the cascade learns "not flat, not pure-edge", it's done.
- CBCL non-faces (the benchmark's negative set) were **curated to be face-like**: face-pose backgrounds, near-symmetric features, similar pixel statistics. Caltech-trained cascades have never seen anything close, so they let them through.

**Fix** (`--neg-source mixed`): pull the ~4 500 CBCL non-faces from the HF train split (×2 with h-flip → 9 096) and use them as the stage-1 seed. Specificity jumped from 0.33 → 0.98 in one training run. This single change is the largest improvement of the whole project.

### 2. Stratified mining (not just stage 1)

The matched seed solved stage 1, but the next 8 stages mined from Caltech only. By stage 9, "Stage negatives: 947" appeared in the log: the cascade had memorised Caltech so completely that only 947 patches out of 1 M still passed it as faces. The cascade had drifted to "distinguish faces from Caltech-leftover-edges", and CBCL non-face patterns crept back in unchallenged.

**Fix** ([violajones.py](violajones.py) `_mine_hard_negatives`): when a seed pool is passed, allocate **half** of each stage's mining budget to it and the other half to Caltech. CBCL non-faces stay in the training mix at every stage. Shortfalls backfill from Caltech. Logged as `[seed] mined X / [caltech] mined Y` per stage.

### 3. Pixel alignment at low resolution

The "CelebA + matched seed at 19×19" run (recall 0.017) is the most surprising failure. CelebA faces are frontal, just like CBCL — so why does the cascade not generalise?

At 19×19, **alignment matters at the pixel level**, not just pose. CBCL crops are tightly cropped 19×19 native captures with eyes at row 8, nose at row 10, mouth at row 13. CelebA crops are 48×48 native with looser framing (more forehead, hair, neck); bilinearly downsized to 19×19, the same landmarks land at row 7 / 10 / 13. A 1-pixel shift on a 19-pixel face is ~5 % of the face — Haar features tile rectangle sums at fixed grid positions, so the feature distribution moves entirely. The cascade learned "eyes at row 8"; CelebA test faces have eyes at row 7 and look like non-faces in feature space.

**Fix**: either move to 24×24 (5 px slack ≈ 20 % of the face, enough that bilinear CelebA still hits the same Haar grid as CBCL), or pre-align CelebA at the dataset level so its landmarks match CBCL's positions before downsampling (the `celeba_aligned` HF split). Adding `--jitter 1` or `--jitter 2` further trains the cascade to be mildly translation-invariant.

### 4. Multi-source positives

CBCL ships only 1 929 unique faces in the HF train split. After augment+jitter that's ~7 700 train positives. Enough for an 11-stage cascade at 19×19, but limited by source diversity — the cascade overfits to CBCL's specific lighting and cropping style.

**Fix** (`--face-source celeba_aligned+cbcl`): combine sources. With aligned CelebA, the alignment problem of § 3 is gone; we can mix freely. The multi-source run buys ~5 pp precision in raw scores (better hard-neg rejection) but only ~0.5 pp F1 after tuning, at 3.3× the compute cost — see [RESULTS.md](RESULTS.md#cbcl-vs-mixed-comparison). Worth it at 24×24, marginal at 19×19.

### 5. Calibration was hitting its own ceiling

[adaboost.py](adaboost.py) `_calibrated_threshold()` historically capped per-stage threshold at `0.5` ("never tighten past majority vote"). At 24×24 with a deep cascade, val-pos scores cluster much higher than 0.5 — calibration wanted to push deep stages to 0.6+ to match the val distribution and got clamped instead. The cap was leaving FPR points on the table.

That's also why post-hoc threshold tuning was so effective: the F1=0.357 cascade reached F1=0.653 just by allowing each stage's threshold to move past 0.5 to its actually-optimal point. No new training, no new features — just the cuts that calibration couldn't reach.

**Fix**: raised the cap to 0.95 in [adaboost.py](adaboost.py); also added [tools/tune_thresholds.py](tools/tune_thresholds.py) as a cheap final-mile optimization (`--objective f1` for benchmark, `recall-at-spec` for "find every face" use cases).

### 6. Hybrid NMS for multi-scale duplicates

Sliding-window detection at growth=1.25 fires the same face at adjacent scales. A 24×24 box and a 30×30 box around the same face have IoU ≈ 0.64 (fuses fine), but a 24×24 box inside a 48×48 box has IoU ≈ 0.25 — below the default 0.3 NMS threshold, so they don't fuse, producing "boxes on boxes" stacks.

**Fix** ([utils.py](utils.py) `non_maximum_supression`, `--nms-metric hybrid`): fuse if **either** IoU > threshold **or** IoMin > 0.7, where IoMin = `intersection / min(area1, area2)`. IoMin is 1.0 for nested boxes regardless of scale ratio. The default `mode="weighted"` then merges the cluster into a single score-weighted-average box rather than dropping the smaller-scale detections.

### 7. Adaptive cascade — calibrated FPR + recall

Getting the adaptive trainer right took three attempts:

**Attempt 1 — FPR at threshold 0.5.** First version replaced `--layers` with a per-stage FPR target. The check measured FPR at threshold 0.5 (un-calibrated majority vote):

```python
fpr = (running_neg_scores / sum_alpha_fpr >= 0.5).mean()
if fpr <= target_stage_fpr: break
```

But after each stage finishes, `calibrate()` lowers the deployed threshold to ~0.15-0.20 to keep `--layer-recall` of faces. **The FPR metric was at the wrong threshold.** Symptom: every stage stopped at round 1 — the first stump alone achieves FPR=0.15 ≤ 0.5 at threshold 0.5, so AdaBoost halted. At the actual deployed threshold the stage's recall was only 80 % and FPR was much higher. The cumulative val-recall check killed the cascade after one stage.

**Attempt 2 — `--min-wcs-per-stage` floor.** Workaround: don't let the FPR early-stop fire until at least N weak classifiers are in. Hid the symptom — every stage stopped exactly at the floor — but the metric was still measured at the wrong threshold, and post-stage calibration was driving the threshold to the 0.95 cap because with so few WCs the score is too coarse to meet `target_recall`. Cascade depth was governed by a magic number.

**Attempt 3 (current) — calibrated-threshold FPR + recall, both checked.**

1. Apply the just-chosen weak classifier to val_pos (features pre-computed and cached in `_cache/xf_val.npy`).
2. Recalibrate `self.threshold` so `target_recall` of val_pos passes.
3. Measure FPR on the stage's training negatives **at that threshold**.
4. Early-stop only when `recall ≥ target_recall AND fpr ≤ target_fpr` simultaneously — V&J §3 verbatim.

The two-condition check matters because intra-stage there's a transient where the threshold is still settling: with 1-2 WCs the score has only 2-3 distinct values, the threshold gets clamped at the `min(alphas)/sum(alphas)` floor (effectively 1.0 → capped to 0.95), and recall is artificially below target. The FPR at that clamped threshold looks small but is meaningless — recall isn't honored. Requiring recall ≥ target gates the early-stop until the threshold has actually settled.

`--min-wcs-per-stage` survives as an inert safeguard (default 1).

**Lesson:** when introducing a "stop when metric X is good enough" rule, verify that X is measured at the same operating point as deployment. A metric at the wrong threshold is worse than no metric — it gives confident misleading numbers.

---

## Other bugs found and fixed

### B1. `min_shift=1` in `find_faces` ignored `self.shift=2`

`ViolaJones.__init__` sets `self.shift = 2` but `find_faces(pil_image, growth=None, min_shift=1)` defaulted to 1 and never read `self.shift`. At scale < 2.0, `int(scale)=1 < min_shift=1` so the window stepped every pixel instead of every 2 pixels. **Quadrupled window count.** For a 640×480 image with a 24×24 base window, this generated ~280 K windows at scale=1.0 instead of ~70 K.

**Fix:** `find_faces(..., min_shift=None)` with `if min_shift is None: min_shift = self.shift` at the top of the method. Detection time dropped 4×.

### B2. Stale feature cache after data source change

`data/<res>_<src>/_cache/xf_pos.npy` is written after the first training run and memmapped on reuse. If `train_pos` changes (different source, resolution, or sample count) but the cache exists, the old features are silently reused — producing a mismatch between feature-matrix dimensions and current `train_pos.shape`.

**Fix (workaround):** always `rm -rf data/<res>_<src>/_cache` before retraining with different data. The cache is safe to reuse only when the exact same `prepare_data.py` command was run.

### B3. Jitter leakage across train/val split

Original multi-source implementation called `jitter_crops()` on the full face array then `split_three_way()`. The same underlying face could end up in train (center crop) AND val (jittered crop) — breaking the independence assumption for calibration.

**Fix:** call `split_three_way()` on `faces_unique` (pre-jitter unique face indices), then apply `jitter_crops()` per split separately.

### B4. Asymmetric NMS for nested boxes

Old code used `intersection / smaller-area` as the overlap metric, but the "smaller area" comparison was order-dependent (which box was `idx` vs `remaining` varied with iteration order). A 24×24 box inside a 48×48 box gave different merge decisions depending on which box was processed first.

**Fix:** `metric='hybrid'` (default): fuse if `IoU > threshold OR IoM > 0.7`, where `IoM = intersection / min(area1, area2)`. Symmetric, and catches nested boxes that plain IoU misses. (Same fix as § 6 above — kept as a bug entry for the symmetry issue too.)

### B5. `build_caltech_pool` capped at `n_imgs × patches_per_image`

`--pool-size 50000000` printed "filled 1 195 160 / 50 000 000" and stopped silently. The original `build_caltech_pool` did a single pass over the Caltech image set, extracting at most `patches_per_image=40` random crops per image. With 29 879 Caltech images that's a hard ceiling of ~1.2 M patches — `--pool-size` was effectively a no-op above that.

**Fix:** multi-pass loop. Each pass re-permutes the image order and draws fresh random offsets, so duplicates across passes are negligible (a 300×300 image has ~80 K possible 19×19 positions).

**Lesson:** when a user-facing knob is gated by an internal limit, surface it. The user shouldn't have to read the function to discover that `--pool-size > n_imgs × patches_per_image` silently caps.

### B6. `val_cbcl_pos` only generated when CBCL was among `face_sources`

The original anchor-set fix had a scope bug: `if 'cbcl' in face_sources:`. A single-source CelebA training run had `face_sources = ['celeba_aligned']` → `val_cbcl_pos` never generated → calibration fell back to augmented `val_pos` (CelebA crops, augmented) → val→test gap reopened *for the model that had no CBCL data anywhere*.

**Fix:** drop the guard. Generate `val_cbcl_pos` whenever the benchmark is CBCL, regardless of which faces train sees. (Later subsumed by the architecture cleanup below.)

**Lesson:** calibration anchor is a property of *the benchmark*, not of the training source. Coupling the two via a "happens to be there" implementation quirk created the false security of "the fix is applied — manifest just says 0, must be an edge case".

### B7. Cumulative val-recall checked on `val_pos`, not the calibration set

Per-stage thresholds were tuned to keep `--layer-recall=0.99` of `val_cbcl_pos` (un-augmented) passing. But the global cascade-recall check counted survivors over the augmented `val_pos` (4× larger with jitter+flip). The shifted variants scored systematically below the calibrated threshold, so each stage dropped cumulative `val_pos` recall by ~8–10 pp instead of ~1 pp.

```
Stage 1: cum val recall 0.956 (should have been ~0.99)
Stage 2: 0.913
Stage 3: 0.902
Stage 4: 0.890 ← below --min-cascade-recall 0.90, stops
```

**Fix:** measure cumulative recall on the same set the per-stage thresholds calibrate against. Consistency restored.

**Lesson:** when there are two "val" sets, every metric that involves a positive pass-rate must declare which one it uses, and they must align. A calibration/validation split where the criteria measure different distributions gives oscillating signals.

### B8. `_mine_from_pool` random memmap access thrashes on >RAM pools

When `caltech_pool.npy` exceeds RAM (50 M patches at 19×19 = ~18 GB, machine has 16 GB), the original random-index sampling triggered a page fault per access — each ~4 KB OS page holds ~11 contiguous patches but the random index only uses 1 of them. Page cache thrashed, swap filled, throughput dropped from ~30 K patches/s to ~8 K patches/s.

**Fix:** chunked sequential traversal. Random order over chunks of 50 K patches (~18 MB each), sequential within a chunk. OS prefetch loves sequential reads on memmap. Random chunk order preserves unbiasedness.

`.copy()` on found patches is load-bearing: without it, the kept patch is a numpy view into the chunk → keeps the whole chunk alive → defeats the chunked-release pattern → memory leak proportional to `#chunks × chunk_size`.

**Lesson:** memmap is not free. Access pattern matters as much as size. Test this by `du -h` of the pool vs free RAM before assuming the old code path works.

### B9. Bash script with `set -euo pipefail` exits silently on empty glob

`scripts/run_19_experiments.sh` returned to the prompt immediately with zero output. The culprit:

```bash
set -euo pipefail
...
stale=$(ls weights/19/cvj_weights_*.pkl 2>/dev/null | wc -l | tr -d ' ')
```

When the glob has no matches, `ls` exits with code 1. `2>/dev/null` only redirects stderr — it doesn't change the exit code. `pipefail` propagates the non-zero exit through the pipeline. `set -e` then kills the script. All before any `echo`.

**Fix:** `shopt -s nullglob` + bash array. Empty globs expand to empty arrays instead of the literal pattern, no `ls` invocation, no pipefail trap.

**Lesson:** under `set -euo pipefail`, every shell glob that can match nothing needs `nullglob` or an explicit `|| true` clause.

---

## Architecture cleanup (2026-05-17)

The codebase had accumulated three coupled concepts that confused everyone: `val_pos` (augmented, internal split), `val_cbcl_pos` (un-augmented anchor, partial coverage), and `cbcl_test_*` (hardcoded benchmark name). Refactored to a single coherent design:

| Before                                | After                                              |
| ------------------------------------- | -------------------------------------------------- |
| `cbcl_test_pos.npy`, `cbcl_test_neg.npy` | `test_pos.npy`, `test_neg.npy`                  |
| `cbcl_neg_seed.npy`                   | `neg_seed.npy`                                     |
| `val_cbcl_pos.npy` (partial)          | (dropped — merged into `val_pos`)                  |
| `val_pos` augmented (968 samples)     | `val_pos` raw, 500 samples, anchored to benchmark  |
| `test_pos.npy` (10 % slice, unused)   | (no internal test split — benchmark IS the test)   |
| `--train-frac 0.8 --val-frac 0.1`     | `--benchmark cbcl --val-size 500`                  |
| `--neg-source caltech\|cbcl\|mixed`   | `--neg-source caltech\|benchmark\|mixed`           |
| `seed_neg_pool`, `val_cal_pos` params | `neg_seed` (single param, single purpose)          |

The new mental model:

```
train_pos    : from --face-source, may be augmented/jittered
val_pos      : from --benchmark, un-augmented, center crop —
               does BOTH per-stage threshold calibration AND the
               cumulative cascade-recall stop criterion
test_pos/neg : from --benchmark test split (untouched)
```

Key benefits:

1. No wasted training data — the old 10 % test slice was generated and saved but never read; `main.py test` always used the separate HF test split.
2. No leakage when face-source includes the benchmark — val indices are reserved before training samples are drawn, then excluded from the training pool.
3. One val concept, used consistently — per-stage threshold tuning and the stop criterion measure the same distribution at the same operating point.
4. Benchmark is a flag, not a hardcoded path — adding FDDB or WIDER FACE later needs ingestion at the HF dataset level only.

---

## Hard-negative mining as a separate tool

Added [tools/mine_hard_negatives.py](tools/mine_hard_negatives.py) to pre-mine hard negatives from a trained cascade, save them to `weights/<res>/<weights-stem>__hardneg.npy`, and feed them into a subsequent training run via `main.py train --hard-neg-pool <path>`.

**Standalone tool, not integrated into prepare_data.** A hard-neg pool is a model-derived artifact — it changes every time the cascade that mined it changes. `prepare_data.py` outputs are meant to be deterministic from raw HF data + flags. Coupling the two would force a full data re-bundle every time the reference model changes (~30 min wasted per iteration). The standalone tool also puts model lineage in the filename (`cvj_weights_1778801054__hardneg.npy`) — you can tell which cascade produced a given hard-neg pool just by looking at it.

**Single-pass design.** Mining iterates `caltech_pool.npy` once, classifies every patch, keeps the ones the cascade misclassifies as faces. Budget = pool size. If the output is smaller than wanted, the fix is to regenerate `caltech_pool.npy` with `--pool-size` 5–10× bigger, *not* to crank a budget knob. `classify(patch)` is deterministic so re-sampling buys nothing.

---

## 19×19 capacity ceiling: jitter saturates the cascade

Empirical observation comparing three runs at 19×19 (CBCL benchmark, 5 K target negatives per stage):

| Run            | jitter | T per stage (early → late)                                |
| -------------- | :----: | --------------------------------------------------------- |
| A              |   0    | 4, 5, 9, 26, 31, 25, 35, 65, 68, 123, 156, 150, 188       |
| B (CBCL@19, jitter=2) | 2 | 8, 7, 47, 326, **500** (cap), 500, 500, 500, ...     |

Run A converged cleanly for 13 stages at `FPR ≤ 0.5` each. Run B hit a cliff at stage 5 — every subsequent stage saturated at the max-WC cap without satisfying the FPR target.

**Diagnosis.** At 19×19 there are 17 268 distinct Haar features. With `--jitter 2` the trained weak classifiers must be tolerant to ±2 px shift (each positive appears as both center crop AND a random shifted variant). To accommodate the shifted variant's lower native scores, each WC's threshold has to "open up" — which lets through hard-negs that look like slightly-shifted faces. By stage 5 the hard-negs from mining are face-like enough that no Haar feature can simultaneously (a) tolerate ±2 px shift on positives AND (b) reject these negatives. The cascade hits an information ceiling.

**Practical bound at 19×19:**

- `jitter=0`: cascade converges to 12–15 stages; brittle to inference-time misalignment (sliding window step=2 misses faces that land off-grid).
- `jitter=1`: midpoint — ~11 stages converging, partial alignment robustness.
- `jitter=2`: cascade saturates after stage 4 at 19×19; not viable.

**24×24 should change this.** 78 K features (vs 17 K) gives roughly 4× more discriminative variety; the shift-tolerance vs hard-neg-rejection trade-off should fit within the model's capacity.

**Lesson:** augmentation interacts with model capacity. At small resolutions the Haar feature space is the bottleneck — adding training diversity (jitter, augment) can exceed the model's ability to represent the resulting decision boundary. The symptom is "stages cap without converging" rather than "model overfits" — a different failure mode that doesn't show up in cross-validation because the cascade structure itself degrades.

### v2 extension — where the ceiling actually sits

The pending experiment listed in RESULTS.md ("19×19 CBCL extended stages") was run by resuming each v1 baseline from its capped final stage with `--max-wcs-per-stage 800 --target-stage-fpr 0.65` (everything else identical to v1). Two distinct regimes appeared:

- **Multi-source runs (CBCL, mixed)** *did* extend cleanly. CBCL went 11 → 15 stages, mixed went 11 → 16. Each new stage early-stopped at FPR ≈ 0.63–0.65 (just below the relaxed target) using 344–778 weak classifiers per stage — roughly 5× more than the equivalent depth in v1, but still finite. Tuned F1 lifted +2.2 to +2.4 pp on both runs (0.634 → 0.658 cbcl; 0.639 → 0.661 mixed), now ahead of the historical 24×24 best.
- **CelebA-aligned-only stayed broken.** v1 capped at stage 3 (FPR=0.598). v2 reached stage 6 but every added stage saturated against the WC cap progressively earlier — stage 6 trained the full 800 WCs and *still* couldn't get FPR below 0.713. The capacity-ceiling break fired again. No amount of `target_stage_fpr` slack rescues a run where the feature space cannot separate the matched-domain hard negatives from the (unanchored, ±1 px jittered) CelebA positives in the first place.

**What the v2 extension actually demonstrated.** The 19×19 ceiling is not a single number — it's a curve. With *matched-domain anchored* positives (CBCL alone or mixed), the cascade can keep adding stages past FPR=0.5 if you raise the cap; each new stage just costs more weak classifiers (~400-800 vs ~150-400 at v1 depth). Beyond that the diminishing return is real: stages 12–16 cost ~25 h of wall time across both runs combined for +2 pp F1. Without anchored positives, the cascade hits the *real* ceiling at stage ~3–6 regardless of how much WC budget you allow — the feature space genuinely cannot represent the decision boundary.

**Practical takeaway:** if a stage caps with `final_fpr > 0.5`, check whether the run has CBCL-style anchored positives before extending. A v1 → v2 resume buys you +2 pp F1 on the well-anchored configurations; on the CelebA-only configuration it just postpones the same failure to a deeper stage.

### v3 filtered — does positive curation rescue the CelebA ceiling? (no.)

The natural follow-up to v2: if CelebA-only saturates because the *aligned* dataset still carries residual misaligned crops, then dropping the worst crops should free the cascade. Built the curation pipeline ([WORKFLOW.md](WORKFLOW.md) §4): score every CelebA-aligned positive with a frozen `cbcl__19_v2` oracle (cumulative AdaBoost margin), drop the bottom 61% (matches CBCL's positive count: ~7700 augmented entries = ~2000 unique faces, vs CBCL's 1929 unique). Pre-mine a 15K very-hard-neg reservoir from the same oracle via the streaming raw miner (`tools/mine_hard_negatives_raw.py`, 326M patches sampled at 0.0046% FPR) to top up late-stage mining shortfalls. Train with identical hyperparameters to cbcl v1.

**Result:** still capped at stage 3, with FPR=0.622 at the same 400-WC budget as cbcl v1's stage 11 (which converged at FPR ≈ 0.47). Tuned F1=0.603 vs cbcl v1's 0.634 — close enough to call "paridad" at the operating-point level, but only because the F1 tuner does heroic work over the 3 trained stages: thresholds collapse to `[0.30, 0.44, 0.52]` and recall is sacrificed (0.917 → 0.504) for precision (0.056 → 0.751). cbcl v1 reaches its 0.634 with 11 stages of natural rejection power; CelebA filtered v3 reaches 0.603 by squeezing every drop out of 3 stages plus tuning. **The very-hard reservoir was never tapped** — training stopped before mining became hard enough to trigger a shortfall.

**What v3 actually proves.** Curation is not the missing piece. The structural delta is the feature space, not the data quality. Visual inspection of the `score_samples.png` grid (the bottom 30% of CelebA-aligned scores cluster on visibly off-axis crops) had suggested filtering would close the gap. It didn't, because *even the cleanest 2000 unique CelebA faces* exceed what 17K Haar features can discriminate against face-like hard negatives while tolerating ±1 px jitter. The capacity ceiling is the resolution, not the dataset.

**Two side effects worth keeping.** First, the curation infrastructure works: F1 went from 0.542 (v1 unfiltered) → 0.603 (v3 filtered) — a real +6 pp lift that confirms the tooling produces a measurable improvement, just not enough to break the ceiling. Second, the streaming raw miner (`tools/mine_hard_negatives_raw.py`) is materially better than the pre-built-pool version for late-stage cascades: 15K very-hard patches at 0.0046% FPR is a usable reservoir, where the legacy pool-based miner against the same `cbcl__19_v2` would have yielded ~5K from a finite 50M pool. Both tools transfer cleanly to 24×24.

**Lesson:** when a capacity-bound configuration shows reasonable per-stage early-stops in the first 1-2 stages, curating the positives can buy a small lift to the trained operating point — but the post-tuning F1 ceiling is set by the cascade *depth*, which the feature space caps. If the cascade saturates at stage 3 with FPR ≈ 0.6, curation alone won't get it to stage 12. The fix is upstream: more features (24×24), not better positives.

---

## What worked vs what didn't

| Change                                          | Effect                                          | Why                                                              |
| ----------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| Matched-domain CBCL seed (stage 1)              | Spec 0.33 → 0.98                                | Training negatives matched test distribution                     |
| 24×24 resolution                                | Enables CelebA, solves alignment                | 5 px slack vs 1 px at 19×19                                      |
| `celeba_aligned` HF split                       | Makes CelebA usable at 19×19                    | Pre-aligned landmarks match CBCL's Haar grid                     |
| Stratified mining (CBCL+Caltech, all stages)    | Prevents specificity drift in deep stages       | Keeps matched-domain negatives throughout                        |
| Post-hoc threshold tuning                       | F1 0.357 → 0.653 on the same 24×24 model         | Bypasses the old 0.5 calibration cap                             |
| Raising calibration cap to 0.95                 | Bakes correct thresholds during training        | Deep stages can now calibrate above 0.5                          |
| Hybrid NMS (IoU OR IoM)                         | Eliminates "boxes on boxes"                     | IoM handles nested multi-scale duplicates                        |
| Score-weighted average NMS                      | Tighter box localization                        | Fuses cluster into weighted centroid                             |
| Adaptive cascade (calibrated FPR+recall)        | Removes the magic `--layers` schedule           | Depth and per-stage WC count emerge from the data                |
| Fix `min_shift` bug                             | 4× faster detection                             | Actually uses `self.shift=2` as intended                         |
| CelebA-only at 19×19 (raw, no alignment)        | Recall 0.017                                    | Pixel alignment mismatch with CBCL benchmark                     |
| CelebA-only at 19×19 (aligned, jitter=2)        | Cascade caps at 3 stages, F1=0.113              | Variety exceeds Haar-feature capacity                            |
| CelebA-only at 19×19 (aligned, filtered drop 0.61) | Still caps at 3 stages, tuned F1=0.603       | Curation lifts operating point +6 pp but doesn't break ceiling   |
| Adding stages to a saturated cascade            | Makes recall worse                              | More rejection on already-broken positives = lower recall        |
| H-flip augmentation                             | Marginal                                        | Symmetric Haar features partially already invariant to it        |
| Positive curation (`--drop-low-score-pos`)      | +6 pp F1 on capacity-bound runs                 | Removes residual misaligned crops; doesn't fix the feature cap   |
| Streaming raw very-hard mining                  | 15K hard negs at 0.0046% FPR (vs ~5K from pool) | No finite intermediate pool — scans HF raw until target reached  |

---

## Suggestions evaluated but not adopted

### `min_w=1, min_h=1` in `build_features`

Adding 1–3 px Haar features would expand the feature set from ~60 K to ~500 K at 24×24.

**Not worth it.** Tiny features are noise at this resolution — a 1×2 Haar on a 24×24 face doesn't encode meaningful structure. Training time scales linearly in feature count; 10× more features means 10× longer training per round. The original paper used `min_size=1` but their 24×24 window was 400-dpi film scans, not bilinearly downsampled 48×48 crops. Stick with `min_w=4, min_h=4`.
