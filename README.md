# Viola-Jones face detector

From-scratch NumPy implementation of [Viola & Jones (2001)](https://doi.org/10.1109/CVPR.2001.990517): Haar-like features, integral image, AdaBoost, and an attentional cascade with hard-negative mining. The detector itself uses no OpenCV.

![Detection example](images/outputs/best/judybats_detected.png)

## Highlights

- **Pure NumPy**, every piece built from scratch: integral image, Haar features, AdaBoost stumps, cascade, multi-scale sliding window, NMS.
- **Adaptive trainer** (paper §3): stage depth and weak-classifier count both emerge from a two-condition early-stop (per-round recall plus FPR at the calibrated operating point), not from hard-coded sizes.
- **Vectorized inference**: one integral image per scale, the whole cascade runs as a batched NumPy reduction over surviving windows.
- **Honest benchmarking**: scored on the CBCL patch set and on in-the-wild FDDB, with [OpenCV cascades as an external baseline](docs/OPENCV_COMPARISON_FINDINGS.md), including a native NumPy port of OpenCV's pretrained cascade that runs inside this same pipeline.

## Install

```bash
git clone https://github.com/salvacarrion/viola-jones.git
cd viola-jones
pip install -r requirements.txt
```

The training data is auto-downloaded on first run from the [`salvacarrion/face-detection`](https://huggingface.co/datasets/salvacarrion/face-detection) HuggingFace dataset.

## Quickstart

```bash
# 1. Prepare data (downloads + caches the dataset on first run)
python tools/prepare_data.py --face-source cbcl --neg-source mixed --resolution 19 --augment

# 2. Train the cascade (quick recipe, a few stages)
python main.py train --data-dir data/19_cbcl --max-stages 6 --max-wcs-per-stage 100

# 3. Evaluate on the CBCL benchmark
python main.py test --data-dir data/19_cbcl

# 4. Run detection on images
python main.py detect --detect-images images/people.png
```

Post-hoc threshold tuning (no retraining, only moves the per-stage cut points):

```bash
python tools/tune_thresholds.py --weights weights/19/cbcl__19_v1.pkl --data-dir data/19_cbcl --objective f1
```

See [docs/WORKFLOW.md](docs/WORKFLOW.md) for the full data-prep and training recipes.

## Results

### CBCL patch benchmark

Per-patch face / non-face classification on the CBCL benchmark (472 faces, 23 573 non-faces). F1<sub>tuned</sub> is the best F1 after post-hoc threshold tuning on the same model. Only canonical models are listed.

| Resolution | Faces                               | Version | Stages      |   F1  | F1<sub>tuned</sub> | Train (approx)<sup>†</sup> |
| :--------: | :---------------------------------- | :-----: | :---------: | :---: | :----------------: | :------------------------: |
|   19×19    | CBCL                                |   v1    |     11      | 0.570 |       0.634        |            ~1 h            |
|   19×19    | CBCL                                | **v2**  |     15      | 0.619 |     **0.658**      |            ~4 h            |
|   19×19    | CelebA<sub>aligned</sub>            |   v1    | 3 (capped)  | 0.113 |       0.542        |            ~1 h            |
|   19×19    | CelebA<sub>aligned</sub>            |   v2    | 6 (capped)  | 0.195 |       0.550        |            ~4 h            |
|   19×19    | CelebA<sub>aligned</sub> (filtered) |   v3    | 3 (capped)  | 0.106 |       0.603        |           ~0.5 h           |
|   19×19    | CelebA<sub>aligned</sub>+CBCL       |   v1    |     11      | 0.596 |       0.639        |            ~4 h            |
|   19×19    | CelebA<sub>aligned</sub>+CBCL       | **v2**  |     16      | 0.628 |     **0.661**      |           ~11 h            |
|   24×24    | CBCL (smoke test)                   |  smoke  | 10 (capped) | 0.505 |       0.660        |            ~5 h            |
|   24×24    | CelebA<sub>aligned</sub>            |   v1    | 9 (capped)  | 0.521 |       0.629        |           ~31 h            |
|   24×24    | CelebA<sub>aligned</sub> ⭐          | **v2**  |     11      | 0.571 |     **0.661**      |           ~95 h            |

⭐ **Project best: `weights/24/celeba_aligned__24_v2_s11_tuned.pkl`** (tuned recall 0.625, specificity 0.995, precision 0.701, F1 0.661). CelebA-only caps at 3 stages at 19×19 but trains an 11-stage cascade at 24×24, which confirms the resolution hypothesis. The benchmark F1 understates it: the test set is CBCL, which this model never trains on, yet on real images it produces the cleanest, most diverse detections of any model here.

<sup>†</sup> Times are approximate and normalized to the `--precompute-sort-index` regime, which is ~5x faster than the original runs (the 24×24 v2 deepening dropped round time from ~280 to ~50 s/round). Raw measured wall-clock and per-stage diagnostics are in [docs/RESULTS.md](docs/RESULTS.md).

### In-the-wild detection (FDDB)

Full-image detection on [FDDB](http://vis-www.cs.umass.edu/fddb/) fold 1 (290 images, 515 faces), IoU-matched against ground truth. This is the fair common ground with OpenCV, since both detectors slide over the same images. AP and recall are reported at IoU 0.5 and at the more lenient 0.3.

| Detector                                | AP@0.5 | R@0.5 | P@0.5 | AP@0.3 | R@0.3 |
| :-------------------------------------- | :----: | :---: | :---: | :----: | :---: |
| OpenCV `alt` (cv2)                      | 0.654  | 0.674 | 0.853 | 0.734  | 0.738 |
| OpenCV `default` (cv2)                  | 0.665  | 0.689 | 0.683 | 0.733  | 0.750 |
| OpenCV `default` (our native port)      | 0.599  | 0.639 | 0.573 | 0.719  | 0.736 |
| Ours 24×24 CelebA v2 ⭐ (min-face 80)    | 0.010  | 0.089 | 0.095 | 0.298  | 0.408 |
| Ours 24×24 CelebA v2 ⭐ (min-face 40)    | 0.001  | 0.033 | 0.005 | 0.073  | 0.272 |
| Ours 19×19 CelebA+CBCL v2               | 0.000  | 0.016 | 0.002 | 0.002  | 0.091 |

**Each detector wins on the domain it was trained for.** On tight CBCL crops our best model reaches F1 0.661 while OpenCV scores 0.000 (its cascade needs a context margin the crops do not have). On in-the-wild FDDB the relationship flips: OpenCV, trained on news photos of the same kind, reaches AP 0.67, while our CelebA-aligned model trails. Two things hold our model back on FDDB: a false-positive flood from negatives that were never in-the-wild scenes, and a box convention tighter than FDDB's ellipse boxes. Raising `--detect-min-face` to 80 removes the small-scale false positives (FDDB has almost no tiny faces) and lifts AP@0.3 from 0.07 to 0.30.

Our best model (red) vs OpenCV default (blue), ground truth in green:

![Ours vs OpenCV](images/outputs/opencv_comparison/2002_08_02_big_img_1231_best.png)

Full analysis, protocol, and the native-port parity check are in [docs/OPENCV_COMPARISON_FINDINGS.md](docs/OPENCV_COMPARISON_FINDINGS.md).

### OpenCV baseline and native port

OpenCV's pretrained cascades double as the external baseline above and can be converted into a native model that runs inside this pipeline (pure NumPy, no cv2 at inference):

```bash
python tools/baseline_opencv.py detect --images images/people.png          # run cv2 directly
python tools/convert_opencv_cascade.py --cascade default                   # -> weights/24/opencv_default.pkl
python main.py detect --weights-path weights/24/opencv_default.pkl --detect-min-score 150
```

The port reproduces OpenCV's `default` cascade with 100% window-level parity (`alt`: 99.97%). `alt2` and `alt_tree` use CART trees instead of stumps and are not supported.

## Repo layout

- `main.py`: CLI for `train` / `test` / `detect`.
- `violajones.py`, `adaboost.py`, `weakclassifier.py`, `features.py`, `utils.py`: the detector.
- `opencv_cascade.py`: native NumPy evaluator for an OpenCV cascade.
- `tools/`: data prep, threshold tuning, per-stage diagnostics, hard-negative mining, OpenCV baseline (`baseline_opencv.py`), FDDB evaluation (`eval_fddb.py`), cascade conversion (`convert_opencv_cascade.py`), and the `reeval.sh` benchmark runner.

## Docs

- [docs/FINDINGS.md](docs/FINDINGS.md): the technical narrative behind each design choice.
- [docs/RESULTS.md](docs/RESULTS.md): full experimental log with per-stage diagnostics and raw timings.
- [docs/OPENCV_COMPARISON_FINDINGS.md](docs/OPENCV_COMPARISON_FINDINGS.md): the OpenCV baseline, FDDB benchmark, and native port.
- [docs/WORKFLOW.md](docs/WORKFLOW.md): data-prep and training recipes.

## Citation

```bibtex
@inproceedings{viola2001rapid,
  author    = {Viola, Paul and Jones, Michael},
  title     = {Rapid object detection using a boosted cascade of simple features},
  booktitle = {Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2001},
  volume    = {1},
  pages     = {I--I},
  doi       = {10.1109/CVPR.2001.990517},
}
```
