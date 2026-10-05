# Viola-Jones face detector

From-scratch NumPy implementation of [Viola & Jones (2001)](https://doi.org/10.1109/CVPR.2001.990517): Haar-like features, integral image, AdaBoost, and an attentional cascade with hard-negative mining. The detector itself uses no OpenCV.

| This repo (best model, ships in `weights/`) | OpenCV's pretrained `haarcascade_frontalface_default` |
| :-: | :-: |
| ![ours on people.png](images/outputs/best/people_detected.png) | ![OpenCV on people.png](images/outputs/opencv_default/people_detected.png) |
| ![ours on judybats.jpg](images/outputs/best/judybats_detected.png) | ![OpenCV on judybats.jpg](images/outputs/opencv_default/judybats_detected.png) |

On clean portraits like these both find every face (OpenCV misses the woman on the right). On a large in-the-wild benchmark OpenCV is clearly more precise: see [Results](#in-the-wild-detection-fddb) and [Limitations](#limitations-and-next-steps). Left: `python main.py detect --detect-min-score 0.8`; right: `python tools/baseline_opencv.py detect`.

## Highlights

- **Pure NumPy**, every piece built from scratch: integral image, Haar features, AdaBoost stumps, cascade, multi-scale sliding window, NMS.
- **Adaptive trainer** (paper §3): stage depth and weak-classifier count both emerge from a two-condition early-stop (per-round recall plus FPR at the calibrated operating point), not from hard-coded sizes.
- **Vectorized inference**: an image pyramid (the native 24×24 window on a downsized image, as OpenCV does), one integral image per scale, and the whole cascade runs as a batched NumPy reduction over surviving windows.
- **Honest benchmarking**: scored on the CBCL patch set and on in-the-wild FDDB, with [OpenCV cascades as an external baseline](docs/OPENCV_COMPARISON_FINDINGS.md), including a native NumPy port of OpenCV's pretrained cascade that runs inside this same pipeline.

## Install

```bash
git clone https://github.com/salvacarrion/viola-jones.git
cd viola-jones
pip install -r requirements.txt
```

The training data is auto-downloaded on first run from the [`salvacarrion/face-detection`](https://huggingface.co/datasets/salvacarrion/face-detection) HuggingFace dataset.

## Quickstart

### Try the pretrained detector

The trained cascades ship with the repo under `weights/` (a few hundred KB each), so no data download or training is needed. The best one, `weights/24/celeba_aligned__24_v2_s11_tuned.pkl`, is used by default:

```bash
python main.py detect --detect-images path/to/photo.jpg --detect-min-score 0.8
# -> images/outputs/photo_detected.png
```

`--detect-min-score 0.8` drops low-confidence boxes (the sample outputs in `images/outputs/best/` use it); pick another model with `--weights-path weights/<res>/<model>.pkl`. See [Limitations](#limitations-and-next-steps) for what it cannot detect.

### Train your own

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

| Resolution | Faces                               | Version | Stages      |   F1  | F1<sub>tuned</sub> | Train (approx)<sup>†</sup> | FDDB AP@0.5<sup>‡</sup> |
| :--------: | :---------------------------------- | :-----: | :---------: | :---: | :----------------: | :------------------------: | :--------------------: |
|   19×19    | CBCL                                |   v1    |     11      | 0.570 |       0.634        |            ~1 h            |         0.459          |
|   19×19    | CBCL                                | **v2**  |     15      | 0.619 |     **0.658**      |            ~4 h            |         0.475          |
|   19×19    | CelebA<sub>aligned</sub>            |   v1    | 3 (capped)  | 0.113 |       0.542        |            ~1 h            |         0.310          |
|   19×19    | CelebA<sub>aligned</sub>            |   v2    | 6 (capped)  | 0.195 |       0.550        |            ~4 h            |         0.362          |
|   19×19    | CelebA<sub>aligned</sub> (filtered) |   v3    | 3 (capped)  | 0.106 |       0.603        |           ~0.5 h           |         0.439          |
|   19×19    | CelebA<sub>aligned</sub>+CBCL       |   v1    |     11      | 0.596 |       0.639        |            ~4 h            |         0.472          |
|   19×19    | CelebA<sub>aligned</sub>+CBCL       | **v2**  |     16      | 0.628 |     **0.661**      |           ~11 h            |         0.483          |
|   24×24    | CBCL (smoke test)                   |  smoke  | 10 (capped) | 0.505 |       0.660        |            ~5 h            |       **0.503**        |
|   24×24    | CelebA<sub>aligned</sub>            |   v1    | 9 (capped)  | 0.521 |       0.629        |           ~31 h            |         0.469          |
|   24×24    | CelebA<sub>aligned</sub> ⭐          | **v2**  |     11      | 0.571 |     **0.661**      |           ~95 h            |         0.471          |

⭐ **Project best: `weights/24/celeba_aligned__24_v2_s11_tuned.pkl`** (tuned recall 0.625, specificity 0.995, precision 0.701, F1 0.661). CelebA-only caps at 3 stages at 19×19 but trains an 11-stage cascade at 24×24, which confirms the resolution hypothesis. The benchmark F1 understates it: the test set is CBCL, which this model never trains on. On in-the-wild FDDB it has the best AP@0.3 (0.589) and recall of all our models; the 24×24 CBCL smoke model trades recall for precision and edges it at IoU 0.5 (0.503 vs 0.471).

<sup>‡</sup> In-the-wild average precision on FDDB folds 2-10 at IoU 0.5, tuned model, boxes mapped to FDDB's face-box convention (see below). OpenCV's `default` cascade scores 0.726 under the same protocol.

<sup>†</sup> Times are approximate and normalized to the `--precompute-sort-index` regime, which is ~5x faster than the original runs (the 24×24 v2 deepening dropped round time from ~280 to ~50 s/round). Raw measured wall-clock and per-stage diagnostics are in [docs/RESULTS.md](docs/RESULTS.md).

### In-the-wild detection (FDDB)

Full-image detection on [FDDB](http://vis-www.cs.umass.edu/fddb/), IoU-matched against ground truth. This is the fair common ground with OpenCV, since both detectors slide over the same images. FDDB is a public benchmark of 2845 news photos with 5171 annotated faces (not shipped here; [download instructions](docs/OPENCV_COMPARISON_FINDINGS.md#data-provenance)). Each detector's boxes are mapped to FDDB's face-box convention by a transform fitted on fold 1, and every number is on the 9 held-out folds 2-10 (2555 images, 4656 faces).

| Detector | AP@0.5 | R@0.5 | P@0.5 | AP@0.3 | R@0.3 | P@0.3 |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| OpenCV `alt` (cv2) | 0.731 | 0.734 | 0.926 | 0.739 | 0.742 | 0.936 |
| OpenCV `default` (cv2) | 0.726 | 0.742 | 0.756 | 0.737 | 0.753 | 0.767 |
| OpenCV `default` (our native port) | 0.703 | 0.720 | 0.637 | 0.730 | 0.745 | 0.659 |
| **Ours 24×24 CelebA v2 ⭐** | **0.471** | 0.596 | 0.250 | **0.589** | 0.716 | 0.300 |
| Ours ⭐ as first evaluated (legacy pyramid, raw boxes) | 0.000 | 0.034 | 0.005 | 0.069 | 0.279 | 0.045 |

**OpenCV is still ahead, but by much less than it first looked.** Until the end of the project this table showed AP@0.3 0.069 for our best model. A post-mortem found two problems on our side, neither of them in training. Our boxes follow the tight eyes-to-mouth crop the cascade was trained on, so even a perfect detection scored IoU ≈ 0.33 against FDDB's forehead-to-chin boxes and was counted as a false positive. And the sliding-window pyramid scaled the Haar rectangles (with integer truncation) instead of the image, which let ~3.5× more background through. Fixing both, without retraining, gives the ⭐ row: recall within a few points of OpenCV, precision still far behind. That residual gap is the trained cascade itself: its negatives were object photos and CBCL patches, not scenes with people, so it lets ~18× more background windows through than OpenCV's. [docs/RETRAINING.md](docs/RETRAINING.md) lists what to change; the full analysis and the per-model numbers are in [docs/OPENCV_COMPARISON_FINDINGS.md](docs/OPENCV_COMPARISON_FINDINGS.md).

**On its own benchmark the relation flips:** on tight CBCL crops our best model reaches F1 0.661 and OpenCV 0.000, since its cascade needs a context margin the crops do not have.

Before (left) and after (right) the two fixes on a held-out FDDB image; ground truth in green, ours in red, OpenCV `default` in blue:

![Before/after on FDDB](images/outputs/fddb_comparison/2002_08_02_big_img_314.png)

### OpenCV baseline and native port

OpenCV's pretrained cascades double as the external baseline above and can be converted into a native model that runs inside this pipeline (pure NumPy, no cv2 at inference):

```bash
python tools/baseline_opencv.py detect --images images/people.png          # run cv2 directly
python tools/convert_opencv_cascade.py --cascade default                   # -> weights/24/opencv_default.pkl
python main.py detect --weights-path weights/24/opencv_default.pkl --detect-min-score 150
```

The port reproduces OpenCV's `default` cascade with 100% window-level parity (`alt`: 99.97%). `alt2` and `alt_tree` use CART trees instead of stumps and are not supported.

## Limitations and next steps

- **Too many false positives in cluttered scenes.** On FDDB the best model finds almost as many faces as OpenCV (recall 0.72 vs 0.75 at IoU 0.3), but only 1 in 4 of its boxes is a face at IoU 0.5 (OpenCV: 3 in 4). Its training negatives were object photos (Caltech-256) and face-like patches (CBCL), never scenes with people, so clothes, hands and crowds get through.
- **Small faces.** The cascade looks at a 24×24 eyes-to-mouth window, so faces narrower than ~35 px are missed (see `clase.png` and `physics.jpg` in `images/outputs/`). Upscale small images first.
- **Frontal faces only.** Like any Viola-Jones cascade trained on aligned frontal faces, profiles and strongly rotated faces are missed.

**Going further needs retraining**, mainly with hard negatives mined from real face-free scenes and positives cropped with some context around the face. A simulation on FDDB suggests this could lift AP@0.5 from 0.47 to roughly 0.6 to 0.7, close to OpenCV's 0.73, which is about the ceiling for this kind of detector. [docs/RETRAINING.md](docs/RETRAINING.md) has the ranked list of changes, code pointers and the estimate.

## Repo layout

- `main.py`: CLI for `train` / `test` / `detect`.
- `violajones.py`, `adaboost.py`, `weakclassifier.py`, `features.py`, `utils.py`: the detector.
- `opencv_cascade.py`: native NumPy evaluator for an OpenCV cascade.
- `tools/`: data prep, threshold tuning, per-stage diagnostics, hard-negative mining, OpenCV baseline (`baseline_opencv.py`), FDDB evaluation and box fitting (`eval_fddb.py`), FDDB figures (`draw_fddb_comparison.py`), cascade conversion (`convert_opencv_cascade.py`), and the `reeval.sh` runner that regenerates every number in the tables ([results_reeval.txt](results_reeval.txt)).

## Docs

- [docs/FINDINGS.md](docs/FINDINGS.md): the technical narrative behind each design choice.
- [docs/RESULTS.md](docs/RESULTS.md): full experimental log with per-stage diagnostics and raw timings.
- [docs/OPENCV_COMPARISON_FINDINGS.md](docs/OPENCV_COMPARISON_FINDINGS.md): the OpenCV baseline, FDDB benchmark, and native port.
- [docs/WORKFLOW.md](docs/WORKFLOW.md): data-prep and training recipes (Spanish).
- [docs/RETRAINING.md](docs/RETRAINING.md): what limits the current models in the wild and what to change if you retrain.

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
