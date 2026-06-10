# Viola-Jones face detector

From-scratch NumPy implementation of [Viola & Jones (2001)](https://doi.org/10.1109/CVPR.2001.990517): Haar-like features + integral image + AdaBoost + attentional cascade with hard-negative mining.

![Detection example](images/outputs/best/judybats_detected.png)

## Installation

```bash
git clone https://github.com/salvacarrion/viola-jones.git
cd viola-jones
pip install -r requirements.txt
```

The training data is auto-downloaded on first run from the [`salvacarrion/face-detection`](https://huggingface.co/datasets/salvacarrion/face-detection) HuggingFace dataset.

## Usage

```bash
# 1. Prepare data (downloads + caches dataset on first run)
python tools/prepare_data.py --face-source cbcl --neg-source mixed --resolution 19 --augment

# 2. Train the cascade (~15 min for the quick recipe)
python main.py train --data-dir data/19_cbcl --max-stages 6 --max-wcs-per-stage 100

# 3. Evaluate on the CBCL benchmark
python main.py test --data-dir data/19_cbcl

# 4. Run detection on images
python main.py detect --detect-images images/people.png
```

Optional post-hoc threshold tuning (no retraining — only moves per-stage cut points):

```bash
python tools/tune_thresholds.py --weights weights/19/cbcl__19.pkl --data-dir data/19_cbcl --objective f1
```

## Results

All metrics on the **CBCL benchmark** (472 faces / 23 573 non-faces). F1<sub>tuned</sub> = best F1 after post-hoc threshold tuning on the same model.

| Resolution | Faces                         | Version | Stages       | Recall | Specificity | Precision |   F1  | F1<sub>tuned</sub> | Train time |
| :--------: | :---------------------------- | :-----: | :----------: | :----: | :---------: | :-------: | :---: | :----------------: | :--------: |
|   19×19    | CBCL                          |   v1    |      11      | 0.657  |    0.987    |   0.503   | 0.570 |       0.634        |    ~6 h    |
|   19×19    | CBCL                          | **v2**  |      15      | 0.600  |    0.993    |   0.639   | 0.619 |     **0.658**      | +14 h 43 m |
|   19×19    | CelebA<sub>aligned</sub>      |   v1    |  3 (capped)  | 0.949  |    0.702    |   0.060   | 0.113 |       0.542        |    ~5 h    |
|   19×19    | CelebA<sub>aligned</sub>      | **v2**  |  6 (capped)  | 0.890  |    0.855    |   0.110   | 0.195 |       0.550        | +13 h 12 m |
|   19×19    | CelebA<sub>aligned</sub> (filtered drop 0.61) | **v3** | 3 (capped) | 0.917 | 0.692 | 0.056 | 0.106 | 0.603 | ~2.2 h |
|   19×19    | CelebA<sub>aligned</sub>+CBCL |   v1    |      11      | 0.653  |    0.989    |   0.548   | 0.596 |       0.639        |   ~20 h    |
|   19×19    | CelebA<sub>aligned</sub>+CBCL | **v2**  |      16      | 0.574  |    0.995    |   0.693   | 0.628 |     **0.661**      | +35 h 19 m |
|   24×24    | CBCL (smoke test)             |  smoke  |  10 (capped) | 0.750  |    0.976    |   0.380   | 0.505 |       0.660        |    ~25 h   |
|   24×24    | CelebA<sub>aligned</sub>      | **v1**  |  9 (capped)  | 0.712  |    0.980    |   0.411   | 0.521 |     **0.629**      |   ~156 h   |
|   24×24    | CelebA<sub>aligned</sub>+CBCL |   TBD   |     TBD      |   —    |      —      |     —     |   —   |         —          |    TBD     |

Shared hyperparameters (19×19 baselines, v1): `--max-stages 20 --max-wcs-per-stage 400 --target-stage-fpr 0.5 --min-cascade-recall 0.95 --target-neg-per-stage 5000 --neg-sample-budget 50000000 --augment --jitter 1`. v2 resumes from v1 with the cap relaxed: `--max-wcs-per-stage 800 --target-stage-fpr 0.65` (everything else unchanged). The v2 tuned 19×19 models now beat the historical 24×24 best (F1=0.653). v3 (CelebA filtered) uses v1 hyperparameters plus the curation pipeline (`--drop-low-score-pos 0.61 --very-hard-neg-pool ...`) — confirmed that positive curation alone doesn't break the 19×19 capacity ceiling.

**24×24 confirms the resolution hypothesis.** CelebA-only — which capped at 3 stages at 19×19 — trains a 9-stage cascade at 24×24 (`--n-faces 10000 --jitter 1 --target-neg-per-stage 10000 --target-stage-fpr 0.5 --drop-low-score-pos 0.02`). Its tuned F1 (0.629) sits ~3 pp under cbcl-on-cbcl (0.660) purely because the benchmark *is* CBCL and this model never sees CBCL faces in training; on real images it produces the cleanest, most diverse detections of any model here. Note the cost: 156 h (deep stages dominate — stage 9 alone was 71 h). Post-hoc tuning lifted it +11 pp (0.521 → 0.629), far cheaper than any extra training.

See [docs/RESULTS.md](docs/RESULTS.md) for per-stage diagnostics and the full experimental log, and [docs/FINDINGS.md](docs/FINDINGS.md) for the technical narrative behind each design choice.

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
