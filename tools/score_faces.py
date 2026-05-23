"""
Score every positive in a prepared data-dir with a frozen oracle cascade.

Produces `<data-dir>/face_scores.npy` (float32, one score per train_pos sample
in the same order). The score is the **cumulative AdaBoost margin** summed
across ALL stages without short-circuiting on rejection:

    score(face) = Σ_stage (stage.score(face) - stage.threshold)

Faces that the cascade clears with comfortable margin accumulate big positive
sums. Faces that the cascade rejects early accumulate negative contributions
from the failing stages and rank low. This is the signal we want for
data-quality filtering: at the bottom of the ranking sit the crops that look
least like canonical aligned-frontal faces — typically alignment failures or
mis-cropped CelebA samples — which AdaBoost would otherwise burn capacity
trying to fit.

Decoupled from the HF dataset on purpose: the score is model-derived (changes
every time the oracle cascade changes), so it lives at the prepared-data
layer next to train_pos.npy, not in the HF dataset itself.

Wire into training with:
    python main.py train --data-dir <dir> --drop-low-score-pos 0.20 ...

Usage:
    python tools/score_faces.py \\
        --weights weights/19/cbcl__19_v2.pkl \\
        --data-dir data/19_celeba_aligned
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_pretty_time, integral_image  # noqa: E402
from violajones import ViolaJones  # noqa: E402


def score_faces(clf: ViolaJones, faces: np.ndarray):
    """Score each face two ways and return both signals.

    Returns:
        scores: float32 cumulative AdaBoost margin summed across ALL stages
            (no short-circuit). Use for ranking faces by oracle confidence.
            **Sign is a proxy, not a verdict** — a face can have positive
            cumulative margin overall yet still be rejected at one
            intermediate stage. For an exact pass/fail use `passed`.
        passed: bool array; True iff the cascade's classify() accepts the
            face (i.e. EVERY stage's margin ≥ 0 at deployment thresholds).
    """
    assert clf.base_width == faces.shape[1] and clf.base_height == faces.shape[2], (
        f"Cascade trained at {clf.base_width}×{clf.base_height}, "
        f"faces are {faces.shape[1]}×{faces.shape[2]}. Resolution mismatch.")
    scores = np.empty(len(faces), dtype=np.float32)
    passed = np.empty(len(faces), dtype=bool)
    for i, face in enumerate(tqdm(faces, desc="Scoring", unit="face")):
        ii = integral_image(face)
        std = max(float(np.std(face)), 1.0)
        total = 0.0
        all_pass = True
        for stage in clf.clfs:
            s = stage.score(ii, std=std)
            total += s - stage.threshold
            if s < stage.threshold:
                all_pass = False
        scores[i] = total
        passed[i] = all_pass
    return scores, passed


def _load_font(size: int):
    """Return a scalable font using ONLY assets Pillow ships with itself.

    Pillow >= 10.1 ships DejaVuSans bundled and exposes it via
    `ImageFont.load_default(size=N)` (scalable TTF, no system deps).
    Older Pillow versions fall back to a fixed bitmap default — text will
    look chunkier but the script still works. No platform-specific paths
    here on purpose: this tool must be portable across mac/linux/windows.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def save_percentile_grid(faces: np.ndarray, scores: np.ndarray,
                         passed: np.ndarray, out_path: str,
                         n_per_band: int = 16, cell_px: int = 128):
    """Save one PNG with rows = percentile bands and cols = sample faces.

    Bands chosen to expose the bottom (where filtering candidates live)
    and the top (sanity-check on what the cascade considers canonical).
    Faces are 19×19 → upsampled with LANCZOS for cleaner edges; row labels
    and per-face score readouts use a TrueType font so text doesn't go
    pixelated when zoomed in.
    """
    bands = [
        ("p00-p05  (worst)",  0.00, 0.05),
        ("p05-p10",           0.05, 0.10),
        ("p10-p20",           0.10, 0.20),
        ("p20-p30",           0.20, 0.30),
        ("p45-p55  (median)", 0.45, 0.55),
        ("p70-p80",           0.70, 0.80),
        ("p90-p95",           0.90, 0.95),
        ("p95-p100 (best)",   0.95, 1.00),
    ]
    n = len(scores)
    order = np.argsort(scores)
    rng = np.random.default_rng(0)

    pad = 6
    label_w = 280
    grid_w = label_w + n_per_band * (cell_px + pad) + pad
    grid_h = len(bands) * (cell_px + pad) + pad
    canvas = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)

    label_font = _load_font(size=max(16, cell_px // 7))
    score_font = _load_font(size=max(12, cell_px // 10))

    for row, (label, lo, hi) in enumerate(bands):
        i_lo = int(round(lo * n))
        i_hi = int(round(hi * n))
        band_idxs = order[i_lo:i_hi]
        if len(band_idxs) == 0:
            continue
        take = min(n_per_band, len(band_idxs))
        pick = rng.choice(len(band_idxs), size=take, replace=False)
        chosen = band_idxs[pick]
        y = pad + row * (cell_px + pad)
        # Row label vertically centered
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_h = bbox[3] - bbox[1]
        draw.text((12, y + (cell_px - text_h) // 2), label,
                  fill=(230, 230, 230), font=label_font)
        for col, idx in enumerate(chosen):
            face = faces[idx]
            img = Image.fromarray(face, mode="L").convert("RGB")
            # LANCZOS upsamples 19×19 → 128×128 with smoother edges than
            # NEAREST. Faces stay clearly recognizable, no boxy artifacts.
            img = img.resize((cell_px, cell_px), Image.LANCZOS)
            border = (220, 60, 60) if not passed[idx] else (80, 200, 80)
            x = label_w + col * (cell_px + pad)
            canvas.paste(img, (x, y))
            draw.rectangle([x - 2, y - 2, x + cell_px + 1, y + cell_px + 1],
                           outline=border, width=2)
            # Per-face score: bottom-left, yellow on dark stripe for legibility.
            txt = f"{scores[idx]:+.2f}"
            tb = draw.textbbox((0, 0), txt, font=score_font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            sx = x + 3
            sy = y + cell_px - th - 4
            draw.rectangle([sx - 2, sy - 1, sx + tw + 2, sy + th + 2],
                           fill=(0, 0, 0, 180))
            draw.text((sx, sy), txt, fill=(255, 240, 80), font=score_font)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    canvas.save(out_path)
    print(f"Saved samples grid -> {out_path}")
    print(f"  Size: {grid_w}×{grid_h} px, cell {cell_px}px (LANCZOS).")
    print("  Green border = cascade passes; red = cascade rejects.")
    print("  Yellow number = cumulative-margin score.")


def _summary(scores: np.ndarray, passed: np.ndarray):
    n = len(scores)
    if n == 0:
        print("\tEmpty array — nothing to summarize.")
        return
    qs = np.quantile(scores, [0.0, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 1.0])
    labels = ["min", "p05", "p10", "p20", "p50", "p80", "p90", "p95", "max"]
    print("\tCumulative-margin distribution (ranking signal):")
    for lab, v in zip(labels, qs):
        print(f"\t  {lab:>3} = {v:+.4f}")
    n_pass = int(passed.sum())
    n_fail = n - n_pass
    n_neg = int((scores < 0).sum())
    print(f"\n\tCascade verdict (deterministic, classify() == 1):")
    print(f"\t  passed:    {n_pass:,}/{n:,}  ({100*n_pass/n:.1f}%)")
    print(f"\t  rejected:  {n_fail:,}/{n:,}  ({100*n_fail/n:.1f}%)")
    print(f"\n\tNote: score < 0 → {n_neg:,} ({100*n_neg/n:.1f}%); "
          f"this overlaps with but isn't identical to 'rejected'. Score is "
          f"the ranking signal; pass/fail is the verdict.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True,
                    help="Path to the oracle cascade .pkl. Its resolution "
                         "must match the data-dir.")
    ap.add_argument("--data-dir", required=True,
                    help="Prepared data directory (contains train_pos.npy).")
    ap.add_argument("--target", default="train_pos",
                    help="Which NPY to score (default: train_pos). Can also "
                         "be 'val_pos' or 'test_pos' for diagnostic runs.")
    ap.add_argument("--out", default=None,
                    help="Output .npy path. Default: "
                         "<data-dir>/face_scores.npy (or "
                         "<data-dir>/<target>_scores.npy for non-default "
                         "targets).")
    ap.add_argument("--save-samples", type=int, default=0, metavar="N",
                    help="If > 0, also save a PNG grid with N example "
                         "faces per percentile band (worst, p10, p20, "
                         "median, p80, p95, best). Default 0 (off). "
                         "Useful for eyeballing the bottom-of-distribution "
                         "before fixing a --drop-low-score-pos value.")
    ap.add_argument("--samples-out", default=None,
                    help="PNG path for the percentile-samples grid. "
                         "Default: <data-dir>/score_samples.png.")
    args = ap.parse_args()

    print(f"Loading cascade: {args.weights}")
    clf = ViolaJones.load(args.weights)
    print(f"\t- {len(clf.clfs)} stages, base {clf.base_width}×{clf.base_height}")

    target_path = os.path.join(args.data_dir, f"{args.target}.npy")
    if not os.path.exists(target_path):
        ap.error(f"Target NPY not found: {target_path}")
    print(f"Loading faces: {target_path}")
    faces = np.load(target_path, mmap_mode="r")
    print(f"\t- {len(faces):,} faces @ {faces.shape[1]}×{faces.shape[2]}")

    print("\nScoring (cumulative margin across all stages, no short-circuit)...")
    start = time.time()
    scores, passed = score_faces(clf, faces)
    print(f"Done in {get_pretty_time(start)}")

    print("\nSummary:")
    _summary(scores, passed)

    if args.out:
        out_path = args.out
    else:
        if args.target == "train_pos":
            out_path = os.path.join(args.data_dir, "face_scores.npy")
        else:
            out_path = os.path.join(args.data_dir, f"{args.target}_scores.npy")
    np.save(out_path, scores)
    print(f"\nSaved -> {out_path}")

    if args.save_samples > 0:
        samples_path = args.samples_out or os.path.join(
            args.data_dir, "score_samples.png")
        # Materialize the faces array fully (was memmap'd) for indexing.
        faces_full = np.asarray(faces)
        save_percentile_grid(faces_full, scores, passed, samples_path,
                             n_per_band=args.save_samples)

    print(f"\nWire into training:\n"
          f"  python main.py train --data-dir {args.data_dir} "
          f"--drop-low-score-pos 0.20 ...")


if __name__ == "__main__":
    main()
