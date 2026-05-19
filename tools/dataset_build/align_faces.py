"""
Landmark-based face alignment pipeline.

Reads the HF face-detection parquet files, detects facial landmarks with
MediaPipe Face Mesh, warps each face so the eyes land on a canonical
position, crops to a fixed output resolution, and saves the result
per-source as both NPY (for training) and a PNG grid (for visual
inspection).

Source parquets expected at --data-dir:
    train-00000-of-00001.parquet   (celeba + fddb, label==1)
    test-00000-of-00001.parquet    (cbcl, label==1 are faces)

Per-source outputs land in --out-dir as:
    aligned_<source>.npy           (N, H, W) uint8 grayscale, aligned
    aligned_<source>_preview.png   visual grid of the first 400 faces
    aligned_<source>_failed.png    grid of source crops we COULDN'T align
                                   (low confidence or pose out of range)

Filtering: faces are rejected if MediaPipe doesn't find one, if the
detected roll exceeds --max-roll (head tilt), or if the eye-distance
is implausibly small (likely false detection on a non-face patch).

Designed for the 48×48 source crops we get from the HF dataset: each
input is upsampled with INTER_CUBIC to UPSAMPLE_SIZE for landmark
detection, then the affine warp is computed in upsampled coordinates
and re-scaled back to the original frame before the final crop. This
avoids cascading interpolations.
"""

import argparse
import io
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyarrow.parquet as pq
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from tqdm.auto import tqdm

# MediaPipe Tasks API requires an explicit model file. The legacy
# `mp.solutions.face_mesh` API is gone in 0.10.x on some wheels, so we use
# the same underlying FaceMesh model via the Tasks loader.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "_models" / "face_landmarker.task"


def ensure_landmarker_model(path):
    """Download face_landmarker.task on first use; cache in tools/_models/."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FaceLandmarker model to {path} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(path))
    print(f"  saved ({path.stat().st_size // 1024} KB)")
    return path

# ---- Canonical geometry (in OUTPUT_SIZE pixel coordinates) -----------------
# Defaults are chosen at 48×48 to match the native CBCL 19×19 framing
# upsampled. CelebA/FDDB sources are 48×48, so alignment at 48×48 is
# lossless from those sources; CBCL passthrough is a 2.5× upsample of the
# native 19×19 (same blur as any other resize from 19×19). Saving at 48×48
# lets prepare_data.py / training pick any smaller window (24, 28, 32) at
# load time with a single, cheap downsample.
DEFAULT_OUTPUT_SIZE = 48
DEFAULT_EYE_Y = 10.0
DEFAULT_EYE_SPACING = 24.0  # horizontal distance between eye centers

# MediaPipe FaceMesh indices for eye corners. Averaging outer+inner gives a
# stable eye-center, less noisy than a single landmark.
LEFT_EYE_IDXS  = [33, 133]
RIGHT_EYE_IDXS = [362, 263]
# Mouth-center landmarks: top of upper lip + bottom of lower lip → vertical
# center of the lips; mouth corners (61, 291) keep the horizontal center
# stable when subjects smile/talk. Averaging four points damps individual
# variation more than picking a single landmark.
MOUTH_IDXS = [13, 14, 61, 291]

# 48×48 inputs are too small for the FaceMesh detector to find landmarks
# reliably. Upsample to this size before detection — bicubic, single pass.
UPSAMPLE_SIZE = 192


def align_face(img_pil, landmarker, dst_pts, output_size,
               max_roll_deg=15.0, min_eye_dist_frac=0.15):
    """
    Align one face to canonical output_size×output_size geometry.

    `dst_pts` is a (2|3, 2) array of canonical landmark positions in the
    output crop: row 0 = left eye, row 1 = right eye, optional row 2 =
    mouth center (all in pixel coords relative to `output_size`).

    Returns:
        (aligned, status) where:
          aligned is an OUTPUT_SIZE×OUTPUT_SIZE uint8 grayscale array on
            success, else None.
          status is one of: 'ok', 'no_face', 'roll_too_high', 'tiny_eyes'.
    """
    rgb = np.array(img_pil.convert('RGB'))
    h0, w0 = rgb.shape[:2]

    # Detection scale-up: MediaPipe wants something nearer to 100×100+.
    side = max(h0, w0)
    if side < UPSAMPLE_SIZE:
        scale = UPSAMPLE_SIZE / side
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
        rgb_big = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        rgb_big = rgb
        scale = 1.0

    # Tasks API: wrap the numpy array in mp.Image, then call detect().
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_big)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, 'no_face'

    lms = result.face_landmarks[0]
    H, W = rgb_big.shape[:2]

    def mean_xy(idxs):
        xs = np.mean([lms[i].x for i in idxs]) * W
        ys = np.mean([lms[i].y for i in idxs]) * H
        return np.array([xs, ys], dtype=np.float32)

    left_eye_big  = mean_xy(LEFT_EYE_IDXS)
    right_eye_big = mean_xy(RIGHT_EYE_IDXS)
    mouth_big     = mean_xy(MOUTH_IDXS) if dst_pts.shape[0] >= 3 else None

    # Sanity checks against the upsampled frame, not the source — robust to
    # whatever resolution the source came in at.
    eye_dist = np.linalg.norm(right_eye_big - left_eye_big)
    if eye_dist < min_eye_dist_frac * max(H, W):
        return None, 'tiny_eyes'

    # Head roll = angle of the eye line vs horizontal. >max_roll → not frontal.
    dy = right_eye_big[1] - left_eye_big[1]
    dx = right_eye_big[0] - left_eye_big[0]
    roll = np.degrees(np.arctan2(dy, dx))
    if abs(roll) > max_roll_deg:
        return None, 'roll_too_high'

    # Map landmark coords back to the original source image, then build the
    # similarity transform straight from source pixels to canonical 24×24.
    # When dst_pts has 3 rows we also feed the mouth — overdetermined system
    # solved by least-squares, which redistributes residuals so the mouth/
    # chin position is consistent across faces even when eye-to-mouth
    # ratios differ between individuals.
    left_src  = left_eye_big  / scale
    right_src = right_eye_big / scale
    src_list = [left_src, right_src]
    if mouth_big is not None:
        src_list.append(mouth_big / scale)
    src_pts = np.stack(src_list, axis=0).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        return None, 'no_face'

    gray = np.array(img_pil.convert('L'))
    aligned = cv2.warpAffine(gray, M, (output_size, output_size),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned.astype(np.uint8), 'ok'


def make_grid(faces, output_size, cols=20, pad=1, bg=255):
    """Build a single-image grid of face crops for visual inspection."""
    n = len(faces)
    if n == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)
    rows = (n + cols - 1) // cols
    H = rows * (output_size + pad) - pad
    W = cols * (output_size + pad) - pad
    grid = np.full((H, W), bg, dtype=np.uint8)
    for i, face in enumerate(faces):
        r, c = divmod(i, cols)
        y = r * (output_size + pad)
        x = c * (output_size + pad)
        grid[y:y + output_size, x:x + output_size] = face
    return grid


def process_source(parquet_path, source, landmarker, out_dir, dst_pts,
                   output_size, limit=None, max_roll_deg=15.0,
                   preview_n=400, passthrough=False):
    """Iterate the parquet, align everything from `source`, write outputs.

    When `passthrough=True`, skip landmark detection and just resize the
    source crop to OUTPUT_SIZE — appropriate for sources that ship in their
    own canonical alignment already (e.g. CBCL at 19×19, where landmark
    detection is unreliable but the crops are pre-aligned by hand).
    """
    pf = pq.ParquetFile(parquet_path)
    aligned = []
    failed_thumbs = []  # original crops we COULDN'T align (only in mesh mode)
    counts = {'ok': 0, 'no_face': 0, 'roll_too_high': 0, 'tiny_eyes': 0}

    total = pf.metadata.num_rows
    desc = f"scan {source}" + (" (passthrough)" if passthrough else "")
    pbar = tqdm(total=total, desc=desc, unit='img', leave=True)
    seen_source = 0

    for batch in pf.iter_batches(batch_size=1000):
        srcs   = batch.column('source').to_pylist()
        labels = batch.column('label').to_pylist()
        imgs   = batch.column('image').to_pylist()
        for src, lbl, ib in zip(srcs, labels, imgs):
            pbar.update(1)
            if src != source or lbl != 1:
                continue
            if limit is not None and seen_source >= limit:
                break
            seen_source += 1
            img = Image.open(io.BytesIO(ib['bytes']))
            if passthrough:
                # Trust the source's native alignment; just resize to output_size.
                gray = np.array(img.convert('L'))
                face = cv2.resize(gray, (output_size, output_size),
                                  interpolation=cv2.INTER_CUBIC).astype(np.uint8)
                counts['ok'] += 1
                aligned.append(face)
            else:
                face, status = align_face(img, landmarker, dst_pts,
                                          output_size,
                                          max_roll_deg=max_roll_deg)
                counts[status] += 1
                if status == 'ok':
                    aligned.append(face)
                elif len(failed_thumbs) < preview_n:
                    failed_thumbs.append(np.array(
                        img.convert('L').resize((output_size, output_size),
                                                Image.BICUBIC), dtype=np.uint8))
        if limit is not None and seen_source >= limit:
            break
    pbar.close()

    total_seen = sum(counts.values())
    if total_seen == 0:
        print(f"  {source}: no rows matched (label==1).")
        return
    print(f"  {source}: {counts['ok']:,} aligned / {total_seen:,} scanned "
          f"({100 * counts['ok'] / total_seen:.1f}%)")
    print(f"    rejects — no_face: {counts['no_face']:,}, "
          f"roll>{max_roll_deg:g}°: {counts['roll_too_high']:,}, "
          f"tiny_eyes: {counts['tiny_eyes']:,}")

    if not aligned:
        return

    arr = np.stack(aligned, axis=0)
    npy_path = out_dir / f"aligned_{source}.npy"
    np.save(npy_path, arr)
    print(f"    NPY -> {npy_path}")

    # Visual previews — 400 faces in a 20×20 grid by default.
    n_prev = min(preview_n, len(arr))
    ok_grid = make_grid(list(arr[:n_prev]), output_size, cols=20)
    ok_path = out_dir / f"aligned_{source}_preview.png"
    Image.fromarray(ok_grid).save(ok_path)
    print(f"    preview ({n_prev}) -> {ok_path}")

    if failed_thumbs:
        fail_grid = make_grid(failed_thumbs, output_size, cols=20)
        fail_path = out_dir / f"aligned_{source}_failed.png"
        Image.fromarray(fail_grid).save(fail_path)
        print(f"    failed-source thumbs ({len(failed_thumbs)}) -> {fail_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/Users/salvacarrion/Desktop/data",
                    help="Directory holding the HF parquet files.")
    ap.add_argument("--out-dir", default="data/aligned",
                    help="Output directory for NPY + preview PNGs.")
    ap.add_argument("--sources", nargs="+",
                    default=["celeba", "fddb", "cbcl"],
                    help="Which sources to align. celeba+fddb come from "
                         "train-*.parquet, cbcl from test-*.parquet.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Per-source cap for quick test runs.")
    ap.add_argument("--max-roll", type=float, default=15.0,
                    help="Reject faces with head-roll > this (degrees). "
                         "VJ Haar features are not rotation-invariant.")
    ap.add_argument("--preview-n", type=int, default=400,
                    help="Faces per preview grid (default 400 = 20×20).")
    ap.add_argument("--min-detection-confidence", type=float, default=0.5,
                    help="MediaPipe FaceLandmarker detection confidence.")
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH),
                    help="Path to face_landmarker.task. Auto-downloaded on "
                         "first run if missing.")
    ap.add_argument("--passthrough-sources", nargs="+", default=["cbcl"],
                    help="Sources to resize-without-landmark-alignment. Use "
                         "for sources that are already hand-aligned at a small "
                         "native resolution where MediaPipe can't detect "
                         "landmarks (default: cbcl).")
    ap.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE,
                    help=f"Aligned crop resolution in pixels (default: "
                         f"{DEFAULT_OUTPUT_SIZE}). Match the source resolution "
                         f"(48 for CelebA/FDDB) to avoid information loss; "
                         f"downsample later at training time. Note: "
                         f"--eye-spacing/--eye-y/--mouth-y are interpreted in "
                         f"THIS resolution, so re-measure with "
                         f"tools/measure_cbcl_geometry.py --output-size if "
                         f"you change it.")
    ap.add_argument("--eye-y", type=float, default=DEFAULT_EYE_Y,
                    help=f"Vertical pixel coord of eye centers in the output "
                         f"crop (default: {DEFAULT_EYE_Y}). Lower = eyes "
                         f"higher in the frame = more chin/mouth visible.")
    ap.add_argument("--eye-spacing", type=float, default=DEFAULT_EYE_SPACING,
                    help=f"Horizontal distance between eye centers in pixels "
                         f"(default: {DEFAULT_EYE_SPACING}). Larger = tighter "
                         f"face framing; smaller = looser with hair/context.")
    ap.add_argument("--mouth-y", type=float, default=None,
                    help="Vertical pixel coord of the mouth center. When set, "
                         "uses 3-point alignment (eyes + mouth) via "
                         "least-squares similarity transform — pins the "
                         "mouth/chin position across individuals with "
                         "differing face proportions. Omit for the 2-point "
                         "(eyes-only) transform.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = ensure_landmarker_model(Path(args.model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=args.min_detection_confidence,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # Build canonical destination points centered horizontally in the crop.
    output_size = args.output_size
    cx = output_size / 2.0
    half = args.eye_spacing / 2.0
    dst_rows = [
        [cx - half, args.eye_y],   # left eye
        [cx + half, args.eye_y],   # right eye
    ]
    if args.mouth_y is not None:
        dst_rows.append([cx, args.mouth_y])  # mouth (horizontally centered)
    dst_pts = np.array(dst_rows, dtype=np.float32)
    mode_tag = "3-point" if args.mouth_y is not None else "2-point"
    print(f"Output {output_size}×{output_size}, {mode_tag} alignment:")
    print(f"  left_eye  = {tuple(dst_pts[0])}")
    print(f"  right_eye = {tuple(dst_pts[1])}")
    if args.mouth_y is not None:
        print(f"  mouth     = {tuple(dst_pts[2])}")

    train_path = data_dir / "train-00000-of-00001.parquet"
    test_path  = data_dir / "test-00000-of-00001.parquet"

    source_to_parquet = {
        'celeba': train_path,
        'fddb':   train_path,
        'cbcl':   test_path,
    }

    for source in args.sources:
        pq_path = source_to_parquet.get(source)
        if pq_path is None or not pq_path.exists():
            print(f"  {source}: parquet not found ({pq_path}), skipping.")
            continue
        print(f"\n=== {source} ===")
        process_source(pq_path, source, landmarker, out_dir, dst_pts,
                       output_size,
                       limit=args.limit, max_roll_deg=args.max_roll,
                       preview_n=args.preview_n,
                       passthrough=(source in args.passthrough_sources))

    landmarker.close()
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
