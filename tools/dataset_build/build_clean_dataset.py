"""
Build a paired (clean + aligned) face dataset from raw sources.

For every face that passes quality filtering we emit TWO 48×48 grayscale
crops at the same index:

  *_clean.npy    — loose portrait framing (eyes upper-third, hair / jaw
                   visible). The "general-purpose face crop" replacement
                   for the existing parquet `celeba` / `fddb` rows.

  *_aligned.npy  — tight VJ-style framing (eyes near the top edge, mouth
                   near the bottom). Geometry measured from CBCL so train
                   and test share the same canonical face placement.

Per-source pipelines:

  CelebA  → reads raw 178×218 JPGs from --celeba-dir and the manual
            landmarks file (--celeba-landmarks). No MediaPipe — every
            face has perfect landmarks by construction. Stops at
            --n-celeba faces that pass quality filters (roll < 15°,
            yaw < 0.20 of inter-ocular distance, sane eye distance).

  CBCL    → loads CBCL train faces from --cbcl-parquet. No raw
            available — emits `cbcl_clean` = `cbcl_aligned` = the
            19×19 native crop resized to 48×48 with INTER_CUBIC. Also
            extracts the label==0 rows separately as `cbcl_negatives`
            for the matched-domain seed.

  FDDB    → opt-in only via `--sources fddb`. Parses COCO --fddb-coco;
            for each bbox crops a padded region from the full-res image
            and runs MediaPipe FaceLandmarker. Yields ~150-200 faces from
            ~40K bboxes (most COCO entries are null-bboxed or fail
            detection). Not included in the default sources because the
            yield is too low to be useful for VJ training.

Outputs into --out-dir:
  <src>_clean.npy / <src>_aligned.npy        paired arrays (N, 48, 48) u8
  <src>_clean_preview.png / <src>_aligned_preview.png    400-face grids
"""

import argparse
import io
import json
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

# ---- Canonical destination geometries (both at 48×48) ----------------------
# CLEAN: loose, hair + jaw visible — comparable to a typical face thumbnail.
# ALIGNED: tight, CBCL-style — face fills the frame, eyebrows near top.
# Only two anchor points (the two eyes) → exactly-determined similarity
# transform. A 3rd anchor (e.g. the mouth) over-constrains the 4-DOF
# similarity system and the least-squares compromise looks like apparent
# "stretching" when source/target proportions disagree, even though the
# warp itself preserves shape. Mouth position is then determined by the
# individual face anatomy, not pinned.
OUTPUT_SIZE = 48
CLEAN_LEFT_EYE   = (15.0, 18.0)
CLEAN_RIGHT_EYE  = (33.0, 18.0)
ALIGNED_LEFT_EYE  = (12.0, 10.0)
ALIGNED_RIGHT_EYE = (36.0, 10.0)

MAX_ROLL_DEG = 15.0
# Yaw rejection: nose horizontal offset from the eye-line midpoint, divided
# by the inter-eye distance. Frontal faces sit near 0; 3/4 profile is ~0.25;
# strict profile is >0.5. We allow mild off-center (the manual landmarker
# isn't pixel-perfect) but reject anything past 3/4 profile.
MAX_YAW_RATIO = 0.20
MIN_EYE_DIST_PX = 8.0  # in the source frame; rejects micro-faces

# MediaPipe model cache (for FDDB landmark detection).
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "_models" / "face_landmarker.task"

# MediaPipe FaceMesh indices.
MP_LEFT_EYE  = [33, 133]
MP_RIGHT_EYE = [362, 263]
MP_NOSE_TIP  = 1  # single landmark; FaceMesh's most stable nose anchor


def ensure_landmarker_model(path):
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FaceLandmarker model to {path} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(path))
    return path


def _warp_to_canonical(gray, src_pts, dst_pts):
    """Single similarity warp from source frame to OUTPUT_SIZE×OUTPUT_SIZE."""
    M, _ = cv2.estimateAffinePartial2D(
        src_pts.astype(np.float32), dst_pts.astype(np.float32),
        method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(gray, M, (OUTPUT_SIZE, OUTPUT_SIZE),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _quality_ok(left_eye, right_eye, nose):
    """Reject by head roll, yaw (frontality), and implausibly small eye distance.

    Yaw heuristic: project the nose onto the eye line and measure the offset
    from the midpoint between the eyes, normalized by inter-eye distance.
    Frontal faces sit close to 0; 3/4 profile crosses ~0.25; full profile
    blows past 0.5. We reject above MAX_YAW_RATIO.
    """
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    eye_dist = float(np.hypot(dx, dy))
    if eye_dist < MIN_EYE_DIST_PX:
        return False, 'tiny_eyes'
    roll = abs(np.degrees(np.arctan2(dy, dx)))
    if roll > MAX_ROLL_DEG:
        return False, 'roll_too_high'
    mid_x = (left_eye[0] + right_eye[0]) / 2.0
    mid_y = (left_eye[1] + right_eye[1]) / 2.0
    # Project nose displacement onto the unit vector along the eye line.
    ux, uy = dx / eye_dist, dy / eye_dist
    nx, ny = nose[0] - mid_x, nose[1] - mid_y
    nose_along_eyes = ux * nx + uy * ny
    yaw_ratio = abs(nose_along_eyes) / eye_dist
    if yaw_ratio > MAX_YAW_RATIO:
        return False, 'profile'
    return True, 'ok'


def _emit_pair(gray, left_eye, right_eye):
    """Warp the source face twice (clean + aligned) using only the two eyes.

    Exactly-determined similarity (4 DOF, 2 correspondences → 4 equations)
    → unique solution, zero LS compromise, face shape preserved 100%.
    Mouth/chin position in the output is whatever the individual anatomy
    dictates — we trust the rest of the face to follow the eyes.
    """
    src = np.stack([left_eye, right_eye], axis=0)
    clean_dst   = np.stack([CLEAN_LEFT_EYE,   CLEAN_RIGHT_EYE])
    aligned_dst = np.stack([ALIGNED_LEFT_EYE, ALIGNED_RIGHT_EYE])
    clean   = _warp_to_canonical(gray, src, clean_dst)
    aligned = _warp_to_canonical(gray, src, aligned_dst)
    return clean, aligned


# ---- CelebA ----------------------------------------------------------------
def parse_celeba_landmarks(txt_path):
    """Yield (filename, left_eye, right_eye, nose) from CelebA file.

    File format (whitespace-separated):
        filename lefteye_x lefteye_y righteye_x righteye_y
                 nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
    """
    with open(txt_path) as f:
        next(f)             # row count
        next(f)             # header
        for line in f:
            parts = line.split()
            fname = parts[0]
            le   = (float(parts[1]), float(parts[2]))
            re   = (float(parts[3]), float(parts[4]))
            nose = (float(parts[5]), float(parts[6]))
            yield fname, le, re, nose


def process_celeba(celeba_dir, landmarks_path, n_target):
    """N CelebA faces with provided landmarks → (clean, aligned) NPYs."""
    clean_list, aligned_list = [], []
    counts = {'ok': 0, 'roll_too_high': 0, 'profile': 0,
              'tiny_eyes': 0, 'load_fail': 0}
    entries = list(parse_celeba_landmarks(landmarks_path))
    pbar = tqdm(entries, desc='celeba', unit='img')
    for fname, le, re, nose in pbar:
        if counts['ok'] >= n_target:
            break
        ok, reason = _quality_ok(le, re, nose)
        if not ok:
            counts[reason] += 1
            continue
        img_path = celeba_dir / fname
        try:
            gray = np.array(Image.open(img_path).convert('L'))
        except Exception:
            counts['load_fail'] += 1
            continue
        clean, aligned = _emit_pair(gray, le, re)
        if clean is None or aligned is None:
            counts['load_fail'] += 1
            continue
        clean_list.append(clean)
        aligned_list.append(aligned)
        counts['ok'] += 1
        pbar.set_postfix(ok=counts['ok'])
    pbar.close()
    print(f"  celeba: ok={counts['ok']:,} | "
          f"roll={counts['roll_too_high']:,} profile={counts['profile']:,} "
          f"tiny={counts['tiny_eyes']:,} load_fail={counts['load_fail']:,}")
    return np.stack(clean_list), np.stack(aligned_list)


# ---- FDDB ------------------------------------------------------------------
def _mediapipe_face_landmarks(rgb, landmarker):
    """Run FaceLandmarker on an RGB array; return (left_eye, right_eye, nose) or None."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    H, W = rgb.shape[:2]
    def mean_xy(idxs):
        return (np.mean([lms[i].x for i in idxs]) * W,
                np.mean([lms[i].y for i in idxs]) * H)
    nose = (lms[MP_NOSE_TIP].x * W, lms[MP_NOSE_TIP].y * H)
    return mean_xy(MP_LEFT_EYE), mean_xy(MP_RIGHT_EYE), nose


def process_fddb(images_dir, coco_path, n_target, landmarker, pad_frac=0.4):
    """FDDB faces: crop padded bbox from full-res image, detect landmarks, emit pair."""
    coco = json.loads(coco_path.read_text())
    id_to_fname = {img['id']: img['file_name'] for img in coco['images']}
    anns = coco['annotations']

    clean_list, aligned_list = [], []
    counts = {'ok': 0, 'bad_bbox': 0, 'no_face': 0,
              'roll_too_high': 0, 'profile': 0,
              'tiny_eyes': 0, 'load_fail': 0}

    # Cache the loaded image so consecutive bboxes from the same photo don't re-read.
    last_fname, last_full = None, None

    pbar = tqdm(anns, desc='fddb', unit='bbox')
    for ann in pbar:
        if counts['ok'] >= n_target:
            break
        bx, by, bw, bh = ann.get('bbox', [None] * 4)
        if not all(v is not None for v in (bx, by, bw, bh)) or bw <= 0 or bh <= 0:
            counts['bad_bbox'] += 1
            continue
        fname = id_to_fname.get(ann['image_id'])
        if fname is None:
            counts['load_fail'] += 1
            continue
        if fname != last_fname:
            try:
                last_full = np.array(Image.open(images_dir / fname).convert('RGB'))
            except Exception:
                counts['load_fail'] += 1
                last_fname = None
                continue
            last_fname = fname
        H_full, W_full = last_full.shape[:2]

        # Pad bbox by `pad_frac` of its size before cropping (gives MediaPipe
        # context). Clip to image bounds.
        pad_x = bw * pad_frac
        pad_y = bh * pad_frac
        x1 = max(0, int(bx - pad_x))
        y1 = max(0, int(by - pad_y))
        x2 = min(W_full, int(bx + bw + pad_x))
        y2 = min(H_full, int(by + bh + pad_y))
        if x2 - x1 < 32 or y2 - y1 < 32:
            counts['bad_bbox'] += 1
            continue
        crop_rgb = last_full[y1:y2, x1:x2]

        lms = _mediapipe_face_landmarks(crop_rgb, landmarker)
        if lms is None:
            counts['no_face'] += 1
            continue
        le_c, re_c, ns_c = lms
        # Lift landmarks back to full-res coords.
        le   = (le_c[0] + x1, le_c[1] + y1)
        re   = (re_c[0] + x1, re_c[1] + y1)
        nose = (ns_c[0] + x1, ns_c[1] + y1)

        ok, reason = _quality_ok(le, re, nose)
        if not ok:
            counts[reason] += 1
            continue
        gray_full = cv2.cvtColor(last_full, cv2.COLOR_RGB2GRAY)
        clean, aligned = _emit_pair(gray_full, le, re)
        if clean is None or aligned is None:
            counts['load_fail'] += 1
            continue
        clean_list.append(clean)
        aligned_list.append(aligned)
        counts['ok'] += 1
        pbar.set_postfix(ok=counts['ok'])
    pbar.close()
    print(f"  fddb: ok={counts['ok']:,} | "
          f"no_face={counts['no_face']:,} roll={counts['roll_too_high']:,} "
          f"profile={counts['profile']:,} tiny={counts['tiny_eyes']:,} "
          f"bad_bbox={counts['bad_bbox']:,} load_fail={counts['load_fail']:,}")
    if not clean_list:
        empty = np.zeros((0, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
        return empty, empty
    return np.stack(clean_list), np.stack(aligned_list)


# ---- CBCL ------------------------------------------------------------------
def process_cbcl(parquet_path):
    """CBCL train rows from HF parquet → resize 19→48.

    Returns three arrays:
        clean_faces, aligned_faces, negatives
    `clean_faces` and `aligned_faces` are identical — CBCL is the canonical
    reference geometry, so the "tight aligned" version IS the original. The
    negatives are the label==0 rows currently misfiled under the train split;
    callers should upload them to the `negatives` split, not train.
    """
    pf = pq.ParquetFile(parquet_path)
    faces = []
    negatives = []
    for batch in tqdm(pf.iter_batches(batch_size=1000),
                      total=pf.num_row_groups, desc='cbcl'):
        srcs   = batch.column('source').to_pylist()
        labels = batch.column('label').to_pylist()
        imgs   = batch.column('image').to_pylist()
        for s, l, ib in zip(srcs, labels, imgs):
            if s != 'cbcl':
                continue
            arr = np.array(Image.open(io.BytesIO(ib['bytes'])).convert('L'))
            resized = cv2.resize(arr, (OUTPUT_SIZE, OUTPUT_SIZE),
                                 interpolation=cv2.INTER_CUBIC)
            (faces if l == 1 else negatives).append(resized)
    empty = np.zeros((0, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    faces_arr = np.stack(faces, axis=0) if faces else empty
    neg_arr   = np.stack(negatives, axis=0) if negatives else empty
    print(f"  cbcl: {len(faces_arr):,} faces, {len(neg_arr):,} negatives")
    return faces_arr, faces_arr.copy(), neg_arr


# ---- Output utilities ------------------------------------------------------
def make_preview(faces, n=400, cols=20, pad=1):
    n = min(n, len(faces))
    if n == 0:
        return np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    rows = (n + cols - 1) // cols
    H = rows * (OUTPUT_SIZE + pad) - pad
    W = cols * (OUTPUT_SIZE + pad) - pad
    grid = np.full((H, W), 255, dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y = r * (OUTPUT_SIZE + pad)
        x = c * (OUTPUT_SIZE + pad)
        grid[y:y + OUTPUT_SIZE, x:x + OUTPUT_SIZE] = faces[i]
    return grid


def save_pair(name, clean_arr, aligned_arr, out_dir):
    if len(clean_arr) == 0:
        print(f"  {name}: empty, skipping save")
        return
    np.save(out_dir / f"{name}_clean.npy", clean_arr)
    np.save(out_dir / f"{name}_aligned.npy", aligned_arr)
    Image.fromarray(make_preview(clean_arr)).save(
        out_dir / f"{name}_clean_preview.png")
    Image.fromarray(make_preview(aligned_arr)).save(
        out_dir / f"{name}_aligned_preview.png")
    print(f"  {name}: saved {len(clean_arr):,} paired faces + previews")


# ---- Main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--celeba-dir',
                    default='/Users/salvacarrion/Desktop/datasets/img_align_celeba',
                    help='Directory of raw CelebA aligned JPGs (178×218 RGB).')
    ap.add_argument('--celeba-landmarks',
                    default='/Users/salvacarrion/Desktop/datasets/list_landmarks_align_celeba.txt',
                    help='CelebA per-image landmark txt file.')
    ap.add_argument('--fddb-images',
                    default='/Users/salvacarrion/Desktop/datasets/Face detection.v1-fddb.coco/train',
                    help='Directory of full-res FDDB JPGs.')
    ap.add_argument('--fddb-coco',
                    default='/Users/salvacarrion/Desktop/datasets/Face detection.v1-fddb.coco/train/_annotations.coco.json',
                    help='COCO annotations JSON for FDDB.')
    ap.add_argument('--cbcl-parquet',
                    default='/Users/salvacarrion/Desktop/data/train-00000-of-00001.parquet',
                    help='HF parquet with the original 19×19 CBCL train faces.')
    ap.add_argument('--out-dir', default='data/clean',
                    help='Where to write the *_clean.npy / *_aligned.npy + previews.')
    ap.add_argument('--n-celeba', type=int, default=50000,
                    help='Cap of accepted CelebA faces (default 50k).')
    ap.add_argument('--n-fddb', type=int, default=20000,
                    help='Cap of accepted FDDB faces (default 20k — FDDB '
                         'tops out around 8-15k passes anyway).')
    ap.add_argument('--sources', nargs='+', default=['celeba', 'cbcl'],
                    help='Which sources to (re)build. Useful for re-running '
                         'one without redoing the others.')
    ap.add_argument('--model-path', default=str(DEFAULT_MODEL_PATH),
                    help='Path to face_landmarker.task (auto-downloaded).')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    landmarker = None
    if 'fddb' in args.sources:
        ensure_landmarker_model(Path(args.model_path))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=args.model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    if 'celeba' in args.sources:
        print("\n=== celeba (raw + provided landmarks) ===")
        clean, aligned = process_celeba(
            Path(args.celeba_dir), Path(args.celeba_landmarks),
            n_target=args.n_celeba)
        save_pair('celeba', clean, aligned, out_dir)

    if 'fddb' in args.sources:
        print("\n=== fddb (raw + MediaPipe) ===")
        clean, aligned = process_fddb(
            Path(args.fddb_images), Path(args.fddb_coco),
            n_target=args.n_fddb, landmarker=landmarker)
        save_pair('fddb', clean, aligned, out_dir)

    if 'cbcl' in args.sources:
        print("\n=== cbcl (parquet passthrough) ===")
        clean, aligned, negatives = process_cbcl(Path(args.cbcl_parquet))
        save_pair('cbcl', clean, aligned, out_dir)
        # Save CBCL negatives separately — these belong in the `negatives`
        # split of the HF dataset, not under train.
        if len(negatives) > 0:
            np.save(out_dir / 'cbcl_negatives.npy', negatives)
            Image.fromarray(make_preview(negatives)).save(
                out_dir / 'cbcl_negatives_preview.png')
            print(f"  cbcl: saved {len(negatives):,} negatives "
                  f"(→ upload to `negatives` split)")

    if landmarker is not None:
        landmarker.close()
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == '__main__':
    main()
