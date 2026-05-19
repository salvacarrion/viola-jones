"""
Measure CBCL's native eye geometry by averaging the pre-aligned face crops.

CBCL was hand-aligned in 2000 for the Viola-Jones benchmark, so every crop
sits at the same canonical position. Averaging all of them produces a clean
"ghost face" where the eyes show up as two strong dark minima — much easier
to localize than on any individual 19×19 noisy crop.

We then find the two darkest local minima in the upper half of the mean
face and report their pixel coordinates in the OUTPUT_SIZE×OUTPUT_SIZE
canonical frame. Pipe the result into tools/align_faces.py via:

    python tools/align_faces.py --eye-spacing <S> --eye-y <Y>

Outputs (under --out-dir):
    mean_cbcl_24.png       — the raw mean face at 24×24
    mean_cbcl_zoom.png     — same upscaled 20× for visual inspection
    mean_cbcl_marked.png   — zoom with detected eye centers marked
"""

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm

DEFAULT_OUTPUT_SIZE = 48


def load_cbcl_faces(parquet_path, output_size):
    """Load every CBCL train face (label==1), resize to output_size×output_size."""
    pf = pq.ParquetFile(parquet_path)
    faces = []
    for batch in tqdm(pf.iter_batches(batch_size=1000),
                      total=pf.num_row_groups, desc="loading CBCL"):
        srcs   = batch.column('source').to_pylist()
        labels = batch.column('label').to_pylist()
        imgs   = batch.column('image').to_pylist()
        for s, l, ib in zip(srcs, labels, imgs):
            if s != 'cbcl' or l != 1:
                continue
            arr = np.array(Image.open(io.BytesIO(ib['bytes'])).convert('L'))
            resized = cv2.resize(arr, (output_size, output_size),
                                 interpolation=cv2.INTER_CUBIC)
            faces.append(resized)
    return np.stack(faces, axis=0)


def detect_eyes_in_mean(mean_face):
    """
    Find the two eye centers in a clean mean face.

    Strategy: search the upper half for the two darkest local minima that
    are also horizontally separated by at least 1/4 of the frame width.
    Mean-of-many-faces averaging makes eye sockets the deepest minima by a
    wide margin, so we don't need anything fancier than NMS.
    """
    h, w = mean_face.shape
    # Upper 2/3: eyes never live below row h/2 in a frontal face. Restrict
    # the search so the nose/mouth shadows can't compete.
    upper = mean_face[: h * 2 // 3].astype(np.float32)
    smoothed = cv2.GaussianBlur(upper, (3, 3), 0.6)

    # Sort all pixel positions by darkness (ascending = darkest first) and
    # greedily pick two with a horizontal separation constraint.
    ys, xs = np.indices(smoothed.shape)
    flat_idx = np.argsort(smoothed.ravel())
    flat_ys = ys.ravel()[flat_idx]
    flat_xs = xs.ravel()[flat_idx]

    min_sep = w * 0.25
    picks = []
    for x, y in zip(flat_xs, flat_ys):
        if not picks:
            picks.append((int(x), int(y)))
        else:
            if abs(int(x) - picks[0][0]) >= min_sep:
                picks.append((int(x), int(y)))
                break

    picks.sort(key=lambda p: p[0])
    return picks


def detect_mouth_in_mean(mean_face, eye_y):
    """
    Find the mouth center in the mean face.

    Strategy: search rows below the eyes (lower 60% of the frame) for the
    darkest pixel that is also reasonably centered horizontally. The mouth
    appears as a horizontal dark band — the darkest single pixel in that
    region is a stable estimate of its center. We further restrict to the
    central horizontal band so a dark nostril shadow off to the side can't
    win when the lip contrast is weak.
    """
    h, w = mean_face.shape
    # Search well below the eyes only.
    row_start = int(eye_y + 0.25 * h)
    region = mean_face[row_start:].astype(np.float32)
    if region.size == 0:
        return (w // 2, h - 1)
    smoothed = cv2.GaussianBlur(region, (3, 3), 0.6)

    # Restrict to the central horizontal band (mouths are centered).
    x_lo = int(w * 0.30)
    x_hi = int(w * 0.70)
    band = smoothed[:, x_lo:x_hi]
    flat_idx = int(np.argmin(band))
    yy, xx = np.unravel_index(flat_idx, band.shape)
    return (int(xx + x_lo), int(yy + row_start))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet",
                    default="/Users/salvacarrion/Desktop/data/train-00000-of-00001.parquet",
                    help="Parquet file containing source='cbcl' label==1 faces.")
    ap.add_argument("--out-dir", default="data/aligned",
                    help="Where to write the mean-face visualizations.")
    ap.add_argument("--zoom", type=int, default=10,
                    help="Pixel-replicating zoom factor for the inspection "
                         "image (default: 10 → 480×480 at output-size=48).")
    ap.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE,
                    help=f"Resolution at which to load and average the CBCL "
                         f"crops (default: {DEFAULT_OUTPUT_SIZE}). The "
                         f"reported eye/mouth pixel coords are in this frame, "
                         f"so use the same --output-size for align_faces.py.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_size = args.output_size

    faces = load_cbcl_faces(Path(args.parquet), output_size)
    print(f"\nLoaded {len(faces):,} CBCL faces at {output_size}×{output_size}.")
    mean_face = faces.mean(axis=0).astype(np.uint8)

    # Save raw mean + zoomed inspection version.
    raw_path = out_dir / f"mean_cbcl_{output_size}.png"
    Image.fromarray(mean_face).save(raw_path)

    zoom = max(1, int(args.zoom))
    big = cv2.resize(mean_face, (output_size * zoom, output_size * zoom),
                     interpolation=cv2.INTER_NEAREST)
    zoom_path = out_dir / "mean_cbcl_zoom.png"
    Image.fromarray(big).save(zoom_path)

    # Detect eyes and mouth.
    eyes = detect_eyes_in_mean(mean_face)
    (lx, ly), (rx, ry) = eyes
    spacing = rx - lx
    eye_y = (ly + ry) / 2.0
    mx, my = detect_mouth_in_mean(mean_face, eye_y)

    # Annotated debug image: zoom + crosshairs at detected eye + mouth.
    marked = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    for (x, y), color in zip([eyes[0], eyes[1], (mx, my)],
                             [(0, 0, 255), (0, 0, 255), (0, 255, 0)]):
        cx, cy = int((x + 0.5) * zoom), int((y + 0.5) * zoom)
        cv2.drawMarker(marked, (cx, cy), color=color,
                       markerType=cv2.MARKER_CROSS,
                       markerSize=zoom * 2, thickness=2)
    marked_path = out_dir / "mean_cbcl_marked.png"
    Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)).save(marked_path)

    print(f"\nSaved:")
    print(f"  {raw_path}        ({output_size}×{output_size} mean)")
    print(f"  {zoom_path}  ({zoom}× zoom, no marks)")
    print(f"  {marked_path} ({zoom}× zoom, eyes=red, mouth=green)")

    print(f"\nDetected landmarks ({output_size}×{output_size} frame):")
    print(f"  left eye  = ({lx}, {ly})")
    print(f"  right eye = ({rx}, {ry})")
    print(f"  mouth     = ({mx}, {my})")
    print(f"  eye spacing = {spacing} px")
    print(f"  eye_y       = {eye_y:.1f}")
    print(f"  mouth_y     = {my}")
    print(f"\nReady-to-use flags for tools/align_faces.py:")
    print(f"  --output-size {output_size} --eye-spacing {spacing} "
          f"--eye-y {eye_y:.1f} --mouth-y {my}")


if __name__ == "__main__":
    main()
