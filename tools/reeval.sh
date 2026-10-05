#!/usr/bin/env bash
#
# Re-evaluation harness for the shipped checkpoints:
#   1) CBCL patch benchmark        (main.py test)               -> F1
#   2) FDDB in-the-wild benchmark  (tools/eval_fddb.py)         -> AP / recall / precision vs OpenCV
#      2b) window diagnostic       (tools/diagnose_fddb_windows.py)
#   3) OpenCV patch baseline       (tools/baseline_opencv.py)
#   4) Native OpenCV port build    (tools/convert_opencv_cascade.py)
#   5) (opt) per-stage diagnose    (tools/diagnose_cascade.py)
#
# Everything is teed to results_reeval.txt.
#
# FDDB protocol: each detector's box transform (its crop convention -> FDDB's
# face box) is fitted on FIT_FOLDS and every metric is reported on the
# disjoint FOLDS, both with raw boxes and with `+box` face boxes.
#
# The committed results_reeval.txt was produced at commit 1c7a789 with every
# training run (the per-model tables in the docs); check that commit out to
# regenerate it. This version only evaluates what ships in weights/.
#
# Usage:
#   tools/reeval.sh                     # FDDB folds 2-10, boxes fitted on fold 1 (~40 min)
#   tools/reeval.sh 2                   # one held-out fold (fast smoke)
#   DIAGNOSE=1 tools/reeval.sh          # also dump the per-stage CBCL diagnose
#
# Env knobs: FOLDS overrides the positional arg; FIT_FOLDS (default 1);
# SKIP_FDDB=1 / SKIP_CBCL=1 to skip a whole section.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

FOLDS="${1:-${FOLDS:-2,3,4,5,6,7,8,9,10}}"
FIT_FOLDS="${FIT_FOLDS:-1}"
RES="$ROOT/results_reeval.txt"
BEST=weights/24/celeba.pkl
OTHERS=(weights/19/celeba_cbcl.pkl weights/19/cbcl.pkl)

{
  echo "# Re-evaluation: $(LC_ALL=C date)"
  echo "# python : $(python --version 2>&1)"
  echo "# fddb   : eval folds=$FOLDS, box transform fitted on folds=$FIT_FOLDS"
} > "$RES"

say()  { echo "$@" | tee -a "$RES"; }
hdr()  { say ""; say "######################## $* ########################"; }
runpy(){ python "$@" 2>/dev/null | tee -a "$RES"; }   # tqdm/cv2 noise -> /dev/null

# ============================================================================
# 1) CBCL patch benchmark (the test set is identical in every data/<res>_*
#    bundle; the data dir only sets the patch size)
# ============================================================================
if [ "${SKIP_CBCL:-0}" != "1" ]; then
  hdr "1) CBCL PATCH BENCHMARK"
  for f in "$BEST" "${OTHERS[@]}"; do
    res="$(basename "$(dirname "$f")")"
    say ""; say "### CBCL $f"
    runpy main.py test --weights-path "$f" --data-dir "data/${res}_cbcl"
  done
fi

# ============================================================================
# 2) FDDB in-the-wild benchmark
#    Every run prints `<name>` (raw boxes) and `<name>+box` (face boxes,
#    transform fitted on FIT_FOLDS) rows.
# ============================================================================
if [ "${SKIP_FDDB:-0}" != "1" ]; then
  hdr "2) FDDB IN-THE-WILD  (eval folds=$FOLDS, box fit folds=$FIT_FOLDS, IoU 0.3 & 0.5)"
  FD=(--folds "$FOLDS" --box-fit-folds "$FIT_FOLDS" --iou 0.3,0.5)

  say ""; say "### FDDB  ours=$BEST (pyramid=image)  +  cv2:default"
  runpy tools/eval_fddb.py --weights "$BEST" --cascade default "${FD[@]}"

  say ""; say "### FDDB  ours=$BEST (pyramid=features, LEGACY pre-fix inference)"
  runpy tools/eval_fddb.py --weights "$BEST" --skip-opencv --pyramid features "${FD[@]}"

  say ""; say "### FDDB  ours=$BEST (min-face=80, sensitivity)"
  runpy tools/eval_fddb.py --weights "$BEST" --skip-opencv --min-face 80 "${FD[@]}"

  for f in "${OTHERS[@]}"; do
    say ""; say "### FDDB  ours=$f"
    runpy tools/eval_fddb.py --weights "$f" --skip-opencv "${FD[@]}"
  done

  say ""; say "### FDDB  native port  weights/24/opencv_default.pkl"
  if [ -f weights/24/opencv_default.pkl ]; then
    runpy tools/eval_fddb.py --weights weights/24/opencv_default.pkl \
          --skip-opencv "${FD[@]}"
  else
    say "  (skipped: run section 4 first to build it)"
  fi

  say ""; say "### FDDB  cv2:alt (reference)"
  runpy tools/eval_fddb.py --skip-ours --cascade alt "${FD[@]}"

  hdr "2b) FDDB WINDOW DIAGNOSTIC (folds=$FIT_FOLDS: background FPR per window + per-stage rejection)"
  runpy tools/diagnose_fddb_windows.py --folds "$FIT_FOLDS" \
        --weights "$BEST" weights/24/opencv_default.pkl
  runpy tools/diagnose_fddb_windows.py --folds "$FIT_FOLDS" --pyramid features \
        --weights "$BEST"
fi

# ============================================================================
# 3) OpenCV patch baseline  (shows it scores ~0 on tight CBCL crops, by design)
# ============================================================================
hdr "3) OPENCV PATCH BASELINE (CBCL)"
runpy tools/baseline_opencv.py benchmark --data-dir data/24_cbcl --cascade default alt alt2

# ============================================================================
# 4) Native OpenCV port  (build + window-level parity vs cv2)
# ============================================================================
hdr "4) NATIVE OPENCV PORT (build + parity)"
runpy tools/convert_opencv_cascade.py --cascade default
runpy tools/convert_opencv_cascade.py --cascade alt

# ============================================================================
# 5) (optional) per-stage diagnose on CBCL
# ============================================================================
if [ "${DIAGNOSE:-0}" = "1" ]; then
  hdr "5) DIAGNOSE (per-stage, best model)"
  runpy tools/diagnose_cascade.py --weights "$BEST" --data-dir data/24_celeba_aligned
fi

hdr "DONE -> $RES"
