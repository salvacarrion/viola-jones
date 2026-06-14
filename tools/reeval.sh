#!/usr/bin/env bash
#
# Re-evaluation harness for the README / docs comparison tables.
#
# Runs, for the CANONICAL models only (not the intermediate s10/s12/ext
# checkpoints):
#   1) CBCL patch benchmark        (main.py test)            -> F1 table
#   2) FDDB in-the-wild benchmark  (tools/eval_fddb.py)      -> AP/recall table
#   3) OpenCV patch baseline       (tools/baseline_opencv.py)
#   4) Native OpenCV port build    (tools/convert_opencv_cascade.py)
#   5) (opt) per-stage diagnose    (tools/diagnose_cascade.py)
#   6) (opt) re-tune thresholds    (tools/tune_thresholds.py)
#
# Everything is teed to results_reeval.txt — send that file back.
#
# Usage:
#   tools/reeval.sh                     # FDDB fold 1 (fast: ~15-25 min total)
#   tools/reeval.sh 1,2,3,4,5,6,7,8,9,10   # full FDDB (definitive headline; +hours for our models)
#   DIAGNOSE=1 tools/reeval.sh          # also dump per-stage diagnose
#   TUNE=1     tools/reeval.sh          # also regenerate the *_tuned.pkl files
#
# Env knobs: FOLDS overrides the positional arg; SKIP_FDDB=1 / SKIP_CBCL=1 to
# skip a whole section.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

FOLDS="${1:-${FOLDS:-1}}"
RES="$ROOT/results_reeval.txt"

{
  echo "# Re-evaluation — $(date)"
  echo "# python : $(python --version 2>&1)"
  echo "# fddb   : folds=$FOLDS"
} > "$RES"

say()  { echo "$@" | tee -a "$RES"; }
hdr()  { say ""; say "######################## $* ########################"; }
runpy(){ python "$@" 2>/dev/null | tee -a "$RES"; }   # tqdm/cv2 noise -> /dev/null

# Canonical models: "name:resdir" (data dir only sets the test-patch size;
# the CBCL test set is identical across all data/<res>_* bundles).
CANON_19=(cbcl__19_v1 cbcl__19_v2 celeba_aligned__19_v1 celeba_aligned__19_v2 \
          celeba_aligned_filtered__19_v1 celeba_aligned+cbcl__19_v1 celeba_aligned+cbcl__19_v2)
CANON_24=(cbcl__24_smoke celeba_aligned__24_v1 celeba_aligned__24_v2_s11)

# ============================================================================
# 1) CBCL patch benchmark  (refreshes the F1 table)
# ============================================================================
if [ "${SKIP_CBCL:-0}" != "1" ]; then
  hdr "1) CBCL PATCH BENCHMARK"
  for m in "${CANON_19[@]}"; do
    for v in "" _tuned; do
      f="weights/19/${m}${v}.pkl"; [ -f "$f" ] || continue
      say ""; say "### CBCL ${m}${v}"
      runpy main.py test --weights-path "$f" --data-dir data/19_cbcl
    done
  done
  for m in "${CANON_24[@]}"; do
    for v in "" _tuned; do
      f="weights/24/${m}${v}.pkl"; [ -f "$f" ] || continue
      say ""; say "### CBCL ${m}${v}"
      runpy main.py test --weights-path "$f" --data-dir data/24_cbcl
    done
  done
fi

# ============================================================================
# 2) FDDB in-the-wild benchmark   (NEW)
#    Append more folds (arg) for the definitive headline numbers.
# ============================================================================
if [ "${SKIP_FDDB:-0}" != "1" ]; then
  hdr "2) FDDB IN-THE-WILD  (folds=$FOLDS, IoU 0.3 & 0.5)"

  say ""; say "### FDDB  ours=celeba_aligned__24_v2_s11_tuned (min-face=40)  +  cv2:default"
  runpy tools/eval_fddb.py --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl \
        --cascade default --folds "$FOLDS" --iou 0.3,0.5

  say ""; say "### FDDB  ours=celeba_aligned__24_v2_s11_tuned (min-face=80, best op-point)"
  runpy tools/eval_fddb.py --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl \
        --skip-opencv --folds "$FOLDS" --min-face 80 --iou 0.3,0.5

  say ""; say "### FDDB  ours=celeba_aligned+cbcl__19_v2_tuned (best 19x19)"
  runpy tools/eval_fddb.py --weights weights/19/celeba_aligned+cbcl__19_v2_tuned.pkl \
        --skip-opencv --folds "$FOLDS" --iou 0.3,0.5

  say ""; say "### FDDB  native port  weights/24/opencv_default.pkl"
  if [ -f weights/24/opencv_default.pkl ]; then
    runpy tools/eval_fddb.py --weights weights/24/opencv_default.pkl \
          --skip-opencv --folds "$FOLDS" --iou 0.3,0.5
  else
    say "  (skipped — run section 4 first to build it)"
  fi

  say ""; say "### FDDB  cv2:alt (reference)"
  runpy tools/eval_fddb.py --skip-ours --cascade alt --folds "$FOLDS" --iou 0.3,0.5
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
# 5) (optional) per-stage diagnose for docs/RESULTS
# ============================================================================
if [ "${DIAGNOSE:-0}" = "1" ]; then
  hdr "5) DIAGNOSE (per-stage, ⭐ model)"
  runpy tools/diagnose_cascade.py \
        --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl \
        --data-dir data/24_celeba_aligned
fi

# ============================================================================
# 6) (optional) regenerate *_tuned.pkl  (the tuned files already exist)
# ============================================================================
if [ "${TUNE:-0}" = "1" ]; then
  hdr "6) RE-TUNE THRESHOLDS (regenerates *_tuned.pkl)"
  for pair in \
    "weights/19/cbcl__19_v1.pkl:data/19_cbcl" \
    "weights/19/cbcl__19_v2.pkl:data/19_cbcl" \
    "weights/19/celeba_aligned__19_v1.pkl:data/19_celeba_aligned" \
    "weights/19/celeba_aligned__19_v2.pkl:data/19_celeba_aligned" \
    "weights/19/celeba_aligned_filtered__19_v1.pkl:data/19_celeba_aligned" \
    "weights/19/celeba_aligned+cbcl__19_v1.pkl:data/19_celeba_aligned+cbcl" \
    "weights/19/celeba_aligned+cbcl__19_v2.pkl:data/19_celeba_aligned+cbcl" \
    "weights/24/cbcl__24_smoke.pkl:data/24_cbcl" \
    "weights/24/celeba_aligned__24_v1.pkl:data/24_celeba_aligned" \
    "weights/24/celeba_aligned__24_v2_s11.pkl:data/24_celeba_aligned" ; do
    w="${pair%%:*}"; d="${pair##*:}"; [ -f "$w" ] || continue
    say ""; say "### TUNE $w"
    runpy tools/tune_thresholds.py --weights "$w" --data-dir "$d" --objective f1
  done
fi

hdr "DONE -> $RES"
