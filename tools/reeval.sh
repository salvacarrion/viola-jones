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
# FDDB protocol: each detector's box transform (its crop convention -> FDDB's
# face box) is fitted on FIT_FOLDS and every metric is reported on the
# disjoint FOLDS, both with raw boxes and with `+box` face boxes.
#
# Usage:
#   tools/reeval.sh                     # FDDB folds 2-10, boxes fitted on fold 1 (~30 min)
#   tools/reeval.sh 2                   # one held-out fold (fast smoke)
#   DIAGNOSE=1 tools/reeval.sh          # also dump per-stage diagnose
#   TUNE=1     tools/reeval.sh          # also regenerate the *_tuned.pkl files
#
# Env knobs: FOLDS overrides the positional arg; FIT_FOLDS (default 1);
# SKIP_FDDB=1 / SKIP_CBCL=1 to skip a whole section.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

FOLDS="${1:-${FOLDS:-2,3,4,5,6,7,8,9,10}}"
FIT_FOLDS="${FIT_FOLDS:-1}"
RES="$ROOT/results_reeval.txt"

{
  echo "# Re-evaluation: $(LC_ALL=C date)"
  echo "# python : $(python --version 2>&1)"
  echo "# fddb   : eval folds=$FOLDS, box transform fitted on folds=$FIT_FOLDS"
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
# 2) FDDB in-the-wild benchmark
#    Every run prints `<name>` (raw boxes) and `<name>+box` (face boxes,
#    transform fitted on FIT_FOLDS) rows.
# ============================================================================
if [ "${SKIP_FDDB:-0}" != "1" ]; then
  hdr "2) FDDB IN-THE-WILD  (eval folds=$FOLDS, box fit folds=$FIT_FOLDS, IoU 0.3 & 0.5)"
  FD=(--folds "$FOLDS" --box-fit-folds "$FIT_FOLDS" --iou 0.3,0.5)

  say ""; say "### FDDB  ours=celeba_aligned__24_v2_s11_tuned (pyramid=image)  +  cv2:default"
  runpy tools/eval_fddb.py --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl \
        --cascade default "${FD[@]}"

  say ""; say "### FDDB  ours=celeba_aligned__24_v2_s11_tuned (pyramid=features, LEGACY pre-fix inference)"
  runpy tools/eval_fddb.py --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl \
        --skip-opencv --pyramid features "${FD[@]}"

  say ""; say "### FDDB  ours=celeba_aligned__24_v2_s11_tuned (min-face=80, sensitivity)"
  runpy tools/eval_fddb.py --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl \
        --skip-opencv --min-face 80 "${FD[@]}"

  # Every other canonical model (tuned thresholds: on FDDB they beat the raw
  # ones), for the per-model FDDB column of the README table.
  for m in "${CANON_19[@]}" "${CANON_24[@]}"; do
    [ "$m" = celeba_aligned__24_v2_s11 ] && continue
    res="${m##*__}"; res="${res%%_*}"
    f="weights/${res}/${m}_tuned.pkl"; [ -f "$f" ] || continue
    say ""; say "### FDDB  ours=${m}_tuned"
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
        --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl weights/24/opencv_default.pkl
  runpy tools/diagnose_fddb_windows.py --folds "$FIT_FOLDS" --pyramid features \
        --weights weights/24/celeba_aligned__24_v2_s11_tuned.pkl
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
