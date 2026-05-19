#!/usr/bin/env bash
# Sequential training of the three 19×19 baselines:
#   cbcl, celeba_aligned, celeba_aligned+cbcl
#
# Each run: train → test → diagnose. Detect (visual outputs) is left for
# manual inspection — run it after the script finishes if you want.
#
# Logs go to logs/<timestamp>/<tag>.log; weights end up as
# weights/19/<tag>__19.pkl. The script aborts on the first error.
#
# Usage:
#   bash scripts/run_19_experiments.sh
#
# Or background (survives terminal close):
#   nohup bash scripts/run_19_experiments.sh > nohup.out 2>&1 &
#   tail -f logs/<latest>/cbcl.log     # follow current train

set -euo pipefail

# ---- Hyperparams shared across all 3 runs (only --data-dir differs) ----
COMMON_ARGS=(
  --max-stages 20
  --max-wcs-per-stage 400
  --target-stage-fpr 0.5
  --min-cascade-recall 0.95
  --target-neg-per-stage 5000
  --neg-sample-budget 50000000
  --min-stage-negatives 1000
)

# ---- Experiments to run, in order ----
# Format: "tag|data_dir"
EXPERIMENTS=(
  # "cbcl|data/19_cbcl"
  "celeba_aligned|data/19_celeba_aligned"
  "celeba_aligned+cbcl|data/19_celeba_aligned+cbcl"
)

# ---- Setup ----
cd "$(dirname "$0")/.."  # repo root
LOG_DIR="logs/19_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
mkdir -p weights/19

# Refuse to start if there are stale cvj_weights_*.pkl that would confuse
# the mv-to-renamed-output step. The user must explicitly clean them.
# `shopt -s nullglob` makes the glob expand to nothing (not the literal
# pattern) when no files match — required so the empty-pool case doesn't
# trip `set -o pipefail` + `set -e` and abort the script silently.
shopt -s nullglob
stale_pkls=(weights/19/cvj_weights_*.pkl)
shopt -u nullglob
if [ "${#stale_pkls[@]}" -gt 0 ]; then
  echo "ERROR: ${#stale_pkls[@]} stale cvj_weights_*.pkl in weights/19/:" >&2
  printf '         %s\n' "${stale_pkls[@]}" >&2
  echo "       Either move them or run: rm weights/19/cvj_weights_*.pkl" >&2
  exit 1
fi

echo "============================================================"
echo "  Three 19×19 baselines"
echo "  Started: $(date)"
echo "  Logs:    $LOG_DIR"
echo "  Hyperparams (shared):"
for arg in "${COMMON_ARGS[@]}"; do echo "    $arg"; done
echo "============================================================"

script_start=$(date +%s)

run_experiment() {
  local tag="$1"
  local data_dir="$2"
  local out_weights="weights/19/${tag}__19.pkl"
  local log_file="${LOG_DIR}/${tag}.log"
  local stage_start=$(date +%s)

  echo ""
  echo "============================================================"
  echo "  [${tag}]  $(date +%H:%M:%S)"
  echo "  data:    ${data_dir}"
  echo "  log:     ${log_file}"
  echo "  weights: ${out_weights}"
  echo "============================================================"

  if [ ! -d "${data_dir}" ]; then
    echo "ERROR: data dir not found: ${data_dir}" >&2
    return 1
  fi

  # Train
  echo "--- Training ---" | tee "${log_file}"
  python main.py train \
    --data-dir "${data_dir}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "${log_file}"

  # Rename the freshly produced checkpoint to the experiment's canonical name.
  # nullglob: when train crashed without writing a .pkl, the glob expands
  # to empty instead of the literal pattern — without it `set -o pipefail`
  # + `set -e` would kill the script before our friendly error message.
  shopt -s nullglob
  candidates=(weights/19/cvj_weights_*.pkl)
  shopt -u nullglob
  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "ERROR: train produced no checkpoint for ${tag}" >&2
    return 1
  fi
  # Newest by mtime (in case more than one survives somehow).
  produced=$(ls -t "${candidates[@]}" | head -1)
  mv "${produced}" "${out_weights}"
  echo "" | tee -a "${log_file}"
  echo "Saved: ${out_weights}" | tee -a "${log_file}"

  # Test
  echo "" | tee -a "${log_file}"
  echo "--- Test ---" | tee -a "${log_file}"
  python main.py test \
    --data-dir "${data_dir}" \
    --weights-path "${out_weights}" \
    2>&1 | tee -a "${log_file}"

  # Diagnose
  echo "" | tee -a "${log_file}"
  echo "--- Diagnose ---" | tee -a "${log_file}"
  python tools/diagnose_cascade.py \
    --weights "${out_weights}" \
    --data-dir "${data_dir}" \
    2>&1 | tee -a "${log_file}"

  local stage_end=$(date +%s)
  local elapsed=$((stage_end - stage_start))
  local hh=$((elapsed / 3600))
  local mm=$(( (elapsed % 3600) / 60 ))
  echo "" | tee -a "${log_file}"
  echo "[${tag}] completed in ${hh}h ${mm}min — $(date +%H:%M:%S)" | tee -a "${log_file}"
}

for spec in "${EXPERIMENTS[@]}"; do
  tag="${spec%%|*}"
  data_dir="${spec##*|}"
  run_experiment "${tag}" "${data_dir}"
done

script_end=$(date +%s)
total=$((script_end - script_start))
total_h=$((total / 3600))
total_m=$(( (total % 3600) / 60 ))

echo ""
echo "============================================================"
echo "  ALL DONE — total: ${total_h}h ${total_m}min"
echo "  Logs:    ${LOG_DIR}/"
echo "  Weights:"
for spec in "${EXPERIMENTS[@]}"; do
  tag="${spec%%|*}"
  echo "    weights/19/${tag}__19.pkl"
done
echo ""
echo "  Run visual detection manually for each:"
echo "    python main.py detect \\"
echo "      --weights-path weights/19/<tag>__19.pkl \\"
echo "      --detect-output images/outputs__19_<tag>"
echo "============================================================"
