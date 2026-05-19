#!/usr/bin/env bash
# Extend the three 19×19 baselines past their natural capacity ceiling.
#
# Each baseline saturated at stage N with final_fpr ≈ 0.55–0.60 (above the
# original target=0.5). We resume from that checkpoint with a relaxed
# target_stage_fpr=0.65 so the capacity-ceiling break (Bug 9 fix) doesn't
# trigger immediately on the existing capped stage, and the cascade can
# add 1–3 more stages before either:
#   (a) the next capacity ceiling around final_fpr ≈ 0.7, or
#   (b) cumulative val recall drops below --min-cascade-recall 0.95
#       (which it will after ~2 extra stages at layer_recall=0.99).
#
# Output checkpoints land as weights/19/<tag>__19_extended.pkl. The original
# baselines (weights/19/<tag>__19.pkl) stay untouched.
#
# Usage:
#   bash scripts/run_19_extend.sh
# Or background:
#   nohup bash scripts/run_19_extend.sh > nohup_extend.out 2>&1 &

set -euo pipefail

# ---- Hyperparams: identical to baselines except target_stage_fpr 0.5 → 0.65 ----
# max-wcs bumped to 500 as safety margin (capped stages should converge
# faster at the looser FPR target, so 500 is rarely hit — it's just headroom).
COMMON_ARGS=(
  --max-stages 20
  --max-wcs-per-stage 800
  --target-stage-fpr 0.65
  --min-cascade-recall 0.95
  --target-neg-per-stage 5000
  --neg-sample-budget 50000000
  --min-stage-negatives 1000
)

# Format: "tag|data_dir|baseline_pkl"
# Baselines are the v1 checkpoints (untuned — extension trains on the raw
# cascade structure; F1-tuning is applied at the end as a separate step).
EXPERIMENTS=(
  "cbcl|data/19_cbcl|weights/19/cbcl__19_v1.1.pkl"
  "celeba_aligned+cbcl|data/19_celeba_aligned+cbcl|weights/19/celeba_aligned+cbcl__19_v1.pkl"
  "celeba_aligned|data/19_celeba_aligned|weights/19/celeba_aligned__19_v1.1.pkl"
)

cd "$(dirname "$0")/.."  # repo root
LOG_DIR="logs/19_extend_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Refuse to start if there are stale cvj_weights_*.pkl that would confuse
# the mv-to-renamed-output step.
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
echo "  19×19 baselines — EXTENSION (relaxed target_stage_fpr)"
echo "  Started: $(date)"
echo "  Logs:    $LOG_DIR"
echo "  Hyperparams (shared):"
for arg in "${COMMON_ARGS[@]}"; do echo "    $arg"; done
echo "============================================================"

script_start=$(date +%s)

run_extension() {
  local tag="$1"
  local data_dir="$2"
  local baseline_pkl="$3"
  local out_weights="weights/19/${tag}__19_v2.pkl"
  local log_file="${LOG_DIR}/${tag}.log"
  local stage_start=$(date +%s)

  echo ""
  echo "============================================================"
  echo "  [${tag}]  $(date +%H:%M:%S)"
  echo "  data:      ${data_dir}"
  echo "  resume:    ${baseline_pkl}"
  echo "  extended:  ${out_weights}"
  echo "  log:       ${log_file}"
  echo "============================================================"

  if [ ! -d "${data_dir}" ]; then
    echo "ERROR: data dir not found: ${data_dir}" >&2
    return 1
  fi
  if [ ! -f "${baseline_pkl}" ]; then
    echo "ERROR: baseline checkpoint not found: ${baseline_pkl}" >&2
    return 1
  fi

  # Train with resume
  echo "--- Training (extension) ---" | tee "${log_file}"
  python main.py train \
    --data-dir "${data_dir}" \
    --resume-from "${baseline_pkl}" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "${log_file}"

  # Rename the extended checkpoint
  shopt -s nullglob
  candidates=(weights/19/cvj_weights_*.pkl)
  shopt -u nullglob
  if [ "${#candidates[@]}" -eq 0 ]; then
    echo "ERROR: train produced no checkpoint for ${tag}" >&2
    return 1
  fi
  produced=$(ls -t "${candidates[@]}" | head -1)
  mv "${produced}" "${out_weights}"
  echo "" | tee -a "${log_file}"
  echo "Saved: ${out_weights}" | tee -a "${log_file}"

  # Test (raw, post-extension)
  echo "" | tee -a "${log_file}"
  echo "--- Test (raw, extended cascade) ---" | tee -a "${log_file}"
  python main.py test \
    --data-dir "${data_dir}" \
    --weights-path "${out_weights}" \
    2>&1 | tee -a "${log_file}"

  # Diagnose
  echo "" | tee -a "${log_file}"
  echo "--- Diagnose (extended) ---" | tee -a "${log_file}"
  python tools/diagnose_cascade.py \
    --weights "${out_weights}" \
    --data-dir "${data_dir}" \
    2>&1 | tee -a "${log_file}"

  # F1 tune on the extended cascade
  echo "" | tee -a "${log_file}"
  echo "--- Tune thresholds on extended cascade ---" | tee -a "${log_file}"
  python tools/tune_thresholds.py \
    --weights "${out_weights}" \
    --data-dir "${data_dir}" \
    --objective f1 \
    2>&1 | tee -a "${log_file}"

  # Test the tuned version
  local tuned_pkl="${out_weights%.pkl}_tuned.pkl"
  if [ -f "${tuned_pkl}" ]; then
    echo "" | tee -a "${log_file}"
    echo "--- Test (tuned, extended cascade) ---" | tee -a "${log_file}"
    python main.py test \
      --data-dir "${data_dir}" \
      --weights-path "${tuned_pkl}" \
      2>&1 | tee -a "${log_file}"
  fi

  local stage_end=$(date +%s)
  local elapsed=$((stage_end - stage_start))
  local hh=$((elapsed / 3600))
  local mm=$(( (elapsed % 3600) / 60 ))
  echo "" | tee -a "${log_file}"
  echo "[${tag}] completed in ${hh}h ${mm}min — $(date +%H:%M:%S)" | tee -a "${log_file}"
}

for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r tag data_dir baseline_pkl <<< "${spec}"
  run_extension "${tag}" "${data_dir}" "${baseline_pkl}"
done

script_end=$(date +%s)
total=$((script_end - script_start))
total_h=$((total / 3600))
total_m=$(( (total % 3600) / 60 ))

echo ""
echo "============================================================"
echo "  EXTENSION DONE — total: ${total_h}h ${total_m}min"
echo "  Logs:    ${LOG_DIR}/"
echo "  v2 (extended) weights:"
for spec in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r tag _ _ <<< "${spec}"
  echo "    weights/19/${tag}__19_v2.pkl        (raw extended)"
  echo "    weights/19/${tag}__19_v2_tuned.pkl  (F1-tuned extended)"
done
echo ""
echo "  Compare v1 (baseline) vs v2 (extended):"
echo "    for tag in cbcl celeba_aligned celeba_aligned+cbcl; do"
echo "      echo \"--- \$tag v1 (tuned) ---\""
echo "      python main.py test --data-dir data/19_\$tag \\\\"
echo "        --weights-path weights/19/\${tag}__19_v1_tuned.pkl"
echo "      echo \"--- \$tag v2 (tuned) ---\""
echo "      python main.py test --data-dir data/19_\$tag \\\\"
echo "        --weights-path weights/19/\${tag}__19_v2_tuned.pkl"
echo "    done"
echo "============================================================"
