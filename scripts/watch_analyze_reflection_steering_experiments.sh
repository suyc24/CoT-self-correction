#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yucheng/experiment/Qwen2.5-Math}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-900}"
MAX_CHECKS="${MAX_CHECKS:-96}"
DATE_TAG="${DATE_TAG:-20260620}"

cd "$ROOT"

set +u
eval "$(/home/yucheng/bin/micromamba shell hook -s bash)"
micromamba activate qwen_math
set -u

OUT_ROOT="outputs/reflection_event_space_20260604"

has_summary() {
  [ -f "$OUT_ROOT/$1/summary.json" ]
}

all_done() {
  local item
  for item in "$@"; do
    if ! has_summary "$item"; then
      return 1
    fi
  done
  return 0
}

run_behavior_analysis() {
  local analysis_dir="$1"
  local report_dir="$2"
  shift 2
  local dirs_csv
  dirs_csv="$(IFS=,; echo "$*")"

  if [ ! -f "$analysis_dir/summary.json" ]; then
    python scripts/analyze_error_ack_behavior_gate_effect.py \
      --root "$OUT_ROOT" \
      --output_dir "$analysis_dir" \
      --dirs "$dirs_csv"
  else
    echo "[Skip] behavior analysis exists: $analysis_dir"
  fi

  if [ ! -f "$report_dir/REPORT.md" ]; then
    python scripts/report_mean_difference_behavior64_study.py \
      --analysis_dir "$analysis_dir" \
      --output_dir "$report_dir"
  else
    echo "[Skip] focus report exists: $report_dir"
  fi

  local position_dir="${analysis_dir}_event_time_position_control"
  if [ ! -f "$position_dir/summary.json" ]; then
    python scripts/analyze_reflection_event_time_position_control.py \
      --analysis_dir "$analysis_dir" \
      --output_dir "$position_dir" \
      --windows 4,8,16,32,64
  else
    echo "[Skip] position-control analysis exists: $position_dir"
  fi
}

start280_dirs=(
  "error_ack_mean_difference_behavior64_start280_n120_alpha0p5_pdecode_20260620"
  "error_ack_mean_difference_behavior64_start280_n120_alpha1p0_pdecode_20260620"
  "error_ack_mean_difference_behavior64_start280_n120_alpha2p0_pdecode_20260620"
  "error_ack_mean_difference_negated_behavior64_start280_n120_alpha2p0_pdecode_20260620"
  "random_rescaled6p029_seed01_behavior64_start280_n120_alpha2p0_pdecode_20260620"
  "random_rescaled6p029_seed02_behavior64_start280_n120_alpha2p0_pdecode_20260620"
)

banstop_dirs=(
  "error_ack_mean_difference_banstop16_behavior64_start400_n40_alpha2p0_pdecode_20260620"
  "error_ack_mean_difference_negated_banstop16_behavior64_start400_n40_alpha2p0_pdecode_20260620"
  "random_rescaled6p029_seed01_banstop16_behavior64_start400_n40_alpha2p0_pdecode_20260620"
)

forcewait_dirs=(
  "error_ack_mean_difference_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620"
  "error_ack_mean_difference_negated_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620"
  "random_rescaled6p029_seed01_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620"
)

for ((i = 1; i <= MAX_CHECKS; i++)); do
  date '+[%F %T] checking completion for steering analyses'

  if all_done "${start280_dirs[@]}"; then
    run_behavior_analysis \
      "$OUT_ROOT/error_ack_mean_difference_behavior64_start280_n120_analysis_${DATE_TAG}" \
      "$OUT_ROOT/error_ack_mean_difference_behavior64_start280_n120_focus_report_${DATE_TAG}" \
      "${start280_dirs[@]}"
  else
    echo "[Wait] start280 not complete."
  fi

  if all_done "${banstop_dirs[@]}"; then
    run_behavior_analysis \
      "$OUT_ROOT/error_ack_banstop16_behavior64_start400_n40_analysis_${DATE_TAG}" \
      "$OUT_ROOT/error_ack_banstop16_behavior64_start400_n40_focus_report_${DATE_TAG}" \
      "${banstop_dirs[@]}"
  else
    echo "[Wait] banstop not complete."
  fi

  if all_done "${forcewait_dirs[@]}"; then
    run_behavior_analysis \
      "$OUT_ROOT/error_ack_forcewait_behavior64_start440_n40_analysis_${DATE_TAG}" \
      "$OUT_ROOT/error_ack_forcewait_behavior64_start440_n40_focus_report_${DATE_TAG}" \
      "${forcewait_dirs[@]}"
  else
    echo "[Wait] forcewait not complete."
  fi

  if [ -f "$OUT_ROOT/error_ack_mean_difference_behavior64_start280_n120_analysis_${DATE_TAG}/summary.json" ] \
    && [ -f "$OUT_ROOT/error_ack_banstop16_behavior64_start400_n40_analysis_${DATE_TAG}/summary.json" ] \
    && [ -f "$OUT_ROOT/error_ack_forcewait_behavior64_start440_n40_analysis_${DATE_TAG}/summary.json" ]; then
    echo "[Done] all steering analyses are available."
    exit 0
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "[Exit] max checks reached before all analyses completed."
