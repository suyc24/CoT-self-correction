#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yucheng/experiment/Qwen2.5-Math}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
MAX_CHECKS="${MAX_CHECKS:-72}"

cd "$ROOT"

names=("mean" "neg" "rand1")
banstop_sessions=("banstop_mean_a20_g6" "banstop_neg_a20_g7" "banstop_rand1_a20_g8")
forcewait_sessions=("forcewait_mean_a20_g6" "forcewait_neg_a20_g7" "forcewait_rand1_a20_g8")
forcewait_gpus=("6" "7" "8")
forcewait_outputs=(
  "outputs/reflection_event_space_20260604/error_ack_mean_difference_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620"
  "outputs/reflection_event_space_20260604/error_ack_mean_difference_negated_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620"
  "outputs/reflection_event_space_20260604/random_rescaled6p029_seed01_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620"
)

session_gone() {
  ! tmux has-session -t "$1" 2>/dev/null
}

forcewait_started_or_done() {
  local session="$1"
  local out="$2"
  if tmux has-session -t "$session" 2>/dev/null; then
    return 0
  fi
  [ -e "$out/behavior_rows.jsonl" ] || [ -e "$out/behavior_rows.partial.jsonl" ] || [ -e "$out/summary.json" ]
}

gpu_available() {
  local gpu="$1"
  local used
  used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v gpu="$gpu" '$1 + 0 == gpu {gsub(/ /, "", $2); print $2}')"
  [ -n "$used" ] || return 1
  [ "$used" -le 1000 ]
}

all_forcewait_started_or_done() {
  local i
  for i in "${!names[@]}"; do
    if ! forcewait_started_or_done "${forcewait_sessions[$i]}" "${forcewait_outputs[$i]}"; then
      return 1
    fi
  done
  return 0
}

for ((i = 1; i <= MAX_CHECKS; i++)); do
  date '+[%F %T] checking banstop/forcewait state'
  if all_forcewait_started_or_done; then
    echo "[Done] all forcewait jobs are started or complete."
    exit 0
  fi

  launched=0
  for j in "${!names[@]}"; do
    name="${names[$j]}"
    if forcewait_started_or_done "${forcewait_sessions[$j]}" "${forcewait_outputs[$j]}"; then
      echo "[Skip] forcewait $name already started or has output."
      continue
    fi
    if ! session_gone "${banstop_sessions[$j]}"; then
      echo "[Wait] banstop $name still running."
      continue
    fi
    if ! gpu_available "${forcewait_gpus[$j]}"; then
      echo "[Wait] GPU ${forcewait_gpus[$j]} not available for forcewait $name."
      continue
    fi
    echo "[Launch] starting forcewait $name on GPU ${forcewait_gpus[$j]}."
    FORCEWAIT_ONLY="$name" bash scripts/launch_error_ack_forcewait_pilot.sh
    launched=1
  done

  if [ "$launched" -eq 0 ]; then
    echo "[Wait] no new forcewait jobs launched this check."
  fi
  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "[Exit] max checks reached before all forcewait jobs started."
