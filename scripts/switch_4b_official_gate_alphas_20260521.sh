#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/yucheng/experiment/Qwen2.5-Math"
OUT_ROOT="outputs/stateful_tamper_attention_20260510/reflection_gate_vllm_38912_aime24_official_20260521/4b_aime24_repeat10_official_baseline_gate_alpha_sweep_38912"
QUEUE_ROOT="logs/vllm_4b_aime24_official_38912_20260521/queue_state"
ORCH_SESSION="rgate4bofficial38912_orchestrator"
LAUNCH_SCRIPT="scripts/launch_vllm_4b_aime24_official_38912_20260521.sh"

cd "${ROOT_DIR}"
echo "[switcher] waiting for 10 complete baseline repeats"

while true; do
  failed=0
  if [[ -d "${QUEUE_ROOT}/baseline/failed" ]]; then
    failed="$(find "${QUEUE_ROOT}/baseline/failed" -maxdepth 1 -type f -name "*.task" | wc -l)"
  fi
  if [[ "${failed}" -ne 0 ]]; then
    echo "[switcher] baseline failed tasks detected; exiting"
    find "${QUEUE_ROOT}/baseline/failed" -maxdepth 1 -type f -name "*.task" -printf "%f\n" | sort
    exit 1
  fi

  complete=0
  for repeat_idx in 1 2 3 4 5 6 7 8 9 10; do
    rows="${OUT_ROOT}/shards/repeat${repeat_idx}_baseline/eval_rows.jsonl"
    if [[ -s "${rows}" ]] && [[ "$(wc -l < "${rows}")" -eq 30 ]]; then
      complete=$((complete + 1))
    fi
  done

  echo "[switcher] complete_baseline_repeats=${complete}/10"
  if [[ "${complete}" -eq 10 ]]; then
    break
  fi
  sleep 60
done

echo "[switcher] baseline complete; restarting orchestrator with alpha 1,0.5,0.25,0.125"
tmux kill-session -t "${ORCH_SESSION}" 2>/dev/null || true
rm -rf "${QUEUE_ROOT}/gate"
find "${OUT_ROOT}/shards" -maxdepth 1 -type d -name "repeat*_gate_*" -exec rm -rf {} +
bash "${LAUNCH_SCRIPT}"
