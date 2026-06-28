#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/yucheng/experiment/Qwen2.5-Math}"
cd "$ROOT"

set +u
eval "$(/home/yucheng/bin/micromamba shell hook -s bash)"
micromamba activate qwen_math
set -u

mkdir -p logs/reflection_event_space_20260604

launch_one() {
  local session="$1"
  local gpu="$2"
  local start="$3"
  local label="$4"
  local cache="$5"
  local out="outputs/reflection_event_space_20260604/${label}_math5_forcewait_start${start}_n60_alpha2p0_pdecode_20260620"
  local log="logs/reflection_event_space_20260604/${out##*/}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[Skip] tmux session exists: $session"
    return
  fi
  if [ -f "$out/summary.json" ]; then
    echo "[Skip] completed output exists: $out"
    return
  fi

  tmux new-session -d -s "$session" \
    "cd '$ROOT' && eval \"\$(/home/yucheng/bin/micromamba shell hook -s bash)\" && micromamba activate qwen_math && CUDA_VISIBLE_DEVICES=$gpu python scripts/run_reflection_patch_rescue.py --input_jsonl outputs/reflection_event_space_20260604/inputs/math_level5_first1000_wrong.jsonl --output_dir '$out' --start_idx '$start' --max_examples 60 --gpu_id 0 --max_stage1_tokens 16384 --max_continuation_tokens 96 --capture_max_position_index 8 --layers 22 --sites post_attn --timings p0 --patch_types delete --intervention_layer 22 --intervention_site post_attn --gate_alpha 2.0 --gate_modes prefill_plus_decode --main_gate_mode prefill_plus_decode --gate_direction_cache_in '$cache' --forced_prefix_text '\$\$\\n\\nWait' --skip_patches --no-save_raw_activations --print_every 5 > '$log' 2>&1"
  echo "[Launch] $session gpu=$gpu start=$start out=$out"
}

starts=(60 120 180)
gpu=0
for start in "${starts[@]}"; do
  launch_one "math5fw_s${start}_mean_g${gpu}" "$gpu" "$start" \
    "error_ack_mean_difference" \
    "outputs/reflection_event_space_20260604/error_ack_as_gate_cache_20260619.pt"
  gpu=$((gpu + 1))

  launch_one "math5fw_s${start}_neg_g${gpu}" "$gpu" "$start" \
    "error_ack_mean_difference_negated" \
    "outputs/reflection_event_space_20260604/error_ack_mean_difference_negated_gate_cache_20260619.pt"
  gpu=$((gpu + 1))

  launch_one "math5fw_s${start}_rand1_g${gpu}" "$gpu" "$start" \
    "random_rescaled6p029_seed01" \
    "outputs/reflection_event_space_20260604/random_rescaled6p029_gate_cache_20260619.pt"
  gpu=$((gpu + 1))
done

tmux ls 2>/dev/null | grep -E "math5fw_s|$" || true
