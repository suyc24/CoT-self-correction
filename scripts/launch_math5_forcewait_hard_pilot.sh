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
  local out="$3"
  local cache="$4"
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
    "cd '$ROOT' && eval \"\$(/home/yucheng/bin/micromamba shell hook -s bash)\" && micromamba activate qwen_math && CUDA_VISIBLE_DEVICES=$gpu python scripts/run_reflection_patch_rescue.py --input_jsonl outputs/reflection_event_space_20260604/inputs/math_level5_first1000_wrong.jsonl --output_dir '$out' --start_idx 0 --max_examples 60 --gpu_id 0 --max_stage1_tokens 16384 --max_continuation_tokens 96 --capture_max_position_index 8 --layers 22 --sites post_attn --timings p0 --patch_types delete --intervention_layer 22 --intervention_site post_attn --gate_alpha 2.0 --gate_modes prefill_plus_decode --main_gate_mode prefill_plus_decode --gate_direction_cache_in '$cache' --forced_prefix_text '\$\$\\n\\nWait' --skip_patches --no-save_raw_activations --print_every 5 > '$log' 2>&1"
  echo "[Launch] $session gpu=$gpu out=$out"
}

launch_one \
  "math5_forcewait_mean_a20_g0" \
  0 \
  "outputs/reflection_event_space_20260604/error_ack_mean_difference_math5_forcewait_start0_n60_alpha2p0_pdecode_20260620" \
  "outputs/reflection_event_space_20260604/error_ack_as_gate_cache_20260619.pt"

launch_one \
  "math5_forcewait_neg_a20_g1" \
  1 \
  "outputs/reflection_event_space_20260604/error_ack_mean_difference_negated_math5_forcewait_start0_n60_alpha2p0_pdecode_20260620" \
  "outputs/reflection_event_space_20260604/error_ack_mean_difference_negated_gate_cache_20260619.pt"

launch_one \
  "math5_forcewait_rand1_a20_g2" \
  2 \
  "outputs/reflection_event_space_20260604/random_rescaled6p029_seed01_math5_forcewait_start0_n60_alpha2p0_pdecode_20260620" \
  "outputs/reflection_event_space_20260604/random_rescaled6p029_gate_cache_20260619.pt"

tmux ls 2>/dev/null | grep -E "math5_forcewait|$" || true
