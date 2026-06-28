#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yucheng/experiment/Qwen2.5-Math"
cd "$ROOT"

set +u
eval "$(/home/yucheng/bin/micromamba shell hook -s bash)"
micromamba activate qwen_math
set -u

mkdir -p logs/reflection_event_space_20260604

should_launch() {
  local name="$1"
  if [ -z "${FORCEWAIT_ONLY:-}" ]; then
    return 0
  fi
  case ",${FORCEWAIT_ONLY}," in
    *",$name,"*) return 0 ;;
    *) return 1 ;;
  esac
}

launch_one() {
  local name="$1"
  shift
  local session="$1"
  local gpu="$2"
  local out="$3"
  local cache="$4"
  local alpha="$5"
  local log="logs/reflection_event_space_20260604/${out##*/}.log"

  if ! should_launch "$name"; then
    echo "[Skip] not selected by FORCEWAIT_ONLY: $name"
    return
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "[Skip] tmux session exists: $session"
    return
  fi
  if [ -f "$out/summary.json" ]; then
    echo "[Skip] completed output exists: $out"
    return
  fi

  tmux new-session -d -s "$session" "cd '$ROOT' && eval \"\$(/home/yucheng/bin/micromamba shell hook -s bash)\" && micromamba activate qwen_math && CUDA_VISIBLE_DEVICES=$gpu python scripts/run_reflection_patch_rescue.py --output_dir '$out' --start_idx 440 --max_examples 40 --gpu_id 0 --max_stage1_tokens 4096 --max_continuation_tokens 64 --capture_max_position_index 8 --layers 22 --sites post_attn --timings p0 --patch_types delete --intervention_layer 22 --intervention_site post_attn --gate_alpha '$alpha' --gate_modes prefill_plus_decode --main_gate_mode prefill_plus_decode --gate_direction_cache_in '$cache' --forced_prefix_text '\$\$\\n\\nWait' --skip_patches --no-save_raw_activations --print_every 5 > '$log' 2>&1"
  echo "[Launch] $session gpu=$gpu out=$out"
}

launch_one "mean" "forcewait_mean_a20_g6" 6 "outputs/reflection_event_space_20260604/error_ack_mean_difference_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620" "outputs/reflection_event_space_20260604/error_ack_as_gate_cache_20260619.pt" "2.0"
launch_one "neg" "forcewait_neg_a20_g7" 7 "outputs/reflection_event_space_20260604/error_ack_mean_difference_negated_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620" "outputs/reflection_event_space_20260604/error_ack_mean_difference_negated_gate_cache_20260619.pt" "2.0"
launch_one "rand1" "forcewait_rand1_a20_g8" 8 "outputs/reflection_event_space_20260604/random_rescaled6p029_seed01_forcewait_behavior64_start440_n40_alpha2p0_pdecode_20260620" "outputs/reflection_event_space_20260604/random_rescaled6p029_gate_cache_20260619.pt" "2.0"

tmux ls | grep -E "forcewait_|$" || true
