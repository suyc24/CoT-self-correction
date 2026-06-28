#!/bin/bash
# Run all four reasoning exploration experiments sequentially.
# Usage: bash scripts/run_all_reasoning_experiments.sh [MODEL] [GPU_IDS]
#
# Example:
#   bash scripts/run_all_reasoning_experiments.sh Qwen/Qwen3-4B 0

# Enable academic acceleration proxy for HuggingFace access
source /etc/network_turbo 2>/dev/null || true
export PYTHONUNBUFFERED=1
export PATH=/root/miniconda3/bin:$PATH

MODEL="${1:-Qwen/Qwen3-4B}"
GPU="${2:-0}"
PYTHON="${3:-python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="outputs/reasoning_exploration_${TIMESTAMP}"

mkdir -p "${BASE_DIR}"

echo "============================================"
echo "Reasoning Exploration Experiments"
echo "Model: ${MODEL}"
echo "GPU: ${GPU}"
echo "Python: ${PYTHON}"
echo "Output: ${BASE_DIR}"
echo "============================================"

# Experiment 1: Multi-sample reasoning trace analysis
echo ""
echo "[EXP1] Starting: Multi-sample reasoning trace analysis..."
${PYTHON} scripts/run_exp1_reasoning_trace.py \
    --model "${MODEL}" \
    --output-dir "${BASE_DIR}/exp1_reasoning_trace" \
    --num-samples 10 \
    --temperature 0.7 \
    --max-tokens 4096 \
    --gpu-ids "${GPU}" \
    2>&1 | tee "${BASE_DIR}/exp1_log.txt"
echo "[EXP1] Done."

# Experiment 4: Problem rephrasing (also uses vLLM, run before HF experiments)
echo ""
echo "[EXP4] Starting: Problem rephrasing analysis..."
${PYTHON} scripts/run_exp4_rephrasing.py \
    --model "${MODEL}" \
    --output-dir "${BASE_DIR}/exp4_rephrasing" \
    --max-tokens 4096 \
    --gpu-ids "${GPU}" \
    2>&1 | tee "${BASE_DIR}/exp4_log.txt"
echo "[EXP4] Done."

# Experiment 2: Token perturbation (uses HF backend)
echo ""
echo "[EXP2] Starting: Token perturbation at key positions..."
${PYTHON} scripts/run_exp2_perturbation.py \
    --model "${MODEL}" \
    --output-dir "${BASE_DIR}/exp2_perturbation" \
    --max-tokens 4096 \
    --gpu-ids "${GPU}" \
    2>&1 | tee "${BASE_DIR}/exp2_log.txt"
echo "[EXP2] Done."

# Experiment 3: Hidden state extraction (uses HF, depends on Exp1 output)
echo ""
echo "[EXP3] Starting: Hidden state extraction and visualization..."
${PYTHON} scripts/run_exp3_hidden_states.py \
    --model "${MODEL}" \
    --output-dir "${BASE_DIR}/exp3_hidden_states" \
    --gpu-ids "${GPU}" \
    --exp1-traces "${BASE_DIR}/exp1_reasoning_trace/full_traces.jsonl" \
    --exp1-analysis "${BASE_DIR}/exp1_reasoning_trace/raw_traces.jsonl" \
    2>&1 | tee "${BASE_DIR}/exp3_log.txt"
echo "[EXP3] Done."

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Results in: ${BASE_DIR}"
echo "============================================"
ls -la "${BASE_DIR}"/*/
