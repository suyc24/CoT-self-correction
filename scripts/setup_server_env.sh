#!/bin/bash
# Setup script for the seetacloud GPU server
# Installs required packages and downloads the Qwen3-4B model

set -e

CONDA_PYTHON=/root/miniconda3/bin/python3
CONDA_PIP=/root/miniconda3/bin/pip
WORK_DIR=/root/autodl-tmp/reasoning_experiments

echo "=== Setting up environment ==="
echo "1. Enabling academic acceleration..."
source /etc/network_turbo 2>/dev/null || true

echo "2. Installing required packages..."
$CONDA_PIP install --no-cache-dir \
    transformers>=4.51.0 \
    accelerate>=0.30.0 \
    vllm \
    scikit-learn \
    umap-learn \
    sentencepiece \
    safetensors \
    sympy \
    numpy \
    regex \
    word2number \
    2>&1 | tail -20

echo "3. Verifying installations..."
$CONDA_PYTHON -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import transformers; print(f'Transformers: {transformers.__version__}')
import sklearn; print(f'Scikit-learn: {sklearn.__version__}')
try:
    import vllm; print(f'vLLM: {vllm.__version__}')
except: print('vLLM: import failed (may need restart)')
"

echo "4. Downloading Qwen3-4B model..."
$CONDA_PYTHON -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen3-4B'
print(f'Downloading tokenizer for {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('Tokenizer downloaded.')

print(f'Downloading model {model_name}...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print('Model downloaded successfully.')
del model
import gc; gc.collect()
torch.cuda.empty_cache()
print('Cache cleared.')
"

echo "5. Creating work directory..."
mkdir -p $WORK_DIR

echo "=== Setup complete ==="
