#!/usr/bin/env python3
"""Download Qwen3-4B model to HuggingFace cache."""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import constants
constants.HF_HUB_ENABLE_HF_TRANSFER = False
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, gc

model_name = "Qwen/Qwen3-4B"
print(f"Downloading tokenizer for {model_name}...")
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("Tokenizer done. Downloading model weights...")
m = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
print(f"Model downloaded. Params: {sum(p.numel() for p in m.parameters())/1e9:.1f}B")
del m; gc.collect()
print("SUCCESS")
