from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, stop_id_sequences: List[List[int]]):
        super().__init__()
        self.stop_id_sequences = [seq for seq in stop_id_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.ndim != 2:
            return False
        for row in input_ids:
            row_ids = row.tolist()
            for stop_seq in self.stop_id_sequences:
                if len(row_ids) >= len(stop_seq) and row_ids[-len(stop_seq):] == stop_seq:
                    return True
        return False


@dataclass(frozen=True)
class AttentionHeadSpec:
    layer_idx: int
    head_idx: int
    num_heads: int
    head_dim: int

    @property
    def label(self) -> str:
        return f"L{self.layer_idx}H{self.head_idx}"


def get_nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for token in path.split("."):
        if not hasattr(current, token):
            return None
        current = getattr(current, token)
    return current


def get_decoder_layers(model: torch.nn.Module) -> Tuple[Any, str]:
    candidates = [
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
    ]
    for path in candidates:
        layers = get_nested_attr(model, path)
        if layers is not None:
            return layers, path
    raise ValueError("Cannot find decoder layers on this model.")


def get_attention_module(layer: torch.nn.Module) -> torch.nn.Module:
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attention"):
        return layer.attention
    raise ValueError("Cannot find self-attention module on decoder layer.")


def infer_attention_head_shape(
    model: torch.nn.Module,
    attn_module: torch.nn.Module,
) -> Tuple[int, int]:
    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(attn_module, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(model.config, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError("Cannot infer num_heads from attention module/config.")

    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None and hasattr(attn_module, "q_proj") and hasattr(attn_module.q_proj, "out_features"):
        head_dim = attn_module.q_proj.out_features // int(num_heads)
    if head_dim is None and hasattr(attn_module, "hidden_size"):
        head_dim = attn_module.hidden_size // int(num_heads)
    if head_dim is None and hasattr(model.config, "hidden_size"):
        head_dim = model.config.hidden_size // int(num_heads)
    if head_dim is None:
        raise ValueError("Cannot infer head_dim from attention module/config.")

    return int(num_heads), int(head_dim)


def list_attention_heads(model: torch.nn.Module) -> Tuple[List[AttentionHeadSpec], List[torch.nn.Module], str]:
    layers, layer_path = get_decoder_layers(model)
    heads: List[AttentionHeadSpec] = []
    attn_modules: List[torch.nn.Module] = []
    for layer_idx, layer in enumerate(layers):
        attn_module = get_attention_module(layer)
        attn_modules.append(attn_module)
        num_heads, head_dim = infer_attention_head_shape(model, attn_module)
        for head_idx in range(num_heads):
            heads.append(
                AttentionHeadSpec(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
            )
    return heads, attn_modules, layer_path


def parse_head_label(text: str) -> Tuple[int, int]:
    match = re.fullmatch(r"[Ll](\d+)[Hh](\d+)", text.strip())
    if not match:
        raise ValueError(f"Invalid head label '{text}'. Expected format LxHy.")
    return int(match.group(1)), int(match.group(2))


def get_input_device_for_model(model: torch.nn.Module) -> torch.device:
    # For models loaded with device_map=auto (accelerate hooks), choose a CUDA device
    # present in hf_device_map if possible.
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        # Prefer embedding module's device for input_ids.
        for k, dev in hf_map.items():
            if "embed_tokens" in str(k):
                if isinstance(dev, str) and dev.startswith("cuda"):
                    return torch.device(dev)
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
        for dev in hf_map.values():
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
    return next(model.parameters()).device


def prepare_inputs_for_model(
    tokenizer,
    text: str,
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    model_device = get_input_device_for_model(model)
    inputs["input_ids"] = inputs["input_ids"].to(model_device)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(model_device)
    return inputs


def resolve_target_token_id(
    tokenizer,
    target_token_text: str,
    target_token_id: int = -1,
) -> int:
    if target_token_id >= 0:
        return int(target_token_id)

    ids = tokenizer.encode(target_token_text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            "Target token text must map to exactly one token when --wait_token_id is not set. "
            f"Got text={target_token_text!r}, token_ids={ids}"
        )
    return int(ids[0])


@torch.no_grad()
def get_next_token_logit(
    model: torch.nn.Module,
    tokenizer,
    prompt_prefix: str,
    target_token_id: int,
) -> float:
    inputs = prepare_inputs_for_model(tokenizer, prompt_prefix, model)
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
    return float(outputs.logits[0, -1, int(target_token_id)].detach().item())


@torch.no_grad()
def generate_continuation(
    model: torch.nn.Module,
    tokenizer,
    prompt_prefix: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    stop_strings: Optional[List[str]] = None,
) -> Tuple[str, str, int]:
    inputs = prepare_inputs_for_model(tokenizer, prompt_prefix, model)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    generation_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask

    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    if stop_strings:
        stop_id_sequences: List[List[int]] = []
        for s in stop_strings:
            if not s:
                continue
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_id_sequences.append(ids)
        if stop_id_sequences:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokenSequences(stop_id_sequences)])

    outputs = model.generate(**generation_kwargs)
    new_token_ids = outputs[0, input_ids.shape[-1] :]
    continuation = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    full_text = prompt_prefix + continuation
    return continuation, full_text, int(new_token_ids.shape[0])


@torch.no_grad()
def forward_with_attentions(
    model: torch.nn.Module,
    input_ids: List[int],
) -> Sequence[Any]:
    device = get_input_device_for_model(model)
    tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tensor)
    outputs = model(
        input_ids=tensor,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=False,
    )
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        raise ValueError("Model did not return attentions. Try eager attention.")
    return attentions


def load_hf_model_and_tokenizer(
    model_name_or_path: str,
    load_in_half: bool,
    use_fast_tokenizer: bool,
    use_safetensors: bool,
    local_files_only: bool,
    device_map: Any,
    attn_implementation: str = "",
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer has no pad/unk/eos token; cannot proceed.")

    model_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
        "use_safetensors": use_safetensors,
        "local_files_only": local_files_only,
    }
    model_kwargs["dtype"] = torch.float16 if load_in_half else "auto"
    if attn_implementation.strip():
        model_kwargs["attn_implementation"] = attn_implementation.strip()

    # Important: do NOT call .cuda() or .half() after loading with device_map=auto.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def load_model_with_retries(
    args: argparse.Namespace,
    device_map_override: Optional[Any] = None,
    log_prefix: str = "[Info]",
):
    attempts: List[Tuple[bool, bool, str]] = [
        (args.use_fast_tokenizer, args.use_safetensors, "user_args"),
        (False, args.use_safetensors, "slow_tokenizer_retry"),
        (False, True, "force_safetensors_retry"),
        (False, False, "disable_safetensors_retry"),
    ]
    unique_attempts: List[Tuple[bool, bool, str]] = []
    seen = set()
    for use_fast, use_safe, tag in attempts:
        key = (use_fast, use_safe)
        if key in seen:
            continue
        seen.add(key)
        unique_attempts.append((use_fast, use_safe, tag))

    errors: List[str] = []
    for use_fast, use_safe, tag in unique_attempts:
        try:
            print(
                f"{log_prefix} Load attempt={tag} "
                f"(use_fast_tokenizer={use_fast}, use_safetensors={use_safe})"
            )
            model, tokenizer = load_hf_model_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                load_in_half=args.load_in_half,
                use_fast_tokenizer=use_fast,
                use_safetensors=use_safe,
                local_files_only=args.local_files_only,
                device_map=device_map_override if device_map_override is not None else args.device_map,
                attn_implementation=str(getattr(args, "attn_implementation", "") or ""),
            )
            return model, tokenizer, use_fast, use_safe
        except Exception as exc:
            err = f"{tag}: {type(exc).__name__}: {exc}"
            errors.append(err)
            print(f"{log_prefix} [Warn] {err}")

    merged = "\n".join(errors[-4:])
    raise RuntimeError(
        "Failed to load model after multiple retries.\n"
        "Common fixes:\n"
        "1) keep use_safetensors=True for Qwen3\n"
        "2) transformers>=4.51.0\n"
        "3) check model name and access\n"
        f"Recent errors:\n{merged}"
    )
