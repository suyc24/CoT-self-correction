from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class HeadSpec:
    layer_idx: int
    head_idx: int
    num_heads: int
    head_dim: int

    @property
    def label(self) -> str:
        return f"L{self.layer_idx}H{self.head_idx}"


class SingleHeadAblationHook:
    """Zero one attention head slice at o_proj input of a single layer."""

    def __init__(
        self,
        attn_module: torch.nn.Module,
        head_idx: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        self.attn_module = attn_module
        self.head_idx = head_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.call_count = 0
        self.first_call_abs_mean_before: Optional[float] = None
        self.first_call_abs_mean_after: Optional[float] = None

    def __enter__(self) -> "SingleHeadAblationHook":
        if not hasattr(self.attn_module, "o_proj"):
            raise ValueError("Attention module has no o_proj; cannot attach single-head ablation hook.")
        self.handle = self.attn_module.o_proj.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _pre_hook(self, module: torch.nn.Module, args: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
        if len(args) == 0:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return None

        start = self.head_idx * self.head_dim
        end = (self.head_idx + 1) * self.head_dim
        if x.size(-1) < end:
            raise ValueError(
                f"o_proj input last dim {x.size(-1)} is smaller than requested slice [{start}:{end}]."
            )

        self.call_count += 1
        if self.call_count == 1:
            self.first_call_abs_mean_before = float(x[..., start:end].detach().abs().mean().item())

        x_masked = x.clone()
        x_masked[..., start:end] = 0

        if self.call_count == 1:
            self.first_call_abs_mean_after = float(x_masked[..., start:end].detach().abs().mean().item())

        if len(args) == 1:
            return (x_masked,)
        return (x_masked, *args[1:])


def get_nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for token in path.split("."):
        if not hasattr(cur, token):
            return None
        cur = getattr(cur, token)
    return cur


def get_decoder_layers(model: torch.nn.Module) -> Tuple[Any, str]:
    candidates = [
        "model.layers",          # Qwen2/Qwen2.5 style
        "transformer.h",         # GPT-like
        "model.decoder.layers",  # some seq2seq decoders
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


def infer_head_shape(
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
        head_dim = attn_module.q_proj.out_features // num_heads
    if head_dim is None and hasattr(attn_module, "hidden_size"):
        head_dim = attn_module.hidden_size // num_heads
    if head_dim is None and hasattr(model.config, "hidden_size"):
        head_dim = model.config.hidden_size // num_heads
    if head_dim is None:
        raise ValueError("Cannot infer head_dim from attention module/config.")

    return int(num_heads), int(head_dim)


def list_all_heads(model: torch.nn.Module) -> Tuple[List[HeadSpec], List[torch.nn.Module], str]:
    layers, layer_path = get_decoder_layers(model)
    all_heads: List[HeadSpec] = []
    attn_modules: List[torch.nn.Module] = []
    for layer_idx, layer in enumerate(layers):
        attn_module = get_attention_module(layer)
        attn_modules.append(attn_module)
        if not hasattr(attn_module, "o_proj"):
            raise ValueError(f"Layer {layer_idx} attention has no o_proj; unsupported for this ablation.")
        num_heads, head_dim = infer_head_shape(model, attn_module)
        for head_idx in range(num_heads):
            all_heads.append(HeadSpec(layer_idx, head_idx, num_heads, head_dim))
    return all_heads, attn_modules, layer_path


def parse_head_label(text: str) -> Tuple[int, int]:
    m = re.fullmatch(r"[Ll](\d+)[Hh](\d+)", text.strip())
    if not m:
        raise ValueError(f"Invalid head label '{text}'. Expected format LxHy, e.g. L0H0.")
    return int(m.group(1)), int(m.group(2))


def filter_heads(all_heads: List[HeadSpec], head_spec: str) -> List[HeadSpec]:
    if not head_spec.strip():
        return all_heads
    head_map = {(h.layer_idx, h.head_idx): h for h in all_heads}
    selected: List[HeadSpec] = []
    for token in head_spec.split(","):
        if not token.strip():
            continue
        layer_idx, head_idx = parse_head_label(token)
        key = (layer_idx, head_idx)
        if key not in head_map:
            raise ValueError(f"Head {token} not found in model.")
        selected.append(head_map[key])
    if not selected:
        raise ValueError("No valid head selected by --head_spec.")
    return selected


def select_single_head(all_heads: List[HeadSpec], args: argparse.Namespace) -> HeadSpec:
    head_map = {(h.layer_idx, h.head_idx): h for h in all_heads}

    target_label = args.ablate_head.strip()
    if not target_label:
        if args.head_spec.strip():
            tokens = [x.strip() for x in args.head_spec.split(",") if x.strip()]
            if not tokens:
                raise ValueError("--head_spec is set but empty after parsing.")
            if len(tokens) > 1:
                print(
                    f"[Warn] --head_spec has {len(tokens)} heads; only the first one will be used: {tokens[0]}"
                )
            target_label = tokens[0]
        else:
            target_label = all_heads[0].label
            print(f"[Info] No --ablate_head specified; defaulting to {target_label}")

    layer_idx, head_idx = parse_head_label(target_label)
    key = (layer_idx, head_idx)
    if key not in head_map:
        raise ValueError(f"Head {target_label} not found in model.")
    return head_map[key]
