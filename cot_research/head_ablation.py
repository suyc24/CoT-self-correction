from __future__ import annotations

import argparse
from contextlib import ExitStack
from typing import Any, List, Optional, Tuple

import torch

from .model_utils import (
    AttentionHeadSpec,
    get_attention_module,
    infer_attention_head_shape,
    list_attention_heads,
    parse_head_label,
)

HeadSpec = AttentionHeadSpec


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


class MultiHeadAblationHookSet:
    """Enable multiple SingleHeadAblationHook instances at the same time."""

    def __init__(self, attn_modules: List[torch.nn.Module], heads: List[HeadSpec]) -> None:
        self.attn_modules = attn_modules
        self.heads = heads
        self._stack: Optional[ExitStack] = None
        self.hooks: List[SingleHeadAblationHook] = []

    def __enter__(self) -> "MultiHeadAblationHookSet":
        stack = ExitStack()
        self.hooks = []
        for head in self.heads:
            hook = SingleHeadAblationHook(
                attn_module=self.attn_modules[head.layer_idx],
                head_idx=head.head_idx,
                num_heads=head.num_heads,
                head_dim=head.head_dim,
            )
            stack.enter_context(hook)
            self.hooks.append(hook)
        self._stack = stack
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

def list_all_heads(model: torch.nn.Module) -> Tuple[List[HeadSpec], List[torch.nn.Module], str]:
    all_heads, attn_modules, layer_path = list_attention_heads(model)
    for layer_idx, attn_module in enumerate(attn_modules):
        if not hasattr(attn_module, "o_proj"):
            raise ValueError(f"Layer {layer_idx} attention has no o_proj; unsupported for this ablation.")
    return list(all_heads), attn_modules, layer_path


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
