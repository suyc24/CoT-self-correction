from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .model_utils import AttentionHeadSpec, list_attention_heads, parse_head_label
from .registry import Registry


INTERVENTION_REGISTRY = Registry()
INTERVENTION_POINT = "self_attn.o_proj.pre"
INTERVENTION_TENSOR_SHAPE = "[batch, seq, hidden] with per-head slice [batch, seq, head_dim]"


HeadTarget = AttentionHeadSpec


@dataclass
class HookDebugState:
    call_count: int = 0
    first_call_abs_mean_before: Optional[float] = None
    first_call_abs_mean_after: Optional[float] = None
    intervention_point: str = INTERVENTION_POINT
    tensor_shape_note: str = INTERVENTION_TENSOR_SHAPE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "first_call_abs_mean_before": self.first_call_abs_mean_before,
            "first_call_abs_mean_after": self.first_call_abs_mean_after,
            "intervention_point": self.intervention_point,
            "tensor_shape_note": self.tensor_shape_note,
        }


class LayerHeadInterventionHook:
    def __init__(self, attn_module: torch.nn.Module, operations: List[Tuple[HeadTarget, float]]) -> None:
        self.attn_module = attn_module
        self.operations = operations
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.debug_state = HookDebugState()

    def __enter__(self) -> "LayerHeadInterventionHook":
        if not hasattr(self.attn_module, "o_proj"):
            raise ValueError("Attention module has no o_proj; cannot attach head intervention hook.")
        self.handle = self.attn_module.o_proj.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _pre_hook(self, module: torch.nn.Module, args: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return None
        x_out = x.clone()
        self.debug_state.call_count += 1
        for target, scale in self.operations:
            start = target.head_idx * target.head_dim
            end = (target.head_idx + 1) * target.head_dim
            if x.size(-1) < end:
                raise ValueError(
                    f"o_proj input last dim {x.size(-1)} is smaller than requested slice [{start}:{end}]."
                )
            if self.debug_state.call_count == 1 and self.debug_state.first_call_abs_mean_before is None:
                self.debug_state.first_call_abs_mean_before = float(x[..., start:end].detach().abs().mean().item())
            x_out[..., start:end] = x_out[..., start:end] * scale
            if self.debug_state.call_count == 1:
                self.debug_state.first_call_abs_mean_after = float(x_out[..., start:end].detach().abs().mean().item())
        if len(args) == 1:
            return (x_out,)
        return (x_out, *args[1:])


class MultiLayerHeadIntervention:
    def __init__(self, attn_modules: List[torch.nn.Module], operations: Iterable[Tuple[HeadTarget, float]]) -> None:
        grouped: Dict[int, List[Tuple[HeadTarget, float]]] = defaultdict(list)
        for target, scale in operations:
            grouped[target.layer_idx].append((target, scale))
        self.attn_modules = attn_modules
        self.grouped = grouped
        self.hooks: List[LayerHeadInterventionHook] = []
        self._stack: Optional[ExitStack] = None

    def __enter__(self) -> "MultiLayerHeadIntervention":
        stack = ExitStack()
        self.hooks = []
        for layer_idx, ops in sorted(self.grouped.items()):
            hook = LayerHeadInterventionHook(self.attn_modules[layer_idx], ops)
            stack.enter_context(hook)
            self.hooks.append(hook)
        self._stack = stack
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def merged_debug_state(self) -> Dict[str, Any]:
        return {
            "hook_count": len(self.hooks),
            "layers": [hook.debug_state.to_dict() for hook in self.hooks],
        }


@INTERVENTION_REGISTRY.register("zero")
def build_zero_intervention(heads: List[HeadTarget], params: Dict[str, Any]) -> List[Tuple[HeadTarget, float]]:
    return [(head, 0.0) for head in heads]


@INTERVENTION_REGISTRY.register("scale")
def build_scale_intervention(heads: List[HeadTarget], params: Dict[str, Any]) -> List[Tuple[HeadTarget, float]]:
    scale = float(params.get("scale", 1.0))
    return [(head, scale) for head in heads]


@INTERVENTION_REGISTRY.register("identity")
def build_identity_intervention(heads: List[HeadTarget], params: Dict[str, Any]) -> List[Tuple[HeadTarget, float]]:
    return [(head, 1.0) for head in heads]

def list_model_heads(model: torch.nn.Module) -> Tuple[List[HeadTarget], List[torch.nn.Module], str]:
    heads, attn_modules, layer_path = list_attention_heads(model)
    return list(heads), attn_modules, layer_path


def resolve_head_targets(model: torch.nn.Module, head_labels: List[str]) -> Tuple[List[HeadTarget], List[torch.nn.Module], str]:
    all_heads, attn_modules, layer_path = list_model_heads(model)
    head_lookup = {head.label: head for head in all_heads}
    selected: List[HeadTarget] = []
    for label in head_labels:
        layer_idx, head_idx = parse_head_label(label)
        key = f"L{layer_idx}H{head_idx}"
        if key not in head_lookup:
            raise ValueError(f"Head {label} not found in model.")
        selected.append(head_lookup[key])
    return selected, attn_modules, layer_path
