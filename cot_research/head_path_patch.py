from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from .hidden_trajectory import replace_tuple_arg


class HeadOProjPatchHooks:
    """Capture or patch per-head attention outputs at the input of `o_proj`.

    Qwen-family attention concatenates per-head attention outputs before the
    output projection. At one-token decode forwards this tensor has shape
    `[batch, 1, num_heads * head_dim]`, so replacing a head slice provides a
    clean head-output patch before downstream residual writing.
    """

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        *,
        num_heads: int,
        head_dim: int,
        capture_layers: Sequence[int] = (),
        patch_vectors: Optional[Mapping[Tuple[int, int], torch.Tensor]] = None,
    ) -> None:
        self.layers = layers
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.capture_layers = [int(x) for x in capture_layers]
        self.patch_vectors = dict(patch_vectors or {})
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.values: Dict[Tuple[int, int], torch.Tensor] = {}
        self.patch_call_count = 0

    def __enter__(self) -> "HeadOProjPatchHooks":
        needed = set(self.capture_layers)
        needed.update(layer_idx for layer_idx, _head_idx in self.patch_vectors)
        for layer_idx in sorted(needed):
            attn = getattr(self.layers[int(layer_idx)], "self_attn", None)
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is None:
                raise ValueError(f"Layer {layer_idx} has no self_attn.o_proj")
            self.handles.append(o_proj.register_forward_pre_hook(self._make_hook(int(layer_idx))))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[1] != 1:
                return None
            if hidden.shape[-1] != self.num_heads * self.head_dim:
                raise ValueError(
                    f"Unexpected o_proj input width {hidden.shape[-1]}, expected {self.num_heads * self.head_dim}"
                )
            heads = hidden.view(hidden.shape[0], hidden.shape[1], self.num_heads, self.head_dim)
            if int(layer_idx) in self.capture_layers:
                for head_idx in range(self.num_heads):
                    self.values[(int(layer_idx), int(head_idx))] = heads[0, -1, head_idx].detach().float().cpu()
            layer_patches = {
                int(head_idx): vec
                for (patch_layer, head_idx), vec in self.patch_vectors.items()
                if int(patch_layer) == int(layer_idx)
            }
            if not layer_patches:
                return None
            patched = hidden.clone()
            patched_heads = patched.view(patched.shape[0], patched.shape[1], self.num_heads, self.head_dim)
            for head_idx, vec in layer_patches.items():
                patched_heads[0, -1, int(head_idx)] = vec.to(device=hidden.device, dtype=hidden.dtype)
                self.patch_call_count += 1
            return replace_tuple_arg(args, 0, patched)

        return hook
