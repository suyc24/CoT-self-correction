from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .generation import _match_stop_sequence_suffix, _sample_next_token_id
from .model_utils import get_input_device_for_model
from .stateful_tampering import logsumexp_token_set


SUPPORTED_TRAJECTORY_SITES = {"block_input", "post_attn", "block_output"}


def tensor_output(output: Any) -> Optional[torch.Tensor]:
    hidden = output[0] if isinstance(output, tuple) else output
    return hidden if isinstance(hidden, torch.Tensor) else None


def replace_tuple_arg(args: Tuple[Any, ...], idx: int, value: Any) -> Tuple[Any, ...]:
    out = list(args)
    out[idx] = value
    return tuple(out)


def tensor_normed(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    norm = float(vec.norm().item())
    if norm < eps:
        return torch.zeros_like(vec)
    return vec / norm


@dataclass(frozen=True)
class AddIntervention:
    layer_idx: int
    site: str
    add_vector: torch.Tensor


class BoundaryTrajectoryHooks:
    """Capture residual-boundary states and optionally add one steering vector.

    The site definitions match `run_stateful_tamper_boundary_patch.py`:
    `post_attn` is the residual after attention (`resid + attn_out`), while
    the intervention is implemented by adding the vector to `attn_out`.
    """

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        *,
        capture_layer_indices: Sequence[int],
        capture_sites: Sequence[str],
        intervention: Optional[AddIntervention] = None,
        capture_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.layers = layers
        self.capture_layer_indices = [int(x) for x in capture_layer_indices]
        self.capture_sites = [str(x) for x in capture_sites]
        self.intervention = intervention
        self.capture_dtype = capture_dtype
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.values: Dict[Tuple[str, int], torch.Tensor] = {}
        self._pre_attn_resid: Dict[int, torch.Tensor] = {}
        self.add_hook_call_count = 0
        self.add_norm: Optional[float] = None

    def __enter__(self) -> "BoundaryTrajectoryHooks":
        needed: Dict[int, set[str]] = {}
        for idx in self.capture_layer_indices:
            needed.setdefault(int(idx), set()).update(self.capture_sites)
        if self.intervention is not None:
            needed.setdefault(int(self.intervention.layer_idx), set()).add(str(self.intervention.site))

        for idx, sites in sorted(needed.items()):
            layer = self.layers[int(idx)]
            if "block_input" in sites:
                self.handles.append(layer.register_forward_pre_hook(self._make_block_input_hook(int(idx))))
            if "post_attn" in sites:
                self.handles.append(layer.input_layernorm.register_forward_pre_hook(self._make_ln_pre_hook(int(idx))))
                self.handles.append(layer.self_attn.register_forward_hook(self._make_attn_post_hook(int(idx))))
            if "block_output" in sites:
                self.handles.append(layer.register_forward_hook(self._make_block_output_hook(int(idx))))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._pre_attn_resid = {}

    def _should_capture(self, layer_idx: int, site: str) -> bool:
        return int(layer_idx) in self.capture_layer_indices and site in self.capture_sites

    def _is_intervention_site(self, layer_idx: int, site: str) -> bool:
        return (
            self.intervention is not None
            and int(self.intervention.layer_idx) == int(layer_idx)
            and str(self.intervention.site) == str(site)
        )

    def _add_vec(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.intervention is None:
            raise ValueError("No intervention configured.")
        return self.intervention.add_vector.to(device=hidden.device, dtype=hidden.dtype)

    def _record_add(self, add_vec: torch.Tensor) -> None:
        self.add_hook_call_count += 1
        self.add_norm = float(add_vec.detach().float().norm().item())

    def _record_capture(self, layer_idx: int, site: str, hidden: torch.Tensor) -> None:
        if self._should_capture(layer_idx, site):
            self.values[(site, int(layer_idx))] = hidden[0, -1].detach().to(device="cpu", dtype=self.capture_dtype)

    def _make_block_input_hook(self, layer_idx: int):
        def hook(module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[1] != 1:
                return None
            patched = hidden
            if self._is_intervention_site(layer_idx, "block_input"):
                patched = hidden.clone()
                add_vec = self._add_vec(patched)
                self._record_add(add_vec)
                patched[0, -1] = patched[0, -1] + add_vec
            self._record_capture(layer_idx, "block_input", patched)
            if patched is not hidden:
                return replace_tuple_arg(args, 0, patched)
            return None

        return hook

    def _make_ln_pre_hook(self, layer_idx: int):
        def hook(module, args):
            if args and isinstance(args[0], torch.Tensor) and args[0].ndim == 3 and args[0].shape[1] == 1:
                self._pre_attn_resid[int(layer_idx)] = args[0].detach()

        return hook

    def _make_attn_post_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            attn_out = tensor_output(output)
            resid = self._pre_attn_resid.pop(int(layer_idx), None)
            if attn_out is None or resid is None or attn_out.ndim != 3 or attn_out.shape[1] != 1:
                return output

            modified_attn = attn_out
            post_attn = resid + attn_out
            if self._is_intervention_site(layer_idx, "post_attn"):
                modified_attn = attn_out.clone()
                add_vec = self._add_vec(modified_attn)
                self._record_add(add_vec)
                modified_attn[0, -1] = modified_attn[0, -1] + add_vec
                post_attn = resid + modified_attn
            self._record_capture(layer_idx, "post_attn", post_attn)
            if modified_attn is not attn_out:
                if isinstance(output, tuple):
                    return (modified_attn,) + output[1:]
                return modified_attn
            return output

        return hook

    def _make_block_output_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = tensor_output(output)
            if hidden is None or hidden.ndim != 3 or hidden.shape[1] != 1:
                return output
            patched = hidden
            if self._is_intervention_site(layer_idx, "block_output"):
                patched = hidden.clone()
                add_vec = self._add_vec(patched)
                self._record_add(add_vec)
                patched[0, -1] = patched[0, -1] + add_vec
            self._record_capture(layer_idx, "block_output", patched)
            if patched is not hidden:
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return output

        return hook


@torch.no_grad()
def prefill_before_final_full_ids(model: torch.nn.Module, full_ids: Sequence[int]) -> Tuple[Any, List[int], int]:
    if len(full_ids) < 2:
        raise ValueError("Need at least two token ids to prefill before the final token.")
    device = get_input_device_for_model(model)
    prefix = [int(x) for x in full_ids[:-1]]
    final_token_id = int(full_ids[-1])
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past = getattr(outputs, "past_key_values", None)
    if past is None:
        raise ValueError("Model did not return past_key_values during prefill.")
    return past, prefix, final_token_id


@torch.no_grad()
def forward_one_with_boundary_hooks(
    model: torch.nn.Module,
    *,
    past: Any,
    full_ids_before_token: Sequence[int],
    token_id: int,
    layers: Sequence[torch.nn.Module],
    capture_layer_indices: Sequence[int],
    capture_sites: Sequence[str],
    intervention: Optional[AddIntervention] = None,
    capture_dtype: torch.dtype = torch.float16,
) -> Tuple[Any, torch.Tensor, Dict[Tuple[str, int], torch.Tensor], Dict[str, Any]]:
    device = get_input_device_for_model(model)
    token_id = int(token_id)
    full_len = len(full_ids_before_token) + 1
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, full_len), dtype=torch.long, device=device)
    captures: Dict[Tuple[str, int], torch.Tensor] = {}
    debug: Dict[str, Any] = {}

    with BoundaryTrajectoryHooks(
        layers,
        capture_layer_indices=capture_layer_indices,
        capture_sites=capture_sites,
        intervention=intervention,
        capture_dtype=capture_dtype,
    ) as hooks:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        captures = dict(hooks.values)
        debug = {
            "add_hook_call_count": int(hooks.add_hook_call_count),
            "add_norm": hooks.add_norm,
        }

    new_past = getattr(outputs, "past_key_values", None)
    if new_past is None:
        raise ValueError("Model did not return past_key_values during trajectory forward.")
    return new_past, outputs.logits[0, -1], captures, debug


def logit_metrics(
    logits: torch.Tensor,
    *,
    token_sets: Dict[str, Sequence[int]],
    token_ids: Sequence[int],
) -> Dict[str, Any]:
    logits_cpu = logits.detach().float().cpu()
    row: Dict[str, Any] = {}
    for name, ids in token_sets.items():
        row[f"{name}_logsum"] = logsumexp_token_set(logits_cpu, ids)
    if "reflect" in token_sets and "stop" in token_sets:
        row["reflect_vs_stop"] = row["reflect_logsum"] - row["stop_logsum"]
    for token_id in token_ids:
        token_id = int(token_id)
        if 0 <= token_id < int(logits_cpu.shape[-1]):
            row[f"logit_tok_{token_id}"] = float(logits_cpu[token_id].item())
    return row


@torch.no_grad()
def run_hidden_trajectory(
    model: torch.nn.Module,
    tokenizer,
    *,
    full_ids: Sequence[int],
    layers: Sequence[torch.nn.Module],
    capture_layer_indices: Sequence[int],
    capture_sites: Sequence[str],
    max_new_tokens: int,
    capture_max_position_index: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    stop_id_sequences: Sequence[Sequence[int]],
    forced_continuation_ids: Optional[Sequence[int]] = None,
    intervention: Optional[AddIntervention] = None,
    token_sets: Optional[Dict[str, Sequence[int]]] = None,
    tracked_token_ids: Sequence[int] = (),
    capture_dtype: torch.dtype = torch.float16,
) -> Dict[str, Any]:
    token_sets = token_sets or {}
    past, ids_before_final, final_token_id = prefill_before_final_full_ids(model, full_ids)
    generated: List[int] = []
    full_ids_current = [int(x) for x in full_ids]
    stop_reason = "max_new_tokens"
    eos_token_id = tokenizer.eos_token_id
    activation_lists: Dict[str, List[torch.Tensor]] = {}
    position_records: List[Dict[str, Any]] = []
    logit_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []

    def append_captures(position_index: int, captures: Dict[Tuple[str, int], torch.Tensor]) -> None:
        for site in capture_sites:
            for layer_idx in capture_layer_indices:
                key = f"L{int(layer_idx)}/{site}"
                value = captures.get((site, int(layer_idx)))
                if value is not None:
                    activation_lists.setdefault(key, []).append(value)

    capture_layers = capture_layer_indices if int(capture_max_position_index) >= 0 else []
    capture_site_list = capture_sites if int(capture_max_position_index) >= 0 else []
    past, logits, captures, debug = forward_one_with_boundary_hooks(
        model,
        past=past,
        full_ids_before_token=ids_before_final,
        token_id=final_token_id,
        layers=layers,
        capture_layer_indices=capture_layers,
        capture_sites=capture_site_list,
        intervention=intervention,
        capture_dtype=capture_dtype,
    )
    append_captures(0, captures)
    position_records.append(
        {
            "position_index": 0,
            "position_label": "p0_final_box_token",
            "token_id": int(final_token_id),
            "token_text": tokenizer.decode([int(final_token_id)], skip_special_tokens=False),
            "absolute_token_index": len(full_ids_current) - 1,
        }
    )
    debug_rows.append({"position_index": 0, **debug})

    forced_ids = [int(x) for x in forced_continuation_ids] if forced_continuation_ids is not None else None
    step_limit = len(forced_ids) if forced_ids is not None else int(max_new_tokens)
    step_limit = min(step_limit, int(max_new_tokens))

    for step_idx in range(max(step_limit, 0)):
        if forced_ids is None:
            next_token_id = _sample_next_token_id(
                logits,
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
            )
        else:
            next_token_id = int(forced_ids[step_idx])

        logit_row = {
            "position_index": int(step_idx),
            "predicts_generated_index": int(step_idx),
            "chosen_token_id": int(next_token_id),
            "chosen_token_text": tokenizer.decode([int(next_token_id)], skip_special_tokens=False),
            "forced": forced_ids is not None,
        }
        logit_row.update(logit_metrics(logits, token_sets=token_sets, token_ids=tracked_token_ids))
        logit_rows.append(logit_row)

        prefix_before_next = list(full_ids_current)
        generated.append(int(next_token_id))
        full_ids_current.append(int(next_token_id))

        next_position_index = step_idx + 1
        use_capture = int(next_position_index) <= int(capture_max_position_index)
        past, logits, captures, debug = forward_one_with_boundary_hooks(
            model,
            past=past,
            full_ids_before_token=prefix_before_next,
            token_id=int(next_token_id),
            layers=layers,
            capture_layer_indices=capture_layer_indices if use_capture else [],
            capture_sites=capture_sites if use_capture else [],
            intervention=intervention,
            capture_dtype=capture_dtype,
        )
        if use_capture:
            append_captures(next_position_index, captures)
        position_records.append(
            {
                "position_index": int(next_position_index),
                "position_label": f"p{next_position_index}_generated_token",
                "token_id": int(next_token_id),
                "token_text": tokenizer.decode([int(next_token_id)], skip_special_tokens=False),
                "absolute_token_index": len(full_ids_current) - 1,
            }
        )
        debug_rows.append({"position_index": int(next_position_index), **debug})

        matched_stop = _match_stop_sequence_suffix(generated, stop_id_sequences)
        if matched_stop is not None:
            stop_reason = "matched_stop_sequence"
            break
        if eos_token_id is not None and int(next_token_id) == int(eos_token_id):
            stop_reason = "eos_token"
            break

    activation_tensors = {
        key: torch.stack(values, dim=0) for key, values in activation_lists.items() if values
    }
    return {
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
        "full_token_ids": full_ids_current,
        "full_text": tokenizer.decode(full_ids_current, skip_special_tokens=False),
        "stop_reason": stop_reason,
        "hit_max_new_tokens": bool(stop_reason == "max_new_tokens" and len(generated) >= int(max_new_tokens)),
        "position_records": position_records,
        "logit_rows": logit_rows,
        "debug_rows": debug_rows,
        "activations": activation_tensors,
    }
