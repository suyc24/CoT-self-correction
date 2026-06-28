from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from .generation import _match_stop_sequence_suffix, _sample_next_token_id
from .hidden_trajectory import logit_metrics, replace_tuple_arg, tensor_output
from .model_utils import get_input_device_for_model


SUPPORTED_PATCH_SITES = {"block_input", "attn_out", "post_attn", "mlp_out", "block_output"}


@dataclass(frozen=True)
class ScheduledAddIntervention:
    layer_idx: int
    site: str
    add_vector: torch.Tensor
    mode: str = "prefill_plus_decode"

    def active_at(self, position_index: int) -> bool:
        if self.mode == "prefill_only":
            return int(position_index) == 0
        if self.mode == "decode_only":
            return int(position_index) >= 1
        if self.mode == "p2_plus":
            return int(position_index) >= 2
        if self.mode == "prefill_plus_decode":
            return int(position_index) >= 0
        raise ValueError(f"Unsupported add intervention mode: {self.mode}")


@dataclass(frozen=True)
class PatchPlan:
    layer_idx: int
    site: str
    vectors_by_position: Mapping[int, torch.Tensor]
    timing: str
    reflection_keywords: Tuple[str, ...] = ()

    def active_at(
        self,
        *,
        position_index: int,
        generated_text_before_current: str,
    ) -> bool:
        pos = int(position_index)
        if pos not in self.vectors_by_position:
            return False
        if self.timing == "p0":
            return pos == 0
        if self.timing == "p1":
            return pos == 1
        if self.timing == "p1_p4":
            return 1 <= pos <= 4
        if self.timing == "p1_p8":
            return 1 <= pos <= 8
        if self.timing == "p1_p16":
            return 1 <= pos <= 16
        if self.timing == "p2_p8":
            return 2 <= pos <= 8
        if self.timing == "p4_p12":
            return 4 <= pos <= 12
        if self.timing == "p8_p16":
            return 8 <= pos <= 16
        if self.timing == "until_marker":
            if pos < 1:
                return False
            lower = generated_text_before_current.lower()
            return not any(str(kw).lower() in lower for kw in self.reflection_keywords if str(kw))
        raise ValueError(f"Unsupported patch timing: {self.timing}")

    def vector_for(self, position_index: int) -> torch.Tensor:
        return self.vectors_by_position[int(position_index)]


@dataclass(frozen=True)
class ScheduledSourceMask:
    token_indices: Tuple[int, ...]
    timing: str = "p1_p16"
    reflection_keywords: Tuple[str, ...] = ()

    def active_at(
        self,
        *,
        position_index: int,
        generated_text_before_current: str,
    ) -> bool:
        pos = int(position_index)
        if self.timing == "p0":
            return pos == 0
        if self.timing == "p1":
            return pos == 1
        if self.timing == "p1_p8":
            return 1 <= pos <= 8
        if self.timing == "p1_p16":
            return 1 <= pos <= 16
        if self.timing == "p2_p16":
            return 2 <= pos <= 16
        if self.timing == "until_marker":
            if pos < 1:
                return False
            lower = generated_text_before_current.lower()
            return not any(str(kw).lower() in lower for kw in self.reflection_keywords if str(kw))
        if self.timing == "always_decode":
            return pos >= 1
        raise ValueError(f"Unsupported source-mask timing: {self.timing}")


def _to_device_vec(vec: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return vec.to(device=hidden.device, dtype=hidden.dtype)


class BoundaryPatchHooks:
    """Capture residual-boundary states and optionally apply add/patch operations.

    Operations are applied to the final token in a one-token incremental forward.
    If both an add intervention and a patch target the same boundary, the patch is
    applied after the add and therefore represents the final boundary state.
    """

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        *,
        position_index: int,
        generated_text_before_current: str,
        capture_layer_indices: Sequence[int],
        capture_sites: Sequence[str],
        add_intervention: Optional[ScheduledAddIntervention] = None,
        patch_plan: Optional[PatchPlan] = None,
        source_mask: Optional[ScheduledSourceMask] = None,
        capture_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.layers = layers
        self.position_index = int(position_index)
        self.generated_text_before_current = str(generated_text_before_current)
        self.capture_layer_indices = [int(x) for x in capture_layer_indices]
        self.capture_sites = [str(x) for x in capture_sites]
        self.add_intervention = add_intervention
        self.patch_plan = patch_plan
        self.source_mask = source_mask
        self.capture_dtype = capture_dtype
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.values: Dict[Tuple[str, int], torch.Tensor] = {}
        self._pre_attn_resid: Dict[int, torch.Tensor] = {}
        self.add_hook_call_count = 0
        self.patch_hook_call_count = 0
        self.add_norm: Optional[float] = None
        self.patch_delta_norm: Optional[float] = None

    def __enter__(self) -> "BoundaryPatchHooks":
        needed: Dict[int, set[str]] = {}
        for idx in self.capture_layer_indices:
            needed.setdefault(int(idx), set()).update(self.capture_sites)
        if self._add_is_active():
            needed.setdefault(int(self.add_intervention.layer_idx), set()).add(str(self.add_intervention.site))  # type: ignore[union-attr]
        if self._patch_is_active():
            needed.setdefault(int(self.patch_plan.layer_idx), set()).add(str(self.patch_plan.site))  # type: ignore[union-attr]

        for idx, sites in sorted(needed.items()):
            layer = self.layers[int(idx)]
            if "block_input" in sites:
                self.handles.append(layer.register_forward_pre_hook(self._make_block_input_hook(int(idx))))
            if "attn_out" in sites or "post_attn" in sites:
                self.handles.append(layer.input_layernorm.register_forward_pre_hook(self._make_ln_pre_hook(int(idx))))
                self.handles.append(layer.self_attn.register_forward_hook(self._make_attn_post_hook(int(idx))))
            if "mlp_out" in sites:
                mlp = getattr(layer, "mlp", None)
                if mlp is None:
                    raise ValueError(f"Layer {idx} has no mlp module")
                self.handles.append(mlp.register_forward_hook(self._make_mlp_out_hook(int(idx))))
            if "block_output" in sites:
                self.handles.append(layer.register_forward_hook(self._make_block_output_hook(int(idx))))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._pre_attn_resid = {}

    def _add_is_active(self) -> bool:
        return self.add_intervention is not None and self.add_intervention.active_at(self.position_index)

    def _patch_is_active(self) -> bool:
        return (
            self.patch_plan is not None
            and self.patch_plan.active_at(
                position_index=self.position_index,
                generated_text_before_current=self.generated_text_before_current,
            )
        )

    def _is_add_site(self, layer_idx: int, site: str) -> bool:
        return (
            self._add_is_active()
            and int(self.add_intervention.layer_idx) == int(layer_idx)  # type: ignore[union-attr]
            and str(self.add_intervention.site) == str(site)  # type: ignore[union-attr]
        )

    def _is_patch_site(self, layer_idx: int, site: str) -> bool:
        return (
            self._patch_is_active()
            and int(self.patch_plan.layer_idx) == int(layer_idx)  # type: ignore[union-attr]
            and str(self.patch_plan.site) == str(site)  # type: ignore[union-attr]
        )

    def _add_vec(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.add_intervention is None:
            raise ValueError("No add intervention configured.")
        return _to_device_vec(self.add_intervention.add_vector, hidden)

    def _patch_vec(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.patch_plan is None:
            raise ValueError("No patch plan configured.")
        return _to_device_vec(self.patch_plan.vector_for(self.position_index), hidden)

    def _record_add(self, add_vec: torch.Tensor) -> None:
        self.add_hook_call_count += 1
        self.add_norm = float(add_vec.detach().float().norm().item())

    def _record_patch(self, current: torch.Tensor, target: torch.Tensor) -> None:
        self.patch_hook_call_count += 1
        self.patch_delta_norm = float((current.detach().float() - target.detach().float()).norm().item())

    def _record_capture(self, layer_idx: int, site: str, hidden: torch.Tensor) -> None:
        if int(layer_idx) in self.capture_layer_indices and site in self.capture_sites:
            self.values[(site, int(layer_idx))] = hidden[0, -1].detach().to(device="cpu", dtype=self.capture_dtype)

    def _make_block_input_hook(self, layer_idx: int):
        def hook(module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            hidden = args[0]
            if hidden.ndim != 3 or hidden.shape[1] != 1:
                return None
            patched = hidden
            if self._is_add_site(layer_idx, "block_input") or self._is_patch_site(layer_idx, "block_input"):
                patched = hidden.clone()
                if self._is_add_site(layer_idx, "block_input"):
                    add_vec = self._add_vec(patched)
                    self._record_add(add_vec)
                    patched[0, -1] = patched[0, -1] + add_vec
                if self._is_patch_site(layer_idx, "block_input"):
                    patch_vec = self._patch_vec(patched)
                    self._record_patch(patched[0, -1], patch_vec)
                    patched[0, -1] = patch_vec
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
            if self._is_add_site(layer_idx, "attn_out") or self._is_patch_site(layer_idx, "attn_out"):
                modified_attn = attn_out.clone()
                if self._is_add_site(layer_idx, "attn_out"):
                    add_vec = self._add_vec(modified_attn)
                    self._record_add(add_vec)
                    modified_attn[0, -1] = modified_attn[0, -1] + add_vec
                if self._is_patch_site(layer_idx, "attn_out"):
                    patch_vec = self._patch_vec(modified_attn)
                    self._record_patch(modified_attn[0, -1], patch_vec)
                    modified_attn[0, -1] = patch_vec
                post_attn = resid + modified_attn

            if self._is_add_site(layer_idx, "post_attn") or self._is_patch_site(layer_idx, "post_attn"):
                if modified_attn is attn_out:
                    modified_attn = attn_out.clone()
                if self._is_add_site(layer_idx, "post_attn"):
                    add_vec = self._add_vec(modified_attn)
                    self._record_add(add_vec)
                    modified_attn[0, -1] = modified_attn[0, -1] + add_vec
                    post_attn = resid + modified_attn
                if self._is_patch_site(layer_idx, "post_attn"):
                    patch_vec = self._patch_vec(modified_attn)
                    self._record_patch(post_attn[0, -1], patch_vec)
                    modified_attn[0, -1] = patch_vec - resid[0, -1]
                    post_attn = resid + modified_attn
            self._record_capture(layer_idx, "attn_out", modified_attn)
            self._record_capture(layer_idx, "post_attn", post_attn)
            if modified_attn is not attn_out:
                if isinstance(output, tuple):
                    return (modified_attn,) + output[1:]
                return modified_attn
            return output

        return hook

    def _make_mlp_out_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            mlp_out = tensor_output(output)
            if mlp_out is None or mlp_out.ndim != 3 or mlp_out.shape[1] != 1:
                return output
            patched = mlp_out
            if self._is_add_site(layer_idx, "mlp_out") or self._is_patch_site(layer_idx, "mlp_out"):
                patched = mlp_out.clone()
                if self._is_add_site(layer_idx, "mlp_out"):
                    add_vec = self._add_vec(patched)
                    self._record_add(add_vec)
                    patched[0, -1] = patched[0, -1] + add_vec
                if self._is_patch_site(layer_idx, "mlp_out"):
                    patch_vec = self._patch_vec(patched)
                    self._record_patch(patched[0, -1], patch_vec)
                    patched[0, -1] = patch_vec
            self._record_capture(layer_idx, "mlp_out", patched)
            if patched is not mlp_out:
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return output

        return hook

    def _make_block_output_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = tensor_output(output)
            if hidden is None or hidden.ndim != 3 or hidden.shape[1] != 1:
                return output
            patched = hidden
            if self._is_add_site(layer_idx, "block_output") or self._is_patch_site(layer_idx, "block_output"):
                patched = hidden.clone()
                if self._is_add_site(layer_idx, "block_output"):
                    add_vec = self._add_vec(patched)
                    self._record_add(add_vec)
                    patched[0, -1] = patched[0, -1] + add_vec
                if self._is_patch_site(layer_idx, "block_output"):
                    patch_vec = self._patch_vec(patched)
                    self._record_patch(patched[0, -1], patch_vec)
                    patched[0, -1] = patch_vec
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
def forward_one_with_patch_hooks(
    model: torch.nn.Module,
    *,
    past: Any,
    full_ids_before_token: Sequence[int],
    token_id: int,
    position_index: int,
    generated_text_before_current: str,
    layers: Sequence[torch.nn.Module],
    capture_layer_indices: Sequence[int],
    capture_sites: Sequence[str],
    add_intervention: Optional[ScheduledAddIntervention] = None,
    patch_plan: Optional[PatchPlan] = None,
    source_mask: Optional[ScheduledSourceMask] = None,
    capture_dtype: torch.dtype = torch.float16,
) -> Tuple[Any, torch.Tensor, Dict[Tuple[str, int], torch.Tensor], Dict[str, Any]]:
    device = get_input_device_for_model(model)
    full_len = len(full_ids_before_token) + 1
    input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, full_len), dtype=torch.long, device=device)
    source_mask_active = False
    source_mask_count = 0
    if source_mask is not None and source_mask.active_at(
        position_index=int(position_index),
        generated_text_before_current=generated_text_before_current,
    ):
        valid = [idx for idx in source_mask.token_indices if 0 <= int(idx) < full_len - 1]
        if valid:
            attention_mask[0, torch.tensor(valid, dtype=torch.long, device=device)] = 0
            source_mask_active = True
            source_mask_count = len(valid)
    with BoundaryPatchHooks(
        layers,
        position_index=int(position_index),
        generated_text_before_current=generated_text_before_current,
        capture_layer_indices=capture_layer_indices,
        capture_sites=capture_sites,
        add_intervention=add_intervention,
        patch_plan=patch_plan,
        source_mask=source_mask,
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
            "patch_hook_call_count": int(hooks.patch_hook_call_count),
            "add_norm": hooks.add_norm,
            "patch_delta_norm": hooks.patch_delta_norm,
            "source_mask_active": bool(source_mask_active),
            "source_mask_count": int(source_mask_count),
        }

    new_past = getattr(outputs, "past_key_values", None)
    if new_past is None:
        raise ValueError("Model did not return past_key_values during trajectory forward.")
    return new_past, outputs.logits[0, -1], captures, debug


@torch.no_grad()
def run_patchable_trajectory(
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
    forced_prefix_ids: Optional[Sequence[int]] = None,
    add_intervention: Optional[ScheduledAddIntervention] = None,
    patch_plan: Optional[PatchPlan] = None,
    source_mask: Optional[ScheduledSourceMask] = None,
    banned_token_ids: Sequence[int] = (),
    ban_until_position: Optional[int] = None,
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
    past, logits, captures, debug = forward_one_with_patch_hooks(
        model,
        past=past,
        full_ids_before_token=ids_before_final,
        token_id=final_token_id,
        position_index=0,
        generated_text_before_current="",
        layers=layers,
        capture_layer_indices=capture_layers,
        capture_sites=capture_site_list,
        add_intervention=add_intervention,
        patch_plan=patch_plan,
        source_mask=source_mask,
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
    forced_prefix = [int(x) for x in forced_prefix_ids] if forced_prefix_ids is not None else []
    banned_ids = [int(x) for x in banned_token_ids]
    step_limit = len(forced_ids) if forced_ids is not None else int(max_new_tokens)
    step_limit = min(step_limit, int(max_new_tokens))

    for step_idx in range(max(step_limit, 0)):
        sampling_logits = logits
        should_ban = banned_ids and (ban_until_position is None or int(step_idx) <= int(ban_until_position))
        if should_ban:
            sampling_logits = logits.clone()
            valid_banned = [x for x in banned_ids if 0 <= int(x) < int(sampling_logits.shape[-1])]
            if valid_banned:
                sampling_logits[..., torch.tensor(valid_banned, dtype=torch.long, device=sampling_logits.device)] = -float("inf")

        if int(step_idx) < len(forced_prefix):
            next_token_id = int(forced_prefix[int(step_idx)])
        elif forced_ids is None:
            next_token_id = _sample_next_token_id(
                sampling_logits,
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
        generated_text_before_current = tokenizer.decode(generated, skip_special_tokens=False)
        generated.append(int(next_token_id))
        full_ids_current.append(int(next_token_id))

        next_position_index = step_idx + 1
        use_capture = int(next_position_index) <= int(capture_max_position_index)
        past, logits, captures, debug = forward_one_with_patch_hooks(
            model,
            past=past,
            full_ids_before_token=prefix_before_next,
            token_id=int(next_token_id),
            position_index=next_position_index,
            generated_text_before_current=generated_text_before_current,
            layers=layers,
            capture_layer_indices=capture_layer_indices if use_capture else [],
            capture_sites=capture_sites if use_capture else [],
            add_intervention=add_intervention,
            patch_plan=patch_plan,
            source_mask=source_mask,
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
