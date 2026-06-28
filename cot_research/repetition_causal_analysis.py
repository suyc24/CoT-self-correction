from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .generation import _match_stop_sequence_suffix, _sample_next_token_id, GenerationBackend
from .model_utils import AttentionHeadSpec, get_input_device_for_model
from .ov_circuit_analysis import extract_head_ov_components
from .repetition_analysis import RepetitionThresholds, analyze_repetition


@dataclass
class StepRecord:
    step_idx: int
    prefix_length: int
    chosen_token_id: int
    chosen_token_text: str
    top_k: List[Dict[str, Any]]
    captured_head_before: Optional[List[float]]
    captured_head_after: Optional[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": int(self.step_idx),
            "prefix_length": int(self.prefix_length),
            "chosen_token_id": int(self.chosen_token_id),
            "chosen_token_text": self.chosen_token_text,
            "top_k": list(self.top_k),
            "captured_head_before": self.captured_head_before,
            "captured_head_after": self.captured_head_after,
        }


class ScheduledSingleHeadHook:
    def __init__(
        self,
        attn_module: torch.nn.Module,
        target: AttentionHeadSpec,
        *,
        default_scale: float = 1.0,
        scale_schedule: Optional[Dict[int, float]] = None,
    ) -> None:
        self.attn_module = attn_module
        self.target = target
        self.default_scale = float(default_scale)
        self.scale_schedule = {int(k): float(v) for k, v in (scale_schedule or {}).items()}
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.current_step_idx: int = 0
        self.captured_before: Optional[torch.Tensor] = None
        self.captured_after: Optional[torch.Tensor] = None

    def __enter__(self) -> "ScheduledSingleHeadHook":
        if not hasattr(self.attn_module, "o_proj"):
            raise ValueError("Attention module has no o_proj; cannot attach scheduled head hook.")
        self.handle = self.attn_module.o_proj.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def set_step(self, step_idx: int) -> None:
        self.current_step_idx = int(step_idx)
        self.captured_before = None
        self.captured_after = None

    def current_scale(self) -> float:
        return float(self.scale_schedule.get(self.current_step_idx, self.default_scale))

    def _pre_hook(self, module: torch.nn.Module, args: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return None
        start = int(self.target.head_idx * self.target.head_dim)
        end = int((self.target.head_idx + 1) * self.target.head_dim)
        x_out = x.clone()
        self.captured_before = x[..., start:end].detach()[0, -1].float().cpu()
        scale = self.current_scale()
        x_out[..., start:end] = x_out[..., start:end] * scale
        self.captured_after = x_out[..., start:end].detach()[0, -1].float().cpu()
        if len(args) == 1:
            return (x_out,)
        return (x_out, *args[1:])


def top_k_from_logits(
    logits_row: torch.Tensor,
    tokenizer,
    *,
    k: int,
) -> List[Dict[str, Any]]:
    k = max(int(k), 1)
    probs = torch.softmax(logits_row.float(), dim=-1)
    top_values, top_indices = torch.topk(logits_row.float(), k=min(k, int(logits_row.shape[-1])))
    out: List[Dict[str, Any]] = []
    for logit, token_id in zip(top_values.tolist(), top_indices.tolist()):
        out.append(
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logit": float(logit),
                "prob": float(probs[int(token_id)].item()),
            }
        )
    return out


def candidate_stats_from_logits(
    logits_row: torch.Tensor,
    tokenizer,
    token_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    probs = torch.softmax(logits_row.float(), dim=-1)
    rows: List[Dict[str, Any]] = []
    for token_id in token_ids:
        token_id = int(token_id)
        rows.append(
            {
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                "logit": float(logits_row[token_id].float().item()),
                "prob": float(probs[token_id].item()),
            }
        )
    return rows


@torch.no_grad()
def forward_prefix_with_capture(
    backend: GenerationBackend,
    prefix_token_ids: Sequence[int],
    *,
    attn_module: torch.nn.Module,
    target: AttentionHeadSpec,
    default_scale: float,
    scale_schedule: Optional[Dict[int, float]] = None,
    step_idx: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    model = backend.model
    if model is None:
        raise ValueError("Prefix forward requires an HF backend with a loaded model.")
    model_device = get_input_device_for_model(model)
    input_ids = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    with ScheduledSingleHeadHook(
        attn_module,
        target,
        default_scale=default_scale,
        scale_schedule=scale_schedule,
    ) as hook:
        hook.set_step(step_idx)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    debug = {
        "captured_head_before": None if hook.captured_before is None else [float(x) for x in hook.captured_before.tolist()],
        "captured_head_after": None if hook.captured_after is None else [float(x) for x in hook.captured_after.tolist()],
        "effective_scale": float(hook.current_scale()),
    }
    return outputs.logits[0, -1].detach().float().cpu(), debug


@torch.no_grad()
def generate_with_head_schedule(
    backend: GenerationBackend,
    prompt_prefix: str,
    generation_config,
    *,
    attn_module: torch.nn.Module,
    target: AttentionHeadSpec,
    default_scale: float = 1.0,
    scale_schedule: Optional[Dict[int, float]] = None,
    forced_token_ids: Optional[Dict[int, int]] = None,
    stop_strings: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("Scheduled generation requires an HF backend with a loaded model and tokenizer.")

    prompt_token_ids = [int(token_id) for token_id in backend.encode(prompt_prefix)]
    stop_id_sequences = [backend.encode(item) for item in (stop_strings or []) if item]
    stop_id_sequences = [item for item in stop_id_sequences if item]
    max_new_tokens = generation_config.max_stage1_tokens if stop_id_sequences else generation_config.max_new_tokens

    model_device = get_input_device_for_model(model)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    generated_token_ids: List[int] = []
    step_records: List[StepRecord] = []
    stop_reason = "max_new_tokens"
    matched_stop_ids: Optional[List[int]] = None
    forced_token_ids = {int(k): int(v) for k, v in (forced_token_ids or {}).items()}

    with ScheduledSingleHeadHook(
        attn_module,
        target,
        default_scale=default_scale,
        scale_schedule=scale_schedule,
    ) as hook:
        hook.set_step(0)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise ValueError("Model did not return past_key_values during scheduled generation.")

        next_token_logits = outputs.logits[0, -1].detach().float().cpu()
        eos_token_id = tokenizer.eos_token_id
        for step_idx in range(max(int(max_new_tokens), 0)):
            prefix_length = len(prompt_token_ids) + len(generated_token_ids)
            step_top_k = top_k_from_logits(next_token_logits, tokenizer, k=top_k)
            chosen_token_id = forced_token_ids.get(step_idx)
            if chosen_token_id is None:
                chosen_token_id = _sample_next_token_id(
                    next_token_logits,
                    do_sample=generation_config.do_sample,
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                )
            chosen_token_id = int(chosen_token_id)
            step_records.append(
                StepRecord(
                    step_idx=int(step_idx),
                    prefix_length=int(prefix_length),
                    chosen_token_id=chosen_token_id,
                    chosen_token_text=tokenizer.decode([chosen_token_id], skip_special_tokens=False),
                    top_k=step_top_k,
                    captured_head_before=None
                    if hook.captured_before is None
                    else [float(x) for x in hook.captured_before.tolist()],
                    captured_head_after=None
                    if hook.captured_after is None
                    else [float(x) for x in hook.captured_after.tolist()],
                )
            )
            generated_token_ids.append(chosen_token_id)

            matched_stop_ids = _match_stop_sequence_suffix(generated_token_ids, stop_id_sequences)
            if matched_stop_ids is not None:
                stop_reason = "matched_stop_sequence"
                break
            if eos_token_id is not None and int(chosen_token_id) == int(eos_token_id):
                stop_reason = "eos_token"
                break

            full_sequence_length = len(prompt_token_ids) + len(generated_token_ids)
            step_input_ids = torch.tensor([[chosen_token_id]], dtype=torch.long, device=model_device)
            step_attention_mask = torch.ones((1, full_sequence_length), dtype=torch.long, device=model_device)
            hook.set_step(step_idx + 1)
            step_outputs = model(
                input_ids=step_input_ids,
                attention_mask=step_attention_mask,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = getattr(step_outputs, "past_key_values", None)
            if past_key_values is None:
                raise ValueError("Model did not return past_key_values during incremental scheduled generation.")
            next_token_logits = step_outputs.logits[0, -1].detach().float().cpu()

    continuation = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    repetition = analyze_repetition(
        continuation,
        token_ids=generated_token_ids,
        thresholds=RepetitionThresholds(),
    )
    return {
        "prompt_prefix": prompt_prefix,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": [int(x) for x in generated_token_ids],
        "continuation": continuation,
        "full_text": prompt_prefix + continuation,
        "step_records": [item.to_dict() for item in step_records],
        "stop_reason": stop_reason,
        "matched_stop_text": backend.decode(matched_stop_ids) if matched_stop_ids else "",
        "repetition_detection": repetition,
    }


def find_first_divergence(token_ids_a: Sequence[int], token_ids_b: Sequence[int]) -> Optional[int]:
    for idx, (a, b) in enumerate(zip(token_ids_a, token_ids_b)):
        if int(a) != int(b):
            return int(idx)
    if len(token_ids_a) != len(token_ids_b):
        return int(min(len(token_ids_a), len(token_ids_b)))
    return None


def choose_loop_escape(
    *,
    baseline_repetitive: bool,
    other_repetitive: bool,
    baseline_token_id: int,
    other_token_id: int,
) -> Dict[str, Any]:
    if baseline_repetitive and (not other_repetitive):
        return {
            "loop_source": "baseline",
            "loop_token_id": int(baseline_token_id),
            "escape_source": "other",
            "escape_token_id": int(other_token_id),
        }
    if (not baseline_repetitive) and other_repetitive:
        return {
            "loop_source": "other",
            "loop_token_id": int(other_token_id),
            "escape_source": "baseline",
            "escape_token_id": int(baseline_token_id),
        }
    return {
        "loop_source": "other",
        "loop_token_id": int(other_token_id),
        "escape_source": "baseline",
        "escape_token_id": int(baseline_token_id),
    }


def compute_direct_write_for_tokens(
    *,
    ov_components: Dict[str, Any],
    head_vector_after: Optional[Sequence[float]],
    token_ids: Sequence[int],
    tokenizer,
) -> List[Dict[str, Any]]:
    if head_vector_after is None:
        return []
    o_proj_slice = ov_components["o_proj_slice"].detach().float().cpu()
    lm_head_weight = ov_components["lm_head_weight"].detach().float().cpu()
    head_vec = torch.tensor(list(head_vector_after), dtype=torch.float32)
    head_residual = torch.matmul(o_proj_slice, head_vec)
    rows: List[Dict[str, Any]] = []
    for token_id in token_ids:
        token_id = int(token_id)
        direct_logit = float(torch.dot(lm_head_weight[token_id], head_residual).item())
        rows.append(
            {
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                "direct_logit_contribution": direct_logit,
            }
        )
    return rows


def build_primary_patch_variants(prefix_token_ids: Sequence[int]) -> List[Dict[str, Any]]:
    ids = [int(x) for x in prefix_token_ids]
    variants: List[Dict[str, Any]] = [{"label": "original", "token_ids": ids}]
    if len(ids) >= 2:
        patch = list(ids)
        patch[-1] = ids[-2]
        variants.append({"label": "patch_tminus1_change_content", "token_ids": patch})
        swap = list(ids)
        swap[-1], swap[-2] = swap[-2], swap[-1]
        variants.append({"label": "swap_tminus1_tminus2_change_position", "token_ids": swap})
    if len(ids) >= 5:
        patch = list(ids)
        patch[-3] = ids[-5]
        patch[-2] = ids[-4]
        variants.append({"label": "patch_tminus2_tminus3_change_content", "token_ids": patch})
    if len(ids) >= 3:
        rotate = list(ids)
        rotate[-3], rotate[-2], rotate[-1] = ids[-2], ids[-1], ids[-3]
        variants.append({"label": "rotate_tminus1_tminus3_change_position", "token_ids": rotate})
    return variants


def summarize_loop_state(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    rep = dict(trajectory.get("repetition_detection") or {})
    return {
        "is_repetitive": bool(rep.get("matched")),
        "repetition_score": int(rep.get("score", 0)),
        "trigger_types": list(rep.get("trigger_types") or []),
    }


def preferred_primary_comparison(
    *,
    baseline: Dict[str, Any],
    ablation: Dict[str, Any],
    scale_trajectories: Sequence[Tuple[str, Dict[str, Any]]],
) -> Tuple[str, Dict[str, Any]]:
    baseline_rep = bool(dict(baseline.get("repetition_detection") or {}).get("matched"))
    ablation_rep = bool(dict(ablation.get("repetition_detection") or {}).get("matched"))
    if baseline_rep != ablation_rep:
        return "ablation", ablation

    if baseline_rep:
        for label, traj in scale_trajectories:
            if not bool(dict(traj.get("repetition_detection") or {}).get("matched")):
                return label, traj
    else:
        for label, traj in scale_trajectories:
            if bool(dict(traj.get("repetition_detection") or {}).get("matched")):
                return label, traj

    if scale_trajectories:
        return scale_trajectories[-1]
    return "ablation", ablation

