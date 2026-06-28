from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .answer_extraction import answers_match, classify_outcome, extract_last_boxed
from .cot_editing import find_last_boxed_span
from .generation import _match_stop_sequence_suffix, _sample_next_token_id
from .model_utils import AttentionHeadSpec, get_input_device_for_model


@dataclass
class BoxTokenSpan:
    box_start: int
    answer_start: int
    answer_end: int
    box_end: int
    answer_text: str
    box_text: str
    roundtrip_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForcedState:
    full_token_ids: List[int]
    forced_token_ids: List[int]
    logits: torch.Tensor
    past_key_values: Any
    attentions: Optional[Sequence[Any]]


@dataclass
class ContinuationResult:
    token_ids: List[int]
    text: str
    stop_reason: str
    step_records: List[Dict[str, Any]]


def token_ids_for_first_tokens(tokenizer, texts: Iterable[str]) -> List[int]:
    out: List[int] = []
    seen = set()
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        token_id = int(ids[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        out.append(token_id)
    return out


def logsumexp_token_set(logits: torch.Tensor, token_ids: Sequence[int]) -> float:
    if not token_ids:
        return float("nan")
    ids = torch.tensor([int(x) for x in token_ids], dtype=torch.long, device=logits.device)
    return float(torch.logsumexp(logits.index_select(0, ids).float(), dim=0).item())


def candidate_stats(logits: torch.Tensor, tokenizer, token_ids: Sequence[int]) -> List[Dict[str, Any]]:
    probs = torch.softmax(logits.float(), dim=-1)
    rows: List[Dict[str, Any]] = []
    for token_id in token_ids:
        token_id = int(token_id)
        rows.append(
            {
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
                "logit": float(logits[token_id].float().item()),
                "prob": float(probs[token_id].item()),
            }
        )
    return rows


def top_k_from_logits(logits: torch.Tensor, tokenizer, k: int = 10) -> List[Dict[str, Any]]:
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(logits.float(), k=min(int(k), int(logits.shape[-1])))
    out: List[Dict[str, Any]] = []
    for value, token_id in zip(values.tolist(), indices.tolist()):
        out.append(
            {
                "token_id": int(token_id),
                "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logit": float(value),
                "prob": float(probs[int(token_id)].item()),
            }
        )
    return out


def _char_to_token_index(tokenizer, text: str, char_index: int) -> int:
    return len(tokenizer.encode(text[:char_index], add_special_tokens=False))


def find_last_boxed_token_span(tokenizer, token_ids: Sequence[int]) -> Optional[BoxTokenSpan]:
    text = tokenizer.decode(list(token_ids), skip_special_tokens=False)
    boxed = find_last_boxed_span(text)
    if boxed is None:
        return None
    box_start_char, box_end_char, answer_text = boxed
    brace_pos = text.find("{", box_start_char + len("\\boxed"))
    if brace_pos < 0:
        return None
    answer_start_char = brace_pos + 1
    answer_end_char = box_end_char - 1

    box_start = _char_to_token_index(tokenizer, text, box_start_char)
    answer_start = _char_to_token_index(tokenizer, text, answer_start_char)
    answer_end = _char_to_token_index(tokenizer, text, answer_end_char)
    box_end = _char_to_token_index(tokenizer, text, box_end_char)
    roundtrip_ok = list(tokenizer.encode(text, add_special_tokens=False)) == [int(x) for x in token_ids]
    return BoxTokenSpan(
        box_start=int(box_start),
        answer_start=int(answer_start),
        answer_end=int(answer_end),
        box_end=int(box_end),
        answer_text=str(answer_text),
        box_text=text[box_start_char:box_end_char],
        roundtrip_ok=bool(roundtrip_ok),
    )


def make_region_map(
    *,
    prompt_len: int,
    span: BoxTokenSpan,
    forced_answer_len: int,
    forced_prefix_len: int,
    current_prefix_len: int,
) -> Dict[str, Tuple[int, int]]:
    answer_start_abs = prompt_len + span.answer_start
    forced_answer_end_abs = answer_start_abs + int(forced_answer_len)
    box_start_abs = prompt_len + span.box_start
    box_end_abs = prompt_len + span.box_end
    before_box_abs = box_start_abs
    regions: Dict[str, Tuple[int, int]] = {
        "prompt": (0, max(prompt_len, 0)),
        "reasoning_before_box": (prompt_len, max(before_box_abs, prompt_len)),
        "box_marker_to_answer": (box_start_abs, max(answer_start_abs, box_start_abs)),
        "forced_answer": (answer_start_abs, max(forced_answer_end_abs, answer_start_abs)),
        "box_suffix": (forced_answer_end_abs, max(forced_prefix_len, forced_answer_end_abs)),
        "forced_box_full": (box_start_abs, max(forced_prefix_len, box_start_abs)),
        "prev_1": (max(current_prefix_len - 1, 0), current_prefix_len),
        "prev_4": (max(current_prefix_len - 4, 0), current_prefix_len),
        "prev_16": (max(current_prefix_len - 16, 0), current_prefix_len),
        "prev_64": (max(current_prefix_len - 64, 0), current_prefix_len),
    }
    if current_prefix_len > forced_prefix_len:
        regions["generated_after_force"] = (forced_prefix_len, current_prefix_len)
    else:
        regions["generated_after_force"] = (current_prefix_len, current_prefix_len)
    return regions


def summarize_attention_regions(
    attentions: Optional[Sequence[Any]],
    regions: Dict[str, Tuple[int, int]],
) -> List[Dict[str, Any]]:
    if attentions is None:
        return []
    rows: List[Dict[str, Any]] = []
    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn is None:
            continue
        attn = layer_attn.detach().float().cpu()
        # Expected shapes:
        # full forward: [batch, heads, query_len, key_len]
        # incremental: [batch, heads, 1, key_len]
        if attn.ndim != 4:
            continue
        vecs = attn[0, :, -1, :]
        key_len = int(vecs.shape[-1])
        probs = vecs.clamp_min(1e-20)
        entropy = -(probs * probs.log()).sum(dim=-1)
        max_values, max_indices = torch.max(vecs, dim=-1)
        for head_idx in range(int(vecs.shape[0])):
            row: Dict[str, Any] = {
                "layer_idx": int(layer_idx),
                "head_idx": int(head_idx),
                "attention_entropy": float(entropy[head_idx].item()),
                "attention_max_value": float(max_values[head_idx].item()),
                "attention_max_index": int(max_indices[head_idx].item()),
                "key_len": int(key_len),
            }
            head_vec = vecs[head_idx]
            for name, (start, end) in regions.items():
                s = max(0, min(int(start), key_len))
                e = max(0, min(int(end), key_len))
                row[f"mass_{name}"] = float(head_vec[s:e].sum().item()) if e > s else 0.0
            rows.append(row)
    return rows


@torch.no_grad()
def prefill_cache(model: torch.nn.Module, input_ids: Sequence[int]) -> Tuple[Any, torch.Tensor]:
    device = get_input_device_for_model(model)
    tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tensor)
    outputs = model(
        input_ids=tensor,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past = getattr(outputs, "past_key_values", None)
    if past is None:
        raise ValueError("Model did not return past_key_values during prefill.")
    return past, outputs.logits[0, -1]


@torch.no_grad()
def force_tokens_from_cache(
    model: torch.nn.Module,
    prefix_token_ids: Sequence[int],
    force_token_ids: Sequence[int],
    *,
    output_attentions_last: bool = False,
) -> ForcedState:
    if not prefix_token_ids:
        raise ValueError("prefix_token_ids must be non-empty.")
    device = get_input_device_for_model(model)
    past, logits = prefill_cache(model, prefix_token_ids)
    full_ids = [int(x) for x in prefix_token_ids]
    attentions = None
    for idx, token_id in enumerate(force_token_ids):
        full_ids.append(int(token_id))
        input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=bool(output_attentions_last and idx == len(force_token_ids) - 1),
            return_dict=True,
        )
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            raise ValueError("Model did not return past_key_values while forcing tokens.")
        logits = outputs.logits[0, -1]
        if output_attentions_last and idx == len(force_token_ids) - 1:
            attentions = getattr(outputs, "attentions", None)
    return ForcedState(
        full_token_ids=full_ids,
        forced_token_ids=[int(x) for x in force_token_ids],
        logits=logits,
        past_key_values=past,
        attentions=attentions,
    )


class SingleHeadLastTokenAblation:
    """Zero one head slice only for q_len=1 calls, typically the final forced token."""

    def __init__(self, attn_module: torch.nn.Module, target: AttentionHeadSpec) -> None:
        self.attn_module = attn_module
        self.target = target
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.call_count = 0
        self.abs_mean_before: Optional[float] = None

    def __enter__(self) -> "SingleHeadLastTokenAblation":
        if not hasattr(self.attn_module, "o_proj"):
            raise ValueError("Attention module has no o_proj; cannot attach ablation hook.")
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
        # We call this hook only around the final forced-token forward. q_len
        # should be 1; if a backend forwards a larger chunk, still zero only the
        # last query position to preserve earlier cached state.
        start = int(self.target.head_idx * self.target.head_dim)
        end = int((self.target.head_idx + 1) * self.target.head_dim)
        x_out = x.clone()
        self.call_count += 1
        self.abs_mean_before = float(x_out[0, -1, start:end].detach().abs().mean().item())
        x_out[0, -1, start:end] = 0
        if len(args) == 1:
            return (x_out,)
        return (x_out, *args[1:])


@torch.no_grad()
def force_tokens_with_final_head_ablation(
    model: torch.nn.Module,
    attn_module: torch.nn.Module,
    target: AttentionHeadSpec,
    prefix_token_ids: Sequence[int],
    force_token_ids: Sequence[int],
) -> Tuple[ForcedState, Dict[str, Any]]:
    if len(force_token_ids) == 0:
        raise ValueError("force_token_ids must be non-empty for final-token ablation.")
    device = get_input_device_for_model(model)
    past, logits = prefill_cache(model, prefix_token_ids)
    full_ids = [int(x) for x in prefix_token_ids]
    for token_id in force_token_ids[:-1]:
        full_ids.append(int(token_id))
        input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            raise ValueError("Model did not return past_key_values while forcing pre-ablation tokens.")
        logits = outputs.logits[0, -1]

    final_token_id = int(force_token_ids[-1])
    full_ids.append(final_token_id)
    input_ids = torch.tensor([[final_token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
    with SingleHeadLastTokenAblation(attn_module, target) as hook:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    past = getattr(outputs, "past_key_values", None)
    if past is None:
        raise ValueError("Model did not return past_key_values after final-token ablation.")
    forced = ForcedState(
        full_token_ids=full_ids,
        forced_token_ids=[int(x) for x in force_token_ids],
        logits=outputs.logits[0, -1],
        past_key_values=past,
        attentions=None,
    )
    debug = {
        "hook_call_count": int(hook.call_count),
        "hook_abs_mean_before": hook.abs_mean_before,
    }
    return forced, debug


@torch.no_grad()
def continue_from_state(
    model: torch.nn.Module,
    tokenizer,
    state: ForcedState,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    stop_id_sequences: Sequence[Sequence[int]] = (),
    capture_attention: bool = False,
    reflect_first_token_ids: Sequence[int] = (),
    region_builder=None,
) -> ContinuationResult:
    device = get_input_device_for_model(model)
    past = state.past_key_values
    logits = state.logits
    full_ids = list(state.full_token_ids)
    generated: List[int] = []
    step_records: List[Dict[str, Any]] = []
    current_attentions = state.attentions
    stop_reason = "max_new_tokens"
    eos_token_id = tokenizer.eos_token_id
    reflect_set = {int(x) for x in reflect_first_token_ids}
    matched_stop: Optional[List[int]] = None

    for step_idx in range(max(int(max_new_tokens), 0)):
        next_token_id = _sample_next_token_id(
            logits,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        record: Dict[str, Any] = {
            "step": int(step_idx),
            "prefix_len": int(len(full_ids)),
            "predicted_token_id": int(next_token_id),
            "predicted_token_text": tokenizer.decode([int(next_token_id)], skip_special_tokens=False),
            "is_reflect_first_token": bool(int(next_token_id) in reflect_set),
        }
        if capture_attention and current_attentions is not None and region_builder is not None:
            regions = region_builder(current_prefix_len=len(full_ids))
            rows = summarize_attention_regions(current_attentions, regions)
            record["attention_region_rows"] = rows if int(next_token_id) in reflect_set else []
        step_records.append(record)

        generated.append(int(next_token_id))
        full_ids.append(int(next_token_id))
        matched_stop = _match_stop_sequence_suffix(generated, stop_id_sequences)
        if matched_stop is not None:
            stop_reason = "matched_stop_sequence"
            break
        if eos_token_id is not None and int(next_token_id) == int(eos_token_id):
            stop_reason = "eos_token"
            break

        input_ids = torch.tensor([[int(next_token_id)]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=bool(capture_attention),
            return_dict=True,
        )
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            raise ValueError("Model did not return past_key_values during continuation.")
        logits = outputs.logits[0, -1]
        current_attentions = getattr(outputs, "attentions", None) if capture_attention else None

    return ContinuationResult(
        token_ids=generated,
        text=tokenizer.decode(generated, skip_special_tokens=True),
        stop_reason=stop_reason,
        step_records=step_records,
    )


def analyze_continuation_text(
    *,
    continuation: str,
    full_text: str,
    correct_answer: Any,
    wrong_answer: Any,
    reflection_keywords: Sequence[str],
) -> Dict[str, Any]:
    lower = continuation.lower()
    reflection_hits = [kw for kw in reflection_keywords if str(kw).lower() in lower]
    first_reflection_pos: Optional[int] = None
    first_reflection_keyword: Optional[str] = None
    for kw in reflection_keywords:
        pos = lower.find(str(kw).lower())
        if pos >= 0 and (first_reflection_pos is None or pos < first_reflection_pos):
            first_reflection_pos = int(pos)
            first_reflection_keyword = str(kw)
    final_boxed_cont = extract_last_boxed(continuation)
    final_boxed_full = extract_last_boxed(full_text)
    return {
        "has_reflection": bool(reflection_hits),
        "matched_reflection_keywords": reflection_hits,
        "first_reflection_pos": first_reflection_pos,
        "first_reflection_keyword": first_reflection_keyword,
        "final_boxed_answer": final_boxed_cont,
        "final_boxed_answer_full_text": final_boxed_full,
        "outcome": classify_outcome(final_boxed_cont, correct_answer, wrong_answer),
        "outcome_full_text": classify_outcome(final_boxed_full, correct_answer, wrong_answer),
        "full_text_final_matches_correct": bool(answers_match(final_boxed_full, correct_answer)),
        "full_text_final_matches_wrong": bool(answers_match(final_boxed_full, wrong_answer)),
    }
