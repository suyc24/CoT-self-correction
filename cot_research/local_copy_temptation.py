from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .answer_extraction import extract_last_boxed
from .cot_accuracy import judge_single_answer, resolve_gold_answer
from .io_utils import load_jsonl, write_json
from .prompt_utils import build_chat_prompt

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None


DEFAULT_K_VALUES = [1, 2, 4, 8]
DEFAULT_TOKEN_SYSTEM_PROMPT = (
    "Continue the current partial draft naturally. Keep the continuation short and do not restart from the beginning."
)
DEFAULT_TOKEN_USER_PROMPT = "Continue the assistant's partial fragment with only a short continuation."
DEFAULT_COT_SYSTEM_PROMPT = (
    "Continue the current reasoning from the partial draft. If the most recent answer is wrong, correct it and end "
    "with a final \\boxed{} answer."
)
DEFAULT_SHARP_USER_PROMPT = (
    "Continue the assistant fragment from exactly the current point with an extremely short continuation."
)
DEFAULT_SHARP_COT_USER_PROMPT = (
    "Continue the partial math reasoning from exactly the current point. Do not restart the solution."
)


@dataclass
class LocalCopyCase:
    example_id: str
    family: str
    system_prompt: str
    user_question: str
    assistant_prefix: str
    max_new_tokens: int
    min_phrase_match_tokens: int = 2
    copy_append_text: str = ""
    correct_answer: Optional[str] = None
    wrong_answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "family": self.family,
            "system_prompt": self.system_prompt,
            "user_question": self.user_question,
            "assistant_prefix": self.assistant_prefix,
            "max_new_tokens": int(self.max_new_tokens),
            "min_phrase_match_tokens": int(self.min_phrase_match_tokens),
            "copy_append_text": self.copy_append_text,
            "correct_answer": self.correct_answer,
            "wrong_answer": self.wrong_answer,
            "metadata": dict(self.metadata or {}),
        }


def parse_int_list(text: str, *, default: Sequence[int]) -> List[int]:
    if not text.strip():
        return [int(item) for item in default]
    values: List[int] = []
    seen = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError(f"All parsed integers must be positive, got {value}.")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("No integers parsed from the provided list.")
    return values


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _dedup_preserve_order(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        token_id = int(value)
        if token_id in seen:
            continue
        seen.add(token_id)
        out.append(token_id)
    return out


def _decode_token(tokenizer, token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def _format_repeated_tail(text: str, repeats: int) -> str:
    parts = [text.strip() for _ in range(max(repeats, 1))]
    return " ".join(part for part in parts if part)


def _extract_first_digit(text: str) -> Optional[str]:
    for char in str(text):
        if char.isdigit():
            return char
    return None


def build_token_family_cases(
    *,
    count: int,
    max_new_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    stimuli = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "+", "-", "*"]
    assistant_templates = [
        "We are close to the end of a short draft.\nAnswer: {stimulus} {stimulus}",
        "Please continue the current scratch line.\nresult = {stimulus} {stimulus}",
        "This draft should continue from the current token tail.\nCurrent tail: {stimulus} {stimulus}",
        "Keep writing from the exact current point.\nShort ending: {stimulus} {stimulus}",
        "Continue the fragment without restarting it.\nMarker: {stimulus} {stimulus}",
        "Only continue the current assistant text.\nSequence tail: {stimulus} {stimulus}",
    ]
    user_prompts = [
        DEFAULT_TOKEN_USER_PROMPT,
        "Keep the continuation brief and continue from the current partial answer.",
        "Continue the partial assistant text from where it stopped.",
    ]
    combos = list(itertools.product(stimuli, assistant_templates, user_prompts))
    rng = random.Random(seed)
    rng.shuffle(combos)
    cases: List[Dict[str, Any]] = []
    for idx, (stimulus, assistant_template, user_prompt) in enumerate(combos[:count]):
        cases.append(
            LocalCopyCase(
                example_id=f"token_{idx:03d}_{stimulus.replace(' ', '_')}",
                family="token",
                system_prompt=DEFAULT_TOKEN_SYSTEM_PROMPT,
                user_question=user_prompt,
                assistant_prefix=assistant_template.format(stimulus=stimulus),
                max_new_tokens=max_new_tokens,
                min_phrase_match_tokens=1,
                copy_append_text=f" {stimulus}",
                metadata={
                    "stimulus": stimulus,
                    "template": assistant_template,
                    "track_phrase_continuation": False,
                },
            ).to_dict()
        )
    return cases


def build_phrase_family_cases(
    *,
    count: int,
    max_new_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    phrases = [
        "the next step is",
        "and then we",
        "this means that",
        "the result is",
        "so we get",
        "we can see",
        "for this reason",
        "on the other hand",
        "one more time",
        "step by step",
        "wait, wait,",
        "again and again",
    ]
    wrappers = [
        "We are in the middle of a draft.\n{tail}",
        "Continue directly from this repeated phrase tail.\n{tail}",
        "This note should continue from the exact current phrase.\n{tail}",
        "Do not restart the sentence; continue from here.\n{tail}",
        "Current line in the draft:\n{tail}",
        "Assistant draft fragment:\n{tail}",
    ]
    user_prompts = [
        DEFAULT_TOKEN_USER_PROMPT,
        "Continue the current assistant sentence naturally from the point where it stopped.",
    ]
    combos = list(itertools.product(phrases, wrappers, user_prompts))
    rng = random.Random(seed)
    rng.shuffle(combos)
    cases: List[Dict[str, Any]] = []
    for idx, (phrase, wrapper, user_prompt) in enumerate(combos[:count]):
        repeated_tail = _format_repeated_tail(phrase, 2)
        cases.append(
            LocalCopyCase(
                example_id=f"phrase_{idx:03d}",
                family="phrase",
                system_prompt=DEFAULT_TOKEN_SYSTEM_PROMPT,
                user_question=user_prompt,
                assistant_prefix=wrapper.format(tail=repeated_tail),
                max_new_tokens=max_new_tokens,
                min_phrase_match_tokens=2,
                copy_append_text=f" {phrase}",
                metadata={
                    "phrase": phrase,
                    "wrapper": wrapper,
                    "track_phrase_continuation": True,
                },
            ).to_dict()
        )
    return cases


def build_sharp_token_family_cases(
    *,
    count: int,
    max_new_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    stimuli = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "=", "+", "-", "*"]
    assistant_templates = [
        "{stimulus} {stimulus}",
        "Current tail:\n{stimulus} {stimulus}",
        "Assistant fragment:\n{stimulus} {stimulus}",
        "Keep going from here:\n{stimulus} {stimulus}",
    ]
    user_prompts = [
        DEFAULT_SHARP_USER_PROMPT,
        "Continue the assistant fragment with exactly one short next token.",
        "Keep writing from the current token with only a minimal continuation.",
    ]
    combos = list(itertools.product(stimuli, assistant_templates, user_prompts))
    rng = random.Random(seed)
    rng.shuffle(combos)
    cases: List[Dict[str, Any]] = []
    for idx, (stimulus, assistant_template, user_prompt) in enumerate(combos[:count]):
        cases.append(
            LocalCopyCase(
                example_id=f"token_sharp_{idx:03d}_{stimulus.replace(' ', '_')}",
                family="token",
                system_prompt=DEFAULT_TOKEN_SYSTEM_PROMPT,
                user_question=user_prompt,
                assistant_prefix=assistant_template.format(stimulus=stimulus),
                max_new_tokens=max_new_tokens,
                min_phrase_match_tokens=1,
                copy_append_text=f" {stimulus}",
                metadata={
                    "stimulus": stimulus,
                    "template": assistant_template,
                    "track_phrase_continuation": False,
                    "force_candidate_from_copy_append": True,
                    "probe_variant": "sharp_prev1",
                    "probe_kind": "exact_prev_token",
                },
            ).to_dict()
        )
    return cases


def build_sharp_phrase_family_cases(
    *,
    count: int,
    max_new_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    phrase_bases = [
        ("the next step", " is"),
        ("the answer should", " be"),
        ("so the result", " is"),
        ("this means", " that"),
        ("on the other", " hand"),
        ("one more", " time"),
        ("step by", " step"),
        ("we can now", " write"),
        ("we should now", " check"),
        ("the final line", " is"),
        ("the current answer", " is"),
        ("we can therefore", " write"),
        ("for this", " reason"),
        ("at this", " point"),
        ("the key idea", " is"),
        ("the next move", " is"),
    ]
    wrappers = [
        "{tail}",
        "Current phrase tail:\n{tail}",
        "Assistant fragment:\n{tail}",
        "Continue directly from here:\n{tail}",
    ]
    user_prompts = [
        DEFAULT_SHARP_USER_PROMPT,
        "Continue the assistant fragment with the next short phrase token only.",
    ]
    combos = list(itertools.product(phrase_bases, wrappers, user_prompts))
    rng = random.Random(seed)
    rng.shuffle(combos)
    cases: List[Dict[str, Any]] = []
    for idx, ((base_prefix, append_text), wrapper, user_prompt) in enumerate(combos[:count]):
        tail = f"{base_prefix}{append_text} {base_prefix}"
        matched_control_tail = base_prefix
        position_shift_tail = f"{base_prefix}{append_text}. Meanwhile, we note a separate detail. {base_prefix}"
        cases.append(
            LocalCopyCase(
                example_id=f"phrase_sharp_{idx:03d}",
                family="phrase",
                system_prompt=DEFAULT_TOKEN_SYSTEM_PROMPT,
                user_question=user_prompt,
                assistant_prefix=wrapper.format(tail=tail),
                max_new_tokens=max_new_tokens,
                min_phrase_match_tokens=1,
                copy_append_text=append_text,
                metadata={
                    "tail": tail,
                    "base_prefix": base_prefix,
                    "append_text": append_text,
                    "matched_control_tail": matched_control_tail,
                    "position_shift_tail": position_shift_tail,
                    "wrapper": wrapper,
                    "track_phrase_continuation": True,
                    "force_candidate_from_copy_append": True,
                    "probe_variant": "sharp_prev1",
                    "probe_kind": "phrase_resume",
                },
            ).to_dict()
        )
    return cases


def build_sharp_cot_family_cases(
    *,
    input_jsonl: str | Path,
    count: int,
    max_new_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = load_jsonl(input_jsonl)
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected_rows = rows[: max(count * 3, count)]
    assistant_templates = [
        (
            "<think>\n"
            "Continue the current reasoning from exactly the current tail.\n"
            "A mistaken draft has started repeating the same wrong answer digit: {digit} {digit}"
        ),
        (
            "<think>\n"
            "We are near the end of a partial solution.\n"
            "Recent wrong digit tail: {digit} {digit}"
        ),
        (
            "<think>\n"
            "Do not restart the solution; continue from the current fragment.\n"
            "Bad answer tail: {digit} {digit}"
        ),
        (
            "<think>\n"
            "The draft is stuck near a wrong final answer.\n"
            "Current wrong digit loop: {digit} {digit}"
        ),
    ]
    cases: List[Dict[str, Any]] = []
    for row in selected_rows:
        wrong_answer = str(row.get("wrong_answer") or "").strip()
        question = str(row.get("question") or row.get("problem") or "").strip()
        correct_answer = str(row.get("correct_answer") or "").strip()
        wrong_digit = _extract_first_digit(wrong_answer)
        if not wrong_digit or not question:
            continue
        idx = len(cases)
        template = assistant_templates[idx % len(assistant_templates)]
        cases.append(
            LocalCopyCase(
                example_id=str(row.get("id") or f"cot_sharp_{idx:03d}"),
                family="cot",
                system_prompt=DEFAULT_COT_SYSTEM_PROMPT,
                user_question=f"{DEFAULT_SHARP_COT_USER_PROMPT}\n\nProblem: {question}",
                assistant_prefix=template.format(digit=wrong_digit),
                max_new_tokens=max_new_tokens,
                min_phrase_match_tokens=1,
                copy_append_text=f" {wrong_digit}",
                correct_answer=correct_answer or None,
                wrong_answer=wrong_answer or None,
                metadata={
                    "question": question,
                    "wrong_digit": wrong_digit,
                    "track_phrase_continuation": False,
                    "force_candidate_from_copy_append": True,
                    "probe_variant": "sharp_prev1",
                    "probe_kind": "cot_wrong_digit_prev1",
                },
            ).to_dict()
        )
        if len(cases) >= count:
            break
    return cases


def build_cot_family_cases(
    *,
    input_jsonl: str | Path,
    count: int,
    max_new_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows = load_jsonl(input_jsonl)
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected_rows = rows[:count]
    assistant_templates = [
        (
            "<think>\n"
            "Let's continue the reasoning carefully.\n"
            "A careless draft near the end says: The final answer is \\boxed{{{wrong}}}. "
            "The final answer is \\boxed{{{wrong}}}."
        ),
        (
            "<think>\n"
            "We should verify the ending carefully.\n"
            "A mistaken draft says: Maybe the answer is \\boxed{{{wrong}}}. "
            "Maybe the answer is \\boxed{{{wrong}}}."
        ),
        (
            "<think>\n"
            "The current draft is near the conclusion.\n"
            "Recent wrong line: So the final answer is \\boxed{{{wrong}}}. "
            "So the final answer is \\boxed{{{wrong}}}."
        ),
        (
            "<think>\n"
            "We need to continue and possibly correct the draft.\n"
            "A hasty ending says: Therefore the answer is \\boxed{{{wrong}}}. "
            "Therefore the answer is \\boxed{{{wrong}}}."
        ),
    ]
    append_templates = [
        " The final answer is \\boxed{{{wrong}}}.",
        " Maybe the answer is \\boxed{{{wrong}}}.",
        " So the final answer is \\boxed{{{wrong}}}.",
        " Therefore the answer is \\boxed{{{wrong}}}.",
    ]
    cases: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected_rows):
        wrong_answer = str(row.get("wrong_answer") or "").strip()
        correct_answer = str(row.get("correct_answer") or "").strip()
        question = str(row.get("question") or row.get("problem") or "").strip()
        if not wrong_answer or not correct_answer or not question:
            continue
        template_idx = idx % len(assistant_templates)
        assistant_prefix = assistant_templates[template_idx].format(wrong=wrong_answer)
        copy_append_text = append_templates[template_idx].format(wrong=wrong_answer)
        user_question = (
            "Continue the current math reasoning from the partial draft below. If the recent draft answer is wrong, "
            "fix it and finish with a final boxed answer.\n\n"
            f"Problem: {question}"
        )
        cases.append(
            LocalCopyCase(
                example_id=str(row.get("id") or f"cot_{idx:03d}"),
                family="cot",
                system_prompt=DEFAULT_COT_SYSTEM_PROMPT,
                user_question=user_question,
                assistant_prefix=assistant_prefix,
                max_new_tokens=max_new_tokens,
                min_phrase_match_tokens=2,
                copy_append_text=copy_append_text,
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                metadata={
                    "question": question,
                    "template_index": template_idx,
                    "track_phrase_continuation": False,
                },
            ).to_dict()
        )
    return cases


def build_local_copy_cases(
    *,
    token_count: int,
    phrase_count: int,
    cot_count: int,
    cot_input_jsonl: str | Path,
    token_max_new_tokens: int,
    phrase_max_new_tokens: int,
    cot_max_new_tokens: int,
    seed: int,
    variant: str = "default",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if variant == "default":
        out.extend(build_token_family_cases(count=token_count, max_new_tokens=token_max_new_tokens, seed=seed))
        out.extend(build_phrase_family_cases(count=phrase_count, max_new_tokens=phrase_max_new_tokens, seed=seed + 17))
        out.extend(
            build_cot_family_cases(
                input_jsonl=cot_input_jsonl,
                count=cot_count,
                max_new_tokens=cot_max_new_tokens,
                seed=seed + 31,
            )
        )
        return out
    if variant == "sharp_prev1":
        out.extend(build_sharp_token_family_cases(count=token_count, max_new_tokens=token_max_new_tokens, seed=seed))
        out.extend(
            build_sharp_phrase_family_cases(count=phrase_count, max_new_tokens=phrase_max_new_tokens, seed=seed + 17)
        )
        out.extend(
            build_sharp_cot_family_cases(
                input_jsonl=cot_input_jsonl,
                count=cot_count,
                max_new_tokens=cot_max_new_tokens,
                seed=seed + 31,
            )
        )
        return out
    raise ValueError(f"Unsupported local-copy case variant: {variant}")
    return out


def build_prompt_prefix(case: Dict[str, Any], tokenizer) -> str:
    return build_chat_prompt(
        tokenizer,
        question=str(case.get("user_question") or ""),
        system_prompt=str(case.get("system_prompt") or ""),
        assistant_prefix=str(case.get("assistant_prefix") or ""),
        enable_thinking=False,
    )


def expected_phrase_token_ids(case: Dict[str, Any], tokenizer) -> List[int]:
    append_text = str(case.get("copy_append_text") or "")
    if not append_text:
        return []
    return tokenizer.encode(append_text, add_special_tokens=False)


def resolve_copy_append_target_token_ids(expected_token_ids: Sequence[int], tokenizer) -> Dict[str, Any]:
    token_ids = [int(token_id) for token_id in expected_token_ids]
    leading_whitespace_token_id: Optional[int] = None
    leading_whitespace_token_index: Optional[int] = None
    semantic_target_token_id: Optional[int] = None
    semantic_target_token_index: Optional[int] = None

    for idx, token_id in enumerate(token_ids):
        token_text = _decode_token(tokenizer, token_id)
        if idx == 0 and token_text and token_text.strip() == "":
            leading_whitespace_token_id = int(token_id)
            leading_whitespace_token_index = int(idx)
        if semantic_target_token_id is None and token_text.strip() != "":
            semantic_target_token_id = int(token_id)
            semantic_target_token_index = int(idx)
            break

    return {
        "expected_token_ids": token_ids,
        "leading_whitespace_token_id": leading_whitespace_token_id,
        "leading_whitespace_token_text": _decode_token(tokenizer, leading_whitespace_token_id),
        "leading_whitespace_token_index": leading_whitespace_token_index,
        "semantic_target_token_id": semantic_target_token_id,
        "semantic_target_token_text": _decode_token(tokenizer, semantic_target_token_id),
        "semantic_target_token_index": semantic_target_token_index,
    }


def _random_fallback_controls(
    *,
    vocab_size: int,
    excluded_token_ids: Sequence[int],
    count: int,
    seed: int,
) -> List[int]:
    if count <= 0 or vocab_size <= 0:
        return []
    rng = random.Random(seed)
    excluded = {int(token_id) for token_id in excluded_token_ids}
    sample: List[int] = []
    while len(sample) < count:
        candidate = int(rng.randrange(vocab_size))
        if candidate in excluded or candidate in sample:
            continue
        sample.append(candidate)
    return sample


def build_matched_control_token_ids(
    *,
    prefix_token_ids: Sequence[int],
    recent_raw_token_ids: Sequence[int],
    recent_token_ids: Sequence[int],
    sample_size: int,
    vocab_size: int,
    seed: int,
) -> List[int]:
    if sample_size <= 0:
        return []
    prefix_counts = Counter(int(token_id) for token_id in prefix_token_ids)
    recent_set = {int(token_id) for token_id in recent_token_ids}
    recent_raw_set = {int(token_id) for token_id in recent_raw_token_ids}
    candidate_pool = [
        int(token_id)
        for token_id in prefix_counts
        if int(token_id) not in recent_set and int(token_id) not in recent_raw_set
    ]
    used = set()
    chosen: List[int] = []
    for idx, recent_token_id in enumerate(recent_token_ids):
        target_count = int(prefix_counts.get(int(recent_token_id), 0))
        best_token = None
        best_key = None
        for candidate in candidate_pool:
            if candidate in used:
                continue
            key = (
                abs(int(prefix_counts.get(candidate, 0)) - target_count),
                -int(prefix_counts.get(candidate, 0)),
                abs(int(candidate) - int(recent_token_id)),
                candidate,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_token = candidate
        if best_token is not None:
            chosen.append(int(best_token))
            used.add(int(best_token))
            continue
        fallback = _random_fallback_controls(
            vocab_size=vocab_size,
            excluded_token_ids=list(recent_set) + list(used),
            count=1,
            seed=seed + idx * 97,
        )
        if fallback:
            chosen.append(int(fallback[0]))
            used.add(int(fallback[0]))
    while len(chosen) < sample_size:
        fallback = _random_fallback_controls(
            vocab_size=vocab_size,
            excluded_token_ids=list(recent_set) + list(used),
            count=1,
            seed=seed + len(chosen) * 193,
        )
        if not fallback:
            break
        chosen.append(int(fallback[0]))
        used.add(int(fallback[0]))
    return chosen[:sample_size]


def token_stats_from_logits(
    logits_row: torch.Tensor,
    token_id: int,
    *,
    log_norm: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if log_norm is None:
        log_norm = torch.logsumexp(logits_row, dim=-1)
    token_id = int(token_id)
    token_logit = logits_row[token_id]
    token_prob = torch.exp(token_logit - log_norm)
    rank = int(torch.count_nonzero(logits_row > token_logit).item()) + 1
    return {
        "token_id": token_id,
        "logit": float(token_logit.detach().item()),
        "prob": float(token_prob.detach().item()),
        "rank": int(rank),
    }


def maybe_token_stats_from_logits(
    logits_row: torch.Tensor,
    token_id: Optional[int],
    *,
    log_norm: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, Any]]:
    if token_id is None:
        return None
    return token_stats_from_logits(logits_row, int(token_id), log_norm=log_norm)


def prob_mass_for_token_ids(
    logits_row: torch.Tensor,
    token_ids: Sequence[int],
    *,
    log_norm: Optional[torch.Tensor] = None,
) -> float:
    if not token_ids:
        return 0.0
    if log_norm is None:
        log_norm = torch.logsumexp(logits_row, dim=-1)
    indices = torch.tensor([int(token_id) for token_id in token_ids], dtype=torch.long, device=logits_row.device)
    selected_logits = torch.index_select(logits_row, 0, indices)
    return float(torch.exp(selected_logits - log_norm).sum().detach().item())


def mean_logit_for_token_ids(logits_row: torch.Tensor, token_ids: Sequence[int]) -> float:
    if not token_ids:
        return 0.0
    indices = torch.tensor([int(token_id) for token_id in token_ids], dtype=torch.long, device=logits_row.device)
    selected_logits = torch.index_select(logits_row, 0, indices)
    return float(selected_logits.detach().float().mean().item())


def build_prompt_step_metrics(
    logits_row: torch.Tensor,
    prompt_token_ids: Sequence[int],
    *,
    k_values: Sequence[int],
    vocab_size: int,
    seed: int,
    fixed_candidate_token_ids: Optional[Dict[int, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    logits_row = logits_row.float()
    log_norm = torch.logsumexp(logits_row, dim=-1)
    prompt_token_ids = [int(token_id) for token_id in prompt_token_ids]
    by_k: Dict[str, Dict[str, Any]] = {}
    for k in k_values:
        recent_raw_token_ids = prompt_token_ids[-int(k) :]
        recent_token_ids = _dedup_preserve_order(recent_raw_token_ids)
        control_token_ids = build_matched_control_token_ids(
            prefix_token_ids=prompt_token_ids,
            recent_raw_token_ids=recent_raw_token_ids,
            recent_token_ids=recent_token_ids,
            sample_size=len(recent_token_ids),
            vocab_size=vocab_size,
            seed=seed + int(k) * 53,
        )
        candidate_token_id = None
        if fixed_candidate_token_ids and int(k) in fixed_candidate_token_ids:
            candidate_token_id = int(fixed_candidate_token_ids[int(k)])
        elif recent_token_ids:
            recent_indices = torch.tensor(recent_token_ids, dtype=torch.long, device=logits_row.device)
            selected_logits = torch.index_select(logits_row, 0, recent_indices)
            candidate_token_id = int(recent_token_ids[int(torch.argmax(selected_logits).item())])
        candidate_stats = None
        if candidate_token_id is not None:
            candidate_stats = token_stats_from_logits(logits_row, candidate_token_id, log_norm=log_norm)
        recent_mass = prob_mass_for_token_ids(logits_row, recent_token_ids, log_norm=log_norm)
        control_mass = prob_mass_for_token_ids(logits_row, control_token_ids, log_norm=log_norm)
        by_k[str(int(k))] = {
            "k": int(k),
            "recent_raw_token_ids": [int(token_id) for token_id in recent_raw_token_ids],
            "recent_token_ids": [int(token_id) for token_id in recent_token_ids],
            "control_token_ids": [int(token_id) for token_id in control_token_ids],
            "recent_mass": recent_mass,
            "control_mass": control_mass,
            "recent_minus_control_gap": recent_mass - control_mass,
            "mean_recent_logit": mean_logit_for_token_ids(logits_row, recent_token_ids),
            "mean_control_logit": mean_logit_for_token_ids(logits_row, control_token_ids),
            "candidate_token_id": None if candidate_stats is None else int(candidate_stats["token_id"]),
            "candidate_token_logit": None if candidate_stats is None else float(candidate_stats["logit"]),
            "candidate_token_prob": None if candidate_stats is None else float(candidate_stats["prob"]),
            "candidate_token_rank": None if candidate_stats is None else int(candidate_stats["rank"]),
            "candidate_in_recent_set": None
            if candidate_token_id is None
            else bool(int(candidate_token_id) in {int(token_id) for token_id in recent_token_ids}),
        }
    return by_k


def compute_realized_local_copy_metrics(
    *,
    first_token_id: Optional[int],
    prompt_metrics_by_k: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[bool]]:
    out: Dict[str, Optional[bool]] = {}
    for k_str, metrics in prompt_metrics_by_k.items():
        if first_token_id is None:
            out[k_str] = None
            continue
        out[k_str] = bool(int(first_token_id) in {int(token_id) for token_id in metrics.get("recent_token_ids") or []})
    return out


def compute_phrase_continuation_metrics(
    *,
    generated_token_ids: Sequence[int],
    expected_token_ids: Sequence[int],
    min_match_tokens: int,
) -> Dict[str, Any]:
    if not expected_token_ids:
        return {
            "available": False,
            "first_token_match": None,
            "strict_match": None,
            "matched_prefix_tokens": 0,
            "evaluated_tokens": 0,
            "prefix_match_rate": 0.0,
        }
    expected = [int(token_id) for token_id in expected_token_ids]
    generated = [int(token_id) for token_id in generated_token_ids[: len(expected)]]
    matched_prefix_tokens = 0
    for expected_token, actual_token in zip(expected, generated):
        if expected_token != actual_token:
            break
        matched_prefix_tokens += 1
    evaluated_tokens = min(len(expected), len(generated))
    strict_need = min(max(int(min_match_tokens), 1), len(expected))
    return {
        "available": True,
        "first_token_match": None if not generated else bool(generated[0] == expected[0]),
        "strict_match": bool(matched_prefix_tokens >= strict_need),
        "matched_prefix_tokens": int(matched_prefix_tokens),
        "evaluated_tokens": int(evaluated_tokens),
        "prefix_match_rate": 0.0 if evaluated_tokens == 0 else round(matched_prefix_tokens / evaluated_tokens, 6),
    }


def judge_accuracy(case: Dict[str, Any], continuation: str) -> Dict[str, Any]:
    final_boxed = extract_last_boxed(continuation)
    judged = judge_single_answer(
        final_answer=final_boxed,
        gold_answer=resolve_gold_answer(case),
    )
    return {
        "available": bool(judged.get("gold_answer") is not None),
        "gold_answer": judged.get("gold_answer"),
        "final_boxed_answer": final_boxed,
        "is_correct": judged.get("is_correct"),
    }


def build_condition_payload(
    *,
    case: Dict[str, Any],
    prompt_prefix: str,
    prompt_token_ids: Sequence[int],
    prompt_metrics_by_k: Dict[str, Dict[str, Any]],
    logits_row: torch.Tensor,
    generation,
    tokenizer,
    label: str,
    intervention_kind: str,
    intervention_scale: float,
    expected_phrase_tokens: Sequence[int],
    semantic_target_token_id: Optional[int] = None,
    leading_whitespace_token_id: Optional[int] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    token_ids = [int(token_id) for token_id in generation.token_ids]
    first_token_id = None if not token_ids else int(token_ids[0])
    log_norm = torch.logsumexp(logits_row.float(), dim=-1)
    semantic_target_prompt_stats = maybe_token_stats_from_logits(
        logits_row.float(),
        semantic_target_token_id,
        log_norm=log_norm,
    )
    whitespace_target_prompt_stats = maybe_token_stats_from_logits(
        logits_row.float(),
        leading_whitespace_token_id,
        log_norm=log_norm,
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_ended = bool(token_ids and eos_token_id is not None and int(token_ids[-1]) == int(eos_token_id))
    max_new_tokens = int(case.get("max_new_tokens") or 0)
    metadata = dict(case.get("metadata") or {})
    track_phrase_continuation = bool(metadata.get("track_phrase_continuation"))
    phrase_metrics = compute_phrase_continuation_metrics(
        generated_token_ids=token_ids,
        expected_token_ids=expected_phrase_tokens if track_phrase_continuation else [],
        min_match_tokens=int(case.get("min_phrase_match_tokens") or 2),
    )
    continuation = str(generation.continuation)
    return {
        "label": label,
        "intervention_kind": intervention_kind,
        "intervention_scale": float(intervention_scale),
        "prompt_prefix": prompt_prefix,
        "prompt_token_count": int(len(prompt_token_ids)),
        "prompt_metrics_by_k": prompt_metrics_by_k,
        "generated_tokens": int(generation.generated_tokens),
        "continuation": continuation,
        "full_text": str(generation.full_text),
        "token_ids": token_ids,
        "first_token_id": first_token_id,
        "first_token_text": _decode_token(tokenizer, first_token_id),
        "target_first_token_id": semantic_target_token_id,
        "target_first_token_text": _decode_token(tokenizer, semantic_target_token_id),
        "target_first_token_prob": None if semantic_target_prompt_stats is None else float(semantic_target_prompt_stats["prob"]),
        "target_first_token_logit": None if semantic_target_prompt_stats is None else float(semantic_target_prompt_stats["logit"]),
        "target_first_token_rank": None if semantic_target_prompt_stats is None else int(semantic_target_prompt_stats["rank"]),
        "realized_target_first_token_match": None
        if first_token_id is None or semantic_target_token_id is None
        else bool(int(first_token_id) == int(semantic_target_token_id)),
        "whitespace_target_token_id": leading_whitespace_token_id,
        "whitespace_target_token_text": _decode_token(tokenizer, leading_whitespace_token_id),
        "whitespace_target_token_prob": None
        if whitespace_target_prompt_stats is None
        else float(whitespace_target_prompt_stats["prob"]),
        "whitespace_target_token_logit": None
        if whitespace_target_prompt_stats is None
        else float(whitespace_target_prompt_stats["logit"]),
        "whitespace_target_token_rank": None
        if whitespace_target_prompt_stats is None
        else int(whitespace_target_prompt_stats["rank"]),
        "realized_whitespace_target_token_match": None
        if first_token_id is None or leading_whitespace_token_id is None
        else bool(int(first_token_id) == int(leading_whitespace_token_id)),
        "realized_local_copy_by_k": compute_realized_local_copy_metrics(
            first_token_id=first_token_id,
            prompt_metrics_by_k=prompt_metrics_by_k,
        ),
        "phrase_continuation": phrase_metrics,
        "hit_max_new_tokens": bool(int(generation.generated_tokens) >= max_new_tokens),
        "ended_early": bool(int(generation.generated_tokens) < max_new_tokens),
        "eos_ended": eos_ended,
        "accuracy": judge_accuracy(case, continuation),
        "debug": debug or {},
    }


def attach_vs_baseline_deltas(
    row: Dict[str, Any],
    *,
    k_values: Sequence[int],
) -> None:
    baseline = dict(row.get("baseline") or {})
    baseline_by_k = dict(baseline.get("prompt_metrics_by_k") or {})
    for condition_label, condition_payload in list(row.items()):
        if condition_label in {"example_id", "family", "case"} or condition_label == "baseline":
            continue
        if not isinstance(condition_payload, dict):
            continue
        current_by_k = dict(condition_payload.get("prompt_metrics_by_k") or {})
        delta_by_k: Dict[str, Dict[str, Any]] = {}
        for k in k_values:
            key = str(int(k))
            base_metrics = dict(baseline_by_k.get(key) or {})
            current_metrics = dict(current_by_k.get(key) or {})
            if not base_metrics or not current_metrics:
                continue
            delta_by_k[key] = {
                "recent_mass_delta": round(
                    float(current_metrics.get("recent_mass", 0.0)) - float(base_metrics.get("recent_mass", 0.0)),
                    6,
                ),
                "control_mass_delta": round(
                    float(current_metrics.get("control_mass", 0.0)) - float(base_metrics.get("control_mass", 0.0)),
                    6,
                ),
                "recent_gap_delta": round(
                    float(current_metrics.get("recent_minus_control_gap", 0.0))
                    - float(base_metrics.get("recent_minus_control_gap", 0.0)),
                    6,
                ),
                "candidate_prob_delta": None
                if current_metrics.get("candidate_token_prob") is None or base_metrics.get("candidate_token_prob") is None
                else round(
                    float(current_metrics.get("candidate_token_prob", 0.0))
                    - float(base_metrics.get("candidate_token_prob", 0.0)),
                    6,
                ),
                "candidate_logit_delta": None
                if current_metrics.get("candidate_token_logit") is None or base_metrics.get("candidate_token_logit") is None
                else round(
                    float(current_metrics.get("candidate_token_logit", 0.0))
                    - float(base_metrics.get("candidate_token_logit", 0.0)),
                    6,
                ),
                "mean_recent_logit_delta": round(
                    float(current_metrics.get("mean_recent_logit", 0.0))
                    - float(base_metrics.get("mean_recent_logit", 0.0)),
                    6,
                ),
                "mean_control_logit_delta": round(
                    float(current_metrics.get("mean_control_logit", 0.0))
                    - float(base_metrics.get("mean_control_logit", 0.0)),
                    6,
                ),
            }
        condition_payload["vs_baseline_by_k"] = delta_by_k
        condition_payload["target_vs_baseline"] = {
            "target_prob_delta": None
            if condition_payload.get("target_first_token_prob") is None or baseline.get("target_first_token_prob") is None
            else round(
                float(condition_payload.get("target_first_token_prob", 0.0))
                - float(baseline.get("target_first_token_prob", 0.0)),
                6,
            ),
            "target_logit_delta": None
            if condition_payload.get("target_first_token_logit") is None or baseline.get("target_first_token_logit") is None
            else round(
                float(condition_payload.get("target_first_token_logit", 0.0))
                - float(baseline.get("target_first_token_logit", 0.0)),
                6,
            ),
            "target_rank_delta": None
            if condition_payload.get("target_first_token_rank") is None or baseline.get("target_first_token_rank") is None
            else round(
                float(condition_payload.get("target_first_token_rank", 0.0))
                - float(baseline.get("target_first_token_rank", 0.0)),
                6,
            ),
            "whitespace_target_prob_delta": None
            if condition_payload.get("whitespace_target_token_prob") is None or baseline.get("whitespace_target_token_prob") is None
            else round(
                float(condition_payload.get("whitespace_target_token_prob", 0.0))
                - float(baseline.get("whitespace_target_token_prob", 0.0)),
                6,
            ),
            "whitespace_target_logit_delta": None
            if condition_payload.get("whitespace_target_token_logit") is None or baseline.get("whitespace_target_token_logit") is None
            else round(
                float(condition_payload.get("whitespace_target_token_logit", 0.0))
                - float(baseline.get("whitespace_target_token_logit", 0.0)),
                6,
            ),
            "whitespace_target_rank_delta": None
            if condition_payload.get("whitespace_target_token_rank") is None or baseline.get("whitespace_target_token_rank") is None
            else round(
                float(condition_payload.get("whitespace_target_token_rank", 0.0))
                - float(baseline.get("whitespace_target_token_rank", 0.0)),
                6,
            ),
        }


def flatten_condition_rows(
    result_rows: Sequence[Dict[str, Any]],
    *,
    condition_order: Sequence[str],
    k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []
    for row in result_rows:
        case = dict(row.get("case") or {})
        for condition_label in condition_order:
            condition = dict(row.get(condition_label) or {})
            if not condition:
                continue
            flat: Dict[str, Any] = {
                "example_id": str(row.get("example_id") or ""),
                "family": str(row.get("family") or ""),
                "condition": condition_label,
                "intervention_kind": str(condition.get("intervention_kind") or ""),
                "intervention_scale": float(condition.get("intervention_scale", 1.0)),
                "generated_tokens": int(condition.get("generated_tokens", 0)),
                "ended_early": bool(condition.get("ended_early")),
                "eos_ended": bool(condition.get("eos_ended")),
                "first_token_id": condition.get("first_token_id"),
                "first_token_text": condition.get("first_token_text") or "",
                "target_first_token_id": condition.get("target_first_token_id"),
                "target_first_token_text": condition.get("target_first_token_text") or "",
                "target_first_token_prob": condition.get("target_first_token_prob"),
                "target_first_token_logit": condition.get("target_first_token_logit"),
                "target_first_token_rank": condition.get("target_first_token_rank"),
                "realized_target_first_token_match": condition.get("realized_target_first_token_match"),
                "whitespace_target_token_id": condition.get("whitespace_target_token_id"),
                "whitespace_target_token_text": condition.get("whitespace_target_token_text") or "",
                "whitespace_target_token_prob": condition.get("whitespace_target_token_prob"),
                "whitespace_target_token_logit": condition.get("whitespace_target_token_logit"),
                "whitespace_target_token_rank": condition.get("whitespace_target_token_rank"),
                "realized_whitespace_target_token_match": condition.get("realized_whitespace_target_token_match"),
                "accuracy_available": bool(dict(condition.get("accuracy") or {}).get("available")),
                "accuracy_is_correct": dict(condition.get("accuracy") or {}).get("is_correct"),
                "final_boxed_answer": dict(condition.get("accuracy") or {}).get("final_boxed_answer"),
                "correct_answer": case.get("correct_answer"),
                "wrong_answer": case.get("wrong_answer"),
                "phrase_metric_available": bool(dict(condition.get("phrase_continuation") or {}).get("available")),
                "phrase_continuation_rate": dict(condition.get("phrase_continuation") or {}).get("strict_match"),
                "phrase_prefix_match_rate": float(dict(condition.get("phrase_continuation") or {}).get("prefix_match_rate", 0.0)),
                "target_prob_delta": dict(condition.get("target_vs_baseline") or {}).get("target_prob_delta"),
                "target_logit_delta": dict(condition.get("target_vs_baseline") or {}).get("target_logit_delta"),
                "target_rank_delta": dict(condition.get("target_vs_baseline") or {}).get("target_rank_delta"),
                "whitespace_target_prob_delta": dict(condition.get("target_vs_baseline") or {}).get("whitespace_target_prob_delta"),
                "whitespace_target_logit_delta": dict(condition.get("target_vs_baseline") or {}).get("whitespace_target_logit_delta"),
                "whitespace_target_rank_delta": dict(condition.get("target_vs_baseline") or {}).get("whitespace_target_rank_delta"),
            }
            prompt_metrics_by_k = dict(condition.get("prompt_metrics_by_k") or {})
            realized_by_k = dict(condition.get("realized_local_copy_by_k") or {})
            delta_by_k = dict(condition.get("vs_baseline_by_k") or {})
            for k in k_values:
                key = str(int(k))
                prompt_metrics = dict(prompt_metrics_by_k.get(key) or {})
                deltas = dict(delta_by_k.get(key) or {})
                flat[f"recent_mass_k{key}"] = float(prompt_metrics.get("recent_mass", 0.0))
                flat[f"control_mass_k{key}"] = float(prompt_metrics.get("control_mass", 0.0))
                flat[f"recent_gap_k{key}"] = float(prompt_metrics.get("recent_minus_control_gap", 0.0))
                flat[f"candidate_prob_k{key}"] = prompt_metrics.get("candidate_token_prob")
                flat[f"candidate_logit_k{key}"] = prompt_metrics.get("candidate_token_logit")
                flat[f"candidate_rank_k{key}"] = prompt_metrics.get("candidate_token_rank")
                flat[f"candidate_in_recent_set_k{key}"] = prompt_metrics.get("candidate_in_recent_set")
                flat[f"mean_recent_logit_k{key}"] = float(prompt_metrics.get("mean_recent_logit", 0.0))
                flat[f"mean_control_logit_k{key}"] = float(prompt_metrics.get("mean_control_logit", 0.0))
                flat[f"realized_local_copy_k{key}"] = realized_by_k.get(key)
                flat[f"recent_mass_delta_k{key}"] = deltas.get("recent_mass_delta")
                flat[f"control_mass_delta_k{key}"] = deltas.get("control_mass_delta")
                flat[f"recent_gap_delta_k{key}"] = deltas.get("recent_gap_delta")
                flat[f"candidate_prob_delta_k{key}"] = deltas.get("candidate_prob_delta")
                flat[f"candidate_logit_delta_k{key}"] = deltas.get("candidate_logit_delta")
                flat[f"mean_recent_logit_delta_k{key}"] = deltas.get("mean_recent_logit_delta")
                flat[f"mean_control_logit_delta_k{key}"] = deltas.get("mean_control_logit_delta")
            flat_rows.append(flat)
    return flat_rows


def _safe_bool_mean(values: Sequence[Optional[bool]]) -> float:
    filtered = [1.0 if bool(item) else 0.0 for item in values if item is not None]
    if not filtered:
        return 0.0
    return round(sum(filtered) / len(filtered), 6)


def aggregate_flat_rows(
    flat_rows: Sequence[Dict[str, Any]],
    *,
    k_values: Sequence[int],
    condition_order: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in flat_rows:
        grouped[(str(row.get("family") or ""), str(row.get("condition") or ""))].append(dict(row))
        grouped[("overall", str(row.get("condition") or ""))].append(dict(row))

    summary_rows: List[Dict[str, Any]] = []
    families = sorted({family for family, _ in grouped if family != "overall"})
    ordered_families = ["overall"] + families
    for family in ordered_families:
        for condition in condition_order:
            rows = grouped.get((family, condition), [])
            if not rows:
                continue
            summary: Dict[str, Any] = {
                "family": family,
                "condition": condition,
                "examples": int(len(rows)),
                "mean_generated_tokens": round(_mean(float(row.get("generated_tokens", 0.0)) for row in rows), 6),
                "early_end_rate": _safe_bool_mean([bool(row.get("ended_early")) for row in rows]),
                "eos_rate": _safe_bool_mean([bool(row.get("eos_ended")) for row in rows]),
            }
            phrase_rows = [row for row in rows if bool(row.get("phrase_metric_available"))]
            summary["phrase_metric_examples"] = int(len(phrase_rows))
            summary["phrase_continuation_rate"] = _safe_bool_mean(
                [row.get("phrase_continuation_rate") for row in phrase_rows]
            )
            summary["phrase_prefix_match_rate"] = round(
                _mean(float(row.get("phrase_prefix_match_rate", 0.0)) for row in phrase_rows),
                6,
            )
            target_rows = [row for row in rows if row.get("target_first_token_id") is not None]
            summary["target_first_token_examples"] = int(len(target_rows))
            summary["target_first_token_match_rate"] = _safe_bool_mean(
                [row.get("realized_target_first_token_match") for row in target_rows]
            )
            summary["target_first_token_prob_mean"] = round(
                _mean(float(row.get("target_first_token_prob", 0.0)) for row in target_rows if row.get("target_first_token_prob") is not None),
                6,
            )
            summary["target_first_token_logit_mean"] = round(
                _mean(float(row.get("target_first_token_logit", 0.0)) for row in target_rows if row.get("target_first_token_logit") is not None),
                6,
            )
            summary["target_first_token_rank_mean"] = round(
                _mean(float(row.get("target_first_token_rank", 0.0)) for row in target_rows if row.get("target_first_token_rank") is not None),
                6,
            )
            summary["target_first_token_prob_delta_mean"] = round(
                _mean(float(row.get("target_prob_delta", 0.0)) for row in rows if row.get("target_prob_delta") is not None),
                6,
            )
            summary["target_first_token_logit_delta_mean"] = round(
                _mean(float(row.get("target_logit_delta", 0.0)) for row in rows if row.get("target_logit_delta") is not None),
                6,
            )
            summary["target_first_token_rank_delta_mean"] = round(
                _mean(float(row.get("target_rank_delta", 0.0)) for row in rows if row.get("target_rank_delta") is not None),
                6,
            )
            whitespace_rows = [row for row in rows if row.get("whitespace_target_token_id") is not None]
            summary["whitespace_target_token_examples"] = int(len(whitespace_rows))
            summary["whitespace_target_token_match_rate"] = _safe_bool_mean(
                [row.get("realized_whitespace_target_token_match") for row in whitespace_rows]
            )
            summary["whitespace_target_token_prob_mean"] = round(
                _mean(
                    float(row.get("whitespace_target_token_prob", 0.0))
                    for row in whitespace_rows
                    if row.get("whitespace_target_token_prob") is not None
                ),
                6,
            )
            summary["whitespace_target_token_logit_mean"] = round(
                _mean(
                    float(row.get("whitespace_target_token_logit", 0.0))
                    for row in whitespace_rows
                    if row.get("whitespace_target_token_logit") is not None
                ),
                6,
            )
            summary["whitespace_target_token_rank_mean"] = round(
                _mean(
                    float(row.get("whitespace_target_token_rank", 0.0))
                    for row in whitespace_rows
                    if row.get("whitespace_target_token_rank") is not None
                ),
                6,
            )
            summary["whitespace_target_token_prob_delta_mean"] = round(
                _mean(
                    float(row.get("whitespace_target_prob_delta", 0.0))
                    for row in rows
                    if row.get("whitespace_target_prob_delta") is not None
                ),
                6,
            )
            summary["whitespace_target_token_logit_delta_mean"] = round(
                _mean(
                    float(row.get("whitespace_target_logit_delta", 0.0))
                    for row in rows
                    if row.get("whitespace_target_logit_delta") is not None
                ),
                6,
            )
            summary["whitespace_target_token_rank_delta_mean"] = round(
                _mean(
                    float(row.get("whitespace_target_rank_delta", 0.0))
                    for row in rows
                    if row.get("whitespace_target_rank_delta") is not None
                ),
                6,
            )
            accuracy_rows = [row for row in rows if bool(row.get("accuracy_available"))]
            summary["accuracy_examples"] = int(len(accuracy_rows))
            if accuracy_rows:
                summary["accuracy_rate"] = round(
                    _mean(1.0 if bool(row.get("accuracy_is_correct")) else 0.0 for row in accuracy_rows),
                    6,
                )
            else:
                summary["accuracy_rate"] = 0.0
            for k in k_values:
                key = str(int(k))
                summary[f"recent_mass_mean_k{key}"] = round(
                    _mean(float(row.get(f"recent_mass_k{key}", 0.0)) for row in rows),
                    6,
                )
                summary[f"control_mass_mean_k{key}"] = round(
                    _mean(float(row.get(f"control_mass_k{key}", 0.0)) for row in rows),
                    6,
                )
                summary[f"recent_gap_mean_k{key}"] = round(
                    _mean(float(row.get(f"recent_gap_k{key}", 0.0)) for row in rows),
                    6,
                )
                summary[f"realized_local_copy_rate_k{key}"] = _safe_bool_mean(
                    [row.get(f"realized_local_copy_k{key}") for row in rows]
                )
                candidate_probs = [row.get(f"candidate_prob_k{key}") for row in rows if row.get(f"candidate_prob_k{key}") is not None]
                candidate_logits = [
                    row.get(f"candidate_logit_k{key}") for row in rows if row.get(f"candidate_logit_k{key}") is not None
                ]
                candidate_ranks = [row.get(f"candidate_rank_k{key}") for row in rows if row.get(f"candidate_rank_k{key}") is not None]
                candidate_in_recent = [
                    row.get(f"candidate_in_recent_set_k{key}")
                    for row in rows
                    if row.get(f"candidate_in_recent_set_k{key}") is not None
                ]
                summary[f"candidate_prob_mean_k{key}"] = round(_mean(float(item) for item in candidate_probs), 6)
                summary[f"candidate_logit_mean_k{key}"] = round(_mean(float(item) for item in candidate_logits), 6)
                summary[f"candidate_rank_mean_k{key}"] = round(_mean(float(item) for item in candidate_ranks), 6)
                summary[f"candidate_in_recent_set_rate_k{key}"] = _safe_bool_mean(candidate_in_recent)
                delta_recent = [row.get(f"recent_mass_delta_k{key}") for row in rows if row.get(f"recent_mass_delta_k{key}") is not None]
                delta_control = [row.get(f"control_mass_delta_k{key}") for row in rows if row.get(f"control_mass_delta_k{key}") is not None]
                delta_gap = [row.get(f"recent_gap_delta_k{key}") for row in rows if row.get(f"recent_gap_delta_k{key}") is not None]
                delta_candidate_prob = [
                    row.get(f"candidate_prob_delta_k{key}")
                    for row in rows
                    if row.get(f"candidate_prob_delta_k{key}") is not None
                ]
                delta_candidate_logit = [
                    row.get(f"candidate_logit_delta_k{key}")
                    for row in rows
                    if row.get(f"candidate_logit_delta_k{key}") is not None
                ]
                delta_recent_logit = [
                    row.get(f"mean_recent_logit_delta_k{key}")
                    for row in rows
                    if row.get(f"mean_recent_logit_delta_k{key}") is not None
                ]
                delta_control_logit = [
                    row.get(f"mean_control_logit_delta_k{key}")
                    for row in rows
                    if row.get(f"mean_control_logit_delta_k{key}") is not None
                ]
                summary[f"recent_mass_delta_mean_k{key}"] = round(_mean(float(item) for item in delta_recent), 6)
                summary[f"control_mass_delta_mean_k{key}"] = round(_mean(float(item) for item in delta_control), 6)
                summary[f"recent_gap_delta_mean_k{key}"] = round(_mean(float(item) for item in delta_gap), 6)
                summary[f"candidate_prob_delta_mean_k{key}"] = round(_mean(float(item) for item in delta_candidate_prob), 6)
                summary[f"candidate_logit_delta_mean_k{key}"] = round(_mean(float(item) for item in delta_candidate_logit), 6)
                summary[f"mean_recent_logit_delta_mean_k{key}"] = round(
                    _mean(float(item) for item in delta_recent_logit),
                    6,
                )
                summary[f"mean_control_logit_delta_mean_k{key}"] = round(
                    _mean(float(item) for item in delta_control_logit),
                    6,
                )
            summary_rows.append(summary)
    return summary_rows


def build_summary_json(
    *,
    summary_rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    condition_order: Sequence[str],
) -> Dict[str, Any]:
    nested: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for row in summary_rows:
        nested[str(row.get("family") or "")][str(row.get("condition") or "")] = dict(row)
    return {
        "k_values": [int(k) for k in k_values],
        "condition_order": [str(item) for item in condition_order],
        "families": nested,
    }


def maybe_write_plots(
    summary_rows: Sequence[Dict[str, Any]],
    *,
    plots_dir: str | Path,
    k_values: Sequence[int],
    condition_order: Sequence[str],
) -> List[str]:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows_by_family: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        rows_by_family[str(row.get("family") or "")][str(row.get("condition") or "")] = dict(row)

    plot_paths: List[str] = []

    def _svg_escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _write_simple_svg_line_chart(
        output_path: Path,
        *,
        title: str,
        ylabel: str,
        x_values: Sequence[int],
        series_rows: Dict[str, Sequence[float]],
    ) -> str:
        width = 720
        height = 440
        left = 72
        right = 24
        top = 48
        bottom = 64
        plot_w = width - left - right
        plot_h = height - top - bottom
        flat_values = [float(value) for values in series_rows.values() for value in values]
        y_min = min(flat_values) if flat_values else 0.0
        y_max = max(flat_values) if flat_values else 1.0
        if math.isclose(y_min, y_max):
            y_min -= 0.5
            y_max += 0.5
        palette = ["#1b6ca8", "#c44536", "#3a7d44", "#7b2cbf", "#f08c00"]

        def x_to_svg(idx: int) -> float:
            if len(x_values) <= 1:
                return left + plot_w / 2.0
            return left + idx * (plot_w / max(len(x_values) - 1, 1))

        def y_to_svg(value: float) -> float:
            ratio = (float(value) - y_min) / max(y_max - y_min, 1e-9)
            return top + plot_h * (1.0 - ratio)

        lines: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
            f'<text x="{width/2:.1f}" y="24" font-size="18" text-anchor="middle" font-family="monospace">{_svg_escape(title)}</text>',
            f'<text x="20" y="{height/2:.1f}" font-size="12" text-anchor="middle" transform="rotate(-90 20 {height/2:.1f})" font-family="monospace">{_svg_escape(ylabel)}</text>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444" stroke-width="1.5" />',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444" stroke-width="1.5" />',
        ]
        for tick_idx in range(5):
            tick_value = y_min + (y_max - y_min) * tick_idx / 4.0
            y = y_to_svg(tick_value)
            lines.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{left - 8}" y="{y + 4:.2f}" font-size="11" text-anchor="end" font-family="monospace">{tick_value:.3f}</text>'
            )
        for idx, x_val in enumerate(x_values):
            x = x_to_svg(idx)
            lines.append(
                f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#444" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{x:.2f}" y="{top + plot_h + 20}" font-size="11" text-anchor="middle" font-family="monospace">{int(x_val)}</text>'
            )
        legend_x = left + 8
        legend_y = height - 18
        for idx, (label, values) in enumerate(series_rows.items()):
            color = palette[idx % len(palette)]
            points = " ".join(f"{x_to_svg(i):.2f},{y_to_svg(float(v)):.2f}" for i, v in enumerate(values))
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />')
            for i, value in enumerate(values):
                lines.append(
                    f'<circle cx="{x_to_svg(i):.2f}" cy="{y_to_svg(float(value)):.2f}" r="3.5" fill="{color}" />'
                )
            tx = legend_x + idx * 138
            lines.append(f'<rect x="{tx}" y="{legend_y - 10}" width="14" height="3" fill="{color}" />')
            lines.append(
                f'<text x="{tx + 20}" y="{legend_y - 2}" font-size="11" font-family="monospace">{_svg_escape(label)}</text>'
            )
        lines.append("</svg>")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return str(output_path)

    def _plot_metric(metric_prefix: str, family: str, title: str, ylabel: str) -> Optional[str]:
        family_rows = rows_by_family.get(family, {})
        if not family_rows:
            return None
        if not MATPLOTLIB_AVAILABLE:
            series_rows: Dict[str, List[float]] = {}
            for condition in condition_order:
                row = family_rows.get(condition)
                if not row:
                    continue
                series_rows[condition] = [float(row.get(f"{metric_prefix}_k{int(k)}", 0.0)) for k in k_values]
            if not series_rows:
                return None
            output_path = plots_dir / f"{family}_{metric_prefix}.svg"
            return _write_simple_svg_line_chart(
                output_path,
                title=title,
                ylabel=ylabel,
                x_values=[int(k) for k in k_values],
                series_rows=series_rows,
            )
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        wrote_any = False
        for condition in condition_order:
            row = family_rows.get(condition)
            if not row:
                continue
            xs = [int(k) for k in k_values]
            ys = [float(row.get(f"{metric_prefix}_k{int(k)}", 0.0)) for k in k_values]
            ax.plot(xs, ys, marker="o", linewidth=2.0, label=condition)
            wrote_any = True
        if not wrote_any:
            plt.close(fig)
            return None
        ax.set_title(title)
        ax.set_xlabel("Recent Window k")
        ax.set_ylabel(ylabel)
        ax.set_xticks([int(k) for k in k_values])
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        output_path = plots_dir / f"{family}_{metric_prefix}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return str(output_path)

    for family in sorted(rows_by_family.keys()):
        for metric_prefix, title, ylabel in [
            ("recent_mass_mean", f"{family}: Recent-Set Probability Mass", "Mean Probability Mass"),
            ("control_mass_mean", f"{family}: Matched Control Probability Mass", "Mean Probability Mass"),
            ("recent_gap_mean", f"{family}: Recent Minus Control Gap", "Mean Probability Gap"),
            ("candidate_prob_mean", f"{family}: Candidate Token Probability", "Mean Probability"),
            ("realized_local_copy_rate", f"{family}: Realized Local Copy Rate", "Rate"),
        ]:
            maybe_path = _plot_metric(metric_prefix, family, title, ylabel)
            if maybe_path is not None:
                plot_paths.append(maybe_path)
    return plot_paths


def write_report(
    path: str | Path,
    *,
    args: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    condition_order: Sequence[str],
) -> None:
    rows_by_family: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        rows_by_family[str(row.get("family") or "")][str(row.get("condition") or "")] = dict(row)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Local-Copy Temptation Experiment\n\n")
        f.write(f"- model: `{args.get('model_name_or_path')}`\n")
        f.write(f"- head_label: `{args.get('head_label')}`\n")
        f.write(f"- prompt_variant: `{args.get('prompt_variant', 'default')}`\n")
        f.write(f"- k_values: `{','.join(str(int(k)) for k in k_values)}`\n")
        f.write(f"- scale_values: `{args.get('scale_values')}`\n")
        f.write(f"- token_count: `{args.get('token_family_count')}`\n")
        f.write(f"- phrase_count: `{args.get('phrase_family_count')}`\n")
        f.write(f"- cot_count: `{args.get('cot_family_count')}`\n\n")

        ordered_families = ["overall"] + sorted(family for family in rows_by_family.keys() if family != "overall")
        for family in ordered_families:
            family_rows = rows_by_family.get(family)
            if not family_rows:
                continue
            f.write(f"## {family.capitalize()}\n\n")
            for condition in condition_order:
                row = family_rows.get(condition)
                if not row:
                    continue
                line_parts = [
                    f"condition=`{condition}`",
                    f"mean_generated_tokens={float(row.get('mean_generated_tokens', 0.0)):.2f}",
                    f"early_end_rate={float(row.get('early_end_rate', 0.0)):.4f}",
                    f"eos_rate={float(row.get('eos_rate', 0.0)):.4f}",
                ]
                if int(row.get("target_first_token_examples", 0)) > 0:
                    line_parts.append(
                        f"target_first_token_match_rate={float(row.get('target_first_token_match_rate', 0.0)):.4f}"
                    )
                    line_parts.append(
                        f"target_first_token_prob={float(row.get('target_first_token_prob_mean', 0.0)):.4f}"
                    )
                if int(row.get("whitespace_target_token_examples", 0)) > 0:
                    line_parts.append(
                        f"whitespace_target_match_rate={float(row.get('whitespace_target_token_match_rate', 0.0)):.4f}"
                    )
                if int(row.get("phrase_metric_examples", 0)) > 0:
                    line_parts.append(f"phrase_continuation_rate={float(row.get('phrase_continuation_rate', 0.0)):.4f}")
                if int(row.get("accuracy_examples", 0)) > 0:
                    line_parts.append(f"accuracy_rate={float(row.get('accuracy_rate', 0.0)):.4f}")
                line_parts.extend(
                    [
                        f"recent_mass_k1={float(row.get('recent_mass_mean_k1', 0.0)):.4f}",
                        f"candidate_prob_k1={float(row.get('candidate_prob_mean_k1', 0.0)):.4f}",
                        f"recent_mass_k8={float(row.get('recent_mass_mean_k8', 0.0)):.4f}",
                        f"copy_rate_k1={float(row.get('realized_local_copy_rate_k1', 0.0)):.4f}",
                        f"copy_rate_k8={float(row.get('realized_local_copy_rate_k8', 0.0)):.4f}",
                    ]
                )
                f.write("- " + ", ".join(line_parts) + "\n")
            f.write("\n")
