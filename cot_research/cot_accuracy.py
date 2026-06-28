from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .answer_extraction import answers_match, extract_last_boxed
from .row_utils import resolve_row_identities, select_continuation_text


def row_identity_values(row: Dict[str, Any]) -> List[str]:
    return resolve_row_identities(row)


def build_row_lookup(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for value in row_identity_values(row):
            lookup[value] = row
    return lookup


def select_cot_text(row: Dict[str, Any]) -> str:
    return select_continuation_text(row)


def extract_final_answer_from_row(row: Dict[str, Any]) -> Optional[str]:
    for key in ["final_boxed_answer", "predicted_answer", "final_answer", "answer"]:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            boxed = extract_last_boxed(text)
            return boxed if boxed is not None else text

    cot_text = select_cot_text(row)
    if cot_text.strip():
        return extract_last_boxed(cot_text)
    return None


def _resolve_boxed_from_text_field(row: Dict[str, Any], key: str) -> Optional[str]:
    value = row.get(key)
    if isinstance(value, str) and value.strip():
        return extract_last_boxed(value)
    return None


def resolve_gold_answer(row: Dict[str, Any]) -> Optional[str]:
    direct_candidates = [
        row.get("correct_answer"),
        row.get("gold_answer"),
        row.get("target_answer"),
    ]
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    direct_candidates.extend(
        [
            metadata.get("correct_answer"),
            metadata.get("gold_answer"),
            metadata.get("target_answer"),
        ]
    )
    for candidate in direct_candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            boxed = extract_last_boxed(text)
            return boxed if boxed is not None else text

    for key in ["reference_solution", "solution"]:
        boxed = _resolve_boxed_from_text_field(row, key)
        if boxed is not None:
            return boxed

    for key in ["reference_messages", "messages"]:
        messages = row.get(key)
        if not isinstance(messages, list):
            continue
        assistant_texts = [
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, dict) and str(item.get("role", "")).lower() == "assistant"
        ]
        for text in reversed(assistant_texts):
            boxed = extract_last_boxed(text)
            if boxed is not None:
                return boxed
    return None


def judge_single_answer(
    *,
    final_answer: Optional[str],
    gold_answer: Optional[str],
) -> Dict[str, Any]:
    if gold_answer is None:
        return {
            "category": "unverifiable",
            "is_correct": None,
            "gold_answer": None,
            "final_answer": final_answer,
        }
    is_correct = bool(answers_match(final_answer, gold_answer))
    return {
        "category": "correct" if is_correct else "wrong",
        "is_correct": is_correct,
        "gold_answer": gold_answer,
        "final_answer": final_answer,
    }


def classify_answer_change(
    *,
    gold_answer: Optional[str],
    baseline_answer: Optional[str],
    intervention_answer: Optional[str],
) -> Dict[str, Any]:
    if gold_answer is None:
        return {
            "category": "unverifiable",
            "baseline_correct": None,
            "intervention_correct": None,
            "gold_answer": None,
            "baseline_answer": baseline_answer,
            "intervention_answer": intervention_answer,
        }

    baseline_correct = bool(answers_match(baseline_answer, gold_answer))
    intervention_correct = bool(answers_match(intervention_answer, gold_answer))
    if (not baseline_correct) and intervention_correct:
        category = "newly_correct"
    elif baseline_correct and (not intervention_correct):
        category = "regressed"
    elif baseline_correct and intervention_correct:
        category = "remained_correct"
    else:
        category = "remained_wrong"

    return {
        "category": category,
        "baseline_correct": baseline_correct,
        "intervention_correct": intervention_correct,
        "gold_answer": gold_answer,
        "baseline_answer": baseline_answer,
        "intervention_answer": intervention_answer,
    }


def judge_single_row(
    row: Dict[str, Any],
    gold_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_row = gold_row or row
    gold_answer = resolve_gold_answer(source_row)
    final_answer = extract_final_answer_from_row(row)
    return judge_single_answer(final_answer=final_answer, gold_answer=gold_answer)


def judge_comparison_row(
    row: Dict[str, Any],
    gold_row: Optional[Dict[str, Any]] = None,
    *,
    baseline_key: str = "baseline",
    intervention_key: str = "intervention",
) -> Dict[str, Any]:
    source_row = gold_row or row
    gold_answer = resolve_gold_answer(source_row)
    baseline_answer = extract_final_answer_from_row(dict(row.get(baseline_key) or {}))
    intervention_answer = extract_final_answer_from_row(dict(row.get(intervention_key) or {}))
    return classify_answer_change(
        gold_answer=gold_answer,
        baseline_answer=baseline_answer,
        intervention_answer=intervention_answer,
    )


def summarize_single_accuracy(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    total = 0
    verifiable = 0
    correct = 0
    for row in rows:
        total += 1
        accuracy = dict(row.get("accuracy") or {})
        category = str(accuracy.get("category", "unverifiable"))
        counter.update([category])
        if category != "unverifiable":
            verifiable += 1
            if bool(accuracy.get("is_correct")):
                correct += 1
    return {
        "total_examples": total,
        "verifiable_examples": verifiable,
        "accuracy_counts": dict(counter),
        "correct_rate_over_verifiable": 0.0 if verifiable == 0 else round(correct / verifiable, 6),
    }


def summarize_comparison_accuracy(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    counter: Counter[str] = Counter()
    total = 0
    verifiable = 0
    baseline_correct = 0
    intervention_correct = 0
    for row in rows:
        total += 1
        comparison = dict(row.get("answer_comparison") or {})
        category = str(comparison.get("category", "unverifiable"))
        counter.update([category])
        if category != "unverifiable":
            verifiable += 1
            if bool(comparison.get("baseline_correct")):
                baseline_correct += 1
            if bool(comparison.get("intervention_correct")):
                intervention_correct += 1
    return {
        "total_examples": total,
        "verifiable_examples": verifiable,
        "accuracy_counts": dict(counter),
        "baseline_correct_rate_over_verifiable": 0.0 if verifiable == 0 else round(baseline_correct / verifiable, 6),
        "intervention_correct_rate_over_verifiable": 0.0 if verifiable == 0 else round(intervention_correct / verifiable, 6),
        "newly_correct_rate_over_verifiable": 0.0 if verifiable == 0 else round(counter.get("newly_correct", 0) / verifiable, 6),
        "regression_rate_over_verifiable": 0.0 if verifiable == 0 else round(counter.get("regressed", 0) / verifiable, 6),
    }
