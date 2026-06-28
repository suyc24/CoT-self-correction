from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .local_copy_temptation import build_local_copy_cases


PHRASE_STATE_ORDER = ["local", "matched_control", "position_shift"]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_sharp_phrase_mechanism_cases(
    *,
    count: int,
    max_new_tokens: int,
    seed: int,
    cot_input_jsonl: str | Path,
) -> List[Dict[str, Any]]:
    base_cases = [
        case
        for case in build_local_copy_cases(
            token_count=0,
            phrase_count=count,
            cot_count=0,
            cot_input_jsonl=cot_input_jsonl,
            token_max_new_tokens=0,
            phrase_max_new_tokens=max_new_tokens,
            cot_max_new_tokens=0,
            seed=seed,
            variant="sharp_prev1",
        )
        if str(case.get("family") or "") == "phrase"
    ]
    out: List[Dict[str, Any]] = []
    for case in base_cases:
        metadata = dict(case.get("metadata") or {})
        wrapper = str(metadata.get("wrapper") or "{tail}")
        state_to_tail = {
            "local": str(case.get("assistant_prefix") or ""),
            "matched_control": wrapper.format(tail=str(metadata.get("matched_control_tail") or metadata.get("base_prefix") or "")),
            "position_shift": wrapper.format(tail=str(metadata.get("position_shift_tail") or metadata.get("tail") or "")),
        }
        for state in PHRASE_STATE_ORDER:
            derived = dict(case)
            derived["state"] = state
            derived["original_example_id"] = str(case.get("example_id") or "")
            derived["example_id"] = f"{case.get('example_id')}_{state}"
            derived["assistant_prefix"] = state_to_tail[state]
            derived_metadata = dict(metadata)
            derived_metadata["state"] = state
            derived["metadata"] = derived_metadata
            out.append(derived)
    return out


def aggregate_phrase_mechanism_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    condition_order: Sequence[str],
    state_order: Sequence[str] = PHRASE_STATE_ORDER,
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("state") or ""), str(row.get("condition") or ""))].append(dict(row))

    summary_rows: List[Dict[str, Any]] = []
    for state in state_order:
        for condition in condition_order:
            state_rows = grouped.get((state, condition), [])
            if not state_rows:
                continue
            summary_rows.append(
                {
                    "state": state,
                    "condition": condition,
                    "examples": int(len(state_rows)),
                    "target_match_rate": round(
                        _mean(1.0 if bool(row.get("realized_target_match")) else 0.0 for row in state_rows),
                        6,
                    ),
                    "target_prob_mean": round(_mean(float(row.get("target_prob", 0.0)) for row in state_rows), 6),
                    "target_logit_mean": round(_mean(float(row.get("target_logit", 0.0)) for row in state_rows), 6),
                    "target_rank_mean": round(_mean(float(row.get("target_rank", 0.0)) for row in state_rows), 6),
                    "control_prob_mean": round(_mean(float(row.get("control_prob_mean", 0.0)) for row in state_rows), 6),
                    "control_logit_mean": round(_mean(float(row.get("control_logit_mean", 0.0)) for row in state_rows), 6),
                    "target_minus_control_prob_mean": round(
                        _mean(float(row.get("target_minus_control_prob", 0.0)) for row in state_rows),
                        6,
                    ),
                    "target_minus_control_logit_mean": round(
                        _mean(float(row.get("target_minus_control_logit", 0.0)) for row in state_rows),
                        6,
                    ),
                    "target_prob_delta_mean": round(
                        _mean(float(row.get("target_prob_delta", 0.0)) for row in state_rows if row.get("target_prob_delta") is not None),
                        6,
                    ),
                    "target_logit_delta_mean": round(
                        _mean(float(row.get("target_logit_delta", 0.0)) for row in state_rows if row.get("target_logit_delta") is not None),
                        6,
                    ),
                    "control_prob_delta_mean": round(
                        _mean(float(row.get("control_prob_delta_mean", 0.0)) for row in state_rows if row.get("control_prob_delta_mean") is not None),
                        6,
                    ),
                    "control_logit_delta_mean": round(
                        _mean(float(row.get("control_logit_delta_mean", 0.0)) for row in state_rows if row.get("control_logit_delta_mean") is not None),
                        6,
                    ),
                    "target_minus_control_logit_delta_mean": round(
                        _mean(
                            float(row.get("target_minus_control_logit_delta", 0.0))
                            for row in state_rows
                            if row.get("target_minus_control_logit_delta") is not None
                        ),
                        6,
                    ),
                    "direct_write_target_logit_mean": round(
                        _mean(
                            float(row.get("direct_write_target_logit", 0.0))
                            for row in state_rows
                            if row.get("direct_write_target_logit") is not None
                        ),
                        6,
                    ),
                    "direct_write_control_logit_mean": round(
                        _mean(
                            float(row.get("direct_write_control_logit_mean", 0.0))
                            for row in state_rows
                            if row.get("direct_write_control_logit_mean") is not None
                        ),
                        6,
                    ),
                    "direct_write_target_minus_control_mean": round(
                        _mean(
                            float(row.get("direct_write_target_minus_control", 0.0))
                            for row in state_rows
                            if row.get("direct_write_target_minus_control") is not None
                        ),
                        6,
                    ),
                }
            )
    return summary_rows


def build_phrase_mechanism_summary_json(
    *,
    summary_rows: Sequence[Dict[str, Any]],
    condition_order: Sequence[str],
    state_order: Sequence[str] = PHRASE_STATE_ORDER,
) -> Dict[str, Any]:
    nested: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for row in summary_rows:
        nested[str(row.get("state") or "")][str(row.get("condition") or "")] = dict(row)
    return {
        "state_order": list(state_order),
        "condition_order": list(condition_order),
        "states": nested,
    }


def write_phrase_mechanism_report(
    path: str | Path,
    *,
    args: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    condition_order: Sequence[str],
    state_order: Sequence[str] = PHRASE_STATE_ORDER,
) -> None:
    rows_by_state: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        rows_by_state[str(row.get("state") or "")][str(row.get("condition") or "")] = dict(row)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Sharp Phrase Copy Mechanism\n\n")
        f.write(f"- model: `{args.get('model_name_or_path')}`\n")
        f.write(f"- head_label: `{args.get('head_label')}`\n")
        f.write(f"- phrase_count: `{args.get('phrase_count')}`\n")
        f.write(f"- scale_values: `{args.get('scale_values')}`\n\n")

        for state in state_order:
            state_rows = rows_by_state.get(state)
            if not state_rows:
                continue
            f.write(f"## {state}\n\n")
            for condition in condition_order:
                row = state_rows.get(condition)
                if not row:
                    continue
                parts = [
                    f"condition=`{condition}`",
                    f"target_match_rate={float(row.get('target_match_rate', 0.0)):.4f}",
                    f"target_prob_mean={float(row.get('target_prob_mean', 0.0)):.4f}",
                    f"target_logit_mean={float(row.get('target_logit_mean', 0.0)):.4f}",
                    f"control_prob_mean={float(row.get('control_prob_mean', 0.0)):.4f}",
                    f"target_minus_control_logit_mean={float(row.get('target_minus_control_logit_mean', 0.0)):.4f}",
                    f"target_logit_delta_mean={float(row.get('target_logit_delta_mean', 0.0)):.4f}",
                    f"direct_write_target_logit_mean={float(row.get('direct_write_target_logit_mean', 0.0)):.4f}",
                    f"direct_write_target_minus_control_mean={float(row.get('direct_write_target_minus_control_mean', 0.0)):.4f}",
                ]
                f.write("- " + ", ".join(parts) + "\n")
            f.write("\n")
