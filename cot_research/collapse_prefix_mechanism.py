from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .repetition_analysis import analyze_repetition, longest_same_token_run


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def decode_token(tokenizer, token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def display_token_text(text: str) -> str:
    if text == "":
        return "<empty>"
    return text.encode("unicode_escape").decode("ascii")


def identify_same_token_collapse(
    token_ids: Sequence[int],
    *,
    same_token_run_threshold: int,
) -> Optional[Dict[str, Any]]:
    same_run = longest_same_token_run(token_ids)
    length = int(same_run.get("length") or 0)
    if length < int(same_token_run_threshold):
        return None
    start = int(same_run.get("start") or 0)
    end = int(same_run.get("end") or 0)
    token_id = same_run.get("token_id")
    if token_id is None or start < 0 or end <= start or end > len(token_ids):
        return None
    return {
        "collapse_type": "same_token_run",
        "collapse_step_idx": int(start),
        "loop_token_id": int(token_id),
        "loop_run_length": int(length),
        "loop_run_start": int(start),
        "loop_run_end": int(end),
    }


def build_short_rollout_metrics(
    *,
    continuation: str,
    token_ids: Sequence[int],
    thresholds,
) -> Dict[str, Any]:
    token_ids = [int(token_id) for token_id in token_ids]
    repetition = analyze_repetition(continuation, token_ids=token_ids, thresholds=thresholds)
    same_run = longest_same_token_run(token_ids)
    first_token_id = None if not token_ids else int(token_ids[0])
    return {
        "generated_tokens": int(len(token_ids)),
        "first_token_id": first_token_id,
        "repetition_detection": repetition,
        "longest_same_token_run": same_run,
    }


def extract_topk_token_rows(
    logits_row,
    tokenizer,
    *,
    k: int,
) -> List[Dict[str, Any]]:
    import torch

    logits_row = logits_row.float()
    log_norm = torch.logsumexp(logits_row, dim=-1)
    top_values, top_indices = torch.topk(logits_row, k=min(int(k), int(logits_row.numel())))
    rows: List[Dict[str, Any]] = []
    for rank_idx, (logit, token_id) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
        prob = float(torch.exp(logits_row[int(token_id)] - log_norm).item())
        rows.append(
            {
                "rank": int(rank_idx),
                "token_id": int(token_id),
                "token_text": decode_token(tokenizer, int(token_id)),
                "logit": float(logit),
                "prob": float(prob),
            }
        )
    return rows


def first_non_loop_top_token_id(
    top_rows: Sequence[Dict[str, Any]],
    *,
    loop_token_id: Optional[int],
) -> Optional[int]:
    loop_token_id = None if loop_token_id is None else int(loop_token_id)
    for row in top_rows:
        token_id = row.get("token_id")
        if token_id is None:
            continue
        if loop_token_id is not None and int(token_id) == loop_token_id:
            continue
        return int(token_id)
    return None


def aggregate_case_rows(case_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    collapse_rows = [dict(row) for row in case_rows if bool(row.get("has_collapse"))]
    baseline_ref_rows = [row for row in collapse_rows if row.get("baseline_same_step_token_id") is not None]

    def _bool_rate(rows: Sequence[Dict[str, Any]], key: str) -> float:
        if not rows:
            return 0.0
        return round(sum(1.0 for row in rows if bool(row.get(key))) / len(rows), 6)

    def _num_mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
        values = [float(row.get(key, 0.0)) for row in rows if row.get(key) is not None]
        return round(_mean(values), 6)

    promoted_counter: Counter[str] = Counter()
    demoted_counter: Counter[str] = Counter()
    loop_token_counter: Counter[str] = Counter()
    for row in collapse_rows:
        promoted = row.get("identity_top1_token_text_display")
        demoted = row.get("zero_top1_token_text_display")
        loop_token = row.get("loop_token_text_display")
        if promoted:
            promoted_counter.update([str(promoted)])
        if demoted:
            demoted_counter.update([str(demoted)])
        if loop_token:
            loop_token_counter.update([str(loop_token)])

    short_conditions = ["identity", "zero", "scale"]
    short_rollout_summary: Dict[str, Dict[str, Any]] = {}
    for condition in short_conditions:
        prefix = f"{condition}_short_"
        short_rollout_summary[condition] = {
            "first_token_is_loop_rate": _bool_rate(collapse_rows, prefix + "first_token_is_loop"),
            "first_token_matches_escape_rate": _bool_rate(collapse_rows, prefix + "first_token_matches_escape"),
            "repetition_hit_rate": _bool_rate(collapse_rows, prefix + "repetition_matched"),
            "mean_longest_same_token_run": _num_mean(collapse_rows, prefix + "longest_same_token_run"),
        }

    return {
        "processed_examples": int(len(case_rows)),
        "collapse_examples": int(len(collapse_rows)),
        "collapse_rate_over_processed": 0.0 if not case_rows else round(len(collapse_rows) / len(case_rows), 6),
        "baseline_same_step_available_examples": int(len(baseline_ref_rows)),
        "identity_top1_is_loop_rate": _bool_rate(collapse_rows, "identity_top1_is_loop"),
        "zero_top1_is_loop_rate": _bool_rate(collapse_rows, "zero_top1_is_loop"),
        "scale_top1_is_loop_rate": _bool_rate(collapse_rows, "scale_top1_is_loop"),
        "identity_top1_matches_baseline_step_rate": _bool_rate(
            baseline_ref_rows,
            "identity_top1_matches_baseline_step",
        ),
        "zero_top1_matches_baseline_step_rate": _bool_rate(
            baseline_ref_rows,
            "zero_top1_matches_baseline_step",
        ),
        "scale_top1_matches_baseline_step_rate": _bool_rate(
            baseline_ref_rows,
            "scale_top1_matches_baseline_step",
        ),
        "escape_token_from_baseline_rate": _bool_rate(collapse_rows, "escape_token_from_baseline"),
        "identity_minus_zero_loop_prob_mean": _num_mean(collapse_rows, "identity_minus_zero_loop_prob"),
        "scale_minus_zero_loop_prob_mean": _num_mean(collapse_rows, "scale_minus_zero_loop_prob"),
        "identity_minus_zero_loop_logit_mean": _num_mean(collapse_rows, "identity_minus_zero_loop_logit"),
        "scale_minus_zero_loop_logit_mean": _num_mean(collapse_rows, "scale_minus_zero_loop_logit"),
        "identity_minus_zero_escape_prob_mean": _num_mean(collapse_rows, "identity_minus_zero_escape_prob"),
        "scale_minus_zero_escape_prob_mean": _num_mean(collapse_rows, "scale_minus_zero_escape_prob"),
        "identity_minus_zero_escape_logit_mean": _num_mean(collapse_rows, "identity_minus_zero_escape_logit"),
        "scale_minus_zero_escape_logit_mean": _num_mean(collapse_rows, "scale_minus_zero_escape_logit"),
        "identity_escape_minus_loop_margin_mean": _num_mean(collapse_rows, "identity_escape_minus_loop_margin"),
        "zero_escape_minus_loop_margin_mean": _num_mean(collapse_rows, "zero_escape_minus_loop_margin"),
        "scale_escape_minus_loop_margin_mean": _num_mean(collapse_rows, "scale_escape_minus_loop_margin"),
        "identity_direct_write_loop_mean": _num_mean(collapse_rows, "identity_direct_write_loop_logit"),
        "identity_direct_write_escape_mean": _num_mean(collapse_rows, "identity_direct_write_escape_logit"),
        "identity_direct_write_escape_minus_loop_mean": _num_mean(
            collapse_rows,
            "identity_direct_write_escape_minus_loop",
        ),
        "scale_direct_write_loop_mean": _num_mean(collapse_rows, "scale_direct_write_loop_logit"),
        "scale_direct_write_escape_mean": _num_mean(collapse_rows, "scale_direct_write_escape_logit"),
        "scale_direct_write_escape_minus_loop_mean": _num_mean(
            collapse_rows,
            "scale_direct_write_escape_minus_loop",
        ),
        "identity_direct_write_prefers_escape_rate": _bool_rate(
            collapse_rows,
            "identity_direct_write_prefers_escape",
        ),
        "scale_direct_write_prefers_escape_rate": _bool_rate(
            collapse_rows,
            "scale_direct_write_prefers_escape",
        ),
        "short_rollout": short_rollout_summary,
        "top_identity_top1_tokens": [
            {"token": token, "count": int(count)}
            for token, count in promoted_counter.most_common(10)
        ],
        "top_zero_top1_tokens": [
            {"token": token, "count": int(count)}
            for token, count in demoted_counter.most_common(10)
        ],
        "top_loop_tokens": [
            {"token": token, "count": int(count)}
            for token, count in loop_token_counter.most_common(10)
        ],
    }


def write_report(
    path: str | Path,
    *,
    args: Dict[str, Any],
    summary: Dict[str, Any],
) -> None:
    short_rollout = dict(summary.get("short_rollout") or {})
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Collapse Prefix Mechanism\n\n")
        f.write(f"- model: `{args.get('model_name_or_path')}`\n")
        f.write(f"- head_label: `{args.get('head_label')}`\n")
        f.write(f"- max_examples: `{args.get('max_examples')}`\n")
        f.write(f"- same_token_run_threshold: `{args.get('same_token_run_threshold')}`\n")
        f.write(f"- short_rollout_tokens: `{args.get('short_rollout_tokens')}`\n\n")

        fields = [
            "collapse_examples",
            "collapse_rate_over_processed",
            "identity_top1_is_loop_rate",
            "zero_top1_is_loop_rate",
            "scale_top1_is_loop_rate",
            "identity_top1_matches_baseline_step_rate",
            "identity_minus_zero_loop_prob_mean",
            "identity_minus_zero_escape_prob_mean",
            "identity_escape_minus_loop_margin_mean",
            "identity_direct_write_escape_minus_loop_mean",
            "identity_direct_write_prefers_escape_rate",
        ]
        f.write("## Main Summary\n\n")
        for key in fields:
            value = summary.get(key)
            if isinstance(value, float):
                f.write(f"- {key}: `{value:.6f}`\n")
            else:
                f.write(f"- {key}: `{value}`\n")

        f.write("\n## Short Rollout\n\n")
        for condition in ["identity", "zero", "scale"]:
            row = dict(short_rollout.get(condition) or {})
            f.write(
                "- "
                + ", ".join(
                    [
                        f"condition=`{condition}`",
                        f"first_token_is_loop_rate={float(row.get('first_token_is_loop_rate', 0.0)):.6f}",
                        f"repetition_hit_rate={float(row.get('repetition_hit_rate', 0.0)):.6f}",
                        f"mean_longest_same_token_run={float(row.get('mean_longest_same_token_run', 0.0)):.6f}",
                    ]
                )
                + "\n"
            )
