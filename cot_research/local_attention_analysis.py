from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    seen = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Expected positive integer, got {value}.")
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def find_last_subsequence_start(sequence: Sequence[int], pattern: Sequence[int]) -> Optional[int]:
    if not sequence or not pattern or len(pattern) > len(sequence):
        return None
    last_start: Optional[int] = None
    pat_len = len(pattern)
    for start in range(0, len(sequence) - pat_len + 1):
        if list(sequence[start : start + pat_len]) == list(pattern):
            last_start = start
    return last_start


def locate_close_think_query(
    *,
    prompt_token_ids: Sequence[int],
    continuation_token_ids: Sequence[int],
    close_think_token_ids: Sequence[int],
) -> Dict[str, int]:
    close_start_in_continuation = find_last_subsequence_start(continuation_token_ids, close_think_token_ids)
    if close_start_in_continuation is None:
        raise ValueError("Generated continuation does not contain the closing </think> token sequence.")
    query_index = len(prompt_token_ids) + close_start_in_continuation - 1
    if query_index < 0:
        raise ValueError("Close-think token starts at the beginning of the sequence; no preceding query token exists.")
    return {
        "close_start_in_continuation": int(close_start_in_continuation),
        "close_start_in_full_sequence": int(len(prompt_token_ids) + close_start_in_continuation),
        "close_token_count": int(len(close_think_token_ids)),
        "query_index_before_close": int(query_index),
        "full_sequence_length": int(len(prompt_token_ids) + len(continuation_token_ids)),
    }


def build_masked_attention_vector(attention_vector: torch.Tensor, mask_prefix_count: int) -> torch.Tensor:
    masked = attention_vector.clone()
    masked[: min(mask_prefix_count, masked.numel())] = 0
    total = float(masked.sum().item())
    if total > 0:
        masked = masked / total
    else:
        masked.zero_()
    return masked


def compute_query_attention_metrics(
    *,
    attentions: Sequence[Any],
    query_index: int,
    prefix_token_counts: Sequence[int],
    mask_prefix_token_counts: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for layer_idx, layer_attn in enumerate(attentions):
        attn = layer_attn[0].detach().float().cpu()
        num_heads = int(attn.shape[0])
        query_count = int(attn.shape[1])
        key_count = int(attn.shape[2])
        if query_index < 0 or query_index >= query_count:
            raise ValueError(
                f"Query index {query_index} is out of range for layer attention with query_count={query_count}."
            )
        for head_idx in range(num_heads):
            attention_vector = attn[head_idx, query_index]
            row: Dict[str, Any] = {
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "head_label": f"L{layer_idx}H{head_idx}",
                "query_index": int(query_index),
                "query_count": query_count,
                "key_count": key_count,
                "attention_l2_norm": float(torch.linalg.vector_norm(attention_vector, ord=2).item()),
                "attention_l1_norm": float(torch.linalg.vector_norm(attention_vector, ord=1).item()),
                "attention_max_value": float(attention_vector.max().item()),
            }
            for prefix_count in prefix_token_counts:
                clipped = min(prefix_count, key_count)
                prefix_mass = float(attention_vector[:clipped].sum().item()) if clipped > 0 else 0.0
                row[f"prefix_mass_k{prefix_count}"] = prefix_mass

            for mask_prefix_count in mask_prefix_token_counts:
                masked_vector = build_masked_attention_vector(attention_vector, mask_prefix_count)
                after_norm = float(torch.linalg.vector_norm(masked_vector, ord=2).item())
                shift_l2 = float(torch.linalg.vector_norm(masked_vector - attention_vector, ord=2).item())
                removed_mass = float(attention_vector[: min(mask_prefix_count, key_count)].sum().item())
                row[f"mask_norm_after_k{mask_prefix_count}"] = after_norm
                row[f"mask_norm_delta_k{mask_prefix_count}"] = after_norm - row["attention_l2_norm"]
                row[f"mask_shift_l2_k{mask_prefix_count}"] = shift_l2
                row[f"mask_removed_prefix_mass_k{mask_prefix_count}"] = removed_mass
            rows.append(row)
    return rows


def annotate_query_metric_ranks(
    rows: Sequence[Dict[str, Any]],
    *,
    prefix_token_counts: Sequence[int],
    mask_prefix_token_counts: Sequence[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("example_id") or "unknown"), []).append(dict(row))

    ranked_rows: List[Dict[str, Any]] = []
    for example_rows in grouped.values():
        for prefix_count in prefix_token_counts:
            metric_key = f"prefix_mass_k{prefix_count}"
            sorted_rows = sorted(
                example_rows,
                key=lambda item: (
                    -float(item.get(metric_key, 0.0)),
                    int(item.get("layer_idx", 0)),
                    int(item.get("head_idx", 0)),
                ),
            )
            for rank, item in enumerate(sorted_rows, start=1):
                item[f"rank_by_{metric_key}"] = rank

        for mask_prefix_count in mask_prefix_token_counts:
            metric_key = f"mask_shift_l2_k{mask_prefix_count}"
            sorted_rows = sorted(
                example_rows,
                key=lambda item: (
                    -float(item.get(metric_key, 0.0)),
                    int(item.get("layer_idx", 0)),
                    int(item.get("head_idx", 0)),
                ),
            )
            for rank, item in enumerate(sorted_rows, start=1):
                item[f"rank_by_{metric_key}"] = rank

        ranked_rows.extend(example_rows)
    return ranked_rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def aggregate_query_metrics(
    rows: Iterable[Dict[str, Any]],
    *,
    prefix_token_counts: Sequence[int],
    mask_prefix_token_counts: Sequence[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["head_label"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for head_label, group in sorted(
        grouped.items(),
        key=lambda item: (
            int(item[1][0].get("layer_idx", 0)),
            int(item[1][0].get("head_idx", 0)),
        ),
    ):
        first = group[0]
        row: Dict[str, Any] = {
            "head_label": head_label,
            "layer_idx": int(first["layer_idx"]),
            "head_idx": int(first["head_idx"]),
            "example_count": len(group),
            "mean_attention_l2_norm": _mean([float(item["attention_l2_norm"]) for item in group]),
            "mean_attention_max_value": _mean([float(item["attention_max_value"]) for item in group]),
        }
        for prefix_count in prefix_token_counts:
            metric_key = f"prefix_mass_k{prefix_count}"
            rank_key = f"rank_by_{metric_key}"
            row[f"mean_{metric_key}"] = _mean([float(item.get(metric_key, 0.0)) for item in group])
            row[f"mean_{rank_key}"] = _mean([float(item.get(rank_key, 0.0)) for item in group])
            row[f"top10_rate_{metric_key}"] = _mean(
                [1.0 if int(item.get(rank_key, 10**9)) <= 10 else 0.0 for item in group]
            )

        for mask_prefix_count in mask_prefix_token_counts:
            delta_key = f"mask_norm_delta_k{mask_prefix_count}"
            shift_key = f"mask_shift_l2_k{mask_prefix_count}"
            rank_key = f"rank_by_{shift_key}"
            removed_key = f"mask_removed_prefix_mass_k{mask_prefix_count}"
            row[f"mean_{delta_key}"] = _mean([float(item.get(delta_key, 0.0)) for item in group])
            row[f"mean_{shift_key}"] = _mean([float(item.get(shift_key, 0.0)) for item in group])
            row[f"mean_{removed_key}"] = _mean([float(item.get(removed_key, 0.0)) for item in group])
            row[f"mean_{rank_key}"] = _mean([float(item.get(rank_key, 0.0)) for item in group])
            row[f"top10_rate_{shift_key}"] = _mean(
                [1.0 if int(item.get(rank_key, 10**9)) <= 10 else 0.0 for item in group]
            )
        summary_rows.append(row)

    for prefix_count in prefix_token_counts:
        metric_key = f"mean_prefix_mass_k{prefix_count}"
        ranked = sorted(summary_rows, key=lambda item: -float(item.get(metric_key, 0.0)))
        for rank, item in enumerate(ranked, start=1):
            item[f"global_rank_by_{metric_key}"] = rank

    for mask_prefix_count in mask_prefix_token_counts:
        metric_key = f"mean_mask_shift_l2_k{mask_prefix_count}"
        ranked = sorted(summary_rows, key=lambda item: -float(item.get(metric_key, 0.0)))
        for rank, item in enumerate(ranked, start=1):
            item[f"global_rank_by_{metric_key}"] = rank

    summary_rows.sort(
        key=lambda item: (
            int(item.get("layer_idx", 0)),
            int(item.get("head_idx", 0)),
        )
    )
    return summary_rows
