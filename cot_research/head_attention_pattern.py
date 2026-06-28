from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple

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


def compute_attention_entropy(attention: torch.Tensor) -> float:
    probs = attention.clamp_min(1e-20)
    return float((-(probs * probs.log()).sum()).item())


def build_distance_bucket_spec() -> List[Tuple[str, int, int | None]]:
    return [
        ("self", 0, 0),
        ("prev_1", 1, 1),
        ("prev_2_4", 2, 4),
        ("prev_5_8", 5, 8),
        ("prev_9_16", 9, 16),
        ("prev_17_32", 17, 32),
        ("prev_33_64", 33, 64),
        ("prev_65_128", 65, 128),
        ("prev_129_256", 129, 256),
        ("prev_257_plus", 257, None),
    ]


def _sum_relative_window(attention: torch.Tensor, start_distance: int, end_distance: int | None) -> float:
    query_pos = attention.numel() - 1
    total = 0.0
    max_distance = query_pos
    if end_distance is None:
        end_distance = max_distance
    for distance in range(start_distance, min(end_distance, max_distance) + 1):
        total += float(attention[query_pos - distance].item())
    return total


def compute_head_pattern_metrics(
    attention: torch.Tensor,
    *,
    cumulative_windows: Sequence[int],
    top_k: int,
) -> Dict[str, Any]:
    query_pos = attention.numel() - 1
    metrics: Dict[str, Any] = {
        "attention_entropy": compute_attention_entropy(attention),
        "attention_l2_norm": float(torch.linalg.vector_norm(attention, ord=2).item()),
        "attention_max_value": float(attention.max().item()),
        "window_token_count": int(attention.numel()),
        "query_index_in_window": int(query_pos),
    }
    for window in cumulative_windows:
        metrics[f"prev_mass_w{window}"] = _sum_relative_window(attention, 1, window)

    for bucket_name, start_distance, end_distance in build_distance_bucket_spec():
        metrics[f"bucket_{bucket_name}"] = _sum_relative_window(attention, start_distance, end_distance)

    top_values, top_indices = torch.topk(attention, k=min(top_k, attention.numel()))
    top_positions: List[Dict[str, Any]] = []
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        rel_distance = query_pos - int(index)
        top_positions.append(
            {
                "index_in_window": int(index),
                "relative_distance": int(rel_distance),
                "attention": float(value),
            }
        )
    metrics["top_positions"] = top_positions
    return metrics


def compute_all_head_pattern_metrics(
    *,
    attentions: Sequence[Any],
    query_index: int,
    cumulative_windows: Sequence[int],
    top_k: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for layer_idx, layer_attn in enumerate(attentions):
        attn = layer_attn[0].detach().float().cpu()
        num_heads = int(attn.shape[0])
        query_count = int(attn.shape[1])
        key_count = int(attn.shape[2])
        if query_count <= 0:
            raise ValueError(f"Layer {layer_idx} returned attention with zero query positions.")
        query_index_in_tensor = query_count - 1
        for head_idx in range(num_heads):
            attention_vector = attn[head_idx, query_index_in_tensor]
            metrics = compute_head_pattern_metrics(
                attention_vector,
                cumulative_windows=cumulative_windows,
                top_k=top_k,
            )
            top_positions = list(metrics.pop("top_positions"))
            top_relative_distance = int(top_positions[0]["relative_distance"]) if top_positions else None
            top_nonself_relative_distance = next(
                (int(item["relative_distance"]) for item in top_positions if int(item["relative_distance"]) != 0),
                None,
            )
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    "head_label": f"L{layer_idx}H{head_idx}",
                    "query_index": int(query_index),
                    "query_index_in_tensor": int(query_index_in_tensor),
                    "query_count": query_count,
                    "key_count": key_count,
                    "top_relative_distance": top_relative_distance,
                    "top_nonself_relative_distance": top_nonself_relative_distance,
                    "is_top_prev_1": 1.0 if top_relative_distance == 1 else 0.0,
                    "is_top_nonself_prev_1": 1.0 if top_nonself_relative_distance == 1 else 0.0,
                    "top_positions": top_positions,
                    **metrics,
                }
            )
    return rows


def aggregate_pattern_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    cumulative_windows: Sequence[int],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["head_label"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for head_label, group in sorted(grouped.items(), key=lambda item: (int(item[1][0]["layer_idx"]), int(item[1][0]["head_idx"]))):
        first = group[0]
        row: Dict[str, Any] = {
            "head_label": head_label,
            "layer_idx": int(first["layer_idx"]),
            "head_idx": int(first["head_idx"]),
            "example_count": len(group),
            "mean_attention_entropy": _mean([float(item["attention_entropy"]) for item in group]),
            "mean_attention_l2_norm": _mean([float(item["attention_l2_norm"]) for item in group]),
            "mean_attention_max_value": _mean([float(item["attention_max_value"]) for item in group]),
        }
        for window in cumulative_windows:
            row[f"mean_prev_mass_w{window}"] = _mean([float(item.get(f"prev_mass_w{window}", 0.0)) for item in group])
        for bucket_name, _, _ in build_distance_bucket_spec():
            row[f"mean_bucket_{bucket_name}"] = _mean([float(item.get(f"bucket_{bucket_name}", 0.0)) for item in group])
        if any("is_top_prev_1" in item for item in group):
            row["prev_1_top1_rate"] = _mean([float(item.get("is_top_prev_1", 0.0)) for item in group])
        if any("is_top_nonself_prev_1" in item for item in group):
            row["prev_1_top_nonself_rate"] = _mean([float(item.get("is_top_nonself_prev_1", 0.0)) for item in group])
        summary_rows.append(row)

    for window in cumulative_windows:
        metric_key = f"mean_prev_mass_w{window}"
        ranked = sorted(summary_rows, key=lambda item: -float(item.get(metric_key, 0.0)))
        for rank, item in enumerate(ranked, start=1):
            item[f"global_rank_by_{metric_key}"] = rank

    for metric_key in ["prev_1_top1_rate", "prev_1_top_nonself_rate"]:
        if any(metric_key in item for item in summary_rows):
            ranked = sorted(summary_rows, key=lambda item: -float(item.get(metric_key, 0.0)))
            for rank, item in enumerate(ranked, start=1):
                item[f"global_rank_by_{metric_key}"] = rank

    return sorted(summary_rows, key=lambda item: (int(item["layer_idx"]), int(item["head_idx"])))


def aggregate_distance_profile(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["head_label"]), []).append(row)

    profile_rows: List[Dict[str, Any]] = []
    for head_label, group in sorted(grouped.items(), key=lambda item: (int(item[1][0]["layer_idx"]), int(item[1][0]["head_idx"]))):
        for bucket_name, _, _ in build_distance_bucket_spec():
            profile_rows.append(
                {
                    "head_label": head_label,
                    "layer_idx": int(group[0]["layer_idx"]),
                    "head_idx": int(group[0]["head_idx"]),
                    "distance_bucket": bucket_name,
                    "mean_mass": _mean([float(item.get(f"bucket_{bucket_name}", 0.0)) for item in group]),
                }
            )
    return profile_rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))
