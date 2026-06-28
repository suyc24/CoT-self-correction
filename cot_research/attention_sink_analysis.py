from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class SinkConfig:
    prefix_token_count: int = 4
    sink_token_texts: List[str] | None = None
    late_query_start: int = -1
    sink_mass_threshold: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if data["sink_token_texts"] is None:
            data["sink_token_texts"] = []
        return data


def parse_comma_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def find_subsequence_positions(sequence: Sequence[int], pattern: Sequence[int]) -> List[int]:
    if not sequence or not pattern or len(pattern) > len(sequence):
        return []
    hits: List[int] = []
    pat_len = len(pattern)
    for start in range(0, len(sequence) - pat_len + 1):
        if list(sequence[start : start + pat_len]) == list(pattern):
            hits.extend(range(start, start + pat_len))
    return hits


def resolve_sink_positions(
    token_ids: Sequence[int],
    tokenizer: Any,
    config: SinkConfig,
) -> Dict[str, Any]:
    key_count = len(token_ids)
    prefix_positions = list(range(min(max(config.prefix_token_count, 0), key_count)))

    matched_token_text_positions: Dict[str, List[int]] = {}
    sink_positions: Set[int] = set(prefix_positions)
    for sink_text in config.sink_token_texts or []:
        pattern = tokenizer.encode(sink_text, add_special_tokens=False)
        if not pattern:
            matched_token_text_positions[sink_text] = []
            continue
        hit_positions = find_subsequence_positions(token_ids, pattern)
        matched_token_text_positions[sink_text] = hit_positions
        sink_positions.update(hit_positions)

    sorted_positions = sorted(sink_positions)
    if config.late_query_start >= 0:
        late_query_start = int(config.late_query_start)
    elif sorted_positions:
        late_query_start = max(sorted_positions) + 1
    else:
        late_query_start = 0

    return {
        "sink_positions": sorted_positions,
        "prefix_positions": prefix_positions,
        "matched_token_text_positions": matched_token_text_positions,
        "sink_token_count": len(sorted_positions),
        "late_query_start": late_query_start,
        "key_count": key_count,
    }


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def compute_head_sink_metrics(
    *,
    attentions: Sequence[Any],
    sink_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sink_positions = list(sink_info.get("sink_positions") or [])
    prefix_positions = list(sink_info.get("prefix_positions") or [])
    late_query_start = int(sink_info.get("late_query_start") or 0)
    sink_token_count = int(sink_info.get("sink_token_count") or 0)
    key_count = int(sink_info.get("key_count") or 0)
    threshold = float(sink_info.get("sink_mass_threshold") or 0.0)

    metrics: List[Dict[str, Any]] = []
    for layer_idx, layer_attn in enumerate(attentions):
        attn = layer_attn[0].detach().float().cpu()
        num_heads = int(attn.shape[0])
        query_count = int(attn.shape[1])
        key_count = int(attn.shape[2])
        late_start = min(max(late_query_start, 0), max(query_count - 1, 0)) if query_count > 0 else 0
        late_queries = list(range(late_start, query_count)) if query_count > 0 else []
        if not late_queries and query_count > 0:
            late_queries = list(range(query_count))

        for head_idx in range(num_heads):
            head_attn = attn[head_idx]
            sink_mass_all = head_attn[:, sink_positions].sum(dim=-1) if sink_positions else head_attn.new_zeros((query_count,))
            sink_mass_late = sink_mass_all[late_queries] if late_queries else sink_mass_all
            prefix_mass_late = (
                head_attn[late_queries][:, prefix_positions].sum(dim=-1)
                if late_queries and prefix_positions
                else head_attn.new_zeros((len(late_queries),))
            )
            pos0_mass_late = head_attn[late_queries, 0] if late_queries and key_count > 0 else head_attn.new_zeros((len(late_queries),))
            top1_indices = head_attn[late_queries].argmax(dim=-1).tolist() if late_queries else []
            top1_sink_hits = sum(1 for idx in top1_indices if idx in sink_positions)
            sink_over_threshold = sum(1 for value in sink_mass_late.tolist() if float(value) >= threshold)
            uniform_sink_mass = (sink_token_count / key_count) if key_count > 0 else 0.0
            mean_sink_mass_late = _safe_mean([float(x) for x in sink_mass_late.tolist()])
            mean_prefix_mass_late = _safe_mean([float(x) for x in prefix_mass_late.tolist()])
            mean_pos0_mass_late = _safe_mean([float(x) for x in pos0_mass_late.tolist()])
            metrics.append(
                {
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    "head_label": f"L{layer_idx}H{head_idx}",
                    "query_count": query_count,
                    "query_count_late": len(late_queries),
                    "key_count": key_count,
                    "sink_token_count": sink_token_count,
                    "mean_sink_mass_all_queries": _safe_mean([float(x) for x in sink_mass_all.tolist()]),
                    "mean_sink_mass_late_queries": mean_sink_mass_late,
                    "mean_prefix_mass_late_queries": mean_prefix_mass_late,
                    "mean_pos0_mass_late_queries": mean_pos0_mass_late,
                    "max_sink_mass_late_queries": 0.0 if not len(sink_mass_late) else float(max(sink_mass_late.tolist())),
                    "top1_sink_rate_late_queries": 0.0 if not late_queries else float(top1_sink_hits / len(late_queries)),
                    "sink_mass_over_threshold_rate_late_queries": 0.0 if not late_queries else float(sink_over_threshold / len(late_queries)),
                    "uniform_sink_mass": uniform_sink_mass,
                    "sink_mass_ratio_late_queries": 0.0 if uniform_sink_mass <= 0 else float(mean_sink_mass_late / uniform_sink_mass),
                }
            )
    return metrics


def annotate_example_head_ranks(
    rows: Sequence[Dict[str, Any]],
    *,
    metric_key: str = "sink_mass_ratio_late_queries",
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("example_id") or "unknown"), []).append(dict(row))

    ranked_rows: List[Dict[str, Any]] = []
    for example_rows in grouped.values():
        sorted_global = sorted(
            example_rows,
            key=lambda item: (
                -float(item.get(metric_key, 0.0)),
                int(item.get("layer_idx", 0)),
                int(item.get("head_idx", 0)),
            ),
        )
        global_rank_lookup = {
            (int(item.get("layer_idx", 0)), int(item.get("head_idx", 0))): rank
            for rank, item in enumerate(sorted_global, start=1)
        }

        by_layer: Dict[int, List[Dict[str, Any]]] = {}
        for item in example_rows:
            by_layer.setdefault(int(item.get("layer_idx", 0)), []).append(item)
        layer_rank_lookup: Dict[Tuple[int, int], int] = {}
        for layer_idx, layer_rows in by_layer.items():
            sorted_layer = sorted(
                layer_rows,
                key=lambda item: (
                    -float(item.get(metric_key, 0.0)),
                    int(item.get("head_idx", 0)),
                ),
            )
            for rank, item in enumerate(sorted_layer, start=1):
                layer_rank_lookup[(layer_idx, int(item.get("head_idx", 0)))] = rank

        for item in example_rows:
            layer_idx = int(item.get("layer_idx", 0))
            head_idx = int(item.get("head_idx", 0))
            key = (layer_idx, head_idx)
            item["example_global_rank_by_sink_mass_ratio"] = int(global_rank_lookup[key])
            item["example_layer_rank_by_sink_mass_ratio"] = int(layer_rank_lookup[key])
            ranked_rows.append(item)
    return ranked_rows


def aggregate_head_metrics(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["head_label"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for head_label, group in sorted(grouped.items()):
        layer_idx = int(group[0]["layer_idx"])
        head_idx = int(group[0]["head_idx"])
        summary_rows.append(
            {
                "head_label": head_label,
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "example_count": len(group),
                "mean_sink_mass_late_queries": _safe_mean([float(item["mean_sink_mass_late_queries"]) for item in group]),
                "mean_sink_mass_ratio_late_queries": _safe_mean([float(item["sink_mass_ratio_late_queries"]) for item in group]),
                "mean_prefix_mass_late_queries": _safe_mean([float(item["mean_prefix_mass_late_queries"]) for item in group]),
                "mean_pos0_mass_late_queries": _safe_mean([float(item["mean_pos0_mass_late_queries"]) for item in group]),
                "mean_top1_sink_rate_late_queries": _safe_mean([float(item["top1_sink_rate_late_queries"]) for item in group]),
                "mean_sink_mass_over_threshold_rate_late_queries": _safe_mean([float(item["sink_mass_over_threshold_rate_late_queries"]) for item in group]),
                "mean_uniform_sink_mass": _safe_mean([float(item["uniform_sink_mass"]) for item in group]),
                "mean_sink_token_count": _safe_mean([float(item["sink_token_count"]) for item in group]),
                "mean_query_count_late": _safe_mean([float(item["query_count_late"]) for item in group]),
                "mean_example_global_rank_by_sink_mass_ratio": _safe_mean(
                    [float(item.get("example_global_rank_by_sink_mass_ratio", 0.0)) for item in group]
                ),
                "mean_example_layer_rank_by_sink_mass_ratio": _safe_mean(
                    [float(item.get("example_layer_rank_by_sink_mass_ratio", 0.0)) for item in group]
                ),
                "example_global_top10_rate_by_sink_mass_ratio": _safe_mean(
                    [1.0 if int(item.get("example_global_rank_by_sink_mass_ratio", 10**9)) <= 10 else 0.0 for item in group]
                ),
                "example_global_top20_rate_by_sink_mass_ratio": _safe_mean(
                    [1.0 if int(item.get("example_global_rank_by_sink_mass_ratio", 10**9)) <= 20 else 0.0 for item in group]
                ),
                "example_layer_top1_rate_by_sink_mass_ratio": _safe_mean(
                    [1.0 if int(item.get("example_layer_rank_by_sink_mass_ratio", 10**9)) == 1 else 0.0 for item in group]
                ),
                "example_layer_top3_rate_by_sink_mass_ratio": _safe_mean(
                    [1.0 if int(item.get("example_layer_rank_by_sink_mass_ratio", 10**9)) <= 3 else 0.0 for item in group]
                ),
            }
        )

    summary_rows.sort(key=lambda row: float(row["mean_sink_mass_ratio_late_queries"]), reverse=True)
    for rank, row in enumerate(summary_rows, start=1):
        row["global_rank_by_sink_mass_ratio"] = rank

    rows_by_layer: Dict[int, List[Dict[str, Any]]] = {}
    for row in summary_rows:
        rows_by_layer.setdefault(int(row["layer_idx"]), []).append(row)
    for layer_rows in rows_by_layer.values():
        layer_rows.sort(key=lambda row: float(row["mean_sink_mass_ratio_late_queries"]), reverse=True)
        layer_mean = _safe_mean([float(item["mean_sink_mass_ratio_late_queries"]) for item in layer_rows])
        for rank, row in enumerate(layer_rows, start=1):
            row["layer_rank_by_sink_mass_ratio"] = rank
            row["layer_mean_sink_mass_ratio"] = layer_mean
            row["delta_vs_layer_mean_sink_mass_ratio"] = float(row["mean_sink_mass_ratio_late_queries"]) - layer_mean
    summary_rows.sort(key=lambda row: int(row["global_rank_by_sink_mass_ratio"]))
    return summary_rows
