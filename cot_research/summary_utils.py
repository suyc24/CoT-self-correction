from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .io_utils import write_csv, write_json

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def summarize_condition_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition_label"])].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for condition_label, group in sorted(grouped.items()):
        total = len(group)
        outcomes = defaultdict(int)
        keyword_hits = 0
        keyword_counts: List[float] = []
        repair_scores: List[float] = []
        generated_tokens: List[float] = []
        for row in group:
            outcomes[str(row.get("outcome", "other_answer"))] += 1
            keyword_stats = row.get("continuation_keyword_stats", {})
            if keyword_stats.get("hit"):
                keyword_hits += 1
            keyword_counts.append(float(keyword_stats.get("count", 0)))
            repair_scores.append(float(row.get("repair_signal_stats", {}).get("score", 0.0)))
            generated_tokens.append(float(row.get("generated_tokens", 0)))
        summary_rows.append(
            {
                "condition_label": condition_label,
                "intervention_kind": group[0].get("intervention_kind", "baseline"),
                "total": total,
                "corrected": outcomes["corrected"],
                "keep_wrong": outcomes["keep_wrong"],
                "other_answer": outcomes["other_answer"],
                "no_boxed_answer": outcomes["no_boxed_answer"],
                "corrected_rate": round(_rate(outcomes["corrected"], total), 6),
                "keep_wrong_rate": round(_rate(outcomes["keep_wrong"], total), 6),
                "keyword_hit_rate": round(_rate(keyword_hits, total), 6),
                "mean_keyword_count": round(_mean(keyword_counts), 6),
                "mean_repair_signal_score": round(_mean(repair_scores), 6),
                "mean_generated_tokens": round(_mean(generated_tokens), 6),
            }
        )
    return summary_rows


def summarize_token_targets(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for item in row.get("token_target_stats", []):
            grouped[(str(row["condition_label"]), str(item["target"]))].append(item)

    output: List[Dict[str, Any]] = []
    for (condition_label, target), items in sorted(grouped.items()):
        output.append(
            {
                "condition_label": condition_label,
                "target": target,
                "count": len(items),
                "mean_text_count_full": round(_mean([float(x.get("text_count_full", 0)) for x in items]), 6),
                "mean_text_count_think": round(_mean([float(x.get("text_count_think", 0)) for x in items]), 6),
                "mean_token_count_full": round(_mean([float(x.get("token_count_full", 0)) for x in items]), 6),
                "mean_token_count_think": round(_mean([float(x.get("token_count_think", 0)) for x in items]), 6),
                "hit_rate_text_think": round(_rate(sum(1 for x in items if int(x.get("text_count_think", 0)) > 0), len(items)), 6),
                "hit_rate_token_think": round(_rate(sum(1 for x in items if int(x.get("token_count_think", 0)) > 0), len(items)), 6),
            }
        )
    return output


def summarize_next_token_targets(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline_by_example: Dict[str, Dict[str, Dict[str, float]]] = {}
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        example_id = str(row["example_id"])
        next_stats = row.get("next_token_stats", {})
        candidate_map = next_stats.get("candidate_stats", {})
        if row.get("intervention_kind") == "baseline":
            baseline_by_example[example_id] = candidate_map

    for row in rows:
        example_id = str(row["example_id"])
        baseline_candidate_map = baseline_by_example.get(example_id, {})
        candidate_stats = row.get("next_token_stats", {}).get("candidate_stats", {})
        candidate_labels = row.get("next_token_stats", {}).get("candidate_labels", {})
        for token_id, stats in candidate_stats.items():
            label = str(candidate_labels.get(token_id, token_id))
            base = baseline_candidate_map.get(token_id, {})
            grouped[(str(row["condition_label"]), label)].append(
                {
                    "logit": float(stats.get("logit", 0.0)),
                    "prob": float(stats.get("prob", 0.0)),
                    "delta_logit": float(stats.get("logit", 0.0)) - float(base.get("logit", 0.0)),
                    "delta_prob": float(stats.get("prob", 0.0)) - float(base.get("prob", 0.0)),
                }
            )

    output: List[Dict[str, Any]] = []
    for (condition_label, label), items in sorted(grouped.items()):
        output.append(
            {
                "condition_label": condition_label,
                "target": label,
                "count": len(items),
                "mean_logit": round(_mean([item["logit"] for item in items]), 6),
                "mean_prob": round(_mean([item["prob"] for item in items]), 6),
                "mean_delta_logit_vs_baseline": round(_mean([item["delta_logit"] for item in items]), 6),
                "mean_delta_prob_vs_baseline": round(_mean([item["delta_prob"] for item in items]), 6),
            }
        )
    return output
