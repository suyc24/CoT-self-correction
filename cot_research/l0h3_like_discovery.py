from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .io_utils import dump_jsonl, load_jsonl, write_csv, write_json


@dataclass(frozen=True)
class CandidateHeadScore:
    head_label: str
    layer_idx: int
    head_idx: int
    wait_anchor_count: int
    mean_no_wait_rate: float
    mean_wait_logit_drop: float
    max_wait_logit_drop: float
    prev_1_top_nonself_rate: float
    prev_1_top1_rate: float
    mean_prev_mass_w1: float
    wait_rank: int
    prev_rank: int
    combined_rank_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "head_label": self.head_label,
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "wait_anchor_count": self.wait_anchor_count,
            "mean_no_wait_rate": self.mean_no_wait_rate,
            "mean_wait_logit_drop": self.mean_wait_logit_drop,
            "max_wait_logit_drop": self.max_wait_logit_drop,
            "prev_1_top_nonself_rate": self.prev_1_top_nonself_rate,
            "prev_1_top1_rate": self.prev_1_top1_rate,
            "mean_prev_mass_w1": self.mean_prev_mass_w1,
            "wait_rank": self.wait_rank,
            "prev_rank": self.prev_rank,
            "combined_rank_score": self.combined_rank_score,
        }


def load_csv_rows(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_subset_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    dump_jsonl(path, list(rows))


def summarize_wait_ablation_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    layer_idx_filter: int = 0,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        layer_idx = int(row.get("layer_idx", 0))
        if layer_idx != layer_idx_filter:
            continue
        grouped.setdefault(str(row["head_label"]), []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for head_label, group in sorted(grouped.items(), key=lambda item: (int(item[1][0]["layer_idx"]), int(item[1][0]["head_idx"]))):
        mean_no_wait_rate = _mean(float(item.get("no_wait_rate", 0.0)) for item in group)
        logit_drops = [max(0.0, -float(item.get("mean_delta_ablated_minus_baseline", 0.0))) for item in group]
        first = group[0]
        summary_rows.append(
            {
                "head_label": head_label,
                "layer_idx": int(first["layer_idx"]),
                "head_idx": int(first["head_idx"]),
                "wait_anchor_count": int(sum(int(item.get("total_anchor_examples", 0)) for item in group)),
                "mean_no_wait_rate": mean_no_wait_rate,
                "mean_wait_logit_drop": _mean(logit_drops),
                "max_wait_logit_drop": max(logit_drops) if logit_drops else 0.0,
            }
        )

    ranked = sorted(
        summary_rows,
        key=lambda item: (
            -float(item["mean_no_wait_rate"]),
            -float(item["mean_wait_logit_drop"]),
            -float(item["max_wait_logit_drop"]),
            int(item["layer_idx"]),
            int(item["head_idx"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["wait_rank"] = rank
    return ranked


def rank_l0h3_like_candidates(
    *,
    wait_rows: Sequence[Dict[str, Any]],
    prev_rows: Sequence[Dict[str, Any]],
    layer_idx_filter: int = 0,
) -> List[CandidateHeadScore]:
    wait_summary = summarize_wait_ablation_candidates(wait_rows, layer_idx_filter=layer_idx_filter)
    prev_lookup = {
        str(row["head_label"]): row
        for row in prev_rows
        if int(row.get("layer_idx", 0)) == layer_idx_filter
    }

    prev_ranked = sorted(
        prev_lookup.values(),
        key=lambda item: (
            -float(item.get("prev_1_top_nonself_rate", 0.0)),
            -float(item.get("prev_1_top1_rate", 0.0)),
            -float(item.get("mean_prev_mass_w1", 0.0)),
            int(item.get("layer_idx", 0)),
            int(item.get("head_idx", 0)),
        ),
    )
    prev_rank_lookup = {str(row["head_label"]): rank for rank, row in enumerate(prev_ranked, start=1)}

    candidates: List[CandidateHeadScore] = []
    for wait_row in wait_summary:
        head_label = str(wait_row["head_label"])
        prev_row = prev_lookup.get(head_label)
        if prev_row is None:
            continue
        wait_rank = int(wait_row.get("wait_rank", 10**9))
        prev_rank = int(prev_rank_lookup.get(head_label, 10**9))
        candidates.append(
            CandidateHeadScore(
                head_label=head_label,
                layer_idx=int(wait_row["layer_idx"]),
                head_idx=int(wait_row["head_idx"]),
                wait_anchor_count=int(wait_row["wait_anchor_count"]),
                mean_no_wait_rate=float(wait_row["mean_no_wait_rate"]),
                mean_wait_logit_drop=float(wait_row["mean_wait_logit_drop"]),
                max_wait_logit_drop=float(wait_row["max_wait_logit_drop"]),
                prev_1_top_nonself_rate=float(prev_row.get("prev_1_top_nonself_rate", 0.0)),
                prev_1_top1_rate=float(prev_row.get("prev_1_top1_rate", 0.0)),
                mean_prev_mass_w1=float(prev_row.get("mean_prev_mass_w1", 0.0)),
                wait_rank=wait_rank,
                prev_rank=prev_rank,
                combined_rank_score=float(wait_rank + prev_rank),
            )
        )

    candidates.sort(
        key=lambda item: (
            item.combined_rank_score,
            -item.mean_no_wait_rate,
            -item.mean_wait_logit_drop,
            -item.prev_1_top_nonself_rate,
            -item.mean_prev_mass_w1,
            item.layer_idx,
            item.head_idx,
        )
    )
    return candidates


def summarize_zero_validation(summary: Dict[str, Any]) -> Dict[str, Any]:
    baseline_repetition = float(summary.get("baseline_repetition_rate", 0.0))
    intervention_repetition = float(summary.get("intervention_repetition_rate", 0.0))
    baseline_correct = float(summary.get("baseline_correct_rate_over_verifiable", 0.0))
    intervention_correct = float(summary.get("intervention_correct_rate_over_verifiable", 0.0))
    return {
        "baseline_repetition_rate": baseline_repetition,
        "intervention_repetition_rate": intervention_repetition,
        "repetition_delta": intervention_repetition - baseline_repetition,
        "repetition_induction_rate": float(summary.get("repetition_induction_rate", summary.get("repetition_induction_rate", 0.0))),
        "repetition_suppression_rate": float(summary.get("repetition_suppression_rate", 0.0)),
        "baseline_correct_rate": baseline_correct,
        "intervention_correct_rate": intervention_correct,
        "accuracy_delta": intervention_correct - baseline_correct,
        "mean_generated_tokens_delta": float(summary.get("mean_generated_tokens_delta", 0.0)),
    }


def summarize_scale_validation(summary: Dict[str, Any], *, boosted_scale: float) -> Dict[str, Any]:
    summary_rows = list(summary.get("summary_by_scale") or [])
    by_scale = {float(row["scale"]): row for row in summary_rows}
    baseline = by_scale.get(1.0)
    boosted = by_scale.get(float(boosted_scale))
    if baseline is None or boosted is None:
        raise ValueError(f"Scale summary missing scale 1.0 or boosted scale {boosted_scale}.")

    baseline_correct = float(baseline.get("correct_rate_over_verifiable", 0.0))
    boosted_correct = float(boosted.get("correct_rate_over_verifiable", 0.0))
    baseline_tokens = float(baseline.get("mean_generated_tokens", 0.0))
    boosted_tokens = float(boosted.get("mean_generated_tokens", 0.0))
    baseline_reflection = float(baseline.get("mean_reflection_count", 0.0))
    boosted_reflection = float(boosted.get("mean_reflection_count", 0.0))
    baseline_repetition = float(baseline.get("repetition_hit_rate", 0.0))
    boosted_repetition = float(boosted.get("repetition_hit_rate", 0.0))
    return {
        "baseline_correct_rate": baseline_correct,
        "boosted_correct_rate": boosted_correct,
        "accuracy_delta": boosted_correct - baseline_correct,
        "baseline_mean_generated_tokens": baseline_tokens,
        "boosted_mean_generated_tokens": boosted_tokens,
        "mean_generated_tokens_delta": boosted_tokens - baseline_tokens,
        "baseline_mean_reflection_count": baseline_reflection,
        "boosted_mean_reflection_count": boosted_reflection,
        "mean_reflection_count_delta": boosted_reflection - baseline_reflection,
        "baseline_repetition_rate": baseline_repetition,
        "boosted_repetition_rate": boosted_repetition,
        "repetition_delta": boosted_repetition - baseline_repetition,
    }


def evaluate_l0h3_like_behavior(
    *,
    zero_validation: Dict[str, Any],
    scale_validation: Dict[str, Any],
    max_accuracy_drop: float = 0.03,
    min_ablation_repetition_delta: float = 0.10,
    max_boost_length_delta: float = -50.0,
    max_boost_reflection_delta: float = -0.10,
) -> Dict[str, Any]:
    ablation_pass = (
        float(zero_validation.get("repetition_delta", 0.0)) >= min_ablation_repetition_delta
    )
    boost_pass = (
        float(scale_validation.get("mean_generated_tokens_delta", 0.0)) <= max_boost_length_delta
        and float(scale_validation.get("mean_reflection_count_delta", 0.0)) <= max_boost_reflection_delta
        and float(scale_validation.get("accuracy_delta", 0.0)) >= -max_accuracy_drop
    )
    return {
        "ablation_repetition_pass": ablation_pass,
        "boost_length_reflection_pass": boost_pass,
        "overall_pass": bool(ablation_pass and boost_pass),
    }


def write_candidate_report(
    *,
    path: str | Path,
    model_name_or_path: str,
    candidates: Sequence[CandidateHeadScore],
    chosen_head: str,
    zero_validation: Dict[str, Any],
    scale_validation: Dict[str, Any],
    behavior_eval: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# L0H3-Like Head Discovery Report\n\n")
        f.write(f"- model: `{model_name_or_path}`\n")
        f.write(f"- chosen_head: `{chosen_head}`\n")
        f.write(f"- ablation_repetition_pass: `{behavior_eval.get('ablation_repetition_pass')}`\n")
        f.write(f"- boost_length_reflection_pass: `{behavior_eval.get('boost_length_reflection_pass')}`\n")
        f.write(f"- overall_pass: `{behavior_eval.get('overall_pass')}`\n\n")

        f.write("## Top Candidates\n\n")
        for item in list(candidates)[:10]:
            f.write(
                "- "
                f"{item.head_label}: combined_rank_score={item.combined_rank_score:.3f}, "
                f"wait_rank={item.wait_rank}, prev_rank={item.prev_rank}, "
                f"mean_no_wait_rate={item.mean_no_wait_rate:.6f}, "
                f"mean_wait_logit_drop={item.mean_wait_logit_drop:.6f}, "
                f"prev_1_top_nonself_rate={item.prev_1_top_nonself_rate:.6f}, "
                f"mean_prev_mass_w1={item.mean_prev_mass_w1:.6f}\n"
            )
        f.write("\n")

        f.write("## Zero Validation\n\n")
        for key, value in zero_validation.items():
            f.write(f"- {key}: `{value}`\n")
        f.write("\n")

        f.write("## Scale Validation\n\n")
        for key, value in scale_validation.items():
            f.write(f"- {key}: `{value}`\n")


def save_discovery_bundle(
    *,
    output_dir: str | Path,
    candidates: Sequence[CandidateHeadScore],
    zero_validation: Dict[str, Any],
    scale_validation: Dict[str, Any],
    behavior_eval: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_scores.csv", [item.to_dict() for item in candidates])
    write_json(output_dir / "zero_validation_summary.json", zero_validation)
    write_json(output_dir / "scale_validation_summary.json", scale_validation)
    write_json(output_dir / "behavior_eval.json", behavior_eval)
    write_json(output_dir / "discovery_metadata.json", metadata)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
