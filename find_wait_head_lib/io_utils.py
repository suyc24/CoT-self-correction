from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from .ablation import HeadSpec


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc
            records.append(obj)
    return records


def ensure_required_fields(records: Iterable[Dict[str, Any]]) -> None:
    required = ["id", "correct_answer", "wrong_answer"]
    for idx, rec in enumerate(records):
        for key in required:
            if key not in rec:
                raise ValueError(f"Record index {idx} missing required field '{key}'.")
        if ("question" not in rec or not str(rec.get("question", "")).strip()) and (
            "prompt_prefix" not in rec or not str(rec.get("prompt_prefix", "")).strip()
        ):
            raise ValueError(
                f"Record index {idx} must provide either non-empty 'question' or non-empty 'prompt_prefix'."
            )


def init_summary_entry() -> Dict[str, Any]:
    return {
        "total": 0,
        "corrected": 0,
        "keep_wrong": 0,
        "other_answer": 0,
        "no_boxed": 0,
        "keyword_hit": 0,
    }


def update_summary(stats: Dict[str, Dict[str, Any]], row: Dict[str, Any]) -> None:
    label = row["head_label"]
    if label not in stats:
        stats[label] = init_summary_entry()
    s = stats[label]
    s["total"] += 1
    s[row["outcome"]] += 1
    if row["self_correction_keyword_hit"]:
        s["keyword_hit"] += 1


def write_summary_csv(
    path: Path,
    stats: Dict[str, Dict[str, Any]],
    head_lookup: Dict[str, HeadSpec],
) -> None:
    baseline_rate = 0.0
    if "baseline" in stats and stats["baseline"]["total"] > 0:
        baseline_rate = stats["baseline"]["corrected"] / stats["baseline"]["total"]

    rows: List[Dict[str, Any]] = []
    for label, s in stats.items():
        total = max(1, s["total"])
        corrected_rate = s["corrected"] / total
        keep_wrong_rate = s["keep_wrong"] / total
        keyword_hit_rate = s["keyword_hit"] / total

        layer_idx = ""
        head_idx = ""
        if label != "baseline" and label in head_lookup:
            layer_idx = head_lookup[label].layer_idx
            head_idx = head_lookup[label].head_idx

        rows.append(
            {
                "head_label": label,
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "total": s["total"],
                "corrected": s["corrected"],
                "keep_wrong": s["keep_wrong"],
                "other_answer": s["other_answer"],
                "no_boxed": s["no_boxed"],
                "keyword_hit": s["keyword_hit"],
                "corrected_rate": round(corrected_rate, 6),
                "keep_wrong_rate": round(keep_wrong_rate, 6),
                "keyword_hit_rate": round(keyword_hit_rate, 6),
                "delta_corrected_rate_vs_baseline": round(corrected_rate - baseline_rate, 6),
            }
        )

    def sort_key(x: Dict[str, Any]) -> Tuple[int, int, int]:
        if x["head_label"] == "baseline":
            return (-1, -1, -1)
        return (0, int(x["layer_idx"]), int(x["head_idx"]))

    rows.sort(key=sort_key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "head_label",
                "layer_idx",
                "head_idx",
                "total",
                "corrected",
                "keep_wrong",
                "other_answer",
                "no_boxed",
                "keyword_hit",
                "corrected_rate",
                "keep_wrong_rate",
                "keyword_hit_rate",
                "delta_corrected_rate_vs_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def select_examples(all_examples: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    examples = all_examples
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    if args.debug:
        if args.debug_example_id:
            target = None
            for ex in examples:
                if str(ex["id"]) == args.debug_example_id:
                    target = ex
                    break
            if target is None:
                raise ValueError(f"--debug_example_id={args.debug_example_id} not found in dataset.")
            return [target]
        return [examples[0]]

    return examples


def parse_parallel_gpu_ids(parallel_gpu_ids: str) -> List[int]:
    if parallel_gpu_ids.strip():
        out: List[int] = []
        seen = set()
        for token in parallel_gpu_ids.split(","):
            token = token.strip()
            if not token:
                continue
            gpu_id = int(token)
            if gpu_id < 0:
                raise ValueError(f"Invalid GPU id: {gpu_id}")
            if gpu_id not in seen:
                seen.add(gpu_id)
                out.append(gpu_id)
        if not out:
            raise ValueError("--parallel_gpu_ids is set but empty after parsing.")
        return out
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def split_head_labels_round_robin(head_labels: List[str], num_buckets: int) -> List[List[str]]:
    if num_buckets <= 0:
        return []
    buckets: List[List[str]] = [[] for _ in range(num_buckets)]
    for i, label in enumerate(head_labels):
        buckets[i % num_buckets].append(label)
    return [b for b in buckets if b]


def should_keep_filtered_row(row: Dict[str, Any]) -> bool:
    return (
        row["condition"] == "single_head_ablation"
        and row["outcome"] != "corrected"
    )


def build_export_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "example_id": row["example_id"],
        "head_label": row["head_label"],
        "layer_idx": row["layer_idx"],
        "head_idx": row["head_idx"],
        "cot_continuation": row["generated_continuation"],
    }


def merge_summary_stats(dst: Dict[str, Dict[str, Any]], src: Dict[str, Dict[str, Any]]) -> None:
    keys = ["total", "corrected", "keep_wrong", "other_answer", "no_boxed", "keyword_hit"]
    for label, item in src.items():
        if label not in dst:
            dst[label] = init_summary_entry()
        for k in keys:
            dst[label][k] += int(item.get(k, 0))


def dump_prepared_examples(path: Path, prepared_examples: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex, prep in prepared_examples:
            f.write(json.dumps({"example": ex, "prep_info": prep}, ensure_ascii=False) + "\n")


def load_prepared_examples(path: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prepared-json on line {line_num}: {exc}") from exc
            out.append((obj["example"], obj["prep_info"]))
    return out


def init_wait_logit_entry() -> Dict[str, Any]:
    return {
        "count": 0,
        "sum_baseline": 0.0,
        "sum_ablated": 0.0,
        "sum_delta": 0.0,
        "sum_abs_delta": 0.0,
        "max_abs_delta": 0.0,
    }


def update_wait_logit_stats(
    stats: Dict[str, Dict[str, Any]],
    head_label: str,
    baseline_logit: float,
    ablated_logit: float,
) -> None:
    if head_label not in stats:
        stats[head_label] = init_wait_logit_entry()
    s = stats[head_label]
    delta = float(ablated_logit - baseline_logit)
    abs_delta = abs(delta)
    s["count"] += 1
    s["sum_baseline"] += float(baseline_logit)
    s["sum_ablated"] += float(ablated_logit)
    s["sum_delta"] += delta
    s["sum_abs_delta"] += abs_delta
    if abs_delta > s["max_abs_delta"]:
        s["max_abs_delta"] = abs_delta


def merge_wait_logit_stats(dst: Dict[str, Dict[str, Any]], src: Dict[str, Dict[str, Any]]) -> None:
    keys = ["count", "sum_baseline", "sum_ablated", "sum_delta", "sum_abs_delta"]
    for label, item in src.items():
        if label not in dst:
            dst[label] = init_wait_logit_entry()
        for k in keys:
            dst[label][k] += float(item.get(k, 0.0))
        dst[label]["max_abs_delta"] = max(
            float(dst[label]["max_abs_delta"]),
            float(item.get("max_abs_delta", 0.0)),
        )


def write_wait_logit_ranking_csv(
    path: Path,
    stats: Dict[str, Dict[str, Any]],
    head_lookup: Dict[str, HeadSpec],
) -> None:
    rows: List[Dict[str, Any]] = []
    for label, s in stats.items():
        count = int(s.get("count", 0))
        if count <= 0:
            continue
        mean_baseline = float(s["sum_baseline"]) / count
        mean_ablated = float(s["sum_ablated"]) / count
        mean_delta = float(s["sum_delta"]) / count
        mean_abs_delta = float(s["sum_abs_delta"]) / count
        max_abs_delta = float(s.get("max_abs_delta", 0.0))

        layer_idx = ""
        head_idx = ""
        if label in head_lookup:
            layer_idx = head_lookup[label].layer_idx
            head_idx = head_lookup[label].head_idx

        rows.append(
            {
                "head_label": label,
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "count": count,
                "mean_logit_baseline_wait": round(mean_baseline, 6),
                "mean_logit_ablated_wait": round(mean_ablated, 6),
                "mean_delta_ablated_minus_baseline": round(mean_delta, 6),
                "mean_abs_delta": round(mean_abs_delta, 6),
                "max_abs_delta": round(max_abs_delta, 6),
            }
        )

    rows.sort(key=lambda x: x["mean_abs_delta"], reverse=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "head_label",
                "layer_idx",
                "head_idx",
                "count",
                "mean_logit_baseline_wait",
                "mean_logit_ablated_wait",
                "mean_delta_ablated_minus_baseline",
                "mean_abs_delta",
                "max_abs_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
