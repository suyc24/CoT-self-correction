#!/usr/bin/env python3
from __future__ import annotations

"""Summarize partially completed worker outputs from test_head_boost_effects.py.

This script is useful when generation finished or was interrupted before the
main process wrote the final summary files. It scans `_worker_*_rows.jsonl`
files, merges them, and rebuilds accuracy/repetition summaries.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.cot_accuracy import summarize_comparison_accuracy
from cot_research.io_utils import load_jsonl, truncate_text, write_csv, write_json

def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_input = root_dir / "outputs" / "head_boost_effects" / "qwen3_1p7b"
    parser = argparse.ArgumentParser(description="Summarize worker output files from head boost experiments")
    parser.add_argument("--input_dir", type=str, default=str(default_input))
    parser.add_argument("--preview_examples", type=int, default=20)
    parser.add_argument("--preview_chars", type=int, default=1200)
    return parser.parse_args()

def dedupe_rows_by_example_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    ordered: List[str] = []
    for row in rows:
        key = str(row.get("example_id") or "")
        if key and key not in deduped:
            ordered.append(key)
        deduped[key] = row
    return [deduped[key] for key in ordered]


def write_preview_markdown(path: Path, rows: List[Dict[str, Any]], preview_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Head Boost Effects Preview\n\n")
        for idx, row in enumerate(rows, start=1):
            f.write(f"## {idx}. {row.get('example_id', 'unknown')}\n\n")
            f.write(f"- source: `{row.get('source')}`\n")
            f.write(f"- repetition_category: `{row.get('repetition_comparison', {}).get('category')}`\n")
            f.write(f"- accuracy_category: `{row.get('answer_comparison', {}).get('category')}`\n")
            f.write(f"- gold_answer: `{row.get('answer_comparison', {}).get('gold_answer')}`\n")
            f.write(f"- baseline_answer: `{row.get('answer_comparison', {}).get('baseline_answer')}`\n")
            f.write(f"- intervention_answer: `{row.get('answer_comparison', {}).get('intervention_answer')}`\n")
            f.write(f"- baseline_repetition_score: `{row.get('repetition_comparison', {}).get('baseline_score')}`\n")
            f.write(f"- intervention_repetition_score: `{row.get('repetition_comparison', {}).get('intervention_score')}`\n")
            f.write(f"- repetition_score_delta: `{row.get('repetition_comparison', {}).get('score_delta')}`\n\n")
            f.write("### Problem\n\n")
            f.write(truncate_text(str(row.get("problem") or row.get("question") or ""), preview_chars) + "\n\n")
            f.write("### Baseline Continuation\n\n```text\n")
            f.write(truncate_text(str(row.get("baseline", {}).get("continuation") or ""), preview_chars))
            f.write("\n```\n\n")
            f.write("### Intervention Continuation\n\n```text\n")
            f.write(truncate_text(str(row.get("intervention", {}).get("continuation") or ""), preview_chars))
            f.write("\n```\n\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    worker_row_paths = sorted(input_dir.glob("*_worker_*_rows.jsonl"))
    worker_skipped_paths = sorted(input_dir.glob("*_worker_*_skipped.json"))
    if not worker_row_paths:
        raise FileNotFoundError(f"No worker row files found under {input_dir}")

    rows_path = input_dir / "rows.jsonl"
    case_summary_path = input_dir / "case_summary.csv"
    accuracy_summary_path = input_dir / "accuracy_summary.csv"
    repetition_summary_path = input_dir / "repetition_summary.csv"
    summary_path = input_dir / "summary.json"
    preview_path = input_dir / "top_examples.md"

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    for path in worker_row_paths:
        result_rows.extend(load_jsonl(path))
    for path in worker_skipped_paths:
        with open(path, "r", encoding="utf-8") as f:
            skipped_rows.extend(json.load(f))
    result_rows = dedupe_rows_by_example_id(result_rows)

    with open(rows_path, "w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    accuracy_summary = summarize_comparison_accuracy(result_rows)
    repetition_counter: Counter[str] = Counter()
    overlap_counter: Counter[str] = Counter()
    baseline_scores: List[float] = []
    intervention_scores: List[float] = []
    baseline_tokens: List[float] = []
    intervention_tokens: List[float] = []
    baseline_hit_cap = 0
    intervention_hit_cap = 0
    case_summary_rows: List[Dict[str, Any]] = []

    for row in result_rows:
        rep = row["repetition_comparison"]
        acc = row["answer_comparison"]
        repetition_counter.update([str(rep["category"])])
        overlap_counter.update([f"{rep['category']}__{acc.get('category', 'unknown')}"])
        baseline_scores.append(float(rep["baseline_score"]))
        intervention_scores.append(float(rep["intervention_score"]))
        baseline_tokens.append(float(row["baseline"].get("generated_tokens", 0)))
        intervention_tokens.append(float(row["intervention"].get("generated_tokens", 0)))
        if bool(row["baseline"].get("hit_max_new_tokens")):
            baseline_hit_cap += 1
        if bool(row["intervention"].get("hit_max_new_tokens")):
            intervention_hit_cap += 1
        case_summary_rows.append(
            {
                "example_id": row["example_id"],
                "source": row.get("source"),
                "repetition_category": rep["category"],
                "accuracy_category": acc.get("category"),
                "baseline_matched": rep["baseline_matched"],
                "intervention_matched": rep["intervention_matched"],
                "baseline_correct": acc.get("baseline_correct"),
                "intervention_correct": acc.get("intervention_correct"),
                "gold_answer": acc.get("gold_answer"),
                "baseline_answer": acc.get("baseline_answer"),
                "intervention_answer": acc.get("intervention_answer"),
                "baseline_score": rep["baseline_score"],
                "intervention_score": rep["intervention_score"],
                "score_delta": rep["score_delta"],
                "baseline_generated_tokens": row["baseline"].get("generated_tokens", 0),
                "intervention_generated_tokens": row["intervention"].get("generated_tokens", 0),
                "baseline_hit_max_new_tokens": row["baseline"].get("hit_max_new_tokens"),
                "intervention_hit_max_new_tokens": row["intervention"].get("hit_max_new_tokens"),
                "generated_tokens_delta": rep["generated_tokens_delta"],
                "baseline_trigger_types": ",".join(rep["baseline_trigger_types"]),
                "intervention_trigger_types": ",".join(rep["intervention_trigger_types"]),
            }
        )

    case_summary_rows.sort(key=lambda item: (item["score_delta"], -item["baseline_score"]))
    write_csv(case_summary_path, case_summary_rows)
    write_csv(
        accuracy_summary_path,
        [{"category": k, "count": v} for k, v in sorted(dict(accuracy_summary.get("accuracy_counts") or {}).items())],
    )
    write_csv(
        repetition_summary_path,
        [{"category": k, "count": v} for k, v in sorted(dict(repetition_counter).items())],
    )

    def _mean(values: List[float]) -> float:
        return 0.0 if not values else sum(values) / len(values)

    first_row = result_rows[0] if result_rows else {}
    summary = {
        "input_dir": str(input_dir),
        "processed_examples": len(result_rows),
        "skipped_examples": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "completed_successfully": False,
        "interrupted": True,
        "recovered_from_worker_outputs": True,
        "head_labels": first_row.get("head_labels", []),
        "intervention_kind": first_row.get("intervention_kind"),
        "intervention_params": first_row.get("intervention_params", {}),
        "accuracy_counts": dict(accuracy_summary.get("accuracy_counts") or {}),
        "repetition_counts": dict(repetition_counter),
        "overlap_counts": dict(overlap_counter),
        "verifiable_examples": int(accuracy_summary.get("verifiable_examples", 0)),
        "baseline_correct_rate_over_verifiable": float(accuracy_summary.get("baseline_correct_rate_over_verifiable", 0.0)),
        "intervention_correct_rate_over_verifiable": float(accuracy_summary.get("intervention_correct_rate_over_verifiable", 0.0)),
        "newly_correct_rate_over_verifiable": float(accuracy_summary.get("newly_correct_rate_over_verifiable", 0.0)),
        "regression_rate_over_verifiable": float(accuracy_summary.get("regression_rate_over_verifiable", 0.0)),
        "baseline_repetition_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["repetition_comparison"]["baseline_matched"]) / len(result_rows), 6),
        "intervention_repetition_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["repetition_comparison"]["intervention_matched"]) / len(result_rows), 6),
        "repetition_suppression_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["repetition_comparison"]["category"] == "suppressed") / len(result_rows), 6),
        "repetition_induction_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["repetition_comparison"]["category"] == "induced_repetition") / len(result_rows), 6),
        "mean_baseline_repetition_score": round(_mean(baseline_scores), 6),
        "mean_intervention_repetition_score": round(_mean(intervention_scores), 6),
        "mean_repetition_score_delta": round(_mean([float(row["repetition_comparison"]["score_delta"]) for row in result_rows]), 6),
        "mean_baseline_generated_tokens": round(_mean(baseline_tokens), 6),
        "mean_intervention_generated_tokens": round(_mean(intervention_tokens), 6),
        "mean_generated_tokens_delta": round(_mean([float(row["repetition_comparison"]["generated_tokens_delta"]) for row in result_rows]), 6),
        "baseline_hit_max_new_tokens_count": baseline_hit_cap,
        "intervention_hit_max_new_tokens_count": intervention_hit_cap,
        "baseline_hit_max_new_tokens_rate": 0.0 if not result_rows else round(baseline_hit_cap / len(result_rows), 6),
        "intervention_hit_max_new_tokens_rate": 0.0 if not result_rows else round(intervention_hit_cap / len(result_rows), 6),
        "worker_row_files": [str(path) for path in worker_row_paths],
        "worker_skipped_files": [str(path) for path in worker_skipped_paths],
    }
    write_json(summary_path, summary)

    preview_priority = {
        "newly_correct": 0,
        "regressed": 1,
        "remained_wrong": 2,
        "remained_correct": 3,
        "unverifiable": 4,
    }
    preview_rows = sorted(
        result_rows,
        key=lambda item: (
            preview_priority.get(str((item.get("answer_comparison") or {}).get("category")), 9),
            item["repetition_comparison"]["score_delta"],
            -item["repetition_comparison"]["baseline_score"],
        ),
    )[: max(args.preview_examples, 0)]
    write_preview_markdown(preview_path, preview_rows, args.preview_chars)

    print("[Done] Worker-output summary rebuilt:")
    print(f"- input_dir: {input_dir}")
    print(f"- worker_row_files: {len(worker_row_paths)}")
    print(f"- rows_jsonl: {rows_path}")
    print(f"- summary_json: {summary_path}")
    print(f"- accuracy_summary_csv: {accuracy_summary_path}")
    print(f"- repetition_summary_csv: {repetition_summary_path}")
    print(f"- preview_md: {preview_path}")


if __name__ == "__main__":
    main()
