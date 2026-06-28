#!/usr/bin/env python3
from __future__ import annotations

"""Judge answer correctness for existing CoT JSONL results.

Supports two common input styles:
- single-condition rows with fields like generated_continuation / final_boxed_answer
- comparison rows with nested baseline / intervention payloads

Gold answers can come from:
- the row itself (correct_answer / reference_solution / solution / messages)
- an external --gold_jsonl joined by example_id/id
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.cot_accuracy import (
    build_row_lookup,
    judge_comparison_row,
    judge_single_row,
    summarize_comparison_accuracy,
    summarize_single_accuracy,
)
from cot_research.io_utils import load_jsonl, truncate_text
from cot_research.summary_utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_input = root_dir / "outputs" / "repetition" / "l0h3_suppression_qwen3_1p7b" / "rows.jsonl"
    default_gold = root_dir / "outputs" / "repetition" / "all_repetition_cases.jsonl"
    default_output = root_dir / "outputs" / "repetition" / "l0h3_suppression_qwen3_1p7b" / "accuracy_judged"

    parser = argparse.ArgumentParser(description="Judge correctness for existing CoT JSONL outputs")
    parser.add_argument("--input_jsonl", type=str, default=str(default_input))
    parser.add_argument("--gold_jsonl", type=str, default=str(default_gold))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "single", "comparison"],
        help="auto: infer from row structure; single: judge top-level row; comparison: judge nested baseline/intervention.",
    )
    parser.add_argument("--preview_examples", type=int, default=30)
    parser.add_argument("--preview_chars", type=int, default=1000)
    return parser.parse_args()

def infer_mode(mode: str, rows: List[Dict[str, Any]]) -> str:
    if mode != "auto":
        return mode
    for row in rows:
        if isinstance(row.get("baseline"), dict) or isinstance(row.get("intervention"), dict):
            return "comparison"
    return "single"


def write_preview_markdown(path: Path, rows: List[Dict[str, Any]], mode: str, preview_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Accuracy Preview\n\n")
        for idx, row in enumerate(rows, start=1):
            f.write(f"## {idx}. {row.get('example_id', 'unknown')}\n\n")
            f.write(f"- source: `{row.get('source')}`\n")
            if mode == "comparison":
                comp = row.get("answer_comparison") or {}
                f.write(f"- category: `{comp.get('category')}`\n")
                f.write(f"- gold_answer: `{comp.get('gold_answer')}`\n")
                f.write(f"- baseline_answer: `{comp.get('baseline_answer')}`\n")
                f.write(f"- intervention_answer: `{comp.get('intervention_answer')}`\n\n")
                f.write("### Baseline Continuation\n\n```text\n")
                f.write(truncate_text(str(row.get("baseline", {}).get("continuation") or ""), preview_chars))
                f.write("\n```\n\n")
                f.write("### Intervention Continuation\n\n```text\n")
                f.write(truncate_text(str(row.get("intervention", {}).get("continuation") or ""), preview_chars))
                f.write("\n```\n\n")
            else:
                acc = row.get("accuracy") or {}
                f.write(f"- category: `{acc.get('category')}`\n")
                f.write(f"- gold_answer: `{acc.get('gold_answer')}`\n")
                f.write(f"- final_answer: `{acc.get('final_answer')}`\n\n")
                f.write("### Continuation\n\n```text\n")
                text = row.get("generated_continuation") or row.get("cot_continuation") or row.get("continuation") or row.get("full_text") or ""
                f.write(truncate_text(str(text), preview_chars))
                f.write("\n```\n\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    gold_lookup: Dict[str, Dict[str, Any]] = {}
    gold_path = Path(args.gold_jsonl) if args.gold_jsonl.strip() else None
    if gold_path is not None and gold_path.exists():
        gold_lookup = build_row_lookup(load_jsonl(gold_path))

    rows = load_jsonl(input_path)
    mode = infer_mode(args.mode, rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out_path = output_dir / "rows_with_accuracy.jsonl"
    summary_json_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    preview_path = output_dir / "preview.md"

    enriched_rows: List[Dict[str, Any]] = []
    overlap_counter: Counter[str] = Counter()
    for row in rows:
        gold_row = None
        for key in ["example_id", "id"]:
            value = row.get(key)
            if value is None:
                continue
            gold_row = gold_lookup.get(str(value))
            if gold_row is not None:
                break

        enriched = dict(row)
        if mode == "comparison":
            answer_comparison = judge_comparison_row(row, gold_row)
            enriched["answer_comparison"] = answer_comparison
            repetition_category = str((row.get("comparison") or {}).get("category", "unknown"))
            overlap_counter.update([f"{repetition_category}__{answer_comparison['category']}"])
        else:
            enriched["accuracy"] = judge_single_row(row, gold_row)
        enriched_rows.append(enriched)

    with open(rows_out_path, "w", encoding="utf-8") as f:
        for row in enriched_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if mode == "comparison":
        summary = summarize_comparison_accuracy(enriched_rows)
        summary["overlap_counts"] = dict(overlap_counter)
        summary_rows = [
            {"category": key, "count": value}
            for key, value in sorted((summary.get("accuracy_counts") or {}).items())
        ]
        preview_priority = {
            "newly_correct": 0,
            "regressed": 1,
            "remained_wrong": 2,
            "remained_correct": 3,
            "unverifiable": 4,
        }
        preview_rows = sorted(
            enriched_rows,
            key=lambda row: preview_priority.get(str((row.get("answer_comparison") or {}).get("category")), 9),
        )[: max(args.preview_examples, 0)]
    else:
        summary = summarize_single_accuracy(enriched_rows)
        summary_rows = [
            {"category": key, "count": value}
            for key, value in sorted((summary.get("accuracy_counts") or {}).items())
        ]
        preview_priority = {"wrong": 0, "correct": 1, "unverifiable": 2}
        preview_rows = sorted(
            enriched_rows,
            key=lambda row: preview_priority.get(str((row.get("accuracy") or {}).get("category")), 9),
        )[: max(args.preview_examples, 0)]

    summary.update(
        {
            "input_jsonl": str(input_path),
            "gold_jsonl": str(gold_path) if gold_path is not None else None,
            "output_dir": str(output_dir),
            "mode": mode,
        }
    )
    write_json(summary_json_path, summary)
    write_csv(summary_csv_path, summary_rows)
    write_preview_markdown(preview_path, preview_rows, mode, args.preview_chars)

    print("[Done] CoT accuracy judging finished:")
    print(f"- input_jsonl: {input_path}")
    print(f"- gold_jsonl: {gold_path}")
    print(f"- mode: {mode}")
    print(f"- rows_with_accuracy: {rows_out_path}")
    print(f"- summary_json: {summary_json_path}")
    print(f"- summary_csv: {summary_csv_path}")
    print(f"- preview_markdown: {preview_path}")


if __name__ == "__main__":
    main()
