#!/usr/bin/env python3
from __future__ import annotations

"""Filter CoT generations that show repetition / looping behavior.

Usage modes:
- Single-file mode: pass --input_jsonl explicitly.
- Batch mode: omit --input_jsonl and scan --input_dir recursively (default: outputs/).

Batch outputs are written under outputs/repetition/ by default.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.io_utils import load_jsonl, peek_first_json_row, truncate_text
from cot_research.repetition_analysis import RepetitionThresholds, analyze_row_repetition
from cot_research.row_utils import select_continuation_text


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Filter obvious dead-loop CoT cases from JSONL files")
    parser.add_argument(
        "--input_jsonl",
        default="",
        help=(
            "Collected CoT JSONL. If omitted, the script will scan --input_dir recursively "
            f"(default: {root_dir / 'outputs'})."
        ),
    )
    parser.add_argument(
        "--input_dir",
        default="",
        help=(
            "Directory to scan recursively for JSONL files in batch mode. "
            f"Default: {root_dir / 'outputs'}"
        ),
    )
    parser.add_argument("--output_jsonl", default="", help="Filtered repetitive-case JSONL in single-file mode")
    parser.add_argument("--summary_json", default="", help="Summary JSON path in single-file mode")
    parser.add_argument("--top_markdown", default="", help="Markdown preview path in single-file mode")
    parser.add_argument(
        "--output_root",
        default=str(root_dir / "outputs" / "repetition"),
        help="Root directory for batch outputs.",
    )
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--same_token_run_threshold", type=int, default=40)
    parser.add_argument("--tail_repeat_min_repeats", type=int, default=6)
    parser.add_argument("--tail_repeat_max_ngram", type=int, default=8)
    parser.add_argument("--tail_repeat_min_span", type=int, default=24)
    parser.add_argument("--line_repeat_threshold", type=int, default=4)
    parser.add_argument("--word_tail_repeat_min_repeats", type=int, default=5)
    parser.add_argument("--word_tail_repeat_max_ngram", type=int, default=8)
    parser.add_argument("--word_tail_repeat_min_span", type=int, default=24)
    parser.add_argument("--min_trigger_count", type=int, default=1)
    parser.add_argument("--preview_chars", type=int, default=1200)
    return parser.parse_args()


def is_under_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False

def is_supported_cot_row(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    return any(
        key in row
        for key in ["generated_continuation", "cot_continuation", "think_text", "full_text", "repetition"]
    )


def is_candidate_jsonl(path: Path, output_root: Path) -> bool:
    name = path.name.lower()
    path_text = str(path).lower()
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        return False
    if is_under_directory(path, output_root):
        return False
    if any(token in name for token in ["wait_logit", "head_wait"]):
        return False
    if any(token in path_text for token in ["/repetition/", "\\repetition\\"]):
        return False
    return True


def discover_input_paths(args: argparse.Namespace, output_root: Path) -> Tuple[List[Path], Optional[Path], bool]:
    if args.input_jsonl.strip():
        input_path = Path(args.input_jsonl)
        if not input_path.exists():
            raise FileNotFoundError(f"--input_jsonl not found: {input_path}")
        return [input_path], input_path.parent, False

    root_dir = Path(__file__).resolve().parents[1]
    input_dir = Path(args.input_dir) if args.input_dir.strip() else root_dir / "outputs"
    if not input_dir.exists():
        raise FileNotFoundError(f"--input_dir not found: {input_dir}")

    discovered: List[Path] = []
    for path in sorted(input_dir.rglob("*.jsonl")):
        if not is_candidate_jsonl(path, output_root):
            continue
        try:
            first_row = peek_first_json_row(path)
        except ValueError as exc:
            print(f"[Warn] Skip invalid JSONL {path}: {exc}")
            continue
        if is_supported_cot_row(first_row):
            discovered.append(path)

    if not discovered:
        raise FileNotFoundError(
            f"No suitable CoT JSONL files found under {input_dir}. "
            "Please pass --input_jsonl explicitly if needed."
        )

    return discovered, input_dir, True

def detect_repetition(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    thresholds = RepetitionThresholds(
        same_token_run_threshold=args.same_token_run_threshold,
        tail_repeat_min_repeats=args.tail_repeat_min_repeats,
        tail_repeat_max_ngram=args.tail_repeat_max_ngram,
        tail_repeat_min_span=args.tail_repeat_min_span,
        line_repeat_threshold=args.line_repeat_threshold,
        line_run_score_multiplier=12,
        word_tail_repeat_min_repeats=args.word_tail_repeat_min_repeats,
        word_tail_repeat_max_ngram=args.word_tail_repeat_max_ngram,
        word_tail_repeat_min_span=args.word_tail_repeat_min_span,
        tail_word_requires_hard_signal=True,
        min_trigger_count=args.min_trigger_count,
    )
    return analyze_row_repetition(row, thresholds=thresholds, reuse_existing=False)

def write_markdown(path: Path, rows: List[Dict[str, Any]], preview_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Repetitive CoT Cases\n\n")
        for idx, row in enumerate(rows, start=1):
            detection = row.get("repetition_detection") or {}
            f.write(f"## {idx}. {row.get('example_id', 'unknown')}\n\n")
            f.write(f"- source: `{row.get('source')}`\n")
            f.write(f"- input_jsonl: `{row.get('source_input_jsonl', '')}`\n")
            f.write(f"- score: `{detection.get('score')}`\n")
            f.write(f"- trigger_types: `{', '.join(detection.get('trigger_types') or [])}`\n")
            f.write(f"- generated_tokens: `{row.get('generated_tokens')}`\n\n")
            f.write("### Problem\n\n")
            f.write(truncate_text(str(row.get("problem") or ""), preview_chars) + "\n\n")
            f.write("### Continuation\n\n```text\n")
            f.write(
                truncate_text(
                    str(
                        row.get("generated_continuation")
                        or row.get("cot_continuation")
                        or row.get("think_text")
                        or row.get("full_text")
                        or ""
                    ),
                    preview_chars,
                )
            )
            f.write("\n```\n\n")


def build_single_output_paths(
    *,
    input_path: Path,
    input_base_dir: Optional[Path],
    output_root: Path,
    explicit_output_jsonl: str,
    explicit_summary_json: str,
    explicit_top_markdown: str,
    batch_mode: bool,
) -> Tuple[Path, Path, Path]:
    if not batch_mode:
        output_path = Path(explicit_output_jsonl) if explicit_output_jsonl else input_path.with_name(input_path.stem + ".repetition_cases.jsonl")
        summary_path = Path(explicit_summary_json) if explicit_summary_json else input_path.with_name(input_path.stem + ".repetition_summary.json")
        markdown_path = Path(explicit_top_markdown) if explicit_top_markdown else input_path.with_name(input_path.stem + ".repetition_top.md")
        return output_path, summary_path, markdown_path

    if input_base_dir is None:
        raise ValueError("batch_mode requires input_base_dir")

    rel_path = input_path.resolve().relative_to(input_base_dir.resolve())
    output_dir = output_root / rel_path.parent / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / "repetition_cases.jsonl",
        output_dir / "repetition_summary.json",
        output_dir / "repetition_top.md",
    )


def process_single_input(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    markdown_path: Path,
) -> Dict[str, Any]:
    rows = load_jsonl(str(input_path))
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    matched_rows: List[Dict[str, Any]] = []
    trigger_counter: Counter[str] = Counter()
    for row in rows:
        detection = detect_repetition(row, args)
        if not detection["matched"]:
            continue
        enriched = dict(row)
        enriched["source_input_jsonl"] = str(input_path)
        enriched["repetition_detection"] = detection
        matched_rows.append(enriched)
        trigger_counter.update(detection["trigger_types"])

    matched_rows.sort(key=lambda row: row["repetition_detection"]["score"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in matched_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    top_rows = matched_rows[: max(args.top_k, 0)]
    write_markdown(markdown_path, top_rows, args.preview_chars)

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "summary_json": str(summary_path),
        "top_markdown": str(markdown_path),
        "total_examples": len(rows),
        "matched_examples": len(matched_rows),
        "match_rate": 0.0 if not rows else round(len(matched_rows) / len(rows), 6),
        "thresholds": {
            "same_token_run_threshold": args.same_token_run_threshold,
            "tail_repeat_min_repeats": args.tail_repeat_min_repeats,
            "tail_repeat_max_ngram": args.tail_repeat_max_ngram,
            "tail_repeat_min_span": args.tail_repeat_min_span,
            "line_repeat_threshold": args.line_repeat_threshold,
            "word_tail_repeat_min_repeats": args.word_tail_repeat_min_repeats,
            "word_tail_repeat_max_ngram": args.word_tail_repeat_max_ngram,
            "word_tail_repeat_min_span": args.word_tail_repeat_min_span,
            "min_trigger_count": args.min_trigger_count,
        },
        "trigger_counts": dict(trigger_counter),
        "top_examples": [
            {
                "example_id": row.get("example_id"),
                "source": row.get("source"),
                "score": row["repetition_detection"]["score"],
                "trigger_types": row["repetition_detection"]["trigger_types"],
                "generated_tokens": row.get("generated_tokens"),
            }
            for row in top_rows
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_paths, input_base_dir, batch_mode = discover_input_paths(args, output_root)
    print(
        "[Info] Repetition scan setup: "
        f"batch_mode={batch_mode}, input_count={len(input_paths)}, output_root={output_root}"
    )

    all_case_rows: List[Dict[str, Any]] = []
    batch_summaries: List[Dict[str, Any]] = []
    batch_errors: List[Dict[str, str]] = []
    for input_path in input_paths:
        output_path, summary_path, markdown_path = build_single_output_paths(
            input_path=input_path,
            input_base_dir=input_base_dir,
            output_root=output_root,
            explicit_output_jsonl=args.output_jsonl,
            explicit_summary_json=args.summary_json,
            explicit_top_markdown=args.top_markdown,
            batch_mode=batch_mode,
        )
        print(f"[Info] Processing input_jsonl: {input_path}")
        try:
            summary = process_single_input(args, input_path, output_path, summary_path, markdown_path)
            batch_summaries.append(summary)
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    all_case_rows.append(json.loads(line))
            print(
                f"[Info] Finished {input_path.name}: "
                f"matched_examples={summary['matched_examples']}/{summary['total_examples']}"
            )
        except Exception as exc:
            batch_errors.append({"input_jsonl": str(input_path), "error": str(exc)})
            print(f"[Warn] Failed processing {input_path}: {exc}")

    if batch_mode:
        all_case_rows.sort(key=lambda row: row["repetition_detection"]["score"], reverse=True)
        aggregate_cases_path = output_root / "all_repetition_cases.jsonl"
        aggregate_markdown_path = output_root / "all_repetition_top.md"
        aggregate_summary_path = output_root / "batch_summary.json"

        with open(aggregate_cases_path, "w", encoding="utf-8") as f:
            for row in all_case_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        write_markdown(aggregate_markdown_path, all_case_rows[: max(args.top_k, 0)], args.preview_chars)

        aggregate_trigger_counts: Counter[str] = Counter()
        total_examples = 0
        matched_examples = 0
        for summary in batch_summaries:
            total_examples += int(summary["total_examples"])
            matched_examples += int(summary["matched_examples"])
            aggregate_trigger_counts.update(summary.get("trigger_counts") or {})

        aggregate_summary = {
            "input_dir": str(input_base_dir),
            "output_root": str(output_root),
            "processed_files": len(batch_summaries),
            "error_files": len(batch_errors),
            "total_examples": total_examples,
            "matched_examples": matched_examples,
            "match_rate": 0.0 if total_examples == 0 else round(matched_examples / total_examples, 6),
            "trigger_counts": dict(aggregate_trigger_counts),
            "per_file": batch_summaries,
            "errors": batch_errors,
            "aggregate_cases_jsonl": str(aggregate_cases_path),
            "aggregate_top_markdown": str(aggregate_markdown_path),
        }
        with open(aggregate_summary_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_summary, f, ensure_ascii=False, indent=2)

        print("[Done] Batch repetition filtering finished:")
        print(f"- input_dir: {input_base_dir}")
        print(f"- output_root: {output_root}")
        print(f"- processed_files: {len(batch_summaries)}")
        print(f"- error_files: {len(batch_errors)}")
        print(f"- total_examples: {total_examples}")
        print(f"- matched_examples: {matched_examples}")
        print(f"- aggregate_cases_jsonl: {aggregate_cases_path}")
        print(f"- aggregate_top_markdown: {aggregate_markdown_path}")
        print(f"- aggregate_summary_json: {aggregate_summary_path}")
        return

    if not batch_summaries:
        raise RuntimeError("No input file was processed successfully.")

    summary = batch_summaries[0]
    print("[Done] Repetition filtering finished:")
    print(f"- input_jsonl: {summary['input_jsonl']}")
    print(f"- output_jsonl: {summary['output_jsonl']}")
    print(f"- summary_json: {summary['summary_json']}")
    print(f"- top_markdown: {summary['top_markdown']}")
    print(f"- total_examples: {summary['total_examples']}")
    print(f"- matched_examples: {summary['matched_examples']}")
    print(f"- trigger_counts: {summary['trigger_counts']}")


if __name__ == "__main__":
    main()
