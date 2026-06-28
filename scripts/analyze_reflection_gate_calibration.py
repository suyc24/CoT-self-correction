#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import answers_match, normalize_answer, parse_simple_number


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_rows(root: Path, include_partial: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for shard in sorted(path for path in (root / "shards").iterdir() if path.is_dir()):
        final_path = shard / "eval_rows.jsonl"
        partial_path = shard / "eval_rows.partial.jsonl"
        if final_path.exists():
            rows.extend(load_jsonl(final_path))
        elif include_partial and partial_path.exists():
            rows.extend(load_jsonl(partial_path))
    rows.sort(
        key=lambda r: (
            str(r.get("repeat_idx", "")),
            int(r.get("global_idx") or 0),
            str(r.get("example_id", "")),
            str(r.get("condition", "")),
            str(r.get("site", "")),
            float(r.get("alpha") or 0.0),
        )
    )
    return rows


def condition_key(row: Dict[str, Any]) -> str:
    if row.get("condition") == "baseline":
        return "baseline"
    return f"{row.get('condition')}_{row.get('site')}_alpha{float(row.get('alpha') or 0.0):g}"


def repeat_key(row: Dict[str, Any]) -> Optional[str]:
    repeat = row.get("repeat_idx")
    if repeat in (None, "") and isinstance(row.get("metadata"), dict):
        repeat = row["metadata"].get("repeat_idx")
    if repeat in (None, ""):
        return None
    return str(repeat)


def strip_units(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", s)
    return s.replace(r"\%", "%")


def numeric_variants(text: Optional[str]) -> List[float]:
    s0 = strip_units(text)
    variants = {s0, s0.replace("%", "")}
    no_words = re.sub(r"(?<!\\)[A-Za-z]+", " ", s0)
    variants.add(no_words)
    variants.add(no_words.replace("%", ""))
    values: List[float] = []
    for variant in variants:
        try:
            value = parse_simple_number(variant)
            if value is not None and math.isfinite(value):
                values.append(float(value))
        except Exception:
            pass
    for variant in variants:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", variant.replace(",", ""))
        if match:
            try:
                values.append(float(match.group(0)))
            except Exception:
                pass
    out: List[float] = []
    for value in values:
        if not any(abs(value - old) <= 1e-9 * max(1.0, abs(value), abs(old)) for old in out):
            out.append(value)
    return out


def nums_match(left: Optional[str], right: Optional[str]) -> bool:
    for left_value in numeric_variants(left):
        for right_value in numeric_variants(right):
            if abs(left_value - right_value) <= 1e-9 * max(1.0, abs(left_value), abs(right_value)):
                return True
    return False


def normalized_contains_answer(text: str, answer: Optional[str]) -> bool:
    if not text or answer is None:
        return False
    tail = str(text)[-2500:]
    answer_norm = normalize_answer(strip_units(answer))
    tail_norm = normalize_answer(strip_units(tail))
    if answer_norm and len(answer_norm) >= 2 and answer_norm in tail_norm:
        return True
    lower = tail.lower()
    cue = max(
        lower.rfind("answer"),
        lower.rfind("final"),
        lower.rfind("therefore"),
        lower.rfind("so "),
        lower.rfind("thus"),
    )
    scope = tail[cue:] if cue >= 0 else tail[-800:]
    return nums_match(scope, answer)


def semantic_correct(row: Dict[str, Any]) -> bool:
    final_boxed = row.get("final_boxed_answer")
    correct_answer = row.get("correct_answer")
    if row.get("continuation_outcome") == "corrected":
        return True
    if answers_match(final_boxed, correct_answer):
        return True
    if final_boxed and nums_match(final_boxed, correct_answer):
        return True
    return normalized_contains_answer(str(row.get("continuation_text") or ""), correct_answer)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(key: str, rows: List[Dict[str, Any]], cap_tokens: int) -> Dict[str, Any]:
    tokens = [int(row.get("generated_tokens") or 0) for row in rows]
    ref_counts = [int(row.get("reflection_keyword_count") or 0) for row in rows]
    benchmark_flags = [bool(row.get("benchmark_correct")) for row in rows]
    correct = sum(benchmark_flags)
    length_caps = sum(1 for value in tokens if value >= cap_tokens)
    return {
        "condition_key": key,
        "n": len(rows),
        "outcomes": dict(Counter(row.get("continuation_outcome") for row in rows)),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "benchmark_correct": correct,
        "benchmark_correct_rate": correct / len(rows) if rows else 0.0,
        "mean_generated_tokens": round(sum(tokens) / len(tokens), 2) if tokens else 0.0,
        "median_generated_tokens": statistics.median(tokens) if tokens else 0,
        "mean_reflection_keyword_count": round(sum(ref_counts) / len(ref_counts), 2) if ref_counts else 0.0,
        "has_reflection_rate": sum(1 for row in rows if row.get("has_reflection")) / len(rows) if rows else 0.0,
        "first_wait_rate": sum(1 for row in rows if row.get("first_wait")) / len(rows) if rows else 0.0,
        "length_cap": length_caps,
        "length_cap_rate": length_caps / len(rows) if rows else 0.0,
    }


def summarize_repeats(by_repeat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in by_repeat:
        by_condition[str(row["condition_key"])].append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(by_condition):
        rows = by_condition[key]
        accuracies = [float(row.get("accuracy") or 0.0) for row in rows]
        token_means = [float(row.get("mean_generated_tokens") or 0.0) for row in rows]
        cap_rates = [float(row.get("length_cap_rate") or 0.0) for row in rows]
        out.append(
            {
                "condition_key": key,
                "n_repeats": len(rows),
                "total_n": sum(int(row.get("n") or 0) for row in rows),
                "mean_accuracy": round(sum(accuracies) / len(accuracies), 6) if accuracies else 0.0,
                "std_accuracy": round(statistics.stdev(accuracies), 6) if len(accuracies) > 1 else 0.0,
                "min_accuracy": min(accuracies) if accuracies else 0.0,
                "max_accuracy": max(accuracies) if accuracies else 0.0,
                "mean_generated_tokens": round(sum(token_means) / len(token_means), 2) if token_means else 0.0,
                "mean_length_cap_rate": round(sum(cap_rates) / len(cap_rates), 6) if cap_rates else 0.0,
            }
        )
    out.sort(key=lambda r: (r["condition_key"] != "baseline", str(r["condition_key"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_partial", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cap_tokens", type=int, default=4096)
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(root, include_partial=args.include_partial)
    by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[condition_key(row)].append(row)

    summary = [summarize(key, by_key[key], args.cap_tokens) for key in sorted(by_key)]
    summary.sort(key=lambda r: (r["condition_key"] != "baseline", str(r["condition_key"])))

    by_repeat_key: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        repeat = repeat_key(row)
        if repeat is not None:
            by_repeat_key[(condition_key(row), repeat)].append(row)
    repeat_summary = [
        {"repeat_idx": repeat, **summarize(key, repeat_rows, args.cap_tokens)}
        for (key, repeat), repeat_rows in sorted(by_repeat_key.items(), key=lambda item: (item[0][0] != "baseline", str(item[0][0]), str(item[0][1])))
    ]
    repeat_aggregate = summarize_repeats(repeat_summary) if repeat_summary else []

    compact_rows = []
    for row in rows:
        compact = {key: value for key, value in row.items() if key != "continuation_text"}
        compact["condition_key"] = condition_key(row)
        compact["continuation_text_chars"] = len(row.get("continuation_text") or "")
        compact_rows.append(compact)

    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "rows.csv", compact_rows)
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if repeat_summary:
        write_csv(output_dir / "summary_by_repeat.csv", repeat_summary)
        (output_dir / "summary_by_repeat.json").write_text(json.dumps(repeat_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(output_dir / "repeat_aggregate.csv", repeat_aggregate)
        (output_dir / "repeat_aggregate.json").write_text(json.dumps(repeat_aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
