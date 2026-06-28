#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

WAIT_FIRST_TOKENS = {"Wait", " wait", " Wait", "wait", "Actually", " Actually", "No", " No"}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[Any]) -> float:
    vals: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(x):
            vals.append(x)
    return sum(vals) / len(vals) if vals else float("nan")


def bool_mean(values: Iterable[Any]) -> float:
    return mean(1.0 if bool(v) else 0.0 for v in values)


def first_is_wait(text: Any) -> bool:
    return str(text or "") in WAIT_FIRST_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stateful tamper head ablation rows.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_slug", default="")
    return parser.parse_args()


def read_all_ablation_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("**/head_ablation_rows.jsonl")):
        for row in load_jsonl(path):
            row = dict(row)
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def summarize_by_head(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("condition")), str(row.get("head_label")))].append(row)
    out: List[Dict[str, Any]] = []
    for (condition, head_label), group in grouped.items():
        baseline_wait = [first_is_wait(row.get("baseline_first_generated_token_text")) for row in group]
        ablated_wait = [first_is_wait(row.get("ablated_first_generated_token_text")) for row in group]
        wait_to_nonwait = [b and not a for b, a in zip(baseline_wait, ablated_wait)]
        nonwait_to_wait = [(not b) and a for b, a in zip(baseline_wait, ablated_wait)]
        ablated_text_available = ["ablated_first_generated_token_text" in row for row in group]
        out.append(
            {
                "condition": condition,
                "head_label": head_label,
                "count": len(group),
                "example_count": len({str(row.get("example_id")) for row in group}),
                "mean_baseline_reflect_vs_stop": mean(row.get("baseline_reflect_vs_stop") for row in group),
                "mean_ablated_reflect_vs_stop": mean(row.get("ablated_reflect_vs_stop") for row in group),
                "mean_delta_ablated_minus_baseline": mean(row.get("delta_ablated_minus_baseline") for row in group),
                "baseline_first_wait_rate": mean(1.0 if x else 0.0 for x in baseline_wait),
                "ablated_first_wait_rate": mean(1.0 if x else 0.0 for x in ablated_wait),
                "delta_first_wait_rate": mean(1.0 if x else 0.0 for x in ablated_wait)
                - mean(1.0 if x else 0.0 for x in baseline_wait),
                "wait_to_nonwait_rate": mean(1.0 if x else 0.0 for x in wait_to_nonwait),
                "nonwait_to_wait_rate": mean(1.0 if x else 0.0 for x in nonwait_to_wait),
                "baseline_has_reflection_rate": bool_mean(row.get("baseline_has_reflection") for row in group),
                "ablated_has_reflection_rate": bool_mean(row.get("ablated_has_reflection") for row in group),
                "delta_has_reflection_rate": bool_mean(row.get("ablated_has_reflection") for row in group)
                - bool_mean(row.get("baseline_has_reflection") for row in group),
                "hook_call_rate": mean(1.0 if int(row.get("hook_call_count", 0)) > 0 else 0.0 for row in group),
                "mean_hook_abs_before": mean(row.get("hook_abs_mean_before") for row in group),
                "ablated_continuation_available_rate": mean(1.0 if x else 0.0 for x in ablated_text_available),
            }
        )
    out.sort(
        key=lambda row: (
            -abs(float(row.get("delta_first_wait_rate", 0.0))),
            -abs(float(row.get("mean_delta_ablated_minus_baseline", 0.0))),
            str(row.get("head_label")),
        )
    )
    return out


def summarize_transitions(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        baseline_token = str(row.get("baseline_first_generated_token_text") or "")
        ablated_token = str(row.get("ablated_first_generated_token_text") or "")
        if baseline_token != ablated_token:
            out.append(
                {
                    "example_id": row.get("example_id"),
                    "condition": row.get("condition"),
                    "head_label": row.get("head_label"),
                    "baseline_first_generated_token_text": baseline_token,
                    "ablated_first_generated_token_text": ablated_token,
                    "baseline_reflect_vs_stop": row.get("baseline_reflect_vs_stop"),
                    "ablated_reflect_vs_stop": row.get("ablated_reflect_vs_stop"),
                    "delta_ablated_minus_baseline": row.get("delta_ablated_minus_baseline"),
                    "baseline_has_reflection": row.get("baseline_has_reflection"),
                    "ablated_has_reflection": row.get("ablated_has_reflection"),
                    "source": row.get("_source_file"),
                }
            )
    out.sort(key=lambda row: (str(row.get("head_label")), str(row.get("example_id"))))
    return out


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_all_ablation_rows(input_root)
    dump_jsonl(output_dir / "merged_head_ablation_rows.jsonl", rows)
    head_summary = summarize_by_head(rows)
    transitions = summarize_transitions(rows)
    write_csv(output_dir / "head_ablation_summary.csv", head_summary)
    write_csv(output_dir / "first_token_transitions.csv", transitions)
    write_json(
        output_dir / "summary.json",
        {
            "model_slug": args.model_slug,
            "input_root": str(input_root),
            "ablation_rows": len(rows),
            "head_count": len(head_summary),
            "transition_count": len(transitions),
        },
    )
    print("[Done] Ablation summary written.")
    print(f"- ablation_rows: {len(rows)}")
    print(f"- head_count: {len(head_summary)}")
    print(f"- transition_count: {len(transitions)}")
    print(f"- output_dir: {output_dir}")


if __name__ == "__main__":
    main()
