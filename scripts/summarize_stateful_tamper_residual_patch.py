#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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


def summarize_patch_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("direction")), int(row.get("layer_idx")))].append(row)
    out: List[Dict[str, Any]] = []
    for (direction, layer_idx), group in grouped.items():
        baseline_wait = [str(row.get("baseline_first_generated_token_text") or "") in WAIT_FIRST_TOKENS for row in group]
        patched_wait = [str(row.get("patched_first_generated_token_text") or "") in WAIT_FIRST_TOKENS for row in group]
        out.append(
            {
                "direction": direction,
                "layer_idx": layer_idx,
                "count": len(group),
                "baseline_first_wait_rate": mean([1.0 if x else 0.0 for x in baseline_wait]),
                "patched_first_wait_rate": mean([1.0 if x else 0.0 for x in patched_wait]),
                "delta_first_wait_rate": mean([1.0 if x else 0.0 for x in patched_wait])
                - mean([1.0 if x else 0.0 for x in baseline_wait]),
                "wait_to_nonwait_rate": mean([1.0 if b and not p else 0.0 for b, p in zip(baseline_wait, patched_wait)]),
                "nonwait_to_wait_rate": mean([1.0 if (not b) and p else 0.0 for b, p in zip(baseline_wait, patched_wait)]),
                "mean_baseline_reflect_vs_stop": mean([row.get("baseline_reflect_vs_stop") for row in group]),
                "mean_patched_reflect_vs_stop": mean([row.get("patched_reflect_vs_stop") for row in group]),
                "mean_delta_reflect_vs_stop": mean([row.get("delta_patched_minus_baseline") for row in group]),
                "baseline_has_reflection_rate": mean([1.0 if row.get("baseline_has_reflection") else 0.0 for row in group]),
                "patched_has_reflection_rate": mean([1.0 if row.get("patched_has_reflection") else 0.0 for row in group]),
                "mean_patch_delta_norm": mean([row.get("patch_delta_norm") for row in group]),
                "patch_hook_call_rate": mean([1.0 if int(row.get("patch_hook_call_count", 0)) > 0 else 0.0 for row in group]),
            }
        )
    out.sort(key=lambda row: (str(row["direction"]), -abs(float(row["delta_first_wait_rate"])), int(row["layer_idx"])))
    return out


def summarize_baselines(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition"))].append(row)
    out: List[Dict[str, Any]] = []
    for condition, group in sorted(grouped.items()):
        first_wait = [str(row.get("first_generated_token_text") or "") in WAIT_FIRST_TOKENS for row in group]
        out.append(
            {
                "condition": condition,
                "count": len(group),
                "example_count": len({str(row.get("example_id")) for row in group}),
                "first_wait_rate": mean([1.0 if x else 0.0 for x in first_wait]),
                "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
                "mean_reflect_vs_stop": mean([row.get("reflect_vs_stop") for row in group]),
            }
        )
    return out


def read_all_jsonl(root: Path, filename: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob(f"**/{filename}")):
        for row in load_jsonl(path):
            row = dict(row)
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize residual patch shard outputs.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_slug", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_rows = read_all_jsonl(input_root, "residual_patch_rows.jsonl")
    baseline_rows = read_all_jsonl(input_root, "baseline_rows.jsonl")
    skipped_rows = read_all_jsonl(input_root, "skipped_rows.jsonl")
    patch_summary = summarize_patch_rows(patch_rows)
    baseline_summary = summarize_baselines(baseline_rows)

    dump_jsonl(output_dir / "merged_residual_patch_rows.jsonl", patch_rows)
    dump_jsonl(output_dir / "merged_baseline_rows.jsonl", baseline_rows)
    dump_jsonl(output_dir / "merged_skipped_rows.jsonl", skipped_rows)
    write_csv(output_dir / "residual_patch_summary.csv", patch_summary)
    write_csv(output_dir / "baseline_summary.csv", baseline_summary)
    write_json(
        output_dir / "summary.json",
        {
            "model_slug": args.model_slug,
            "input_root": str(input_root),
            "patch_rows": len(patch_rows),
            "baseline_rows": len(baseline_rows),
            "skipped_rows": len(skipped_rows),
            "summary_rows": len(patch_summary),
        },
    )
    print("[Done] Residual patch summary written.")
    print(f"- patch_rows: {len(patch_rows)}")
    print(f"- baseline_rows: {len(baseline_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")
    print(f"- output_dir: {output_dir}")


if __name__ == "__main__":
    main()
