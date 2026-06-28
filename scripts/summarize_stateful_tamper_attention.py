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

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stateful tamper behavior and attention chunk outputs.")
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_slug", type=str, default="")
    return parser.parse_args()


def read_all_jsonl(root: Path, filename: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob(f"**/{filename}")):
        for row in load_jsonl(path):
            row = dict(row)
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def mean(values: Iterable[float]) -> float:
    vals: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(x):
            continue
        vals.append(x)
    return sum(vals) / len(vals) if vals else float("nan")


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None and not math.isnan(float(x)) and not math.isnan(float(y))
    ]
    if len(pairs) < 3:
        return float("nan")
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, _ in pairs)
    vy = sum((y - my) ** 2 for _, y in pairs)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def summarize_behavior(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition"))].append(row)
    table: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"total_rows": len(rows), "conditions": {}}
    for condition, group in sorted(grouped.items()):
        first_counts: Dict[str, int] = defaultdict(int)
        for row in group:
            first_counts[str(row.get("first_generated_token_text") or "")] += 1
        out = {
            "condition": condition,
            "count": len(group),
            "example_count": len({str(row.get("example_id")) for row in group}),
            "first_wait_rate": mean([1.0 if str(row.get("first_generated_token_text")) in WAIT_FIRST_TOKENS else 0.0 for row in group]),
            "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
            "mean_reflect_vs_stop": mean([float(row.get("reflect_vs_stop", float("nan"))) for row in group]),
            "mean_wrong_answer_token_len": mean([float(row.get("wrong_answer_token_len", 0.0)) for row in group]),
            "correct_full_rate": mean([1.0 if row.get("outcome_full_text") in {"correct", "corrected"} else 0.0 for row in group]),
            "keep_wrong_full_rate": mean([1.0 if row.get("outcome_full_text") == "keep_wrong" else 0.0 for row in group]),
            "first_token_counts_json": json.dumps(dict(sorted(first_counts.items(), key=lambda item: -item[1])[:20]), ensure_ascii=False),
        }
        table.append(out)
        summary["conditions"][condition] = out
    return table, summary


def add_behavior_labels(attention_rows: List[Dict[str, Any]], behavior_rows: List[Dict[str, Any]]) -> None:
    lookup = {
        (str(row.get("example_id")), str(row.get("condition"))): row
        for row in behavior_rows
    }
    for row in attention_rows:
        key = (str(row.get("example_id")), str(row.get("condition")))
        behavior = lookup.get(key, {})
        first_token = str(behavior.get("first_generated_token_text") or "")
        row["first_generated_token_text"] = first_token
        row["first_is_wait"] = first_token in WAIT_FIRST_TOKENS
        row["has_reflection"] = bool(behavior.get("has_reflection"))
        try:
            row["reflect_vs_stop"] = float(behavior.get("reflect_vs_stop"))
        except (TypeError, ValueError):
            row["reflect_vs_stop"] = float("nan")
        row["outcome_full_text"] = behavior.get("outcome_full_text")


def summarize_attention_by_condition(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    mass_keys = sorted(k for k in rows[0].keys() if k.startswith("mass_"))
    grouped: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("condition")),
                str(row.get("stage")),
                int(row.get("layer_idx")),
                int(row.get("head_idx")),
            )
        ].append(row)
    out_rows: List[Dict[str, Any]] = []
    for (condition, stage, layer_idx, head_idx), group in grouped.items():
        out: Dict[str, Any] = {
            "condition": condition,
            "stage": stage,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "head_label": f"L{layer_idx}H{head_idx}",
            "count": len(group),
            "mean_attention_entropy": mean([float(row.get("attention_entropy", float("nan"))) for row in group]),
            "mean_reflect_vs_stop": mean([float(row.get("reflect_vs_stop", float("nan"))) for row in group]),
            "first_wait_rate": mean([1.0 if row.get("first_is_wait") else 0.0 for row in group]),
        }
        for key in mass_keys:
            out[f"mean_{key}"] = mean([float(row.get(key, 0.0)) for row in group])
            out[f"corr_{key}_reflect_vs_stop"] = pearson(
                [float(row.get(key, 0.0)) for row in group],
                [float(row.get("reflect_vs_stop", float("nan"))) for row in group],
            )
        out_rows.append(out)
    out_rows.sort(
        key=lambda row: (
            str(row["condition"]),
            str(row["stage"]),
            -float(row.get("mean_mass_forced_box_full", 0.0)),
        )
    )
    return out_rows


def summarize_tamper_wait_split(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tamper_rows = [row for row in rows if row.get("condition") == "tamper"]
    if not tamper_rows:
        return []
    mass_keys = sorted(k for k in tamper_rows[0].keys() if k.startswith("mass_"))
    grouped: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in tamper_rows:
        grouped[(str(row.get("stage")), int(row.get("layer_idx")), int(row.get("head_idx")))].append(row)
    out_rows: List[Dict[str, Any]] = []
    for (stage, layer_idx, head_idx), group in grouped.items():
        wait_group = [row for row in group if row.get("first_is_wait")]
        stop_group = [row for row in group if not row.get("first_is_wait")]
        out: Dict[str, Any] = {
            "stage": stage,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "head_label": f"L{layer_idx}H{head_idx}",
            "count": len(group),
            "wait_count": len(wait_group),
            "stop_count": len(stop_group),
            "mean_reflect_vs_stop": mean([float(row.get("reflect_vs_stop", float("nan"))) for row in group]),
        }
        for key in mass_keys:
            wait_mean = mean([float(row.get(key, 0.0)) for row in wait_group])
            stop_mean = mean([float(row.get(key, 0.0)) for row in stop_group])
            out[f"wait_mean_{key}"] = wait_mean
            out[f"stop_mean_{key}"] = stop_mean
            out[f"delta_wait_minus_stop_{key}"] = wait_mean - stop_mean
            out[f"corr_{key}_reflect_vs_stop"] = pearson(
                [float(row.get(key, 0.0)) for row in group],
                [float(row.get("reflect_vs_stop", float("nan"))) for row in group],
            )
        out_rows.append(out)
    out_rows.sort(
        key=lambda row: (
            -abs(float(row.get("delta_wait_minus_stop_mass_forced_answer", 0.0))),
            -abs(float(row.get("corr_mass_forced_answer_reflect_vs_stop", 0.0)))
            if not math.isnan(float(row.get("corr_mass_forced_answer_reflect_vs_stop", float("nan"))))
            else 0.0,
        )
    )
    return out_rows


def summarize_tamper_clean_delta(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_condition = summarize_attention_by_condition(rows)
    keyed = {
        (str(row["stage"]), int(row["layer_idx"]), int(row["head_idx"]), str(row["condition"])): row
        for row in by_condition
    }
    mass_keys = sorted(k[len("mean_"):] for k in by_condition[0].keys() if k.startswith("mean_mass_")) if by_condition else []
    out_rows: List[Dict[str, Any]] = []
    base_keys = {(stage, layer, head) for stage, layer, head, _condition in keyed.keys()}
    for stage, layer_idx, head_idx in sorted(base_keys):
        tamper = keyed.get((stage, layer_idx, head_idx, "tamper"))
        clean = keyed.get((stage, layer_idx, head_idx, "clean_force"))
        if not tamper or not clean:
            continue
        out: Dict[str, Any] = {
            "stage": stage,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "head_label": f"L{layer_idx}H{head_idx}",
            "tamper_count": tamper.get("count"),
            "clean_count": clean.get("count"),
            "tamper_first_wait_rate": tamper.get("first_wait_rate"),
            "clean_first_wait_rate": clean.get("first_wait_rate"),
            "delta_reflect_vs_stop": float(tamper.get("mean_reflect_vs_stop", 0.0))
            - float(clean.get("mean_reflect_vs_stop", 0.0)),
        }
        for key in mass_keys:
            t = float(tamper.get(f"mean_{key}", 0.0))
            c = float(clean.get(f"mean_{key}", 0.0))
            out[f"tamper_mean_{key}"] = t
            out[f"clean_mean_{key}"] = c
            out[f"delta_tamper_minus_clean_{key}"] = t - c
        out_rows.append(out)
    out_rows.sort(
        key=lambda row: (
            -abs(float(row.get("delta_tamper_minus_clean_mass_forced_answer", 0.0))),
            -abs(float(row.get("delta_tamper_minus_clean_mass_forced_box_full", 0.0))),
        )
    )
    return out_rows


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    behavior_rows = read_all_jsonl(input_root, "behavior_rows.jsonl")
    attention_rows = read_all_jsonl(input_root, "attention_after_forced_box_rows.jsonl")
    skipped_rows = read_all_jsonl(input_root, "skipped_rows.jsonl")
    add_behavior_labels(attention_rows, behavior_rows)

    dump_jsonl(output_dir / "merged_behavior_rows.jsonl", behavior_rows)
    dump_jsonl(output_dir / "merged_attention_after_forced_box_rows.jsonl", attention_rows)
    dump_jsonl(output_dir / "merged_skipped_rows.jsonl", skipped_rows)

    behavior_table, behavior_summary = summarize_behavior(behavior_rows)
    write_csv(output_dir / "behavior_summary.csv", behavior_table)
    write_json(
        output_dir / "summary.json",
        {
            "model_slug": args.model_slug,
            "input_root": str(input_root),
            "behavior": behavior_summary,
            "attention_rows": len(attention_rows),
            "skipped_rows": len(skipped_rows),
        },
    )
    write_csv(output_dir / "head_attention_by_condition.csv", summarize_attention_by_condition(attention_rows))
    write_csv(output_dir / "head_tamper_wait_vs_stop.csv", summarize_tamper_wait_split(attention_rows))
    write_csv(output_dir / "head_tamper_minus_clean_delta.csv", summarize_tamper_clean_delta(attention_rows))

    print("[Done] Summary written.")
    print(f"- behavior_rows: {len(behavior_rows)}")
    print(f"- attention_rows: {len(attention_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")
    print(f"- output_dir: {output_dir}")


if __name__ == "__main__":
    main()
