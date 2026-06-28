#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_head_locality_module():
    module_path = ROOT_DIR / "cot_research" / "head_locality.py"
    spec = importlib.util.spec_from_file_location("head_locality_standalone", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEAD_LOCALITY = _load_head_locality_module()
classify_head_locality = HEAD_LOCALITY.classify_head_locality
summarize_locality_rows = HEAD_LOCALITY.summarize_locality_rows
build_locality_report = HEAD_LOCALITY.build_locality_report


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Classify attention heads into local/global groups from summary CSV.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(root_dir / "experiment_results" / "experiments" / "qwen3_1p7b_prev1_heads_20260407" / "merged" / "head_prev_attention_summary.csv"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "head_locality_classification" / "qwen3_1p7b"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--local_near_window", type=int, default=16)
    parser.add_argument("--local_mass_threshold", type=float, default=0.8)
    parser.add_argument("--self_mass_threshold", type=float, default=0.5)
    parser.add_argument("--prev_local_threshold", type=float, default=0.5)
    parser.add_argument("--criterion", type=str, default="near_total", choices=["near_total", "prev_only"], help="Use self+prev_window or only prev_window mass for local/global classification.")
    return parser.parse_args()


def _coerce_value(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return ""
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except Exception:
        return text


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {str(key): _coerce_value(value) for key, value in raw_row.items()}
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    classified = [
        classify_head_locality(
            row,
            local_near_window=args.local_near_window,
            local_mass_threshold=args.local_mass_threshold,
            self_mass_threshold=args.self_mass_threshold,
            prev_local_threshold=args.prev_local_threshold,
            criterion=args.criterion,
        )
        for row in rows
    ]
    classified.sort(
        key=lambda item: (
            str(item.get("locality_label") or ""),
            -float(item.get("near_total_mass", 0.0)),
            -float(item.get("mean_prev_mass_w16", 0.0)),
        )
    )
    summary = summarize_locality_rows(classified)

    write_csv(output_dir / "head_locality_classification.csv", classified)
    write_csv(output_dir / "local_heads.csv", [row for row in classified if str(row.get("locality_label")) == "local"])
    write_csv(output_dir / "global_heads.csv", [row for row in classified if str(row.get("locality_label")) == "global"])
    write_csv(output_dir / "self_local_heads.csv", [row for row in classified if str(row.get("locality_subtype")) == "self_local"])
    write_csv(output_dir / "recent_local_heads.csv", [row for row in classified if str(row.get("locality_subtype")) == "recent_local"])
    write_csv(output_dir / "mixed_local_heads.csv", [row for row in classified if str(row.get("locality_subtype")) == "mixed_local"])
    write_json(
        output_dir / "summary.json",
        {
            "args": vars(args),
            "summary": summary,
        },
    )
    report = build_locality_report(
        model_name_or_path=args.model_name_or_path,
        source_summary_csv=str(input_csv),
        rows=classified,
        summary=summary,
        local_near_window=args.local_near_window,
        local_mass_threshold=args.local_mass_threshold,
        self_mass_threshold=args.self_mass_threshold,
        prev_local_threshold=args.prev_local_threshold,
        criterion=args.criterion,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    print("[Done] head locality classification finished:")
    print(f"- output_dir: {output_dir}")
    print(f"- head_count: {len(classified)}")
    print(f"- label_counts: {summary.get('label_counts', {})}")


if __name__ == "__main__":
    main()
