#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and rank reflection-gate steering summaries.")
    parser.add_argument("--root", required=True, help="Directory containing one or more steering run output dirs.")
    parser.add_argument("--output_dir", required=True, help="Directory for merged CSV summaries.")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def finite(value: float) -> bool:
    return math.isfinite(value)


def add_run_meta(rows: Iterable[Dict[str, str]], run_dir: Path, root: Path) -> List[Dict[str, Any]]:
    config = load_json(run_dir / "run_config.json")
    summary = load_json(run_dir / "summary.json")
    rel = str(run_dir.relative_to(root)) if run_dir != root else run_dir.name
    out: List[Dict[str, Any]] = []
    for row in rows:
        enriched: Dict[str, Any] = {
            "run": rel,
            "run_dir": str(run_dir),
            "model_name_or_path": config.get("model_name_or_path", ""),
            "usable_examples": summary.get("usable_examples", ""),
            **row,
        }
        out.append(enriched)
    return out


def run_dirs(root: Path) -> List[Path]:
    dirs = set()
    for path in root.rglob("steering_summary.csv"):
        dirs.add(path.parent)
    for path in root.rglob("baseline_summary.csv"):
        dirs.add(path.parent)
    return sorted(dirs)


def key_for(row: Dict[str, Any], *, direction_type: Optional[str] = None) -> Tuple[Any, ...]:
    return (
        row.get("run"),
        row.get("layer_idx"),
        row.get("site"),
        row.get("steer_condition"),
        row.get("alpha"),
        direction_type if direction_type is not None else row.get("direction_type"),
    )


def build_index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    return {key_for(row): row for row in rows}


def control_row(
    index: Dict[Tuple[Any, ...], Dict[str, Any]],
    row: Dict[str, Any],
    direction_type: str,
) -> Optional[Dict[str, Any]]:
    key = key_for(row, direction_type=direction_type)
    return index.get(key)


def delta_wait(row: Optional[Dict[str, Any]]) -> float:
    if row is None:
        return float("nan")
    return to_float(row.get("delta_first_wait_rate"))


def delta_rvs(row: Optional[Dict[str, Any]]) -> float:
    if row is None:
        return float("nan")
    return to_float(row.get("mean_delta_reflect_vs_stop"))


def attach_control_deltas(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = build_index(rows)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("direction_type") != "gate":
            continue
        random = control_row(index, row, "random")
        logit = control_row(index, row, "logit")
        gate_dw = delta_wait(row)
        random_dw = delta_wait(random)
        logit_dw = delta_wait(logit)
        gate_dr = delta_rvs(row)
        random_dr = delta_rvs(random)
        logit_dr = delta_rvs(logit)
        enriched = {
            **row,
            "gate_delta_first_wait_rate": gate_dw,
            "random_delta_first_wait_rate": random_dw,
            "logit_delta_first_wait_rate": logit_dw,
            "gate_minus_random_delta_first_wait": gate_dw - random_dw if finite(random_dw) else "",
            "gate_minus_logit_delta_first_wait": gate_dw - logit_dw if finite(logit_dw) else "",
            "gate_delta_rvs": gate_dr,
            "random_delta_rvs": random_dr,
            "logit_delta_rvs": logit_dr,
            "gate_minus_random_delta_rvs": gate_dr - random_dr if finite(random_dr) else "",
            "gate_minus_logit_delta_rvs": gate_dr - logit_dr if finite(logit_dr) else "",
        }
        out.append(enriched)
    return out


def condition_lookup(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    return {
        (
            row.get("run"),
            row.get("layer_idx"),
            row.get("site"),
            row.get("alpha"),
            row.get("direction_type"),
            row.get("steer_condition"),
        ): row
        for row in rows
    }


def add_clean_contrast(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = condition_lookup(rows)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("direction_type") != "gate":
            continue
        if row.get("steer_condition") != "coherent_plus":
            continue
        clean = index.get(
            (
                row.get("run"),
                row.get("layer_idx"),
                row.get("site"),
                row.get("alpha"),
                row.get("direction_type"),
                "clean_plus",
            )
        )
        coherent_dw = delta_wait(row)
        clean_dw = delta_wait(clean)
        out.append(
            {
                **row,
                "coherent_delta_first_wait_rate": coherent_dw,
                "clean_delta_first_wait_rate": clean_dw,
                "coherent_minus_clean_delta_first_wait": coherent_dw - clean_dw if finite(clean_dw) else "",
                "coherent_delta_rvs": delta_rvs(row),
                "clean_delta_rvs": delta_rvs(clean),
                "coherent_minus_clean_delta_rvs": delta_rvs(row) - delta_rvs(clean) if finite(delta_rvs(clean)) else "",
            }
        )
    return out


def sort_opening(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        to_float(row.get("coherent_delta_first_wait_rate")),
        to_float(row.get("coherent_minus_clean_delta_first_wait")),
        to_float(row.get("mean_delta_reflect_vs_stop")),
    )


def sort_closing(row: Dict[str, Any]) -> Tuple[float, float, float]:
    gate_close = -to_float(row.get("gate_delta_first_wait_rate"))
    random_close = -to_float(row.get("random_delta_first_wait_rate"))
    logit_close = -to_float(row.get("logit_delta_first_wait_rate"))
    best_control = max([x for x in [random_close, logit_close] if finite(x)] or [float("nan")])
    control_adjusted = gate_close - best_control if finite(best_control) else float("nan")
    return (gate_close, control_adjusted, -to_float(row.get("gate_delta_rvs")))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    dirs = run_dirs(root)

    baseline_rows: List[Dict[str, Any]] = []
    steering_rows: List[Dict[str, Any]] = []
    for run_dir in dirs:
        baseline_rows.extend(add_run_meta(read_csv(run_dir / "baseline_summary.csv"), run_dir, root))
        steering_rows.extend(add_run_meta(read_csv(run_dir / "steering_summary.csv"), run_dir, root))

    write_csv(output_dir / "combined_baseline_summary.csv", baseline_rows)
    write_csv(output_dir / "combined_steering_summary.csv", steering_rows)

    gate_with_controls = attach_control_deltas(steering_rows)
    write_csv(output_dir / "gate_control_comparison.csv", gate_with_controls)

    opening = add_clean_contrast(steering_rows)
    opening.sort(key=sort_opening, reverse=True)
    write_csv(output_dir / "ranked_coherent_opening.csv", opening)

    closing = [row for row in gate_with_controls if row.get("steer_condition") == "tamper_minus"]
    closing.sort(key=sort_closing, reverse=True)
    write_csv(output_dir / "ranked_tamper_closing.csv", closing)

    summary = {
        "root": str(root),
        "run_dirs": len(dirs),
        "baseline_rows": len(baseline_rows),
        "steering_rows": len(steering_rows),
        "top_coherent_opening": opening[:5],
        "top_tamper_closing": closing[:5],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] merged {len(dirs)} run dirs")
    print(f"- output_dir: {output_dir}")
    print(f"- steering rows: {len(steering_rows)}")


if __name__ == "__main__":
    main()
