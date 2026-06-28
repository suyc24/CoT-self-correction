#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.io_utils import write_json
from analyze_hidden_trajectory_movie import (
    first_pass,
    iter_activation_paths,
    load_behavior,
    load_logit_rows,
    second_pass,
    summarize_logits,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate patch-rescue/delete activations into behavior, logit, and hidden metrics."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--write_detail_rows", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_positions", type=int, default=64)
    return parser.parse_args()


def mean(values: Sequence[Any]) -> float:
    xs: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            xs.append(x)
    return sum(xs) / len(xs) if xs else float("nan")


def summarize_patch_behavior(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("condition_kind", "baseline")),
                str(row.get("condition")),
                str(row.get("patch_type", "")),
                str(row.get("patch_layer_idx", "")),
                str(row.get("patch_site", "")),
                str(row.get("patch_timing", "")),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (kind, condition, patch_type, layer_idx, site, timing), group in sorted(grouped.items()):
        out.append(
            {
                "condition_kind": kind,
                "condition": condition,
                "patch_type": patch_type,
                "patch_layer_idx": layer_idx,
                "patch_site": site,
                "patch_timing": timing,
                "base_condition": group[0].get("base_condition", ""),
                "source_condition": group[0].get("source_condition", ""),
                "gate_mode": group[0].get("gate_mode", ""),
                "sweep": group[0].get("sweep", ""),
                "count": len(group),
                "first_wait_rate": mean([1.0 if row.get("first_wait") else 0.0 for row in group]),
                "surface_reflection_rate": mean([1.0 if row.get("surface_reflection") else 0.0 for row in group]),
                "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
                "semantic_repair_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
                "functional_repair_rate": mean([1.0 if row.get("functional_repair") else 0.0 for row in group]),
                "final_correction_rate": mean([1.0 if row.get("final_correction") else 0.0 for row in group]),
                "mean_generated_tokens": mean([row.get("generated_tokens") for row in group]),
                "cap_rate": mean([1.0 if row.get("hit_max_new_tokens") else 0.0 for row in group]),
                "mean_p0_reflect_vs_stop": mean([row.get("p0_reflect_vs_stop") for row in group]),
                "mean_patch_hook_calls": mean([row.get("patch_hook_calls") for row in group]),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activation_paths = iter_activation_paths(root)
    if not activation_paths:
        raise ValueError(f"No activation traces found under {root}")

    behavior_rows, behavior_lookup = load_behavior(root)
    logit_rows = load_logit_rows(root)
    directions, centroids, probes, stats = first_pass(
        activation_paths,
        behavior_lookup,
        int(args.max_positions),
    )
    metric_summary, metric_detail = second_pass(
        activation_paths,
        directions=directions,
        centroids=centroids,
        probes=probes,
        max_positions=int(args.max_positions),
        write_detail_rows=bool(args.write_detail_rows),
    )

    write_csv(output_dir / "behavior_summary.csv", summarize_patch_behavior(behavior_rows))
    write_csv(output_dir / "logit_summary.csv", summarize_logits(logit_rows))
    write_csv(output_dir / "hidden_metric_summary.csv", metric_summary)
    if args.write_detail_rows:
        write_csv(output_dir / "hidden_metric_rows.csv", metric_detail)
    write_csv(
        output_dir / "gate_direction_rows.csv",
        [
            {"layer_idx": key[0], "site": key[1], "direction_norm": float(vec.norm().item())}
            for key, vec in sorted(directions.items())
        ],
    )
    write_csv(
        output_dir / "repair_probe_rows.csv",
        [
            {
                "mode": key[0],
                "layer_idx": key[1],
                "site": key[2],
                "position_index": key[3],
                "probe_direction_norm": float(probe[:-1].norm().item()),
                "probe_threshold": float(probe[-1].item()),
            }
            for key, probe in sorted(probes.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
        ],
    )
    write_json(
        output_dir / "summary.json",
        {
            **stats,
            "behavior_rows": len(behavior_rows),
            "logit_rows": len(logit_rows),
            "metric_summary_rows": len(metric_summary),
            "metric_detail_rows": len(metric_detail),
            "gate_directions": len(directions),
            "repair_probes": len(probes),
            "activation_root": str(root),
        },
    )
    print("[Done] Patch rescue/delete analysis finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- activation_files: {len(activation_paths)}")
    print(f"- hidden_metric_summary_rows: {len(metric_summary)}")


if __name__ == "__main__":
    main()
