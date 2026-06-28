#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.hidden_trajectory import tensor_normed
from cot_research.io_utils import load_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate hidden trajectory movie activations into gate/centroid/probe metrics."
    )
    parser.add_argument("--root", required=True, help="Experiment root containing shards or activation_traces.")
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


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_activation_paths(root: Path) -> List[Path]:
    return sorted(root.glob("**/activation_traces/*.pt"))


def iter_jsonl(root: Path, name: str) -> List[Path]:
    return sorted(root.glob(f"**/{name}"))


def parse_activation_key(key: str) -> Tuple[int, str]:
    layer_text, site = key.split("/", 1)
    return int(layer_text.lstrip("L")), site


def add_vec(agg: Dict[Any, Dict[str, Any]], key: Any, vec: torch.Tensor) -> None:
    vec = vec.detach().float().cpu()
    item = agg.setdefault(key, {"count": 0, "sum": torch.zeros_like(vec)})
    item["count"] += 1
    item["sum"] += vec


def finalize_vec_agg(agg: Dict[Any, Dict[str, Any]]) -> Dict[Any, torch.Tensor]:
    out: Dict[Any, torch.Tensor] = {}
    for key, item in agg.items():
        count = int(item["count"])
        if count > 0:
            out[key] = item["sum"] / float(count)
    return out


def add_scalar(agg: Dict[Any, Dict[str, Any]], key: Any, values: Dict[str, float]) -> None:
    item = agg.setdefault(key, {"count": 0, "sums": defaultdict(float)})
    item["count"] += 1
    for name, value in values.items():
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            item["sums"][name] += x


def finalize_scalar_rows(agg: Dict[Any, Dict[str, Any]], key_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, item in sorted(agg.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        count = int(item["count"])
        row = {name: value for name, value in zip(key_names, key)}
        row["count"] = count
        for name, total in item["sums"].items():
            row[f"mean_{name}"] = float(total / max(count, 1))
        rows.append(row)
    return rows


def load_behavior(root: Path) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for path in iter_jsonl(root, "behavior_rows.jsonl"):
        rows.extend(load_jsonl(path))
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        lookup[(str(row.get("example_id")), str(row.get("mode")), str(row.get("condition")))] = row
    return rows, lookup


def summarize_behavior(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("mode")), str(row.get("condition")))].append(row)
    out: List[Dict[str, Any]] = []
    for (mode, condition), group in sorted(grouped.items()):
        token_counter = Counter(str(row.get("first_generated_token_text") or "") for row in group)
        out.append(
            {
                "mode": mode,
                "condition": condition,
                "count": len(group),
                "first_wait_rate": mean([1.0 if row.get("first_wait") else 0.0 for row in group]),
                "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
                "semantic_repair_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
                "mean_generated_tokens": mean([row.get("generated_tokens") for row in group]),
                "cap_rate": mean([1.0 if row.get("hit_max_new_tokens") else 0.0 for row in group]),
                "mean_p0_reflect_vs_stop": mean([row.get("p0_reflect_vs_stop") for row in group]),
                "top_first_tokens": json.dumps(token_counter.most_common(8), ensure_ascii=False),
            }
        )
    return out


def load_logit_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in iter_jsonl(root, "logit_rows.jsonl"):
        rows.extend(load_jsonl(path))
    return rows


def summarize_logits(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric = [
        "reflect_logsum",
        "stop_logsum",
        "reflect_vs_stop",
        "wait_logsum",
        "check_logsum",
        "actually_logsum",
        "finalize_logsum",
        "newline_logsum",
    ]
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("mode")), str(row.get("condition")), int(row.get("position_index", -1)))].append(row)
    out: List[Dict[str, Any]] = []
    for (mode, condition, pos), group in sorted(grouped.items()):
        item: Dict[str, Any] = {"mode": mode, "condition": condition, "position_index": pos, "count": len(group)}
        for name in numeric:
            item[f"mean_{name}"] = mean([row.get(name) for row in group])
        out.append(item)
    return out


def first_pass(
    activation_paths: Sequence[Path],
    behavior_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    max_positions: int,
) -> Tuple[Dict[Any, torch.Tensor], Dict[Any, torch.Tensor], Dict[Any, torch.Tensor], Dict[str, Any]]:
    direction_diffs: Dict[Any, Dict[str, Any]] = {}
    centroid_agg: Dict[Any, Dict[str, Any]] = {}
    probe_pos_agg: Dict[Any, Dict[str, Any]] = {}
    probe_neg_agg: Dict[Any, Dict[str, Any]] = {}
    stats = {"activation_files": len(activation_paths), "examples_with_direction": 0}

    for path in tqdm(activation_paths, desc="First pass activations", dynamic_ncols=True):
        payload = torch.load(path, map_location="cpu")
        example_id = str(payload.get("example_id"))
        runs = payload.get("runs") or {}
        free = runs.get("free") or {}
        t_run = free.get("T") or {}
        c_run = free.get("C") or {}
        t_acts = t_run.get("activations") or {}
        c_acts = c_run.get("activations") or {}
        saw_direction = False
        for act_key, t_tensor in t_acts.items():
            c_tensor = c_acts.get(act_key)
            if c_tensor is None or int(t_tensor.shape[0]) < 1 or int(c_tensor.shape[0]) < 1:
                continue
            layer_idx, site = parse_activation_key(act_key)
            diff = t_tensor[0].float() - c_tensor[0].float()
            add_vec(direction_diffs, (layer_idx, site), tensor_normed(diff))
            saw_direction = True
        if saw_direction:
            stats["examples_with_direction"] += 1

        for mode, mode_runs in runs.items():
            if not isinstance(mode_runs, dict):
                continue
            for condition, run in mode_runs.items():
                acts = run.get("activations") or {}
                behavior = behavior_lookup.get((example_id, str(mode), str(condition)), {})
                for act_key, tensor in acts.items():
                    layer_idx, site = parse_activation_key(act_key)
                    n_pos = min(int(tensor.shape[0]), int(max_positions) + 1)
                    for pos in range(n_pos):
                        vec = tensor[pos].float()
                        if condition in {"T", "C"}:
                            add_vec(centroid_agg, (str(mode), str(condition), layer_idx, site, pos), vec)
                        if condition == "T" and bool(behavior.get("has_reflection")):
                            add_vec(probe_pos_agg, (str(mode), layer_idx, site, pos), vec)
                        elif condition == "C" or (condition == "T" and not bool(behavior.get("has_reflection"))):
                            add_vec(probe_neg_agg, (str(mode), layer_idx, site, pos), vec)

    direction_means = finalize_vec_agg(direction_diffs)
    directions = {key: tensor_normed(vec) for key, vec in direction_means.items()}
    centroids = finalize_vec_agg(centroid_agg)
    probe_pos = finalize_vec_agg(probe_pos_agg)
    probe_neg = finalize_vec_agg(probe_neg_agg)
    probes: Dict[Any, torch.Tensor] = {}
    for key, pos_mean in probe_pos.items():
        neg_mean = probe_neg.get(key)
        if neg_mean is None:
            continue
        unit = tensor_normed(pos_mean - neg_mean)
        if float(unit.norm().item()) > 0:
            threshold = float(torch.dot(unit, ((pos_mean + neg_mean) / 2.0).float()).item())
            probes[key] = torch.cat([unit, torch.tensor([threshold], dtype=torch.float32)], dim=0)
    return directions, centroids, probes, stats


def second_pass(
    activation_paths: Sequence[Path],
    *,
    directions: Dict[Any, torch.Tensor],
    centroids: Dict[Any, torch.Tensor],
    probes: Dict[Any, torch.Tensor],
    max_positions: int,
    write_detail_rows: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_agg: Dict[Any, Dict[str, Any]] = {}
    detail_rows: List[Dict[str, Any]] = []
    for path in tqdm(activation_paths, desc="Second pass metrics", dynamic_ncols=True):
        payload = torch.load(path, map_location="cpu")
        example_id = str(payload.get("example_id"))
        global_idx = payload.get("global_idx")
        runs = payload.get("runs") or {}
        for mode, mode_runs in runs.items():
            if not isinstance(mode_runs, dict):
                continue
            for condition, run in mode_runs.items():
                acts = run.get("activations") or {}
                for act_key, tensor in acts.items():
                    layer_idx, site = parse_activation_key(act_key)
                    direction = directions.get((layer_idx, site))
                    n_pos = min(int(tensor.shape[0]), int(max_positions) + 1)
                    for pos in range(n_pos):
                        vec = tensor[pos].float()
                        values: Dict[str, float] = {}
                        if direction is not None:
                            values["gate_proj"] = float(torch.dot(vec, direction).item())
                        mu_t = centroids.get((str(mode), "T", layer_idx, site, pos))
                        mu_c = centroids.get((str(mode), "C", layer_idx, site, pos))
                        if mu_t is not None and mu_c is not None:
                            dist_t = float((vec - mu_t).norm().item())
                            dist_c = float((vec - mu_c).norm().item())
                            values["dist_to_T"] = dist_t
                            values["dist_to_C"] = dist_c
                            values["relative_position"] = dist_c - dist_t
                        probe = probes.get((str(mode), layer_idx, site, pos))
                        if probe is not None:
                            unit = probe[:-1]
                            threshold = float(probe[-1].item())
                            values["repair_probe_margin"] = float(torch.dot(vec, unit).item() - threshold)
                        if values:
                            key = (str(mode), str(condition), layer_idx, site, pos)
                            add_scalar(summary_agg, key, values)
                            if write_detail_rows:
                                detail_rows.append(
                                    {
                                        "example_id": example_id,
                                        "global_idx": global_idx,
                                        "mode": str(mode),
                                        "condition": str(condition),
                                        "layer_idx": layer_idx,
                                        "site": site,
                                        "position_index": pos,
                                        **values,
                                    }
                                )
    summary_rows = finalize_scalar_rows(
        summary_agg,
        ["mode", "condition", "layer_idx", "site", "position_index"],
    )
    c_lookup = {
        (row["mode"], row["layer_idx"], row["site"], row["position_index"]): row
        for row in summary_rows
        if row["condition"] == "C"
    }
    t_lookup = {
        (row["mode"], row["layer_idx"], row["site"], row["position_index"]): row
        for row in summary_rows
        if row["condition"] == "T"
    }
    for row in summary_rows:
        key = (row["mode"], row["layer_idx"], row["site"], row["position_index"])
        c_row = c_lookup.get(key)
        t_row = t_lookup.get(key)
        if c_row is not None and "mean_gate_proj" in row and "mean_gate_proj" in c_row:
            row["gate_proj_minus_C"] = float(row["mean_gate_proj"] - c_row["mean_gate_proj"])
        if t_row is not None and "mean_gate_proj" in row and "mean_gate_proj" in t_row:
            row["gate_proj_minus_T"] = float(row["mean_gate_proj"] - t_row["mean_gate_proj"])
        if c_row is not None and "mean_repair_probe_margin" in row and "mean_repair_probe_margin" in c_row:
            row["probe_margin_minus_C"] = float(row["mean_repair_probe_margin"] - c_row["mean_repair_probe_margin"])
        if t_row is not None and "mean_repair_probe_margin" in row and "mean_repair_probe_margin" in t_row:
            row["probe_margin_minus_T"] = float(row["mean_repair_probe_margin"] - t_row["mean_repair_probe_margin"])
    return summary_rows, detail_rows


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

    write_csv(output_dir / "behavior_summary.csv", summarize_behavior(behavior_rows))
    write_csv(output_dir / "logit_summary.csv", summarize_logits(logit_rows))
    write_csv(output_dir / "hidden_metric_summary.csv", metric_summary)
    if args.write_detail_rows:
        write_csv(output_dir / "hidden_metric_rows.csv", metric_detail)

    direction_rows = [
        {
            "layer_idx": key[0],
            "site": key[1],
            "direction_norm": float(vec.norm().item()),
        }
        for key, vec in sorted(directions.items())
    ]
    probe_rows = [
        {
            "mode": key[0],
            "layer_idx": key[1],
            "site": key[2],
            "position_index": key[3],
            "probe_direction_norm": float(probe[:-1].norm().item()),
            "probe_threshold": float(probe[-1].item()),
        }
        for key, probe in sorted(probes.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
    ]
    write_csv(output_dir / "gate_direction_rows.csv", direction_rows)
    write_csv(output_dir / "repair_probe_rows.csv", probe_rows)
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
    print("[Done] Hidden trajectory movie analysis finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- activation_files: {len(activation_paths)}")
    print(f"- hidden_metric_summary_rows: {len(metric_summary)}")


if __name__ == "__main__":
    main()
