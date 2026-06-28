#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.hidden_trajectory import tensor_normed
from cot_research.io_utils import load_jsonl, write_json
from analyze_hidden_trajectory_movie import (
    add_scalar,
    add_vec,
    finalize_scalar_rows,
    finalize_vec_agg,
    iter_activation_paths,
    iter_jsonl,
    parse_activation_key,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reflection mechanism D1/D2/D3/D6 outputs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment", choices=["D1", "D2", "D3", "D6"], required=True)
    parser.add_argument("--max_positions", type=int, default=64)
    parser.add_argument("--write_detail_rows", action=argparse.BooleanOptionalAction, default=False)
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


def load_behavior(root: Path) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str], Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for path in iter_jsonl(root, "behavior_rows.jsonl"):
        rows.extend(load_jsonl(path))
    lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        lookup[(str(row.get("example_id")), str(row.get("mode", "free")), str(row.get("condition")))] = row
    return rows, lookup


def load_logit_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in iter_jsonl(root, "logit_rows.jsonl"):
        rows.extend(load_jsonl(path))
    return rows


def summarize_behavior(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("experiment")),
                str(row.get("condition_kind")),
                str(row.get("condition")),
                str(row.get("patch_type", "")),
                str(row.get("patch_layer", "")),
                str(row.get("patch_timing", "")),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (experiment, kind, condition, patch_type, patch_layer, patch_timing), group in sorted(grouped.items()):
        tokens = [int(row.get("generated_tokens") or 0) for row in group]
        out.append(
            {
                "experiment": experiment,
                "condition_kind": kind,
                "condition": condition,
                "patch_type": patch_type,
                "patch_layer": patch_layer,
                "patch_site": group[0].get("patch_site", ""),
                "patch_timing": patch_timing,
                "base_condition": group[0].get("base_condition", ""),
                "source_condition": group[0].get("source_condition", ""),
                "count": len(group),
                "surface_reflection_rate": mean([1.0 if row.get("surface_reflection") else 0.0 for row in group]),
                "explicit_error_ack_rate": mean([1.0 if row.get("explicit_error_ack") else 0.0 for row in group]),
                "recompute_or_revise_rate": mean([1.0 if row.get("recompute_or_revise") else 0.0 for row in group]),
                "semantic_repair_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
                "functional_repair_rate": mean([1.0 if row.get("functional_repair") else 0.0 for row in group]),
                "silent_answer_correction_rate": mean([1.0 if row.get("silent_answer_correction") else 0.0 for row in group]),
                "recover_A_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
                "reject_B_rate": mean([1.0 if row.get("reject_B") else 0.0 for row in group]),
                "uses_restored_anchor_rate": mean([1.0 if row.get("uses_restored_anchor") else 0.0 for row in group]),
                "new_answer_not_A_not_B_rate": mean([1.0 if row.get("new_answer_not_A_not_B") else 0.0 for row in group]),
                "explicit_search_rate": mean([1.0 if row.get("explicit_search") else 0.0 for row in group]),
                "mean_generated_tokens": mean(tokens),
                "median_generated_tokens": float(median(tokens)) if tokens else float("nan"),
                "cap_rate": mean([1.0 if row.get("cap") else 0.0 for row in group]),
                "no_box_rate": mean([1.0 if row.get("no_box") else 0.0 for row in group]),
                "mean_p0_reflect_vs_stop": mean([row.get("p0_reflect_vs_stop") for row in group]),
                "mean_patch_hook_calls": mean([row.get("patch_hook_calls") for row in group]),
                "mean_add_hook_calls": mean([row.get("add_hook_calls") for row in group]),
            }
        )
    return out


def summarize_logits(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric = [
        "reflect_logsum",
        "stop_logsum",
        "reflect_vs_stop",
        "marker_logsum",
        "marker_vs_stop",
        "finalize_logsum",
        "finalize_vs_reflect",
        "wait_logsum",
        "check_logsum",
        "actually_logsum",
        "newline_logsum",
    ]
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("mode", "free")), str(row.get("condition")), int(row.get("position_index", -1)))].append(row)
    out: List[Dict[str, Any]] = []
    for (mode, condition, pos), group in sorted(grouped.items()):
        item: Dict[str, Any] = {"mode": mode, "condition": condition, "position_index": pos, "count": len(group)}
        for name in numeric:
            item[f"mean_{name}"] = mean([row.get(name) for row in group])
        out.append(item)
    return out


def reference_conditions(experiment: str) -> Tuple[str, str]:
    if experiment == "D1":
        return "IW", "CW"
    return "T", "C"


def first_pass(
    activation_paths: Sequence[Path],
    behavior_lookup: Dict[Tuple[str, str, str], Dict[str, Any]],
    max_positions: int,
    *,
    t_condition: str,
    c_condition: str,
) -> Tuple[Dict[Any, torch.Tensor], Dict[Any, torch.Tensor], Dict[Any, torch.Tensor], Dict[str, Any]]:
    direction_diffs: Dict[Any, Dict[str, Any]] = {}
    centroid_agg: Dict[Any, Dict[str, Any]] = {}
    probe_pos_agg: Dict[Any, Dict[str, Any]] = {}
    probe_neg_agg: Dict[Any, Dict[str, Any]] = {}
    stats = {"activation_files": len(activation_paths), "examples_with_direction": 0}

    for path in tqdm(activation_paths, desc="First pass activations", dynamic_ncols=True):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            stats["bad_activation_files"] = int(stats.get("bad_activation_files", 0)) + 1
            stats.setdefault("bad_activation_file_paths", []).append(str(path))
            stats.setdefault("bad_activation_errors", []).append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        example_id = str(payload.get("example_id"))
        runs = payload.get("runs") or {}
        free = runs.get("free") or {}
        t_run = free.get(t_condition) or {}
        c_run = free.get(c_condition) or {}
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
                        if condition in {t_condition, c_condition}:
                            add_vec(centroid_agg, (str(mode), str(condition), layer_idx, site, pos), vec)
                        if condition == t_condition and bool(behavior.get("surface_reflection") or behavior.get("has_reflection")):
                            add_vec(probe_pos_agg, (str(mode), layer_idx, site, pos), vec)
                        elif condition == c_condition or (condition == t_condition and not bool(behavior.get("surface_reflection") or behavior.get("has_reflection"))):
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
    t_condition: str,
    c_condition: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_agg: Dict[Any, Dict[str, Any]] = {}
    detail_rows: List[Dict[str, Any]] = []
    for path in tqdm(activation_paths, desc="Second pass metrics", dynamic_ncols=True):
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
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
                        mu_t = centroids.get((str(mode), t_condition, layer_idx, site, pos))
                        mu_c = centroids.get((str(mode), c_condition, layer_idx, site, pos))
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
    summary_rows = finalize_scalar_rows(summary_agg, ["mode", "condition", "layer_idx", "site", "position_index"])
    c_lookup = {
        (row["mode"], row["layer_idx"], row["site"], row["position_index"]): row
        for row in summary_rows
        if row["condition"] == c_condition
    }
    t_lookup = {
        (row["mode"], row["layer_idx"], row["site"], row["position_index"]): row
        for row in summary_rows
        if row["condition"] == t_condition
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


def compact_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], *, max_rows: int = 18) -> str:
    rows = list(rows)[:max_rows]
    if not rows:
        return "(empty)"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def write_report(output_dir: Path, *, experiment: str, summary: Dict[str, Any], behavior_rows: Sequence[Dict[str, Any]], hidden_rows: Sequence[Dict[str, Any]]) -> None:
    behavior_core = [
        row
        for row in behavior_rows
        if row.get("condition_kind")
        in {
            "factorial",
            "factorial_intervention",
            "baseline",
            "marker_loop",
            "marker_control",
            "anchor_restore",
            "anchor_gate_marker",
            "gate_marker",
            "component_patch",
        }
    ]
    hidden_core = [
        row
        for row in hidden_rows
        if int(row.get("layer_idx", -1)) == 22 and row.get("site") == "post_attn" and int(row.get("position_index", -1)) in {1, 2, 4, 8}
    ]
    hidden_grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in hidden_core:
        hidden_grouped[str(row.get("condition"))].append(row)
    hidden_summary: List[Dict[str, Any]] = []
    for condition, group in sorted(hidden_grouped.items()):
        hidden_summary.append(
            {
                "condition": condition,
                "gate_proj_minus_C": mean([row.get("gate_proj_minus_C") for row in group]),
                "repair_probe_margin": mean([row.get("mean_repair_probe_margin") for row in group]),
                "relative_position": mean([row.get("mean_relative_position") for row in group]),
            }
        )

    title_map = {
        "D1": "D1 Truth x Local Coherence Factorial",
        "D2": "D2 Token-Hidden Closed Loop",
        "D3": "D3 Gate State + Alternative Evidence Sufficiency",
        "D6": "D6 Attention-to-MLP Stabilization",
    }
    lines = [
        f"# {title_map.get(experiment, experiment)}",
        "",
        "## 1. 实验目的",
        {
            "D1": "区分 L22 repair/gate state 更像 local inconsistency、wrong-answer detection、need-to-revise 还是 commitment inhibition。",
            "D2": "测试 forced Wait / neutral prefix 是否通过重新制造或绕开 L22 hidden repair state 来恢复修复。",
            "D3": "测试 coherent wrong context 中，T repair state 是否需要 restored clean answer anchors 才足以诱导 repair。",
            "D6": "区分 attention 写入 alarm、MLP 放大/稳定 branch、block_output 承载行为状态这几种可能路径。",
        }.get(experiment, ""),
        "",
        "## 2. 运行规模",
        f"- n_attempted: {summary.get('n_attempted')}",
        f"- n_usable: {summary.get('n_usable')}",
        f"- n_skipped: {summary.get('n_skipped')}",
        f"- skip_reasons: `{json.dumps(summary.get('skip_reasons', {}), ensure_ascii=False)}`",
        "",
        "## 3. Behavior Summary",
        compact_table(
            behavior_core,
            [
                "condition",
                "count",
                "surface_reflection_rate",
                "explicit_error_ack_rate",
                "semantic_repair_rate",
                "functional_repair_rate",
                "silent_answer_correction_rate",
                "mean_generated_tokens",
            ],
            max_rows=24,
        ),
        "",
        "## 4. Hidden Summary: L22 post_attn p1-p8",
        compact_table(
            hidden_summary,
            ["condition", "gate_proj_minus_C", "repair_probe_margin", "relative_position"],
            max_rows=24,
        ),
        "",
        "## 5. 关键发现",
        "- 自动报告先列核心表；具体结论需要结合完整 CSV 做人工解读。",
        "- 优先查看 behavior summary 与 L22 post_attn p1-p8 hidden summary 是否同向变化。",
        "- patch 条件请查看对应 `*_patch_summary.csv` 或完整 `behavior_summary.csv`。",
        "",
        "## 6. 不确定点",
        "- explicit/silent/functional repair 仍是启发式文本判定，需要样本级 audit。",
        "- hidden probe 以本实验内 contrast 训练，跨实验比较时应看相对趋势。",
        "",
        "## 7. 下一步建议",
        "- Stage 1 完成后，优先人工审计 D1/D2/D3 中分歧最大的样本。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_paths = iter_activation_paths(root)
    behavior_rows, behavior_lookup = load_behavior(root)
    logit_rows = load_logit_rows(root)
    t_condition, c_condition = reference_conditions(args.experiment)

    directions: Dict[Any, torch.Tensor] = {}
    centroids: Dict[Any, torch.Tensor] = {}
    probes: Dict[Any, torch.Tensor] = {}
    stats = {"activation_files": len(activation_paths), "examples_with_direction": 0}
    metric_summary: List[Dict[str, Any]] = []
    metric_detail: List[Dict[str, Any]] = []
    if activation_paths:
        directions, centroids, probes, stats = first_pass(
            activation_paths,
            behavior_lookup,
            int(args.max_positions),
            t_condition=t_condition,
            c_condition=c_condition,
        )
        metric_summary, metric_detail = second_pass(
            activation_paths,
            directions=directions,
            centroids=centroids,
            probes=probes,
            max_positions=int(args.max_positions),
            write_detail_rows=bool(args.write_detail_rows),
            t_condition=t_condition,
            c_condition=c_condition,
        )

    behavior_summary = summarize_behavior(behavior_rows)
    logit_summary = summarize_logits(logit_rows)
    write_csv(output_dir / "behavior_summary.csv", behavior_summary)
    write_csv(output_dir / "hidden_summary.csv", metric_summary)
    write_csv(output_dir / "hidden_metric_summary.csv", metric_summary)
    if args.write_detail_rows:
        write_csv(output_dir / "hidden_metric_rows.csv", metric_detail)
    write_csv(output_dir / "logit_summary.csv", logit_summary)
    write_csv(
        output_dir / "gate_direction_rows.csv",
        [{"layer_idx": key[0], "site": key[1], "direction_norm": float(vec.norm().item())} for key, vec in sorted(directions.items())],
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
    if args.experiment == "D1":
        write_csv(output_dir / "factorial_behavior_summary.csv", behavior_summary)
        write_csv(output_dir / "factorial_hidden_summary.csv", metric_summary)
        write_csv(output_dir / "factorial_patch_summary.csv", [row for row in behavior_summary if row.get("condition_kind") == "factorial_patch"])
        write_csv(output_dir / "factorial_layer_site_heatmap_data.csv", metric_summary)
    elif args.experiment == "D2":
        write_csv(output_dir / "token_hidden_behavior_summary.csv", behavior_summary)
        write_csv(output_dir / "token_hidden_hidden_summary.csv", metric_summary)
        write_csv(output_dir / "forced_wait_delete_patch_summary.csv", [row for row in behavior_summary if str(row.get("condition")).startswith("T_gateoff_force_wait")])
        write_csv(output_dir / "neutral_prefix_delete_patch_summary.csv", [row for row in behavior_summary if "neutral_prefix" in str(row.get("condition"))])
    elif args.experiment == "D3":
        write_csv(output_dir / "anchor_restore_behavior_summary.csv", behavior_summary)
        write_csv(output_dir / "anchor_restore_hidden_summary.csv", metric_summary)
        write_csv(output_dir / "anchor_restore_interaction_table.csv", behavior_summary)
    elif args.experiment == "D6":
        write_csv(output_dir / "component_patch_behavior_summary.csv", behavior_summary)
        write_csv(output_dir / "component_patch_hidden_summary.csv", metric_summary)
        write_csv(output_dir / "layer_component_timing_heatmap.csv", metric_summary)

    reason_counts = Counter(str(row.get("reason")) for path in iter_jsonl(root, "skipped_rows.jsonl") for row in load_jsonl(path))
    summary = {
        **stats,
        "experiment": args.experiment,
        "n_attempted": len({(row.get("example_id"), row.get("global_idx")) for row in behavior_rows}) + sum(reason_counts.values()),
        "n_usable": len({row.get("example_id") for row in behavior_rows}),
        "n_skipped": sum(reason_counts.values()),
        "skip_reasons": dict(reason_counts),
        "behavior_rows": len(behavior_rows),
        "logit_rows": len(logit_rows),
        "metric_summary_rows": len(metric_summary),
        "metric_detail_rows": len(metric_detail),
        "gate_directions": len(directions),
        "repair_probes": len(probes),
        "activation_root": str(root),
        "t_condition": t_condition,
        "c_condition": c_condition,
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, experiment=args.experiment, summary=summary, behavior_rows=behavior_summary, hidden_rows=metric_summary)
    print(f"[Done] Reflection mechanism analysis {args.experiment} finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- behavior_rows: {len(behavior_rows)}")
    print(f"- hidden_summary_rows: {len(metric_summary)}")


if __name__ == "__main__":
    main()
