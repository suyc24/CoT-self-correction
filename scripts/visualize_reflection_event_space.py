#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_reflection_event_space import (  # noqa: E402
    build_cluster_matrix,
    finite_float,
    mean,
    pca_reduce,
    silhouette_score_torch,
    standardize_all,
    torch_kmeans,
)


COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#0f766e",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write SVG visualizations for reflection event-space clusters.")
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--site", default="post_attn")
    parser.add_argument("--feature_kinds", default="h_pre,h_marker,delta_marker,delta_post")
    parser.add_argument("--pca_components", type=int, default=20)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--width", type=int, default=980)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def scope_predicate(scope: str, row: Mapping[str, Any]) -> bool:
    event_type = str(row.get("event_type", ""))
    condition = str(row.get("condition", ""))
    reflection_like = event_type in {"natural_marker", "forced_marker", "silent_correction"}
    if scope == "natural_baseline_reflection":
        return event_type == "natural_marker" and condition in {"T", "C"}
    if scope == "intervened_reflection":
        return reflection_like and condition not in {"T", "C"}
    if scope == "all_reflection":
        return reflection_like
    if scope == "all_events":
        return True
    raise ValueError(f"Unknown scope: {scope}")


def event_label(row: Mapping[str, Any]) -> str:
    event_type = str(row.get("event_type", ""))
    condition = str(row.get("condition", ""))
    if condition in {"T", "C"}:
        return f"baseline:{event_type}"
    if str(row.get("forced_prefix_name", "")):
        return f"forced:{event_type}"
    if str(row.get("intervention_type", "")):
        return f"gate:{event_type}"
    return f"control:{event_type}"


def cluster_scope(
    *,
    events: List[Dict[str, Any]],
    feature_sets: Mapping[str, Mapping[str, Any]],
    scope: str,
    layer_idx: int,
    site: str,
    feature_kinds: Sequence[str],
    pca_components: int,
    k: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_indices, X_raw_all = build_cluster_matrix(feature_sets, layer_idx=layer_idx, site=site, feature_kinds=feature_kinds)
    selected_positions = [pos for pos, event_idx in enumerate(all_indices) if scope_predicate(scope, events[event_idx])]
    if len(selected_positions) < 3:
        return [{"scope": scope, "status": "skipped_too_few_events", "count": len(selected_positions)}], {}
    selected_event_indices = [all_indices[pos] for pos in selected_positions]
    X_raw = X_raw_all[torch.tensor(selected_positions, dtype=torch.long)]
    X_std, mean_vec, std_vec = standardize_all(X_raw.float())
    X_pca, explained = pca_reduce(X_std, int(pca_components))
    k_eff = max(2, min(int(k), int(X_pca.shape[0])))
    labels, inertia = torch_kmeans(X_pca, k_eff, seed=seed, n_init=10, max_iter=100)
    silhouette = silhouette_score_torch(X_pca, labels, max_points=500, seed=seed)

    rows: List[Dict[str, Any]] = []
    for cluster_id in range(k_eff):
        member_positions = [i for i, label in enumerate(labels.tolist()) if int(label) == cluster_id]
        member_events = [events[selected_event_indices[i]] for i in member_positions]
        type_counts = Counter(str(row.get("event_type")) for row in member_events)
        label_counts = Counter(event_label(row) for row in member_events)
        rows.append(
            {
                "scope": scope,
                "k": k_eff,
                "cluster_id": cluster_id,
                "count": len(member_events),
                "dominant_event_type": type_counts.most_common(1)[0][0] if type_counts else "",
                "event_type_counts": json.dumps(dict(type_counts), ensure_ascii=False),
                "behavior_label_counts": json.dumps(dict(label_counts), ensure_ascii=False),
                "silhouette": silhouette,
                "inertia": inertia,
                "semantic_repair_rate": mean(row.get("semantic_repair") for row in member_events),
                "functional_repair_rate": mean(row.get("functional_repair") for row in member_events),
                "silent_answer_correction_rate": mean(row.get("silent_answer_correction") for row in member_events),
                "mean_generated_tokens": mean(row.get("generated_tokens") for row in member_events),
            }
        )

    payload = {
        "scope": scope,
        "event_indices": selected_event_indices,
        "x_raw": X_raw,
        "x_standardized": X_std,
        "x_pca": X_pca,
        "labels": labels,
        "pca_explained_variance": explained,
        "standardize_mean": mean_vec,
        "standardize_std": std_vec,
    }
    return rows, payload


def scale(values: Sequence[float], lo: float, hi: float) -> List[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [(lo + hi) / 2.0 for _ in values]
    v_min = min(finite)
    v_max = max(finite)
    if abs(v_max - v_min) < 1e-12:
        return [(lo + hi) / 2.0 for _ in values]
    return [lo + (v - v_min) / (v_max - v_min) * (hi - lo) for v in values]


def write_scatter_svg(
    path: Path,
    *,
    title: str,
    events: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    color_by: str,
    width: int,
    height: int,
) -> None:
    X = payload["x_pca"]
    labels = payload["labels"]
    event_indices = payload["event_indices"]
    if int(X.shape[1]) < 2:
        return
    xs = [float(x) for x in X[:, 0].tolist()]
    ys = [float(y) for y in X[:, 1].tolist()]
    sx = scale(xs, 80, width - 260)
    sy = scale(ys, height - 80, 70)
    if color_by == "cluster":
        color_labels = [f"cluster {int(x)}" for x in labels.tolist()]
    elif color_by == "event_type":
        color_labels = [str(events[event_indices[i]].get("event_type")) for i in range(len(event_indices))]
    else:
        color_labels = [event_label(events[event_indices[i]]) for i in range(len(event_indices))]
    unique = sorted(set(color_labels))
    color_map = {label: COLORS[i % len(COLORS)] for i, label in enumerate(unique)}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
        f'<text x="24" y="58" font-family="Arial" font-size="12" fill="#555">PC1 vs PC2; color_by={html.escape(color_by)}</text>',
        f'<line x1="80" y1="{height - 80}" x2="{width - 260}" y2="{height - 80}" stroke="#444"/>',
        f'<line x1="80" y1="70" x2="80" y2="{height - 80}" stroke="#444"/>',
        f'<text x="{(80 + width - 260) / 2:.1f}" y="{height - 28}" font-family="Arial" font-size="13">PC1</text>',
        f'<text x="24" y="{(70 + height - 80) / 2:.1f}" transform="rotate(-90 24 {(70 + height - 80) / 2:.1f})" font-family="Arial" font-size="13">PC2</text>',
    ]
    for i, event_idx in enumerate(event_indices):
        row = events[event_idx]
        label = color_labels[i]
        tooltip = f"{row.get('event_idx')} | {row.get('condition')} | {row.get('event_type')} | cluster={int(labels[i].item())}"
        parts.append(
            f'<circle cx="{sx[i]:.2f}" cy="{sy[i]:.2f}" r="4.2" fill="{color_map[label]}" fill-opacity="0.78">'
            f"<title>{html.escape(tooltip)}</title></circle>"
        )
    legend_x = width - 230
    legend_y = 80
    parts.append(f'<text x="{legend_x}" y="{legend_y - 18}" font-family="Arial" font-size="14" font-weight="700">Legend</text>')
    for i, label in enumerate(unique[:24]):
        y = legend_y + i * 22
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color_map[label]}"/>')
        parts.append(f'<text x="{legend_x + 12}" y="{y + 4}" font-family="Arial" font-size="12">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(path: Path, cluster_rows: Sequence[Mapping[str, Any]], plot_paths: Sequence[Path]) -> None:
    lines = [
        "# 反思事件空间可视化",
        "",
        "## Scope 定义",
        "- `natural_baseline_reflection`：只看无干涉条件 `T/C` 下自然生成 marker 的反思事件。",
        "- `intervened_reflection`：看 gate/forced-prefix/ban-marker 等 intervention 条件下的反思或 silent correction 事件。",
        "- `all_reflection`：把自然、forced、silent correction 反思相关事件放在一起。",
        "- `all_events`：包含 nonreflection termination，用来观察反思和非反思是否分开。",
        "",
        "## 聚类摘要",
    ]
    for scope in sorted(set(str(row.get("scope")) for row in cluster_rows)):
        rows = [row for row in cluster_rows if str(row.get("scope")) == scope and row.get("cluster_id") != ""]
        lines.extend(["", f"### {scope}"])
        if not rows:
            lines.append("（没有足够事件）")
            continue
        lines.append("| cluster | count | dominant | semantic_repair | functional_repair | silent_correction | mean_tokens |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                "| {cluster_id} | {count} | {dominant_event_type} | {semantic_repair_rate:.3f} | "
                "{functional_repair_rate:.3f} | {silent_answer_correction_rate:.3f} | {mean_generated_tokens:.1f} |".format(
                    cluster_id=row.get("cluster_id", ""),
                    count=row.get("count", ""),
                    dominant_event_type=row.get("dominant_event_type", ""),
                    semantic_repair_rate=finite_float(row.get("semantic_repair_rate"), 0.0),
                    functional_repair_rate=finite_float(row.get("functional_repair_rate"), 0.0),
                    silent_answer_correction_rate=finite_float(row.get("silent_answer_correction_rate"), 0.0),
                    mean_generated_tokens=finite_float(row.get("mean_generated_tokens"), 0.0),
                )
            )
    lines.extend(["", "## 图像文件"])
    for plot_path in plot_paths:
        lines.append(f"- [{plot_path.name}](plots/{plot_path.name})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "visualizations"
    plots_dir = output_dir / "plots"
    payload = torch.load(analysis_dir / "event_features.pt", map_location="cpu")
    events = payload["events"]
    feature_sets = payload["feature_sets"]
    scopes = ["natural_baseline_reflection", "intervened_reflection", "all_reflection", "all_events"]
    cluster_rows: List[Dict[str, Any]] = []
    plot_paths: List[Path] = []
    for scope in scopes:
        rows, scope_payload = cluster_scope(
            events=events,
            feature_sets=feature_sets,
            scope=scope,
            layer_idx=int(args.layer),
            site=str(args.site),
            feature_kinds=parse_csv_list(args.feature_kinds),
            pca_components=int(args.pca_components),
            k=int(args.k),
            seed=int(args.seed),
        )
        cluster_rows.extend(rows)
        if scope_payload:
            for color_by in ["cluster", "event_type", "behavior"]:
                plot_path = plots_dir / f"{scope}_{color_by}.svg"
                write_scatter_svg(
                    plot_path,
                    title=f"{scope} ({color_by})",
                    events=events,
                    payload=scope_payload,
                    color_by=color_by,
                    width=int(args.width),
                    height=int(args.height),
                )
                plot_paths.append(plot_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scope_cluster_summary.csv", cluster_rows)
    write_report(output_dir / "VIS_REPORT.md", cluster_rows, plot_paths)
    torch.save({"cluster_rows": cluster_rows, "scopes": scopes}, output_dir / "visualization_payload.pt")
    print(f"[Done] Wrote visualizations to {output_dir}")
    print(f"- plots: {len(plot_paths)}")
    print(f"- cluster_rows: {len(cluster_rows)}")


if __name__ == "__main__":
    main()
