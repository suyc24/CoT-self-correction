#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finer-grained reflection-event clustering with cohesion diagnostics."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="L22/post_attn")
    parser.add_argument("--feature_kinds", default="h_pre,h_marker,delta_marker,delta_post")
    parser.add_argument("--pca_dims", type=int, default=48)
    parser.add_argument("--cluster_ks", default="8,12,16,24,32,40")
    parser.add_argument("--final_k_natural", type=int, default=32)
    parser.add_argument("--final_k_tampered", type=int, default=20)
    parser.add_argument("--final_k_combined", type=int, default=40)
    parser.add_argument("--sample_for_metrics", type=int, default=30000)
    parser.add_argument("--max_scatter_points", type=int, default=25000)
    parser.add_argument("--kmeans_steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_ks(text: str) -> List[int]:
    out: List[int] = []
    for part in parse_csv_list(text):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(k for k in out if k > 1))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def shard_dirs(root: Path) -> List[Path]:
    if (root / "event_rows.jsonl").exists():
        return [root]
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "event_rows.jsonl").exists()])


def load_all(root: Path, feature_key: str, feature_kinds: Sequence[str]) -> Tuple[List[Dict[str, Any]], torch.Tensor, Dict[str, Any]]:
    rows_all: List[Dict[str, Any]] = []
    xs: List[torch.Tensor] = []
    meta: Dict[str, Any] = {"shards": [], "bad_shards": []}
    offset = 0
    for shard in shard_dirs(root):
        rows = load_jsonl(shard / "event_rows.jsonl")
        feat_path = shard / "event_features.pt"
        if not rows or not feat_path.exists():
            meta["bad_shards"].append(str(shard))
            continue
        obj = torch.load(feat_path, map_location="cpu")
        features = obj.get("features", {})
        if feature_key not in features:
            meta["bad_shards"].append(str(shard))
            continue
        parts = []
        ok = True
        for kind in feature_kinds:
            t = features[feature_key].get(kind)
            if t is None or t.ndim != 2 or t.shape[0] == 0:
                ok = False
                break
            parts.append(t.float())
        if not ok:
            meta["bad_shards"].append(str(shard))
            continue
        x = torch.cat(parts, dim=1)
        n = min(len(rows), int(x.shape[0]))
        for i in range(n):
            row = dict(rows[i])
            row["global_event_index"] = offset + i
            row["shard"] = shard.name
            rows_all.append(row)
        xs.append(x[:n])
        offset += n
        meta["shards"].append({"path": str(shard), "events": n})
    if not xs:
        return rows_all, torch.empty((0, 0)), meta
    return rows_all, torch.cat(xs, dim=0), meta


def standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1e-6)


def pca_project(x: torch.Tensor, dims: int) -> Tuple[torch.Tensor, List[float]]:
    x_std = standardize(x)
    q = min(int(dims), int(x_std.shape[0]) - 1, int(x_std.shape[1]))
    _, s, v = torch.pca_lowrank(x_std, q=q)
    z = x_std @ v
    var = s.pow(2)
    ratios = (var / var.sum().clamp_min(1e-12)).tolist()
    return z.contiguous(), [float(v) for v in ratios]


def sample_indices(n: int, max_n: int, seed: int) -> torch.Tensor:
    if n <= max_n:
        return torch.arange(n)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randperm(n, generator=gen)[:max_n].sort().values


def kmeans_plus_plus_init(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    n = int(x.shape[0])
    centers = torch.empty((k, x.shape[1]), dtype=x.dtype)
    first = int(torch.randint(0, n, (1,), generator=gen).item())
    centers[0] = x[first]
    closest = torch.cdist(x, centers[:1]).squeeze(1).pow(2)
    for i in range(1, k):
        probs = closest / closest.sum().clamp_min(1e-12)
        idx = int(torch.multinomial(probs, 1, generator=gen).item())
        centers[i] = x[idx]
        closest = torch.minimum(closest, torch.cdist(x, centers[i : i + 1]).squeeze(1).pow(2))
    return centers


def kmeans(x: torch.Tensor, k: int, steps: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
    n = int(x.shape[0])
    k = min(int(k), n)
    if n == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0, x.shape[1])), float("nan")
    if n <= k:
        labels = torch.arange(n)
        return labels, x.clone(), 0.0
    centers = kmeans_plus_plus_init(x, k, seed)
    labels = torch.full((n,), -1, dtype=torch.long)
    for _ in range(int(steps)):
        new_labels = torch.cdist(x, centers).argmin(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if bool(mask.any()):
                centers[cluster_id] = x[mask].mean(dim=0)
            else:
                farthest = torch.cdist(x, centers).min(dim=1).values.argmax()
                centers[cluster_id] = x[int(farthest)]
    inertia = torch.cdist(x, centers).min(dim=1).values.pow(2).mean().item()
    return labels, centers, float(inertia)


def assign_to_centers(x: torch.Tensor, centers: torch.Tensor, chunk: int = 20000) -> torch.Tensor:
    labels = []
    for start in range(0, int(x.shape[0]), chunk):
        labels.append(torch.cdist(x[start : start + chunk], centers).argmin(dim=1).cpu())
    return torch.cat(labels, dim=0)


def silhouette(x: torch.Tensor, labels: torch.Tensor, max_n: int, seed: int) -> float:
    n = int(x.shape[0])
    if n < 3 or labels.unique().numel() < 2:
        return float("nan")
    idx = sample_indices(n, max_n, seed)
    x = x[idx]
    labels = labels[idx]
    n = int(x.shape[0])
    distances = torch.cdist(x, x)
    values = []
    unique = labels.unique().tolist()
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = distances[i][same].mean().item() if bool(same.any()) else 0.0
        b = float("inf")
        for label in unique:
            if int(label) == int(labels[i]):
                continue
            mask = labels == int(label)
            if bool(mask.any()):
                b = min(b, distances[i][mask].mean().item())
        values.append((b - a) / max(a, b, 1e-12))
    return float(sum(values) / len(values))


def cluster_radii(x: torch.Tensor, labels: torch.Tensor, centers: torch.Tensor) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    distances = torch.norm(x - centers[labels], dim=1)
    for cluster_id in sorted(set(int(v) for v in labels.tolist())):
        d = distances[labels == cluster_id]
        if d.numel() == 0:
            continue
        out[cluster_id] = {
            "mean_radius": float(d.mean().item()),
            "p50_radius": float(torch.quantile(d, 0.50).item()),
            "p90_radius": float(torch.quantile(d, 0.90).item()),
            "max_radius": float(d.max().item()),
        }
    return out


def as_bool(row: Mapping[str, Any], key: str) -> bool:
    return str(row.get(key)).lower() in {"true", "1", "yes"}


def mean_float(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key) or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return sum(values) / max(len(values), 1)


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        p = count / total
        value -= p * math.log(p + 1e-12)
    return float(value)


def label_cluster(row: Mapping[str, Any]) -> str:
    marker = str(row.get("dominant_marker", ""))
    ack = float(row.get("explicit_error_ack_rate") or 0.0)
    repair = float(row.get("repair_language_rate") or 0.0)
    correct = float(row.get("final_answer_correct_rate") or 0.0)
    pos = float(row.get("mean_relative_position") or 0.0)
    tamper_match = float(row.get("final_answer_matches_tamper_rate") or 0.0)
    if ack >= 0.75 and repair >= 0.70 and pos <= 0.20:
        return "立即不一致警报"
    if marker == "wrong" and repair >= 0.60:
        return "显式错误承认修复"
    if marker == "check" and repair >= 0.55:
        return "重新检查修复"
    if marker in {"wait", "hold_on"} and ack >= 0.45 and repair >= 0.60:
        return "停顿后重算"
    if marker in {"but", "however", "actually"} and repair >= 0.75:
        return "强转折修复"
    if marker in {"but", "however", "actually"} and repair >= 0.35:
        return "弱转折重路由"
    if repair <= 0.10 and correct <= 0.05:
        return "空转或失败停顿"
    if pos >= 0.60 and correct <= 0.10:
        return "晚期失败纠缠"
    if tamper_match >= 0.08:
        return "干涉残留未完全修复"
    return "混合反思"


def representative_examples(
    rows: Sequence[Mapping[str, Any]],
    x: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    cluster_id: int,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    idx = torch.nonzero(labels == int(cluster_id), as_tuple=False).flatten()
    if idx.numel() == 0:
        return []
    d = torch.norm(x[idx] - centers[int(cluster_id)], dim=1)
    order = torch.argsort(d)[:limit]
    reps = []
    for j in order.tolist():
        local = int(idx[j].item())
        row = rows[local]
        reps.append(
            {
                "example_id": row.get("example_id"),
                "marker": row.get("marker_kind"),
                "step": row.get("event_step"),
                "relative_position": row.get("relative_position"),
                "error_ack": row.get("future_explicit_error_ack"),
                "repair": row.get("future_repair_language"),
                "correct": row.get("final_answer_correct"),
                "tamper_match": row.get("final_answer_matches_tamper"),
                "distance": round(float(d[j].item()), 4),
            }
        )
    return reps


def summarize_clusters(
    scope: str,
    rows: Sequence[Mapping[str, Any]],
    z: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    sil: float,
) -> List[Dict[str, Any]]:
    radii = cluster_radii(z, labels, centers)
    out = []
    raw_labels = labels.tolist()
    total = len(rows)
    for cluster_id in sorted(set(int(v) for v in raw_labels)):
        idx = [i for i, value in enumerate(raw_labels) if int(value) == cluster_id]
        cluster_rows = [rows[i] for i in idx]
        marker_counts = Counter(str(r.get("marker_kind")) for r in cluster_rows)
        scope_counts = Counter(str(r.get("scope")) for r in cluster_rows)
        n = len(cluster_rows)
        row = {
            "scope": scope,
            "cluster_id": cluster_id,
            "count": n,
            "fraction": n / max(total, 1),
            "dominant_marker": marker_counts.most_common(1)[0][0] if marker_counts else "",
            "marker_entropy": entropy(marker_counts),
            "marker_counts": json.dumps(dict(marker_counts), ensure_ascii=False),
            "scope_counts": json.dumps(dict(scope_counts), ensure_ascii=False),
            "silhouette": sil,
            "explicit_error_ack_rate": sum(as_bool(r, "future_explicit_error_ack") for r in cluster_rows) / max(n, 1),
            "repair_language_rate": sum(as_bool(r, "future_repair_language") for r in cluster_rows) / max(n, 1),
            "final_answer_correct_rate": sum(as_bool(r, "final_answer_correct") for r in cluster_rows) / max(n, 1),
            "final_answer_matches_tamper_rate": sum(as_bool(r, "final_answer_matches_tamper") for r in cluster_rows) / max(n, 1),
            "mean_relative_position": mean_float(cluster_rows, "relative_position"),
            "mean_event_step": mean_float(cluster_rows, "event_step"),
            **radii.get(cluster_id, {}),
        }
        row["type_label"] = label_cluster(row)
        row["representative_events"] = json.dumps(
            representative_examples(cluster_rows, z[idx], labels[idx] * 0, centers[cluster_id : cluster_id + 1], 0),
            ensure_ascii=False,
        )
        out.append(row)
    return out


def run_k_sweep(scope: str, z: torch.Tensor, ks: Sequence[int], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = []
    sample_idx = sample_indices(int(z.shape[0]), int(args.sample_for_metrics), int(args.seed) + len(scope))
    sample = z[sample_idx]
    for k in ks:
        if int(sample.shape[0]) < k:
            continue
        labels, centers, inertia = kmeans(sample, int(k), steps=int(args.kmeans_steps), seed=int(args.seed) + int(k) + len(scope))
        sil = silhouette(sample, labels, max_n=min(5000, int(args.sample_for_metrics)), seed=int(args.seed) + int(k))
        radii = cluster_radii(sample, labels, centers)
        sizes = Counter(int(v) for v in labels.tolist())
        rows.append(
            {
                "scope": scope,
                "k": int(k),
                "sample_size": int(sample.shape[0]),
                "silhouette": sil,
                "inertia": inertia,
                "mean_cluster_size": sum(sizes.values()) / max(len(sizes), 1),
                "min_cluster_size": min(sizes.values()) if sizes else 0,
                "max_cluster_size": max(sizes.values()) if sizes else 0,
                "mean_radius": sum(v["mean_radius"] for v in radii.values()) / max(len(radii), 1),
                "mean_p90_radius": sum(v["p90_radius"] for v in radii.values()) / max(len(radii), 1),
            }
        )
    return rows


def svg_k_sweep(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    scopes = sorted(set(str(r["scope"]) for r in rows))
    colors = ["#2563eb", "#dc2626", "#16a34a"]
    width, height = 900, 560
    left, top, plot_w, plot_h = 70, 50, 760, 420
    ks = [float(r["k"]) for r in rows]
    vals = [float(r["mean_p90_radius"]) for r in rows]
    xmin, xmax = min(ks), max(ks)
    ymin, ymax = min(vals), max(vals)

    def sx(value: float) -> float:
        return left + plot_w * (value - xmin) / max(xmax - xmin, 1e-9)

    def sy(value: float) -> float:
        return top + plot_h - plot_h * (value - ymin) / max(ymax - ymin, 1e-9)

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="30" font-family="Arial" font-size="20" fill="#111827">Fine clustering cohesion: lower p90 radius is tighter</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
    ]
    for si, scope in enumerate(scopes):
        sr = sorted([r for r in rows if str(r["scope"]) == scope], key=lambda r: int(r["k"]))
        points = [(sx(float(r["k"])), sy(float(r["mean_p90_radius"]))) for r in sr]
        if len(points) >= 2:
            d = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            items.append(f'<polyline points="{d}" fill="none" stroke="{colors[si % len(colors)]}" stroke-width="2"/>')
        for x, y in points:
            items.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors[si % len(colors)]}"/>')
        items.append(
            f'<text x="{left + 20 + si * 240}" y="{height - 40}" font-family="Arial" font-size="14" fill="{colors[si % len(colors)]}">{scope}</text>'
        )
    items.append('<text x="420" y="535" font-family="Arial" font-size="13" fill="#374151">k</text>')
    items.append('<text x="12" y="265" font-family="Arial" font-size="13" fill="#374151" transform="rotate(-90 12,265)">mean p90 radius</text>')
    items.append("</svg>")
    path.write_text("".join(items), encoding="utf-8")


def svg_scatter(path: Path, title: str, z: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if z.shape[0] == 0:
        path.write_text("", encoding="utf-8")
        return
    idx = sample_indices(int(z.shape[0]), int(args.max_scatter_points), int(args.seed) + len(title))
    z = z[idx]
    labels = labels[idx]
    x = z[:, 0]
    y = z[:, 1] if z.shape[1] > 1 else torch.zeros_like(x)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    def sx(value: torch.Tensor) -> float:
        return 50 + 720 * (float(value) - xmin) / max(xmax - xmin, 1e-9)

    def sy(value: torch.Tensor) -> float:
        return 560 - 500 * (float(value) - ymin) / max(ymax - ymin, 1e-9)

    colors = [
        "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed", "#0891b2", "#be185d", "#4b5563",
        "#ea580c", "#059669", "#9333ea", "#0f766e", "#b91c1c", "#1d4ed8", "#a16207", "#475569",
    ]
    circles = []
    for i in range(z.shape[0]):
        color = colors[int(labels[i]) % len(colors)]
        circles.append(f'<circle cx="{sx(x[i]):.2f}" cy="{sy(y[i]):.2f}" r="2.4" fill="{color}" opacity="0.58"/>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="630" viewBox="0 0 850 630">'
        '<rect width="850" height="630" fill="white"/>'
        f'<text x="50" y="32" font-family="Arial" font-size="20" fill="#111827">{title}</text>'
        '<line x1="50" y1="560" x2="770" y2="560" stroke="#9ca3af"/>'
        '<line x1="50" y1="60" x2="50" y2="560" stroke="#9ca3af"/>'
        + "".join(circles)
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def write_report(path: Path, args: argparse.Namespace, summary: Mapping[str, Any], k_rows: Sequence[Mapping[str, Any]], cluster_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# 细粒度自然/干涉反思 Hidden-State 聚类报告",
        "",
        "## 方法说明",
        "- 这版不再只固定八个簇，而是先扫描多个聚类数，再用更高的聚类数生成细簇。",
        "- 每个细簇额外报告簇内平均半径和 90% 半径；半径越小，说明簇内事件越接近。",
        "- 类型标签不是监督标签，是根据 marker、后续错误承认、修复语言、最终正确率和相对位置给出的启发式解释。",
        "",
        "## 数据概览",
        f"- 事件总数：`{summary.get('events')}`",
        f"- 使用表示：`{args.feature_key}` / `{args.feature_kinds}`",
        f"- PCA 维度：`{args.pca_dims}`",
        f"- 最终聚类数：自然 `{args.final_k_natural}`，干涉后 `{args.final_k_tampered}`，合并 `{args.final_k_combined}`",
        "",
        "## k 扫描结论",
        "更大的 k 会稳定降低簇内半径，但 silhouette 不一定同步升高；因此这里把 k 扫描当作“分得更细是否更紧”的诊断，而不是唯一选型标准。",
        "",
    ]
    for scope in sorted(set(str(r["scope"]) for r in k_rows)):
        rows = [r for r in k_rows if str(r["scope"]) == scope]
        best_sil = max(rows, key=lambda r: float(r["silhouette"]))
        tightest = min(rows, key=lambda r: float(r["mean_p90_radius"]))
        lines.append(
            f"- `{scope}`：最高 silhouette 在 k=`{best_sil['k']}`，值 `{float(best_sil['silhouette']):.3f}`；"
            f"最小 90% 半径在 k=`{tightest['k']}`，值 `{float(tightest['mean_p90_radius']):.3f}`。"
        )
    lines += ["", "## 主要细簇类型", ""]
    for scope in ["natural_baseline", "tampered_continuation", "combined"]:
        rows = sorted([r for r in cluster_rows if str(r["scope"]) == scope], key=lambda r: int(r["count"]), reverse=True)[:12]
        if not rows:
            continue
        lines.append(f"### {scope}")
        lines.append("| cluster | count | marker | 类型 | error_ack | repair | correct | p90_radius |")
        lines.append("|---:|---:|---|---|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                "| {cluster_id} | {count} | {dominant_marker} | {type_label} | {ack:.3f} | {repair:.3f} | {correct:.3f} | {radius:.3f} |".format(
                    cluster_id=row.get("cluster_id"),
                    count=row.get("count"),
                    dominant_marker=row.get("dominant_marker"),
                    type_label=row.get("type_label"),
                    ack=float(row.get("explicit_error_ack_rate") or 0.0),
                    repair=float(row.get("repair_language_rate") or 0.0),
                    correct=float(row.get("final_answer_correct_rate") or 0.0),
                    radius=float(row.get("p90_radius") or 0.0),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_kinds = parse_csv_list(args.feature_kinds)
    ks = parse_ks(args.cluster_ks)
    rows, x, meta = load_all(Path(args.root), args.feature_key, feature_kinds)
    if x.numel() == 0:
        raise RuntimeError("No features loaded.")
    scopes = {
        "natural_baseline": [i for i, r in enumerate(rows) if r.get("scope") == "natural_baseline"],
        "tampered_continuation": [i for i, r in enumerate(rows) if r.get("scope") == "tampered_continuation"],
        "combined": list(range(len(rows))),
    }
    final_ks = {
        "natural_baseline": int(args.final_k_natural),
        "tampered_continuation": int(args.final_k_tampered),
        "combined": int(args.final_k_combined),
    }
    k_sweep_rows: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []
    pca_info: Dict[str, Any] = {}
    for scope, idx in scopes.items():
        if len(idx) < 3:
            continue
        sub_rows = [rows[i] for i in idx]
        sub_x = x[idx]
        z, ratios = pca_project(sub_x, int(args.pca_dims))
        pca_info[scope] = [round(float(v), 6) for v in ratios[: min(20, len(ratios))]]
        k_sweep_rows.extend(run_k_sweep(scope, z, ks, args))
        k = min(final_ks[scope], len(sub_rows))
        metric_idx = sample_indices(int(z.shape[0]), int(args.sample_for_metrics), int(args.seed) + len(scope) + 17)
        metric_labels, metric_centers, _ = kmeans(
            z[metric_idx],
            k,
            steps=int(args.kmeans_steps),
            seed=int(args.seed) + k + len(scope) + 31,
        )
        labels = assign_to_centers(z, metric_centers)
        sil = silhouette(z, labels, max_n=min(5000, int(args.sample_for_metrics)), seed=int(args.seed) + k + 47)
        # Recompute centers on full assigned data for better cohesion stats.
        centers = torch.empty_like(metric_centers)
        for cluster_id in range(k):
            mask = labels == cluster_id
            centers[cluster_id] = z[mask].mean(dim=0) if bool(mask.any()) else metric_centers[cluster_id]
        labels = assign_to_centers(z, centers)
        cluster_rows.extend(summarize_clusters(scope, sub_rows, z, labels, centers, sil))
        for local_i, row in enumerate(sub_rows):
            assignment_rows.append(
                {
                    "scope": scope,
                    "global_event_index": row.get("global_event_index"),
                    "cluster_id": int(labels[local_i]),
                    "marker_kind": row.get("marker_kind"),
                    "example_id": row.get("example_id"),
                }
            )
        svg_scatter(output_dir / "plots" / f"{scope}_fine_cluster.svg", scope, z[:, :2], labels, args)
    svg_k_sweep(output_dir / "plots" / "k_sweep_p90_radius.svg", k_sweep_rows)
    summary = {
        "root": str(args.root),
        "events": len(rows),
        "feature_key": args.feature_key,
        "feature_kinds": feature_kinds,
        "scope_counts": dict(Counter(str(r.get("scope")) for r in rows)),
        "marker_counts": dict(Counter(str(r.get("marker_kind")) for r in rows)),
        "pca_explained": pca_info,
        **meta,
    }
    write_csv(output_dir / "k_sweep.csv", k_sweep_rows)
    write_csv(output_dir / "fine_cluster_summary.csv", cluster_rows)
    write_csv(output_dir / "fine_event_cluster_assignments.csv", assignment_rows)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "REPORT.md", args, summary, k_sweep_rows, cluster_rows)
    print(f"[Done] refined clusters rows={len(cluster_rows)} assignments={len(assignment_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
