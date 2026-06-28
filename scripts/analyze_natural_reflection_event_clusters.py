#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster first-pass and post-tamper natural reflection hidden-state events.")
    p.add_argument("--root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--feature_key", default="L22/post_attn")
    p.add_argument("--feature_kinds", default="h_pre,h_marker,delta_marker,delta_post")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--pca_dims", type=int, default=32)
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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
    return (x.float() - x.float().mean(dim=0, keepdim=True)) / x.float().std(dim=0, keepdim=True).clamp_min(1e-6)


def pca(x: torch.Tensor, dims: int) -> Tuple[torch.Tensor, List[float]]:
    x = standardize(x)
    q = min(int(dims), x.shape[0] - 1, x.shape[1])
    u, s, v = torch.pca_lowrank(x, q=q)
    z = x @ v
    var = s.pow(2)
    ratios = (var / var.sum().clamp_min(1e-12)).tolist()
    return z, [float(v) for v in ratios]


def kmeans(x: torch.Tensor, k: int, steps: int = 80, seed: int = 1234) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n = int(x.shape[0])
    if n <= k:
        return torch.arange(n)
    centers = x[torch.randperm(n, generator=generator)[:k]].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(steps):
        new_labels = torch.cdist(x, centers).argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if bool(mask.any()):
                centers[cluster_id] = x[mask].mean(dim=0)
    return labels


def silhouette(x: torch.Tensor, labels: torch.Tensor) -> float:
    n = int(x.shape[0])
    if n < 3 or labels.unique().numel() < 2:
        return float("nan")
    if n > 2500:
        idx = torch.linspace(0, n - 1, 2500).long()
        x = x[idx]
        labels = labels[idx]
        n = int(x.shape[0])
    distances = torch.cdist(x, x)
    values = []
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = distances[i][same].mean().item() if bool(same.any()) else 0.0
        b = float("inf")
        for label in labels.unique().tolist():
            if int(label) == int(labels[i]):
                continue
            mask = labels == int(label)
            if bool(mask.any()):
                b = min(b, distances[i][mask].mean().item())
        values.append((b - a) / max(a, b, 1e-12))
    return float(sum(values) / len(values))


def as_bool(row: Dict[str, Any], key: str) -> bool:
    return str(row.get(key)).lower() in {"true", "1", "yes"}


def summarize(scope: str, rows: List[Dict[str, Any]], labels: torch.Tensor, sil: float) -> List[Dict[str, Any]]:
    out = []
    raw_labels = labels.tolist()
    for cluster_id in sorted(set(int(x) for x in raw_labels)):
        idx = [i for i, value in enumerate(raw_labels) if int(value) == cluster_id]
        cluster_rows = [rows[i] for i in idx]
        marker_counts = Counter(str(r.get("marker_kind")) for r in cluster_rows)
        scope_counts = Counter(str(r.get("scope")) for r in cluster_rows)
        n = len(cluster_rows)
        out.append({
            "scope": scope,
            "cluster_id": cluster_id,
            "count": n,
            "fraction": n / max(len(rows), 1),
            "dominant_marker": marker_counts.most_common(1)[0][0] if marker_counts else "",
            "marker_counts": json.dumps(dict(marker_counts), ensure_ascii=False),
            "scope_counts": json.dumps(dict(scope_counts), ensure_ascii=False),
            "silhouette": sil,
            "explicit_error_ack_rate": sum(as_bool(r, "future_explicit_error_ack") for r in cluster_rows) / max(n, 1),
            "repair_language_rate": sum(as_bool(r, "future_repair_language") for r in cluster_rows) / max(n, 1),
            "final_answer_correct_rate": sum(as_bool(r, "final_answer_correct") for r in cluster_rows) / max(n, 1),
            "final_answer_matches_tamper_rate": sum(as_bool(r, "final_answer_matches_tamper") for r in cluster_rows) / max(n, 1),
            "mean_relative_position": sum(float(r.get("relative_position") or 0.0) for r in cluster_rows) / max(n, 1),
            "mean_event_step": sum(float(r.get("event_step") or 0.0) for r in cluster_rows) / max(n, 1),
        })
    return out


def svg_scatter(path: Path, title: str, z2: torch.Tensor, labels: torch.Tensor, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if z2.shape[0] == 0:
        path.write_text("", encoding="utf-8")
        return
    x = z2[:, 0]
    y = z2[:, 1] if z2.shape[1] > 1 else torch.zeros_like(x)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    def sx(value):
        return 50 + 700 * (float(value) - xmin) / max(xmax - xmin, 1e-9)
    def sy(value):
        return 550 - 500 * (float(value) - ymin) / max(ymax - ymin, 1e-9)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#7c3aed", "#0891b2", "#be185d", "#4b5563", "#ea580c", "#059669"]
    circles = []
    for i in range(z2.shape[0]):
        color = colors[int(labels[i]) % len(colors)]
        stroke = "#111827" if str(rows[i].get("scope")) == "tampered_continuation" else "none"
        circles.append(f"<circle cx=\"{sx(x[i]):.2f}\" cy=\"{sy(y[i]):.2f}\" r=\"3.2\" fill=\"{color}\" stroke=\"{stroke}\" stroke-width=\"0.8\" opacity=\"0.72\"/>")
    body = "".join(circles)
    svg = f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"820\" height=\"620\" viewBox=\"0 0 820 620\"><rect width=\"820\" height=\"620\" fill=\"white\"/><text x=\"50\" y=\"32\" font-family=\"Arial\" font-size=\"20\" fill=\"#111827\">{title}</text><line x1=\"50\" y1=\"550\" x2=\"760\" y2=\"550\" stroke=\"#9ca3af\"/><line x1=\"50\" y1=\"50\" x2=\"50\" y2=\"550\" stroke=\"#9ca3af\"/>{body}</svg>"
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = [item.strip() for item in args.feature_kinds.split(",") if item.strip()]
    rows, x, meta = load_all(Path(args.root), args.feature_key, kinds)
    if x.numel() == 0:
        raise RuntimeError("No event features loaded.")
    scopes = {
        "natural_baseline": [i for i, r in enumerate(rows) if r.get("scope") == "natural_baseline"],
        "tampered_continuation": [i for i, r in enumerate(rows) if r.get("scope") == "tampered_continuation"],
        "combined": list(range(len(rows))),
    }
    cluster_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []
    kind_text = ", ".join(kinds)
    report = ["# 自然反思 Hidden-State 聚类报告", "", "## 数据概览", f"- 事件总数：`{len(rows)}`", f"- 使用表示：`{args.feature_key}` 的 `{kind_text}`", f"- 聚类数：`{int(args.k)}`", ""]
    for scope, idx in scopes.items():
        if len(idx) < max(3, int(args.k)):
            report += [f"## {scope}", f"事件数 `{len(idx)}` 太少，未聚类。", ""]
            continue
        sub_rows = [rows[i] for i in idx]
        sub_x = x[idx]
        z, ratios = pca(sub_x, int(args.pca_dims))
        labels = kmeans(z, int(args.k), seed=1234 + len(idx))
        sil = silhouette(z[:, : min(20, z.shape[1])], labels)
        cluster_rows.extend(summarize(scope, sub_rows, labels, sil))
        for local_i, row in enumerate(sub_rows):
            assignment_rows.append({"scope": scope, "global_event_index": row.get("global_event_index"), "cluster_id": int(labels[local_i]), "marker_kind": row.get("marker_kind"), "example_id": row.get("example_id")})
        svg_scatter(out / "plots" / f"{scope}_cluster.svg", scope, z[:, :2], labels, sub_rows)
        report += [f"## {scope}", f"- 事件数：`{len(idx)}`", f"- silhouette：`{sil:.3f}`", f"- PCA 前五维解释率：`{[round(float(v), 4) for v in ratios[:5]]}`", ""]
    write_csv(out / "cluster_summary.csv", cluster_rows)
    write_csv(out / "event_cluster_assignments.csv", assignment_rows)
    write_json(out / "summary.json", {"root": str(args.root), "events": len(rows), "feature_key": args.feature_key, "feature_kinds": kinds, "scope_counts": dict(Counter(str(r.get("scope")) for r in rows)), "marker_counts": dict(Counter(str(r.get("marker_kind")) for r in rows)), **meta})
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[Done] clustered events={len(rows)} rows={len(cluster_rows)} output={out}")


if __name__ == "__main__":
    main()
