#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.cluster import AgglomerativeClustering, Birch, KMeans, MiniBatchKMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
except Exception:  # pragma: no cover
    AgglomerativeClustering = None
    Birch = None
    KMeans = None
    MiniBatchKMeans = None
    GaussianMixture = None
    adjusted_mutual_info_score = None
    calinski_harabasz_score = None
    davies_bouldin_score = None
    normalized_mutual_info_score = None
    silhouette_score = None

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clustering methods on natural reflection hidden states."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="L22/post_attn")
    parser.add_argument(
        "--feature_sets",
        default="h_pre;h_pre,h_marker,delta_marker,delta_post",
        help="Semicolon-separated feature-kind sets.",
    )
    parser.add_argument("--scope", default="natural_baseline")
    parser.add_argument("--sample_size", type=int, default=30000)
    parser.add_argument("--metric_sample_size", type=int, default=8000)
    parser.add_argument("--pca_dims", type=int, default=48)
    parser.add_argument("--ks", default="8,12,16,24,32,40")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--kmeans_steps", type=int, default=80)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=250)
    return parser.parse_args()


def parse_feature_sets(text: str) -> List[List[str]]:
    out = []
    for part in text.split(";"):
        kinds = [item.strip() for item in part.split(",") if item.strip()]
        if kinds:
            out.append(kinds)
    return out


def parse_ks(text: str) -> List[int]:
    out: List[int] = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(k for k in out if k > 1))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def shard_dirs(root: Path) -> List[Path]:
    if (root / "event_rows.jsonl").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "event_rows.jsonl").exists())


def load_rows_and_features(
    root: Path,
    feature_key: str,
    feature_kinds: Sequence[str],
    scope: str,
) -> Tuple[List[Dict[str, Any]], torch.Tensor, Dict[str, Any]]:
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
        keep = [i for i in range(n) if str(rows[i].get("scope")) == scope]
        if not keep:
            meta["shards"].append({"path": str(shard), "events": 0})
            continue
        keep_t = torch.tensor(keep, dtype=torch.long)
        for i in keep:
            row = dict(rows[i])
            row["global_event_index"] = offset
            row["shard"] = shard.name
            rows_all.append(row)
            offset += 1
        xs.append(x[keep_t])
        meta["shards"].append({"path": str(shard), "events": len(keep)})
    if not xs:
        return rows_all, torch.empty((0, 0)), meta
    return rows_all, torch.cat(xs, dim=0), meta


def standardize(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1e-6)


def pca_project(x: torch.Tensor, dims: int) -> Tuple[torch.Tensor, List[float]]:
    x = standardize(x)
    q = min(int(dims), int(x.shape[0]) - 1, int(x.shape[1]))
    _, s, v = torch.pca_lowrank(x, q=q)
    z = x @ v
    var = s.pow(2)
    ratios = (var / var.sum().clamp_min(1e-12)).tolist()
    return z.contiguous(), [float(v) for v in ratios]


def sample_indices(n: int, max_n: int, seed: int) -> torch.Tensor:
    if max_n <= 0 or n <= max_n:
        return torch.arange(n)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randperm(n, generator=gen)[:max_n].sort().values


def l2_normalize_np(x: "np.ndarray") -> "np.ndarray":
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom < 1e-12] = 1.0
    return x / denom


def torch_kmeans(x: torch.Tensor, k: int, steps: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    n = int(x.shape[0])
    k = min(k, n)
    centers = x[torch.randperm(n, generator=gen)[:k]].clone()
    labels = torch.full((n,), -1, dtype=torch.long)
    for _ in range(steps):
        new_labels = torch.cdist(x, centers).argmin(dim=1)
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for cid in range(k):
            mask = labels == cid
            if bool(mask.any()):
                centers[cid] = x[mask].mean(dim=0)
    return labels


def behavior_label(row: Mapping[str, Any]) -> str:
    ack = str(row.get("future_explicit_error_ack")).lower() in {"true", "1", "yes"}
    repair = str(row.get("future_repair_language")).lower() in {"true", "1", "yes"}
    correct = str(row.get("final_answer_correct")).lower() in {"true", "1", "yes"}
    marker = str(row.get("marker_kind"))
    if marker == "wrong":
        return "wrong_marker"
    if ack and repair and correct:
        return "ack_repair_correct"
    if ack and repair:
        return "ack_repair"
    if repair and correct:
        return "repair_correct"
    if repair:
        return "repair_only"
    if not repair and not correct:
        return "no_repair_wrong"
    return "other"


def cluster_size_stats(labels: Sequence[int]) -> Dict[str, float]:
    counts = Counter(int(v) for v in labels if int(v) >= 0)
    total = sum(counts.values())
    if total == 0 or not counts:
        return {
            "effective_clusters": 0,
            "noise_fraction": sum(1 for v in labels if int(v) < 0) / max(len(labels), 1),
            "min_cluster_size": 0,
            "max_cluster_size": 0,
            "size_entropy": 0.0,
        }
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return {
        "effective_clusters": len(counts),
        "noise_fraction": sum(1 for v in labels if int(v) < 0) / max(len(labels), 1),
        "min_cluster_size": min(counts.values()),
        "max_cluster_size": max(counts.values()),
        "size_entropy": entropy,
    }


def purity(labels: Sequence[int], targets: Sequence[str]) -> float:
    grouped: Dict[int, Counter[str]] = {}
    total = 0
    hit = 0
    for label, target in zip(labels, targets):
        label = int(label)
        if label < 0:
            continue
        grouped.setdefault(label, Counter())[str(target)] += 1
    for counter in grouped.values():
        total += sum(counter.values())
        hit += counter.most_common(1)[0][1]
    return hit / max(total, 1)


def metric_scores(
    x: "np.ndarray",
    labels: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
    metric_sample_size: int,
    seed: int,
) -> Dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int64)
    if metric_sample_size > 0 and len(labels_np) > metric_sample_size:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(labels_np), size=metric_sample_size, replace=False))
        x_metric = x[idx]
        labels_metric = labels_np[idx]
        rows_metric = [rows[int(i)] for i in idx.tolist()]
    else:
        x_metric = x
        labels_metric = labels_np
        rows_metric = rows
    labels_metric_np = np.asarray(labels_metric, dtype=np.int64)
    non_noise = labels_np >= 0
    metric_non_noise = labels_metric_np >= 0
    unique = sorted(set(int(v) for v in labels_metric_np[metric_non_noise].tolist()))
    out: Dict[str, float] = cluster_size_stats(labels_np.tolist())
    if len(unique) >= 2 and int(metric_non_noise.sum()) > len(unique):
        x_eval = x_metric[metric_non_noise]
        y_eval = labels_metric_np[metric_non_noise]
        try:
            out["silhouette"] = float(silhouette_score(x_eval, y_eval, metric=metric_name))
        except Exception:
            out["silhouette"] = float("nan")
        try:
            out["davies_bouldin"] = float(davies_bouldin_score(x_eval, y_eval))
        except Exception:
            out["davies_bouldin"] = float("nan")
        try:
            out["calinski_harabasz"] = float(calinski_harabasz_score(x_eval, y_eval))
        except Exception:
            out["calinski_harabasz"] = float("nan")
    else:
        out.update({"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")})
    markers = [str(r.get("marker_kind")) for r in rows_metric]
    behaviors = [behavior_label(r) for r in rows_metric]
    metric_labels = labels_metric_np
    if normalized_mutual_info_score is not None:
        out["marker_nmi"] = float(normalized_mutual_info_score(markers, metric_labels))
        out["behavior_nmi"] = float(normalized_mutual_info_score(behaviors, metric_labels))
        out["marker_ami"] = float(adjusted_mutual_info_score(markers, metric_labels))
        out["behavior_ami"] = float(adjusted_mutual_info_score(behaviors, metric_labels))
    out["marker_purity"] = float(purity(metric_labels.tolist(), markers))
    out["behavior_purity"] = float(purity(metric_labels.tolist(), behaviors))
    return out


def run_methods(
    z_np: "np.ndarray",
    rows: Sequence[Mapping[str, Any]],
    ks: Sequence[int],
    args: argparse.Namespace,
    feature_set_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    result_rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {}
    if np is None:
        z_t = torch.tensor(z_np, dtype=torch.float32)
        for k in ks:
            labels = torch_kmeans(z_t, k=k, steps=int(args.kmeans_steps), seed=int(args.seed) + k).tolist()
            result_rows.append(
                {
                    "feature_set": feature_set_name,
                    "method": "torch_kmeans",
                    "k": k,
                    **metric_scores(z_np, labels, rows, "euclidean", int(args.metric_sample_size), int(args.seed) + k),
                }
            )
        return result_rows, payload

    for k in ks:
        if KMeans is not None:
            model = KMeans(n_clusters=k, n_init=10, random_state=int(args.seed), max_iter=300)
            labels = model.fit_predict(z_np)
            result_rows.append({"feature_set": feature_set_name, "method": "kmeans", "k": k, **metric_scores(z_np, labels, rows, "euclidean", int(args.metric_sample_size), int(args.seed) + k)})
        if MiniBatchKMeans is not None:
            model = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=int(args.seed), batch_size=4096, max_iter=300)
            labels = model.fit_predict(z_np)
            result_rows.append({"feature_set": feature_set_name, "method": "minibatch_kmeans", "k": k, **metric_scores(z_np, labels, rows, "euclidean", int(args.metric_sample_size), int(args.seed) + k)})
        if KMeans is not None:
            z_sphere = l2_normalize_np(z_np)
            model = KMeans(n_clusters=k, n_init=10, random_state=int(args.seed), max_iter=300)
            labels = model.fit_predict(z_sphere)
            result_rows.append({"feature_set": feature_set_name, "method": "spherical_kmeans", "k": k, **metric_scores(z_sphere, labels, rows, "cosine", int(args.metric_sample_size), int(args.seed) + k + 11)})
        if GaussianMixture is not None:
            for cov in ["diag", "full"]:
                try:
                    model = GaussianMixture(n_components=k, covariance_type=cov, random_state=int(args.seed), max_iter=200, reg_covar=1e-5)
                    labels = model.fit_predict(z_np)
                    result_rows.append({"feature_set": feature_set_name, "method": f"gmm_{cov}", "k": k, **metric_scores(z_np, labels, rows, "euclidean", int(args.metric_sample_size), int(args.seed) + k)})
                except Exception as exc:
                    result_rows.append({"feature_set": feature_set_name, "method": f"gmm_{cov}", "k": k, "status": f"failed:{type(exc).__name__}"})
        if Birch is not None:
            try:
                model = Birch(n_clusters=k, threshold=0.5)
                labels = model.fit_predict(z_np)
                result_rows.append({"feature_set": feature_set_name, "method": "birch", "k": k, **metric_scores(z_np, labels, rows, "euclidean", int(args.metric_sample_size), int(args.seed) + k)})
            except Exception as exc:
                result_rows.append({"feature_set": feature_set_name, "method": "birch", "k": k, "status": f"failed:{type(exc).__name__}"})
        if AgglomerativeClustering is not None and len(rows) <= 12000:
            for linkage, metric in [("ward", "euclidean"), ("average", "cosine")]:
                try:
                    kwargs = {"n_clusters": k, "linkage": linkage}
                    if linkage != "ward":
                        kwargs["metric"] = metric
                    model = AgglomerativeClustering(**kwargs)
                    x_for_model = z_np if metric == "euclidean" else l2_normalize_np(z_np)
                    labels = model.fit_predict(x_for_model)
                    result_rows.append({"feature_set": feature_set_name, "method": f"agglomerative_{linkage}_{metric}", "k": k, **metric_scores(x_for_model, labels, rows, metric, int(args.metric_sample_size), int(args.seed) + k + 23)})
                except Exception as exc:
                    result_rows.append({"feature_set": feature_set_name, "method": f"agglomerative_{linkage}_{metric}", "k": k, "status": f"failed:{type(exc).__name__}"})
    if hdbscan is not None and np is not None:
        for min_cluster_size in [int(args.hdbscan_min_cluster_size), int(args.hdbscan_min_cluster_size) * 2, int(args.hdbscan_min_cluster_size) * 4]:
            try:
                model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(10, min_cluster_size // 10), metric="euclidean")
                labels = model.fit_predict(z_np)
                result_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "method": "hdbscan",
                        "k": "",
                        "min_cluster_size": min_cluster_size,
                        **metric_scores(z_np, labels, rows, "euclidean", int(args.metric_sample_size), int(args.seed) + min_cluster_size),
                    }
                )
            except Exception as exc:
                result_rows.append({"feature_set": feature_set_name, "method": "hdbscan", "k": "", "min_cluster_size": min_cluster_size, "status": f"failed:{type(exc).__name__}"})
    return result_rows, payload


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


def best_rows(rows: Sequence[Mapping[str, Any]], top_n: int = 20) -> List[Mapping[str, Any]]:
    valid = [r for r in rows if str(r.get("status", "")) == "" and str(r.get("silhouette", "nan")).lower() != "nan"]
    return sorted(valid, key=lambda r: (float(r.get("silhouette") or -999), float(r.get("behavior_nmi") or 0.0)), reverse=True)[:top_n]


def write_report(path: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# 自然反思状态聚类方法横向比较",
        "",
        "## 设置",
        f"- 只使用 scope：`{summary['scope']}`",
        f"- 总自然反思事件：`{summary['events']}`",
        f"- 每个 feature set 抽样：`{summary['sample_size']}`",
        f"- PCA 维度：`{summary['pca_dims']}`",
        "",
        "## 指标解释",
        "- silhouette 越高越好，衡量簇内近、簇间远。",
        "- Davies-Bouldin 越低越好。",
        "- marker/behavior NMI 越高，说明簇和可解释标签越相关。",
        "- purity 容易被大 k 抬高，只作为辅助。",
        "",
        "## Top Methods By Silhouette",
        "| feature_set | method | k | silhouette | DB | behavior_nmi | marker_nmi | clusters | noise |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_rows(rows, top_n=25):
        lines.append(
            "| {feature_set} | {method} | {k} | {sil:.4f} | {db:.3f} | {bnmi:.4f} | {mnmi:.4f} | {clusters} | {noise:.3f} |".format(
                feature_set=row.get("feature_set", ""),
                method=row.get("method", ""),
                k=row.get("k", ""),
                sil=float(row.get("silhouette") or float("nan")),
                db=float(row.get("davies_bouldin") or float("nan")),
                bnmi=float(row.get("behavior_nmi") or 0.0),
                mnmi=float(row.get("marker_nmi") or 0.0),
                clusters=row.get("effective_clusters", ""),
                noise=float(row.get("noise_fraction") or 0.0),
            )
        )
    lines += [
        "",
        "## 结论提示",
        "- 如果所有方法 silhouette 仍低，优先考虑反思状态是连续流形，而不是清楚离散簇。",
        "- 如果球面 K-means 或 average-cosine 层次聚类更好，说明方向信息比向量长度更重要。",
        "- 如果 GMM 更好，说明簇形状可能不是球形。",
        "- 如果 HDBSCAN 产生大量 noise，但少数簇很纯，说明只有高密度核心反思类型清楚，边界区域是连续过渡。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if np is None:
        print("[Warn] numpy unavailable; only torch kmeans fallback will run.", file=sys.stderr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = parse_feature_sets(args.feature_sets)
    ks = parse_ks(args.ks)
    all_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "root": str(args.root),
        "scope": args.scope,
        "feature_key": args.feature_key,
        "feature_sets": feature_sets,
        "sample_size": args.sample_size,
        "pca_dims": args.pca_dims,
        "ks": ks,
        "sklearn_available": KMeans is not None,
        "hdbscan_available": hdbscan is not None,
    }
    for fi, kinds in enumerate(feature_sets):
        feature_set_name = "+".join(kinds)
        rows, x, meta = load_rows_and_features(Path(args.root), args.feature_key, kinds, args.scope)
        if x.numel() == 0:
            continue
        idx = sample_indices(int(x.shape[0]), int(args.sample_size), int(args.seed) + fi)
        sub_rows = [rows[int(i)] for i in idx.tolist()]
        sub_x = x[idx]
        z, ratios = pca_project(sub_x, int(args.pca_dims))
        z_np = z.numpy() if np is not None else []
        method_rows, _ = run_methods(z_np, sub_rows, ks, args, feature_set_name) if np is not None else ([], {})
        if np is None:
            labels = torch_kmeans(z, k=12, steps=int(args.kmeans_steps), seed=int(args.seed)).tolist()
            method_rows = [{"feature_set": feature_set_name, "method": "torch_kmeans", "k": 12, **metric_scores(z.numpy(), labels, sub_rows, "euclidean", int(args.metric_sample_size), int(args.seed))}]
        for row in method_rows:
            row["feature_kinds"] = ",".join(kinds)
            row["sample_size"] = len(sub_rows)
            all_rows.append(dict(row))
        summary.setdefault("feature_set_summaries", {})[feature_set_name] = {
            "loaded_events": len(rows),
            "sampled_events": len(sub_rows),
            "pca_explained_first10": [round(float(v), 6) for v in ratios[:10]],
            **meta,
        }
        summary["events"] = len(rows)
    write_csv(output_dir / "method_summary.csv", all_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "REPORT.md", summary, all_rows)
    print(f"[Done] methods={len(all_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
