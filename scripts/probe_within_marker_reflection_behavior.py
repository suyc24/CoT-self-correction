#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Within-marker controlled probes for reflection behavior.")
    p.add_argument("--root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--feature_key", default="L22/post_attn")
    p.add_argument("--feature_kind", default="h_pre")
    p.add_argument("--scope", default="natural_baseline")
    p.add_argument("--markers", default="wait,but,check,however,wrong")
    p.add_argument("--labels", default="error_ack,repair,productive_repair,failed_reflection")
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--max_train", type=int, default=90000)
    p.add_argument("--max_test", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--min_events", type=int, default=1500)
    return p.parse_args()


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


def load_all(root: Path, feature_key: str, feature_kind: str) -> Tuple[List[Dict[str, Any]], torch.Tensor, Dict[str, Any]]:
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
        tensor = obj.get("features", {}).get(feature_key, {}).get(feature_kind)
        if tensor is None or tensor.ndim != 2 or tensor.shape[0] == 0:
            meta["bad_shards"].append(str(shard))
            continue
        n = min(len(rows), int(tensor.shape[0]))
        for i in range(n):
            row = dict(rows[i])
            row["global_event_index"] = offset + i
            row["shard"] = shard.name
            rows_all.append(row)
        xs.append(tensor[:n].float())
        offset += n
        meta["shards"].append({"path": str(shard), "events": n})
    if not xs:
        return rows_all, torch.empty((0, 0)), meta
    return rows_all, torch.cat(xs, dim=0), meta


def as_bool(row: Mapping[str, Any], key: str) -> bool:
    return str(row.get(key)).lower() in {"true", "1", "yes"}


def label_value(row: Mapping[str, Any], label: str) -> float:
    if label == "repair":
        return float(as_bool(row, "future_repair_language"))
    if label == "error_ack":
        return float(as_bool(row, "future_explicit_error_ack"))
    if label == "productive_repair":
        return float(as_bool(row, "future_repair_language") and as_bool(row, "final_answer_correct"))
    if label == "failed_reflection":
        return float((not as_bool(row, "future_repair_language")) and (not as_bool(row, "final_answer_correct")))
    raise ValueError(label)


def control_features(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    feats = []
    for row in rows:
        rel = float(row.get("relative_position") or 0.0)
        step = math.log1p(float(row.get("event_step") or 0.0))
        feats.append([rel, rel * rel, step, step * step])
    return torch.tensor(feats, dtype=torch.float32)


def split_by_example(rows: Sequence[Mapping[str, Any]], test_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    examples = sorted(set(str(r.get("example_id")) for r in rows))
    rng = random.Random(seed)
    rng.shuffle(examples)
    test_examples = set(examples[: max(1, int(round(len(examples) * test_frac)))])
    train, test = [], []
    for i, row in enumerate(rows):
        (test if str(row.get("example_id")) in test_examples else train).append(i)
    return torch.tensor(train, dtype=torch.long), torch.tensor(test, dtype=torch.long)


def one_event_per_example_indices(rows: Sequence[Mapping[str, Any]], seed: int) -> torch.Tensor:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row.get("example_id"))].append(i)
    rng = random.Random(seed)
    chosen = [rng.choice(v) for _, v in sorted(groups.items())]
    return torch.tensor(sorted(chosen), dtype=torch.long)


def subsample(idx: torch.Tensor, max_n: int, seed: int) -> torch.Tensor:
    if max_n <= 0 or idx.numel() <= max_n:
        return idx
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return idx[torch.randperm(idx.numel(), generator=gen)[:max_n]].sort().values


def standardize_train_test(x_train: torch.Tensor, x_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def residualize(
    h_train: torch.Tensor,
    h_test: torch.Tensor,
    c_train: torch.Tensor,
    c_test: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    c_train, c_test = standardize_train_test(c_train, c_test)
    c_train = torch.cat([torch.ones((c_train.shape[0], 1)), c_train], dim=1)
    c_test = torch.cat([torch.ones((c_test.shape[0], 1)), c_test], dim=1)
    reg = float(ridge) * torch.eye(c_train.shape[1])
    reg[0, 0] = 0.0
    w = torch.linalg.solve(c_train.T @ c_train + reg, c_train.T @ h_train)
    return h_train - c_train @ w, h_test - c_test @ w


def train_probe(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, args: argparse.Namespace, seed: int) -> torch.Tensor:
    x_train, x_test = standardize_train_test(x_train.float(), x_test.float())
    torch.manual_seed(seed)
    model = torch.nn.Linear(int(x_train.shape[1]), 1)
    pos = y_train.sum().clamp_min(1.0)
    neg = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=neg / pos)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 99)
    n = int(x_train.shape[0])
    for _ in range(int(args.epochs)):
        order = torch.randperm(n, generator=gen)
        for start in range(0, n, int(args.batch_size)):
            batch = order[start : start + int(args.batch_size)]
            logits = model(x_train[batch]).squeeze(1)
            loss = loss_fn(logits, y_train[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(x_test).squeeze(1)).cpu()


def auc_score(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, y_score.numel() + 1, dtype=torch.float32)
    return float((ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(y_true: torch.Tensor, y_score: torch.Tensor) -> Dict[str, float]:
    pred = y_score >= 0.5
    yb = y_true.bool()
    tp = int((pred & yb).sum().item())
    tn = int((~pred & ~yb).sum().item())
    fp = int((pred & ~yb).sum().item())
    fn = int((~pred & yb).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "auc": auc_score(y_true, y_score),
        "balanced_accuracy": (recall + tnr) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def run_subset(
    rows: Sequence[Mapping[str, Any]],
    h: torch.Tensor,
    marker: str,
    label: str,
    mode: str,
    args: argparse.Namespace,
    seed: int,
) -> List[Dict[str, Any]]:
    idx = [i for i, row in enumerate(rows) if str(row.get("scope")) == str(args.scope) and str(row.get("marker_kind")) == marker]
    if len(idx) < int(args.min_events):
        return []
    sub_rows = [rows[i] for i in idx]
    sub_h = h[idx]
    if mode == "one_per_example":
        keep = one_event_per_example_indices(sub_rows, seed)
        if keep.numel() < 200:
            return []
        sub_rows = [sub_rows[int(i)] for i in keep.tolist()]
        sub_h = sub_h[keep]
    y = torch.tensor([label_value(row, label) for row in sub_rows], dtype=torch.float32)
    if y.sum() < 30 or (y.numel() - y.sum()) < 30:
        return []
    controls = control_features(sub_rows)
    train_idx, test_idx = split_by_example(sub_rows, float(args.test_frac), seed)
    train_idx = subsample(train_idx, int(args.max_train), seed + 1)
    test_idx = subsample(test_idx, int(args.max_test), seed + 2)
    y_train, y_test = y[train_idx], y[test_idx]
    h_train, h_test = sub_h[train_idx], sub_h[test_idx]
    c_train, c_test = controls[train_idx], controls[test_idx]
    h_res_train, h_res_test = residualize(h_train, h_test, c_train, c_test, float(args.ridge))
    experiments = {
        "position_controls": (c_train, c_test),
        "h_pre": (h_train, h_test),
        "h_pre_residualized_position": (h_res_train, h_res_test),
    }
    out = []
    for j, (feature, (x_train, x_test)) in enumerate(experiments.items()):
        scores = train_probe(x_train, y_train, x_test, args, seed + j * 100)
        out.append(
            {
                "scope": args.scope,
                "marker": marker,
                "label": label,
                "mode": mode,
                "feature": feature,
                "train_n": int(train_idx.numel()),
                "test_n": int(test_idx.numel()),
                "train_positive_rate": float(y_train.mean().item()),
                "test_positive_rate": float(y_test.mean().item()),
                **metrics(y_test, scores),
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
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


def write_report(path: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# 同一反思词内部行为预测报告",
        "",
        "## 目的",
        "排除 marker identity 主导的解释：只在同一种反思词内部，检验 pre-marker hidden state 是否仍能预测后续反思行为。",
        "",
        "## 数据",
        f"- scope：`{summary['scope']}`",
        f"- 事件总数：`{summary['events']}`",
        f"- marker 计数：`{summary['marker_counts']}`",
        "",
        "## 结果",
        "| marker | label | mode | feature | AUC | bal_acc | F1 | pos_test | n_test |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["marker"], r["label"], r["mode"], r["feature"])):
        lines.append(
            "| {marker} | {label} | {mode} | {feature} | {auc:.4f} | {balanced_accuracy:.4f} | {f1:.4f} | {pos:.3f} | {n} |".format(
                marker=row["marker"],
                label=row["label"],
                mode=row["mode"],
                feature=row["feature"],
                auc=float(row["auc"]),
                balanced_accuracy=float(row["balanced_accuracy"]),
                f1=float(row["f1"]),
                pos=float(row["test_positive_rate"]),
                n=row["test_n"],
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, h, meta = load_all(Path(args.root), args.feature_key, args.feature_kind)
    markers = [x.strip() for x in args.markers.split(",") if x.strip()]
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    out: List[Dict[str, Any]] = []
    seed = int(args.seed)
    for marker in markers:
        for label in labels:
            for mode in ["all_events", "one_per_example"]:
                out.extend(run_subset(rows, h, marker, label, mode, args, seed))
                seed += 1000
    scoped = [r for r in rows if str(r.get("scope")) == str(args.scope)]
    summary = {
        "root": str(args.root),
        "scope": args.scope,
        "events": len(scoped),
        "feature_key": args.feature_key,
        "feature_kind": args.feature_kind,
        "marker_counts": dict(Counter(str(r.get("marker_kind")) for r in scoped)),
        **meta,
    }
    write_csv(output_dir / "within_marker_probe_summary.csv", out)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "REPORT.md", out, summary)
    print(f"[Done] rows={len(out)} output={output_dir}")


if __name__ == "__main__":
    main()
