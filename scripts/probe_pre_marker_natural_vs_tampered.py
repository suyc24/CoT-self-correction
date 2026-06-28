#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe natural-vs-tampered reflection events using only pre-marker hidden states."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_key", default="L22/post_attn")
    parser.add_argument("--feature_kind", default="h_pre")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_train", type=int, default=180000)
    parser.add_argument("--max_test", type=int, default=80000)
    return parser.parse_args()


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
        features = obj.get("features", {})
        tensor = features.get(feature_key, {}).get(feature_kind)
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


def split_by_example(rows: Sequence[Mapping[str, Any]], test_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    examples = sorted(set(str(row.get("example_id")) for row in rows))
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_test = max(1, int(round(len(examples) * float(test_frac))))
    test_examples = set(examples[:n_test])
    train_idx = []
    test_idx = []
    for i, row in enumerate(rows):
        if str(row.get("example_id")) in test_examples:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return torch.tensor(train_idx, dtype=torch.long), torch.tensor(test_idx, dtype=torch.long)


def subsample_indices(idx: torch.Tensor, max_n: int, seed: int) -> torch.Tensor:
    if max_n <= 0 or idx.numel() <= max_n:
        return idx
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    choice = torch.randperm(idx.numel(), generator=gen)[:max_n]
    return idx[choice].sort().values


def labels_from_rows(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    values = [1.0 if str(row.get("scope")) == "tampered_continuation" else 0.0 for row in rows]
    return torch.tensor(values, dtype=torch.float32)


def numeric_baseline_features(rows: Sequence[Mapping[str, Any]], mode: str) -> torch.Tensor:
    marker_vocab = ["wait", "but", "check", "hold_on", "however", "wrong", "actually"]
    marker_to_idx = {name: i for i, name in enumerate(marker_vocab)}
    feats = []
    for row in rows:
        rel = float(row.get("relative_position") or 0.0)
        step = math.log1p(float(row.get("event_step") or 0.0))
        if mode == "position":
            feats.append([rel, step])
        elif mode == "marker":
            v = [0.0] * len(marker_vocab)
            marker = str(row.get("marker_kind"))
            if marker in marker_to_idx:
                v[marker_to_idx[marker]] = 1.0
            feats.append(v)
        elif mode == "marker_position":
            v = [rel, step] + [0.0] * len(marker_vocab)
            marker = str(row.get("marker_kind"))
            if marker in marker_to_idx:
                v[2 + marker_to_idx[marker]] = 1.0
            feats.append(v)
        else:
            raise ValueError(f"unknown baseline mode: {mode}")
    return torch.tensor(feats, dtype=torch.float32)


def standardize_train_test(x_train: torch.Tensor, x_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def train_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    args: argparse.Namespace,
    seed_offset: int = 0,
) -> torch.Tensor:
    torch.manual_seed(int(args.seed) + int(seed_offset))
    x_train, x_test = standardize_train_test(x_train.float(), x_test.float())
    model = torch.nn.Linear(int(x_train.shape[1]), 1)
    pos = y_train.sum().clamp_min(1.0)
    neg = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=(neg / pos))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    n = int(x_train.shape[0])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed) + int(seed_offset) + 991)
    for _ in range(int(args.epochs)):
        order = torch.randperm(n, generator=gen)
        for start in range(0, n, int(args.batch_size)):
            batch = order[start : start + int(args.batch_size)]
            logits = model(x_train[batch]).squeeze(1)
            loss = criterion(logits, y_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(model(x_test).squeeze(1)).cpu()


def auc_score(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.float()
    y_score = y_score.float()
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, y_score.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[pos].sum().item()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics(y_true: torch.Tensor, y_score: torch.Tensor) -> Dict[str, float]:
    pred = y_score >= 0.5
    yb = y_true.bool()
    tp = int((pred & yb).sum().item())
    tn = int((~pred & ~yb).sum().item())
    fp = int((pred & ~yb).sum().item())
    fn = int((~pred & yb).sum().item())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    tpr = recall
    tnr = tn / max(tn + fp, 1)
    return {
        "auc": auc_score(y_true, y_score),
        "accuracy": float(acc),
        "balanced_accuracy": float((tpr + tnr) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


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
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Pre-Marker Hidden State 自然/干涉分类报告",
        "",
        "## 问题",
        "只使用反思词生成之前的 hidden state，判断这个反思事件来自自然生成还是人工改 box 后的 continuation。",
        "",
        "## 数据",
        f"- 事件总数：`{summary['events']}`",
        f"- 训练事件：`{summary['train_events']}`",
        f"- 测试事件：`{summary['test_events']}`",
        f"- 训练/测试按题目切分，同一道题不会同时出现在两边。",
        "",
        "## 结果",
        "| feature | AUC | accuracy | balanced_accuracy | F1 | recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {auc:.4f} | {accuracy:.4f} | {balanced_accuracy:.4f} | {f1:.4f} | {recall:.4f} |".format(
                feature=row["feature"],
                auc=float(row["auc"]),
                accuracy=float(row["accuracy"]),
                balanced_accuracy=float(row["balanced_accuracy"]),
                f1=float(row["f1"]),
                recall=float(row["recall"]),
            )
        )
    lines += [
        "",
        "## 解释",
        "- `h_pre` 是主实验：反思词还没生成时的模型状态。",
        "- `position` 是泄漏检查：干涉后反思通常更早出现，如果这个 baseline 很强，说明分类器可能在利用推理阶段差异。",
        "- `marker` 和 `marker_position` 不是主结论，只用于判断表面 marker 分布和位置能解释多少。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, x, meta = load_all(Path(args.root), args.feature_key, args.feature_kind)
    if x.numel() == 0:
        raise RuntimeError("No features loaded.")
    y = labels_from_rows(rows)
    train_idx, test_idx = split_by_example(rows, args.test_frac, args.seed)
    train_idx = subsample_indices(train_idx, int(args.max_train), int(args.seed) + 1)
    test_idx = subsample_indices(test_idx, int(args.max_test), int(args.seed) + 2)
    y_train = y[train_idx]
    y_test = y[test_idx]
    result_rows: List[Dict[str, Any]] = []
    experiments = [("h_pre", x), ("position", numeric_baseline_features(rows, "position")), ("marker", numeric_baseline_features(rows, "marker")), ("marker_position", numeric_baseline_features(rows, "marker_position"))]
    for offset, (name, feat) in enumerate(experiments):
        scores = train_linear_probe(feat[train_idx], y_train, feat[test_idx], args, seed_offset=offset * 100)
        row = {"feature": name, **metrics(y_test, scores)}
        result_rows.append(row)
    summary = {
        "root": str(args.root),
        "events": len(rows),
        "feature_key": args.feature_key,
        "feature_kind": args.feature_kind,
        "scope_counts": dict(Counter(str(r.get("scope")) for r in rows)),
        "train_events": int(train_idx.numel()),
        "test_events": int(test_idx.numel()),
        "train_scope_counts": dict(Counter("tampered_continuation" if float(v) == 1.0 else "natural_baseline" for v in y_train.tolist())),
        "test_scope_counts": dict(Counter("tampered_continuation" if float(v) == 1.0 else "natural_baseline" for v in y_test.tolist())),
        **meta,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "classifier_summary.csv", result_rows)
    write_report(output_dir / "REPORT.md", summary, result_rows)
    print(f"[Done] rows={len(result_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
