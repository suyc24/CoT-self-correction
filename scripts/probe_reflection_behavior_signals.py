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


MARKERS = ["wait", "but", "check", "hold_on", "however", "wrong", "actually"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Controlled probes for reflection behavior signals in pre-marker hidden states."
    )
    p.add_argument("--root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--feature_key", default="L22/post_attn")
    p.add_argument("--feature_kind", default="h_pre")
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--max_train", type=int, default=160000)
    p.add_argument("--max_test", type=int, default=80000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--position_bins", type=int, default=20)
    p.add_argument("--log_step_bins", type=int, default=20)
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
    if label == "correct":
        return float(as_bool(row, "final_answer_correct"))
    if label == "productive_repair":
        return float(as_bool(row, "future_repair_language") and as_bool(row, "final_answer_correct"))
    if label == "failed_reflection":
        return float((not as_bool(row, "future_repair_language")) and (not as_bool(row, "final_answer_correct")))
    raise ValueError(label)


def control_features(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    marker_to_idx = {name: i for i, name in enumerate(MARKERS)}
    feats = []
    for row in rows:
        rel = float(row.get("relative_position") or 0.0)
        step = math.log1p(float(row.get("event_step") or 0.0))
        vec = [rel, rel * rel, step, step * step]
        marker = str(row.get("marker_kind"))
        onehot = [0.0] * len(MARKERS)
        if marker in marker_to_idx:
            onehot[marker_to_idx[marker]] = 1.0
        feats.append(vec + onehot)
    return torch.tensor(feats, dtype=torch.float32)


def split_by_example(rows: Sequence[Mapping[str, Any]], test_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    examples = sorted(set(str(r.get("example_id")) for r in rows))
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_test = max(1, int(round(len(examples) * test_frac)))
    test_examples = set(examples[:n_test])
    train, test = [], []
    for i, row in enumerate(rows):
        (test if str(row.get("example_id")) in test_examples else train).append(i)
    return torch.tensor(train, dtype=torch.long), torch.tensor(test, dtype=torch.long)


def subsample(idx: torch.Tensor, max_n: int, seed: int) -> torch.Tensor:
    if max_n <= 0 or idx.numel() <= max_n:
        return idx
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    chosen = torch.randperm(idx.numel(), generator=gen)[:max_n]
    return idx[chosen].sort().values


def standardize_train_test(x_train: torch.Tensor, x_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x_train - mean) / std, (x_test - mean) / std


def residualize_hidden(
    h_train: torch.Tensor,
    h_test: torch.Tensor,
    c_train: torch.Tensor,
    c_test: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    c_train, c_test = standardize_train_test(c_train, c_test)
    ones_train = torch.ones((c_train.shape[0], 1), dtype=c_train.dtype)
    ones_test = torch.ones((c_test.shape[0], 1), dtype=c_test.dtype)
    c_train = torch.cat([ones_train, c_train], dim=1)
    c_test = torch.cat([ones_test, c_test], dim=1)
    reg = float(ridge) * torch.eye(c_train.shape[1], dtype=c_train.dtype)
    reg[0, 0] = 0.0
    w = torch.linalg.solve(c_train.T @ c_train + reg, c_train.T @ h_train)
    return h_train - c_train @ w, h_test - c_test @ w


def train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    args: argparse.Namespace,
    seed_offset: int,
) -> torch.Tensor:
    x_train, x_test = standardize_train_test(x_train.float(), x_test.float())
    torch.manual_seed(int(args.seed) + seed_offset)
    model = torch.nn.Linear(int(x_train.shape[1]), 1)
    pos = y_train.sum().clamp_min(1.0)
    neg = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(neg / pos))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed) + seed_offset + 991)
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
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "balanced_accuracy": (recall + tnr) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
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
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_scope_label(
    scope: str,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    h: torch.Tensor,
    controls: torch.Tensor,
    args: argparse.Namespace,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    idx = [i for i, row in enumerate(rows) if str(row.get("scope")) == scope]
    scoped_rows = [rows[i] for i in idx]
    if len(scoped_rows) < 1000:
        return []
    scoped_h = h[idx]
    scoped_c = controls[idx]
    y = torch.tensor([label_value(row, label) for row in scoped_rows], dtype=torch.float32)
    if y.sum() < 50 or (y.numel() - y.sum()) < 50:
        return []
    train_idx, test_idx = split_by_example(scoped_rows, float(args.test_frac), int(args.seed) + seed_offset)
    train_idx = subsample(train_idx, int(args.max_train), int(args.seed) + seed_offset + 1)
    test_idx = subsample(test_idx, int(args.max_test), int(args.seed) + seed_offset + 2)
    y_train, y_test = y[train_idx], y[test_idx]
    h_train, h_test = scoped_h[train_idx], scoped_h[test_idx]
    c_train, c_test = scoped_c[train_idx], scoped_c[test_idx]
    h_res_train, h_res_test = residualize_hidden(h_train, h_test, c_train, c_test, float(args.ridge))
    experiments = {
        "controls": (c_train, c_test),
        "h_pre": (h_train, h_test),
        "h_pre_plus_controls": (torch.cat([h_train, c_train], dim=1), torch.cat([h_test, c_test], dim=1)),
        "h_pre_residualized_controls": (h_res_train, h_res_test),
    }
    out = []
    for i, (feature, (x_train, x_test)) in enumerate(experiments.items()):
        scores = train_probe(x_train, y_train, x_test, args, seed_offset + i * 100)
        out.append(
            {
                "task": "within_scope_behavior",
                "scope_train": scope,
                "scope_test": scope,
                "label": label,
                "feature": feature,
                "train_n": int(train_idx.numel()),
                "test_n": int(test_idx.numel()),
                "train_positive_rate": float(y_train.mean().item()),
                "test_positive_rate": float(y_test.mean().item()),
                **metrics(y_test, scores),
            }
        )
    return out


def run_cross_scope_label(
    train_scope: str,
    test_scope: str,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    h: torch.Tensor,
    controls: torch.Tensor,
    args: argparse.Namespace,
    seed_offset: int,
) -> List[Dict[str, Any]]:
    train_idx_all = [i for i, row in enumerate(rows) if str(row.get("scope")) == train_scope]
    test_idx_all = [i for i, row in enumerate(rows) if str(row.get("scope")) == test_scope]
    if len(train_idx_all) < 1000 or len(test_idx_all) < 1000:
        return []
    train_idx = subsample(torch.tensor(train_idx_all, dtype=torch.long), int(args.max_train), int(args.seed) + seed_offset)
    test_idx = subsample(torch.tensor(test_idx_all, dtype=torch.long), int(args.max_test), int(args.seed) + seed_offset + 1)
    y = torch.tensor([label_value(row, label) for row in rows], dtype=torch.float32)
    y_train, y_test = y[train_idx], y[test_idx]
    if y_train.sum() < 50 or (y_train.numel() - y_train.sum()) < 50 or y_test.sum() < 50 or (y_test.numel() - y_test.sum()) < 50:
        return []
    h_train, h_test = h[train_idx], h[test_idx]
    c_train, c_test = controls[train_idx], controls[test_idx]
    h_res_train, h_res_test = residualize_hidden(h_train, h_test, c_train, c_test, float(args.ridge))
    experiments = {
        "controls": (c_train, c_test),
        "h_pre": (h_train, h_test),
        "h_pre_residualized_controls": (h_res_train, h_res_test),
    }
    out = []
    for i, (feature, (x_train, x_test)) in enumerate(experiments.items()):
        scores = train_probe(x_train, y_train, x_test, args, seed_offset + i * 100)
        out.append(
            {
                "task": "cross_scope_behavior",
                "scope_train": train_scope,
                "scope_test": test_scope,
                "label": label,
                "feature": feature,
                "train_n": int(train_idx.numel()),
                "test_n": int(test_idx.numel()),
                "train_positive_rate": float(y_train.mean().item()),
                "test_positive_rate": float(y_test.mean().item()),
                **metrics(y_test, scores),
            }
        )
    return out


def bin_key(row: Mapping[str, Any], args: argparse.Namespace) -> Tuple[str, int, int]:
    marker = str(row.get("marker_kind"))
    rel = max(0.0, min(0.999999, float(row.get("relative_position") or 0.0)))
    rel_bin = int(rel * int(args.position_bins))
    step = math.log1p(float(row.get("event_step") or 0.0))
    step_norm = max(0.0, min(0.999999, step / math.log1p(16384.0)))
    step_bin = int(step_norm * int(args.log_step_bins))
    return marker, rel_bin, step_bin


def matched_condition_indices(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> torch.Tensor:
    groups: Dict[Tuple[str, int, int, str], List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        scope = str(row.get("scope"))
        if scope not in {"natural_baseline", "tampered_continuation"}:
            continue
        groups[(*bin_key(row, args), scope)].append(i)
    rng = random.Random(int(args.seed) + 404)
    matched = []
    base_keys = set(key[:3] for key in groups)
    for key in sorted(base_keys):
        a = groups.get((*key, "natural_baseline"), [])
        b = groups.get((*key, "tampered_continuation"), [])
        m = min(len(a), len(b))
        if m < 5:
            continue
        rng.shuffle(a)
        rng.shuffle(b)
        matched.extend(a[:m])
        matched.extend(b[:m])
    return torch.tensor(sorted(matched), dtype=torch.long)


def run_matched_condition(rows: Sequence[Mapping[str, Any]], h: torch.Tensor, controls: torch.Tensor, args: argparse.Namespace) -> List[Dict[str, Any]]:
    matched_idx = matched_condition_indices(rows, args)
    if matched_idx.numel() < 1000:
        return []
    matched_rows = [rows[int(i)] for i in matched_idx.tolist()]
    y = torch.tensor([1.0 if str(row.get("scope")) == "tampered_continuation" else 0.0 for row in matched_rows], dtype=torch.float32)
    train_local, test_local = split_by_example(matched_rows, float(args.test_frac), int(args.seed) + 808)
    train_local = subsample(train_local, int(args.max_train), int(args.seed) + 809)
    test_local = subsample(test_local, int(args.max_test), int(args.seed) + 810)
    global_train = matched_idx[train_local]
    global_test = matched_idx[test_local]
    y_train, y_test = y[train_local], y[test_local]
    h_train, h_test = h[global_train], h[global_test]
    c_train, c_test = controls[global_train], controls[global_test]
    h_res_train, h_res_test = residualize_hidden(h_train, h_test, c_train, c_test, float(args.ridge))
    experiments = {
        "controls": (c_train, c_test),
        "h_pre": (h_train, h_test),
        "h_pre_residualized_controls": (h_res_train, h_res_test),
    }
    out = []
    for i, (feature, (x_train, x_test)) in enumerate(experiments.items()):
        scores = train_probe(x_train, y_train, x_test, args, 9000 + i * 100)
        out.append(
            {
                "task": "matched_natural_vs_tampered",
                "scope_train": "matched",
                "scope_test": "matched",
                "label": "tampered_condition",
                "feature": feature,
                "train_n": int(train_local.numel()),
                "test_n": int(test_local.numel()),
                "train_positive_rate": float(y_train.mean().item()),
                "test_positive_rate": float(y_test.mean().item()),
                **metrics(y_test, scores),
            }
        )
    return out


def write_report(path: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# 反思行为 Hidden-State 控制变量 Probe 报告",
        "",
        "## 核心问题",
        "从无监督聚类转向可检验命题：反思词生成前的 hidden state 是否预测接下来的真实反思行为，并且这种预测是否超过 marker/position 控制变量。",
        "",
        "## 数据",
        f"- 事件总数：`{summary['events']}`",
        f"- scope 计数：`{summary['scope_counts']}`",
        f"- 表示：`{summary['feature_key']}/{summary['feature_kind']}`",
        "",
        "## 结果表",
        "| task | train | test | label | feature | AUC | bal_acc | F1 | pos_rate_test | n_test |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (str(r.get("task")), str(r.get("scope_train")), str(r.get("label")), str(r.get("feature")))):
        lines.append(
            "| {task} | {scope_train} | {scope_test} | {label} | {feature} | {auc:.4f} | {balanced_accuracy:.4f} | {f1:.4f} | {pos:.3f} | {n} |".format(
                task=row.get("task"),
                scope_train=row.get("scope_train"),
                scope_test=row.get("scope_test"),
                label=row.get("label"),
                feature=row.get("feature"),
                auc=float(row.get("auc") or float("nan")),
                balanced_accuracy=float(row.get("balanced_accuracy") or float("nan")),
                f1=float(row.get("f1") or float("nan")),
                pos=float(row.get("test_positive_rate") or 0.0),
                n=row.get("test_n"),
            )
        )
    lines += [
        "",
        "## 读法",
        "- `controls` 只含 marker、相对位置和 log step。",
        "- `h_pre_residualized_controls` 先从 hidden state 中线性回归掉 controls，再训练 probe。",
        "- 如果 residualized hidden 仍显著强于 controls，说明结果不只是 marker/position 泄漏。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, h, meta = load_all(Path(args.root), args.feature_key, args.feature_kind)
    if h.numel() == 0:
        raise RuntimeError("No hidden features loaded.")
    controls = control_features(rows)
    result_rows: List[Dict[str, Any]] = []
    labels = ["repair", "error_ack", "correct", "productive_repair", "failed_reflection"]
    scopes = ["natural_baseline", "tampered_continuation"]
    seed_offset = 0
    for scope in scopes:
        for label in labels:
            result_rows.extend(run_scope_label(scope, label, rows, h, controls, args, seed_offset))
            seed_offset += 1000
    for label in ["repair", "error_ack", "productive_repair", "failed_reflection"]:
        result_rows.extend(run_cross_scope_label("natural_baseline", "tampered_continuation", label, rows, h, controls, args, seed_offset))
        seed_offset += 1000
        result_rows.extend(run_cross_scope_label("tampered_continuation", "natural_baseline", label, rows, h, controls, args, seed_offset))
        seed_offset += 1000
    result_rows.extend(run_matched_condition(rows, h, controls, args))
    summary = {
        "root": str(args.root),
        "events": len(rows),
        "feature_key": args.feature_key,
        "feature_kind": args.feature_kind,
        "scope_counts": dict(Counter(str(r.get("scope")) for r in rows)),
        "marker_counts": dict(Counter(str(r.get("marker_kind")) for r in rows)),
        **meta,
    }
    write_csv(output_dir / "probe_summary.csv", result_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "REPORT.md", result_rows, summary)
    print(f"[Done] probe_rows={len(result_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
