#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


MARKERS = ["wait", "but", "check", "hold_on", "however", "wrong", "actually", "cn_wrong", "cn_recheck"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strict audit for whether pre-reflection hidden states predict behavior beyond marker/position leakage."
    )
    p.add_argument("--root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--feature_key", default="L22/post_attn")
    p.add_argument("--feature_kind", default="h_pre")
    p.add_argument("--scope", default="natural_baseline")
    p.add_argument("--labels", default="error_ack,repair,productive_repair,failed_reflection")
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--position_bins", type=int, default=32)
    p.add_argument("--step_bins", type=int, default=32)
    p.add_argument("--max_train", type=int, default=120000)
    p.add_argument("--max_test", type=int, default=60000)
    p.add_argument("--max_hidden_events", type=int, default=180000)
    p.add_argument("--event_caps", default="1,3,all")
    p.add_argument("--matched_max_per_cell", type=int, default=400)
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def shard_dirs(root: Path) -> List[Path]:
    if (root / "event_rows.jsonl").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "event_rows.jsonl").exists())


def load_all(root: Path, feature_key: str, feature_kind: str, scope: str) -> Tuple[List[Dict[str, Any]], torch.Tensor, Dict[str, Any]]:
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
        if tensor is None or tensor.ndim != 2:
            meta["bad_shards"].append(str(shard))
            continue
        n = min(len(rows), int(tensor.shape[0]))
        keep = [i for i in range(n) if str(rows[i].get("scope")) == scope]
        if keep:
            for i in keep:
                row = dict(rows[i])
                row["global_event_index"] = offset + int(i)
                row["shard"] = shard.name
                rows_all.append(row)
            xs.append(tensor[torch.tensor(keep, dtype=torch.long)].float())
        offset += n
        meta["shards"].append({"path": str(shard), "events": len(keep)})
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def bin_index(value: float, bins: int) -> int:
    if bins <= 1:
        return 0
    return max(0, min(int(bins) - 1, int(math.floor(max(0.0, min(0.999999, value)) * int(bins)))))


def row_cell(row: Mapping[str, Any], position_bins: int, step_bins: int, *, include_subject: bool) -> Tuple[Any, ...]:
    rel = safe_float(row.get("relative_position"))
    step = math.log1p(max(0.0, safe_float(row.get("event_step")))) / math.log1p(16384.0)
    base: List[Any] = [
        str(row.get("marker_kind")),
        str(row.get("marker_token_id")),
        bin_index(rel, position_bins),
        bin_index(step, step_bins),
    ]
    if include_subject:
        base.append(str(row.get("subject") or ""))
    return tuple(base)


def top_values(rows: Sequence[Mapping[str, Any]], key: str, limit: int) -> List[str]:
    counts = Counter(str(r.get(key) or "") for r in rows)
    return [k for k, _ in counts.most_common(limit) if k]


def one_hot(value: str, vocab: Mapping[str, int]) -> List[float]:
    vec = [0.0] * len(vocab)
    if value in vocab:
        vec[vocab[value]] = 1.0
    return vec


def control_features(rows: Sequence[Mapping[str, Any]], position_bins: int, step_bins: int) -> Tuple[torch.Tensor, List[str]]:
    token_vocab = {v: i for i, v in enumerate(top_values(rows, "marker_token_id", 48))}
    text_vocab = {v: i for i, v in enumerate(top_values(rows, "marker_token_text", 48))}
    subject_vocab = {v: i for i, v in enumerate(top_values(rows, "subject", 16))}
    marker_vocab = {m: i for i, m in enumerate(MARKERS)}
    feature_names = (
        ["relative_position", "relative_position_sq", "log_step", "log_step_sq"]
        + [f"position_bin_{i}" for i in range(position_bins)]
        + [f"step_bin_{i}" for i in range(step_bins)]
        + [f"marker_{m}" for m in marker_vocab]
        + [f"token_id_{v}" for v in token_vocab]
        + [f"token_text_{v}" for v in text_vocab]
        + [f"subject_{v}" for v in subject_vocab]
    )
    feats: List[List[float]] = []
    for row in rows:
        rel = max(0.0, min(1.0, safe_float(row.get("relative_position"))))
        step = math.log1p(max(0.0, safe_float(row.get("event_step")))) / math.log1p(16384.0)
        pbin = bin_index(rel, position_bins)
        sbin = bin_index(step, step_bins)
        pvec = [0.0] * position_bins
        svec = [0.0] * step_bins
        pvec[pbin] = 1.0
        svec[sbin] = 1.0
        feats.append(
            [rel, rel * rel, step, step * step]
            + pvec
            + svec
            + one_hot(str(row.get("marker_kind")), marker_vocab)
            + one_hot(str(row.get("marker_token_id") or ""), token_vocab)
            + one_hot(str(row.get("marker_token_text") or ""), text_vocab)
            + one_hot(str(row.get("subject") or ""), subject_vocab)
        )
    return torch.tensor(feats, dtype=torch.float32), feature_names


def select_event_cap(rows: Sequence[Mapping[str, Any]], cap: str, seed: int) -> torch.Tensor:
    if cap == "all":
        return torch.arange(len(rows), dtype=torch.long)
    n_cap = int(cap)
    by_ex: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_ex[str(row.get("example_id"))].append(i)
    rng = random.Random(seed)
    keep: List[int] = []
    for idxs in by_ex.values():
        idxs = sorted(idxs, key=lambda i: safe_float(rows[i].get("event_step")))
        if len(idxs) > n_cap:
            # Keep the earliest event and sample the rest to avoid making the task purely "first marker".
            head = idxs[:1]
            tail = idxs[1:]
            rng.shuffle(tail)
            idxs = head + tail[: max(0, n_cap - 1)]
        keep.extend(idxs)
    return torch.tensor(sorted(keep), dtype=torch.long)


def matched_indices(
    rows: Sequence[Mapping[str, Any]],
    y: torch.Tensor,
    *,
    position_bins: int,
    step_bins: int,
    max_per_cell: int,
    seed: int,
) -> torch.Tensor:
    rng = random.Random(seed)
    cells: Dict[Tuple[Any, ...], Dict[int, List[int]]] = defaultdict(lambda: {0: [], 1: []})
    for i, row in enumerate(rows):
        cells[row_cell(row, position_bins, step_bins, include_subject=True)][int(y[i].item())].append(i)
    keep: List[int] = []
    for group in cells.values():
        pos = group[1]
        neg = group[0]
        n = min(len(pos), len(neg), int(max_per_cell))
        if n <= 0:
            continue
        rng.shuffle(pos)
        rng.shuffle(neg)
        keep.extend(pos[:n])
        keep.extend(neg[:n])
    return torch.tensor(sorted(keep), dtype=torch.long)


def split_by_example(rows: Sequence[Mapping[str, Any]], indices: torch.Tensor, test_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    examples = sorted({str(rows[int(i)].get("example_id")) for i in indices.tolist()})
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_test = max(1, int(round(len(examples) * float(test_frac))))
    test_examples = set(examples[:n_test])
    train: List[int] = []
    test: List[int] = []
    for i in indices.tolist():
        (test if str(rows[int(i)].get("example_id")) in test_examples else train).append(int(i))
    return torch.tensor(train, dtype=torch.long), torch.tensor(test, dtype=torch.long)


def cap_indices(idx: torch.Tensor, max_n: int, seed: int) -> torch.Tensor:
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
    h_train = torch.nan_to_num(h_train.float(), nan=0.0, posinf=0.0, neginf=0.0)
    h_test = torch.nan_to_num(h_test.float(), nan=0.0, posinf=0.0, neginf=0.0)
    c_train = torch.nan_to_num(c_train.float(), nan=0.0, posinf=0.0, neginf=0.0)
    c_test = torch.nan_to_num(c_test.float(), nan=0.0, posinf=0.0, neginf=0.0)
    c_train, c_test = standardize_train_test(c_train, c_test)
    c_train = torch.cat([torch.ones((c_train.shape[0], 1)), c_train], dim=1)
    c_test = torch.cat([torch.ones((c_test.shape[0], 1)), c_test], dim=1)
    reg = float(ridge) * torch.eye(c_train.shape[1])
    reg[0, 0] = 0.0
    lhs = torch.nan_to_num(c_train.T @ c_train + reg, nan=0.0, posinf=0.0, neginf=0.0)
    rhs = torch.nan_to_num(c_train.T @ h_train, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        w = torch.linalg.solve(lhs, rhs)
    except torch._C._LinAlgError:
        w = torch.linalg.pinv(lhs) @ rhs
    return h_train - c_train @ w, h_test - c_test @ w


def train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> torch.Tensor:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, x_test = standardize_train_test(x_train.float(), x_test.float())
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    torch.manual_seed(seed)
    model = torch.nn.Linear(int(x_train.shape[1]), 1).to(device)
    pos = y_train.sum().clamp_min(1.0)
    neg = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(neg / pos))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    n = int(x_train.shape[0])
    for _ in range(int(epochs)):
        order = torch.randperm(n, device=device)
        for start in range(0, n, int(batch_size)):
            batch = order[start : start + int(batch_size)]
            logits = model(x_train[batch]).squeeze(1)
            loss = loss_fn(logits, y_train[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(x_test).squeeze(1)).detach().cpu()


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


def metric_row(y_true: torch.Tensor, y_score: torch.Tensor) -> Dict[str, Any]:
    pred = y_score >= 0.5
    yb = y_true.bool()
    tp = int((pred & yb).sum().item())
    tn = int((~pred & ~yb).sum().item())
    fp = int((pred & ~yb).sum().item())
    fn = int((~pred & yb).sum().item())
    acc = (tp + tn) / max(1, len(yb))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = tpr
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "auc": auc_score(y_true, y_score),
        "accuracy": acc,
        "balanced_accuracy": 0.5 * (tpr + tnr),
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
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    best = sorted(
        [r for r in rows if str(r.get("feature")) == "h_pre_residualized_controls"],
        key=lambda r: float(r.get("auc") or 0.0),
        reverse=True,
    )[:12]
    control_pairs = []
    by_key: Dict[Tuple[str, str, str], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        key = (str(r.get("label")), str(r.get("sample_mode")), str(r.get("matching")))
        by_key[key][str(r.get("feature"))] = r
    for key, group in by_key.items():
        if "controls" in group and "h_pre_residualized_controls" in group:
            control_pairs.append(
                {
                    "label": key[0],
                    "sample": key[1],
                    "matching": key[2],
                    "control_auc": group["controls"].get("auc"),
                    "residual_auc": group["h_pre_residualized_controls"].get("auc"),
                    "delta": float(group["h_pre_residualized_controls"].get("auc") or 0.0)
                    - float(group["controls"].get("auc") or 0.0),
                }
            )
    control_pairs = sorted(control_pairs, key=lambda r: float(r["delta"]), reverse=True)[:12]

    def md_table(items: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
        if not items:
            return "_无_"
        lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
        for item in items:
            vals = []
            for key in keys:
                value = item.get(key, "")
                if isinstance(value, float):
                    vals.append(f"{value:.4f}")
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    text = [
        "# 反思前 hidden-state 严格审计",
        "",
        "## 结论口径",
        "",
        "这轮不是继续追求聚类轮廓，而是检查反思词生成前的 hidden state 是否仍能在同题分组、marker/位置/token 控制、同控制桶匹配以后预测后续反思行为。",
        "",
        "## 数据规模",
        "",
        f"- 事件数：{summary.get('events')}",
        f"- 题目数：{summary.get('examples')}",
        f"- scope：`{summary.get('scope')}`",
        f"- hidden：`{summary.get('feature_key')}` / `{summary.get('feature_kind')}`",
        "",
        "## 最强残差信号",
        "",
        md_table(best, ["label", "sample_mode", "matching", "train_n", "test_n", "test_positive_rate", "auc", "balanced_accuracy", "f1", "permutation_p"]),
        "",
        "## 相对控制基线的增量",
        "",
        md_table(control_pairs, ["label", "sample", "matching", "control_auc", "residual_auc", "delta"]),
        "",
        "## 100% 信心自检",
        "",
        "- 若残差 AUC 在每题限量和同 marker/位置/token 桶匹配后仍明显高于控制基线，说明结果不只是位置或 marker 泄漏。",
        "- 若只在全事件上强、每题限量后弱，说明之前样本相关性放大了结论。",
        "- 若同桶匹配后下降到接近 chance，下一步必须转向更局部的干预或更精细行为标注，不能把它包装成清晰类型学。",
        "- 即使审计通过，也还需要因果 patch/steering 才能达到强论文证据。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    rows, h, meta = load_all(Path(args.root), args.feature_key, args.feature_kind, args.scope)
    if h.numel() == 0:
        raise RuntimeError("No hidden features loaded.")
    controls, control_names = control_features(rows, int(args.position_bins), int(args.step_bins))
    labels = [x.strip() for x in str(args.labels).split(",") if x.strip()]
    event_caps = [x.strip() for x in str(args.event_caps).split(",") if x.strip()]
    out_rows: List[Dict[str, Any]] = []
    perm_rows: List[Dict[str, Any]] = []

    for label_idx, label in enumerate(labels):
        y_all = torch.tensor([label_value(row, label) for row in rows], dtype=torch.float32)
        if y_all.sum().item() < 10 or (1 - y_all).sum().item() < 10:
            continue
        for cap in event_caps:
            base_idx = select_event_cap(rows, cap, int(args.seed) + label_idx)
            matching_specs = [("unmatched", base_idx)]
            matched = matched_indices(
                [rows[int(i)] for i in base_idx.tolist()],
                y_all[base_idx],
                position_bins=int(args.position_bins),
                step_bins=int(args.step_bins),
                max_per_cell=int(args.matched_max_per_cell),
                seed=int(args.seed) + 101 + label_idx,
            )
            if matched.numel() >= 100:
                matching_specs.append(("matched_marker_token_position_subject", base_idx[matched]))
            for matching_name, idx in matching_specs:
                if idx.numel() < 100:
                    continue
                train_idx, test_idx = split_by_example(rows, idx, float(args.test_frac), int(args.seed) + 203 + label_idx)
                train_idx = cap_indices(train_idx, int(args.max_train), int(args.seed) + 307 + label_idx)
                test_idx = cap_indices(test_idx, int(args.max_test), int(args.seed) + 409 + label_idx)
                y_train = y_all[train_idx]
                y_test = y_all[test_idx]
                if y_train.sum().item() < 5 or y_test.sum().item() < 5 or (1 - y_train).sum().item() < 5 or (1 - y_test).sum().item() < 5:
                    continue
                h_train = h[train_idx]
                h_test = h[test_idx]
                c_train = controls[train_idx]
                c_test = controls[test_idx]
                h_res_train, h_res_test = residualize_hidden(h_train, h_test, c_train, c_test, float(args.ridge))
                feature_sets = [
                    ("controls", c_train, c_test),
                    ("h_pre", h_train, h_test),
                    ("h_pre_plus_controls", torch.cat([h_train, c_train], dim=1), torch.cat([h_test, c_test], dim=1)),
                    ("h_pre_residualized_controls", h_res_train, h_res_test),
                ]
                for feature_offset, (feature_name, x_train, x_test) in enumerate(feature_sets):
                    if feature_name.startswith("h_pre") and int(args.max_hidden_events) > 0 and x_train.shape[0] > int(args.max_hidden_events):
                        reduced = cap_indices(torch.arange(x_train.shape[0]), int(args.max_hidden_events), int(args.seed) + feature_offset)
                        x_train_use = x_train[reduced]
                        y_train_use = y_train[reduced]
                    else:
                        x_train_use = x_train
                        y_train_use = y_train
                    score = train_probe(
                        x_train_use,
                        y_train_use,
                        x_test,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        seed=int(args.seed) + 1000 * label_idx + 17 * feature_offset,
                        device=str(args.device),
                    )
                    row = {
                        "label": label,
                        "sample_mode": f"cap_{cap}",
                        "matching": matching_name,
                        "feature": feature_name,
                        "train_n": int(y_train_use.numel()),
                        "test_n": int(y_test.numel()),
                        "train_positive_rate": float(y_train_use.mean().item()),
                        "test_positive_rate": float(y_test.mean().item()),
                    }
                    row.update(metric_row(y_test, score))
                    if feature_name == "h_pre_residualized_controls" and int(args.permutations) > 0:
                        auc = float(row["auc"])
                        ge = 0
                        valid = 0
                        for pidx in range(int(args.permutations)):
                            gen = torch.Generator(device="cpu")
                            gen.manual_seed(int(args.seed) + 90000 + 1000 * label_idx + pidx)
                            perm_y = y_train_use[torch.randperm(y_train_use.numel(), generator=gen)]
                            perm_score = train_probe(
                                x_train_use,
                                perm_y,
                                x_test,
                                epochs=max(3, int(args.epochs) // 2),
                                batch_size=int(args.batch_size),
                                lr=float(args.lr),
                                weight_decay=float(args.weight_decay),
                                seed=int(args.seed) + 80000 + pidx,
                                device=str(args.device),
                            )
                            perm_auc = auc_score(y_test, perm_score)
                            if math.isfinite(perm_auc):
                                valid += 1
                                ge += int(perm_auc >= auc)
                                perm_rows.append({
                                    "label": label,
                                    "sample_mode": f"cap_{cap}",
                                    "matching": matching_name,
                                    "permutation": pidx,
                                    "auc": perm_auc,
                                })
                        row["permutation_p"] = (ge + 1) / (valid + 1) if valid else ""
                    out_rows.append(row)
                    write_csv(output_dir / "audit_summary.partial.csv", out_rows)
                    write_csv(output_dir / "permutation_summary.partial.csv", perm_rows)

    summary = {
        "root": str(args.root),
        "scope": args.scope,
        "events": len(rows),
        "examples": len({str(r.get("example_id")) for r in rows}),
        "feature_key": args.feature_key,
        "feature_kind": args.feature_kind,
        "labels": labels,
        "position_bins": int(args.position_bins),
        "step_bins": int(args.step_bins),
        "control_feature_count": len(control_names),
        **meta,
    }
    write_csv(output_dir / "audit_summary.csv", out_rows)
    write_csv(output_dir / "permutation_summary.csv", perm_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, out_rows, summary)
    print(f"[Done] wrote {output_dir}")


if __name__ == "__main__":
    main()
