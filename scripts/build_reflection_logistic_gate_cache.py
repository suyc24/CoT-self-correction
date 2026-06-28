#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_reflection_hidden_signal_strength import (  # noqa: E402
    auc_score,
    control_features,
    label_value,
    load_all,
    matched_indices,
    metric_row,
    residualize_hidden,
    split_by_example,
    standardize_train_test,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a gate-compatible steering cache from a controlled logistic probe over natural reflection hidden states."
    )
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feature_key", default="L22/post_attn")
    p.add_argument("--feature_kind", default="h_pre")
    p.add_argument("--scope", default="natural_baseline")
    p.add_argument("--label", default="error_ack")
    p.add_argument("--layer_idx", type=int, default=22)
    p.add_argument("--site", default="post_attn")
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--position_bins", type=int, default=32)
    p.add_argument("--step_bins", type=int, default=32)
    p.add_argument("--matched_max_per_cell", type=int, default=800)
    p.add_argument("--max_train", type=int, default=160000)
    p.add_argument("--max_test", type=int, default=60000)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--scale_quantile", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def cap_indices(idx: torch.Tensor, max_n: int, seed: int) -> torch.Tensor:
    if max_n <= 0 or int(idx.numel()) <= int(max_n):
        return idx
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    chosen = torch.randperm(idx.numel(), generator=gen)[: int(max_n)]
    return idx[chosen].sort().values


def bool_label(row: Mapping[str, Any], label: str) -> bool:
    return bool(label_value(row, label))


def unit(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.float().norm().clamp_min(1e-12)


def finite_float(value: Any, default: float = 1.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def train_logistic_direction(
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
) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train_std, x_test_std = standardize_train_test(x_train.float(), x_test.float())
    mean = x_train.float().mean(dim=0)
    std = x_train.float().std(dim=0).clamp_min(1e-6)
    x_train_std = x_train_std.to(device)
    x_test_std = x_test_std.to(device)
    y_train = y_train.float().to(device)
    torch.manual_seed(int(seed))
    model = torch.nn.Linear(int(x_train_std.shape[1]), 1).to(device)
    pos = y_train.sum().clamp_min(1.0)
    neg = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(neg / pos))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    n = int(x_train_std.shape[0])
    for _ in range(int(epochs)):
        order = torch.randperm(n, device=device)
        for start in range(0, n, int(batch_size)):
            batch = order[start : start + int(batch_size)]
            logits = model(x_train_std[batch]).squeeze(1)
            loss = loss_fn(logits, y_train[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        scores = torch.sigmoid(model(x_test_std).squeeze(1)).detach().cpu()
        w_std = model.weight.detach().cpu().squeeze(0).float()
        b_std = float(model.bias.detach().cpu().item())
    raw_direction = unit(w_std / std)
    meta = {
        "standardized_bias": b_std,
        "weight_std_norm": float(w_std.norm().item()),
        "raw_direction_norm": float(raw_direction.norm().item()),
        "train_feature_mean_norm": float(mean.norm().item()),
        "train_feature_std_mean": float(std.mean().item()),
    }
    return raw_direction, meta, scores


def projection_scale(x: torch.Tensor, direction: torch.Tensor, q: float) -> float:
    proj = x.float() @ direction.float()
    centered = (proj - proj.median()).abs()
    scale = float(torch.quantile(centered, float(q)).item())
    if not math.isfinite(scale) or scale <= 1e-6:
        scale = float(proj.std().item())
    if not math.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return scale


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    rows, h, meta = load_all(Path(args.root), args.feature_key, args.feature_kind, args.scope)
    if h.numel() == 0:
        raise RuntimeError("No hidden features loaded.")
    controls, control_names = control_features(rows, int(args.position_bins), int(args.step_bins))
    y = torch.tensor([bool_label(row, str(args.label)) for row in rows], dtype=torch.float32)
    idx = matched_indices(
        rows,
        y,
        position_bins=int(args.position_bins),
        step_bins=int(args.step_bins),
        max_per_cell=int(args.matched_max_per_cell),
        seed=int(args.seed),
    )
    if idx.numel() < 100:
        raise RuntimeError(f"Too few matched examples: {idx.numel()}")
    train_idx, test_idx = split_by_example(rows, idx, float(args.test_frac), int(args.seed) + 17)
    train_idx = cap_indices(train_idx, int(args.max_train), int(args.seed) + 29)
    test_idx = cap_indices(test_idx, int(args.max_test), int(args.seed) + 31)
    h_res_train, h_res_test = residualize_hidden(
        h[train_idx],
        h[test_idx],
        controls[train_idx],
        controls[test_idx],
        float(args.ridge),
    )
    y_train = y[train_idx]
    y_test = y[test_idx]
    direction, direction_meta, test_scores = train_logistic_direction(
        h_res_train,
        y_train,
        h_res_test,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed) + 43,
        device=str(args.device),
    )
    metrics = metric_row(y_test, test_scores)
    scale = projection_scale(h_res_train, direction, float(args.scale_quantile))
    pos_count = int(y.sum().item())
    neg_count = int(y.numel() - y.sum().item())
    item = {
        "direction_type": "gate",
        "source_direction_type": f"logistic_{args.label}",
        "wrapped_behavior_direction": str(args.label),
        "layer_idx": int(args.layer_idx),
        "site": str(args.site),
        "label": str(args.label),
        "n_pairs": int(min(pos_count, neg_count)),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "matched_count": int(idx.numel()),
        "train_n": int(train_idx.numel()),
        "test_n": int(test_idx.numel()),
        "train_positive_rate": float(y_train.mean().item()),
        "test_positive_rate": float(y_test.mean().item()),
        "scale": scale,
        "feature_key": str(args.feature_key),
        "feature_kind": str(args.feature_kind),
        "scope": str(args.scope),
        "ridge": float(args.ridge),
        "position_bins": int(args.position_bins),
        "step_bins": int(args.step_bins),
        "matched_max_per_cell": int(args.matched_max_per_cell),
        "control_feature_count": len(control_names),
        "probe_auc": finite_float(metrics.get("auc"), float("nan")),
        "probe_balanced_accuracy": finite_float(metrics.get("balanced_accuracy"), float("nan")),
        "probe_f1": finite_float(metrics.get("f1"), float("nan")),
        "direction": direction.cpu(),
        **direction_meta,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "directions": [item],
        "direction_rows": [{k: v for k, v in item.items() if k != "direction"}],
        "meta": {
            **meta,
            "events": len(rows),
            "test_metrics": metrics,
            "script": Path(__file__).name,
        },
    }
    torch.save(payload, out)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "direction_rows": payload["direction_rows"],
                "meta": payload["meta"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[Done] wrote {out} matched={matched} train={train} test={test} auc={auc:.4f} scale={scale:.4f}".format(
            out=out,
            matched=int(idx.numel()),
            train=int(train_idx.numel()),
            test=int(test_idx.numel()),
            auc=finite_float(metrics.get("auc"), float("nan")),
            scale=scale,
        )
    )


if __name__ == "__main__":
    main()
