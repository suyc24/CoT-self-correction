#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

MARKERS = ["wait", "but", "check", "hold_on", "however", "wrong", "actually"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build steering direction cache from natural reflection behavior labels.")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--feature_key", default="L22/post_attn")
    p.add_argument("--feature_kind", default="h_pre")
    p.add_argument("--scope", default="natural_baseline")
    p.add_argument("--labels", default="error_ack,repair,productive_repair,failed_reflection")
    p.add_argument("--layer_idx", type=int, default=22)
    p.add_argument("--site", default="post_attn")
    p.add_argument("--residualize_controls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--min_positive", type=int, default=500)
    p.add_argument("--min_negative", type=int, default=500)
    p.add_argument("--scale_quantile", type=float, default=0.5)
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


def load_all(root: Path, feature_key: str, feature_kind: str, scope: str) -> Tuple[List[Dict[str, Any]], torch.Tensor, Dict[str, Any]]:
    rows_all: List[Dict[str, Any]] = []
    xs: List[torch.Tensor] = []
    meta: Dict[str, Any] = {"shards": [], "bad_shards": []}
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
            rows_all.extend(dict(rows[i]) for i in keep)
            xs.append(tensor[torch.tensor(keep, dtype=torch.long)].float())
        meta["shards"].append({"path": str(shard), "events": len(keep)})
    if not xs:
        return rows_all, torch.empty((0, 0)), meta
    return rows_all, torch.cat(xs, dim=0), meta


def as_bool(row: Mapping[str, Any], key: str) -> bool:
    return str(row.get(key)).lower() in {"true", "1", "yes"}


def label_value(row: Mapping[str, Any], label: str) -> bool:
    if label == "repair":
        return as_bool(row, "future_repair_language")
    if label == "error_ack":
        return as_bool(row, "future_explicit_error_ack")
    if label == "productive_repair":
        return as_bool(row, "future_repair_language") and as_bool(row, "final_answer_correct")
    if label == "failed_reflection":
        return (not as_bool(row, "future_repair_language")) and (not as_bool(row, "final_answer_correct"))
    raise ValueError(label)


def controls(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    marker_to_idx = {m: i for i, m in enumerate(MARKERS)}
    out = []
    for row in rows:
        rel = float(row.get("relative_position") or 0.0)
        step = math.log1p(float(row.get("event_step") or 0.0))
        onehot = [0.0] * len(MARKERS)
        marker = str(row.get("marker_kind"))
        if marker in marker_to_idx:
            onehot[marker_to_idx[marker]] = 1.0
        out.append([rel, rel * rel, step, step * step] + onehot)
    return torch.tensor(out, dtype=torch.float32)


def residualize(x: torch.Tensor, c: torch.Tensor, ridge: float) -> torch.Tensor:
    c = (c - c.mean(dim=0, keepdim=True)) / c.std(dim=0, keepdim=True).clamp_min(1e-6)
    c = torch.cat([torch.ones((c.shape[0], 1)), c], dim=1)
    reg = float(ridge) * torch.eye(c.shape[1])
    reg[0, 0] = 0.0
    w = torch.linalg.solve(c.T @ c + reg, c.T @ x)
    return x - c @ w


def unit(x: torch.Tensor) -> torch.Tensor:
    return x.float() / x.float().norm().clamp_min(1e-12)


def main() -> None:
    args = parse_args()
    rows, x, meta = load_all(Path(args.root), args.feature_key, args.feature_kind, args.scope)
    if x.numel() == 0:
        raise RuntimeError("No features loaded.")
    x_used = residualize(x, controls(rows), float(args.ridge)) if args.residualize_controls else x
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    directions = []
    direction_rows = []
    for label in labels:
        y = torch.tensor([label_value(row, label) for row in rows], dtype=torch.bool)
        pos_n = int(y.sum().item())
        neg_n = int((~y).sum().item())
        if pos_n < int(args.min_positive) or neg_n < int(args.min_negative):
            continue
        pos_mean = x_used[y].mean(dim=0)
        neg_mean = x_used[~y].mean(dim=0)
        direction = unit(pos_mean - neg_mean)
        proj = x_used @ direction
        scale = float(torch.quantile((proj - proj.median()).abs(), float(args.scale_quantile)).item())
        if not math.isfinite(scale) or scale <= 1e-6:
            scale = float(proj.std().item())
        row = {
            "direction_type": label,
            "layer_idx": int(args.layer_idx),
            "site": str(args.site),
            "label": label,
            "n_pairs": min(pos_n, neg_n),
            "positive_count": pos_n,
            "negative_count": neg_n,
            "scale": scale,
            "residualize_controls": bool(args.residualize_controls),
            "feature_key": args.feature_key,
            "feature_kind": args.feature_kind,
            "scope": args.scope,
            "marker_counts": json.dumps(dict(Counter(str(r.get("marker_kind")) for r in rows)), ensure_ascii=False),
        }
        directions.append({**row, "direction": direction.cpu()})
        direction_rows.append(row)
    payload = {"directions": directions, "direction_rows": direction_rows, "meta": {**meta, "events": len(rows)}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    (out.with_suffix(".json")).write_text(json.dumps({"direction_rows": direction_rows, "meta": payload["meta"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] directions={len(directions)} output={out}")


if __name__ == "__main__":
    main()
