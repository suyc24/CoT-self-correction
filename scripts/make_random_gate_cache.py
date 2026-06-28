#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a gate cache with random unit directions matching an existing cache scale.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--source_direction_type", default="random_control")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    src = torch.load(args.input, map_location="cpu")
    out_items: List[Dict[str, Any]] = []
    out_rows: List[Dict[str, Any]] = []
    for item in src.get("directions", []):
        direction = item.get("direction")
        if direction is None:
            continue
        rand = torch.randn_like(direction.float())
        rand = rand / rand.norm().clamp_min(1e-12)
        copied = dict(item)
        copied["direction"] = rand.cpu()
        copied["direction_type"] = "gate"
        copied["source_direction_type"] = str(args.source_direction_type)
        copied["wrapped_behavior_direction"] = "random_control"
        copied["random_seed"] = int(args.seed)
        out_items.append(copied)
        out_rows.append({k: v for k, v in copied.items() if k != "direction"})
    if not out_items:
        raise RuntimeError("No directions found.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"directions": out_items, "direction_rows": out_rows, "source": str(args.input)}
    torch.save(payload, out)
    out.with_suffix(".json").write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] wrote {out} directions={len(out_items)}")


if __name__ == "__main__":
    main()
