#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wrap one behavior direction as a gate cache for existing intervention scripts.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--direction_type", default="error_ack")
    p.add_argument("--layer_idx", type=int, default=22)
    p.add_argument("--site", default="post_attn")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu")
    matches: List[Dict[str, Any]] = []
    for item in payload.get("directions", []):
        if (
            str(item.get("direction_type")) == str(args.direction_type)
            and int(item.get("layer_idx")) == int(args.layer_idx)
            and str(item.get("site")) == str(args.site)
        ):
            copied = dict(item)
            copied["source_direction_type"] = copied.get("direction_type")
            copied["direction_type"] = "gate"
            copied["wrapped_behavior_direction"] = str(args.direction_type)
            matches.append(copied)
    if not matches:
        raise RuntimeError(f"No matching direction {args.direction_type} L{args.layer_idx}/{args.site}")
    out_payload = {
        "directions": matches,
        "direction_rows": [
            {k: v for k, v in item.items() if k != "direction"}
            for item in matches
        ],
        "source": str(args.input),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out)
    out.with_suffix(".json").write_text(json.dumps(out_payload["direction_rows"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
