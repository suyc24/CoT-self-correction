#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy a gate cache while changing the layer/site metadata.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layer_idx", type=int, required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--note", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.input, map_location="cpu")
    out_items: List[Dict[str, Any]] = []
    for item in payload.get("directions", []):
        copied = dict(item)
        copied["source_layer_idx"] = copied.get("layer_idx")
        copied["source_site"] = copied.get("site")
        copied["layer_idx"] = int(args.layer_idx)
        copied["site"] = str(args.site)
        copied["retarget_note"] = str(args.note)
        out_items.append(copied)
    if not out_items:
        raise RuntimeError("No directions found.")
    out_payload = {
        **payload,
        "directions": out_items,
        "direction_rows": [{k: v for k, v in item.items() if k != "direction"} for item in out_items],
        "retargeted_from": str(args.input),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out)
    out.with_suffix(".json").write_text(json.dumps(out_payload["direction_rows"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] wrote {out} layer={int(args.layer_idx)} site={args.site}")


if __name__ == "__main__":
    main()
