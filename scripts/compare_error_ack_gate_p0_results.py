#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare p0 alpha sweep analyses across gate directions.")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--analysis",
        action="append",
        default=[],
        help="label:path/to/analysis_dir. Can be passed multiple times.",
    )
    return p.parse_args()


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def parse_specs(specs: Sequence[str]) -> List[tuple[str, Path]]:
    out: List[tuple[str, Path]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Expected label:path, got {spec!r}")
        label, path = spec.split(":", 1)
        out.append((label, Path(path)))
    return out


def load_comparison(specs: Sequence[tuple[str, Path]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    mono_rows: List[Dict[str, Any]] = []
    for label, path in specs:
        for row in read_csv(path / "summary.csv"):
            summary_rows.append(
                {
                    "direction": label,
                    "alpha": safe_float(row.get("alpha")),
                    "pair": row.get("pair"),
                    "n": int(float(row.get("n") or 0)),
                    "mean_directional_delta": safe_float(row.get("mean_directional_delta")),
                    "median_directional_delta": safe_float(row.get("median_directional_delta")),
                    "positive_rate": safe_float(row.get("positive_rate")),
                    "one_sided_sign_p": safe_float(row.get("one_sided_sign_p")),
                    "analysis_dir": str(path),
                }
            )
        for row in read_csv(path / "alpha_monotonic_summary.csv"):
            mono_rows.append(
                {
                    "direction": label,
                    "pair": row.get("pair"),
                    "n_examples": int(float(row.get("n_examples") or 0)),
                    "monotone_rate": safe_float(row.get("monotone_rate")),
                    "positive_all_alpha_rate": safe_float(row.get("positive_all_alpha_rate")),
                    "mean_last_minus_first": safe_float(row.get("mean_last_minus_first")),
                    "analysis_dir": str(path),
                }
            )
    return summary_rows, mono_rows


def write_report(path: Path, summary_rows: Sequence[Mapping[str, Any]], mono_rows: Sequence[Mapping[str, Any]]) -> None:
    top_alpha = sorted(
        [r for r in summary_rows if abs(safe_float(r.get("alpha")) - 2.0) < 1e-9],
        key=lambda r: (str(r.get("pair")), -safe_float(r.get("mean_directional_delta"))),
    )
    lines = [
        "# error-ack gate p0 对照汇总",
        "",
        "## alpha=2 主对照",
        "",
        "| 方向 | 对比 | n | 平均方向性位移 | 中位数 | 方向一致率 | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_alpha:
        lines.append(
            "| {direction} | {pair} | {n} | {mean:.4f} | {median:.4f} | {pos:.3f} | {p:.3g} |".format(
                direction=row.get("direction"),
                pair=row.get("pair"),
                n=int(row.get("n") or 0),
                mean=safe_float(row.get("mean_directional_delta")),
                median=safe_float(row.get("median_directional_delta")),
                pos=safe_float(row.get("positive_rate")),
                p=safe_float(row.get("one_sided_sign_p")),
            )
        )
    lines.extend(
        [
            "",
            "## alpha 单调性",
            "",
            "| 方向 | 对比 | 题目数 | 单调率 | 所有 alpha 正向率 | 最大减最小平均增量 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in mono_rows:
        lines.append(
            "| {direction} | {pair} | {n} | {mono:.3f} | {pos:.3f} | {gain:.4f} |".format(
                direction=row.get("direction"),
                pair=row.get("pair"),
                n=int(row.get("n_examples") or 0),
                mono=safe_float(row.get("monotone_rate")),
                pos=safe_float(row.get("positive_all_alpha_rate")),
                gain=safe_float(row.get("mean_last_minus_first")),
            )
        )
    lines.extend(
        [
            "",
            "## 解释口径",
            "",
            "这个表用于区分三个问题：方向是否可读、干预幅度是否足够、方向是否特异。只有当 rescaled logistic 的位移明显超过 random control，且 alpha 关系稳定时，才值得进入行为级扩大实验。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = parse_specs(args.analysis)
    summary_rows, mono_rows = load_comparison(specs)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "p0_direction_comparison.csv", summary_rows)
    write_csv(out / "p0_monotonic_comparison.csv", mono_rows)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "analyses": [{"label": label, "path": str(path)} for label, path in specs],
                "summary_rows": len(summary_rows),
                "monotonic_rows": len(mono_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(out / "REPORT.md", summary_rows, mono_rows)
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
