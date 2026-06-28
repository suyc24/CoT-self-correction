#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


KEY_METRICS = ["surface_reflection", "semantic_repair", "functional_repair", "final_correction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Chinese consolidated report for error-ack behavior steering.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--study", action="append", default=[], help="name:path/to/analysis_dir")
    parser.add_argument("--locus_analysis_dir", default="")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    x = safe_float(value)
    return "" if not math.isfinite(x) else f"{x:.{digits}f}"


def direction_label(direction: str) -> str:
    if "mean_difference_negated" in direction:
        return "反向 mean-difference"
    if "random_rescaled6p029_seed02" in direction:
        return "随机方向 seed02"
    if "random_rescaled6p029" in direction:
        return "随机方向 seed01"
    if "mean_difference" in direction:
        if "alpha0p5" in direction:
            return "mean-difference alpha=0.5"
        if "alpha1p0" in direction:
            return "mean-difference alpha=1.0"
        if "alpha2p0" in direction:
            return "mean-difference alpha=2.0"
        return "mean-difference"
    return direction


def iter_focus_rows(study_name: str, analysis_dir: Path) -> Iterable[Dict[str, Any]]:
    for row in read_csv(analysis_dir / "binary_summary.csv"):
        if row.get("pair") != "T_remove" or row.get("metric") not in KEY_METRICS:
            continue
        yield {
            "study": study_name,
            "direction": direction_label(str(row.get("direction", ""))),
            "metric": row.get("metric", ""),
            "n": row.get("n", ""),
            "base": row.get("base_rate", ""),
            "treated": row.get("treated_rate", ""),
            "delta": row.get("directional_delta_rate", ""),
            "p": row.get("one_sided_discordant_p", ""),
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
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_studies(items: Sequence[str]) -> List[tuple[str, Path]]:
    out: List[tuple[str, Path]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"--study must be name:path, got {item!r}")
        name, path = item.split(":", 1)
        out.append((name, Path(path)))
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    studies = parse_studies(args.study)
    focus_rows: List[Dict[str, Any]] = []
    for name, path in studies:
        focus_rows.extend(iter_focus_rows(name, path))
    write_csv(out / "cross_sample_t_remove_summary.csv", focus_rows)

    locus_rows: List[Dict[str, str]] = []
    if args.locus_analysis_dir:
        locus_rows = [
            row
            for row in read_csv(Path(args.locus_analysis_dir) / "binary_summary.csv")
            if row.get("pair") == "T_remove" and row.get("metric") in KEY_METRICS
        ]
        write_csv(out / "locus_t_remove_summary.csv", locus_rows)

    lines = [
        "# 自然反思 error-ack hidden 方向：阶段性合并报告",
        "",
        "## 当前可汇报结论",
        "",
        "最强结果不是聚类，而是从自然反思 hidden state 中构造的 error-ack 方向，能在错误 box 后的 64 token 续写中剂量依赖地改变反思与修复行为。该结果已经在两个不重叠样本块上复现，并且反向方向产生相反行为变化，随机方向没有同等幅度的行为效应。",
        "",
        "## 跨样本 T 错误 box 结果",
        "",
        "| 样本块 | 方向 | 行为 | n | baseline | intervention | 方向性变化 | p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in focus_rows:
        if "alpha=2.0" not in str(row["direction"]) and "alpha=1.0" not in str(row["direction"]) and "随机" not in str(row["direction"]) and "反向" not in str(row["direction"]):
            continue
        lines.append(
            f"| {row['study']} | {row['direction']} | {row['metric']} | {row['n']} | {fmt(row['base'])} | {fmt(row['treated'])} | {fmt(row['delta'])} | {fmt(row['p'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## 论文价值判断",
            "",
            "这个结果具备继续推进价值：它已经从“hidden state 可分类”升级到“自然反思方向可以因果改变后续行为”。这比低 silhouette 聚类更接近可汇报主线，因为它有行为因果效应、剂量曲线、反向对照和随机对照。",
            "",
            "但它还不够支撑强论文主张。主要漏洞是：",
            "",
            "- 方向会影响续写长度，因此目前不能说它只控制反思语义；更准确说法是控制错误承认、反思/修复启动以及短窗口延展。",
            "- 现有层位点实验把 L22 方向重定向到别的层，不是每层独立学习方向；因此不能声称 L22/post_attn 是唯一机制位置。",
            "- 目前强结论主要来自错误 box 场景；还需要验证在自然错误、非 box 篡改或不同题型上是否成立。",
            "- p0 logit 对随机方向敏感，不能作为核心证据，只能作为辅助局部指标。",
            "",
            "## 下一轮实验门槛",
            "",
            "如果第三个不重叠样本块继续复现 alpha=2.0 的大效应，且随机方向仍弱，那么结果可以作为组会主结果。如果每层独立方向也显示特定层/阶段更强，才有机会进入机制性论文主线。如果长度控制后行为效应消失，结论必须降级为 continuation-length steering。",
            "",
        ]
    )
    if locus_rows:
        lines.extend(
            [
                "## 层位点试验备注",
                "",
                "已有层位点试验显示同一个方向重定向到 L19-L22 附近都能产生行为变化。这说明方向可能是宽残差流 steering vector，而不是已经定位到单个层的局部机制。下一步必须每层独立学习方向，再比较行为效应。",
                "",
            ]
        )

    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "studies": [{"name": name, "path": str(path)} for name, path in studies],
                "focus_rows": len(focus_rows),
                "locus_rows": len(locus_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
