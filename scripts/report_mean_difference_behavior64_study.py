#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


KEY_BINARY = ["surface_reflection", "semantic_repair", "functional_repair", "final_correction"]
KEY_CONT = ["generated_tokens", "p0_reflect_vs_stop"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a Chinese report for mean-difference error-ack behavior64 study.")
    p.add_argument("--analysis_dir", required=True)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


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


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def short_name(direction: str) -> str:
    if "mean_difference_negated" in direction:
        return "mean_negated"
    if "mean_difference" in direction:
        if "alpha0p5" in direction:
            return "mean_alpha0.5"
        if "alpha1p0" in direction:
            return "mean_alpha1"
        if "alpha2p0" in direction:
            return "mean_alpha2"
        return "mean"
    if "random" in direction and "seed01" in direction:
        return "random_seed01"
    if "random" in direction and "seed02" in direction:
        return "random_seed02"
    return direction


def select_binary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("pair")) != "T_remove":
            continue
        if str(row.get("metric")) not in KEY_BINARY:
            continue
        item = dict(row)
        item["short_direction"] = short_name(str(row.get("direction")))
        out.append(item)
    return out


def select_cont(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("pair")) != "T_remove":
            continue
        if str(row.get("metric")) not in KEY_CONT:
            continue
        item = dict(row)
        item["short_direction"] = short_name(str(row.get("direction")))
        out.append(item)
    return out


def sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    order = {
        "mean_alpha0.5": 0,
        "mean_alpha1": 1,
        "mean_alpha2": 2,
        "mean_negated": 3,
        "random_seed01": 4,
        "random_seed02": 5,
    }
    return (order.get(str(row.get("short_direction")), 99), str(row.get("metric")))


def fmt(x: Any, digits: int = 3) -> str:
    val = safe_float(x)
    if not math.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def write_report(path: Path, binary_rows: Sequence[Mapping[str, Any]], cont_rows: Sequence[Mapping[str, Any]]) -> None:
    binary_rows = sorted(binary_rows, key=sort_key)
    cont_rows = sorted(cont_rows, key=sort_key)
    lines = [
        "# mean-difference error-ack 行为干预报告",
        "",
        "## 核心问题",
        "",
        "检验从自然反思 hidden state 中得到的 mean-difference error-ack 方向，是否能在错误 box 场景中稳定改变后续 64 token 的真实反思与修复行为，而不是只改变第一个 token 的 logit。",
        "",
        "## T 错误 box：二值行为",
        "",
        "| 方向 | 行为 | n | base | treated | 方向性变化 | 正向discordant | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in binary_rows:
        lines.append(
            "| {direction} | {metric} | {n} | {base} | {treated} | {delta} | {good} | {p} |".format(
                direction=row.get("short_direction"),
                metric=row.get("metric"),
                n=int(float(row.get("n") or 0)),
                base=fmt(row.get("base_rate")),
                treated=fmt(row.get("treated_rate")),
                delta=fmt(row.get("directional_delta_rate")),
                good=fmt(row.get("direction_good_rate_among_discordant")),
                p=fmt(row.get("one_sided_discordant_p"), 4),
            )
        )
    lines.extend(
        [
            "",
            "## T 错误 box：连续指标",
            "",
            "| 方向 | 指标 | n | 平均方向性变化 | 中位数 | 正向率 | p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cont_rows:
        lines.append(
            "| {direction} | {metric} | {n} | {mean} | {median} | {good} | {p} |".format(
                direction=row.get("short_direction"),
                metric=row.get("metric"),
                n=int(float(row.get("n") or 0)),
                mean=fmt(row.get("mean_directional_delta")),
                median=fmt(row.get("median_directional_delta")),
                good=fmt(row.get("direction_good_rate")),
                p=fmt(row.get("one_sided_sign_p"), 4),
            )
        )
    lines.extend(
        [
            "",
            "## 目前可汇报的形状",
            "",
            "如果 mean_alpha0.5/1/2 在 T_remove 上呈剂量依赖，mean_negated 反向增强反思，random_seed01/02 不产生同等行为变化，那么主结论可以从“p0 logit 被移动”升级为“自然反思 hidden 方向可以因果控制错误 box 后的反思/修复行为”。",
            "",
            "## 100% 信心自检",
            "",
            "- 还需要扩大样本确认效果不是 40/120 题局部现象。",
            "- 还需要层位点消融，确认不是任意层任意方向都能做到。",
            "- 还需要把行为改变和答案正确性、副作用长度分开报告，避免把“压制输出”误解为“压制反思”。",
            "- 如果 random controls 在完整样本上也改变行为，就必须把结论降级为大幅扰动效应，而不是 error-ack 方向效应。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    analysis = Path(args.analysis_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    binary = select_binary(read_csv(analysis / "binary_summary.csv"))
    cont = select_cont(read_csv(analysis / "continuous_summary.csv"))
    write_csv(out / "t_remove_binary_focus.csv", binary)
    write_csv(out / "t_remove_continuous_focus.csv", cont)
    (out / "summary.json").write_text(
        json.dumps({"binary_rows": len(binary), "continuous_rows": len(cont), "analysis_dir": str(analysis)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out / "REPORT.md", binary, cont)
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
