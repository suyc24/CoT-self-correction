#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REFLECTION_MARKERS = [
    "wait",
    "actually",
    "however",
    "but",
    "hold on",
    "let me check",
    "mistake",
    "incorrect",
    "wrong",
    "recheck",
    "recalculate",
    "不对",
    "等等",
    "等一下",
    "重新",
    "检查",
    "错误",
]

STOP_MARKERS = [
    "<|im_end|>",
    "</think>",
    "therefore",
    "thus",
    "hence",
    "the answer",
    "so the answer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reflection vs termination competition in paired behavior rows.")
    parser.add_argument("--analysis_dir", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
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


def fmt(value: Any, digits: int = 3) -> str:
    x = safe_float(value)
    return "" if not math.isfinite(x) else f"{x:.{digits}f}"


def short_direction(direction: str) -> str:
    if "mean_difference_negated" in direction:
        return "mean_negated_alpha2"
    if "random_rescaled6p029_seed02" in direction:
        return "random_seed02_alpha2"
    if "random_rescaled6p029" in direction:
        return "random_seed01_alpha2"
    if "alpha0p5" in direction:
        return "mean_alpha0.5"
    if "alpha1p0" in direction:
        return "mean_alpha1"
    if "alpha2p0" in direction:
        return "mean_alpha2"
    return direction


def study_name(path: Path) -> str:
    name = path.name
    for prefix in ["error_ack_mean_difference_behavior64_", "error_ack_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in ["_analysis_20260619", "_analysis_20260620"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def first_marker(text: str, markers: Sequence[str]) -> Tuple[int, str]:
    lower = text.lower()
    best = -1
    best_marker = ""
    for marker in markers:
        pos = lower.find(marker.lower())
        if pos >= 0 and (best < 0 or pos < best):
            best = pos
            best_marker = marker
    return best, best_marker


def approx_token_index(text: str, char_pos: int) -> float:
    if char_pos < 0:
        return float("nan")
    prefix = text[:char_pos]
    pieces = [p for p in re.split(r"\s+|(?=[,.;:!?])|(?<=[,.;:!?])", prefix) if p]
    return float(len(pieces))


def text_features(text: str) -> Dict[str, Any]:
    reflect_pos, reflect_marker = first_marker(text, REFLECTION_MARKERS)
    stop_pos, stop_marker = first_marker(text, STOP_MARKERS)
    has_reflect = reflect_pos >= 0
    has_stop = stop_pos >= 0
    reflect_before_stop = bool(has_reflect and (not has_stop or reflect_pos < stop_pos))
    stop_before_reflect = bool(has_stop and (not has_reflect or stop_pos <= reflect_pos))
    stripped = text.strip().lower()
    return {
        "has_reflect_marker": has_reflect,
        "has_stop_marker": has_stop,
        "reflect_before_stop": reflect_before_stop,
        "stop_before_reflect": stop_before_reflect,
        "first_reflect_char": reflect_pos,
        "first_stop_char": stop_pos,
        "first_reflect_marker": reflect_marker,
        "first_stop_marker": stop_marker,
        "first_reflect_token_approx": approx_token_index(text, reflect_pos),
        "first_stop_token_approx": approx_token_index(text, stop_pos),
        "starts_with_stop": bool(stripped.startswith("$$") or stripped.startswith("<|im_end|>") or stripped.startswith("</think>")),
    }


def mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(1.0 for row in rows if bool(row.get(key))) / len(rows)


def finite_mean(values: Iterable[Any]) -> float:
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def binom_tail_ge(n: int, k: int) -> float:
    if n <= 0:
        return float("nan")
    return min(1.0, sum(math.comb(n, i) * (0.5**n) for i in range(k, n + 1)))


def build_event_rows(analysis_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    study = study_name(analysis_dir)
    for row in read_csv(analysis_dir / "paired_behavior_rows.csv"):
        if row.get("pair") != "T_remove":
            continue
        base = text_features(row.get("base_continuation_text", ""))
        treated = text_features(row.get("treated_continuation_text", ""))
        item: Dict[str, Any] = {
            "study": study,
            "direction": short_direction(str(row.get("dir"))),
            "example_id": row.get("example_id"),
            "global_idx": row.get("global_idx"),
            "base_generated_tokens": row.get("base_generated_tokens"),
            "treated_generated_tokens": row.get("treated_generated_tokens"),
            "delta_generated_tokens": row.get("delta_generated_tokens"),
            "base_continuation_text": row.get("base_continuation_text", ""),
            "treated_continuation_text": row.get("treated_continuation_text", ""),
        }
        for key, value in base.items():
            item[f"base_{key}"] = value
        for key, value in treated.items():
            item[f"treated_{key}"] = value
        out.append(item)
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("study")), str(row.get("direction")))].append(row)
    out: List[Dict[str, Any]] = []
    for (study, direction), group in sorted(grouped.items()):
        for metric in ["reflect_before_stop", "stop_before_reflect", "starts_with_stop", "has_reflect_marker"]:
            base_rate = mean_bool(group, f"base_{metric}")
            treated_rate = mean_bool(group, f"treated_{metric}")
            if metric in {"stop_before_reflect", "starts_with_stop"}:
                directional_delta = treated_rate - base_rate
                good = sum(
                    1
                    for row in group
                    if int(bool(row.get(f"treated_{metric}"))) - int(bool(row.get(f"base_{metric}"))) > 0
                )
                bad = sum(
                    1
                    for row in group
                    if int(bool(row.get(f"treated_{metric}"))) - int(bool(row.get(f"base_{metric}"))) < 0
                )
            else:
                directional_delta = base_rate - treated_rate
                good = sum(
                    1
                    for row in group
                    if int(bool(row.get(f"base_{metric}"))) - int(bool(row.get(f"treated_{metric}"))) > 0
                )
                bad = sum(
                    1
                    for row in group
                    if int(bool(row.get(f"base_{metric}"))) - int(bool(row.get(f"treated_{metric}"))) < 0
                )
            discordant = good + bad
            out.append(
                {
                    "study": study,
                    "direction": direction,
                    "metric": metric,
                    "n": len(group),
                    "base_rate": base_rate,
                    "treated_rate": treated_rate,
                    "directional_delta_rate": directional_delta,
                    "direction_good": good,
                    "direction_bad": bad,
                    "direction_good_rate_among_discordant": good / discordant if discordant else float("nan"),
                    "one_sided_discordant_p": binom_tail_ge(discordant, good),
                }
            )
        out.append(
            {
                "study": study,
                "direction": direction,
                "metric": "first_stop_token_approx",
                "n": len(group),
                "base_mean": finite_mean(row.get("base_first_stop_token_approx") for row in group),
                "treated_mean": finite_mean(row.get("treated_first_stop_token_approx") for row in group),
                "raw_delta_mean": finite_mean(
                    safe_float(row.get("treated_first_stop_token_approx")) - safe_float(row.get("base_first_stop_token_approx"))
                    for row in group
                ),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_rows: List[Dict[str, Any]] = []
    for item in args.analysis_dir:
        event_rows.extend(build_event_rows(Path(item)))
    summary_rows = summarize(event_rows)
    write_csv(out / "reflection_termination_events.csv", event_rows)
    write_csv(out / "reflection_termination_summary.csv", summary_rows)

    focus = [
        row
        for row in summary_rows
        if row.get("direction") in {"mean_alpha1", "mean_alpha2", "mean_negated_alpha2", "random_seed01_alpha2", "random_seed02_alpha2"}
        and row.get("metric") in {"reflect_before_stop", "stop_before_reflect", "starts_with_stop"}
    ]
    lines = [
        "# 反思-终止门控分析",
        "",
        "## 核心读法",
        "",
        "这个分析不再只问 64 token 内有没有反思，而是问反思 marker 和终止/收束 marker 谁先出现。若 mean-difference T_remove 主要提高先终止比例、降低先反思比例，而随机方向没有同幅度变化，说明该 hidden 方向更像反思-终止竞争门控。",
        "",
        "| 样本块 | 方向 | 指标 | n | baseline | intervention | 方向性变化 | 正向discordant | p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus:
        lines.append(
            f"| {row['study']} | {row['direction']} | {row['metric']} | {row['n']} | {fmt(row.get('base_rate'))} | {fmt(row.get('treated_rate'))} | {fmt(row.get('directional_delta_rate'))} | {fmt(row.get('direction_good_rate_among_discordant'))} | {fmt(row.get('one_sided_discordant_p'), 4)} |"
        )
    lines.extend(
        [
            "",
            "## 当前结论边界",
            "",
            "如果禁止早停实验中该效应消失，主结论应表述为“自然反思 hidden 方向控制错误答案后的继续推理/终止竞争”。如果禁止早停后仍改变语义修复，则可以再升级为“反思语义程序方向”。",
            "",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"events": len(event_rows), "summary_rows": len(summary_rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
