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
    parser = argparse.ArgumentParser(
        description="Position-controlled event-time audit for reflection vs termination behavior."
    )
    parser.add_argument("--analysis_dir", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--windows", default="4,8,16,32,64")
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


def finite_mean(values: Iterable[Any]) -> float:
    xs = [safe_float(v) for v in values]
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    x = safe_float(value)
    return "" if not math.isfinite(x) else f"{x:.{digits}f}"


def binom_tail_ge(n: int, k: int) -> float:
    if n <= 0:
        return float("nan")
    return min(1.0, sum(math.comb(n, i) * (0.5**n) for i in range(k, n + 1)))


def parse_windows(raw: str) -> List[int]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return sorted(set(out))


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


def marker_pattern(markers: Sequence[str]) -> re.Pattern[str]:
    escaped = sorted((re.escape(x) for x in markers), key=len, reverse=True)
    return re.compile("|".join(escaped), flags=re.IGNORECASE)


REFLECT_RE = marker_pattern(REFLECTION_MARKERS)
STOP_RE = marker_pattern(STOP_MARKERS)
TOKEN_RE = re.compile(r"<\|[^>]+?\|>|</?think>|\\boxed|[A-Za-z]+|[0-9]+|[^\s]", flags=re.IGNORECASE)


def token_index_for_char(text: str, char_pos: int) -> float:
    if char_pos < 0:
        return float("nan")
    return float(sum(1 for match in TOKEN_RE.finditer(text) if match.start() < char_pos))


def first_match(text: str, pattern: re.Pattern[str]) -> Tuple[int, str]:
    match = pattern.search(text)
    if match is None:
        return -1, ""
    return match.start(), match.group(0)


def event_features(text: str, generated_tokens: Any) -> Dict[str, Any]:
    reflect_char, reflect_marker = first_match(text, REFLECT_RE)
    stop_char, stop_marker = first_match(text, STOP_RE)
    reflect_tok = token_index_for_char(text, reflect_char)
    stop_tok = token_index_for_char(text, stop_char)
    gen_len = safe_float(generated_tokens)
    has_reflect = math.isfinite(reflect_tok)
    has_stop = math.isfinite(stop_tok)
    if has_reflect and (not has_stop or reflect_tok < stop_tok):
        first_event = "reflect"
        first_event_token = reflect_tok
    elif has_stop:
        first_event = "stop"
        first_event_token = stop_tok
    else:
        first_event = "censored"
        first_event_token = gen_len
    return {
        "generated_tokens": gen_len,
        "first_reflect_token": reflect_tok,
        "first_stop_token": stop_tok,
        "first_reflect_marker": reflect_marker,
        "first_stop_marker": stop_marker,
        "first_event": first_event,
        "first_event_token": first_event_token,
    }


def bool_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(1.0 for row in rows if bool(row.get(key))) / len(rows)


def directional_sign_counts(
    rows: Sequence[Mapping[str, Any]],
    base_key: str,
    treated_key: str,
    direction: str,
) -> Tuple[int, int, int]:
    good = bad = 0
    for row in rows:
        base = int(bool(row.get(base_key)))
        treated = int(bool(row.get(treated_key)))
        if base == treated:
            continue
        if direction == "decrease":
            good += int(base > treated)
            bad += int(treated > base)
        elif direction == "increase":
            good += int(treated > base)
            bad += int(base > treated)
        else:
            raise ValueError(direction)
    return good, bad, good + bad


def build_rows(analysis_dir: Path, windows: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    study = study_name(analysis_dir)
    for row in read_csv(analysis_dir / "paired_behavior_rows.csv"):
        if row.get("pair") != "T_remove":
            continue
        base = event_features(row.get("base_continuation_text", ""), row.get("base_generated_tokens"))
        treated = event_features(row.get("treated_continuation_text", ""), row.get("treated_generated_tokens"))
        item: Dict[str, Any] = {
            "study": study,
            "direction": short_direction(str(row.get("dir"))),
            "example_id": row.get("example_id"),
            "global_idx": row.get("global_idx"),
            "base_text": row.get("base_continuation_text", ""),
            "treated_text": row.get("treated_continuation_text", ""),
        }
        for key, value in base.items():
            item[f"base_{key}"] = value
        for key, value in treated.items():
            item[f"treated_{key}"] = value
        for window in windows:
            for side, feats in [("base", base), ("treated", treated)]:
                reflect_t = safe_float(feats["first_reflect_token"])
                stop_t = safe_float(feats["first_stop_token"])
                gen_len = safe_float(feats["generated_tokens"])
                reflect_before_stop = math.isfinite(reflect_t) and reflect_t <= window and (
                    not math.isfinite(stop_t) or reflect_t < stop_t
                )
                stop_before_reflect = math.isfinite(stop_t) and stop_t <= window and (
                    not math.isfinite(reflect_t) or stop_t <= reflect_t
                )
                alive_to_window = math.isfinite(gen_len) and gen_len >= window and not stop_before_reflect
                item[f"{side}_reflect_by_{window}"] = reflect_before_stop
                item[f"{side}_stop_by_{window}"] = stop_before_reflect
                item[f"{side}_alive_to_{window}"] = alive_to_window
            item[f"both_alive_to_{window}"] = bool(item[f"base_alive_to_{window}"] and item[f"treated_alive_to_{window}"])
        rows.append(item)
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], windows: Sequence[int]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("study")), str(row.get("direction")))].append(row)
    out: List[Dict[str, Any]] = []
    for (study, direction), group in sorted(grouped.items()):
        out.append(
            {
                "study": study,
                "direction": direction,
                "metric": "first_event_token",
                "window": "",
                "n": len(group),
                "base_mean": finite_mean(row.get("base_first_event_token") for row in group),
                "treated_mean": finite_mean(row.get("treated_first_event_token") for row in group),
                "raw_delta_mean": finite_mean(
                    safe_float(row.get("treated_first_event_token")) - safe_float(row.get("base_first_event_token"))
                    for row in group
                ),
            }
        )
        for window in windows:
            for metric, trend in [("reflect", "decrease"), ("stop", "increase")]:
                base_key = f"base_{metric}_by_{window}"
                treated_key = f"treated_{metric}_by_{window}"
                good, bad, discordant = directional_sign_counts(group, base_key, treated_key, trend)
                out.append(
                    {
                        "study": study,
                        "direction": direction,
                        "metric": f"{metric}_by_window",
                        "window": window,
                        "subset": "all_pairs",
                        "n": len(group),
                        "base_rate": bool_rate(group, base_key),
                        "treated_rate": bool_rate(group, treated_key),
                        "directional_delta_rate": (
                            bool_rate(group, base_key) - bool_rate(group, treated_key)
                            if trend == "decrease"
                            else bool_rate(group, treated_key) - bool_rate(group, base_key)
                        ),
                        "direction_good": good,
                        "direction_bad": bad,
                        "direction_good_rate_among_discordant": good / discordant if discordant else float("nan"),
                        "one_sided_discordant_p": binom_tail_ge(discordant, good),
                    }
                )
                alive_group = [row for row in group if bool(row.get(f"both_alive_to_{window}"))]
                good, bad, discordant = directional_sign_counts(alive_group, base_key, treated_key, trend)
                out.append(
                    {
                        "study": study,
                        "direction": direction,
                        "metric": f"{metric}_by_window",
                        "window": window,
                        "subset": "both_alive_to_window",
                        "n": len(alive_group),
                        "base_rate": bool_rate(alive_group, base_key),
                        "treated_rate": bool_rate(alive_group, treated_key),
                        "directional_delta_rate": (
                            bool_rate(alive_group, base_key) - bool_rate(alive_group, treated_key)
                            if trend == "decrease"
                            else bool_rate(alive_group, treated_key) - bool_rate(alive_group, base_key)
                        ),
                        "direction_good": good,
                        "direction_bad": bad,
                        "direction_good_rate_among_discordant": good / discordant if discordant else float("nan"),
                        "one_sided_discordant_p": binom_tail_ge(discordant, good),
                    }
                )
    return out


def report_lines(summary_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    focus = [
        row
        for row in summary_rows
        if row.get("direction") in {"mean_alpha1", "mean_alpha2", "mean_negated_alpha2", "random_seed01_alpha2", "random_seed02_alpha2"}
        and row.get("metric") in {"reflect_by_window", "stop_by_window"}
        and str(row.get("window")) in {"8", "16", "32", "64"}
    ]
    lines = [
        "# 反思事件时间与位置控制审计",
        "",
        "## 读法",
        "",
        "这里把续写看成反思事件和终止事件的竞争。每个窗口只问：在同样的前若干 token 观察范围内，反思或终止是否已经先发生。`both_alive_to_window` 子集进一步要求 baseline 和 intervention 在该窗口内都没有提前终止，用来检查效果是否仍存在于等观察机会的样本中。",
        "",
        "| 样本块 | 方向 | 窗口 | 子集 | 指标 | n | baseline | intervention | 方向性变化 | 正向discordant | p |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus:
        lines.append(
            f"| {row['study']} | {row['direction']} | {row['window']} | {row.get('subset','')} | {row['metric']} | {row['n']} | {fmt(row.get('base_rate'))} | {fmt(row.get('treated_rate'))} | {fmt(row.get('directional_delta_rate'))} | {fmt(row.get('direction_good_rate_among_discordant'))} | {fmt(row.get('one_sided_discordant_p'), 4)} |"
        )
    lines.extend(
        [
            "",
            "## 当前解释",
            "",
            "如果 all-pairs 显著而 both-alive 子集不显著，说明主要效应来自提前终止或可观察长度变化；如果 both-alive 子集仍显著，才更接近“进入反思路径后的语义改变”。",
            "",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_rows: List[Dict[str, Any]] = []
    for item in args.analysis_dir:
        event_rows.extend(build_rows(Path(item), windows))
    summary_rows = summarize(event_rows, windows)
    write_csv(out / "event_time_rows.csv", event_rows)
    write_csv(out / "event_time_position_summary.csv", summary_rows)
    (out / "REPORT.md").write_text("\n".join(report_lines(summary_rows)), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"events": len(event_rows), "summary_rows": len(summary_rows), "windows": windows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
