#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


BINARY_METRICS = ["surface_reflection", "semantic_repair", "functional_repair", "final_correction"]
LENGTH_WINDOWS = [0, 2, 5, 10, 10**9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit behavior steering after restricting paired examples by generation-length change.")
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


def as_bool(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def binom_tail_ge(n: int, k: int) -> float:
    if n <= 0:
        return float("nan")
    return min(1.0, sum(math.comb(n, i) * (0.5**n) for i in range(k, n + 1)))


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


def summarize(rows: Sequence[Mapping[str, Any]], study: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pair") != "T_remove":
            continue
        grouped[(short_direction(str(row.get("dir"))), str(row.get("gate_mode")))].append(row)
    for (direction, gate_mode), group in sorted(grouped.items()):
        for window in LENGTH_WINDOWS:
            selected = [
                row
                for row in group
                if abs(safe_float(row.get("delta_generated_tokens"))) <= float(window)
            ]
            label = f"<= {window}" if window < 10**8 else "all"
            for metric in BINARY_METRICS:
                good = bad = same = 0
                base_vals: List[float] = []
                treated_vals: List[float] = []
                for row in selected:
                    base = as_bool(row.get(f"base_{metric}"))
                    treated = as_bool(row.get(f"treated_{metric}"))
                    base_vals.append(float(base))
                    treated_vals.append(float(treated))
                    delta = int(treated) - int(base)
                    directional = -delta
                    if directional > 0:
                        good += 1
                    elif directional < 0:
                        bad += 1
                    else:
                        same += 1
                discordant = good + bad
                out.append(
                    {
                        "study": study,
                        "direction": direction,
                        "gate_mode": gate_mode,
                        "length_delta_abs_window": label,
                        "metric": metric,
                        "n": len(selected),
                        "base_rate": sum(base_vals) / len(base_vals) if base_vals else float("nan"),
                        "treated_rate": sum(treated_vals) / len(treated_vals) if treated_vals else float("nan"),
                        "directional_delta_rate": (sum(base_vals) - sum(treated_vals)) / len(selected) if selected else float("nan"),
                        "direction_good": good,
                        "direction_bad": bad,
                        "unchanged": same,
                        "direction_good_rate_among_discordant": good / discordant if discordant else float("nan"),
                        "one_sided_discordant_p": binom_tail_ge(discordant, good),
                    }
                )
    return out


def fmt(value: Any, digits: int = 3) -> str:
    x = safe_float(value)
    return "" if not math.isfinite(x) else f"{x:.{digits}f}"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for item in args.analysis_dir:
        path = Path(item)
        study = path.name.replace("error_ack_mean_difference_behavior64_", "").replace("_analysis_20260619", "")
        rows.extend(summarize(read_csv(path / "paired_behavior_rows.csv"), study))
    write_csv(out / "length_sensitivity_summary.csv", rows)
    focus = [
        row
        for row in rows
        if row["direction"] in {"mean_alpha2", "mean_alpha1", "random_seed01_alpha2", "random_seed02_alpha2"}
        and row["metric"] in {"surface_reflection", "semantic_repair"}
        and row["length_delta_abs_window"] in {"<= 0", "<= 2", "all"}
    ]
    lines = [
        "# 长度混淆审计",
        "",
        "## 读法",
        "",
        "这里把同一题的 baseline 和 intervention 配对后，按续写 token 数变化筛选。若只在长度变化很大的样本中有效，行为结论会受到长度混淆；若等长或近似等长子集仍有方向一致变化，说明不完全是输出变短导致。",
        "",
        "| 样本块 | 方向 | 长度变化窗口 | 行为 | n | baseline | intervention | 方向性变化 | 正向discordant | p |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus:
        lines.append(
            f"| {row['study']} | {row['direction']} | {row['length_delta_abs_window']} | {row['metric']} | {row['n']} | {fmt(row['base_rate'])} | {fmt(row['treated_rate'])} | {fmt(row['directional_delta_rate'])} | {fmt(row['direction_good_rate_among_discordant'])} | {fmt(row['one_sided_discordant_p'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            "这不是最终因果控制长度的证明，只是排雷。如果等长子集样本太少或效果消失，下一步应改为固定 continuation token budget 后对相同前缀长度打标签，或报告到首次反思 marker 的时间，而不是只看 64 token 内是否出现反思。",
            "",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"rows": len(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
