#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


BINARY_METRICS = [
    "first_wait",
    "surface_reflection",
    "has_reflection",
    "explicit_repair_marker",
    "semantic_repair",
    "functional_repair",
    "final_correction",
    "hit_max_new_tokens",
]

CONTINUOUS_METRICS = [
    "generated_tokens",
    "p0_reflect_vs_stop",
    "p0_wait_logsum",
]

PAIRS = [
    ("C_add", "C_gateon", "C", 1.0),
    ("T_remove", "T_gateoff", "T", -1.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired behavior analysis for error-ack gate interventions.")
    p.add_argument("--root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--dirs",
        default="error_ack_logistic_behavior64_alpha*_20260619",
        help="Glob or comma-separated directories under root.",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        x = safe_float(value)
        if math.isfinite(x):
            out.append(x)
    return out


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    vals = sorted(xs)
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def binom_tail_ge(n: int, k: int) -> float:
    if n <= 0:
        return float("nan")
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (0.5**n)
    return min(1.0, max(0.0, total))


def infer_alpha(path: Path) -> float:
    match = re.search(r"alpha([0-9]+p[0-9]+|[0-9]+)", path.name)
    if not match:
        raise ValueError(path.name)
    return float(match.group(1).replace("p", "."))


def infer_mode(path: Path) -> str:
    if "pdecode" in path.name:
        return "prefill_plus_decode"
    if "prefill" in path.name:
        return "prefill_only"
    return ""


def resolve_dirs(root: Path, spec: str) -> List[Path]:
    dirs: List[Path] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        p = Path(part)
        if any(ch in part for ch in "*?[]"):
            matches = sorted(root.glob(part))
        elif p.is_absolute():
            matches = [p]
        else:
            matches = [root / part]
        for path in matches:
            if path.is_dir() and (path / "behavior_rows.jsonl").exists() and path not in dirs:
                dirs.append(path)
    return sorted(dirs, key=lambda p: (infer_alpha(p), infer_mode(p), p.name))


def condition_matches(condition: str, prefix: str) -> bool:
    return condition == prefix or condition.startswith(prefix + "_")


def key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    global_idx = row.get("global_idx")
    return (
        str(row.get("example_id")),
        int(global_idx) if global_idx is not None else -1,
        str(row.get("mode") or "free"),
    )


def paired_rows_for_dir(path: Path) -> List[Dict[str, Any]]:
    alpha = infer_alpha(path)
    gate_mode = infer_mode(path)
    grouped: Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path / "behavior_rows.jsonl"):
        grouped[key(row)][str(row.get("condition"))] = row
    out: List[Dict[str, Any]] = []
    for k, by_cond in sorted(grouped.items()):
        example_id, global_idx, mode = k
        for pair_name, treated_prefix, base_condition, expected_sign in PAIRS:
            base = by_cond.get(base_condition)
            treated_items = [
                row
                for cond, row in by_cond.items()
                if condition_matches(cond, treated_prefix)
                and (not gate_mode or cond.endswith(gate_mode))
            ]
            if base is None or not treated_items:
                continue
            treated = treated_items[0]
            item: Dict[str, Any] = {
                "alpha": alpha,
                "gate_mode": gate_mode,
                "dir": path.name,
                "pair": pair_name,
                "example_id": example_id,
                "global_idx": global_idx,
                "mode": mode,
                "treated_condition": treated.get("condition"),
                "base_condition": base_condition,
                "expected_sign": expected_sign,
                "base_continuation_text": base.get("continuation_text", ""),
                "treated_continuation_text": treated.get("continuation_text", ""),
            }
            for metric in BINARY_METRICS:
                b = as_bool(base.get(metric))
                t = as_bool(treated.get(metric))
                item[f"base_{metric}"] = b
                item[f"treated_{metric}"] = t
                item[f"delta_{metric}"] = int(t) - int(b)
                item[f"directional_delta_{metric}"] = (int(t) - int(b)) * expected_sign
            for metric in CONTINUOUS_METRICS:
                delta = safe_float(treated.get(metric)) - safe_float(base.get(metric))
                item[f"base_{metric}"] = safe_float(base.get(metric))
                item[f"treated_{metric}"] = safe_float(treated.get(metric))
                item[f"delta_{metric}"] = delta
                item[f"directional_delta_{metric}"] = delta * expected_sign
            out.append(item)
    return out


def summarize_binary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for metric in BINARY_METRICS:
            grouped[
                (
                    str(row.get("dir")),
                    safe_float(row.get("alpha")),
                    str(row.get("gate_mode")),
                    str(row.get("pair")),
                    metric,
                )
            ].append(row)
    out: List[Dict[str, Any]] = []
    for (direction, alpha, gate_mode, pair, metric), group in sorted(grouped.items()):
        base_rate = mean([float(as_bool(row.get(f"base_{metric}"))) for row in group])
        treated_rate = mean([float(as_bool(row.get(f"treated_{metric}"))) for row in group])
        good = sum(1 for row in group if safe_float(row.get(f"directional_delta_{metric}")) > 0)
        bad = sum(1 for row in group if safe_float(row.get(f"directional_delta_{metric}")) < 0)
        same = len(group) - good - bad
        discordant = good + bad
        out.append(
            {
                "direction": direction,
                "alpha": alpha,
                "gate_mode": gate_mode,
                "pair": pair,
                "metric": metric,
                "n": len(group),
                "base_rate": base_rate,
                "treated_rate": treated_rate,
                "raw_delta_rate": treated_rate - base_rate,
                "directional_delta_rate": (treated_rate - base_rate) * (1.0 if pair == "C_add" else -1.0),
                "direction_good": good,
                "direction_bad": bad,
                "unchanged": same,
                "direction_good_rate_among_discordant": good / discordant if discordant else float("nan"),
                "one_sided_discordant_p": binom_tail_ge(discordant, good),
            }
        )
    return out


def summarize_continuous(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for metric in CONTINUOUS_METRICS:
            grouped[
                (
                    str(row.get("dir")),
                    safe_float(row.get("alpha")),
                    str(row.get("gate_mode")),
                    str(row.get("pair")),
                    metric,
                )
            ].append(row)
    out: List[Dict[str, Any]] = []
    for (direction, alpha, gate_mode, pair, metric), group in sorted(grouped.items()):
        vals = finite(row.get(f"directional_delta_{metric}") for row in group)
        good = sum(1 for v in vals if v > 0)
        bad = sum(1 for v in vals if v < 0)
        nz = good + bad
        out.append(
            {
                "direction": direction,
                "alpha": alpha,
                "gate_mode": gate_mode,
                "pair": pair,
                "metric": metric,
                "n": len(vals),
                "mean_directional_delta": mean(vals),
                "median_directional_delta": median(vals) if vals else float("nan"),
                "q25_directional_delta": quantile(vals, 0.25),
                "q75_directional_delta": quantile(vals, 0.75),
                "direction_good_rate": good / nz if nz else float("nan"),
                "one_sided_sign_p": binom_tail_ge(nz, good),
            }
        )
    return out


def top_effects(binary_rows: Sequence[Mapping[str, Any]], continuous_rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    candidates: List[Mapping[str, Any]] = []
    for row in binary_rows:
        if str(row.get("metric")) in {"surface_reflection", "has_reflection", "explicit_repair_marker", "semantic_repair", "functional_repair"}:
            candidates.append(row)
    return sorted(
        candidates,
        key=lambda r: (safe_float(r.get("directional_delta_rate")), -safe_float(r.get("one_sided_discordant_p"))),
        reverse=True,
    )[:12]


def write_report(path: Path, binary_rows: Sequence[Mapping[str, Any]], continuous_rows: Sequence[Mapping[str, Any]], paired_n: int) -> None:
    lines = [
        "# error-ack logistic gate 短窗口行为分析",
        "",
        "## 目标",
        "",
        "检验 error-ack hidden 方向是否不只移动第一个 token 的 logit，而是真的改变 64 token 续写里的反思与修复行为。",
        "",
        f"- 配对记录数：{paired_n}",
        "",
        "## 最相关行为指标",
        "",
        "| 方向 | alpha | 注入方式 | 对比 | 行为 | base | treated | 方向性变化 | discordant 正向率 | p |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_effects(binary_rows, continuous_rows):
        lines.append(
            "| {direction} | {alpha:g} | {mode} | {pair} | {metric} | {base:.3f} | {treated:.3f} | {delta:.3f} | {good:.3f} | {p:.3g} |".format(
                direction=row.get("direction"),
                alpha=safe_float(row.get("alpha")),
                mode=row.get("gate_mode"),
                pair=row.get("pair"),
                metric=row.get("metric"),
                base=safe_float(row.get("base_rate")),
                treated=safe_float(row.get("treated_rate")),
                delta=safe_float(row.get("directional_delta_rate")),
                good=safe_float(row.get("direction_good_rate_among_discordant")),
                p=safe_float(row.get("one_sided_discordant_p")),
            )
        )
    lines.extend(
        [
            "",
            "## 自检",
            "",
            "如果行为指标没有随方向改变，即使 p0 logit 结果很显著，也只能说明有局部决策边界方向，不能声称我们控制了反思行为。",
            "如果行为指标方向一致但修复率不动，下一步要把结论限定为“错误承认/反思启动方向”，并另找 repair 方向。",
            "如果行为和修复都显著，再扩大样本并做随机方向、反向方向、层位点 ablation，才接近可汇报的强证据链。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = resolve_dirs(root, args.dirs)
    if not dirs:
        raise RuntimeError("No completed behavior dirs found.")
    paired: List[Dict[str, Any]] = []
    for path in dirs:
        paired.extend(paired_rows_for_dir(path))
    if not paired:
        raise RuntimeError("No paired behavior rows found.")
    binary = summarize_binary(paired)
    continuous = summarize_continuous(paired)
    write_csv(out / "paired_behavior_rows.csv", paired)
    write_csv(out / "binary_summary.csv", binary)
    write_csv(out / "continuous_summary.csv", continuous)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "dirs": [str(p) for p in dirs],
                "paired_rows": len(paired),
                "binary_summary_rows": len(binary),
                "continuous_summary_rows": len(continuous),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(out / "REPORT.md", binary, continuous, len(paired))
    print(f"[Done] wrote {out} paired_rows={len(paired)} dirs={len(dirs)}")


if __name__ == "__main__":
    main()
