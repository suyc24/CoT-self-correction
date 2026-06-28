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


METRICS = [
    "reflect_vs_stop",
    "reflect_logsum",
    "stop_logsum",
    "wait_logsum",
    "check_logsum",
    "actually_logsum",
    "finalize_logsum",
    "newline_logsum",
]

PAIRS = [
    ("C_add", "C_gateon_prefill_only", "C", 1.0),
    ("T_remove", "T_gateoff_prefill_only", "T", -1.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired analysis for error-ack direction p0 alpha sweep."
    )
    p.add_argument("--root", required=True, help="Directory containing alpha sweep output dirs.")
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--alpha_dirs",
        default="error_ack_p0_alpha*_20260619",
        help="Glob, comma-separated paths, or comma-separated directory names under root.",
    )
    p.add_argument("--metric", default="reflect_vs_stop")
    p.add_argument("--position_index", type=int, default=0)
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
                fields.append(key)
                seen.add(key)
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


def finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        x = safe_float(value)
        if math.isfinite(x):
            out.append(x)
    return out


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


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


def binom_tail_ge(n: int, k: int, p: float = 0.5) -> float:
    if n <= 0:
        return float("nan")
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(1.0, max(0.0, total))


def infer_alpha(path: Path) -> float:
    match = re.search(r"alpha([0-9]+p[0-9]+|[0-9]+)", path.name)
    if not match:
        raise ValueError(f"Cannot infer alpha from {path}")
    return float(match.group(1).replace("p", "."))


def resolve_alpha_dirs(root: Path, spec: str) -> List[Path]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    dirs: List[Path] = []
    for part in parts:
        candidate = Path(part)
        if any(ch in part for ch in "*?[]"):
            matches = sorted(root.glob(part))
        elif candidate.is_absolute():
            matches = [candidate]
        else:
            matches = [root / part]
        for path in matches:
            try:
                infer_alpha(path)
            except ValueError:
                continue
            if path.is_dir() and (path / "logit_rows.jsonl").exists() and path not in dirs:
                dirs.append(path)
    return sorted(dirs, key=infer_alpha)


def row_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    global_idx = row.get("global_idx")
    return (
        str(row.get("example_id")),
        int(global_idx) if global_idx is not None else -1,
        str(row.get("mode") or "free"),
    )


def load_condition_map(path: Path, position_index: int) -> Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]]:
    rows = read_jsonl(path / "logit_rows.jsonl")
    out: Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pos = row.get("position_index")
        if (int(pos) if pos is not None else -1) != int(position_index):
            continue
        out[row_key(row)][str(row.get("condition"))] = row
    return out


def paired_rows_for_dir(path: Path, position_index: int) -> List[Dict[str, Any]]:
    alpha = infer_alpha(path)
    grouped = load_condition_map(path, position_index)
    out: List[Dict[str, Any]] = []
    for key, by_condition in sorted(grouped.items()):
        example_id, global_idx, mode = key
        for pair_name, treated, base, expected_sign in PAIRS:
            a = by_condition.get(treated)
            b = by_condition.get(base)
            if a is None or b is None:
                continue
            item: Dict[str, Any] = {
                "alpha": alpha,
                "alpha_dir": path.name,
                "pair": pair_name,
                "example_id": example_id,
                "global_idx": global_idx,
                "mode": mode,
                "treated_condition": treated,
                "base_condition": base,
                "expected_sign": expected_sign,
                "treated_chosen_token_text": a.get("chosen_token_text"),
                "base_chosen_token_text": b.get("chosen_token_text"),
            }
            for metric in METRICS:
                delta = safe_float(a.get(metric)) - safe_float(b.get(metric))
                item[f"delta_{metric}"] = delta
                item[f"directional_delta_{metric}"] = delta * expected_sign
                item[f"treated_{metric}"] = safe_float(a.get(metric))
                item[f"base_{metric}"] = safe_float(b.get(metric))
            out.append(item)
    return out


def summarize(rows: Sequence[Mapping[str, Any]], metric: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(safe_float(row.get("alpha")), str(row.get("pair")))].append(row)
    out: List[Dict[str, Any]] = []
    for (alpha, pair), group in sorted(grouped.items()):
        vals = finite(row.get(f"directional_delta_{metric}") for row in group)
        raw_vals = finite(row.get(f"delta_{metric}") for row in group)
        pos = sum(1 for x in vals if x > 0)
        neg = sum(1 for x in vals if x < 0)
        nonzero = pos + neg
        out.append(
            {
                "alpha": alpha,
                "pair": pair,
                "n": len(vals),
                "mean_directional_delta": mean(vals),
                "median_directional_delta": median(vals) if vals else float("nan"),
                "sd_directional_delta": sd(vals),
                "q05_directional_delta": quantile(vals, 0.05),
                "q25_directional_delta": quantile(vals, 0.25),
                "q75_directional_delta": quantile(vals, 0.75),
                "q95_directional_delta": quantile(vals, 0.95),
                "positive_rate": pos / nonzero if nonzero else float("nan"),
                "negative_rate": neg / nonzero if nonzero else float("nan"),
                "zero_count": len(vals) - nonzero,
                "one_sided_sign_p": binom_tail_ge(nonzero, pos),
                "mean_raw_delta": mean(raw_vals),
            }
        )
    return out


def monotonic_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("pair")), int(row.get("global_idx") or -1), str(row.get("example_id")))].append(row)
    out: List[Dict[str, Any]] = []
    for (pair, global_idx, example_id), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda r: safe_float(r.get("alpha")))
        vals = [safe_float(r.get(f"directional_delta_{metric}")) for r in ordered]
        alphas = [safe_float(r.get("alpha")) for r in ordered]
        if len(vals) < 2 or any(not math.isfinite(v) for v in vals):
            continue
        monotone_nondec = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
        strictly_pos_all = all(v > 0 for v in vals)
        out.append(
            {
                "pair": pair,
                "example_id": example_id,
                "global_idx": global_idx,
                "n_alpha": len(vals),
                "alphas": ",".join(f"{a:g}" for a in alphas),
                "directional_deltas": ",".join(f"{v:.6g}" for v in vals),
                "monotone_nondec": monotone_nondec,
                "positive_all_alphas": strictly_pos_all,
                "last_minus_first": vals[-1] - vals[0],
            }
        )
    return out


def summarize_monotonic(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("pair"))].append(row)
    out: List[Dict[str, Any]] = []
    for pair, group in sorted(grouped.items()):
        n = len(group)
        mono = sum(1 for row in group if str(row.get("monotone_nondec")).lower() == "true")
        pos_all = sum(1 for row in group if str(row.get("positive_all_alphas")).lower() == "true")
        last = finite(row.get("last_minus_first") for row in group)
        out.append(
            {
                "pair": pair,
                "n_examples": n,
                "monotone_rate": mono / n if n else float("nan"),
                "positive_all_alpha_rate": pos_all / n if n else float("nan"),
                "mean_last_minus_first": mean(last),
                "median_last_minus_first": median(last) if last else float("nan"),
            }
        )
    return out


def write_report(
    path: Path,
    *,
    alpha_dirs: Sequence[Path],
    metric: str,
    summary_rows: Sequence[Mapping[str, Any]],
    monotonic_summary: Sequence[Mapping[str, Any]],
    n_paired: int,
) -> None:
    lines = [
        "# error-ack 方向 p0 alpha sweep 成对分析",
        "",
        "## 实验问题",
        "",
        "检验从自然反思事件中学到的 error-ack hidden 方向，是否能在 forced-box 场景里稳定移动第一个生成位置的“继续反思 vs 停止/收尾”logit 边界。",
        "",
        "## 数据范围",
        "",
        f"- alpha 目录数：{len(alpha_dirs)}",
        f"- 配对记录数：{n_paired}",
        f"- 主指标：{metric}",
        "",
        "## 主要成对结果",
        "",
        "| alpha | 对比 | 题目数 | 平均方向性位移 | 中位数 | 方向一致率 | 符号检验 p |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {alpha:g} | {pair} | {n} | {mean:.4f} | {med:.4f} | {pos:.3f} | {p:.3g} |".format(
                alpha=safe_float(row.get("alpha")),
                pair=row.get("pair"),
                n=int(row.get("n") or 0),
                mean=safe_float(row.get("mean_directional_delta")),
                med=safe_float(row.get("median_directional_delta")),
                pos=safe_float(row.get("positive_rate")),
                p=safe_float(row.get("one_sided_sign_p")),
            )
        )
    lines.extend(
        [
            "",
            "## alpha 单调性",
            "",
            "| 对比 | 题目数 | 单调率 | 所有 alpha 方向正确率 | 最大 alpha 减最小 alpha 平均增量 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in monotonic_summary:
        lines.append(
            "| {pair} | {n} | {mono:.3f} | {pos:.3f} | {gain:.4f} |".format(
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
            "## 当前判断",
            "",
            "这个分析若显示高方向一致率和清晰 alpha 单调性，就支持一个比聚类更强的说法：自然反思前状态里存在可读且可干预的 error-ack 方向，它不是只由 marker 或位置解释的表面差异。",
            "",
            "但这还不是最终论文级结论：它主要证明 logit 边界被移动，并不等同于长轨迹反思行为已经被稳定翻转。下一步必须扩大样本并做短窗口行为级干预，确认这个方向能改变实际生成文本，而不只是改变第一步分数。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alpha_dirs = resolve_alpha_dirs(root, args.alpha_dirs)
    if not alpha_dirs:
        raise RuntimeError(f"No alpha dirs found under {root} with spec {args.alpha_dirs!r}.")
    rows: List[Dict[str, Any]] = []
    for path in alpha_dirs:
        rows.extend(paired_rows_for_dir(path, int(args.position_index)))
    if not rows:
        raise RuntimeError("No paired rows found.")
    summary_rows = summarize(rows, str(args.metric))
    mono = monotonic_rows(rows, str(args.metric))
    mono_summary = summarize_monotonic(mono)
    write_csv(out / "paired_deltas.csv", rows)
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "alpha_monotonic_examples.csv", mono)
    write_csv(out / "alpha_monotonic_summary.csv", mono_summary)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "alpha_dirs": [str(p) for p in alpha_dirs],
                "metric": args.metric,
                "position_index": int(args.position_index),
                "paired_rows": len(rows),
                "summary_rows": list(summary_rows),
                "monotonic_summary": list(mono_summary),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        out / "REPORT.md",
        alpha_dirs=alpha_dirs,
        metric=str(args.metric),
        summary_rows=summary_rows,
        monotonic_summary=mono_summary,
        n_paired=len(rows),
    )
    print(f"[Done] wrote {out}")


if __name__ == "__main__":
    main()
