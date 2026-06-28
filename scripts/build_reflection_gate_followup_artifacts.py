#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiment_results" / "experiments" / "reflection_gate_followup_20260521"
DATA = OUT / "data"
FIG = OUT / "figures"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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


def condition_alpha(condition: str) -> Optional[float]:
    if condition == "baseline":
        return 0.0
    marker = "alpha"
    if marker not in condition:
        return None
    try:
        return float(condition.split(marker, 1)[1])
    except ValueError:
        return None


def condition_family(condition: str) -> str:
    if condition == "baseline":
        return "baseline"
    if "gate_orth_logit" in condition:
        return "gate_orth_logit"
    if "logit" in condition:
        return "logit"
    if "random" in condition:
        return "random"
    if "gate_off" in condition:
        return "gate"
    return condition.split("_alpha", 1)[0]


def model_tag_from_path(path: Path) -> str:
    name = path.parts[-3]
    if name.startswith("1p7b_"):
        return "Qwen3-1.7B"
    if name.startswith("4b_"):
        return "Qwen3-4B"
    if name.startswith("8b_"):
        return "Qwen3-8B"
    return name


def dataset_from_path(path: Path) -> str:
    name = path.parts[-3]
    if "aime24" in name:
        return "AIME24"
    if "math45" in name:
        return "MATH45 first100"
    return name


def collect_vllm_summaries() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(
        (ROOT / "outputs" / "stateful_tamper_attention_20260510" / "reflection_gate_vllm_16384_20260520").glob(
            "*_baseline_gate_alpha_sweep_16384/merged/summary.json"
        )
    ):
        model = model_tag_from_path(path)
        dataset = dataset_from_path(path)
        for item in read_json(path):
            condition = str(item["condition_key"])
            rows.append(
                {
                    "source": str(path.relative_to(ROOT)),
                    "model": model,
                    "dataset": dataset,
                    "condition_key": condition,
                    "family": condition_family(condition),
                    "alpha": condition_alpha(condition),
                    "n": item.get("n"),
                    "correct": item.get("correct"),
                    "accuracy": item.get("accuracy"),
                    "mean_generated_tokens": item.get("mean_generated_tokens"),
                    "median_generated_tokens": item.get("median_generated_tokens"),
                    "mean_reflection_keyword_count": item.get("mean_reflection_keyword_count"),
                    "length_cap": item.get("length_cap"),
                    "length_cap_rate": item.get("length_cap_rate"),
                    "outcomes": item.get("outcomes"),
                }
            )
    return rows


def collect_direction_summaries() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = ROOT / "outputs" / "stateful_tamper_attention_20260510" / "reflection_gate_vllm_16384_20260520" / "direction_build"
    for path in sorted(root.glob("*/summary.json")):
        tag = path.parent.name
        if tag.startswith("1p7b_"):
            model = "Qwen3-1.7B"
        elif tag.startswith("4b_"):
            model = "Qwen3-4B"
        elif tag.startswith("8b_"):
            model = "Qwen3-8B"
        else:
            model = tag
        payload = read_json(path)
        for item in payload.get("directions", []):
            rows.append(
                {
                    "source": str(path.relative_to(ROOT)),
                    "model": model,
                    "direction_type": item.get("direction_type"),
                    "layer_idx": item.get("layer_idx"),
                    "site": item.get("site"),
                    "scale": item.get("scale"),
                    "n_pairs": item.get("n_pairs"),
                    "direction_baseline_condition": item.get("direction_baseline_condition"),
                    "direction_contrast": item.get("direction_contrast"),
                    "gate_logit_cosine": item.get("gate_logit_cosine"),
                }
            )
    return rows


def baseline_map(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(r["model"], r["dataset"]): r for r in rows if r["condition_key"] == "baseline"}


def compute_failure_rows(vllm_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    selected = {
        ("Qwen3-4B", "MATH45 first100"): ["gate_off_post_attn_alpha0.25", "gate_off_post_attn_alpha1"],
        ("Qwen3-8B", "MATH45 first100"): ["gate_off_post_attn_alpha0.5", "gate_off_post_attn_alpha1"],
        ("Qwen3-1.7B", "MATH45 first100"): ["gate_off_block_output_alpha0.25", "gate_off_block_output_alpha1"],
        ("Qwen3-4B", "AIME24"): ["gate_off_post_attn_alpha0.25", "gate_off_post_attn_alpha1"],
        ("Qwen3-8B", "AIME24"): ["gate_off_post_attn_alpha0.5", "gate_off_post_attn_alpha1"],
        ("Qwen3-1.7B", "AIME24"): ["gate_off_block_output_alpha0.25", "gate_off_block_output_alpha1"],
    }
    roots = {
        ("Qwen3-4B", "MATH45 first100"): ROOT
        / "outputs/stateful_tamper_attention_20260510/reflection_gate_vllm_16384_20260520/4b_math45_first100_baseline_gate_alpha_sweep_16384/merged/rows.jsonl",
        ("Qwen3-8B", "MATH45 first100"): ROOT
        / "outputs/stateful_tamper_attention_20260510/reflection_gate_vllm_16384_20260520/8b_math45_first100_baseline_gate_alpha_sweep_16384/merged/rows.jsonl",
        ("Qwen3-1.7B", "MATH45 first100"): ROOT
        / "outputs/stateful_tamper_attention_20260510/reflection_gate_vllm_16384_20260520/1p7b_math45_first100_baseline_gate_alpha_sweep_16384/merged/rows.jsonl",
        ("Qwen3-4B", "AIME24"): ROOT
        / "outputs/stateful_tamper_attention_20260510/reflection_gate_vllm_16384_20260520/4b_aime24_baseline_gate_alpha_sweep_16384/merged/rows.jsonl",
        ("Qwen3-8B", "AIME24"): ROOT
        / "outputs/stateful_tamper_attention_20260510/reflection_gate_vllm_16384_20260520/8b_aime24_baseline_gate_alpha_sweep_16384/merged/rows.jsonl",
        ("Qwen3-1.7B", "AIME24"): ROOT
        / "outputs/stateful_tamper_attention_20260510/reflection_gate_vllm_16384_20260520/1p7b_aime24_baseline_gate_alpha_sweep_16384/merged/rows.jsonl",
    }
    for key, conditions in selected.items():
        path = roots[key]
        if not path.exists():
            continue
        by_example: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for row in read_jsonl(path):
            cond = "baseline" if row.get("condition") == "baseline" else f"{row.get('condition')}_{row.get('site')}_alpha{float(row.get('alpha') or 0.0):g}"
            by_example[str(row.get("example_id"))][cond] = row
        for cond in conditions:
            pairs = [(eid, p["baseline"], p[cond]) for eid, p in by_example.items() if "baseline" in p and cond in p]
            if not pairs:
                continue
            cap = 16384
            lost = [p for p in pairs if p[1].get("benchmark_correct") and not p[2].get("benchmark_correct")]
            rescued = [p for p in pairs if not p[1].get("benchmark_correct") and p[2].get("benchmark_correct")]
            cap_rescue = [
                p
                for p in pairs
                if int(p[1].get("generated_tokens") or 0) >= cap and int(p[2].get("generated_tokens") or 0) < cap
            ]
            new_cap = [
                p
                for p in pairs
                if int(p[1].get("generated_tokens") or 0) < cap and int(p[2].get("generated_tokens") or 0) >= cap
            ]
            new_no_box = [
                p
                for p in pairs
                if p[1].get("continuation_outcome") != "no_boxed_answer"
                and p[2].get("continuation_outcome") == "no_boxed_answer"
            ]
            early_wrong = [
                p
                for p in lost
                if int(p[2].get("generated_tokens") or 0) < 0.6 * max(1, int(p[1].get("generated_tokens") or 0))
            ]
            out.append(
                {
                    "model": key[0],
                    "dataset": key[1],
                    "condition_key": cond,
                    "n_pairs": len(pairs),
                    "lost_correctness": len(lost),
                    "rescued_correctness": len(rescued),
                    "cap_rescues": len(cap_rescue),
                    "new_caps": len(new_cap),
                    "new_no_box": len(new_no_box),
                    "early_wrong_lost": len(early_wrong),
                    "lost_examples": ";".join(x[0] for x in lost[:8]),
                    "rescued_examples": ";".join(x[0] for x in rescued[:8]),
                    "new_no_box_examples": ";".join(x[0] for x in new_no_box[:8]),
                }
            )
    return out


def parse_projection_remote_if_present() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base = ROOT / "outputs/stateful_tamper_attention_20260510/reflection_gate_projection_ablation_20260521"
    if not base.exists():
        return [], []
    summaries: List[Dict[str, Any]] = []
    baselines: List[Dict[str, Any]] = []
    for root in sorted(path for path in base.iterdir() if path.is_dir()):
        summary_path = root / "projection_summary.csv"
        baseline_path = root / "baseline_summary.csv"
        if summary_path.exists():
            for row in read_csv(summary_path):
                row["run_id"] = root.name
                row["source"] = str(summary_path.relative_to(ROOT))
                summaries.append(row)
        if baseline_path.exists():
            for row in read_csv(baseline_path):
                row["run_id"] = root.name
                row["source"] = str(baseline_path.relative_to(ROOT))
                baselines.append(row)
    return baselines, summaries


def svg_bar_chart(
    path: Path,
    *,
    title: str,
    labels: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float], str]],
    ylabel: str,
    width: int = 980,
    height: int = 420,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    margin_l, margin_r, margin_t, margin_b = 70, 30, 52, 100
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_val = max([0.0] + [float(v) for _, vals, _ in series for v in vals])
    max_val = max_val * 1.15 if max_val > 0 else 1.0
    group_w = plot_w / max(1, len(labels))
    bar_w = group_w / (len(series) + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="18" y="{margin_t + plot_h/2}" transform="rotate(-90 18 {margin_t + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="12">{ylabel}</text>',
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" stroke="#333"/>',
    ]
    for i in range(5):
        val = max_val * i / 4
        y = margin_t + plot_h - (val / max_val) * plot_h
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_l - 8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{val:.0f}</text>')
    for i, label in enumerate(labels):
        x0 = margin_l + i * group_w
        parts.append(
            f'<text x="{x0 + group_w/2:.1f}" y="{height - 56}" text-anchor="end" transform="rotate(-35 {x0 + group_w/2:.1f} {height - 56})" font-family="Arial" font-size="11">{label}</text>'
        )
        for j, (name, vals, color) in enumerate(series):
            val = float(vals[i])
            h = (val / max_val) * plot_h
            x = x0 + (j + 0.5) * bar_w
            y = margin_t + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.82:.1f}" height="{h:.1f}" fill="{color}"/>')
    legend_x = margin_l
    for name, _vals, color in series:
        parts.append(f'<rect x="{legend_x}" y="{height - 28}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{height - 18}" font-family="Arial" font-size="12">{name}</text>')
        legend_x += 170
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_figures(vllm: Sequence[Dict[str, Any]], paired: Sequence[Dict[str, str]], projection: Sequence[Dict[str, str]]) -> None:
    math_rows = [r for r in vllm if r["dataset"] == "MATH45 first100" and r["family"] in {"baseline", "gate"}]
    selected = []
    for model, alpha in [("Qwen3-1.7B", 0.25), ("Qwen3-4B", 0.25), ("Qwen3-8B", 0.5)]:
        base = next(r for r in math_rows if r["model"] == model and r["family"] == "baseline")
        gate = next(r for r in math_rows if r["model"] == model and abs(float(r["alpha"]) - alpha) < 1e-9)
        selected.append((model.replace("Qwen3-", ""), base, gate))
    svg_bar_chart(
        FIG / "vllm_math45_operating_points.svg",
        title="vLLM 16k MATH45 Gate-Off Operating Points",
        labels=[x[0] for x in selected],
        series=[
            ("baseline tokens", [x[1]["mean_generated_tokens"] for x in selected], "#7f8c8d"),
            ("gate tokens", [x[2]["mean_generated_tokens"] for x in selected], "#2f80ed"),
            ("baseline refkw x100", [100 * float(x[1]["mean_reflection_keyword_count"]) for x in selected], "#b0b7c3"),
            ("gate refkw x100", [100 * float(x[2]["mean_reflection_keyword_count"]) for x in selected], "#27ae60"),
        ],
        ylabel="tokens / refkw x100",
    )
    paired_sel = [r for r in paired if r.get("comparison", "").startswith(("4B_gateorth", "8B_gateorth"))]
    labels = [r["comparison"].replace("_gateorth_vs_logit_", "\n").replace("alpha", "a") for r in paired_sel]
    svg_bar_chart(
        FIG / "paired_gateorth_vs_logit.svg",
        title="Gate-Orth-Logit vs Direct Logit (paired all100)",
        labels=labels,
        series=[
            ("token reduction", [abs(float(r["mean_delta_tokens_a_minus_b"])) for r in paired_sel], "#8e44ad"),
            ("refkw reduction x100", [100 * abs(float(r["mean_delta_refkw_a_minus_b"])) for r in paired_sel], "#16a085"),
        ],
        ylabel="absolute reduction",
        width=1100,
    )
    if projection:
        labels = [r["projection_type"] for r in projection]
        svg_bar_chart(
            FIG / "projection_ablation_forced_tamper.svg",
            title="Projection Ablation on Held-Out Forced-Tamper States",
            labels=labels,
            series=[
                ("has reflection %", [100 * float(r["has_reflection_rate"]) for r in projection], "#e67e22"),
                ("reflect-vs-stop", [max(0.0, float(r["mean_reflect_vs_stop"])) for r in projection], "#34495e"),
                ("refkw x10", [10 * float(r["mean_reflection_keyword_count"]) for r in projection], "#27ae60"),
            ],
            ylabel="rate / margin / count",
            width=1100,
        )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    vllm = collect_vllm_summaries()
    directions = collect_direction_summaries()
    paired = read_csv(ROOT / "experiment_results/reports/reflection_gate_supplements_20260518/paired_stats_summary.csv")
    failure_rows = compute_failure_rows(vllm)
    projection_baseline, projection_summary = parse_projection_remote_if_present()
    write_csv(DATA / "vllm_16384_gate_sweeps.csv", vllm)
    write_csv(DATA / "direction_build_summary.csv", directions)
    write_csv(DATA / "paired_gateorth_vs_logit_summary.csv", paired)
    write_csv(DATA / "vllm_failure_case_counts.csv", failure_rows)
    write_csv(DATA / "projection_ablation_baseline_summary.csv", projection_baseline)
    write_csv(DATA / "projection_ablation_summary.csv", projection_summary)
    make_figures(vllm, paired, projection_summary)
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "generated": "2026-05-21",
                "data_files": sorted(str(p.relative_to(OUT)) for p in DATA.glob("*")),
                "figure_files": sorted(str(p.relative_to(OUT)) for p in FIG.glob("*")),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Done] wrote artifacts under {OUT}")


if __name__ == "__main__":
    main()
