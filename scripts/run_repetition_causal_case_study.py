#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.cot_accuracy import judge_single_row
from cot_research.datasets import convert_numinamath_row, load_numinamath_dataset
from cot_research.generation import create_backend
from cot_research.head_intervention import resolve_head_targets
from cot_research.io_utils import load_jsonl, write_csv, write_json
from cot_research.ov_circuit_analysis import extract_head_ov_components
from cot_research.repetition_analysis import RepetitionThresholds, analyze_repetition
from cot_research.repetition_causal_analysis import (
    build_primary_patch_variants,
    candidate_stats_from_logits,
    choose_loop_escape,
    compute_direct_write_for_tokens,
    find_first_divergence,
    forward_prefix_with_capture,
    generate_with_head_schedule,
    preferred_primary_comparison,
)
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Single-case causal analysis of L0H3 on repetition onset/suppression."
    )
    parser.add_argument("--output_dir", type=str, default=str(root_dir / "outputs" / "repetition_causal_case_study"))
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset_name", type=str, default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_cache_dir", type=str, default=str(root_dir / "evaluation" / "data" / "temp"))
    parser.add_argument("--dataset_local_files_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source_filter", type=str, default="")
    parser.add_argument(
        "--repeat_examples_jsonl",
        type=str,
        default=str(root_dir / "outputs" / "repetition" / "all_repetition_cases.jsonl"),
        help="Optional prior repetition-case JSONL; used to seed the 5 repetitive examples if available.",
    )
    parser.add_argument("--selection_pool_size", type=int, default=512)
    parser.add_argument("--repeat_case_count", type=int, default=5)
    parser.add_argument("--nonrepeat_case_count", type=int, default=5)
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--scale_values", type=str, default="1.5,2.0")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--timeline_window_before", type=int, default=3)
    parser.add_argument("--timeline_window_after", type=int, default=6)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Use eager attention for stable internal analysis.",
    )
    return parser.parse_args()


def parse_scale_values(text: str) -> List[float]:
    values: List[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(float(chunk))
    if not values:
        raise ValueError("No scale values parsed.")
    return values


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        enable_thinking=args.enable_thinking,
        capture_step_scores=False,
    )


def normalize_numinamath_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    dataset = load_numinamath_dataset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        local_files_only=args.dataset_local_files_only,
        source_filter=args.source_filter,
        shuffle=args.shuffle,
        seed=args.seed,
        start_idx=0,
        sample_stride=1,
        max_examples=args.selection_pool_size,
    )
    rows: List[Dict[str, Any]] = []
    for local_index, raw_row in enumerate(dataset):
        rows.append(
            convert_numinamath_row(
                dict(raw_row),
                local_index,
                dataset_name=args.dataset_name,
                dataset_split=args.dataset_split,
            )
        )
    return rows


def maybe_load_repeat_seed_rows(args: argparse.Namespace, dataset_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    path = Path(args.repeat_examples_jsonl)
    if not path.exists():
        return []
    dataset_by_id = {str(row.get("example_id")): row for row in dataset_rows}
    raw_rows = load_jsonl(str(path))
    seed_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        example_id = str(row.get("example_id") or "")
        if example_id in dataset_by_id:
            seed_rows.append(dict(dataset_by_id[example_id]))
            continue
        question = str(row.get("problem") or row.get("question") or "").strip()
        if not question:
            continue
        seed_rows.append(
            {
                "example_id": example_id or f"repeat_seed:{len(seed_rows)}",
                "id": row.get("id") or row.get("example_id") or len(seed_rows),
                "source": row.get("source"),
                "question": question,
                "problem": question,
                "reference_solution": row.get("reference_solution"),
                "correct_answer": row.get("correct_answer"),
                "gold_answer": row.get("gold_answer"),
                "metadata": dict(row.get("metadata") or {}),
            }
        )
    return seed_rows


def screen_rows_for_selection(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map="auto",
            load_in_half=args.load_in_half,
            use_fast_tokenizer=args.use_fast_tokenizer,
            use_safetensors=args.use_safetensors,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    )
    generation_config = build_generation_config(args)
    repeat_rows: List[Dict[str, Any]] = []
    nonrepeat_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        prompt = backend.build_prompt(str(row["question"]), generation_config)
        generation = backend.generate(prompt, generation_config)
        rep = judge_repetition(generation.continuation, generation.token_ids)
        judged = dict(row)
        judged["baseline_generation"] = {
            "continuation": generation.continuation,
            "generated_tokens": generation.generated_tokens,
            "repetition_detection": rep,
        }
        if bool(rep["matched"]):
            repeat_rows.append(judged)
        else:
            nonrepeat_rows.append(judged)
        if idx % max(args.print_every, 1) == 0:
            print(
                f"[Info] selection_screened={idx} repeat={len(repeat_rows)} nonrepeat={len(nonrepeat_rows)}"
            )
        if len(repeat_rows) >= args.repeat_case_count and len(nonrepeat_rows) >= args.nonrepeat_case_count:
            break
    return repeat_rows, nonrepeat_rows


def judge_repetition(continuation: str, token_ids: Sequence[int]) -> Dict[str, Any]:
    return analyze_repetition(
        continuation,
        token_ids=token_ids,
        thresholds=RepetitionThresholds(),
    )


def make_case_dir(output_dir: Path, example_id: str) -> Path:
    safe = example_id.replace("/", "_").replace(":", "_")
    path = output_dir / "cases" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_candidate_ids(
    *,
    baseline_step: Dict[str, Any],
    other_step: Dict[str, Any],
    loop_token_id: int,
    escape_token_id: int,
) -> List[int]:
    values = [int(loop_token_id), int(escape_token_id)]
    for item in list(baseline_step.get("top_k") or []) + list(other_step.get("top_k") or []):
        values.append(int(item["token_id"]))
    dedup: List[int] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        dedup.append(value)
    return dedup


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def step_map(trajectory: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["step_idx"]): item for item in trajectory.get("step_records") or []}


def choose_primary_pair(
    baseline: Dict[str, Any],
    ablation: Dict[str, Any],
    scales: Sequence[Tuple[str, Dict[str, Any]]],
) -> Tuple[str, Dict[str, Any]]:
    return preferred_primary_comparison(
        baseline=baseline,
        ablation=ablation,
        scale_trajectories=scales,
    )


def compute_effect_metric(
    baseline_stats: Dict[int, Dict[str, Any]],
    ablation_stats: Dict[int, Dict[str, Any]],
    *,
    loop_token_id: int,
    escape_token_id: int,
) -> float:
    def _lookup(stats: Dict[int, Dict[str, Any]], token_id: int) -> float:
        item = stats.get(int(token_id))
        if item is None:
            return 0.0
        return float(item["logit"])

    return (_lookup(baseline_stats, escape_token_id) - _lookup(baseline_stats, loop_token_id)) - (
        _lookup(ablation_stats, escape_token_id) - _lookup(ablation_stats, loop_token_id)
    )


def build_stats_lookup(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(item["token_id"]): dict(item) for item in rows}


def summarize_conclusion(
    *,
    loop_delta: float,
    escape_delta: float,
    baseline_repetitive: bool,
    other_repetitive: bool,
) -> str:
    if loop_delta < min(-0.5, -abs(escape_delta) * 1.2):
        return "更像是直接压低 loop token 的竞争力，从而阻止复读。"
    if escape_delta > max(0.5, abs(loop_delta) * 0.8):
        return "更像是直接抬高 escape token，从而把生成带离复读轨道。"
    if baseline_repetitive != other_repetitive:
        return "更像是通过改变候选 token 的整体竞争格局，间接阻止或诱发复读。"
    return "这条样本里没有出现明确的复读状态翻转，更像是弱影响或分散影响。"


def plot_timeline(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    compare_label: str,
) -> None:
    if not rows:
        return
    if not MATPLOTLIB_AVAILABLE:
        svg_path = path if path.suffix.lower() == ".svg" else path.with_suffix(".svg")
        _write_svg_timeline(svg_path, rows, compare_label=compare_label)
        return
    xs = [int(row["step_idx"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(xs, [float(row["baseline_loop_logit"]) for row in rows], label="baseline loop")
    axes[0].plot(xs, [float(row["compare_loop_logit"]) for row in rows], label=f"{compare_label} loop")
    axes[0].plot(xs, [float(row["baseline_escape_logit"]) for row in rows], label="baseline escape")
    axes[0].plot(xs, [float(row["compare_escape_logit"]) for row in rows], label=f"{compare_label} escape")
    axes[0].set_ylabel("logit")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(xs, [float(row["baseline_loop_prob"]) for row in rows], label="baseline loop")
    axes[1].plot(xs, [float(row["compare_loop_prob"]) for row in rows], label=f"{compare_label} loop")
    axes[1].plot(xs, [float(row["baseline_escape_prob"]) for row in rows], label="baseline escape")
    axes[1].plot(xs, [float(row["compare_escape_prob"]) for row in rows], label=f"{compare_label} escape")
    axes[1].set_ylabel("prob")
    axes[1].legend(loc="best", fontsize=8)

    axes[2].plot(xs, [float(row["baseline_loop_direct"]) for row in rows], label="baseline loop direct")
    axes[2].plot(xs, [float(row["compare_loop_direct"]) for row in rows], label=f"{compare_label} loop direct")
    axes[2].plot(xs, [float(row["baseline_escape_direct"]) for row in rows], label="baseline escape direct")
    axes[2].plot(xs, [float(row["compare_escape_direct"]) for row in rows], label=f"{compare_label} escape direct")
    axes[2].set_ylabel("L0H3 direct")
    axes[2].set_xlabel("generation step")
    axes[2].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_svg_timeline(path: Path, rows: Sequence[Dict[str, Any]], *, compare_label: str) -> None:
    width = 960
    panel_height = 220
    margin_left = 72
    margin_right = 32
    margin_top = 32
    margin_bottom = 44
    panel_gap = 28
    total_height = margin_top + margin_bottom + panel_height * 3 + panel_gap * 2
    plot_width = width - margin_left - margin_right
    path.parent.mkdir(parents=True, exist_ok=True)

    def _panel_bounds(idx: int) -> Tuple[float, float]:
        top = margin_top + idx * (panel_height + panel_gap)
        bottom = top + panel_height
        return float(top), float(bottom)

    def _x(step_idx: int, xs: Sequence[int]) -> float:
        if len(xs) <= 1:
            return float(margin_left + plot_width / 2.0)
        return float(margin_left + (step_idx - xs[0]) / (xs[-1] - xs[0]) * plot_width)

    def _y(value: float, ymin: float, ymax: float, top: float, bottom: float) -> float:
        if ymax <= ymin:
            return (top + bottom) / 2.0
        ratio = (value - ymin) / (ymax - ymin)
        return float(bottom - ratio * (bottom - top))

    def _polyline(values: Sequence[float], xs: Sequence[int], ymin: float, ymax: float, top: float, bottom: float) -> str:
        pts = [f"{_x(step, xs):.2f},{_y(val, ymin, ymax, top, bottom):.2f}" for step, val in zip(xs, values)]
        return " ".join(pts)

    def _panel_series(key_specs: Sequence[Tuple[str, str]]) -> Tuple[float, float, List[Tuple[str, str, List[float]]]]:
        series: List[Tuple[str, str, List[float]]] = []
        values_all: List[float] = []
        for label, key in key_specs:
            values = [float(row[key]) for row in rows]
            values_all.extend(values)
            series.append((label, key, values))
        ymin = min(values_all) if values_all else 0.0
        ymax = max(values_all) if values_all else 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0
        pad = (ymax - ymin) * 0.08
        return ymin - pad, ymax + pad, series

    xs = [int(row["step_idx"]) for row in rows]
    colors = ["#d1495b", "#edae49", "#00798c", "#30638e"]
    panels = [
        (
            "logit",
            [
                ("baseline loop", "baseline_loop_logit"),
                (f"{compare_label} loop", "compare_loop_logit"),
                ("baseline escape", "baseline_escape_logit"),
                (f"{compare_label} escape", "compare_escape_logit"),
            ],
        ),
        (
            "prob",
            [
                ("baseline loop", "baseline_loop_prob"),
                (f"{compare_label} loop", "compare_loop_prob"),
                ("baseline escape", "baseline_escape_prob"),
                (f"{compare_label} escape", "compare_escape_prob"),
            ],
        ),
        (
            "L0H3 direct",
            [
                ("baseline loop", "baseline_loop_direct"),
                (f"{compare_label} loop", "compare_loop_direct"),
                ("baseline escape", "baseline_escape_direct"),
                (f"{compare_label} escape", "compare_escape_direct"),
            ],
        ),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:monospace;font-size:12px;fill:#222} .small{font-size:11px} .axis{stroke:#666;stroke-width:1} .grid{stroke:#ddd;stroke-width:1} .legend-line{stroke-width:3;fill:none}</style>',
        f'<text x="{margin_left}" y="20">Loop vs escape timeline ({compare_label})</text>',
    ]

    for panel_idx, (ylabel, key_specs) in enumerate(panels):
        top, bottom = _panel_bounds(panel_idx)
        ymin, ymax, series = _panel_series(key_specs)
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{bottom}" x2="{width - margin_right}" y2="{bottom}"/>')
        parts.append(f'<line class="axis" x1="{margin_left}" y1="{top}" x2="{margin_left}" y2="{bottom}"/>')
        for frac, label in [(0.0, f"{ymin:.2f}"), (0.5, f"{(ymin+ymax)/2:.2f}"), (1.0, f"{ymax:.2f}")]:
            y = bottom - frac * (bottom - top)
            parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}"/>')
            parts.append(f'<text class="small" x="6" y="{y + 4:.2f}">{label}</text>')
        parts.append(f'<text x="8" y="{top + 14:.2f}">{ylabel}</text>')
        for idx, (label, _key, values) in enumerate(series):
            poly = _polyline(values, xs, ymin, ymax, top, bottom)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2"/>')
            legend_x = width - margin_right - 180
            legend_y = top + 16 + idx * 16
            parts.append(f'<line class="legend-line" x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 18}" y2="{legend_y}" stroke="{colors[idx % len(colors)]}"/>')
            parts.append(f'<text class="small" x="{legend_x + 24}" y="{legend_y + 4}">{label}</text>')
        if panel_idx == len(panels) - 1:
            for step in xs:
                x = _x(step, xs)
                parts.append(f'<text class="small" x="{x - 6:.2f}" y="{bottom + 18:.2f}">{step}</text>')
            parts.append(f'<text x="{margin_left + plot_width / 2 - 40:.2f}" y="{bottom + 34:.2f}">generation step</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def analyze_case(
    row: Dict[str, Any],
    args_dict: Dict[str, Any],
    gpu_id: int,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map={"": gpu_id},
            load_in_half=args.load_in_half,
            use_fast_tokenizer=args.use_fast_tokenizer,
            use_safetensors=args.use_safetensors,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    )
    if backend.model is None or backend.tokenizer is None:
        raise ValueError("Causal analysis requires HF backend internals.")

    generation_config = build_generation_config(args)
    target, attn_modules, _ = resolve_head_targets(backend.model, [args.head_label])
    if len(target) != 1:
        raise ValueError("Expected exactly one head target.")
    target = target[0]
    attn_module = attn_modules[target.layer_idx]
    ov_components = extract_head_ov_components(backend.model, layer_idx=target.layer_idx, head_idx=target.head_idx)
    scale_values = parse_scale_values(args.scale_values)

    question = str(row["question"])
    prompt_prefix = backend.build_prompt(question, generation_config)
    baseline = generate_with_head_schedule(
        backend,
        prompt_prefix,
        generation_config,
        attn_module=attn_module,
        target=target,
        default_scale=1.0,
        top_k=args.top_k,
    )
    ablation = generate_with_head_schedule(
        backend,
        prompt_prefix,
        generation_config,
        attn_module=attn_module,
        target=target,
        default_scale=0.0,
        top_k=args.top_k,
    )
    scale_runs: List[Tuple[str, Dict[str, Any]]] = []
    for scale in scale_values:
        label = f"scale_{scale:g}"
        scale_runs.append(
            (
                label,
                generate_with_head_schedule(
                    backend,
                    prompt_prefix,
                    generation_config,
                    attn_module=attn_module,
                    target=target,
                    default_scale=float(scale),
                    top_k=args.top_k,
                ),
            )
        )

    primary_label, primary = choose_primary_pair(baseline, ablation, scale_runs)
    divergence_step = find_first_divergence(baseline["generated_token_ids"], primary["generated_token_ids"])
    if divergence_step is None:
        divergence_step = min(len(baseline["generated_token_ids"]), len(primary["generated_token_ids"])) - 1
        divergence_step = max(int(divergence_step), 0)

    baseline_map = step_map(baseline)
    primary_map = step_map(primary)
    baseline_step = dict(baseline_map[divergence_step])
    primary_step = dict(primary_map[divergence_step])
    loop_escape = choose_loop_escape(
        baseline_repetitive=bool(dict(baseline["repetition_detection"]).get("matched")),
        other_repetitive=bool(dict(primary["repetition_detection"]).get("matched")),
        baseline_token_id=int(baseline_step["chosen_token_id"]),
        other_token_id=int(primary_step["chosen_token_id"]),
    )
    loop_token_id = int(loop_escape["loop_token_id"])
    escape_token_id = int(loop_escape["escape_token_id"])

    common_generated_prefix = baseline["generated_token_ids"][:divergence_step]
    divergence_prefix_token_ids = list(baseline["prompt_token_ids"]) + [int(x) for x in common_generated_prefix]
    candidate_ids = collect_candidate_ids(
        baseline_step=baseline_step,
        other_step=primary_step,
        loop_token_id=loop_token_id,
        escape_token_id=escape_token_id,
    )

    compare_scale = 0.0 if primary_label == "ablation" else float(primary_label.split("_", 1)[1])
    baseline_logits, baseline_debug = forward_prefix_with_capture(
        backend,
        divergence_prefix_token_ids,
        attn_module=attn_module,
        target=target,
        default_scale=1.0,
    )
    primary_logits, primary_debug = forward_prefix_with_capture(
        backend,
        divergence_prefix_token_ids,
        attn_module=attn_module,
        target=target,
        default_scale=compare_scale if primary_label != "ablation" else 0.0,
    )
    baseline_candidate_stats = candidate_stats_from_logits(baseline_logits, backend.tokenizer, candidate_ids)
    primary_candidate_stats = candidate_stats_from_logits(primary_logits, backend.tokenizer, candidate_ids)
    baseline_candidate_lookup = build_stats_lookup(baseline_candidate_stats)
    primary_candidate_lookup = build_stats_lookup(primary_candidate_stats)
    baseline_direct = compute_direct_write_for_tokens(
        ov_components=ov_components,
        head_vector_after=baseline_debug.get("captured_head_after"),
        token_ids=candidate_ids,
        tokenizer=backend.tokenizer,
    )
    primary_direct = compute_direct_write_for_tokens(
        ov_components=ov_components,
        head_vector_after=primary_debug.get("captured_head_after"),
        token_ids=candidate_ids,
        tokenizer=backend.tokenizer,
    )
    baseline_direct_lookup = {int(item["token_id"]): item for item in baseline_direct}
    primary_direct_lookup = {int(item["token_id"]): item for item in primary_direct}

    forced_escape = generate_with_head_schedule(
        backend,
        backend.decode(divergence_prefix_token_ids),
        generation_config,
        attn_module=attn_module,
        target=target,
        default_scale=1.0,
        forced_token_ids={0: escape_token_id},
        top_k=args.top_k,
    )
    forced_loop = generate_with_head_schedule(
        backend,
        backend.decode(divergence_prefix_token_ids),
        generation_config,
        attn_module=attn_module,
        target=target,
        default_scale=1.0,
        forced_token_ids={0: loop_token_id},
        top_k=args.top_k,
    )

    timed_runs = {
        "ablate_only_divergence": generate_with_head_schedule(
            backend,
            prompt_prefix,
            generation_config,
            attn_module=attn_module,
            target=target,
            default_scale=1.0,
            scale_schedule={int(divergence_step): 0.0},
            top_k=args.top_k,
        ),
        "scale_only_divergence": generate_with_head_schedule(
            backend,
            prompt_prefix,
            generation_config,
            attn_module=attn_module,
            target=target,
            default_scale=1.0,
            scale_schedule={int(divergence_step): compare_scale if primary_label != "ablation" else scale_values[-1]},
            top_k=args.top_k,
        ),
        "ablate_post_divergence_3": generate_with_head_schedule(
            backend,
            prompt_prefix,
            generation_config,
            attn_module=attn_module,
            target=target,
            default_scale=1.0,
            scale_schedule={
                int(divergence_step + 1): 0.0,
                int(divergence_step + 2): 0.0,
                int(divergence_step + 3): 0.0,
            },
            top_k=args.top_k,
        ),
    }

    patch_rows: List[Dict[str, Any]] = []
    for variant in build_primary_patch_variants(divergence_prefix_token_ids):
        patched_prefix_ids = [int(x) for x in variant["token_ids"]]
        base_logits, _ = forward_prefix_with_capture(
            backend,
            patched_prefix_ids,
            attn_module=attn_module,
            target=target,
            default_scale=1.0,
        )
        abl_logits, _ = forward_prefix_with_capture(
            backend,
            patched_prefix_ids,
            attn_module=attn_module,
            target=target,
            default_scale=0.0,
        )
        base_stats = build_stats_lookup(candidate_stats_from_logits(base_logits, backend.tokenizer, [loop_token_id, escape_token_id]))
        abl_stats = build_stats_lookup(candidate_stats_from_logits(abl_logits, backend.tokenizer, [loop_token_id, escape_token_id]))
        patch_rows.append(
            {
                "patch_label": str(variant["label"]),
                "loop_token_id": loop_token_id,
                "escape_token_id": escape_token_id,
                "baseline_loop_logit": float(base_stats[loop_token_id]["logit"]),
                "baseline_escape_logit": float(base_stats[escape_token_id]["logit"]),
                "ablation_loop_logit": float(abl_stats[loop_token_id]["logit"]),
                "ablation_escape_logit": float(abl_stats[escape_token_id]["logit"]),
                "l0h3_effect_metric": float(
                    compute_effect_metric(base_stats, abl_stats, loop_token_id=loop_token_id, escape_token_id=escape_token_id)
                ),
            }
        )

    window_start = max(0, int(divergence_step) - int(args.timeline_window_before))
    window_end = int(divergence_step) + int(args.timeline_window_after)
    timeline_rows: List[Dict[str, Any]] = []
    for step_idx in range(window_start, min(window_end + 1, len(baseline["generated_token_ids"]), len(primary["generated_token_ids"]))):
        base_prefix = list(baseline["prompt_token_ids"]) + [int(x) for x in baseline["generated_token_ids"][:step_idx]]
        compare_prefix = list(primary["prompt_token_ids"]) + [int(x) for x in primary["generated_token_ids"][:step_idx]]
        base_logits, base_debug = forward_prefix_with_capture(
            backend,
            base_prefix,
            attn_module=attn_module,
            target=target,
            default_scale=1.0,
        )
        compare_logits, compare_debug = forward_prefix_with_capture(
            backend,
            compare_prefix,
            attn_module=attn_module,
            target=target,
            default_scale=compare_scale if primary_label != "ablation" else 0.0,
        )
        base_stats = build_stats_lookup(candidate_stats_from_logits(base_logits, backend.tokenizer, [loop_token_id, escape_token_id]))
        compare_stats = build_stats_lookup(candidate_stats_from_logits(compare_logits, backend.tokenizer, [loop_token_id, escape_token_id]))
        base_direct = {
            int(item["token_id"]): item
            for item in compute_direct_write_for_tokens(
                ov_components=ov_components,
                head_vector_after=base_debug.get("captured_head_after"),
                token_ids=[loop_token_id, escape_token_id],
                tokenizer=backend.tokenizer,
            )
        }
        compare_direct = {
            int(item["token_id"]): item
            for item in compute_direct_write_for_tokens(
                ov_components=ov_components,
                head_vector_after=compare_debug.get("captured_head_after"),
                token_ids=[loop_token_id, escape_token_id],
                tokenizer=backend.tokenizer,
            )
        }
        timeline_rows.append(
            {
                "step_idx": int(step_idx),
                "baseline_actual_token_id": int(baseline["generated_token_ids"][step_idx]),
                "baseline_actual_token_text": token_text(backend.tokenizer, int(baseline["generated_token_ids"][step_idx])),
                "compare_actual_token_id": int(primary["generated_token_ids"][step_idx]),
                "compare_actual_token_text": token_text(backend.tokenizer, int(primary["generated_token_ids"][step_idx])),
                "baseline_loop_logit": float(base_stats[loop_token_id]["logit"]),
                "baseline_loop_prob": float(base_stats[loop_token_id]["prob"]),
                "baseline_escape_logit": float(base_stats[escape_token_id]["logit"]),
                "baseline_escape_prob": float(base_stats[escape_token_id]["prob"]),
                "compare_loop_logit": float(compare_stats[loop_token_id]["logit"]),
                "compare_loop_prob": float(compare_stats[loop_token_id]["prob"]),
                "compare_escape_logit": float(compare_stats[escape_token_id]["logit"]),
                "compare_escape_prob": float(compare_stats[escape_token_id]["prob"]),
                "baseline_loop_direct": float(base_direct.get(loop_token_id, {}).get("direct_logit_contribution", 0.0)),
                "baseline_escape_direct": float(base_direct.get(escape_token_id, {}).get("direct_logit_contribution", 0.0)),
                "compare_loop_direct": float(compare_direct.get(loop_token_id, {}).get("direct_logit_contribution", 0.0)),
                "compare_escape_direct": float(compare_direct.get(escape_token_id, {}).get("direct_logit_contribution", 0.0)),
            }
        )

    step_table_rows: List[Dict[str, Any]] = []
    max_steps = min(len(baseline["step_records"]), len(primary["step_records"]))
    for step_idx in range(max_steps):
        base_step = dict(baseline["step_records"][step_idx])
        cmp_step = dict(primary["step_records"][step_idx])
        base_direct_all = {
            int(item["token_id"]): item
            for item in compute_direct_write_for_tokens(
                ov_components=ov_components,
                head_vector_after=base_step.get("captured_head_after"),
                token_ids=[int(item["token_id"]) for item in base_step.get("top_k") or []] + [loop_token_id, escape_token_id],
                tokenizer=backend.tokenizer,
            )
        }
        cmp_direct_all = {
            int(item["token_id"]): item
            for item in compute_direct_write_for_tokens(
                ov_components=ov_components,
                head_vector_after=cmp_step.get("captured_head_after"),
                token_ids=[int(item["token_id"]) for item in cmp_step.get("top_k") or []] + [loop_token_id, escape_token_id],
                tokenizer=backend.tokenizer,
            )
        }
        step_table_rows.append(
            {
                "step_idx": int(step_idx),
                "baseline_actual_token": str(base_step["chosen_token_text"]),
                "compare_actual_token": str(cmp_step["chosen_token_text"]),
                "baseline_top_k": json.dumps(base_step.get("top_k") or [], ensure_ascii=False),
                "compare_top_k": json.dumps(cmp_step.get("top_k") or [], ensure_ascii=False),
                "loop_token_id": int(loop_token_id),
                "loop_token_text": token_text(backend.tokenizer, loop_token_id),
                "escape_token_id": int(escape_token_id),
                "escape_token_text": token_text(backend.tokenizer, escape_token_id),
                "baseline_loop_direct": float(base_direct_all.get(loop_token_id, {}).get("direct_logit_contribution", 0.0)),
                "baseline_escape_direct": float(base_direct_all.get(escape_token_id, {}).get("direct_logit_contribution", 0.0)),
                "compare_loop_direct": float(cmp_direct_all.get(loop_token_id, {}).get("direct_logit_contribution", 0.0)),
                "compare_escape_direct": float(cmp_direct_all.get(escape_token_id, {}).get("direct_logit_contribution", 0.0)),
            }
        )

    loop_delta = float(baseline_direct_lookup.get(loop_token_id, {}).get("direct_logit_contribution", 0.0)) - float(
        primary_direct_lookup.get(loop_token_id, {}).get("direct_logit_contribution", 0.0)
    )
    escape_delta = float(baseline_direct_lookup.get(escape_token_id, {}).get("direct_logit_contribution", 0.0)) - float(
        primary_direct_lookup.get(escape_token_id, {}).get("direct_logit_contribution", 0.0)
    )

    return {
        "example_id": str(row.get("example_id")),
        "source": row.get("source"),
        "question": question,
        "gold": judge_single_row(row),
        "baseline": baseline,
        "ablation": ablation,
        "scales": {label: traj for label, traj in scale_runs},
        "primary_compare_label": primary_label,
        "primary_compare": primary,
        "divergence_step": int(divergence_step),
        "loop_escape": {
            **loop_escape,
            "loop_token_text": token_text(backend.tokenizer, loop_token_id),
            "escape_token_text": token_text(backend.tokenizer, escape_token_id),
        },
        "divergence_candidate_stats": {
            "baseline": baseline_candidate_stats,
            primary_label: primary_candidate_stats,
            "baseline_direct": baseline_direct,
            f"{primary_label}_direct": primary_direct,
            "baseline_vs_primary_deltas": [
                {
                    "token_id": int(item["token_id"]),
                    "token_text": item["token_text"],
                    "logit_delta": float(item["logit"] - primary_candidate_lookup[int(item["token_id"])]["logit"]),
                    "prob_delta": float(item["prob"] - primary_candidate_lookup[int(item["token_id"])]["prob"]),
                }
                for item in baseline_candidate_stats
                if int(item["token_id"]) in primary_candidate_lookup
            ],
        },
        "forced_continuations": {
            "force_escape_identity": forced_escape,
            "force_loop_identity": forced_loop,
        },
        "timed_interventions": timed_runs,
        "local_prefix_patches": patch_rows,
        "timeline_rows": timeline_rows,
        "step_table_rows": step_table_rows,
        "case_conclusion": summarize_conclusion(
            loop_delta=loop_delta,
            escape_delta=escape_delta,
            baseline_repetitive=bool(dict(baseline["repetition_detection"]).get("matched")),
            other_repetitive=bool(dict(primary["repetition_detection"]).get("matched")),
        ),
    }


def write_case_outputs(case_dir: Path, case_result: Dict[str, Any]) -> None:
    write_json(case_dir / "case_result.json", case_result)
    write_csv(case_dir / "step_table.csv", case_result["step_table_rows"])
    write_csv(case_dir / "timeline.csv", case_result["timeline_rows"])
    write_csv(case_dir / "local_prefix_patches.csv", case_result["local_prefix_patches"])
    plot_timeline(case_dir / "timeline.svg", case_result["timeline_rows"], compare_label=str(case_result["primary_compare_label"]))
    with open(case_dir / "conclusion.md", "w", encoding="utf-8") as f:
        f.write(f"# {case_result['example_id']}\n\n")
        f.write(f"- primary_compare: `{case_result['primary_compare_label']}`\n")
        f.write(f"- divergence_step: `{case_result['divergence_step']}`\n")
        f.write(
            f"- baseline_repetitive: `{bool(dict(case_result['baseline']['repetition_detection']).get('matched'))}`\n"
        )
        f.write(
            f"- primary_repetitive: `{bool(dict(case_result['primary_compare']['repetition_detection']).get('matched'))}`\n"
        )
        f.write(f"- conclusion: {case_result['case_conclusion']}\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", vars(args))

    dataset_rows = normalize_numinamath_rows(args)
    repeat_seed_rows = maybe_load_repeat_seed_rows(args, dataset_rows)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(repeat_seed_rows)
        rng.shuffle(dataset_rows)

    selected_repeat: List[Dict[str, Any]] = []
    selected_ids = set()
    for row in repeat_seed_rows:
        row_id = str(row.get("example_id"))
        if row_id in selected_ids:
            continue
        selected_repeat.append(dict(row))
        selected_ids.add(row_id)
        if len(selected_repeat) >= args.repeat_case_count:
            break

    scan_pool = [row for row in dataset_rows if str(row.get("example_id")) not in selected_ids]
    screened_repeat, screened_nonrepeat = screen_rows_for_selection(scan_pool, args)
    for row in screened_repeat:
        row_id = str(row.get("example_id"))
        if row_id in selected_ids:
            continue
        selected_repeat.append(dict(row))
        selected_ids.add(row_id)
        if len(selected_repeat) >= args.repeat_case_count:
            break
    selected_nonrepeat: List[Dict[str, Any]] = []
    for row in screened_nonrepeat:
        row_id = str(row.get("example_id"))
        if row_id in selected_ids:
            continue
        selected_nonrepeat.append(dict(row))
        selected_ids.add(row_id)
        if len(selected_nonrepeat) >= args.nonrepeat_case_count:
            break

    if len(selected_repeat) < args.repeat_case_count or len(selected_nonrepeat) < args.nonrepeat_case_count:
        raise ValueError(
            f"Not enough selected cases: repeat={len(selected_repeat)}/{args.repeat_case_count}, "
            f"nonrepeat={len(selected_nonrepeat)}/{args.nonrepeat_case_count}. "
            "Provide a repetition-case JSONL or enlarge the selection pool."
        )

    selected_cases = selected_repeat[: args.repeat_case_count] + selected_nonrepeat[: args.nonrepeat_case_count]
    write_json(output_dir / "selected_cases.json", {"repeat": selected_repeat[: args.repeat_case_count], "nonrepeat": selected_nonrepeat[: args.nonrepeat_case_count]})

    gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    if not gpu_ids:
        raise ValueError("This experiment requires explicit or visible CUDA GPUs.")
    if args.parallel_workers > 0:
        worker_count = min(int(args.parallel_workers), len(gpu_ids), len(selected_cases))
    else:
        worker_count = min(len(gpu_ids), len(selected_cases))
    selected_gpu_ids = gpu_ids[:worker_count]

    results: List[Dict[str, Any]] = []
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as pool:
        futures = []
        for idx, row in enumerate(selected_cases):
            gpu_id = selected_gpu_ids[idx % worker_count]
            futures.append(pool.submit(analyze_case, row, vars(args), gpu_id))
        for future in as_completed(futures):
            case_result = future.result()
            results.append(case_result)
            case_dir = make_case_dir(output_dir, str(case_result["example_id"]))
            write_case_outputs(case_dir, case_result)
            print(
                f"[Info] finished_case={case_result['example_id']} primary_compare={case_result['primary_compare_label']} "
                f"baseline_repetitive={bool(dict(case_result['baseline']['repetition_detection']).get('matched'))} "
                f"primary_repetitive={bool(dict(case_result['primary_compare']['repetition_detection']).get('matched'))}"
            )

    results = sorted(results, key=lambda item: str(item["example_id"]))
    summary_rows: List[Dict[str, Any]] = []
    for item in results:
        baseline_rep = bool(dict(item["baseline"]["repetition_detection"]).get("matched"))
        primary_rep = bool(dict(item["primary_compare"]["repetition_detection"]).get("matched"))
        forced_escape_rep = bool(dict(item["forced_continuations"]["force_escape_identity"]["repetition_detection"]).get("matched"))
        forced_loop_rep = bool(dict(item["forced_continuations"]["force_loop_identity"]["repetition_detection"]).get("matched"))
        summary_rows.append(
            {
                "example_id": item["example_id"],
                "source": item.get("source"),
                "primary_compare": item["primary_compare_label"],
                "divergence_step": item["divergence_step"],
                "baseline_repetitive": baseline_rep,
                "primary_repetitive": primary_rep,
                "forced_escape_repetitive": forced_escape_rep,
                "forced_loop_repetitive": forced_loop_rep,
                "case_conclusion": item["case_conclusion"],
            }
        )
    write_csv(output_dir / "summary.csv", summary_rows)
    write_json(output_dir / "summary.json", {"cases": summary_rows})
    print("[Done] repetition causal case study finished:")
    print(f"- output_dir: {output_dir}")
    print(f"- analyzed_cases: {len(results)}")
    print(f"- summary_csv: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
