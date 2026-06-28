#!/usr/bin/env python3
from __future__ import annotations

"""Sweep a list of local attention heads at multiple scales on LoopBench-style datasets.

For each (head, scale) combination the script measures repetition rate on the dataset.
A head is flagged as a *repetition suppression head* when the repetition rate strictly
decreases as scale goes 0 → 1 → 1.5 (ablated → baseline → amplified).

Typical usage (1.7B, 3 GPUs):
    python scripts/run_local_heads_repetition_screen.py \
      --model_name_or_path Qwen/Qwen3-1.7B \
      --heads L0H9,L0H7,L0H5,L0H15,L0H14,L0H1,L0H3,L0H12,L2H9,L1H14,\
L1H3,L1H2,L1H6,L4H8,L0H2,L2H12,L1H7,L3H8,L2H4,L1H1,L2H1,L0H8,\
L24H0,L4H4,L4H1,L5H13,L0H10,L2H13,L4H11 \
      --scales 0,1,1.5 \
      --input_jsonl evaluation/data/loopbench_inspired/test.jsonl \
      --output_dir outputs/loopbench_rep_screen_qwen3_1p7b \
      --parallel_gpu_ids 0,1,2 \
      --max_new_tokens 16384 \
      --no-do_sample --temperature 0.0
"""

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.generation import create_backend
from cot_research.head_intervention import (
    INTERVENTION_REGISTRY,
    MultiLayerHeadIntervention,
    resolve_head_targets,
)
from cot_research.io_utils import load_jsonl, write_csv, write_json
from cot_research.repetition_analysis import (
    LoopBenchThresholds,
    RepetitionThresholds,
    analyze_loopbench_repetition,
    analyze_repetition,
)
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import (
    parse_parallel_gpu_ids,
    seed_everything,
)
from cot_research.schemas import BackendConfig, GenerationConfig


DEFAULT_SCALES = [0.0, 1.0, 1.5]

# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Sweep local attention heads at scales [0, 1, 1.5] on loopbench_inspired "
            "test.jsonl and identify heads with strictly decreasing repetition rates."
        )
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="",
        help=(
            "Comma-separated head labels to screen, e.g. L0H3,L1H2,L2H9. "
            "If omitted, the script will load head labels from --heads_csv."
        ),
    )
    parser.add_argument(
        "--heads_csv",
        type=str,
        default=str(
            root_dir
            / "experiment_results"
            / "experiments"
            / "phase5_head_locality"
            / "head_locality_classification_qwen3_1p7b_20260414_1"
            / "data"
            / "classification"
            / "local_heads.csv"
        ),
        help=(
            "CSV file containing at least a head_label column. Used when --heads is omitted. "
            "Default: 1.7B local_heads.csv from phase5 head locality classification."
        ),
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SCALES),
        help="Comma-separated scale values to test (default: 0,1,1.5).",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(root_dir / "evaluation" / "data" / "loopbench_reconstructed_v2" / "test.jsonl"),
        help="Path to the test JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "loopbench_rep_screen"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_examples", type=int, default=-1, help="-1 = use all.")
    parser.add_argument(
        "--shuffle_examples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomly shuffle examples before slicing to --max_examples (uses --seed).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--parallel_gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU ids; workers are distributed round-robin.",
    )
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for HF generation during intervention sweeps.",
    )
    parser.add_argument(
        "--keep_worker_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print_every", type=int, default=25)
    # Repetition thresholds
    parser.add_argument("--same_token_run_threshold", type=int, default=40)
    parser.add_argument("--tail_repeat_min_repeats", type=int, default=6)
    parser.add_argument("--tail_repeat_max_ngram", type=int, default=8)
    parser.add_argument("--tail_repeat_min_span", type=int, default=24)
    parser.add_argument("--line_repeat_threshold", type=int, default=4)
    parser.add_argument("--line_run_score_multiplier", type=int, default=12)
    parser.add_argument("--word_tail_repeat_min_repeats", type=int, default=5)
    parser.add_argument("--word_tail_repeat_max_ngram", type=int, default=8)
    parser.add_argument("--word_tail_repeat_min_span", type=int, default=24)
    parser.add_argument(
        "--tail_word_requires_hard_signal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min_trigger_count", type=int, default=1)
    parser.add_argument(
        "--analysis_mode",
        type=str,
        choices=["generic", "loopbench"],
        default="loopbench",
        help="Which repetition detector to use for intervention outputs.",
    )
    parser.add_argument("--numerical_loop_min_repeated_span", type=int, default=500)
    parser.add_argument("--statement_loop_min_repeat_count", type=int, default=4)
    parser.add_argument("--numerical_same_digit_run_threshold", type=int, default=500)
    parser.add_argument(
        "--baseline_rows_jsonl",
        type=str,
        default="",
        help="Optional comma-separated rows.jsonl files whose scale=1 outputs will be reused as baseline.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def parse_head_list(text: str) -> List[str]:
    return [tok.strip() for tok in text.split(",") if tok.strip()]


def load_head_list_from_csv(path: Path) -> List[str]:
    rows: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "head_label" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing required column `head_label`.")
        for row in reader:
            head_label = str(row.get("head_label") or "").strip()
            if head_label and head_label not in rows:
                rows.append(head_label)
    return rows


def parse_scale_list(text: str) -> List[float]:
    values: List[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            values.append(float(tok))
    if not values:
        raise ValueError("Scale list is empty.")
    return values


def format_scale_tag(scale: float) -> str:
    text = f"{scale:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def make_thresholds(args: argparse.Namespace) -> RepetitionThresholds:
    return RepetitionThresholds(
        same_token_run_threshold=args.same_token_run_threshold,
        tail_repeat_min_repeats=args.tail_repeat_min_repeats,
        tail_repeat_max_ngram=args.tail_repeat_max_ngram,
        tail_repeat_min_span=args.tail_repeat_min_span,
        line_repeat_threshold=args.line_repeat_threshold,
        line_run_score_multiplier=args.line_run_score_multiplier,
        word_tail_repeat_min_repeats=args.word_tail_repeat_min_repeats,
        word_tail_repeat_max_ngram=args.word_tail_repeat_max_ngram,
        word_tail_repeat_min_span=args.word_tail_repeat_min_span,
        tail_word_requires_hard_signal=args.tail_word_requires_hard_signal,
        min_trigger_count=args.min_trigger_count,
    )


def make_loopbench_thresholds(args: argparse.Namespace) -> LoopBenchThresholds:
    return LoopBenchThresholds(
        numerical_loop_min_repeated_span=args.numerical_loop_min_repeated_span,
        statement_loop_min_repeat_count=args.statement_loop_min_repeat_count,
        numerical_same_digit_run_threshold=args.numerical_same_digit_run_threshold,
    )


def parse_path_list(text: str) -> List[Path]:
    return [Path(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def load_baseline_rows(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    baseline_by_id: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Baseline rows JSONL not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                example_id = str(row.get("example_id") or row.get("id") or "").strip()
                if example_id:
                    baseline_by_id[example_id] = row
    return baseline_by_id


def distribute_heads_round_robin(
    head_labels: List[str], num_workers: int
) -> List[List[str]]:
    """Distribute heads across workers using round-robin assignment."""
    buckets: List[List[str]] = [[] for _ in range(num_workers)]
    for idx, head in enumerate(head_labels):
        buckets[idx % num_workers].append(head)
    return [b for b in buckets if b]


def summarize_scale_rows(
    rows: List[Dict[str, Any]],
    head_label: str,
    scale: float,
    gpu_id: int,
    skipped_count: int,
) -> Dict[str, Any]:
    generated = [int(r["generated_tokens"]) for r in rows]
    rep_hits = sum(1 for r in rows if bool(r["is_repetitive"]))
    hit_max = sum(1 for r in rows if bool(r["hit_max_new_tokens"]))
    count = len(rows)
    total_tokens = sum(generated)
    return {
        "head_label": head_label,
        "scale": scale,
        "gpu_id": gpu_id,
        "example_count": count,
        "skipped_count": skipped_count,
        "rep_rate": round(rep_hits / count, 6) if count else 0.0,
        "rep_count": rep_hits,
        "hit_max_rate": round(hit_max / count, 6) if count else 0.0,
        "mean_generated_tokens": round(total_tokens / count, 6) if count else 0.0,
        "median_generated_tokens": float(median(generated)) if generated else 0.0,
    }


# --------------------------------------------------------------------------- #
# Worker                                                                       #
# --------------------------------------------------------------------------- #

def run_worker(
    worker_id: int,
    gpu_id: int,
    head_labels: List[str],
    scales: List[float],
    rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_rows_path: str,
    worker_summary_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    thresholds = make_thresholds(args)
    loopbench_thresholds = make_loopbench_thresholds(args)
    baseline_scale_1_by_id: Dict[str, Dict[str, Any]] = dict(args_dict.get("baseline_scale_1_by_id") or {})

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
        )
    )
    if not backend.supports_intervention or backend.model is None:
        raise ValueError("HF backend with intervention support required.")

    generation_config = GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        enable_thinking=args.enable_thinking,
        capture_step_scores=False,
    )

    all_result_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    first_layer_path = ""
    prepared_rows: List[Tuple[str, str]] = []

    for idx, row in enumerate(rows):
        example_id = str(row.get("id") or row.get("example_id") or idx)
        try:
            prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
        except Exception as exc:
            skipped.append(
                {
                    "head_label": "__prompt_prepare__",
                    "scale": None,
                    "example_id": example_id,
                    "reason": str(exc),
                }
            )
            continue
        prepared_rows.append((example_id, prompt_prefix))

    if not prepared_rows:
        raise ValueError("No valid rows left after prompt preparation.")
    prompt_prepare_skipped_count = len(skipped)

    # Optimization: scale=1.0 is identity — either reuse provided baseline rows
    # or compute it once and reuse across all heads in this worker.
    baseline_scale_1_rows: Optional[List[Dict[str, Any]]] = None
    baseline_scale_1_skipped: int = 0

    for head_label in head_labels:
        targets, attn_modules, layer_path = resolve_head_targets(backend.model, [head_label])
        if not first_layer_path:
            first_layer_path = layer_path

        for scale in scales:
            scale_tag = format_scale_tag(scale)
            scale_rows: List[Dict[str, Any]] = []
            head_skipped_count = prompt_prepare_skipped_count

            # Reuse scale=1 (identity) baseline — identical output for all heads.
            if scale == 1.0 and baseline_scale_1_rows is not None:
                scale_rows = [
                    {**r, "head_label": head_label}
                    for r in baseline_scale_1_rows
                ]
                head_skipped_count = baseline_scale_1_skipped
                all_result_rows.extend(scale_rows)
                print(
                    f"[Worker {worker_id}|GPU{gpu_id}|{head_label}|scale={scale_tag}] "
                    f"reused baseline ({len(scale_rows)} rows, skipped generation)"
                )
            elif scale == 1.0 and baseline_scale_1_by_id:
                missing_ids: List[str] = []
                for example_id, _prompt_prefix in prepared_rows:
                    baseline_row = baseline_scale_1_by_id.get(example_id)
                    if baseline_row is None:
                        missing_ids.append(example_id)
                        continue
                    row_out = {
                        "head_label": head_label,
                        "scale": scale,
                        "example_id": example_id,
                        "generated_tokens": int(baseline_row.get("generated_tokens") or 0),
                        "is_repetitive": bool(baseline_row.get("is_repetitive")),
                        "hit_max_new_tokens": bool(baseline_row.get("hit_max_new_tokens")),
                    }
                    scale_rows.append(row_out)
                    all_result_rows.append(row_out)
                if missing_ids:
                    raise ValueError(
                        f"Missing {len(missing_ids)} baseline rows for scale=1, "
                        f"examples like: {missing_ids[:5]}"
                    )
                print(
                    f"[Worker {worker_id}|GPU{gpu_id}|{head_label}|scale={scale_tag}] "
                    f"reused provided baseline ({len(scale_rows)} rows)"
                )
            else:
                operations = INTERVENTION_REGISTRY.get_required("scale")(
                    targets, {"scale": scale}
                )
                iterator = tqdm(
                    range(0, len(prepared_rows), max(int(args.batch_size), 1)),
                    desc=f"worker={worker_id} gpu={gpu_id} head={head_label} scale={scale_tag}",
                    dynamic_ncols=True,
                    leave=False,
                )
                for batch_start in iterator:
                    batch_pairs = prepared_rows[batch_start : batch_start + max(int(args.batch_size), 1)]
                    example_ids = [item[0] for item in batch_pairs]
                    prompt_prefixes = [item[1] for item in batch_pairs]

                    seed_everything(args.seed + batch_start)
                    with MultiLayerHeadIntervention(attn_modules, operations):
                        generations = backend.generate_many(prompt_prefixes, generation_config)

                    for example_id, generation in zip(example_ids, generations):
                        if args.analysis_mode == "loopbench":
                            repetition = analyze_loopbench_repetition(
                                generation.continuation,
                                thresholds=loopbench_thresholds,
                            )
                        else:
                            repetition = analyze_repetition(
                                generation.continuation,
                                token_ids=generation.token_ids,
                                existing_repetition=None,
                                thresholds=thresholds,
                            )
                        row_out: Dict[str, Any] = {
                            "head_label": head_label,
                            "scale": scale,
                            "example_id": example_id,
                            "generated_tokens": int(generation.generated_tokens),
                            "is_repetitive": bool(repetition["matched"]),
                            "repetition_triggers": json.dumps(
                                repetition.get("triggers") or [], ensure_ascii=False
                            ),
                            "hit_max_new_tokens": bool(
                                generation.generated_tokens >= args.max_new_tokens
                            ),
                        }
                        scale_rows.append(row_out)
                        all_result_rows.append(row_out)

                    if len(scale_rows) % max(args.print_every, 1) == 0:
                        print(
                            f"[Worker {worker_id}|GPU{gpu_id}|{head_label}|scale={scale_tag}] "
                            f"processed={len(scale_rows)} rep_so_far="
                            f"{sum(1 for r in scale_rows if r['is_repetitive'])}/{len(scale_rows)}"
                        )

                # Cache after first successful scale=1 run — reused for all subsequent heads
                if scale == 1.0:
                    baseline_scale_1_rows = list(scale_rows)
                    baseline_scale_1_skipped = head_skipped_count

            summary = summarize_scale_rows(
                scale_rows, head_label, scale, gpu_id, head_skipped_count
            )
            summary["decoder_layer_path"] = layer_path
            summary_rows.append(summary)
            print(
                f"[Worker {worker_id}|GPU{gpu_id}] done head={head_label} scale={scale_tag} "
                f"rep_rate={summary['rep_rate']:.4f} n={summary['example_count']}"
            )

    write_csv(worker_rows_path, all_result_rows)
    write_json(
        worker_summary_path,
        {"summary_rows": summary_rows, "skipped": skipped},
    )
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "head_labels": head_labels,
        "scales": scales,
        "row_count": len(all_result_rows),
        "skipped_count": len(skipped),
        "worker_rows_path": worker_rows_path,
        "worker_summary_path": worker_summary_path,
        "first_layer_path": first_layer_path,
    }


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #

def build_screening_results(
    head_scale_summary: List[Dict[str, Any]],
    scales: List[float],
    *,
    strictly_decreasing_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    For each head, pivot the scale summaries and check if rep_rate strictly
    decreases as scale increases (0 → 1 → 1.5).
    """
    by_head: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for row in head_scale_summary:
        head = str(row["head_label"])
        scale = float(row["scale"])
        by_head.setdefault(head, {})[scale] = row

    results: List[Dict[str, Any]] = []
    sorted_scales = sorted(scales)

    for head, scale_map in sorted(by_head.items()):
        row_out: Dict[str, Any] = {"head_label": head}
        rep_rates: List[Optional[float]] = []
        for scale in sorted_scales:
            srow = scale_map.get(scale)
            tag = format_scale_tag(scale)
            if srow is not None:
                rep = float(srow["rep_rate"])
                row_out[f"rep_rate_scale_{tag}"] = rep
                row_out[f"n_scale_{tag}"] = int(srow["example_count"])
                row_out[f"hit_max_scale_{tag}"] = float(srow["hit_max_rate"])
                rep_rates.append(rep)
            else:
                row_out[f"rep_rate_scale_{tag}"] = None
                row_out[f"n_scale_{tag}"] = None
                row_out[f"hit_max_scale_{tag}"] = None
                rep_rates.append(None)

        # Strict decrease check: each consecutive pair must strictly decrease
        if None in rep_rates or len(rep_rates) < 2:
            row_out["strictly_decreasing"] = False
            row_out["all_scales_present"] = False
        else:
            row_out["all_scales_present"] = True
            strictly_decreasing = all(
                rep_rates[i] > rep_rates[i + 1] + strictly_decreasing_threshold  # type: ignore[operator]
                for i in range(len(rep_rates) - 1)
            )
            row_out["strictly_decreasing"] = strictly_decreasing

        # Pairwise deltas (positive = decreasing)
        for i in range(len(sorted_scales) - 1):
            s0, s1 = sorted_scales[i], sorted_scales[i + 1]
            tag0, tag1 = format_scale_tag(s0), format_scale_tag(s1)
            r0 = row_out.get(f"rep_rate_scale_{tag0}")
            r1 = row_out.get(f"rep_rate_scale_{tag1}")
            key = f"delta_{tag0}_to_{tag1}"
            if r0 is not None and r1 is not None:
                row_out[key] = round(float(r0) - float(r1), 6)
            else:
                row_out[key] = None

        results.append(row_out)

    return sorted(results, key=lambda r: (not bool(r["strictly_decreasing"]), r["head_label"]))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    if str(args.heads).strip():
        head_labels = parse_head_list(args.heads)
        head_source = "cli"
    else:
        heads_csv_path = Path(args.heads_csv)
        if not heads_csv_path.exists():
            raise FileNotFoundError(f"Head CSV not found: {heads_csv_path}")
        head_labels = load_head_list_from_csv(heads_csv_path)
        head_source = str(heads_csv_path)
    scales = parse_scale_list(args.scales)

    if not head_labels:
        raise ValueError("No head labels resolved from --heads / --heads_csv.")

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    all_rows = load_jsonl(str(input_path))
    if getattr(args, "shuffle_examples", False):
        rng = random.Random(args.seed)
        rng.shuffle(all_rows)
        print(f"[Info] Shuffled examples with seed={args.seed}")
    if args.max_examples > 0:
        all_rows = all_rows[: args.max_examples]
    if not all_rows:
        raise ValueError("No rows loaded from input JSONL.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = output_dir / "run_config.json"
    rows_path = output_dir / "rows.csv"
    head_scale_summary_path = output_dir / "head_scale_summary.csv"
    screening_results_path = output_dir / "screening_results.csv"
    summary_json_path = output_dir / "summary.json"

    available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    if not available_gpu_ids:
        raise ValueError("Provide --parallel_gpu_ids (e.g. 0,1,2).")

    baseline_paths = parse_path_list(args.baseline_rows_jsonl)
    baseline_scale_1_by_id: Dict[str, Dict[str, Any]] = {}
    if baseline_paths:
        baseline_scale_1_by_id = load_baseline_rows(baseline_paths)
        selected_ids = {
            str(row.get("id") or row.get("example_id") or idx)
            for idx, row in enumerate(all_rows)
        }
        baseline_scale_1_by_id = {
            key: value
            for key, value in baseline_scale_1_by_id.items()
            if key in selected_ids
        }
        if len(baseline_scale_1_by_id) != len(selected_ids):
            missing = sorted(selected_ids - set(baseline_scale_1_by_id.keys()))
            raise ValueError(
                f"Baseline rows cover {len(baseline_scale_1_by_id)}/{len(selected_ids)} selected examples. "
                f"Missing examples like: {missing[:5]}"
            )

    num_heads = len(head_labels)
    if args.parallel_workers > 0:
        worker_count = min(args.parallel_workers, len(available_gpu_ids), num_heads)
    else:
        worker_count = min(len(available_gpu_ids), num_heads)
    if worker_count <= 0:
        raise ValueError("No workers available.")

    head_buckets = distribute_heads_round_robin(head_labels, worker_count)
    worker_count = len(head_buckets)
    selected_gpu_ids = available_gpu_ids[:worker_count]

    write_json(
        run_config_path,
        {
            "args": vars(args),
            "head_labels": head_labels,
            "scales": scales,
            "example_count": len(all_rows),
            "selected_gpu_ids": selected_gpu_ids,
            "worker_count": worker_count,
            "worker_head_buckets": head_buckets,
            "baseline_rows_jsonl": [str(path) for path in baseline_paths],
        },
    )

    print(
        f"[Info] Repetition screen: model={args.model_name_or_path}, "
        f"heads={num_heads}, scales={scales}, examples={len(all_rows)}, "
        f"workers={worker_count}, gpus={selected_gpu_ids}, "
        f"max_new_tokens={args.max_new_tokens}, do_sample={args.do_sample}, "
        f"head_source={head_source}"
    )

    worker_rows_paths: List[Path] = []
    worker_summary_paths: List[Path] = []
    worker_returns: List[Dict[str, Any]] = []

    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as pool:
        futures = []
        for worker_id, (head_bucket, gpu_id) in enumerate(
            zip(head_buckets, selected_gpu_ids)
        ):
            bucket_tag = "_".join(h.lower() for h in head_bucket[:3])
            if len(head_bucket) > 3:
                bucket_tag += f"_plus{len(head_bucket) - 3}more"
            wrp = output_dir / f"_worker_{worker_id}_{bucket_tag}_rows.csv"
            wsp = output_dir / f"_worker_{worker_id}_{bucket_tag}_summary.json"
            futures.append(
                pool.submit(
                    run_worker,
                    worker_id,
                    gpu_id,
                    head_bucket,
                    scales,
                    all_rows,
                    {**vars(args), "baseline_scale_1_by_id": baseline_scale_1_by_id},
                    str(wrp),
                    str(wsp),
                )
            )
        for fut in as_completed(futures):
            ret = fut.result()
            worker_returns.append(ret)
            worker_rows_paths.append(Path(ret["worker_rows_path"]))
            worker_summary_paths.append(Path(ret["worker_summary_path"]))
            print(
                f"[Done] worker={ret['worker_id']} gpu={ret['gpu_id']} "
                f"heads={ret['head_labels']} rows={ret['row_count']} "
                f"skipped={ret['skipped_count']}"
            )

    # Merge results
    all_result_rows: List[Dict[str, Any]] = []
    head_scale_summary: List[Dict[str, Any]] = []
    all_skipped: List[Dict[str, Any]] = []
    first_layer_path = ""

    for ret in sorted(worker_returns, key=lambda r: int(r["worker_id"])):
        p = Path(ret["worker_rows_path"])
        if p.exists() and p.stat().st_size > 0:
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_result_rows.append(
                        {
                            "head_label": row["head_label"],
                            "scale": float(row["scale"]),
                            "example_id": row["example_id"],
                            "generated_tokens": int(row["generated_tokens"]),
                            "is_repetitive": row["is_repetitive"].lower() == "true",
                            "hit_max_new_tokens": row["hit_max_new_tokens"].lower() == "true",
                        }
                    )
        sp = Path(ret["worker_summary_path"])
        if sp.exists():
            blob = json.loads(sp.read_text(encoding="utf-8"))
            head_scale_summary.extend(blob.get("summary_rows", []))
            all_skipped.extend(blob.get("skipped", []))
        if not first_layer_path:
            first_layer_path = str(ret.get("first_layer_path") or "")

    # Sort rows: by head, then scale, then example_id
    all_result_rows.sort(
        key=lambda r: (str(r["head_label"]), float(r["scale"]), str(r["example_id"]))
    )
    head_scale_summary.sort(
        key=lambda r: (str(r["head_label"]), float(r["scale"]))
    )

    # Recompute global summaries (aggregate worker shards)
    global_summary: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for srow in head_scale_summary:
        key = (str(srow["head_label"]), float(srow["scale"]))
        if key not in global_summary:
            global_summary[key] = dict(srow)
        else:
            # Merge counts from multiple shards (shouldn't happen but be safe)
            existing = global_summary[key]
            existing["example_count"] = int(existing["example_count"]) + int(srow["example_count"])
            existing["rep_count"] = int(existing["rep_count"]) + int(srow["rep_count"])
            existing["skipped_count"] = int(existing["skipped_count"]) + int(srow["skipped_count"])
            n = existing["example_count"]
            existing["rep_rate"] = round(existing["rep_count"] / n, 6) if n else 0.0

    flat_summary = list(global_summary.values())
    flat_summary.sort(key=lambda r: (str(r["head_label"]), float(r["scale"])))

    screening_results = build_screening_results(flat_summary, scales)

    write_csv(rows_path, all_result_rows)
    write_csv(head_scale_summary_path, flat_summary)
    write_csv(screening_results_path, screening_results)

    strictly_decreasing_heads = [
        r["head_label"] for r in screening_results if r.get("strictly_decreasing")
    ]
    write_json(
        summary_json_path,
        {
            "model_name_or_path": args.model_name_or_path,
            "head_labels_tested": head_labels,
            "scales": scales,
            "example_count": len(all_rows),
            "total_rows": len(all_result_rows),
            "total_skipped": len(all_skipped),
            "selected_gpu_ids": selected_gpu_ids,
            "first_decoder_layer_path": first_layer_path,
            "strictly_decreasing_heads": strictly_decreasing_heads,
            "strictly_decreasing_count": len(strictly_decreasing_heads),
            "head_scale_summary": flat_summary,
            "screening_results": screening_results,
        },
    )

    if not args.keep_worker_outputs:
        for path in worker_rows_paths + worker_summary_paths:
            if path.exists():
                path.unlink()

    print("\n[Done] Repetition screen complete.")
    print(f"- rows:              {rows_path}")
    print(f"- head_scale_summary: {head_scale_summary_path}")
    print(f"- screening_results: {screening_results_path}")
    print(f"- summary_json:      {summary_json_path}")
    print(f"\nStrictly decreasing rep-rate heads ({len(strictly_decreasing_heads)}):")
    for h in strictly_decreasing_heads:
        print(f"  {h}")


if __name__ == "__main__":
    main()
