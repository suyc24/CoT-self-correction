#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import extract_last_boxed
from cot_research.generation import create_backend, generate_with_next_token_attention_trace
from cot_research.head_attention_pattern import (
    aggregate_distance_profile,
    aggregate_pattern_rows,
    compute_all_head_pattern_metrics,
    parse_int_list,
)
from cot_research.io_utils import dump_jsonl, load_jsonl, truncate_text, write_csv, write_json
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_input = root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"
    default_output = root_dir / "outputs" / "next_token_current_probe" / "qwen3_4b_current_top10"

    parser = argparse.ArgumentParser(
        description="Rank heads by attention to the current last token at the next-token prediction step."
    )
    parser.add_argument("--input_jsonl", type=str, default=str(default_input))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--keep_worker_outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_examples", type=int, default=9)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target_head", type=str, default="L1H4")
    parser.add_argument("--cumulative_windows", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--top_k_positions", type=int, default=8)
    parser.add_argument("--preview_examples", type=int, default=9)
    parser.add_argument("--preview_chars", type=int, default=1200)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Before closing </think>, include your interim result in \\boxed{}."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--stage1_stop_string", type=str, default="</think>")
    parser.add_argument("--max_stage1_tokens", type=int, default=8192)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    return parser.parse_args()


def build_target_head_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("prev_mass_w") or key.startswith("bucket_"):
            metrics[key] = value
        elif key in {
            "attention_entropy",
            "attention_l2_norm",
            "attention_max_value",
            "top_relative_distance",
            "top_nonself_relative_distance",
            "query_index",
            "query_count",
            "key_count",
            "is_top_current",
        }:
            metrics[key] = value
    return metrics


def write_report(
    *,
    path: Path,
    target_head: str,
    summary_rows: Sequence[Dict[str, Any]],
    preview_rows: Sequence[Dict[str, Any]],
) -> None:
    by_head = {str(item["head_label"]): item for item in summary_rows}
    ranked = sorted(summary_rows, key=lambda item: -float(item.get("mean_bucket_self", 0.0)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Current-Token Next-Step Attention Report\n\n")
        f.write("- metric: `attention to current last token at the next-token prediction step`\n")
        f.write(f"- target_head: `{target_head}`\n\n")
        target_row = by_head.get(target_head)
        f.write("## Target Head\n\n")
        if target_row is None:
            f.write(f"- {target_head}: not found\n\n")
        else:
            f.write(
                f"- head=`{target_head}`, mean_bucket_self={target_row.get('mean_bucket_self', 0.0):.6f}, "
                f"current_top1_rate={target_row.get('current_top1_rate', 0.0):.6f}, "
                f"global_rank_by_mean_bucket_self={target_row.get('global_rank_by_mean_bucket_self', 0)}\n\n"
            )
        f.write("## Top Heads By Current-Token Mass\n\n")
        for row in ranked[:15]:
            f.write(
                f"- {row['head_label']}: mean_bucket_self={row.get('mean_bucket_self', 0.0):.6f}, "
                f"current_top1_rate={row.get('current_top1_rate', 0.0):.6f}, "
                f"mean_prev_mass_w1={row.get('mean_prev_mass_w1', 0.0):.6f}\n"
            )
        f.write("\n## Example Preview\n\n")
        for row in preview_rows:
            f.write(f"### {row.get('example_id', 'unknown')}\n\n")
            f.write(f"- prefix_last_token_text: `{row.get('prefix_last_token_text', '')}`\n")
            f.write(f"- predicted_token_text: `{row.get('predicted_token_text', '')}`\n")
            f.write(f"- query_index: `{row.get('query_index')}`\n")
            f.write(f"- stop_reason: `{row.get('stop_reason')}`\n")
            f.write(f"- target_head_metrics: `{row.get('target_head_metrics')}`\n\n")
            f.write("```text\n")
            f.write(str(row.get("generated_continuation_preview") or ""))
            f.write("\n```\n\n")


def process_rows(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    cumulative_windows = parse_int_list(args.cumulative_windows)
    if 1 not in cumulative_windows:
        raise ValueError("--cumulative_windows must include 1.")

    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map=device_map_override,
            load_in_half=args.load_in_half,
            use_fast_tokenizer=args.use_fast_tokenizer,
            use_safetensors=args.use_safetensors,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    )
    if backend.model is None:
        raise ValueError("This script requires an HF backend.")

    generation_config = GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        stage1_stop_string=args.stage1_stop_string,
        max_stage1_tokens=args.max_stage1_tokens,
        max_new_tokens=args.max_stage1_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        capture_step_scores=False,
    )

    metric_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []

    iterator = tqdm(rows, desc=progress_desc, dynamic_ncols=True, leave=False)
    for row_idx, row in enumerate(iterator, start=1):
        example_id = str(row.get("example_id") or row.get("id") or f"row_{row_idx}")
        try:
            prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
            generation, trace = generate_with_next_token_attention_trace(
                backend,
                prompt_prefix,
                generation_config,
                stop_strings=[args.stage1_stop_string],
            )
            if trace.get("stop_reason") != "matched_stop_sequence":
                raise ValueError(f"Generation stopped with stop_reason={trace.get('stop_reason')}.")

            captured_attentions = trace.get("captured_attentions")
            query_index = trace.get("captured_query_index")
            predicted_token_id = trace.get("captured_predicted_token_id")
            prefix_last_token_id = trace.get("captured_prefix_last_token_id")
            if captured_attentions is None or query_index is None or predicted_token_id is None or prefix_last_token_id is None:
                raise ValueError("Missing captured next-token attention trace.")

            example_metric_rows = compute_all_head_pattern_metrics(
                attentions=captured_attentions,
                query_index=int(query_index),
                cumulative_windows=cumulative_windows,
                top_k=args.top_k_positions,
            )
            for item in example_metric_rows:
                is_top_current = 1.0 if int(item.get("top_relative_distance", -1)) == 0 else 0.0
                item["is_top_current"] = is_top_current
                item["example_id"] = example_id
                item["source"] = row.get("source")
                item["predicted_token_id"] = int(predicted_token_id)
                item["predicted_token_text"] = str(trace.get("captured_predicted_token_text") or "")
                item["prefix_last_token_id"] = int(prefix_last_token_id)
                item["prefix_last_token_text"] = str(trace.get("captured_prefix_last_token_text") or "")
            metric_rows.extend(example_metric_rows)

            target_item = next((item for item in example_metric_rows if str(item["head_label"]) == args.target_head), None)
            generated_preview = truncate_text(generation.continuation, args.preview_chars)
            final_boxed_answer = extract_last_boxed(generation.continuation)
            prompt_tokens = len(trace.get("prompt_token_ids") or [])
            example_rows.append(
                {
                    "example_id": example_id,
                    "prompt_tokens": prompt_tokens,
                    "continuation_tokens": generation.generated_tokens,
                    "query_index": int(query_index),
                    "prefix_last_token_id": int(prefix_last_token_id),
                    "prefix_last_token_text": str(trace.get("captured_prefix_last_token_text") or ""),
                    "predicted_token_id": int(predicted_token_id),
                    "predicted_token_text": str(trace.get("captured_predicted_token_text") or ""),
                    "stop_reason": str(trace.get("stop_reason") or ""),
                    "matched_stop_text": str(trace.get("matched_stop_text") or ""),
                    "final_boxed_answer": final_boxed_answer,
                    "generated_continuation_preview": generated_preview,
                }
            )
            target_rows.append(
                {
                    "example_id": example_id,
                    "query_index": int(query_index),
                    "prefix_last_token_text": str(trace.get("captured_prefix_last_token_text") or ""),
                    "predicted_token_text": str(trace.get("captured_predicted_token_text") or ""),
                    "stop_reason": str(trace.get("stop_reason") or ""),
                    "target_head": args.target_head,
                    "target_head_metrics": build_target_head_metrics(target_item or {}),
                    "target_head_top_positions": list((target_item or {}).get("top_positions") or []),
                    "generated_continuation_preview": generated_preview,
                }
            )
        except Exception as exc:
            skipped_rows.append({"example_id": example_id, "reason": str(exc)})

    return metric_rows, example_rows, target_rows, skipped_rows


def run_worker(
    worker_id: int,
    gpu_id: int,
    shard_rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_metrics_path: str,
    worker_examples_path: str,
    worker_targets_path: str,
    worker_skipped_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    metric_rows, example_rows, target_rows, skipped_rows = process_rows(
        shard_rows,
        args,
        device_map_override={"": gpu_id},
        progress_desc=f"Worker {worker_id} GPU{gpu_id}",
    )
    dump_jsonl(worker_metrics_path, metric_rows)
    dump_jsonl(worker_examples_path, example_rows)
    dump_jsonl(worker_targets_path, target_rows)
    write_json(worker_skipped_path, skipped_rows)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "metric_count": len(metric_rows),
        "example_count": len(example_rows),
        "target_count": len(target_rows),
        "skipped_count": len(skipped_rows),
        "worker_metrics_path": worker_metrics_path,
        "worker_examples_path": worker_examples_path,
        "worker_targets_path": worker_targets_path,
        "worker_skipped_path": worker_skipped_path,
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    rows = rows[args.start_idx :]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise ValueError("No rows selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    if args.parallel and torch.cuda.is_available() and len(available_gpu_ids) > 1 and len(rows) > 1:
        worker_count = args.parallel_workers if args.parallel_workers > 0 else len(available_gpu_ids)
        worker_count = min(worker_count, len(available_gpu_ids), len(rows))
        row_shards = split_examples_contiguous(rows, worker_count)
        worker_gpu_ids = available_gpu_ids[:worker_count]
        metric_rows: List[Dict[str, Any]] = []
        example_rows: List[Dict[str, Any]] = []
        target_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []
        worker_metrics_paths: List[Path] = []
        worker_examples_paths: List[Path] = []
        worker_targets_paths: List[Path] = []
        worker_skipped_paths: List[Path] = []
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as pool:
            futures = []
            for worker_id, shard_rows in enumerate(row_shards):
                gpu_id = worker_gpu_ids[worker_id]
                worker_metrics_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_metrics.jsonl"
                worker_examples_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_examples.jsonl"
                worker_targets_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_targets.jsonl"
                worker_skipped_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_skipped.json"
                futures.append(
                    pool.submit(
                        run_worker,
                        worker_id,
                        gpu_id,
                        shard_rows,
                        vars(args),
                        str(worker_metrics_path),
                        str(worker_examples_path),
                        str(worker_targets_path),
                        str(worker_skipped_path),
                    )
                )
                worker_metrics_paths.append(worker_metrics_path)
                worker_examples_paths.append(worker_examples_path)
                worker_targets_paths.append(worker_targets_path)
                worker_skipped_paths.append(worker_skipped_path)
            for fut in as_completed(futures):
                fut.result()
        for path in worker_metrics_paths:
            metric_rows.extend(load_jsonl(path))
        for path in worker_examples_paths:
            example_rows.extend(load_jsonl(path))
        for path in worker_targets_paths:
            target_rows.extend(load_jsonl(path))
        for path in worker_skipped_paths:
            if path.exists():
                skipped_rows.extend(json.loads(path.read_text(encoding="utf-8")))
        if not args.keep_worker_outputs:
            for path in worker_metrics_paths + worker_examples_paths + worker_targets_paths + worker_skipped_paths:
                if path.exists():
                    path.unlink()
    else:
        device_map_override = {"": args.gpu_id} if args.gpu_id >= 0 else args.device_map
        metric_rows, example_rows, target_rows, skipped_rows = process_rows(
            rows,
            args,
            device_map_override=device_map_override,
            progress_desc="Current-token next-step probe",
        )

    summary_rows = aggregate_pattern_rows(metric_rows, cumulative_windows=parse_int_list(args.cumulative_windows))
    for row in summary_rows:
        group = [item for item in metric_rows if str(item["head_label"]) == str(row["head_label"])]
        row["current_top1_rate"] = 0.0 if not group else round(sum(float(item.get("is_top_current", 0.0)) for item in group) / len(group), 6)
    ranked_self = sorted(summary_rows, key=lambda item: -float(item.get("mean_bucket_self", 0.0)))
    for rank, item in enumerate(ranked_self, start=1):
        item["global_rank_by_mean_bucket_self"] = rank
    ranked_current_top1 = sorted(summary_rows, key=lambda item: -float(item.get("current_top1_rate", 0.0)))
    for rank, item in enumerate(ranked_current_top1, start=1):
        item["global_rank_by_current_top1_rate"] = rank
    distance_rows = aggregate_distance_profile(metric_rows)

    dump_jsonl(output_dir / "per_head_query_metrics.jsonl", metric_rows)
    write_csv(output_dir / "example_generation_summary.csv", example_rows)
    write_csv(output_dir / "target_head_by_example.csv", target_rows)
    write_csv(output_dir / "head_current_attention_summary.csv", summary_rows)
    write_csv(output_dir / "head_distance_profile.csv", distance_rows)
    write_json(
        output_dir / "summary.json",
        {
            "input_jsonl": args.input_jsonl,
            "output_dir": str(output_dir),
            "model_name_or_path": args.model_name_or_path,
            "max_examples": len(rows),
            "skipped_count": len(skipped_rows),
            "top10_by_mean_bucket_self": [
                {
                    "head_label": row["head_label"],
                    "mean_bucket_self": row.get("mean_bucket_self", 0.0),
                    "current_top1_rate": row.get("current_top1_rate", 0.0),
                }
                for row in ranked_self[:10]
            ],
        },
    )
    dump_jsonl(output_dir / "target_head_by_example.jsonl", target_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    write_report(
        path=output_dir / "report.md",
        target_head=args.target_head,
        summary_rows=summary_rows,
        preview_rows=target_rows[: args.preview_examples],
    )


if __name__ == "__main__":
    main()
