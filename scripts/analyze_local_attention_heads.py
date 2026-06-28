#!/usr/bin/env python3
from __future__ import annotations

"""Rank all heads by prev-1 attention on the generated </think> token query.

Experiment logic:
1. Build the prompt prefix for each example.
2. Generate stage-1 reasoning with a manual HF decoding loop that records the
   query attention of each generated token as it is produced.
3. Stop when the generated token suffix matches </think>.
4. Analyze only the query attention vector of the generated </think> token.
5. Rank all heads by how much that query attends to the immediately previous
   token and summarize whether the top attended position is prev-1.

Outputs:
- per_head_query_metrics.jsonl
- example_generation_summary.csv
- head_prev_attention_summary.csv
- head_distance_profile.csv
- target_head_by_example.jsonl
- summary.json
- report.md
"""

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
from cot_research.generation import create_backend, generate_with_query_attention_trace
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
    default_output = root_dir / "outputs" / "local_attention_probe" / "qwen3_1p7b_prev1"

    parser = argparse.ArgumentParser(
        description="Analyze all heads on the generated </think> token query and rank prev-1 behavior."
    )
    parser.add_argument("--input_jsonl", type=str, default=str(default_input))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard examples across multiple GPUs using multiprocessing.",
    )
    parser.add_argument(
        "--parallel_gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU ids for parallel mode, e.g. '0,1,2,3'. Default: all visible GPUs.",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=0,
        help="Number of worker processes. 0 means one worker per selected GPU.",
    )
    parser.add_argument(
        "--keep_worker_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep intermediate worker files under output_dir.",
    )
    parser.add_argument("--max_examples", type=int, default=9)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target_head", type=str, default="L0H3")
    parser.add_argument(
        "--cumulative_windows",
        type=str,
        default="1,2,4,8,16,32,64",
        help="Comma-separated backward window sizes. Must include 1 for prev-1 ranking.",
    )
    parser.add_argument("--top_k_positions", type=int, default=8)
    parser.add_argument(
        "--require_boxed_before_close",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optionally require a boxed answer before </think>. Disabled by default for pure attention analysis.",
    )
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
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Attention implementation to request when loading the model. 'eager' is recommended for output_attentions.",
    )
    return parser.parse_args()


def has_boxed_before_close(continuation: str) -> bool:
    close_pos = continuation.lower().rfind("</think>")
    if close_pos < 0:
        return False
    boxed_pos = continuation.rfind("\\boxed", 0, close_pos)
    return boxed_pos >= 0


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
            "is_top_prev_1",
            "is_top_nonself_prev_1",
            "query_index",
            "query_count",
            "key_count",
        }:
            metrics[key] = value
    return metrics


def write_report(
    *,
    path: Path,
    target_head: str,
    summary_rows: Sequence[Dict[str, Any]],
    cumulative_windows: Sequence[int],
    preview_rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_head = {str(item["head_label"]): item for item in summary_rows}

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Prev-1 Query Attention Report\n\n")
        f.write("- direct_query_capture: `true`\n")
        f.write(f"- target_head: `{target_head}`\n")
        f.write(f"- cumulative_windows: `{list(cumulative_windows)}`\n\n")

        f.write("## Target Head\n\n")
        target_row = by_head.get(target_head)
        if target_row is None:
            f.write(f"- {target_head}: not found in analyzed heads\n\n")
        else:
            f.write(f"- head: `{target_head}`\n")
            f.write(
                "- "
                f"mean_prev_mass_w1={target_row.get('mean_prev_mass_w1', 0.0):.6f}, "
                f"mean_bucket_self={target_row.get('mean_bucket_self', 0.0):.6f}, "
                f"prev_1_top1_rate={target_row.get('prev_1_top1_rate', 0.0):.6f}, "
                f"prev_1_top_nonself_rate={target_row.get('prev_1_top_nonself_rate', 0.0):.6f}\n"
            )
            f.write(
                "- "
                f"global_rank_by_mean_prev_mass_w1={target_row.get('global_rank_by_mean_prev_mass_w1', 0)}, "
                f"global_rank_by_prev_1_top1_rate={target_row.get('global_rank_by_prev_1_top1_rate', 0)}, "
                f"global_rank_by_prev_1_top_nonself_rate={target_row.get('global_rank_by_prev_1_top_nonself_rate', 0)}\n"
            )
            for window in cumulative_windows:
                f.write(f"- mean_prev_mass_w{window}={target_row.get(f'mean_prev_mass_w{window}', 0.0):.6f}\n")
        f.write("\n")

        f.write("## Top Heads By Prev-1 Mass\n\n")
        ranked_by_mass = sorted(summary_rows, key=lambda item: -float(item.get("mean_prev_mass_w1", 0.0)))
        for row in ranked_by_mass[:15]:
            f.write(
                "- "
                f"{row['head_label']}: "
                f"mean_prev_mass_w1={row.get('mean_prev_mass_w1', 0.0):.6f}, "
                f"mean_bucket_self={row.get('mean_bucket_self', 0.0):.6f}, "
                f"prev_1_top1_rate={row.get('prev_1_top1_rate', 0.0):.6f}\n"
            )
        f.write("\n")

        f.write("## Top Heads By Prev-1 Top-1 Rate\n\n")
        ranked_by_top1 = sorted(summary_rows, key=lambda item: -float(item.get("prev_1_top1_rate", 0.0)))
        for row in ranked_by_top1[:15]:
            f.write(
                "- "
                f"{row['head_label']}: "
                f"prev_1_top1_rate={row.get('prev_1_top1_rate', 0.0):.6f}, "
                f"prev_1_top_nonself_rate={row.get('prev_1_top_nonself_rate', 0.0):.6f}, "
                f"mean_prev_mass_w1={row.get('mean_prev_mass_w1', 0.0):.6f}\n"
            )
        f.write("\n")

        f.write("## Example Preview\n\n")
        for row in preview_rows:
            f.write(f"### {row.get('example_id', 'unknown')}\n\n")
            f.write(f"- matched_stop_text: `{row.get('matched_stop_text', '')}`\n")
            f.write(f"- query_token_text: `{row.get('query_token_text', '')}`\n")
            f.write(f"- query_index: `{row.get('query_index')}`\n")
            f.write(f"- stop_reason: `{row.get('stop_reason')}`\n")
            f.write(f"- final_boxed_answer: `{row.get('final_boxed_answer')}`\n")
            f.write(f"- target_head_metrics: `{row.get('target_head_metrics')}`\n")
            f.write(f"- target_head_top_positions: `{row.get('target_head_top_positions')}`\n\n")
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
        raise ValueError("--cumulative_windows must include 1 so prev-1 ranking is defined.")

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
            generation, trace = generate_with_query_attention_trace(
                backend,
                prompt_prefix,
                generation_config,
                stop_strings=[args.stage1_stop_string],
            )
            if trace.get("stop_reason") != "matched_stop_sequence":
                raise ValueError(f"Generation stopped with stop_reason={trace.get('stop_reason')} before matching </think>.")
            if args.require_boxed_before_close and not has_boxed_before_close(generation.continuation):
                raise ValueError("Generated continuation does not contain a boxed answer before </think>.")

            captured_attentions = trace.get("captured_attentions")
            query_index = trace.get("captured_query_index")
            query_token_id = trace.get("captured_token_id")
            if captured_attentions is None or query_index is None or query_token_id is None:
                raise ValueError("Missing captured query attentions for the generated </think> token.")

            example_metric_rows = compute_all_head_pattern_metrics(
                attentions=captured_attentions,
                query_index=int(query_index),
                cumulative_windows=cumulative_windows,
                top_k=args.top_k_positions,
            )
            for item in example_metric_rows:
                item["example_id"] = example_id
                item["source"] = row.get("source")
                item["query_token_id"] = int(query_token_id)
                item["query_token_text"] = str(trace.get("captured_token_text") or "")
            metric_rows.extend(example_metric_rows)

            target_item = next((item for item in example_metric_rows if str(item["head_label"]) == args.target_head), None)
            final_boxed_answer = extract_last_boxed(generation.continuation)
            prompt_tokens = len(trace.get("prompt_token_ids") or [])
            generated_preview = truncate_text(generation.continuation, args.preview_chars)

            example_rows.append(
                {
                    "example_id": example_id,
                    "source": row.get("source"),
                    "problem": row.get("problem") or row.get("question"),
                    "prompt_tokens": prompt_tokens,
                    "continuation_tokens": generation.generated_tokens,
                    "full_sequence_tokens": prompt_tokens + generation.generated_tokens,
                    "query_index": int(query_index),
                    "query_token_id": int(query_token_id),
                    "query_token_text": str(trace.get("captured_token_text") or ""),
                    "matched_stop_text": str(trace.get("matched_stop_text") or ""),
                    "stop_reason": str(trace.get("stop_reason") or ""),
                    "matched_stop_token_ids": json.dumps(trace.get("matched_stop_token_ids") or []),
                    "trace_steps": len(trace.get("step_trace") or []),
                    "final_boxed_answer": final_boxed_answer,
                    "generated_continuation_preview": generated_preview,
                }
            )
            target_rows.append(
                {
                    "example_id": example_id,
                    "problem": row.get("problem") or row.get("question"),
                    "query_index": int(query_index),
                    "query_token_id": int(query_token_id),
                    "query_token_text": str(trace.get("captured_token_text") or ""),
                    "matched_stop_text": str(trace.get("matched_stop_text") or ""),
                    "full_sequence_tokens": prompt_tokens + generation.generated_tokens,
                    "stop_reason": str(trace.get("stop_reason") or ""),
                    "final_boxed_answer": final_boxed_answer,
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
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    cumulative_windows = parse_int_list(args.cumulative_windows)
    if 1 not in cumulative_windows:
        raise ValueError("--cumulative_windows must include 1 so prev-1 ranking is defined.")

    rows = load_jsonl(input_path)
    if args.start_idx > 0:
        rows = rows[args.start_idx :]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_head_metrics_path = output_dir / "per_head_query_metrics.jsonl"
    example_summary_path = output_dir / "example_generation_summary.csv"
    head_summary_path = output_dir / "head_prev_attention_summary.csv"
    distance_profile_path = output_dir / "head_distance_profile.csv"
    target_path = output_dir / "target_head_by_example.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    run_config_path = output_dir / "run_config.json"

    if not rows:
        dump_jsonl(per_head_metrics_path, [])
        write_csv(example_summary_path, [])
        write_csv(head_summary_path, [])
        write_csv(distance_profile_path, [])
        dump_jsonl(target_path, [])
        write_json(summary_path, {"input_jsonl": str(input_path), "processed_examples": 0})
        write_json(run_config_path, {"args": vars(args)})
        report_path.write_text("# No rows\n", encoding="utf-8")
        print("[Done] No rows to analyze.")
        return

    available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    torch_cuda_available = torch.cuda.is_available()
    can_parallel = args.parallel and torch_cuda_available and len(available_gpu_ids) > 1 and len(rows) > 1
    if can_parallel and args.parallel_workers > 0:
        worker_count = min(args.parallel_workers, len(available_gpu_ids), len(rows))
    elif can_parallel:
        worker_count = min(len(available_gpu_ids), len(rows))
    else:
        worker_count = 1
    parallel_enabled = can_parallel and worker_count > 1

    print(
        "[Info] Prev-1 attention probe setup: "
        f"examples={len(rows)}, target_head={args.target_head}, cumulative_windows={cumulative_windows}, "
        f"cuda_available={torch_cuda_available}, available_gpu_ids={available_gpu_ids}, "
        f"parallel_enabled={parallel_enabled}, worker_count={worker_count}"
    )

    metric_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    if parallel_enabled:
        worker_gpu_ids = available_gpu_ids[:worker_count]
        example_shards = split_examples_contiguous(rows, worker_count)
        worker_metric_paths: List[Path] = []
        worker_example_paths: List[Path] = []
        worker_target_paths: List[Path] = []
        worker_skipped_paths: List[Path] = []
        worker_returns: List[Dict[str, Any]] = []
        try:
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(example_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_rows in enumerate(example_shards):
                    gpu_id = worker_gpu_ids[worker_id]
                    worker_metric_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_metrics.jsonl"
                    worker_example_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_examples.jsonl"
                    worker_target_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_targets.jsonl"
                    worker_skipped_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_skipped.json"
                    futures.append(
                        pool.submit(
                            run_worker,
                            worker_id,
                            gpu_id,
                            shard_rows,
                            vars(args),
                            str(worker_metric_path),
                            str(worker_example_path),
                            str(worker_target_path),
                            str(worker_skipped_path),
                        )
                    )
                for fut in as_completed(futures):
                    worker_ret = fut.result()
                    worker_returns.append(worker_ret)
                    worker_metric_paths.append(Path(worker_ret["worker_metrics_path"]))
                    worker_example_paths.append(Path(worker_ret["worker_examples_path"]))
                    worker_target_paths.append(Path(worker_ret["worker_targets_path"]))
                    worker_skipped_paths.append(Path(worker_ret["worker_skipped_path"]))
                    print(
                        f"[Info] Worker {worker_ret['worker_id']} GPU{worker_ret['gpu_id']} "
                        f"finished: metrics={worker_ret['metric_count']} examples={worker_ret['example_count']} skipped={worker_ret['skipped_count']}"
                    )

            for worker_ret in sorted(worker_returns, key=lambda item: int(item["worker_id"])):
                metric_rows.extend(load_jsonl(worker_ret["worker_metrics_path"]))
                example_rows.extend(load_jsonl(worker_ret["worker_examples_path"]))
                target_rows.extend(load_jsonl(worker_ret["worker_targets_path"]))
                with open(worker_ret["worker_skipped_path"], "r", encoding="utf-8") as f:
                    skipped_rows.extend(json.load(f))
        finally:
            if not args.keep_worker_outputs:
                for path in worker_metric_paths + worker_example_paths + worker_target_paths + worker_skipped_paths:
                    if path.exists():
                        path.unlink()
    else:
        metric_rows, example_rows, target_rows, skipped_rows = process_rows(
            rows,
            args,
            device_map_override={"": args.gpu_id} if args.gpu_id >= 0 else args.device_map,
            progress_desc="Prev-1 attention probe",
        )

    summary_rows = aggregate_pattern_rows(metric_rows, cumulative_windows=cumulative_windows)
    ranked_summary_rows = sorted(
        summary_rows,
        key=lambda item: (
            int(item.get(f"global_rank_by_mean_prev_mass_w1", 10**9)),
            int(item.get("layer_idx", 0)),
            int(item.get("head_idx", 0)),
        ),
    )
    distance_profile_rows = aggregate_distance_profile(metric_rows)
    dump_jsonl(per_head_metrics_path, metric_rows)
    write_csv(example_summary_path, example_rows)
    write_csv(head_summary_path, ranked_summary_rows)
    write_csv(distance_profile_path, distance_profile_rows)
    dump_jsonl(target_path, target_rows)

    target_summary = next((row for row in summary_rows if str(row["head_label"]) == args.target_head), None)
    summary = {
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "model_name_or_path": args.model_name_or_path,
        "target_head": args.target_head,
        "requested_examples": len(rows),
        "analyzed_examples": len(example_rows),
        "skipped_examples": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "parallel_enabled": parallel_enabled,
        "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
        "parallel_workers": worker_count,
        "cumulative_windows": cumulative_windows,
        "top_k_positions": args.top_k_positions,
        "require_boxed_before_close": bool(args.require_boxed_before_close),
        "target_head_summary": target_summary or {},
    }
    write_json(summary_path, summary)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "cumulative_windows": cumulative_windows,
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
            "parallel_workers": worker_count,
        },
    )

    write_report(
        path=report_path,
        target_head=args.target_head,
        summary_rows=ranked_summary_rows,
        cumulative_windows=cumulative_windows,
        preview_rows=target_rows[: max(args.preview_examples, 0)],
    )

    print("[Done] Prev-1 attention analysis finished:")
    print(f"- output_dir: {output_dir}")
    print(f"- analyzed_examples: {len(example_rows)}")
    print(f"- skipped_examples: {len(skipped_rows)}")
    print(f"- per_head_query_metrics_jsonl: {per_head_metrics_path}")
    print(f"- example_generation_summary_csv: {example_summary_path}")
    print(f"- head_prev_attention_summary_csv: {head_summary_path}")
    print(f"- head_distance_profile_csv: {distance_profile_path}")
    print(f"- target_head_by_example_jsonl: {target_path}")
    print(f"- summary_json: {summary_path}")
    print(f"- report_md: {report_path}")


if __name__ == "__main__":
    main()
