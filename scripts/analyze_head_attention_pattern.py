#!/usr/bin/env python3
from __future__ import annotations

"""Analyze the attention pattern of a single head near the final </think> boundary.

For each example:
1. Generate stage-1 CoT until the closing </think>.
2. Keep examples whose continuation contains a boxed answer before the final </think>.
3. Re-run the model with output_attentions=True on a trailing token window ending at
   the query position immediately before the final </think> token sequence.
4. Extract one target head's attention vector and summarize where it attends.

Outputs:
- per_example_pattern.jsonl
- per_example_pattern.csv
- distance_profile.csv
- summary.json
- report.md
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import extract_last_boxed
from cot_research.generation import create_backend
from cot_research.head_attention_pattern import (
    aggregate_distance_profile,
    build_distance_bucket_spec,
    compute_head_pattern_metrics,
    parse_int_list,
)
from cot_research.io_utils import dump_jsonl, load_jsonl, truncate_text, write_csv, write_json
from cot_research.local_attention_analysis import locate_close_think_query
from cot_research.model_utils import forward_with_attentions, parse_head_label
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_input = root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"
    default_output = root_dir / "outputs" / "head_attention_pattern" / "qwen3_1p7b_l0h3"

    parser = argparse.ArgumentParser(description="Analyze the attention pattern of one target head.")
    parser.add_argument("--input_jsonl", type=str, default=str(default_input))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--keep_worker_outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--target_head", type=str, default="L0H3")
    parser.add_argument(
        "--cumulative_windows",
        type=str,
        default="1,4,8,16,32,64,128,256",
        help="Comma-separated backward window sizes. Each statistic excludes self and sums attention to the previous k tokens.",
    )
    parser.add_argument(
        "--analysis_window_tokens",
        type=int,
        default=2048,
        help="Trailing token window length ending at the query before </think> used for attention analysis.",
    )
    parser.add_argument("--top_k_positions", type=int, default=10)
    parser.add_argument("--preview_examples", type=int, default=20)
    parser.add_argument("--preview_chars", type=int, default=800)
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


def has_boxed_before_close(continuation: str) -> bool:
    close_pos = continuation.lower().rfind("</think>")
    if close_pos < 0:
        return False
    boxed_pos = continuation.rfind("\\boxed", 0, close_pos)
    return boxed_pos >= 0


def build_top_position_rows(
    attention_vector: torch.Tensor,
    *,
    top_positions: Sequence[Dict[str, Any]],
    window_start_in_full: int,
    window_token_ids: Sequence[int],
    decode_token,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in top_positions:
        idx_in_window = int(item["index_in_window"])
        token_id = int(window_token_ids[idx_in_window])
        rows.append(
            {
                "index_in_window": idx_in_window,
                "index_in_full_sequence": window_start_in_full + idx_in_window,
                "relative_distance": int(item["relative_distance"]),
                "attention": float(item["attention"]),
                "token_id": token_id,
                "token_text": decode_token([token_id]),
            }
        )
    return rows


def process_rows(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    layer_idx, head_idx = parse_head_label(args.target_head)
    cumulative_windows = parse_int_list(args.cumulative_windows)
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
        raise ValueError("This script requires an HF backend with a loaded model.")

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
    close_think_token_ids = backend.encode(args.stage1_stop_string)
    if not close_think_token_ids:
        raise ValueError("The configured stage1_stop_string tokenizes to an empty sequence.")

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    iterator = tqdm(rows, desc=progress_desc, dynamic_ncols=True, leave=False)
    for row_idx, row in enumerate(iterator, start=1):
        example_id = str(row.get("example_id") or row.get("id") or f"row_{row_idx}")
        try:
            prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
            generation = backend.generate(
                prompt_prefix,
                generation_config,
                stop_strings=[args.stage1_stop_string],
            )
            if args.stage1_stop_string.lower() not in generation.continuation.lower():
                raise ValueError("Generated continuation does not contain the closing </think> string.")
            if not has_boxed_before_close(generation.continuation):
                raise ValueError("Generated continuation does not contain a boxed answer before </think>.")

            prompt_token_ids = backend.encode(prompt_prefix)
            full_token_ids = list(prompt_token_ids) + list(generation.token_ids)
            query_info = locate_close_think_query(
                prompt_token_ids=prompt_token_ids,
                continuation_token_ids=generation.token_ids,
                close_think_token_ids=close_think_token_ids,
            )
            query_index = int(query_info["query_index_before_close"])
            window_start = max(0, query_index + 1 - args.analysis_window_tokens)
            window_token_ids = full_token_ids[window_start : query_index + 1]
            attentions = forward_attentions(backend.model, window_token_ids)
            layer_attn = attentions[layer_idx][0].detach().float().cpu()
            if head_idx >= int(layer_attn.shape[0]):
                raise ValueError(f"Head {args.target_head} exceeds available heads in layer {layer_idx}.")
            attention_vector = layer_attn[head_idx, -1]
            metrics = compute_head_pattern_metrics(
                attention_vector,
                cumulative_windows=cumulative_windows,
                top_k=args.top_k_positions,
            )
            top_positions = build_top_position_rows(
                attention_vector,
                top_positions=metrics.pop("top_positions"),
                window_start_in_full=window_start,
                window_token_ids=window_token_ids,
                decode_token=backend.decode,
            )

            result_rows.append(
                {
                    "example_id": example_id,
                    "source": row.get("source"),
                    "problem": row.get("problem") or row.get("question"),
                    "prompt_tokens": len(prompt_token_ids),
                    "continuation_tokens": generation.generated_tokens,
                    "full_sequence_tokens": len(full_token_ids),
                    "query_index_before_close": query_index,
                    "analysis_window_start": window_start,
                    "analysis_window_tokens": len(window_token_ids),
                    "final_boxed_answer": extract_last_boxed(generation.continuation),
                    "head_label": args.target_head,
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    **metrics,
                    "top_positions": top_positions,
                    "generated_continuation_preview": truncate_text(generation.continuation, args.preview_chars),
                }
            )
        except Exception as exc:
            skipped_rows.append({"example_id": example_id, "reason": str(exc)})
    return result_rows, skipped_rows


def run_worker(
    worker_id: int,
    gpu_id: int,
    shard_rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_rows_path: str,
    worker_skipped_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    result_rows, skipped_rows = process_rows(
        shard_rows,
        args,
        device_map_override={"": gpu_id},
        progress_desc=f"Worker {worker_id} GPU{gpu_id}",
    )
    dump_jsonl(worker_rows_path, result_rows)
    write_json(worker_skipped_path, skipped_rows)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "row_count": len(result_rows),
        "skipped_count": len(skipped_rows),
        "worker_rows_path": worker_rows_path,
        "worker_skipped_path": worker_skipped_path,
    }


def write_report(path: Path, summary: Dict[str, Any], preview_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Head Attention Pattern Report\n\n")
        f.write(f"- target_head: `{summary['target_head']}`\n")
        f.write(f"- analyzed_examples: `{summary['analyzed_examples']}`\n")
        f.write(f"- skipped_examples: `{summary['skipped_examples']}`\n")
        f.write(f"- mean_attention_entropy: `{summary['mean_attention_entropy']:.6f}`\n")
        for key, value in summary.items():
            if key.startswith("mean_prev_mass_w"):
                f.write(f"- {key}: `{value:.6f}`\n")
        for key, value in summary.items():
            if key.startswith("mean_bucket_"):
                f.write(f"- {key}: `{value:.6f}`\n")
        f.write("\n## Example Preview\n\n")
        for row in preview_rows:
            f.write(f"### {row['example_id']}\n\n")
            f.write(f"- final_boxed_answer: `{row.get('final_boxed_answer')}`\n")
            f.write(f"- query_index_before_close: `{row['query_index_before_close']}`\n")
            f.write(f"- analysis_window_tokens: `{row['analysis_window_tokens']}`\n")
            f.write(f"- top_positions: `{row.get('top_positions')}`\n\n")
            f.write("```text\n")
            f.write(str(row.get("generated_continuation_preview") or ""))
            f.write("\n```\n\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    all_rows = load_jsonl(input_path)
    if args.max_examples > 0:
        all_rows = all_rows[: args.max_examples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "per_example_pattern.jsonl"
    rows_csv_path = output_dir / "per_example_pattern.csv"
    distance_profile_path = output_dir / "distance_profile.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    run_config_path = output_dir / "run_config.json"

    available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    torch_cuda_available = torch.cuda.is_available()
    can_parallel = args.parallel and torch_cuda_available and len(available_gpu_ids) > 1 and len(all_rows) > 1
    if can_parallel and args.parallel_workers > 0:
        worker_count = min(args.parallel_workers, len(available_gpu_ids), len(all_rows))
    elif can_parallel:
        worker_count = min(len(available_gpu_ids), len(all_rows))
    else:
        worker_count = 1
    parallel_enabled = can_parallel and worker_count > 1

    print(
        "[Info] Head attention pattern setup: "
        f"examples={len(all_rows)}, target_head={args.target_head}, "
        f"cuda_available={torch_cuda_available}, available_gpu_ids={available_gpu_ids}, "
        f"parallel_enabled={parallel_enabled}, worker_count={worker_count}"
    )

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    if parallel_enabled:
        worker_gpu_ids = available_gpu_ids[:worker_count]
        example_shards = split_examples_contiguous(all_rows, worker_count)
        worker_rows_paths: List[Path] = []
        worker_skipped_paths: List[Path] = []
        worker_returns: List[Dict[str, Any]] = []
        try:
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(example_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_rows in enumerate(example_shards):
                    gpu_id = worker_gpu_ids[worker_id]
                    worker_rows_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_rows.jsonl"
                    worker_skipped_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_skipped.json"
                    futures.append(
                        pool.submit(
                            run_worker,
                            worker_id,
                            gpu_id,
                            shard_rows,
                            vars(args),
                            str(worker_rows_path),
                            str(worker_skipped_path),
                        )
                    )
                for fut in as_completed(futures):
                    worker_ret = fut.result()
                    worker_returns.append(worker_ret)
                    worker_rows_paths.append(Path(worker_ret["worker_rows_path"]))
                    worker_skipped_paths.append(Path(worker_ret["worker_skipped_path"]))
                    print(
                        f"[Info] Worker {worker_ret['worker_id']} GPU{worker_ret['gpu_id']} finished: "
                        f"rows={worker_ret['row_count']} skipped={worker_ret['skipped_count']}"
                    )

            for worker_ret in sorted(worker_returns, key=lambda item: int(item["worker_id"])):
                result_rows.extend(load_jsonl(worker_ret["worker_rows_path"]))
                with open(worker_ret["worker_skipped_path"], "r", encoding="utf-8") as f:
                    skipped_rows.extend(json.load(f))
        finally:
            if not args.keep_worker_outputs:
                for path in worker_rows_paths + worker_skipped_paths:
                    if path.exists():
                        path.unlink()
    else:
        result_rows, skipped_rows = process_rows(
            all_rows,
            args,
            device_map_override={"": args.gpu_id} if args.gpu_id >= 0 else args.device_map,
            progress_desc="Head attention pattern",
        )

    cumulative_windows = parse_int_list(args.cumulative_windows)
    rows_for_csv: List[Dict[str, Any]] = []
    for row in result_rows:
        flat = {key: value for key, value in row.items() if key not in {"top_positions", "generated_continuation_preview"}}
        rows_for_csv.append(flat)

    summary = {
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "model_name_or_path": args.model_name_or_path,
        "target_head": args.target_head,
        "requested_examples": len(all_rows),
        "analyzed_examples": len(result_rows),
        "skipped_examples": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "parallel_enabled": parallel_enabled,
        "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
        "parallel_workers": worker_count,
    }
    if result_rows:
        profile = aggregate_distance_profile(result_rows)
        head_summary = aggregate_distance_profile([row for row in result_rows])
        write_csv(distance_profile_path, profile)
        summary.update(
            {
                "mean_attention_entropy": sum(float(row["attention_entropy"]) for row in result_rows) / len(result_rows),
                "mean_attention_l2_norm": sum(float(row["attention_l2_norm"]) for row in result_rows) / len(result_rows),
            }
        )
        for window in cumulative_windows:
            summary[f"mean_prev_mass_w{window}"] = sum(float(row.get(f"prev_mass_w{window}", 0.0)) for row in result_rows) / len(result_rows)
        for bucket_name, _, _ in build_distance_bucket_spec():
            summary[f"mean_bucket_{bucket_name}"] = sum(float(row.get(f"bucket_{bucket_name}", 0.0)) for row in result_rows) / len(result_rows)
    else:
        write_csv(distance_profile_path, [])

    dump_jsonl(rows_path, result_rows)
    write_csv(rows_csv_path, rows_for_csv)
    write_json(summary_path, summary)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
            "parallel_workers": worker_count,
        },
    )
    write_report(report_path, summary, result_rows[: max(args.preview_examples, 0)])

    print("[Done] Head attention pattern analysis finished:")
    print(f"- output_dir: {output_dir}")
    print(f"- analyzed_examples: {len(result_rows)}")
    print(f"- skipped_examples: {len(skipped_rows)}")
    print(f"- rows_jsonl: {rows_path}")
    print(f"- rows_csv: {rows_csv_path}")
    print(f"- distance_profile_csv: {distance_profile_path}")
    print(f"- summary_json: {summary_path}")
    print(f"- report_md: {report_path}")


if __name__ == "__main__":
    main()
