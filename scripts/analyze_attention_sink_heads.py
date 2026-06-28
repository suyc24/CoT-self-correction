#!/usr/bin/env python3
from __future__ import annotations

"""Analyze whether specific heads behave like attention sink heads.

The script computes per-head attention mass directed to designated sink positions,
by default the first few tokens of the analyzed sequence. This is useful for
checking hypotheses like whether `L0H3` behaves as an attention sink head.

Outputs:
- head_sink_summary.csv
- target_heads_by_example.jsonl
- summary.json
- report.md
- run_config.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.attention_sink_analysis import (
    SinkConfig,
    aggregate_head_metrics,
    annotate_example_head_ranks,
    compute_head_sink_metrics,
    parse_comma_list,
    resolve_sink_positions,
)
from cot_research.head_intervention import list_model_heads
from cot_research.io_utils import load_jsonl
from cot_research.model_utils import forward_with_attentions, load_hf_model_and_tokenizer
from cot_research.prompt_utils import build_chat_prompt
from cot_research.row_utils import resolve_row_identities, select_continuation_text
from cot_research.runtime_utils import parse_parallel_gpu_ids, resolve_device_map, split_examples_contiguous
from cot_research.summary_utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_input = root_dir / "outputs" / "repetition" / "all_repetition_cases.jsonl"
    default_output = root_dir / "outputs" / "attention_sink" / "qwen3_1p7b"

    parser = argparse.ArgumentParser(description="Analyze attention sink behavior for model heads")
    parser.add_argument("--input_jsonl", type=str, default=str(default_input))
    parser.add_argument(
        "--exclude_jsonl",
        type=str,
        default="",
        help="Optional JSONL whose example_id/id rows will be excluded from --input_jsonl before sampling.",
    )
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
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target_heads", type=str, default="L0H3")
    parser.add_argument(
        "--analysis_text_field",
        type=str,
        default="auto",
        choices=["auto", "full_text", "prompt", "continuation", "prompt_plus_continuation"],
    )
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--truncate_side", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--prefix_token_count", type=int, default=4)
    parser.add_argument(
        "--sink_token_texts",
        type=str,
        default="",
        help="Comma-separated extra sink marker texts to include, e.g. '<think>,</think>'.",
    )
    parser.add_argument("--late_query_start", type=int, default=-1)
    parser.add_argument("--sink_mass_threshold", type=float, default=0.1)
    parser.add_argument("--preview_examples", type=int, default=20)
    parser.add_argument("--preview_heads", type=int, default=20)
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
        help="Attention implementation to request when loading the model. 'eager' is recommended for output_attentions.",
    )
    return parser.parse_args()


def build_exclude_id_set(path: Path) -> set[str]:
    excluded = set()
    for row in load_jsonl(path):
        excluded.update(resolve_row_identities(row))
    return excluded


def load_model_and_tokenizer(args: argparse.Namespace):
    return load_hf_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        load_in_half=args.load_in_half,
        use_fast_tokenizer=args.use_fast_tokenizer,
        use_safetensors=args.use_safetensors,
        local_files_only=args.local_files_only,
        device_map=resolve_device_map(args.device_map, args.gpu_id),
        attn_implementation=args.attn_implementation,
    )


def resolve_analysis_text(row: Dict[str, Any], tokenizer, args: argparse.Namespace) -> Tuple[str, str]:
    full_text = str(row.get("full_text") or "")
    prompt = str(row.get("prompt") or row.get("prompt_prefix") or "")
    continuation = select_continuation_text(row)
    question = str(row.get("problem") or row.get("question") or "")
    prompt_from_question = ""
    if question.strip():
        prompt_from_question = build_chat_prompt(
            tokenizer,
            question=question,
            system_prompt=args.system_prompt,
            assistant_prefix=args.assistant_prefix,
            enable_thinking=args.enable_thinking,
        )

    if args.analysis_text_field == "full_text":
        if full_text.strip():
            return full_text, "full_text"
        raise ValueError("Requested full_text analysis but row has no full_text.")
    if args.analysis_text_field == "prompt":
        text = prompt or prompt_from_question
        if text.strip():
            return text, "prompt"
        raise ValueError("Requested prompt analysis but row has no prompt/prompt_prefix/question.")
    if args.analysis_text_field == "continuation":
        if continuation.strip():
            return continuation, "continuation"
        raise ValueError("Requested continuation analysis but row has no continuation text.")
    if args.analysis_text_field == "prompt_plus_continuation":
        base_prompt = prompt or prompt_from_question
        if base_prompt.strip() and continuation.strip():
            return base_prompt + continuation, "prompt_plus_continuation"
        raise ValueError("Requested prompt_plus_continuation but row lacks prompt/question or continuation.")

    if full_text.strip():
        return full_text, "full_text"
    if prompt.strip() and continuation.strip():
        return prompt + continuation, "prompt_plus_continuation"
    if prompt_from_question.strip() and continuation.strip():
        return prompt_from_question + continuation, "prompt_plus_continuation"
    if prompt.strip():
        return prompt, "prompt"
    if prompt_from_question.strip():
        return prompt_from_question, "prompt"
    if continuation.strip():
        return continuation, "continuation"
    raise ValueError("Cannot resolve analysis text for row.")


def tokenize_text(tokenizer, text: str, max_input_tokens: int, truncate_side: str) -> List[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if max_input_tokens > 0 and len(token_ids) > max_input_tokens:
        if truncate_side == "left":
            token_ids = token_ids[-max_input_tokens:]
        else:
            token_ids = token_ids[:max_input_tokens]
    return list(token_ids)


def write_report(
    path: Path,
    summary_rows: List[Dict[str, Any]],
    target_heads: List[str],
    preview_examples: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_by_head = {str(row["head_label"]): row for row in summary_rows}
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Attention Sink Report\n\n")
        f.write("## Target Heads\n\n")
        for head in target_heads:
            row = summary_by_head.get(head)
            if row is None:
                f.write(f"- {head}: not found\n")
                continue
            f.write(
                "- "
                f"{head}: global_rank={row.get('global_rank_by_sink_mass_ratio')}, "
                f"layer_rank={row.get('layer_rank_by_sink_mass_ratio')}, "
                f"mean_sink_ratio={row.get('mean_sink_mass_ratio_late_queries'):.4f}, "
                f"mean_sink_mass={row.get('mean_sink_mass_late_queries'):.4f}, "
                f"mean_pos0_mass={row.get('mean_pos0_mass_late_queries'):.4f}, "
                f"mean_example_global_rank={row.get('mean_example_global_rank_by_sink_mass_ratio'):.2f}, "
                f"mean_example_layer_rank={row.get('mean_example_layer_rank_by_sink_mass_ratio'):.2f}, "
                f"example_layer_top1_rate={row.get('example_layer_top1_rate_by_sink_mass_ratio'):.4f}\n"
            )
        f.write("\n## Top Heads\n\n")
        for row in summary_rows[:20]:
            f.write(
                "- "
                f"{row['head_label']}: global_rank={row['global_rank_by_sink_mass_ratio']}, "
                f"layer_rank={row.get('layer_rank_by_sink_mass_ratio')}, "
                f"sink_ratio={row['mean_sink_mass_ratio_late_queries']:.4f}, "
                f"sink_mass={row['mean_sink_mass_late_queries']:.4f}, "
                f"pos0_mass={row['mean_pos0_mass_late_queries']:.4f}, "
                f"example_layer_top1_rate={row.get('example_layer_top1_rate_by_sink_mass_ratio', 0.0):.4f}\n"
            )
        f.write("\n## Example Preview\n\n")
        for item in preview_examples:
            f.write(f"### {item.get('example_id', 'unknown')}\n\n")
            f.write(f"- analysis_source: `{item.get('analysis_source')}`\n")
            f.write(f"- token_count: `{item.get('token_count')}`\n")
            f.write(f"- sink_positions: `{item.get('sink_positions')}`\n")
            f.write(f"- target_head_metrics: `{item.get('target_head_metrics')}`\n\n")


def process_rows(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]], str, int]:
    model, tokenizer = load_model_and_tokenizer(argparse.Namespace(**{**vars(args), "device_map": device_map_override, "gpu_id": -1}))
    all_heads, _, layer_path = list_model_heads(model)
    target_heads = parse_comma_list(args.target_heads)
    target_head_set = set(target_heads)
    sink_config = SinkConfig(
        prefix_token_count=args.prefix_token_count,
        sink_token_texts=parse_comma_list(args.sink_token_texts),
        late_query_start=args.late_query_start,
        sink_mass_threshold=args.sink_mass_threshold,
    )

    per_head_metric_rows: List[Dict[str, Any]] = []
    target_example_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    iterator = tqdm(rows, desc=progress_desc, dynamic_ncols=True, leave=False)
    for row in iterator:
        example_id = str(row.get("example_id") or row.get("id") or "unknown")
        try:
            text, analysis_source = resolve_analysis_text(row, tokenizer, args)
            token_ids = tokenize_text(tokenizer, text, args.max_input_tokens, args.truncate_side)
            if len(token_ids) < 2:
                raise ValueError("Tokenized sequence too short for attention analysis.")
            sink_info = resolve_sink_positions(token_ids, tokenizer, sink_config)
            sink_info["sink_mass_threshold"] = args.sink_mass_threshold
            attentions = forward_attentions(model, token_ids)
            metric_rows = compute_head_sink_metrics(attentions=attentions, sink_info=sink_info)
            for item in metric_rows:
                item["example_id"] = example_id
                item["analysis_source"] = analysis_source
            metric_rows = annotate_example_head_ranks(metric_rows)
            per_head_metric_rows.extend(metric_rows)

            target_metrics = [
                {
                    "head_label": item["head_label"],
                    "mean_sink_mass_late_queries": item["mean_sink_mass_late_queries"],
                    "sink_mass_ratio_late_queries": item["sink_mass_ratio_late_queries"],
                    "mean_pos0_mass_late_queries": item["mean_pos0_mass_late_queries"],
                    "top1_sink_rate_late_queries": item["top1_sink_rate_late_queries"],
                    "example_global_rank_by_sink_mass_ratio": item["example_global_rank_by_sink_mass_ratio"],
                    "example_layer_rank_by_sink_mass_ratio": item["example_layer_rank_by_sink_mass_ratio"],
                }
                for item in metric_rows
                if item["head_label"] in target_head_set
            ]
            target_example_rows.append(
                {
                    "example_id": example_id,
                    "analysis_source": analysis_source,
                    "token_count": len(token_ids),
                    "sink_positions": sink_info["sink_positions"],
                    "sink_token_count": sink_info["sink_token_count"],
                    "late_query_start": sink_info["late_query_start"],
                    "target_head_metrics": target_metrics,
                }
            )
        except Exception as exc:
            skipped_rows.append({"example_id": example_id, "reason": str(exc)})

    return per_head_metric_rows, target_example_rows, skipped_rows, layer_path, len(all_heads)


def run_worker(
    worker_id: int,
    gpu_id: int,
    shard_rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_metrics_path: str,
    worker_targets_path: str,
    worker_skipped_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    metric_rows, target_example_rows, skipped_rows, layer_path, num_total_heads = process_rows(
        shard_rows,
        args,
        device_map_override={"": gpu_id},
        progress_desc=f"Worker {worker_id} GPU{gpu_id}",
    )
    with open(worker_metrics_path, "w", encoding="utf-8") as f:
        for row in metric_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(worker_targets_path, "w", encoding="utf-8") as f:
        for row in target_example_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(worker_skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped_rows, f, ensure_ascii=False, indent=2)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "metric_count": len(metric_rows),
        "target_example_count": len(target_example_rows),
        "skipped_count": len(skipped_rows),
        "worker_metrics_path": worker_metrics_path,
        "worker_targets_path": worker_targets_path,
        "worker_skipped_path": worker_skipped_path,
        "layer_path": layer_path,
        "num_total_heads": num_total_heads,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    exclude_path = Path(args.exclude_jsonl) if args.exclude_jsonl.strip() else None
    if exclude_path is not None and not exclude_path.exists():
        raise FileNotFoundError(f"Exclude JSONL not found: {exclude_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "head_sink_summary.csv"
    target_example_path = output_dir / "target_heads_by_example.jsonl"
    summary_json_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    run_config_path = output_dir / "run_config.json"

    rows = load_jsonl(input_path)
    excluded_ids: set[str] = set()
    if exclude_path is not None:
        excluded_ids = build_exclude_id_set(exclude_path)
        rows = [
            row
            for row in rows
            if not any(identity in excluded_ids for identity in resolve_row_identities(row))
        ]
    if args.start_idx > 0:
        rows = rows[args.start_idx :]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        write_json(
            summary_json_path,
            {
                "input_jsonl": str(input_path),
                "exclude_jsonl": str(exclude_path) if exclude_path is not None else "",
                "processed_examples": 0,
            },
        )
        write_json(run_config_path, {"args": vars(args)})
        summary_csv_path.write_text("", encoding="utf-8")
        target_example_path.write_text("", encoding="utf-8")
        report_path.write_text("# No rows\n", encoding="utf-8")
        print("[Done] No rows to analyze.")
        return

    per_head_metric_rows: List[Dict[str, Any]] = []
    target_example_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    target_heads = parse_comma_list(args.target_heads)
    sink_config = SinkConfig(
        prefix_token_count=args.prefix_token_count,
        sink_token_texts=parse_comma_list(args.sink_token_texts),
        late_query_start=args.late_query_start,
        sink_mass_threshold=args.sink_mass_threshold,
    )
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
        "[Info] Attention sink setup: "
        f"examples={len(rows)}, excluded={len(excluded_ids)}, "
        f"cuda_available={torch_cuda_available}, available_gpu_ids={available_gpu_ids}, "
        f"parallel_enabled={parallel_enabled}, worker_count={worker_count}"
    )

    layer_path = ""
    num_total_heads = 0
    if parallel_enabled:
        worker_gpu_ids = available_gpu_ids[:worker_count]
        example_shards = split_examples_contiguous(rows, worker_count)
        worker_metrics_paths: List[Path] = []
        worker_targets_paths: List[Path] = []
        worker_skipped_paths: List[Path] = []
        worker_returns: List[Dict[str, Any]] = []
        try:
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(example_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_rows in enumerate(example_shards):
                    gpu_id = worker_gpu_ids[worker_id]
                    worker_metrics_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_metrics.jsonl"
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
                            str(worker_targets_path),
                            str(worker_skipped_path),
                        )
                    )
                for fut in as_completed(futures):
                    worker_ret = fut.result()
                    worker_returns.append(worker_ret)
                    worker_metrics_paths.append(Path(worker_ret["worker_metrics_path"]))
                    worker_targets_paths.append(Path(worker_ret["worker_targets_path"]))
                    worker_skipped_paths.append(Path(worker_ret["worker_skipped_path"]))
                    if not layer_path:
                        layer_path = str(worker_ret.get("layer_path") or "")
                    if not num_total_heads:
                        num_total_heads = int(worker_ret.get("num_total_heads") or 0)
                    print(
                        f"[Info] Worker {worker_ret['worker_id']} GPU{worker_ret['gpu_id']} finished: "
                        f"target_examples={worker_ret['target_example_count']} skipped={worker_ret['skipped_count']}"
                    )

            for worker_ret in sorted(worker_returns, key=lambda item: int(item["worker_id"])):
                with open(worker_ret["worker_metrics_path"], "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            per_head_metric_rows.append(json.loads(line))
                with open(worker_ret["worker_targets_path"], "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            target_example_rows.append(json.loads(line))
                with open(worker_ret["worker_skipped_path"], "r", encoding="utf-8") as f:
                    skipped_rows.extend(json.load(f))
        finally:
            if not args.keep_worker_outputs:
                for path in worker_metrics_paths + worker_targets_paths + worker_skipped_paths:
                    if path.exists():
                        path.unlink()
    else:
        per_head_metric_rows, target_example_rows, skipped_rows, layer_path, num_total_heads = process_rows(
            rows,
            args,
            device_map_override=resolve_device_map(args.device_map, args.gpu_id),
            progress_desc="Attention sink analysis",
        )

    summary_rows = aggregate_head_metrics(per_head_metric_rows)
    write_csv(summary_csv_path, summary_rows)
    with open(target_example_path, "w", encoding="utf-8") as f:
        for row in target_example_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    target_summary = [row for row in summary_rows if row["head_label"] in target_head_set]
    summary = {
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "exclude_jsonl": str(exclude_path) if exclude_path is not None else "",
        "excluded_examples": len(excluded_ids),
        "processed_examples": len(target_example_rows),
        "skipped_examples": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "model_name_or_path": args.model_name_or_path,
        "decoder_layer_path": layer_path,
        "num_total_heads": num_total_heads,
        "target_heads": target_heads,
        "sink_config": sink_config.to_dict(),
        "analysis_text_field": args.analysis_text_field,
        "max_input_tokens": args.max_input_tokens,
        "truncate_side": args.truncate_side,
        "parallel_enabled": parallel_enabled,
        "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
        "parallel_workers": worker_count,
        "target_head_hypothesis_support": [
            {
                "head_label": row["head_label"],
                "global_rank_by_sink_mass_ratio": row["global_rank_by_sink_mass_ratio"],
                "layer_rank_by_sink_mass_ratio": row.get("layer_rank_by_sink_mass_ratio"),
                "mean_sink_mass_ratio_late_queries": row["mean_sink_mass_ratio_late_queries"],
                "mean_pos0_mass_late_queries": row["mean_pos0_mass_late_queries"],
                "mean_example_global_rank_by_sink_mass_ratio": row.get("mean_example_global_rank_by_sink_mass_ratio"),
                "mean_example_layer_rank_by_sink_mass_ratio": row.get("mean_example_layer_rank_by_sink_mass_ratio"),
                "example_global_top10_rate_by_sink_mass_ratio": row.get("example_global_top10_rate_by_sink_mass_ratio"),
                "example_layer_top1_rate_by_sink_mass_ratio": row.get("example_layer_top1_rate_by_sink_mass_ratio"),
            }
            for row in target_summary
        ],
        "top_heads_by_sink_mass_ratio": summary_rows[: max(args.preview_heads, 0)],
        "target_head_summary": target_summary,
    }
    write_json(summary_json_path, summary)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "sink_config": sink_config.to_dict(),
            "decoder_layer_path": layer_path,
            "target_heads": target_heads,
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
            "parallel_workers": worker_count,
        },
    )
    write_report(report_path, summary_rows, target_heads, target_example_rows[: max(args.preview_examples, 0)])

    print("[Done] Attention sink analysis finished:")
    print(f"- input_jsonl: {input_path}")
    print(f"- output_dir: {output_dir}")
    print(f"- processed_examples: {len(target_example_rows)}")
    print(f"- skipped_examples: {len(skipped_rows)}")
    print(f"- head_sink_summary_csv: {summary_csv_path}")
    print(f"- target_heads_by_example_jsonl: {target_example_path}")
    print(f"- summary_json: {summary_json_path}")
    print(f"- report_md: {report_path}")


if __name__ == "__main__":
    main()
