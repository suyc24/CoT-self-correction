#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import extract_last_boxed
from cot_research.generation import create_backend
from cot_research.io_utils import dump_jsonl, load_jsonl, truncate_text, write_csv, write_json
from cot_research.model_utils import parse_head_label
from cot_research.ov_circuit_analysis import (
    compute_static_ov_metrics_for_source_tokens,
    generate_with_filtered_head_attention,
)
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the L0H3 mechanism test with a Section 3.1 / Appendix E style OV-circuit analysis: "
            "capture real high-attention token pairs during generation, then test whether the attended token's "
            "own idealized direct-logit contribution is negative."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "ov_circuit" / "qwen3_1p7b_l0h3_20260408"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target_head", type=str, default="L0H3")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--static_gpu_id", type=int, default=-1)
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard examples across multiple GPUs using multiprocessing.",
    )
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--keep_worker_outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_examples", type=int, default=108)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=5)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--stage1_stop_string", type=str, default="</think>")
    parser.add_argument("--max_stage1_tokens", type=int, default=1024)
    parser.add_argument(
        "--require_matched_stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, keep only examples that actually emitted the configured stop string.",
    )
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
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

    parser.add_argument(
        "--attention_threshold",
        type=float,
        default=0.10,
        help="Filter dynamic source/destination pairs with head attention >= this threshold, matching Appendix E.",
    )
    parser.add_argument(
        "--static_score_batch_size",
        type=int,
        default=256,
        help="How many observed source token types to score together during static OV ranking.",
    )
    parser.add_argument(
        "--top_k_suppressed",
        type=int,
        default=10,
        help="Whether the attended token is among the top-k most suppressed logits in its OV column.",
    )
    parser.add_argument(
        "--bottom_fraction",
        type=float,
        default=0.05,
        help="Also mark whether the attended token lies in the most-suppressed bottom fraction of the vocabulary.",
    )
    parser.add_argument("--preview_examples", type=int, default=12)
    parser.add_argument("--preview_pairs", type=int, default=20)
    parser.add_argument("--preview_chars", type=int, default=800)
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl)
    if args.start_idx > 0:
        rows = rows[args.start_idx :]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    return rows


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
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


def build_example_summary_row(
    *,
    row: Dict[str, Any],
    example_id: str,
    generation,
    trace: Dict[str, Any],
    preview_chars: int,
) -> Dict[str, Any]:
    pair_rows = list(trace.get("filtered_pair_rows") or [])
    step_summaries = list(trace.get("step_summaries") or [])
    prompt_tokens = len(trace.get("prompt_token_ids") or [])
    top_attentions = [float(item.get("top_attention", 0.0)) for item in step_summaries]
    return {
        "example_id": example_id,
        "source": row.get("source"),
        "problem": row.get("problem") or row.get("question"),
        "prompt_tokens": int(prompt_tokens),
        "continuation_tokens": int(generation.generated_tokens),
        "full_sequence_tokens": int(prompt_tokens + generation.generated_tokens),
        "stop_reason": str(trace.get("stop_reason") or ""),
        "matched_stop_text": str(trace.get("matched_stop_text") or ""),
        "steps_traced": int(len(step_summaries)),
        "filtered_pair_count_all": int(len(pair_rows)),
        "filtered_pair_count_self": int(sum(1 for item in pair_rows if bool(item["is_self_source"]))),
        "filtered_pair_count_nonself": int(sum(1 for item in pair_rows if not bool(item["is_self_source"]))),
        "filtered_pair_count_generated_nonself": int(
            sum(1 for item in pair_rows if bool(item["is_generated_source"]) and not bool(item["is_self_source"]))
        ),
        "filtered_pair_count_prev1": int(sum(1 for item in pair_rows if bool(item["is_prev_1_source"]))),
        "unique_source_token_types": int(len({int(item["source_token_id"]) for item in pair_rows})),
        "mean_top_attention_over_steps": round(_mean(top_attentions), 6),
        "max_top_attention_over_steps": round(max(top_attentions) if top_attentions else 0.0, 6),
        "final_boxed_answer": extract_last_boxed(generation.continuation),
        "generated_continuation_preview": truncate_text(generation.continuation, preview_chars),
    }


def build_flat_source_token_rows(
    static_rows: Sequence[Dict[str, Any]],
    *,
    pair_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pair_count_by_token: Dict[int, int] = {}
    nonself_count_by_token: Dict[int, int] = {}
    generated_nonself_count_by_token: Dict[int, int] = {}
    attention_sum_by_token: Dict[int, float] = {}
    for pair in pair_rows:
        token_id = int(pair["source_token_id"])
        pair_count_by_token[token_id] = pair_count_by_token.get(token_id, 0) + 1
        attention_sum_by_token[token_id] = attention_sum_by_token.get(token_id, 0.0) + float(pair["attention"])
        if not bool(pair["is_self_source"]):
            nonself_count_by_token[token_id] = nonself_count_by_token.get(token_id, 0) + 1
        if bool(pair["is_generated_source"]) and not bool(pair["is_self_source"]):
            generated_nonself_count_by_token[token_id] = generated_nonself_count_by_token.get(token_id, 0) + 1

    flat_rows: List[Dict[str, Any]] = []
    for row in static_rows:
        token_id = int(row["source_token_id"])
        pair_count = int(pair_count_by_token.get(token_id, 0))
        flat_rows.append(
            {
                "source_token_id": token_id,
                "source_token_text": row["source_token_text"],
                "pair_count_all": pair_count,
                "pair_count_nonself": int(nonself_count_by_token.get(token_id, 0)),
                "pair_count_generated_nonself": int(generated_nonself_count_by_token.get(token_id, 0)),
                "mean_attention_when_observed": round(
                    float(attention_sum_by_token.get(token_id, 0.0) / max(pair_count, 1)),
                    6,
                ),
                "self_logit_contribution": float(row["self_logit_contribution"]),
                "self_rank_from_bottom": int(row["self_rank_from_bottom"]),
                "self_percentile_from_bottom": float(row["self_percentile_from_bottom"]),
                "self_is_negative": bool(row["self_is_negative"]),
                "self_is_topk_suppressed": bool(row["self_is_topk_suppressed"]),
                "self_is_bottom_fraction_suppressed": bool(row["self_is_bottom_fraction_suppressed"]),
                "top_suppressed_token_ids": json.dumps(row["top_suppressed_token_ids"], ensure_ascii=False),
                "top_suppressed_token_texts": json.dumps(row["top_suppressed_token_texts"], ensure_ascii=False),
                "top_suppressed_scores": json.dumps(row["top_suppressed_scores"], ensure_ascii=False),
            }
        )
    flat_rows.sort(key=lambda item: (-int(item["pair_count_all"]), int(item["self_rank_from_bottom"]), int(item["source_token_id"])))
    return flat_rows


def merge_pair_rows_with_static_metrics(
    pair_rows: Sequence[Dict[str, Any]],
    static_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    static_by_token = {int(row["source_token_id"]): row for row in static_rows}
    merged_rows: List[Dict[str, Any]] = []
    for pair in pair_rows:
        static = static_by_token.get(int(pair["source_token_id"]))
        merged = dict(pair)
        if static is not None:
            self_logit = float(static["self_logit_contribution"])
            merged["source_self_logit_contribution"] = self_logit
            merged["attention_scaled_source_self_logit"] = float(pair["attention"]) * self_logit
            merged["source_self_rank_from_bottom"] = int(static["self_rank_from_bottom"])
            merged["source_self_percentile_from_bottom"] = float(static["self_percentile_from_bottom"])
            merged["source_self_is_negative"] = bool(static["self_is_negative"])
            merged["source_self_is_topk_suppressed"] = bool(static["self_is_topk_suppressed"])
            merged["source_self_is_bottom_fraction_suppressed"] = bool(static["self_is_bottom_fraction_suppressed"])
        else:
            merged["source_self_logit_contribution"] = None
            merged["attention_scaled_source_self_logit"] = None
            merged["source_self_rank_from_bottom"] = None
            merged["source_self_percentile_from_bottom"] = None
            merged["source_self_is_negative"] = None
            merged["source_self_is_topk_suppressed"] = None
            merged["source_self_is_bottom_fraction_suppressed"] = None
        merged_rows.append(merged)
    return merged_rows


def summarize_pair_subset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "pair_count": 0,
            "example_count": 0,
            "source_token_type_count": 0,
            "mean_attention": 0.0,
            "mean_source_self_logit_contribution": 0.0,
            "mean_attention_scaled_source_self_logit": 0.0,
            "negative_self_logit_rate": 0.0,
            "topk_suppressed_rate": 0.0,
            "bottom_fraction_suppressed_rate": 0.0,
            "attention_weighted_negative_self_logit_rate": 0.0,
        }

    valid_rows = [row for row in rows if row.get("source_self_logit_contribution") is not None]
    attn_sum = sum(float(row["attention"]) for row in valid_rows)
    weighted_negative = sum(
        float(row["attention"]) for row in valid_rows if bool(row["source_self_is_negative"])
    )
    return {
        "pair_count": int(len(rows)),
        "example_count": int(len({str(row["example_id"]) for row in rows})),
        "source_token_type_count": int(len({int(row["source_token_id"]) for row in rows})),
        "mean_attention": round(_mean(float(row["attention"]) for row in rows), 6),
        "mean_source_self_logit_contribution": round(
            _mean(float(row["source_self_logit_contribution"]) for row in valid_rows),
            6,
        ),
        "mean_attention_scaled_source_self_logit": round(
            _mean(float(row["attention_scaled_source_self_logit"]) for row in valid_rows),
            6,
        ),
        "negative_self_logit_rate": round(
            _mean(1.0 if bool(row["source_self_is_negative"]) else 0.0 for row in valid_rows),
            6,
        ),
        "topk_suppressed_rate": round(
            _mean(1.0 if bool(row["source_self_is_topk_suppressed"]) else 0.0 for row in valid_rows),
            6,
        ),
        "bottom_fraction_suppressed_rate": round(
            _mean(1.0 if bool(row["source_self_is_bottom_fraction_suppressed"]) else 0.0 for row in valid_rows),
            6,
        ),
        "attention_weighted_negative_self_logit_rate": round(
            float(weighted_negative / attn_sum) if attn_sum > 0 else 0.0,
            6,
        ),
    }


def build_pair_subset_summaries(pair_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    subsets = {
        "all_pairs": list(pair_rows),
        "self_pairs": [row for row in pair_rows if bool(row["is_self_source"])],
        "nonself_pairs": [row for row in pair_rows if not bool(row["is_self_source"])],
        "prompt_source_pairs": [row for row in pair_rows if bool(row["is_prompt_source"])],
        "generated_source_pairs": [row for row in pair_rows if bool(row["is_generated_source"])],
        "generated_nonself_pairs": [
            row for row in pair_rows if bool(row["is_generated_source"]) and not bool(row["is_self_source"])
        ],
        "prev1_pairs": [row for row in pair_rows if bool(row["is_prev_1_source"])],
        "generated_prev1_pairs": [
            row for row in pair_rows if bool(row["is_prev_1_source"]) and bool(row["is_generated_source"])
        ],
    }
    return {name: summarize_pair_subset(rows) for name, rows in subsets.items()}


def build_pair_preview_rows(pair_rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        pair_rows,
        key=lambda row: (
            -float(row.get("attention", 0.0)),
            float(row.get("source_self_rank_from_bottom") or 10**9),
            str(row.get("example_id") or ""),
            int(row.get("query_index") or 0),
        ),
    )
    preview_rows: List[Dict[str, Any]] = []
    for row in ranked[: max(int(limit), 0)]:
        preview_rows.append(
            {
                "example_id": row["example_id"],
                "query_index": int(row["query_index"]),
                "query_token_text": row["query_token_text"],
                "source_index": int(row["source_index"]),
                "source_token_text": row["source_token_text"],
                "attention": float(row["attention"]),
                "relative_distance": int(row["relative_distance"]),
                "is_self_source": bool(row["is_self_source"]),
                "is_generated_source": bool(row["is_generated_source"]),
                "source_self_logit_contribution": row.get("source_self_logit_contribution"),
                "source_self_rank_from_bottom": row.get("source_self_rank_from_bottom"),
                "source_self_is_topk_suppressed": row.get("source_self_is_topk_suppressed"),
            }
        )
    return preview_rows


def write_report(
    *,
    path: Path,
    args: argparse.Namespace,
    static_summary: Dict[str, Any],
    pair_subset_summaries: Dict[str, Dict[str, Any]],
    flat_source_rows: Sequence[Dict[str, Any]],
    preview_pairs: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    main_dynamic = pair_subset_summaries.get("generated_nonself_pairs", {})
    all_dynamic = pair_subset_summaries.get("all_pairs", {})

    with open(path, "w", encoding="utf-8") as f:
        f.write("# L0H3 OV Circuit Analysis\n\n")
        f.write("- paper_reference: `Section 3.1 + Appendix E style OV analysis, no replay`\n")
        f.write(f"- model: `{args.model_name_or_path}`\n")
        f.write(f"- target_head: `{args.target_head}`\n")
        f.write(f"- attention_threshold: `{args.attention_threshold}`\n")
        f.write(f"- top_k_suppressed: `{args.top_k_suppressed}`\n")
        f.write(f"- bottom_fraction: `{args.bottom_fraction}`\n\n")

        f.write("## Static OV Summary\n\n")
        f.write(
            "- "
            f"observed_source_token_type_count={static_summary.get('observed_source_token_type_count', 0)}, "
            f"negative_self_logit_rate={static_summary.get('negative_self_logit_rate', 0.0):.6f}, "
            f"topk_suppressed_rate={static_summary.get('topk_suppressed_rate', 0.0):.6f}, "
            f"bottom_fraction_rate={static_summary.get('bottom_fraction_rate', 0.0):.6f}\n"
        )
        f.write(
            "- "
            f"kv_head_idx={static_summary.get('kv_head_idx')}, "
            f"num_kv_heads={static_summary.get('num_kv_heads')}, "
            f"num_query_heads_per_kv_head={static_summary.get('num_query_heads_per_kv_head')}\n\n"
        )

        f.write("## Dynamic Pair Summary\n\n")
        f.write(
            "- "
            f"all_pairs: count={all_dynamic.get('pair_count', 0)}, "
            f"negative_self_logit_rate={all_dynamic.get('negative_self_logit_rate', 0.0):.6f}, "
            f"topk_suppressed_rate={all_dynamic.get('topk_suppressed_rate', 0.0):.6f}\n"
        )
        f.write(
            "- "
            f"generated_nonself_pairs: count={main_dynamic.get('pair_count', 0)}, "
            f"negative_self_logit_rate={main_dynamic.get('negative_self_logit_rate', 0.0):.6f}, "
            f"topk_suppressed_rate={main_dynamic.get('topk_suppressed_rate', 0.0):.6f}, "
            f"mean_attention_scaled_source_self_logit={main_dynamic.get('mean_attention_scaled_source_self_logit', 0.0):.6f}\n\n"
        )

        f.write("## Frequent Source Tokens\n\n")
        for row in flat_source_rows[:10]:
            f.write(
                "- "
                f"id={row['source_token_id']} text={row['source_token_text']!r}: "
                f"pair_count_all={row['pair_count_all']}, "
                f"pair_count_generated_nonself={row['pair_count_generated_nonself']}, "
                f"self_logit={row['self_logit_contribution']:.6f}, "
                f"self_rank_from_bottom={row['self_rank_from_bottom']}, "
                f"topk={row['self_is_topk_suppressed']}\n"
            )
        f.write("\n")

        f.write("## Pair Preview\n\n")
        for row in preview_pairs:
            f.write(
                "- "
                f"{row['example_id']}: "
                f"query={row['query_token_text']!r}@{row['query_index']}, "
                f"source={row['source_token_text']!r}@{row['source_index']}, "
                f"attention={row['attention']:.4f}, "
                f"distance={row['relative_distance']}, "
                f"self_logit={row.get('source_self_logit_contribution')}, "
                f"rank_from_bottom={row.get('source_self_rank_from_bottom')}\n"
            )


def process_rows(
    rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    layer_idx, head_idx = parse_head_label(args.target_head)
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
    generation_config = build_generation_config(args)

    example_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    iterator = tqdm(rows, desc=progress_desc, dynamic_ncols=True, leave=False)
    for row_idx, row in enumerate(iterator, start=1):
        example_id = str(row.get("example_id") or row.get("id") or f"row_{row_idx}")
        try:
            prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
            generation, trace = generate_with_filtered_head_attention(
                backend,
                prompt_prefix,
                generation_config,
                layer_idx=layer_idx,
                head_idx=head_idx,
                attention_threshold=args.attention_threshold,
                stop_strings=[args.stage1_stop_string] if args.stage1_stop_string else None,
            )
            if bool(args.require_matched_stop) and trace.get("stop_reason") != "matched_stop_sequence":
                raise ValueError(f"Generation stopped with stop_reason={trace.get('stop_reason')} before matching </think>.")

            for pair in trace.get("filtered_pair_rows") or []:
                item = dict(pair)
                item["example_id"] = example_id
                pair_rows.append(item)
            example_rows.append(
                build_example_summary_row(
                    row=row,
                    example_id=example_id,
                    generation=generation,
                    trace=trace,
                    preview_chars=args.preview_chars,
                )
            )
            if args.print_every > 0 and row_idx % int(args.print_every) == 0:
                print(
                    "[Info] "
                    f"processed={row_idx} kept={len(example_rows)} skipped={len(skipped_rows)} "
                    f"example_id={example_id} filtered_pairs={len(trace.get('filtered_pair_rows') or [])}"
                )
        except Exception as exc:
            skipped_rows.append({"example_id": example_id, "reason": str(exc)})
    return example_rows, pair_rows, skipped_rows


def run_worker(
    worker_id: int,
    gpu_id: int,
    shard_rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_examples_path: str,
    worker_pairs_path: str,
    worker_skipped_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    example_rows, pair_rows, skipped_rows = process_rows(
        shard_rows,
        args,
        device_map_override={"": gpu_id},
        progress_desc=f"Worker {worker_id} GPU{gpu_id}",
    )
    dump_jsonl(worker_examples_path, example_rows)
    dump_jsonl(worker_pairs_path, pair_rows)
    write_json(worker_skipped_path, skipped_rows)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "example_count": len(example_rows),
        "pair_count": len(pair_rows),
        "skipped_count": len(skipped_rows),
        "worker_examples_path": worker_examples_path,
        "worker_pairs_path": worker_pairs_path,
        "worker_skipped_path": worker_skipped_path,
    }


def run_static_ov_scoring(
    *,
    args: argparse.Namespace,
    source_token_ids: Sequence[int],
    static_gpu_id: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    layer_idx, head_idx = parse_head_label(args.target_head)
    device_map_override: Any = {"": static_gpu_id} if static_gpu_id >= 0 else args.device_map
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
    if backend.model is None or backend.tokenizer is None:
        raise ValueError("Static OV scoring requires an HF backend with a loaded model and tokenizer.")
    rows, summary = compute_static_ov_metrics_for_source_tokens(
        backend.model,
        backend.tokenizer,
        layer_idx=layer_idx,
        head_idx=head_idx,
        source_token_ids=source_token_ids,
        score_batch_size=args.static_score_batch_size,
        top_k_suppressed=args.top_k_suppressed,
        bottom_fraction=args.bottom_fraction,
    )
    return rows, summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    rows = load_rows(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    example_summary_path = output_dir / "example_summary.csv"
    pair_rows_path = output_dir / "pair_rows.jsonl"
    source_token_metrics_jsonl_path = output_dir / "source_token_ov_metrics.jsonl"
    source_token_metrics_csv_path = output_dir / "source_token_ov_metrics.csv"
    pair_preview_path = output_dir / "pair_preview.jsonl"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    run_config_path = output_dir / "run_config.json"

    if not rows:
        write_csv(example_summary_path, [])
        dump_jsonl(pair_rows_path, [])
        dump_jsonl(source_token_metrics_jsonl_path, [])
        write_csv(source_token_metrics_csv_path, [])
        dump_jsonl(pair_preview_path, [])
        write_json(summary_path, {"processed_examples": 0, "skipped_examples": 0})
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
        "[Info] OV-circuit analysis setup: "
        f"examples={len(rows)}, target_head={args.target_head}, attention_threshold={args.attention_threshold}, "
        f"cuda_available={torch_cuda_available}, available_gpu_ids={available_gpu_ids}, "
        f"parallel_enabled={parallel_enabled}, worker_count={worker_count}"
    )

    example_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    if parallel_enabled:
        worker_gpu_ids = available_gpu_ids[:worker_count]
        example_shards = split_examples_contiguous(rows, worker_count)
        worker_example_paths: List[Path] = []
        worker_pair_paths: List[Path] = []
        worker_skipped_paths: List[Path] = []
        worker_returns: List[Dict[str, Any]] = []
        try:
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(example_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_rows in enumerate(example_shards):
                    gpu_id = worker_gpu_ids[worker_id]
                    worker_example_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_examples.jsonl"
                    worker_pair_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_pairs.jsonl"
                    worker_skipped_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_skipped.json"
                    futures.append(
                        pool.submit(
                            run_worker,
                            worker_id,
                            gpu_id,
                            shard_rows,
                            vars(args),
                            str(worker_example_path),
                            str(worker_pair_path),
                            str(worker_skipped_path),
                        )
                    )
                for fut in as_completed(futures):
                    worker_ret = fut.result()
                    worker_returns.append(worker_ret)
                    worker_example_paths.append(Path(worker_ret["worker_examples_path"]))
                    worker_pair_paths.append(Path(worker_ret["worker_pairs_path"]))
                    worker_skipped_paths.append(Path(worker_ret["worker_skipped_path"]))
                    print(
                        f"[Info] Worker {worker_ret['worker_id']} GPU{worker_ret['gpu_id']} finished: "
                        f"examples={worker_ret['example_count']} pairs={worker_ret['pair_count']} skipped={worker_ret['skipped_count']}"
                    )
            for worker_ret in sorted(worker_returns, key=lambda item: int(item["worker_id"])):
                example_rows.extend(load_jsonl(worker_ret["worker_examples_path"]))
                pair_rows.extend(load_jsonl(worker_ret["worker_pairs_path"]))
                with open(worker_ret["worker_skipped_path"], "r", encoding="utf-8") as f:
                    skipped_rows.extend(json.load(f))
        finally:
            if not args.keep_worker_outputs:
                for path in worker_example_paths + worker_pair_paths + worker_skipped_paths:
                    if path.exists():
                        path.unlink()
    else:
        example_rows, pair_rows, skipped_rows = process_rows(
            rows,
            args,
            device_map_override={"": args.gpu_id} if args.gpu_id >= 0 else args.device_map,
            progress_desc="OV circuit analysis",
        )

    source_token_ids = sorted({int(row["source_token_id"]) for row in pair_rows})
    if args.static_gpu_id >= 0:
        static_gpu_id = int(args.static_gpu_id)
    elif available_gpu_ids:
        static_gpu_id = int(available_gpu_ids[0])
    else:
        static_gpu_id = int(args.gpu_id)

    static_rows, static_summary = run_static_ov_scoring(
        args=args,
        source_token_ids=source_token_ids,
        static_gpu_id=static_gpu_id,
    )
    merged_pair_rows = merge_pair_rows_with_static_metrics(pair_rows, static_rows)
    flat_source_rows = build_flat_source_token_rows(static_rows, pair_rows=merged_pair_rows)
    pair_subset_summaries = build_pair_subset_summaries(merged_pair_rows)
    preview_pairs = build_pair_preview_rows(merged_pair_rows, args.preview_pairs)

    write_csv(example_summary_path, example_rows)
    dump_jsonl(pair_rows_path, merged_pair_rows)
    dump_jsonl(source_token_metrics_jsonl_path, static_rows)
    write_csv(source_token_metrics_csv_path, flat_source_rows)
    dump_jsonl(pair_preview_path, preview_pairs)

    summary = {
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "model_name_or_path": args.model_name_or_path,
        "target_head": args.target_head,
        "requested_examples": len(rows),
        "processed_examples": len(example_rows),
        "skipped_examples": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "parallel_enabled": parallel_enabled,
        "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
        "parallel_workers": worker_count,
        "attention_threshold": float(args.attention_threshold),
        "top_k_suppressed": int(args.top_k_suppressed),
        "bottom_fraction": float(args.bottom_fraction),
        "static_gpu_id": int(static_gpu_id),
        "static_summary": static_summary,
        "pair_subset_summaries": pair_subset_summaries,
        "preview_pair_count": len(preview_pairs),
    }
    write_json(summary_path, summary)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
            "parallel_workers": worker_count,
            "static_gpu_id": static_gpu_id,
        },
    )
    write_report(
        path=report_path,
        args=args,
        static_summary=static_summary,
        pair_subset_summaries=pair_subset_summaries,
        flat_source_rows=flat_source_rows,
        preview_pairs=preview_pairs,
    )

    print("[Done] OV-circuit analysis finished:")
    print(f"- output_dir: {output_dir}")
    print(f"- processed_examples: {len(example_rows)}")
    print(f"- skipped_examples: {len(skipped_rows)}")
    print(f"- example_summary_csv: {example_summary_path}")
    print(f"- pair_rows_jsonl: {pair_rows_path}")
    print(f"- source_token_ov_metrics_jsonl: {source_token_metrics_jsonl_path}")
    print(f"- source_token_ov_metrics_csv: {source_token_metrics_csv_path}")
    print(f"- pair_preview_jsonl: {pair_preview_path}")
    print(f"- summary_json: {summary_path}")
    print(f"- report_md: {report_path}")


if __name__ == "__main__":
    main()
