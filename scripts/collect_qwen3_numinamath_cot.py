#!/usr/bin/env python3
from __future__ import annotations

"""Collect Qwen3-1.7B CoT generations on AI-MO/NuminaMath-CoT with vLLM.

The script is designed for repetition-loop analysis, so each output row stores:
- dataset metadata (`source`, `problem`, optional reference solution/messages)
- raw model output (`generated_continuation`, `full_text`, `think_text`)
- answer extraction (`final_boxed_answer`)
- generation metadata (`generated_tokens`, `closed_think`, `finish_reason`)
- repetition heuristics on generated token ids/text

Typical usage:
    python scripts/collect_qwen3_numinamath_cot.py \
      --output_jsonl outputs/numinamath_qwen3_1p7b_cot.jsonl \
      --max_examples 5000 \
      --parallel_gpu_ids 0,1,2,3 \
      --request_batch_size 1024 \
      --resume
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm

from cot_research.answer_extraction import extract_last_boxed
from cot_research.datasets import load_numinamath_dataset
from cot_research.prompt_utils import build_chat_prompt
from cot_research.repetition_analysis import (
    longest_same_token_run,
    repeated_line_stats,
    repeated_suffix_stats as shared_repeated_suffix_stats,
)
from cot_research.row_utils import build_problem_text, build_reference_solution
from cot_research.runtime_utils import parse_parallel_gpu_ids
from cot_research.text_analysis import extract_continuation_think_text, extract_think_segments


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_output = root_dir / "outputs" / "numinamath_qwen3_1p7b_cot.jsonl"

    parser = argparse.ArgumentParser(description="Collect Qwen3-1.7B CoT data from AI-MO/NuminaMath-CoT with vLLM")
    parser.add_argument("--dataset_name", type=str, default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_cache_dir", type=str, default=str(root_dir / "evaluation" / "data" / "temp"))
    parser.add_argument(
        "--dataset_local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only load the dataset from local cache.",
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_jsonl", type=str, default=str(default_output))
    parser.add_argument("--run_config_json", type=str, default="")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use multiple GPUs through vLLM tensor parallelism when multiple GPUs are selected.",
    )
    parser.add_argument(
        "--parallel_gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU ids, e.g. '0,1,2,3'. Uses CUDA_VISIBLE_DEVICES remapping for vLLM.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=0,
        help="vLLM tensor_parallel_size. 0 means auto: selected GPU count if --parallel, else 1.",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384)
    parser.add_argument(
        "--enable_chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vLLM chunked prefill to improve mixed prefill/decode utilization.",
    )
    parser.add_argument(
        "--enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable CUDA graph capture in vLLM. Usually slower; keep false unless debugging.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=1024,
        help="How many prompts to hand to one llm.generate call.",
    )
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--source_filter",
        type=str,
        default="",
        help="Comma-separated source filter against the dataset `source` field.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_token_ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=20)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def detect_visible_gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def configure_cuda_visible_devices(args: argparse.Namespace) -> List[int]:
    requested_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids, default_to_visible=False)
    if requested_gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in requested_gpu_ids)
        return requested_gpu_ids
    existing = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if existing:
        return [int(token.strip()) for token in existing.split(",") if token.strip()]
    return list(range(detect_visible_gpu_count()))


def repeated_suffix_token_stats(token_ids: Sequence[int], n_values: Sequence[int] = (1, 2, 4, 8, 16)) -> List[Dict[str, Any]]:
    return [
        {
            "ngram_size": int(item["ngram_size"]),
            "repeat_count_at_tail": int(item["repeat_count_at_tail"]),
            "pattern_token_ids": list(item["pattern"]),
        }
        for item in shared_repeated_suffix_stats(token_ids, n_values=n_values)
    ]


def load_completed_example_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            done.add(str(obj["example_id"]))
    return done


def make_example_id(row: Dict[str, Any], local_index: int) -> str:
    source = str(row.get("source", "unknown")).strip() or "unknown"
    if "id" in row:
        return f"{source}:{row['id']}"
    if "idx" in row:
        return f"{source}:{row['idx']}"
    problem = build_problem_text(row)
    digest = hashlib.sha1(problem.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def chunked(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("--request_batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_config_path = Path(args.run_config_json) if args.run_config_json else output_path.with_suffix(".run_config.json")

    requested_gpu_ids = configure_cuda_visible_devices(args)

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "This script requires the `vllm` package. Install it into the existing qwen_math environment, for example:\n"
            "  pip install vllm\n"
            "No new environment is required."
        ) from exc

    visible_gpu_count = detect_visible_gpu_count()
    if args.tensor_parallel_size > 0:
        tensor_parallel_size = args.tensor_parallel_size
    elif args.parallel and visible_gpu_count > 0:
        tensor_parallel_size = visible_gpu_count
    else:
        tensor_parallel_size = 1
    if visible_gpu_count > 0:
        tensor_parallel_size = max(1, min(tensor_parallel_size, visible_gpu_count))
    else:
        tensor_parallel_size = 1

    dataset = load_numinamath_dataset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        local_files_only=args.dataset_local_files_only,
        source_filter=args.source_filter,
        shuffle=args.shuffle,
        seed=args.seed,
        start_idx=args.start_idx,
        sample_stride=args.sample_stride,
        max_examples=args.max_examples,
    )
    completed_ids = load_completed_example_ids(output_path) if args.resume else set()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    pending_examples: List[Dict[str, Any]] = []
    skipped = 0
    for local_index, raw_row in enumerate(dataset):
        row = dict(raw_row)
        example_id = make_example_id(row, local_index)
        if example_id in completed_ids:
            skipped += 1
            continue
        problem = build_problem_text(row)
        prompt = build_chat_prompt(
            tokenizer=tokenizer,
            question=problem,
            system_prompt=args.system_prompt,
            assistant_prefix=args.assistant_prefix,
            enable_thinking=args.enable_thinking,
        )
        pending_examples.append(
            {
                "dataset_local_index": local_index,
                "example_id": example_id,
                "row": row,
                "problem": problem,
                "prompt": prompt,
            }
        )

    sampling_params = SamplingParams(
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 1.0,
        max_tokens=args.max_new_tokens,
        n=1,
    )

    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
    )

    run_config = {
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "dataset_cache_dir": args.dataset_cache_dir,
        "dataset_local_files_only": args.dataset_local_files_only,
        "model_name_or_path": args.model_name_or_path,
        "parallel": args.parallel,
        "requested_gpu_ids": requested_gpu_ids,
        "visible_gpu_count": visible_gpu_count,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enforce_eager": args.enforce_eager,
        "request_batch_size": args.request_batch_size,
        "max_examples": args.max_examples,
        "start_idx": args.start_idx,
        "sample_stride": args.sample_stride,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "source_filter": args.source_filter,
        "resume": args.resume,
        "save_token_ids": args.save_token_ids,
        "system_prompt": args.system_prompt,
        "assistant_prefix": args.assistant_prefix,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "output_jsonl": str(output_path),
        "pending_examples": len(pending_examples),
        "resume_skipped_examples": skipped,
    }
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(
        "[Info] Collection setup: "
        f"pending_examples={len(pending_examples)}, "
        f"resume_skipped={skipped}, "
        f"requested_gpu_ids={requested_gpu_ids}, "
        f"visible_gpu_count={visible_gpu_count}, "
        f"tensor_parallel_size={tensor_parallel_size}, "
        f"request_batch_size={args.request_batch_size}, "
        f"max_num_seqs={args.max_num_seqs}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}, "
        f"chunked_prefill={args.enable_chunked_prefill}"
    )

    write_mode = "a" if args.resume else "w"
    written = 0
    with open(output_path, write_mode, encoding="utf-8") as out_f:
        progress = tqdm(total=len(pending_examples), desc="Collect CoT (vLLM)", dynamic_ncols=True)
        for batch in chunked(pending_examples, args.request_batch_size):
            prompts = [item["prompt"] for item in batch]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for a batch of {len(batch)} prompts."
                )
            for item, request_output in zip(batch, outputs):
                if not request_output.outputs:
                    raise RuntimeError(f"No completion returned for example_id={item['example_id']}")
                completion = request_output.outputs[0]
                continuation = completion.text
                token_ids = list(completion.token_ids)
                full_text = item["prompt"] + continuation
                assistant_text = full_text[len(item["prompt"]) - len(args.assistant_prefix) :]
                think_segments = extract_think_segments(assistant_text)
                think_text = (
                    "\n\n".join(think_segments)
                    if think_segments
                    else extract_continuation_think_text(continuation)
                )
                export_row: Dict[str, Any] = {
                    "example_id": item["example_id"],
                    "dataset_name": args.dataset_name,
                    "dataset_split": args.dataset_split,
                    "dataset_local_index": int(item["dataset_local_index"]),
                    "source": item["row"].get("source"),
                    "problem": item["problem"],
                    "reference_solution": build_reference_solution(item["row"]),
                    "reference_messages": item["row"].get("messages"),
                    "prompt": item["prompt"],
                    "generated_continuation": continuation,
                    "full_text": full_text,
                    "think_text": think_text,
                    "final_boxed_answer": extract_last_boxed(continuation),
                    "generated_tokens": len(token_ids),
                    "closed_think": "</think>" in continuation.lower(),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "repetition": {
                        "longest_same_token_run": longest_same_token_run(token_ids),
                        "tail_repeated_suffix": repeated_suffix_token_stats(token_ids),
                        "repeated_line_stats": repeated_line_stats(continuation),
                    },
                }
                if args.save_token_ids:
                    export_row["generated_token_ids"] = token_ids
                out_f.write(json.dumps(export_row, ensure_ascii=False) + "\n")
                written += 1
                progress.update(1)

                if written % max(args.print_every, 1) == 0:
                    print(
                        f"[Info] written={written} skipped={skipped} "
                        f"last_example_id={item['example_id']} generated_tokens={len(token_ids)}"
                    )
        progress.close()

    print("[Done] Collection finished:")
    print(f"- output_jsonl: {output_path}")
    print(f"- run_config_json: {run_config_path}")
    print(f"- written_examples: {written}")
    print(f"- skipped_examples: {skipped}")
    print(f"- tensor_parallel_size: {tensor_parallel_size}")


if __name__ == "__main__":
    main()
