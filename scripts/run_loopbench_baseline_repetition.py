#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tqdm import tqdm

from cot_research.generation import create_backend
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.loopbench_reconstructed import (
    PAPER_BASELINE_DECODING,
    RECONSTRUCTED_SYSTEM_PROMPT,
)
from cot_research.repetition_analysis import (
    LoopBenchThresholds,
    analyze_loopbench_repetition,
)
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LoopBench baseline repetition evaluation with batched vLLM generation."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "loopbench_reconstructed_v2" / "test.jsonl"),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--parallel_gpu_ids", type=str, required=True)
    parser.add_argument("--request_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--system_prompt", type=str, default=RECONSTRUCTED_SYSTEM_PROMPT)
    parser.add_argument("--assistant_prefix", type=str, default="")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=PAPER_BASELINE_DECODING["temperature"])
    parser.add_argument("--top_k", type=int, default=PAPER_BASELINE_DECODING["top_k"])
    parser.add_argument("--top_p", type=float, default=PAPER_BASELINE_DECODING["top_p"])
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--backend_type", type=str, default="vllm")
    parser.add_argument("--tensor_parallel_size", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--enable_chunked_prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce_eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--numerical_loop_min_repeated_span", type=int, default=500)
    parser.add_argument("--statement_loop_min_repeat_count", type=int, default=4)
    parser.add_argument("--numerical_same_digit_run_threshold", type=int, default=500)
    return parser.parse_args()


def _chunked(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("--request_batch_size must be >= 1")
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def _parse_gpu_ids(text: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        gpu_id = int(token)
        if gpu_id < 0:
            raise ValueError(f"Invalid GPU id: {gpu_id}")
        if gpu_id not in seen:
            seen.add(gpu_id)
            out.append(gpu_id)
    return out


def _aggregate_subtask_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_subtask: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_subtask.setdefault(str(row["subtask"]), []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    for subtask, chunk in sorted(by_subtask.items()):
        total = len(chunk)
        summary_rows.append(
            {
                "subtask": subtask,
                "example_count": total,
                "numerical_loop_rate": round(sum(1 for row in chunk if row["numerical_loop"]) / total, 6),
                "statement_loop_rate": round(sum(1 for row in chunk if row["statement_loop"]) / total, 6),
                "repetition_rate": round(sum(1 for row in chunk if row["is_repetitive"]) / total, 6),
                "hit_max_rate": round(sum(1 for row in chunk if row["hit_max_new_tokens"]) / total, 6),
                "mean_generated_tokens": round(sum(int(row["generated_tokens"]) for row in chunk) / total, 6),
            }
        )
    return summary_rows


def _build_messages(question: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    rows = load_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows loaded from {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    rows_jsonl_path = output_dir / "rows.jsonl"
    rows_csv_path = output_dir / "rows.csv"
    subtask_summary_path = output_dir / "subtask_summary.csv"
    summary_json_path = output_dir / "summary.json"

    requested_gpu_ids = _parse_gpu_ids(args.parallel_gpu_ids)
    if not requested_gpu_ids:
        raise ValueError("Provide --parallel_gpu_ids.")
    if args.tensor_parallel_size > 0:
        tensor_parallel_size = int(args.tensor_parallel_size)
    else:
        tensor_parallel_size = len(requested_gpu_ids)
    tensor_parallel_size = max(1, tensor_parallel_size)

    thresholds = LoopBenchThresholds(
        numerical_loop_min_repeated_span=args.numerical_loop_min_repeated_span,
        statement_loop_min_repeat_count=args.statement_loop_min_repeat_count,
        numerical_same_digit_run_threshold=args.numerical_same_digit_run_threshold,
    )

    backend = create_backend(
        BackendConfig(
            backend_type=args.backend_type,
            model_name_or_path=args.model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_chunked_prefill=args.enable_chunked_prefill,
            enforce_eager=args.enforce_eager,
            load_in_half=args.load_in_half,
            use_fast_tokenizer=args.use_fast_tokenizer,
            use_safetensors=args.use_safetensors,
            local_files_only=args.local_files_only,
        )
    )
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

    message_batches: List[List[Dict[str, str]]] = []
    prompts: List[str] = []
    for row in rows:
        question = str(row.get("question") or "")
        if not question:
            raise ValueError(f"Example {row.get('id')} is missing a question.")
        messages = _build_messages(question, args.system_prompt)
        message_batches.append(messages)
        prompts.append(backend.build_prompt(question, generation_config))

    write_json(
        run_config_path,
        {
            "args": vars(args),
            "total_examples": len(rows),
            "requested_gpu_ids": requested_gpu_ids,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "paper_decoding_reference": dict(PAPER_BASELINE_DECODING),
            "effective_decoding": {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_new_tokens": args.max_new_tokens,
            },
            "system_prompt_note": "Figure 10 prompt is only partially public; this run uses the closest public reconstruction.",
        },
    )

    print(
        f"[Info] LoopBench baseline repetition: model={args.model_name_or_path}, examples={len(rows)}, "
        f"backend={args.backend_type}, visible_gpus={os.environ.get('CUDA_VISIBLE_DEVICES','')}, "
        f"requested_gpus={requested_gpu_ids}, tp={tensor_parallel_size}, batch_size={args.request_batch_size}, "
        f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_k={args.top_k}, "
        f"top_p={args.top_p}, repetition_penalty={args.repetition_penalty}"
    )

    result_rows: List[Dict[str, Any]] = []
    index_batches = _chunked(list(range(len(rows))), args.request_batch_size)
    for batch_idx, batch_indices in enumerate(tqdm(index_batches, desc="batches", dynamic_ncols=True), start=1):
        batch_messages = [message_batches[idx] for idx in batch_indices]
        generations = backend.generate_many_chat(batch_messages, generation_config)
        for local_idx, generation in enumerate(generations):
            row = rows[batch_indices[local_idx]]
            example_id = str(row.get("id") or batch_indices[local_idx])
            metadata = dict(row.get("metadata") or {})
            repetition = analyze_loopbench_repetition(
                generation.continuation,
                thresholds=thresholds,
            )
            result_rows.append(
                {
                    "example_id": example_id,
                    "subtask": str(metadata.get("subtask") or "unknown"),
                    "task_category": str(metadata.get("task_category") or "unknown"),
                    "solver_status": str(metadata.get("solver_status") or "validated"),
                    "generated_tokens": int(generation.generated_tokens),
                    "is_repetitive": bool(repetition["matched"]),
                    "numerical_loop": bool(repetition["numerical_loop"]),
                    "statement_loop": bool(repetition["statement_loop"]),
                    "hit_max_new_tokens": bool(generation.generated_tokens >= args.max_new_tokens),
                    "digit_stream_length": int(repetition["digit_stream_length"]),
                    "sentence_count": int(repetition["sentence_count"]),
                    "line_count": int(repetition["line_count"]),
                    "clause_count": int(repetition["clause_count"]),
                    "same_digit_run": int(repetition["same_digit_run"]),
                    "numerical_tail": json.dumps(repetition["numerical_tail"], ensure_ascii=False),
                    "statement_tail": json.dumps(repetition["statement_tail"], ensure_ascii=False),
                    "sentence_tail": json.dumps(repetition["sentence_tail"], ensure_ascii=False),
                    "line_tail": json.dumps(repetition["line_tail"], ensure_ascii=False),
                    "clause_tail": json.dumps(repetition["clause_tail"], ensure_ascii=False),
                    "generated_continuation": generation.continuation,
                }
            )
        if batch_idx % max(args.print_every, 1) == 0:
            rep_so_far = sum(1 for item in result_rows if item["is_repetitive"])
            print(f"[Progress] batches={batch_idx}/{len(index_batches)} examples={len(result_rows)}/{len(rows)} rep_so_far={rep_so_far}")

    subtask_summary_rows = _aggregate_subtask_summary(result_rows)
    total = len(result_rows)
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "example_count": total,
        "numerical_loop_rate": round(sum(1 for row in result_rows if row["numerical_loop"]) / total, 6),
        "statement_loop_rate": round(sum(1 for row in result_rows if row["statement_loop"]) / total, 6),
        "repetition_rate": round(sum(1 for row in result_rows if row["is_repetitive"]) / total, 6),
        "hit_max_rate": round(sum(1 for row in result_rows if row["hit_max_new_tokens"]) / total, 6),
        "mean_generated_tokens": round(sum(int(row["generated_tokens"]) for row in result_rows) / total, 6),
        "backend_type": args.backend_type,
        "tensor_parallel_size": tensor_parallel_size,
        "paper_decoding_reference": dict(PAPER_BASELINE_DECODING),
        "effective_decoding": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    dump_jsonl(rows_jsonl_path, result_rows)
    write_csv(rows_csv_path, result_rows)
    write_csv(subtask_summary_path, subtask_summary_rows)
    write_json(summary_json_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
