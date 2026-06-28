#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import extract_last_boxed
from cot_research.cot_accuracy import judge_single_answer, resolve_gold_answer
from cot_research.datasets import convert_numinamath_row, load_numinamath_dataset
from cot_research.generation import create_backend
from cot_research.head_intervention import INTERVENTION_REGISTRY, MultiLayerHeadIntervention, resolve_head_targets
from cot_research.io_utils import load_jsonl, write_csv, write_json
from cot_research.repetition_analysis import RepetitionThresholds, analyze_repetition
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig
from cot_research.text_analysis import analyze_text_keywords, extract_continuation_think_text


DEFAULT_SCALES = [0.5, 1.0, 1.2, 1.4, 1.6, 2.0]
DEFAULT_REFLECTION_KEYWORDS = [
    "wait",
    "hold on",
    "let me check",
    "let me double check",
    "let me think",
    "i should check",
    "i need to check",
    "actually",
    "on second thought",
]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run scale-L0H3 experiments and summarize CoT length, reflection frequency, repetition frequency, and answer correctness"
    )
    parser.add_argument(
        "--input_source",
        type=str,
        default="numinamath",
        choices=["jsonl", "numinamath"],
        help="Load rows either from an existing JSONL or directly from AI-MO/NuminaMath-CoT.",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "l0h3_scale_wait_length_qwen3_1p7b"),
    )
    parser.add_argument("--dataset_name", type=str, default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_cache_dir", type=str, default=str(root_dir / "evaluation" / "data" / "temp"))
    parser.add_argument(
        "--dataset_local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only load NuminaMath from local cache.",
    )
    parser.add_argument(
        "--source_filter",
        type=str,
        default="",
        help="Optional comma-separated NuminaMath source filter.",
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--scales", type=str, default=",".join(str(x) for x in DEFAULT_SCALES))
    parser.add_argument(
        "--reflection_keywords",
        type=str,
        default=",".join(DEFAULT_REFLECTION_KEYWORDS),
        help="Comma-separated reflection lexicon matched inside continuation think text.",
    )
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--parallel_gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU ids. Scales are distributed round-robin across these GPUs.",
    )
    parser.add_argument("--parallel_workers", type=int, default=0)
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
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print_every", type=int, default=25)

    parser.add_argument("--same_token_run_threshold", type=int, default=24)
    parser.add_argument("--tail_repeat_min_repeats", type=int, default=4)
    parser.add_argument("--tail_repeat_max_ngram", type=int, default=32)
    parser.add_argument("--line_repeat_threshold", type=int, default=3)
    parser.add_argument("--word_tail_repeat_min_repeats", type=int, default=4)
    parser.add_argument("--word_tail_repeat_max_ngram", type=int, default=24)
    parser.add_argument("--min_trigger_count", type=int, default=1)
    return parser.parse_args()


def parse_scale_list(text: str) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Scale list is empty.")
    return values


def parse_keyword_list(text: str) -> List[str]:
    values = [token.strip() for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Reflection keyword list is empty.")
    return values


def format_scale_tag(scale: float) -> str:
    text = f"{scale:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def make_thresholds(args: argparse.Namespace) -> RepetitionThresholds:
    return RepetitionThresholds(
        same_token_run_threshold=args.same_token_run_threshold,
        tail_repeat_min_repeats=args.tail_repeat_min_repeats,
        tail_repeat_max_ngram=args.tail_repeat_max_ngram,
        line_repeat_threshold=args.line_repeat_threshold,
        word_tail_repeat_min_repeats=args.word_tail_repeat_min_repeats,
        word_tail_repeat_max_ngram=args.word_tail_repeat_max_ngram,
        min_trigger_count=args.min_trigger_count,
    )


def select_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if args.input_source == "jsonl":
        rows = load_jsonl(args.input_jsonl)
        rows = rows[args.start_idx :]
        if args.shuffle:
            rng = random.Random(args.seed)
            rng.shuffle(rows)
        if args.max_examples > 0:
            rows = rows[: args.max_examples]
        if not rows:
            raise ValueError("No rows selected for the experiment.")
        return rows, {
            "input_source": "jsonl",
            "dataset_name": None,
            "dataset_split": None,
            "source_filter": None,
        }

    dataset = load_numinamath_dataset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        local_files_only=args.dataset_local_files_only,
        source_filter=args.source_filter,
        shuffle=args.shuffle,
        seed=args.seed,
        start_idx=args.start_idx,
        sample_stride=1,
        max_examples=args.max_examples,
    )
    rows: List[Dict[str, Any]] = []
    for local_index, raw_row in enumerate(dataset):
        rows.append(
            convert_numinamath_row(
                dict(raw_row),
                args.start_idx + local_index,
                dataset_name=args.dataset_name,
                dataset_split=args.dataset_split,
            )
        )
    if not rows:
        raise ValueError("No NuminaMath rows selected for the experiment.")
    return rows, {
        "input_source": "numinamath",
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "source_filter": args.source_filter,
    }


def split_scales_round_robin(scales: Sequence[float], num_workers: int) -> List[List[float]]:
    buckets: List[List[float]] = [[] for _ in range(num_workers)]
    for idx, scale in enumerate(scales):
        buckets[idx % num_workers].append(scale)
    return [bucket for bucket in buckets if bucket]


def build_worker_tasks(
    scales: Sequence[float],
    rows: Sequence[Dict[str, Any]],
    worker_count: int,
) -> List[Dict[str, Any]]:
    if worker_count <= 0:
        return []
    if not scales:
        return []
    if worker_count <= len(scales):
        scale_buckets = split_scales_round_robin(scales, worker_count)
        return [{"scales": list(bucket), "rows": list(rows)} for bucket in scale_buckets]

    per_scale_task_counts = [1 for _ in scales]
    extra = worker_count - len(scales)
    for idx in range(extra):
        per_scale_task_counts[idx % len(scales)] += 1

    tasks: List[Dict[str, Any]] = []
    for scale, task_count in zip(scales, per_scale_task_counts):
        row_shards = split_examples_contiguous(rows, task_count)
        for shard in row_shards:
            tasks.append(
                {
                    "scales": [float(scale)],
                    "rows": list(shard),
                }
            )
    return tasks


def analyze_reflection(continuation: str, reflection_keywords: Sequence[str]) -> Dict[str, Any]:
    think_text = extract_continuation_think_text(continuation)
    stats = analyze_text_keywords(think_text, reflection_keywords)
    return {
        "reflection_count": int(stats["count"]),
        "has_reflection": bool(stats["hit"]),
        "matched_keywords": list(stats["matched_keywords"]),
    }


def summarize_scale_rows(
    rows: List[Dict[str, Any]],
    scale: float,
    gpu_id: int,
    skipped_count: int,
) -> Dict[str, Any]:
    generated = [int(row["generated_tokens"]) for row in rows]
    reflection_counts = [int(row["reflection_count"]) for row in rows]
    reflection_hits = sum(1 for row in rows if bool(row["has_reflection"]))
    repetition_hits = sum(1 for row in rows if bool(row["is_repetitive"]))
    hit_max = sum(1 for row in rows if bool(row["hit_max_new_tokens"]))
    total_tokens = sum(generated)
    total_reflection = sum(reflection_counts)
    verifiable = sum(1 for row in rows if str(row["accuracy_category"]) != "unverifiable")
    correct = sum(1 for row in rows if bool(row["is_correct"]))
    keyword_counter: Counter[str] = Counter()
    for row in rows:
        keyword_counter.update(row.get("matched_keywords") or [])
    count = len(rows)
    return {
        "scale": scale,
        "gpu_id": gpu_id,
        "example_count": count,
        "skipped_count": skipped_count,
        "verifiable_examples": verifiable,
        "correct_examples": correct,
        "correct_rate_over_verifiable": round(correct / verifiable, 6) if verifiable else 0.0,
        "mean_generated_tokens": round(total_tokens / count, 6) if count else 0.0,
        "median_generated_tokens": float(median(generated)) if generated else 0.0,
        "min_generated_tokens": min(generated) if generated else 0,
        "max_generated_tokens": max(generated) if generated else 0,
        "mean_reflection_count": round(total_reflection / count, 6) if count else 0.0,
        "median_reflection_count": float(median(reflection_counts)) if reflection_counts else 0.0,
        "total_reflection_count": total_reflection,
        "reflection_hit_rate": round(reflection_hits / count, 6) if count else 0.0,
        "reflection_per_1k_tokens": round((total_reflection / total_tokens) * 1000.0, 6) if total_tokens else 0.0,
        "repetition_hit_rate": round(repetition_hits / count, 6) if count else 0.0,
        "hit_max_new_tokens_rate": round(hit_max / count, 6) if count else 0.0,
        "matched_keyword_counts": dict(sorted(keyword_counter.items())),
    }


def run_scale_worker(
    worker_id: int,
    gpu_id: int,
    scales: List[float],
    rows: List[Dict[str, Any]],
    reflection_keywords: List[str],
    args_dict: Dict[str, Any],
    worker_rows_path: str,
    worker_summary_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    thresholds = make_thresholds(args)
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
        raise ValueError("This script requires an HF backend with intervention support.")

    generation_config = GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        capture_step_scores=False,
    )

    targets, attn_modules, layer_path = resolve_head_targets(backend.model, [args.head_label])

    result_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for scale in scales:
        operations = INTERVENTION_REGISTRY.get_required("scale")(targets, {"scale": scale})
        scale_rows: List[Dict[str, Any]] = []
        scale_tag = format_scale_tag(scale)
        iterator = tqdm(rows, desc=f"scale={scale_tag} gpu={gpu_id}", dynamic_ncols=True, leave=False)
        for idx, row in enumerate(iterator):
            try:
                prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
            except Exception as exc:
                skipped.append(
                    {
                        "example_id": row.get("id") or row.get("example_id") or idx,
                        "scale": scale,
                        "reason": str(exc),
                    }
                )
                continue

            seed_everything(args.seed + idx)
            with MultiLayerHeadIntervention(attn_modules, operations):
                generation = backend.generate(prompt_prefix, generation_config)

            reflection_stats = analyze_reflection(generation.continuation, reflection_keywords)
            repetition_detection = analyze_repetition(
                generation.continuation,
                token_ids=generation.token_ids,
                existing_repetition=None,
                thresholds=thresholds,
            )
            final_boxed_answer = extract_last_boxed(generation.continuation)
            accuracy = judge_single_answer(
                final_answer=final_boxed_answer,
                gold_answer=resolve_gold_answer(row),
            )
            row_out = {
                "example_id": row.get("example_id") or row.get("id") or idx,
                "scale": scale,
                "generated_tokens": int(generation.generated_tokens),
                "reflection_count": int(reflection_stats["reflection_count"]),
                "has_reflection": bool(reflection_stats["has_reflection"]),
                "matched_keywords": list(reflection_stats["matched_keywords"]),
                "is_repetitive": bool(repetition_detection["matched"]),
                "final_answer": accuracy.get("final_answer"),
                "gold_answer": accuracy.get("gold_answer"),
                "is_correct": accuracy.get("is_correct"),
                "accuracy_category": accuracy.get("category"),
                "hit_max_new_tokens": bool(generation.generated_tokens >= args.max_new_tokens),
            }
            scale_rows.append(row_out)
            result_rows.append(
                {
                    **row_out,
                    "matched_keywords_json": json.dumps(row_out["matched_keywords"], ensure_ascii=False),
                }
            )
            if (idx + 1) % max(args.print_every, 1) == 0:
                print(
                    f"[Info] scale={scale} gpu={gpu_id} processed={idx + 1} "
                    f"kept={len(scale_rows)} skipped={len(skipped)}"
                )

        summary = summarize_scale_rows(
            scale_rows,
            scale,
            gpu_id,
            sum(1 for item in skipped if float(item["scale"]) == float(scale)),
        )
        summary["decoder_layer_path"] = layer_path
        summary_rows.append(summary)

    write_csv(worker_rows_path, result_rows)
    write_json(worker_summary_path, {"summary_rows": summary_rows, "skipped": skipped})
    return {
        "scales": scales,
        "gpu_id": gpu_id,
        "row_count": len(result_rows),
        "skipped_count": len(skipped),
        "worker_rows_path": worker_rows_path,
        "worker_summary_path": worker_summary_path,
        "decoder_layer_path": layer_path,
    }


def main() -> None:
    args = parse_args()
    scales = parse_scale_list(args.scales)
    reflection_keywords = parse_keyword_list(args.reflection_keywords)
    rows, input_metadata = select_rows(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    rows_path = output_dir / "rows.csv"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"

    available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    if not available_gpu_ids:
        raise ValueError("This experiment requires explicit GPU ids or visible CUDA devices.")

    max_parallel_tasks = max(1, len(scales) * max(1, len(rows)))
    if args.parallel_workers > 0:
        worker_count = min(args.parallel_workers, len(available_gpu_ids), max_parallel_tasks)
    else:
        worker_count = min(len(available_gpu_ids), max_parallel_tasks)
    if worker_count <= 0:
        raise ValueError("No worker GPU available.")

    worker_tasks = build_worker_tasks(scales, rows, worker_count)
    worker_count = min(worker_count, len(worker_tasks))
    selected_gpu_ids = available_gpu_ids[:worker_count]
    worker_tasks = worker_tasks[:worker_count]
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "input_metadata": input_metadata,
            "scales": scales,
            "reflection_keywords": reflection_keywords,
            "selected_gpu_ids": selected_gpu_ids,
            "selected_example_count": len(rows),
            "worker_tasks": [
                {
                    "scales": task["scales"],
                    "example_count": len(task["rows"]),
                }
                for task in worker_tasks
            ],
        },
    )

    print(
        "[Info] Running scale experiment: "
        f"model={args.model_name_or_path}, head={args.head_label}, scales={scales}, "
        f"examples={len(rows)}, gpus={selected_gpu_ids}, max_new_tokens={args.max_new_tokens}, "
        f"do_sample={args.do_sample}, temperature={args.temperature}, "
        f"reflection_keywords={reflection_keywords}, input_source={input_metadata['input_source']}"
    )

    worker_rows_paths: List[Path] = []
    worker_summary_paths: List[Path] = []
    worker_returns: List[Dict[str, Any]] = []
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(worker_tasks), mp_context=mp_ctx) as pool:
        futures = []
        for worker_id, (task, gpu_id) in enumerate(zip(worker_tasks, selected_gpu_ids)):
            scale_bucket = task["scales"]
            row_shard = task["rows"]
            bucket_tag = "_".join(format_scale_tag(scale) for scale in scale_bucket)
            worker_rows_path = output_dir / f"worker_{worker_id}_scales_{bucket_tag}_rows.csv"
            worker_summary_path = output_dir / f"worker_{worker_id}_scales_{bucket_tag}_summary.json"
            futures.append(
                pool.submit(
                    run_scale_worker,
                    worker_id,
                    gpu_id,
                    scale_bucket,
                    row_shard,
                    reflection_keywords,
                    vars(args),
                    str(worker_rows_path),
                    str(worker_summary_path),
                )
            )
        for fut in as_completed(futures):
            worker_ret = fut.result()
            worker_returns.append(worker_ret)
            worker_rows_paths.append(Path(worker_ret["worker_rows_path"]))
            worker_summary_paths.append(Path(worker_ret["worker_summary_path"]))
            print(
                f"[Info] Finished worker gpu={worker_ret['gpu_id']} scales={worker_ret['scales']} "
                f"rows={worker_ret['row_count']} skipped={worker_ret['skipped_count']}"
            )

    all_rows: List[Dict[str, Any]] = []
    decoder_layer_path = ""
    skipped_counts_by_scale: Dict[float, int] = {float(scale): 0 for scale in scales}
    for worker_ret in worker_returns:
        with open(worker_ret["worker_rows_path"], "r", encoding="utf-8") as f:
            import csv

            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(
                    {
                        "example_id": row["example_id"],
                        "scale": float(row["scale"]),
                        "generated_tokens": int(row["generated_tokens"]),
                        "reflection_count": int(row["reflection_count"]),
                        "has_reflection": row["has_reflection"].lower() == "true",
                        "matched_keywords": json.loads(row["matched_keywords_json"]),
                        "is_repetitive": row["is_repetitive"].lower() == "true",
                        "final_answer": row["final_answer"],
                        "gold_answer": row["gold_answer"],
                        "is_correct": row["is_correct"].lower() == "true" if row["is_correct"] not in {"", "None"} else None,
                        "accuracy_category": row["accuracy_category"],
                        "hit_max_new_tokens": row["hit_max_new_tokens"].lower() == "true",
                    }
                )
        summary_blob = json.loads(Path(worker_ret["worker_summary_path"]).read_text(encoding="utf-8"))
        for skipped in summary_blob.get("skipped", []):
            try:
                skipped_counts_by_scale[float(skipped["scale"])] += 1
            except Exception:
                pass
        if not decoder_layer_path and summary_blob.get("summary_rows"):
            decoder_layer_path = str(summary_blob["summary_rows"][0].get("decoder_layer_path") or "")

    all_rows = sorted(all_rows, key=lambda item: (float(item["scale"]), str(item["example_id"])))
    summary_rows = []
    for scale in sorted(scales):
        scale_value = float(scale)
        scale_rows = [row for row in all_rows if float(row["scale"]) == scale_value]
        summary = summarize_scale_rows(
            scale_rows,
            scale_value,
            -1,
            skipped_counts_by_scale.get(scale_value, 0),
        )
        summary["decoder_layer_path"] = decoder_layer_path
        summary_rows.append(summary)
    write_csv(rows_path, all_rows)
    write_csv(summary_csv_path, summary_rows)
    write_json(
        summary_json_path,
        {
            "model_name_or_path": args.model_name_or_path,
            "head_label": args.head_label,
            "scales": scales,
            "reflection_keywords": reflection_keywords,
            "input_metadata": input_metadata,
            "example_count": len(rows),
            "selected_gpu_ids": selected_gpu_ids,
            "worker_tasks": [
                {
                    "scales": task["scales"],
                    "example_count": len(task["rows"]),
                }
                for task in worker_tasks
            ],
            "decoder_layer_path": decoder_layer_path,
            "summary_by_scale": summary_rows,
        },
    )

    if not args.keep_worker_outputs:
        for path in worker_rows_paths + worker_summary_paths:
            if path.exists():
                path.unlink()

    print("[Done] Outputs written:")
    print(f"- rows_csv: {rows_path}")
    print(f"- summary_csv: {summary_csv_path}")
    print(f"- summary_json: {summary_json_path}")


if __name__ == "__main__":
    main()
