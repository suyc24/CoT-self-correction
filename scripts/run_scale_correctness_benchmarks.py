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
from statistics import median, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import extract_last_boxed
from cot_research.cot_accuracy import judge_single_answer, resolve_gold_answer
from cot_research.datasets import load_hf_dataset_records
from cot_research.generation import create_backend
from cot_research.head_intervention import INTERVENTION_REGISTRY, MultiLayerHeadIntervention, resolve_head_targets
from cot_research.io_utils import load_jsonl, write_csv, write_json
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig

DEFAULT_DATASETS = ["math500", "aime25", "minerva_math"]
DEFAULT_SCALES = [1.0, 1.4]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run scale-head correctness sweeps across math benchmarks."
    )
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--max_examples_per_dataset", type=int, default=1000)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--dataset_cache_dir", type=str, default=str(root_dir / "evaluation" / "data" / "temp"))
    parser.add_argument(
        "--dataset_local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only load Hugging Face benchmark datasets from local cache.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "scale_correctness_benchmarks_qwen3_1p7b"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--scales", type=str, default=",".join(str(x) for x in DEFAULT_SCALES))
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
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--keep_worker_outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=25)
    return parser.parse_args()


def parse_scale_list(text: str) -> List[float]:
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Scale list is empty.")
    return values


def parse_dataset_list(text: str) -> List[str]:
    values = [token.strip() for token in text.split(",") if token.strip()]
    if not values:
        raise ValueError("Dataset list is empty.")
    return values


def format_scale_tag(scale: float) -> str:
    text = f"{scale:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg").replace(".", "p")


def _shuffle_and_limit(rows: List[Dict[str, Any]], *, shuffle: bool, seed: int, limit: int) -> List[Dict[str, Any]]:
    out = list(rows)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(out)
    if limit > 0:
        out = out[: min(limit, len(out))]
    return out


def _convert_math500_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = str(row.get("problem") or row.get("question") or row.get("prompt") or "").strip()
    answer = row.get("answer")
    solution = row.get("solution")
    return {
        "example_id": f"math500:{idx}",
        "id": idx,
        "dataset_name": "math500",
        "source": "math500",
        "question": question,
        "problem": question,
        "correct_answer": answer,
        "gold_answer": answer,
        "reference_solution": solution,
        "metadata": {"dataset_name": "math500", "dataset_local_index": idx},
    }


def _convert_aime25_row(row: Dict[str, Any], idx: int, subset_name: str) -> Dict[str, Any]:
    question = str(row.get("problem") or row.get("question") or row.get("prompt") or "").strip()
    answer = row.get("answer") or row.get("final_answer") or row.get("solution")
    solution = row.get("solution")
    source = f"aime25_{subset_name.lower().replace(' ', '_').replace('-', '_')}"
    return {
        "example_id": f"{source}:{idx}",
        "id": idx,
        "dataset_name": "aime25",
        "source": source,
        "question": question,
        "problem": question,
        "correct_answer": answer,
        "gold_answer": answer,
        "reference_solution": solution,
        "metadata": {"dataset_name": "aime25", "subset_name": subset_name, "dataset_local_index": idx},
    }


def _convert_minerva_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = str(row.get("problem") or "").strip()
    solution = row.get("solution")
    gold_answer = resolve_gold_answer({"solution": solution})
    return {
        "example_id": f"minerva_math:{idx}",
        "id": idx,
        "dataset_name": "minerva_math",
        "source": str(row.get("type") or "minerva_math"),
        "question": question,
        "problem": question,
        "reference_solution": solution,
        "correct_answer": gold_answer,
        "gold_answer": gold_answer,
        "metadata": {
            "dataset_name": "minerva_math",
            "dataset_local_index": idx,
            "source": row.get("type"),
            "gold_answer": gold_answer,
        },
    }


def load_dataset_rows(dataset_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    dataset_name = dataset_name.strip().lower()
    if dataset_name == "math500":
        raw = load_hf_dataset_records(
            dataset_name="HuggingFaceH4/MATH-500",
            dataset_split="test",
            cache_dir=args.dataset_cache_dir,
            local_files_only=args.dataset_local_files_only,
        )
        rows = [_convert_math500_row(dict(row), idx) for idx, row in enumerate(raw)]
        rows = _shuffle_and_limit(rows, shuffle=args.shuffle, seed=args.seed, limit=args.max_examples_per_dataset)
        return {
            "dataset_name": "math500",
            "requested_examples": args.max_examples_per_dataset,
            "available_examples": len(raw),
            "selected_examples": len(rows),
            "rows": rows,
        }

    if dataset_name == "aime25":
        source_groups: List[List[Tuple[str, Optional[str], str, str]]] = [
            [
                ("opencompass/AIME2025", "AIME2025-I", "AIME2025-I", "test"),
                ("opencompass/AIME2025", "AIME2025-II", "AIME2025-II", "test"),
            ],
            [
                ("MathArena/aime_2025_I", None, "aime_2025_I", "train"),
                ("MathArena/aime_2025_II", None, "aime_2025_II", "train"),
            ],
        ]
        subset_rows: List[Dict[str, Any]] = []
        loaded_subsets: List[str] = []
        failures: List[str] = []
        for group in source_groups:
            group_rows: List[Dict[str, Any]] = []
            group_loaded: List[str] = []
            group_failed = False
            for dataset_id, config_name, subset_name, dataset_split in group:
                try:
                    raw = load_hf_dataset_records(
                        dataset_name=dataset_id,
                        config_name=config_name,
                        dataset_split=dataset_split,
                        cache_dir=args.dataset_cache_dir,
                        local_files_only=args.dataset_local_files_only,
                    )
                    converted = [_convert_aime25_row(dict(row), idx, subset_name) for idx, row in enumerate(raw)]
                    group_rows.extend(converted)
                    group_loaded.append(f"{dataset_id}:{config_name or 'default'}")
                except Exception as exc:
                    failures.append(f"{dataset_id}:{config_name or 'default'} -> {exc}")
                    group_failed = True
                    break
            if group_rows and not group_failed:
                subset_rows = group_rows
                loaded_subsets = group_loaded
                break
        # de-duplicate if both preferred and fallback sources loaded
        dedup: Dict[str, Dict[str, Any]] = {}
        for row in subset_rows:
            key = str(row.get("question", "")).strip() or str(row["example_id"])
            dedup[key] = row
        rows = list(dedup.values())
        rows = _shuffle_and_limit(rows, shuffle=args.shuffle, seed=args.seed, limit=args.max_examples_per_dataset)
        if not rows:
            raise RuntimeError("Failed to load aime25 from all candidate sources: " + " | ".join(failures))
        return {
            "dataset_name": "aime25",
            "requested_examples": args.max_examples_per_dataset,
            "available_examples": len(dedup),
            "selected_examples": len(rows),
            "loaded_subsets": loaded_subsets,
            "rows": rows,
        }

    if dataset_name == "minerva_math":
        path = ROOT_DIR / "evaluation" / "data" / "minerva_math" / "test.jsonl"
        raw = load_jsonl(str(path))
        rows = [_convert_minerva_row(dict(row), idx) for idx, row in enumerate(raw)]
        rows = _shuffle_and_limit(rows, shuffle=args.shuffle, seed=args.seed, limit=args.max_examples_per_dataset)
        return {
            "dataset_name": "minerva_math",
            "requested_examples": args.max_examples_per_dataset,
            "available_examples": len(raw),
            "selected_examples": len(rows),
            "rows": rows,
        }

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_tasks(
    dataset_infos: List[Dict[str, Any]],
    scales: Sequence[float],
    num_repeats: int,
    worker_count: int,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for info in dataset_infos:
        for repeat_idx in range(int(num_repeats)):
            for scale in scales:
                tasks.append(
                    {
                        "dataset_name": info["dataset_name"],
                        "scale": float(scale),
                        "repeat_idx": int(repeat_idx),
                        "rows": list(info["rows"]),
                    }
                )
    if worker_count <= len(tasks):
        return tasks

    while len(tasks) < worker_count:
        largest_idx = max(range(len(tasks)), key=lambda idx: len(tasks[idx]["rows"]))
        largest = tasks[largest_idx]
        if len(largest["rows"]) <= 1:
            break
        shard_a, shard_b = split_examples_contiguous(largest["rows"], 2)
        tasks[largest_idx] = {
            "dataset_name": largest["dataset_name"],
            "scale": largest["scale"],
            "repeat_idx": largest["repeat_idx"],
            "rows": list(shard_a),
        }
        tasks.insert(
            largest_idx + 1,
            {
                "dataset_name": largest["dataset_name"],
                "scale": largest["scale"],
                "repeat_idx": largest["repeat_idx"],
                "rows": list(shard_b),
            },
        )
    return tasks


def summarize_rows(
    rows: List[Dict[str, Any]],
    dataset_name: str,
    scale: float,
    *,
    repeat_idx: Optional[int] = None,
) -> Dict[str, Any]:
    generated = [int(row["generated_tokens"]) for row in rows]
    count = len(rows)
    verifiable = sum(1 for row in rows if str(row["accuracy_category"]) != "unverifiable")
    correct = sum(1 for row in rows if bool(row["is_correct"]))
    hit_max = sum(1 for row in rows if bool(row["hit_max_new_tokens"]))
    total_tokens = sum(generated)
    return {
        "dataset_name": dataset_name,
        "scale": scale,
        "repeat_idx": repeat_idx,
        "example_count": count,
        "verifiable_examples": verifiable,
        "correct_examples": correct,
        "correct_rate_over_verifiable": round(correct / verifiable, 6) if verifiable else 0.0,
        "mean_generated_tokens": round(total_tokens / count, 6) if count else 0.0,
        "median_generated_tokens": float(median(generated)) if generated else 0.0,
        "hit_max_new_tokens_rate": round(hit_max / count, 6) if count else 0.0,
    }


def aggregate_repeat_summaries(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["dataset_name"]), float(row["scale"]))
        grouped.setdefault(key, []).append(row)

    merged: List[Dict[str, Any]] = []
    for (dataset_name, scale), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        correct_rates = [float(row["correct_rate_over_verifiable"]) for row in rows]
        mean_tokens = [float(row["mean_generated_tokens"]) for row in rows]
        median_tokens = [float(row["median_generated_tokens"]) for row in rows]
        hit_max_rates = [float(row["hit_max_new_tokens_rate"]) for row in rows]
        example_counts = [int(row["example_count"]) for row in rows]
        verifiable_counts = [int(row["verifiable_examples"]) for row in rows]
        correct_counts = [int(row["correct_examples"]) for row in rows]
        merged.append(
            {
                "dataset_name": dataset_name,
                "scale": float(scale),
                "num_repeats": len(rows),
                "example_count_mean": round(sum(example_counts) / len(example_counts), 6) if rows else 0.0,
                "verifiable_examples_mean": round(sum(verifiable_counts) / len(verifiable_counts), 6) if rows else 0.0,
                "correct_examples_mean": round(sum(correct_counts) / len(correct_counts), 6) if rows else 0.0,
                "correct_rate_over_verifiable_mean": round(sum(correct_rates) / len(correct_rates), 6) if rows else 0.0,
                "correct_rate_over_verifiable_std": round(pstdev(correct_rates), 6) if len(correct_rates) > 1 else 0.0,
                "mean_generated_tokens_mean": round(sum(mean_tokens) / len(mean_tokens), 6) if rows else 0.0,
                "mean_generated_tokens_std": round(pstdev(mean_tokens), 6) if len(mean_tokens) > 1 else 0.0,
                "median_generated_tokens_mean": round(sum(median_tokens) / len(median_tokens), 6) if rows else 0.0,
                "hit_max_new_tokens_rate_mean": round(sum(hit_max_rates) / len(hit_max_rates), 6) if rows else 0.0,
                "hit_max_new_tokens_rate_std": round(pstdev(hit_max_rates), 6) if len(hit_max_rates) > 1 else 0.0,
            }
        )
    return merged


def run_worker(
    worker_id: int,
    gpu_id: int,
    task: Dict[str, Any],
    args_dict: Dict[str, Any],
    worker_rows_path: str,
    worker_summary_path: str,
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
    operations = INTERVENTION_REGISTRY.get_required("scale")(targets, {"scale": task["scale"]})

    dataset_name = str(task["dataset_name"])
    scale = float(task["scale"])
    repeat_idx = int(task.get("repeat_idx", 0))
    rows = list(task["rows"])
    out_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    iterator = tqdm(
        rows,
        desc=f"{dataset_name} repeat={repeat_idx} scale={format_scale_tag(scale)} gpu={gpu_id}",
        dynamic_ncols=True,
        leave=False,
    )
    for idx, row in enumerate(iterator):
        try:
            prompt_prefix = backend.build_prompt(str(row["question"]), generation_config)
        except Exception as exc:
            skipped.append({"dataset_name": dataset_name, "scale": scale, "example_id": row.get("example_id"), "reason": str(exc)})
            continue
        seed_everything(args.seed + repeat_idx * 100000 + idx)
        with MultiLayerHeadIntervention(attn_modules, operations):
            generation = backend.generate(prompt_prefix, generation_config)
        final_answer = extract_last_boxed(generation.continuation)
        accuracy = judge_single_answer(
            final_answer=final_answer,
            gold_answer=resolve_gold_answer(row),
        )
        out_row = {
            "dataset_name": dataset_name,
            "repeat_idx": repeat_idx,
            "example_id": row.get("example_id") or row.get("id") or idx,
            "source": row.get("source"),
            "scale": scale,
            "generated_tokens": int(generation.generated_tokens),
            "final_answer": accuracy.get("final_answer"),
            "gold_answer": accuracy.get("gold_answer"),
            "is_correct": accuracy.get("is_correct"),
            "accuracy_category": accuracy.get("category"),
            "hit_max_new_tokens": bool(generation.generated_tokens >= args.max_new_tokens),
        }
        out_rows.append(out_row)
        if (idx + 1) % max(args.print_every, 1) == 0:
            print(
                f"[Info] dataset={dataset_name} repeat={repeat_idx} scale={scale} gpu={gpu_id} processed={idx + 1} kept={len(out_rows)} skipped={len(skipped)}"
            )

    write_csv(worker_rows_path, out_rows)
    summary = summarize_rows(out_rows, dataset_name, scale, repeat_idx=repeat_idx)
    summary["gpu_id"] = gpu_id
    summary["skipped_count"] = len(skipped)
    summary["decoder_layer_path"] = layer_path
    write_json(worker_summary_path, {"summary": summary, "skipped": skipped})
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "dataset_name": dataset_name,
        "scale": scale,
        "repeat_idx": repeat_idx,
        "row_count": len(out_rows),
        "skipped_count": len(skipped),
        "worker_rows_path": worker_rows_path,
        "worker_summary_path": worker_summary_path,
        "decoder_layer_path": layer_path,
    }


def main() -> None:
    args = parse_args()
    dataset_names = parse_dataset_list(args.datasets)
    scales = parse_scale_list(args.scales)

    dataset_infos = [load_dataset_rows(name, args) for name in dataset_names]
    total_examples = sum(len(info["rows"]) for info in dataset_infos)
    if total_examples == 0:
        raise ValueError("No examples selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    rows_path = output_dir / "rows.csv"
    per_repeat_summary_csv_path = output_dir / "per_repeat_summary.csv"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"

    available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    if not available_gpu_ids:
        raise ValueError("This experiment requires explicit GPU ids or visible CUDA devices.")

    max_parallel_tasks = sum(max(1, len(info["rows"]) * len(scales) * int(args.num_repeats)) for info in dataset_infos)
    if args.parallel_workers > 0:
        worker_count = min(args.parallel_workers, len(available_gpu_ids), max_parallel_tasks)
    else:
        worker_count = min(len(available_gpu_ids), max_parallel_tasks)
    if worker_count <= 0:
        raise ValueError("No worker GPU available.")

    tasks = build_tasks(dataset_infos, scales, int(args.num_repeats), worker_count)
    worker_count = min(worker_count, len(tasks))
    selected_gpu_ids = available_gpu_ids[:worker_count]

    write_json(
        run_config_path,
        {
            "args": vars(args),
            "datasets": [
                {key: value for key, value in info.items() if key != "rows"}
                for info in dataset_infos
            ],
            "scales": scales,
            "selected_gpu_ids": selected_gpu_ids,
            "worker_tasks": [
                {
                    "dataset_name": task["dataset_name"],
                    "repeat_idx": task.get("repeat_idx", 0),
                    "scale": task["scale"],
                    "example_count": len(task["rows"]),
                }
                for task in tasks
            ],
        },
    )

    print(
        "[Info] Running scale benchmark correctness sweep: "
        f"model={args.model_name_or_path}, head={args.head_label}, datasets={dataset_names}, scales={scales}, "
        f"selected_examples={total_examples}, num_repeats={args.num_repeats}, gpus={selected_gpu_ids}, max_new_tokens={args.max_new_tokens}, "
        f"do_sample={args.do_sample}, temperature={args.temperature}"
    )

    worker_rows_paths: List[Path] = []
    worker_summary_paths: List[Path] = []
    worker_returns: List[Dict[str, Any]] = []
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as pool:
        futures = []
        for worker_id, task in enumerate(tasks):
            gpu_id = selected_gpu_ids[worker_id % worker_count]
            dataset_tag = str(task["dataset_name"])
            scale_tag = format_scale_tag(float(task["scale"]))
            repeat_tag = f"repeat_{int(task.get('repeat_idx', 0)):02d}"
            worker_rows_path = output_dir / f"worker_{worker_id}_{dataset_tag}_{repeat_tag}_scale_{scale_tag}_rows.csv"
            worker_summary_path = output_dir / f"worker_{worker_id}_{dataset_tag}_{repeat_tag}_scale_{scale_tag}_summary.json"
            futures.append(
                pool.submit(
                    run_worker,
                    worker_id,
                    gpu_id,
                    task,
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
                f"[Info] Finished worker gpu={worker_ret['gpu_id']} dataset={worker_ret['dataset_name']} repeat={worker_ret['repeat_idx']} scale={worker_ret['scale']} rows={worker_ret['row_count']} skipped={worker_ret['skipped_count']}"
            )

    all_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    decoder_layer_path = ""
    for worker_ret in worker_returns:
        with open(worker_ret["worker_rows_path"], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append(
                    {
                        "dataset_name": row["dataset_name"],
                        "repeat_idx": int(row["repeat_idx"]) if row.get("repeat_idx", "") not in {"", "None"} else 0,
                        "example_id": row["example_id"],
                        "source": row["source"],
                        "scale": float(row["scale"]),
                        "generated_tokens": int(row["generated_tokens"]),
                        "final_answer": row["final_answer"],
                        "gold_answer": row["gold_answer"],
                        "is_correct": row["is_correct"].lower() == "true" if row["is_correct"] not in {"", "None"} else None,
                        "accuracy_category": row["accuracy_category"],
                        "hit_max_new_tokens": row["hit_max_new_tokens"].lower() == "true",
                    }
                )
        summary_blob = json.loads(Path(worker_ret["worker_summary_path"]).read_text(encoding="utf-8"))
        summary = dict(summary_blob["summary"])
        summary_rows.append(summary)
        if not decoder_layer_path:
            decoder_layer_path = str(summary.get("decoder_layer_path") or "")

    all_rows = sorted(all_rows, key=lambda item: (item["dataset_name"], int(item["repeat_idx"]), float(item["scale"]), str(item["example_id"])))

    per_repeat_summary_rows: List[Dict[str, Any]] = []
    for dataset_name in dataset_names:
        for repeat_idx in range(int(args.num_repeats)):
            for scale in scales:
                scale_rows = [
                    row for row in all_rows
                    if row["dataset_name"] == dataset_name
                    and int(row["repeat_idx"]) == int(repeat_idx)
                    and float(row["scale"]) == float(scale)
                ]
                if not scale_rows:
                    continue
                summary = summarize_rows(scale_rows, dataset_name, float(scale), repeat_idx=repeat_idx)
                summary["gpu_id"] = -1
                summary["skipped_count"] = sum(
                    int(item.get("skipped_count", 0))
                    for item in summary_rows
                    if item.get("dataset_name") == dataset_name
                    and int(item.get("repeat_idx", -1)) == int(repeat_idx)
                    and float(item.get("scale", -999)) == float(scale)
                )
                summary["decoder_layer_path"] = decoder_layer_path
                per_repeat_summary_rows.append(summary)

    merged_summary_rows = aggregate_repeat_summaries(per_repeat_summary_rows)

    baseline_by_dataset = {
        row["dataset_name"]: row
        for row in merged_summary_rows
        if abs(float(row["scale"]) - 1.0) < 1e-9
    }
    for row in merged_summary_rows:
        baseline = baseline_by_dataset.get(row["dataset_name"])
        if baseline is None:
            continue
        row["delta_correct_rate_vs_scale1.0_pp"] = round(
            100 * (float(row["correct_rate_over_verifiable_mean"]) - float(baseline["correct_rate_over_verifiable_mean"])), 4
        )
        row["delta_mean_tokens_vs_scale1.0"] = round(
            float(row["mean_generated_tokens_mean"]) - float(baseline["mean_generated_tokens_mean"]), 3
        )

    write_csv(rows_path, all_rows)
    write_csv(per_repeat_summary_csv_path, per_repeat_summary_rows)
    write_csv(summary_csv_path, merged_summary_rows)
    write_json(
        summary_json_path,
        {
            "model_name_or_path": args.model_name_or_path,
            "head_label": args.head_label,
            "datasets": [
                {key: value for key, value in info.items() if key != "rows"}
                for info in dataset_infos
            ],
            "scales": scales,
            "num_repeats": int(args.num_repeats),
            "selected_gpu_ids": selected_gpu_ids,
            "worker_tasks": [
                {
                    "dataset_name": task["dataset_name"],
                    "repeat_idx": task.get("repeat_idx", 0),
                    "scale": task["scale"],
                    "example_count": len(task["rows"]),
                }
                for task in tasks
            ],
            "decoder_layer_path": decoder_layer_path,
            "summary_by_dataset_scale_repeat": per_repeat_summary_rows,
            "summary_by_dataset_scale": merged_summary_rows,
        },
    )

    if not args.keep_worker_outputs:
        for path in worker_rows_paths + worker_summary_paths:
            if path.exists():
                path.unlink()

    print("[Done] Outputs written:")
    print(f"- rows_csv: {rows_path}")
    print(f"- per_repeat_summary_csv: {per_repeat_summary_csv_path}")
    print(f"- summary_csv: {summary_csv_path}")
    print(f"- summary_json: {summary_json_path}")


if __name__ == "__main__":
    main()
