#!/usr/bin/env python3
from __future__ import annotations

"""Compare baseline vs boosted-head generation on accuracy and repetition.

Typical usage:
    conda run -n qwen_math python scripts/test_head_boost_effects.py \
      --input_source numinamath \
      --head_labels L0H3,L0H7 \
      --scale 1.2 \
      --max_examples 1000 \
      --parallel_gpu_ids 0,1,2,3,4,5,6,7,8 \
      --parallel_workers 9
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, IO, List, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import extract_last_boxed
from cot_research.attention_sink_analysis import parse_comma_list
from cot_research.cot_accuracy import judge_comparison_row, summarize_comparison_accuracy
from cot_research.datasets import convert_numinamath_row, load_numinamath_dataset
from cot_research.generation import create_backend
from cot_research.head_intervention import INTERVENTION_REGISTRY, MultiLayerHeadIntervention, resolve_head_targets
from cot_research.io_utils import load_jsonl, truncate_text
from cot_research.repetition_analysis import RepetitionThresholds, analyze_repetition, select_continuation_text
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, resolve_device_map, seed_everything, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig
from cot_research.summary_utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    default_input = root_dir / "outputs" / "numinamath_qwen3_1p7b_cot.jsonl"
    default_output = root_dir / "outputs" / "head_boost_effects" / "qwen3_1p7b"

    parser = argparse.ArgumentParser(description="Compare boosted attention heads against baseline on accuracy and repetition")
    parser.add_argument(
        "--input_source",
        type=str,
        default="numinamath",
        choices=["jsonl", "numinamath"],
        help="Load evaluation rows either from an existing JSONL or directly from AI-MO/NuminaMath-CoT.",
    )
    parser.add_argument("--input_jsonl", type=str, default=str(default_input))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--dataset_name", type=str, default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_cache_dir", type=str, default=str(root_dir / "evaluation" / "data" / "temp"))
    parser.add_argument(
        "--dataset_local_files_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only load the dataset from local cache.",
    )
    parser.add_argument(
        "--source_filter",
        type=str,
        default="",
        help="Optional comma-separated dataset source filter when --input_source=numinamath.",
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=-1, help="If set >=0, place the whole model on this GPU.")
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-process multi-GPU execution by sharding examples across GPUs.",
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
        help="Number of worker processes in parallel mode. 0 means one worker per selected GPU.",
    )
    parser.add_argument(
        "--keep_worker_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep intermediate worker result files under output_dir.",
    )
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--head_labels", type=str, default="L0H3")
    parser.add_argument(
        "--intervention_kind",
        type=str,
        default="scale",
        choices=list(INTERVENTION_REGISTRY.names()),
    )
    parser.add_argument("--scale", type=float, default=1.2, help="Used when --intervention_kind=scale.")
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default="rerun",
        choices=["stored", "rerun"],
        help="Rerun baseline by default so max_new_tokens is controlled consistently. Use stored only with --input_source=jsonl.",
    )
    parser.add_argument("--preview_examples", type=int, default=20)
    parser.add_argument("--preview_chars", type=int, default=1200)
    parser.add_argument("--save_token_ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=10)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--same_token_run_threshold", type=int, default=24)
    parser.add_argument("--tail_repeat_min_repeats", type=int, default=4)
    parser.add_argument("--tail_repeat_max_ngram", type=int, default=32)
    parser.add_argument("--line_repeat_threshold", type=int, default=3)
    parser.add_argument("--word_tail_repeat_min_repeats", type=int, default=4)
    parser.add_argument("--word_tail_repeat_max_ngram", type=int, default=24)
    parser.add_argument("--min_trigger_count", type=int, default=1)
    return parser.parse_args()

def load_experiment_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if args.input_source == "jsonl":
        input_path = Path(args.input_jsonl)
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {input_path}")
        rows = load_jsonl(input_path)
        metadata = {
            "input_source": "jsonl",
            "input_jsonl": str(input_path),
            "dataset_name": None,
            "dataset_split": None,
        }
        return rows, metadata

    dataset = load_numinamath_dataset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
        local_files_only=args.dataset_local_files_only,
        source_filter=args.source_filter,
        shuffle=False,
        seed=args.seed,
        start_idx=0,
        sample_stride=1,
        max_examples=-1,
    )
    rows: List[Dict[str, Any]] = []
    for local_index, raw_row in enumerate(dataset):
        rows.append(
            convert_numinamath_row(
                dict(raw_row),
                local_index,
                dataset_name=args.dataset_name,
                dataset_split=args.dataset_split,
            )
        )
    metadata = {
        "input_source": "numinamath",
        "input_jsonl": "",
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
    }
    return rows, metadata

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

def build_condition_payload(
    *,
    label: str,
    continuation: str,
    full_text: str,
    generated_tokens: int,
    token_ids: List[int] | None,
    existing_repetition: Dict[str, Any] | None,
    thresholds: RepetitionThresholds,
    save_token_ids: bool,
    max_new_tokens: int,
    debug: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    detection = analyze_repetition(
        continuation,
        token_ids=token_ids,
        existing_repetition=existing_repetition,
        thresholds=thresholds,
    )
    payload: Dict[str, Any] = {
        "label": label,
        "continuation": continuation,
        "full_text": full_text,
        "generated_tokens": generated_tokens,
        "hit_max_new_tokens": bool(generated_tokens >= max_new_tokens),
        "final_boxed_answer": extract_last_boxed(continuation),
        "repetition_detection": detection,
        "debug": debug or {},
    }
    if save_token_ids:
        payload["token_ids"] = token_ids or []
    return payload


def build_stored_baseline(
    row: Dict[str, Any],
    thresholds: RepetitionThresholds,
    save_token_ids: bool,
    max_new_tokens: int,
) -> Dict[str, Any]:
    continuation = select_continuation_text(row)
    token_ids = row.get("generated_token_ids")
    if token_ids is not None:
        token_ids = [int(token_id) for token_id in token_ids]
    full_text = str(row.get("full_text") or continuation)
    generated_tokens = int(row.get("generated_tokens") or len(token_ids or []))
    return build_condition_payload(
        label="baseline",
        continuation=continuation,
        full_text=full_text,
        generated_tokens=generated_tokens,
        token_ids=token_ids,
        existing_repetition=row.get("repetition") or (row.get("repetition_detection") or {}).get("repetition"),
        thresholds=thresholds,
        save_token_ids=save_token_ids,
        max_new_tokens=max_new_tokens,
        debug={"baseline_mode": "stored"},
    )


def append_jsonl_line(handle: IO[str] | None, row: Dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def classify_repetition_comparison(baseline: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, Any]:
    base_det = baseline["repetition_detection"]
    int_det = intervention["repetition_detection"]
    score_delta = int(int_det["score"]) - int(base_det["score"])
    generated_tokens_delta = int(intervention.get("generated_tokens", 0)) - int(baseline.get("generated_tokens", 0))

    if base_det["matched"] and not int_det["matched"]:
        category = "suppressed"
    elif (not base_det["matched"]) and int_det["matched"]:
        category = "induced_repetition"
    elif base_det["matched"] and int_det["matched"] and score_delta < 0:
        category = "improved_still_repetitive"
    elif base_det["matched"] and int_det["matched"] and score_delta > 0:
        category = "worsened_still_repetitive"
    elif score_delta < 0:
        category = "improved_nonrepetitive"
    elif score_delta > 0:
        category = "worsened_nonrepetitive"
    else:
        category = "unchanged"

    return {
        "category": category,
        "baseline_matched": bool(base_det["matched"]),
        "intervention_matched": bool(int_det["matched"]),
        "score_delta": score_delta,
        "generated_tokens_delta": generated_tokens_delta,
        "baseline_score": int(base_det["score"]),
        "intervention_score": int(int_det["score"]),
        "baseline_trigger_types": list(base_det.get("trigger_types") or []),
        "intervention_trigger_types": list(int_det.get("trigger_types") or []),
    }


def write_preview_markdown(path: Path, rows: List[Dict[str, Any]], preview_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Head Boost Effects Preview\n\n")
        for idx, row in enumerate(rows, start=1):
            f.write(f"## {idx}. {row.get('example_id', 'unknown')}\n\n")
            f.write(f"- source: `{row.get('source')}`\n")
            f.write(f"- repetition_category: `{row.get('repetition_comparison', {}).get('category')}`\n")
            f.write(f"- accuracy_category: `{row.get('answer_comparison', {}).get('category')}`\n")
            f.write(f"- gold_answer: `{row.get('answer_comparison', {}).get('gold_answer')}`\n")
            f.write(f"- baseline_answer: `{row.get('answer_comparison', {}).get('baseline_answer')}`\n")
            f.write(f"- intervention_answer: `{row.get('answer_comparison', {}).get('intervention_answer')}`\n")
            f.write(f"- baseline_repetition_score: `{row.get('repetition_comparison', {}).get('baseline_score')}`\n")
            f.write(f"- intervention_repetition_score: `{row.get('repetition_comparison', {}).get('intervention_score')}`\n")
            f.write(f"- repetition_score_delta: `{row.get('repetition_comparison', {}).get('score_delta')}`\n\n")
            f.write("### Problem\n\n")
            f.write(truncate_text(str(row.get("problem") or row.get("question") or ""), preview_chars) + "\n\n")
            f.write("### Baseline Continuation\n\n```text\n")
            f.write(truncate_text(str(row.get("baseline", {}).get("continuation") or ""), preview_chars))
            f.write("\n```\n\n")
            f.write("### Intervention Continuation\n\n```text\n")
            f.write(truncate_text(str(row.get("intervention", {}).get("continuation") or ""), preview_chars))
            f.write("\n```\n\n")


def process_rows(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
    stream_result_path: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], str, bool]:
    thresholds = make_thresholds(args)
    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map=device_map_override,
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

    head_labels = parse_comma_list(args.head_labels)
    targets, attn_modules, layer_path = resolve_head_targets(backend.model, head_labels)
    operations = INTERVENTION_REGISTRY.get_required(args.intervention_kind)(
        targets,
        {"scale": args.scale},
    )
    intervention_label = f"{args.intervention_kind}[{','.join(head_labels)}]"

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    interrupted = False
    stream_handle: IO[str] | None = None
    try:
        if stream_result_path:
            stream_handle = open(stream_result_path, "w", encoding="utf-8")
        iterator = tqdm(rows, desc=progress_desc, dynamic_ncols=True, leave=False)
        for idx, row in enumerate(iterator, start=1):
            try:
                prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
            except Exception as exc:
                skipped_rows.append(
                    {
                        "example_id": str(row.get("example_id") or f"row_{idx}"),
                        "reason": f"prompt_resolution_failed: {exc}",
                    }
                )
                continue

            try:
                if args.baseline_mode == "stored":
                    baseline = build_stored_baseline(row, thresholds, args.save_token_ids, args.max_new_tokens)
                else:
                    seed_everything(args.seed)
                    baseline_generation = backend.generate(prompt_prefix, generation_config)
                    baseline = build_condition_payload(
                        label="baseline",
                        continuation=baseline_generation.continuation,
                        full_text=baseline_generation.full_text,
                        generated_tokens=baseline_generation.generated_tokens,
                        token_ids=baseline_generation.token_ids,
                        existing_repetition=None,
                        thresholds=thresholds,
                        save_token_ids=args.save_token_ids,
                        max_new_tokens=args.max_new_tokens,
                        debug={"baseline_mode": "rerun"},
                    )

                seed_everything(args.seed)
                with MultiLayerHeadIntervention(attn_modules, operations) as hook_set:
                    intervention_generation = backend.generate(prompt_prefix, generation_config)
                    intervention = build_condition_payload(
                        label=intervention_label,
                        continuation=intervention_generation.continuation,
                        full_text=intervention_generation.full_text,
                        generated_tokens=intervention_generation.generated_tokens,
                        token_ids=intervention_generation.token_ids,
                        existing_repetition=None,
                        thresholds=thresholds,
                        save_token_ids=args.save_token_ids,
                        max_new_tokens=args.max_new_tokens,
                        debug=hook_set.merged_debug_state(),
                    )
            except KeyboardInterrupt:
                interrupted = True
                break

            repetition_comparison = classify_repetition_comparison(baseline, intervention)
            answer_comparison = judge_comparison_row({"baseline": baseline, "intervention": intervention}, row)
            result_row = {
                "example_id": row.get("example_id") or row.get("id") or f"row_{idx}",
                "source": row.get("source"),
                "problem": row.get("problem") or row.get("question"),
                "input_prompt": prompt_prefix,
                "baseline": baseline,
                "intervention": intervention,
                "repetition_comparison": repetition_comparison,
                "answer_comparison": answer_comparison,
                "head_labels": head_labels,
                "intervention_kind": args.intervention_kind,
                "intervention_params": {"scale": args.scale} if args.intervention_kind == "scale" else {},
            }
            result_rows.append(result_row)
            append_jsonl_line(stream_handle, result_row)

            if idx % max(args.print_every, 1) == 0:
                print(
                    f"[Info] pid={os.getpid()} processed={idx} kept={len(result_rows)} skipped={len(skipped_rows)} "
                    f"latest_accuracy={answer_comparison['category']} "
                    f"latest_repetition={repetition_comparison['category']} "
                    f"latest_example_id={result_row['example_id']}"
                )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if stream_handle is not None:
            stream_handle.close()

    return result_rows, skipped_rows, layer_path, interrupted


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
    result_rows, skipped_rows, layer_path, interrupted = process_rows(
        shard_rows,
        args,
        device_map_override={"": gpu_id},
        progress_desc=f"Worker {worker_id} GPU{gpu_id}",
        stream_result_path=worker_rows_path,
    )
    with open(worker_skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped_rows, f, ensure_ascii=False, indent=2)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "row_count": len(result_rows),
        "skipped_count": len(skipped_rows),
        "worker_rows_path": worker_rows_path,
        "worker_skipped_path": worker_skipped_path,
        "layer_path": layer_path,
        "interrupted": interrupted,
    }


def read_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dedupe_rows_by_example_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    ordered: List[str] = []
    for row in rows:
        key = str(row.get("example_id") or "")
        if key and key not in deduped:
            ordered.append(key)
        deduped[key] = row
    return [deduped[key] for key in ordered]


def build_summary_outputs(
    *,
    args: argparse.Namespace,
    input_metadata: Dict[str, Any],
    output_dir: Path,
    rows_path: Path,
    case_summary_path: Path,
    accuracy_summary_path: Path,
    repetition_summary_path: Path,
    summary_path: Path,
    preview_path: Path,
    run_config_path: Path,
    result_rows: List[Dict[str, Any]],
    skipped_rows: List[Dict[str, str]],
    layer_path: str,
    parallel_enabled: bool,
    available_gpu_ids: List[int],
    worker_count: int,
    interrupted: bool,
) -> None:
    with open(rows_path, "w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    accuracy_summary = summarize_comparison_accuracy(result_rows)
    repetition_counter: Counter[str] = Counter()
    overlap_counter: Counter[str] = Counter()
    baseline_scores: List[float] = []
    intervention_scores: List[float] = []
    baseline_tokens: List[float] = []
    intervention_tokens: List[float] = []
    baseline_hit_cap = 0
    intervention_hit_cap = 0
    case_summary_rows: List[Dict[str, Any]] = []
    for row in result_rows:
        rep = row["repetition_comparison"]
        acc = row["answer_comparison"]
        repetition_counter.update([str(rep["category"])])
        overlap_counter.update([f"{rep['category']}__{acc.get('category', 'unknown')}"])
        baseline_scores.append(float(rep["baseline_score"]))
        intervention_scores.append(float(rep["intervention_score"]))
        baseline_tokens.append(float(row["baseline"].get("generated_tokens", 0)))
        intervention_tokens.append(float(row["intervention"].get("generated_tokens", 0)))
        if bool(row["baseline"].get("hit_max_new_tokens")):
            baseline_hit_cap += 1
        if bool(row["intervention"].get("hit_max_new_tokens")):
            intervention_hit_cap += 1
        case_summary_rows.append(
            {
                "example_id": row["example_id"],
                "source": row.get("source"),
                "repetition_category": rep["category"],
                "accuracy_category": acc.get("category"),
                "baseline_matched": rep["baseline_matched"],
                "intervention_matched": rep["intervention_matched"],
                "baseline_correct": acc.get("baseline_correct"),
                "intervention_correct": acc.get("intervention_correct"),
                "gold_answer": acc.get("gold_answer"),
                "baseline_answer": acc.get("baseline_answer"),
                "intervention_answer": acc.get("intervention_answer"),
                "baseline_score": rep["baseline_score"],
                "intervention_score": rep["intervention_score"],
                "score_delta": rep["score_delta"],
                "baseline_generated_tokens": row["baseline"].get("generated_tokens", 0),
                "intervention_generated_tokens": row["intervention"].get("generated_tokens", 0),
                "baseline_hit_max_new_tokens": row["baseline"].get("hit_max_new_tokens"),
                "intervention_hit_max_new_tokens": row["intervention"].get("hit_max_new_tokens"),
                "generated_tokens_delta": rep["generated_tokens_delta"],
                "baseline_trigger_types": ",".join(rep["baseline_trigger_types"]),
                "intervention_trigger_types": ",".join(rep["intervention_trigger_types"]),
            }
        )
    case_summary_rows.sort(key=lambda item: (item["score_delta"], -item["baseline_score"]))
    write_csv(case_summary_path, case_summary_rows)

    accuracy_summary_rows = [
        {"category": category, "count": count}
        for category, count in sorted(dict(accuracy_summary.get("accuracy_counts") or {}).items())
    ]
    write_csv(accuracy_summary_path, accuracy_summary_rows)
    repetition_summary_rows = [
        {"category": category, "count": count}
        for category, count in sorted(dict(repetition_counter).items())
    ]
    write_csv(repetition_summary_path, repetition_summary_rows)

    def _mean(values: List[float]) -> float:
        return 0.0 if not values else sum(values) / len(values)

    summary = {
        "input_source": input_metadata["input_source"],
        "input_jsonl": input_metadata["input_jsonl"],
        "dataset_name": input_metadata["dataset_name"],
        "dataset_split": input_metadata["dataset_split"],
        "output_dir": str(output_dir),
        "processed_examples": len(result_rows),
        "skipped_examples": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "completed_successfully": not interrupted,
        "interrupted": interrupted,
        "baseline_mode": args.baseline_mode,
        "model_name_or_path": args.model_name_or_path,
        "head_labels": parse_comma_list(args.head_labels),
        "intervention_kind": args.intervention_kind,
        "intervention_params": {"scale": args.scale} if args.intervention_kind == "scale" else {},
        "parallel_enabled": parallel_enabled,
        "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
        "parallel_workers": worker_count,
        "accuracy_counts": dict(accuracy_summary.get("accuracy_counts") or {}),
        "repetition_counts": dict(repetition_counter),
        "overlap_counts": dict(overlap_counter),
        "verifiable_examples": int(accuracy_summary.get("verifiable_examples", 0)),
        "baseline_correct_rate_over_verifiable": float(accuracy_summary.get("baseline_correct_rate_over_verifiable", 0.0)),
        "intervention_correct_rate_over_verifiable": float(accuracy_summary.get("intervention_correct_rate_over_verifiable", 0.0)),
        "newly_correct_rate_over_verifiable": float(accuracy_summary.get("newly_correct_rate_over_verifiable", 0.0)),
        "regression_rate_over_verifiable": float(accuracy_summary.get("regression_rate_over_verifiable", 0.0)),
        "baseline_repetition_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["repetition_comparison"]["baseline_matched"]) / len(result_rows), 6),
        "intervention_repetition_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["repetition_comparison"]["intervention_matched"]) / len(result_rows), 6),
        "repetition_suppression_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["repetition_comparison"]["category"] == "suppressed") / len(result_rows), 6),
        "repetition_induction_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["repetition_comparison"]["category"] == "induced_repetition") / len(result_rows), 6),
        "mean_baseline_repetition_score": round(_mean(baseline_scores), 6),
        "mean_intervention_repetition_score": round(_mean(intervention_scores), 6),
        "mean_repetition_score_delta": round(_mean([float(row["repetition_comparison"]["score_delta"]) for row in result_rows]), 6),
        "mean_baseline_generated_tokens": round(_mean(baseline_tokens), 6),
        "mean_intervention_generated_tokens": round(_mean(intervention_tokens), 6),
        "mean_generated_tokens_delta": round(_mean([float(row["repetition_comparison"]["generated_tokens_delta"]) for row in result_rows]), 6),
        "baseline_hit_max_new_tokens_count": baseline_hit_cap,
        "intervention_hit_max_new_tokens_count": intervention_hit_cap,
        "baseline_hit_max_new_tokens_rate": 0.0 if not result_rows else round(baseline_hit_cap / len(result_rows), 6),
        "intervention_hit_max_new_tokens_rate": 0.0 if not result_rows else round(intervention_hit_cap / len(result_rows), 6),
        "decoder_layer_path": layer_path,
    }
    write_json(summary_path, summary)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
            "parallel_workers": worker_count,
            "decoder_layer_path": layer_path,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
        },
    )

    preview_priority = {
        "newly_correct": 0,
        "regressed": 1,
        "remained_wrong": 2,
        "remained_correct": 3,
        "unverifiable": 4,
    }
    preview_rows = sorted(
        result_rows,
        key=lambda item: (
            preview_priority.get(str((item.get("answer_comparison") or {}).get("category")), 9),
            item["repetition_comparison"]["score_delta"],
            -item["repetition_comparison"]["baseline_score"],
        ),
    )[: max(args.preview_examples, 0)]
    write_preview_markdown(preview_path, preview_rows, args.preview_chars)


def main() -> None:
    args = parse_args()
    if args.input_source == "numinamath" and args.baseline_mode != "rerun":
        raise ValueError("--input_source numinamath requires --baseline_mode rerun.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    case_summary_path = output_dir / "case_summary.csv"
    accuracy_summary_path = output_dir / "accuracy_summary.csv"
    repetition_summary_path = output_dir / "repetition_summary.csv"
    summary_path = output_dir / "summary.json"
    preview_path = output_dir / "top_examples.md"
    run_config_path = output_dir / "run_config.json"

    all_rows, input_metadata = load_experiment_rows(args)
    if args.start_idx > 0:
        all_rows = all_rows[args.start_idx :]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(all_rows)
    if args.max_examples > 0:
        all_rows = all_rows[: args.max_examples]

    if not all_rows:
        write_json(
            summary_path,
            {
                "input_source": input_metadata["input_source"],
                "input_jsonl": input_metadata["input_jsonl"],
                "dataset_name": input_metadata["dataset_name"],
                "dataset_split": input_metadata["dataset_split"],
                "processed_examples": 0,
                "message": "No rows to process.",
            },
        )
        rows_path.write_text("", encoding="utf-8")
        case_summary_path.write_text("", encoding="utf-8")
        accuracy_summary_path.write_text("", encoding="utf-8")
        repetition_summary_path.write_text("", encoding="utf-8")
        preview_path.write_text("# No rows\n", encoding="utf-8")
        write_json(run_config_path, {"args": vars(args)})
        print("[Done] No rows available; wrote empty outputs.")
        return

    if args.parallel_gpu_ids.strip():
        available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    elif args.gpu_id >= 0:
        available_gpu_ids = [args.gpu_id]
    else:
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
        "[Info] Head boost experiment setup: "
        f"examples={len(all_rows)}, cuda_available={torch_cuda_available}, "
        f"available_gpu_ids={available_gpu_ids}, parallel_enabled={parallel_enabled}, "
        f"worker_count={worker_count}, head_labels={parse_comma_list(args.head_labels)}, "
        f"intervention_kind={args.intervention_kind}, scale={args.scale}, baseline_mode={args.baseline_mode}"
    )

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    layer_path = ""
    interrupted = False
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
                    worker_rows_paths.append(worker_rows_path)
                    worker_skipped_paths.append(worker_skipped_path)
                try:
                    for fut in as_completed(futures):
                        worker_ret = fut.result()
                        worker_returns.append(worker_ret)
                        if not layer_path:
                            layer_path = str(worker_ret.get("layer_path") or "")
                        print(
                            f"[Info] Worker {worker_ret['worker_id']} GPU{worker_ret['gpu_id']} "
                            f"finished: rows={worker_ret['row_count']} skipped={worker_ret['skipped_count']}"
                        )
                except KeyboardInterrupt:
                    interrupted = True
                    print("[Warn] Interrupted while waiting for worker completion. Aggregating available partial outputs.")
        finally:
            for path in worker_rows_paths:
                result_rows.extend(read_jsonl_if_exists(path))
            result_rows = dedupe_rows_by_example_id(result_rows)
            for path in worker_skipped_paths:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        skipped_rows.extend(json.load(f))
            if not args.keep_worker_outputs and not interrupted:
                for path in worker_rows_paths + worker_skipped_paths:
                    if path.exists():
                        path.unlink()
    else:
        result_rows, skipped_rows, layer_path, interrupted = process_rows(
            all_rows,
            args,
            device_map_override=resolve_device_map(args.device_map, args.gpu_id),
            progress_desc="Head boost experiment",
        )

    result_rows = dedupe_rows_by_example_id(result_rows)
    build_summary_outputs(
        args=args,
        input_metadata=input_metadata,
        output_dir=output_dir,
        rows_path=rows_path,
        case_summary_path=case_summary_path,
        accuracy_summary_path=accuracy_summary_path,
        repetition_summary_path=repetition_summary_path,
        summary_path=summary_path,
        preview_path=preview_path,
        run_config_path=run_config_path,
        result_rows=result_rows,
        skipped_rows=skipped_rows,
        layer_path=layer_path,
        parallel_enabled=parallel_enabled,
        available_gpu_ids=available_gpu_ids,
        worker_count=worker_count,
        interrupted=interrupted,
    )

    if interrupted:
        print("[Done] Head boost experiment interrupted; partial outputs written:")
    else:
        print("[Done] Head boost experiment finished:")
    print(f"- input_source: {input_metadata['input_source']}")
    if input_metadata["input_jsonl"]:
        print(f"- input_jsonl: {input_metadata['input_jsonl']}")
    if input_metadata["dataset_name"]:
        print(f"- dataset: {input_metadata['dataset_name']} [{input_metadata['dataset_split']}]")
    print(f"- output_dir: {output_dir}")
    print(f"- processed_examples: {len(result_rows)}")
    print(f"- skipped_examples: {len(skipped_rows)}")
    print(f"- rows_jsonl: {rows_path}")
    print(f"- summary_json: {summary_path}")
    print(f"- accuracy_summary_csv: {accuracy_summary_path}")
    print(f"- repetition_summary_csv: {repetition_summary_path}")
    print(f"- preview_md: {preview_path}")


if __name__ == "__main__":
    main()
