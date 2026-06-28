#!/usr/bin/env python3
from __future__ import annotations

"""Test whether manual L0H3 intervention suppresses repetition on collected CoT cases.

Default workflow:
1. Read repetitive examples from outputs/repetition/all_repetition_cases.jsonl.
2. Re-run baseline generation on the same prompt.
3. Re-run the same prompt with an L0H3 intervention on Qwen3-1.7B.
4. Compare repetition metrics before vs after intervention.

Typical usage:
    python scripts/test_l0h3_repetition_suppression.py

    python scripts/test_l0h3_repetition_suppression.py \
      --baseline_mode rerun \
      --intervention_kind scale \
      --scale 0.2 \
      --max_examples 100
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from collections import Counter
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import extract_last_boxed
from cot_research.cot_accuracy import judge_comparison_row, summarize_comparison_accuracy
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
    default_input = root_dir / "outputs" / "repetition" / "all_repetition_cases.jsonl"
    default_output = root_dir / "outputs" / "repetition" / "l0h3_suppression_qwen3_1p7b"

    parser = argparse.ArgumentParser(description="Test L0H3 intervention for repetition suppression on CoT cases")
    parser.add_argument("--input_jsonl", type=str, default=str(default_input))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
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
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument(
        "--intervention_kind",
        type=str,
        default="zero",
        choices=list(INTERVENTION_REGISTRY.names()),
    )
    parser.add_argument("--scale", type=float, default=0.0, help="Used when --intervention_kind=scale.")
    parser.add_argument(
        "--baseline_mode",
        type=str,
        default="rerun",
        choices=["stored", "rerun"],
        help="Rerun baseline generation by default. Use stored only if you explicitly want to reuse saved generations.",
    )
    parser.add_argument("--preview_examples", type=int, default=20)
    parser.add_argument("--preview_chars", type=int, default=1200)
    parser.add_argument("--save_token_ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)

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

    parser.add_argument("--same_token_run_threshold", type=int, default=40)
    parser.add_argument("--tail_repeat_min_repeats", type=int, default=6)
    parser.add_argument("--tail_repeat_max_ngram", type=int, default=8)
    parser.add_argument("--tail_repeat_min_span", type=int, default=24)
    parser.add_argument("--line_repeat_threshold", type=int, default=4)
    parser.add_argument("--word_tail_repeat_min_repeats", type=int, default=5)
    parser.add_argument("--word_tail_repeat_max_ngram", type=int, default=8)
    parser.add_argument("--word_tail_repeat_min_span", type=int, default=24)
    parser.add_argument("--min_trigger_count", type=int, default=1)
    return parser.parse_args()

def write_preview_markdown(path: Path, rows: List[Dict[str, Any]], preview_chars: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# L0H3 Repetition Suppression Cases\n\n")
        for idx, row in enumerate(rows, start=1):
            f.write(f"## {idx}. {row.get('example_id', 'unknown')}\n\n")
            f.write(f"- source: `{row.get('source')}`\n")
            f.write(f"- comparison: `{row.get('comparison', {}).get('category')}`\n")
            f.write(f"- accuracy: `{row.get('answer_comparison', {}).get('category')}`\n")
            f.write(f"- gold_answer: `{row.get('answer_comparison', {}).get('gold_answer')}`\n")
            f.write(f"- baseline_answer: `{row.get('answer_comparison', {}).get('baseline_answer')}`\n")
            f.write(f"- intervention_answer: `{row.get('answer_comparison', {}).get('intervention_answer')}`\n")
            f.write(f"- baseline_repetitive: `{row.get('comparison', {}).get('baseline_matched')}`\n")
            f.write(f"- intervention_repetitive: `{row.get('comparison', {}).get('intervention_matched')}`\n\n")
            f.write("### Problem\n\n")
            f.write(truncate_text(str(row.get("problem") or row.get("question") or ""), preview_chars) + "\n\n")
            f.write("### Baseline Continuation\n\n```text\n")
            f.write(truncate_text(str(row.get("baseline", {}).get("continuation") or ""), preview_chars))
            f.write("\n```\n\n")
            f.write("### Intervention Continuation\n\n```text\n")
            f.write(truncate_text(str(row.get("intervention", {}).get("continuation") or ""), preview_chars))
            f.write("\n```\n\n")


def make_thresholds(args: argparse.Namespace) -> RepetitionThresholds:
    return RepetitionThresholds(
        same_token_run_threshold=args.same_token_run_threshold,
        tail_repeat_min_repeats=args.tail_repeat_min_repeats,
        tail_repeat_max_ngram=args.tail_repeat_max_ngram,
        tail_repeat_min_span=args.tail_repeat_min_span,
        line_repeat_threshold=args.line_repeat_threshold,
        line_run_score_multiplier=12,
        word_tail_repeat_min_repeats=args.word_tail_repeat_min_repeats,
        word_tail_repeat_max_ngram=args.word_tail_repeat_max_ngram,
        word_tail_repeat_min_span=args.word_tail_repeat_min_span,
        tail_word_requires_hard_signal=True,
        min_trigger_count=args.min_trigger_count,
    )

def build_condition_payload(
    *,
    label: str,
    continuation: str,
    full_text: str,
    generated_tokens: int,
    token_ids: Optional[List[int]],
    existing_repetition: Optional[Dict[str, Any]],
    thresholds: RepetitionThresholds,
    save_token_ids: bool,
    max_new_tokens: int,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    repetition_eval = analyze_repetition(
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
        "repetition_detection": repetition_eval,
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
        debug={"baseline_mode": "stored", "source_repetition_detection": row.get("repetition_detection") or {}},
    )


def classify_comparison(baseline: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, Any]:
    base_det = baseline["repetition_detection"]
    int_det = intervention["repetition_detection"]

    if base_det["matched"] and not int_det["matched"]:
        category = "suppressed"
    elif base_det["matched"] and int_det["matched"]:
        category = "still_repetitive"
    elif (not base_det["matched"]) and int_det["matched"]:
        category = "induced_repetition"
    else:
        category = "nonrepetitive"

    return {
        "category": category,
        "baseline_matched": bool(base_det["matched"]),
        "intervention_matched": bool(int_det["matched"]),
    }


def process_rows(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], str]:
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

    targets, attn_modules, layer_path = resolve_head_targets(backend.model, [args.head_label])
    operations = INTERVENTION_REGISTRY.get_required(args.intervention_kind)(
        targets,
        {"scale": args.scale},
    )

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
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
                label=f"{args.intervention_kind}[{args.head_label}]",
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

        comparison = classify_comparison(baseline, intervention)
        answer_comparison = judge_comparison_row(
            {
                "baseline": baseline,
                "intervention": intervention,
            },
            row,
        )
        result_row = {
            "example_id": row.get("example_id") or row.get("id") or f"row_{idx}",
            "source": row.get("source"),
            "problem": row.get("problem") or row.get("question"),
            "source_input_jsonl": row.get("source_input_jsonl"),
            "input_prompt": prompt_prefix,
            "stored_repetition_detection": row.get("repetition_detection") or {},
            "baseline": baseline,
            "intervention": intervention,
            "comparison": comparison,
            "answer_comparison": answer_comparison,
            "head_label": args.head_label,
            "intervention_kind": args.intervention_kind,
            "intervention_params": {"scale": args.scale} if args.intervention_kind == "scale" else {},
        }
        result_rows.append(result_row)

        if idx % max(args.print_every, 1) == 0:
            print(
                f"[Info] pid={os.getpid()} processed={idx} kept={len(result_rows)} skipped={len(skipped_rows)} "
                f"latest_category={comparison['category']} latest_example_id={result_row['example_id']}"
            )

    return result_rows, skipped_rows, layer_path


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

    result_rows, skipped_rows, layer_path = process_rows(
        shard_rows,
        args,
        device_map_override={"": gpu_id},
        progress_desc=f"Worker {worker_id} GPU{gpu_id}",
    )

    with open(worker_rows_path, "w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input repetition file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    case_summary_path = output_dir / "case_summary.csv"
    accuracy_summary_path = output_dir / "accuracy_summary.csv"
    summary_path = output_dir / "summary.json"
    preview_path = output_dir / "top_examples.md"
    run_config_path = output_dir / "run_config.json"

    thresholds = make_thresholds(args)
    all_rows = load_jsonl(input_path)
    if args.max_examples > 0:
        all_rows = all_rows[: args.max_examples]

    if not all_rows:
        write_json(summary_path, {"input_jsonl": str(input_path), "processed_examples": 0, "message": "No rows to process."})
        rows_path.write_text("", encoding="utf-8")
        case_summary_path.write_text("", encoding="utf-8")
        preview_path.write_text("# No rows\n", encoding="utf-8")
        write_json(run_config_path, {"args": vars(args), "thresholds": thresholds.to_dict()})
        print("[Done] No repetition rows found in input; wrote empty outputs.")
        return

    if args.parallel_gpu_ids.strip():
        available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    elif args.gpu_id >= 0:
        available_gpu_ids = [args.gpu_id]
    else:
        available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    torch_cuda_available = torch.cuda.is_available()
    can_parallel = (
        args.parallel
        and torch_cuda_available
        and len(available_gpu_ids) > 1
        and len(all_rows) > 1
    )
    if can_parallel and args.parallel_workers > 0:
        worker_count = min(args.parallel_workers, len(available_gpu_ids), len(all_rows))
    elif can_parallel:
        worker_count = min(len(available_gpu_ids), len(all_rows))
    else:
        worker_count = 1
    parallel_enabled = can_parallel and worker_count > 1

    print(
        "[Info] L0H3 suppression setup: "
        f"examples={len(all_rows)}, cuda_available={torch_cuda_available}, "
        f"available_gpu_ids={available_gpu_ids}, parallel_enabled={parallel_enabled}, "
        f"worker_count={worker_count}, intervention_kind={args.intervention_kind}, scale={args.scale}"
    )

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    layer_path = ""
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
                    if not layer_path:
                        layer_path = str(worker_ret.get("layer_path") or "")
                    print(
                        f"[Info] Worker {worker_ret['worker_id']} GPU{worker_ret['gpu_id']} "
                        f"finished: rows={worker_ret['row_count']} skipped={worker_ret['skipped_count']}"
                    )

            for worker_ret in sorted(worker_returns, key=lambda item: int(item["worker_id"])):
                with open(worker_ret["worker_rows_path"], "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        result_rows.append(json.loads(line))
                with open(worker_ret["worker_skipped_path"], "r", encoding="utf-8") as f:
                    skipped_rows.extend(json.load(f))
        finally:
            if not args.keep_worker_outputs:
                for path in worker_rows_paths:
                    if path.exists():
                        path.unlink()
                for path in worker_skipped_paths:
                    if path.exists():
                        path.unlink()
    else:
        result_rows, skipped_rows, layer_path = process_rows(
            all_rows,
            args,
            device_map_override=resolve_device_map(args.device_map, args.gpu_id),
            progress_desc="L0H3 repetition test",
        )

    comparison_counter: Counter[str] = Counter()
    for row in result_rows:
        comparison_counter.update([str(row["comparison"]["category"])])

    with open(rows_path, "w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    case_summary_rows: List[Dict[str, Any]] = []
    overlap_counter: Counter[str] = Counter()
    for row in result_rows:
        comparison = row["comparison"]
        answer_comparison = row.get("answer_comparison") or {}
        overlap_counter.update([f"{comparison['category']}__{answer_comparison.get('category', 'unknown')}"])
        case_summary_rows.append(
            {
                "example_id": row["example_id"],
                "source": row.get("source"),
                "category": comparison["category"],
                "answer_category": answer_comparison.get("category"),
                "baseline_matched": comparison["baseline_matched"],
                "intervention_matched": comparison["intervention_matched"],
                "baseline_correct": answer_comparison.get("baseline_correct"),
                "intervention_correct": answer_comparison.get("intervention_correct"),
                "gold_answer": answer_comparison.get("gold_answer"),
                "baseline_answer": answer_comparison.get("baseline_answer"),
                "intervention_answer": answer_comparison.get("intervention_answer"),
                "baseline_generated_tokens": row["baseline"].get("generated_tokens", 0),
                "intervention_generated_tokens": row["intervention"].get("generated_tokens", 0),
            }
        )

    case_summary_rows.sort(key=lambda item: (item["baseline_matched"], item["intervention_matched"], item["answer_category"]))
    write_csv(case_summary_path, case_summary_rows)
    accuracy_summary = summarize_comparison_accuracy(result_rows)
    accuracy_counts = dict(accuracy_summary.get("accuracy_counts") or {})
    accuracy_summary_rows = [
        {"category": category, "count": count}
        for category, count in sorted(accuracy_counts.items())
    ]
    write_csv(accuracy_summary_path, accuracy_summary_rows)

    summary = {
        "input_jsonl": str(input_path),
        "output_dir": str(output_dir),
        "processed_examples": len(result_rows),
        "skipped_examples": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "baseline_mode": args.baseline_mode,
        "model_name_or_path": args.model_name_or_path,
        "head_label": args.head_label,
        "intervention_kind": args.intervention_kind,
        "intervention_params": {"scale": args.scale} if args.intervention_kind == "scale" else {},
        "thresholds": thresholds.to_dict(),
        "parallel_enabled": parallel_enabled,
        "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
        "parallel_workers": worker_count,
        "comparison_counts": dict(comparison_counter),
        "accuracy_counts": dict(accuracy_summary.get("accuracy_counts") or {}),
        "overlap_counts": dict(overlap_counter),
        "verifiable_examples": int(accuracy_summary.get("verifiable_examples", 0)),
        "baseline_correct_rate_over_verifiable": float(accuracy_summary.get("baseline_correct_rate_over_verifiable", 0.0)),
        "intervention_correct_rate_over_verifiable": float(accuracy_summary.get("intervention_correct_rate_over_verifiable", 0.0)),
        "newly_correct_rate_over_verifiable": float(accuracy_summary.get("newly_correct_rate_over_verifiable", 0.0)),
        "regression_rate_over_verifiable": float(accuracy_summary.get("regression_rate_over_verifiable", 0.0)),
        "baseline_match_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["comparison"]["baseline_matched"]) / len(result_rows), 6),
        "intervention_match_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["comparison"]["intervention_matched"]) / len(result_rows), 6),
        "suppression_rate": 0.0 if not result_rows else round(sum(1 for row in result_rows if row["comparison"]["category"] == "suppressed") / len(result_rows), 6),
        "decoder_layer_path": layer_path,
    }
    write_json(summary_path, summary)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "thresholds": thresholds.to_dict(),
            "decoder_layer_path": layer_path,
            "resolved_targets": [args.head_label],
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
            "parallel_workers": worker_count,
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
            0 if item["comparison"]["category"] == "suppressed" else 1,
        ),
    )[: max(args.preview_examples, 0)]
    write_preview_markdown(preview_path, preview_rows, args.preview_chars)

    print("[Done] L0H3 repetition suppression test finished:")
    print(f"- input_jsonl: {input_path}")
    print(f"- output_dir: {output_dir}")
    print(f"- processed_examples: {len(result_rows)}")
    print(f"- skipped_examples: {len(skipped_rows)}")
    print(f"- comparison_counts: {dict(comparison_counter)}")
    print(f"- accuracy_counts: {dict(accuracy_summary.get('accuracy_counts') or {})}")
    print(f"- rows_jsonl: {rows_path}")
    print(f"- case_summary_csv: {case_summary_path}")
    print(f"- accuracy_summary_csv: {accuracy_summary_path}")
    print(f"- summary_json: {summary_path}")
    print(f"- preview_markdown: {preview_path}")


if __name__ == "__main__":
    main()
