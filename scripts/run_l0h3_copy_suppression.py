#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, IO, List, Optional, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.copy_suppression import (
    build_condition_step_metric,
    build_generated_copy_step_rows,
    sample_analysis_step_indices,
    summarize_generated_copy_step_rows,
)
from cot_research.generation import create_backend
from cot_research.head_intervention import INTERVENTION_REGISTRY, MultiLayerHeadIntervention, resolve_head_targets
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.model_utils import get_input_device_for_model
from cot_research.repetition_analysis import RepetitionThresholds, analyze_repetition
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, resolve_device_map, seed_everything, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run ARIS-style experiments for the hypothesis that Qwen3-1.7B L0H3 suppresses copying, "
            "especially recent-context copy impulses, and does so by pushing down candidate copy tokens."
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
        default=str(root_dir / "outputs" / "copy_suppression" / "qwen3_1p7b_l0h3_20260408"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--boost_scale", type=float, default=1.4)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-process multi-GPU execution by sharding examples across GPUs.",
    )
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--keep_worker_outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_examples", type=int, default=108)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--save_token_ids", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--recent_window", type=int, default=32)
    parser.add_argument("--max_selected_copy_steps_per_example", type=int, default=8)
    parser.add_argument("--max_selected_control_steps_per_example", type=int, default=4)
    parser.add_argument("--claim3_min_candidate_prob", type=float, default=0.05)
    parser.add_argument("--claim3_max_candidate_rank", type=int, default=5)

    parser.add_argument("--same_token_run_threshold", type=int, default=24)
    parser.add_argument("--tail_repeat_min_repeats", type=int, default=4)
    parser.add_argument("--tail_repeat_max_ngram", type=int, default=32)
    parser.add_argument("--line_repeat_threshold", type=int, default=3)
    parser.add_argument("--word_tail_repeat_min_repeats", type=int, default=4)
    parser.add_argument("--word_tail_repeat_max_ngram", type=int, default=24)
    parser.add_argument("--min_trigger_count", type=int, default=1)
    return parser.parse_args()


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


def append_jsonl_line(handle: IO[str] | None, row: Dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


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


def token_text(tokenizer, token_id: Optional[int]) -> str:
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def run_generation_condition(
    *,
    backend,
    prompt_prefix: str,
    generation_config: GenerationConfig,
    thresholds: RepetitionThresholds,
    recent_window: int,
    label: str,
    operations: Optional[List[Tuple[Any, float]]],
    attn_modules: List[torch.nn.Module],
    save_token_ids: bool,
) -> Tuple[Dict[str, Any], List[int]]:
    debug: Dict[str, Any] = {}
    if operations:
        with MultiLayerHeadIntervention(attn_modules, operations) as hook_set:
            generation = backend.generate(prompt_prefix, generation_config)
            debug = hook_set.merged_debug_state()
    else:
        generation = backend.generate(prompt_prefix, generation_config)

    copy_step_rows = build_generated_copy_step_rows(generation.token_ids, recent_window=recent_window)
    payload: Dict[str, Any] = {
        "label": label,
        "continuation": generation.continuation,
        "full_text": generation.full_text,
        "generated_tokens": int(generation.generated_tokens),
        "repetition_detection": analyze_repetition(
            generation.continuation,
            token_ids=generation.token_ids,
            thresholds=thresholds,
        ),
        "copy_metrics": summarize_generated_copy_step_rows(copy_step_rows),
        "debug": debug,
    }
    if save_token_ids:
        payload["token_ids"] = [int(token_id) for token_id in generation.token_ids]
    return payload, [int(token_id) for token_id in generation.token_ids]


@torch.no_grad()
def forward_logits_for_full_sequence(
    model: torch.nn.Module,
    full_token_ids: List[int],
    *,
    attn_modules: List[torch.nn.Module],
    operations: Optional[List[Tuple[Any, float]]] = None,
) -> torch.Tensor:
    model_device = get_input_device_for_model(model)
    input_ids = torch.tensor([full_token_ids], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    if operations:
        with MultiLayerHeadIntervention(attn_modules, operations):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits[0]


def analyze_replay_for_example(
    *,
    backend,
    tokenizer,
    attn_modules: List[torch.nn.Module],
    prompt_token_ids: List[int],
    baseline_token_ids: List[int],
    zero_operations: List[Tuple[Any, float]],
    scale_operations: List[Tuple[Any, float]],
    args: argparse.Namespace,
    example_seed: int,
    example_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not baseline_token_ids:
        return {
            "total_replay_steps": 0,
            "selected_step_count": 0,
            "recent_copy_step_count": 0,
            "claim3_candidate_step_count": 0,
        }, []

    full_token_ids = [int(token_id) for token_id in prompt_token_ids] + [int(token_id) for token_id in baseline_token_ids]
    step_rows = build_generated_copy_step_rows(baseline_token_ids, recent_window=args.recent_window)
    if not step_rows:
        return {
            "total_replay_steps": 0,
            "selected_step_count": 0,
            "recent_copy_step_count": 0,
            "claim3_candidate_step_count": 0,
        }, []

    baseline_logits = forward_logits_for_full_sequence(
        backend.model,
        full_token_ids,
        attn_modules=attn_modules,
        operations=None,
    )
    vocab_size = int(baseline_logits.shape[-1])

    baseline_step_metrics: List[Dict[str, Any]] = []
    prompt_token_count = len(prompt_token_ids)
    for step_row in step_rows:
        logit_pos = prompt_token_count - 1 + int(step_row["step_idx"])
        baseline_step_metrics.append(
            build_condition_step_metric(
                baseline_logits[logit_pos],
                step_row,
                vocab_size=vocab_size,
                random_seed=example_seed + int(step_row["step_idx"]) * 17,
            )
        )
    del baseline_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate_token_ids = [item.get("candidate_token_id") for item in baseline_step_metrics]

    zero_logits = forward_logits_for_full_sequence(
        backend.model,
        full_token_ids,
        attn_modules=attn_modules,
        operations=zero_operations,
    )
    zero_step_metrics: List[Dict[str, Any]] = []
    for step_row, candidate_token_id in zip(step_rows, candidate_token_ids):
        logit_pos = prompt_token_count - 1 + int(step_row["step_idx"])
        zero_step_metrics.append(
            build_condition_step_metric(
                zero_logits[logit_pos],
                step_row,
                vocab_size=vocab_size,
                random_seed=example_seed + int(step_row["step_idx"]) * 17,
                fixed_candidate_token_id=candidate_token_id,
            )
        )
    del zero_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scale_logits = forward_logits_for_full_sequence(
        backend.model,
        full_token_ids,
        attn_modules=attn_modules,
        operations=scale_operations,
    )
    scale_step_metrics: List[Dict[str, Any]] = []
    for step_row, candidate_token_id in zip(step_rows, candidate_token_ids):
        logit_pos = prompt_token_count - 1 + int(step_row["step_idx"])
        scale_step_metrics.append(
            build_condition_step_metric(
                scale_logits[logit_pos],
                step_row,
                vocab_size=vocab_size,
                random_seed=example_seed + int(step_row["step_idx"]) * 17,
                fixed_candidate_token_id=candidate_token_id,
            )
        )
    del scale_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sampled_indices = set(
        sample_analysis_step_indices(
            step_rows,
            max_recent_copy_steps=args.max_selected_copy_steps_per_example,
            max_control_steps=args.max_selected_control_steps_per_example,
            seed=example_seed,
        )
    )

    step_count = 0
    recent_copy_step_count = 0
    claim3_step_count = 0

    sum_baseline_recent_mass = 0.0
    sum_zero_recent_mass = 0.0
    sum_scale_recent_mass = 0.0
    sum_baseline_older_mass = 0.0
    sum_zero_older_mass = 0.0
    sum_scale_older_mass = 0.0
    sum_baseline_random_mass = 0.0
    sum_zero_random_mass = 0.0
    sum_scale_random_mass = 0.0

    sum_zero_recent_delta = 0.0
    sum_scale_recent_delta = 0.0
    sum_zero_older_delta = 0.0
    sum_scale_older_delta = 0.0
    sum_zero_random_delta = 0.0
    sum_scale_random_delta = 0.0

    recent_copy_sum_zero_recent_delta = 0.0
    recent_copy_sum_scale_recent_delta = 0.0
    recent_copy_sum_zero_random_delta = 0.0
    recent_copy_sum_scale_random_delta = 0.0

    scale_recent_more_negative_than_random = 0
    zero_recent_more_positive_than_random = 0
    scale_recent_more_negative_than_older = 0
    zero_recent_more_positive_than_older = 0

    claim3_zero_prob_delta_sum = 0.0
    claim3_scale_prob_delta_sum = 0.0
    claim3_zero_logit_delta_sum = 0.0
    claim3_scale_logit_delta_sum = 0.0
    claim3_ordered_prob_sign_count = 0
    claim3_ordered_logit_sign_count = 0
    claim3_zero_rank_better_count = 0
    claim3_scale_rank_worse_count = 0
    claim3_is_actual_recent_copy_count = 0

    selected_step_rows: List[Dict[str, Any]] = []
    for step_row, baseline_metric, zero_metric, scale_metric in zip(
        step_rows,
        baseline_step_metrics,
        zero_step_metrics,
        scale_step_metrics,
    ):
        step_idx = int(step_row["step_idx"])
        step_count += 1
        if bool(step_row["target_is_recent_copy"]):
            recent_copy_step_count += 1

        baseline_recent_mass = float(baseline_metric["recent_mass"])
        zero_recent_mass = float(zero_metric["recent_mass"])
        scale_recent_mass = float(scale_metric["recent_mass"])
        baseline_older_mass = float(baseline_metric["older_seen_mass"])
        zero_older_mass = float(zero_metric["older_seen_mass"])
        scale_older_mass = float(scale_metric["older_seen_mass"])
        baseline_random_mass = float(baseline_metric["random_control_mass"])
        zero_random_mass = float(zero_metric["random_control_mass"])
        scale_random_mass = float(scale_metric["random_control_mass"])

        zero_recent_delta = zero_recent_mass - baseline_recent_mass
        scale_recent_delta = scale_recent_mass - baseline_recent_mass
        zero_older_delta = zero_older_mass - baseline_older_mass
        scale_older_delta = scale_older_mass - baseline_older_mass
        zero_random_delta = zero_random_mass - baseline_random_mass
        scale_random_delta = scale_random_mass - baseline_random_mass

        sum_baseline_recent_mass += baseline_recent_mass
        sum_zero_recent_mass += zero_recent_mass
        sum_scale_recent_mass += scale_recent_mass
        sum_baseline_older_mass += baseline_older_mass
        sum_zero_older_mass += zero_older_mass
        sum_scale_older_mass += scale_older_mass
        sum_baseline_random_mass += baseline_random_mass
        sum_zero_random_mass += zero_random_mass
        sum_scale_random_mass += scale_random_mass

        sum_zero_recent_delta += zero_recent_delta
        sum_scale_recent_delta += scale_recent_delta
        sum_zero_older_delta += zero_older_delta
        sum_scale_older_delta += scale_older_delta
        sum_zero_random_delta += zero_random_delta
        sum_scale_random_delta += scale_random_delta

        if scale_recent_delta < scale_random_delta:
            scale_recent_more_negative_than_random += 1
        if zero_recent_delta > zero_random_delta:
            zero_recent_more_positive_than_random += 1
        if scale_recent_delta < scale_older_delta:
            scale_recent_more_negative_than_older += 1
        if zero_recent_delta > zero_older_delta:
            zero_recent_more_positive_than_older += 1

        if bool(step_row["target_is_recent_copy"]):
            recent_copy_sum_zero_recent_delta += zero_recent_delta
            recent_copy_sum_scale_recent_delta += scale_recent_delta
            recent_copy_sum_zero_random_delta += zero_random_delta
            recent_copy_sum_scale_random_delta += scale_random_delta

        candidate_prob = baseline_metric.get("candidate_token_prob")
        candidate_logit = baseline_metric.get("candidate_token_logit")
        candidate_rank = baseline_metric.get("candidate_token_rank")
        claim3_eligible = (
            candidate_prob is not None
            and candidate_logit is not None
            and candidate_rank is not None
            and float(candidate_prob) >= float(args.claim3_min_candidate_prob)
            and int(candidate_rank) <= int(args.claim3_max_candidate_rank)
        )
        if claim3_eligible:
            claim3_step_count += 1
            zero_candidate_prob = float(zero_metric["candidate_token_prob"])
            scale_candidate_prob = float(scale_metric["candidate_token_prob"])
            zero_candidate_logit = float(zero_metric["candidate_token_logit"])
            scale_candidate_logit = float(scale_metric["candidate_token_logit"])
            zero_candidate_rank = int(zero_metric["candidate_token_rank"])
            scale_candidate_rank = int(scale_metric["candidate_token_rank"])

            claim3_zero_prob_delta_sum += zero_candidate_prob - float(candidate_prob)
            claim3_scale_prob_delta_sum += scale_candidate_prob - float(candidate_prob)
            claim3_zero_logit_delta_sum += zero_candidate_logit - float(candidate_logit)
            claim3_scale_logit_delta_sum += scale_candidate_logit - float(candidate_logit)
            if zero_candidate_prob > float(candidate_prob) > scale_candidate_prob:
                claim3_ordered_prob_sign_count += 1
            if zero_candidate_logit > float(candidate_logit) > scale_candidate_logit:
                claim3_ordered_logit_sign_count += 1
            if zero_candidate_rank < int(candidate_rank):
                claim3_zero_rank_better_count += 1
            if scale_candidate_rank > int(candidate_rank):
                claim3_scale_rank_worse_count += 1
            if bool(step_row["target_is_recent_copy"]):
                claim3_is_actual_recent_copy_count += 1
            sampled_indices.add(step_idx)

        if step_idx in sampled_indices:
            selected_step_rows.append(
                {
                    "example_id": example_id,
                    "step_idx": step_idx,
                    "prefix_generated_tokens": int(step_row["prefix_generated_tokens"]),
                    "actual_token_id": int(step_row["target_token_id"]),
                    "actual_token_text": token_text(tokenizer, int(step_row["target_token_id"])),
                    "actual_is_recent_copy": bool(step_row["target_is_recent_copy"]),
                    "actual_is_older_copy": bool(step_row["target_is_older_copy"]),
                    "actual_is_any_copy": bool(step_row["target_is_any_copy"]),
                    "last_occurrence_distance": step_row.get("last_occurrence_distance"),
                    "recent_unique_count": int(step_row["recent_unique_count"]),
                    "older_seen_unique_count": int(step_row["older_seen_unique_count"]),
                    "baseline_recent_mass": baseline_recent_mass,
                    "zero_recent_mass": zero_recent_mass,
                    "scale_recent_mass": scale_recent_mass,
                    "baseline_random_control_mass": baseline_random_mass,
                    "zero_random_control_mass": zero_random_mass,
                    "scale_random_control_mass": scale_random_mass,
                    "baseline_older_seen_mass": baseline_older_mass,
                    "zero_older_seen_mass": zero_older_mass,
                    "scale_older_seen_mass": scale_older_mass,
                    "zero_minus_baseline_recent_mass": zero_recent_delta,
                    "scale_minus_baseline_recent_mass": scale_recent_delta,
                    "zero_minus_baseline_random_mass": zero_random_delta,
                    "scale_minus_baseline_random_mass": scale_random_delta,
                    "baseline_actual_token_prob": float(baseline_metric["actual_token_prob"]),
                    "zero_actual_token_prob": float(zero_metric["actual_token_prob"]),
                    "scale_actual_token_prob": float(scale_metric["actual_token_prob"]),
                    "baseline_actual_token_rank": int(baseline_metric["actual_token_rank"]),
                    "zero_actual_token_rank": int(zero_metric["actual_token_rank"]),
                    "scale_actual_token_rank": int(scale_metric["actual_token_rank"]),
                    "candidate_token_id": baseline_metric.get("candidate_token_id"),
                    "candidate_token_text": token_text(tokenizer, baseline_metric.get("candidate_token_id")),
                    "baseline_candidate_token_prob": baseline_metric.get("candidate_token_prob"),
                    "zero_candidate_token_prob": zero_metric.get("candidate_token_prob"),
                    "scale_candidate_token_prob": scale_metric.get("candidate_token_prob"),
                    "baseline_candidate_token_logit": baseline_metric.get("candidate_token_logit"),
                    "zero_candidate_token_logit": zero_metric.get("candidate_token_logit"),
                    "scale_candidate_token_logit": scale_metric.get("candidate_token_logit"),
                    "baseline_candidate_token_rank": baseline_metric.get("candidate_token_rank"),
                    "zero_candidate_token_rank": zero_metric.get("candidate_token_rank"),
                    "scale_candidate_token_rank": scale_metric.get("candidate_token_rank"),
                    "claim3_eligible": claim3_eligible,
                }
            )

    def safe_mean(total: float, count: int) -> float:
        return 0.0 if count == 0 else round(total / count, 6)

    replay_summary = {
        "total_replay_steps": int(step_count),
        "selected_step_count": int(len(selected_step_rows)),
        "recent_copy_step_count": int(recent_copy_step_count),
        "claim3_candidate_step_count": int(claim3_step_count),
        "sum_baseline_recent_mass_all": round(sum_baseline_recent_mass, 6),
        "sum_zero_recent_mass_all": round(sum_zero_recent_mass, 6),
        "sum_scale_recent_mass_all": round(sum_scale_recent_mass, 6),
        "sum_baseline_older_seen_mass_all": round(sum_baseline_older_mass, 6),
        "sum_zero_older_seen_mass_all": round(sum_zero_older_mass, 6),
        "sum_scale_older_seen_mass_all": round(sum_scale_older_mass, 6),
        "sum_baseline_random_control_mass_all": round(sum_baseline_random_mass, 6),
        "sum_zero_random_control_mass_all": round(sum_zero_random_mass, 6),
        "sum_scale_random_control_mass_all": round(sum_scale_random_mass, 6),
        "mean_baseline_recent_mass_all": safe_mean(sum_baseline_recent_mass, step_count),
        "mean_zero_recent_mass_all": safe_mean(sum_zero_recent_mass, step_count),
        "mean_scale_recent_mass_all": safe_mean(sum_scale_recent_mass, step_count),
        "mean_zero_minus_baseline_recent_mass_all": safe_mean(sum_zero_recent_delta, step_count),
        "mean_scale_minus_baseline_recent_mass_all": safe_mean(sum_scale_recent_delta, step_count),
        "mean_zero_minus_baseline_older_seen_mass_all": safe_mean(sum_zero_older_delta, step_count),
        "mean_scale_minus_baseline_older_seen_mass_all": safe_mean(sum_scale_older_delta, step_count),
        "mean_zero_minus_baseline_random_control_mass_all": safe_mean(sum_zero_random_delta, step_count),
        "mean_scale_minus_baseline_random_control_mass_all": safe_mean(sum_scale_random_delta, step_count),
        "zero_recent_minus_random_gap_mean": safe_mean(sum_zero_recent_delta - sum_zero_random_delta, step_count),
        "scale_recent_minus_random_gap_mean": safe_mean(sum_scale_recent_delta - sum_scale_random_delta, step_count),
        "zero_recent_minus_older_gap_mean": safe_mean(sum_zero_recent_delta - sum_zero_older_delta, step_count),
        "scale_recent_minus_older_gap_mean": safe_mean(sum_scale_recent_delta - sum_scale_older_delta, step_count),
        "scale_recent_more_negative_than_random_rate": safe_mean(scale_recent_more_negative_than_random, step_count),
        "zero_recent_more_positive_than_random_rate": safe_mean(zero_recent_more_positive_than_random, step_count),
        "scale_recent_more_negative_than_older_rate": safe_mean(scale_recent_more_negative_than_older, step_count),
        "zero_recent_more_positive_than_older_rate": safe_mean(zero_recent_more_positive_than_older, step_count),
        "mean_zero_minus_baseline_recent_mass_on_recent_copy_steps": safe_mean(
            recent_copy_sum_zero_recent_delta,
            recent_copy_step_count,
        ),
        "mean_scale_minus_baseline_recent_mass_on_recent_copy_steps": safe_mean(
            recent_copy_sum_scale_recent_delta,
            recent_copy_step_count,
        ),
        "mean_zero_minus_baseline_random_mass_on_recent_copy_steps": safe_mean(
            recent_copy_sum_zero_random_delta,
            recent_copy_step_count,
        ),
        "mean_scale_minus_baseline_random_mass_on_recent_copy_steps": safe_mean(
            recent_copy_sum_scale_random_delta,
            recent_copy_step_count,
        ),
        "claim3_zero_candidate_prob_delta_mean": safe_mean(claim3_zero_prob_delta_sum, claim3_step_count),
        "claim3_scale_candidate_prob_delta_mean": safe_mean(claim3_scale_prob_delta_sum, claim3_step_count),
        "claim3_zero_candidate_logit_delta_mean": safe_mean(claim3_zero_logit_delta_sum, claim3_step_count),
        "claim3_scale_candidate_logit_delta_mean": safe_mean(claim3_scale_logit_delta_sum, claim3_step_count),
        "claim3_ordered_prob_sign_rate": safe_mean(claim3_ordered_prob_sign_count, claim3_step_count),
        "claim3_ordered_logit_sign_rate": safe_mean(claim3_ordered_logit_sign_count, claim3_step_count),
        "claim3_zero_candidate_rank_better_rate": safe_mean(claim3_zero_rank_better_count, claim3_step_count),
        "claim3_scale_candidate_rank_worse_rate": safe_mean(claim3_scale_rank_worse_count, claim3_step_count),
        "claim3_candidate_is_actual_recent_copy_rate": safe_mean(
            claim3_is_actual_recent_copy_count,
            claim3_step_count,
        ),
    }
    return replay_summary, selected_step_rows


def build_case_summary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    baseline = dict(row["baseline"])
    zero = dict(row["zero"])
    scale = dict(row["scale"])
    replay = dict(row["replay_summary"])
    base_copy = dict(baseline["copy_metrics"])
    zero_copy = dict(zero["copy_metrics"])
    scale_copy = dict(scale["copy_metrics"])
    return {
        "example_id": row["example_id"],
        "baseline_generated_tokens": int(baseline["generated_tokens"]),
        "zero_generated_tokens": int(zero["generated_tokens"]),
        "scale_generated_tokens": int(scale["generated_tokens"]),
        "zero_minus_baseline_generated_tokens": int(zero["generated_tokens"]) - int(baseline["generated_tokens"]),
        "scale_minus_baseline_generated_tokens": int(scale["generated_tokens"]) - int(baseline["generated_tokens"]),
        "baseline_recent_copy_rate": float(base_copy["recent_copy_rate_over_all_steps"]),
        "zero_recent_copy_rate": float(zero_copy["recent_copy_rate_over_all_steps"]),
        "scale_recent_copy_rate": float(scale_copy["recent_copy_rate_over_all_steps"]),
        "zero_minus_baseline_recent_copy_rate": float(zero_copy["recent_copy_rate_over_all_steps"]) - float(base_copy["recent_copy_rate_over_all_steps"]),
        "scale_minus_baseline_recent_copy_rate": float(scale_copy["recent_copy_rate_over_all_steps"]) - float(base_copy["recent_copy_rate_over_all_steps"]),
        "baseline_any_copy_rate": float(base_copy["any_copy_rate"]),
        "zero_any_copy_rate": float(zero_copy["any_copy_rate"]),
        "scale_any_copy_rate": float(scale_copy["any_copy_rate"]),
        "baseline_repetition_matched": bool(baseline["repetition_detection"]["matched"]),
        "zero_repetition_matched": bool(zero["repetition_detection"]["matched"]),
        "scale_repetition_matched": bool(scale["repetition_detection"]["matched"]),
        "replay_total_steps": int(replay.get("total_replay_steps", 0)),
        "replay_recent_copy_steps": int(replay.get("recent_copy_step_count", 0)),
        "replay_claim3_candidate_steps": int(replay.get("claim3_candidate_step_count", 0)),
        "replay_mean_zero_minus_baseline_recent_mass_all": float(replay.get("mean_zero_minus_baseline_recent_mass_all", 0.0)),
        "replay_mean_scale_minus_baseline_recent_mass_all": float(replay.get("mean_scale_minus_baseline_recent_mass_all", 0.0)),
        "replay_zero_recent_minus_random_gap_mean": float(replay.get("zero_recent_minus_random_gap_mean", 0.0)),
        "replay_scale_recent_minus_random_gap_mean": float(replay.get("scale_recent_minus_random_gap_mean", 0.0)),
        "claim3_zero_candidate_prob_delta_mean": float(replay.get("claim3_zero_candidate_prob_delta_mean", 0.0)),
        "claim3_scale_candidate_prob_delta_mean": float(replay.get("claim3_scale_candidate_prob_delta_mean", 0.0)),
        "claim3_ordered_prob_sign_rate": float(replay.get("claim3_ordered_prob_sign_rate", 0.0)),
        "claim3_ordered_logit_sign_rate": float(replay.get("claim3_ordered_logit_sign_rate", 0.0)),
    }


def process_rows(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
    stream_result_path: str = "",
    stream_step_path: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]], str, bool]:
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
    zero_operations = INTERVENTION_REGISTRY.get_required("zero")(targets, {})
    scale_operations = INTERVENTION_REGISTRY.get_required("scale")(targets, {"scale": args.boost_scale})

    result_rows: List[Dict[str, Any]] = []
    all_selected_step_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    interrupted = False
    result_handle: IO[str] | None = None
    step_handle: IO[str] | None = None
    try:
        if stream_result_path:
            result_handle = open(stream_result_path, "w", encoding="utf-8")
        if stream_step_path:
            step_handle = open(stream_step_path, "w", encoding="utf-8")

        iterator = tqdm(rows, desc=progress_desc, dynamic_ncols=True, leave=False)
        for idx, row in enumerate(iterator, start=1):
            example_id = str(row.get("example_id") or row.get("id") or f"row_{idx}")
            try:
                prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"prompt_resolution_failed: {exc}"})
                continue

            prompt_token_ids = backend.encode(prompt_prefix)
            example_seed = args.seed + idx * 1009
            try:
                seed_everything(example_seed)
                baseline_payload, baseline_token_ids = run_generation_condition(
                    backend=backend,
                    prompt_prefix=prompt_prefix,
                    generation_config=generation_config,
                    thresholds=thresholds,
                    recent_window=args.recent_window,
                    label="baseline",
                    operations=None,
                    attn_modules=attn_modules,
                    save_token_ids=args.save_token_ids,
                )

                seed_everything(example_seed)
                zero_payload, _ = run_generation_condition(
                    backend=backend,
                    prompt_prefix=prompt_prefix,
                    generation_config=generation_config,
                    thresholds=thresholds,
                    recent_window=args.recent_window,
                    label="zero",
                    operations=zero_operations,
                    attn_modules=attn_modules,
                    save_token_ids=args.save_token_ids,
                )

                seed_everything(example_seed)
                scale_payload, _ = run_generation_condition(
                    backend=backend,
                    prompt_prefix=prompt_prefix,
                    generation_config=generation_config,
                    thresholds=thresholds,
                    recent_window=args.recent_window,
                    label=f"scale_{args.boost_scale:g}",
                    operations=scale_operations,
                    attn_modules=attn_modules,
                    save_token_ids=args.save_token_ids,
                )

                replay_summary, example_selected_step_rows = analyze_replay_for_example(
                    backend=backend,
                    tokenizer=backend.tokenizer,
                    attn_modules=attn_modules,
                    prompt_token_ids=prompt_token_ids,
                    baseline_token_ids=baseline_token_ids,
                    zero_operations=zero_operations,
                    scale_operations=scale_operations,
                    args=args,
                    example_seed=example_seed,
                    example_id=example_id,
                )
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"processing_failed: {exc}"})
                continue

            result_row = {
                "example_id": example_id,
                "problem": row.get("question") or row.get("problem") or "",
                "baseline": baseline_payload,
                "zero": zero_payload,
                "scale": scale_payload,
                "replay_summary": replay_summary,
                "head_label": args.head_label,
                "boost_scale": args.boost_scale,
            }
            result_rows.append(result_row)
            append_jsonl_line(result_handle, result_row)
            for step_row in example_selected_step_rows:
                all_selected_step_rows.append(step_row)
                append_jsonl_line(step_handle, step_row)

            if idx % max(args.print_every, 1) == 0:
                print(
                    f"[Info] pid={os.getpid()} processed={idx} kept={len(result_rows)} skipped={len(skipped_rows)} "
                    f"example_id={example_id} "
                    f"baseline_recent_copy_rate={baseline_payload['copy_metrics']['recent_copy_rate_over_all_steps']:.4f} "
                    f"zero_recent_copy_rate={zero_payload['copy_metrics']['recent_copy_rate_over_all_steps']:.4f} "
                    f"scale_recent_copy_rate={scale_payload['copy_metrics']['recent_copy_rate_over_all_steps']:.4f}"
                )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if result_handle is not None:
            result_handle.close()
        if step_handle is not None:
            step_handle.close()

    return result_rows, all_selected_step_rows, skipped_rows, layer_path, interrupted


def run_worker(
    worker_id: int,
    gpu_id: int,
    shard_rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_rows_path: str,
    worker_selected_steps_path: str,
    worker_skipped_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    result_rows, selected_step_rows, skipped_rows, layer_path, interrupted = process_rows(
        shard_rows,
        args,
        device_map_override={"": gpu_id},
        progress_desc=f"Worker {worker_id} GPU{gpu_id}",
        stream_result_path=worker_rows_path,
        stream_step_path=worker_selected_steps_path,
    )
    write_json(worker_skipped_path, skipped_rows)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "row_count": len(result_rows),
        "selected_step_count": len(selected_step_rows),
        "skipped_count": len(skipped_rows),
        "worker_rows_path": worker_rows_path,
        "worker_selected_steps_path": worker_selected_steps_path,
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


def dedupe_rows_by_key(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    ordered: List[str] = []
    for row in rows:
        row_key = str(row.get(key) or "")
        if row_key and row_key not in deduped:
            ordered.append(row_key)
        deduped[row_key] = row
    return [deduped[row_key] for row_key in ordered]


def aggregate_generation_condition(result_rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    condition_rows = [dict(row.get(key) or {}) for row in result_rows]
    total_examples = len(condition_rows)
    total_tokens = sum(int(dict(item.get("copy_metrics") or {}).get("total_generated_tokens", 0)) for item in condition_rows)
    total_recent_copy = sum(int(dict(item.get("copy_metrics") or {}).get("recent_copy_count", 0)) for item in condition_rows)
    total_any_copy = sum(int(dict(item.get("copy_metrics") or {}).get("any_copy_count", 0)) for item in condition_rows)
    total_eligible_recent = sum(int(dict(item.get("copy_metrics") or {}).get("eligible_recent_context_steps", 0)) for item in condition_rows)
    repetition_hits = sum(1 for item in condition_rows if bool(dict(item.get("repetition_detection") or {}).get("matched")))
    mean_generated_tokens = 0.0 if total_examples == 0 else round(sum(int(item.get("generated_tokens", 0)) for item in condition_rows) / total_examples, 6)
    return {
        "examples": total_examples,
        "total_generated_tokens": int(total_tokens),
        "recent_copy_count": int(total_recent_copy),
        "any_copy_count": int(total_any_copy),
        "eligible_recent_context_steps": int(total_eligible_recent),
        "recent_copy_rate_over_all_steps": 0.0 if total_tokens == 0 else round(total_recent_copy / total_tokens, 6),
        "recent_copy_rate_over_eligible_steps": 0.0
        if total_eligible_recent == 0
        else round(total_recent_copy / total_eligible_recent, 6),
        "any_copy_rate": 0.0 if total_tokens == 0 else round(total_any_copy / total_tokens, 6),
        "repetition_hit_rate": 0.0 if total_examples == 0 else round(repetition_hits / total_examples, 6),
        "mean_generated_tokens": mean_generated_tokens,
    }


def aggregate_replay(result_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    replay_rows = [dict(row.get("replay_summary") or {}) for row in result_rows]
    total_steps = sum(int(row.get("total_replay_steps", 0)) for row in replay_rows)
    recent_copy_steps = sum(int(row.get("recent_copy_step_count", 0)) for row in replay_rows)
    claim3_steps = sum(int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)

    def _sum(name: str) -> float:
        return float(sum(float(row.get(name, 0.0)) for row in replay_rows))

    def _mean_from_sum(sum_name: str, count: int) -> float:
        total = _sum(sum_name)
        return 0.0 if count == 0 else round(total / count, 6)

    return {
        "total_replay_steps": int(total_steps),
        "recent_copy_step_count": int(recent_copy_steps),
        "claim3_candidate_step_count": int(claim3_steps),
        "mean_baseline_recent_mass_all": _mean_from_sum("sum_baseline_recent_mass_all", total_steps),
        "mean_zero_recent_mass_all": _mean_from_sum("sum_zero_recent_mass_all", total_steps),
        "mean_scale_recent_mass_all": _mean_from_sum("sum_scale_recent_mass_all", total_steps),
        "mean_baseline_older_seen_mass_all": _mean_from_sum("sum_baseline_older_seen_mass_all", total_steps),
        "mean_zero_older_seen_mass_all": _mean_from_sum("sum_zero_older_seen_mass_all", total_steps),
        "mean_scale_older_seen_mass_all": _mean_from_sum("sum_scale_older_seen_mass_all", total_steps),
        "mean_baseline_random_control_mass_all": _mean_from_sum("sum_baseline_random_control_mass_all", total_steps),
        "mean_zero_random_control_mass_all": _mean_from_sum("sum_zero_random_control_mass_all", total_steps),
        "mean_scale_random_control_mass_all": _mean_from_sum("sum_scale_random_control_mass_all", total_steps),
        "mean_zero_minus_baseline_recent_mass_all": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("mean_zero_minus_baseline_recent_mass_all", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "mean_scale_minus_baseline_recent_mass_all": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("mean_scale_minus_baseline_recent_mass_all", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "mean_zero_minus_baseline_random_control_mass_all": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("mean_zero_minus_baseline_random_control_mass_all", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "mean_scale_minus_baseline_random_control_mass_all": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("mean_scale_minus_baseline_random_control_mass_all", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "mean_zero_minus_baseline_older_seen_mass_all": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("mean_zero_minus_baseline_older_seen_mass_all", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "mean_scale_minus_baseline_older_seen_mass_all": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("mean_scale_minus_baseline_older_seen_mass_all", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "scale_recent_more_negative_than_random_rate": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("scale_recent_more_negative_than_random_rate", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "zero_recent_more_positive_than_random_rate": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("zero_recent_more_positive_than_random_rate", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "scale_recent_more_negative_than_older_rate": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("scale_recent_more_negative_than_older_rate", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "zero_recent_more_positive_than_older_rate": 0.0
        if total_steps == 0
        else round(
            sum(float(row.get("zero_recent_more_positive_than_older_rate", 0.0)) * int(row.get("total_replay_steps", 0)) for row in replay_rows)
            / total_steps,
            6,
        ),
        "mean_scale_minus_baseline_recent_mass_on_recent_copy_steps": 0.0
        if recent_copy_steps == 0
        else round(
            sum(
                float(row.get("mean_scale_minus_baseline_recent_mass_on_recent_copy_steps", 0.0))
                * int(row.get("recent_copy_step_count", 0))
                for row in replay_rows
            )
            / recent_copy_steps,
            6,
        ),
        "mean_zero_minus_baseline_recent_mass_on_recent_copy_steps": 0.0
        if recent_copy_steps == 0
        else round(
            sum(
                float(row.get("mean_zero_minus_baseline_recent_mass_on_recent_copy_steps", 0.0))
                * int(row.get("recent_copy_step_count", 0))
                for row in replay_rows
            )
            / recent_copy_steps,
            6,
        ),
        "claim3_zero_candidate_prob_delta_mean": 0.0
        if claim3_steps == 0
        else round(
            sum(float(row.get("claim3_zero_candidate_prob_delta_mean", 0.0)) * int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)
            / claim3_steps,
            6,
        ),
        "claim3_scale_candidate_prob_delta_mean": 0.0
        if claim3_steps == 0
        else round(
            sum(float(row.get("claim3_scale_candidate_prob_delta_mean", 0.0)) * int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)
            / claim3_steps,
            6,
        ),
        "claim3_zero_candidate_logit_delta_mean": 0.0
        if claim3_steps == 0
        else round(
            sum(float(row.get("claim3_zero_candidate_logit_delta_mean", 0.0)) * int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)
            / claim3_steps,
            6,
        ),
        "claim3_scale_candidate_logit_delta_mean": 0.0
        if claim3_steps == 0
        else round(
            sum(float(row.get("claim3_scale_candidate_logit_delta_mean", 0.0)) * int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)
            / claim3_steps,
            6,
        ),
        "claim3_ordered_prob_sign_rate": 0.0
        if claim3_steps == 0
        else round(
            sum(float(row.get("claim3_ordered_prob_sign_rate", 0.0)) * int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)
            / claim3_steps,
            6,
        ),
        "claim3_ordered_logit_sign_rate": 0.0
        if claim3_steps == 0
        else round(
            sum(float(row.get("claim3_ordered_logit_sign_rate", 0.0)) * int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)
            / claim3_steps,
            6,
        ),
        "claim3_zero_candidate_rank_better_rate": 0.0
        if claim3_steps == 0
        else round(
            sum(float(row.get("claim3_zero_candidate_rank_better_rate", 0.0)) * int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)
            / claim3_steps,
            6,
        ),
        "claim3_scale_candidate_rank_worse_rate": 0.0
        if claim3_steps == 0
        else round(
            sum(float(row.get("claim3_scale_candidate_rank_worse_rate", 0.0)) * int(row.get("claim3_candidate_step_count", 0)) for row in replay_rows)
            / claim3_steps,
            6,
        ),
    }


def write_report(path: Path, *, args: argparse.Namespace, claim1: Dict[str, Any], claim2: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# L0H3 Copy Suppression Experiment\n\n")
        f.write(f"- model: `{args.model_name_or_path}`\n")
        f.write(f"- head_label: `{args.head_label}`\n")
        f.write(f"- boost_scale: `{args.boost_scale}`\n")
        f.write(f"- max_examples: `{args.max_examples}`\n")
        f.write(f"- recent_window: `{args.recent_window}`\n")
        f.write(f"- max_new_tokens: `{args.max_new_tokens}`\n\n")
        f.write("## Claim 1\n\n")
        f.write(
            f"- baseline_recent_copy_rate={claim1['baseline']['recent_copy_rate_over_all_steps']:.6f}, "
            f"zero_recent_copy_rate={claim1['zero']['recent_copy_rate_over_all_steps']:.6f}, "
            f"scale_recent_copy_rate={claim1['scale']['recent_copy_rate_over_all_steps']:.6f}\n"
        )
        f.write(
            f"- baseline_mean_generated_tokens={claim1['baseline']['mean_generated_tokens']:.2f}, "
            f"zero_mean_generated_tokens={claim1['zero']['mean_generated_tokens']:.2f}, "
            f"scale_mean_generated_tokens={claim1['scale']['mean_generated_tokens']:.2f}\n\n"
        )
        f.write("## Claim 2 and 3\n\n")
        f.write(
            f"- mean_scale_minus_baseline_recent_mass_all={claim2['mean_scale_minus_baseline_recent_mass_all']:.6f}\n"
        )
        f.write(
            f"- mean_scale_minus_baseline_random_control_mass_all={claim2['mean_scale_minus_baseline_random_control_mass_all']:.6f}\n"
        )
        f.write(
            f"- claim3_scale_candidate_prob_delta_mean={claim2['claim3_scale_candidate_prob_delta_mean']:.6f}\n"
        )
        f.write(
            f"- claim3_zero_candidate_prob_delta_mean={claim2['claim3_zero_candidate_prob_delta_mean']:.6f}\n"
        )


def build_summary_outputs(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    result_rows: List[Dict[str, Any]],
    selected_step_rows: List[Dict[str, Any]],
    skipped_rows: List[Dict[str, str]],
    layer_path: str,
    parallel_enabled: bool,
    available_gpu_ids: List[int],
    worker_count: int,
    interrupted: bool,
) -> None:
    rows_path = output_dir / "rows.jsonl"
    case_summary_path = output_dir / "case_summary.csv"
    selected_steps_path = output_dir / "selected_step_rows.jsonl"
    claim1_path = output_dir / "claim1_summary.json"
    claim2_path = output_dir / "claim2_summary.json"
    claim3_path = output_dir / "claim3_summary.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    run_config_path = output_dir / "run_config.json"

    dump_jsonl(rows_path, result_rows)
    dump_jsonl(selected_steps_path, selected_step_rows)
    case_summary_rows = [build_case_summary_row(row) for row in result_rows]
    write_csv(case_summary_path, case_summary_rows)

    baseline_summary = aggregate_generation_condition(result_rows, "baseline")
    zero_summary = aggregate_generation_condition(result_rows, "zero")
    scale_summary = aggregate_generation_condition(result_rows, "scale")
    replay_summary = aggregate_replay(result_rows)

    claim1_summary = {
        "baseline": baseline_summary,
        "zero": zero_summary,
        "scale": scale_summary,
        "zero_minus_baseline_recent_copy_rate": round(
            float(zero_summary["recent_copy_rate_over_all_steps"]) - float(baseline_summary["recent_copy_rate_over_all_steps"]),
            6,
        ),
        "scale_minus_baseline_recent_copy_rate": round(
            float(scale_summary["recent_copy_rate_over_all_steps"]) - float(baseline_summary["recent_copy_rate_over_all_steps"]),
            6,
        ),
        "zero_minus_baseline_mean_generated_tokens": round(
            float(zero_summary["mean_generated_tokens"]) - float(baseline_summary["mean_generated_tokens"]),
            6,
        ),
        "scale_minus_baseline_mean_generated_tokens": round(
            float(scale_summary["mean_generated_tokens"]) - float(baseline_summary["mean_generated_tokens"]),
            6,
        ),
    }
    claim2_summary = {
        key: replay_summary[key]
        for key in [
            "total_replay_steps",
            "recent_copy_step_count",
            "mean_baseline_recent_mass_all",
            "mean_zero_recent_mass_all",
            "mean_scale_recent_mass_all",
            "mean_zero_minus_baseline_recent_mass_all",
            "mean_scale_minus_baseline_recent_mass_all",
            "mean_zero_minus_baseline_random_control_mass_all",
            "mean_scale_minus_baseline_random_control_mass_all",
            "mean_zero_minus_baseline_older_seen_mass_all",
            "mean_scale_minus_baseline_older_seen_mass_all",
            "scale_recent_more_negative_than_random_rate",
            "zero_recent_more_positive_than_random_rate",
            "scale_recent_more_negative_than_older_rate",
            "zero_recent_more_positive_than_older_rate",
            "mean_zero_minus_baseline_recent_mass_on_recent_copy_steps",
            "mean_scale_minus_baseline_recent_mass_on_recent_copy_steps",
        ]
    }
    claim3_summary = {
        key: replay_summary[key]
        for key in [
            "claim3_candidate_step_count",
            "claim3_zero_candidate_prob_delta_mean",
            "claim3_scale_candidate_prob_delta_mean",
            "claim3_zero_candidate_logit_delta_mean",
            "claim3_scale_candidate_logit_delta_mean",
            "claim3_ordered_prob_sign_rate",
            "claim3_ordered_logit_sign_rate",
            "claim3_zero_candidate_rank_better_rate",
            "claim3_scale_candidate_rank_worse_rate",
        ]
    }

    write_json(claim1_path, claim1_summary)
    write_json(claim2_path, claim2_summary)
    write_json(claim3_path, claim3_summary)
    write_json(
        summary_path,
        {
            "output_dir": str(output_dir),
            "processed_examples": len(result_rows),
            "skipped_examples": len(skipped_rows),
            "skipped_rows": skipped_rows,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
            "model_name_or_path": args.model_name_or_path,
            "head_label": args.head_label,
            "boost_scale": args.boost_scale,
            "recent_window": args.recent_window,
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids,
            "parallel_workers": worker_count,
            "decoder_layer_path": layer_path,
            "claim1": claim1_summary,
            "claim2": claim2_summary,
            "claim3": claim3_summary,
        },
    )
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
    write_report(report_path, args=args, claim1=claim1_summary, claim2=replay_summary)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_rows(args)
    if not all_rows:
        write_json(output_dir / "summary.json", {"message": "No rows to process.", "processed_examples": 0})
        write_json(output_dir / "run_config.json", {"args": vars(args)})
        (output_dir / "rows.jsonl").write_text("", encoding="utf-8")
        (output_dir / "selected_step_rows.jsonl").write_text("", encoding="utf-8")
        print("[Done] No rows to process.")
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
        "[Info] Copy-suppression experiment setup: "
        f"examples={len(all_rows)}, cuda_available={torch_cuda_available}, "
        f"available_gpu_ids={available_gpu_ids}, parallel_enabled={parallel_enabled}, "
        f"worker_count={worker_count}, head_label={args.head_label}, boost_scale={args.boost_scale}, "
        f"recent_window={args.recent_window}, max_new_tokens={args.max_new_tokens}"
    )

    result_rows: List[Dict[str, Any]] = []
    selected_step_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    layer_path = ""
    interrupted = False

    if parallel_enabled:
        worker_gpu_ids = available_gpu_ids[:worker_count]
        example_shards = split_examples_contiguous(all_rows, worker_count)
        worker_rows_paths: List[Path] = []
        worker_selected_step_paths: List[Path] = []
        worker_skipped_paths: List[Path] = []
        try:
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(example_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_rows in enumerate(example_shards):
                    gpu_id = worker_gpu_ids[worker_id]
                    worker_rows_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_rows.jsonl"
                    worker_selected_steps_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_selected_steps.jsonl"
                    worker_skipped_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_skipped.json"
                    futures.append(
                        pool.submit(
                            run_worker,
                            worker_id,
                            gpu_id,
                            shard_rows,
                            vars(args),
                            str(worker_rows_path),
                            str(worker_selected_steps_path),
                            str(worker_skipped_path),
                        )
                    )
                    worker_rows_paths.append(worker_rows_path)
                    worker_selected_step_paths.append(worker_selected_steps_path)
                    worker_skipped_paths.append(worker_skipped_path)
                try:
                    for fut in as_completed(futures):
                        worker_ret = fut.result()
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
            result_rows = dedupe_rows_by_key(result_rows, "example_id")
            for path in worker_selected_step_paths:
                selected_step_rows.extend(read_jsonl_if_exists(path))
            for path in worker_skipped_paths:
                if path.exists():
                    skipped_rows.extend(json.loads(path.read_text(encoding="utf-8")))
            if not args.keep_worker_outputs and not interrupted:
                for path in worker_rows_paths + worker_selected_step_paths + worker_skipped_paths:
                    if path.exists():
                        path.unlink()
    else:
        result_rows, selected_step_rows, skipped_rows, layer_path, interrupted = process_rows(
            all_rows,
            args,
            device_map_override=resolve_device_map(args.device_map, args.gpu_id),
            progress_desc="L0H3 copy-suppression",
        )

    result_rows = dedupe_rows_by_key(result_rows, "example_id")

    build_summary_outputs(
        args=args,
        output_dir=output_dir,
        result_rows=result_rows,
        selected_step_rows=selected_step_rows,
        skipped_rows=skipped_rows,
        layer_path=layer_path,
        parallel_enabled=parallel_enabled,
        available_gpu_ids=available_gpu_ids,
        worker_count=worker_count,
        interrupted=interrupted,
    )

    if interrupted:
        print("[Done] Copy-suppression experiment interrupted; partial outputs written:")
    else:
        print("[Done] Copy-suppression experiment finished:")
    print(f"- output_dir: {output_dir}")
    print(f"- processed_examples: {len(result_rows)}")
    print(f"- skipped_examples: {len(skipped_rows)}")
    print(f"- rows_jsonl: {output_dir / 'rows.jsonl'}")
    print(f"- selected_steps_jsonl: {output_dir / 'selected_step_rows.jsonl'}")
    print(f"- summary_json: {output_dir / 'summary.json'}")
    print(f"- report_md: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
