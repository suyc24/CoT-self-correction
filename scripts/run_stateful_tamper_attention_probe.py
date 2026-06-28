#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import answers_match
from cot_research.generation import create_backend
from cot_research.head_ablation import list_all_heads
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.model_utils import parse_head_label
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig
from cot_research.stateful_tampering import (
    analyze_continuation_text,
    candidate_stats,
    continue_from_state,
    find_last_boxed_token_span,
    force_tokens_from_cache,
    force_tokens_with_final_head_ablation,
    logsumexp_token_set,
    make_region_map,
    summarize_attention_regions,
    token_ids_for_first_tokens,
    top_k_from_logits,
)


DEFAULT_REFLECTION_KEYWORDS = [
    "wait",
    "Wait",
    "actually",
    "Actually",
    "hold on",
    "let me check",
    "mistake",
    "incorrect",
    "等等",
    "等一下",
    "不对",
    "重新",
    "检查",
    "重算",
]

DEFAULT_REFLECT_FIRST_TEXTS = [
    " Wait",
    " wait",
    "Wait",
    "wait",
    " Actually",
    " actually",
    "Actually",
    " But",
    " but",
    " However",
    " No",
    " no",
    "等等",
    "等一下",
    "不对",
    "重新",
    "检查",
]

DEFAULT_STOP_FIRST_TEXTS = [
    "</think>",
    "\n</think>",
    " </think>",
    " Therefore",
    " Therefore,",
    " Thus",
    " Thus,",
    " So",
    " Hence",
    " The",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stateful forced boxed-answer tampering with attention-region diagnostics."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stateful_tamper_attention_probe"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device_map", type=str, default="")
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", type=str, default="eager")
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
    parser.add_argument("--max_stage1_tokens", type=int, default=4096)
    parser.add_argument("--max_continuation_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k_logprobs", type=int, default=10)
    parser.add_argument("--capture_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture_before_reflect_attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--conditions", type=str, default="tamper,clean_force")
    parser.add_argument("--ablate_heads", type=str, default="")
    parser.add_argument("--continue_ablation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_ablation_continuation_tokens", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)
    return parser.parse_args()


def parse_conditions(text: str) -> List[str]:
    out = [item.strip() for item in text.split(",") if item.strip()]
    if not out:
        raise ValueError("--conditions must contain at least one condition.")
    supported = {"tamper", "clean_force"}
    unknown = [item for item in out if item not in supported]
    if unknown:
        raise ValueError(f"Unsupported conditions: {unknown}. Supported={sorted(supported)}")
    return out


def select_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl)
    selected = rows[args.start_idx : args.start_idx + args.max_examples]
    if not selected:
        raise ValueError("No examples selected.")
    return selected


def build_backend(args: argparse.Namespace):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu_id))
    device_map: Any = {"": int(args.gpu_id)}
    if args.device_map.strip():
        device_map = args.device_map
    return create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map=device_map,
            load_in_half=args.load_in_half,
            use_fast_tokenizer=args.use_fast_tokenizer,
            use_safetensors=args.use_safetensors,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    )


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        stage1_stop_string=args.stage1_stop_string,
        max_stage1_tokens=args.max_stage1_tokens,
        max_new_tokens=args.max_continuation_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=True,
        capture_step_scores=False,
    )


def stop_id_sequences(backend, texts: Iterable[str]) -> List[List[int]]:
    seqs: List[List[int]] = []
    for text in texts:
        if not text:
            continue
        ids = backend.encode(text)
        if ids:
            seqs.append([int(x) for x in ids])
    return seqs


def mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def summarize_behavior(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    out: Dict[str, Any] = {"total_rows": len(rows), "conditions": {}}
    for condition, group in sorted(by_condition.items()):
        out["conditions"][condition] = {
            "count": len(group),
            "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
            "correct_full_rate": mean([1.0 if row.get("outcome_full_text") == "correct" else 0.0 for row in group]),
            "wrong_full_rate": mean([1.0 if row.get("outcome_full_text") == "wrong" else 0.0 for row in group]),
            "mean_reflect_vs_stop": mean([float(row.get("reflect_vs_stop", float("nan"))) for row in group]),
            "mean_forced_answer_token_len": mean([float(row.get("forced_answer_token_len", 0)) for row in group]),
            "first_token_counts": {},
        }
        counts: Dict[str, int] = defaultdict(int)
        for row in group:
            counts[str(row.get("first_generated_token_text") or "")] += 1
        out["conditions"][condition]["first_token_counts"] = dict(sorted(counts.items(), key=lambda item: -item[1])[:20])
    return out


def summarize_attention(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("condition")),
            str(row.get("stage")),
            int(row.get("layer_idx")),
            int(row.get("head_idx")),
        )
        grouped[key].append(row)
    summary: List[Dict[str, Any]] = []
    mass_keys = sorted(k for k in rows[0].keys() if k.startswith("mass_")) if rows else []
    for (condition, stage, layer_idx, head_idx), group in grouped.items():
        out: Dict[str, Any] = {
            "condition": condition,
            "stage": stage,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "head_label": f"L{layer_idx}H{head_idx}",
            "count": len(group),
            "mean_attention_entropy": mean([float(x.get("attention_entropy", float("nan"))) for x in group]),
            "mean_attention_max_value": mean([float(x.get("attention_max_value", float("nan"))) for x in group]),
        }
        for key in mass_keys:
            out[f"mean_{key}"] = mean([float(x.get(key, 0.0)) for x in group])
        summary.append(out)
    sort_key = "mean_mass_forced_box_full"
    summary.sort(key=lambda row: float(row.get(sort_key, 0.0)), reverse=True)
    return summary


def attach_attention_metadata(
    rows: List[Dict[str, Any]],
    *,
    example_id: Any,
    condition: str,
    stage: str,
    prompt_len: int,
    clean_box_answer: str,
    wrong_answer: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        merged = dict(row)
        merged.update(
            {
                "example_id": example_id,
                "condition": condition,
                "stage": stage,
                "head_label": f"L{row['layer_idx']}H{row['head_idx']}",
                "prompt_len": int(prompt_len),
                "clean_box_answer": clean_box_answer,
                "wrong_answer": wrong_answer,
            }
        )
        out.append(merged)
    return out


def parse_head_specs(model, spec_text: str):
    if not spec_text.strip():
        return [], [], ""
    all_heads, attn_modules, layer_path = list_all_heads(model)
    lookup = {(head.layer_idx, head.head_idx): head for head in all_heads}
    selected = []
    for item in spec_text.split(","):
        if not item.strip():
            continue
        key = parse_head_label(item.strip())
        if key not in lookup:
            raise ValueError(f"Head {item!r} not found.")
        selected.append(lookup[key])
    return selected, attn_modules, layer_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = parse_conditions(args.conditions)
    seed_everything(args.seed)

    backend = build_backend(args)
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("This script requires an HF backend.")
    gen_config = build_generation_config(args)
    examples = select_examples(args)

    reflect_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    tracked_token_ids = sorted(set(reflect_token_ids + stop_token_ids))
    stop_sequences = stop_id_sequences(backend, [])
    ablate_heads, attn_modules, layer_path = parse_head_specs(model, args.ablate_heads)

    run_config = {
        "model_name_or_path": args.model_name_or_path,
        "input_jsonl": args.input_jsonl,
        "max_examples": args.max_examples,
        "start_idx": args.start_idx,
        "gpu_id": args.gpu_id,
        "conditions": conditions,
        "capture_attention": bool(args.capture_attention),
        "capture_before_reflect_attention": bool(args.capture_before_reflect_attention),
        "reflect_token_ids": reflect_token_ids,
        "stop_token_ids": stop_token_ids,
        "reflect_token_texts": [
            tokenizer.decode([token_id], skip_special_tokens=False) for token_id in reflect_token_ids
        ],
        "stop_token_texts": [
            tokenizer.decode([token_id], skip_special_tokens=False) for token_id in stop_token_ids
        ],
        "ablate_heads": [head.label for head in ablate_heads],
        "continue_ablation": bool(args.continue_ablation),
        "max_ablation_continuation_tokens": int(args.max_ablation_continuation_tokens),
        "layer_path": layer_path,
        "generation": asdict(gen_config),
    }
    write_json(output_dir / "run_config.json", run_config)

    behavior_rows: List[Dict[str, Any]] = []
    attention_rows: List[Dict[str, Any]] = []
    before_reflect_attention_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Stateful tamper probe", dynamic_ncols=True)
    for local_idx, example in enumerate(iterator):
        example_id = example.get("id") or example.get("example_id") or str(local_idx + args.start_idx)
        question = str(example.get("question") or example.get("problem") or "")
        correct_answer = example.get("correct_answer")
        wrong_answer = str(example.get("wrong_answer") or "").strip()
        if not question or correct_answer is None or not wrong_answer:
            skipped_rows.append({"example_id": example_id, "reason": "missing_question_or_answers"})
            continue

        try:
            seed_everything(args.seed + local_idx)
            prompt = backend.build_prompt(question, gen_config)
            prompt_ids = backend.encode(prompt)
            clean = backend.generate(
                prompt,
                gen_config,
                stop_strings=[args.stage1_stop_string] if args.stage1_stop_string else None,
            )
            clean_gen_ids = [int(x) for x in clean.token_ids]
            span = find_last_boxed_token_span(tokenizer, clean_gen_ids)
            if span is None:
                skipped_rows.append({"example_id": example_id, "reason": "no_boxed_span"})
                continue
            if not args.allow_nonmatching_clean and not answers_match(span.answer_text, correct_answer):
                skipped_rows.append(
                    {
                        "example_id": example_id,
                        "reason": "clean_boxed_answer_not_correct",
                        "clean_boxed_answer": span.answer_text,
                        "correct_answer": correct_answer,
                    }
                )
                continue

            prefix_ids = prompt_ids + clean_gen_ids[: span.answer_start]
            clean_force_ids = clean_gen_ids[span.answer_start : span.box_end]
            wrong_answer_ids = backend.encode(wrong_answer)
            wrong_force_ids = wrong_answer_ids + clean_gen_ids[span.answer_end : span.box_end]
            clean_prefix_text = tokenizer.decode(prompt_ids + clean_gen_ids[: span.box_end], skip_special_tokens=False)
            prefix_before_answer_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)

            condition_force_ids = {
                "tamper": wrong_force_ids,
                "clean_force": clean_force_ids,
            }

            for condition in conditions:
                force_ids = condition_force_ids[condition]
                forced_answer_len = len(wrong_answer_ids) if condition == "tamper" else span.answer_end - span.answer_start
                state = force_tokens_from_cache(
                    model,
                    prefix_ids,
                    force_ids,
                    output_attentions_last=bool(args.capture_attention),
                )
                forced_prefix_len = len(state.full_token_ids)

                def region_builder(*, current_prefix_len: int):
                    return make_region_map(
                        prompt_len=len(prompt_ids),
                        span=span,
                        forced_answer_len=forced_answer_len,
                        forced_prefix_len=forced_prefix_len,
                        current_prefix_len=current_prefix_len,
                    )

                regions = region_builder(current_prefix_len=forced_prefix_len)
                if args.capture_attention:
                    raw_attn_rows = summarize_attention_regions(state.attentions, regions)
                    attention_rows.extend(
                        attach_attention_metadata(
                            raw_attn_rows,
                            example_id=example_id,
                            condition=condition,
                            stage="after_forced_box",
                            prompt_len=len(prompt_ids),
                            clean_box_answer=span.answer_text,
                            wrong_answer=wrong_answer,
                        )
                    )

                logits = state.logits.detach().float().cpu()
                reflect_logsum = logsumexp_token_set(logits, reflect_token_ids)
                stop_logsum = logsumexp_token_set(logits, stop_token_ids)
                continuation = continue_from_state(
                    model,
                    tokenizer,
                    state,
                    max_new_tokens=args.max_continuation_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_id_sequences=stop_sequences,
                    capture_attention=bool(args.capture_before_reflect_attention),
                    reflect_first_token_ids=reflect_token_ids,
                    region_builder=region_builder,
                )
                raw_continuation_text = tokenizer.decode(continuation.token_ids, skip_special_tokens=False)
                forced_prefix_text = tokenizer.decode(state.full_token_ids, skip_special_tokens=False)
                full_text = tokenizer.decode(state.full_token_ids + continuation.token_ids, skip_special_tokens=False)
                analysis = analyze_continuation_text(
                    continuation=raw_continuation_text,
                    full_text=full_text,
                    correct_answer=correct_answer,
                    wrong_answer=wrong_answer,
                    reflection_keywords=DEFAULT_REFLECTION_KEYWORDS,
                )

                for step in continuation.step_records:
                    for row in step.get("attention_region_rows") or []:
                        merged = dict(row)
                        merged.update(
                            {
                                "example_id": example_id,
                                "condition": condition,
                                "stage": "before_reflect_token",
                                "head_label": f"L{row['layer_idx']}H{row['head_idx']}",
                                "prompt_len": len(prompt_ids),
                                "clean_box_answer": span.answer_text,
                                "wrong_answer": wrong_answer,
                                "predicted_token_id": step.get("predicted_token_id"),
                                "predicted_token_text": step.get("predicted_token_text"),
                                "generation_step": step.get("step"),
                            }
                        )
                        before_reflect_attention_rows.append(merged)

                first_token_id = continuation.token_ids[0] if continuation.token_ids else None
                behavior_rows.append(
                    {
                        "example_id": example_id,
                        "condition": condition,
                        "question": question,
                        "correct_answer": str(correct_answer),
                        "wrong_answer": wrong_answer,
                        "clean_box_answer": span.answer_text,
                        "clean_answer_matches_correct": bool(answers_match(span.answer_text, correct_answer)),
                        "clean_stage_generated_tokens": clean.generated_tokens,
                        "clean_stage_contains_stop": bool(args.stage1_stop_string and args.stage1_stop_string in clean.continuation),
                        "box_span": span.to_dict(),
                        "prompt_token_len": len(prompt_ids),
                        "prefix_before_answer_token_len": len(prefix_ids),
                        "clean_force_token_len": len(clean_force_ids),
                        "wrong_answer_token_len": len(wrong_answer_ids),
                        "forced_answer_token_len": forced_answer_len,
                        "forced_token_len": len(force_ids),
                        "forced_prefix_token_len": len(state.full_token_ids),
                        "reflect_logsum": reflect_logsum,
                        "stop_logsum": stop_logsum,
                        "reflect_vs_stop": float(reflect_logsum - stop_logsum),
                        "tracked_next_token_stats": candidate_stats(logits, tokenizer, tracked_token_ids),
                        "next_token_topk": top_k_from_logits(logits, tokenizer, args.top_k_logprobs),
                        "first_generated_token_id": first_token_id,
                        "first_generated_token_text": tokenizer.decode([first_token_id], skip_special_tokens=False)
                        if first_token_id is not None
                        else "",
                        "continuation_token_len": len(continuation.token_ids),
                        "continuation_stop_reason": continuation.stop_reason,
                        "continuation_text": raw_continuation_text,
                        "forced_prefix_text": forced_prefix_text,
                        "prefix_before_answer_text": prefix_before_answer_text,
                        "clean_prefix_through_box_text": clean_prefix_text,
                        "full_text": full_text,
                        **analysis,
                    }
                )

                for head in ablate_heads:
                    ablated_state, debug = force_tokens_with_final_head_ablation(
                        model,
                        attn_modules[head.layer_idx],
                        head,
                        prefix_ids,
                        force_ids,
                    )
                    ablated_logits = ablated_state.logits.detach().float().cpu()
                    ablated_reflect = logsumexp_token_set(ablated_logits, reflect_token_ids)
                    ablated_stop = logsumexp_token_set(ablated_logits, stop_token_ids)
                    ablation_row = {
                        "example_id": example_id,
                        "condition": condition,
                        "head_label": head.label,
                        "layer_idx": head.layer_idx,
                        "head_idx": head.head_idx,
                        "baseline_reflect_vs_stop": float(reflect_logsum - stop_logsum),
                        "ablated_reflect_vs_stop": float(ablated_reflect - ablated_stop),
                        "delta_ablated_minus_baseline": float(
                            (ablated_reflect - ablated_stop) - (reflect_logsum - stop_logsum)
                        ),
                        "baseline_reflect_logsum": reflect_logsum,
                        "ablated_reflect_logsum": ablated_reflect,
                        "baseline_stop_logsum": stop_logsum,
                        "ablated_stop_logsum": ablated_stop,
                        "baseline_first_generated_token_id": first_token_id,
                        "baseline_first_generated_token_text": tokenizer.decode(
                            [first_token_id], skip_special_tokens=False
                        )
                        if first_token_id is not None
                        else "",
                        "baseline_has_reflection": bool(analysis.get("has_reflection")),
                        "baseline_outcome_full_text": analysis.get("outcome_full_text"),
                        **debug,
                    }
                    if args.continue_ablation:
                        ablation_max_new = (
                            int(args.max_ablation_continuation_tokens)
                            if int(args.max_ablation_continuation_tokens) > 0
                            else int(args.max_continuation_tokens)
                        )
                        ablated_continuation = continue_from_state(
                            model,
                            tokenizer,
                            ablated_state,
                            max_new_tokens=ablation_max_new,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            stop_id_sequences=stop_sequences,
                            capture_attention=False,
                            reflect_first_token_ids=reflect_token_ids,
                            region_builder=None,
                        )
                        ablated_cont_text = tokenizer.decode(
                            ablated_continuation.token_ids, skip_special_tokens=False
                        )
                        ablated_full_text = tokenizer.decode(
                            ablated_state.full_token_ids + ablated_continuation.token_ids,
                            skip_special_tokens=False,
                        )
                        ablated_analysis = analyze_continuation_text(
                            continuation=ablated_cont_text,
                            full_text=ablated_full_text,
                            correct_answer=correct_answer,
                            wrong_answer=wrong_answer,
                            reflection_keywords=DEFAULT_REFLECTION_KEYWORDS,
                        )
                        ablated_first_token_id = (
                            ablated_continuation.token_ids[0] if ablated_continuation.token_ids else None
                        )
                        ablation_row.update(
                            {
                                "ablated_first_generated_token_id": ablated_first_token_id,
                                "ablated_first_generated_token_text": tokenizer.decode(
                                    [ablated_first_token_id], skip_special_tokens=False
                                )
                                if ablated_first_token_id is not None
                                else "",
                                "ablated_first_is_reflect_token": bool(
                                    ablated_first_token_id is not None
                                    and int(ablated_first_token_id) in set(reflect_token_ids)
                                ),
                                "ablated_continuation_token_len": len(ablated_continuation.token_ids),
                                "ablated_continuation_stop_reason": ablated_continuation.stop_reason,
                                "ablated_continuation_text": ablated_cont_text,
                                "ablated_has_reflection": bool(ablated_analysis.get("has_reflection")),
                                "ablated_first_reflection_keyword": ablated_analysis.get(
                                    "first_reflection_keyword"
                                ),
                                "ablated_outcome_full_text": ablated_analysis.get("outcome_full_text"),
                                "ablated_full_text_final_matches_correct": bool(
                                    ablated_analysis.get("full_text_final_matches_correct")
                                ),
                                "ablated_full_text_final_matches_wrong": bool(
                                    ablated_analysis.get("full_text_final_matches_wrong")
                                ),
                            }
                        )
                    ablation_rows.append(ablation_row)

            if (local_idx + 1) % max(int(args.print_every), 1) == 0:
                iterator.set_postfix(
                    {
                        "kept_rows": len(behavior_rows),
                        "skipped": len(skipped_rows),
                        "attn_rows": len(attention_rows),
                    }
                )
        except Exception as exc:
            skipped_rows.append(
                {
                    "example_id": example_id,
                    "reason": "exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    dump_jsonl(output_dir / "behavior_rows.jsonl", behavior_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    if attention_rows:
        dump_jsonl(output_dir / "attention_after_forced_box_rows.jsonl", attention_rows)
        write_csv(output_dir / "attention_after_forced_box_summary.csv", summarize_attention(attention_rows))
    if before_reflect_attention_rows:
        dump_jsonl(output_dir / "attention_before_reflect_rows.jsonl", before_reflect_attention_rows)
        write_csv(output_dir / "attention_before_reflect_summary.csv", summarize_attention(before_reflect_attention_rows))
    if ablation_rows:
        dump_jsonl(output_dir / "head_ablation_rows.jsonl", ablation_rows)

    summary = summarize_behavior(behavior_rows)
    summary["skipped_count"] = len(skipped_rows)
    summary["skipped_reasons"] = dict(defaultdict(int))
    reason_counts: Dict[str, int] = defaultdict(int)
    for row in skipped_rows:
        reason_counts[str(row.get("reason"))] += 1
    summary["skipped_reasons"] = dict(reason_counts)
    summary["attention_after_forced_box_rows"] = len(attention_rows)
    summary["attention_before_reflect_rows"] = len(before_reflect_attention_rows)
    summary["head_ablation_rows"] = len(ablation_rows)
    write_json(output_dir / "summary.json", summary)

    print("[Done] Stateful tamper attention probe finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- behavior_rows: {len(behavior_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")
    print(f"- attention_after_forced_box_rows: {len(attention_rows)}")
    print(f"- attention_before_reflect_rows: {len(before_reflect_attention_rows)}")
    print(f"- head_ablation_rows: {len(ablation_rows)}")


if __name__ == "__main__":
    main()
