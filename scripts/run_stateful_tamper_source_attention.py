#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.answer_extraction import answers_match
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import (
    analyze_continuation_text,
    candidate_stats,
    continue_from_state,
    find_last_boxed_token_span,
    force_tokens_from_cache,
    logsumexp_token_set,
    make_region_map,
    summarize_attention_regions,
    token_ids_for_first_tokens,
    top_k_from_logits,
)
HELPER_PATH = SCRIPT_DIR / "run_stateful_tamper_attention_probe.py"
HELPER_SPEC = importlib.util.spec_from_file_location("_stateful_tamper_attention_probe_helpers", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load helper script at {HELPER_PATH}")
HELPERS = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPERS)

DEFAULT_REFLECT_FIRST_TEXTS = HELPERS.DEFAULT_REFLECT_FIRST_TEXTS
DEFAULT_REFLECTION_KEYWORDS = HELPERS.DEFAULT_REFLECTION_KEYWORDS
DEFAULT_STOP_FIRST_TEXTS = HELPERS.DEFAULT_STOP_FIRST_TEXTS
build_backend = HELPERS.build_backend
build_generation_config = HELPERS.build_generation_config
parse_conditions = HELPERS.parse_conditions
select_examples = HELPERS.select_examples
stop_id_sequences = HELPERS.stop_id_sequences

WAIT_FIRST_TOKENS = {" Wait", " wait", "Wait", "wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token-level attention source probe for stateful forced tampering."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--max_examples", type=int, default=40)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device_map", default="")
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument(
        "--system_prompt",
        default=(
            "Please reason step by step in <think>...</think>. "
            "Before closing </think>, include your interim result in \\boxed{}."
        ),
    )
    parser.add_argument("--assistant_prefix", default="<think>\n")
    parser.add_argument("--stage1_stop_string", default="</think>")
    parser.add_argument("--max_stage1_tokens", type=int, default=2048)
    parser.add_argument("--max_continuation_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k_logprobs", type=int, default=10)
    parser.add_argument("--top_k_sources", type=int, default=8)
    parser.add_argument("--context_radius", type=int, default=4)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--conditions", default="tamper,clean_force")
    parser.add_argument("--layers", default="19,20")
    parser.add_argument("--print_every", type=int, default=5)
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("Expected at least one layer.")
    return out


def mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def region_for_index(regions: Dict[str, Tuple[int, int]], idx: int) -> str:
    priority = [
        "forced_answer",
        "box_marker_to_answer",
        "box_suffix",
        "forced_box_full",
        "prev_1",
        "prev_4",
        "prev_16",
        "prev_64",
        "reasoning_before_box",
        "prompt",
        "generated_after_force",
    ]
    for name in priority:
        span = regions.get(name)
        if span is None:
            continue
        if int(span[0]) <= int(idx) < int(span[1]):
            return name
    for name, (start, end) in regions.items():
        if int(start) <= int(idx) < int(end):
            return name
    return "unknown"


def context_text(tokenizer, token_ids: Sequence[int], idx: int, radius: int) -> str:
    start = max(0, int(idx) - int(radius))
    end = min(len(token_ids), int(idx) + int(radius) + 1)
    return tokenizer.decode([int(x) for x in token_ids[start:end]], skip_special_tokens=False)


def source_rows_from_attentions(
    attentions: Any,
    *,
    tokenizer,
    full_token_ids: Sequence[int],
    regions: Dict[str, Tuple[int, int]],
    layers: Sequence[int],
    top_k: int,
    context_radius: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if attentions is None:
        return rows
    selected = set(int(x) for x in layers)
    for layer_idx, layer_attn in enumerate(attentions):
        if layer_idx not in selected or layer_attn is None:
            continue
        attn = layer_attn.detach().float().cpu()
        if attn.ndim != 4:
            continue
        vecs = attn[0, :, -1, :]
        key_len = int(vecs.shape[-1])
        for head_idx in range(int(vecs.shape[0])):
            head_vec = vecs[head_idx]
            k = min(int(top_k), key_len)
            values, indices = torch.topk(head_vec, k=k)
            for rank, (value, key_idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                idx = int(key_idx)
                token_id = int(full_token_ids[idx]) if idx < len(full_token_ids) else -1
                rows.append(
                    {
                        "layer_idx": int(layer_idx),
                        "head_idx": int(head_idx),
                        "head_label": f"L{layer_idx}H{head_idx}",
                        "source_rank": int(rank),
                        "source_index": int(idx),
                        "source_token_id": int(token_id),
                        "source_token_text": tokenizer.decode([token_id], skip_special_tokens=False)
                        if token_id >= 0
                        else "",
                        "source_region": region_for_index(regions, idx),
                        "source_attention": float(value),
                        "source_context": context_text(tokenizer, full_token_ids, idx, context_radius),
                        "distance_from_query": int(key_len - 1 - idx),
                    }
                )
    return rows


def summarize_region_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["condition"]),
                str(row["behavior_group"]),
                int(row["layer_idx"]),
                int(row["head_idx"]),
            )
        ].append(row)
    mass_keys = sorted(k for k in rows[0].keys() if k.startswith("mass_")) if rows else []
    out: List[Dict[str, Any]] = []
    for (condition, behavior_group, layer_idx, head_idx), group in grouped.items():
        item: Dict[str, Any] = {
            "condition": condition,
            "behavior_group": behavior_group,
            "layer_idx": layer_idx,
            "head_idx": head_idx,
            "head_label": f"L{layer_idx}H{head_idx}",
            "count": len(group),
            "mean_reflect_vs_stop": mean([float(x.get("reflect_vs_stop", float("nan"))) for x in group]),
            "first_wait_rate": mean([1.0 if x.get("first_wait") else 0.0 for x in group]),
            "has_reflection_rate": mean([1.0 if x.get("has_reflection") else 0.0 for x in group]),
            "mean_attention_entropy": mean([float(x.get("attention_entropy", float("nan"))) for x in group]),
            "mean_attention_max_value": mean([float(x.get("attention_max_value", float("nan"))) for x in group]),
        }
        for key in mass_keys:
            item[f"mean_{key}"] = mean([float(x.get(key, 0.0)) for x in group])
        out.append(item)
    out.sort(key=lambda row: (row["condition"], row["behavior_group"], row["layer_idx"], row["head_idx"]))
    return out


def summarize_top_sources(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    top1 = [row for row in rows if int(row.get("source_rank", 0)) == 1]
    grouped: Dict[Tuple[str, str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in top1:
        grouped[
            (
                str(row["condition"]),
                str(row["behavior_group"]),
                int(row["layer_idx"]),
                int(row["head_idx"]),
                str(row["source_region"]),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (condition, behavior_group, layer_idx, head_idx, source_region), group in grouped.items():
        out.append(
            {
                "condition": condition,
                "behavior_group": behavior_group,
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "head_label": f"L{layer_idx}H{head_idx}",
                "top1_source_region": source_region,
                "top1_count": len(group),
                "mean_top1_attention": mean([float(x["source_attention"]) for x in group]),
                "example_count": len({str(x["example_id"]) for x in group}),
            }
        )
    out.sort(key=lambda row: (-int(row["top1_count"]), row["condition"], row["layer_idx"], row["head_idx"]))
    return out


def attach_common(rows: Iterable[Dict[str, Any]], common: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(common)
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    conditions = parse_conditions(args.conditions)
    layers = parse_int_list(args.layers)
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

    write_json(
        output_dir / "run_config.json",
        {
            "model_name_or_path": args.model_name_or_path,
            "input_jsonl": args.input_jsonl,
            "max_examples": args.max_examples,
            "start_idx": args.start_idx,
            "gpu_id": args.gpu_id,
            "conditions": conditions,
            "layers": layers,
            "top_k_sources": int(args.top_k_sources),
            "context_radius": int(args.context_radius),
            "reflect_token_ids": reflect_token_ids,
            "stop_token_ids": stop_token_ids,
            "generation": asdict(gen_config),
        },
    )

    behavior_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Stateful source attention", dynamic_ncols=True)
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
            condition_force_ids = {
                "clean_force": clean_force_ids,
                "tamper": wrong_answer_ids + clean_gen_ids[span.answer_end : span.box_end],
            }

            for condition in conditions:
                force_ids = condition_force_ids[condition]
                forced_answer_len = len(wrong_answer_ids) if condition == "tamper" else span.answer_end - span.answer_start
                state = force_tokens_from_cache(
                    model,
                    prefix_ids,
                    force_ids,
                    output_attentions_last=True,
                )
                forced_prefix_len = len(state.full_token_ids)
                regions = make_region_map(
                    prompt_len=len(prompt_ids),
                    span=span,
                    forced_answer_len=forced_answer_len,
                    forced_prefix_len=forced_prefix_len,
                    current_prefix_len=forced_prefix_len,
                )
                logits = state.logits.detach().float().cpu()
                reflect_logsum = logsumexp_token_set(logits, reflect_token_ids)
                stop_logsum = logsumexp_token_set(logits, stop_token_ids)
                reflect_vs_stop = float(reflect_logsum - stop_logsum)
                continuation = continue_from_state(
                    model,
                    tokenizer,
                    state,
                    max_new_tokens=args.max_continuation_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_id_sequences=stop_sequences,
                    capture_attention=False,
                    reflect_first_token_ids=reflect_token_ids,
                    region_builder=None,
                )
                continuation_text = tokenizer.decode(continuation.token_ids, skip_special_tokens=False)
                full_text = tokenizer.decode(state.full_token_ids + continuation.token_ids, skip_special_tokens=False)
                analysis = analyze_continuation_text(
                    continuation=continuation_text,
                    full_text=full_text,
                    correct_answer=correct_answer,
                    wrong_answer=wrong_answer,
                    reflection_keywords=DEFAULT_REFLECTION_KEYWORDS,
                )
                first_token_id = continuation.token_ids[0] if continuation.token_ids else None
                first_token_text = (
                    tokenizer.decode([first_token_id], skip_special_tokens=False) if first_token_id is not None else ""
                )
                first_wait = first_token_text in WAIT_FIRST_TOKENS
                behavior_group = "first_wait" if first_wait else "non_wait"
                common = {
                    "example_id": example_id,
                    "condition": condition,
                    "behavior_group": behavior_group,
                    "first_wait": bool(first_wait),
                    "has_reflection": bool(analysis.get("has_reflection")),
                    "correct_answer": str(correct_answer),
                    "wrong_answer": wrong_answer,
                    "clean_box_answer": span.answer_text,
                    "prompt_token_len": len(prompt_ids),
                    "forced_prefix_token_len": forced_prefix_len,
                    "reflect_vs_stop": reflect_vs_stop,
                    "first_generated_token_id": first_token_id,
                    "first_generated_token_text": first_token_text,
                }
                behavior_rows.append(
                    {
                        **common,
                        "question": question,
                        "continuation_token_len": len(continuation.token_ids),
                        "continuation_text": continuation_text,
                        "next_token_topk": top_k_from_logits(logits, tokenizer, args.top_k_logprobs),
                        "tracked_next_token_stats": candidate_stats(logits, tokenizer, tracked_token_ids),
                        **analysis,
                    }
                )

                raw_region_rows = [
                    row for row in summarize_attention_regions(state.attentions, regions) if int(row["layer_idx"]) in set(layers)
                ]
                region_rows.extend(attach_common(raw_region_rows, common))
                raw_source_rows = source_rows_from_attentions(
                    state.attentions,
                    tokenizer=tokenizer,
                    full_token_ids=state.full_token_ids,
                    regions=regions,
                    layers=layers,
                    top_k=args.top_k_sources,
                    context_radius=args.context_radius,
                )
                source_rows.extend(attach_common(raw_source_rows, common))
        except Exception as exc:
            skipped_rows.append({"example_id": example_id, "reason": "exception", "error": repr(exc)})

        if args.print_every > 0 and (local_idx + 1) % args.print_every == 0:
            iterator.set_postfix(rows=len(behavior_rows), skipped=len(skipped_rows))

    dump_jsonl(output_dir / "behavior_rows.jsonl", behavior_rows)
    dump_jsonl(output_dir / "attention_region_rows.jsonl", region_rows)
    dump_jsonl(output_dir / "source_topk_rows.jsonl", source_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    write_csv(output_dir / "source_region_summary.csv", summarize_region_rows(region_rows))
    write_csv(output_dir / "top1_source_region_summary.csv", summarize_top_sources(source_rows))
    write_json(
        output_dir / "summary.json",
        {
            "behavior_rows": len(behavior_rows),
            "attention_region_rows": len(region_rows),
            "source_topk_rows": len(source_rows),
            "skipped_rows": len(skipped_rows),
            "layers": layers,
            "conditions": conditions,
        },
    )


if __name__ == "__main__":
    main()
