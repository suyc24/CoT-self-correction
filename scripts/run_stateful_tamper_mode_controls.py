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
from typing import Any, Dict, List, Sequence

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
    ForcedState,
    analyze_continuation_text,
    candidate_stats,
    continue_from_state,
    find_last_boxed_token_span,
    force_tokens_from_cache,
    logsumexp_token_set,
    prefill_cache,
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
stop_id_sequences = HELPERS.stop_id_sequences

WAIT_FIRST_TOKENS = {" Wait", " wait", "Wait", "wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stateful tamper against fresh-cache/text-only tamper controls."
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
    parser.add_argument("--max_continuation_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k_logprobs", type=int, default=10)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--coherent_window_chars",
        type=int,
        default=1200,
        help="For fresh_local_coherent_tamper_text, replace answer strings only in this final prefix window.",
    )
    parser.add_argument(
        "--coherent_answer_k_values",
        default="",
        help=(
            "Optional comma-separated k values. For each k, add a fresh_prev{k}_answer_coherent_tamper_text "
            "condition that replaces the k nearest clean/correct-answer string occurrences before the box."
        ),
    )
    parser.add_argument("--print_every", type=int, default=5)
    return parser.parse_args()


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


def select_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl)
    selected = rows[args.start_idx : args.start_idx + args.max_examples]
    if not selected:
        raise ValueError("No examples selected.")
    return selected


def parse_int_values(text: str) -> List[int]:
    values: List[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"k values must be positive, got {value}")
        if value not in values:
            values.append(value)
    return values


def answer_replacement_targets(clean_answer: Any, correct_answer: Any, wrong_answer: str) -> List[str]:
    targets: List[str] = []
    for target in [str(clean_answer), str(correct_answer)]:
        target = target.strip()
        if target and target not in targets and target != wrong_answer:
            targets.append(target)
    return targets


def replace_answer_occurrences_in_tail(
    text: str,
    *,
    targets: Sequence[str],
    wrong_answer: str,
    max_replacements: int,
) -> tuple[str, Dict[str, Any]]:
    """Replace the nearest non-overlapping target occurrences before the box."""
    spans: List[tuple[int, int, str]] = []
    for target in targets:
        start = 0
        while True:
            idx = text.find(target, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(target), target))
            start = idx + max(1, len(target))

    selected: List[tuple[int, int, str]] = []
    occupied: List[tuple[int, int]] = []
    for start, end, target in sorted(spans, key=lambda item: (item[1], item[0]), reverse=True):
        if any(not (end <= old_start or start >= old_end) for old_start, old_end in occupied):
            continue
        selected.append((start, end, target))
        occupied.append((start, end))
        if len(selected) >= max_replacements:
            break

    out = text
    for start, end, _target in sorted(selected, key=lambda item: item[0], reverse=True):
        out = out[:start] + wrong_answer + out[end:]

    return out, {
        "replacement_count": len(selected),
        "replacement_targets": list(targets),
        "replacement_spans_from_box": [
            {"start_from_prefix": start, "end_from_prefix": end, "target": target}
            for start, end, target in sorted(selected, key=lambda item: item[0])
        ],
    }


@torch.no_grad()
def fresh_state_from_full_ids(model: torch.nn.Module, full_token_ids: Sequence[int]) -> ForcedState:
    past, logits = prefill_cache(model, full_token_ids)
    return ForcedState(
        full_token_ids=[int(x) for x in full_token_ids],
        forced_token_ids=[],
        logits=logits,
        past_key_values=past,
        attentions=None,
    )


def summarize_behavior(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    out: List[Dict[str, Any]] = []
    for condition, group in sorted(grouped.items()):
        out.append(
            {
                "condition": condition,
                "count": len(group),
                "first_wait_rate": mean([1.0 if row.get("first_wait") else 0.0 for row in group]),
                "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
                "correct_full_rate": mean([1.0 if row.get("outcome_full_text") == "correct" else 0.0 for row in group]),
                "wrong_full_rate": mean([1.0 if row.get("outcome_full_text") == "wrong" else 0.0 for row in group]),
                "mean_reflect_vs_stop": mean([float(row.get("reflect_vs_stop", float("nan"))) for row in group]),
                "first_token_counts": dict(
                    sorted(
                        {
                            token: sum(1 for row in group if str(row.get("first_generated_token_text") or "") == token)
                            for token in {str(row.get("first_generated_token_text") or "") for row in group}
                        }.items(),
                        key=lambda item: -item[1],
                    )[:20]
                ),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    backend = build_backend(args)
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("This script requires an HF backend.")
    gen_config = build_generation_config(args)
    examples = select_examples(args)
    coherent_answer_k_values = parse_int_values(args.coherent_answer_k_values)

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
            "reflect_token_ids": reflect_token_ids,
            "stop_token_ids": stop_token_ids,
            "generation": asdict(gen_config),
            "conditions": [
                "stateful_clean_force",
                "stateful_tamper",
                "fresh_clean_text",
                "fresh_tamper_text",
                "fresh_local_coherent_tamper_text",
            ]
            + [f"fresh_prev{k}_answer_coherent_tamper_text" for k in coherent_answer_k_values],
            "coherent_answer_k_values": coherent_answer_k_values,
        },
    )

    behavior_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Stateful mode controls", dynamic_ncols=True)
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
            tamper_force_ids = wrong_answer_ids + clean_gen_ids[span.answer_end : span.box_end]
            clean_full_ids = prefix_ids + clean_force_ids
            tamper_full_ids = prefix_ids + tamper_force_ids
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
            tamper_force_text = tokenizer.decode(tamper_force_ids, skip_special_tokens=False)
            window_chars = max(0, int(args.coherent_window_chars))
            if window_chars > 0:
                prefix_head = prefix_text[:-window_chars]
                prefix_tail = prefix_text[-window_chars:]
            else:
                prefix_head = ""
                prefix_tail = prefix_text
            replacement_count = 0
            replacement_targets = []
            for target in [str(span.answer_text), str(correct_answer)]:
                target = target.strip()
                if target and target not in replacement_targets and target != wrong_answer:
                    replacement_targets.append(target)
            for target in replacement_targets:
                replacement_count += prefix_tail.count(target)
                prefix_tail = prefix_tail.replace(target, wrong_answer)
            coherent_tamper_text = prefix_head + prefix_tail + tamper_force_text
            coherent_tamper_ids = backend.encode(coherent_tamper_text)
            answer_k_ids_by_condition: Dict[str, tuple[List[int], Dict[str, Any]]] = {}
            for k in coherent_answer_k_values:
                k_text, k_meta = replace_answer_occurrences_in_tail(
                    prefix_text,
                    targets=replacement_targets,
                    wrong_answer=wrong_answer,
                    max_replacements=int(k),
                )
                answer_k_ids_by_condition[f"fresh_prev{k}_answer_coherent_tamper_text"] = (
                    backend.encode(k_text + tamper_force_text),
                    k_meta,
                )

            condition_states = {
                "stateful_clean_force": force_tokens_from_cache(model, prefix_ids, clean_force_ids),
                "stateful_tamper": force_tokens_from_cache(model, prefix_ids, tamper_force_ids),
                "fresh_clean_text": fresh_state_from_full_ids(model, clean_full_ids),
                "fresh_tamper_text": fresh_state_from_full_ids(model, tamper_full_ids),
                "fresh_local_coherent_tamper_text": fresh_state_from_full_ids(model, coherent_tamper_ids),
            }
            for condition, (condition_ids, _meta) in answer_k_ids_by_condition.items():
                condition_states[condition] = fresh_state_from_full_ids(model, condition_ids)

            for condition, state in condition_states.items():
                answer_k_meta = answer_k_ids_by_condition.get(condition, (None, None))[1]
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
                behavior_rows.append(
                    {
                        "example_id": example_id,
                        "condition": condition,
                        "question": question,
                        "correct_answer": str(correct_answer),
                        "wrong_answer": wrong_answer,
                        "clean_box_answer": span.answer_text,
                        "local_coherent_replacement_count": int(replacement_count),
                        "local_coherent_replacement_targets": replacement_targets,
                        "answer_k_replacement_count": (
                            int(answer_k_meta["replacement_count"]) if answer_k_meta is not None else None
                        ),
                        "answer_k_replacement_targets": (
                            answer_k_meta["replacement_targets"] if answer_k_meta is not None else None
                        ),
                        "answer_k_requested": (
                            int(condition.split("_prev", 1)[1].split("_answer", 1)[0])
                            if condition.startswith("fresh_prev")
                            else None
                        ),
                        "answer_k_replacement_spans_from_box": (
                            answer_k_meta["replacement_spans_from_box"] if answer_k_meta is not None else None
                        ),
                        "reflect_vs_stop": float(reflect_logsum - stop_logsum),
                        "reflect_logsum": float(reflect_logsum),
                        "stop_logsum": float(stop_logsum),
                        "first_generated_token_id": first_token_id,
                        "first_generated_token_text": first_token_text,
                        "first_wait": bool(first_token_text in WAIT_FIRST_TOKENS),
                        "continuation_token_len": len(continuation.token_ids),
                        "continuation_text": continuation_text,
                        "tracked_next_token_stats": candidate_stats(logits, tokenizer, tracked_token_ids),
                        "next_token_topk": top_k_from_logits(logits, tokenizer, args.top_k_logprobs),
                        **analysis,
                    }
                )
        except Exception as exc:
            skipped_rows.append({"example_id": example_id, "reason": "exception", "error": repr(exc)})

        if args.print_every > 0 and (local_idx + 1) % args.print_every == 0:
            iterator.set_postfix(rows=len(behavior_rows), skipped=len(skipped_rows))

    dump_jsonl(output_dir / "behavior_rows.jsonl", behavior_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    summary_rows = summarize_behavior(behavior_rows)
    write_csv(output_dir / "behavior_summary.csv", summary_rows)
    write_json(
        output_dir / "summary.json",
        {
            "behavior_rows": len(behavior_rows),
            "skipped_rows": len(skipped_rows),
            "conditions": [row["condition"] for row in summary_rows],
        },
    )


if __name__ == "__main__":
    main()
