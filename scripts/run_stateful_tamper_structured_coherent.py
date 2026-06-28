#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.answer_extraction import answers_match, extract_last_boxed
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import GenerationConfig
from cot_research.stateful_tampering import (
    ForcedState,
    analyze_continuation_text,
    candidate_stats,
    continue_from_state,
    find_last_boxed_token_span,
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
BANNED_REPAIR_WORDS = [
    "wait",
    "actually",
    "mistake",
    "incorrect",
    "inconsistent",
    "discrepancy",
    "misinterpret",
    "misunderstand",
    "wrong",
    "however",
    "instead",
    "correction",
    "revise",
    "fix",
    "target answer",
    "target boxed",
    "controlled experiment",
    "must have",
    "appears",
    "不对",
    "等等",
    "错误",
    "修正",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured coherent wrong-transcript controls for boxed-answer tampering."
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
    parser.add_argument("--rewrite_max_tokens_local", type=int, default=900)
    parser.add_argument("--rewrite_max_tokens_full", type=int, default=1800)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k_logprobs", type=int, default=10)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coherent_window_chars", type=int, default=1200)
    parser.add_argument("--include_stateful", action=argparse.BooleanOptionalAction, default=False)
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
                "valid_rewrite_rate": mean([1.0 if row.get("rewrite_valid", True) else 0.0 for row in group]),
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


def strip_code_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def extract_tagged(text: str, tag: str = "solution") -> str:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", flags=re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    return strip_code_fence(text)


def replace_last_boxed_answer(text: str, wrong_answer: str) -> Tuple[str, bool]:
    marker = "\\boxed"
    pos = text.rfind(marker)
    if pos < 0:
        return text.rstrip() + f"\n\\boxed{{{wrong_answer}}}", False
    brace_pos = text.find("{", pos + len(marker))
    if brace_pos < 0:
        return text.rstrip() + f"\n\\boxed{{{wrong_answer}}}", False
    depth = 0
    for idx in range(brace_pos, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: brace_pos + 1] + str(wrong_answer) + text[idx:], True
    return text.rstrip() + f"\n\\boxed{{{wrong_answer}}}", False


def answer_literal_pattern(answer: str) -> Optional[re.Pattern[str]]:
    ans = str(answer or "").strip()
    if not ans:
        return None
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", ans):
        return re.compile(rf"(?<![\d.]){re.escape(ans)}(?![\d.])")
    return re.compile(re.escape(ans), flags=re.IGNORECASE)


def contains_standalone_answer(text: str, answer: str) -> bool:
    pattern = answer_literal_pattern(answer)
    return bool(pattern and pattern.search(text))


def sanitize_solution_text(
    text: str,
    wrong_answer: str,
    *,
    clean_answer: str = "",
    forbid_clean_answer: bool = False,
) -> Dict[str, Any]:
    s = extract_tagged(text, "solution")
    # Keep only assistant reasoning content; remove accidental chat/control fragments.
    for marker in ["</think>", "<|im_end|>", "<|endoftext|>"]:
        if marker in s:
            s = s.split(marker, 1)[0]
    s = strip_code_fence(s).strip()
    for prefix in ["<solution>", "<think>"]:
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()
    s = re.sub(r"</?solution>", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"</?think>", "", s, flags=re.IGNORECASE).strip()
    s, had_box = replace_last_boxed_answer(s, wrong_answer)
    boxed = extract_last_boxed(s)
    valid_box = bool(boxed is not None and answers_match(boxed, wrong_answer))
    lowered = s.lower()
    banned_hits = [word for word in BANNED_REPAIR_WORDS if word.lower() in lowered]
    clean_answer_hit = bool(forbid_clean_answer and contains_standalone_answer(s, clean_answer))
    return {
        "text": s.strip(),
        "had_box": bool(had_box),
        "boxed_answer": boxed,
        "valid_box": bool(valid_box),
        "banned_hits": banned_hits,
        "clean_answer_hit": clean_answer_hit,
        "valid": bool(valid_box and not banned_hits and not clean_answer_hit),
    }


def make_answer_token_coherent(
    tokenizer,
    prefix_ids: Sequence[int],
    tamper_force_text: str,
    clean_answer: str,
    correct_answer: str,
    wrong_answer: str,
    window_chars: int,
) -> Tuple[List[int], Dict[str, Any]]:
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    if window_chars > 0:
        prefix_head = prefix_text[:-window_chars]
        prefix_tail = prefix_text[-window_chars:]
    else:
        prefix_head = ""
        prefix_tail = prefix_text
    replacement_targets: List[str] = []
    for target in [str(clean_answer), str(correct_answer)]:
        target = target.strip()
        if target and target not in replacement_targets and target != wrong_answer:
            replacement_targets.append(target)
    replacement_count = 0
    for target in replacement_targets:
        replacement_count += prefix_tail.count(target)
        prefix_tail = prefix_tail.replace(target, wrong_answer)
    coherent_tamper_text = prefix_head + prefix_tail + tamper_force_text
    return tokenizer.encode(coherent_tamper_text, add_special_tokens=False), {
        "replacement_count": int(replacement_count),
        "replacement_targets": replacement_targets,
        "text": coherent_tamper_text,
    }


def make_rewrite_prompt(
    *,
    mode: str,
    question: str,
    clean_solution: str,
    clean_answer: str,
    wrong_answer: str,
) -> str:
    if mode == "local_step":
        return f"""You are writing the final few sentences of a first-pass math solution.

Continue from the earlier solution prefix below. Write only the final 2-5 sentences. The continuation must sound like the original solver confidently finishing their solution, not like an editor forcing an answer.

Requirements:
- Output only final continuation sentences, not the whole solution.
- End with exactly one final boxed answer: \\boxed{{{wrong_answer}}}
- The final boxed answer must be {wrong_answer}, not {clean_answer}.
- The last calculation or stated final quantity before the box must directly support {wrong_answer}.
- Do not state {clean_answer} as the final quantity anywhere in the continuation.
- Do not mention a target answer, experiment, discrepancy, alternative, reinterpretation, or any correction process.
- Do not question the answer.
- Do not include <think> or </think>.
- Do not use these words anywhere: Wait, actually, mistake, incorrect, inconsistent, discrepancy, wrong, however, target, correction, revise.
- Wrap the output in <solution>...</solution> tags.

Problem:
{question}

Earlier solution prefix to continue from:
{clean_solution}

Final answer to write naturally in the solution: {wrong_answer}
"""
    if mode == "full_chain":
        return f"""You are writing a complete first-pass math solution.

Write an internally coherent alternative solution that naturally arrives at the specified boxed answer. You may make one early arithmetic slip or unit choice, but carry it through consistently. The text must read as a normal confident solution, not as a correction, critique, or forced edit.

Requirements:
- Output only the assistant reasoning text that comes after the existing '<think>\\n' prefix.
- End with exactly one final boxed answer: \\boxed{{{wrong_answer}}}
- The final boxed answer must be {wrong_answer}, not {clean_answer}.
- Do not mention a target answer, experiment, discrepancy, alternative, reinterpretation, or any correction process.
- Do not question the answer.
- Do not include <think> or </think>.
- Do not use these words anywhere: Wait, actually, mistake, incorrect, inconsistent, discrepancy, wrong, however, target, correction, revise.
- Wrap the output in <solution>...</solution> tags.

Problem:
{question}

Final answer to write naturally in the solution: {wrong_answer}
"""
    raise ValueError(mode)


def generate_rewrite(backend, *, prompt: str, max_new_tokens: int) -> str:
    cfg = GenerationConfig(
        system_prompt="You are a precise mathematical-solution editor. Follow the requested output format exactly.",
        assistant_prefix="<solution>\n",
        stage1_stop_string="",
        max_stage1_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        enable_thinking=False,
        capture_step_scores=False,
    )
    prompt_text = backend.build_prompt(prompt, cfg)
    result = backend.generate(prompt_text, cfg, stop_strings=None)
    return result.continuation


def split_local_context(clean_text_before_box: str, keep_back_chars: int = 1500) -> Tuple[str, str]:
    text = clean_text_before_box.rstrip()
    if len(text) <= keep_back_chars:
        cut = max(0, int(len(text) * 0.35))
    else:
        target = max(0, len(text) - keep_back_chars)
        candidates = [text.rfind("\n\n", 0, target), text.rfind(". ", 0, target), text.rfind("\n", 0, target)]
        cut = max(candidates)
        if cut < max(0, len(text) - 1800):
            cut = target
    kept = text[:cut].rstrip()
    context_tail = kept[-1200:]
    return kept, context_tail


def run_condition(
    *,
    model: torch.nn.Module,
    tokenizer,
    full_ids: Sequence[int],
    condition: str,
    example_id: str,
    question: str,
    correct_answer: str,
    wrong_answer: str,
    clean_box_answer: str,
    rewrite_meta: Dict[str, Any],
    reflect_token_ids: Sequence[int],
    stop_token_ids: Sequence[int],
    tracked_token_ids: Sequence[int],
    stop_sequences: Sequence[Sequence[int]],
    max_continuation_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k_logprobs: int,
) -> Dict[str, Any]:
    state = fresh_state_from_full_ids(model, full_ids)
    logits = state.logits.detach().float().cpu()
    reflect_logsum = logsumexp_token_set(logits, reflect_token_ids)
    stop_logsum = logsumexp_token_set(logits, stop_token_ids)
    continuation = continue_from_state(
        model,
        tokenizer,
        state,
        max_new_tokens=max_continuation_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
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
    first_token_text = tokenizer.decode([first_token_id], skip_special_tokens=False) if first_token_id is not None else ""
    return {
        "example_id": example_id,
        "condition": condition,
        "question": question,
        "correct_answer": str(correct_answer),
        "wrong_answer": str(wrong_answer),
        "clean_box_answer": str(clean_box_answer),
        "reflect_vs_stop": float(reflect_logsum - stop_logsum),
        "reflect_logsum": float(reflect_logsum),
        "stop_logsum": float(stop_logsum),
        "first_generated_token_id": first_token_id,
        "first_generated_token_text": first_token_text,
        "first_wait": bool(first_token_text in WAIT_FIRST_TOKENS),
        "continuation_token_len": len(continuation.token_ids),
        "continuation_text": continuation_text,
        "tracked_next_token_stats": candidate_stats(logits, tokenizer, tracked_token_ids),
        "next_token_topk": top_k_from_logits(logits, tokenizer, top_k_logprobs),
        **rewrite_meta,
        **analysis,
    }


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

    reflect_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    tracked_token_ids = sorted(set(reflect_token_ids + stop_token_ids))
    stop_sequences = stop_id_sequences(backend, [])

    conditions = [
        "fresh_clean_text",
        "fresh_tamper_text",
        "fresh_answer_token_coherent_tamper_text",
        "fresh_local_step_coherent_wrong_text",
        "fresh_full_chain_coherent_wrong_text",
    ]

    write_json(
        output_dir / "run_config.json",
        {
            "model_name_or_path": args.model_name_or_path,
            "input_jsonl": args.input_jsonl,
            "max_examples": args.max_examples,
            "start_idx": args.start_idx,
            "gpu_id": args.gpu_id,
            "conditions": conditions,
            "coherent_window_chars": args.coherent_window_chars,
            "rewrite_max_tokens_local": args.rewrite_max_tokens_local,
            "rewrite_max_tokens_full": args.rewrite_max_tokens_full,
            "reflect_token_ids": reflect_token_ids,
            "stop_token_ids": stop_token_ids,
            "generation": asdict(gen_config),
        },
    )

    behavior_rows: List[Dict[str, Any]] = []
    rewrite_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Structured coherent controls", dynamic_ncols=True)
    for local_idx, example in enumerate(iterator):
        example_id = str(example.get("id") or example.get("example_id") or str(local_idx + args.start_idx))
        question = str(example.get("question") or example.get("problem") or "")
        correct_answer = str(example.get("correct_answer")) if example.get("correct_answer") is not None else ""
        wrong_answer = str(example.get("wrong_answer") or "").strip()
        if not question or not correct_answer or not wrong_answer:
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
            tamper_force_text = tokenizer.decode(tamper_force_ids, skip_special_tokens=False)
            clean_solution_text = tokenizer.decode(clean_gen_ids[: span.box_end], skip_special_tokens=False)
            clean_text_before_box = tokenizer.decode(clean_gen_ids[: span.box_start], skip_special_tokens=False)
            local_kept_text, local_context_tail = split_local_context(clean_text_before_box)

            answer_token_ids, answer_token_meta = make_answer_token_coherent(
                tokenizer,
                prefix_ids,
                tamper_force_text,
                span.answer_text,
                correct_answer,
                wrong_answer,
                int(args.coherent_window_chars),
            )

            local_prompt = make_rewrite_prompt(
                mode="local_step",
                question=question,
                clean_solution=local_context_tail,
                clean_answer=span.answer_text,
                wrong_answer=wrong_answer,
            )
            full_prompt = make_rewrite_prompt(
                mode="full_chain",
                question=question,
                clean_solution="",
                clean_answer=span.answer_text,
                wrong_answer=wrong_answer,
            )
            local_raw = generate_rewrite(backend, prompt=local_prompt, max_new_tokens=args.rewrite_max_tokens_local)
            full_raw = generate_rewrite(backend, prompt=full_prompt, max_new_tokens=args.rewrite_max_tokens_full)
            local_final_info = sanitize_solution_text(
                local_raw,
                wrong_answer,
                clean_answer=span.answer_text,
                forbid_clean_answer=True,
            )
            full_info = sanitize_solution_text(
                full_raw,
                wrong_answer,
                clean_answer=span.answer_text,
                forbid_clean_answer=False,
            )
            local_combined_text = (local_kept_text.rstrip() + "\n" + local_final_info["text"].lstrip()).strip()
            local_info = dict(local_final_info)
            local_info["text"] = local_combined_text
            local_info["local_kept_chars"] = len(local_kept_text)
            local_ids = prompt_ids + backend.encode(local_info["text"])
            full_ids = prompt_ids + backend.encode(full_info["text"])

            rewrite_rows.extend(
                [
                    {
                        "example_id": example_id,
                        "rewrite_mode": "answer_token",
                        "valid": bool(answer_token_meta.get("replacement_count", 0) > 0),
                        "replacement_count": answer_token_meta.get("replacement_count"),
                        "replacement_targets": answer_token_meta.get("replacement_targets"),
                        "text": answer_token_meta.get("text"),
                    },
                    {
                        "example_id": example_id,
                        "rewrite_mode": "local_step",
                        "raw_text": local_raw,
                        **{f"rewrite_{k}": v for k, v in local_info.items() if k != "text"},
                        "text": local_info["text"],
                    },
                    {
                        "example_id": example_id,
                        "rewrite_mode": "full_chain",
                        "raw_text": full_raw,
                        **{f"rewrite_{k}": v for k, v in full_info.items() if k != "text"},
                        "text": full_info["text"],
                    },
                ]
            )

            condition_inputs = [
                ("fresh_clean_text", clean_full_ids, {"rewrite_valid": True, "rewrite_mode": "clean"}),
                ("fresh_tamper_text", tamper_full_ids, {"rewrite_valid": True, "rewrite_mode": "tamper"}),
                (
                    "fresh_answer_token_coherent_tamper_text",
                    answer_token_ids,
                    {
                        "rewrite_valid": bool(answer_token_meta.get("replacement_count", 0) > 0),
                        "rewrite_mode": "answer_token",
                        "replacement_count": answer_token_meta.get("replacement_count"),
                    },
                ),
                (
                    "fresh_local_step_coherent_wrong_text",
                    local_ids,
                    {
                        "rewrite_valid": bool(local_info["valid"]),
                        "rewrite_mode": "local_step",
                        "rewrite_had_box": bool(local_info["had_box"]),
                        "rewrite_boxed_answer": local_info["boxed_answer"],
                        "rewrite_valid_box": bool(local_info["valid_box"]),
                        "rewrite_banned_hits": local_info["banned_hits"],
                        "rewrite_clean_answer_hit": bool(local_info.get("clean_answer_hit", False)),
                    },
                ),
                (
                    "fresh_full_chain_coherent_wrong_text",
                    full_ids,
                    {
                        "rewrite_valid": bool(full_info["valid"]),
                        "rewrite_mode": "full_chain",
                        "rewrite_had_box": bool(full_info["had_box"]),
                        "rewrite_boxed_answer": full_info["boxed_answer"],
                        "rewrite_valid_box": bool(full_info["valid_box"]),
                        "rewrite_banned_hits": full_info["banned_hits"],
                        "rewrite_clean_answer_hit": bool(full_info.get("clean_answer_hit", False)),
                    },
                ),
            ]

            for condition, full_token_ids, meta in condition_inputs:
                behavior_rows.append(
                    run_condition(
                        model=model,
                        tokenizer=tokenizer,
                        full_ids=full_token_ids,
                        condition=condition,
                        example_id=example_id,
                        question=question,
                        correct_answer=correct_answer,
                        wrong_answer=wrong_answer,
                        clean_box_answer=span.answer_text,
                        rewrite_meta=meta,
                        reflect_token_ids=reflect_token_ids,
                        stop_token_ids=stop_token_ids,
                        tracked_token_ids=tracked_token_ids,
                        stop_sequences=stop_sequences,
                        max_continuation_tokens=args.max_continuation_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k_logprobs=args.top_k_logprobs,
                    )
                )
        except Exception as exc:
            skipped_rows.append({"example_id": example_id, "reason": "exception", "error": repr(exc)})

        if args.print_every > 0 and (local_idx + 1) % args.print_every == 0:
            iterator.set_postfix(rows=len(behavior_rows), rewrites=len(rewrite_rows), skipped=len(skipped_rows))

    dump_jsonl(output_dir / "behavior_rows.jsonl", behavior_rows)
    dump_jsonl(output_dir / "rewrite_rows.jsonl", rewrite_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    summary_rows = summarize_behavior(behavior_rows)
    write_csv(output_dir / "behavior_summary.csv", summary_rows)
    write_json(
        output_dir / "summary.json",
        {
            "behavior_rows": len(behavior_rows),
            "rewrite_rows": len(rewrite_rows),
            "skipped_rows": len(skipped_rows),
            "conditions": [row["condition"] for row in summary_rows],
        },
    )


if __name__ == "__main__":
    main()
