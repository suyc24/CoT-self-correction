from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import torch

from .ablation import HeadSpec, SingleHeadAblationHook
from .model_utils import generate_continuation
from .text_utils import (
    build_stage1_prompt,
    classify_outcome,
    detect_self_correction_keywords,
    extract_last_boxed,
    extract_think_segments,
    tamper_think_answer,
)


def prepare_example_prefix(
    model: torch.nn.Module,
    tokenizer,
    example: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    has_prompt_prefix = bool(str(example.get("prompt_prefix", "")).strip())
    has_question = bool(str(example.get("question", "")).strip())

    if has_prompt_prefix and args.legacy_use_prompt_prefix_first:
        prefix = str(example["prompt_prefix"])
        return {
            "ok": True,
            "mode": "legacy_prompt_prefix",
            "tampered_prefix": prefix,
            "stage1_prompt": None,
            "stage1_full_text": None,
            "stage1_generated_tokens": None,
            "tamper_method": "provided_prefix",
            "tampered_from": None,
            "tampered_to": None,
        }

    if not has_question:
        if has_prompt_prefix:
            prefix = str(example["prompt_prefix"])
            return {
                "ok": True,
                "mode": "legacy_prompt_prefix_fallback",
                "tampered_prefix": prefix,
                "stage1_prompt": None,
                "stage1_full_text": None,
                "stage1_generated_tokens": None,
                "tamper_method": "provided_prefix_fallback",
                "tampered_from": None,
                "tampered_to": None,
            }
        return {
            "ok": False,
            "reason": "record has neither question nor prompt_prefix",
        }

    stage1_prompt = build_stage1_prompt(
        tokenizer=tokenizer,
        question=str(example["question"]),
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        enable_thinking=args.enable_thinking,
    )
    _, stage1_full, stage1_tokens = generate_continuation(
        model=model,
        tokenizer=tokenizer,
        prompt_prefix=stage1_prompt,
        max_new_tokens=args.max_stage1_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_strings=[args.stage1_stop_string] if args.stage1_stop_string else None,
    )

    # Important: ignore any </think> that appears in system/user prompt text.
    # Only consider closures generated after assistant prompt begins.
    assistant_start_idx = stage1_prompt.lower().rfind("<think>")
    if assistant_start_idx < 0:
        if args.assistant_prefix and stage1_prompt.endswith(args.assistant_prefix):
            assistant_start_idx = len(stage1_prompt) - len(args.assistant_prefix)
        else:
            assistant_start_idx = len(stage1_prompt)

    close_pos = stage1_full.lower().find("</think>", len(stage1_prompt))
    if close_pos >= 0:
        stage1_until_think_end = stage1_full[: close_pos + len("</think>")]
    else:
        stage1_until_think_end = stage1_full

    stage1_assistant_stream = stage1_until_think_end[assistant_start_idx:]
    tamper_info = tamper_think_answer(
        stage1_full_text=stage1_assistant_stream,
        correct_answer=example["correct_answer"],
        wrong_answer=example["wrong_answer"],
        strict_tamper=args.strict_tamper,
    )
    if not tamper_info.get("ok", False):
        if has_prompt_prefix:
            prefix = str(example["prompt_prefix"])
            return {
                "ok": True,
                "mode": "legacy_prompt_prefix_fallback_after_stage1_fail",
                "tampered_prefix": prefix,
                "stage1_prompt": stage1_prompt,
                "stage1_full_text": stage1_until_think_end,
                "stage1_generated_tokens": stage1_tokens,
                "tamper_method": "stage1_fail_then_fallback_prefix",
                "tampered_from": None,
                "tampered_to": None,
                "stage1_fail_reason": tamper_info.get("reason"),
            }
        return {
            "ok": False,
            "reason": tamper_info.get("reason", "unknown_stage1_tamper_failure"),
            "stage1_prompt": stage1_prompt,
            "stage1_full_text": stage1_until_think_end,
        }

    return {
        "ok": True,
        "mode": "two_stage_auto_tamper",
        "tampered_prefix": stage1_until_think_end[:assistant_start_idx] + tamper_info["tampered_prefix"],
        "stage1_prompt": stage1_prompt,
        "stage1_full_text": stage1_until_think_end,
        "stage1_assistant_stream": stage1_assistant_stream,
        "stage1_generated_tokens": stage1_tokens,
        "tamper_method": tamper_info.get("tamper_method"),
        "tampered_from": tamper_info.get("tampered_from"),
        "tampered_to": tamper_info.get("tampered_to"),
        "think_content_before_tamper": tamper_info.get("think_content_before_tamper"),
        "think_content_after_tamper": tamper_info.get("think_content_after_tamper"),
    }


def analyze_generation(
    example: Dict[str, Any],
    prep_info: Dict[str, Any],
    condition: str,
    continuation: str,
    full_text: str,
    generated_tokens: int,
    keywords: List[str],
    head: Optional[HeadSpec],
    hook: Optional[SingleHeadAblationHook],
) -> Dict[str, Any]:
    think_segments = extract_think_segments(full_text)
    think_text = "\n\n".join(think_segments)
    hit, matched = detect_self_correction_keywords(think_text, keywords)
    boxed_continuation = extract_last_boxed(continuation)
    boxed_full_text = extract_last_boxed(full_text)
    outcome = classify_outcome(boxed_continuation, example["correct_answer"], example["wrong_answer"])
    outcome_full_text = classify_outcome(boxed_full_text, example["correct_answer"], example["wrong_answer"])

    row: Dict[str, Any] = {
        "example_id": example["id"],
        "condition": condition,
        "flow_mode": prep_info.get("mode"),
        "layer_idx": head.layer_idx if head is not None else None,
        "head_idx": head.head_idx if head is not None else None,
        "head_label": head.label if head is not None else "baseline",
        "question": example.get("question"),
        "prompt_prefix": prep_info.get("tampered_prefix"),
        "correct_answer": example["correct_answer"],
        "wrong_answer": example["wrong_answer"],
        "stage1_prompt": prep_info.get("stage1_prompt"),
        "stage1_full_text": prep_info.get("stage1_full_text"),
        "stage1_generated_tokens": prep_info.get("stage1_generated_tokens"),
        "tamper_method": prep_info.get("tamper_method"),
        "tampered_from": prep_info.get("tampered_from"),
        "tampered_to": prep_info.get("tampered_to"),
        "generated_continuation": continuation,
        "full_text": full_text,
        "think_text": think_text,
        "think_segment_count": len(think_segments),
        "self_correction_keyword_hit": hit,
        "matched_keywords": matched,
        "final_boxed_answer": boxed_continuation,
        "final_boxed_answer_full_text": boxed_full_text,
        "outcome": outcome,
        "outcome_full_text": outcome_full_text,
        "generated_tokens": generated_tokens,
    }

    if hook is not None:
        row["hook_call_count"] = hook.call_count
        row["hook_first_call_abs_mean_before"] = hook.first_call_abs_mean_before
        row["hook_first_call_abs_mean_after"] = hook.first_call_abs_mean_after

    return row
