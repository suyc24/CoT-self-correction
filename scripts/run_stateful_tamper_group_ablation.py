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
from cot_research.head_ablation import MultiHeadAblationHookSet, list_all_heads
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import AttentionHeadSpec, get_input_device_for_model, parse_head_label
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig
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

DEFAULT_REFLECTION_KEYWORDS = [
    "wait", "Wait", "actually", "Actually", "hold on", "let me check",
    "mistake", "incorrect", "等等", "等一下", "不对", "重新", "检查", "重算",
]

DEFAULT_REFLECT_FIRST_TEXTS = [
    " Wait", " wait", "Wait", "wait", " Actually", " actually", "Actually",
    " But", " but", " However", " No", " no", "等等", "等一下", "不对", "重新", "检查",
]

DEFAULT_STOP_FIRST_TEXTS = [
    "</think>", "\n</think>", " </think>", " Therefore", " Therefore,",
    " Thus", " Thus,", " So", " Hence", " The",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stateful tamper group head ablation at the final forced token.")
    parser.add_argument("--input_jsonl", type=str, default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--head_group", type=str, required=True)
    parser.add_argument("--group_label", type=str, default="")
    parser.add_argument("--max_examples", type=int, default=40)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device_map", type=str, default="")
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--system_prompt", type=str, default=("Please reason step by step in <think>...</think>. Before closing </think>, include your interim result in \\boxed{}."))
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--stage1_stop_string", type=str, default="</think>")
    parser.add_argument("--max_stage1_tokens", type=int, default=2048)
    parser.add_argument("--max_continuation_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k_logprobs", type=int, default=10)
    parser.add_argument("--condition", choices=["tamper", "clean_force"], default="tamper")
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=5)
    return parser.parse_args()


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


def select_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl)
    selected = rows[args.start_idx : args.start_idx + args.max_examples]
    if not selected:
        raise ValueError("No examples selected.")
    return selected


def token_id_sequences(backend, texts: Iterable[str]) -> List[List[int]]:
    seqs: List[List[int]] = []
    for text in texts:
        ids = backend.encode(text)
        if ids:
            seqs.append([int(x) for x in ids])
    return seqs


def parse_head_group(model: torch.nn.Module, spec_text: str) -> Tuple[List[AttentionHeadSpec], List[torch.nn.Module], str]:
    all_heads, attn_modules, layer_path = list_all_heads(model)
    lookup = {(head.layer_idx, head.head_idx): head for head in all_heads}
    selected: List[AttentionHeadSpec] = []
    for item in spec_text.split(','):
        item = item.strip()
        if not item:
            continue
        key = parse_head_label(item)
        if key not in lookup:
            raise ValueError(f"Head {item!r} not found.")
        selected.append(lookup[key])
    if not selected:
        raise ValueError("--head_group did not select any heads.")
    return selected, attn_modules, layer_path


@torch.no_grad()
def force_tokens_with_final_group_ablation(
    model: torch.nn.Module,
    attn_modules: List[torch.nn.Module],
    targets: Sequence[AttentionHeadSpec],
    prefix_token_ids: Sequence[int],
    force_token_ids: Sequence[int],
) -> Tuple[ForcedState, Dict[str, Any]]:
    if len(force_token_ids) == 0:
        raise ValueError("force_token_ids must be non-empty for final-token group ablation.")
    device = get_input_device_for_model(model)
    past, logits = prefill_cache(model, prefix_token_ids)
    full_ids = [int(x) for x in prefix_token_ids]
    for token_id in force_token_ids[:-1]:
        full_ids.append(int(token_id))
        input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            raise ValueError("Model did not return past_key_values while forcing pre-ablation tokens.")
        logits = outputs.logits[0, -1]

    final_token_id = int(force_token_ids[-1])
    full_ids.append(final_token_id)
    input_ids = torch.tensor([[final_token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
    with MultiHeadAblationHookSet(attn_modules, list(targets)) as hookset:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    past = getattr(outputs, "past_key_values", None)
    if past is None:
        raise ValueError("Model did not return past_key_values after group ablation.")
    means = [h.first_call_abs_mean_before for h in hookset.hooks if h.first_call_abs_mean_before is not None]
    debug = {
        "hook_call_count": int(sum(h.call_count for h in hookset.hooks)),
        "hook_head_count": int(len(targets)),
        "hook_abs_mean_before": float(sum(means) / len(means)) if means else None,
    }
    return ForcedState(
        full_token_ids=full_ids,
        forced_token_ids=[int(x) for x in force_token_ids],
        logits=outputs.logits[0, -1],
        past_key_values=past,
        attentions=None,
    ), debug


def mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    return sum(vals) / len(vals) if vals else float('nan')


def summarize_behavior(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"total_rows": 0}
    return {
        "total_rows": len(rows),
        "first_wait_rate": mean([1.0 if str(r.get("first_generated_token_text")) in {"Wait", " wait", " Wait", "wait"} else 0.0 for r in rows]),
        "has_reflection_rate": mean([1.0 if r.get("has_reflection") else 0.0 for r in rows]),
        "mean_reflect_vs_stop": mean([float(r.get("reflect_vs_stop", float('nan'))) for r in rows]),
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
    heads, attn_modules, layer_path = parse_head_group(model, args.head_group)
    group_label = args.group_label.strip() or "GROUP[" + "+".join(head.label for head in heads) + "]"

    reflect_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    tracked_token_ids = sorted(set(reflect_token_ids + stop_token_ids))
    stop_sequences = token_id_sequences(backend, [])

    write_json(output_dir / "run_config.json", {
        "model_name_or_path": args.model_name_or_path,
        "input_jsonl": args.input_jsonl,
        "max_examples": args.max_examples,
        "start_idx": args.start_idx,
        "condition": args.condition,
        "head_group": [head.label for head in heads],
        "group_label": group_label,
        "layer_path": layer_path,
        "reflect_token_ids": reflect_token_ids,
        "stop_token_ids": stop_token_ids,
        "generation": asdict(gen_config),
    })

    behavior_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Stateful group ablation", dynamic_ncols=True)
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
                skipped_rows.append({
                    "example_id": example_id,
                    "reason": "clean_boxed_answer_not_correct",
                    "clean_boxed_answer": span.answer_text,
                    "correct_answer": correct_answer,
                })
                continue

            prefix_ids = prompt_ids + clean_gen_ids[: span.answer_start]
            clean_force_ids = clean_gen_ids[span.answer_start : span.box_end]
            wrong_answer_ids = backend.encode(wrong_answer)
            wrong_force_ids = wrong_answer_ids + clean_gen_ids[span.answer_end : span.box_end]
            force_ids = wrong_force_ids if args.condition == "tamper" else clean_force_ids

            baseline_state = force_tokens_from_cache(model, prefix_ids, force_ids, output_attentions_last=False)
            baseline_logits = baseline_state.logits.detach().float().cpu()
            baseline_reflect = logsumexp_token_set(baseline_logits, reflect_token_ids)
            baseline_stop = logsumexp_token_set(baseline_logits, stop_token_ids)
            baseline_cont = continue_from_state(
                model,
                tokenizer,
                baseline_state,
                max_new_tokens=args.max_continuation_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_id_sequences=stop_sequences,
                capture_attention=False,
                reflect_first_token_ids=reflect_token_ids,
                region_builder=None,
            )
            baseline_cont_text = tokenizer.decode(baseline_cont.token_ids, skip_special_tokens=False)
            baseline_full_text = tokenizer.decode(baseline_state.full_token_ids + baseline_cont.token_ids, skip_special_tokens=False)
            baseline_analysis = analyze_continuation_text(
                continuation=baseline_cont_text,
                full_text=baseline_full_text,
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                reflection_keywords=DEFAULT_REFLECTION_KEYWORDS,
            )
            baseline_first_id = baseline_cont.token_ids[0] if baseline_cont.token_ids else None
            behavior_rows.append({
                "example_id": example_id,
                "condition": args.condition,
                "question": question,
                "correct_answer": str(correct_answer),
                "wrong_answer": wrong_answer,
                "clean_box_answer": span.answer_text,
                "reflect_vs_stop": float(baseline_reflect - baseline_stop),
                "tracked_next_token_stats": candidate_stats(baseline_logits, tokenizer, tracked_token_ids),
                "next_token_topk": top_k_from_logits(baseline_logits, tokenizer, args.top_k_logprobs),
                "first_generated_token_id": baseline_first_id,
                "first_generated_token_text": tokenizer.decode([baseline_first_id], skip_special_tokens=False) if baseline_first_id is not None else "",
                "continuation_text": baseline_cont_text,
                **baseline_analysis,
            })

            ablated_state, debug = force_tokens_with_final_group_ablation(
                model,
                attn_modules,
                heads,
                prefix_ids,
                force_ids,
            )
            ablated_logits = ablated_state.logits.detach().float().cpu()
            ablated_reflect = logsumexp_token_set(ablated_logits, reflect_token_ids)
            ablated_stop = logsumexp_token_set(ablated_logits, stop_token_ids)
            ablated_cont = continue_from_state(
                model,
                tokenizer,
                ablated_state,
                max_new_tokens=args.max_continuation_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_id_sequences=stop_sequences,
                capture_attention=False,
                reflect_first_token_ids=reflect_token_ids,
                region_builder=None,
            )
            ablated_cont_text = tokenizer.decode(ablated_cont.token_ids, skip_special_tokens=False)
            ablated_full_text = tokenizer.decode(ablated_state.full_token_ids + ablated_cont.token_ids, skip_special_tokens=False)
            ablated_analysis = analyze_continuation_text(
                continuation=ablated_cont_text,
                full_text=ablated_full_text,
                correct_answer=correct_answer,
                wrong_answer=wrong_answer,
                reflection_keywords=DEFAULT_REFLECTION_KEYWORDS,
            )
            ablated_first_id = ablated_cont.token_ids[0] if ablated_cont.token_ids else None
            ablation_rows.append({
                "example_id": example_id,
                "condition": args.condition,
                "head_label": group_label,
                "head_group": [head.label for head in heads],
                "layer_idx": -1,
                "head_idx": -1,
                "baseline_reflect_vs_stop": float(baseline_reflect - baseline_stop),
                "ablated_reflect_vs_stop": float(ablated_reflect - ablated_stop),
                "delta_ablated_minus_baseline": float((ablated_reflect - ablated_stop) - (baseline_reflect - baseline_stop)),
                "baseline_reflect_logsum": baseline_reflect,
                "ablated_reflect_logsum": ablated_reflect,
                "baseline_stop_logsum": baseline_stop,
                "ablated_stop_logsum": ablated_stop,
                "baseline_first_generated_token_id": baseline_first_id,
                "baseline_first_generated_token_text": tokenizer.decode([baseline_first_id], skip_special_tokens=False) if baseline_first_id is not None else "",
                "baseline_has_reflection": bool(baseline_analysis.get("has_reflection")),
                "baseline_outcome_full_text": baseline_analysis.get("outcome_full_text"),
                "ablated_first_generated_token_id": ablated_first_id,
                "ablated_first_generated_token_text": tokenizer.decode([ablated_first_id], skip_special_tokens=False) if ablated_first_id is not None else "",
                "ablated_first_is_reflect_token": bool(ablated_first_id is not None and int(ablated_first_id) in set(reflect_token_ids)),
                "ablated_continuation_token_len": len(ablated_cont.token_ids),
                "ablated_continuation_stop_reason": ablated_cont.stop_reason,
                "ablated_continuation_text": ablated_cont_text,
                "ablated_has_reflection": bool(ablated_analysis.get("has_reflection")),
                "ablated_first_reflection_keyword": ablated_analysis.get("first_reflection_keyword"),
                "ablated_outcome_full_text": ablated_analysis.get("outcome_full_text"),
                "ablated_full_text_final_matches_correct": bool(ablated_analysis.get("full_text_final_matches_correct")),
                "ablated_full_text_final_matches_wrong": bool(ablated_analysis.get("full_text_final_matches_wrong")),
                **debug,
            })

            if (local_idx + 1) % max(int(args.print_every), 1) == 0:
                iterator.set_postfix({"kept_rows": len(behavior_rows), "skipped": len(skipped_rows)})
        except Exception as exc:
            skipped_rows.append({
                "example_id": example_id,
                "reason": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    dump_jsonl(output_dir / "behavior_rows.jsonl", behavior_rows)
    dump_jsonl(output_dir / "head_ablation_rows.jsonl", ablation_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    reason_counts: Dict[str, int] = defaultdict(int)
    for row in skipped_rows:
        reason_counts[str(row.get("reason"))] += 1
    write_json(output_dir / "summary.json", {
        "behavior": summarize_behavior(behavior_rows),
        "ablation_rows": len(ablation_rows),
        "skipped_count": len(skipped_rows),
        "skipped_reasons": dict(reason_counts),
        "head_group": [head.label for head in heads],
        "group_label": group_label,
    })
    print("[Done] Stateful tamper group ablation finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- behavior_rows: {len(behavior_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")
    print(f"- head_ablation_rows: {len(ablation_rows)}")


if __name__ == "__main__":
    main()
