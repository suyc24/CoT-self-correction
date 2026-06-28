#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers, get_input_device_for_model
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig
from cot_research.stateful_tampering import (
    ForcedState,
    analyze_continuation_text,
    continue_from_state,
    find_last_boxed_token_span,
    logsumexp_token_set,
    token_ids_for_first_tokens,
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

WAIT_FIRST_TOKENS = {"Wait", " wait", " Wait", "wait", "Actually", " Actually", "No", " No"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stateful forced tamper residual/block-output patching at the final forced token."
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
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
    parser.add_argument("--max_stage1_tokens", type=int, default=2048)
    parser.add_argument("--max_continuation_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--layers",
        type=str,
        default="",
        help="Comma-separated decoder layer indices. Empty means all layers.",
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="clean_to_tamper,tamper_to_clean",
        help="Comma-separated: clean_to_tamper,tamper_to_clean.",
    )
    parser.add_argument("--print_every", type=int, default=5)
    return parser.parse_args()


def parse_int_list(text: str, n_layers: int) -> List[int]:
    if not text.strip():
        return list(range(n_layers))
    out: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {idx} outside [0, {n_layers}).")
        out.append(idx)
    if not out:
        raise ValueError("--layers did not select any layer.")
    return out


def parse_directions(text: str) -> List[str]:
    supported = {"clean_to_tamper", "tamper_to_clean"}
    out = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in out if x not in supported]
    if unknown:
        raise ValueError(f"Unsupported directions: {unknown}")
    if not out:
        raise ValueError("--directions must not be empty.")
    return out


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


def stop_id_sequences(backend, texts: Iterable[str]) -> List[List[int]]:
    seqs: List[List[int]] = []
    for text in texts:
        ids = backend.encode(text)
        if ids:
            seqs.append([int(x) for x in ids])
    return seqs


def mean(values: Iterable[Any]) -> float:
    vals: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(x):
            vals.append(x)
    return sum(vals) / len(vals) if vals else float("nan")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class MultiLayerFinalTokenCapture:
    def __init__(self, layers: Sequence[torch.nn.Module], layer_indices: Sequence[int]) -> None:
        self.layers = layers
        self.layer_indices = list(layer_indices)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.values: Dict[int, torch.Tensor] = {}

    def __enter__(self) -> "MultiLayerFinalTokenCapture":
        for idx in self.layer_indices:
            self.handles.append(self.layers[idx].register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden, torch.Tensor) and hidden.ndim == 3 and hidden.shape[1] == 1:
                self.values[layer_idx] = hidden[0, -1].detach().float().cpu()
        return hook


class SingleLayerFinalTokenPatch:
    def __init__(self, layer: torch.nn.Module, patch_vector: torch.Tensor) -> None:
        self.layer = layer
        self.patch_vector = patch_vector
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.call_count = 0
        self.delta_norm: Optional[float] = None

    def __enter__(self) -> "SingleLayerFinalTokenPatch":
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[1] != 1:
            return output
        patched = hidden.clone()
        patch_vec = self.patch_vector.to(device=patched.device, dtype=patched.dtype)
        self.call_count += 1
        self.delta_norm = float((patched[0, -1].detach().float() - patch_vec.float()).norm().item())
        patched[0, -1] = patch_vec
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched


@torch.no_grad()
def prefill_before_final(
    model: torch.nn.Module,
    prefix_token_ids: Sequence[int],
    force_token_ids: Sequence[int],
) -> Tuple[Any, List[int]]:
    if not prefix_token_ids or not force_token_ids:
        raise ValueError("prefix_token_ids and force_token_ids must be non-empty.")
    device = get_input_device_for_model(model)
    input_ids = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=False,
        return_dict=True,
    )
    past = getattr(outputs, "past_key_values", None)
    if past is None:
        raise ValueError("Model did not return past_key_values during prefill.")
    full_ids = [int(x) for x in prefix_token_ids]
    for token_id in force_token_ids[:-1]:
        full_ids.append(int(token_id))
        step_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
        outputs = model(
            input_ids=step_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past = getattr(outputs, "past_key_values", None)
        if past is None:
            raise ValueError("Model did not return past_key_values while forcing tokens.")
    return past, full_ids


@torch.no_grad()
def run_final_forward(
    model: torch.nn.Module,
    past: Any,
    full_ids_before_final: Sequence[int],
    final_token_id: int,
    *,
    layers: Sequence[torch.nn.Module],
    capture_layers: Sequence[int] = (),
    patch_layer_idx: Optional[int] = None,
    patch_vector: Optional[torch.Tensor] = None,
) -> Tuple[ForcedState, Dict[int, torch.Tensor], Dict[str, Any]]:
    device = get_input_device_for_model(model)
    full_ids = [int(x) for x in full_ids_before_final] + [int(final_token_id)]
    input_ids = torch.tensor([[int(final_token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
    captures: Dict[int, torch.Tensor] = {}
    debug: Dict[str, Any] = {}

    capture_ctx = MultiLayerFinalTokenCapture(layers, capture_layers) if capture_layers else None
    patch_ctx = (
        SingleLayerFinalTokenPatch(layers[int(patch_layer_idx)], patch_vector)
        if patch_layer_idx is not None and patch_vector is not None
        else None
    )
    if capture_ctx is not None:
        capture_ctx.__enter__()
    if patch_ctx is not None:
        patch_ctx.__enter__()
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        if patch_ctx is not None:
            debug["patch_hook_call_count"] = int(patch_ctx.call_count)
            debug["patch_delta_norm"] = patch_ctx.delta_norm
            patch_ctx.__exit__(None, None, None)
        if capture_ctx is not None:
            captures = dict(capture_ctx.values)
            capture_ctx.__exit__(None, None, None)

    new_past = getattr(outputs, "past_key_values", None)
    if new_past is None:
        raise ValueError("Model did not return past_key_values during final forward.")
    return (
        ForcedState(
            full_token_ids=full_ids,
            forced_token_ids=[int(final_token_id)],
            logits=outputs.logits[0, -1],
            past_key_values=new_past,
            attentions=None,
        ),
        captures,
        debug,
    )


def continuation_analysis(
    *,
    model: torch.nn.Module,
    tokenizer,
    state: ForcedState,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    stop_sequences: Sequence[Sequence[int]],
    reflect_token_ids: Sequence[int],
    correct_answer: Any,
    wrong_answer: Any,
) -> Dict[str, Any]:
    continuation = continue_from_state(
        model,
        tokenizer,
        state,
        max_new_tokens=max_new_tokens,
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
    first_id = continuation.token_ids[0] if continuation.token_ids else None
    return {
        "first_generated_token_id": first_id,
        "first_generated_token_text": tokenizer.decode([first_id], skip_special_tokens=False)
        if first_id is not None
        else "",
        "continuation_token_len": len(continuation.token_ids),
        "continuation_stop_reason": continuation.stop_reason,
        "continuation_text": continuation_text,
        **analysis,
    }


def summarize_patch_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("direction")), int(row.get("layer_idx")))].append(row)
    out: List[Dict[str, Any]] = []
    for (direction, layer_idx), group in grouped.items():
        baseline_wait = [str(row.get("baseline_first_generated_token_text") or "") in WAIT_FIRST_TOKENS for row in group]
        patched_wait = [str(row.get("patched_first_generated_token_text") or "") in WAIT_FIRST_TOKENS for row in group]
        out.append(
            {
                "direction": direction,
                "layer_idx": layer_idx,
                "count": len(group),
                "baseline_first_wait_rate": mean([1.0 if x else 0.0 for x in baseline_wait]),
                "patched_first_wait_rate": mean([1.0 if x else 0.0 for x in patched_wait]),
                "delta_first_wait_rate": mean([1.0 if x else 0.0 for x in patched_wait])
                - mean([1.0 if x else 0.0 for x in baseline_wait]),
                "wait_to_nonwait_rate": mean([1.0 if b and not p else 0.0 for b, p in zip(baseline_wait, patched_wait)]),
                "nonwait_to_wait_rate": mean([1.0 if (not b) and p else 0.0 for b, p in zip(baseline_wait, patched_wait)]),
                "mean_baseline_reflect_vs_stop": mean([row.get("baseline_reflect_vs_stop") for row in group]),
                "mean_patched_reflect_vs_stop": mean([row.get("patched_reflect_vs_stop") for row in group]),
                "mean_delta_reflect_vs_stop": mean([row.get("delta_patched_minus_baseline") for row in group]),
                "baseline_has_reflection_rate": mean([1.0 if row.get("baseline_has_reflection") else 0.0 for row in group]),
                "patched_has_reflection_rate": mean([1.0 if row.get("patched_has_reflection") else 0.0 for row in group]),
                "mean_patch_delta_norm": mean([row.get("patch_delta_norm") for row in group]),
                "patch_hook_call_rate": mean([1.0 if int(row.get("patch_hook_call_count", 0)) > 0 else 0.0 for row in group]),
            }
        )
    out.sort(key=lambda row: (str(row["direction"]), -abs(float(row["delta_first_wait_rate"])), int(row["layer_idx"])))
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
    decoder_layers, layer_path = get_decoder_layers(model)
    layers = list(decoder_layers)
    selected_layers = parse_int_list(args.layers, len(layers))
    directions = parse_directions(args.directions)
    gen_config = build_generation_config(args)
    examples = select_examples(args)

    reflect_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    stop_sequences = stop_id_sequences(backend, [])

    write_json(
        output_dir / "run_config.json",
        {
            "model_name_or_path": args.model_name_or_path,
            "input_jsonl": args.input_jsonl,
            "max_examples": args.max_examples,
            "start_idx": args.start_idx,
            "layers": selected_layers,
            "directions": directions,
            "layer_path": layer_path,
            "reflect_token_ids": reflect_token_ids,
            "stop_token_ids": stop_token_ids,
            "generation": asdict(gen_config),
        },
    )

    baseline_rows: List[Dict[str, Any]] = []
    patch_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Stateful residual patch", dynamic_ncols=True)
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

            condition_force_ids = {
                "clean_force": clean_force_ids,
                "tamper": tamper_force_ids,
            }
            condition_data: Dict[str, Dict[str, Any]] = {}
            for condition, force_ids in condition_force_ids.items():
                past_before_final, ids_before_final = prefill_before_final(model, prefix_ids, force_ids)
                state, captures, _debug = run_final_forward(
                    model,
                    past_before_final,
                    ids_before_final,
                    int(force_ids[-1]),
                    layers=layers,
                    capture_layers=selected_layers,
                )
                logits = state.logits.detach().float().cpu()
                reflect_vs_stop = logsumexp_token_set(logits, reflect_token_ids) - logsumexp_token_set(
                    logits, stop_token_ids
                )
                analysis = continuation_analysis(
                    model=model,
                    tokenizer=tokenizer,
                    state=state,
                    max_new_tokens=args.max_continuation_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop_sequences=stop_sequences,
                    reflect_token_ids=reflect_token_ids,
                    correct_answer=correct_answer,
                    wrong_answer=wrong_answer,
                )
                condition_data[condition] = {
                    "past_before_final": past_before_final,
                    "ids_before_final": ids_before_final,
                    "final_token_id": int(force_ids[-1]),
                    "state": state,
                    "captures": captures,
                    "reflect_vs_stop": float(reflect_vs_stop),
                    "analysis": analysis,
                }
                baseline_rows.append(
                    {
                        "example_id": example_id,
                        "condition": condition,
                        "correct_answer": str(correct_answer),
                        "wrong_answer": wrong_answer,
                        "clean_box_answer": span.answer_text,
                        "reflect_vs_stop": float(reflect_vs_stop),
                        **analysis,
                    }
                )

            for layer_idx in selected_layers:
                clean_vec = condition_data["clean_force"]["captures"].get(layer_idx)
                tamper_vec = condition_data["tamper"]["captures"].get(layer_idx)
                if clean_vec is None or tamper_vec is None:
                    skipped_rows.append(
                        {
                            "example_id": example_id,
                            "reason": "missing_layer_capture",
                            "layer_idx": layer_idx,
                        }
                    )
                    continue
                patch_specs: List[Tuple[str, str, str, torch.Tensor]] = []
                if "clean_to_tamper" in directions:
                    patch_specs.append(("clean_to_tamper", "clean_force", "tamper", clean_vec))
                if "tamper_to_clean" in directions:
                    patch_specs.append(("tamper_to_clean", "tamper", "clean_force", tamper_vec))
                for direction, source_condition, target_condition, patch_vector in patch_specs:
                    target = condition_data[target_condition]
                    patched_state, _captures, debug = run_final_forward(
                        model,
                        target["past_before_final"],
                        target["ids_before_final"],
                        int(target["final_token_id"]),
                        layers=layers,
                        patch_layer_idx=layer_idx,
                        patch_vector=patch_vector,
                    )
                    patched_logits = patched_state.logits.detach().float().cpu()
                    patched_rvs = logsumexp_token_set(patched_logits, reflect_token_ids) - logsumexp_token_set(
                        patched_logits, stop_token_ids
                    )
                    patched_analysis = continuation_analysis(
                        model=model,
                        tokenizer=tokenizer,
                        state=patched_state,
                        max_new_tokens=args.max_continuation_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        stop_sequences=stop_sequences,
                        reflect_token_ids=reflect_token_ids,
                        correct_answer=correct_answer,
                        wrong_answer=wrong_answer,
                    )
                    target_analysis = target["analysis"]
                    patch_rows.append(
                        {
                            "example_id": example_id,
                            "direction": direction,
                            "source_condition": source_condition,
                            "target_condition": target_condition,
                            "layer_idx": int(layer_idx),
                            "baseline_reflect_vs_stop": float(target["reflect_vs_stop"]),
                            "patched_reflect_vs_stop": float(patched_rvs),
                            "delta_patched_minus_baseline": float(patched_rvs - float(target["reflect_vs_stop"])),
                            "baseline_first_generated_token_text": target_analysis.get(
                                "first_generated_token_text", ""
                            ),
                            "patched_first_generated_token_text": patched_analysis.get(
                                "first_generated_token_text", ""
                            ),
                            "baseline_has_reflection": bool(target_analysis.get("has_reflection")),
                            "patched_has_reflection": bool(patched_analysis.get("has_reflection")),
                            "baseline_outcome_full_text": target_analysis.get("outcome_full_text"),
                            "patched_outcome_full_text": patched_analysis.get("outcome_full_text"),
                            "patched_continuation_text": patched_analysis.get("continuation_text", ""),
                            **debug,
                        }
                    )

            if (local_idx + 1) % max(int(args.print_every), 1) == 0:
                iterator.set_postfix(
                    {
                        "baseline": len(baseline_rows),
                        "patch": len(patch_rows),
                        "skipped": len(skipped_rows),
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

    dump_jsonl(output_dir / "baseline_rows.jsonl", baseline_rows)
    dump_jsonl(output_dir / "residual_patch_rows.jsonl", patch_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    summary_rows = summarize_patch_rows(patch_rows)
    write_csv(output_dir / "residual_patch_summary.csv", summary_rows)
    reason_counts: Dict[str, int] = defaultdict(int)
    for row in skipped_rows:
        reason_counts[str(row.get("reason"))] += 1
    write_json(
        output_dir / "summary.json",
        {
            "baseline_rows": len(baseline_rows),
            "patch_rows": len(patch_rows),
            "skipped_rows": len(skipped_rows),
            "skipped_reasons": dict(reason_counts),
            "layers": selected_layers,
            "directions": directions,
            "summary_rows": len(summary_rows),
        },
    )

    print("[Done] Stateful residual patch finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- baseline_rows: {len(baseline_rows)}")
    print(f"- patch_rows: {len(patch_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
