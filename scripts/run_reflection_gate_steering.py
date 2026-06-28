#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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

from cot_research.answer_extraction import answers_match
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers, get_input_device_for_model
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import (
    ForcedState,
    analyze_continuation_text,
    continue_from_state,
    find_last_boxed_token_span,
    logsumexp_token_set,
    token_ids_for_first_tokens,
)

HELPER_PATH = SCRIPT_DIR / "run_stateful_tamper_attention_probe.py"
HELPER_SPEC = importlib.util.spec_from_file_location("_stateful_tamper_attention_probe_helpers", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load helper script at {HELPER_PATH}")
HELPERS = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPERS)

BOUNDARY_PATH = SCRIPT_DIR / "run_stateful_tamper_boundary_patch.py"
BOUNDARY_SPEC = importlib.util.spec_from_file_location("_stateful_boundary_helpers", BOUNDARY_PATH)
if BOUNDARY_SPEC is None or BOUNDARY_SPEC.loader is None:
    raise ImportError(f"Could not load boundary helper script at {BOUNDARY_PATH}")
BOUNDARY = importlib.util.module_from_spec(BOUNDARY_SPEC)
BOUNDARY_SPEC.loader.exec_module(BOUNDARY)

RESIDUAL_PATH = SCRIPT_DIR / "run_stateful_tamper_residual_patch.py"
RESIDUAL_SPEC = importlib.util.spec_from_file_location("_stateful_residual_helpers", RESIDUAL_PATH)
if RESIDUAL_SPEC is None or RESIDUAL_SPEC.loader is None:
    raise ImportError(f"Could not load residual helper script at {RESIDUAL_PATH}")
RESIDUAL = importlib.util.module_from_spec(RESIDUAL_SPEC)
RESIDUAL_SPEC.loader.exec_module(RESIDUAL)

DEFAULT_REFLECT_FIRST_TEXTS = HELPERS.DEFAULT_REFLECT_FIRST_TEXTS
DEFAULT_REFLECTION_KEYWORDS = HELPERS.DEFAULT_REFLECTION_KEYWORDS
DEFAULT_STOP_FIRST_TEXTS = HELPERS.DEFAULT_STOP_FIRST_TEXTS
WAIT_FIRST_TOKENS = {" Wait", " wait", "Wait", "wait"}

build_backend = HELPERS.build_backend
build_generation_config = HELPERS.build_generation_config
stop_id_sequences = HELPERS.stop_id_sequences
MultiBoundaryFinalTokenCapture = BOUNDARY.MultiBoundaryFinalTokenCapture
run_final_forward_boundary = BOUNDARY.run_final_forward_boundary
tensor_output = BOUNDARY.tensor_output
replace_tuple_arg = BOUNDARY.replace_tuple_arg
parse_sites = BOUNDARY.parse_sites
parse_int_list = RESIDUAL.parse_int_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and steer a residual reflection-gate direction from fresh tamper vs coherent tamper."
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
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    parser.add_argument("--assistant_prefix", default="")
    parser.add_argument("--stage1_stop_string", default="</think>")
    parser.add_argument("--max_stage1_tokens", type=int, default=8192)
    parser.add_argument("--max_continuation_tokens", type=int, default=8192)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coherent_window_chars", type=int, default=1200)
    parser.add_argument("--layers", default="19,20", help="Comma-separated decoder layer indices.")
    parser.add_argument("--sites", default="post_attn,block_output")
    parser.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    parser.add_argument(
        "--direction_types",
        default="gate,random,logit",
        help="Comma-separated controls: gate,random,logit.",
    )
    parser.add_argument(
        "--steer_conditions",
        default="coherent_plus,tamper_minus,clean_plus",
        help="Comma-separated: coherent_plus,tamper_minus,clean_plus.",
    )
    parser.add_argument(
        "--scale_mode",
        default="resid_std",
        choices=["resid_std", "diff_norm", "unit"],
        help="How to scale normalized directions before multiplying by alpha.",
    )
    parser.add_argument("--save_direction_tensors", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop_at_think_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=5)
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise ValueError("No float values selected.")
    return out


def parse_choice_list(text: str, supported: Sequence[str], name: str) -> List[str]:
    supported_set = set(supported)
    out: List[str] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in supported_set:
            raise ValueError(f"Unsupported {name}: {item}; expected one of {sorted(supported_set)}")
        if item not in out:
            out.append(item)
    if not out:
        raise ValueError(f"No {name} selected.")
    return out


def mean(values: Sequence[Any]) -> float:
    xs: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            xs.append(x)
    return sum(xs) / len(xs) if xs else float("nan")


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
def prefill_before_final_full_ids(model: torch.nn.Module, full_ids: Sequence[int]) -> Tuple[Any, List[int], int]:
    if len(full_ids) < 2:
        raise ValueError("Need at least two token ids to prefill before final token.")
    device = get_input_device_for_model(model)
    prefix = [int(x) for x in full_ids[:-1]]
    final_token_id = int(full_ids[-1])
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
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
    return past, prefix, final_token_id


def make_answer_token_coherent_ids(
    tokenizer,
    *,
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
    targets: List[str] = []
    for target in [str(clean_answer), str(correct_answer)]:
        target = target.strip()
        if target and target not in targets and target != wrong_answer:
            targets.append(target)
    replacement_count = 0
    for target in targets:
        replacement_count += prefix_tail.count(target)
        prefix_tail = prefix_tail.replace(target, wrong_answer)
    text = prefix_head + prefix_tail + tamper_force_text
    return tokenizer.encode(text, add_special_tokens=False), {
        "replacement_count": int(replacement_count),
        "replacement_targets": targets,
    }


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
    first_text = tokenizer.decode([first_id], skip_special_tokens=False) if first_id is not None else ""
    return {
        "first_generated_token_id": first_id,
        "first_generated_token_text": first_text,
        "first_wait": bool(first_text in WAIT_FIRST_TOKENS),
        "continuation_token_len": len(continuation.token_ids),
        "continuation_stop_reason": continuation.stop_reason,
        "continuation_text": continuation_text,
        **analysis,
    }


def tensor_normed(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    norm = float(vec.norm().item())
    if norm < eps:
        return torch.zeros_like(vec)
    return vec / norm


class BoundaryAdd:
    def __init__(self, layer: torch.nn.Module, site: str, add_vector: torch.Tensor) -> None:
        self.layer = layer
        self.site = site
        self.add_vector = add_vector
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.call_count = 0
        self.add_norm: Optional[float] = None
        self._pre_attn_resid: Optional[torch.Tensor] = None

    def __enter__(self) -> "BoundaryAdd":
        if self.site == "block_input":
            self.handles.append(self.layer.register_forward_pre_hook(self._block_input_hook))
        elif self.site == "post_attn":
            self.handles.append(self.layer.input_layernorm.register_forward_pre_hook(self._ln_pre_hook))
            self.handles.append(self.layer.self_attn.register_forward_hook(self._attn_post_hook))
        elif self.site == "block_output":
            self.handles.append(self.layer.register_forward_hook(self._block_output_hook))
        else:
            raise ValueError(f"Unsupported site {self.site}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _add_vec(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.add_vector.to(device=hidden.device, dtype=hidden.dtype)

    def _record(self, add_vec: torch.Tensor) -> None:
        self.call_count += 1
        self.add_norm = float(add_vec.detach().float().norm().item())

    def _block_input_hook(self, module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            return None
        hidden = args[0]
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            return None
        patched = hidden.clone()
        add_vec = self._add_vec(patched)
        self._record(add_vec)
        patched[0, -1] = patched[0, -1] + add_vec
        return replace_tuple_arg(args, 0, patched)

    def _ln_pre_hook(self, module, args):
        if args and isinstance(args[0], torch.Tensor) and args[0].ndim == 3 and args[0].shape[1] == 1:
            self._pre_attn_resid = args[0]

    def _attn_post_hook(self, module, inputs, output):
        attn_out = tensor_output(output)
        if attn_out is None or self._pre_attn_resid is None or attn_out.ndim != 3 or attn_out.shape[1] != 1:
            self._pre_attn_resid = None
            return output
        modified = attn_out.clone()
        add_vec = self._add_vec(modified)
        self._record(add_vec)
        modified[0, -1] = modified[0, -1] + add_vec
        self._pre_attn_resid = None
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def _block_output_hook(self, module, inputs, output):
        hidden = tensor_output(output)
        if hidden is None or hidden.ndim != 3 or hidden.shape[1] != 1:
            return output
        patched = hidden.clone()
        add_vec = self._add_vec(patched)
        self._record(add_vec)
        patched[0, -1] = patched[0, -1] + add_vec
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched


@torch.no_grad()
def run_final_forward_add(
    model: torch.nn.Module,
    past: Any,
    full_ids_before_final: Sequence[int],
    final_token_id: int,
    *,
    layers: Sequence[torch.nn.Module],
    add_layer_idx: int,
    add_site: str,
    add_vector: torch.Tensor,
) -> Tuple[ForcedState, Dict[str, Any]]:
    device = get_input_device_for_model(model)
    full_ids = [int(x) for x in full_ids_before_final] + [int(final_token_id)]
    input_ids = torch.tensor([[int(final_token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
    add_ctx = BoundaryAdd(layers[int(add_layer_idx)], str(add_site), add_vector)
    add_ctx.__enter__()
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
        debug = {"add_hook_call_count": int(add_ctx.call_count), "add_norm": add_ctx.add_norm}
        add_ctx.__exit__(None, None, None)
    new_past = getattr(outputs, "past_key_values", None)
    if new_past is None:
        raise ValueError("Model did not return past_key_values during steering forward.")
    return (
        ForcedState(
            full_token_ids=full_ids,
            forced_token_ids=[int(final_token_id)],
            logits=outputs.logits[0, -1],
            past_key_values=new_past,
            attentions=None,
        ),
        debug,
    )


def summarize_rows(rows: Sequence[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    out: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        first_wait = [bool(row.get("first_wait")) for row in group]
        baseline_wait = [bool(row.get("baseline_first_wait")) for row in group]
        row = {name: value for name, value in zip(group_keys, key)}
        row.update(
            {
                "count": len(group),
                "baseline_first_wait_rate": mean([1.0 if x else 0.0 for x in baseline_wait]),
                "first_wait_rate": mean([1.0 if x else 0.0 for x in first_wait]),
                "delta_first_wait_rate": mean([1.0 if x else 0.0 for x in first_wait])
                - mean([1.0 if x else 0.0 for x in baseline_wait]),
                "baseline_has_reflection_rate": mean([1.0 if row.get("baseline_has_reflection") else 0.0 for row in group]),
                "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
                "mean_baseline_reflect_vs_stop": mean([row.get("baseline_reflect_vs_stop") for row in group]),
                "mean_reflect_vs_stop": mean([row.get("reflect_vs_stop") for row in group]),
                "mean_delta_reflect_vs_stop": mean([row.get("delta_reflect_vs_stop") for row in group]),
                "mean_add_norm": mean([row.get("add_norm") for row in group]),
            }
        )
        out.append(row)
    out.sort(key=lambda r: tuple(str(r.get(k)) for k in group_keys))
    return out


def get_logit_direction(model: torch.nn.Module, reflect_ids: Sequence[int], stop_ids: Sequence[int]) -> Optional[torch.Tensor]:
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        return None
    with torch.no_grad():
        w = weight.detach().float().cpu()
        reflect = w[torch.tensor([int(x) for x in reflect_ids], dtype=torch.long)].mean(dim=0)
        stop = w[torch.tensor([int(x) for x in stop_ids], dtype=torch.long)].mean(dim=0)
        return tensor_normed(reflect - stop)


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
    selected_sites = parse_sites(args.sites)
    alphas = parse_float_list(args.alphas)
    direction_types = parse_choice_list(args.direction_types, ["gate", "random", "logit"], "direction_types")
    steer_conditions = parse_choice_list(
        args.steer_conditions,
        ["coherent_plus", "tamper_minus", "clean_plus"],
        "steer_conditions",
    )
    gen_config = build_generation_config(args)
    examples = select_examples(args)

    reflect_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    stop_sequences = stop_id_sequences(backend, [])
    clean_stop_strings = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    logit_dir = get_logit_direction(model, reflect_token_ids, stop_token_ids)

    write_json(
        output_dir / "run_config.json",
        {
            "model_name_or_path": args.model_name_or_path,
            "input_jsonl": args.input_jsonl,
            "max_examples": args.max_examples,
            "start_idx": args.start_idx,
            "gpu_id": args.gpu_id,
            "layers": selected_layers,
            "sites": selected_sites,
            "alphas": alphas,
            "direction_types": direction_types,
            "steer_conditions": steer_conditions,
            "scale_mode": args.scale_mode,
            "coherent_window_chars": args.coherent_window_chars,
            "stop_at_think_end": args.stop_at_think_end,
            "layer_path": layer_path,
            "reflect_token_ids": reflect_token_ids,
            "stop_token_ids": stop_token_ids,
            "generation": asdict(gen_config),
        },
    )

    baseline_rows: List[Dict[str, Any]] = []
    steering_examples: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Collect gate activations", dynamic_ncols=True)
    for local_idx, example in enumerate(iterator):
        example_id = str(example.get("id") or example.get("example_id") or str(local_idx + args.start_idx))
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
                stop_strings=clean_stop_strings,
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
            coherent_ids, coherent_meta = make_answer_token_coherent_ids(
                tokenizer,
                prefix_ids=prefix_ids,
                tamper_force_text=tamper_force_text,
                clean_answer=span.answer_text,
                correct_answer=str(correct_answer),
                wrong_answer=wrong_answer,
                window_chars=int(args.coherent_window_chars),
            )
            condition_full_ids = {
                "clean": clean_full_ids,
                "tamper": tamper_full_ids,
                "coherent": coherent_ids,
            }
            condition_data: Dict[str, Dict[str, Any]] = {}
            for condition, full_ids in condition_full_ids.items():
                past, ids_before_final, final_token_id = prefill_before_final_full_ids(model, full_ids)
                state, captures, _debug = run_final_forward_boundary(
                    model,
                    past,
                    ids_before_final,
                    final_token_id,
                    layers=layers,
                    capture_layers=selected_layers,
                    capture_sites=selected_sites,
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
                    "full_ids": [int(x) for x in full_ids],
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
                        "coherent_replacement_count": int(coherent_meta["replacement_count"]),
                        "reflect_vs_stop": float(reflect_vs_stop),
                        "delta_reflect_vs_stop": 0.0,
                        "baseline_reflect_vs_stop": float(reflect_vs_stop),
                        "baseline_first_wait": bool(analysis.get("first_wait")),
                        "baseline_has_reflection": bool(analysis.get("has_reflection")),
                        **analysis,
                    }
                )
            steering_examples.append(
                {
                    "example_id": example_id,
                    "question": question,
                    "correct_answer": str(correct_answer),
                    "wrong_answer": wrong_answer,
                    "clean_box_answer": span.answer_text,
                    "coherent_replacement_count": int(coherent_meta["replacement_count"]),
                    "conditions": condition_data,
                }
            )
            if args.print_every > 0 and (local_idx + 1) % args.print_every == 0:
                iterator.set_postfix({"usable": len(steering_examples), "skipped": len(skipped_rows)})
        except Exception as exc:
            skipped_rows.append({"example_id": example_id, "reason": "exception_collect", "error": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    direction_rows: List[Dict[str, Any]] = []
    directions: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 98765)
    for layer_idx in selected_layers:
        for site in selected_sites:
            diffs: List[torch.Tensor] = []
            diff_norms: List[float] = []
            resid_stds: List[float] = []
            for ex in steering_examples:
                tamper = ex["conditions"]["tamper"]["captures"].get((site, layer_idx))
                coherent = ex["conditions"]["coherent"]["captures"].get((site, layer_idx))
                clean_vec = ex["conditions"]["clean"]["captures"].get((site, layer_idx))
                if tamper is None or coherent is None:
                    continue
                diff = tamper.float() - coherent.float()
                diff_norms.append(float(diff.norm().item()))
                diffs.append(tensor_normed(diff))
                for vec in [tamper, coherent, clean_vec]:
                    if isinstance(vec, torch.Tensor):
                        resid_stds.append(float(vec.float().std().item()))
            if not diffs:
                continue
            gate_dir = tensor_normed(torch.stack(diffs, dim=0).mean(dim=0))
            dim = int(gate_dir.numel())
            random_dir = tensor_normed(torch.randn(dim, generator=generator))
            scale = 1.0
            if args.scale_mode == "resid_std":
                scale = mean(resid_stds)
            elif args.scale_mode == "diff_norm":
                scale = mean(diff_norms)
            elif args.scale_mode == "unit":
                scale = 1.0
            if not math.isfinite(scale) or scale == 0:
                scale = 1.0
            for direction_type, direction in [
                ("gate", gate_dir),
                ("random", random_dir),
                ("logit", logit_dir if logit_dir is not None and logit_dir.numel() == dim else None),
            ]:
                if direction_type not in direction_types or direction is None:
                    continue
                directions[(direction_type, layer_idx, site)] = {
                    "direction": tensor_normed(direction),
                    "scale": float(scale),
                    "n_pairs": len(diffs),
                    "mean_diff_norm": mean(diff_norms),
                    "mean_resid_std": mean(resid_stds),
                }
                direction_rows.append(
                    {
                        "direction_type": direction_type,
                        "layer_idx": int(layer_idx),
                        "site": site,
                        "n_pairs": len(diffs),
                        "scale": float(scale),
                        "mean_diff_norm": mean(diff_norms),
                        "mean_resid_std": mean(resid_stds),
                        "direction_norm": float(tensor_normed(direction).norm().item()),
                    }
                )

    steering_rows: List[Dict[str, Any]] = []
    iterator = tqdm(list(directions.items()), desc="Run gate steering", dynamic_ncols=True)
    for (direction_type, layer_idx, site), direction_info in iterator:
        unit_dir = direction_info["direction"]
        scale = float(direction_info["scale"])
        for alpha in alphas:
            if float(alpha) == 0.0 and direction_type != "gate":
                continue
            add_base = unit_dir * (float(alpha) * scale)
            for steer_condition in steer_conditions:
                if steer_condition == "coherent_plus":
                    target_condition = "coherent"
                    signed_add = add_base
                elif steer_condition == "tamper_minus":
                    target_condition = "tamper"
                    signed_add = -add_base
                elif steer_condition == "clean_plus":
                    target_condition = "clean"
                    signed_add = add_base
                else:
                    raise ValueError(steer_condition)
                for ex in steering_examples:
                    target = ex["conditions"][target_condition]
                    if (site, layer_idx) not in target["captures"]:
                        continue
                    try:
                        past, ids_before_final, final_token_id = prefill_before_final_full_ids(
                            model,
                            target["full_ids"],
                        )
                        patched_state, debug = run_final_forward_add(
                            model,
                            past,
                            ids_before_final,
                            int(final_token_id),
                            layers=layers,
                            add_layer_idx=int(layer_idx),
                            add_site=site,
                            add_vector=signed_add,
                        )
                        logits = patched_state.logits.detach().float().cpu()
                        rvs = logsumexp_token_set(logits, reflect_token_ids) - logsumexp_token_set(logits, stop_token_ids)
                        analysis = continuation_analysis(
                            model=model,
                            tokenizer=tokenizer,
                            state=patched_state,
                            max_new_tokens=args.max_continuation_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            stop_sequences=stop_sequences,
                            reflect_token_ids=reflect_token_ids,
                            correct_answer=ex["correct_answer"],
                            wrong_answer=ex["wrong_answer"],
                        )
                        baseline_analysis = target["analysis"]
                        steering_rows.append(
                            {
                                "example_id": ex["example_id"],
                                "direction_type": direction_type,
                                "layer_idx": int(layer_idx),
                                "site": site,
                                "alpha": float(alpha),
                                "scale": float(scale),
                                "steer_condition": steer_condition,
                                "target_condition": target_condition,
                                "correct_answer": ex["correct_answer"],
                                "wrong_answer": ex["wrong_answer"],
                                "clean_box_answer": ex["clean_box_answer"],
                                "coherent_replacement_count": int(ex["coherent_replacement_count"]),
                                "baseline_reflect_vs_stop": float(target["reflect_vs_stop"]),
                                "reflect_vs_stop": float(rvs),
                                "delta_reflect_vs_stop": float(rvs - float(target["reflect_vs_stop"])),
                                "baseline_first_generated_token_text": baseline_analysis.get("first_generated_token_text", ""),
                                "baseline_first_wait": bool(baseline_analysis.get("first_wait")),
                                "first_generated_token_text": analysis.get("first_generated_token_text", ""),
                                "first_wait": bool(analysis.get("first_wait")),
                                "baseline_has_reflection": bool(baseline_analysis.get("has_reflection")),
                                "has_reflection": bool(analysis.get("has_reflection")),
                                "baseline_outcome_full_text": baseline_analysis.get("outcome_full_text"),
                                "outcome_full_text": analysis.get("outcome_full_text"),
                                "continuation_text": analysis.get("continuation_text", ""),
                                **debug,
                            }
                        )
                    except Exception as exc:
                        skipped_rows.append(
                            {
                                "example_id": ex["example_id"],
                                "reason": "exception_steer",
                                "direction_type": direction_type,
                                "layer_idx": int(layer_idx),
                                "site": site,
                                "alpha": float(alpha),
                                "steer_condition": steer_condition,
                                "error": repr(exc),
                            }
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    dump_jsonl(output_dir / "baseline_rows.jsonl", baseline_rows)
    dump_jsonl(output_dir / "direction_rows.jsonl", direction_rows)
    dump_jsonl(output_dir / "steering_rows.jsonl", steering_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    write_csv(output_dir / "baseline_summary.csv", summarize_rows(baseline_rows, ["condition"]))
    write_csv(output_dir / "direction_summary.csv", direction_rows)
    write_csv(
        output_dir / "steering_summary.csv",
        summarize_rows(steering_rows, ["direction_type", "layer_idx", "site", "steer_condition", "alpha"]),
    )
    write_json(
        output_dir / "summary.json",
        {
            "usable_examples": len(steering_examples),
            "baseline_rows": len(baseline_rows),
            "direction_rows": len(direction_rows),
            "steering_rows": len(steering_rows),
            "skipped_rows": len(skipped_rows),
            "layers": selected_layers,
            "sites": selected_sites,
            "alphas": alphas,
            "direction_types": direction_types,
            "steer_conditions": steer_conditions,
        },
    )
    if args.save_direction_tensors:
        tensor_dir = output_dir / "direction_tensors"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        for (direction_type, layer_idx, site), info in directions.items():
            torch.save(
                {"direction": info["direction"], "scale": info["scale"]},
                tensor_dir / f"{direction_type}_L{layer_idx}_{site}.pt",
            )
    print("[Done] Reflection gate steering finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_examples: {len(steering_examples)}")
    print(f"- steering_rows: {len(steering_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
