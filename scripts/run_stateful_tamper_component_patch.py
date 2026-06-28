#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from cot_research.io_utils import dump_jsonl, write_json
from cot_research.model_utils import get_decoder_layers, get_input_device_for_model
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import (
    ForcedState,
    find_last_boxed_token_span,
    logsumexp_token_set,
    token_ids_for_first_tokens,
)

from run_stateful_tamper_residual_patch import (
    DEFAULT_REFLECT_FIRST_TEXTS,
    DEFAULT_REFLECTION_KEYWORDS,
    DEFAULT_STOP_FIRST_TEXTS,
    WAIT_FIRST_TOKENS,
    build_backend,
    build_generation_config,
    continuation_analysis,
    mean,
    parse_directions,
    parse_int_list,
    prefill_before_final,
    select_examples,
    stop_id_sequences,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stateful forced tamper attention-output / MLP-output patching at the final forced token."
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
    parser.add_argument("--max_continuation_tokens", type=int, default=16)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layers", type=str, required=True, help="Comma-separated decoder layer indices.")
    parser.add_argument(
        "--components",
        type=str,
        default="attn,mlp",
        help="Comma-separated components: attn,self_attn,mlp.",
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="clean_to_tamper,tamper_to_clean",
        help="Comma-separated: clean_to_tamper,tamper_to_clean.",
    )
    parser.add_argument("--print_every", type=int, default=5)
    return parser.parse_args()


def parse_components(text: str) -> List[str]:
    aliases = {"attn": "attn", "self_attn": "attn", "attention": "attn", "mlp": "mlp"}
    out: List[str] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in aliases:
            raise ValueError(f"Unsupported component {item!r}; expected attn or mlp.")
        norm = aliases[item]
        if norm not in out:
            out.append(norm)
    if not out:
        raise ValueError("--components did not select any component.")
    return out


def get_component_module(layer: torch.nn.Module, component: str) -> torch.nn.Module:
    if component == "attn":
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        if hasattr(layer, "attention"):
            return layer.attention
    if component == "mlp":
        if hasattr(layer, "mlp"):
            return layer.mlp
        if hasattr(layer, "feed_forward"):
            return layer.feed_forward
    raise ValueError(f"Cannot find component {component!r} on layer {layer!r}.")


def tensor_output(output: Any) -> Optional[torch.Tensor]:
    hidden = output[0] if isinstance(output, tuple) else output
    return hidden if isinstance(hidden, torch.Tensor) else None


class MultiComponentFinalTokenCapture:
    def __init__(self, specs: Sequence[Tuple[str, int, torch.nn.Module]]) -> None:
        self.specs = list(specs)
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.values: Dict[Tuple[str, int], torch.Tensor] = {}

    def __enter__(self) -> "MultiComponentFinalTokenCapture":
        for component, layer_idx, module in self.specs:
            self.handles.append(module.register_forward_hook(self._make_hook(component, layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_hook(self, component: str, layer_idx: int):
        key = (component, int(layer_idx))

        def hook(module, inputs, output):
            hidden = tensor_output(output)
            if hidden is not None and hidden.ndim == 3 and hidden.shape[1] == 1:
                self.values[key] = hidden[0, -1].detach().float().cpu()

        return hook


class SingleComponentFinalTokenPatch:
    def __init__(self, module: torch.nn.Module, patch_vector: torch.Tensor) -> None:
        self.module = module
        self.patch_vector = patch_vector
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.call_count = 0
        self.delta_norm: Optional[float] = None

    def __enter__(self) -> "SingleComponentFinalTokenPatch":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, inputs, output):
        hidden = tensor_output(output)
        if hidden is None or hidden.ndim != 3 or hidden.shape[1] != 1:
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
def run_final_forward_component(
    model: torch.nn.Module,
    past: Any,
    full_ids_before_final: Sequence[int],
    final_token_id: int,
    *,
    capture_specs: Sequence[Tuple[str, int, torch.nn.Module]] = (),
    patch_module: Optional[torch.nn.Module] = None,
    patch_vector: Optional[torch.Tensor] = None,
) -> Tuple[ForcedState, Dict[Tuple[str, int], torch.Tensor], Dict[str, Any]]:
    device = get_input_device_for_model(model)
    full_ids = [int(x) for x in full_ids_before_final] + [int(final_token_id)]
    input_ids = torch.tensor([[int(final_token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
    captures: Dict[Tuple[str, int], torch.Tensor] = {}
    debug: Dict[str, Any] = {}

    capture_ctx = MultiComponentFinalTokenCapture(capture_specs) if capture_specs else None
    patch_ctx = SingleComponentFinalTokenPatch(patch_module, patch_vector) if patch_module is not None and patch_vector is not None else None
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


def summarize_component_patch_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("component")), str(row.get("direction")), int(row.get("layer_idx")))].append(row)
    out: List[Dict[str, Any]] = []
    for (component, direction, layer_idx), group in grouped.items():
        baseline_wait = [str(row.get("baseline_first_generated_token_text") or "") in WAIT_FIRST_TOKENS for row in group]
        patched_wait = [str(row.get("patched_first_generated_token_text") or "") in WAIT_FIRST_TOKENS for row in group]
        out.append(
            {
                "component": component,
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
    out.sort(key=lambda row: (str(row["component"]), str(row["direction"]), int(row["layer_idx"])))
    return out


def summarize_baselines(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("condition"))].append(row)
    out: List[Dict[str, Any]] = []
    for condition, group in sorted(grouped.items()):
        first_wait = [str(row.get("first_generated_token_text") or "") in WAIT_FIRST_TOKENS for row in group]
        out.append(
            {
                "condition": condition,
                "count": len(group),
                "example_count": len({str(row.get("example_id")) for row in group}),
                "first_wait_rate": mean([1.0 if x else 0.0 for x in first_wait]),
                "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
                "mean_reflect_vs_stop": mean([row.get("reflect_vs_stop") for row in group]),
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
    decoder_layers, layer_path = get_decoder_layers(model)
    layers = list(decoder_layers)
    selected_layers = parse_int_list(args.layers, len(layers))
    components = parse_components(args.components)
    directions = parse_directions(args.directions)
    gen_config = build_generation_config(args)
    examples = select_examples(args)

    component_modules: Dict[Tuple[str, int], torch.nn.Module] = {}
    for layer_idx in selected_layers:
        for component in components:
            component_modules[(component, layer_idx)] = get_component_module(layers[layer_idx], component)
    capture_specs = [(component, layer_idx, module) for (component, layer_idx), module in component_modules.items()]

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
            "components": components,
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

    iterator = tqdm(examples, desc="Stateful component patch", dynamic_ncols=True)
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

            condition_data: Dict[str, Dict[str, Any]] = {}
            for condition, force_ids in {"clean_force": clean_force_ids, "tamper": tamper_force_ids}.items():
                past_before_final, ids_before_final = prefill_before_final(model, prefix_ids, force_ids)
                state, captures, _debug = run_final_forward_component(
                    model,
                    past_before_final,
                    ids_before_final,
                    int(force_ids[-1]),
                    capture_specs=capture_specs,
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
                for component in components:
                    clean_vec = condition_data["clean_force"]["captures"].get((component, layer_idx))
                    tamper_vec = condition_data["tamper"]["captures"].get((component, layer_idx))
                    if clean_vec is None or tamper_vec is None:
                        skipped_rows.append(
                            {
                                "example_id": example_id,
                                "reason": "missing_component_capture",
                                "layer_idx": layer_idx,
                                "component": component,
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
                        patched_state, _captures, debug = run_final_forward_component(
                            model,
                            target["past_before_final"],
                            target["ids_before_final"],
                            int(target["final_token_id"]),
                            patch_module=component_modules[(component, layer_idx)],
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
                                "component": component,
                                "direction": direction,
                                "source_condition": source_condition,
                                "target_condition": target_condition,
                                "layer_idx": int(layer_idx),
                                "baseline_reflect_vs_stop": float(target["reflect_vs_stop"]),
                                "patched_reflect_vs_stop": float(patched_rvs),
                                "delta_patched_minus_baseline": float(patched_rvs - float(target["reflect_vs_stop"])),
                                "baseline_first_generated_token_text": target_analysis.get("first_generated_token_text", ""),
                                "patched_first_generated_token_text": patched_analysis.get("first_generated_token_text", ""),
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
    dump_jsonl(output_dir / "component_patch_rows.jsonl", patch_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    write_csv(output_dir / "component_patch_summary.csv", summarize_component_patch_rows(patch_rows))
    write_csv(output_dir / "baseline_summary.csv", summarize_baselines(baseline_rows))
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
            "components": components,
            "directions": directions,
            "summary_rows": len(summarize_component_patch_rows(patch_rows)),
        },
    )

    print("[Done] Stateful component patch finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- baseline_rows: {len(baseline_rows)}")
    print(f"- patch_rows: {len(patch_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
