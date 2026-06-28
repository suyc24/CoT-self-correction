#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, replace
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
from cot_research.model_utils import get_decoder_layers
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import (
    analyze_continuation_text,
    continue_from_state,
    find_last_boxed_token_span,
    logsumexp_token_set,
    token_ids_for_first_tokens,
)

STEERING_PATH = SCRIPT_DIR / "run_reflection_gate_steering.py"
STEERING_SPEC = importlib.util.spec_from_file_location("_reflection_gate_steering", STEERING_PATH)
if STEERING_SPEC is None or STEERING_SPEC.loader is None:
    raise ImportError(f"Could not load {STEERING_PATH}")
STEERING = importlib.util.module_from_spec(STEERING_SPEC)
STEERING_SPEC.loader.exec_module(STEERING)

HELPER_PATH = SCRIPT_DIR / "run_stateful_tamper_attention_probe.py"
HELPER_SPEC = importlib.util.spec_from_file_location("_stateful_helpers", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load {HELPER_PATH}")
HELPERS = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPERS)

DEFAULT_REFLECT_FIRST_TEXTS = HELPERS.DEFAULT_REFLECT_FIRST_TEXTS
DEFAULT_REFLECTION_KEYWORDS = HELPERS.DEFAULT_REFLECTION_KEYWORDS
DEFAULT_STOP_FIRST_TEXTS = HELPERS.DEFAULT_STOP_FIRST_TEXTS
WAIT_FIRST_TOKENS = {" Wait", " wait", "Wait", "wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Projection-ablate a learned reflection-gate component from forced-tamper "
            "boundary states on held-out examples."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--direction_cache_in", default="")
    parser.add_argument("--direction_start_idx", type=int, default=0)
    parser.add_argument("--direction_max_examples", type=int, default=40)
    parser.add_argument("--eval_start_idx", type=int, default=40)
    parser.add_argument("--eval_max_examples", type=int, default=24)
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
    parser.add_argument("--max_continuation_tokens", type=int, default=96)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coherent_window_chars", type=int, default=1200)
    parser.add_argument("--layers", default="22")
    parser.add_argument("--sites", default="post_attn")
    parser.add_argument(
        "--projection_types",
        default="gate,gate_orth_logit,logit,random,random_matched_norm,logit_matched_norm",
        help=(
            "Comma-separated controls. gate/logit/random remove each direction's actual projection. "
            "*_matched_norm removes a vector with the gate projection norm along the control direction."
        ),
    )
    parser.add_argument("--scale_mode", choices=["diff_norm", "unit"], default="diff_norm")
    parser.add_argument("--stop_at_think_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=4)
    return parser.parse_args()


def parse_int_list(text: str, n_layers: int) -> List[int]:
    out: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx = n_layers + idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index out of range: {item}")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("No layers selected.")
    return out


def parse_choice_list(text: str, supported: Sequence[str], name: str) -> List[str]:
    supported_set = set(supported)
    out: List[str] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in supported_set:
            raise ValueError(f"Unsupported {name}: {item}; supported={sorted(supported_set)}")
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
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tensor_normed(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    norm = float(vec.norm().item())
    if norm < eps:
        return torch.zeros_like(vec)
    return vec / norm


def count_reflection_keywords(text: str) -> Tuple[int, List[str], Optional[int], Optional[str]]:
    lower = text.lower()
    hits: List[str] = []
    first_pos: Optional[int] = None
    first_kw: Optional[str] = None
    total = 0
    for kw in DEFAULT_REFLECTION_KEYWORDS:
        kw_s = str(kw)
        count = lower.count(kw_s.lower())
        if count:
            total += count
            hits.append(kw_s)
            pos = lower.find(kw_s.lower())
            if pos >= 0 and (first_pos is None or pos < first_pos):
                first_pos = int(pos)
                first_kw = kw_s
    return int(total), hits, first_pos, first_kw


def select_rows(path: Path, start: int, max_examples: int) -> List[Dict[str, Any]]:
    rows = load_jsonl(path)
    if max_examples <= 0:
        return rows[start:]
    return rows[start : start + max_examples]


def row_id(row: Dict[str, Any], fallback: int) -> str:
    for key in ["id", "example_id", "unique_id", "idx"]:
        if row.get(key) is not None:
            return str(row.get(key))
    return str(fallback)


def row_question(row: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt"]:
        if row.get(key):
            return str(row.get(key))
    return ""


def row_answer(row: Dict[str, Any]) -> Optional[str]:
    for key in ["correct_answer", "answer", "target"]:
        if row.get(key) is not None:
            return str(row.get(key))
    return None


def load_cached_gate_direction(path: Path, layer_idx: int, site: str) -> Optional[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    for item in payload.get("directions", []):
        if (
            str(item.get("direction_type")) == "gate"
            and int(item.get("layer_idx")) == int(layer_idx)
            and str(item.get("site")) == str(site)
        ):
            direction = tensor_normed(item["direction"])
            return {
                "direction": direction,
                "scale": float(item.get("scale") or item.get("mean_diff_norm") or 1.0),
                "n_pairs": int(item.get("n_pairs") or 0),
                "source": str(path),
                "mean_diff_norm": item.get("mean_diff_norm"),
                "mean_resid_std": item.get("mean_resid_std"),
            }
    return None


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


def continuation_analysis(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    state: Any,
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
    first_id = continuation.token_ids[0] if continuation.token_ids else None
    first_text = tokenizer.decode([first_id], skip_special_tokens=False) if first_id is not None else ""
    ref_count, hits, first_pos, first_kw = count_reflection_keywords(continuation_text)
    analysis = analyze_continuation_text(
        continuation=continuation_text,
        full_text=full_text,
        correct_answer=correct_answer,
        wrong_answer=wrong_answer,
        reflection_keywords=DEFAULT_REFLECTION_KEYWORDS,
    )
    return {
        "first_generated_token_id": first_id,
        "first_generated_token_text": first_text,
        "first_wait": bool(first_text in WAIT_FIRST_TOKENS),
        "continuation_token_len": len(continuation.token_ids),
        "continuation_stop_reason": continuation.stop_reason,
        "reflection_keyword_count": int(ref_count),
        "matched_reflection_keywords": hits,
        "first_reflection_pos": first_pos,
        "first_reflection_keyword": first_kw,
        "continuation_text": continuation_text,
        **analysis,
    }


@torch.no_grad()
def prepare_examples(
    *,
    args: argparse.Namespace,
    backend: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    layers: Sequence[torch.nn.Module],
    selected_layers: Sequence[int],
    selected_sites: Sequence[str],
    examples: Sequence[Dict[str, Any]],
    run_continuations: bool,
    split_name: str,
    reflect_token_ids: Sequence[int],
    stop_token_ids: Sequence[int],
    stop_sequences: Sequence[Sequence[int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    gen_config = STEERING.build_generation_config(args)
    stage_gen_config = replace(gen_config, max_new_tokens=int(args.max_stage1_tokens))
    clean_stop_strings = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    prepared: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    iterator = tqdm(list(enumerate(examples)), desc=f"Prepare {split_name}", dynamic_ncols=True)
    for local_idx, example in iterator:
        global_idx = (
            args.direction_start_idx + local_idx if split_name == "direction" else args.eval_start_idx + local_idx
        )
        example_id = row_id(example, global_idx)
        question = row_question(example)
        correct_answer = row_answer(example)
        wrong_answer = str(example.get("wrong_answer") or "").strip()
        if not question or correct_answer is None or not wrong_answer:
            skipped_rows.append({"split": split_name, "example_id": example_id, "reason": "missing_question_or_answers"})
            continue
        try:
            seed_everything(args.seed + global_idx)
            prompt = backend.build_prompt(question, gen_config)
            prompt_ids = backend.encode(prompt)
            clean = backend.generate(prompt, stage_gen_config, stop_strings=clean_stop_strings)
            clean_gen_ids = [int(x) for x in clean.token_ids]
            span = find_last_boxed_token_span(tokenizer, clean_gen_ids)
            if span is None:
                skipped_rows.append({"split": split_name, "example_id": example_id, "reason": "no_boxed_span"})
                continue
            if not args.allow_nonmatching_clean and not answers_match(span.answer_text, correct_answer):
                skipped_rows.append(
                    {
                        "split": split_name,
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
            coherent_ids, coherent_meta = STEERING.make_answer_token_coherent_ids(
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
                past, ids_before_final, final_token_id = STEERING.prefill_before_final_full_ids(model, full_ids)
                state, captures, _debug = STEERING.run_final_forward_boundary(
                    model,
                    past,
                    ids_before_final,
                    final_token_id,
                    layers=layers,
                    capture_layers=selected_layers,
                    capture_sites=selected_sites,
                )
                logits = state.logits.detach().float().cpu()
                rvs = logsumexp_token_set(logits, reflect_token_ids) - logsumexp_token_set(logits, stop_token_ids)
                entry: Dict[str, Any] = {
                    "full_ids": [int(x) for x in full_ids],
                    "captures": captures,
                    "reflect_vs_stop": float(rvs),
                    "state": state if run_continuations else None,
                }
                if run_continuations:
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
                    entry["analysis"] = analysis
                    baseline_rows.append(
                        {
                            "split": split_name,
                            "example_id": example_id,
                            "global_idx": global_idx,
                            "condition": f"{condition}_baseline",
                            "target_condition": condition,
                            "correct_answer": str(correct_answer),
                            "wrong_answer": wrong_answer,
                            "clean_box_answer": span.answer_text,
                            "coherent_replacement_count": int(coherent_meta["replacement_count"]),
                            "reflect_vs_stop": float(rvs),
                            "projection_coeff": 0.0,
                            "add_norm": 0.0,
                            **analysis,
                        }
                    )
                condition_data[condition] = entry
            prepared.append(
                {
                    "split": split_name,
                    "example_id": example_id,
                    "global_idx": global_idx,
                    "question": question,
                    "correct_answer": str(correct_answer),
                    "wrong_answer": wrong_answer,
                    "clean_box_answer": span.answer_text,
                    "coherent_replacement_count": int(coherent_meta["replacement_count"]),
                    "conditions": condition_data,
                }
            )
            if args.print_every > 0 and (local_idx + 1) % args.print_every == 0:
                iterator.set_postfix({"usable": len(prepared), "skipped": len(skipped_rows)})
        except Exception as exc:
            skipped_rows.append({"split": split_name, "example_id": example_id, "reason": "exception", "error": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return prepared, baseline_rows, skipped_rows


def build_gate_directions_from_examples(
    examples: Sequence[Dict[str, Any]],
    selected_layers: Sequence[int],
    selected_sites: Sequence[str],
    scale_mode: str,
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for layer_idx in selected_layers:
        for site in selected_sites:
            diffs: List[torch.Tensor] = []
            diff_norms: List[float] = []
            for ex in examples:
                tamper = ex["conditions"]["tamper"]["captures"].get((site, layer_idx))
                coherent = ex["conditions"]["coherent"]["captures"].get((site, layer_idx))
                if not isinstance(tamper, torch.Tensor) or not isinstance(coherent, torch.Tensor):
                    continue
                diff = tamper.float().cpu() - coherent.float().cpu()
                diff_norms.append(float(diff.norm().item()))
                diffs.append(tensor_normed(diff))
            if not diffs:
                continue
            direction = tensor_normed(torch.stack(diffs, dim=0).mean(dim=0))
            scale = mean(diff_norms) if scale_mode == "diff_norm" else 1.0
            if not math.isfinite(scale) or scale == 0:
                scale = 1.0
            out[(int(layer_idx), site)] = {
                "direction": direction,
                "scale": float(scale),
                "n_pairs": len(diffs),
                "source": "built_in_run",
                "mean_diff_norm": mean(diff_norms),
            }
    return out


def make_direction_controls(
    *,
    gate_info: Dict[str, Any],
    logit_dir: Optional[torch.Tensor],
    seed: int,
    projection_types: Sequence[str],
) -> Dict[str, torch.Tensor]:
    gate_dir = tensor_normed(gate_info["direction"])
    directions: Dict[str, torch.Tensor] = {}
    if "gate" in projection_types:
        directions["gate"] = gate_dir
    if logit_dir is not None and int(logit_dir.numel()) == int(gate_dir.numel()):
        logit_unit = tensor_normed(logit_dir)
        if "logit" in projection_types or "logit_matched_norm" in projection_types:
            directions["logit"] = logit_unit
        if "gate_orth_logit" in projection_types:
            directions["gate_orth_logit"] = tensor_normed(gate_dir - torch.dot(gate_dir, logit_unit) * logit_unit)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 99173)
    random_unit = tensor_normed(torch.randn(int(gate_dir.numel()), generator=generator))
    if "random" in projection_types or "random_matched_norm" in projection_types:
        directions["random"] = random_unit
    return directions


def projection_coeff(
    *,
    projection_type: str,
    diff: torch.Tensor,
    direction: torch.Tensor,
    gate_coeff: float,
) -> float:
    actual = float(torch.dot(diff.float().cpu(), direction.float().cpu()).item())
    if projection_type in {"random_matched_norm", "logit_matched_norm"}:
        return float(gate_coeff)
    return actual


def summarize(rows: Sequence[Dict[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    out: List[Dict[str, Any]] = []
    for key, group in groups.items():
        row = {name: value for name, value in zip(group_keys, key)}
        row.update(
            {
                "count": len(group),
                "first_wait_rate": mean([1.0 if x.get("first_wait") else 0.0 for x in group]),
                "has_reflection_rate": mean([1.0 if x.get("has_reflection") else 0.0 for x in group]),
                "mean_reflection_keyword_count": mean([x.get("reflection_keyword_count") for x in group]),
                "mean_continuation_token_len": mean([x.get("continuation_token_len") for x in group]),
                "mean_reflect_vs_stop": mean([x.get("reflect_vs_stop") for x in group]),
                "mean_delta_reflect_vs_stop": mean([x.get("delta_reflect_vs_stop") for x in group]),
                "mean_projection_coeff": mean([x.get("projection_coeff") for x in group]),
                "mean_add_norm": mean([x.get("add_norm") for x in group]),
                "correct_full_text_rate": mean([1.0 if x.get("full_text_final_matches_correct") else 0.0 for x in group]),
                "wrong_full_text_rate": mean([1.0 if x.get("full_text_final_matches_wrong") else 0.0 for x in group]),
            }
        )
        out.append(row)
    out.sort(key=lambda r: tuple(str(r.get(k)) for k in group_keys))
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    backend = STEERING.build_backend(args)
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("This script requires an HF backend.")
    decoder_layers, layer_path = get_decoder_layers(model)
    layers = list(decoder_layers)
    selected_layers = parse_int_list(args.layers, len(layers))
    selected_sites = STEERING.parse_sites(args.sites)
    projection_types = parse_choice_list(
        args.projection_types,
        ["gate", "gate_orth_logit", "logit", "random", "random_matched_norm", "logit_matched_norm"],
        "projection_types",
    )
    reflect_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    stop_sequences = HELPERS.stop_id_sequences(backend, [])
    logit_dir = get_logit_direction(model, reflect_token_ids, stop_token_ids)

    input_path = Path(args.input_jsonl)
    direction_examples: List[Dict[str, Any]] = []
    direction_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    if not args.direction_cache_in:
        direction_examples, _direction_baselines, skipped = prepare_examples(
            args=args,
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            selected_layers=selected_layers,
            selected_sites=selected_sites,
            examples=select_rows(input_path, args.direction_start_idx, args.direction_max_examples),
            run_continuations=False,
            split_name="direction",
            reflect_token_ids=reflect_token_ids,
            stop_token_ids=stop_token_ids,
            stop_sequences=stop_sequences,
        )
        skipped_rows.extend(skipped)
        gate_infos = build_gate_directions_from_examples(
            direction_examples,
            selected_layers,
            selected_sites,
            args.scale_mode,
        )
    else:
        gate_infos = {}
        cache_path = Path(args.direction_cache_in)
        for layer_idx in selected_layers:
            for site in selected_sites:
                info = load_cached_gate_direction(cache_path, int(layer_idx), site)
                if info is None:
                    skipped_rows.append(
                        {
                            "split": "direction",
                            "reason": "missing_cached_direction",
                            "direction_cache_in": str(cache_path),
                            "layer_idx": int(layer_idx),
                            "site": site,
                        }
                    )
                    continue
                gate_infos[(int(layer_idx), site)] = info

    for (layer_idx, site), info in gate_infos.items():
        gate_dir = tensor_normed(info["direction"])
        logit_cosine = None
        if logit_dir is not None and int(logit_dir.numel()) == int(gate_dir.numel()):
            logit_cosine = float(torch.dot(gate_dir, tensor_normed(logit_dir)).item())
        direction_rows.append(
            {
                "layer_idx": int(layer_idx),
                "site": site,
                "direction_type": "gate",
                "n_pairs": info.get("n_pairs"),
                "scale": info.get("scale"),
                "mean_diff_norm": info.get("mean_diff_norm"),
                "mean_resid_std": info.get("mean_resid_std"),
                "source": info.get("source"),
                "gate_logit_cosine": logit_cosine,
            }
        )

    eval_examples, baseline_rows, skipped = prepare_examples(
        args=args,
        backend=backend,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        selected_layers=selected_layers,
        selected_sites=selected_sites,
        examples=select_rows(input_path, args.eval_start_idx, args.eval_max_examples),
        run_continuations=True,
        split_name="eval",
        reflect_token_ids=reflect_token_ids,
        stop_token_ids=stop_token_ids,
        stop_sequences=stop_sequences,
    )
    skipped_rows.extend(skipped)

    projection_rows: List[Dict[str, Any]] = []
    iterator = tqdm(eval_examples, desc="Projection ablate", dynamic_ncols=True)
    for ex in iterator:
        for layer_idx in selected_layers:
            for site in selected_sites:
                gate_info = gate_infos.get((int(layer_idx), site))
                if gate_info is None:
                    continue
                controls = make_direction_controls(
                    gate_info=gate_info,
                    logit_dir=logit_dir,
                    seed=args.seed + int(layer_idx) * 17,
                    projection_types=projection_types,
                )
                tamper_capture = ex["conditions"]["tamper"]["captures"].get((site, int(layer_idx)))
                coherent_capture = ex["conditions"]["coherent"]["captures"].get((site, int(layer_idx)))
                if not isinstance(tamper_capture, torch.Tensor) or not isinstance(coherent_capture, torch.Tensor):
                    continue
                diff = tamper_capture.float().cpu() - coherent_capture.float().cpu()
                gate_coeff = float(torch.dot(diff, tensor_normed(gate_info["direction"])).item())
                baseline_tamper = ex["conditions"]["tamper"]
                baseline_analysis = baseline_tamper.get("analysis") or {}
                for projection_type in projection_types:
                    base_type = projection_type.replace("_matched_norm", "")
                    direction = controls.get(base_type)
                    if direction is None:
                        continue
                    coeff = projection_coeff(
                        projection_type=projection_type,
                        diff=diff,
                        direction=direction,
                        gate_coeff=gate_coeff,
                    )
                    add_vector = -float(coeff) * direction
                    try:
                        past, ids_before_final, final_token_id = STEERING.prefill_before_final_full_ids(
                            model,
                            baseline_tamper["full_ids"],
                        )
                        patched_state, debug = STEERING.run_final_forward_add(
                            model,
                            past,
                            ids_before_final,
                            int(final_token_id),
                            layers=layers,
                            add_layer_idx=int(layer_idx),
                            add_site=site,
                            add_vector=add_vector,
                        )
                        logits = patched_state.logits.detach().float().cpu()
                        rvs = logsumexp_token_set(logits, reflect_token_ids) - logsumexp_token_set(
                            logits, stop_token_ids
                        )
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
                        projection_rows.append(
                            {
                                "split": "eval",
                                "example_id": ex["example_id"],
                                "global_idx": ex["global_idx"],
                                "condition": f"{projection_type}_projection_ablation",
                                "projection_type": projection_type,
                                "base_direction_type": base_type,
                                "target_condition": "tamper",
                                "layer_idx": int(layer_idx),
                                "site": site,
                                "direction_source": gate_info.get("source"),
                                "direction_n_pairs": gate_info.get("n_pairs"),
                                "correct_answer": ex["correct_answer"],
                                "wrong_answer": ex["wrong_answer"],
                                "clean_box_answer": ex["clean_box_answer"],
                                "coherent_replacement_count": int(ex["coherent_replacement_count"]),
                                "baseline_reflect_vs_stop": float(baseline_tamper["reflect_vs_stop"]),
                                "reflect_vs_stop": float(rvs),
                                "delta_reflect_vs_stop": float(rvs - float(baseline_tamper["reflect_vs_stop"])),
                                "baseline_first_wait": bool(baseline_analysis.get("first_wait")),
                                "baseline_has_reflection": bool(baseline_analysis.get("has_reflection")),
                                "baseline_reflection_keyword_count": baseline_analysis.get("reflection_keyword_count"),
                                "projection_coeff": float(coeff),
                                "gate_projection_coeff": float(gate_coeff),
                                "add_norm": float(add_vector.norm().item()),
                                **debug,
                                **analysis,
                            }
                        )
                    except Exception as exc:
                        skipped_rows.append(
                            {
                                "split": "eval",
                                "example_id": ex["example_id"],
                                "reason": "projection_exception",
                                "projection_type": projection_type,
                                "layer_idx": int(layer_idx),
                                "site": site,
                                "error": repr(exc),
                            }
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    all_rows = baseline_rows + projection_rows
    compact_rows = []
    for row in all_rows:
        compact = {key: value for key, value in row.items() if key != "continuation_text"}
        compact["continuation_text_chars"] = len(row.get("continuation_text") or "")
        compact_rows.append(compact)

    dump_jsonl(output_dir / "baseline_rows.jsonl", baseline_rows)
    dump_jsonl(output_dir / "projection_rows.jsonl", projection_rows)
    dump_jsonl(output_dir / "all_rows.jsonl", all_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    write_csv(output_dir / "all_rows.csv", compact_rows)
    write_csv(output_dir / "direction_summary.csv", direction_rows)
    write_csv(output_dir / "baseline_summary.csv", summarize(baseline_rows, ["condition", "target_condition"]))
    write_csv(
        output_dir / "projection_summary.csv",
        summarize(projection_rows, ["projection_type", "layer_idx", "site"]),
    )
    write_json(
        output_dir / "summary.json",
        {
            "model_name_or_path": args.model_name_or_path,
            "input_jsonl": args.input_jsonl,
            "direction_cache_in": args.direction_cache_in,
            "direction_start_idx": args.direction_start_idx,
            "direction_max_examples": args.direction_max_examples,
            "eval_start_idx": args.eval_start_idx,
            "eval_max_examples": args.eval_max_examples,
            "usable_direction_examples": len(direction_examples),
            "usable_eval_examples": len(eval_examples),
            "baseline_rows": len(baseline_rows),
            "projection_rows": len(projection_rows),
            "skipped_rows": len(skipped_rows),
            "layers": selected_layers,
            "sites": selected_sites,
            "projection_types": projection_types,
            "scale_mode": args.scale_mode,
            "coherent_window_chars": args.coherent_window_chars,
            "layer_path": layer_path,
            "reflect_token_ids": reflect_token_ids,
            "stop_token_ids": stop_token_ids,
            "generation": asdict(STEERING.build_generation_config(args)),
            "direction_summary": direction_rows,
        },
    )
    print("[Done] Reflection gate projection ablation finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_eval_examples: {len(eval_examples)}")
    print(f"- projection_rows: {len(projection_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
