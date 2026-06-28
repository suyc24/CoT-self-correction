#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = ROOT_DIR / "evaluation"
for p in (EVALUATION_DIR, ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.answer_extraction import classify_outcome, extract_last_boxed
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import (
    find_last_boxed_token_span,
    logsumexp_token_set,
    token_ids_for_first_tokens,
)
from grader import math_equal
from parser import extract_answer as benchmark_extract_answer
from parser import strip_string as benchmark_strip_string

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

BoundaryAdd = STEERING.BoundaryAdd
build_backend = STEERING.build_backend
build_generation_config = STEERING.build_generation_config
get_logit_direction = STEERING.get_logit_direction
make_answer_token_coherent_ids = STEERING.make_answer_token_coherent_ids
parse_sites = STEERING.parse_sites
prefill_before_final_full_ids = STEERING.prefill_before_final_full_ids
run_final_forward_boundary = STEERING.run_final_forward_boundary
tensor_normed = STEERING.tensor_normed

DEFAULT_REFLECTION_KEYWORDS = HELPERS.DEFAULT_REFLECTION_KEYWORDS
DEFAULT_REFLECT_FIRST_TEXTS = HELPERS.DEFAULT_REFLECT_FIRST_TEXTS
DEFAULT_STOP_FIRST_TEXTS = HELPERS.DEFAULT_STOP_FIRST_TEXTS
WAIT_FIRST_TOKENS = {" Wait", " wait", "Wait", "wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a learned reflection-gate direction throughout normal generation."
    )
    parser.add_argument(
        "--direction_input_jsonl",
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--eval_input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--direction_max_examples", type=int, default=40)
    parser.add_argument("--direction_start_idx", type=int, default=0)
    parser.add_argument(
        "--direction_cache_in",
        default="",
        help="Load precomputed direction tensors from this torch cache instead of rebuilding them.",
    )
    parser.add_argument(
        "--direction_cache_out",
        default="",
        help="Write computed direction tensors to this torch cache for reuse by eval shards.",
    )
    parser.add_argument("--eval_max_examples", type=int, default=40)
    parser.add_argument("--eval_start_idx", type=int, default=0)
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
    parser.add_argument("--benchmark_data_name", default="math")
    parser.add_argument("--layers", default="22")
    parser.add_argument("--sites", default="post_attn")
    parser.add_argument("--alphas", default="0,0.5,1")
    parser.add_argument(
        "--direction_types",
        default="gate,random,logit",
        help="Comma-separated controls: gate,random,logit,gate_orth_logit.",
    )
    parser.add_argument(
        "--direction_baseline_condition",
        choices=["coherent", "clean"],
        default="coherent",
        help=(
            "Baseline condition subtracted from tamper when building the reflection-gate direction. "
            "The original setting is tamper - coherent; clean gives a tamper - clean robustness control."
        ),
    )
    parser.add_argument("--scale_mode", choices=["diff_norm", "resid_std", "unit"], default="diff_norm")
    parser.add_argument(
        "--intervention_mode",
        choices=["decode_only", "all_positions"],
        default="decode_only",
        help="decode_only leaves the prompt prefill untouched and edits generated-token decoding steps.",
    )
    parser.add_argument(
        "--baseline_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only run normal baseline trajectories; skip direction construction and interventions.",
    )
    parser.add_argument("--skip_baseline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stop_at_think_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=5)
    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    out = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not out:
        raise ValueError("No layer indices supplied.")
    return out


def parse_float_list(text: str) -> List[float]:
    out = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not out:
        raise ValueError("No alpha values supplied.")
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
    vals: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            vals.append(x)
    return sum(vals) / len(vals) if vals else float("nan")


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
    return total, hits, first_pos, first_kw


def benchmark_judge_answer(text: str, correct_answer: Optional[str], data_name: str) -> Dict[str, Any]:
    pred = benchmark_extract_answer(text, data_name)
    pred = benchmark_strip_string(pred, skip_unit=False)
    reference = None if correct_answer is None else benchmark_strip_string(str(correct_answer), skip_unit=False)
    correct = bool(reference is not None and math_equal(pred, reference))
    return {
        "benchmark_pred_answer": pred,
        "benchmark_reference_answer": reference,
        "benchmark_correct": correct,
    }


def analyze_generation(
    tokenizer,
    token_ids: Sequence[int],
    text: str,
    correct_answer: Optional[str],
    benchmark_data_name: str,
) -> Dict[str, Any]:
    first_id = int(token_ids[0]) if token_ids else None
    first_text = tokenizer.decode([first_id], skip_special_tokens=False) if first_id is not None else ""
    reflection_count, hits, first_pos, first_kw = count_reflection_keywords(text)
    final_box = extract_last_boxed(text)
    benchmark = benchmark_judge_answer(text, correct_answer, benchmark_data_name)
    return {
        "first_generated_token_id": first_id,
        "first_generated_token_text": first_text,
        "first_wait": bool(first_text in WAIT_FIRST_TOKENS),
        "has_reflection": bool(hits),
        "reflection_keyword_count": int(reflection_count),
        "matched_reflection_keywords": hits,
        "first_reflection_pos": first_pos,
        "first_reflection_keyword": first_kw,
        "final_boxed_answer": final_box,
        "continuation_outcome": classify_outcome(final_box, correct_answer, None),
        **benchmark,
        "generated_tokens": len(token_ids),
        "continuation_text": text,
    }


class FullTrajectoryAdd(BoundaryAdd):
    def __init__(self, layer: torch.nn.Module, site: str, add_vector: torch.Tensor, *, all_positions: bool) -> None:
        super().__init__(layer, site, add_vector)
        self.all_positions = bool(all_positions)

    def _slice_ok(self, hidden: torch.Tensor) -> bool:
        return hidden.ndim == 3 and (self.all_positions or hidden.shape[1] == 1)

    def _block_input_hook(self, module, args):
        if not args or not isinstance(args[0], torch.Tensor):
            return None
        hidden = args[0]
        if not self._slice_ok(hidden):
            return None
        patched = hidden.clone()
        add_vec = self._add_vec(patched)
        self._record(add_vec)
        if self.all_positions:
            patched[0, :, :] = patched[0, :, :] + add_vec
        else:
            patched[0, -1] = patched[0, -1] + add_vec
        return STEERING.replace_tuple_arg(args, 0, patched)

    def _ln_pre_hook(self, module, args):
        if args and isinstance(args[0], torch.Tensor) and self._slice_ok(args[0]):
            self._pre_attn_resid = args[0]

    def _attn_post_hook(self, module, inputs, output):
        attn_out = STEERING.tensor_output(output)
        if attn_out is None or self._pre_attn_resid is None or not self._slice_ok(attn_out):
            self._pre_attn_resid = None
            return output
        modified = attn_out.clone()
        add_vec = self._add_vec(modified)
        self._record(add_vec)
        if self.all_positions:
            modified[0, :, :] = modified[0, :, :] + add_vec
        else:
            modified[0, -1] = modified[0, -1] + add_vec
        self._pre_attn_resid = None
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def _block_output_hook(self, module, inputs, output):
        hidden = STEERING.tensor_output(output)
        if hidden is None or not self._slice_ok(hidden):
            return output
        patched = hidden.clone()
        add_vec = self._add_vec(patched)
        self._record(add_vec)
        if self.all_positions:
            patched[0, :, :] = patched[0, :, :] + add_vec
        else:
            patched[0, -1] = patched[0, -1] + add_vec
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        return patched


def generate_with_add(
    *,
    backend,
    gen_config,
    prompt: str,
    stop_strings: Optional[List[str]],
    layer,
    site: str,
    add_vector: Optional[torch.Tensor],
    all_positions: bool,
) -> Tuple[Any, Dict[str, Any]]:
    if add_vector is None:
        result = backend.generate(prompt, gen_config, stop_strings=stop_strings)
        return result, {"add_hook_call_count": 0, "add_norm": 0.0}
    ctx = FullTrajectoryAdd(layer, site, add_vector, all_positions=all_positions)
    ctx.__enter__()
    try:
        result = backend.generate(prompt, gen_config, stop_strings=stop_strings)
    finally:
        debug = {"add_hook_call_count": int(ctx.call_count), "add_norm": ctx.add_norm}
        ctx.__exit__(None, None, None)
    return result, debug


@torch.no_grad()
def build_directions(args: argparse.Namespace, backend, layers: Sequence[torch.nn.Module]) -> Tuple[Dict[Tuple[str, int, str], Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    tokenizer = backend.tokenizer
    selected_layers = parse_int_list(args.layers)
    selected_sites = parse_sites(args.sites)
    rows = load_jsonl(args.direction_input_jsonl)
    examples = rows[args.direction_start_idx : args.direction_start_idx + args.direction_max_examples]
    gen_config = build_generation_config(args)
    stop_strings = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    reflect_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_token_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    logit_dir = get_logit_direction(backend.model, reflect_token_ids, stop_token_ids)

    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for local_idx, ex in enumerate(tqdm(examples, desc="Build gate direction", dynamic_ncols=True)):
        ex_id = row_id(ex, args.direction_start_idx + local_idx)
        question = row_question(ex)
        correct_answer = row_answer(ex)
        wrong_answer = str(ex.get("wrong_answer") or "").strip()
        if not question or correct_answer is None or not wrong_answer:
            skipped.append({"example_id": ex_id, "reason": "missing_fields"})
            continue
        try:
            seed_everything(args.seed + local_idx)
            prompt = backend.build_prompt(question, gen_config)
            prompt_ids = backend.encode(prompt)
            clean = backend.generate(prompt, gen_config, stop_strings=stop_strings)
            clean_gen_ids = [int(x) for x in clean.token_ids]
            span = find_last_boxed_token_span(tokenizer, clean_gen_ids)
            if span is None:
                skipped.append({"example_id": ex_id, "reason": "no_boxed_span"})
                continue
            if not STEERING.answers_match(span.answer_text, correct_answer):
                skipped.append({"example_id": ex_id, "reason": "clean_box_not_correct", "clean_box": span.answer_text})
                continue
            prefix_ids = prompt_ids + clean_gen_ids[: span.answer_start]
            wrong_answer_ids = backend.encode(wrong_answer)
            tamper_force_ids = wrong_answer_ids + clean_gen_ids[span.answer_end : span.box_end]
            tamper_force_text = tokenizer.decode(tamper_force_ids, skip_special_tokens=False)
            coherent_ids, _meta = make_answer_token_coherent_ids(
                tokenizer,
                prefix_ids=prefix_ids,
                tamper_force_text=tamper_force_text,
                clean_answer=span.answer_text,
                correct_answer=str(correct_answer),
                wrong_answer=wrong_answer,
                window_chars=1200,
            )
            tamper_full_ids = prefix_ids + tamper_force_ids
            clean_full_ids = prompt_ids + clean_gen_ids[: span.box_end]
            captures_by_condition: Dict[str, Dict[Tuple[str, int], torch.Tensor]] = {}
            condition_full_ids = [
                ("tamper", tamper_full_ids),
                ("coherent", coherent_ids),
                ("clean", clean_full_ids),
            ]
            for condition, full_ids in condition_full_ids:
                past, ids_before_final, final_token_id = prefill_before_final_full_ids(backend.model, full_ids)
                _state, captures, _debug = run_final_forward_boundary(
                    backend.model,
                    past,
                    ids_before_final,
                    final_token_id,
                    layers=layers,
                    capture_layers=selected_layers,
                    capture_sites=selected_sites,
                )
                captures_by_condition[condition] = captures
            records.append(
                {
                    "example_id": ex_id,
                    "captures": captures_by_condition,
                }
            )
        except Exception as exc:
            skipped.append({"example_id": ex_id, "reason": "exception", "error": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    directions: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    direction_rows: List[Dict[str, Any]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 1717)
    direction_types = parse_choice_list(
        args.direction_types,
        ["gate", "random", "logit", "gate_orth_logit"],
        "direction_types",
    )
    for layer_idx in selected_layers:
        for site in selected_sites:
            diffs: List[torch.Tensor] = []
            diff_norms: List[float] = []
            resid_stds: List[float] = []
            for rec in records:
                tamper = rec["captures"]["tamper"].get((site, layer_idx))
                baseline = rec["captures"][args.direction_baseline_condition].get((site, layer_idx))
                if tamper is None or baseline is None:
                    continue
                diff = tamper.float() - baseline.float()
                diffs.append(tensor_normed(diff))
                diff_norms.append(float(diff.norm().item()))
                resid_stds.append(float(tamper.float().std().item()))
                resid_stds.append(float(baseline.float().std().item()))
            if not diffs:
                continue
            gate_dir = tensor_normed(torch.stack(diffs, dim=0).mean(dim=0))
            dim = int(gate_dir.numel())
            random_dir = tensor_normed(torch.randn(dim, generator=generator))
            logit_dir_for_site = logit_dir if logit_dir is not None and logit_dir.numel() == dim else None
            gate_orth_logit_dir = None
            gate_logit_cosine = None
            if logit_dir_for_site is not None:
                logit_unit = tensor_normed(logit_dir_for_site.float())
                gate_unit = tensor_normed(gate_dir.float())
                gate_logit_cosine = float(torch.dot(gate_unit.cpu(), logit_unit.cpu()).item())
                orth = gate_unit - torch.dot(gate_unit, logit_unit.to(gate_unit.device)) * logit_unit.to(gate_unit.device)
                if float(orth.norm().item()) > 1e-8:
                    gate_orth_logit_dir = tensor_normed(orth)
            scale = 1.0
            if args.scale_mode == "diff_norm":
                scale = mean(diff_norms)
            elif args.scale_mode == "resid_std":
                scale = mean(resid_stds)
            if not math.isfinite(scale) or scale == 0:
                scale = 1.0
            for direction_type, direction in [
                ("gate", gate_dir),
                ("random", random_dir),
                ("logit", logit_dir_for_site),
                ("gate_orth_logit", gate_orth_logit_dir),
            ]:
                if direction_type not in direction_types or direction is None:
                    continue
                directions[(direction_type, layer_idx, site)] = {
                    "direction": tensor_normed(direction),
                    "scale": float(scale),
                    "n_pairs": len(diffs),
                    "mean_diff_norm": mean(diff_norms),
                    "mean_resid_std": mean(resid_stds),
                    "direction_baseline_condition": args.direction_baseline_condition,
                    "direction_contrast": f"tamper_minus_{args.direction_baseline_condition}",
                }
                direction_rows.append(
                    {
                        "direction_type": direction_type,
                        "layer_idx": layer_idx,
                        "site": site,
                        "direction_baseline_condition": args.direction_baseline_condition,
                        "direction_contrast": f"tamper_minus_{args.direction_baseline_condition}",
                        "n_pairs": len(diffs),
                        "scale": float(scale),
                        "mean_diff_norm": mean(diff_norms),
                        "mean_resid_std": mean(resid_stds),
                        "gate_logit_cosine": gate_logit_cosine,
                    }
                )
    skipped.extend({"example_id": rec["example_id"], "reason": "used_for_direction"} for rec in records[:0])
    return directions, direction_rows, skipped


def save_direction_cache(
    path: Path,
    directions: Dict[Tuple[str, int, str], Dict[str, Any]],
    direction_rows: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "directions": [
            {
                "direction_type": key[0],
                "layer_idx": int(key[1]),
                "site": key[2],
                **{k: v for k, v in info.items() if k != "direction"},
                "direction": info["direction"].detach().float().cpu(),
            }
            for key, info in directions.items()
        ],
        "direction_rows": list(direction_rows),
    }
    torch.save(payload, path)


def load_direction_cache(path: Path) -> Tuple[Dict[Tuple[str, int, str], Dict[str, Any]], List[Dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu")
    directions: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for item in payload.get("directions", []):
        key = (str(item["direction_type"]), int(item["layer_idx"]), str(item["site"]))
        info = {k: v for k, v in item.items() if k not in {"direction_type", "layer_idx", "site"}}
        info["direction"] = tensor_normed(info["direction"].float())
        directions[key] = info
    return directions, list(payload.get("direction_rows", []))


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    keys = ["condition", "direction_type", "layer_idx", "site", "alpha"]
    for row in rows:
        grouped[tuple(row.get(k) for k in keys)].append(row)
    out: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        row = {k: v for k, v in zip(keys, key)}
        token_counter = Counter(str(x.get("first_generated_token_text") or "") for x in group)
        outcome_counter = Counter(str(x.get("continuation_outcome") or "") for x in group)
        row.update(
            {
                "count": len(group),
                "first_wait_rate": mean([1.0 if x.get("first_wait") else 0.0 for x in group]),
                "has_reflection_rate": mean([1.0 if x.get("has_reflection") else 0.0 for x in group]),
                "mean_reflection_keyword_count": mean([x.get("reflection_keyword_count") for x in group]),
                "correct": sum(1 for x in group if x.get("benchmark_correct")),
                "accuracy": mean([1.0 if x.get("benchmark_correct") else 0.0 for x in group]),
                "benchmark_correct_rate": mean([1.0 if x.get("benchmark_correct") else 0.0 for x in group]),
                "no_boxed_rate": mean([1.0 if x.get("continuation_outcome") == "no_boxed_answer" else 0.0 for x in group]),
                "mean_generated_tokens": mean([x.get("generated_tokens") for x in group]),
                "mean_add_hook_call_count": mean([x.get("add_hook_call_count") for x in group]),
                "top_first_tokens": json.dumps(token_counter.most_common(8), ensure_ascii=False),
                "outcome_counts": json.dumps(dict(outcome_counter), ensure_ascii=False),
            }
        )
        out.append(row)
    out.sort(key=lambda r: tuple(str(r.get(k)) for k in keys))
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    backend = build_backend(args)
    if backend.model is None:
        raise ValueError("HF model backend is required.")
    decoder_layers, layer_path = get_decoder_layers(backend.model)
    layers = list(decoder_layers)
    selected_layers = parse_int_list(args.layers)
    selected_sites = parse_sites(args.sites)
    alphas = parse_float_list(args.alphas)
    all_positions = args.intervention_mode == "all_positions"
    gen_config = build_generation_config(args)
    stop_strings = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None

    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "layer_path": layer_path,
            "selected_layers": selected_layers,
            "selected_sites": selected_sites,
            "alphas": alphas,
            "generation": asdict(gen_config),
        },
    )

    if args.baseline_only:
        directions = {}
        direction_rows = []
        skipped_rows = []
    elif args.direction_cache_in:
        directions, direction_rows = load_direction_cache(Path(args.direction_cache_in))
        skipped_rows = []
    else:
        directions, direction_rows, skipped_rows = build_directions(args, backend, layers)
        if args.direction_cache_out:
            save_direction_cache(Path(args.direction_cache_out), directions, direction_rows)
    eval_rows_raw = load_jsonl(args.eval_input_jsonl)
    eval_examples = eval_rows_raw[args.eval_start_idx : args.eval_start_idx + args.eval_max_examples]
    eval_rows: List[Dict[str, Any]] = []

    direction_items = list(directions.items())
    iterator = tqdm(eval_examples, desc="Full-trajectory gate-off eval", dynamic_ncols=True)
    for local_idx, ex in enumerate(iterator):
        ex_id = row_id(ex, args.eval_start_idx + local_idx)
        question = row_question(ex)
        correct_answer = row_answer(ex)
        if not question:
            skipped_rows.append({"example_id": ex_id, "stage": "eval", "reason": "missing_question"})
            continue
        prompt = backend.build_prompt(question, gen_config)
        for baseline_alpha in ([] if args.skip_baseline else [0.0]):
            try:
                seed_everything(args.seed + 100000 + local_idx)
                result, debug = generate_with_add(
                    backend=backend,
                    gen_config=gen_config,
                    prompt=prompt,
                    stop_strings=stop_strings,
                    layer=None,
                    site="",
                    add_vector=None,
                    all_positions=all_positions,
                )
                analysis = analyze_generation(
                    backend.tokenizer,
                    result.token_ids,
                    result.continuation,
                    correct_answer,
                    args.benchmark_data_name,
                )
                eval_rows.append(
                    {
                        "example_id": ex_id,
                        "condition": "baseline",
                        "direction_type": "none",
                        "layer_idx": "",
                        "site": "",
                        "alpha": baseline_alpha,
                        "question": question,
                        "correct_answer": correct_answer,
                        **analysis,
                        **debug,
                    }
                )
            except Exception as exc:
                skipped_rows.append({"example_id": ex_id, "stage": "baseline", "reason": "exception", "error": repr(exc)})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        for (direction_type, layer_idx, site), info in direction_items:
            unit_dir = info["direction"]
            scale = float(info["scale"])
            for alpha in alphas:
                if float(alpha) == 0.0:
                    continue
                add_vector = -unit_dir * (float(alpha) * scale)
                try:
                    seed_everything(args.seed + 100000 + local_idx)
                    result, debug = generate_with_add(
                        backend=backend,
                        gen_config=gen_config,
                        prompt=prompt,
                        stop_strings=stop_strings,
                        layer=layers[int(layer_idx)],
                        site=site,
                        add_vector=add_vector,
                        all_positions=all_positions,
                    )
                    analysis = analyze_generation(
                        backend.tokenizer,
                        result.token_ids,
                        result.continuation,
                        correct_answer,
                        args.benchmark_data_name,
                    )
                    eval_rows.append(
                        {
                            "example_id": ex_id,
                            "condition": "gate_off" if direction_type == "gate" else f"{direction_type}_minus",
                            "direction_type": direction_type,
                            "layer_idx": int(layer_idx),
                            "site": site,
                            "alpha": float(alpha),
                            "scale": scale,
                            "question": question,
                            "correct_answer": correct_answer,
                            **analysis,
                            **debug,
                        }
                    )
                except Exception as exc:
                    skipped_rows.append(
                        {
                            "example_id": ex_id,
                            "stage": "eval_intervention",
                            "reason": "exception",
                            "direction_type": direction_type,
                            "layer_idx": int(layer_idx),
                            "site": site,
                            "alpha": float(alpha),
                            "error": repr(exc),
                        }
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        if args.print_every > 0 and (local_idx + 1) % args.print_every == 0:
            iterator.set_postfix({"rows": len(eval_rows), "skipped": len(skipped_rows)})
            dump_jsonl(output_dir / "eval_rows.partial.jsonl", eval_rows)
            dump_jsonl(output_dir / "skipped_rows.partial.jsonl", skipped_rows)
            write_csv(output_dir / "summary.partial.csv", summarize(eval_rows))

    dump_jsonl(output_dir / "eval_rows.jsonl", eval_rows)
    dump_jsonl(output_dir / "direction_rows.jsonl", direction_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    write_csv(output_dir / "summary.csv", summarize(eval_rows))
    write_json(
        output_dir / "summary.json",
        {
            "eval_rows": len(eval_rows),
            "direction_rows": len(direction_rows),
            "skipped_rows": len(skipped_rows),
            "directions": [
                {
                    "direction_type": k[0],
                    "layer_idx": k[1],
                    "site": k[2],
                    "scale": v["scale"],
                    "n_pairs": v["n_pairs"],
                    "gate_logit_cosine": v.get("gate_logit_cosine"),
                    "direction_baseline_condition": v.get("direction_baseline_condition"),
                    "direction_contrast": v.get("direction_contrast"),
                }
                for k, v in directions.items()
            ],
        },
    )
    print("[Done] Full-trajectory gate-off eval finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- eval_rows: {len(eval_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
