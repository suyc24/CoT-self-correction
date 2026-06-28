#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.patch_trajectory import PatchPlan, ScheduledAddIntervention, run_patchable_trajectory
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import analyze_continuation_text, token_ids_for_first_tokens
from run_reflection_hidden_trajectory_movie import (
    DEFAULT_REFLECT_FIRST_TEXTS,
    DEFAULT_REFLECTION_KEYWORDS,
    DEFAULT_STOP_FIRST_TEXTS,
    WAIT_FIRST_TOKENS,
    build_backend,
    build_generation_config,
    load_gate_direction_cache,
    parse_csv_list,
    parse_layer_spec,
    prepare_forced_box_example,
    row_answer,
    row_id,
    row_question,
    safe_name,
    stop_id_sequences,
    summarize_behavior as summarize_basic_behavior,
    tensor_normed,
)


SUPPORTED_TIMINGS = {"p0", "p1", "p1_p8", "until_marker"}
SUPPORTED_PATCH_TYPES = {"delete", "induce", "rescue", "cancel"}
SUPPORTED_GATE_MODES = {"prefill_only", "decode_only", "prefill_plus_decode"}

REPAIR_NEGATION_KEYWORDS = [
    "wrong",
    "incorrect",
    "mistake",
    "not correct",
    "actually",
    "check",
    "recalculate",
    "不对",
    "错误",
    "重新",
    "检查",
]


@dataclass(frozen=True)
class PatchSpec:
    condition: str
    patch_type: str
    base_condition: str
    source_condition: str
    layer_idx: int
    site: str
    timing: str
    gate_mode: str
    sweep: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch-delete, patch-induce, gate-off rescue, and gate-on cancellation for forced-box reflection."
    )
    parser.add_argument(
        "--input_jsonl",
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=20)
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
    parser.add_argument("--stop_at_think_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_stage1_tokens", type=int, default=4096)
    parser.add_argument("--max_continuation_tokens", type=int, default=96)
    parser.add_argument("--capture_max_position_index", type=int, default=64)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coherent_window_chars", type=int, default=1200)
    parser.add_argument("--layers", default="19-22")
    parser.add_argument("--sites", default="post_attn,block_output")
    parser.add_argument("--timings", default="p0,p1,p1_p8,until_marker")
    parser.add_argument("--patch_types", default="delete,induce,rescue,cancel")
    parser.add_argument("--intervention_layer", type=int, default=22)
    parser.add_argument("--intervention_site", default="post_attn")
    parser.add_argument("--gate_alpha", type=float, default=0.5)
    parser.add_argument("--gate_modes", default="prefill_only,decode_only,prefill_plus_decode")
    parser.add_argument("--main_gate_mode", default="prefill_plus_decode")
    parser.add_argument("--timing_sweep_layer", type=int, default=22)
    parser.add_argument("--timing_sweep_site", default="post_attn")
    parser.add_argument("--full_cross", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gate_direction_cache_in", required=True)
    parser.add_argument("--continuation_stop_strings", default="")
    parser.add_argument(
        "--forced_prefix_text",
        default="",
        help="Optional decoded text forced at the beginning of every continuation, e.g. '$$\\n\\nWait'.",
    )
    parser.add_argument(
        "--ban_stop_tokens_until",
        type=int,
        default=-1,
        help="If non-negative, ban EOS and stop/finalization first-token ids through this continuation position.",
    )
    parser.add_argument("--save_raw_activations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_patches", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print_every", type=int, default=1)
    return parser.parse_args()


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_choice_list(values: Sequence[str], supported: set[str], label: str) -> List[str]:
    out: List[str] = []
    for value in values:
        if value not in supported:
            raise ValueError(f"Unsupported {label} {value!r}; supported={sorted(supported)}")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError(f"No {label} selected.")
    return out


def condition_full_ids(prepared: Dict[str, Any], condition: str) -> List[int]:
    if condition == "T" or condition.startswith("T_gateoff"):
        return list(prepared["tamper_full_ids"])
    if condition == "C" or condition.startswith("C_gateon"):
        return list(prepared["coherent_full_ids"])
    raise ValueError(f"Unsupported condition: {condition}")


def base_condition_for_patch(patch_type: str, gate_mode: str) -> str:
    if patch_type == "delete":
        return "T"
    if patch_type == "induce":
        return "C"
    if patch_type == "rescue":
        return f"T_gateoff_{gate_mode}"
    if patch_type == "cancel":
        return f"C_gateon_{gate_mode}"
    raise ValueError(patch_type)


def source_condition_for_patch(patch_type: str) -> str:
    if patch_type in {"induce", "rescue"}:
        return "T"
    if patch_type in {"delete", "cancel"}:
        return "C"
    raise ValueError(patch_type)


def patch_condition_name(
    *,
    patch_type: str,
    gate_mode: str,
    layer_idx: int,
    site: str,
    timing: str,
) -> str:
    mode_part = "" if patch_type in {"delete", "induce"} else f"_{gate_mode}"
    return f"patch_{patch_type}{mode_part}_L{int(layer_idx)}_{site}_{timing}"


def build_patch_specs(
    *,
    layers: Sequence[int],
    sites: Sequence[str],
    timings: Sequence[str],
    patch_types: Sequence[str],
    gate_modes: Sequence[str],
    main_gate_mode: str,
    timing_sweep_layer: int,
    timing_sweep_site: str,
    full_cross: bool,
) -> List[PatchSpec]:
    specs: Dict[str, PatchSpec] = {}

    def add_spec(patch_type: str, gate_mode: str, layer_idx: int, site: str, timing: str, sweep: str) -> None:
        condition = patch_condition_name(
            patch_type=patch_type,
            gate_mode=gate_mode,
            layer_idx=layer_idx,
            site=site,
            timing=timing,
        )
        specs[condition] = PatchSpec(
            condition=condition,
            patch_type=patch_type,
            base_condition=base_condition_for_patch(patch_type, gate_mode),
            source_condition=source_condition_for_patch(patch_type),
            layer_idx=int(layer_idx),
            site=str(site),
            timing=str(timing),
            gate_mode=str(gate_mode),
            sweep=sweep,
        )

    if full_cross:
        for patch_type in patch_types:
            modes = gate_modes if patch_type in {"rescue", "cancel"} else [main_gate_mode]
            for mode in modes:
                for layer_idx in layers:
                    for site in sites:
                        for timing in timings:
                            add_spec(patch_type, mode, layer_idx, site, timing, "full_cross")
        return sorted(specs.values(), key=lambda x: x.condition)

    for patch_type in patch_types:
        mode = main_gate_mode
        for layer_idx in layers:
            for site in sites:
                add_spec(patch_type, mode, layer_idx, site, "p0", "layer_site_p0")

    if timing_sweep_layer in layers and timing_sweep_site in sites:
        for patch_type in patch_types:
            mode = main_gate_mode
            for timing in timings:
                add_spec(patch_type, mode, timing_sweep_layer, timing_sweep_site, timing, "timing_main_site")

        for patch_type in [x for x in patch_types if x in {"rescue", "cancel"}]:
            for mode in gate_modes:
                add_spec(patch_type, mode, timing_sweep_layer, timing_sweep_site, "p0", "gate_mode_main_site")

    return sorted(specs.values(), key=lambda x: x.condition)


def add_intervention_for_condition(
    *,
    condition: str,
    gate_dir: torch.Tensor,
    gate_scale: float,
    gate_alpha: float,
    layer_idx: int,
    site: str,
) -> Optional[ScheduledAddIntervention]:
    if condition.startswith("T_gateoff_"):
        mode = condition.removeprefix("T_gateoff_")
        add = -gate_dir * (float(gate_alpha) * float(gate_scale))
    elif condition.startswith("C_gateon_"):
        mode = condition.removeprefix("C_gateon_")
        add = gate_dir * (float(gate_alpha) * float(gate_scale))
    else:
        return None
    if mode not in SUPPORTED_GATE_MODES:
        raise ValueError(f"Unsupported gate mode parsed from {condition}: {mode}")
    return ScheduledAddIntervention(
        layer_idx=int(layer_idx),
        site=str(site),
        add_vector=add.detach().float().cpu(),
        mode=mode,
    )


def run_analysis_payload(
    *,
    tokenizer,
    trajectory: Dict[str, Any],
    correct_answer: str,
    wrong_answer: str,
) -> Dict[str, Any]:
    generated_ids = [int(x) for x in trajectory["generated_token_ids"]]
    continuation_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    full_text = str(trajectory["full_text"])
    analysis = analyze_continuation_text(
        continuation=continuation_text,
        full_text=full_text,
        correct_answer=correct_answer,
        wrong_answer=wrong_answer,
        reflection_keywords=DEFAULT_REFLECTION_KEYWORDS,
    )
    first_id = generated_ids[0] if generated_ids else None
    first_text = tokenizer.decode([first_id], skip_special_tokens=False) if first_id is not None else ""
    p0_logits = trajectory["logit_rows"][0] if trajectory.get("logit_rows") else {}
    semantic_repair = bool(analysis.get("full_text_final_matches_correct"))
    surface_reflection = bool(analysis.get("has_reflection"))
    lower_cont = continuation_text.lower()
    explicit_repair_marker = any(kw.lower() in lower_cont for kw in REPAIR_NEGATION_KEYWORDS)
    return {
        "first_generated_token_id": first_id,
        "first_generated_token_text": first_text,
        "first_wait": bool(first_text in WAIT_FIRST_TOKENS),
        "generated_tokens": len(generated_ids),
        "hit_max_new_tokens": bool(trajectory.get("hit_max_new_tokens")),
        "stop_reason": trajectory.get("stop_reason"),
        "continuation_text": continuation_text,
        "surface_reflection": surface_reflection,
        "semantic_repair": semantic_repair,
        "final_correction": bool(analysis.get("full_text_final_matches_correct")),
        "functional_repair": bool(semantic_repair and (surface_reflection or explicit_repair_marker)),
        "explicit_repair_marker": bool(explicit_repair_marker),
        "p0_reflect_vs_stop": p0_logits.get("reflect_vs_stop"),
        "p0_wait_logsum": p0_logits.get("wait_logsum"),
        **analysis,
    }


def summarize_behavior(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("condition_kind", "baseline")),
                str(row.get("condition")),
                str(row.get("patch_type", "")),
                str(row.get("patch_layer_idx", "")),
                str(row.get("patch_site", "")),
                str(row.get("patch_timing", "")),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (kind, condition, patch_type, layer_idx, site, timing), group in sorted(grouped.items()):
        item = {
            "condition_kind": kind,
            "condition": condition,
            "patch_type": patch_type,
            "patch_layer_idx": layer_idx,
            "patch_site": site,
            "patch_timing": timing,
            "base_condition": group[0].get("base_condition", ""),
            "source_condition": group[0].get("source_condition", ""),
            "gate_mode": group[0].get("gate_mode", ""),
            "count": len(group),
            "first_wait_rate": mean([1.0 if row.get("first_wait") else 0.0 for row in group]),
            "surface_reflection_rate": mean([1.0 if row.get("surface_reflection") else 0.0 for row in group]),
            "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
            "semantic_repair_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
            "functional_repair_rate": mean([1.0 if row.get("functional_repair") else 0.0 for row in group]),
            "final_correction_rate": mean([1.0 if row.get("final_correction") else 0.0 for row in group]),
            "mean_generated_tokens": mean([row.get("generated_tokens") for row in group]),
            "cap_rate": mean([1.0 if row.get("hit_max_new_tokens") else 0.0 for row in group]),
            "mean_p0_reflect_vs_stop": mean([row.get("p0_reflect_vs_stop") for row in group]),
            "mean_patch_hook_calls": mean([row.get("patch_hook_calls") for row in group]),
        }
        out.append(item)
    return out


def activation_entry(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_token_ids": trajectory["generated_token_ids"],
        "generated_text": trajectory["generated_text"],
        "full_token_ids": trajectory["full_token_ids"],
        "stop_reason": trajectory["stop_reason"],
        "hit_max_new_tokens": trajectory["hit_max_new_tokens"],
        "position_records": trajectory["position_records"],
        "debug_rows": trajectory["debug_rows"],
        "activations": trajectory["activations"],
    }


def patch_vectors_from_source(source_run: Dict[str, Any], *, layer_idx: int, site: str, max_position: int) -> Dict[int, torch.Tensor]:
    key = f"L{int(layer_idx)}/{site}"
    tensor = (source_run.get("activations") or {}).get(key)
    if tensor is None:
        return {}
    n_pos = min(int(tensor.shape[0]), int(max_position) + 1)
    return {int(pos): tensor[pos].detach().float().cpu() for pos in range(n_pos)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    activation_dir = output_dir / "activation_traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    backend = build_backend(args)
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("This script requires an HF backend.")
    decoder_layers, layer_path = get_decoder_layers(model)
    layers = list(decoder_layers)
    selected_layers = parse_layer_spec(args.layers, len(layers))
    selected_sites = validate_choice_list(parse_csv_list(args.sites), {"post_attn", "block_output", "block_input"}, "site")
    selected_timings = validate_choice_list(parse_csv_list(args.timings), SUPPORTED_TIMINGS, "timing")
    selected_patch_types = validate_choice_list(parse_csv_list(args.patch_types), SUPPORTED_PATCH_TYPES, "patch type")
    gate_modes = validate_choice_list(parse_csv_list(args.gate_modes), SUPPORTED_GATE_MODES, "gate mode")
    if args.main_gate_mode not in gate_modes:
        raise ValueError("--main_gate_mode must be included in --gate_modes")

    gen_config = build_generation_config(args)
    stage1_stop = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    continuation_stop_strings = parse_csv_list(args.continuation_stop_strings)
    stop_sequences = stop_id_sequences(backend, continuation_stop_strings)
    forced_prefix_text = str(args.forced_prefix_text).replace("\\$", "$").replace("\\n", "\n")
    forced_prefix_ids = tokenizer.encode(forced_prefix_text, add_special_tokens=False) if forced_prefix_text else []

    reflect_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    wait_ids = token_ids_for_first_tokens(tokenizer, [" Wait", " wait", "Wait", "wait"])
    check_ids = token_ids_for_first_tokens(tokenizer, [" check", " Check", "check", " verify", " recalculate"])
    actually_ids = token_ids_for_first_tokens(tokenizer, [" Actually", " actually", "Actually"])
    finalize_ids = token_ids_for_first_tokens(tokenizer, [" Therefore", " Thus", " So", " Hence", "</think>"])
    newline_ids = token_ids_for_first_tokens(tokenizer, ["\n", "\n\n"])
    eos_ids = [int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else []
    banned_stop_ids = sorted(set(eos_ids + stop_ids + finalize_ids))
    token_sets = {
        "reflect": reflect_ids,
        "stop": stop_ids,
        "wait": wait_ids,
        "check": check_ids,
        "actually": actually_ids,
        "finalize": finalize_ids,
        "newline": newline_ids,
    }
    tracked_token_ids = sorted(set(reflect_ids + stop_ids + wait_ids + check_ids + actually_ids + finalize_ids + newline_ids))

    gate_info = load_gate_direction_cache(Path(args.gate_direction_cache_in), args.intervention_layer, args.intervention_site)
    if gate_info is None:
        raise ValueError(f"No gate direction for L{args.intervention_layer}/{args.intervention_site} in {args.gate_direction_cache_in}")
    gate_dir = tensor_normed(gate_info["direction"])
    gate_scale = float(gate_info["scale"])

    patch_specs = build_patch_specs(
        layers=selected_layers,
        sites=selected_sites,
        timings=selected_timings,
        patch_types=selected_patch_types,
        gate_modes=gate_modes,
        main_gate_mode=args.main_gate_mode,
        timing_sweep_layer=int(args.timing_sweep_layer),
        timing_sweep_site=str(args.timing_sweep_site),
        full_cross=bool(args.full_cross),
    )
    baseline_conditions = ["T", "C"]
    for mode in gate_modes:
        baseline_conditions.append(f"T_gateoff_{mode}")
        baseline_conditions.append(f"C_gateon_{mode}")
    for spec in patch_specs:
        if spec.base_condition not in baseline_conditions:
            baseline_conditions.append(spec.base_condition)

    rows = load_jsonl(args.input_jsonl)
    examples = rows[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "layer_path": layer_path,
            "selected_layers": selected_layers,
            "selected_sites": selected_sites,
            "selected_timings": selected_timings,
            "selected_patch_types": selected_patch_types,
            "gate_modes": gate_modes,
            "baseline_conditions": baseline_conditions,
            "patch_specs": [asdict(spec) for spec in patch_specs],
            "reflect_token_ids": reflect_ids,
            "stop_token_ids": stop_ids,
            "wait_token_ids": wait_ids,
            "check_token_ids": check_ids,
            "actually_token_ids": actually_ids,
            "finalize_token_ids": finalize_ids,
            "newline_token_ids": newline_ids,
            "eos_token_ids": eos_ids,
            "banned_stop_token_ids": banned_stop_ids,
            "ban_stop_tokens_until": int(args.ban_stop_tokens_until),
            "forced_prefix_text": forced_prefix_text,
            "forced_prefix_ids": forced_prefix_ids,
            "generation": asdict(gen_config),
            "gate_direction": {
                "source": gate_info.get("source"),
                "n_pairs": gate_info.get("n_pairs"),
                "scale": gate_scale,
                "mean_diff_norm": gate_info.get("mean_diff_norm"),
                "layer_idx": int(args.intervention_layer),
                "site": args.intervention_site,
            },
        },
    )
    write_csv(output_dir / "patch_specs.csv", [asdict(spec) for spec in patch_specs])

    behavior_rows: List[Dict[str, Any]] = []
    logit_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    common_run_kwargs = {
        "layers": layers,
        "capture_layer_indices": selected_layers,
        "capture_sites": selected_sites,
        "max_new_tokens": int(args.max_continuation_tokens),
        "capture_max_position_index": int(args.capture_max_position_index),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "stop_id_sequences": stop_sequences,
        "banned_token_ids": banned_stop_ids if int(args.ban_stop_tokens_until) >= 0 else [],
        "ban_until_position": int(args.ban_stop_tokens_until) if int(args.ban_stop_tokens_until) >= 0 else None,
        "forced_prefix_ids": forced_prefix_ids,
        "token_sets": token_sets,
        "tracked_token_ids": tracked_token_ids,
    }

    iterator = tqdm(examples, desc="Patch rescue/delete", dynamic_ncols=True)
    for local_idx, ex in enumerate(iterator):
        global_idx = int(args.start_idx) + local_idx
        ex_id = row_id(ex, global_idx)
        question = row_question(ex)
        correct_answer = row_answer(ex)
        wrong_answer = str(ex.get("wrong_answer") or "").strip()
        if not question or correct_answer is None or not wrong_answer:
            skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "missing_fields"})
            continue
        try:
            seed_everything(int(args.seed) + global_idx)
            prepared, skip = prepare_forced_box_example(
                backend=backend,
                tokenizer=tokenizer,
                gen_config=gen_config,
                example=ex,
                example_id=ex_id,
                correct_answer=str(correct_answer),
                wrong_answer=wrong_answer,
                stop_strings=stage1_stop,
                allow_nonmatching_clean=bool(args.allow_nonmatching_clean),
                coherent_window_chars=int(args.coherent_window_chars),
            )
            if prepared is None:
                skipped_rows.append({"global_idx": global_idx, **(skip or {"example_id": ex_id, "reason": "prepare_failed"})})
                continue

            activation_payload: Dict[str, Any] = {
                "example_id": ex_id,
                "global_idx": global_idx,
                "question": question,
                "correct_answer": str(correct_answer),
                "wrong_answer": wrong_answer,
                "metadata": {
                    "clean_box_answer": prepared["clean_box_answer"],
                    "box_span": prepared["box_span"],
                    "coherent_replacement_count": prepared["coherent_replacement_count"],
                    "coherent_replacement_targets": prepared["coherent_replacement_targets"],
                    "prompt_token_len": len(prepared["prompt_ids"]),
                    "clean_force_token_len": prepared["clean_force_token_len"],
                    "tamper_force_token_len": prepared["tamper_force_token_len"],
                    "wrong_answer_token_len": prepared["wrong_answer_token_len"],
                },
                "runs": {"free": {}},
            }

            reference_runs: Dict[str, Dict[str, Any]] = {}
            for condition in baseline_conditions:
                seed_everything(int(args.seed) + 100000 + global_idx)
                trajectory = run_patchable_trajectory(
                    model,
                    tokenizer,
                    full_ids=condition_full_ids(prepared, condition),
                    add_intervention=add_intervention_for_condition(
                        condition=condition,
                        gate_dir=gate_dir,
                        gate_scale=gate_scale,
                        gate_alpha=float(args.gate_alpha),
                        layer_idx=int(args.intervention_layer),
                        site=str(args.intervention_site),
                    ),
                    patch_plan=None,
                    forced_continuation_ids=None,
                    **common_run_kwargs,
                )
                reference_runs[condition] = trajectory
                analysis = run_analysis_payload(
                    tokenizer=tokenizer,
                    trajectory=trajectory,
                    correct_answer=str(correct_answer),
                    wrong_answer=wrong_answer,
                )
                behavior_rows.append(
                    {
                        "example_id": ex_id,
                        "global_idx": global_idx,
                        "mode": "free",
                        "condition": condition,
                        "condition_kind": "baseline",
                        "correct_answer": str(correct_answer),
                        "wrong_answer": wrong_answer,
                        "clean_box_answer": prepared["clean_box_answer"],
                        "coherent_replacement_count": prepared["coherent_replacement_count"],
                        **analysis,
                    }
                )
                for row in trajectory.get("logit_rows") or []:
                    logit_rows.append({"example_id": ex_id, "global_idx": global_idx, "mode": "free", "condition": condition, **row})
                if args.save_raw_activations:
                    activation_payload["runs"]["free"][condition] = activation_entry(trajectory)

            if not bool(args.skip_patches):
                source_replays: Dict[Tuple[str, str], Dict[str, Any]] = {}
                for source_condition, base_condition in sorted({(s.source_condition, s.base_condition) for s in patch_specs}):
                    base_tokens = reference_runs[base_condition]["generated_token_ids"]
                    trajectory = run_patchable_trajectory(
                        model,
                        tokenizer,
                        full_ids=condition_full_ids(prepared, source_condition),
                        add_intervention=None,
                        patch_plan=None,
                        forced_continuation_ids=base_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        **{k: v for k, v in common_run_kwargs.items() if k not in {"do_sample", "temperature", "top_p"}},
                    )
                    source_replays[(source_condition, base_condition)] = trajectory

                for spec_idx, spec in enumerate(patch_specs):
                    source_run = source_replays.get((spec.source_condition, spec.base_condition))
                    if source_run is None:
                        skipped_rows.append(
                            {
                                "example_id": ex_id,
                                "global_idx": global_idx,
                                "reason": "missing_source_replay",
                                "condition": spec.condition,
                            }
                        )
                        continue
                    vectors = patch_vectors_from_source(
                        source_run,
                        layer_idx=int(spec.layer_idx),
                        site=str(spec.site),
                        max_position=int(args.capture_max_position_index),
                    )
                    if not vectors:
                        skipped_rows.append(
                            {
                                "example_id": ex_id,
                                "global_idx": global_idx,
                                "reason": "missing_source_patch_vectors",
                                "condition": spec.condition,
                                "source_condition": spec.source_condition,
                                "base_condition": spec.base_condition,
                                "layer_idx": int(spec.layer_idx),
                                "site": spec.site,
                            }
                        )
                        continue
                    patch_plan = PatchPlan(
                        layer_idx=int(spec.layer_idx),
                        site=str(spec.site),
                        vectors_by_position=vectors,
                        timing=str(spec.timing),
                        reflection_keywords=tuple(DEFAULT_REFLECTION_KEYWORDS),
                    )
                    seed_everything(int(args.seed) + 300000 + global_idx + spec_idx)
                    trajectory = run_patchable_trajectory(
                        model,
                        tokenizer,
                        full_ids=condition_full_ids(prepared, spec.base_condition),
                        add_intervention=add_intervention_for_condition(
                            condition=spec.base_condition,
                            gate_dir=gate_dir,
                            gate_scale=gate_scale,
                            gate_alpha=float(args.gate_alpha),
                            layer_idx=int(args.intervention_layer),
                            site=str(args.intervention_site),
                        ),
                        patch_plan=patch_plan,
                        forced_continuation_ids=None,
                        **common_run_kwargs,
                    )
                    analysis = run_analysis_payload(
                        tokenizer=tokenizer,
                        trajectory=trajectory,
                        correct_answer=str(correct_answer),
                        wrong_answer=wrong_answer,
                    )
                    debug_rows = trajectory.get("debug_rows") or []
                    patch_hook_calls = sum(int(row.get("patch_hook_call_count") or 0) for row in debug_rows)
                    behavior_rows.append(
                        {
                            "example_id": ex_id,
                            "global_idx": global_idx,
                            "mode": "free",
                            "condition": spec.condition,
                            "condition_kind": "patch",
                            "patch_type": spec.patch_type,
                            "base_condition": spec.base_condition,
                            "source_condition": spec.source_condition,
                            "patch_layer_idx": int(spec.layer_idx),
                            "patch_site": spec.site,
                            "patch_timing": spec.timing,
                            "gate_mode": spec.gate_mode,
                            "sweep": spec.sweep,
                            "patch_hook_calls": int(patch_hook_calls),
                            "correct_answer": str(correct_answer),
                            "wrong_answer": wrong_answer,
                            "clean_box_answer": prepared["clean_box_answer"],
                            "coherent_replacement_count": prepared["coherent_replacement_count"],
                            **analysis,
                        }
                    )
                    for row in trajectory.get("logit_rows") or []:
                        logit_rows.append({"example_id": ex_id, "global_idx": global_idx, "mode": "free", "condition": spec.condition, **row})
                    if args.save_raw_activations:
                        activation_payload["runs"]["free"][spec.condition] = activation_entry(trajectory)

            if args.save_raw_activations:
                torch.save(activation_payload, activation_dir / f"{global_idx:05d}_{safe_name(ex_id)}.pt")

            if args.print_every > 0 and (local_idx + 1) % int(args.print_every) == 0:
                iterator.set_postfix({"rows": len(behavior_rows), "patches": len(patch_specs), "skipped": len(skipped_rows)})
                dump_jsonl(output_dir / "behavior_rows.partial.jsonl", behavior_rows)
                dump_jsonl(output_dir / "logit_rows.partial.jsonl", logit_rows)
                dump_jsonl(output_dir / "skipped_rows.partial.jsonl", skipped_rows)
                write_csv(output_dir / "behavior_summary.partial.csv", summarize_behavior(behavior_rows))
                write_csv(output_dir / "behavior_summary_basic.partial.csv", summarize_basic_behavior(behavior_rows))
        except Exception as exc:
            skipped_rows.append(
                {
                    "example_id": ex_id,
                    "global_idx": global_idx,
                    "reason": "exception",
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    dump_jsonl(output_dir / "behavior_rows.jsonl", behavior_rows)
    dump_jsonl(output_dir / "logit_rows.jsonl", logit_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    summary_rows = summarize_behavior(behavior_rows)
    write_csv(output_dir / "behavior_summary.csv", summary_rows)
    write_csv(output_dir / "behavior_summary_basic.csv", summarize_basic_behavior(behavior_rows))
    reason_counts = Counter(str(row.get("reason")) for row in skipped_rows)
    write_json(
        output_dir / "summary.json",
        {
            "behavior_rows": len(behavior_rows),
            "logit_rows": len(logit_rows),
            "skipped_rows": len(skipped_rows),
            "skipped_reasons": dict(reason_counts),
            "usable_examples": len({row["example_id"] for row in behavior_rows}),
            "baseline_conditions": baseline_conditions,
            "patch_specs": len(patch_specs),
            "summary_rows": len(summary_rows),
        },
    )
    print("[Done] Reflection patch rescue/delete finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_examples: {len({row['example_id'] for row in behavior_rows})}")
    print(f"- behavior_rows: {len(behavior_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
