#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.evidence_regions import annotate_forced_box_regions, serialize_regions
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.patch_trajectory import PatchPlan, ScheduledAddIntervention, ScheduledSourceMask, run_patchable_trajectory
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
from run_reflection_patch_rescue import activation_entry, patch_vectors_from_source


SUPPORTED_MASK_TIMINGS = {"p1", "p1_p8", "p1_p16", "p2_p16", "until_marker"}
DEFAULT_MASK_REGIONS = [
    "box_wrong",
    "nearest_clean_anchor",
    "prev_k_clean_anchors",
    "local_evidence_window",
    "inconsistency_anchors",
    "conflict_all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidence-to-gate maintenance experiment: source-region attention masks, rescue-then-mask, and marker controls."
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
    parser.add_argument("--local_window_chars", type=int, default=1200)
    parser.add_argument("--prev_k_anchors", type=int, default=4)
    parser.add_argument("--layers", default="19-22")
    parser.add_argument("--sites", default="post_attn,block_output")
    parser.add_argument("--mask_regions", default=",".join(DEFAULT_MASK_REGIONS))
    parser.add_argument("--mask_timings", default="p1_p8,p1_p16")
    parser.add_argument("--intervention_layer", type=int, default=22)
    parser.add_argument("--intervention_site", default="post_attn")
    parser.add_argument("--gate_alpha", type=float, default=0.5)
    parser.add_argument("--gate_mode", default="prefill_plus_decode")
    parser.add_argument("--rescue_layer", type=int, default=22)
    parser.add_argument("--rescue_site", default="post_attn")
    parser.add_argument("--gate_direction_cache_in", required=True)
    parser.add_argument("--continuation_stop_strings", default="")
    parser.add_argument("--save_raw_activations", action=argparse.BooleanOptionalAction, default=True)
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


def condition_full_ids(prepared: Dict[str, Any], condition: str) -> List[int]:
    if condition.startswith("T"):
        return list(prepared["tamper_full_ids"])
    if condition.startswith("C"):
        return list(prepared["coherent_full_ids"])
    raise ValueError(condition)


def make_gate_add(
    *,
    condition: str,
    gate_dir: torch.Tensor,
    gate_scale: float,
    gate_alpha: float,
    layer_idx: int,
    site: str,
    gate_mode: str,
) -> Optional[ScheduledAddIntervention]:
    if condition.startswith("T_gateoff"):
        add = -gate_dir * (float(gate_alpha) * float(gate_scale))
    elif condition.startswith("C_gateon"):
        add = gate_dir * (float(gate_alpha) * float(gate_scale))
    else:
        return None
    return ScheduledAddIntervention(
        layer_idx=int(layer_idx),
        site=str(site),
        add_vector=add.detach().float().cpu(),
        mode=str(gate_mode),
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
    surface_reflection = bool(analysis.get("has_reflection"))
    semantic_repair = bool(analysis.get("full_text_final_matches_correct"))
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
        "final_correction": semantic_repair,
        "functional_repair": bool(semantic_repair and surface_reflection),
        "p0_reflect_vs_stop": p0_logits.get("reflect_vs_stop"),
        "p0_wait_logsum": p0_logits.get("wait_logsum"),
        **analysis,
    }


def summarize_behavior(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("condition_kind", "baseline")),
                str(row.get("condition")),
                str(row.get("mask_region", "")),
                str(row.get("mask_timing", "")),
                str(row.get("control_type", "")),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (kind, condition, mask_region, mask_timing, control_type), group in sorted(grouped.items()):
        out.append(
            {
                "condition_kind": kind,
                "condition": condition,
                "mask_region": mask_region,
                "mask_timing": mask_timing,
                "control_type": control_type,
                "count": len(group),
                "first_wait_rate": mean([1.0 if row.get("first_wait") else 0.0 for row in group]),
                "surface_reflection_rate": mean([1.0 if row.get("surface_reflection") else 0.0 for row in group]),
                "semantic_repair_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
                "functional_repair_rate": mean([1.0 if row.get("functional_repair") else 0.0 for row in group]),
                "final_correction_rate": mean([1.0 if row.get("final_correction") else 0.0 for row in group]),
                "mean_generated_tokens": mean([row.get("generated_tokens") for row in group]),
                "cap_rate": mean([1.0 if row.get("hit_max_new_tokens") else 0.0 for row in group]),
                "mean_p0_reflect_vs_stop": mean([row.get("p0_reflect_vs_stop") for row in group]),
                "mean_source_mask_calls": mean([row.get("source_mask_calls") for row in group]),
                "mean_patch_hook_calls": mean([row.get("patch_hook_calls") for row in group]),
            }
        )
    return out


def source_mask_for(region_indices: Sequence[int], timing: str) -> ScheduledSourceMask:
    return ScheduledSourceMask(
        token_indices=tuple(int(x) for x in region_indices),
        timing=str(timing),
        reflection_keywords=tuple(DEFAULT_REFLECTION_KEYWORDS),
    )


def debug_counts(trajectory: Dict[str, Any]) -> Dict[str, int]:
    debug_rows = trajectory.get("debug_rows") or []
    return {
        "source_mask_calls": sum(1 for row in debug_rows if bool(row.get("source_mask_active"))),
        "source_masked_tokens": sum(int(row.get("source_mask_count") or 0) for row in debug_rows),
        "patch_hook_calls": sum(int(row.get("patch_hook_call_count") or 0) for row in debug_rows),
        "add_hook_calls": sum(int(row.get("add_hook_call_count") or 0) for row in debug_rows),
    }


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
    selected_sites = parse_csv_list(args.sites)
    mask_regions = parse_csv_list(args.mask_regions)
    mask_timings = parse_csv_list(args.mask_timings)
    unsupported_timings = [x for x in mask_timings if x not in SUPPORTED_MASK_TIMINGS]
    if unsupported_timings:
        raise ValueError(f"Unsupported mask timings: {unsupported_timings}")

    gen_config = build_generation_config(args)
    stage1_stop = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    continuation_stop_strings = parse_csv_list(args.continuation_stop_strings)
    stop_sequences = stop_id_sequences(backend, continuation_stop_strings)

    reflect_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    wait_ids = token_ids_for_first_tokens(tokenizer, [" Wait", " wait", "Wait", "wait"])
    check_ids = token_ids_for_first_tokens(tokenizer, [" check", " Check", "check", " verify", " recalculate"])
    actually_ids = token_ids_for_first_tokens(tokenizer, [" Actually", " actually", "Actually"])
    finalize_ids = token_ids_for_first_tokens(tokenizer, [" Therefore", " Thus", " So", " Hence", "</think>"])
    newline_ids = token_ids_for_first_tokens(tokenizer, ["\n", "\n\n"])
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
    banned_marker_ids = sorted(set(reflect_ids + wait_ids + check_ids + actually_ids))
    force_wait_ids = token_ids_for_first_tokens(tokenizer, [" Wait"])
    neutral_prefix_ids = backend.encode(" The answer is")
    neutral_prefix_ids = neutral_prefix_ids[:4]

    gate_info = load_gate_direction_cache(Path(args.gate_direction_cache_in), args.intervention_layer, args.intervention_site)
    if gate_info is None:
        raise ValueError(f"No gate direction for L{args.intervention_layer}/{args.intervention_site} in {args.gate_direction_cache_in}")
    gate_dir = tensor_normed(gate_info["direction"])
    gate_scale = float(gate_info["scale"])

    rows = load_jsonl(args.input_jsonl)
    examples = rows[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "layer_path": layer_path,
            "selected_layers": selected_layers,
            "selected_sites": selected_sites,
            "mask_regions": mask_regions,
            "mask_timings": mask_timings,
            "reflect_token_ids": reflect_ids,
            "stop_token_ids": stop_ids,
            "banned_marker_ids": banned_marker_ids,
            "force_wait_ids": force_wait_ids,
            "neutral_prefix_ids": neutral_prefix_ids,
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

    behavior_rows: List[Dict[str, Any]] = []
    logit_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []

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
        "token_sets": token_sets,
        "tracked_token_ids": tracked_token_ids,
    }

    baseline_conditions = ["T", "C", "T_gateoff", "C_gateon"]

    iterator = tqdm(examples, desc="Evidence-to-gate maintenance", dynamic_ncols=True)
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
            regions = annotate_forced_box_regions(
                tokenizer,
                prepared=prepared,
                correct_answer=str(correct_answer),
                local_window_chars=int(args.local_window_chars),
                prev_k=int(args.prev_k_anchors),
            )
            for name, region in regions.items():
                region_rows.append(
                    {
                        "example_id": ex_id,
                        "global_idx": global_idx,
                        "region": name,
                        "n_tokens": len(region.token_indices),
                        "token_indices": json.dumps(list(region.token_indices)),
                        "meta": json.dumps(dict(region.meta), ensure_ascii=False),
                    }
                )
            missing_regions = [name for name in mask_regions if name not in regions or not regions[name].token_indices]
            if missing_regions:
                skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "empty_regions", "regions": ",".join(missing_regions)})

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
                    "regions": serialize_regions(regions),
                },
                "runs": {"free": {}},
            }

            reference_runs: Dict[str, Dict[str, Any]] = {}

            def record_run(condition: str, trajectory: Dict[str, Any], extra: Dict[str, Any]) -> None:
                analysis = run_analysis_payload(
                    tokenizer=tokenizer,
                    trajectory=trajectory,
                    correct_answer=str(correct_answer),
                    wrong_answer=wrong_answer,
                )
                counts = debug_counts(trajectory)
                behavior_rows.append(
                    {
                        "example_id": ex_id,
                        "global_idx": global_idx,
                        "mode": "free",
                        "condition": condition,
                        "correct_answer": str(correct_answer),
                        "wrong_answer": wrong_answer,
                        "clean_box_answer": prepared["clean_box_answer"],
                        "coherent_replacement_count": prepared["coherent_replacement_count"],
                        **extra,
                        **counts,
                        **analysis,
                    }
                )
                for row in trajectory.get("logit_rows") or []:
                    logit_rows.append({"example_id": ex_id, "global_idx": global_idx, "mode": "free", "condition": condition, **row})
                if args.save_raw_activations:
                    activation_payload["runs"]["free"][condition] = activation_entry(trajectory)

            for condition in baseline_conditions:
                seed_everything(int(args.seed) + 100000 + global_idx)
                trajectory = run_patchable_trajectory(
                    model,
                    tokenizer,
                    full_ids=condition_full_ids(prepared, condition),
                    add_intervention=make_gate_add(
                        condition=condition,
                        gate_dir=gate_dir,
                        gate_scale=gate_scale,
                        gate_alpha=float(args.gate_alpha),
                        layer_idx=int(args.intervention_layer),
                        site=str(args.intervention_site),
                        gate_mode=str(args.gate_mode),
                    ),
                    patch_plan=None,
                    source_mask=None,
                    forced_continuation_ids=None,
                    **common_run_kwargs,
                )
                reference_runs[condition] = trajectory
                record_run(condition, trajectory, {"condition_kind": "baseline", "control_type": condition})

            for region_name in mask_regions:
                region = regions.get(region_name)
                if region is None or not region.token_indices:
                    continue
                for timing in mask_timings:
                    condition = f"T_mask_{region_name}_{timing}"
                    trajectory = run_patchable_trajectory(
                        model,
                        tokenizer,
                        full_ids=condition_full_ids(prepared, "T"),
                        add_intervention=None,
                        patch_plan=None,
                        source_mask=source_mask_for(region.token_indices, timing),
                        forced_continuation_ids=None,
                        **common_run_kwargs,
                    )
                    record_run(
                        condition,
                        trajectory,
                        {
                            "condition_kind": "source_mask",
                            "control_type": "C1",
                            "mask_region": region_name,
                            "mask_timing": timing,
                        },
                    )

            conflict_region = regions.get("conflict_all")
            if conflict_region is not None and conflict_region.token_indices:
                c3_specs = [
                    ("T_evidence_masked_p1p16", None, "p1_p16", "C3_mask_only"),
                    ("T_rescue_p1_only_then_mask", "p1", "p2_p16", "C3_rescue_then_mask"),
                    ("T_rescue_p1p8_then_mask", "p1_p8", "p1_p16", "C3_rescue_with_mask"),
                ]
                for condition, patch_timing, mask_timing, control_type in c3_specs:
                    patch_plan = None
                    if patch_timing is not None:
                        vectors = patch_vectors_from_source(
                            reference_runs["T"],
                            layer_idx=int(args.rescue_layer),
                            site=str(args.rescue_site),
                            max_position=int(args.capture_max_position_index),
                        )
                        patch_plan = PatchPlan(
                            layer_idx=int(args.rescue_layer),
                            site=str(args.rescue_site),
                            vectors_by_position=vectors,
                            timing=str(patch_timing),
                            reflection_keywords=tuple(DEFAULT_REFLECTION_KEYWORDS),
                        )
                    trajectory = run_patchable_trajectory(
                        model,
                        tokenizer,
                        full_ids=condition_full_ids(prepared, "T"),
                        add_intervention=None,
                        patch_plan=patch_plan,
                        source_mask=source_mask_for(conflict_region.token_indices, str(mask_timing)),
                        forced_continuation_ids=None,
                        **common_run_kwargs,
                    )
                    record_run(
                        condition,
                        trajectory,
                        {
                            "condition_kind": "maintenance",
                            "control_type": control_type,
                            "mask_region": "conflict_all",
                            "mask_timing": mask_timing,
                            "patch_timing": patch_timing or "",
                            "patch_layer_idx": int(args.rescue_layer) if patch_timing else "",
                            "patch_site": str(args.rescue_site) if patch_timing else "",
                        },
                    )

            c4_specs: List[Tuple[str, str, Optional[List[int]], List[int], Optional[int]]] = [
                ("T_gateoff_force_wait", "T_gateoff", [int(force_wait_ids[0])] if force_wait_ids else [], [], None),
                ("T_ban_markers_p1p16", "T", None, banned_marker_ids, 15),
                ("T_force_neutral_prefix4", "T", neutral_prefix_ids, [], None),
                ("C_gateon_ban_markers_p1p16", "C_gateon", None, banned_marker_ids, 15),
            ]
            for condition, base_condition, forced_prefix, banned_ids, ban_until in c4_specs:
                trajectory = run_patchable_trajectory(
                    model,
                    tokenizer,
                    full_ids=condition_full_ids(prepared, base_condition),
                    add_intervention=make_gate_add(
                        condition=base_condition,
                        gate_dir=gate_dir,
                        gate_scale=gate_scale,
                        gate_alpha=float(args.gate_alpha),
                        layer_idx=int(args.intervention_layer),
                        site=str(args.intervention_site),
                        gate_mode=str(args.gate_mode),
                    ),
                    patch_plan=None,
                    source_mask=None,
                    forced_prefix_ids=forced_prefix,
                    banned_token_ids=banned_ids,
                    ban_until_position=ban_until,
                    forced_continuation_ids=None,
                    **common_run_kwargs,
                )
                record_run(
                    condition,
                    trajectory,
                    {
                        "condition_kind": "marker_control",
                        "control_type": "C4",
                        "base_condition": base_condition,
                        "forced_prefix_len": len(forced_prefix or []),
                        "banned_token_count": len(banned_ids),
                        "ban_until_position": ban_until if ban_until is not None else "",
                    },
                )

            if args.save_raw_activations:
                torch.save(activation_payload, activation_dir / f"{global_idx:05d}_{safe_name(ex_id)}.pt")

            if args.print_every > 0 and (local_idx + 1) % int(args.print_every) == 0:
                iterator.set_postfix({"rows": len(behavior_rows), "skipped": len(skipped_rows)})
                dump_jsonl(output_dir / "behavior_rows.partial.jsonl", behavior_rows)
                dump_jsonl(output_dir / "logit_rows.partial.jsonl", logit_rows)
                dump_jsonl(output_dir / "skipped_rows.partial.jsonl", skipped_rows)
                write_csv(output_dir / "region_rows.partial.csv", region_rows)
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
    write_csv(output_dir / "region_rows.csv", region_rows)
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
            "summary_rows": len(summary_rows),
            "mask_regions": mask_regions,
            "mask_timings": mask_timings,
        },
    )
    print("[Done] Evidence-to-gate maintenance finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_examples: {len({row['example_id'] for row in behavior_rows})}")
    print(f"- behavior_rows: {len(behavior_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
