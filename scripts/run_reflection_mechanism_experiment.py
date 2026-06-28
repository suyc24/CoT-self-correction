#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.answer_extraction import answers_match, extract_last_boxed
from cot_research.cot_editing import find_last_boxed_span
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
from run_reflection_patch_rescue import activation_entry, patch_vectors_from_source


MARKER_TEXTS = [
    " Wait",
    " wait",
    "Wait",
    "wait",
    " Actually",
    " actually",
    "Actually",
    " check",
    " Check",
    "check",
    " recheck",
    " mistake",
    " wrong",
    " reconsider",
    " verify",
    " Hmm",
    " But",
    " However",
]

EXPLICIT_ERROR_KEYWORDS = [
    "wrong",
    "incorrect",
    "mistake",
    "inconsistent",
    "not correct",
    "previous answer",
    "boxed answer",
    "actually",
    "wait",
    "不对",
    "错误",
    "矛盾",
]

RECOMPUTE_KEYWORDS = [
    "recompute",
    "recalculate",
    "calculate",
    "check",
    "verify",
    "let's compute",
    "let me compute",
    "重新",
    "重算",
    "检查",
]


@dataclass(frozen=True)
class ConditionSpec:
    condition: str
    full_id_key: str
    condition_kind: str
    gate_sign: int = 0
    gate_mode: str = "prefill_plus_decode"
    forced_prefix_name: str = ""
    banned_markers_until: Optional[int] = None
    extra: Mapping[str, Any] = None


@dataclass(frozen=True)
class PatchSpec:
    condition: str
    experiment: str
    patch_type: str
    base_condition: str
    source_condition: str
    layer_idx: int
    site: str
    timing: str
    condition_kind: str
    sweep: str
    base_gate_sign: int = 0
    base_gate_mode: str = "prefill_plus_decode"
    base_forced_prefix_name: str = ""
    base_banned_markers_until: Optional[int] = None
    source_gate_sign: int = 0
    source_gate_mode: str = "prefill_plus_decode"
    source_forced_prefix_name: str = ""
    source_banned_markers_until: Optional[int] = None
    extra: Mapping[str, Any] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified runner for reflection mechanism experiments D1/D2/D3/D6."
    )
    parser.add_argument(
        "--experiment",
        choices=["D1", "D2", "D3", "D6"],
        required=True,
        help="D1=factorial truth/coherence, D2=token-hidden loop, D3=gate+evidence sufficiency, D6=attention-to-MLP stabilization.",
    )
    parser.add_argument(
        "--input_jsonl",
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--model_size_label", default="qwen3_4b")
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
    parser.add_argument("--system_prompt", default="Please reason step by step, and put your final answer within \\boxed{}.")
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
    parser.add_argument(
        "--runtime_patch_capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture patch source layer/site tensors during the run even when they are not "
            "listed in --layers/--sites. Saved activation traces are still filtered to "
            "--layers/--sites to keep disk usage bounded."
        ),
    )
    parser.add_argument("--intervention_layer", type=int, default=22)
    parser.add_argument("--intervention_site", default="post_attn")
    parser.add_argument("--gate_alpha", type=float, default=0.5)
    parser.add_argument("--gate_direction_cache_in", required=True)
    parser.add_argument("--continuation_stop_strings", default="")
    parser.add_argument("--d1_patch_layers", default="19-22")
    parser.add_argument("--d1_patch_sites", default="post_attn,block_output")
    parser.add_argument("--d1_patch_timings", default="p0,p1,p1_p8")
    parser.add_argument("--d2_patch_sites", default="post_attn,block_output")
    parser.add_argument("--d3_restore_ks", default="1,2,4,8")
    parser.add_argument("--d6_patch_layers", default="19-22")
    parser.add_argument("--d6_patch_sites", default="attn_out,post_attn,mlp_out,block_output")
    parser.add_argument("--d6_patch_timings", default="p0,p1,p1_p4,p1_p8,p8_p16,until_marker")
    parser.add_argument("--save_activations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_logits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_per_example", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
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


def sha1_text(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="replace")).hexdigest()


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for item in parse_csv_list(text):
        if "-" in item:
            left, right = item.split("-", 1)
            out.extend(range(int(left), int(right) + 1))
        else:
            out.append(int(item))
    return out


def replace_last_boxed_answer(text: str, answer: str) -> Optional[str]:
    span = find_last_boxed_span(text)
    if span is None:
        return None
    box_start, box_end, _old = span
    brace_pos = text.find("{", box_start + len("\\boxed"))
    if brace_pos < 0:
        return None
    return text[: brace_pos + 1] + str(answer) + text[box_end - 1 :]


def replace_last_k_occurrences(text: str, old: str, new: str, k: int) -> Tuple[str, int]:
    old = str(old)
    if not old or int(k) <= 0:
        return text, 0
    out = str(text)
    count = 0
    search_end = len(out)
    for _ in range(int(k)):
        idx = out.rfind(old, 0, search_end)
        if idx < 0:
            break
        out = out[:idx] + str(new) + out[idx + len(old) :]
        search_end = idx
        count += 1
    return out, count


def restore_k_anchors_in_coherent(coherent_text: str, *, wrong_answer: str, correct_answer: str, k: int) -> Tuple[Optional[str], int]:
    span = find_last_boxed_span(coherent_text)
    if span is None:
        return None, 0
    box_start, _box_end, _old = span
    prefix = coherent_text[:box_start]
    suffix = coherent_text[box_start:]
    restored_prefix, count = replace_last_k_occurrences(prefix, str(wrong_answer), str(correct_answer), int(k))
    return restored_prefix + suffix, count


def build_full_id_map(tokenizer, backend, prepared: Mapping[str, Any], correct_answer: str, wrong_answer: str, restore_ks: Sequence[int]) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    full_ids: Dict[str, List[int]] = {
        "CC": [int(x) for x in prepared["clean_full_ids"]],
        "IW": [int(x) for x in prepared["tamper_full_ids"]],
        "T": [int(x) for x in prepared["tamper_full_ids"]],
        "CW": [int(x) for x in prepared["coherent_full_ids"]],
        "C": [int(x) for x in prepared["coherent_full_ids"]],
    }
    meta: Dict[str, Any] = {"restore_counts": {}}
    coherent_text = tokenizer.decode(full_ids["CW"], skip_special_tokens=False)
    ic_text = replace_last_boxed_answer(coherent_text, str(correct_answer))
    if ic_text is not None:
        full_ids["IC"] = backend.encode(ic_text)
    for k in restore_ks:
        text, count = restore_k_anchors_in_coherent(
            coherent_text,
            wrong_answer=str(wrong_answer),
            correct_answer=str(correct_answer),
            k=int(k),
        )
        meta["restore_counts"][str(k)] = int(count)
        if text is not None:
            full_ids[f"C_restore_{int(k)}_anchors"] = backend.encode(text)
    return full_ids, meta


def gate_add(
    *,
    gate_sign: int,
    gate_mode: str,
    gate_dir: torch.Tensor,
    gate_scale: float,
    gate_alpha: float,
    layer_idx: int,
    site: str,
) -> Optional[ScheduledAddIntervention]:
    if int(gate_sign) == 0:
        return None
    add = gate_dir * (float(gate_sign) * float(gate_alpha) * float(gate_scale))
    return ScheduledAddIntervention(
        layer_idx=int(layer_idx),
        site=str(site),
        add_vector=add.detach().float().cpu(),
        mode=str(gate_mode),
    )


def debug_counts(trajectory: Mapping[str, Any]) -> Dict[str, int]:
    rows = trajectory.get("debug_rows") or []
    return {
        "patch_hook_calls": sum(int(row.get("patch_hook_call_count") or 0) for row in rows),
        "add_hook_calls": sum(int(row.get("add_hook_call_count") or 0) for row in rows),
        "source_mask_calls": sum(1 for row in rows if bool(row.get("source_mask_active"))),
        "source_masked_tokens": sum(int(row.get("source_mask_count") or 0) for row in rows),
    }


def text_metrics(continuation: str, final_answer: Any, *, correct_answer: str, wrong_answer: str, surface_reflection: bool) -> Dict[str, Any]:
    lower = str(continuation).lower()
    explicit_error_ack = any(kw in lower for kw in EXPLICIT_ERROR_KEYWORDS)
    recompute = any(kw in lower for kw in RECOMPUTE_KEYWORDS)
    final_is_correct = bool(answers_match(final_answer, correct_answer))
    final_is_wrong = bool(answers_match(final_answer, wrong_answer))
    no_box = final_answer is None
    return {
        "explicit_error_ack": bool(explicit_error_ack),
        "recompute_or_revise": bool(recompute),
        "semantic_repair": final_is_correct,
        "final_correction": final_is_correct,
        "functional_repair": bool(final_is_correct and (surface_reflection or explicit_error_ack or recompute)),
        "silent_answer_correction": bool(final_is_correct and not surface_reflection and not explicit_error_ack),
        "no_box": bool(no_box),
        "final_matches_wrong": bool(final_is_wrong),
        "new_answer_not_A_not_B": bool((not no_box) and (not final_is_correct) and (not final_is_wrong)),
        "reject_B": bool(explicit_error_ack and str(wrong_answer).strip() in str(continuation)),
        "uses_restored_anchor": bool(str(correct_answer).strip() and str(correct_answer).strip() in str(continuation)),
        "explicit_search": bool(recompute),
    }


def run_analysis_payload(
    *,
    tokenizer,
    trajectory: Mapping[str, Any],
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
    p0_logits = (trajectory.get("logit_rows") or [{}])[0]
    p0_marker_vs_stop = None
    if p0_logits.get("marker_logsum") is not None and p0_logits.get("stop_logsum") is not None:
        p0_marker_vs_stop = p0_logits.get("marker_logsum") - p0_logits.get("stop_logsum")
    p0_finalize_vs_reflect = None
    if p0_logits.get("finalize_logsum") is not None and p0_logits.get("reflect_logsum") is not None:
        p0_finalize_vs_reflect = p0_logits.get("finalize_logsum") - p0_logits.get("reflect_logsum")
    surface_reflection = bool(analysis.get("has_reflection"))
    final_answer = analysis.get("final_boxed_answer_full_text")
    marker_position = analysis.get("first_reflection_pos")
    return {
        "first_generated_token_id": first_id,
        "first_generated_token_text": first_text,
        "first_wait": bool(first_text in WAIT_FIRST_TOKENS),
        "generated_tokens": len(generated_ids),
        "hit_max_new_tokens": bool(trajectory.get("hit_max_new_tokens")),
        "cap": bool(trajectory.get("hit_max_new_tokens")),
        "stop_reason": trajectory.get("stop_reason"),
        "continuation_text": continuation_text,
        "final_answer": final_answer,
        "surface_reflection": surface_reflection,
        "first_marker_position": marker_position,
        "first_marker_text": analysis.get("first_reflection_keyword"),
        "p0_reflect_vs_stop": p0_logits.get("reflect_vs_stop"),
        "p0_marker_logit_margin": p0_marker_vs_stop,
        "p0_finalization_logit_margin": p0_finalize_vs_reflect,
        **analysis,
        **text_metrics(
            continuation_text,
            final_answer,
            correct_answer=str(correct_answer),
            wrong_answer=str(wrong_answer),
            surface_reflection=surface_reflection,
        ),
    }


def summarize_behavior(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("experiment")),
                str(row.get("condition_kind")),
                str(row.get("condition")),
                str(row.get("patch_type", "")),
                str(row.get("patch_layer", "")),
                str(row.get("patch_timing", "")),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (experiment, kind, condition, patch_type, patch_layer, patch_timing), group in sorted(grouped.items()):
        token_counts = [int(row.get("generated_tokens") or 0) for row in group]
        item = {
            "experiment": experiment,
            "condition_kind": kind,
            "condition": condition,
            "patch_type": patch_type,
            "patch_layer": patch_layer,
            "patch_site": group[0].get("patch_site", ""),
            "patch_timing": patch_timing,
            "base_condition": group[0].get("base_condition", ""),
            "source_condition": group[0].get("source_condition", ""),
            "count": len(group),
            "surface_reflection_rate": mean([1.0 if row.get("surface_reflection") else 0.0 for row in group]),
            "explicit_error_ack_rate": mean([1.0 if row.get("explicit_error_ack") else 0.0 for row in group]),
            "recompute_or_revise_rate": mean([1.0 if row.get("recompute_or_revise") else 0.0 for row in group]),
            "semantic_repair_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
            "functional_repair_rate": mean([1.0 if row.get("functional_repair") else 0.0 for row in group]),
            "silent_answer_correction_rate": mean([1.0 if row.get("silent_answer_correction") else 0.0 for row in group]),
            "recover_A_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
            "reject_B_rate": mean([1.0 if row.get("reject_B") else 0.0 for row in group]),
            "uses_restored_anchor_rate": mean([1.0 if row.get("uses_restored_anchor") else 0.0 for row in group]),
            "new_answer_not_A_not_B_rate": mean([1.0 if row.get("new_answer_not_A_not_B") else 0.0 for row in group]),
            "explicit_search_rate": mean([1.0 if row.get("explicit_search") else 0.0 for row in group]),
            "mean_generated_tokens": mean(token_counts),
            "median_generated_tokens": float(median(token_counts)) if token_counts else float("nan"),
            "cap_rate": mean([1.0 if row.get("cap") else 0.0 for row in group]),
            "no_box_rate": mean([1.0 if row.get("no_box") else 0.0 for row in group]),
            "mean_p0_reflect_vs_stop": mean([row.get("p0_reflect_vs_stop") for row in group]),
            "mean_patch_hook_calls": mean([row.get("patch_hook_calls") for row in group]),
            "mean_add_hook_calls": mean([row.get("add_hook_calls") for row in group]),
        }
        out.append(item)
    return out


def summarize_logits(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric = [
        "reflect_logsum",
        "stop_logsum",
        "reflect_vs_stop",
        "marker_logsum",
        "marker_vs_stop",
        "finalize_logsum",
        "finalize_vs_reflect",
        "wait_logsum",
        "check_logsum",
        "actually_logsum",
        "newline_logsum",
    ]
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("mode", "free")), str(row.get("condition")), int(row.get("position_index", -1)))].append(row)
    out: List[Dict[str, Any]] = []
    for (mode, condition, pos), group in sorted(grouped.items()):
        item: Dict[str, Any] = {"mode": mode, "condition": condition, "position_index": pos, "count": len(group)}
        for name in numeric:
            item[f"mean_{name}"] = mean([row.get(name) for row in group])
        out.append(item)
    return out


def condition_specs_for(experiment: str) -> List[ConditionSpec]:
    if experiment == "D1":
        return [
            ConditionSpec("CC", "CC", "factorial"),
            ConditionSpec("IW", "IW", "factorial"),
            ConditionSpec("CW", "CW", "factorial"),
            ConditionSpec("IC", "IC", "factorial"),
            ConditionSpec("IW_gateoff", "IW", "factorial_intervention", gate_sign=-1),
            ConditionSpec("IC_gateoff", "IC", "factorial_intervention", gate_sign=-1),
            ConditionSpec("CW_gateon", "CW", "factorial_intervention", gate_sign=1),
            ConditionSpec("CC_gateon", "CC", "factorial_intervention", gate_sign=1),
        ]
    if experiment == "D2":
        return [
            ConditionSpec("T", "T", "baseline"),
            ConditionSpec("C", "C", "baseline"),
            ConditionSpec("T_gateoff", "T", "baseline", gate_sign=-1),
            ConditionSpec("T_gateoff_force_wait", "T", "marker_loop", gate_sign=-1, forced_prefix_name="wait"),
            ConditionSpec(
                "T_gateoff_force_wait_gateoff_after_wait",
                "T",
                "marker_loop",
                gate_sign=-1,
                gate_mode="p2_plus",
                forced_prefix_name="wait",
            ),
            ConditionSpec("T_force_neutral_prefix4", "T", "marker_loop", forced_prefix_name="neutral4"),
            ConditionSpec("T_ban_markers_p1_p16", "T", "marker_control", banned_markers_until=15),
            ConditionSpec("C_gateon_ban_markers_p1_p16", "C", "marker_control", gate_sign=1, banned_markers_until=15),
        ]
    if experiment == "D3":
        specs = [
            ConditionSpec("T", "T", "baseline"),
            ConditionSpec("C", "C", "baseline"),
            ConditionSpec("C_gateon_force_wait", "C", "gate_marker", gate_sign=1, forced_prefix_name="wait"),
        ]
        for k in [1, 2, 4, 8]:
            specs.append(ConditionSpec(f"C_restore_{k}_anchors", f"C_restore_{k}_anchors", "anchor_restore", extra={"restore_k": k}))
        specs.extend(
            [
                ConditionSpec(
                    "C_restore_1_anchor_gateon_force_wait",
                    "C_restore_1_anchors",
                    "anchor_gate_marker",
                    gate_sign=1,
                    forced_prefix_name="wait",
                    extra={"restore_k": 1},
                ),
                ConditionSpec(
                    "C_restore_4_anchors_gateon_force_wait",
                    "C_restore_4_anchors",
                    "anchor_gate_marker",
                    gate_sign=1,
                    forced_prefix_name="wait",
                    extra={"restore_k": 4},
                ),
            ]
        )
        return specs
    if experiment == "D6":
        return [
            ConditionSpec("T", "T", "baseline"),
            ConditionSpec("C", "C", "baseline"),
            ConditionSpec("T_gateoff", "T", "baseline", gate_sign=-1),
            ConditionSpec("C_gateon", "C", "baseline", gate_sign=1),
        ]
    raise ValueError(experiment)


def d1_patch_specs(args: argparse.Namespace) -> List[PatchSpec]:
    layers = parse_layer_spec(args.d1_patch_layers, 10_000)
    sites = parse_csv_list(args.d1_patch_sites)
    timings = parse_csv_list(args.d1_patch_timings)
    pairs = [
        ("IW_from_CW", "IW", "CW", "delete"),
        ("IC_from_CC", "IC", "CC", "delete_correct_inconsistent"),
        ("CW_from_IW", "CW", "IW", "induce"),
        ("CC_from_IC", "CC", "IC", "induce_correct"),
    ]
    specs: List[PatchSpec] = []
    for label, base, source, patch_type in pairs:
        for layer in layers:
            for site in sites:
                for timing in timings:
                    specs.append(
                        PatchSpec(
                            condition=f"D1_patch_{label}_L{layer}_{site}_{timing}",
                            experiment="D1",
                            patch_type=patch_type,
                            base_condition=base,
                            source_condition=source,
                            layer_idx=int(layer),
                            site=str(site),
                            timing=str(timing),
                            condition_kind="factorial_patch",
                            sweep="layer_site_timing",
                        )
                    )
    return specs


def d2_patch_specs(args: argparse.Namespace) -> List[PatchSpec]:
    specs: List[PatchSpec] = []
    patch_layer = int(args.intervention_layer)
    for site in parse_csv_list(args.d2_patch_sites):
        specs.extend(
            [
                PatchSpec(
                    condition=f"T_gateoff_force_wait_delete_L{patch_layer}_{site}_p2_p8",
                    experiment="D2",
                    patch_type="delete",
                    base_condition="T_gateoff_force_wait",
                    source_condition="C",
                    layer_idx=patch_layer,
                    site=site,
                    timing="p2_p8",
                    condition_kind="forced_wait_patch",
                    sweep="forced_wait_delete",
                    base_gate_sign=-1,
                    base_forced_prefix_name="wait",
                ),
                PatchSpec(
                    condition=f"T_gateoff_force_wait_patch_C_L{patch_layer}_{site}_p2_p8",
                    experiment="D2",
                    patch_type="patch_C",
                    base_condition="T_gateoff_force_wait",
                    source_condition="C",
                    layer_idx=patch_layer,
                    site=site,
                    timing="p2_p8",
                    condition_kind="forced_wait_patch",
                    sweep="forced_wait_patch_C",
                    base_gate_sign=-1,
                    base_forced_prefix_name="wait",
                ),
                PatchSpec(
                    condition=f"T_force_neutral_prefix4_delete_L{patch_layer}_{site}_p4_p12",
                    experiment="D2",
                    patch_type="delete",
                    base_condition="T_force_neutral_prefix4",
                    source_condition="C",
                    layer_idx=patch_layer,
                    site=site,
                    timing="p4_p12",
                    condition_kind="neutral_prefix_patch",
                    sweep="neutral_delete",
                    base_forced_prefix_name="neutral4",
                ),
                PatchSpec(
                    condition=f"T_ban_markers_patch_T_L{patch_layer}_{site}_p1_p8",
                    experiment="D2",
                    patch_type="patch_T",
                    base_condition="T_ban_markers_p1_p16",
                    source_condition="T",
                    layer_idx=patch_layer,
                    site=site,
                    timing="p1_p8",
                    condition_kind="marker_ban_patch",
                    sweep="ban_marker_repair_state",
                    base_banned_markers_until=15,
                ),
            ]
        )
    return specs


def d3_patch_specs(args: argparse.Namespace) -> List[PatchSpec]:
    specs = [
        PatchSpec(
            condition="C_patch_T_state_L22_post_attn_p1_p8",
            experiment="D3",
            patch_type="patch_T_state",
            base_condition="C",
            source_condition="T",
            layer_idx=22,
            site="post_attn",
            timing="p1_p8",
            condition_kind="state_patch",
            sweep="main",
        )
    ]
    for k in parse_int_list(args.d3_restore_ks):
        specs.append(
            PatchSpec(
                condition=f"C_restore_{k}_anchors_patch_T_state_L22_post_attn_p1_p8",
                experiment="D3",
                patch_type="patch_T_state",
                base_condition=f"C_restore_{k}_anchors",
                source_condition="T",
                layer_idx=22,
                site="post_attn",
                timing="p1_p8",
                condition_kind="anchor_restore_state_patch",
                sweep="anchor_restore",
                extra={"restore_k": int(k)},
            )
        )
    return specs


def d6_patch_specs(args: argparse.Namespace) -> List[PatchSpec]:
    layers = parse_layer_spec(args.d6_patch_layers, 10_000)
    sites = parse_csv_list(args.d6_patch_sites)
    timings = parse_csv_list(args.d6_patch_timings)
    pairs = [
        ("delete", "T", "C", 0, "T_to_C"),
        ("rescue", "T_gateoff", "T", -1, "Tgateoff_to_T"),
        ("induce", "C", "T", 0, "C_to_T"),
        ("cancel", "C_gateon", "C", 1, "Cgateon_to_C"),
    ]
    specs: List[PatchSpec] = []
    for patch_type, base, source, base_gate_sign, label in pairs:
        for layer in layers:
            for site in sites:
                for timing in timings:
                    specs.append(
                        PatchSpec(
                            condition=f"D6_{patch_type}_{label}_L{layer}_{site}_{timing}",
                            experiment="D6",
                            patch_type=patch_type,
                            base_condition=base,
                            source_condition=source,
                            layer_idx=int(layer),
                            site=str(site),
                            timing=str(timing),
                            condition_kind="component_patch",
                            sweep="layer_component_timing",
                            base_gate_sign=int(base_gate_sign),
                        )
                    )
    return specs


def patch_specs_for(args: argparse.Namespace) -> List[PatchSpec]:
    if args.experiment == "D1":
        return d1_patch_specs(args)
    if args.experiment == "D2":
        return d2_patch_specs(args)
    if args.experiment == "D3":
        return d3_patch_specs(args)
    if args.experiment == "D6":
        return d6_patch_specs(args)
    return []


def filtered_activation_entry(
    trajectory: Dict[str, Any],
    *,
    save_layers: Sequence[int],
    save_sites: Sequence[str],
) -> Dict[str, Any]:
    entry = activation_entry(trajectory)
    allowed: Set[str] = {
        f"L{int(layer_idx)}/{site}"
        for layer_idx in save_layers
        for site in save_sites
    }
    entry["activations"] = {
        key: value
        for key, value in (entry.get("activations") or {}).items()
        if key in allowed
    }
    return entry


def full_id_key_for_condition(condition: str) -> str:
    if condition in {"CC", "IW", "CW", "IC", "T", "C"}:
        return condition
    if condition.startswith("T_") or condition.startswith("Tgate") or condition.startswith("Tgateoff"):
        return "T"
    if condition.startswith("T"):
        return "T"
    if condition.startswith("C_gateon") or condition.startswith("Cgateon"):
        return "C"
    if condition.startswith("C_restore_") and "_gateon_" in condition:
        parts = condition.split("_gateon_", 1)[0]
        return parts
    return condition


def forced_prefix_ids(name: str, *, wait_ids: Sequence[int], neutral_ids: Sequence[int]) -> Optional[List[int]]:
    if not name:
        return None
    if name == "wait":
        return [int(wait_ids[0])] if wait_ids else []
    if name == "neutral4":
        return [int(x) for x in neutral_ids[:4]]
    raise ValueError(f"Unsupported forced prefix name: {name}")


def condition_extra(spec: ConditionSpec) -> Dict[str, Any]:
    return dict(spec.extra or {})


def patch_extra(spec: PatchSpec) -> Dict[str, Any]:
    return dict(spec.extra or {})


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    activation_dir = output_dir / "activation_traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_activations:
        activation_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (output_dir / "summary.json").exists():
        print(f"[Resume] Existing summary found, skipping: {output_dir}")
        return
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
    gen_config = build_generation_config(args)
    stage1_stop = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    continuation_stop_strings = parse_csv_list(args.continuation_stop_strings)
    stop_sequences = stop_id_sequences(backend, continuation_stop_strings)

    reflect_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    marker_ids = token_ids_for_first_tokens(tokenizer, MARKER_TEXTS)
    wait_ids = token_ids_for_first_tokens(tokenizer, [" Wait", "Wait", " wait", "wait"])
    check_ids = token_ids_for_first_tokens(tokenizer, [" check", " Check", "check", " verify", " recalculate"])
    actually_ids = token_ids_for_first_tokens(tokenizer, [" Actually", " actually", "Actually"])
    finalize_ids = token_ids_for_first_tokens(tokenizer, [" Therefore", " Thus", " So", " Hence", "</think>", " final", " answer"])
    newline_ids = token_ids_for_first_tokens(tokenizer, ["\n", "\n\n"])
    neutral_ids = backend.encode(" The answer is")[:4]
    token_sets = {
        "reflect": reflect_ids,
        "stop": stop_ids,
        "marker": sorted(set(marker_ids + reflect_ids)),
        "wait": wait_ids,
        "check": check_ids,
        "actually": actually_ids,
        "finalize": finalize_ids,
        "newline": newline_ids,
    }
    tracked_token_ids = sorted(set(reflect_ids + stop_ids + marker_ids + wait_ids + check_ids + actually_ids + finalize_ids + newline_ids))
    banned_marker_ids = sorted(set(marker_ids + reflect_ids + wait_ids + check_ids + actually_ids))

    gate_info = load_gate_direction_cache(Path(args.gate_direction_cache_in), args.intervention_layer, args.intervention_site)
    if gate_info is None:
        raise ValueError(f"No gate direction for L{args.intervention_layer}/{args.intervention_site} in {args.gate_direction_cache_in}")
    gate_dir = tensor_normed(gate_info["direction"])
    gate_scale = float(gate_info["scale"])
    condition_specs = condition_specs_for(args.experiment)
    patch_specs = patch_specs_for(args)
    runtime_layers = list(selected_layers)
    runtime_sites = list(selected_sites)
    if bool(args.runtime_patch_capture):
        runtime_layers = sorted(set(runtime_layers) | {int(spec.layer_idx) for spec in patch_specs})
        runtime_sites = sorted(set(runtime_sites) | {str(spec.site) for spec in patch_specs})

    rows = load_jsonl(args.input_jsonl)
    examples = rows[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    run_metadata = {
        "experiment": args.experiment,
        "model_size_label": args.model_size_label,
        "n_attempted": len(examples),
        "start_idx": int(args.start_idx),
        "max_examples": int(args.max_examples),
    }
    config = {
        **vars(args),
        "layer_path": layer_path,
        "selected_layers": selected_layers,
        "selected_sites": selected_sites,
        "runtime_layers": runtime_layers,
        "runtime_sites": runtime_sites,
        "condition_specs": [asdict(spec) for spec in condition_specs],
        "patch_specs": [asdict(spec) for spec in patch_specs],
        "reflect_token_ids": reflect_ids,
        "stop_token_ids": stop_ids,
        "marker_token_ids": marker_ids,
        "banned_marker_ids": banned_marker_ids,
        "neutral_prefix_ids": neutral_ids,
        "generation": asdict(gen_config),
        "gate_direction": {
            "source": gate_info.get("source"),
            "n_pairs": gate_info.get("n_pairs"),
            "scale": gate_scale,
            "mean_diff_norm": gate_info.get("mean_diff_norm"),
            "layer_idx": int(args.intervention_layer),
            "site": args.intervention_site,
        },
        "run_metadata": run_metadata,
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "run_config.json", config)
    write_json(output_dir / "run_metadata.json", run_metadata)
    write_csv(output_dir / "patch_specs.csv", [asdict(spec) for spec in patch_specs])

    if args.dry_run:
        print("[Dry run] Wrote config only.")
        print(f"- output_dir: {output_dir}")
        print(f"- conditions: {len(condition_specs)}")
        print(f"- patch_specs: {len(patch_specs)}")
        return

    common_run_kwargs = {
        "layers": layers,
        "capture_layer_indices": runtime_layers,
        "capture_sites": runtime_sites,
        "max_new_tokens": int(args.max_continuation_tokens),
        "capture_max_position_index": int(args.capture_max_position_index),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "stop_id_sequences": stop_sequences,
        "token_sets": token_sets,
        "tracked_token_ids": tracked_token_ids,
    }

    behavior_rows: List[Dict[str, Any]] = []
    logit_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc=f"Reflection mechanism {args.experiment}", dynamic_ncols=True)
    for local_idx, ex in enumerate(iterator):
        global_idx = int(args.start_idx) + local_idx
        ex_id = row_id(ex, global_idx)
        question = row_question(ex)
        correct_answer = row_answer(ex)
        wrong_answer = str(ex.get("wrong_answer") or "").strip()
        if not question or correct_answer is None or not wrong_answer or str(correct_answer).strip() == wrong_answer:
            skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "missing_or_bad_fields"})
            continue
        try:
            seed_everything(int(args.seed) + global_idx)
            restore_ks = parse_int_list(args.d3_restore_ks)
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
            full_ids, derived_meta = build_full_id_map(
                tokenizer,
                backend,
                prepared,
                str(correct_answer),
                wrong_answer,
                restore_ks=restore_ks,
            )
            missing = [spec.full_id_key for spec in condition_specs if spec.full_id_key not in full_ids]
            missing.extend(
                [
                    full_id_key_for_condition(spec.base_condition)
                    for spec in patch_specs
                    if full_id_key_for_condition(spec.base_condition) not in full_ids
                ]
            )
            missing.extend(
                [
                    full_id_key_for_condition(spec.source_condition)
                    for spec in patch_specs
                    if full_id_key_for_condition(spec.source_condition) not in full_ids
                ]
            )
            missing = sorted(set(missing))
            if missing:
                skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "missing_derived_conditions", "conditions": ",".join(missing)})
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
                    **derived_meta,
                },
                "runs": {"free": {}},
            }
            reference_runs: Dict[str, Dict[str, Any]] = {}

            def record(condition: str, trajectory: Dict[str, Any], extra: Dict[str, Any]) -> None:
                analysis = run_analysis_payload(
                    tokenizer=tokenizer,
                    trajectory=trajectory,
                    correct_answer=str(correct_answer),
                    wrong_answer=wrong_answer,
                )
                row = {
                    "example_id": ex_id,
                    "problem_id": ex_id,
                    "global_idx": global_idx,
                    "mode": "free",
                    "condition": condition,
                    "experiment": args.experiment,
                    "model": args.model_size_label,
                    "seed": int(args.seed) + global_idx,
                    "prompt_hash": sha1_text(prepared["prompt"]),
                    "transcript_hash": sha1_text(trajectory.get("full_text", "")),
                    "clean_answer": str(correct_answer),
                    "wrong_answer": wrong_answer,
                    "clean_box_answer": prepared["clean_box_answer"],
                    "coherent_replacement_count": prepared["coherent_replacement_count"],
                    **extra,
                    **debug_counts(trajectory),
                    **analysis,
                }
                behavior_rows.append(row)
                if args.save_logits:
                    for logit_row in trajectory.get("logit_rows") or []:
                        enriched = {
                            "example_id": ex_id,
                            "global_idx": global_idx,
                            "mode": "free",
                            "condition": condition,
                            "experiment": args.experiment,
                            **logit_row,
                        }
                        if "marker_logsum" in enriched and "stop_logsum" in enriched:
                            enriched["marker_vs_stop"] = enriched["marker_logsum"] - enriched["stop_logsum"]
                        if "finalize_logsum" in enriched and "reflect_logsum" in enriched:
                            enriched["finalize_vs_reflect"] = enriched["finalize_logsum"] - enriched["reflect_logsum"]
                        logit_rows.append(enriched)
                if args.save_activations:
                    activation_payload["runs"]["free"][condition] = filtered_activation_entry(
                        trajectory,
                        save_layers=selected_layers,
                        save_sites=selected_sites,
                    )

            for spec_idx, spec in enumerate(condition_specs):
                seed_everything(int(args.seed) + 100000 + global_idx + spec_idx)
                trajectory = run_patchable_trajectory(
                    model,
                    tokenizer,
                    full_ids=full_ids[spec.full_id_key],
                    add_intervention=gate_add(
                        gate_sign=spec.gate_sign,
                        gate_mode=spec.gate_mode,
                        gate_dir=gate_dir,
                        gate_scale=gate_scale,
                        gate_alpha=float(args.gate_alpha),
                        layer_idx=int(args.intervention_layer),
                        site=str(args.intervention_site),
                    ),
                    patch_plan=None,
                    forced_prefix_ids=forced_prefix_ids(
                        spec.forced_prefix_name,
                        wait_ids=wait_ids,
                        neutral_ids=neutral_ids,
                    ),
                    banned_token_ids=banned_marker_ids if spec.banned_markers_until is not None else [],
                    ban_until_position=spec.banned_markers_until,
                    forced_continuation_ids=None,
                    **common_run_kwargs,
                )
                reference_runs[spec.condition] = trajectory
                if spec.full_id_key not in reference_runs:
                    reference_runs[spec.full_id_key] = trajectory
                record(
                    spec.condition,
                    trajectory,
                    {
                        "condition_kind": spec.condition_kind,
                        "intervention_type": "gate_add" if spec.gate_sign else "",
                        "gate_sign": spec.gate_sign,
                        "gate_mode": spec.gate_mode if spec.gate_sign else "",
                        "forced_prefix_name": spec.forced_prefix_name,
                        "banned_markers_until": spec.banned_markers_until if spec.banned_markers_until is not None else "",
                        "patch_type": "",
                        "patch_layer": "",
                        "patch_site": "",
                        "patch_timing": "",
                        "alpha": float(args.gate_alpha) if spec.gate_sign else "",
                        **condition_extra(spec),
                    },
                )

            source_replays: Dict[Tuple[str, str, Tuple[Any, ...]], Dict[str, Any]] = {}
            for spec_idx, spec in enumerate(patch_specs):
                if spec.base_condition in reference_runs:
                    base_generated = reference_runs[spec.base_condition]["generated_token_ids"]
                else:
                    skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "missing_base_run", "condition": spec.condition})
                    continue
                source_key = (
                    spec.source_condition,
                    spec.base_condition,
                    (spec.source_gate_sign, spec.source_gate_mode, spec.source_forced_prefix_name, spec.source_banned_markers_until),
                )
                if source_key not in source_replays:
                    source_replays[source_key] = run_patchable_trajectory(
                        model,
                        tokenizer,
                        full_ids=full_ids[full_id_key_for_condition(spec.source_condition)],
                        add_intervention=gate_add(
                            gate_sign=spec.source_gate_sign,
                            gate_mode=spec.source_gate_mode,
                            gate_dir=gate_dir,
                            gate_scale=gate_scale,
                            gate_alpha=float(args.gate_alpha),
                            layer_idx=int(args.intervention_layer),
                            site=str(args.intervention_site),
                        ),
                        patch_plan=None,
                        forced_prefix_ids=forced_prefix_ids(
                            spec.source_forced_prefix_name,
                            wait_ids=wait_ids,
                            neutral_ids=neutral_ids,
                        ),
                        banned_token_ids=banned_marker_ids if spec.source_banned_markers_until is not None else [],
                        ban_until_position=spec.source_banned_markers_until,
                        forced_continuation_ids=base_generated,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        **{k: v for k, v in common_run_kwargs.items() if k not in {"do_sample", "temperature", "top_p"}},
                    )
                source_run = source_replays[source_key]
                vectors = patch_vectors_from_source(
                    source_run,
                    layer_idx=int(spec.layer_idx),
                    site=str(spec.site),
                    max_position=int(args.capture_max_position_index),
                )
                if not vectors:
                    skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "missing_patch_vectors", "condition": spec.condition})
                    continue
                patch_plan = PatchPlan(
                    layer_idx=int(spec.layer_idx),
                    site=str(spec.site),
                    vectors_by_position=vectors,
                    timing=str(spec.timing),
                    reflection_keywords=tuple(DEFAULT_REFLECTION_KEYWORDS),
                )
                trajectory = run_patchable_trajectory(
                    model,
                    tokenizer,
                    full_ids=full_ids[full_id_key_for_condition(spec.base_condition)],
                    add_intervention=gate_add(
                        gate_sign=spec.base_gate_sign,
                        gate_mode=spec.base_gate_mode,
                        gate_dir=gate_dir,
                        gate_scale=gate_scale,
                        gate_alpha=float(args.gate_alpha),
                        layer_idx=int(args.intervention_layer),
                        site=str(args.intervention_site),
                    ),
                    patch_plan=patch_plan,
                    forced_prefix_ids=forced_prefix_ids(
                        spec.base_forced_prefix_name,
                        wait_ids=wait_ids,
                        neutral_ids=neutral_ids,
                    ),
                    banned_token_ids=banned_marker_ids if spec.base_banned_markers_until is not None else [],
                    ban_until_position=spec.base_banned_markers_until,
                    forced_continuation_ids=None,
                    **common_run_kwargs,
                )
                record(
                    spec.condition,
                    trajectory,
                    {
                        "condition_kind": spec.condition_kind,
                        "intervention_type": "patch",
                        "patch_type": spec.patch_type,
                        "base_condition": spec.base_condition,
                        "source_condition": spec.source_condition,
                        "patch_layer": int(spec.layer_idx),
                        "patch_site": spec.site,
                        "patch_timing": spec.timing,
                        "sweep": spec.sweep,
                        "gate_sign": spec.base_gate_sign,
                        "gate_mode": spec.base_gate_mode if spec.base_gate_sign else "",
                        "forced_prefix_name": spec.base_forced_prefix_name,
                        "banned_markers_until": spec.base_banned_markers_until if spec.base_banned_markers_until is not None else "",
                        "alpha": float(args.gate_alpha) if spec.base_gate_sign else "",
                        **patch_extra(spec),
                    },
                )

            if args.save_activations:
                torch.save(activation_payload, activation_dir / f"{global_idx:05d}_{safe_name(ex_id)}.pt")

            if args.print_every > 0 and (local_idx + 1) % int(args.print_every) == 0:
                iterator.set_postfix({"rows": len(behavior_rows), "skipped": len(skipped_rows)})
                dump_jsonl(output_dir / "behavior_rows.partial.jsonl", behavior_rows)
                dump_jsonl(output_dir / "per_example.partial.jsonl", behavior_rows)
                dump_jsonl(output_dir / "logit_rows.partial.jsonl", logit_rows)
                dump_jsonl(output_dir / "skipped_rows.partial.jsonl", skipped_rows)
                write_csv(output_dir / "behavior_summary.partial.csv", summarize_behavior(behavior_rows))
                write_csv(output_dir / f"{args.experiment.lower()}_behavior_summary.partial.csv", summarize_behavior(behavior_rows))
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
    if args.save_per_example:
        dump_jsonl(output_dir / "per_example.jsonl", behavior_rows)
    dump_jsonl(output_dir / "logit_rows.jsonl", logit_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    behavior_summary = summarize_behavior(behavior_rows)
    write_csv(output_dir / "behavior_summary.csv", behavior_summary)
    write_csv(output_dir / f"{args.experiment.lower()}_behavior_summary.csv", behavior_summary)
    if args.experiment == "D1":
        write_csv(output_dir / "factorial_behavior_summary.csv", behavior_summary)
        write_csv(output_dir / "factorial_patch_summary.csv", [row for row in behavior_summary if row.get("condition_kind") == "factorial_patch"])
    elif args.experiment == "D2":
        write_csv(output_dir / "token_hidden_behavior_summary.csv", behavior_summary)
        write_csv(output_dir / "forced_wait_delete_patch_summary.csv", [row for row in behavior_summary if str(row.get("condition")).startswith("T_gateoff_force_wait")])
        write_csv(output_dir / "neutral_prefix_delete_patch_summary.csv", [row for row in behavior_summary if "neutral_prefix" in str(row.get("condition"))])
    elif args.experiment == "D3":
        write_csv(output_dir / "anchor_restore_behavior_summary.csv", behavior_summary)
        write_csv(output_dir / "anchor_restore_interaction_table.csv", behavior_summary)
    elif args.experiment == "D6":
        write_csv(output_dir / "component_patch_behavior_summary.csv", behavior_summary)
        write_csv(output_dir / "layer_component_timing_heatmap.csv", behavior_summary)
    write_csv(output_dir / "behavior_summary_basic.csv", summarize_basic_behavior(behavior_rows))
    write_csv(output_dir / "logit_summary.csv", summarize_logits(logit_rows))
    reason_counts = Counter(str(row.get("reason")) for row in skipped_rows)
    summary = {
        **run_metadata,
        "behavior_rows": len(behavior_rows),
        "logit_rows": len(logit_rows),
        "skipped_rows": len(skipped_rows),
        "n_skipped": len(skipped_rows),
        "skip_reasons": dict(reason_counts),
        "usable_examples": len({row["example_id"] for row in behavior_rows}),
        "n_usable": len({row["example_id"] for row in behavior_rows}),
        "condition_count": len(condition_specs),
        "patch_spec_count": len(patch_specs),
        "summary_rows": len(behavior_summary),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_metadata.json", {**run_metadata, **summary})
    print(f"[Done] Reflection mechanism {args.experiment} finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_examples: {summary['usable_examples']}")
    print(f"- behavior_rows: {len(behavior_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
