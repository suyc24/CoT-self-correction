#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import answers_match
from cot_research.generation import create_backend
from cot_research.hidden_trajectory import (
    AddIntervention,
    run_hidden_trajectory,
    tensor_normed,
)
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig
from cot_research.stateful_tampering import (
    analyze_continuation_text,
    find_last_boxed_token_span,
    token_ids_for_first_tokens,
)


DEFAULT_REFLECTION_KEYWORDS = [
    "wait",
    "actually",
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

WAIT_FIRST_TOKENS = {" Wait", " wait", "Wait", "wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hidden trajectory movie for forced-box reflection gate: free-running plus "
            "teacher-forced replay under gate/logit/random controls."
        )
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
    parser.add_argument("--max_continuation_tokens", type=int, default=128)
    parser.add_argument(
        "--capture_max_position_index",
        type=int,
        default=64,
        help="Capture p0..pN activations; generation/replay can still continue beyond N with interventions.",
    )
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coherent_window_chars", type=int, default=1200)
    parser.add_argument("--layers", default="18-24")
    parser.add_argument("--sites", default="post_attn,block_output")
    parser.add_argument("--intervention_layer", type=int, default=22)
    parser.add_argument("--intervention_site", default="post_attn")
    parser.add_argument("--gate_alpha", type=float, default=0.5)
    parser.add_argument("--logit_alpha", type=float, default=0.5)
    parser.add_argument("--random_alpha", type=float, default=0.5)
    parser.add_argument(
        "--gate_direction_cache_in",
        default="",
        help="Torch cache from run_reflection_gate_full_trajectory.py containing the gate direction.",
    )
    parser.add_argument(
        "--gate_direction_cache_out",
        default="",
        help="If no input cache is supplied, save the locally built intervention direction here.",
    )
    parser.add_argument(
        "--direction_max_examples",
        type=int,
        default=40,
        help="Fallback direction-build size when --gate_direction_cache_in is not supplied.",
    )
    parser.add_argument("--direction_start_idx", type=int, default=0)
    parser.add_argument(
        "--free_conditions",
        default="T,C,T_gateoff,T_logit,T_random,C_gateon",
        help="Comma-separated free-running conditions.",
    )
    parser.add_argument(
        "--replay_conditions",
        default="T,C,T_gateoff,T_logit,T_random,C_gateon",
        help="Comma-separated teacher-forced replay conditions using the free-running T continuation.",
    )
    parser.add_argument("--continuation_stop_strings", default="")
    parser.add_argument("--save_raw_activations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    return parser.parse_args()


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    out: List[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            values = range(start, end + step, step)
        else:
            values = [int(item)]
        for idx in values:
            if idx < 0:
                idx = n_layers + idx
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index out of range: {idx}")
            if idx not in out:
                out.append(idx)
    if not out:
        raise ValueError("No layers selected.")
    return out


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


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
            load_in_half=bool(args.load_in_half),
            use_fast_tokenizer=bool(args.use_fast_tokenizer),
            use_safetensors=bool(args.use_safetensors),
            local_files_only=bool(args.local_files_only),
            attn_implementation=args.attn_implementation,
        )
    )


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        stage1_stop_string=args.stage1_stop_string,
        max_stage1_tokens=int(args.max_stage1_tokens),
        max_new_tokens=int(args.max_continuation_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        enable_thinking=True,
        capture_step_scores=False,
    )


def stop_id_sequences(backend, texts: Iterable[str]) -> List[List[int]]:
    out: List[List[int]] = []
    for text in texts:
        text = str(text)
        if not text:
            continue
        ids = backend.encode(text)
        if ids:
            out.append([int(x) for x in ids])
    return out


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


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:120] or "example"


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
    if int(window_chars) > 0:
        prefix_head = prefix_text[: -int(window_chars)]
        prefix_tail = prefix_text[-int(window_chars) :]
    else:
        prefix_head = ""
        prefix_tail = prefix_text
    targets: List[str] = []
    for target in [str(clean_answer), str(correct_answer)]:
        target = target.strip()
        if target and target != str(wrong_answer) and target not in targets:
            targets.append(target)
    replacement_count = 0
    for target in targets:
        replacement_count += prefix_tail.count(target)
        prefix_tail = prefix_tail.replace(target, str(wrong_answer))
    text = prefix_head + prefix_tail + tamper_force_text
    return tokenizer.encode(text, add_special_tokens=False), {
        "replacement_count": int(replacement_count),
        "replacement_targets": targets,
    }


def prepare_forced_box_example(
    *,
    backend,
    tokenizer,
    gen_config: GenerationConfig,
    example: Dict[str, Any],
    example_id: str,
    correct_answer: str,
    wrong_answer: str,
    stop_strings: Optional[List[str]],
    allow_nonmatching_clean: bool,
    coherent_window_chars: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    prompt = backend.build_prompt(row_question(example), gen_config)
    prompt_ids = backend.encode(prompt)
    clean_config = gen_config if stop_strings else replace(gen_config, max_new_tokens=int(gen_config.max_stage1_tokens))
    clean = backend.generate(prompt, clean_config, stop_strings=stop_strings)
    clean_gen_ids = [int(x) for x in clean.token_ids]
    span = find_last_boxed_token_span(tokenizer, clean_gen_ids)
    if span is None:
        return None, {"example_id": example_id, "reason": "no_boxed_span"}
    if not allow_nonmatching_clean and not answers_match(span.answer_text, correct_answer):
        return None, {
            "example_id": example_id,
            "reason": "clean_boxed_answer_not_correct",
            "clean_boxed_answer": span.answer_text,
            "correct_answer": correct_answer,
        }

    prefix_ids = prompt_ids + clean_gen_ids[: span.answer_start]
    clean_force_ids = clean_gen_ids[span.answer_start : span.box_end]
    wrong_answer_ids = backend.encode(wrong_answer)
    tamper_force_ids = wrong_answer_ids + clean_gen_ids[span.answer_end : span.box_end]
    tamper_full_ids = prefix_ids + tamper_force_ids
    clean_full_ids = prefix_ids + clean_force_ids
    tamper_force_text = tokenizer.decode(tamper_force_ids, skip_special_tokens=False)
    coherent_ids, coherent_meta = make_answer_token_coherent_ids(
        tokenizer,
        prefix_ids=prefix_ids,
        tamper_force_text=tamper_force_text,
        clean_answer=span.answer_text,
        correct_answer=str(correct_answer),
        wrong_answer=str(wrong_answer),
        window_chars=int(coherent_window_chars),
    )
    return (
        {
            "example_id": example_id,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "clean_generated_ids": clean_gen_ids,
            "clean_full_ids": clean_full_ids,
            "tamper_full_ids": tamper_full_ids,
            "coherent_full_ids": coherent_ids,
            "clean_box_answer": span.answer_text,
            "wrong_answer_token_len": len(wrong_answer_ids),
            "clean_force_token_len": len(clean_force_ids),
            "tamper_force_token_len": len(tamper_force_ids),
            "coherent_replacement_count": int(coherent_meta["replacement_count"]),
            "coherent_replacement_targets": coherent_meta["replacement_targets"],
            "box_span": span.to_dict(),
        },
        None,
    )


def load_gate_direction_cache(path: Path, layer_idx: int, site: str) -> Optional[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    for item in payload.get("directions", []):
        if (
            str(item.get("direction_type")) == "gate"
            and int(item.get("layer_idx")) == int(layer_idx)
            and str(item.get("site")) == str(site)
        ):
            direction = tensor_normed(item["direction"])
            scale = float(item.get("scale") or item.get("mean_diff_norm") or 1.0)
            if not math.isfinite(scale) or scale == 0:
                scale = 1.0
            return {
                "direction": direction,
                "scale": scale,
                "n_pairs": int(item.get("n_pairs") or 0),
                "mean_diff_norm": item.get("mean_diff_norm"),
                "source": str(path),
            }
    return None


@torch.no_grad()
def build_gate_direction(
    *,
    args: argparse.Namespace,
    backend,
    layers: Sequence[torch.nn.Module],
    gen_config: GenerationConfig,
) -> Dict[str, Any]:
    from cot_research.hidden_trajectory import run_hidden_trajectory

    rows = load_jsonl(args.input_jsonl)
    examples = rows[int(args.direction_start_idx) : int(args.direction_start_idx) + int(args.direction_max_examples)]
    tokenizer = backend.tokenizer
    stop_strings = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    stop_sequences: List[List[int]] = []
    diffs: List[torch.Tensor] = []
    diff_norms: List[float] = []
    skipped: List[Dict[str, Any]] = []
    for local_idx, row in enumerate(tqdm(examples, desc="Build fallback gate direction", dynamic_ncols=True)):
        ex_id = row_id(row, int(args.direction_start_idx) + local_idx)
        question = row_question(row)
        correct = row_answer(row)
        wrong = str(row.get("wrong_answer") or "").strip()
        if not question or correct is None or not wrong:
            skipped.append({"example_id": ex_id, "reason": "missing_fields"})
            continue
        seed_everything(int(args.seed) + local_idx)
        prepared, skip = prepare_forced_box_example(
            backend=backend,
            tokenizer=tokenizer,
            gen_config=gen_config,
            example=row,
            example_id=ex_id,
            correct_answer=str(correct),
            wrong_answer=wrong,
            stop_strings=stop_strings,
            allow_nonmatching_clean=bool(args.allow_nonmatching_clean),
            coherent_window_chars=int(args.coherent_window_chars),
        )
        if prepared is None:
            skipped.append(skip or {"example_id": ex_id, "reason": "prepare_failed"})
            continue
        common = {
            "layers": layers,
            "capture_layer_indices": [int(args.intervention_layer)],
            "capture_sites": [str(args.intervention_site)],
            "max_new_tokens": 0,
            "capture_max_position_index": 0,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "stop_id_sequences": stop_sequences,
        }
        t_run = run_hidden_trajectory(model=backend.model, tokenizer=tokenizer, full_ids=prepared["tamper_full_ids"], **common)
        c_run = run_hidden_trajectory(model=backend.model, tokenizer=tokenizer, full_ids=prepared["coherent_full_ids"], **common)
        key = f"L{int(args.intervention_layer)}/{args.intervention_site}"
        t_vec = t_run["activations"].get(key)
        c_vec = c_run["activations"].get(key)
        if t_vec is None or c_vec is None:
            skipped.append({"example_id": ex_id, "reason": "missing_capture"})
            continue
        diff = t_vec[0].float() - c_vec[0].float()
        diffs.append(tensor_normed(diff))
        diff_norms.append(float(diff.norm().item()))
    if not diffs:
        raise ValueError(f"Could not build fallback gate direction; skipped={skipped[:5]}")
    direction = tensor_normed(torch.stack(diffs, dim=0).mean(dim=0))
    scale = mean(diff_norms)
    if not math.isfinite(scale) or scale == 0:
        scale = 1.0
    info = {
        "direction": direction,
        "scale": float(scale),
        "n_pairs": len(diffs),
        "mean_diff_norm": mean(diff_norms),
        "source": "built_in_run_reflection_hidden_trajectory_movie",
        "skipped": skipped,
    }
    if args.gate_direction_cache_out:
        out_path = Path(args.gate_direction_cache_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "directions": [
                    {
                        "direction_type": "gate",
                        "layer_idx": int(args.intervention_layer),
                        "site": str(args.intervention_site),
                        "direction": direction.detach().float().cpu(),
                        "scale": float(scale),
                        "n_pairs": len(diffs),
                        "mean_diff_norm": mean(diff_norms),
                    }
                ],
                "skipped": skipped,
            },
            out_path,
        )
    return info


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


def marker_positions(tokenizer, token_ids: Sequence[int], keywords: Sequence[str]) -> Dict[str, Any]:
    lower_keywords = [str(kw).lower() for kw in keywords if str(kw)]
    for idx in range(len(token_ids)):
        text = tokenizer.decode(list(token_ids[: idx + 1]), skip_special_tokens=False).lower()
        if any(kw in text for kw in lower_keywords):
            marker_pos = idx + 1
            return {
                "reflection_marker_generated_index": int(idx),
                "reflection_marker_position_index": int(marker_pos),
                "pre_reflection_marker_position_index": max(int(marker_pos) - 1, 0),
                "post_reflection_marker_position_index": int(marker_pos) + 1,
            }
    return {
        "reflection_marker_generated_index": None,
        "reflection_marker_position_index": None,
        "pre_reflection_marker_position_index": None,
        "post_reflection_marker_position_index": None,
    }


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
    return {
        "first_generated_token_id": first_id,
        "first_generated_token_text": first_text,
        "first_wait": bool(first_text in WAIT_FIRST_TOKENS),
        "generated_tokens": len(generated_ids),
        "hit_max_new_tokens": bool(trajectory.get("hit_max_new_tokens")),
        "stop_reason": trajectory.get("stop_reason"),
        "continuation_text": continuation_text,
        "semantic_repair": bool(analysis.get("full_text_final_matches_correct")),
        "p0_reflect_vs_stop": p0_logits.get("reflect_vs_stop"),
        "p0_wait_logsum": p0_logits.get("wait_logsum"),
        "p0_stop_logsum": p0_logits.get("stop_logsum"),
        **marker_positions(tokenizer, generated_ids, DEFAULT_REFLECTION_KEYWORDS),
        **analysis,
    }


def summarize_behavior(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("mode")), str(row.get("condition")))].append(row)
    out: List[Dict[str, Any]] = []
    for (mode, condition), group in sorted(grouped.items()):
        token_counter = Counter(str(row.get("first_generated_token_text") or "") for row in group)
        out.append(
            {
                "mode": mode,
                "condition": condition,
                "count": len(group),
                "first_wait_rate": mean([1.0 if row.get("first_wait") else 0.0 for row in group]),
                "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in group]),
                "semantic_repair_rate": mean([1.0 if row.get("semantic_repair") else 0.0 for row in group]),
                "mean_generated_tokens": mean([row.get("generated_tokens") for row in group]),
                "cap_rate": mean([1.0 if row.get("hit_max_new_tokens") else 0.0 for row in group]),
                "mean_p0_reflect_vs_stop": mean([row.get("p0_reflect_vs_stop") for row in group]),
                "top_first_tokens": json.dumps(token_counter.most_common(8), ensure_ascii=False),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    activation_dir = output_dir / "activation_traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))

    backend = build_backend(args)
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("This script requires an HF backend.")
    decoder_layers, layer_path = get_decoder_layers(model)
    layers = list(decoder_layers)
    selected_layers = parse_layer_spec(args.layers, len(layers))
    selected_sites = parse_csv_list(args.sites)
    unknown_sites = [site for site in selected_sites if site not in {"block_input", "post_attn", "block_output"}]
    if unknown_sites:
        raise ValueError(f"Unsupported sites: {unknown_sites}")
    free_conditions = parse_csv_list(args.free_conditions)
    replay_conditions = parse_csv_list(args.replay_conditions)
    supported_conditions = {"T", "C", "T_gateoff", "T_logit", "T_random", "C_gateon"}
    for cond in free_conditions + replay_conditions:
        if cond not in supported_conditions:
            raise ValueError(f"Unsupported condition {cond}; supported={sorted(supported_conditions)}")

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

    gate_info: Optional[Dict[str, Any]] = None
    if args.gate_direction_cache_in:
        gate_info = load_gate_direction_cache(Path(args.gate_direction_cache_in), args.intervention_layer, args.intervention_site)
        if gate_info is None:
            raise ValueError(
                f"No gate direction for L{args.intervention_layer}/{args.intervention_site} in {args.gate_direction_cache_in}"
            )
    else:
        gate_info = build_gate_direction(args=args, backend=backend, layers=layers, gen_config=gen_config)

    gate_dir = tensor_normed(gate_info["direction"])
    gate_scale = float(gate_info["scale"])
    logit_dir = get_logit_direction(model, reflect_ids, stop_ids)
    if logit_dir is None or logit_dir.numel() != gate_dir.numel():
        raise ValueError("Could not build a logit direction compatible with the gate direction.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed) + 515151)
    random_dir = tensor_normed(torch.randn(gate_dir.numel(), generator=generator))

    rows = load_jsonl(args.input_jsonl)
    examples = rows[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "layer_path": layer_path,
            "selected_layers": selected_layers,
            "selected_sites": selected_sites,
            "free_conditions": free_conditions,
            "replay_conditions": replay_conditions,
            "reflect_token_ids": reflect_ids,
            "stop_token_ids": stop_ids,
            "wait_token_ids": wait_ids,
            "check_token_ids": check_ids,
            "actually_token_ids": actually_ids,
            "finalize_token_ids": finalize_ids,
            "newline_token_ids": newline_ids,
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

    def condition_full_ids(prepared: Dict[str, Any], condition: str) -> List[int]:
        if condition.startswith("T"):
            return list(prepared["tamper_full_ids"])
        if condition.startswith("C"):
            return list(prepared["coherent_full_ids"])
        raise ValueError(condition)

    def condition_intervention(condition: str) -> Optional[AddIntervention]:
        if condition == "T_gateoff":
            add = -gate_dir * (float(args.gate_alpha) * gate_scale)
        elif condition == "T_logit":
            add = -logit_dir * (float(args.logit_alpha) * gate_scale)
        elif condition == "T_random":
            add = -random_dir * (float(args.random_alpha) * gate_scale)
        elif condition == "C_gateon":
            add = gate_dir * (float(args.gate_alpha) * gate_scale)
        else:
            return None
        return AddIntervention(
            layer_idx=int(args.intervention_layer),
            site=str(args.intervention_site),
            add_vector=add.detach().float().cpu(),
        )

    iterator = tqdm(examples, desc="Hidden trajectory movie", dynamic_ncols=True)
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
                "runs": {"free": {}, "replay": {}},
            }

            t_free_tokens: Optional[List[int]] = None
            for condition in free_conditions:
                seed_everything(int(args.seed) + 100000 + global_idx)
                trajectory = run_hidden_trajectory(
                    model,
                    tokenizer,
                    full_ids=condition_full_ids(prepared, condition),
                    layers=layers,
                    capture_layer_indices=selected_layers,
                    capture_sites=selected_sites,
                    max_new_tokens=int(args.max_continuation_tokens),
                    capture_max_position_index=int(args.capture_max_position_index),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    stop_id_sequences=stop_sequences,
                    forced_continuation_ids=None,
                    intervention=condition_intervention(condition),
                    token_sets=token_sets,
                    tracked_token_ids=tracked_token_ids,
                )
                if condition == "T":
                    t_free_tokens = [int(x) for x in trajectory["generated_token_ids"]]
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
                    activation_payload["runs"]["free"][condition] = {
                        "generated_token_ids": trajectory["generated_token_ids"],
                        "generated_text": trajectory["generated_text"],
                        "full_token_ids": trajectory["full_token_ids"],
                        "stop_reason": trajectory["stop_reason"],
                        "hit_max_new_tokens": trajectory["hit_max_new_tokens"],
                        "position_records": trajectory["position_records"],
                        "debug_rows": trajectory["debug_rows"],
                        "activations": trajectory["activations"],
                    }
                del trajectory

            if t_free_tokens is None:
                raise ValueError("T free-running condition did not run; cannot build replay.")

            for condition in replay_conditions:
                seed_everything(int(args.seed) + 200000 + global_idx)
                trajectory = run_hidden_trajectory(
                    model,
                    tokenizer,
                    full_ids=condition_full_ids(prepared, condition),
                    layers=layers,
                    capture_layer_indices=selected_layers,
                    capture_sites=selected_sites,
                    max_new_tokens=min(int(args.max_continuation_tokens), len(t_free_tokens)),
                    capture_max_position_index=int(args.capture_max_position_index),
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    stop_id_sequences=stop_sequences,
                    forced_continuation_ids=t_free_tokens,
                    intervention=condition_intervention(condition),
                    token_sets=token_sets,
                    tracked_token_ids=tracked_token_ids,
                )
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
                        "mode": "replay",
                        "condition": condition,
                        "correct_answer": str(correct_answer),
                        "wrong_answer": wrong_answer,
                        "clean_box_answer": prepared["clean_box_answer"],
                        "coherent_replacement_count": prepared["coherent_replacement_count"],
                        **analysis,
                    }
                )
                for row in trajectory.get("logit_rows") or []:
                    logit_rows.append({"example_id": ex_id, "global_idx": global_idx, "mode": "replay", "condition": condition, **row})
                if args.save_raw_activations:
                    activation_payload["runs"]["replay"][condition] = {
                        "generated_token_ids": trajectory["generated_token_ids"],
                        "generated_text": trajectory["generated_text"],
                        "full_token_ids": trajectory["full_token_ids"],
                        "stop_reason": trajectory["stop_reason"],
                        "hit_max_new_tokens": trajectory["hit_max_new_tokens"],
                        "position_records": trajectory["position_records"],
                        "debug_rows": trajectory["debug_rows"],
                        "activations": trajectory["activations"],
                    }
                del trajectory

            if args.save_raw_activations:
                torch.save(activation_payload, activation_dir / f"{global_idx:05d}_{safe_name(ex_id)}.pt")

            if args.print_every > 0 and (local_idx + 1) % int(args.print_every) == 0:
                iterator.set_postfix({"rows": len(behavior_rows), "skipped": len(skipped_rows)})
                dump_jsonl(output_dir / "behavior_rows.partial.jsonl", behavior_rows)
                dump_jsonl(output_dir / "logit_rows.partial.jsonl", logit_rows)
                dump_jsonl(output_dir / "skipped_rows.partial.jsonl", skipped_rows)
                write_csv(output_dir / "behavior_summary.partial.csv", summarize_behavior(behavior_rows))
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
    reason_counts = Counter(str(row.get("reason")) for row in skipped_rows)
    write_json(
        output_dir / "summary.json",
        {
            "behavior_rows": len(behavior_rows),
            "logit_rows": len(logit_rows),
            "skipped_rows": len(skipped_rows),
            "skipped_reasons": dict(reason_counts),
            "usable_examples": len({row["example_id"] for row in behavior_rows}),
            "summary_rows": summary_rows,
        },
    )
    print("[Done] Hidden trajectory movie collection finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_examples: {len({row['example_id'] for row in behavior_rows})}")
    print(f"- behavior_rows: {len(behavior_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
