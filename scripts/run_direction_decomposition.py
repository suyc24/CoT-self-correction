#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.patch_trajectory import ScheduledAddIntervention, run_patchable_trajectory
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import token_ids_for_first_tokens
from run_reflection_hidden_trajectory_movie import (
    DEFAULT_REFLECT_FIRST_TEXTS,
    DEFAULT_REFLECTION_KEYWORDS,
    DEFAULT_STOP_FIRST_TEXTS,
    build_backend,
    build_generation_config,
    parse_csv_list,
    prepare_forced_box_example,
    row_answer,
    row_id,
    row_question,
    stop_id_sequences,
    tensor_normed,
)
from run_reflection_mechanism_experiment import MARKER_TEXTS, run_analysis_payload, summarize_behavior


FINALIZE_TEXTS = ["</think>", " final", " answer", " Therefore", " therefore", " Hence", " hence", " \\boxed", " done"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D5 repair/commit/marker/silent-correction direction decomposition.")
    parser.add_argument("--input_jsonl", default=str(ROOT_DIR / "evaluation/data/self_correction_ablation/test_questions.jsonl"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--model_size_label", default="qwen3_4b")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=90)
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
    parser.add_argument("--max_direction_tokens", type=int, default=16)
    parser.add_argument("--max_continuation_tokens", type=int, default=96)
    parser.add_argument("--capture_max_position_index", type=int, default=8)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coherent_window_chars", type=int, default=1200)
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--site", default="post_attn")
    parser.add_argument("--positions", default="1,2,4,8")
    parser.add_argument("--alphas", default="0.25,0.5")
    parser.add_argument("--intervention_mode", default="decode_only")
    parser.add_argument("--continuation_stop_strings", default="")
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
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for item in parse_csv_list(text):
        if "-" in item:
            left, right = item.split("-", 1)
            out.extend(range(int(left), int(right) + 1))
        else:
            out.append(int(item))
    return out


def parse_float_list(text: str) -> List[float]:
    return [float(x) for x in parse_csv_list(text)]


def vector_at(run: Mapping[str, Any], *, layer: int, site: str, positions: Sequence[int]) -> Optional[torch.Tensor]:
    key = f"L{int(layer)}/{site}"
    tensor = (run.get("activations") or {}).get(key)
    if tensor is None:
        return None
    vecs: List[torch.Tensor] = []
    for pos in positions:
        if 0 <= int(pos) < int(tensor.shape[0]):
            vecs.append(tensor[int(pos)].detach().float().cpu())
    if not vecs:
        return None
    return torch.stack(vecs, dim=0).mean(dim=0)


def add_vec(agg: Dict[str, List[torch.Tensor]], name: str, vec: Optional[torch.Tensor]) -> None:
    if vec is not None:
        agg.setdefault(name, []).append(vec.detach().float().cpu())


def mean_vec(vecs: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    if not vecs:
        return None
    return torch.stack([v.float().cpu() for v in vecs], dim=0).mean(dim=0)


def direction_from_diff(pos: Sequence[torch.Tensor], neg: Sequence[torch.Tensor]) -> Tuple[Optional[torch.Tensor], float]:
    p = mean_vec(pos)
    n = mean_vec(neg)
    if p is None or n is None:
        return None, float("nan")
    diff = p - n
    return tensor_normed(diff), float(diff.norm().item())


def orthogonalize(vec: torch.Tensor, basis: Sequence[torch.Tensor]) -> torch.Tensor:
    out = vec.detach().float().cpu().clone()
    for b in basis:
        unit = tensor_normed(b.detach().float().cpu())
        out = out - torch.dot(out, unit) * unit
    return tensor_normed(out)


def logit_direction(model: torch.nn.Module, marker_ids: Sequence[int], finalize_ids: Sequence[int]) -> Optional[torch.Tensor]:
    weight = getattr(getattr(model, "lm_head", None), "weight", None)
    if weight is None or not marker_ids or not finalize_ids:
        return None
    marker = weight[torch.tensor(sorted(set(marker_ids)), device=weight.device)].detach().float().mean(dim=0).cpu()
    finalize = weight[torch.tensor(sorted(set(finalize_ids)), device=weight.device)].detach().float().mean(dim=0).cpu()
    return tensor_normed(marker - finalize)


def condition_run(
    *,
    model,
    tokenizer,
    layers,
    full_ids: Sequence[int],
    add_intervention: Optional[ScheduledAddIntervention],
    forced_prefix_ids: Optional[Sequence[int]],
    max_new_tokens: int,
    capture_max_position_index: int,
    capture_layer: int,
    capture_site: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    stop_sequences: Sequence[Sequence[int]],
    token_sets: Mapping[str, Sequence[int]],
    tracked_token_ids: Sequence[int],
) -> Dict[str, Any]:
    return run_patchable_trajectory(
        model,
        tokenizer,
        full_ids=full_ids,
        layers=layers,
        capture_layer_indices=[int(capture_layer)],
        capture_sites=[str(capture_site)],
        max_new_tokens=int(max_new_tokens),
        capture_max_position_index=int(capture_max_position_index),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        stop_id_sequences=stop_sequences,
        forced_prefix_ids=forced_prefix_ids,
        add_intervention=add_intervention,
        token_sets=dict(token_sets),
        tracked_token_ids=tracked_token_ids,
    )


def compact_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], max_rows: int = 20) -> str:
    rows = list(rows)[:max_rows]
    if not rows:
        return "(empty)"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, summary: Mapping[str, Any], cosine_rows: Sequence[Dict[str, Any]], intervention_summary: Sequence[Dict[str, Any]]) -> None:
    core = [r for r in intervention_summary if r.get("condition_kind") == "direction_intervention"]
    lines = [
        "# D5 Direction Decomposition",
        "",
        "## 1. 实验目的",
        "把 d_gate 拆成 inconsistency alarm、marker、commitment、length、silent-correction 等方向，并测试正交化后是否仍能干预行为。",
        "",
        "## 2. 运行规模",
        f"- n_attempted: {summary.get('n_attempted')}",
        f"- n_usable: {summary.get('n_usable')}",
        f"- n_skipped: {summary.get('n_skipped')}",
        f"- skip_reasons: `{json.dumps(summary.get('skip_reasons', {}), ensure_ascii=False)}`",
        "",
        "## 3. Direction Cosine Matrix",
        compact_table(cosine_rows, ["direction_a", "direction_b", "cosine"], 24),
        "",
        "## 4. Intervention Summary",
        compact_table(core, ["condition", "direction_name", "base_condition", "sign", "alpha", "count", "surface_reflection_rate", "semantic_repair_rate", "mean_generated_tokens"], 24),
        "",
        "## 5. 关键发现",
        "- `direction_cosine_matrix.csv` 显示 d_gate 与 marker/commit/silent/length 的重叠程度。",
        "- `orthogonalized_direction_summary.csv` 检查从 d_gate 中去掉 marker 或 commit 分量后还剩多少行为效应。",
        "- d_truth 在本脚本中用 `CW - CC` 近似，因此更像 wrong-final-answer contrast，而不是完整 truth factorization。",
        "",
        "## 6. 不确定点",
        "- 方向由同一批样本的平均 hidden contrast 构造，linear basis 不是唯一分解。",
        "- explicit/silent/functional repair 仍是启发式文本标签。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    backend = build_backend(args)
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("HF backend is required.")
    decoder_layers, layer_path = get_decoder_layers(model)
    layers = list(decoder_layers)
    positions = parse_int_list(args.positions)
    alphas = parse_float_list(args.alphas)
    gen_config = build_generation_config(args)
    stage1_stop = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    stop_sequences = stop_id_sequences(backend, parse_csv_list(args.continuation_stop_strings))
    reflect_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)
    marker_ids = token_ids_for_first_tokens(tokenizer, MARKER_TEXTS)
    finalize_ids = token_ids_for_first_tokens(tokenizer, FINALIZE_TEXTS)
    wait_ids = token_ids_for_first_tokens(tokenizer, [" Wait", "Wait", " wait", "wait"])
    neutral_ids = backend.encode(" The answer is")[:4]
    token_sets = {
        "reflect": reflect_ids,
        "stop": stop_ids,
        "marker": sorted(set(marker_ids + reflect_ids)),
        "finalize": finalize_ids,
        "wait": wait_ids,
    }
    tracked = sorted(set(reflect_ids + stop_ids + marker_ids + finalize_ids + wait_ids))
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "layer_path": layer_path,
            "positions": positions,
            "alphas": alphas,
            "reflect_token_ids": reflect_ids,
            "marker_token_ids": marker_ids,
            "finalize_token_ids": finalize_ids,
        },
    )

    examples = load_jsonl(args.input_jsonl)[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    prepared_examples: List[Dict[str, Any]] = []
    base_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []
    vecs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    length_records: List[Tuple[int, torch.Tensor]] = []

    for local_idx, ex in enumerate(tqdm(examples, desc="D5 first pass directions", dynamic_ncols=True)):
        global_idx = int(args.start_idx) + local_idx
        ex_id = row_id(ex, global_idx)
        correct = row_answer(ex)
        wrong = str(ex.get("wrong_answer") or "").strip()
        if not row_question(ex) or correct is None or not wrong or str(correct).strip() == wrong:
            skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "missing_or_bad_fields"})
            continue
        try:
            seed_everything(int(args.seed) + global_idx)
            prepared, skip = prepare_forced_box_example(
                backend=backend,
                tokenizer=tokenizer,
                gen_config=gen_config,
                example=ex,
                example_id=ex_id,
                correct_answer=str(correct),
                wrong_answer=wrong,
                stop_strings=stage1_stop,
                allow_nonmatching_clean=bool(args.allow_nonmatching_clean),
                coherent_window_chars=int(args.coherent_window_chars),
            )
            if prepared is None:
                skipped_rows.append({"global_idx": global_idx, **(skip or {"example_id": ex_id, "reason": "prepare_failed"})})
                continue
            full_ids = {
                "T": [int(x) for x in prepared["tamper_full_ids"]],
                "C": [int(x) for x in prepared["coherent_full_ids"]],
                "CC": [int(x) for x in prepared["clean_full_ids"]],
            }
            runs = {
                "T": condition_run(
                    model=model,
                    tokenizer=tokenizer,
                    layers=layers,
                    full_ids=full_ids["T"],
                    add_intervention=None,
                    forced_prefix_ids=None,
                    max_new_tokens=int(args.max_direction_tokens),
                    capture_max_position_index=int(args.capture_max_position_index),
                    capture_layer=int(args.layer),
                    capture_site=str(args.site),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    stop_sequences=stop_sequences,
                    token_sets=token_sets,
                    tracked_token_ids=tracked,
                ),
                "C": condition_run(
                    model=model,
                    tokenizer=tokenizer,
                    layers=layers,
                    full_ids=full_ids["C"],
                    add_intervention=None,
                    forced_prefix_ids=None,
                    max_new_tokens=int(args.max_direction_tokens),
                    capture_max_position_index=int(args.capture_max_position_index),
                    capture_layer=int(args.layer),
                    capture_site=str(args.site),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    stop_sequences=stop_sequences,
                    token_sets=token_sets,
                    tracked_token_ids=tracked,
                ),
                "CC": condition_run(
                    model=model,
                    tokenizer=tokenizer,
                    layers=layers,
                    full_ids=full_ids["CC"],
                    add_intervention=None,
                    forced_prefix_ids=None,
                    max_new_tokens=int(args.max_direction_tokens),
                    capture_max_position_index=int(args.capture_max_position_index),
                    capture_layer=int(args.layer),
                    capture_site=str(args.site),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    stop_sequences=stop_sequences,
                    token_sets=token_sets,
                    tracked_token_ids=tracked,
                ),
                "T_force_wait": condition_run(
                    model=model,
                    tokenizer=tokenizer,
                    layers=layers,
                    full_ids=full_ids["T"],
                    add_intervention=None,
                    forced_prefix_ids=wait_ids[:1],
                    max_new_tokens=int(args.max_direction_tokens),
                    capture_max_position_index=int(args.capture_max_position_index),
                    capture_layer=int(args.layer),
                    capture_site=str(args.site),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    stop_sequences=stop_sequences,
                    token_sets=token_sets,
                    tracked_token_ids=tracked,
                ),
                "T_neutral4": condition_run(
                    model=model,
                    tokenizer=tokenizer,
                    layers=layers,
                    full_ids=full_ids["T"],
                    add_intervention=None,
                    forced_prefix_ids=neutral_ids[:4],
                    max_new_tokens=int(args.max_direction_tokens),
                    capture_max_position_index=int(args.capture_max_position_index),
                    capture_layer=int(args.layer),
                    capture_site=str(args.site),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    stop_sequences=stop_sequences,
                    token_sets=token_sets,
                    tracked_token_ids=tracked,
                ),
            }
            record = {"example_id": ex_id, "global_idx": global_idx, "correct_answer": str(correct), "wrong_answer": wrong, "full_ids": full_ids}
            prepared_examples.append(record)
            for name, run in runs.items():
                payload = run_analysis_payload(tokenizer=tokenizer, trajectory=run, correct_answer=str(correct), wrong_answer=wrong)
                base_rows.append({"example_id": ex_id, "global_idx": global_idx, "condition": name, "condition_kind": "direction_base", **payload})
                vec = vector_at(run, layer=int(args.layer), site=str(args.site), positions=positions)
                add_vec(vecs, name, vec)
                if vec is not None:
                    length_records.append((int(payload.get("generated_tokens") or 0), vec))
                    if name == "T" and bool(payload.get("surface_reflection")):
                        add_vec(vecs, "T_explicit_repair", vec)
            if args.print_every > 0 and (local_idx + 1) % int(args.print_every) == 0:
                dump_jsonl(output_dir / "base_rows.partial.jsonl", base_rows)
        except Exception as exc:
            skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "exception", "error_type": type(exc).__name__, "error": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    directions: Dict[str, Dict[str, Any]] = {}
    specs = {
        "d_inconsistency": ("T", "C"),
        "d_truth_wrong_minus_correct": ("C", "CC"),
        "d_marker": ("T_force_wait", "T_neutral4"),
        "d_commit": ("C", "T"),
        "d_silent_correction": ("T_neutral4", "T_force_wait"),
        "d_explicit_repair": ("T_explicit_repair", "C"),
    }
    for name, (pos_name, neg_name) in specs.items():
        direction, scale = direction_from_diff(vecs.get(pos_name, []), vecs.get(neg_name, []))
        if direction is not None:
            directions[name] = {"direction": direction, "scale": scale, "pos": pos_name, "neg": neg_name}
    if length_records:
        token_counts = [tokens for tokens, _vec in length_records]
        med = median(token_counts)
        long_vecs = [vec for tokens, vec in length_records if tokens >= med]
        short_vecs = [vec for tokens, vec in length_records if tokens < med]
        direction, scale = direction_from_diff(long_vecs, short_vecs)
        if direction is not None:
            directions["d_length"] = {"direction": direction, "scale": scale, "pos": "long", "neg": "short", "median_tokens": med}
    logit_dir = logit_direction(model, marker_ids, finalize_ids)
    if logit_dir is not None and "d_inconsistency" in directions and logit_dir.numel() == directions["d_inconsistency"]["direction"].numel():
        directions["d_logit_marker"] = {"direction": logit_dir, "scale": directions["d_inconsistency"]["scale"], "pos": "marker_unembed", "neg": "finalize_unembed"}
    if "d_inconsistency" in directions and "d_marker" in directions:
        directions["d_gate_orth_marker"] = {
            "direction": orthogonalize(directions["d_inconsistency"]["direction"], [directions["d_marker"]["direction"]]),
            "scale": directions["d_inconsistency"]["scale"],
            "pos": "d_inconsistency",
            "neg": "d_marker",
        }
    if "d_inconsistency" in directions and "d_commit" in directions:
        directions["d_gate_orth_commit"] = {
            "direction": orthogonalize(directions["d_inconsistency"]["direction"], [directions["d_commit"]["direction"]]),
            "scale": directions["d_inconsistency"]["scale"],
            "pos": "d_inconsistency",
            "neg": "d_commit",
        }
    if all(name in directions for name in ["d_inconsistency", "d_marker", "d_commit", "d_length"]):
        directions["d_gate_orth_marker_commit_length"] = {
            "direction": orthogonalize(
                directions["d_inconsistency"]["direction"],
                [directions["d_marker"]["direction"], directions["d_commit"]["direction"], directions["d_length"]["direction"]],
            ),
            "scale": directions["d_inconsistency"]["scale"],
            "pos": "d_inconsistency",
            "neg": "marker_commit_length",
        }

    cosine_rows: List[Dict[str, Any]] = []
    for a, av in sorted(directions.items()):
        for b, bv in sorted(directions.items()):
            cosine_rows.append({"direction_a": a, "direction_b": b, "cosine": float(torch.dot(av["direction"], bv["direction"]).item())})
    direction_rows = [
        {
            "direction_name": name,
            "scale": value.get("scale"),
            "pos": value.get("pos"),
            "neg": value.get("neg"),
            "norm": float(value["direction"].norm().item()),
        }
        for name, value in sorted(directions.items())
    ]

    intervention_rows: List[Dict[str, Any]] = []
    intervention_names = [
        name
        for name in [
            "d_inconsistency",
            "d_marker",
            "d_commit",
            "d_silent_correction",
            "d_gate_orth_marker",
            "d_gate_orth_commit",
            "d_gate_orth_marker_commit_length",
        ]
        if name in directions
    ]
    for ex in tqdm(prepared_examples, desc="D5 direction interventions", dynamic_ncols=True):
        for direction_name in intervention_names:
            direction = directions[direction_name]["direction"]
            scale = float(directions[direction_name].get("scale") or 1.0)
            if not math.isfinite(scale) or scale == 0:
                scale = 1.0
            for alpha in alphas:
                for base_condition, sign in [("T", -1), ("C", 1)]:
                    add = direction * (float(sign) * float(alpha) * scale)
                    intervention = ScheduledAddIntervention(
                        layer_idx=int(args.layer),
                        site=str(args.site),
                        add_vector=add.detach().float().cpu(),
                        mode=str(args.intervention_mode),
                    )
                    run = condition_run(
                        model=model,
                        tokenizer=tokenizer,
                        layers=layers,
                        full_ids=ex["full_ids"][base_condition],
                        add_intervention=intervention,
                        forced_prefix_ids=None,
                        max_new_tokens=int(args.max_continuation_tokens),
                        capture_max_position_index=-1,
                        capture_layer=int(args.layer),
                        capture_site=str(args.site),
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        stop_sequences=stop_sequences,
                        token_sets=token_sets,
                        tracked_token_ids=tracked,
                    )
                    payload = run_analysis_payload(tokenizer=tokenizer, trajectory=run, correct_answer=ex["correct_answer"], wrong_answer=ex["wrong_answer"])
                    intervention_rows.append(
                        {
                            "example_id": ex["example_id"],
                            "global_idx": ex["global_idx"],
                            "condition": f"{base_condition}_{'add' if sign > 0 else 'subtract'}_{direction_name}_a{alpha}",
                            "condition_kind": "direction_intervention",
                            "base_condition": base_condition,
                            "direction_name": direction_name,
                            "sign": sign,
                            "alpha": float(alpha),
                            **payload,
                        }
                    )

    all_behavior_rows = base_rows + intervention_rows
    behavior_summary = summarize_behavior(
        [
            {
                "experiment": "D5",
                "condition_kind": row.get("condition_kind", ""),
                "condition": row.get("condition", ""),
                "patch_type": "",
                "patch_layer": "",
                "patch_timing": "",
                **row,
            }
            for row in all_behavior_rows
        ]
    )
    for row in behavior_summary:
        parts = str(row.get("condition", "")).split("_")
        for direction_name in directions:
            if direction_name in str(row.get("condition", "")):
                row["direction_name"] = direction_name
        row["base_condition"] = next((x for x in ["T", "C"] if str(row.get("condition", "")).startswith(x + "_")), row.get("condition", ""))
        row["sign"] = -1 if "_subtract_" in str(row.get("condition", "")) else (1 if "_add_" in str(row.get("condition", "")) else "")
        for alpha in alphas:
            if f"a{alpha}" in str(row.get("condition", "")):
                row["alpha"] = float(alpha)

    dump_jsonl(output_dir / "base_rows.jsonl", base_rows)
    dump_jsonl(output_dir / "direction_intervention_rows.jsonl", intervention_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    write_csv(output_dir / "direction_rows.csv", direction_rows)
    write_csv(output_dir / "direction_cosine_matrix.csv", cosine_rows)
    write_csv(output_dir / "direction_intervention_summary.csv", behavior_summary)
    write_csv(output_dir / "orthogonalized_direction_summary.csv", [row for row in behavior_summary if "orth" in str(row.get("condition"))])
    write_csv(output_dir / "direction_hidden_effects.csv", direction_rows)
    reason_counts = Counter(str(row.get("reason")) for row in skipped_rows)
    summary = {
        "experiment": "D5",
        "n_attempted": len(examples),
        "n_usable": len(prepared_examples),
        "n_skipped": len(skipped_rows),
        "skip_reasons": dict(reason_counts),
        "base_rows": len(base_rows),
        "intervention_rows": len(intervention_rows),
        "directions": len(directions),
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, cosine_rows, behavior_summary)
    print("[Done] D5 direction decomposition finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_examples: {len(prepared_examples)}")
    print(f"- intervention_rows: {len(intervention_rows)}")


if __name__ == "__main__":
    main()
