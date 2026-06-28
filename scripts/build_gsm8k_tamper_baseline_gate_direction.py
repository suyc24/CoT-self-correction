#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = ROOT_DIR / "evaluation"
for path in (EVALUATION_DIR, ROOT_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.runtime_utils import seed_everything

FULL_TRAJ_PATH = SCRIPT_DIR / "run_reflection_gate_full_trajectory.py"
FULL_TRAJ_SPEC = importlib.util.spec_from_file_location("_reflection_gate_full_trajectory", FULL_TRAJ_PATH)
if FULL_TRAJ_SPEC is None or FULL_TRAJ_SPEC.loader is None:
    raise ImportError(f"Could not load {FULL_TRAJ_PATH}")
FULL_TRAJ = importlib.util.module_from_spec(FULL_TRAJ_SPEC)
FULL_TRAJ_SPEC.loader.exec_module(FULL_TRAJ)

build_backend = FULL_TRAJ.build_backend
build_generation_config = FULL_TRAJ.build_generation_config
find_last_boxed_token_span = FULL_TRAJ.find_last_boxed_token_span
mean = FULL_TRAJ.mean
parse_int_list = FULL_TRAJ.parse_int_list
parse_sites = FULL_TRAJ.parse_sites
prefill_before_final_full_ids = FULL_TRAJ.prefill_before_final_full_ids
run_final_forward_boundary = FULL_TRAJ.run_final_forward_boundary
save_direction_cache = FULL_TRAJ.save_direction_cache
tensor_normed = FULL_TRAJ.tensor_normed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tamper-minus-baseline reflection gate direction from standard GSM8K CoT."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--cache_out", required=True)
    parser.add_argument("--max_examples", type=int, default=100)
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
    parser.add_argument("--max_stage1_tokens", type=int, default=1)
    parser.add_argument("--max_continuation_tokens", type=int, default=1)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--layers", default="22")
    parser.add_argument("--sites", default="post_attn")
    return parser.parse_args()


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


def row_id(row: Dict[str, Any], fallback: int) -> str:
    for key in ["id", "example_id", "unique_id", "idx"]:
        if row.get(key) is not None:
            return str(row[key])
    return str(fallback)


def row_question(row: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt"]:
        if row.get(key):
            return str(row[key]).strip()
    return ""


def row_answer(row: Dict[str, Any]) -> str:
    for key in ["correct_answer", "answer", "target"]:
        if row.get(key) is not None:
            value = str(row[key]).strip()
            if "####" in value:
                value = value.split("####", 1)[1].strip()
            return value.replace(",", "")
    return ""


def row_wrong_answer(row: Dict[str, Any]) -> str:
    if row.get("wrong_answer") is not None:
        return str(row["wrong_answer"]).strip()
    answer = row_answer(row)
    try:
        value = float(answer)
        wrong = int(value + 1) if value.is_integer() else value + 1.0
        return str(wrong)
    except ValueError:
        nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
        if nums:
            return str(int(float(nums[-1])) + 1)
    return ""


def row_cot(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("gt_cot"):
        return str(metadata["gt_cot"]).strip()
    for key in ["cot", "reasoning", "solution"]:
        if row.get(key):
            return str(row[key]).strip()
    answer = str(row.get("answer") or "")
    if "####" in answer:
        return answer.split("####", 1)[0].strip()
    return ""


def clean_continuation(prompt: str, cot: str, answer: str) -> str:
    final = f"Therefore, the final answer is \\boxed{{{answer}}}."
    if prompt.rstrip().endswith("<think>"):
        return f"{cot}\n</think>\n\n{final}"
    return f"<think>\n{cot}\n</think>\n\n{final}"


@torch.no_grad()
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
    gen_config = build_generation_config(args)

    rows = load_jsonl(args.input_jsonl)
    examples = rows[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    records: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for local_idx, row in enumerate(tqdm(examples, desc="GSM8K tamper-baseline captures", dynamic_ncols=True)):
        global_idx = int(args.start_idx) + local_idx
        ex_id = row_id(row, global_idx)
        question = row_question(row)
        answer = row_answer(row)
        wrong = row_wrong_answer(row)
        cot = row_cot(row)
        if not question or not answer or not wrong or not cot:
            skipped.append({"example_id": ex_id, "reason": "missing_fields"})
            continue
        try:
            prompt = backend.build_prompt(question, gen_config)
            prompt_ids = backend.encode(prompt)
            clean_ids = backend.encode(clean_continuation(prompt, cot, answer))
            span = find_last_boxed_token_span(backend.tokenizer, clean_ids)
            if span is None:
                skipped.append({"example_id": ex_id, "reason": "no_boxed_span_in_standard_cot"})
                continue
            wrong_ids = backend.encode(wrong)
            prefix_ids = prompt_ids + clean_ids[: span.answer_start]
            clean_full_ids = prompt_ids + clean_ids[: span.box_end]
            tamper_full_ids = prefix_ids + wrong_ids + clean_ids[span.answer_end : span.box_end]

            captures_by_condition: Dict[str, Dict[Tuple[str, int], torch.Tensor]] = {}
            for condition, full_ids in [("tamper", tamper_full_ids), ("baseline", clean_full_ids)]:
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
                    "global_idx": global_idx,
                    "correct_answer": answer,
                    "wrong_answer": wrong,
                    "captures": captures_by_condition,
                }
            )
        except Exception as exc:
            skipped.append({"example_id": ex_id, "reason": "exception", "error": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    directions: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    direction_rows: List[Dict[str, Any]] = []
    for layer_idx in selected_layers:
        for site in selected_sites:
            diffs: List[torch.Tensor] = []
            diff_norms: List[float] = []
            resid_stds: List[float] = []
            for rec in records:
                tamper = rec["captures"]["tamper"].get((site, layer_idx))
                baseline = rec["captures"]["baseline"].get((site, layer_idx))
                if tamper is None or baseline is None:
                    continue
                diff = tamper.float() - baseline.float()
                diffs.append(tensor_normed(diff))
                diff_norms.append(float(diff.norm().item()))
                resid_stds.append(float(tamper.float().std().item()))
                resid_stds.append(float(baseline.float().std().item()))
            if not diffs:
                continue
            direction = tensor_normed(torch.stack(diffs, dim=0).mean(dim=0))
            scale = mean(diff_norms)
            directions[("gate", layer_idx, site)] = {
                "direction": direction,
                "scale": float(scale),
                "n_pairs": len(diffs),
                "mean_diff_norm": mean(diff_norms),
                "mean_resid_std": mean(resid_stds),
                "direction_baseline_condition": "baseline",
                "direction_contrast": "tamper_minus_baseline",
                "source": "gsm8k_standard_cot",
            }
            direction_rows.append(
                {
                    "direction_type": "gate",
                    "layer_idx": int(layer_idx),
                    "site": site,
                    "n_pairs": len(diffs),
                    "scale": float(scale),
                    "mean_diff_norm": mean(diff_norms),
                    "mean_resid_std": mean(resid_stds),
                    "direction_norm": float(direction.norm().item()),
                    "direction_baseline_condition": "baseline",
                    "direction_contrast": "tamper_minus_baseline",
                    "source": "gsm8k_standard_cot",
                }
            )

    save_direction_cache(Path(args.cache_out), directions, direction_rows)
    dump_jsonl(output_dir / "used_rows.jsonl", [{k: v for k, v in row.items() if k != "captures"} for row in records])
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped)
    dump_jsonl(output_dir / "direction_rows.jsonl", direction_rows)
    write_csv(output_dir / "direction_summary.csv", direction_rows)
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "layer_path": layer_path,
            "selected_layers": selected_layers,
            "selected_sites": selected_sites,
            "n_records": len(records),
            "n_skipped": len(skipped),
        },
    )
    print(json.dumps({"records": len(records), "skipped": len(skipped), "directions": len(directions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
