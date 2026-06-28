#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.generation import create_backend
from cot_research.head_intervention import resolve_head_targets
from cot_research.io_utils import load_jsonl, write_csv, write_json
from cot_research.ov_circuit_analysis import extract_head_ov_components
from cot_research.repetition_analysis import RepetitionThresholds, analyze_repetition
from cot_research.repetition_causal_analysis import (
    candidate_stats_from_logits,
    choose_loop_escape,
    find_first_divergence,
    forward_prefix_with_capture,
    generate_with_head_schedule,
)
from cot_research.runtime_utils import parse_parallel_gpu_ids
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Sweep L0H3 scale on already-known repetitive seeds and track loop/escape logit trends."
    )
    parser.add_argument(
        "--repeat_examples_jsonl",
        type=str,
        default=str(root_dir / "outputs" / "repetition" / "all_repetition_cases.jsonl"),
        help="Seed pool of previously detected repetitive CoT examples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "repetition_scale_logit_sweep_qwen3_1p7b"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--scale_values", type=str, default="1.2,1.5,2.0,4.0")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_cases", type=int, default=-1)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument(
        "--keep_worker_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep worker jsonl/json files under output_dir.",
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    return parser.parse_args()


def parse_scale_values(text: str) -> List[float]:
    values: List[float] = []
    seen = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value <= 0:
            raise ValueError(f"Scale must be positive, got {value}.")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("No scale values parsed.")
    return values


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        enable_thinking=args.enable_thinking,
        capture_step_scores=False,
    )


def load_seed_rows(path: Path, max_cases: int) -> List[Dict[str, Any]]:
    rows = load_jsonl(str(path))
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        question = str(row.get("problem") or row.get("question") or "").strip()
        if not question:
            continue
        example_id = str(row.get("example_id") or row.get("id") or "")
        if not example_id:
            example_id = f"seed_{len(out)}"
        if example_id in seen:
            continue
        seen.add(example_id)
        out.append(
            {
                "example_id": example_id,
                "source": row.get("source"),
                "question": question,
                "problem": question,
                "reference_solution": row.get("reference_solution"),
                "correct_answer": row.get("correct_answer"),
                "gold_answer": row.get("gold_answer"),
                "source_input_jsonl": row.get("source_input_jsonl"),
            }
        )
        if max_cases > 0 and len(out) >= max_cases:
            break
    return out


def step_map(trajectory: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(item["step_idx"]): item for item in trajectory.get("step_records") or []}


def build_stats_lookup(rows: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(item["token_id"]): dict(item) for item in rows}


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def judge_repetition(continuation: str, token_ids: Sequence[int]) -> Dict[str, Any]:
    return analyze_repetition(
        continuation,
        token_ids=token_ids,
        thresholds=RepetitionThresholds(),
    )


def choose_anchor_scale(scale_runs: Sequence[Tuple[float, Dict[str, Any]]]) -> Tuple[float, Dict[str, Any]]:
    for scale, traj in scale_runs:
        if not bool(dict(traj.get("repetition_detection") or {}).get("matched")):
            return float(scale), traj
    return float(scale_runs[-1][0]), scale_runs[-1][1]


def analyze_one_row(
    row: Dict[str, Any],
    *,
    backend,
    generation_config: GenerationConfig,
    attn_module: torch.nn.Module,
    target,
    scale_values: Sequence[float],
    top_k: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    question = str(row.get("question") or row.get("problem") or "").strip()
    if not question:
        return None, {"example_id": str(row.get("example_id") or "unknown"), "reason": "missing_question"}

    prompt_prefix = backend.build_prompt(question, generation_config)
    baseline = generate_with_head_schedule(
        backend,
        prompt_prefix,
        generation_config,
        attn_module=attn_module,
        target=target,
        default_scale=1.0,
        top_k=top_k,
    )
    baseline_repetitive = bool(dict(baseline.get("repetition_detection") or {}).get("matched"))
    if not baseline_repetitive:
        return None, {"example_id": str(row.get("example_id") or "unknown"), "reason": "baseline_not_repetitive"}

    scale_runs: List[Tuple[float, Dict[str, Any]]] = []
    for scale in scale_values:
        scale_runs.append(
            (
                float(scale),
                generate_with_head_schedule(
                    backend,
                    prompt_prefix,
                    generation_config,
                    attn_module=attn_module,
                    target=target,
                    default_scale=float(scale),
                    top_k=top_k,
                ),
            )
        )

    anchor_scale, anchor_traj = choose_anchor_scale(scale_runs)
    divergence_step = find_first_divergence(baseline["generated_token_ids"], anchor_traj["generated_token_ids"])
    if divergence_step is None:
        divergence_step = min(len(baseline["generated_token_ids"]), len(anchor_traj["generated_token_ids"])) - 1
        divergence_step = max(int(divergence_step), 0)

    baseline_steps = step_map(baseline)
    anchor_steps = step_map(anchor_traj)
    baseline_step = dict(baseline_steps[divergence_step])
    anchor_step = dict(anchor_steps[divergence_step])
    loop_escape = choose_loop_escape(
        baseline_repetitive=True,
        other_repetitive=bool(dict(anchor_traj.get("repetition_detection") or {}).get("matched")),
        baseline_token_id=int(baseline_step["chosen_token_id"]),
        other_token_id=int(anchor_step["chosen_token_id"]),
    )
    loop_token_id = int(loop_escape["loop_token_id"])
    escape_token_id = int(loop_escape["escape_token_id"])
    divergence_prefix_token_ids = list(baseline["prompt_token_ids"]) + [int(x) for x in baseline["generated_token_ids"][:divergence_step]]

    per_scale_rows: List[Dict[str, Any]] = []
    baseline_row: Optional[Dict[str, Any]] = None
    for scale_label, trajectory in [(1.0, baseline)] + [(float(scale), traj) for scale, traj in scale_runs]:
        logits, _debug = forward_prefix_with_capture(
            backend,
            divergence_prefix_token_ids,
            attn_module=attn_module,
            target=target,
            default_scale=float(scale_label),
        )
        candidate_rows = candidate_stats_from_logits(logits, backend.tokenizer, [loop_token_id, escape_token_id])
        candidate_lookup = build_stats_lookup(candidate_rows)
        row_out = {
            "example_id": str(row.get("example_id")),
            "source": row.get("source"),
            "anchor_scale": float(anchor_scale),
            "scale": float(scale_label),
            "divergence_step": int(divergence_step),
            "loop_token_id": int(loop_token_id),
            "loop_token_text": token_text(backend.tokenizer, loop_token_id),
            "escape_token_id": int(escape_token_id),
            "escape_token_text": token_text(backend.tokenizer, escape_token_id),
            "trajectory_repetitive": bool(dict(trajectory.get("repetition_detection") or {}).get("matched")),
            "generated_tokens": int(len(trajectory.get("generated_token_ids") or [])),
            "divergence_actual_token_id": int((step_map(trajectory).get(divergence_step) or {}).get("chosen_token_id", -1)),
            "divergence_actual_token_text": str((step_map(trajectory).get(divergence_step) or {}).get("chosen_token_text", "")),
            "loop_logit": float(candidate_lookup[loop_token_id]["logit"]),
            "loop_prob": float(candidate_lookup[loop_token_id]["prob"]),
            "escape_logit": float(candidate_lookup[escape_token_id]["logit"]),
            "escape_prob": float(candidate_lookup[escape_token_id]["prob"]),
            "escape_minus_loop_logit": float(candidate_lookup[escape_token_id]["logit"] - candidate_lookup[loop_token_id]["logit"]),
            "escape_minus_loop_prob": float(candidate_lookup[escape_token_id]["prob"] - candidate_lookup[loop_token_id]["prob"]),
            "top_k": json.dumps(
                generate_top_k_from_logits(logits, backend, top_k),
                ensure_ascii=False,
            ),
        }
        per_scale_rows.append(row_out)
        if float(scale_label) == 1.0:
            baseline_row = dict(row_out)

    if baseline_row is None:
        return None, {"example_id": str(row.get("example_id") or "unknown"), "reason": "missing_baseline_row"}

    for item in per_scale_rows:
        item["loop_logit_delta_vs_baseline"] = float(item["loop_logit"] - baseline_row["loop_logit"])
        item["escape_logit_delta_vs_baseline"] = float(item["escape_logit"] - baseline_row["escape_logit"])
        item["loop_prob_delta_vs_baseline"] = float(item["loop_prob"] - baseline_row["loop_prob"])
        item["escape_prob_delta_vs_baseline"] = float(item["escape_prob"] - baseline_row["escape_prob"])
        item["loop_logit_down_vs_baseline"] = bool(item["loop_logit"] < baseline_row["loop_logit"])
        item["escape_logit_up_vs_baseline"] = bool(item["escape_logit"] > baseline_row["escape_logit"])
        item["both_effect_vs_baseline"] = bool(item["loop_logit_down_vs_baseline"] and item["escape_logit_up_vs_baseline"])

    scale_lookup = {float(item["scale"]): item for item in per_scale_rows}
    return {
        "example_id": str(row.get("example_id")),
        "source": row.get("source"),
        "question": question,
        "anchor_scale": float(anchor_scale),
        "divergence_step": int(divergence_step),
        "loop_token_id": int(loop_token_id),
        "loop_token_text": token_text(backend.tokenizer, loop_token_id),
        "escape_token_id": int(escape_token_id),
        "escape_token_text": token_text(backend.tokenizer, escape_token_id),
        "baseline_generated_tokens": int(len(baseline.get("generated_token_ids") or [])),
        "baseline_repetitive": True,
        "scale_rows": per_scale_rows,
        "scales_that_escape": [float(scale) for scale, traj in scale_runs if not bool(dict(traj.get("repetition_detection") or {}).get("matched"))],
        "scale_repetition_map": {str(scale): bool(dict(traj.get("repetition_detection") or {}).get("matched")) for scale, traj in scale_runs},
        "max_scale_both_effect": max((float(item["scale"]) for item in per_scale_rows if float(item["scale"]) != 1.0 and item["both_effect_vs_baseline"]), default=None),
        "baseline_row": baseline_row,
        "scale_lookup": scale_lookup,
    }, None


def generate_top_k_from_logits(logits: torch.Tensor, backend, top_k: int) -> List[Dict[str, Any]]:
    probs = torch.softmax(logits.float(), dim=-1)
    top_values, top_indices = torch.topk(logits.float(), k=min(int(top_k), int(logits.shape[-1])))
    rows: List[Dict[str, Any]] = []
    for logit, token_id in zip(top_values.tolist(), top_indices.tolist()):
        rows.append(
            {
                "token_id": int(token_id),
                "token_text": backend.tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "logit": float(logit),
                "prob": float(probs[int(token_id)].item()),
            }
        )
    return rows


def run_worker(
    worker_id: int,
    gpu_id: int,
    shard_rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_rows_path: str,
    worker_skipped_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map={"": gpu_id},
            load_in_half=args.load_in_half,
            use_fast_tokenizer=args.use_fast_tokenizer,
            use_safetensors=args.use_safetensors,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    )
    if backend.model is None or backend.tokenizer is None:
        raise ValueError("This experiment requires HF backend internals.")
    generation_config = build_generation_config(args)
    targets, attn_modules, _ = resolve_head_targets(backend.model, [args.head_label])
    if len(targets) != 1:
        raise ValueError("Expected exactly one target head.")
    target = targets[0]
    attn_module = attn_modules[target.layer_idx]
    _ = extract_head_ov_components(backend.model, layer_idx=target.layer_idx, head_idx=target.head_idx)
    scale_values = parse_scale_values(args.scale_values)

    kept = 0
    skipped_rows: List[Dict[str, str]] = []
    with open(worker_rows_path, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(shard_rows, start=1):
            result, skipped = analyze_one_row(
                row,
                backend=backend,
                generation_config=generation_config,
                attn_module=attn_module,
                target=target,
                scale_values=scale_values,
                top_k=args.top_k,
            )
            if skipped is not None:
                skipped_rows.append(skipped)
                continue
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            kept += 1
            if idx % max(int(args.print_every), 1) == 0:
                print(
                    f"[Info] worker={worker_id} gpu={gpu_id} processed={idx} kept={kept} skipped={len(skipped_rows)} latest_example_id={result['example_id']}"
                )

    with open(worker_skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped_rows, f, ensure_ascii=False, indent=2)

    return {
        "worker_id": int(worker_id),
        "gpu_id": int(gpu_id),
        "kept": int(kept),
        "skipped": int(len(skipped_rows)),
        "worker_rows_path": worker_rows_path,
        "worker_skipped_path": worker_skipped_path,
    }


def split_examples_contiguous(rows: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    if num_shards <= 0 or not rows:
        return []
    shard_size = (len(rows) + num_shards - 1) // num_shards
    shards = [rows[idx : idx + shard_size] for idx in range(0, len(rows), shard_size)]
    return [shard for shard in shards if shard]


def main() -> None:
    args = parse_args()
    input_path = Path(args.repeat_examples_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"repeat_examples_jsonl not found: {input_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = load_seed_rows(input_path, args.max_cases)
    if not seed_rows:
        raise ValueError(f"No directly callable seed rows found in {input_path}")

    gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    if not gpu_ids:
        raise ValueError("This experiment requires explicit or visible CUDA GPUs.")
    if args.parallel_workers > 0:
        worker_count = min(int(args.parallel_workers), len(gpu_ids), len(seed_rows))
    else:
        worker_count = min(len(gpu_ids), len(seed_rows))
    selected_gpu_ids = gpu_ids[:worker_count]
    example_shards = split_examples_contiguous(seed_rows, worker_count)

    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "seed_row_count": len(seed_rows),
            "worker_count": worker_count,
            "selected_gpu_ids": selected_gpu_ids,
            "scale_values_parsed": parse_scale_values(args.scale_values),
        },
    )

    print(
        "[Info] repetition scale sweep setup: "
        f"seed_rows={len(seed_rows)}, worker_count={worker_count}, gpus={selected_gpu_ids}, scales={parse_scale_values(args.scale_values)}"
    )

    worker_returns: List[Dict[str, Any]] = []
    worker_rows_paths: List[Path] = []
    worker_skipped_paths: List[Path] = []
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_ctx) as pool:
        futures = []
        for worker_id, shard_rows in enumerate(example_shards):
            gpu_id = selected_gpu_ids[worker_id]
            worker_rows_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_rows.jsonl"
            worker_skipped_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_skipped.json"
            futures.append(
                pool.submit(
                    run_worker,
                    worker_id,
                    gpu_id,
                    shard_rows,
                    vars(args),
                    str(worker_rows_path),
                    str(worker_skipped_path),
                )
            )
        for future in as_completed(futures):
            worker_ret = future.result()
            worker_returns.append(worker_ret)
            worker_rows_paths.append(Path(worker_ret["worker_rows_path"]))
            worker_skipped_paths.append(Path(worker_ret["worker_skipped_path"]))
            print(
                f"[Info] worker_finished id={worker_ret['worker_id']} gpu={worker_ret['gpu_id']} kept={worker_ret['kept']} skipped={worker_ret['skipped']}"
            )

    case_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    for worker_ret in sorted(worker_returns, key=lambda item: int(item["worker_id"])):
        with open(worker_ret["worker_rows_path"], "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    case_rows.append(json.loads(line))
        with open(worker_ret["worker_skipped_path"], "r", encoding="utf-8") as f:
            skipped_rows.extend(json.load(f))

    per_scale_rows: List[Dict[str, Any]] = []
    for case in case_rows:
        per_scale_rows.extend(list(case.get("scale_rows") or []))

    summary_by_scale: List[Dict[str, Any]] = []
    grouped: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_scale_rows:
        grouped[float(row["scale"])].append(row)

    for scale in sorted(grouped):
        rows = grouped[scale]
        count = len(rows)
        if count == 0:
            continue
        summary_by_scale.append(
            {
                "scale": float(scale),
                "example_count": int(count),
                "repetition_hit_rate": round(sum(1 for row in rows if bool(row["trajectory_repetitive"])) / count, 6),
                "mean_generated_tokens": round(sum(float(row["generated_tokens"]) for row in rows) / count, 6),
                "mean_loop_logit": round(sum(float(row["loop_logit"]) for row in rows) / count, 6),
                "mean_escape_logit": round(sum(float(row["escape_logit"]) for row in rows) / count, 6),
                "mean_loop_prob": round(sum(float(row["loop_prob"]) for row in rows) / count, 6),
                "mean_escape_prob": round(sum(float(row["escape_prob"]) for row in rows) / count, 6),
                "mean_escape_minus_loop_logit": round(sum(float(row["escape_minus_loop_logit"]) for row in rows) / count, 6),
                "loop_logit_down_rate_vs_baseline": round(sum(1 for row in rows if bool(row["loop_logit_down_vs_baseline"])) / count, 6),
                "escape_logit_up_rate_vs_baseline": round(sum(1 for row in rows if bool(row["escape_logit_up_vs_baseline"])) / count, 6),
                "both_effect_rate_vs_baseline": round(sum(1 for row in rows if bool(row["both_effect_vs_baseline"])) / count, 6),
                "mean_loop_logit_delta_vs_baseline": round(sum(float(row["loop_logit_delta_vs_baseline"]) for row in rows) / count, 6),
                "mean_escape_logit_delta_vs_baseline": round(sum(float(row["escape_logit_delta_vs_baseline"]) for row in rows) / count, 6),
                "mean_loop_prob_delta_vs_baseline": round(sum(float(row["loop_prob_delta_vs_baseline"]) for row in rows) / count, 6),
                "mean_escape_prob_delta_vs_baseline": round(sum(float(row["escape_prob_delta_vs_baseline"]) for row in rows) / count, 6),
            }
        )

    case_summary_rows: List[Dict[str, Any]] = []
    anchor_scale_counter: Counter[float] = Counter()
    for case in case_rows:
        anchor_scale = case.get("anchor_scale")
        if anchor_scale is not None:
            anchor_scale_counter.update([float(anchor_scale)])
        baseline_row = dict(case.get("baseline_row") or {})
        case_summary_rows.append(
            {
                "example_id": case["example_id"],
                "source": case.get("source"),
                "anchor_scale": case.get("anchor_scale"),
                "divergence_step": case.get("divergence_step"),
                "loop_token_text": case.get("loop_token_text"),
                "escape_token_text": case.get("escape_token_text"),
                "baseline_loop_logit": baseline_row.get("loop_logit"),
                "baseline_escape_logit": baseline_row.get("escape_logit"),
                "baseline_generated_tokens": case.get("baseline_generated_tokens"),
                "scales_that_escape": json.dumps(case.get("scales_that_escape") or []),
                "scale_repetition_map": json.dumps(case.get("scale_repetition_map") or {}, ensure_ascii=False),
                "max_scale_both_effect": case.get("max_scale_both_effect"),
            }
        )

    write_csv(output_dir / "case_summary.csv", case_summary_rows)
    write_csv(output_dir / "scale_logit_rows.csv", per_scale_rows)
    write_csv(output_dir / "scale_summary.csv", summary_by_scale)
    with open(output_dir / "case_results.jsonl", "w", encoding="utf-8") as f:
        for case in case_rows:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    write_json(
        output_dir / "summary.json",
        {
            "seed_row_count": len(seed_rows),
            "baseline_repetitive_case_count": len(case_rows),
            "skipped_count": len(skipped_rows),
            "skip_reasons": dict(Counter(str(item.get("reason")) for item in skipped_rows)),
            "anchor_scale_counts": {str(key): int(value) for key, value in sorted(anchor_scale_counter.items())},
            "summary_by_scale": summary_by_scale,
        },
    )
    write_json(output_dir / "skipped_rows.json", skipped_rows)

    if not args.keep_worker_outputs:
        for path in worker_rows_paths + worker_skipped_paths:
            if path.exists():
                path.unlink()

    print("[Done] repetition scale logit sweep finished:")
    print(f"- output_dir: {output_dir}")
    print(f"- seed_row_count: {len(seed_rows)}")
    print(f"- baseline_repetitive_case_count: {len(case_rows)}")
    print(f"- skipped_count: {len(skipped_rows)}")
    print(f"- scale_summary_csv: {output_dir / 'scale_summary.csv'}")


if __name__ == "__main__":
    main()
