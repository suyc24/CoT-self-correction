#!/usr/bin/env python3
from __future__ import annotations

"""Exp 1: Cross-model RSH Localization — Stage 0 (sentinel) + Stage 1 (full sweep).

Stage 0: Per-layer random sentinel scan — for each layer, randomly sample k heads per
round for R rounds, zero-ablate, and measure Δrep. Flags layers with signal.

Stage 1: Full-head sweep on target layers with prompt-paired bootstrap CI + BH FDR.

Usage (1.7B calibration, Stage 0+1 on GPU 0):
    python scripts/run_exp1_rsh_localization.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --stage 0 \
        --parallel_gpu_ids 0 \
        --subtasks square_root \
        --max_examples 50 \
        --output_dir experiment_results/experiments/phase7_exp1/1p7b_stage0

    python scripts/run_exp1_rsh_localization.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --stage 1 \
        --target_layers 0 \
        --parallel_gpu_ids 0 \
        --subtasks square_root,newtons_iteration \
        --max_examples_per_subtask 50 \
        --output_dir experiment_results/experiments/phase7_exp1/1p7b_stage1 \
        --baseline_rows_jsonl experiment_results/experiments/phase7_exp1/1p7b_stage0/baseline_rows.jsonl
"""

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.bootstrap_stats import (
    apply_fdr_to_stats,
    compute_head_ablation_stats,
    recall_at_k,
)
from cot_research.generation import create_backend
from cot_research.head_intervention import (
    INTERVENTION_REGISTRY,
    MultiLayerHeadIntervention,
    list_model_heads,
    resolve_head_targets,
)
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.model_utils import (
    get_attention_module,
    get_decoder_layers,
    infer_attention_head_shape,
)
from cot_research.repetition_analysis import LoopBenchThresholds, analyze_loopbench_repetition
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 1: Cross-model RSH Localization")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--stage", type=int, required=True, choices=[0, 1], help="0=sentinel scan, 1=full sweep")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "loopbench_reconstructed_v2" / "test.jsonl"),
    )
    parser.add_argument("--subtasks", type=str, default="square_root",
                        help="Comma-separated subtask names to filter")
    parser.add_argument("--max_examples_per_subtask", type=int, default=50)
    parser.add_argument("--max_examples", type=int, default=-1, help="Override: total examples regardless of subtask")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--parallel_gpu_ids", type=str, required=True)
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Cap generation length (4096 sufficient for loop detection, 4x faster than 16K)")
    parser.add_argument(
        "--system_prompt", type=str,
        default="Please reason step by step in <think>...</think>. Put your final answer within \\boxed{} after the reasoning.",
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep_worker_outputs", action=argparse.BooleanOptionalAction, default=False)
    # Stage 0 params
    parser.add_argument("--sentinel_k", type=int, default=4, help="Heads per layer per round in Stage 0")
    parser.add_argument("--sentinel_rounds", type=int, default=3, help="Independent sampling rounds in Stage 0")
    # Stage 1 params
    parser.add_argument("--target_layers", type=str, default="0",
                        help="Comma-separated layer indices to sweep in Stage 1")
    parser.add_argument("--bootstrap_B", type=int, default=10000)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    # Baseline reuse
    parser.add_argument("--baseline_rows_jsonl", type=str, default="",
                        help="Path to baseline (scale=1) rows for reuse")
    # Repetition thresholds (LoopBench)
    parser.add_argument("--numerical_loop_min_repeated_span", type=int, default=500)
    parser.add_argument("--statement_loop_min_repeat_count", type=int, default=4)
    parser.add_argument("--numerical_same_digit_run_threshold", type=int, default=500)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def load_filtered_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl)
    subtasks = [s.strip() for s in args.subtasks.split(",") if s.strip()]
    if subtasks:
        filtered = []
        for st in subtasks:
            st_rows = [r for r in rows if (r.get("metadata") or {}).get("subtask") == st]
            if args.max_examples_per_subtask > 0:
                st_rows = st_rows[:args.max_examples_per_subtask]
            filtered.extend(st_rows)
        rows = filtered
    if args.max_examples > 0:
        rows = rows[:args.max_examples]
    return rows


# --------------------------------------------------------------------------- #
# Stage 0: Sentinel scan                                                       #
# --------------------------------------------------------------------------- #

def build_sentinel_head_list(
    model: torch.nn.Module,
    k: int,
    rounds: int,
    seed: int,
) -> Tuple[List[str], Dict[int, List[str]]]:
    """Generate random head samples per layer for sentinel scan."""
    layers, _ = get_decoder_layers(model)
    rng = random.Random(seed)
    per_layer: Dict[int, List[str]] = {}
    all_heads: List[str] = []

    for li in range(len(layers)):
        attn = get_attention_module(layers[li])
        num_heads, _ = infer_attention_head_shape(model, attn)
        available = list(range(num_heads))
        selected = set()
        for _ in range(rounds):
            sample = rng.sample(available, min(k, len(available)))
            selected.update(sample)
        layer_heads = sorted(selected)
        labels = [f"L{li}H{hi}" for hi in layer_heads]
        per_layer[li] = labels
        all_heads.extend(labels)

    return all_heads, per_layer


# --------------------------------------------------------------------------- #
# Worker: single-head zero-ablation generation                                 #
# --------------------------------------------------------------------------- #

def run_ablation_worker(
    worker_id: int,
    gpu_id: int,
    head_labels: List[str],
    rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_output_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    thresholds = LoopBenchThresholds(
        numerical_loop_min_repeated_span=args.numerical_loop_min_repeated_span,
        statement_loop_min_repeat_count=args.statement_loop_min_repeat_count,
        numerical_same_digit_run_threshold=args.numerical_same_digit_run_threshold,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map={"": gpu_id},
            load_in_half=args.load_in_half,
            use_fast_tokenizer=False,
            use_safetensors=True,
            local_files_only=args.local_files_only,
        )
    )
    model = backend.model

    generation_config = GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )

    # Prepare prompts
    prepared = []
    for idx, row in enumerate(rows):
        example_id = str(row.get("id", idx))
        try:
            prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
            prepared.append((example_id, prompt_prefix))
        except Exception as exc:
            print(f"[Worker {worker_id}] Skip {example_id}: {exc}")

    # Generate baseline (scale=1) once
    baseline_by_id: Dict[str, bool] = {}
    baseline_rows_data: List[Dict[str, Any]] = []

    provided_baseline = args_dict.get("_baseline_by_id")
    if provided_baseline:
        baseline_by_id = provided_baseline
    else:
        print(f"[Worker {worker_id}|GPU{gpu_id}] Generating baseline ({len(prepared)} prompts)...")
        for eid, prompt in tqdm(prepared, desc=f"W{worker_id} baseline", leave=False):
            seed_everything(args.seed)
            gen = backend.generate(prompt, generation_config)
            rep = analyze_loopbench_repetition(gen.continuation, thresholds=thresholds)
            is_rep = bool(rep["matched"])
            baseline_by_id[eid] = is_rep
            baseline_rows_data.append({
                "example_id": eid, "is_repetitive": is_rep,
                "generated_tokens": gen.generated_tokens,
            })

    # Zero-ablate each head
    all_rows: List[Dict[str, Any]] = []
    for head_label in head_labels:
        targets, attn_modules, _ = resolve_head_targets(model, [head_label])
        operations = INTERVENTION_REGISTRY.get_required("zero")(targets, {})

        for eid, prompt in tqdm(prepared, desc=f"W{worker_id} {head_label}", leave=False):
            seed_everything(args.seed)
            with MultiLayerHeadIntervention(attn_modules, operations):
                gen = backend.generate(prompt, generation_config)
            rep = analyze_loopbench_repetition(gen.continuation, thresholds=thresholds)
            all_rows.append({
                "head_label": head_label,
                "example_id": eid,
                "baseline_rep": baseline_by_id.get(eid, False),
                "ablation_rep": bool(rep["matched"]),
                "generated_tokens": gen.generated_tokens,
            })

        done_count = sum(1 for r in all_rows if r["head_label"] == head_label)
        abl_reps = sum(1 for r in all_rows if r["head_label"] == head_label and r["ablation_rep"])
        print(f"[Worker {worker_id}|GPU{gpu_id}] {head_label}: {abl_reps}/{done_count} repetitive after ablation")

    # Write results
    output_path = Path(worker_output_path)
    dump_jsonl(output_path, all_rows)
    if baseline_rows_data:
        dump_jsonl(output_path.parent / f"_worker_{worker_id}_baseline.jsonl", baseline_rows_data)

    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "head_labels": head_labels,
        "row_count": len(all_rows),
        "baseline_count": len(baseline_rows_data),
        "worker_output_path": worker_output_path,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_filtered_rows(args)
    if not rows:
        raise ValueError("No examples loaded")
    print(f"[Info] Loaded {len(rows)} examples")

    gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    if not gpu_ids:
        raise ValueError("Provide --parallel_gpu_ids")

    # Load provided baseline if available
    provided_baseline: Optional[Dict[str, bool]] = None
    if args.baseline_rows_jsonl:
        bl_path = Path(args.baseline_rows_jsonl)
        if bl_path.exists():
            bl_rows = load_jsonl(bl_path)
            provided_baseline = {
                str(r.get("example_id", "")): bool(r.get("is_repetitive", False))
                for r in bl_rows
            }
            print(f"[Info] Loaded {len(provided_baseline)} baseline rows from {bl_path}")

    # Determine head list based on stage
    if args.stage == 0:
        # Need model briefly to get architecture
        tmp_backend = create_backend(
            BackendConfig(
                backend_type="hf",
                model_name_or_path=args.model_name_or_path,
                device_map={"": gpu_ids[0]},
                load_in_half=args.load_in_half,
                use_fast_tokenizer=False, use_safetensors=True,
                local_files_only=args.local_files_only,
            )
        )
        head_labels, per_layer = build_sentinel_head_list(
            tmp_backend.model, args.sentinel_k, args.sentinel_rounds, args.seed,
        )
        del tmp_backend
        torch.cuda.empty_cache()

        write_json(output_dir / "sentinel_plan.json", {
            "sentinel_k": args.sentinel_k,
            "sentinel_rounds": args.sentinel_rounds,
            "per_layer_heads": {str(k): v for k, v in per_layer.items()},
            "total_heads": len(head_labels),
        })
        print(f"[Stage 0] Sentinel scan: {len(head_labels)} heads across {len(per_layer)} layers")

    elif args.stage == 1:
        target_layers = [int(x.strip()) for x in args.target_layers.split(",") if x.strip()]
        # Need model briefly to enumerate heads
        tmp_backend = create_backend(
            BackendConfig(
                backend_type="hf",
                model_name_or_path=args.model_name_or_path,
                device_map={"": gpu_ids[0]},
                load_in_half=args.load_in_half,
                use_fast_tokenizer=False, use_safetensors=True,
                local_files_only=args.local_files_only,
            )
        )
        all_heads, _, _ = list_model_heads(tmp_backend.model)
        head_labels = [h.label for h in all_heads if h.layer_idx in target_layers]
        del tmp_backend
        torch.cuda.empty_cache()

        print(f"[Stage 1] Full sweep: {len(head_labels)} heads in layers {target_layers}")
    else:
        raise ValueError(f"Unknown stage {args.stage}")

    write_json(output_dir / "run_config.json", {
        "args": vars(args),
        "head_labels": head_labels,
        "n_examples": len(rows),
        "gpu_ids": gpu_ids,
    })

    # Distribute heads across workers
    num_workers = min(len(gpu_ids), len(head_labels))
    if args.parallel_workers > 0:
        num_workers = min(args.parallel_workers, num_workers)

    head_buckets: List[List[str]] = [[] for _ in range(num_workers)]
    for idx, h in enumerate(head_labels):
        head_buckets[idx % num_workers].append(h)
    head_buckets = [b for b in head_buckets if b]
    num_workers = len(head_buckets)

    print(f"[Info] {num_workers} workers on GPUs {gpu_ids[:num_workers]}")

    # Run workers
    args_dict = vars(args)
    if provided_baseline:
        args_dict["_baseline_by_id"] = provided_baseline

    worker_outputs: List[Dict[str, Any]] = []
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as pool:
        futures = []
        for wi, (bucket, gid) in enumerate(zip(head_buckets, gpu_ids[:num_workers])):
            wpath = str(output_dir / f"_worker_{wi}_rows.jsonl")
            futures.append(pool.submit(
                run_ablation_worker, wi, gid, bucket, rows, args_dict, wpath,
            ))
        for fut in as_completed(futures):
            ret = fut.result()
            worker_outputs.append(ret)
            print(f"[Done] Worker {ret['worker_id']}: {ret['row_count']} rows")

    # Merge results
    all_rows: List[Dict[str, Any]] = []
    for ret in sorted(worker_outputs, key=lambda r: r["worker_id"]):
        wp = Path(ret["worker_output_path"])
        if wp.exists():
            all_rows.extend(load_jsonl(wp))

    # Merge baselines from workers
    baseline_rows: List[Dict[str, Any]] = []
    for wi in range(num_workers):
        bp = output_dir / f"_worker_{wi}_baseline.jsonl"
        if bp.exists():
            baseline_rows.extend(load_jsonl(bp))
    if baseline_rows:
        seen = set()
        deduped = []
        for r in baseline_rows:
            eid = r.get("example_id")
            if eid not in seen:
                seen.add(eid)
                deduped.append(r)
        dump_jsonl(output_dir / "baseline_rows.jsonl", deduped)

    dump_jsonl(output_dir / "all_ablation_rows.jsonl", all_rows)
    print(f"[Info] Total ablation rows: {len(all_rows)}")

    # Compute bootstrap statistics per head
    by_head: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_head[r["head_label"]].append(r)

    stats_list: List[Dict[str, Any]] = []
    for head_label, head_rows in sorted(by_head.items()):
        baseline_rep = [bool(r["baseline_rep"]) for r in head_rows]
        ablation_rep = [bool(r["ablation_rep"]) for r in head_rows]
        stats = compute_head_ablation_stats(
            head_label, baseline_rep, ablation_rep,
            B=args.bootstrap_B, seed=args.seed,
        )
        stats_list.append(stats)

    stats_list = apply_fdr_to_stats(stats_list, alpha=args.fdr_alpha)

    write_csv(output_dir / "head_stats.csv", stats_list)

    # Summary
    significant = [s for s in stats_list if s["significant_bh"]]
    sig_sorted = sorted(significant, key=lambda s: s["q_value_bh"])

    summary = {
        "model": args.model_name_or_path,
        "stage": args.stage,
        "n_examples": len(rows),
        "n_heads_tested": len(stats_list),
        "n_significant_bh": len(significant),
        "significant_heads": [
            {"head": s["head_label"], "q": round(s["q_value_bh"], 6),
             "mean_delta": round(s["mean_delta_rep"], 4), "d": round(s["cohens_d"], 3)}
            for s in sig_sorted
        ],
        "bootstrap_B": args.bootstrap_B,
        "fdr_alpha": args.fdr_alpha,
    }

    if args.stage == 0:
        # Layer-level heatmap
        layer_signal: Dict[int, Dict[str, Any]] = {}
        for s in stats_list:
            li = int(s["head_label"].split("H")[0].replace("L", ""))
            if li not in layer_signal:
                layer_signal[li] = {"max_delta": -999, "max_head": "", "any_significant": False, "heads_tested": 0}
            layer_signal[li]["heads_tested"] += 1
            if s["mean_delta_rep"] > layer_signal[li]["max_delta"]:
                layer_signal[li]["max_delta"] = s["mean_delta_rep"]
                layer_signal[li]["max_head"] = s["head_label"]
            if s["significant_bh"]:
                layer_signal[li]["any_significant"] = True

        # Flag layers with signal: any head > 95th percentile of all layer max deltas
        all_max_deltas = [v["max_delta"] for v in layer_signal.values()]
        p95 = float(np.percentile(all_max_deltas, 95)) if all_max_deltas else 0.0
        signal_layers = [li for li, v in layer_signal.items()
                         if v["max_delta"] > p95 or v["any_significant"]]
        summary["layer_signal"] = {str(k): v for k, v in sorted(layer_signal.items())}
        summary["signal_layers"] = sorted(signal_layers)
        summary["p95_threshold"] = p95

    write_json(output_dir / "summary.json", summary)

    # Cleanup worker files
    if not args.keep_worker_outputs:
        for wi in range(num_workers):
            for pattern in [f"_worker_{wi}_rows.jsonl", f"_worker_{wi}_baseline.jsonl"]:
                p = output_dir / pattern
                if p.exists():
                    p.unlink()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Exp 1 Stage {args.stage} — {args.model_name_or_path}")
    print(f"{'='*60}")
    print(f"Heads tested: {len(stats_list)}, Significant (BH q<{args.fdr_alpha}): {len(significant)}")
    if significant:
        for s in sig_sorted[:10]:
            print(f"  {s['head_label']:>6s}: Δrep={s['mean_delta_rep']:+.3f}  "
                  f"q={s['q_value_bh']:.4f}  d={s['cohens_d']:.3f}  "
                  f"rank={s['rank_by_q']}")
    if args.stage == 0 and "signal_layers" in summary:
        print(f"\nSignal layers: {summary['signal_layers']}")
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
