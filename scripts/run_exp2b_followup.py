#!/usr/bin/env python3
from __future__ import annotations

"""Exp 2b Follow-up: Targeted grouped ablation with C2-discovered heads.

Tests the L0H1+L0H26 combination that produced 14-16% loop rate on held-out,
plus leave-one-out analysis to identify key drivers.

Usage:
    CUDA_VISIBLE_DEVICES=1,3,4,5,6,7,8 python scripts/run_exp2b_followup.py \
        --model_name_or_path Qwen/Qwen3-4B \
        --parallel_gpu_ids 0,1,2,3,4,5,6 \
        --max_examples_per_subtask 50 \
        --max_new_tokens 1024 \
        --output_dir experiment_results/experiments/phase7_exp2b/4b_followup
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.generation import create_backend
from cot_research.head_intervention import (
    INTERVENTION_REGISTRY,
    MultiLayerHeadIntervention,
    resolve_head_targets,
)
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.repetition_analysis import LoopBenchThresholds, analyze_loopbench_repetition
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig


# The C2_23 configuration that produced 16% loop rate on held-out
C2_BEST_GROUP = ["L0H26", "L35H11", "L12H27", "L22H18", "L16H10", "L24H27", "L14H8", "L0H1"]

# Original top-8 SAC for comparison
SAC_TOP8 = ["L0H2", "L35H0", "L12H25", "L22H12", "L16H27", "L24H24", "L14H30", "L0H22"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 2b Follow-up: C2-discovered group")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--input_jsonl", type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "loopbench_reconstructed_v2" / "test.jsonl"),
    )
    parser.add_argument("--subtasks", type=str, default="square_root,newtons_iteration")
    parser.add_argument("--max_examples_per_subtask", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--parallel_gpu_ids", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--system_prompt", type=str,
                        default="Please reason step by step in <think>...</think>. Put your final answer within \\boxed{} after the reasoning.")
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n_random_controls", type=int, default=20)
    parser.add_argument("--numerical_loop_min_repeated_span", type=int, default=500)
    parser.add_argument("--statement_loop_min_repeat_count", type=int, default=4)
    parser.add_argument("--numerical_same_digit_run_threshold", type=int, default=500)
    return parser.parse_args()


def load_filtered_rows(args) -> List[Dict]:
    rows = load_jsonl(args.input_jsonl)
    subtask_names = [s.strip() for s in args.subtasks.split(",")]
    filtered = [r for r in rows if any(st in r.get("id", "") for st in subtask_names)]
    if args.max_examples_per_subtask > 0:
        by_subtask: Dict[str, List] = defaultdict(list)
        for r in filtered:
            for st in subtask_names:
                if st in r.get("id", ""):
                    by_subtask[st].append(r)
                    break
        result = []
        for st in subtask_names:
            result.extend(by_subtask[st][:args.max_examples_per_subtask])
        return result
    return filtered


def run_conditions_on_gpu(
    worker_id: int,
    gpu_id: int,
    conditions: List[Dict[str, Any]],
    rows: List[Dict],
    args_dict: Dict[str, Any],
    worker_output_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    mp.current_process().name = f"Worker-{worker_id}"

    thresholds = LoopBenchThresholds(
        numerical_loop_min_repeated_span=args.numerical_loop_min_repeated_span,
        statement_loop_min_repeat_count=args.statement_loop_min_repeat_count,
        numerical_same_digit_run_threshold=args.numerical_same_digit_run_threshold,
    )

    backend = create_backend(
        BackendConfig(
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

    prepared = []
    for idx, row in enumerate(rows):
        example_id = str(row.get("id", idx))
        try:
            prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
            prepared.append((example_id, prompt_prefix))
        except Exception as exc:
            print(f"[Worker {worker_id}] Skip {example_id}: {exc}")

    all_results: List[Dict[str, Any]] = []

    for cond in conditions:
        cond_name = cond["name"]
        head_labels = cond["head_labels"]

        if head_labels:
            targets, attn_modules, _ = resolve_head_targets(model, head_labels)
            operations = INTERVENTION_REGISTRY.get_required("zero")(targets, {})
        else:
            targets, attn_modules, operations = None, None, None

        loop_count = 0
        for eid, prompt in tqdm(prepared, desc=f"W{worker_id} {cond_name}", leave=False):
            seed_everything(args.seed)
            if attn_modules is not None:
                with MultiLayerHeadIntervention(attn_modules, operations):
                    gen = backend.generate(prompt, generation_config)
            else:
                gen = backend.generate(prompt, generation_config)

            rep = analyze_loopbench_repetition(gen.continuation, thresholds=thresholds)
            is_rep = bool(rep["matched"])
            if is_rep:
                loop_count += 1
            all_results.append({
                "condition": cond_name,
                "condition_type": cond["type"],
                "head_labels": ",".join(head_labels) if head_labels else "",
                "k": cond.get("k", 0),
                "example_id": eid,
                "is_repetitive": is_rep,
                "generated_tokens": gen.generated_tokens,
            })

        print(f"[Worker {worker_id}|GPU{gpu_id}] {cond_name}: {loop_count}/{len(prepared)} repetitive")

    dump_jsonl(Path(worker_output_path), all_results)
    return {"worker_id": worker_id, "row_count": len(all_results)}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_filtered_rows(args)
    if not rows:
        raise ValueError("No examples loaded")
    print(f"[Info] Loaded {len(rows)} examples")

    rng = random.Random(args.seed)

    conditions: List[Dict[str, Any]] = []

    # Baseline
    conditions.append({"name": "baseline", "type": "baseline", "head_labels": [], "k": 0})

    # C2-best full group (the winning configuration)
    conditions.append({
        "name": "c2_best_full",
        "type": "c2_best",
        "head_labels": C2_BEST_GROUP,
        "k": 8,
    })

    # SAC top-8 for comparison
    conditions.append({
        "name": "sac_top8",
        "type": "sac_top8",
        "head_labels": SAC_TOP8,
        "k": 8,
    })

    # Swap experiment: SAC top-8 but replace L0 heads with L0H1+L0H26
    swap_group = [h for h in SAC_TOP8 if not h.startswith("L0")] + ["L0H1", "L0H26"]
    conditions.append({
        "name": "sac_top8_l0swap",
        "type": "l0_swap",
        "head_labels": swap_group,
        "k": 8,
    })

    # Leave-one-out from C2-best (8 conditions)
    for i, head in enumerate(C2_BEST_GROUP):
        loo_group = [h for h in C2_BEST_GROUP if h != head]
        conditions.append({
            "name": f"loo_remove_{head}",
            "type": "leave_one_out",
            "head_labels": loo_group,
            "k": 7,
        })

    # L0-only ablation: just L0H1+L0H26
    conditions.append({
        "name": "l0_pair_only",
        "type": "l0_pair",
        "head_labels": ["L0H1", "L0H26"],
        "k": 2,
    })

    # Random 8-head controls
    all_heads = [f"L{l}H{h}" for l in range(36) for h in range(32)]
    for i in range(args.n_random_controls):
        group = rng.sample([h for h in all_heads if h not in C2_BEST_GROUP], 8)
        conditions.append({
            "name": f"random_ctrl_{i:02d}",
            "type": "random_control",
            "head_labels": group,
            "k": 8,
        })

    n_conds = len(conditions)
    print(f"[Info] {n_conds} conditions: 1 baseline, 1 c2_best, 1 sac_top8, 1 l0_swap, "
          f"8 leave-one-out, 1 l0_pair, {args.n_random_controls} random controls")

    write_json(output_dir / "run_config.json", {
        "c2_best_group": C2_BEST_GROUP,
        "sac_top8": SAC_TOP8,
        "swap_group": swap_group,
        "n_conditions": n_conds,
        "n_examples": len(rows),
    })

    gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    n_workers = len(gpu_ids)
    print(f"[Info] {n_workers} workers on GPUs {gpu_ids}")

    worker_assignments: List[List[Dict]] = [[] for _ in range(n_workers)]
    for i, cond in enumerate(conditions):
        worker_assignments[i % n_workers].append(cond)

    args_dict = vars(args)

    if n_workers == 1:
        wpath = str(output_dir / "_worker_0_rows.jsonl")
        run_conditions_on_gpu(0, gpu_ids[0], worker_assignments[0], rows, args_dict, wpath)
    else:
        ctx = mp.get_context("spawn")
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            for wi in range(n_workers):
                if not worker_assignments[wi]:
                    continue
                wpath = str(output_dir / f"_worker_{wi}_rows.jsonl")
                future = executor.submit(
                    run_conditions_on_gpu,
                    wi, gpu_ids[wi], worker_assignments[wi], rows, args_dict, wpath,
                )
                futures[future] = wi
            for future in as_completed(futures):
                wi = futures[future]
                try:
                    result = future.result()
                    print(f"[Worker {wi}] Done: {result['row_count']} rows")
                except Exception as exc:
                    print(f"[Worker {wi}] FAILED: {exc}")
                    raise

    # Merge
    all_rows: List[Dict] = []
    for wi in range(n_workers):
        wp = output_dir / f"_worker_{wi}_rows.jsonl"
        if wp.exists():
            all_rows.extend(load_jsonl(wp))
    dump_jsonl(output_dir / "all_results.jsonl", all_rows)

    # Stats
    cond_stats: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "n_rep": 0})
    for r in all_rows:
        cn = r["condition"]
        cond_stats[cn]["n"] += 1
        if r["is_repetitive"]:
            cond_stats[cn]["n_rep"] += 1
        cond_stats[cn]["type"] = r["condition_type"]
        cond_stats[cn]["k"] = r.get("k", 0)
        cond_stats[cn]["head_labels"] = r.get("head_labels", "")

    stats_list = []
    for cn, s in sorted(cond_stats.items()):
        loop_rate = s["n_rep"] / s["n"] if s["n"] > 0 else 0
        stats_list.append({
            "condition": cn, "type": s["type"], "k": s["k"],
            "head_labels": s["head_labels"],
            "n_examples": s["n"], "n_repetitive": s["n_rep"],
            "loop_rate": round(loop_rate, 4),
        })
    write_csv(output_dir / "condition_stats.csv", stats_list)

    # Summary
    summary: Dict[str, Any] = {"model": args.model_name_or_path, "n_examples": len(rows)}

    for s in stats_list:
        if s["type"] in ("baseline", "c2_best", "sac_top8", "l0_swap", "l0_pair"):
            summary[s["condition"]] = {"loop_rate": s["loop_rate"], "n_rep": s["n_repetitive"], "n": s["n_examples"]}
        elif s["type"] == "leave_one_out":
            summary.setdefault("leave_one_out", {})[s["condition"]] = s["loop_rate"]

    ctrl_rates = [s["loop_rate"] for s in stats_list if s["type"] == "random_control"]
    c2_rate = summary.get("c2_best_full", {}).get("loop_rate", 0)
    n_exceed = sum(1 for cr in ctrl_rates if cr >= c2_rate)
    summary["random_ctrl_p"] = round((n_exceed + 1) / (len(ctrl_rates) + 1), 4)
    summary["random_ctrl_mean"] = round(np.mean(ctrl_rates), 4) if ctrl_rates else None

    write_json(output_dir / "summary.json", summary)

    # Print
    print("\n" + "=" * 60)
    print(f"Exp 2b Follow-up — {args.model_name_or_path}")
    print("=" * 60)
    for key in ["baseline", "c2_best_full", "sac_top8", "sac_top8_l0swap", "l0_pair_only"]:
        if key in summary:
            d = summary[key]
            print(f"  {key}: {d['loop_rate']:.1%} ({d['n_rep']}/{d['n']})")
    print(f"\nLeave-one-out from C2-best:")
    for name, rate in sorted(summary.get("leave_one_out", {}).items()):
        removed = name.replace("loo_remove_", "")
        print(f"  remove {removed}: {rate:.1%}")
    print(f"\nRandom controls: mean={summary['random_ctrl_mean']:.1%}, p_vs_c2_best={summary['random_ctrl_p']:.4f}")
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
