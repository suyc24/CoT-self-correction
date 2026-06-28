#!/usr/bin/env python3
from __future__ import annotations

"""Exp 2b: Grouped Ablation — distributed RSH hypothesis test.

Triggered by Exp 1 null result (0/128 significant heads in 4B).
Tests whether grouped ablation of top-SAC heads produces repetition that no
single head ablation can.

Design:
  Phase 1 (discovery): Top-k grouped ablation (k=2,3,5,8) + individual heads
                        + baseline + random controls, on discovery prompts.
  Phase 2 (held-out):  If Phase 1 shows signal, replicate on held-out prompts.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python scripts/run_exp2b_grouped_ablation.py \
        --model_name_or_path Qwen/Qwen3-4B \
        --landscape_csv experiment_results/experiments/phase7_exp0/4b_calibration/landscape_ranked.csv \
        --parallel_gpu_ids 0,1,2,3,4,5 \
        --max_examples_per_subtask 50 \
        --max_new_tokens 1024 \
        --output_dir experiment_results/experiments/phase7_exp2b/4b_discovery
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
    parser = argparse.ArgumentParser(description="Exp 2b: Grouped Ablation")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--input_jsonl", type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "loopbench_reconstructed_v2" / "test.jsonl"),
    )
    parser.add_argument("--landscape_csv", type=str, required=True,
                        help="Path to Exp 0 landscape_ranked.csv for SAC-based head selection")
    parser.add_argument("--subtasks", type=str, default="square_root,newtons_iteration",
                        help="Discovery-set subtask names")
    parser.add_argument("--max_examples_per_subtask", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--parallel_gpu_ids", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
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
    # Grouped ablation params
    parser.add_argument("--top_k_values", type=str, default="2,3,5,8",
                        help="Comma-separated k values for dose curve")
    parser.add_argument("--n_random_controls", type=int, default=10,
                        help="Number of random control permutations per k")
    parser.add_argument("--selection_criterion", type=str, default="sac",
                        choices=["sac", "dla_rp"], help="Which Exp 0 score to rank heads by")
    # Repetition thresholds
    parser.add_argument("--numerical_loop_min_repeated_span", type=int, default=500)
    parser.add_argument("--statement_loop_min_repeat_count", type=int, default=4)
    parser.add_argument("--numerical_same_digit_run_threshold", type=int, default=500)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Head selection from Exp 0 landscape                                          #
# --------------------------------------------------------------------------- #

def load_landscape_ranking(csv_path: str, criterion: str = "sac") -> List[Dict[str, Any]]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    col = "sac_mean" if criterion == "sac" else "dla_mean_repeat_prone"
    reverse = True if criterion == "sac" else False  # SAC: higher=more self-attending; DLA_RP: more negative=more suppression
    if criterion == "dla_rp":
        rows.sort(key=lambda r: float(r[col]))
    else:
        rows.sort(key=lambda r: float(r[col]), reverse=True)
    return rows


def select_top_k_heads(ranked: List[Dict], k: int) -> List[str]:
    return [r["head_label"] for r in ranked[:k]]


def build_random_control_groups(
    all_head_labels: List[str],
    top_k_labels: List[str],
    k: int,
    n_perms: int,
    rng: random.Random,
) -> List[List[str]]:
    pool = [h for h in all_head_labels if h not in top_k_labels]
    groups = []
    for _ in range(n_perms):
        groups.append(rng.sample(pool, min(k, len(pool))))
    return groups


def build_layer_matched_control_groups(
    ranked: List[Dict],
    top_k_labels: List[str],
    k: int,
    n_perms: int,
    rng: random.Random,
) -> List[List[str]]:
    layer_pools: Dict[int, List[str]] = defaultdict(list)
    for r in ranked:
        if r["head_label"] not in top_k_labels:
            layer_pools[int(r["layer_idx"])].append(r["head_label"])

    top_k_layers = []
    for r in ranked:
        if r["head_label"] in top_k_labels:
            top_k_layers.append(int(r["layer_idx"]))

    groups = []
    for _ in range(n_perms):
        group = []
        for layer in top_k_layers:
            pool = layer_pools.get(layer, [])
            if pool:
                group.append(rng.choice(pool))
        if len(group) == k:
            groups.append(group)
    return groups


def build_sac_matched_control_groups(
    ranked: List[Dict],
    top_k_labels: List[str],
    k: int,
    n_perms: int,
    rng: random.Random,
    sac_threshold: float = 0.5,
) -> List[List[str]]:
    high_sac_pool = [
        r["head_label"] for r in ranked
        if float(r["sac_mean"]) >= sac_threshold and r["head_label"] not in top_k_labels
    ]
    groups = []
    for _ in range(n_perms):
        if len(high_sac_pool) >= k:
            groups.append(rng.sample(high_sac_pool, k))
    return groups


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Worker: run a list of ablation conditions on one GPU                         #
# --------------------------------------------------------------------------- #

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

    output_path = Path(worker_output_path)
    dump_jsonl(output_path, all_results)

    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "conditions": [c["name"] for c in conditions],
        "row_count": len(all_results),
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

    ranked = load_landscape_ranking(args.landscape_csv, args.selection_criterion)
    all_head_labels = [r["head_label"] for r in ranked]
    k_values = [int(k) for k in args.top_k_values.split(",")]
    max_k = max(k_values)
    top_heads = select_top_k_heads(ranked, max_k)
    print(f"[Info] Top-{max_k} heads by {args.selection_criterion}: {top_heads}")

    rng = random.Random(args.seed)

    conditions: List[Dict[str, Any]] = []

    # Baseline (no ablation)
    conditions.append({"name": "baseline", "type": "baseline", "head_labels": [], "k": 0})

    # Top-k grouped ablation (dose curve)
    for k in k_values:
        group = top_heads[:k]
        conditions.append({
            "name": f"top{k}_grouped",
            "type": "top_k",
            "head_labels": group,
            "k": k,
        })

    # Individual head ablation (for additivity check)
    for h in top_heads:
        conditions.append({
            "name": f"individual_{h}",
            "type": "individual",
            "head_labels": [h],
            "k": 1,
        })

    # C1: Random k-group controls (at k=max_k)
    c1_groups = build_random_control_groups(all_head_labels, top_heads, max_k, args.n_random_controls, rng)
    for i, group in enumerate(c1_groups):
        conditions.append({
            "name": f"C1_random_{max_k}_{i:02d}",
            "type": "C1_random",
            "head_labels": group,
            "k": max_k,
        })

    # C2: Layer-matched controls (at k=max_k)
    c2_groups = build_layer_matched_control_groups(ranked, top_heads, max_k, args.n_random_controls, rng)
    for i, group in enumerate(c2_groups):
        conditions.append({
            "name": f"C2_layer_{max_k}_{i:02d}",
            "type": "C2_layer_matched",
            "head_labels": group,
            "k": max_k,
        })

    # C3: SAC-matched controls (at k=max_k)
    c3_groups = build_sac_matched_control_groups(ranked, top_heads, max_k, args.n_random_controls, rng)
    for i, group in enumerate(c3_groups):
        conditions.append({
            "name": f"C3_sac_{max_k}_{i:02d}",
            "type": "C3_sac_matched",
            "head_labels": group,
            "k": max_k,
        })

    n_conds = len(conditions)
    print(f"[Info] {n_conds} conditions total: 1 baseline, {len(k_values)} top-k, "
          f"{max_k} individual, {len(c1_groups)} C1, {len(c2_groups)} C2, {len(c3_groups)} C3")

    # Save run config
    write_json(output_dir / "run_config.json", {
        "model": args.model_name_or_path,
        "selection_criterion": args.selection_criterion,
        "top_heads": top_heads,
        "k_values": k_values,
        "n_random_controls": args.n_random_controls,
        "n_examples": len(rows),
        "subtasks": args.subtasks,
        "n_conditions": n_conds,
        "conditions": [{"name": c["name"], "type": c["type"], "head_labels": c["head_labels"], "k": c["k"]} for c in conditions],
    })

    # Distribute conditions across GPUs
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

    # Merge worker outputs
    all_rows: List[Dict] = []
    for wi in range(n_workers):
        wp = output_dir / f"_worker_{wi}_rows.jsonl"
        if wp.exists():
            all_rows.extend(load_jsonl(wp))
    dump_jsonl(output_dir / "all_results.jsonl", all_rows)

    # Compute per-condition statistics
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
            "condition": cn,
            "type": s["type"],
            "k": s["k"],
            "head_labels": s["head_labels"],
            "n_examples": s["n"],
            "n_repetitive": s["n_rep"],
            "loop_rate": round(loop_rate, 4),
        })
    write_csv(output_dir / "condition_stats.csv", stats_list)

    # Compute grouped ablation tests
    baseline_rate = 0.0
    for s in stats_list:
        if s["condition"] == "baseline":
            baseline_rate = s["loop_rate"]
            break

    # Per top-k: bootstrap test vs controls
    summary = {
        "model": args.model_name_or_path,
        "selection_criterion": args.selection_criterion,
        "top_heads": top_heads,
        "n_examples": len(rows),
        "baseline_loop_rate": baseline_rate,
        "dose_curve": {},
        "individual_rates": {},
        "control_rates": {"C1": [], "C2": [], "C3": []},
    }

    for s in stats_list:
        if s["type"] == "top_k":
            k = s["k"]
            summary["dose_curve"][str(k)] = {
                "loop_rate": s["loop_rate"],
                "n_rep": s["n_repetitive"],
                "n": s["n_examples"],
            }
        elif s["type"] == "individual":
            summary["individual_rates"][s["condition"].replace("individual_", "")] = s["loop_rate"]
        elif s["type"].startswith("C"):
            ctrl_type = s["type"].split("_")[0]
            summary["control_rates"][ctrl_type].append(s["loop_rate"])

    # Additivity analysis
    individual_sum = sum(summary["individual_rates"].values())
    for k_str, dose in summary["dose_curve"].items():
        top_k_indiv_sum = sum(
            summary["individual_rates"].get(h, 0)
            for h in top_heads[:int(k_str)]
        )
        dose["individual_sum"] = round(top_k_indiv_sum, 4)
        if top_k_indiv_sum > 0:
            dose["synergy_ratio"] = round(dose["loop_rate"] / top_k_indiv_sum, 4)
        else:
            dose["synergy_ratio"] = "inf" if dose["loop_rate"] > 0 else "nan"

    # Control comparison (permutation test for top-max_k vs controls)
    top_max_k_rate = summary["dose_curve"].get(str(max_k), {}).get("loop_rate", 0)
    for ctrl_type in ["C1", "C2", "C3"]:
        ctrl_rates = summary["control_rates"][ctrl_type]
        if ctrl_rates:
            n_exceed = sum(1 for cr in ctrl_rates if cr >= top_max_k_rate)
            p_val = (n_exceed + 1) / (len(ctrl_rates) + 1)
            summary[f"{ctrl_type}_permutation_p"] = round(p_val, 4)
            summary[f"{ctrl_type}_mean_rate"] = round(np.mean(ctrl_rates), 4)
        else:
            summary[f"{ctrl_type}_permutation_p"] = None
            summary[f"{ctrl_type}_mean_rate"] = None

    # Judgment
    has_signal = any(d["loop_rate"] > 0 for d in summary["dose_curve"].values())
    monotone = True
    prev = 0.0
    for k in sorted(int(x) for x in summary["dose_curve"]):
        rate = summary["dose_curve"][str(k)]["loop_rate"]
        if rate < prev:
            monotone = False
        prev = rate

    if not has_signal:
        judgment = "no_signal"
    elif has_signal and monotone:
        ctrl_sig = all(
            summary.get(f"{ct}_permutation_p", 1.0) is not None
            and summary.get(f"{ct}_permutation_p", 1.0) < 0.05
            for ct in ["C1", "C2", "C3"]
            if summary.get(f"{ct}_permutation_p") is not None
        )
        judgment = "distributed_signal_significant" if ctrl_sig else "distributed_signal_tentative"
    else:
        judgment = "non_monotone_signal"

    summary["judgment"] = judgment

    write_json(output_dir / "summary.json", summary)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Exp 2b Grouped Ablation — {args.model_name_or_path}")
    print("=" * 60)
    print(f"Selection: top-{max_k} by {args.selection_criterion}")
    print(f"Top heads: {top_heads}")
    print(f"Baseline loop rate: {baseline_rate:.1%}")
    print(f"\nDose curve:")
    for k in sorted(int(x) for x in summary["dose_curve"]):
        d = summary["dose_curve"][str(k)]
        print(f"  k={k}: loop_rate={d['loop_rate']:.1%} (indiv_sum={d['individual_sum']:.1%}, synergy={d['synergy_ratio']})")
    print(f"\nIndividual rates:")
    for h, rate in summary["individual_rates"].items():
        print(f"  {h}: {rate:.1%}")
    for ct in ["C1", "C2", "C3"]:
        p = summary.get(f"{ct}_permutation_p")
        m = summary.get(f"{ct}_mean_rate")
        if p is not None:
            print(f"\n{ct}: mean_rate={m:.1%}, p_vs_top={p:.4f}")
    print(f"\nJudgment: {judgment}")
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
