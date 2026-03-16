#!/usr/bin/env python3
"""Single-head attention ablation for Qwen CoT continuation self-correction analysis.

Input JSONL fields (required):
- id
- correct_answer
- wrong_answer

Recommended field:
- question

Optional legacy field:
- prompt_prefix

This script runs:
1) Stage-1 generation from question until first </think>.
2) Replace the answer near end of <think> with wrong_answer and remove </think>.
3) Baseline continuation generation from tampered prefix.
4) Single-head ablation runs (one head masked at a time).
5) Analysis focused on text inside <think>...</think>.

Outputs:
- ablation_no_reflect_wrong_only.jsonl
- head_summary.csv
- run_config.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from find_wait_head_lib.ablation import (
    MultiHeadAblationHookSet,
    SingleHeadAblationHook,
    filter_heads,
    list_all_heads,
    parse_head_label,
)
from find_wait_head_lib.constants import ABLATION_POSITION, DEFAULT_KEYWORDS
from find_wait_head_lib.io_utils import (
    build_export_row,
    dump_prepared_examples,
    ensure_required_fields,
    init_summary_entry,
    init_wait_logit_entry,
    load_jsonl,
    load_prepared_examples,
    merge_summary_stats,
    merge_wait_logit_stats,
    parse_parallel_gpu_ids,
    select_examples,
    should_keep_filtered_row,
    split_head_labels_round_robin,
    update_wait_logit_stats,
    update_summary,
    write_wait_logit_per_example_csvs,
    write_wait_logit_ranking_csv,
    write_summary_csv,
)
from find_wait_head_lib.model_utils import (
    generate_continuation,
    get_next_token_logit,
    load_model_with_retries,
    resolve_target_token_id,
)
from find_wait_head_lib.parallel_utils import run_ablation_worker, run_prepare_and_baseline_worker
from find_wait_head_lib.pipeline import analyze_generation, prepare_example_prefix
from find_wait_head_lib.text_utils import parse_keywords


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    default_input = root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"
    default_output = root_dir / "outputs" / "self_correction_full"

    parser = argparse.ArgumentParser(description="Qwen CoT continuation self-correction head ablation")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--input_jsonl", type=str, default=str(default_input))
    parser.add_argument("--output_dir", type=str, default=str(default_output))
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument(
        "--parallel_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Parallelize ablation heads across multiple GPUs via multiprocessing.",
    )
    parser.add_argument(
        "--parallel_gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU ids for parallel ablation, e.g. '0,1,2,3'. Default: all visible GPUs.",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=0,
        help="Number of worker processes for parallel ablation. 0 means auto by selected parallel mode.",
    )
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="auto",
        choices=["auto", "head", "example", "none"],
        help="Parallelization strategy: head-split, example-split, auto-select, or none.",
    )
    parser.add_argument(
        "--baseline_gpu_id",
        type=int,
        default=0,
        help="GPU id used for stage1+baseline when --parallel_heads is enabled.",
    )
    parser.add_argument(
        "--keep_worker_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep intermediate worker jsonl files under output_dir.",
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Before closing </think>, include your interim result in \\\\boxed{}."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--stage1_stop_string", type=str, default="</think>")
    parser.add_argument("--strict_tamper", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--max_stage1_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action="store_true", help="Default is deterministic (greedy).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model/tokenizer only from local cache (recommended for restricted network).",
    )
    parser.add_argument("--load_in_half", dest="load_in_half", action="store_true", default=True)
    parser.add_argument("--no_load_in_half", dest="load_in_half", action="store_false")

    parser.add_argument(
        "--keywords",
        type=str,
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated keyword list for self-correction signal detection.",
    )
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument(
        "--head_spec",
        type=str,
        default="",
        help="Optional head subset, e.g. 'L0H0,L1H3'. Default: all heads.",
    )
    parser.add_argument(
        "--joint_ablation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, ablate all heads selected by --head_spec simultaneously (instead of one-by-one).",
    )
    parser.add_argument(
        "--ablate_head",
        type=str,
        default="L0H0",
        help="Single head to ablate (format: LxHy). This script now defaults to one-head ablation.",
    )
    parser.add_argument(
        "--print_cot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print Stage1 CoT, tampered prefix, baseline continuation, and ablated continuation to terminal.",
    )

    parser.add_argument("--debug", action="store_true", help="Run 1 sample + 1 specified head with verbose logs.")
    parser.add_argument("--debug_example_id", type=str, default="")
    parser.add_argument("--debug_head", type=str, default="", help="Head in format LxHy, required for --debug.")
    parser.add_argument(
        "--wait_token_text",
        type=str,
        default="Wait",
        help="Target token text for wait-logit tracking. Must map to exactly one token if --wait_token_id<0.",
    )
    parser.add_argument(
        "--wait_token_id",
        type=int,
        default=-1,
        help="Optional explicit token id for wait-logit tracking. If set >=0, overrides --wait_token_text.",
    )
    parser.add_argument(
        "--legacy_use_prompt_prefix_first",
        action="store_true",
        help="If set, use record['prompt_prefix'] directly when present, instead of two-stage auto-tamper from question.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    keywords = parse_keywords(args.keywords)
    if not keywords:
        raise ValueError("Keyword list is empty.")

    all_examples = load_jsonl(args.input_jsonl)
    if not all_examples:
        raise ValueError("Input dataset is empty.")
    ensure_required_fields(all_examples)
    examples = select_examples(all_examples, args)

    if args.debug and not args.debug_head:
        raise ValueError("--debug requires --debug_head (e.g., L0H0).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "ablation_no_reflect_wrong_only.jsonl"
    summary_path = output_dir / "head_summary.csv"
    config_path = output_dir / "run_config.json"
    wait_logit_path = output_dir / "head_wait_token_logits.jsonl"
    wait_logit_rank_path = output_dir / "head_wait_token_logit_ranking.csv"
    wait_logit_per_example_dir = output_dir / "wait_logit_by_example"

    available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    torch_cuda_available = torch.cuda.is_available()
    torch_cuda_count = torch.cuda.device_count() if torch_cuda_available else 0
    print(
        "[Info] GPU visibility: "
        f"cuda_available={torch_cuda_available}, "
        f"torch_visible_gpu_count={torch_cuda_count}, "
        f"active_parallel_gpu_ids={available_gpu_ids}"
    )
    can_parallel = (
        args.parallel_heads
        and (not args.debug)
        and torch_cuda_available
        and len(available_gpu_ids) > 1
    )
    if can_parallel and args.baseline_gpu_id not in available_gpu_ids:
        raise ValueError(
            f"--baseline_gpu_id={args.baseline_gpu_id} not in active GPUs {available_gpu_ids}."
        )

    baseline_device_map_override: Optional[Any] = None
    if can_parallel:
        # Keep stage1+baseline on a dedicated GPU to avoid auto-sharding/offload slowdown.
        baseline_device_map_override = {"": args.baseline_gpu_id}

    print(f"[Info] Loading model: {args.model_name_or_path}")
    model, tokenizer, loaded_use_fast, loaded_use_safetensors = load_model_with_retries(
        args,
        device_map_override=baseline_device_map_override,
    )

    all_heads, attn_modules, layer_path = list_all_heads(model)
    head_lookup = {h.label: h for h in all_heads}

    if args.debug:
        layer_idx, head_idx = parse_head_label(args.debug_head)
        key = (layer_idx, head_idx)
        debug_candidate = {(h.layer_idx, h.head_idx): h for h in all_heads}
        if key not in debug_candidate:
            raise ValueError(f"Debug head {args.debug_head} not found in model.")
        selected_heads = [debug_candidate[key]]
    else:
        selected_heads = filter_heads(all_heads, args.head_spec)

    wait_token_id = resolve_target_token_id(tokenizer, args.wait_token_text, args.wait_token_id)
    wait_token_decoded = tokenizer.decode([wait_token_id], skip_special_tokens=False)
    print(
        "[Info] Wait-token tracking: "
        f"id={wait_token_id}, text_arg={args.wait_token_text!r}, decoded={wait_token_decoded!r}"
    )

    if args.joint_ablation:
        print(
            "[Info] Joint ablation enabled: "
            f"simultaneously masking {len(selected_heads)} heads."
        )

    stats: Dict[str, Dict[str, Any]] = defaultdict(init_summary_entry)
    wait_logit_stats: Dict[str, Dict[str, Any]] = defaultdict(init_wait_logit_entry)
    baseline_rows: Dict[str, Dict[str, Any]] = {}
    baseline_wait_logits: Dict[str, float] = {}
    prepared_examples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    stage_parallel_enabled = can_parallel and args.parallel_mode != "none" and len(examples) > 1
    stage_parallel_workers = 0
    if stage_parallel_enabled:
        stage_gpu_pool = [g for g in available_gpu_ids if g != args.baseline_gpu_id]
        if not stage_gpu_pool:
            stage_gpu_pool = list(available_gpu_ids)
        if args.parallel_workers > 0:
            stage_worker_count = min(args.parallel_workers, len(stage_gpu_pool), len(examples))
        else:
            stage_worker_count = min(len(stage_gpu_pool), len(examples))
        if stage_worker_count <= 0:
            stage_worker_count = 1
        stage_parallel_workers = stage_worker_count

        print(
            "[Info] Parallel Stage1+Baseline enabled: "
            f"workers={stage_worker_count}, gpus={stage_gpu_pool[:stage_worker_count]}"
        )
        stage_gpu_ids = stage_gpu_pool[:stage_worker_count]
        example_shards: List[List[Dict[str, Any]]] = [[] for _ in range(stage_worker_count)]
        for i, ex in enumerate(examples):
            example_shards[i % stage_worker_count].append(ex)
        example_shards = [x for x in example_shards if x]

        shard_paths: List[Path] = []
        worker_prepared_paths: List[Path] = []
        worker_baseline_paths: List[Path] = []
        try:
            for worker_id, shard in enumerate(example_shards):
                shard_path = output_dir / f"_stage_examples_worker_{worker_id}.jsonl"
                with open(shard_path, "w", encoding="utf-8") as f:
                    for ex in shard:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                shard_paths.append(shard_path)

            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(example_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_path in enumerate(shard_paths):
                    gpu_id = stage_gpu_ids[worker_id]
                    worker_prepared = output_dir / f"_stage_worker_{worker_id}_gpu{gpu_id}_prepared.jsonl"
                    worker_baseline = output_dir / f"_stage_worker_{worker_id}_gpu{gpu_id}_baseline.jsonl"
                    futures.append(
                        pool.submit(
                            run_prepare_and_baseline_worker,
                            worker_id,
                            gpu_id,
                            str(shard_path),
                            vars(args),
                            keywords,
                            str(worker_prepared),
                            str(worker_baseline),
                            wait_token_id,
                        )
                    )

                for fut in as_completed(futures):
                    worker_ret = fut.result()
                    merge_summary_stats(stats, worker_ret["stats"])
                    worker_prepared_paths.append(Path(worker_ret["worker_prepared_output_path"]))
                    worker_baseline_paths.append(Path(worker_ret["worker_baseline_output_path"]))
                    print(
                        f"[Info] PrepWorker {worker_ret['worker_id']} (GPU {worker_ret['gpu_id']}) done: "
                        f"prepared={worker_ret['prepared_count']}, skipped={worker_ret['skipped_count']}"
                    )

            for p in sorted(worker_prepared_paths):
                if p.exists():
                    prepared_examples.extend(load_prepared_examples(str(p)))

            for p in sorted(worker_baseline_paths):
                if not p.exists():
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        ex_id = str(obj["example_id"])
                        baseline_wait_logits[ex_id] = float(obj["baseline_wait_logit"])
                        baseline_rows[ex_id] = obj["baseline_row"]

            if args.print_cot:
                for ex, prep_info in prepared_examples:
                    ex_id = str(ex["id"])
                    print("\n" + "=" * 80)
                    print(f"[CoT-Stage1] example_id={ex.get('id')}")
                    print(prep_info.get("stage1_full_text"))
                    print("-" * 80)
                    print(f"[CoT-TamperedPrefix] example_id={ex.get('id')}")
                    print(prep_info.get("tampered_prefix"))
                    baseline = baseline_rows.get(ex_id)
                    if baseline is not None:
                        print("-" * 80)
                        print(f"[CoT-BaselineContinuation] example_id={ex.get('id')}")
                        print(baseline.get("generated_continuation"))
                    if ex_id in baseline_wait_logits:
                        print(
                            f"[WaitLogit-Baseline] example_id={ex.get('id')} "
                            f"value={baseline_wait_logits[ex_id]:.6f}"
                        )
        finally:
            if not args.keep_worker_outputs:
                for p in shard_paths:
                    if p.exists():
                        p.unlink()
                for p in worker_prepared_paths:
                    if p.exists():
                        p.unlink()
                for p in worker_baseline_paths:
                    if p.exists():
                        p.unlink()
    else:
        for ex in tqdm(examples, desc="Stage1+Tamper", dynamic_ncols=True):
            prep_info = prepare_example_prefix(
                model=model,
                tokenizer=tokenizer,
                example=ex,
                args=args,
            )
            if not prep_info.get("ok", False):
                print(f"[Warn] Skip example {ex.get('id')} due to preparation failure: {prep_info.get('reason')}")
                continue
            prepared_examples.append((ex, prep_info))
            if args.print_cot:
                print("\n" + "=" * 80)
                print(f"[CoT-Stage1] example_id={ex.get('id')}")
                print(prep_info.get("stage1_full_text"))
                print("-" * 80)
                print(f"[CoT-TamperedPrefix] example_id={ex.get('id')}")
                print(prep_info.get("tampered_prefix"))

        for ex, prep_info in tqdm(prepared_examples, desc="Baseline", dynamic_ncols=True):
            baseline_wait_logit = get_next_token_logit(
                model=model,
                tokenizer=tokenizer,
                prompt_prefix=prep_info["tampered_prefix"],
                target_token_id=wait_token_id,
            )
            continuation, full_text, gen_tokens = generate_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt_prefix=prep_info["tampered_prefix"],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            row = analyze_generation(
                example=ex,
                prep_info=prep_info,
                condition="baseline",
                continuation=continuation,
                full_text=full_text,
                generated_tokens=gen_tokens,
                keywords=keywords,
                head=None,
                hook=None,
            )
            update_summary(stats, row)
            baseline_rows[str(ex["id"])] = row
            baseline_wait_logits[str(ex["id"])] = baseline_wait_logit
            if args.print_cot:
                print("\n" + "=" * 80)
                print(f"[CoT-BaselineContinuation] example_id={ex.get('id')}")
                print(row["generated_continuation"])
                print(f"[WaitLogit-Baseline] example_id={ex.get('id')} value={baseline_wait_logit:.6f}")

    parallel_disabled_reason = ""
    if not args.parallel_heads:
        parallel_disabled_reason = "--no-parallel_heads is set"
    elif args.parallel_mode == "none":
        parallel_disabled_reason = "--parallel_mode none is set"
    elif args.debug:
        parallel_disabled_reason = "--debug is set"
    elif not torch_cuda_available:
        parallel_disabled_reason = "torch.cuda.is_available() is False"
    elif len(available_gpu_ids) <= 1:
        parallel_disabled_reason = (
            f"only {len(available_gpu_ids)} visible GPU id ({available_gpu_ids})"
        )

    if can_parallel and prepared_examples:
        if args.parallel_mode == "none":
            parallel_execution_mode = "none"
        elif args.parallel_mode == "head":
            parallel_execution_mode = "head"
        elif args.parallel_mode == "example":
            parallel_execution_mode = "example"
        else:
            # auto mode:
            # - single-head or joint-ablation => example-split (head-split is meaningless).
            # - multi-head => pick the split dimension with larger exploitable parallelism.
            num_heads_for_parallel = len(selected_heads) if not args.joint_ablation else 1
            num_examples_for_parallel = len(prepared_examples)
            if args.joint_ablation:
                parallel_execution_mode = "example"
            elif num_heads_for_parallel == 1:
                parallel_execution_mode = "example" if num_examples_for_parallel > 1 else "none"
            elif num_heads_for_parallel >= len(available_gpu_ids):
                parallel_execution_mode = "head"
            elif num_examples_for_parallel >= len(available_gpu_ids):
                parallel_execution_mode = "example"
            elif num_heads_for_parallel > 1:
                parallel_execution_mode = "head"
            elif num_examples_for_parallel > 1:
                parallel_execution_mode = "example"
            else:
                parallel_execution_mode = "none"
    else:
        parallel_execution_mode = "none"

    if args.joint_ablation and parallel_execution_mode == "head":
        print("[Warn] joint_ablation is incompatible with head-split mode; fallback to example-split mode.")
        parallel_execution_mode = "example" if can_parallel and prepared_examples else "none"

    use_parallel_heads = parallel_execution_mode == "head"
    use_parallel_examples = parallel_execution_mode == "example"
    use_parallel_any = parallel_execution_mode in {"head", "example"}

    if use_parallel_any:
        print(
            "[Info] Parallel ablation enabled: "
            f"mode={parallel_execution_mode}, gpus={available_gpu_ids}, "
            f"selected_heads={len(selected_heads)}, prepared_examples={len(prepared_examples)}"
        )
    else:
        if parallel_disabled_reason:
            print(f"[Info] Parallel ablation disabled: {parallel_disabled_reason}")
        else:
            print("[Info] Parallel ablation disabled by auto strategy.")
        print("[Info] Using single-process ablation loop.")

    config = {
        "model_name_or_path": args.model_name_or_path,
        "input_jsonl": os.path.abspath(args.input_jsonl),
        "num_examples_raw": len(examples),
        "num_examples_prepared": len(prepared_examples),
        "system_prompt": args.system_prompt,
        "assistant_prefix": args.assistant_prefix,
        "stage1_stop_string": args.stage1_stop_string,
        "strict_tamper": args.strict_tamper,
        "max_stage1_tokens": args.max_stage1_tokens,
        "max_new_tokens": args.max_new_tokens,
        "device_map": args.device_map,
        "parallel_heads": args.parallel_heads,
        "parallel_mode": args.parallel_mode,
        "stage_parallel_enabled": stage_parallel_enabled,
        "stage_parallel_workers": stage_parallel_workers,
        "joint_ablation": args.joint_ablation,
        "parallel_execution_mode": parallel_execution_mode,
        "use_parallel_heads": use_parallel_heads,
        "use_parallel_examples": use_parallel_examples,
        "parallel_gpu_ids": available_gpu_ids,
        "parallel_workers": args.parallel_workers,
        "baseline_gpu_id": args.baseline_gpu_id,
        "keep_worker_outputs": args.keep_worker_outputs,
        "baseline_device_map_override": baseline_device_map_override,
        "enable_thinking": args.enable_thinking,
        "use_fast_tokenizer": args.use_fast_tokenizer,
        "use_safetensors": args.use_safetensors,
        "local_files_only": args.local_files_only,
        "loaded_use_fast_tokenizer": loaded_use_fast,
        "loaded_use_safetensors": loaded_use_safetensors,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "ablation_position": ABLATION_POSITION,
        "decoder_layer_path": layer_path,
        "num_layers": len(attn_modules),
        "num_total_heads": len(all_heads),
        "num_selected_heads": len(selected_heads),
        "selected_heads": [h.label for h in selected_heads],
        "ablate_head": selected_heads[0].label if len(selected_heads) == 1 else None,
        "wait_token_text": args.wait_token_text,
        "wait_token_id": wait_token_id,
        "wait_token_decoded": wait_token_decoded,
        "keywords": keywords,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if not prepared_examples:
        print("[Warn] No examples available after stage1+tamper preparation. Writing empty outputs.")
        write_summary_csv(summary_path, {}, head_lookup)
        write_wait_logit_ranking_csv(wait_logit_rank_path, {}, head_lookup)
        raw_path.touch()
        wait_logit_path.touch()
        wait_logit_per_example_dir.mkdir(parents=True, exist_ok=True)
        print("\n[Done] Outputs written:")
        print(f"- Filtered ablation JSONL:  {raw_path}")
        print("- Filtered row count:       0")
        print(f"- Wait logit JSONL:         {wait_logit_path}")
        print(f"- Wait logit ranking CSV:   {wait_logit_rank_path}")
        print(f"- Wait logit per-example:   {wait_logit_per_example_dir}")
        print(f"- Head summary CSV:        {summary_path}")
        print(f"- Run config:              {config_path}")
        print(f"- Ablation position:       {ABLATION_POSITION}")
        return

    if use_parallel_any:
        # Free baseline model memory before spawning per-GPU workers.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    kept_rows = 0
    if use_parallel_heads:
        head_labels = [h.label for h in selected_heads]
        if args.parallel_workers > 0:
            worker_count = min(args.parallel_workers, len(available_gpu_ids), len(head_labels))
        else:
            worker_count = min(len(available_gpu_ids), len(head_labels))

        worker_gpu_ids = available_gpu_ids[:worker_count]
        head_splits = split_head_labels_round_robin(head_labels, worker_count)
        if not head_splits:
            raise ValueError("Parallel ablation has no head split to run.")

        prepared_jsonl_path = output_dir / "_prepared_examples_for_workers.jsonl"
        dump_prepared_examples(prepared_jsonl_path, prepared_examples)

        worker_output_paths: List[Path] = []
        worker_wait_logit_paths: List[Path] = []
        try:
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(head_splits), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, head_chunk in enumerate(head_splits):
                    gpu_id = worker_gpu_ids[worker_id]
                    worker_output = output_dir / f"_worker_{worker_id}_gpu{gpu_id}.jsonl"
                    worker_wait_logit = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_wait_logits.jsonl"
                    futures.append(
                        pool.submit(
                            run_ablation_worker,
                            worker_id,
                            gpu_id,
                            head_chunk,
                            str(prepared_jsonl_path),
                            vars(args),
                            keywords,
                            str(worker_output),
                            str(worker_wait_logit),
                            wait_token_id,
                            baseline_wait_logits,
                        )
                    )

                for fut in as_completed(futures):
                    worker_ret = fut.result()
                    merge_summary_stats(stats, worker_ret["stats"])
                    merge_wait_logit_stats(wait_logit_stats, worker_ret["wait_logit_stats"])
                    kept_rows += int(worker_ret["kept_rows"])
                    worker_output_paths.append(Path(worker_ret["worker_output_path"]))
                    worker_wait_logit_paths.append(Path(worker_ret["worker_wait_logit_path"]))
                    print(
                        f"[Info] Worker {worker_ret['worker_id']} (GPU {worker_ret['gpu_id']}) done: "
                        f"heads={worker_ret['head_count']}, kept_rows={worker_ret['kept_rows']}"
                    )

            with open(raw_path, "w", encoding="utf-8") as raw_f:
                for p in sorted(worker_output_paths):
                    if not p.exists():
                        continue
                    with open(p, "r", encoding="utf-8") as wf:
                        for line in wf:
                            raw_f.write(line)
            with open(wait_logit_path, "w", encoding="utf-8") as wait_f:
                for p in sorted(worker_wait_logit_paths):
                    if not p.exists():
                        continue
                    with open(p, "r", encoding="utf-8") as wf:
                        for line in wf:
                            wait_f.write(line)
        finally:
            if not args.keep_worker_outputs:
                for p in worker_output_paths:
                    if p.exists():
                        p.unlink()
                for p in worker_wait_logit_paths:
                    if p.exists():
                        p.unlink()
                if prepared_jsonl_path.exists():
                    prepared_jsonl_path.unlink()
    elif use_parallel_examples:
        head_labels = [h.label for h in selected_heads]
        if args.parallel_workers > 0:
            worker_count = min(args.parallel_workers, len(available_gpu_ids), len(prepared_examples))
        else:
            worker_count = min(len(available_gpu_ids), len(prepared_examples))
        if worker_count <= 0:
            worker_count = 1

        worker_gpu_ids = available_gpu_ids[:worker_count]
        example_shards: List[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = [[] for _ in range(worker_count)]
        for i, item in enumerate(prepared_examples):
            example_shards[i % worker_count].append(item)
        example_shards = [x for x in example_shards if x]
        if not example_shards:
            raise ValueError("Parallel example mode has no shard to run.")

        shard_paths: List[Path] = []
        worker_output_paths = []
        worker_wait_logit_paths = []
        try:
            for worker_id, shard in enumerate(example_shards):
                shard_path = output_dir / f"_prepared_examples_worker_{worker_id}.jsonl"
                dump_prepared_examples(shard_path, shard)
                shard_paths.append(shard_path)

            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(example_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_path in enumerate(shard_paths):
                    gpu_id = worker_gpu_ids[worker_id]
                    worker_output = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_ex.jsonl"
                    worker_wait_logit = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_ex_wait_logits.jsonl"
                    futures.append(
                        pool.submit(
                            run_ablation_worker,
                            worker_id,
                            gpu_id,
                            head_labels,
                            str(shard_path),
                            vars(args),
                            keywords,
                            str(worker_output),
                            str(worker_wait_logit),
                            wait_token_id,
                            baseline_wait_logits,
                        )
                    )

                for fut in as_completed(futures):
                    worker_ret = fut.result()
                    merge_summary_stats(stats, worker_ret["stats"])
                    merge_wait_logit_stats(wait_logit_stats, worker_ret["wait_logit_stats"])
                    kept_rows += int(worker_ret["kept_rows"])
                    worker_output_paths.append(Path(worker_ret["worker_output_path"]))
                    worker_wait_logit_paths.append(Path(worker_ret["worker_wait_logit_path"]))
                    print(
                        f"[Info] Worker {worker_ret['worker_id']} (GPU {worker_ret['gpu_id']}) done: "
                        f"shard_examples={len(example_shards[worker_ret['worker_id']])}, "
                        f"kept_rows={worker_ret['kept_rows']}"
                    )

            with open(raw_path, "w", encoding="utf-8") as raw_f:
                for p in sorted(worker_output_paths):
                    if not p.exists():
                        continue
                    with open(p, "r", encoding="utf-8") as wf:
                        for line in wf:
                            raw_f.write(line)
            with open(wait_logit_path, "w", encoding="utf-8") as wait_f:
                for p in sorted(worker_wait_logit_paths):
                    if not p.exists():
                        continue
                    with open(p, "r", encoding="utf-8") as wf:
                        for line in wf:
                            wait_f.write(line)
        finally:
            if not args.keep_worker_outputs:
                for p in worker_output_paths:
                    if p.exists():
                        p.unlink()
                for p in worker_wait_logit_paths:
                    if p.exists():
                        p.unlink()
                for p in shard_paths:
                    if p.exists():
                        p.unlink()
    else:
        with open(raw_path, "w", encoding="utf-8") as raw_f, open(wait_logit_path, "w", encoding="utf-8") as wait_f:
            if args.joint_ablation:
                joint_head_label = "JOINT[" + ",".join(h.label for h in selected_heads) + "]"
                desc = f"Ablation {joint_head_label}"
                for ex, prep_info in tqdm(prepared_examples, desc=desc, dynamic_ncols=True, leave=False):
                    with MultiHeadAblationHookSet(attn_modules=attn_modules, heads=selected_heads):
                        ablated_wait_logit = get_next_token_logit(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_prefix=prep_info["tampered_prefix"],
                            target_token_id=wait_token_id,
                        )
                    with MultiHeadAblationHookSet(attn_modules=attn_modules, heads=selected_heads):
                        continuation, full_text, gen_tokens = generate_continuation(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_prefix=prep_info["tampered_prefix"],
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )

                    row = analyze_generation(
                        example=ex,
                        prep_info=prep_info,
                        condition="multi_head_joint_ablation",
                        continuation=continuation,
                        full_text=full_text,
                        generated_tokens=gen_tokens,
                        keywords=keywords,
                        head=None,
                        hook=None,
                    )
                    row["head_label"] = joint_head_label
                    row["layer_idx"] = None
                    row["head_idx"] = None
                    update_summary(stats, row)

                    baseline_wait_logit = float(baseline_wait_logits[str(ex["id"])])
                    update_wait_logit_stats(wait_logit_stats, joint_head_label, baseline_wait_logit, ablated_wait_logit)
                    wait_f.write(
                        json.dumps(
                            {
                                "example_id": ex["id"],
                                "head_label": joint_head_label,
                                "layer_idx": None,
                                "head_idx": None,
                                "ablated_heads": [h.label for h in selected_heads],
                                "logitbaseline_wait_token": baseline_wait_logit,
                                "logitablated_wait_token": ablated_wait_logit,
                                "delta_ablated_minus_baseline": float(ablated_wait_logit - baseline_wait_logit),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if args.print_cot:
                        print("\n" + "=" * 80)
                        print(
                            f"[CoT-AblatedContinuation] example_id={ex.get('id')} "
                            f"head={joint_head_label}"
                        )
                        print(row["generated_continuation"])
                        print(
                            "[WaitLogit] "
                            f"example_id={ex.get('id')} head={joint_head_label} "
                            f"baseline={baseline_wait_logit:.6f} "
                            f"ablated={ablated_wait_logit:.6f} "
                            f"delta={ablated_wait_logit - baseline_wait_logit:.6f}"
                        )

                    if should_keep_filtered_row(row):
                        raw_f.write(json.dumps(build_export_row(row), ensure_ascii=False) + "\n")
                        kept_rows += 1
            else:
                for head in selected_heads:
                    attn_module = attn_modules[head.layer_idx]
                    desc = f"Ablation {head.label}"
                    for ex, prep_info in tqdm(prepared_examples, desc=desc, dynamic_ncols=True, leave=False):
                        with SingleHeadAblationHook(
                            attn_module=attn_module,
                            head_idx=head.head_idx,
                            num_heads=head.num_heads,
                            head_dim=head.head_dim,
                        ):
                            ablated_wait_logit = get_next_token_logit(
                                model=model,
                                tokenizer=tokenizer,
                                prompt_prefix=prep_info["tampered_prefix"],
                                target_token_id=wait_token_id,
                            )
                        with SingleHeadAblationHook(
                            attn_module=attn_module,
                            head_idx=head.head_idx,
                            num_heads=head.num_heads,
                            head_dim=head.head_dim,
                        ) as hook:
                            continuation, full_text, gen_tokens = generate_continuation(
                                model=model,
                                tokenizer=tokenizer,
                                prompt_prefix=prep_info["tampered_prefix"],
                                max_new_tokens=args.max_new_tokens,
                                do_sample=args.do_sample,
                                temperature=args.temperature,
                                top_p=args.top_p,
                            )

                        row = analyze_generation(
                            example=ex,
                            prep_info=prep_info,
                            condition="single_head_ablation",
                            continuation=continuation,
                            full_text=full_text,
                            generated_tokens=gen_tokens,
                            keywords=keywords,
                            head=head,
                            hook=hook,
                        )
                        update_summary(stats, row)
                        baseline_wait_logit = float(baseline_wait_logits[str(ex["id"])])
                        update_wait_logit_stats(wait_logit_stats, head.label, baseline_wait_logit, ablated_wait_logit)
                        wait_f.write(
                            json.dumps(
                                {
                                    "example_id": ex["id"],
                                    "head_label": head.label,
                                    "layer_idx": head.layer_idx,
                                    "head_idx": head.head_idx,
                                    "logitbaseline_wait_token": baseline_wait_logit,
                                    "logitablated_wait_token": ablated_wait_logit,
                                    "delta_ablated_minus_baseline": float(ablated_wait_logit - baseline_wait_logit),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        if args.print_cot:
                            print("\n" + "=" * 80)
                            print(
                                f"[CoT-AblatedContinuation] example_id={ex.get('id')} "
                                f"head={head.label}"
                            )
                            print(row["generated_continuation"])
                            print(
                                "[WaitLogit] "
                                f"example_id={ex.get('id')} head={head.label} "
                                f"baseline={baseline_wait_logit:.6f} "
                                f"ablated={ablated_wait_logit:.6f} "
                                f"delta={ablated_wait_logit - baseline_wait_logit:.6f}"
                            )

                        if should_keep_filtered_row(row):
                            raw_f.write(json.dumps(build_export_row(row), ensure_ascii=False) + "\n")
                            kept_rows += 1

                        if args.debug:
                            baseline = baseline_rows[str(ex["id"])]
                            print("\n" + "=" * 80)
                            print(f"[Debug] Example id: {ex['id']}")
                            print(f"[Debug] Head: {head.label}")
                            print(f"[Debug] Hook calls: {hook.call_count}")
                            print(
                                "[Debug] Hook first-call abs-mean before/after: "
                                f"{hook.first_call_abs_mean_before} -> {hook.first_call_abs_mean_after}"
                            )
                            print("-" * 80)
                            print("[Baseline] outcome:", baseline["outcome"])
                            print("[Baseline] final_boxed:", baseline["final_boxed_answer"])
                            print("[Baseline] keyword_hit:", baseline["self_correction_keyword_hit"])
                            print("[Baseline] matched_keywords:", baseline["matched_keywords"])
                            print("[Baseline] think_text:\n", baseline["think_text"])
                            print("-" * 80)
                            print("[Ablated] outcome:", row["outcome"])
                            print("[Ablated] final_boxed:", row["final_boxed_answer"])
                            print("[Ablated] keyword_hit:", row["self_correction_keyword_hit"])
                            print("[Ablated] matched_keywords:", row["matched_keywords"])
                            print("[Ablated] think_text:\n", row["think_text"])
                            print("-" * 80)
                            print("[Ablated] tamper_method:", row["tamper_method"])
                            print(
                                "[Ablated] tampered_from -> tampered_to:",
                                row["tampered_from"],
                                "->",
                                row["tampered_to"],
                            )
                            print("-" * 80)
                            print("[Stage1] full_text:\n", row["stage1_full_text"])
                            print("-" * 80)
                            print("[Ablated] full_text:\n", row["full_text"])

    write_summary_csv(summary_path, dict(stats), head_lookup)
    write_wait_logit_ranking_csv(wait_logit_rank_path, dict(wait_logit_stats), head_lookup)
    per_example_count = write_wait_logit_per_example_csvs(wait_logit_path, wait_logit_per_example_dir)

    print("\n[Done] Outputs written:")
    print(f"- Filtered ablation JSONL:  {raw_path}")
    print(f"- Filtered row count:       {kept_rows}")
    print(f"- Wait logit JSONL:         {wait_logit_path}")
    print(f"- Wait logit ranking CSV:   {wait_logit_rank_path}")
    print(f"- Wait logit per-example:   {wait_logit_per_example_dir} ({per_example_count} files)")
    print(f"- Parallel ablation mode:   {parallel_execution_mode}")
    print(f"- Head summary CSV:        {summary_path}")
    print(f"- Run config:              {config_path}")
    print(f"- Ablation position:       {ABLATION_POSITION}")


if __name__ == "__main__":
    main()
