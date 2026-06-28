#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.generation import create_backend
from cot_research.head_ablation import HeadSpec, SingleHeadAblationHook, list_all_heads
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig
from cot_research.text_analysis import analyze_text_keywords, extract_continuation_think_text


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Ablate first-layer heads before the first/last wait position and measure "
            "wait-logit changes plus no-wait continuations."
        )
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "reflection_heads_wait_prefix_ablation_qwen3_1p7b"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument(
        "--keep_worker_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    parser.add_argument("--max_stage1_tokens", type=int, default=8192)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait_keyword", type=str, default="wait")
    parser.add_argument(
        "--wait_token_text",
        type=str,
        default="Wait",
        help="Single-token text used for next-token wait-logit tracking at anchor prefixes.",
    )
    parser.add_argument("--wait_token_id", type=int, default=-1)
    parser.add_argument("--print_every", type=int, default=10)
    return parser.parse_args()


def select_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl)
    rows = rows[args.start_idx : args.start_idx + args.max_examples]
    if not rows:
        raise ValueError("No rows selected.")
    return rows


def build_backend(args: argparse.Namespace, device_map: Any):
    return create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map=device_map,
            load_in_half=args.load_in_half,
            use_fast_tokenizer=args.use_fast_tokenizer,
            use_safetensors=args.use_safetensors,
            local_files_only=args.local_files_only,
        )
    )


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        stage1_stop_string="</think>",
        max_stage1_tokens=args.max_stage1_tokens,
        max_new_tokens=args.max_stage1_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        capture_step_scores=False,
    )


def run_stage1_generation(backend, generation_config: GenerationConfig, prompt_prefix: str) -> Dict[str, Any]:
    result = backend.generate(prompt_prefix, generation_config, stop_strings=[generation_config.stage1_stop_string])
    return {
        "continuation": result.continuation,
        "generated_tokens": int(result.generated_tokens),
    }


def resolve_wait_token_id(backend, args: argparse.Namespace) -> int:
    if args.wait_token_id >= 0:
        return int(args.wait_token_id)
    token_ids = backend.encode(args.wait_token_text)
    if len(token_ids) != 1:
        raise ValueError(
            "wait_token_text must map to exactly one token when --wait_token_id is not set. "
            f"Got text={args.wait_token_text!r}, token_ids={token_ids}"
        )
    return int(token_ids[0])


def analyze_wait_positions(continuation: str, wait_keyword: str) -> Dict[str, Any]:
    think_text = extract_continuation_think_text(continuation)
    stats = analyze_text_keywords(think_text, [wait_keyword])
    positions = list(stats["positions"])
    return {
        "think_text": think_text,
        "wait_count": int(stats["count"]),
        "has_wait": bool(stats["hit"]),
        "positions": positions,
    }


def build_anchor_record(
    *,
    example_id: Any,
    question: Optional[str],
    prompt_prefix: str,
    continuation: str,
    generated_tokens: int,
    wait_keyword: str,
    wait_token_text: str,
    wait_token_logits: Dict[str, float],
) -> List[Dict[str, Any]]:
    wait_stats = analyze_wait_positions(continuation, wait_keyword)
    if not wait_stats["has_wait"]:
        return []
    think_text = str(wait_stats["think_text"])
    positions = list(wait_stats["positions"])
    anchors: List[Tuple[str, Dict[str, Any]]] = [
        ("first_wait", positions[0]),
        ("last_wait", positions[-1]),
    ]
    rows: List[Dict[str, Any]] = []
    for anchor_kind, pos in anchors:
        anchor_start = int(pos["start"])
        anchor_end = int(pos["end"])
        anchor_prefix = prompt_prefix + think_text[:anchor_start]
        rows.append(
            {
                "example_id": example_id,
                "question": question,
                "generated_tokens": generated_tokens,
                "baseline_wait_count": int(wait_stats["wait_count"]),
                "baseline_think_text": think_text,
                "anchor_kind": anchor_kind,
                "anchor_start": anchor_start,
                "anchor_end": anchor_end,
                "anchor_wait_text": think_text[anchor_start:anchor_end],
                "anchor_prefix": anchor_prefix,
                "anchor_prefix_think_text": think_text[:anchor_start],
                "baseline_wait_logit": float(wait_token_logits[anchor_kind]),
                "wait_token_text": wait_token_text,
            }
        )
    return rows


def run_baseline(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    gpu_id: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, str, List[HeadSpec]]:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    backend = build_backend(args, {"": gpu_id})
    generation_config = build_generation_config(args)
    wait_token_id = resolve_wait_token_id(backend, args)
    all_heads, _, layer_path = list_all_heads(backend.model)
    layer0_heads = [head for head in all_heads if head.layer_idx == 0]
    baseline_rows: List[Dict[str, Any]] = []
    anchor_rows: List[Dict[str, Any]] = []
    iterator = tqdm(rows, desc="Baseline stage1", dynamic_ncols=True)
    for idx, row in enumerate(iterator):
        prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
        seed_everything(args.seed + idx)
        generation = run_stage1_generation(backend, generation_config, prompt_prefix)
        wait_stats = analyze_wait_positions(generation["continuation"], args.wait_keyword)
        question = row.get("question") or row.get("problem")
        wait_logit_lookup: Dict[str, float] = {}
        if wait_stats["has_wait"]:
            first_pos = int(wait_stats["positions"][0]["start"])
            prefix_before_first = prompt_prefix + str(wait_stats["think_text"])[:first_pos]
            wait_logit_lookup["first_wait"] = float(
                backend.next_token_stats(prefix_before_first, [wait_token_id])[str(wait_token_id)]["logit"]
            )
            last_pos = int(wait_stats["positions"][-1]["start"])
            prefix_before_last = prompt_prefix + str(wait_stats["think_text"])[:last_pos]
            wait_logit_lookup["last_wait"] = float(
                backend.next_token_stats(prefix_before_last, [wait_token_id])[str(wait_token_id)]["logit"]
            )
        out = {
            "example_id": row.get("id") or row.get("example_id") or idx,
            "question": question,
            "generated_tokens": generation["generated_tokens"],
            "wait_count": int(wait_stats["wait_count"]),
            "has_wait": bool(wait_stats["has_wait"]),
            "first_wait_pos": int(wait_stats["positions"][0]["start"]) if wait_stats["has_wait"] else None,
            "last_wait_pos": int(wait_stats["positions"][-1]["start"]) if wait_stats["has_wait"] else None,
            "baseline_wait_logit_before_first_wait": wait_logit_lookup.get("first_wait"),
            "baseline_wait_logit_before_last_wait": wait_logit_lookup.get("last_wait"),
            "prompt_prefix": prompt_prefix,
        }
        baseline_rows.append(out)
        if wait_stats["has_wait"]:
            anchor_rows.extend(
                build_anchor_record(
                    example_id=out["example_id"],
                    question=question,
                    prompt_prefix=prompt_prefix,
                    continuation=generation["continuation"],
                    generated_tokens=generation["generated_tokens"],
                    wait_keyword=args.wait_keyword,
                    wait_token_text=args.wait_token_text,
                    wait_token_logits=wait_logit_lookup,
                )
            )
        if (idx + 1) % max(args.print_every, 1) == 0:
            print(
                f"[Info] baseline processed={idx + 1} examples_with_wait="
                f"{sum(1 for item in baseline_rows if item['has_wait'])} anchors={len(anchor_rows)}"
            )
    return baseline_rows, anchor_rows, wait_token_id, layer_path, layer0_heads


def split_heads_round_robin(heads: List[HeadSpec], num_buckets: int) -> List[List[HeadSpec]]:
    buckets: List[List[HeadSpec]] = [[] for _ in range(num_buckets)]
    for idx, head in enumerate(heads):
        buckets[idx % num_buckets].append(head)
    return [bucket for bucket in buckets if bucket]


def summarize_head_anchor_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["anchor_kind"]), str(row["head_label"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (anchor_kind, head_label), items in grouped.items():
        total = len(items)
        suppressed = sum(1 for item in items if not bool(item["ablated_has_wait"]))
        baseline_logits = [float(item["baseline_wait_logit"]) for item in items]
        ablated_logits = [float(item["ablated_wait_logit"]) for item in items]
        deltas = [float(item["delta_ablated_minus_baseline"]) for item in items]
        ablated_wait_counts = [int(item["ablated_wait_count"]) for item in items]
        first = items[0]
        summary_rows.append(
            {
                "anchor_kind": anchor_kind,
                "head_label": head_label,
                "layer_idx": int(first["layer_idx"]),
                "head_idx": int(first["head_idx"]),
                "total_anchor_examples": total,
                "no_wait_examples": suppressed,
                "no_wait_rate": round(suppressed / total, 6) if total else 0.0,
                "mean_baseline_wait_logit": round(sum(baseline_logits) / total, 6) if total else 0.0,
                "mean_ablated_wait_logit": round(sum(ablated_logits) / total, 6) if total else 0.0,
                "mean_delta_ablated_minus_baseline": round(sum(deltas) / total, 6) if total else 0.0,
                "mean_ablated_wait_count": round(sum(ablated_wait_counts) / total, 6) if total else 0.0,
            }
        )
    summary_rows.sort(key=lambda item: (item["anchor_kind"], -item["no_wait_rate"], item["mean_delta_ablated_minus_baseline"]))
    return summary_rows


def run_ablation_worker(
    worker_id: int,
    gpu_id: int,
    head_specs: List[Dict[str, Any]],
    anchor_rows: List[Dict[str, Any]],
    wait_token_id: int,
    args_dict: Dict[str, Any],
    worker_details_path: str,
    worker_no_wait_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    backend = build_backend(args, {"": gpu_id})
    generation_config = build_generation_config(args)
    _, attn_modules, layer_path = list_all_heads(backend.model)

    detail_rows: List[Dict[str, Any]] = []
    no_wait_rows: List[Dict[str, Any]] = []
    for head_payload in head_specs:
        layer_idx = int(head_payload["layer_idx"])
        head_idx = int(head_payload["head_idx"])
        head_label = str(head_payload["head_label"])
        num_heads = int(head_payload["num_heads"])
        head_dim = int(head_payload["head_dim"])
        attn_module = attn_modules[layer_idx]
        for ex_idx, anchor in enumerate(anchor_rows):
            seed_everything(args.seed + ex_idx)
            with SingleHeadAblationHook(attn_module, head_idx, num_heads, head_dim):
                ablated_wait_logit = float(
                    backend.next_token_stats(anchor["anchor_prefix"], [wait_token_id])[str(wait_token_id)]["logit"]
                )
            seed_everything(args.seed + ex_idx)
            with SingleHeadAblationHook(attn_module, head_idx, num_heads, head_dim):
                generation = run_stage1_generation(backend, generation_config, anchor["anchor_prefix"])
            wait_stats = analyze_wait_positions(generation["continuation"], args.wait_keyword)
            detail_row = {
                "example_id": anchor["example_id"],
                "anchor_kind": anchor["anchor_kind"],
                "head_label": head_label,
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "baseline_wait_logit": float(anchor["baseline_wait_logit"]),
                "ablated_wait_logit": ablated_wait_logit,
                "delta_ablated_minus_baseline": round(ablated_wait_logit - float(anchor["baseline_wait_logit"]), 6),
                "baseline_wait_count": int(anchor["baseline_wait_count"]),
                "ablated_wait_count": int(wait_stats["wait_count"]),
                "ablated_has_wait": bool(wait_stats["has_wait"]),
                "gpu_id": gpu_id,
            }
            detail_rows.append(detail_row)
            if not wait_stats["has_wait"]:
                no_wait_rows.append(
                    {
                        "example_id": anchor["example_id"],
                        "question": anchor["question"],
                        "anchor_kind": anchor["anchor_kind"],
                        "head_label": head_label,
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "baseline_wait_count": int(anchor["baseline_wait_count"]),
                        "baseline_wait_logit": float(anchor["baseline_wait_logit"]),
                        "ablated_wait_logit": ablated_wait_logit,
                        "delta_ablated_minus_baseline": round(ablated_wait_logit - float(anchor["baseline_wait_logit"]), 6),
                        "anchor_prefix_think_text": anchor["anchor_prefix_think_text"],
                        "ablated_generated_continuation": generation["continuation"],
                        "modified_think_text": str(anchor["anchor_prefix_think_text"]) + str(wait_stats["think_text"]),
                    }
                )
        print(
            f"[Info] worker={worker_id} gpu={gpu_id} head={head_label} "
            f"processed_anchors={len(anchor_rows)}"
        )

    write_csv(worker_details_path, detail_rows)
    dump_jsonl(worker_no_wait_path, no_wait_rows)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "worker_details_path": worker_details_path,
        "worker_no_wait_path": worker_no_wait_path,
        "head_count": len(head_specs),
        "decoder_layer_path": layer_path,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_csv_path = output_dir / "baseline_wait_anchors.csv"
    details_csv_path = output_dir / "wait_logit_delta_details.csv"
    summary_csv_path = output_dir / "head_wait_prefix_summary.csv"
    no_wait_jsonl_path = output_dir / "no_wait_cot_cases.jsonl"
    summary_json_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"

    rows = select_rows(args)
    gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    if len(gpu_ids) < 1:
        raise ValueError("At least one GPU is required.")

    baseline_gpu = gpu_ids[0]
    print(
        "[Info] Starting first-layer reflection-head prefix ablation experiment: "
        f"model={args.model_name_or_path}, examples={len(rows)}, gpus={gpu_ids}, "
        f"max_stage1_tokens={args.max_stage1_tokens}, do_sample={args.do_sample}"
    )

    baseline_rows, anchor_rows, wait_token_id, layer_path, layer0_heads = run_baseline(rows, args, baseline_gpu)
    write_csv(baseline_csv_path, anchor_rows)
    if not anchor_rows:
        write_json(
            summary_json_path,
            {
                "message": "No baseline wait anchors found.",
                "baseline_examples": len(baseline_rows),
                "anchor_examples": 0,
                "decoder_layer_path": layer_path,
            },
        )
        write_json(
            run_config_path,
            {
                "args": vars(args),
                "gpu_ids": gpu_ids,
                "decoder_layer_path": layer_path,
                "wait_token_id": wait_token_id,
            },
        )
        print("[Done] No wait anchors found in baseline.")
        return

    worker_count = min(len(gpu_ids), len(layer0_heads))
    if args.parallel_workers > 0:
        worker_count = min(worker_count, args.parallel_workers)
    worker_gpu_ids = gpu_ids[:worker_count]
    head_buckets = split_heads_round_robin(layer0_heads, worker_count)

    worker_detail_paths: List[Path] = []
    worker_no_wait_paths: List[Path] = []
    worker_returns: List[Dict[str, Any]] = []
    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(head_buckets), mp_context=mp_ctx) as pool:
        futures = []
        for worker_id, head_bucket in enumerate(head_buckets):
            gpu_id = worker_gpu_ids[worker_id]
            head_payloads = [
                {
                    "head_label": head.label,
                    "layer_idx": head.layer_idx,
                    "head_idx": head.head_idx,
                    "num_heads": head.num_heads,
                    "head_dim": head.head_dim,
                }
                for head in head_bucket
            ]
            worker_details_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_details.csv"
            worker_no_wait_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_no_wait.jsonl"
            futures.append(
                pool.submit(
                    run_ablation_worker,
                    worker_id,
                    gpu_id,
                    head_payloads,
                    anchor_rows,
                    wait_token_id,
                    vars(args),
                    str(worker_details_path),
                    str(worker_no_wait_path),
                )
            )
        for fut in as_completed(futures):
            worker_ret = fut.result()
            worker_returns.append(worker_ret)
            worker_detail_paths.append(Path(worker_ret["worker_details_path"]))
            worker_no_wait_paths.append(Path(worker_ret["worker_no_wait_path"]))
            print(
                f"[Info] worker={worker_ret['worker_id']} gpu={worker_ret['gpu_id']} finished "
                f"heads={worker_ret['head_count']}"
            )

    import csv

    detail_rows: List[Dict[str, Any]] = []
    no_wait_rows: List[Dict[str, Any]] = []
    for worker_ret in sorted(worker_returns, key=lambda item: int(item["worker_id"])):
        with open(worker_ret["worker_details_path"], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                detail_rows.append(
                    {
                        "example_id": row["example_id"],
                        "anchor_kind": row["anchor_kind"],
                        "head_label": row["head_label"],
                        "layer_idx": int(row["layer_idx"]),
                        "head_idx": int(row["head_idx"]),
                        "baseline_wait_logit": float(row["baseline_wait_logit"]),
                        "ablated_wait_logit": float(row["ablated_wait_logit"]),
                        "delta_ablated_minus_baseline": float(row["delta_ablated_minus_baseline"]),
                        "baseline_wait_count": int(row["baseline_wait_count"]),
                        "ablated_wait_count": int(row["ablated_wait_count"]),
                        "ablated_has_wait": row["ablated_has_wait"].lower() == "true",
                        "gpu_id": int(row["gpu_id"]),
                    }
                )
        no_wait_rows.extend(load_jsonl(worker_ret["worker_no_wait_path"]))

    detail_rows.sort(key=lambda item: (item["anchor_kind"], item["layer_idx"], item["head_idx"], str(item["example_id"])))
    summary_rows = summarize_head_anchor_rows(detail_rows)
    no_wait_rows.sort(key=lambda item: (item["anchor_kind"], item["head_label"], str(item["example_id"])))
    write_csv(details_csv_path, detail_rows)
    write_csv(summary_csv_path, summary_rows)
    dump_jsonl(no_wait_jsonl_path, no_wait_rows)

    write_json(
        summary_json_path,
        {
            "model_name_or_path": args.model_name_or_path,
            "input_jsonl": args.input_jsonl,
            "baseline_examples": len(baseline_rows),
            "baseline_examples_with_wait": sum(1 for row in baseline_rows if row["has_wait"]),
            "anchor_rows": len(anchor_rows),
            "wait_token_text": args.wait_token_text,
            "wait_token_id": wait_token_id,
            "parallel_gpu_ids": worker_gpu_ids,
            "decoder_layer_path": layer_path,
            "analyzed_heads": [head.label for head in layer0_heads],
            "top_rows": summary_rows[:12],
            "no_wait_cot_cases": len(no_wait_rows),
        },
    )
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "worker_gpu_ids": worker_gpu_ids,
            "decoder_layer_path": layer_path,
            "wait_token_id": wait_token_id,
            "analyzed_heads": [head.label for head in layer0_heads],
        },
    )

    if not args.keep_worker_outputs:
        for path in worker_detail_paths + worker_no_wait_paths:
            if path.exists():
                path.unlink()

    print("[Done] Outputs written:")
    print(f"- baseline_wait_anchors_csv: {baseline_csv_path}")
    print(f"- wait_logit_delta_details_csv: {details_csv_path}")
    print(f"- head_wait_prefix_summary_csv: {summary_csv_path}")
    print(f"- no_wait_cot_cases_jsonl: {no_wait_jsonl_path}")
    print(f"- summary_json: {summary_json_path}")


if __name__ == "__main__":
    main()
