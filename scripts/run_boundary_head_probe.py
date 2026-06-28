#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Dict, IO, List, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.boundary_probe import (
    STATE_ORDER,
    aggregate_boundary_rows,
    build_boundary_probe_cases,
    build_boundary_summary_json,
    build_direct_write_components,
    build_target_control_token_ids,
    compute_direct_write_stats,
    flatten_boundary_row_for_csv,
    format_scale_label,
    forward_prompt_with_capture,
    mean_logit_for_token_ids,
    maybe_write_boundary_plots,
    mean_prob_for_token_ids,
    parse_scale_values,
    top_k_from_logits,
    write_boundary_report,
)
from cot_research.generation import create_backend
from cot_research.head_intervention import resolve_head_targets
from cot_research.io_utils import dump_jsonl, write_csv, write_json
from cot_research.local_copy_temptation import (
    DEFAULT_K_VALUES,
    build_prompt_prefix,
    build_prompt_step_metrics,
    expected_phrase_token_ids,
    maybe_token_stats_from_logits,
    parse_int_list,
    resolve_copy_append_target_token_ids,
)
from cot_research.model_utils import get_input_device_for_model
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything, split_examples_contiguous
from cot_research.schemas import BackendConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run boundary-distribution probes for a single attention head, focusing on local-copy pressure at the "
            "critical next-token boundary rather than visible long-range repetition."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "boundary_head_probe" / "qwen3_4b_l0h1_20260414"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--head_label", type=str, default="L0H1")
    parser.add_argument("--scale_values", type=str, default="1.2,1.5")
    parser.add_argument("--recent_k_values", type=str, default="1,2,4,8")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--control_sample_size", type=int, default=8)
    parser.add_argument(
        "--cot_input_jsonl",
        type=str,
        default=str(root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--sharp_phrase_count", type=int, default=96)
    parser.add_argument("--wrong_tail_count", type=int, default=96)
    parser.add_argument("--control_count", type=int, default=96)
    parser.add_argument("--sharp_phrase_max_new_tokens", type=int, default=4)
    parser.add_argument("--wrong_tail_max_new_tokens", type=int, default=32)
    parser.add_argument("--control_max_new_tokens", type=int, default=4)
    parser.add_argument("--include_direct_write", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-process multi-GPU execution by sharding cases across GPUs.",
    )
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--keep_worker_outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=12)
    parser.add_argument("--save_prompt_text", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def append_jsonl_line(handle: IO[str] | None, row: Dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def build_condition_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {"label": "baseline", "scale": 1.0},
        {"label": "ablation", "scale": 0.0},
    ]
    for scale in parse_scale_values(args.scale_values):
        specs.append({"label": format_scale_label(scale), "scale": float(scale)})
    return specs


def _populate_k_metrics(
    row: Dict[str, Any],
    prompt_metrics_by_k: Dict[str, Dict[str, Any]],
    *,
    k_values: Sequence[int],
) -> None:
    for k in k_values:
        key = str(int(k))
        metrics = dict(prompt_metrics_by_k.get(key) or {})
        row[f"recent_mass_k{key}"] = float(metrics.get("recent_mass", 0.0))
        row[f"control_mass_k{key}"] = float(metrics.get("control_mass", 0.0))
        row[f"recent_gap_k{key}"] = float(metrics.get("recent_minus_control_gap", 0.0))
        row[f"recent_mass_delta_k{key}"] = None
        row[f"control_mass_delta_k{key}"] = None
        row[f"recent_gap_delta_k{key}"] = None


def _attach_baseline_deltas(
    row: Dict[str, Any],
    *,
    baseline_row: Dict[str, Any],
    k_values: Sequence[int],
) -> None:
    for field in [
        "target_prob",
        "target_logit",
        "target_rank",
        "target_control_prob_gap",
        "target_control_logit_gap",
        "eos_prob",
        "eos_logit",
        "eos_rank",
    ]:
        delta_key = f"{field}_delta"
        current = row.get(field)
        baseline = baseline_row.get(field)
        if current is None or baseline is None:
            row[delta_key] = None
        else:
            row[delta_key] = round(float(current) - float(baseline), 6)
    for k in k_values:
        key = str(int(k))
        for metric_name in ["recent_mass", "control_mass", "recent_gap"]:
            current = row.get(f"{metric_name}_k{key}")
            baseline = baseline_row.get(f"{metric_name}_k{key}")
            delta_key = f"{metric_name}_delta_k{key}"
            if current is None or baseline is None:
                row[delta_key] = None
            else:
                row[delta_key] = round(float(current) - float(baseline), 6)


def process_cases(
    cases: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
    stream_result_path: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], str, bool]:
    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map=device_map_override,
            load_in_half=args.load_in_half,
            use_fast_tokenizer=args.use_fast_tokenizer,
            use_safetensors=args.use_safetensors,
            local_files_only=args.local_files_only,
        )
    )
    if not backend.supports_intervention or backend.model is None or backend.tokenizer is None:
        raise ValueError("This script requires an HF backend with intervention support.")

    k_values = parse_int_list(args.recent_k_values, default=DEFAULT_K_VALUES)
    condition_specs = build_condition_specs(args)
    targets, attn_modules, layer_path = resolve_head_targets(backend.model, [args.head_label])
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one head target, got {len(targets)}.")
    target = targets[0]
    attn_module = attn_modules[target.layer_idx]
    model_device = get_input_device_for_model(backend.model)
    direct_write_components = None
    if args.include_direct_write:
        try:
            direct_write_components = build_direct_write_components(
                backend.model,
                layer_idx=target.layer_idx,
                head_idx=target.head_idx,
            )
        except Exception as exc:
            print(
                f"[Warn] direct_write disabled for {args.head_label}: {exc}"
            )

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    interrupted = False
    result_handle: IO[str] | None = None
    try:
        if stream_result_path:
            result_handle = open(stream_result_path, "w", encoding="utf-8")
        iterator = tqdm(cases, desc=progress_desc, dynamic_ncols=True, leave=False)
        for idx, case in enumerate(iterator, start=1):
            example_id = str(case.get("example_id") or f"case_{idx}")
            try:
                prompt_prefix = build_prompt_prefix(case, backend.tokenizer)
                prompt_token_ids = backend.encode(prompt_prefix)
                expected_phrase_tokens = expected_phrase_token_ids(case, backend.tokenizer)
                target_info = resolve_copy_append_target_token_ids(expected_phrase_tokens, backend.tokenizer)
                target_token_id = target_info.get("semantic_target_token_id")
                if target_token_id is None:
                    raise ValueError("No semantic target token found from copy_append_text.")
                vocab_size = int(backend.model.config.vocab_size)
                control_token_ids = build_target_control_token_ids(
                    prompt_token_ids=prompt_token_ids,
                    target_token_id=int(target_token_id),
                    vocab_size=vocab_size,
                    control_sample_size=args.control_sample_size,
                    seed=int(args.seed) + idx * 97,
                )
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"prompt_setup_failed: {exc}"})
                continue

            baseline_row: Dict[str, Any] | None = None
            for spec in condition_specs:
                try:
                    seed_everything(int(args.seed) + idx * 1009)
                    logits_row, debug = forward_prompt_with_capture(
                        backend.model,
                        prompt_token_ids,
                        attn_module=attn_module,
                        target=target,
                        scale=float(spec["scale"]),
                        model_device=model_device,
                    )
                    log_norm = torch.logsumexp(logits_row.float(), dim=-1)
                    target_stats = maybe_token_stats_from_logits(
                        logits_row.float(),
                        int(target_token_id),
                        log_norm=log_norm,
                    )
                    eos_token_id = getattr(backend.tokenizer, "eos_token_id", None)
                    eos_stats = maybe_token_stats_from_logits(
                        logits_row.float(),
                        None if eos_token_id is None else int(eos_token_id),
                        log_norm=log_norm,
                    )
                    prompt_metrics_by_k = build_prompt_step_metrics(
                        logits_row.float(),
                        prompt_token_ids,
                        k_values=k_values,
                        vocab_size=vocab_size,
                        seed=int(args.seed) + idx * 193,
                    )
                    top_k = top_k_from_logits(logits_row.float(), backend.tokenizer, k=args.top_k)
                    predicted_token_id = None if not top_k else int(top_k[0]["token_id"])
                    control_prob_mean = mean_prob_for_token_ids(
                        logits_row.float(),
                        control_token_ids,
                        log_norm=log_norm,
                    )
                    control_logit_mean = mean_logit_for_token_ids(logits_row.float(), control_token_ids)
                    direct_write_stats = {
                        "direct_write_target_logit": None,
                        "direct_write_control_logit_mean": None,
                        "direct_write_target_minus_control_logit": None,
                    }
                    if direct_write_components is not None:
                        direct_write_stats = compute_direct_write_stats(
                            components=direct_write_components,
                            head_vector_after=debug.get("captured_head_after"),
                            target_token_id=int(target_token_id),
                            control_token_ids=control_token_ids,
                        )
                    row: Dict[str, Any] = {
                        "example_id": example_id,
                        "pair_id": str(case.get("pair_id") or example_id),
                        "family": str(case.get("family") or ""),
                        "state": str(case.get("state") or case.get("family") or ""),
                        "condition": str(spec["label"]),
                        "intervention_scale": float(spec["scale"]),
                        "prompt_token_count": int(len(prompt_token_ids)),
                        "target_token_id": int(target_token_id),
                        "target_token_text": target_info.get("semantic_target_token_text") or "",
                        "control_token_ids": [int(token_id) for token_id in control_token_ids],
                        "control_token_texts": [
                            backend.tokenizer.decode([int(token_id)], skip_special_tokens=False)
                            for token_id in control_token_ids
                        ],
                        "top1_token_id": predicted_token_id,
                        "top1_token_text": ""
                        if predicted_token_id is None
                        else backend.tokenizer.decode([int(predicted_token_id)], skip_special_tokens=False),
                        "top1_is_target": None
                        if predicted_token_id is None
                        else bool(int(predicted_token_id) == int(target_token_id)),
                        "target_prob": None if target_stats is None else float(target_stats["prob"]),
                        "target_logit": None if target_stats is None else float(target_stats["logit"]),
                        "target_rank": None if target_stats is None else int(target_stats["rank"]),
                        "control_prob_mean": float(control_prob_mean),
                        "control_logit_mean": float(control_logit_mean),
                        "target_control_prob_gap": None
                        if target_stats is None
                        else float(float(target_stats["prob"]) - float(control_prob_mean)),
                        "target_control_logit_gap": None
                        if target_stats is None
                        else float(float(target_stats["logit"]) - float(control_logit_mean)),
                        "eos_prob": None if eos_stats is None else float(eos_stats["prob"]),
                        "eos_logit": None if eos_stats is None else float(eos_stats["logit"]),
                        "eos_rank": None if eos_stats is None else int(eos_stats["rank"]),
                        "target_prob_delta": None,
                        "target_logit_delta": None,
                        "target_rank_delta": None,
                        "target_control_prob_gap_delta": None,
                        "target_control_logit_gap_delta": None,
                        "eos_prob_delta": None,
                        "eos_logit_delta": None,
                        "eos_rank_delta": None,
                        "direct_write_target_logit": direct_write_stats["direct_write_target_logit"],
                        "direct_write_control_logit_mean": direct_write_stats["direct_write_control_logit_mean"],
                        "direct_write_target_minus_control_logit": direct_write_stats[
                            "direct_write_target_minus_control_logit"
                        ],
                        "top_k": top_k,
                        "prompt_metrics_by_k": prompt_metrics_by_k,
                        "debug": debug,
                    }
                    if args.save_prompt_text:
                        row["prompt_prefix"] = prompt_prefix
                    _populate_k_metrics(row, prompt_metrics_by_k, k_values=k_values)
                    if baseline_row is None:
                        baseline_row = dict(row)
                    else:
                        _attach_baseline_deltas(row, baseline_row=baseline_row, k_values=k_values)
                    result_rows.append(row)
                    append_jsonl_line(result_handle, row)
                except KeyboardInterrupt:
                    interrupted = True
                    break
                except Exception as exc:
                    skipped_rows.append({"example_id": example_id, "reason": f"{spec['label']}_failed: {exc}"})
                    break
            if interrupted:
                break
            if idx % max(int(args.print_every), 1) == 0:
                rows_for_example = [row for row in result_rows if str(row.get("example_id") or "") == example_id]
                rows_by_condition = {str(row.get("condition") or ""): row for row in rows_for_example}
                baseline = rows_by_condition.get("baseline")
                ablation = rows_by_condition.get("ablation")
                if baseline is not None and ablation is not None:
                    print(
                        f"[Info] pid={os.getpid()} processed={idx} kept_rows={len(result_rows)} skipped={len(skipped_rows)} "
                        f"example_id={example_id} family={case.get('family')} "
                        f"baseline_target_prob={float(baseline.get('target_prob') or 0.0):.4f} "
                        f"ablation_target_prob={float(ablation.get('target_prob') or 0.0):.4f} "
                        f"baseline_recent_mass_k1={float(baseline.get('recent_mass_k1') or 0.0):.4f} "
                        f"ablation_recent_mass_k1={float(ablation.get('recent_mass_k1') or 0.0):.4f}"
                    )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if result_handle is not None:
            result_handle.close()
    return result_rows, skipped_rows, layer_path, interrupted


def run_worker(
    worker_id: int,
    gpu_id: int,
    shard_cases: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_rows_path: str,
    worker_skipped_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    result_rows, skipped_rows, layer_path, interrupted = process_cases(
        shard_cases,
        args,
        device_map_override={"": gpu_id},
        progress_desc=f"Worker {worker_id} GPU{gpu_id}",
        stream_result_path=worker_rows_path,
    )
    write_json(worker_skipped_path, skipped_rows)
    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "row_count": len(result_rows),
        "skipped_count": len(skipped_rows),
        "worker_rows_path": worker_rows_path,
        "worker_skipped_path": worker_skipped_path,
        "layer_path": layer_path,
        "interrupted": interrupted,
    }


def read_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate_outputs(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    result_rows: List[Dict[str, Any]],
    skipped_rows: List[Dict[str, str]],
    layer_path: str,
    condition_order: Sequence[str],
    k_values: Sequence[int],
    parallel_enabled: bool,
    available_gpu_ids: Sequence[int],
    worker_count: int,
    interrupted: bool,
) -> None:
    rows_path = output_dir / "rows.jsonl"
    flat_rows_path = output_dir / "sample_condition_rows.csv"
    summary_rows_path = output_dir / "condition_family_summary.csv"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"
    report_path = output_dir / "report.md"
    plots_dir = output_dir / "plots"

    dump_jsonl(rows_path, result_rows)
    csv_rows = [flatten_boundary_row_for_csv(row, k_values=k_values) for row in result_rows]
    write_csv(flat_rows_path, csv_rows)
    summary_rows = aggregate_boundary_rows(result_rows, k_values=k_values, condition_order=condition_order)
    write_csv(summary_rows_path, summary_rows)
    plot_paths = maybe_write_boundary_plots(
        summary_rows,
        plots_dir=plots_dir,
        k_values=k_values,
        condition_order=condition_order,
    )
    summary_json = build_boundary_summary_json(
        summary_rows=summary_rows,
        k_values=k_values,
        condition_order=condition_order,
    )
    summary_json.update(
        {
            "output_dir": str(output_dir),
            "processed_case_conditions": int(len(result_rows)),
            "processed_examples": int(len({str(row.get('example_id') or '') for row in result_rows})),
            "processed_pairs": int(len({str(row.get('pair_id') or '') for row in result_rows})),
            "skipped_examples": len(skipped_rows),
            "skipped_rows": skipped_rows,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
            "model_name_or_path": args.model_name_or_path,
            "head_label": args.head_label,
            "scale_values": parse_scale_values(args.scale_values),
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": list(available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids),
            "parallel_workers": int(worker_count),
            "decoder_layer_path": layer_path,
            "condition_order": list(condition_order),
            "state_order": list(STATE_ORDER),
            "plot_paths": plot_paths,
        }
    )
    write_json(summary_path, summary_json)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "condition_order": list(condition_order),
            "state_order": list(STATE_ORDER),
            "recent_k_values": [int(k) for k in k_values],
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": list(available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids),
            "parallel_workers": int(worker_count),
            "decoder_layer_path": layer_path,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
        },
    )
    write_boundary_report(
        report_path,
        args=vars(args),
        summary_rows=summary_rows,
        k_values=k_values,
        condition_order=condition_order,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_int_list(args.recent_k_values, default=DEFAULT_K_VALUES)
    condition_order = ["baseline", "ablation"] + [
        format_scale_label(scale) for scale in parse_scale_values(args.scale_values)
    ]
    all_cases = build_boundary_probe_cases(
        sharp_phrase_count=args.sharp_phrase_count,
        wrong_tail_count=args.wrong_tail_count,
        control_count=args.control_count,
        cot_input_jsonl=args.cot_input_jsonl,
        sharp_phrase_max_new_tokens=args.sharp_phrase_max_new_tokens,
        wrong_tail_max_new_tokens=args.wrong_tail_max_new_tokens,
        control_max_new_tokens=args.control_max_new_tokens,
        seed=args.seed,
    )
    if not all_cases:
        write_json(output_dir / "summary.json", {"message": "No cases to process.", "processed_case_conditions": 0})
        write_json(output_dir / "run_config.json", {"args": vars(args)})
        (output_dir / "rows.jsonl").write_text("", encoding="utf-8")
        print("[Done] No cases to process.")
        return

    if args.parallel_gpu_ids.strip():
        available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    elif args.gpu_id >= 0:
        available_gpu_ids = [args.gpu_id]
    else:
        available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)

    torch_cuda_available = torch.cuda.is_available()
    can_parallel = args.parallel and torch_cuda_available and len(available_gpu_ids) > 1 and len(all_cases) > 1
    if can_parallel and args.parallel_workers > 0:
        worker_count = min(args.parallel_workers, len(available_gpu_ids), len(all_cases))
    elif can_parallel:
        worker_count = min(len(available_gpu_ids), len(all_cases))
    else:
        worker_count = 1
    parallel_enabled = can_parallel and worker_count > 1

    family_counts: Dict[str, int] = {}
    for case in all_cases:
        family = str(case.get("family") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1

    print(
        "[Info] Boundary head probe setup: "
        f"cases={len(all_cases)}, family_counts={family_counts}, cuda_available={torch_cuda_available}, "
        f"available_gpu_ids={available_gpu_ids}, parallel_enabled={parallel_enabled}, worker_count={worker_count}, "
        f"head_label={args.head_label}, scale_values={parse_scale_values(args.scale_values)}, recent_k_values={k_values}"
    )

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    layer_path = ""
    interrupted = False

    if parallel_enabled:
        worker_gpu_ids = available_gpu_ids[:worker_count]
        case_shards = split_examples_contiguous(all_cases, worker_count)
        worker_rows_paths: List[Path] = []
        worker_skipped_paths: List[Path] = []
        try:
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(case_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_cases in enumerate(case_shards):
                    gpu_id = worker_gpu_ids[worker_id]
                    worker_rows_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_rows.jsonl"
                    worker_skipped_path = output_dir / f"_worker_{worker_id}_gpu{gpu_id}_skipped.json"
                    futures.append(
                        pool.submit(
                            run_worker,
                            worker_id,
                            gpu_id,
                            shard_cases,
                            vars(args),
                            str(worker_rows_path),
                            str(worker_skipped_path),
                        )
                    )
                    worker_rows_paths.append(worker_rows_path)
                    worker_skipped_paths.append(worker_skipped_path)
                try:
                    for fut in as_completed(futures):
                        worker_ret = fut.result()
                        if not layer_path:
                            layer_path = str(worker_ret.get("layer_path") or "")
                        print(
                            f"[Info] Worker {worker_ret['worker_id']} GPU{worker_ret['gpu_id']} finished: "
                            f"rows={worker_ret['row_count']} skipped={worker_ret['skipped_count']}"
                        )
                        if bool(worker_ret.get("interrupted")):
                            interrupted = True
                except KeyboardInterrupt:
                    interrupted = True
                    for fut in futures:
                        fut.cancel()
                    raise
        except KeyboardInterrupt:
            interrupted = True
        finally:
            for path in worker_rows_paths:
                result_rows.extend(read_jsonl_if_exists(path))
            for path in worker_skipped_paths:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, list):
                        skipped_rows.extend(payload)
            if not args.keep_worker_outputs:
                for path in worker_rows_paths + worker_skipped_paths:
                    if path.exists():
                        path.unlink()
    else:
        device_map_override: Any = args.device_map
        if args.gpu_id >= 0:
            device_map_override = {"": args.gpu_id}
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu_id)
        result_rows, skipped_rows, layer_path, interrupted = process_cases(
            all_cases,
            args,
            device_map_override=device_map_override,
            progress_desc="Boundary head probe",
            stream_result_path=str(output_dir / "_stream_rows.jsonl"),
        )
        stream_path = output_dir / "_stream_rows.jsonl"
        if stream_path.exists() and not args.keep_worker_outputs:
            stream_path.unlink()

    aggregate_outputs(
        args=args,
        output_dir=output_dir,
        result_rows=result_rows,
        skipped_rows=skipped_rows,
        layer_path=layer_path,
        condition_order=condition_order,
        k_values=k_values,
        parallel_enabled=parallel_enabled,
        available_gpu_ids=available_gpu_ids,
        worker_count=worker_count,
        interrupted=interrupted,
    )

    print(
        "[Done] Boundary head probe finished: "
        f"processed_case_conditions={len(result_rows)} skipped_examples={len(skipped_rows)} interrupted={interrupted}"
    )


if __name__ == "__main__":
    main()
