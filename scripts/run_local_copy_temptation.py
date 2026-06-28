#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Dict, IO, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.generation import create_backend
from cot_research.head_intervention import INTERVENTION_REGISTRY, MultiLayerHeadIntervention, resolve_head_targets
from cot_research.io_utils import dump_jsonl, write_csv, write_json
from cot_research.local_copy_temptation import (
    DEFAULT_K_VALUES,
    aggregate_flat_rows,
    attach_vs_baseline_deltas,
    build_condition_payload,
    build_local_copy_cases,
    build_prompt_prefix,
    build_prompt_step_metrics,
    build_summary_json,
    expected_phrase_token_ids,
    flatten_condition_rows,
    maybe_write_plots,
    parse_int_list,
    resolve_copy_append_target_token_ids,
    write_report,
)
from cot_research.model_utils import get_input_device_for_model
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run a local-copy temptation experiment to test whether a specified attention head suppresses copying "
            "from the most recent local context."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "local_copy_temptation" / "qwen3_1p7b_l0h3_20260408"),
    )
    parser.add_argument("--prompt_variant", type=str, default="default", choices=["default", "sharp_prev1"])
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--ablation_kind", type=str, default="zero", choices=list(INTERVENTION_REGISTRY.names()))
    parser.add_argument("--scale_kind", type=str, default="scale", choices=list(INTERVENTION_REGISTRY.names()))
    parser.add_argument("--scale_values", type=str, default="1.2,1.5")
    parser.add_argument("--recent_k_values", type=str, default="1,2,4,8")
    parser.add_argument(
        "--cot_input_jsonl",
        type=str,
        default=str(root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )

    parser.add_argument("--token_family_count", type=int, default=72)
    parser.add_argument("--phrase_family_count", type=int, default=72)
    parser.add_argument("--cot_family_count", type=int, default=72)
    parser.add_argument("--token_max_new_tokens", type=int, default=5)
    parser.add_argument("--phrase_max_new_tokens", type=int, default=8)
    parser.add_argument("--cot_max_new_tokens", type=int, default=96)

    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable multi-process multi-GPU execution by sharding examples across GPUs.",
    )
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--keep_worker_outputs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--save_prompt_text", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
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


def parse_scale_values(text: str) -> List[float]:
    if not text.strip():
        return [1.2]
    values: List[float] = []
    seen = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value <= 0.0:
            raise ValueError(f"All scale values must be positive, got {value}.")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("No scale values were parsed from --scale_values.")
    return values


def format_scale_label(scale: float) -> str:
    return f"scale_{scale:g}"


def build_condition_specs(args: argparse.Namespace, targets) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "label": "baseline",
            "kind": "identity",
            "scale": 1.0,
            "operations": None,
        }
    ]
    ablation_ops = INTERVENTION_REGISTRY.get_required(args.ablation_kind)(targets, {})
    specs.append(
        {
            "label": "ablation",
            "kind": args.ablation_kind,
            "scale": 0.0,
            "operations": ablation_ops,
        }
    )
    for scale in parse_scale_values(args.scale_values):
        scale_ops = INTERVENTION_REGISTRY.get_required(args.scale_kind)(targets, {"scale": scale})
        specs.append(
            {
                "label": format_scale_label(scale),
                "kind": args.scale_kind,
                "scale": float(scale),
                "operations": scale_ops,
            }
        )
    return specs


@torch.no_grad()
def forward_prompt_logits(
    model: torch.nn.Module,
    prompt_token_ids: Sequence[int],
    *,
    attn_modules: Sequence[torch.nn.Module],
    operations: Optional[List[Tuple[Any, float]]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    model_device = get_input_device_for_model(model)
    input_ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    debug: Dict[str, Any] = {}
    if operations:
        with MultiLayerHeadIntervention(list(attn_modules), operations) as hook_set:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            debug = hook_set.merged_debug_state()
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits[0, -1].detach(), debug


def generate_with_optional_intervention(
    *,
    backend,
    prompt_prefix: str,
    generation_config: GenerationConfig,
    attn_modules: Sequence[torch.nn.Module],
    operations: Optional[List[Tuple[Any, float]]],
) -> Tuple[Any, Dict[str, Any]]:
    debug: Dict[str, Any] = {}
    if operations:
        with MultiLayerHeadIntervention(list(attn_modules), operations) as hook_set:
            generation = backend.generate(prompt_prefix, generation_config)
            debug = hook_set.merged_debug_state()
    else:
        generation = backend.generate(prompt_prefix, generation_config)
    return generation, debug


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
    if not backend.supports_intervention or backend.model is None:
        raise ValueError("This script requires an HF backend with intervention support.")

    k_values = parse_int_list(args.recent_k_values, default=DEFAULT_K_VALUES)
    targets, attn_modules, layer_path = resolve_head_targets(backend.model, [args.head_label])
    condition_specs = build_condition_specs(args, targets)
    condition_order = [str(spec["label"]) for spec in condition_specs]
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
                target_token_info = resolve_copy_append_target_token_ids(expected_phrase_tokens, backend.tokenizer)
                metadata = dict(case.get("metadata") or {})
                forced_candidate_token_ids = None
                semantic_target_token_id = target_token_info.get("semantic_target_token_id")
                leading_whitespace_token_id = target_token_info.get("leading_whitespace_token_id")
                if bool(metadata.get("force_candidate_from_copy_append")) and semantic_target_token_id is not None:
                    forced_target_id = int(semantic_target_token_id)
                    forced_candidate_token_ids = {int(k): forced_target_id for k in k_values}
                generation_config = GenerationConfig(
                    system_prompt="",
                    assistant_prefix="",
                    max_new_tokens=int(case.get("max_new_tokens") or 0),
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    enable_thinking=False,
                    capture_step_scores=False,
                )

                example_seed = int(args.seed) + idx * 1009
                seed_everything(example_seed)
                baseline_logits, baseline_prompt_debug = forward_prompt_logits(
                    backend.model,
                    prompt_token_ids,
                    attn_modules=attn_modules,
                    operations=None,
                )
                vocab_size = int(baseline_logits.shape[-1])
                baseline_prompt_metrics = build_prompt_step_metrics(
                    baseline_logits,
                    prompt_token_ids,
                    k_values=k_values,
                    vocab_size=vocab_size,
                    seed=example_seed,
                    fixed_candidate_token_ids=forced_candidate_token_ids,
                )
                baseline_candidate_ids = {
                    int(k_str): int(metrics["candidate_token_id"])
                    for k_str, metrics in baseline_prompt_metrics.items()
                    if metrics.get("candidate_token_id") is not None
                }
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"prompt_setup_failed: {exc}"})
                continue

            row: Dict[str, Any] = {
                "example_id": example_id,
                "family": str(case.get("family") or ""),
                "case": dict(case),
            }
            if args.save_prompt_text:
                row["prompt_prefix"] = prompt_prefix

            condition_success = True
            for spec in condition_specs:
                label = str(spec["label"])
                try:
                    if label == "baseline":
                        current_logits = baseline_logits
                        prompt_metrics_by_k = baseline_prompt_metrics
                        prompt_debug = baseline_prompt_debug
                    else:
                        seed_everything(example_seed)
                        current_logits, prompt_debug = forward_prompt_logits(
                            backend.model,
                            prompt_token_ids,
                            attn_modules=attn_modules,
                            operations=spec["operations"],
                        )
                        prompt_metrics_by_k = build_prompt_step_metrics(
                            current_logits,
                            prompt_token_ids,
                            k_values=k_values,
                            vocab_size=vocab_size,
                            seed=example_seed,
                            fixed_candidate_token_ids=baseline_candidate_ids,
                        )

                    seed_everything(example_seed)
                    generation, generation_debug = generate_with_optional_intervention(
                        backend=backend,
                        prompt_prefix=prompt_prefix,
                        generation_config=generation_config,
                        attn_modules=attn_modules,
                        operations=spec["operations"],
                    )
                    payload = build_condition_payload(
                        case=case,
                        prompt_prefix=prompt_prefix,
                        prompt_token_ids=prompt_token_ids,
                        prompt_metrics_by_k=prompt_metrics_by_k,
                        logits_row=current_logits,
                        generation=generation,
                        tokenizer=backend.tokenizer,
                        label=label,
                        intervention_kind=str(spec["kind"]),
                        intervention_scale=float(spec["scale"]),
                        expected_phrase_tokens=expected_phrase_tokens,
                        semantic_target_token_id=semantic_target_token_id,
                        leading_whitespace_token_id=leading_whitespace_token_id,
                        debug={
                            "prompt_forward": prompt_debug,
                            "generation": generation_debug,
                        },
                    )
                    if not args.save_prompt_text:
                        payload.pop("prompt_prefix", None)
                    row[label] = payload
                except KeyboardInterrupt:
                    interrupted = True
                    condition_success = False
                    break
                except Exception as exc:
                    skipped_rows.append({"example_id": example_id, "reason": f"{label}_failed: {exc}"})
                    condition_success = False
                    break
            if interrupted:
                break
            if not condition_success:
                continue

            attach_vs_baseline_deltas(row, k_values=k_values)
            result_rows.append(row)
            append_jsonl_line(result_handle, row)
            if idx % max(args.print_every, 1) == 0:
                baseline = dict(row.get("baseline") or {})
                ablation = dict(row.get("ablation") or {})
                baseline_copy = dict(baseline.get("realized_local_copy_by_k") or {})
                ablation_copy = dict(ablation.get("realized_local_copy_by_k") or {})
                print(
                    f"[Info] pid={os.getpid()} processed={idx} kept={len(result_rows)} skipped={len(skipped_rows)} "
                    f"example_id={example_id} family={row['family']} "
                    f"baseline_recent_mass_k1={baseline.get('prompt_metrics_by_k', {}).get('1', {}).get('recent_mass', 0.0):.4f} "
                    f"ablation_recent_mass_k1={ablation.get('prompt_metrics_by_k', {}).get('1', {}).get('recent_mass', 0.0):.4f} "
                    f"baseline_copy_k1={baseline_copy.get('1')} ablation_copy_k1={ablation_copy.get('1')}"
                )
            del baseline_logits
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
    flat_rows = flatten_condition_rows(result_rows, condition_order=condition_order, k_values=k_values)
    write_csv(flat_rows_path, flat_rows)
    summary_rows = aggregate_flat_rows(flat_rows, k_values=k_values, condition_order=condition_order)
    write_csv(summary_rows_path, summary_rows)
    plot_paths = maybe_write_plots(
        summary_rows,
        plots_dir=plots_dir,
        k_values=k_values,
        condition_order=condition_order,
    )
    summary_json = build_summary_json(summary_rows=summary_rows, k_values=k_values, condition_order=condition_order)
    summary_json.update(
        {
            "output_dir": str(output_dir),
            "processed_examples": len(result_rows),
            "skipped_examples": len(skipped_rows),
            "skipped_rows": skipped_rows,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
            "model_name_or_path": args.model_name_or_path,
            "head_label": args.head_label,
            "scale_values": parse_scale_values(args.scale_values),
            "recent_k_values": [int(k) for k in k_values],
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": list(available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids),
            "parallel_workers": int(worker_count),
            "decoder_layer_path": layer_path,
            "plot_paths": plot_paths,
            "condition_order": list(condition_order),
        }
    )
    write_json(summary_path, summary_json)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "condition_order": list(condition_order),
            "recent_k_values": [int(k) for k in k_values],
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": list(available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids),
            "parallel_workers": int(worker_count),
            "decoder_layer_path": layer_path,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
        },
    )
    write_report(
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
    scale_values = parse_scale_values(args.scale_values)
    condition_order = ["baseline", "ablation"] + [format_scale_label(scale) for scale in scale_values]

    all_cases = build_local_copy_cases(
        token_count=args.token_family_count,
        phrase_count=args.phrase_family_count,
        cot_count=args.cot_family_count,
        cot_input_jsonl=args.cot_input_jsonl,
        token_max_new_tokens=args.token_max_new_tokens,
        phrase_max_new_tokens=args.phrase_max_new_tokens,
        cot_max_new_tokens=args.cot_max_new_tokens,
        seed=args.seed,
        variant=args.prompt_variant,
    )
    if not all_cases:
        write_json(output_dir / "summary.json", {"message": "No cases to process.", "processed_examples": 0})
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
        "[Info] Local-copy temptation experiment setup: "
        f"examples={len(all_cases)}, family_counts={family_counts}, cuda_available={torch_cuda_available}, "
        f"available_gpu_ids={available_gpu_ids}, parallel_enabled={parallel_enabled}, worker_count={worker_count}, "
        f"head_label={args.head_label}, prompt_variant={args.prompt_variant}, "
        f"scale_values={scale_values}, recent_k_values={k_values}"
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
                        if worker_ret.get("interrupted"):
                            interrupted = True
                except KeyboardInterrupt:
                    interrupted = True
                    raise
        except KeyboardInterrupt:
            interrupted = True
        finally:
            collected_rows: List[Dict[str, Any]] = []
            collected_skipped: List[Dict[str, str]] = []
            for path in worker_rows_paths:
                collected_rows.extend(read_jsonl_if_exists(path))
            for path in worker_skipped_paths:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        collected_skipped.extend(json.load(f))
            result_rows = collected_rows
            skipped_rows = collected_skipped
            if not args.keep_worker_outputs:
                for path in worker_rows_paths + worker_skipped_paths:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
    else:
        if args.gpu_id >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
        result_rows, skipped_rows, layer_path, interrupted = process_cases(
            all_cases,
            args,
            device_map_override={"": args.gpu_id} if args.gpu_id >= 0 else args.device_map,
            progress_desc="Local-copy temptation",
        )

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

    if interrupted:
        print("[Done] Local-copy temptation experiment interrupted; partial outputs written:")
    else:
        print("[Done] Local-copy temptation experiment finished:")
    print(f"- output_dir: {output_dir}")
    print(f"- processed_examples: {len(result_rows)}")
    print(f"- skipped_examples: {len(skipped_rows)}")
    print(f"- rows_jsonl: {output_dir / 'rows.jsonl'}")
    print(f"- summary_json: {output_dir / 'summary.json'}")
    print(f"- summary_csv: {output_dir / 'condition_family_summary.csv'}")
    print(f"- report_md: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
