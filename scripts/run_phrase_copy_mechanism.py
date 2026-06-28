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
from cot_research.head_intervention import resolve_head_targets
from cot_research.io_utils import dump_jsonl, write_csv, write_json
from cot_research.local_copy_temptation import (
    _decode_token,
    build_matched_control_token_ids,
    build_prompt_prefix,
    expected_phrase_token_ids,
    maybe_token_stats_from_logits,
    resolve_copy_append_target_token_ids,
)
from cot_research.model_utils import AttentionHeadSpec, get_input_device_for_model
from cot_research.ov_circuit_analysis import extract_head_ov_components
from cot_research.phrase_copy_mechanism import (
    PHRASE_STATE_ORDER,
    aggregate_phrase_mechanism_rows,
    build_phrase_mechanism_summary_json,
    build_sharp_phrase_mechanism_cases,
    write_phrase_mechanism_report,
)
from cot_research.runtime_utils import parse_parallel_gpu_ids, seed_everything, split_examples_contiguous
from cot_research.schemas import BackendConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run a phrase-only local-copy mechanism experiment with matched-control and position-shift states. "
            "The script measures both prompt-boundary logits and the target head's direct OV write onto the "
            "continuation token."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(root_dir / "outputs" / "phrase_copy_mechanism" / "qwen3_1p7b_l0h3_20260408"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--ablation_kind", type=str, default="zero")
    parser.add_argument("--scale_values", type=str, default="1.2,1.5")
    parser.add_argument("--phrase_count", type=int, default=96)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--control_sample_size", type=int, default=8)
    parser.add_argument(
        "--cot_input_jsonl",
        type=str,
        default=str(root_dir / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )

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


class SingleHeadPreOProjHook:
    def __init__(self, attn_module: torch.nn.Module, target: AttentionHeadSpec, scale: float) -> None:
        self.attn_module = attn_module
        self.target = target
        self.scale = float(scale)
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.captured_before: Optional[torch.Tensor] = None
        self.captured_after: Optional[torch.Tensor] = None

    def __enter__(self) -> "SingleHeadPreOProjHook":
        if not hasattr(self.attn_module, "o_proj"):
            raise ValueError("Attention module has no o_proj; cannot attach prompt-boundary capture hook.")
        self.handle = self.attn_module.o_proj.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _pre_hook(self, module: torch.nn.Module, args: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return None
        start = int(self.target.head_idx * self.target.head_dim)
        end = int((self.target.head_idx + 1) * self.target.head_dim)
        x_out = x.clone()
        self.captured_before = x[..., start:end].detach()[0, -1].float().cpu()
        x_out[..., start:end] = x_out[..., start:end] * self.scale
        self.captured_after = x_out[..., start:end].detach()[0, -1].float().cpu()
        if len(args) == 1:
            return (x_out,)
        return (x_out, *args[1:])


@torch.no_grad()
def forward_prompt_with_capture(
    model: torch.nn.Module,
    prompt_token_ids: Sequence[int],
    *,
    attn_module: torch.nn.Module,
    target: AttentionHeadSpec,
    scale: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    model_device = get_input_device_for_model(model)
    input_ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    with SingleHeadPreOProjHook(attn_module, target, scale) as hook:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    debug = {
        "capture_scale": float(scale),
        "captured_head_before_abs_mean": None
        if hook.captured_before is None
        else float(hook.captured_before.abs().mean().item()),
        "captured_head_after_abs_mean": None
        if hook.captured_after is None
        else float(hook.captured_after.abs().mean().item()),
        "captured_head_before": None
        if hook.captured_before is None
        else [float(value) for value in hook.captured_before.tolist()],
        "captured_head_after": None
        if hook.captured_after is None
        else [float(value) for value in hook.captured_after.tolist()],
    }
    return outputs.logits[0, -1].detach().float().cpu(), debug


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def mean_prob_for_token_ids(
    logits_row: torch.Tensor,
    token_ids: Sequence[int],
    *,
    log_norm: Optional[torch.Tensor] = None,
) -> float:
    if not token_ids:
        return 0.0
    if log_norm is None:
        log_norm = torch.logsumexp(logits_row, dim=-1)
    values = []
    for token_id in token_ids:
        token_logit = logits_row[int(token_id)]
        values.append(float(torch.exp(token_logit - log_norm).item()))
    return _mean(values)


def mean_logit_for_token_ids(logits_row: torch.Tensor, token_ids: Sequence[int]) -> float:
    if not token_ids:
        return 0.0
    values = [float(logits_row[int(token_id)].item()) for token_id in token_ids]
    return _mean(values)


def compute_direct_write_stats(
    *,
    ov_components: Dict[str, Any],
    head_vector_after: Sequence[float] | None,
    target_token_id: int,
    control_token_ids: Sequence[int],
) -> Dict[str, Optional[float]]:
    if head_vector_after is None:
        return {
            "direct_write_target_logit": None,
            "direct_write_control_logit_mean": None,
            "direct_write_target_minus_control": None,
        }

    o_proj_slice = ov_components["o_proj_slice"].detach().float().cpu()
    lm_head_weight = ov_components["lm_head_weight"].detach()
    head_vec = torch.tensor(list(head_vector_after), dtype=torch.float32)
    head_residual = torch.matmul(o_proj_slice, head_vec)

    target_token_id = int(target_token_id)
    target_logit = float(torch.dot(lm_head_weight[target_token_id].detach().float().cpu(), head_residual).item())

    control_logits: List[float] = []
    if control_token_ids:
        control_indices = torch.tensor([int(token_id) for token_id in control_token_ids], dtype=torch.long)
        control_weights = torch.index_select(lm_head_weight.detach().float().cpu(), 0, control_indices)
        control_logits = [float(value) for value in torch.matmul(control_weights, head_residual).tolist()]
    control_logit_mean = _mean(control_logits) if control_logits else None
    return {
        "direct_write_target_logit": float(target_logit),
        "direct_write_control_logit_mean": None if control_logit_mean is None else float(control_logit_mean),
        "direct_write_target_minus_control": None
        if control_logit_mean is None
        else float(target_logit - control_logit_mean),
    }


def build_condition_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "label": "baseline",
            "kind": "identity",
            "scale": 1.0,
        },
        {
            "label": "ablation",
            "kind": args.ablation_kind,
            "scale": 0.0,
        },
    ]
    for scale in parse_scale_values(args.scale_values):
        specs.append(
            {
                "label": format_scale_label(scale),
                "kind": "scale",
                "scale": float(scale),
            }
        )
    return specs


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

    targets, attn_modules, layer_path = resolve_head_targets(backend.model, [args.head_label])
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one head target, got {len(targets)}.")
    target = targets[0]
    attn_module = attn_modules[target.layer_idx]
    ov_components = extract_head_ov_components(
        backend.model,
        layer_idx=target.layer_idx,
        head_idx=target.head_idx,
    )
    condition_specs = build_condition_specs(args)
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
                    raise ValueError("No semantic target token found for the phrase continuation.")
                vocab_size = int(backend.model.config.vocab_size)
                control_token_ids = build_matched_control_token_ids(
                    prefix_token_ids=prompt_token_ids,
                    recent_raw_token_ids=[int(target_token_id)],
                    recent_token_ids=[int(target_token_id)],
                    sample_size=max(int(args.control_sample_size), 1),
                    vocab_size=vocab_size,
                    seed=int(args.seed) + idx * 97,
                )
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"prompt_setup_failed: {exc}"})
                continue

            baseline_row: Optional[Dict[str, Any]] = None
            metadata = dict(case.get("metadata") or {})
            for spec in condition_specs:
                try:
                    seed_everything(int(args.seed) + idx * 1009)
                    logits_row, debug = forward_prompt_with_capture(
                        backend.model,
                        prompt_token_ids,
                        attn_module=attn_module,
                        target=target,
                        scale=float(spec["scale"]),
                    )
                    log_norm = torch.logsumexp(logits_row, dim=-1)
                    target_stats = maybe_token_stats_from_logits(
                        logits_row,
                        int(target_token_id),
                        log_norm=log_norm,
                    )
                    predicted_token_id = int(torch.argmax(logits_row).item())
                    control_prob_mean = mean_prob_for_token_ids(logits_row, control_token_ids, log_norm=log_norm)
                    control_logit_mean = mean_logit_for_token_ids(logits_row, control_token_ids)
                    direct_write_stats = compute_direct_write_stats(
                        ov_components=ov_components,
                        head_vector_after=debug.get("captured_head_after"),
                        target_token_id=int(target_token_id),
                        control_token_ids=control_token_ids,
                    )
                    row = {
                        "example_id": example_id,
                        "original_example_id": str(case.get("original_example_id") or example_id),
                        "family": "phrase",
                        "state": str(case.get("state") or metadata.get("state") or ""),
                        "condition": str(spec["label"]),
                        "intervention_kind": str(spec["kind"]),
                        "intervention_scale": float(spec["scale"]),
                        "prompt_token_count": int(len(prompt_token_ids)),
                        "target_token_id": int(target_token_id),
                        "target_token_text": _decode_token(backend.tokenizer, int(target_token_id)),
                        "control_token_ids": [int(token_id) for token_id in control_token_ids],
                        "control_token_texts": [
                            _decode_token(backend.tokenizer, int(token_id)) for token_id in control_token_ids
                        ],
                        "predicted_token_id": int(predicted_token_id),
                        "predicted_token_text": _decode_token(backend.tokenizer, int(predicted_token_id)),
                        "realized_target_match": bool(int(predicted_token_id) == int(target_token_id)),
                        "target_prob": None if target_stats is None else float(target_stats["prob"]),
                        "target_logit": None if target_stats is None else float(target_stats["logit"]),
                        "target_rank": None if target_stats is None else int(target_stats["rank"]),
                        "control_prob_mean": float(control_prob_mean),
                        "control_logit_mean": float(control_logit_mean),
                        "target_minus_control_prob": None
                        if target_stats is None
                        else float(float(target_stats["prob"]) - float(control_prob_mean)),
                        "target_minus_control_logit": None
                        if target_stats is None
                        else float(float(target_stats["logit"]) - float(control_logit_mean)),
                        "target_prob_delta": None,
                        "target_logit_delta": None,
                        "control_prob_delta_mean": None,
                        "control_logit_delta_mean": None,
                        "target_minus_control_logit_delta": None,
                        "direct_write_target_logit": direct_write_stats["direct_write_target_logit"],
                        "direct_write_control_logit_mean": direct_write_stats["direct_write_control_logit_mean"],
                        "direct_write_target_minus_control": direct_write_stats["direct_write_target_minus_control"],
                        "base_prefix": str(metadata.get("base_prefix") or ""),
                        "append_text": str(metadata.get("append_text") or ""),
                        "assistant_prefix": str(case.get("assistant_prefix") or ""),
                        "debug": debug,
                    }
                    if args.save_prompt_text:
                        row["prompt_prefix"] = prompt_prefix

                    if baseline_row is None:
                        baseline_row = dict(row)
                    else:
                        row["target_prob_delta"] = None
                        if row.get("target_prob") is not None and baseline_row.get("target_prob") is not None:
                            row["target_prob_delta"] = round(
                                float(row["target_prob"]) - float(baseline_row["target_prob"]),
                                6,
                            )
                        row["target_logit_delta"] = None
                        if row.get("target_logit") is not None and baseline_row.get("target_logit") is not None:
                            row["target_logit_delta"] = round(
                                float(row["target_logit"]) - float(baseline_row["target_logit"]),
                                6,
                            )
                        row["control_prob_delta_mean"] = round(
                            float(row["control_prob_mean"]) - float(baseline_row["control_prob_mean"]),
                            6,
                        )
                        row["control_logit_delta_mean"] = round(
                            float(row["control_logit_mean"]) - float(baseline_row["control_logit_mean"]),
                            6,
                        )
                        if row.get("target_minus_control_logit") is not None and baseline_row.get(
                            "target_minus_control_logit"
                        ) is not None:
                            row["target_minus_control_logit_delta"] = round(
                                float(row["target_minus_control_logit"])
                                - float(baseline_row["target_minus_control_logit"]),
                                6,
                            )
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
                recent_rows = [row for row in result_rows if row["example_id"] == example_id]
                recent_by_condition = {str(row["condition"]): row for row in recent_rows}
                base = recent_by_condition.get("baseline")
                abl = recent_by_condition.get("ablation")
                if base is not None and abl is not None:
                    print(
                        f"[Info] pid={os.getpid()} processed={idx} kept_rows={len(result_rows)} skipped={len(skipped_rows)} "
                        f"example_id={example_id} state={case.get('state')} "
                        f"baseline_target_logit={base.get('target_logit', 0.0):.4f} "
                        f"ablation_target_logit={abl.get('target_logit', 0.0):.4f} "
                        f"baseline_direct_write={base.get('direct_write_target_logit', 0.0):.4f}"
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
    parallel_enabled: bool,
    available_gpu_ids: Sequence[int],
    worker_count: int,
    interrupted: bool,
) -> None:
    rows_path = output_dir / "rows.jsonl"
    flat_rows_path = output_dir / "sample_condition_rows.csv"
    summary_rows_path = output_dir / "condition_state_summary.csv"
    summary_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"
    report_path = output_dir / "report.md"

    dump_jsonl(rows_path, result_rows)

    csv_rows: List[Dict[str, Any]] = []
    for row in result_rows:
        flat = dict(row)
        flat["control_token_ids"] = json.dumps(flat.get("control_token_ids") or [], ensure_ascii=False)
        flat["control_token_texts"] = json.dumps(flat.get("control_token_texts") or [], ensure_ascii=False)
        flat["debug"] = json.dumps(flat.get("debug") or {}, ensure_ascii=False)
        csv_rows.append(flat)
    write_csv(flat_rows_path, csv_rows)

    summary_rows = aggregate_phrase_mechanism_rows(
        result_rows,
        condition_order=condition_order,
        state_order=PHRASE_STATE_ORDER,
    )
    write_csv(summary_rows_path, summary_rows)

    summary_json = build_phrase_mechanism_summary_json(
        summary_rows=summary_rows,
        condition_order=condition_order,
        state_order=PHRASE_STATE_ORDER,
    )
    summary_json.update(
        {
            "output_dir": str(output_dir),
            "processed_case_conditions": int(len(result_rows)),
            "processed_examples": int(len({str(row.get('example_id') or '') for row in result_rows})),
            "processed_original_examples": int(len({str(row.get('original_example_id') or '') for row in result_rows})),
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
            "state_order": list(PHRASE_STATE_ORDER),
        }
    )
    write_json(summary_path, summary_json)

    write_json(
        run_config_path,
        {
            "args": vars(args),
            "condition_order": list(condition_order),
            "state_order": list(PHRASE_STATE_ORDER),
            "parallel_enabled": parallel_enabled,
            "parallel_gpu_ids": list(available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids),
            "parallel_workers": int(worker_count),
            "decoder_layer_path": layer_path,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
        },
    )

    write_phrase_mechanism_report(
        report_path,
        args=vars(args),
        summary_rows=summary_rows,
        condition_order=condition_order,
        state_order=PHRASE_STATE_ORDER,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_order = ["baseline", "ablation"] + [format_scale_label(scale) for scale in parse_scale_values(args.scale_values)]

    all_cases = build_sharp_phrase_mechanism_cases(
        count=args.phrase_count,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        cot_input_jsonl=args.cot_input_jsonl,
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

    state_counts: Dict[str, int] = {}
    for case in all_cases:
        state = str(case.get("state") or "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1

    print(
        "[Info] Phrase copy mechanism setup: "
        f"examples={len(all_cases)}, state_counts={state_counts}, cuda_available={torch_cuda_available}, "
        f"available_gpu_ids={available_gpu_ids}, parallel_enabled={parallel_enabled}, worker_count={worker_count}, "
        f"head_label={args.head_label}, scale_values={parse_scale_values(args.scale_values)}"
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

            for worker_rows_path in worker_rows_paths:
                result_rows.extend(read_jsonl_if_exists(worker_rows_path))
            for worker_skipped_path in worker_skipped_paths:
                if worker_skipped_path.exists():
                    with open(worker_skipped_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    if isinstance(payload, list):
                        skipped_rows.extend(payload)

            if not args.keep_worker_outputs:
                for path in worker_rows_paths + worker_skipped_paths:
                    if path.exists():
                        path.unlink()
        except KeyboardInterrupt:
            interrupted = True
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
            progress_desc="Phrase mechanism",
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
        parallel_enabled=parallel_enabled,
        available_gpu_ids=available_gpu_ids,
        worker_count=worker_count,
        interrupted=interrupted,
    )

    print(
        "[Done] Phrase copy mechanism finished: "
        f"processed_case_conditions={len(result_rows)} skipped_examples={len(skipped_rows)} interrupted={interrupted}"
    )


if __name__ == "__main__":
    main()
