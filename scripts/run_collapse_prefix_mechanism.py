#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, IO, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.collapse_prefix_mechanism import (
    aggregate_case_rows,
    build_short_rollout_metrics,
    decode_token,
    display_token_text,
    extract_topk_token_rows,
    first_non_loop_top_token_id,
    identify_same_token_collapse,
    write_report,
)
from cot_research.copy_suppression import token_stats_from_logits
from cot_research.generation import create_backend
from cot_research.head_intervention import INTERVENTION_REGISTRY, MultiLayerHeadIntervention, resolve_head_targets
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.model_utils import AttentionHeadSpec, get_input_device_for_model
from cot_research.ov_circuit_analysis import extract_head_ov_components
from cot_research.repetition_analysis import RepetitionThresholds
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import parse_parallel_gpu_ids, resolve_device_map, seed_everything, split_examples_contiguous
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the collapse-prefix mechanism of a target head on actual zero-ablation loop trajectories. "
            "For each same-token collapse onset, the script replays the exact zero prefix with the head on/off/scaled, "
            "measures loop-token vs escape-token logits, and runs short rescue rollouts."
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
        default=str(root_dir / "outputs" / "collapse_prefix_mechanism" / "qwen3_1p7b_l0h3_20260408"),
    )
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--scale_value", type=float, default=1.5)
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
    parser.add_argument("--max_examples", type=int, default=108)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--print_every", type=int, default=4)

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--short_rollout_tokens", type=int, default=32)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--same_token_run_threshold", type=int, default=24)
    parser.add_argument("--tail_repeat_min_repeats", type=int, default=4)
    parser.add_argument("--tail_repeat_max_ngram", type=int, default=32)
    parser.add_argument("--line_repeat_threshold", type=int, default=3)
    parser.add_argument("--word_tail_repeat_min_repeats", type=int, default=4)
    parser.add_argument("--word_tail_repeat_max_ngram", type=int, default=24)
    parser.add_argument("--min_trigger_count", type=int, default=1)
    parser.add_argument("--step_top_k", type=int, default=8)
    return parser.parse_args()


def make_thresholds(args: argparse.Namespace) -> RepetitionThresholds:
    return RepetitionThresholds(
        same_token_run_threshold=args.same_token_run_threshold,
        tail_repeat_min_repeats=args.tail_repeat_min_repeats,
        tail_repeat_max_ngram=args.tail_repeat_max_ngram,
        line_repeat_threshold=args.line_repeat_threshold,
        word_tail_repeat_min_repeats=args.word_tail_repeat_min_repeats,
        word_tail_repeat_max_ngram=args.word_tail_repeat_max_ngram,
        min_trigger_count=args.min_trigger_count,
    )


def append_jsonl_line(handle: IO[str] | None, row: Dict[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def load_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl)
    if args.start_idx > 0:
        rows = rows[args.start_idx :]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.max_examples > 0:
        rows = rows[: args.max_examples]
    return rows


class SingleHeadPreOProjHook:
    def __init__(self, attn_module: torch.nn.Module, target: AttentionHeadSpec, scale: float) -> None:
        self.attn_module = attn_module
        self.target = target
        self.scale = float(scale)
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
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
        x_out[..., start:end] = x_out[..., start:end] * self.scale
        self.captured_after = x_out[..., start:end].detach()[0, -1].float().cpu()
        if len(args) == 1:
            return (x_out,)
        return (x_out, *args[1:])


@torch.no_grad()
def forward_prefix_with_capture(
    model: torch.nn.Module,
    prefix_token_ids: Sequence[int],
    *,
    attn_module: torch.nn.Module,
    target: AttentionHeadSpec,
    scale: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    model_device = get_input_device_for_model(model)
    input_ids = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    with SingleHeadPreOProjHook(attn_module, target, scale) as hook:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits[0, -1].detach().float().cpu(), hook.captured_after


@torch.no_grad()
def generate_from_token_ids(
    model: torch.nn.Module,
    tokenizer,
    prefix_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    attn_modules: Sequence[torch.nn.Module],
    operations: Optional[List[Tuple[Any, float]]] = None,
) -> Dict[str, Any]:
    model_device = get_input_device_for_model(model)
    input_ids = torch.tensor([list(prefix_token_ids)], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    generation_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)
    if operations:
        with MultiLayerHeadIntervention(list(attn_modules), operations):
            outputs = model.generate(**generation_kwargs)
    else:
        outputs = model.generate(**generation_kwargs)
    sequences = outputs.sequences[0]
    new_token_ids = [int(token_id) for token_id in sequences[input_ids.shape[-1] :].tolist()]
    continuation = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return {
        "generated_tokens": int(len(new_token_ids)),
        "token_ids": new_token_ids,
        "continuation": continuation,
    }


def build_direct_write_components(model: torch.nn.Module, *, layer_idx: int, head_idx: int) -> Dict[str, Any]:
    ov = extract_head_ov_components(model, layer_idx=layer_idx, head_idx=head_idx)
    return {
        "o_proj_slice": ov["o_proj_slice"].detach().float().cpu(),
        "lm_head_weight": ov["lm_head_weight"].detach().float().cpu(),
    }


def compute_direct_write_pair(
    components: Dict[str, Any],
    *,
    head_vector_after: Optional[torch.Tensor],
    loop_token_id: int,
    escape_token_id: Optional[int],
) -> Dict[str, Optional[float]]:
    if head_vector_after is None:
        return {
            "loop_logit": None,
            "escape_logit": None,
            "escape_minus_loop": None,
        }
    o_proj_slice = components["o_proj_slice"]
    lm_head_weight = components["lm_head_weight"]
    head_residual = torch.matmul(o_proj_slice, head_vector_after.float().cpu())
    loop_logit = float(torch.dot(lm_head_weight[int(loop_token_id)], head_residual).item())
    escape_logit: Optional[float] = None
    if escape_token_id is not None:
        escape_logit = float(torch.dot(lm_head_weight[int(escape_token_id)], head_residual).item())
    return {
        "loop_logit": float(loop_logit),
        "escape_logit": None if escape_logit is None else float(escape_logit),
        "escape_minus_loop": None if escape_logit is None else float(escape_logit - loop_logit),
    }


def flatten_case_row(case_row: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(case_row)
    for key in ["identity_top_rows", "zero_top_rows", "scale_top_rows"]:
        flat[key] = json.dumps(flat.get(key) or [], ensure_ascii=False)
    return flat


def process_rows(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    device_map_override: Any,
    progress_desc: str,
    stream_result_path: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], str, bool]:
    thresholds = make_thresholds(args)
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

    generation_config = GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        capture_step_scores=False,
    )
    targets, attn_modules, layer_path = resolve_head_targets(backend.model, [args.head_label])
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target head, got {len(targets)}.")
    target = targets[0]
    attn_module = attn_modules[target.layer_idx]
    zero_operations = INTERVENTION_REGISTRY.get_required("zero")(targets, {})
    scale_operations = INTERVENTION_REGISTRY.get_required("scale")(targets, {"scale": args.scale_value})
    direct_write_components = build_direct_write_components(
        backend.model,
        layer_idx=target.layer_idx,
        head_idx=target.head_idx,
    )

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    interrupted = False
    result_handle: IO[str] | None = None
    try:
        if stream_result_path:
            result_handle = open(stream_result_path, "w", encoding="utf-8")

        iterator = tqdm(rows, desc=progress_desc, dynamic_ncols=True, leave=False)
        for idx, row in enumerate(iterator, start=1):
            example_id = str(row.get("example_id") or row.get("id") or f"row_{idx}")
            try:
                prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
                prompt_token_ids = backend.encode(prompt_prefix)
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"prompt_resolution_failed: {exc}"})
                continue

            example_seed = int(args.seed) + idx * 1009
            try:
                seed_everything(example_seed)
                baseline_generation = generate_from_token_ids(
                    backend.model,
                    backend.tokenizer,
                    prompt_token_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    attn_modules=attn_modules,
                    operations=None,
                )
                seed_everything(example_seed)
                zero_generation = generate_from_token_ids(
                    backend.model,
                    backend.tokenizer,
                    prompt_token_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    attn_modules=attn_modules,
                    operations=zero_operations,
                )
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"full_generation_failed: {exc}"})
                continue

            baseline_token_ids = [int(token_id) for token_id in baseline_generation["token_ids"]]
            zero_token_ids = [int(token_id) for token_id in zero_generation["token_ids"]]
            zero_same_run = identify_same_token_collapse(
                zero_token_ids,
                same_token_run_threshold=args.same_token_run_threshold,
            )

            case_row: Dict[str, Any] = {
                "example_id": example_id,
                "problem": row.get("question") or row.get("problem") or "",
                "prompt_token_count": int(len(prompt_token_ids)),
                "baseline_generated_tokens": int(len(baseline_token_ids)),
                "zero_generated_tokens": int(len(zero_token_ids)),
                "has_collapse": bool(zero_same_run is not None),
                "zero_longest_same_token_run_length": int(
                    0 if zero_same_run is None else zero_same_run["loop_run_length"]
                ),
            }

            if zero_same_run is None:
                result_rows.append(case_row)
                append_jsonl_line(result_handle, case_row)
                if idx % max(args.print_every, 1) == 0:
                    print(
                        f"[Info] pid={os.getpid()} processed={idx} kept={len(result_rows)} skipped={len(skipped_rows)} "
                        f"example_id={example_id} has_collapse=False"
                    )
                continue

            collapse_step_idx = int(zero_same_run["collapse_step_idx"])
            loop_token_id = int(zero_same_run["loop_token_id"])
            collapse_prefix_token_ids = prompt_token_ids + zero_token_ids[:collapse_step_idx]
            baseline_same_step_token_id = None
            if collapse_step_idx < len(baseline_token_ids):
                baseline_same_step_token_id = int(baseline_token_ids[collapse_step_idx])

            condition_scales = {
                "identity": 1.0,
                "zero": 0.0,
                "scale": float(args.scale_value),
            }
            condition_logits: Dict[str, torch.Tensor] = {}
            condition_head_after: Dict[str, Optional[torch.Tensor]] = {}
            condition_top_rows: Dict[str, List[Dict[str, Any]]] = {}
            try:
                for label, scale in condition_scales.items():
                    logits_row, head_after = forward_prefix_with_capture(
                        backend.model,
                        collapse_prefix_token_ids,
                        attn_module=attn_module,
                        target=target,
                        scale=scale,
                    )
                    condition_logits[label] = logits_row
                    condition_head_after[label] = head_after
                    condition_top_rows[label] = extract_topk_token_rows(
                        logits_row,
                        backend.tokenizer,
                        k=args.step_top_k,
                    )
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"collapse_forward_failed: {exc}"})
                continue

            identity_top_rows = condition_top_rows["identity"]
            identity_top1_token_id = int(identity_top_rows[0]["token_id"])
            identity_non_loop_token_id = first_non_loop_top_token_id(
                identity_top_rows,
                loop_token_id=loop_token_id,
            )
            escape_token_id = None
            escape_token_from_baseline = False
            if baseline_same_step_token_id is not None and int(baseline_same_step_token_id) != int(loop_token_id):
                escape_token_id = int(baseline_same_step_token_id)
                escape_token_from_baseline = True
            elif identity_non_loop_token_id is not None:
                escape_token_id = int(identity_non_loop_token_id)

            case_row.update(
                {
                    "collapse_type": str(zero_same_run["collapse_type"]),
                    "collapse_step_idx": int(collapse_step_idx),
                    "loop_token_id": int(loop_token_id),
                    "loop_token_text": decode_token(backend.tokenizer, loop_token_id),
                    "loop_token_text_display": display_token_text(decode_token(backend.tokenizer, loop_token_id)),
                    "loop_run_length": int(zero_same_run["loop_run_length"]),
                    "baseline_same_step_token_id": baseline_same_step_token_id,
                    "baseline_same_step_token_text": decode_token(backend.tokenizer, baseline_same_step_token_id),
                    "baseline_same_step_token_text_display": display_token_text(
                        decode_token(backend.tokenizer, baseline_same_step_token_id)
                    ),
                    "escape_token_id": escape_token_id,
                    "escape_token_text": decode_token(backend.tokenizer, escape_token_id),
                    "escape_token_text_display": display_token_text(decode_token(backend.tokenizer, escape_token_id)),
                    "escape_token_from_baseline": bool(escape_token_from_baseline),
                    "identity_top1_token_id": identity_top1_token_id,
                    "identity_top1_token_text": decode_token(backend.tokenizer, identity_top1_token_id),
                    "identity_top1_token_text_display": display_token_text(
                        decode_token(backend.tokenizer, identity_top1_token_id)
                    ),
                    "zero_top1_token_id": int(condition_top_rows["zero"][0]["token_id"]),
                    "zero_top1_token_text": decode_token(backend.tokenizer, int(condition_top_rows["zero"][0]["token_id"])),
                    "zero_top1_token_text_display": display_token_text(
                        decode_token(backend.tokenizer, int(condition_top_rows["zero"][0]["token_id"]))
                    ),
                    "scale_top1_token_id": int(condition_top_rows["scale"][0]["token_id"]),
                    "scale_top1_token_text": decode_token(backend.tokenizer, int(condition_top_rows["scale"][0]["token_id"])),
                    "scale_top1_token_text_display": display_token_text(
                        decode_token(backend.tokenizer, int(condition_top_rows["scale"][0]["token_id"]))
                    ),
                    "identity_top_rows": condition_top_rows["identity"],
                    "zero_top_rows": condition_top_rows["zero"],
                    "scale_top_rows": condition_top_rows["scale"],
                    "identity_top1_is_loop": bool(identity_top1_token_id == loop_token_id),
                    "zero_top1_is_loop": bool(int(condition_top_rows["zero"][0]["token_id"]) == loop_token_id),
                    "scale_top1_is_loop": bool(int(condition_top_rows["scale"][0]["token_id"]) == loop_token_id),
                    "identity_top1_matches_baseline_step": bool(
                        baseline_same_step_token_id is not None and identity_top1_token_id == baseline_same_step_token_id
                    ),
                    "zero_top1_matches_baseline_step": bool(
                        baseline_same_step_token_id is not None
                        and int(condition_top_rows["zero"][0]["token_id"]) == baseline_same_step_token_id
                    ),
                    "scale_top1_matches_baseline_step": bool(
                        baseline_same_step_token_id is not None
                        and int(condition_top_rows["scale"][0]["token_id"]) == baseline_same_step_token_id
                    ),
                }
            )

            for label in ["identity", "zero", "scale"]:
                logits_row = condition_logits[label]
                log_norm = torch.logsumexp(logits_row.float(), dim=-1)
                loop_stats = token_stats_from_logits(logits_row, loop_token_id, log_norm=log_norm)
                escape_stats = None
                if escape_token_id is not None:
                    escape_stats = token_stats_from_logits(logits_row, int(escape_token_id), log_norm=log_norm)
                direct_write = compute_direct_write_pair(
                    direct_write_components,
                    head_vector_after=condition_head_after[label],
                    loop_token_id=loop_token_id,
                    escape_token_id=escape_token_id,
                )
                case_row[f"{label}_loop_prob"] = float(loop_stats["prob"])
                case_row[f"{label}_loop_logit"] = float(loop_stats["logit"])
                case_row[f"{label}_loop_rank"] = int(loop_stats["rank"])
                case_row[f"{label}_escape_prob"] = None if escape_stats is None else float(escape_stats["prob"])
                case_row[f"{label}_escape_logit"] = None if escape_stats is None else float(escape_stats["logit"])
                case_row[f"{label}_escape_rank"] = None if escape_stats is None else int(escape_stats["rank"])
                case_row[f"{label}_escape_minus_loop_margin"] = (
                    None
                    if escape_stats is None
                    else float(float(escape_stats["prob"]) - float(loop_stats["prob"]))
                )
                case_row[f"{label}_direct_write_loop_logit"] = direct_write["loop_logit"]
                case_row[f"{label}_direct_write_escape_logit"] = direct_write["escape_logit"]
                case_row[f"{label}_direct_write_escape_minus_loop"] = direct_write["escape_minus_loop"]
                case_row[f"{label}_direct_write_prefers_escape"] = (
                    None
                    if direct_write["escape_minus_loop"] is None
                    else bool(float(direct_write["escape_minus_loop"]) > 0.0)
                )

            case_row["identity_minus_zero_loop_prob"] = (
                float(case_row["identity_loop_prob"]) - float(case_row["zero_loop_prob"])
            )
            case_row["scale_minus_zero_loop_prob"] = (
                float(case_row["scale_loop_prob"]) - float(case_row["zero_loop_prob"])
            )
            case_row["identity_minus_zero_loop_logit"] = (
                float(case_row["identity_loop_logit"]) - float(case_row["zero_loop_logit"])
            )
            case_row["scale_minus_zero_loop_logit"] = (
                float(case_row["scale_loop_logit"]) - float(case_row["zero_loop_logit"])
            )
            case_row["identity_minus_zero_escape_prob"] = None
            case_row["scale_minus_zero_escape_prob"] = None
            case_row["identity_minus_zero_escape_logit"] = None
            case_row["scale_minus_zero_escape_logit"] = None
            if escape_token_id is not None:
                case_row["identity_minus_zero_escape_prob"] = (
                    float(case_row["identity_escape_prob"]) - float(case_row["zero_escape_prob"])
                )
                case_row["scale_minus_zero_escape_prob"] = (
                    float(case_row["scale_escape_prob"]) - float(case_row["zero_escape_prob"])
                )
                case_row["identity_minus_zero_escape_logit"] = (
                    float(case_row["identity_escape_logit"]) - float(case_row["zero_escape_logit"])
                )
                case_row["scale_minus_zero_escape_logit"] = (
                    float(case_row["scale_escape_logit"]) - float(case_row["zero_escape_logit"])
                )

            rollout_operations = {
                "identity": None,
                "zero": zero_operations,
                "scale": scale_operations,
            }
            try:
                for label in ["identity", "zero", "scale"]:
                    seed_everything(example_seed)
                    rollout = generate_from_token_ids(
                        backend.model,
                        backend.tokenizer,
                        collapse_prefix_token_ids,
                        max_new_tokens=args.short_rollout_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        attn_modules=attn_modules,
                        operations=rollout_operations[label],
                    )
                    rollout_metrics = build_short_rollout_metrics(
                        continuation=str(rollout["continuation"]),
                        token_ids=rollout["token_ids"],
                        thresholds=thresholds,
                    )
                    first_token_id = rollout_metrics["first_token_id"]
                    case_row[f"{label}_short_generated_tokens"] = int(rollout_metrics["generated_tokens"])
                    case_row[f"{label}_short_first_token_id"] = first_token_id
                    case_row[f"{label}_short_first_token_text"] = decode_token(backend.tokenizer, first_token_id)
                    case_row[f"{label}_short_first_token_text_display"] = display_token_text(
                        decode_token(backend.tokenizer, first_token_id)
                    )
                    case_row[f"{label}_short_first_token_is_loop"] = (
                        None if first_token_id is None else bool(int(first_token_id) == loop_token_id)
                    )
                    case_row[f"{label}_short_first_token_matches_escape"] = (
                        None
                        if first_token_id is None or escape_token_id is None
                        else bool(int(first_token_id) == int(escape_token_id))
                    )
                    case_row[f"{label}_short_repetition_matched"] = bool(
                        dict(rollout_metrics["repetition_detection"]).get("matched")
                    )
                    case_row[f"{label}_short_repetition_score"] = int(
                        dict(rollout_metrics["repetition_detection"]).get("score", 0)
                    )
                    case_row[f"{label}_short_longest_same_token_run"] = int(
                        dict(rollout_metrics["longest_same_token_run"]).get("length", 0)
                    )
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as exc:
                skipped_rows.append({"example_id": example_id, "reason": f"short_rollout_failed: {exc}"})
                continue

            result_rows.append(case_row)
            append_jsonl_line(result_handle, case_row)
            if idx % max(args.print_every, 1) == 0:
                print(
                    f"[Info] pid={os.getpid()} processed={idx} kept={len(result_rows)} skipped={len(skipped_rows)} "
                    f"example_id={example_id} collapse_step={collapse_step_idx} loop_token={case_row['loop_token_text_display']} "
                    f"identity_loop_prob={case_row['identity_loop_prob']:.4f} zero_loop_prob={case_row['zero_loop_prob']:.4f} "
                    f"identity_short_loop={case_row['identity_short_first_token_is_loop']} zero_short_loop={case_row['zero_short_first_token_is_loop']}"
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
    shard_rows: List[Dict[str, Any]],
    args_dict: Dict[str, Any],
    worker_rows_path: str,
    worker_skipped_path: str,
) -> Dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    result_rows, skipped_rows, layer_path, interrupted = process_rows(
        shard_rows,
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
    parallel_enabled: bool,
    available_gpu_ids: Sequence[int],
    worker_count: int,
    interrupted: bool,
) -> None:
    rows_path = output_dir / "rows.jsonl"
    case_summary_path = output_dir / "case_summary.csv"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    run_config_path = output_dir / "run_config.json"

    dump_jsonl(rows_path, result_rows)
    write_csv(case_summary_path, [flatten_case_row(row) for row in result_rows])

    summary = aggregate_case_rows(result_rows)
    summary.update(
        {
            "output_dir": str(output_dir),
            "processed_examples": int(len(result_rows)),
            "skipped_examples": int(len(skipped_rows)),
            "skipped_rows": skipped_rows,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
            "model_name_or_path": args.model_name_or_path,
            "head_label": args.head_label,
            "scale_value": float(args.scale_value),
            "same_token_run_threshold": int(args.same_token_run_threshold),
            "short_rollout_tokens": int(args.short_rollout_tokens),
            "parallel_enabled": bool(parallel_enabled),
            "parallel_gpu_ids": list(available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids),
            "parallel_workers": int(worker_count),
            "decoder_layer_path": layer_path,
        }
    )
    write_json(summary_path, summary)
    write_json(
        run_config_path,
        {
            "args": vars(args),
            "parallel_enabled": bool(parallel_enabled),
            "parallel_gpu_ids": list(available_gpu_ids[:worker_count] if parallel_enabled else available_gpu_ids),
            "parallel_workers": int(worker_count),
            "decoder_layer_path": layer_path,
            "completed_successfully": not interrupted,
            "interrupted": interrupted,
        },
    )
    write_report(report_path, args=vars(args), summary=summary)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_rows(args)
    if not all_rows:
        write_json(output_dir / "summary.json", {"message": "No rows to process.", "processed_examples": 0})
        write_json(output_dir / "run_config.json", {"args": vars(args)})
        (output_dir / "rows.jsonl").write_text("", encoding="utf-8")
        print("[Done] No rows to process.")
        return

    if args.parallel_gpu_ids.strip():
        available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)
    elif args.gpu_id >= 0:
        available_gpu_ids = [args.gpu_id]
    else:
        available_gpu_ids = parse_parallel_gpu_ids(args.parallel_gpu_ids)

    torch_cuda_available = torch.cuda.is_available()
    can_parallel = args.parallel and torch_cuda_available and len(available_gpu_ids) > 1 and len(all_rows) > 1
    if can_parallel and args.parallel_workers > 0:
        worker_count = min(args.parallel_workers, len(available_gpu_ids), len(all_rows))
    elif can_parallel:
        worker_count = min(len(available_gpu_ids), len(all_rows))
    else:
        worker_count = 1
    parallel_enabled = can_parallel and worker_count > 1

    print(
        "[Info] Collapse-prefix mechanism setup: "
        f"examples={len(all_rows)}, cuda_available={torch_cuda_available}, "
        f"available_gpu_ids={available_gpu_ids}, parallel_enabled={parallel_enabled}, worker_count={worker_count}, "
        f"head_label={args.head_label}, scale_value={args.scale_value}, same_token_run_threshold={args.same_token_run_threshold}"
    )

    result_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, str]] = []
    layer_path = ""
    interrupted = False

    if parallel_enabled:
        worker_gpu_ids = available_gpu_ids[:worker_count]
        row_shards = split_examples_contiguous(all_rows, worker_count)
        worker_rows_paths: List[Path] = []
        worker_skipped_paths: List[Path] = []
        try:
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(row_shards), mp_context=mp_ctx) as pool:
                futures = []
                for worker_id, shard_rows in enumerate(row_shards):
                    gpu_id = worker_gpu_ids[worker_id]
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
        device_map_override = resolve_device_map(args.device_map, args.gpu_id)
        if args.gpu_id >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
        result_rows, skipped_rows, layer_path, interrupted = process_rows(
            all_rows,
            args,
            device_map_override=device_map_override,
            progress_desc="Collapse mechanism",
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
        parallel_enabled=parallel_enabled,
        available_gpu_ids=available_gpu_ids,
        worker_count=worker_count,
        interrupted=interrupted,
    )

    print(
        "[Done] Collapse-prefix mechanism finished: "
        f"processed_examples={len(result_rows)} skipped_examples={len(skipped_rows)} interrupted={interrupted}"
    )


if __name__ == "__main__":
    main()
