#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.generation import _sample_next_token_id
from cot_research.head_path_patch import HeadOProjPatchHooks
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.patch_trajectory import forward_one_with_patch_hooks, prefill_before_final_full_ids
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import token_ids_for_first_tokens
from run_reflection_hidden_trajectory_movie import (
    DEFAULT_REFLECT_FIRST_TEXTS,
    DEFAULT_STOP_FIRST_TEXTS,
    build_backend,
    build_generation_config,
    load_gate_direction_cache,
    parse_layer_spec,
    prepare_forced_box_example,
    row_answer,
    row_id,
    row_question,
    tensor_normed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen attention heads by causal effect on downstream L22 gate projection at p1."
    )
    parser.add_argument(
        "--input_jsonl",
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device_map", default="")
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument(
        "--system_prompt",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    parser.add_argument("--assistant_prefix", default="")
    parser.add_argument("--stage1_stop_string", default="</think>")
    parser.add_argument("--stop_at_think_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_stage1_tokens", type=int, default=4096)
    parser.add_argument("--max_continuation_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coherent_window_chars", type=int, default=1200)
    parser.add_argument("--head_layers", default="19-22")
    parser.add_argument("--target_layer", type=int, default=22)
    parser.add_argument("--target_site", default="post_attn")
    parser.add_argument("--gate_direction_cache_in", required=True)
    parser.add_argument("--print_every", type=int, default=1)
    return parser.parse_args()


def mean(values: Sequence[Any]) -> float:
    xs: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            xs.append(x)
    return sum(xs) / len(xs) if xs else float("nan")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def condition_full_ids(prepared: Dict[str, Any], condition: str) -> List[int]:
    if condition == "T":
        return list(prepared["tamper_full_ids"])
    if condition == "C":
        return list(prepared["coherent_full_ids"])
    raise ValueError(condition)


def gate_projection(captures: Mapping[Tuple[str, int], torch.Tensor], *, layer_idx: int, site: str, direction: torch.Tensor) -> float:
    vec = captures.get((str(site), int(layer_idx)))
    if vec is None:
        return float("nan")
    return float(torch.dot(vec.detach().float().cpu(), direction.detach().float().cpu()).item())


@torch.no_grad()
def run_p0_p1(
    *,
    model: torch.nn.Module,
    tokenizer,
    full_ids: Sequence[int],
    layers: Sequence[torch.nn.Module],
    next_token_id: Optional[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    target_layer: int,
    target_site: str,
    head_layers: Sequence[int],
    num_heads: int,
    head_dim: int,
    capture_heads: bool = False,
    patch_vectors: Optional[Mapping[Tuple[int, int], torch.Tensor]] = None,
) -> Dict[str, Any]:
    past, ids_before_final, final_token_id = prefill_before_final_full_ids(model, full_ids)
    past, logits0, _captures0, _debug0 = forward_one_with_patch_hooks(
        model,
        past=past,
        full_ids_before_token=ids_before_final,
        token_id=final_token_id,
        position_index=0,
        generated_text_before_current="",
        layers=layers,
        capture_layer_indices=[],
        capture_sites=[],
    )
    if next_token_id is None:
        next_token_id = _sample_next_token_id(
            logits0,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
        )
    generated_text_before_current = ""
    with HeadOProjPatchHooks(
        layers,
        num_heads=int(num_heads),
        head_dim=int(head_dim),
        capture_layers=head_layers if capture_heads else [],
        patch_vectors=patch_vectors,
    ) as head_hooks:
        past1, logits1, captures1, debug1 = forward_one_with_patch_hooks(
            model,
            past=past,
            full_ids_before_token=full_ids,
            token_id=int(next_token_id),
            position_index=1,
            generated_text_before_current=generated_text_before_current,
            layers=layers,
            capture_layer_indices=[int(target_layer)],
            capture_sites=[str(target_site)],
        )
        head_values = dict(head_hooks.values)
        patch_call_count = int(head_hooks.patch_call_count)
    return {
        "next_token_id": int(next_token_id),
        "next_token_text": tokenizer.decode([int(next_token_id)], skip_special_tokens=False),
        "logits0": logits0.detach().float().cpu(),
        "logits1": logits1.detach().float().cpu(),
        "captures1": captures1,
        "head_values": head_values,
        "patch_call_count": patch_call_count,
        "debug1": debug1,
    }


def summarize_effects(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("direction")), int(row.get("layer_idx")), int(row.get("head_idx")))].append(row)
    out: List[Dict[str, Any]] = []
    for (direction, layer_idx, head_idx), group in sorted(grouped.items()):
        out.append(
            {
                "direction": direction,
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "count": len(group),
                "mean_effect": mean([row.get("effect") for row in group]),
                "mean_abs_effect": mean([abs(float(row.get("effect"))) for row in group if row.get("effect") is not None]),
                "mean_base_gate_proj": mean([row.get("base_gate_proj") for row in group]),
                "mean_patched_gate_proj": mean([row.get("patched_gate_proj") for row in group]),
                "patch_call_rate": mean([1.0 if int(row.get("patch_call_count") or 0) > 0 else 0.0 for row in group]),
            }
        )
    out.sort(key=lambda row: (str(row["direction"]), -abs(float(row["mean_effect"])), int(row["layer_idx"]), int(row["head_idx"])))
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    backend = build_backend(args)
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("This script requires an HF backend.")
    decoder_layers, layer_path = get_decoder_layers(model)
    layers = list(decoder_layers)
    head_layers = parse_layer_spec(args.head_layers, len(layers))
    num_heads = int(getattr(model.config, "num_attention_heads"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    config_head_dim = getattr(model.config, "head_dim", None)
    if config_head_dim is not None:
        head_dim = int(config_head_dim)
    else:
        first_attn = getattr(layers[0], "self_attn", None)
        first_o_proj = getattr(first_attn, "o_proj", None)
        o_proj_in_features = getattr(first_o_proj, "in_features", None)
        if o_proj_in_features is not None and int(o_proj_in_features) % num_heads == 0:
            head_dim = int(o_proj_in_features) // num_heads
        else:
            head_dim = hidden_size // num_heads

    gen_config = build_generation_config(args)
    stage1_stop = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    gate_info = load_gate_direction_cache(Path(args.gate_direction_cache_in), args.target_layer, args.target_site)
    if gate_info is None:
        raise ValueError(f"No gate direction for L{args.target_layer}/{args.target_site} in {args.gate_direction_cache_in}")
    gate_dir = tensor_normed(gate_info["direction"])

    reflect_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_REFLECT_FIRST_TEXTS)
    stop_ids = token_ids_for_first_tokens(tokenizer, DEFAULT_STOP_FIRST_TEXTS)

    rows = load_jsonl(args.input_jsonl)
    examples = rows[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "layer_path": layer_path,
            "head_layers": head_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "reflect_token_ids": reflect_ids,
            "stop_token_ids": stop_ids,
            "generation": asdict(gen_config),
            "gate_direction": {
                "source": gate_info.get("source"),
                "n_pairs": gate_info.get("n_pairs"),
                "target_layer": int(args.target_layer),
                "target_site": str(args.target_site),
            },
        },
    )

    effect_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="Head gate-effect screen", dynamic_ncols=True)
    for local_idx, ex in enumerate(iterator):
        global_idx = int(args.start_idx) + local_idx
        ex_id = row_id(ex, global_idx)
        question = row_question(ex)
        correct_answer = row_answer(ex)
        wrong_answer = str(ex.get("wrong_answer") or "").strip()
        if not question or correct_answer is None or not wrong_answer:
            skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "missing_fields"})
            continue
        try:
            seed_everything(int(args.seed) + global_idx)
            prepared, skip = prepare_forced_box_example(
                backend=backend,
                tokenizer=tokenizer,
                gen_config=gen_config,
                example=ex,
                example_id=ex_id,
                correct_answer=str(correct_answer),
                wrong_answer=wrong_answer,
                stop_strings=stage1_stop,
                allow_nonmatching_clean=bool(args.allow_nonmatching_clean),
                coherent_window_chars=int(args.coherent_window_chars),
            )
            if prepared is None:
                skipped_rows.append({"global_idx": global_idx, **(skip or {"example_id": ex_id, "reason": "prepare_failed"})})
                continue

            t_free = run_p0_p1(
                model=model,
                tokenizer=tokenizer,
                full_ids=condition_full_ids(prepared, "T"),
                layers=layers,
                next_token_id=None,
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                target_layer=int(args.target_layer),
                target_site=str(args.target_site),
                head_layers=head_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                capture_heads=True,
            )
            forced_token = int(t_free["next_token_id"])
            t_replay = t_free
            c_replay = run_p0_p1(
                model=model,
                tokenizer=tokenizer,
                full_ids=condition_full_ids(prepared, "C"),
                layers=layers,
                next_token_id=forced_token,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                target_layer=int(args.target_layer),
                target_site=str(args.target_site),
                head_layers=head_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                capture_heads=True,
            )
            t_gate = gate_projection(
                t_replay["captures1"],
                layer_idx=int(args.target_layer),
                site=str(args.target_site),
                direction=gate_dir,
            )
            c_gate = gate_projection(
                c_replay["captures1"],
                layer_idx=int(args.target_layer),
                site=str(args.target_site),
                direction=gate_dir,
            )
            baseline_rows.extend(
                [
                    {
                        "example_id": ex_id,
                        "global_idx": global_idx,
                        "condition": "T_replay_p1",
                        "forced_token_id": forced_token,
                        "forced_token_text": t_replay["next_token_text"],
                        "gate_proj": t_gate,
                    },
                    {
                        "example_id": ex_id,
                        "global_idx": global_idx,
                        "condition": "C_replay_p1",
                        "forced_token_id": forced_token,
                        "forced_token_text": c_replay["next_token_text"],
                        "gate_proj": c_gate,
                    },
                ]
            )
            for layer_idx in head_layers:
                for head_idx in range(num_heads):
                    key = (int(layer_idx), int(head_idx))
                    t_head = t_replay["head_values"].get(key)
                    c_head = c_replay["head_values"].get(key)
                    if t_head is None or c_head is None:
                        continue
                    patched_c = run_p0_p1(
                        model=model,
                        tokenizer=tokenizer,
                        full_ids=condition_full_ids(prepared, "C"),
                        layers=layers,
                        next_token_id=forced_token,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        target_layer=int(args.target_layer),
                        target_site=str(args.target_site),
                        head_layers=head_layers,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        capture_heads=False,
                        patch_vectors={key: t_head},
                    )
                    patched_c_gate = gate_projection(
                        patched_c["captures1"],
                        layer_idx=int(args.target_layer),
                        site=str(args.target_site),
                        direction=gate_dir,
                    )
                    effect_rows.append(
                        {
                            "example_id": ex_id,
                            "global_idx": global_idx,
                            "direction": "T_to_C",
                            "layer_idx": int(layer_idx),
                            "head_idx": int(head_idx),
                            "forced_token_id": forced_token,
                            "base_gate_proj": c_gate,
                            "source_gate_proj": t_gate,
                            "patched_gate_proj": patched_c_gate,
                            "effect": patched_c_gate - c_gate,
                            "patch_call_count": patched_c["patch_call_count"],
                        }
                    )

                    patched_t = run_p0_p1(
                        model=model,
                        tokenizer=tokenizer,
                        full_ids=condition_full_ids(prepared, "T"),
                        layers=layers,
                        next_token_id=forced_token,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        target_layer=int(args.target_layer),
                        target_site=str(args.target_site),
                        head_layers=head_layers,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        capture_heads=False,
                        patch_vectors={key: c_head},
                    )
                    patched_t_gate = gate_projection(
                        patched_t["captures1"],
                        layer_idx=int(args.target_layer),
                        site=str(args.target_site),
                        direction=gate_dir,
                    )
                    effect_rows.append(
                        {
                            "example_id": ex_id,
                            "global_idx": global_idx,
                            "direction": "C_to_T",
                            "layer_idx": int(layer_idx),
                            "head_idx": int(head_idx),
                            "forced_token_id": forced_token,
                            "base_gate_proj": t_gate,
                            "source_gate_proj": c_gate,
                            "patched_gate_proj": patched_t_gate,
                            "effect": patched_t_gate - t_gate,
                            "patch_call_count": patched_t["patch_call_count"],
                        }
                    )
            if args.print_every > 0 and (local_idx + 1) % int(args.print_every) == 0:
                iterator.set_postfix({"effects": len(effect_rows), "skipped": len(skipped_rows)})
                dump_jsonl(output_dir / "head_effect_rows.partial.jsonl", effect_rows)
                dump_jsonl(output_dir / "baseline_rows.partial.jsonl", baseline_rows)
                dump_jsonl(output_dir / "skipped_rows.partial.jsonl", skipped_rows)
                write_csv(output_dir / "head_effect_summary.partial.csv", summarize_effects(effect_rows))
        except Exception as exc:
            skipped_rows.append(
                {
                    "example_id": ex_id,
                    "global_idx": global_idx,
                    "reason": "exception",
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    dump_jsonl(output_dir / "head_effect_rows.jsonl", effect_rows)
    dump_jsonl(output_dir / "baseline_rows.jsonl", baseline_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    write_csv(output_dir / "head_effect_summary.csv", summarize_effects(effect_rows))
    reason_counts = Counter(str(row.get("reason")) for row in skipped_rows)
    write_json(
        output_dir / "summary.json",
        {
            "effect_rows": len(effect_rows),
            "baseline_rows": len(baseline_rows),
            "skipped_rows": len(skipped_rows),
            "skipped_reasons": dict(reason_counts),
            "usable_examples": len({row["example_id"] for row in baseline_rows}),
            "head_layers": head_layers,
            "num_heads": num_heads,
        },
    )
    print("[Done] Head gate-effect screen finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_examples: {len({row['example_id'] for row in baseline_rows})}")
    print(f"- effect_rows: {len(effect_rows)}")
    print(f"- skipped_rows: {len(skipped_rows)}")


if __name__ == "__main__":
    main()
