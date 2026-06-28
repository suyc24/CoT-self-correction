#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT_DIR, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.answer_extraction import answers_match
from cot_research.cot_editing import find_last_boxed_span
from cot_research.generation import _sample_next_token_id
from cot_research.head_path_patch import HeadOProjPatchHooks
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.patch_trajectory import BoundaryPatchHooks, prefill_before_final_full_ids
from cot_research.runtime_utils import seed_everything
from cot_research.stateful_tampering import analyze_continuation_text, token_ids_for_first_tokens
from run_reflection_hidden_trajectory_movie import (
    DEFAULT_REFLECT_FIRST_TEXTS,
    DEFAULT_REFLECTION_KEYWORDS,
    DEFAULT_STOP_FIRST_TEXTS,
    build_backend,
    build_generation_config,
    load_gate_direction_cache,
    parse_csv_list,
    parse_layer_spec,
    prepare_forced_box_example,
    row_answer,
    row_id,
    row_question,
    tensor_normed,
)


FALLBACK_TOP_HEADS = [(21, 3), (20, 1), (22, 26), (19, 20)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D4 source-constrained head path patching.")
    parser.add_argument("--input_jsonl", default=str(ROOT_DIR / "evaluation/data/self_correction_ablation/test_questions.jsonl"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--model_size_label", default="qwen3_4b")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=45)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device_map", default="")
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--system_prompt", default="Please reason step by step, and put your final answer within \\boxed{}.")
    parser.add_argument("--assistant_prefix", default="")
    parser.add_argument("--stage1_stop_string", default="</think>")
    parser.add_argument("--stop_at_think_end", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_stage1_tokens", type=int, default=4096)
    parser.add_argument("--max_continuation_tokens", type=int, default=8)
    parser.add_argument("--max_probe_tokens", type=int, default=8)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--allow_nonmatching_clean", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coherent_window_chars", type=int, default=1200)
    parser.add_argument("--head_layers", default="19-22")
    parser.add_argument("--positions", default="1,2,4,8")
    parser.add_argument("--regions", default="box_wrong,nearest_clean_anchor,local_evidence_window,inconsistency_anchors,conflict_all")
    parser.add_argument("--ks", default="4,8,16")
    parser.add_argument("--target_layer", type=int, default=22)
    parser.add_argument("--target_site", default="post_attn")
    parser.add_argument("--gate_direction_cache_in", required=True)
    parser.add_argument("--c2_summary_csv", default="")
    parser.add_argument("--top_heads", default="")
    parser.add_argument("--include_pattern_value_controls", action=argparse.BooleanOptionalAction, default=False)
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
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for item in parse_csv_list(text):
        if "-" in item:
            left, right = item.split("-", 1)
            out.extend(range(int(left), int(right) + 1))
        else:
            out.append(int(item))
    return out


def parse_heads(text: str) -> List[Tuple[int, int]]:
    heads: List[Tuple[int, int]] = []
    for item in parse_csv_list(text):
        item = item.upper().replace("L", "").replace("H", ":")
        if ":" in item:
            left, right = item.split(":", 1)
        else:
            left, right = item.split(".", 1)
        heads.append((int(left), int(right)))
    return heads


def load_top_heads(path: str, fallback: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not path:
        return list(fallback)
    p = Path(path)
    if not p.exists():
        return list(fallback)
    rows: List[Dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    scored: List[Tuple[float, Tuple[int, int]]] = []
    for row in rows:
        try:
            layer = int(row.get("layer_idx") or row.get("layer") or -1)
            head = int(row.get("head_idx") or row.get("head") or -1)
            effect = float(row.get("mean_effect") or row.get("effect") or 0.0)
        except ValueError:
            continue
        if layer >= 0 and head >= 0:
            scored.append((abs(effect), (layer, head)))
    scored.sort(reverse=True, key=lambda x: x[0])
    out: List[Tuple[int, int]] = []
    for _score, head in scored:
        if head not in out:
            out.append(head)
    return out or list(fallback)


def token_char_offsets(tokenizer, token_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
    pieces: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        pieces.append(piece)
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return "".join(pieces), offsets


def chars_to_tokens(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> Tuple[int, ...]:
    return tuple(i for i, (left, right) in enumerate(offsets) if right > int(start) and left < int(end))


def find_answer_spans(text: str, answer: str, *, end: int) -> List[Tuple[int, int]]:
    answer = str(answer).strip()
    spans: List[Tuple[int, int]] = []
    if not answer:
        return spans
    start = 0
    while True:
        idx = text.find(answer, start, end)
        if idx < 0:
            break
        spans.append((idx, idx + len(answer)))
        start = idx + max(1, len(answer))
    return spans


def annotate_regions(tokenizer, full_ids: Sequence[int], *, prompt_len: int, correct_answer: str, local_window_chars: int = 1200) -> Dict[str, Tuple[int, ...]]:
    text, offsets = token_char_offsets(tokenizer, full_ids)
    n_tokens = len(full_ids)
    boxed = find_last_boxed_span(text)
    if boxed is None:
        return {"prompt": tuple(range(min(prompt_len, n_tokens)))}
    box_start, box_end, _old = boxed
    brace = text.find("{", box_start, box_end)
    answer_start = brace + 1 if brace >= 0 else box_start
    answer_end = max(answer_start, box_end - 1)
    box_wrong = chars_to_tokens(offsets, answer_start, answer_end)
    box_format = tuple(i for i in range(max(0, min(box_wrong or (0,)) - 8), min(n_tokens, max(box_wrong or (0,)) + 9))) if box_wrong else ()
    anchors = find_answer_spans(text, correct_answer, end=box_start)
    nearest = chars_to_tokens(offsets, *(anchors[-1])) if anchors else ()
    prev: List[int] = []
    for left, right in anchors[-4:]:
        prev.extend(chars_to_tokens(offsets, left, right))
    local_start = max(0, box_start - int(local_window_chars))
    local = chars_to_tokens(offsets, local_start, box_start)
    nonlocal_reasoning = chars_to_tokens(offsets, 0, local_start)
    prompt = tuple(range(0, min(int(prompt_len), n_tokens)))

    def uniq(values: Iterable[int]) -> Tuple[int, ...]:
        return tuple(sorted({int(x) for x in values if 0 <= int(x) < n_tokens}))

    inconsistency = uniq(list(box_wrong) + list(nearest) + prev)
    conflict_all = uniq(list(inconsistency) + list(local))
    return {
        "box_wrong": uniq(box_wrong),
        "box_format": uniq(box_format),
        "nearest_clean_anchor": uniq(nearest),
        "prev_k_clean_anchors": uniq(prev),
        "local_evidence_window": uniq(local),
        "nonlocal_reasoning": uniq(nonlocal_reasoning),
        "prompt": uniq(prompt),
        "inconsistency_anchors": inconsistency,
        "conflict_all": conflict_all,
    }


def get_past_values(past: Any, layer_idx: int) -> Optional[torch.Tensor]:
    if hasattr(past, "value_cache"):
        return past.value_cache[int(layer_idx)]
    if hasattr(past, "layers"):
        layer = past.layers[int(layer_idx)]
        for name in ("values", "value_states", "value_cache"):
            value = getattr(layer, name, None)
            if value is not None:
                return value
    if isinstance(past, (tuple, list)):
        item = past[int(layer_idx)]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            return item[1]
    return None


def repeat_kv_values(values: torch.Tensor, num_heads: int) -> torch.Tensor:
    vals = values[0]
    if vals.shape[0] == int(num_heads):
        return vals
    repeat = int(num_heads) // int(vals.shape[0])
    return vals.repeat_interleave(repeat, dim=0)


def gate_projection(captures: Mapping[Tuple[str, int], torch.Tensor], *, layer_idx: int, site: str, direction: torch.Tensor) -> float:
    vec = captures.get((str(site), int(layer_idx)))
    if vec is None:
        return float("nan")
    return float(torch.dot(vec.detach().float().cpu(), direction.detach().float().cpu()).item())


@torch.no_grad()
def forward_one_detailed(
    model: torch.nn.Module,
    *,
    past: Any,
    full_ids_before_token: Sequence[int],
    token_id: int,
    position_index: int,
    layers: Sequence[torch.nn.Module],
    head_layers: Sequence[int],
    num_heads: int,
    head_dim: int,
    target_layer: int,
    target_site: str,
    gate_dir: torch.Tensor,
    regions: Mapping[str, Sequence[int]],
    patch_vectors: Optional[Mapping[Tuple[int, int], torch.Tensor]] = None,
) -> Tuple[Any, torch.Tensor, Dict[str, Any]]:
    device = next(model.parameters()).device
    values_before: Dict[int, Optional[torch.Tensor]] = {int(layer): get_past_values(past, int(layer)) for layer in head_layers}
    input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    full_len = len(full_ids_before_token) + 1
    attention_mask = torch.ones((1, full_len), dtype=torch.long, device=device)
    with HeadOProjPatchHooks(
        layers,
        num_heads=int(num_heads),
        head_dim=int(head_dim),
        capture_layers=head_layers,
        patch_vectors=patch_vectors,
    ) as head_hooks, BoundaryPatchHooks(
        layers,
        position_index=int(position_index),
        generated_text_before_current="",
        capture_layer_indices=[int(target_layer)],
        capture_sites=[str(target_site)],
    ) as boundary_hooks:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        captures = dict(boundary_hooks.values)
        head_values = {key: value.detach().float().cpu() for key, value in head_hooks.values.items()}
        patch_call_count = int(head_hooks.patch_call_count)
    attentions = getattr(outputs, "attentions", None)
    contribs: Dict[Tuple[str, int, int], torch.Tensor] = {}
    masses: Dict[Tuple[str, int, int], float] = {}
    if attentions is not None:
        for layer_idx in head_layers:
            layer_attn = attentions[int(layer_idx)]
            values = values_before.get(int(layer_idx))
            if values is None:
                continue
            vals = repeat_kv_values(values, int(num_heads)).detach().float()
            attn = layer_attn[0, :, -1, : vals.shape[1]].detach().float()
            for region_name, token_indices in regions.items():
                valid = [int(i) for i in token_indices if 0 <= int(i) < vals.shape[1]]
                if not valid:
                    zero = torch.zeros((int(num_heads), int(head_dim)), dtype=torch.float32)
                    for head_idx in range(int(num_heads)):
                        contribs[(region_name, int(layer_idx), head_idx)] = zero[head_idx]
                        masses[(region_name, int(layer_idx), head_idx)] = 0.0
                    continue
                idx = torch.tensor(valid, dtype=torch.long, device=vals.device)
                weights = attn[:, idx]
                region_vals = vals[:, idx, :]
                contribution = (weights.unsqueeze(-1) * region_vals).sum(dim=1).cpu()
                mass = weights.sum(dim=1).cpu()
                for head_idx in range(int(num_heads)):
                    contribs[(region_name, int(layer_idx), head_idx)] = contribution[head_idx].detach().float()
                    masses[(region_name, int(layer_idx), head_idx)] = float(mass[head_idx].item())
    details = {
        "gate_proj": gate_projection(captures, layer_idx=int(target_layer), site=str(target_site), direction=gate_dir),
        "head_values": head_values,
        "contribs": contribs,
        "masses": masses,
        "patch_call_count": patch_call_count,
        "captures": captures,
    }
    return getattr(outputs, "past_key_values"), outputs.logits[0, -1], details


@torch.no_grad()
def generate_probe_tokens(
    model: torch.nn.Module,
    tokenizer,
    full_ids: Sequence[int],
    *,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[int]:
    past, ids_before_final, final_token_id = prefill_before_final_full_ids(model, full_ids)
    device = next(model.parameters()).device
    input_ids = torch.tensor([[int(final_token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(ids_before_final) + 1), dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True, return_dict=True)
    past = getattr(outputs, "past_key_values")
    logits = outputs.logits[0, -1]
    generated: List[int] = []
    current_full = list(full_ids)
    for _ in range(int(max_tokens)):
        token_id = _sample_next_token_id(logits, do_sample=bool(do_sample), temperature=float(temperature), top_p=float(top_p))
        generated.append(int(token_id))
        prefix = list(current_full)
        current_full.append(int(token_id))
        input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, len(prefix) + 1), dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True, return_dict=True)
        past = getattr(outputs, "past_key_values")
        logits = outputs.logits[0, -1]
    return generated


def run_to_position(
    *,
    model: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    full_ids: Sequence[int],
    forced_tokens: Sequence[int],
    position: int,
    head_layers: Sequence[int],
    num_heads: int,
    head_dim: int,
    target_layer: int,
    target_site: str,
    gate_dir: torch.Tensor,
    regions: Mapping[str, Sequence[int]],
    patch_vectors: Optional[Mapping[Tuple[int, int], torch.Tensor]] = None,
) -> Dict[str, Any]:
    past, ids_before_final, final_token_id = prefill_before_final_full_ids(model, full_ids)
    device = next(model.parameters()).device
    input_ids = torch.tensor([[int(final_token_id)]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(ids_before_final) + 1), dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True, return_dict=True)
    past = getattr(outputs, "past_key_values")
    logits = outputs.logits[0, -1]
    current_full = list(full_ids)
    detail: Dict[str, Any] = {}
    for step_idx in range(int(position)):
        token_id = int(forced_tokens[step_idx])
        prefix = list(current_full)
        current_full.append(token_id)
        if step_idx + 1 == int(position):
            dynamic_regions = dict(regions)
            dynamic_regions["generated_prefix"] = tuple(range(len(full_ids), len(prefix)))
            past, logits, detail = forward_one_detailed(
                model,
                past=past,
                full_ids_before_token=prefix,
                token_id=token_id,
                position_index=step_idx + 1,
                layers=layers,
                head_layers=head_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                target_layer=target_layer,
                target_site=target_site,
                gate_dir=gate_dir,
                regions=dynamic_regions,
                patch_vectors=patch_vectors,
            )
        else:
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
            attention_mask = torch.ones((1, len(prefix) + 1), dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True, return_dict=True)
            past = getattr(outputs, "past_key_values")
            logits = outputs.logits[0, -1]
    return detail


def summarize_effects(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("direction")),
                str(row.get("group_name")),
                str(row.get("patch_type")),
                str(row.get("region")),
                int(row.get("position")),
                str(row.get("control_type")),
            )
        ].append(row)
    out: List[Dict[str, Any]] = []
    for (direction, group_name, patch_type, region, position, control_type), group in sorted(grouped.items()):
        out.append(
            {
                "direction": direction,
                "group_name": group_name,
                "control_type": control_type,
                "patch_type": patch_type,
                "region": region,
                "position": position,
                "count": len(group),
                "mean_effect": mean([row.get("effect") for row in group]),
                "mean_abs_effect": mean([abs(float(row.get("effect"))) for row in group if row.get("effect") is not None]),
                "mean_base_gate_proj": mean([row.get("base_gate_proj") for row in group]),
                "mean_source_gate_proj": mean([row.get("source_gate_proj") for row in group]),
                "mean_patched_gate_proj": mean([row.get("patched_gate_proj") for row in group]),
                "patch_call_rate": mean([1.0 if int(row.get("patch_call_count") or 0) > 0 else 0.0 for row in group]),
            }
        )
    out.sort(key=lambda row: (str(row["direction"]), str(row["patch_type"]), -abs(float(row["mean_effect"])), str(row["group_name"])))
    return out


def compact_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], max_rows: int = 20) -> str:
    rows = list(rows)[:max_rows]
    if not rows:
        return "(empty)"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals: List[str] = []
        for col in columns:
            value = row.get(col, "")
            vals.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, summary: Mapping[str, Any], effect_summary: Sequence[Dict[str, Any]]) -> None:
    top = sorted(effect_summary, key=lambda r: abs(float(r.get("mean_effect") or 0.0)), reverse=True)[:16]
    control = [r for r in effect_summary if r.get("patch_type") == "source_contribution" and r.get("region") in {"conflict_all", "inconsistency_anchors"}]
    lines = [
        "# D4 Source-Constrained Head Path Patching",
        "",
        "## 1. 实验目的",
        "验证 C2 candidate heads 是否能把 box/evidence inconsistency 的 source-region contribution 路由到 L22 gate state。",
        "",
        "## 2. 运行规模",
        f"- n_attempted: {summary.get('n_attempted')}",
        f"- n_usable: {summary.get('n_usable')}",
        f"- n_skipped: {summary.get('n_skipped')}",
        f"- skip_reasons: `{json.dumps(summary.get('skip_reasons', {}), ensure_ascii=False)}`",
        "",
        "## 3. Top Hidden Effects",
        compact_table(top, ["direction", "group_name", "control_type", "patch_type", "region", "position", "count", "mean_effect", "mean_abs_effect"], 20),
        "",
        "## 4. Source Contribution Focus",
        compact_table(control, ["direction", "group_name", "control_type", "region", "position", "count", "mean_effect", "mean_abs_effect"], 24),
        "",
        "## 5. 关键发现",
        "- 该脚本的核心读数是 downstream L22 gate_proj 的因果位移，不按 Wait logit 选头。",
        "- source-constrained patch 使用 exact attention-weighted value contribution：`head = head - base_region_contrib + source_region_contrib`。",
        "- top-gate group 与 random / same-layer-random / dynamic attention-mass controls 的差异见 `head_group_hidden_summary.csv`。",
        "",
        "## 6. 不确定点",
        "- 本轮默认不做完整 free-running 行为 patch；D4 主要是 hidden path causality。",
        "- pattern-only/value-only 是可选扩展；默认报告重点放在 full-head 与 region contribution。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    backend = build_backend(args)
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("HF backend is required.")
    decoder_layers, layer_path = get_decoder_layers(model)
    layers = list(decoder_layers)
    head_layers = parse_layer_spec(args.head_layers, len(layers))
    num_heads = int(getattr(model.config, "num_attention_heads"))
    first_attn = getattr(layers[0], "self_attn")
    o_proj = getattr(first_attn, "o_proj")
    head_dim = int(getattr(model.config, "head_dim", 0) or (int(o_proj.in_features) // num_heads))
    positions = parse_int_list(args.positions)
    regions = parse_csv_list(args.regions)
    ks = parse_int_list(args.ks)
    top_heads = parse_heads(args.top_heads) if args.top_heads else load_top_heads(args.c2_summary_csv, FALLBACK_TOP_HEADS)
    all_heads = [(layer, head) for layer in head_layers for head in range(num_heads)]

    gate_info = load_gate_direction_cache(Path(args.gate_direction_cache_in), args.target_layer, args.target_site)
    if gate_info is None:
        raise ValueError(f"No gate direction for L{args.target_layer}/{args.target_site}")
    gate_dir = tensor_normed(gate_info["direction"])
    gen_config = build_generation_config(args)
    stage1_stop = [args.stage1_stop_string] if args.stop_at_think_end and args.stage1_stop_string else None
    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "layer_path": layer_path,
            "head_layers": head_layers,
            "positions": positions,
            "regions": regions,
            "ks": ks,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "top_heads_loaded": top_heads[:32],
            "gate_direction": {"source": gate_info.get("source"), "n_pairs": gate_info.get("n_pairs")},
        },
    )

    rows = load_jsonl(args.input_jsonl)
    examples = rows[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    effect_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    attention_mass_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    iterator = tqdm(examples, desc="D4 source-constrained head patch", dynamic_ncols=True)
    for local_idx, ex in enumerate(iterator):
        global_idx = int(args.start_idx) + local_idx
        ex_id = row_id(ex, global_idx)
        question = row_question(ex)
        correct = row_answer(ex)
        wrong = str(ex.get("wrong_answer") or "").strip()
        if not question or correct is None or not wrong or str(correct).strip() == wrong:
            skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "missing_or_bad_fields"})
            continue
        try:
            seed_everything(int(args.seed) + global_idx)
            prepared, skip = prepare_forced_box_example(
                backend=backend,
                tokenizer=tokenizer,
                gen_config=gen_config,
                example=ex,
                example_id=ex_id,
                correct_answer=str(correct),
                wrong_answer=wrong,
                stop_strings=stage1_stop,
                allow_nonmatching_clean=bool(args.allow_nonmatching_clean),
                coherent_window_chars=int(args.coherent_window_chars),
            )
            if prepared is None:
                skipped_rows.append({"global_idx": global_idx, **(skip or {"example_id": ex_id, "reason": "prepare_failed"})})
                continue
            t_ids = [int(x) for x in prepared["tamper_full_ids"]]
            c_ids = [int(x) for x in prepared["coherent_full_ids"]]
            forced = generate_probe_tokens(
                model,
                tokenizer,
                t_ids,
                max_tokens=max(max(positions), int(args.max_probe_tokens)),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
            prompt_len = len(prepared["prompt_ids"])
            region_map = {
                "T": annotate_regions(tokenizer, t_ids, prompt_len=prompt_len, correct_answer=str(correct), local_window_chars=int(args.coherent_window_chars)),
                "C": annotate_regions(tokenizer, c_ids, prompt_len=prompt_len, correct_answer=str(correct), local_window_chars=int(args.coherent_window_chars)),
            }
            for position in positions:
                t_detail = run_to_position(
                    model=model,
                    layers=layers,
                    full_ids=t_ids,
                    forced_tokens=forced,
                    position=int(position),
                    head_layers=head_layers,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    target_layer=int(args.target_layer),
                    target_site=str(args.target_site),
                    gate_dir=gate_dir,
                    regions={name: region_map["T"].get(name, ()) for name in regions},
                )
                c_detail = run_to_position(
                    model=model,
                    layers=layers,
                    full_ids=c_ids,
                    forced_tokens=forced,
                    position=int(position),
                    head_layers=head_layers,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    target_layer=int(args.target_layer),
                    target_site=str(args.target_site),
                    gate_dir=gate_dir,
                    regions={name: region_map["C"].get(name, ()) for name in regions},
                )
                baseline_rows.extend(
                    [
                        {"example_id": ex_id, "global_idx": global_idx, "condition": "T_replay", "position": int(position), "gate_proj": t_detail["gate_proj"]},
                        {"example_id": ex_id, "global_idx": global_idx, "condition": "C_replay", "position": int(position), "gate_proj": c_detail["gate_proj"]},
                    ]
                )
                for layer, head in all_heads:
                    attention_mass_rows.append(
                        {
                            "example_id": ex_id,
                            "global_idx": global_idx,
                            "condition": "T_replay",
                            "position": int(position),
                            "layer_idx": layer,
                            "head_idx": head,
                            "conflict_all_attention_mass": t_detail["masses"].get(("conflict_all", layer, head), 0.0),
                            "inconsistency_attention_mass": t_detail["masses"].get(("inconsistency_anchors", layer, head), 0.0),
                        }
                    )
                rng = random.Random(int(args.seed) + global_idx * 1009 + int(position))
                for k in ks:
                    top_group = list(top_heads[: int(k)])
                    random_group = rng.sample(all_heads, min(int(k), len(all_heads)))
                    top_layers = [layer for layer, _head in top_group] or head_layers
                    same_layer_pool = [(layer, head) for layer, head in all_heads if layer in top_layers]
                    same_layer_random = rng.sample(same_layer_pool, min(int(k), len(same_layer_pool)))
                    mass_ranked = sorted(
                        all_heads,
                        key=lambda lh: t_detail["masses"].get(("conflict_all", lh[0], lh[1]), 0.0),
                        reverse=True,
                    )[: int(k)]
                    groups = [
                        (f"top{k}_gate_effect", "top_gate_effect", top_group),
                        (f"random{k}", "random_same_K", random_group),
                        (f"same_layer_random{k}", "same_layer_random_K", same_layer_random),
                        (f"top{k}_attention_mass_dynamic", "topK_by_attention_mass", mass_ranked),
                    ]
                    for group_name, control_type, group_heads in groups:
                        for direction, base_name, source_name, base_ids, base_detail, source_detail in [
                            ("T_to_C", "C", "T", c_ids, c_detail, t_detail),
                            ("C_to_T", "T", "C", t_ids, t_detail, c_detail),
                        ]:
                            patch_vectors = {
                                (layer, head): source_detail["head_values"][(layer, head)]
                                for layer, head in group_heads
                                if (layer, head) in source_detail["head_values"]
                            }
                            if patch_vectors:
                                patched = run_to_position(
                                    model=model,
                                    layers=layers,
                                    full_ids=base_ids,
                                    forced_tokens=forced,
                                    position=int(position),
                                    head_layers=head_layers,
                                    num_heads=num_heads,
                                    head_dim=head_dim,
                                    target_layer=int(args.target_layer),
                                    target_site=str(args.target_site),
                                    gate_dir=gate_dir,
                                    regions={name: region_map[base_name].get(name, ()) for name in regions},
                                    patch_vectors=patch_vectors,
                                )
                                effect_rows.append(
                                    {
                                        "example_id": ex_id,
                                        "global_idx": global_idx,
                                        "direction": direction,
                                        "position": int(position),
                                        "group_name": group_name,
                                        "control_type": control_type,
                                        "K": int(k),
                                        "patch_type": "full_head_output",
                                        "region": "full",
                                        "base_gate_proj": base_detail["gate_proj"],
                                        "source_gate_proj": source_detail["gate_proj"],
                                        "patched_gate_proj": patched["gate_proj"],
                                        "effect": patched["gate_proj"] - base_detail["gate_proj"],
                                        "patch_call_count": patched.get("patch_call_count", 0),
                                    }
                                )
                            for region in regions:
                                patch_vectors = {}
                                for layer, head in group_heads:
                                    base_head = base_detail["head_values"].get((layer, head))
                                    base_contrib = base_detail["contribs"].get((region, layer, head))
                                    source_contrib = source_detail["contribs"].get((region, layer, head))
                                    if base_head is None or base_contrib is None or source_contrib is None:
                                        continue
                                    patch_vectors[(layer, head)] = base_head - base_contrib + source_contrib
                                if not patch_vectors:
                                    continue
                                patched = run_to_position(
                                    model=model,
                                    layers=layers,
                                    full_ids=base_ids,
                                    forced_tokens=forced,
                                    position=int(position),
                                    head_layers=head_layers,
                                    num_heads=num_heads,
                                    head_dim=head_dim,
                                    target_layer=int(args.target_layer),
                                    target_site=str(args.target_site),
                                    gate_dir=gate_dir,
                                    regions={name: region_map[base_name].get(name, ()) for name in regions},
                                    patch_vectors=patch_vectors,
                                )
                                effect_rows.append(
                                    {
                                        "example_id": ex_id,
                                        "global_idx": global_idx,
                                        "direction": direction,
                                        "position": int(position),
                                        "group_name": group_name,
                                        "control_type": control_type,
                                        "K": int(k),
                                        "patch_type": "source_contribution",
                                        "region": region,
                                        "base_gate_proj": base_detail["gate_proj"],
                                        "source_gate_proj": source_detail["gate_proj"],
                                        "patched_gate_proj": patched["gate_proj"],
                                        "effect": patched["gate_proj"] - base_detail["gate_proj"],
                                        "patch_call_count": patched.get("patch_call_count", 0),
                                    }
                                )
            if args.print_every > 0 and (local_idx + 1) % int(args.print_every) == 0:
                iterator.set_postfix({"effects": len(effect_rows), "skipped": len(skipped_rows)})
                dump_jsonl(output_dir / "head_patch_effect_rows.partial.jsonl", effect_rows)
                write_csv(output_dir / "head_group_hidden_summary.partial.csv", summarize_effects(effect_rows))
        except Exception as exc:
            skipped_rows.append({"example_id": ex_id, "global_idx": global_idx, "reason": "exception", "error_type": type(exc).__name__, "error": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    dump_jsonl(output_dir / "head_patch_effect_rows.jsonl", effect_rows)
    dump_jsonl(output_dir / "baseline_rows.jsonl", baseline_rows)
    dump_jsonl(output_dir / "attention_mass_rows.jsonl", attention_mass_rows)
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped_rows)
    effect_summary = summarize_effects(effect_rows)
    write_csv(output_dir / "head_group_hidden_summary.csv", effect_summary)
    write_csv(output_dir / "source_contribution_effects.csv", [row for row in effect_summary if row.get("patch_type") == "source_contribution"])
    write_csv(output_dir / "head_group_behavior_summary.csv", [])
    write_csv(output_dir / "pattern_vs_value_effects.csv", [])
    write_csv(output_dir / "random_control_summary.csv", [row for row in effect_summary if "random" in str(row.get("control_type"))])
    reason_counts = Counter(str(row.get("reason")) for row in skipped_rows)
    summary = {
        "experiment": "D4",
        "n_attempted": len(examples),
        "n_usable": len({row["example_id"] for row in baseline_rows}),
        "n_skipped": len(skipped_rows),
        "skip_reasons": dict(reason_counts),
        "effect_rows": len(effect_rows),
        "baseline_rows": len(baseline_rows),
        "attention_mass_rows": len(attention_mass_rows),
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary, effect_summary)
    print("[Done] D4 source-constrained head patch finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- usable_examples: {summary['n_usable']}")
    print(f"- effect_rows: {len(effect_rows)}")


if __name__ == "__main__":
    main()
