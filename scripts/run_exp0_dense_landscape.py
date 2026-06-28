#!/usr/bin/env python3
from __future__ import annotations

"""Exp 0: Dense Mechanistic Landscape — DLA + SAC + DCS for all heads.

Computes three RSH-specific scoring functions across all attention heads:
  0a: Direct Logit Attribution (DLA) — per-head contribution to repeat-vs-alt logit margin
  0b: Self-Attention Concentration (SAC) — diagonal self-attention mass
  0c: Downstream Composition Score (DCS) — weight-space overlap with downstream heads

Usage (1.7B calibration on GPU 0):
    python scripts/run_exp0_dense_landscape.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --gpu_id 0 \
        --subtask square_root \
        --max_examples 50 \
        --output_dir experiment_results/experiments/phase7_exp0/1p7b_calibration

DCS-only (no data needed, runs on CPU or any GPU):
    python scripts/run_exp0_dense_landscape.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --gpu_id 0 \
        --dcs_only \
        --dcs_source_head L0H3 \
        --dcs_downstream_layers 1,2,3,4,5 \
        --output_dir experiment_results/experiments/phase7_exp0/1p7b_dcs
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.generation import create_backend
from cot_research.head_intervention import list_model_heads, resolve_head_targets
from cot_research.io_utils import load_jsonl, write_csv, write_json
from cot_research.model_utils import (
    get_attention_module,
    get_decoder_layers,
    get_input_device_for_model,
    infer_attention_head_shape,
    parse_head_label,
)
from cot_research.repetition_analysis import LoopBenchThresholds, analyze_loopbench_repetition
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 0: Dense Mechanistic Landscape")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "loopbench_reconstructed_v2" / "test.jsonl"),
    )
    parser.add_argument("--subtask", type=str, default="square_root")
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "Please reason step by step in <think>...</think>. "
            "Put your final answer within \\boxed{} after the reasoning."
        ),
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dcs_only", action="store_true", help="Only compute DCS (no forward pass needed)")
    parser.add_argument("--dcs_source_head", type=str, default="L0H3")
    parser.add_argument("--dcs_downstream_layers", type=str, default="1,2,3,4,5")
    parser.add_argument(
        "--max_gen_positions",
        type=int,
        default=4096,
        help="Max generated positions to analyze per prompt (truncate long generations for memory)",
    )
    parser.add_argument(
        "--baseline_trajectories_jsonl",
        type=str,
        default="",
        help="Pre-computed baseline trajectories to skip generation step",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# DLA + SAC computation                                                        #
# --------------------------------------------------------------------------- #


def _get_lm_head_weight(model: torch.nn.Module) -> torch.Tensor:
    for path in ["lm_head.weight", "model.lm_head.weight"]:
        parts = path.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if isinstance(obj, torch.Tensor):
            return obj
    raise ValueError("Cannot find lm_head weight")


@torch.no_grad()
def compute_dla_sac_for_prompt(
    model: torch.nn.Module,
    tokenizer,
    prompt_prefix: str,
    generation_config: GenerationConfig,
    *,
    max_gen_positions: int = 4096,
) -> Dict[str, Any]:
    """Single-prompt DLA + SAC computation.

    1. Generate baseline trajectory (greedy)
    2. Forward pass on full trajectory with hooks to capture pre-o_proj
    3. Compute DLA and SAC per head per position
    """
    device = get_input_device_for_model(model)
    layers, layer_path = get_decoder_layers(model)
    num_layers = len(layers)
    lm_head_weight = _get_lm_head_weight(model)

    # Step 1: Generate baseline trajectory (capped at max_gen_positions to save time)
    prompt_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        gen_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_gen_positions,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full_ids = gen_output[0].tolist()
    gen_ids = full_ids[len(prompt_ids):]
    if len(gen_ids) > max_gen_positions:
        gen_ids = gen_ids[:max_gen_positions]
    full_ids_truncated = prompt_ids + gen_ids
    num_gen = len(gen_ids)
    if num_gen < 2:
        return {"num_gen_tokens": num_gen, "skipped": True, "reason": "too_short"}

    # Step 2: Forward pass to get logits + hook pre_o_proj + attentions
    full_tensor = torch.tensor([full_ids_truncated], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(full_tensor)

    captured_pre_oproj = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, args):
            if args and isinstance(args[0], torch.Tensor):
                captured_pre_oproj[layer_idx] = args[0].detach()
            return None
        return hook_fn

    for li in range(num_layers):
        attn_module = get_attention_module(layers[li])
        h = attn_module.o_proj.register_forward_pre_hook(make_hook(li))
        hooks.append(h)

    try:
        outputs = model(
            input_ids=full_tensor,
            attention_mask=attn_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for h in hooks:
            h.remove()

    logits = outputs.logits[0]  # [seq_len, vocab_size]
    attentions = getattr(outputs, "attentions", None)  # tuple of [1, num_heads, seq_len, seq_len] or None
    has_attentions = attentions is not None and attentions[0] is not None

    # Step 3: Compute tok_repeat, tok_best_alt, repeat-prone positions
    prompt_len = len(prompt_ids)
    gen_start = prompt_len
    gen_end = prompt_len + num_gen

    # For position t (0-indexed in full sequence):
    # tok_repeat(t) = full_ids_truncated[t-1] (the previous token)
    # tok_best_alt(t) = argmax(logits[t]) excluding tok_repeat(t)
    # DLA is computed only at generated positions (gen_start <= t < gen_end)

    results_per_head = defaultdict(lambda: {
        "dla_all": [], "dla_repeat_prone": [], "dla_non_repeat": [],
        "sac_all": [],
    })

    gen_positions_info = []
    for t in range(gen_start, gen_end):
        tok_repeat = full_ids_truncated[t - 1]
        logit_t = logits[t]  # [vocab_size]

        # Determine tok_best_alt
        logit_repeat = float(logit_t[tok_repeat].item())
        logit_t_masked = logit_t.clone()
        logit_t_masked[tok_repeat] = float("-inf")
        tok_best_alt = int(logit_t_masked.argmax().item())
        logit_best_alt = float(logit_t[tok_best_alt].item())

        # Check if repeat-prone: tok_repeat in top-3 logits
        top3_ids = logit_t.topk(3).indices.tolist()
        is_repeat_prone = tok_repeat in top3_ids

        # Compute unembed direction: W_unembed[:, tok_repeat] - W_unembed[:, tok_best_alt]
        unembed_diff = lm_head_weight[tok_repeat] - lm_head_weight[tok_best_alt]  # [hidden_dim]

        gen_positions_info.append({
            "t": t,
            "tok_repeat": tok_repeat,
            "tok_best_alt": tok_best_alt,
            "logit_margin": logit_repeat - logit_best_alt,
            "is_repeat_prone": is_repeat_prone,
        })

        # Compute DLA for each head at this position
        for li in range(num_layers):
            pre_oproj = captured_pre_oproj[li]  # [1, seq_len, num_heads*head_dim]
            attn_module = get_attention_module(layers[li])
            num_heads, head_dim = infer_attention_head_shape(model, attn_module)
            o_proj_weight = attn_module.o_proj.weight.detach()  # [hidden_dim, num_heads*head_dim]

            for hi in range(num_heads):
                h_start = hi * head_dim
                h_end = (hi + 1) * head_dim
                head_pre = pre_oproj[0, t, h_start:h_end]  # [head_dim]
                head_output = o_proj_weight[:, h_start:h_end] @ head_pre  # [hidden_dim]
                dla_val = float(torch.dot(head_output, unembed_diff).item())

                label = f"L{li}H{hi}"
                results_per_head[label]["dla_all"].append(dla_val)
                if is_repeat_prone:
                    results_per_head[label]["dla_repeat_prone"].append(dla_val)
                else:
                    results_per_head[label]["dla_non_repeat"].append(dla_val)

        # Compute SAC for each head at this position (only if attentions available)
        if has_attentions:
            for li in range(num_layers):
                attn_weights = attentions[li][0]  # [num_heads, seq_len, seq_len]
                attn_module = get_attention_module(layers[li])
                num_heads, head_dim = infer_attention_head_shape(model, attn_module)
                for hi in range(num_heads):
                    sac_val = float(attn_weights[hi, t, t].item())
                    label = f"L{li}H{hi}"
                    results_per_head[label]["sac_all"].append(sac_val)

    # Free captured tensors
    del captured_pre_oproj, attentions, logits, outputs
    torch.cuda.empty_cache()

    # Aggregate per head
    head_summaries = []
    for label, data in sorted(results_per_head.items()):
        li, hi = parse_head_label(label)
        head_summaries.append({
            "head_label": label,
            "layer_idx": li,
            "head_idx": hi,
            "dla_mean_all": float(np.mean(data["dla_all"])) if data["dla_all"] else 0.0,
            "dla_mean_repeat_prone": float(np.mean(data["dla_repeat_prone"])) if data["dla_repeat_prone"] else None,
            "dla_mean_non_repeat": float(np.mean(data["dla_non_repeat"])) if data["dla_non_repeat"] else None,
            "n_repeat_prone": len(data["dla_repeat_prone"]),
            "n_non_repeat": len(data["dla_non_repeat"]),
            "sac_mean": float(np.mean(data["sac_all"])) if data["sac_all"] else 0.0,
        })

    # Repetition analysis on generated text
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    rep_result = analyze_loopbench_repetition(gen_text)

    return {
        "num_gen_tokens": num_gen,
        "skipped": False,
        "is_repetitive": bool(rep_result["matched"]),
        "head_summaries": head_summaries,
        "n_positions_analyzed": len(gen_positions_info),
        "n_repeat_prone_positions": sum(1 for p in gen_positions_info if p["is_repeat_prone"]),
    }


# --------------------------------------------------------------------------- #
# DCS computation (weight-only)                                                #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_dcs(
    model: torch.nn.Module,
    source_layer_idx: int,
    source_head_idx: int,
    downstream_layer_idxs: List[int],
) -> List[Dict[str, Any]]:
    layers, layer_path = get_decoder_layers(model)

    src_attn = get_attention_module(layers[source_layer_idx])
    src_num_heads, src_head_dim = infer_attention_head_shape(model, src_attn)
    src_o_proj = src_attn.o_proj.weight.detach().float()  # [hidden, num_heads*head_dim]
    h_start = source_head_idx * src_head_dim
    h_end = (source_head_idx + 1) * src_head_dim
    W_O_h = src_o_proj[:, h_start:h_end]  # [hidden, head_dim]
    W_O_h_norm = W_O_h.norm(p="fro")

    results = []
    for dl in downstream_layer_idxs:
        if dl >= len(layers):
            continue
        dst_attn = get_attention_module(layers[dl])
        dst_num_heads, dst_head_dim = infer_attention_head_shape(model, dst_attn)
        num_kv_heads = getattr(dst_attn, "num_key_value_heads", dst_num_heads)
        q_per_kv = dst_num_heads // num_kv_heads

        W_K = dst_attn.k_proj.weight.detach().float()  # [num_kv_heads*head_dim, hidden]
        W_V = dst_attn.v_proj.weight.detach().float()

        for dhi in range(dst_num_heads):
            kv_hi = dhi // q_per_kv
            k_start = kv_hi * dst_head_dim
            k_end = (kv_hi + 1) * dst_head_dim

            W_K_j = W_K[k_start:k_end, :]  # [head_dim, hidden]
            W_V_j = W_V[k_start:k_end, :]

            composition_K = W_K_j @ W_O_h  # [head_dim, head_dim]
            composition_V = W_V_j @ W_O_h

            dcs_k = float(composition_K.norm(p="fro").item() / (W_K_j.norm(p="fro").item() * W_O_h_norm.item() + 1e-12))
            dcs_v = float(composition_V.norm(p="fro").item() / (W_V_j.norm(p="fro").item() * W_O_h_norm.item() + 1e-12))

            results.append({
                "source_head": f"L{source_layer_idx}H{source_head_idx}",
                "downstream_head": f"L{dl}H{dhi}",
                "downstream_layer": dl,
                "downstream_head_idx": dhi,
                "dcs_k": dcs_k,
                "dcs_v": dcs_v,
            })

    return results


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "run_config.json", {"args": vars(args)})

    # Load model — eager attention required for output_attentions=True
    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map={"": args.gpu_id},
            load_in_half=args.load_in_half,
            use_fast_tokenizer=False,
            use_safetensors=True,
            local_files_only=args.local_files_only,
            attn_implementation="eager",
        )
    )
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("HF backend required")

    generation_config = GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        enable_thinking=args.enable_thinking,
    )

    # DCS computation (always run)
    src_layer, src_head = parse_head_label(args.dcs_source_head)
    ds_layers = [int(x.strip()) for x in args.dcs_downstream_layers.split(",") if x.strip()]
    print(f"[DCS] Computing DCS for {args.dcs_source_head} → layers {ds_layers}")
    dcs_results = compute_dcs(model, src_layer, src_head, ds_layers)
    write_csv(output_dir / "dcs_scores.csv", dcs_results)

    # DCS summary: top downstream heads
    dcs_k_sorted = sorted(dcs_results, key=lambda x: x["dcs_k"], reverse=True)
    all_dcs_k_vals = [r["dcs_k"] for r in dcs_results]
    dcs_k_95th = float(np.percentile(all_dcs_k_vals, 95)) if all_dcs_k_vals else 0.0
    dcs_top_heads = [r for r in dcs_k_sorted if r["dcs_k"] > dcs_k_95th]
    write_json(output_dir / "dcs_summary.json", {
        "source_head": args.dcs_source_head,
        "downstream_layers": ds_layers,
        "total_downstream_heads": len(dcs_results),
        "dcs_k_95th_percentile": dcs_k_95th,
        "top_heads_above_95th": [r["downstream_head"] for r in dcs_top_heads],
        "top_10_by_dcs_k": [{
            "head": r["downstream_head"], "dcs_k": round(r["dcs_k"], 6), "dcs_v": round(r["dcs_v"], 6),
        } for r in dcs_k_sorted[:10]],
    })
    print(f"[DCS] DCS 95th percentile: {dcs_k_95th:.4f}")
    print(f"[DCS] Top heads above 95th: {[r['downstream_head'] for r in dcs_top_heads[:5]]}")

    if args.dcs_only:
        print("[Done] DCS-only mode complete.")
        return

    # DLA + SAC computation
    rows = load_jsonl(args.input_jsonl)
    if args.subtask:
        rows = [r for r in rows if (r.get("metadata") or {}).get("subtask") == args.subtask]
    if args.max_examples > 0:
        rows = rows[:args.max_examples]
    if not rows:
        raise ValueError(f"No examples found for subtask={args.subtask}")

    print(f"[DLA+SAC] Processing {len(rows)} examples from subtask={args.subtask}")

    all_head_summaries = defaultdict(lambda: {
        "dla_all": [], "dla_repeat_prone": [], "dla_non_repeat": [], "sac_all": [],
    })
    prompt_results = []

    for idx, row in enumerate(tqdm(rows, desc="DLA+SAC")):
        example_id = str(row.get("id", idx))
        try:
            prompt_prefix = resolve_prompt_prefix_from_row(row, backend, generation_config)
        except Exception as exc:
            print(f"[Warning] Skipping {example_id}: {exc}")
            continue

        result = compute_dla_sac_for_prompt(
            model, tokenizer, prompt_prefix, generation_config,
            max_gen_positions=args.max_gen_positions,
        )

        if result.get("skipped"):
            print(f"[Warning] Skipped {example_id}: {result.get('reason')}")
            continue

        for hs in result["head_summaries"]:
            label = hs["head_label"]
            all_head_summaries[label]["dla_all"].append(hs["dla_mean_all"])
            if hs["dla_mean_repeat_prone"] is not None:
                all_head_summaries[label]["dla_repeat_prone"].append(hs["dla_mean_repeat_prone"])
            if hs["dla_mean_non_repeat"] is not None:
                all_head_summaries[label]["dla_non_repeat"].append(hs["dla_mean_non_repeat"])
            all_head_summaries[label]["sac_all"].append(hs["sac_mean"])

        prompt_results.append({
            "example_id": example_id,
            "num_gen_tokens": result["num_gen_tokens"],
            "is_repetitive": result["is_repetitive"],
            "n_positions_analyzed": result["n_positions_analyzed"],
            "n_repeat_prone_positions": result["n_repeat_prone_positions"],
        })

        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # Aggregate across prompts
    landscape_rows = []
    for label, data in sorted(all_head_summaries.items()):
        li, hi = parse_head_label(label)
        landscape_rows.append({
            "head_label": label,
            "layer_idx": li,
            "head_idx": hi,
            "dla_mean_all": float(np.mean(data["dla_all"])) if data["dla_all"] else 0.0,
            "dla_std_all": float(np.std(data["dla_all"])) if len(data["dla_all"]) > 1 else 0.0,
            "dla_mean_repeat_prone": float(np.mean(data["dla_repeat_prone"])) if data["dla_repeat_prone"] else None,
            "dla_mean_non_repeat": float(np.mean(data["dla_non_repeat"])) if data["dla_non_repeat"] else None,
            "sac_mean": float(np.mean(data["sac_all"])) if data["sac_all"] else 0.0,
            "n_prompts": len(data["dla_all"]),
        })

    write_csv(output_dir / "landscape.csv", landscape_rows)
    write_csv(output_dir / "prompt_results.csv", prompt_results)

    # Rankings
    # DLA rank (lower = more negative = more suppressive)
    dla_sorted = sorted(landscape_rows, key=lambda r: r["dla_mean_all"])
    for rank, r in enumerate(dla_sorted, start=1):
        r["dla_rank"] = rank

    # DLA at repeat-prone positions
    rp_rows = [r for r in landscape_rows if r["dla_mean_repeat_prone"] is not None]
    rp_sorted = sorted(rp_rows, key=lambda r: r["dla_mean_repeat_prone"])
    for rank, r in enumerate(rp_sorted, start=1):
        r["dla_rp_rank"] = rank

    # SAC rank (descending)
    sac_sorted = sorted(landscape_rows, key=lambda r: r["sac_mean"], reverse=True)
    for rank, r in enumerate(sac_sorted, start=1):
        r["sac_rank"] = rank

    write_csv(output_dir / "landscape_ranked.csv", landscape_rows)

    # Calibration report
    total_heads = len(landscape_rows)
    target_head_label = args.dcs_source_head
    target_row = next((r for r in landscape_rows if r["head_label"] == target_head_label), None)

    calibration = {
        "model": args.model_name_or_path,
        "subtask": args.subtask,
        "n_examples": len(prompt_results),
        "total_heads": total_heads,
        "target_head": target_head_label,
    }
    if target_row:
        calibration.update({
            "target_dla_mean_all": target_row["dla_mean_all"],
            "target_dla_rank": target_row.get("dla_rank"),
            "target_dla_rank_percentile": round(target_row.get("dla_rank", total_heads) / total_heads * 100, 2),
            "target_dla_rp_mean": target_row.get("dla_mean_repeat_prone"),
            "target_dla_rp_rank": target_row.get("dla_rp_rank"),
            "target_sac_mean": target_row["sac_mean"],
            "target_sac_rank": target_row.get("sac_rank"),
        })
        for k in [1, 3, 5, 10]:
            top_k_labels = [r["head_label"] for r in dla_sorted[:k]]
            calibration[f"recall_at_{k}_dla"] = 1.0 if target_head_label in top_k_labels else 0.0
        for k in [1, 3, 5, 10]:
            top_k_labels = [r["head_label"] for r in rp_sorted[:k]]
            calibration[f"recall_at_{k}_dla_rp"] = 1.0 if target_head_label in top_k_labels else 0.0
    else:
        calibration["target_found"] = False

    write_json(output_dir / "calibration.json", calibration)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Exp 0 Dense Mechanistic Landscape — {args.model_name_or_path}")
    print(f"{'='*60}")
    print(f"Examples: {len(prompt_results)}, Total heads: {total_heads}")
    if target_row:
        print(f"\n{target_head_label} calibration:")
        print(f"  DLA rank (all): {target_row.get('dla_rank')}/{total_heads} "
              f"(mean={target_row['dla_mean_all']:.4f})")
        if target_row.get("dla_rp_rank"):
            print(f"  DLA rank (repeat-prone): {target_row.get('dla_rp_rank')}/{len(rp_rows)} "
                  f"(mean={target_row.get('dla_mean_repeat_prone'):.4f})")
        print(f"  SAC rank: {target_row.get('sac_rank')}/{total_heads} "
              f"(mean={target_row['sac_mean']:.4f})")
        print(f"  Pass criterion (DLA rank ≤ 10): "
              f"{'PASS' if (target_row.get('dla_rank', 999) <= 10) else 'FAIL'}")
    print(f"\nTop 10 heads by DLA (most negative = most suppressive):")
    for r in dla_sorted[:10]:
        print(f"  {r['head_label']:>6s}: DLA={r['dla_mean_all']:+.4f}  SAC={r['sac_mean']:.3f}")
    print(f"\nDCS top heads (>{dcs_k_95th:.4f}):")
    for r in dcs_top_heads[:5]:
        print(f"  {r['downstream_head']:>6s}: DCS_K={r['dcs_k']:.4f}  DCS_V={r['dcs_v']:.4f}")


if __name__ == "__main__":
    main()
