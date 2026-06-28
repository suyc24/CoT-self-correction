#!/usr/bin/env python3
from __future__ import annotations

"""Exp 3 Phase A: Descriptive analysis of L0H3 mechanism.

For ALL 100 samples (not just collapse-induced), compare baseline vs zero(L0H3):
  - Residual stream norm (post Layer 0 attention, pre MLP)
  - L0H3 output vector norm / fraction of residual
  - L0H3 output vs residual cosine similarity
  - Layer 1-5 attention entropy
  - Post-Layer-0 LayerNorm scale changes
Then stratify by outcome: Group A (ablation → loop) vs Group B (ablation → normal).

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/run_exp3a_descriptive.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --gpu_id 0 \
        --max_examples 100 \
        --output_dir experiment_results/experiments/phase7_exp3a/1p7b
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    resolve_head_targets,
)
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.model_utils import (
    get_attention_module,
    get_decoder_layers,
    get_input_device_for_model,
    infer_attention_head_shape,
)
from cot_research.repetition_analysis import LoopBenchThresholds, analyze_loopbench_repetition
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 3 Phase A: Descriptive analysis")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument(
        "--input_jsonl", type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "loopbench_reconstructed_v2" / "test.jsonl"),
    )
    parser.add_argument("--subtasks", type=str, default="square_root,newtons_iteration")
    parser.add_argument("--max_examples_per_subtask", type=int, default=50)
    parser.add_argument("--max_examples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_analysis_positions", type=int, default=512,
                        help="Max generated positions to capture detailed metrics")
    parser.add_argument(
        "--system_prompt", type=str,
        default="Please reason step by step in <think>...</think>. Put your final answer within \\boxed{} after the reasoning.",
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--downstream_layers", type=int, default=5,
                        help="Number of layers beyond L0 to capture attention entropy")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Hooks                                                                        #
# --------------------------------------------------------------------------- #

class HeadOutputCapture:
    """Captures pre-o_proj head slice during generation (per-step)."""
    def __init__(self, attn_module: torch.nn.Module, head_idx: int, head_dim: int):
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.attn_module = attn_module
        self.handle = None
        self.captured: List[torch.Tensor] = []

    def __enter__(self):
        self.handle = self.attn_module.o_proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()

    def _hook(self, module, args):
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
            s = self.head_idx * self.head_dim
            e = (self.head_idx + 1) * self.head_dim
            self.captured.append(x[0, :, s:e].detach().cpu())

    def reset(self):
        self.captured = []


class ResidualCapture:
    """Captures hidden_states output of a decoder layer."""
    def __init__(self, layer_module: torch.nn.Module):
        self.layer_module = layer_module
        self.handle = None
        self.captured: List[torch.Tensor] = []

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()

    def _hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.captured.append(hidden[0].detach().cpu())

    def reset(self):
        self.captured = []


class LayerNormScaleCapture:
    """Captures the scale (RMS or std) applied by a LayerNorm/RMSNorm."""
    def __init__(self, ln_module: torch.nn.Module):
        self.ln_module = ln_module
        self.handle = None
        self.scales: List[float] = []

    def __enter__(self):
        self.handle = self.ln_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()

    def _hook(self, module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            inp = input[0]
        else:
            inp = input
        if isinstance(inp, torch.Tensor):
            rms = float(inp[0, -1].float().pow(2).mean().sqrt().item())
            self.scales.append(rms)

    def reset(self):
        self.scales = []


# --------------------------------------------------------------------------- #
# Per-example analysis                                                         #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def analyze_example(
    backend,
    prompt_prefix: str,
    gen_config: GenerationConfig,
    target_head_label: str,
    downstream_layers: int,
    max_analysis_positions: int,
) -> Dict[str, Any]:
    model = backend.model
    tokenizer = backend.tokenizer

    target_heads, attn_modules, _ = resolve_head_targets(model, [target_head_label])
    target = target_heads[0]
    layers, _ = get_decoder_layers(model)
    layer0 = layers[0]
    attn0 = get_attention_module(layer0)
    o_proj_weight = attn0.o_proj.weight.detach().cpu().float()

    thresholds = LoopBenchThresholds()

    # --- Baseline generation ---
    seed_everything(1234)
    baseline_result = backend.generate(prompt_prefix, gen_config)
    baseline_rep = analyze_loopbench_repetition(baseline_result.continuation, thresholds=thresholds)
    baseline_is_rep = bool(baseline_rep["matched"])

    # --- Zero(L0H3) generation with hooks ---
    head_cap = HeadOutputCapture(attn0, target.head_idx, target.head_dim)
    resid_cap = ResidualCapture(layer0)

    # LayerNorm capture: post_attention_layernorm (before MLP in layer 0)
    ln_cap = None
    if hasattr(layer0, "post_attention_layernorm"):
        ln_cap = LayerNormScaleCapture(layer0.post_attention_layernorm)

    zero_ops = [(target, 0.0)]
    ctx_managers = [MultiLayerHeadIntervention(attn_modules, zero_ops), head_cap, resid_cap]
    if ln_cap:
        ctx_managers.append(ln_cap)

    seed_everything(1234)
    with ctx_managers[0], ctx_managers[1], ctx_managers[2]:
        if ln_cap:
            with ln_cap:
                zero_result = backend.generate(prompt_prefix, gen_config)
        else:
            zero_result = backend.generate(prompt_prefix, gen_config)

    zero_rep = analyze_loopbench_repetition(zero_result.continuation, thresholds=thresholds)
    zero_is_rep = bool(zero_rep["matched"])

    # --- Also run baseline with hooks (for baseline metrics) ---
    head_cap_bl = HeadOutputCapture(attn0, target.head_idx, target.head_dim)
    resid_cap_bl = ResidualCapture(layer0)
    ln_cap_bl = None
    if hasattr(layer0, "post_attention_layernorm"):
        ln_cap_bl = LayerNormScaleCapture(layer0.post_attention_layernorm)

    seed_everything(1234)
    with head_cap_bl, resid_cap_bl:
        if ln_cap_bl:
            with ln_cap_bl:
                _ = backend.generate(prompt_prefix, gen_config)
        else:
            _ = backend.generate(prompt_prefix, gen_config)

    # --- Extract per-position metrics ---
    def extract_gen_metrics(head_captured, resid_captured, ln_scales, max_pos):
        norms = []
        cosines = []
        fracs = []
        resid_norms = []

        if len(head_captured) < 2:
            return {"head_norms": [], "cosines": [], "fracs": [], "resid_norms": [], "ln_scales": ln_scales[:max_pos]}

        for i in range(1, min(len(head_captured), max_pos + 1)):
            h_out = head_captured[i][-1].float()  # [head_dim]
            r_out = resid_captured[i][-1].float()  # [hidden_dim]

            h_start = target.head_idx * target.head_dim
            h_end = (target.head_idx + 1) * target.head_dim
            h_contrib = o_proj_weight[:, h_start:h_end] @ h_out

            h_norm = float(h_contrib.norm().item())
            r_norm = float(r_out.norm().item())
            cos = 0.0
            if h_norm > 1e-8 and r_norm > 1e-8:
                cos = float(torch.dot(h_contrib, r_out).item() / (h_norm * r_norm))

            norms.append(h_norm)
            resid_norms.append(r_norm)
            cosines.append(cos)
            fracs.append(h_norm / max(r_norm, 1e-8))

        return {
            "head_norms": norms, "cosines": cosines, "fracs": fracs,
            "resid_norms": resid_norms,
            "ln_scales": (ln_scales[1:max_pos+1] if ln_scales else []),
        }

    bl_metrics = extract_gen_metrics(
        head_cap_bl.captured, resid_cap_bl.captured,
        ln_cap_bl.scales if ln_cap_bl else [],
        max_analysis_positions,
    )
    zero_metrics = extract_gen_metrics(
        head_cap.captured, resid_cap.captured,
        ln_cap.scales if ln_cap else [],
        max_analysis_positions,
    )

    # Aggregate
    def agg(vals):
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

    return {
        "baseline_is_rep": baseline_is_rep,
        "zero_is_rep": zero_is_rep,
        "group": "A" if zero_is_rep and not baseline_is_rep else ("B" if not zero_is_rep else "both_rep" if baseline_is_rep and zero_is_rep else "neither"),
        "baseline_gen_tokens": baseline_result.generated_tokens,
        "zero_gen_tokens": zero_result.generated_tokens,
        "n_positions_baseline": len(bl_metrics["head_norms"]),
        "n_positions_zero": len(zero_metrics["head_norms"]),
        # Baseline metrics
        "bl_head_norm": agg(bl_metrics["head_norms"]),
        "bl_resid_norm": agg(bl_metrics["resid_norms"]),
        "bl_cosine": agg(bl_metrics["cosines"]),
        "bl_frac": agg(bl_metrics["fracs"]),
        "bl_ln_scale": agg(bl_metrics["ln_scales"]),
        # Zero metrics
        "zero_resid_norm": agg(zero_metrics["resid_norms"]),
        "zero_ln_scale": agg(zero_metrics["ln_scales"]),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", {"args": vars(args)})

    # Load data
    rows = load_jsonl(args.input_jsonl)
    subtasks = [s.strip() for s in args.subtasks.split(",") if s.strip()]
    if subtasks:
        filtered = []
        for st in subtasks:
            st_rows = [r for r in rows if (r.get("metadata") or {}).get("subtask") == st]
            if args.max_examples_per_subtask > 0:
                st_rows = st_rows[:args.max_examples_per_subtask]
            filtered.extend(st_rows)
        rows = filtered
    if args.max_examples > 0:
        rows = rows[:args.max_examples]

    print(f"[Exp 3A] {len(rows)} examples, head={args.head_label}")

    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map={"": args.gpu_id},
            load_in_half=args.load_in_half,
            use_fast_tokenizer=False, use_safetensors=True,
            local_files_only=args.local_files_only,
        )
    )
    gen_config = GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=False, temperature=0.0,
        enable_thinking=args.enable_thinking,
    )

    results = []
    for idx, row in enumerate(tqdm(rows, desc="Exp3A")):
        eid = str(row.get("id", idx))
        try:
            prompt = resolve_prompt_prefix_from_row(row, backend, gen_config)
        except Exception as exc:
            print(f"[Skip] {eid}: {exc}")
            continue

        r = analyze_example(
            backend, prompt, gen_config,
            args.head_label, args.downstream_layers, args.max_analysis_positions,
        )
        r["example_id"] = eid
        results.append(r)

        if (idx + 1) % 10 == 0:
            n_a = sum(1 for r in results if r["group"] == "A")
            n_b = sum(1 for r in results if r["group"] == "B")
            print(f"[Progress] {idx+1}/{len(rows)}: Group A={n_a}, Group B={n_b}")
            torch.cuda.empty_cache()

    # Save detailed results
    dump_jsonl(output_dir / "per_example.jsonl", results)

    # Summary table
    groups = {"A": [], "B": [], "both_rep": [], "neither": []}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    summary_rows = []
    for gname, gresults in sorted(groups.items()):
        if not gresults:
            continue
        def mean_of(key, subkey):
            vals = [r[key][subkey] for r in gresults if r[key][subkey] is not None]
            return float(np.mean(vals)) if vals else None

        summary_rows.append({
            "group": gname,
            "count": len(gresults),
            "bl_head_norm_mean": mean_of("bl_head_norm", "mean"),
            "bl_resid_norm_mean": mean_of("bl_resid_norm", "mean"),
            "bl_cosine_mean": mean_of("bl_cosine", "mean"),
            "bl_frac_mean": mean_of("bl_frac", "mean"),
            "bl_ln_scale_mean": mean_of("bl_ln_scale", "mean"),
            "zero_resid_norm_mean": mean_of("zero_resid_norm", "mean"),
            "zero_ln_scale_mean": mean_of("zero_ln_scale", "mean"),
            "resid_norm_delta": (
                (mean_of("zero_resid_norm", "mean") or 0) - (mean_of("bl_resid_norm", "mean") or 0)
            ),
            "ln_scale_delta": (
                (mean_of("zero_ln_scale", "mean") or 0) - (mean_of("bl_ln_scale", "mean") or 0)
            ),
        })

    write_csv(output_dir / "group_summary.csv", summary_rows)

    # Overall summary
    n_total = len(results)
    n_a = len(groups.get("A", []))
    n_b = len(groups.get("B", []))
    summary = {
        "model": args.model_name_or_path,
        "head": args.head_label,
        "n_total": n_total,
        "n_group_A": n_a, "n_group_B": n_b,
        "n_both_rep": len(groups.get("both_rep", [])),
        "n_neither": len(groups.get("neither", [])),
        "ablation_loop_rate": round(n_a / max(n_total, 1), 4),
        "group_summary": summary_rows,
    }
    write_json(output_dir / "summary.json", summary)

    print(f"\n{'='*60}")
    print(f"Exp 3A Descriptive — {args.model_name_or_path} {args.head_label}")
    print(f"{'='*60}")
    print(f"Total: {n_total}, Group A (ablation→loop): {n_a}, Group B (ablation→normal): {n_b}")
    for row in summary_rows:
        print(f"\n  Group {row['group']} (n={row['count']}):")
        print(f"    Baseline head norm:  {row['bl_head_norm_mean']:.4f}" if row['bl_head_norm_mean'] else "")
        print(f"    Baseline resid norm: {row['bl_resid_norm_mean']:.4f}" if row['bl_resid_norm_mean'] else "")
        print(f"    Baseline cosine:     {row['bl_cosine_mean']:.4f}" if row['bl_cosine_mean'] else "")
        print(f"    Resid norm Δ(zero-bl): {row['resid_norm_delta']:+.4f}")
        print(f"    LN scale Δ(zero-bl):  {row['ln_scale_delta']:+.4f}")


if __name__ == "__main__":
    main()
