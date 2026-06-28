#!/usr/bin/env python3
from __future__ import annotations

"""Exp 3B: Causal Mediation — norm/direction decomposition + project-out.

Sub-experiments:
  3b-1: Zero(target) + norm-preserve patch — keep zero direction, restore baseline norm
  3b-2: Zero(target) + dir-preserve patch  — keep zero norm, restore baseline direction
  3b-3: Baseline + project-out target mean direction from residual

Patch locus: post-attention, pre-MLP residual in the target head's layer.
Hook architecture: input_layernorm.forward_pre_hook (capture residual) +
                   self_attn.forward_hook (modify output to achieve patched residual).

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/run_exp3b_causal_mediation.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --gpu_id 0 \
        --head_label L0H3 \
        --max_examples 100 \
        --output_dir experiment_results/experiments/phase7_exp3b/1p7b_L0H3
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.generation import create_backend
from cot_research.head_intervention import (
    MultiLayerHeadIntervention,
    resolve_head_targets,
)
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.model_utils import (
    get_attention_module,
    get_decoder_layers,
)
from cot_research.repetition_analysis import LoopBenchThresholds, analyze_loopbench_repetition
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 3B: Causal mediation analysis")
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
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--system_prompt", type=str,
        default="Please reason step by step in <think>...</think>. "
                "Put your final answer within \\boxed{} after the reasoning.",
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Hooks: mid-layer residual capture and patching                               #
# --------------------------------------------------------------------------- #

class MidLayerResidualCapture:
    """Captures post-attention, pre-MLP residual at each generation step.

    Hooks input_layernorm (to get pre-attn residual) and self_attn output
    (to compute residual + attn_output). Only captures during autoregressive
    steps (seq_len=1), not during prefill.
    """

    def __init__(self, layer: torch.nn.Module):
        self.layer = layer
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._residual_before: Optional[torch.Tensor] = None
        self._is_prefill = True
        self.captured_norms: List[float] = []
        self.captured_dirs: List[torch.Tensor] = []

    def __enter__(self):
        h1 = self.layer.input_layernorm.register_forward_pre_hook(self._ln_pre)
        h2 = self.layer.self_attn.register_forward_hook(self._attn_post)
        self._handles = [h1, h2]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _ln_pre(self, module, args):
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
            self._is_prefill = x.shape[1] > 1
            if not self._is_prefill:
                self._residual_before = x.detach()

    def _attn_post(self, module, input, output):
        if self._is_prefill or self._residual_before is None:
            self._residual_before = None
            return
        attn_output = output[0] if isinstance(output, tuple) else output
        r = self._residual_before + attn_output.detach()
        r_flat = r[0, 0].float()
        norm = float(r_flat.norm().item())
        self.captured_norms.append(norm)
        if norm > 1e-8:
            self.captured_dirs.append((r_flat / norm).cpu())
        else:
            self.captured_dirs.append(torch.zeros_like(r_flat).cpu())
        self._residual_before = None

    def reset(self):
        self.captured_norms = []
        self.captured_dirs = []
        self._residual_before = None


class HeadOutputCapture:
    """Captures pre-o_proj head slice during generation steps only."""

    def __init__(self, attn_module: torch.nn.Module, head_idx: int, head_dim: int):
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.attn_module = attn_module
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
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
            if x.shape[1] == 1:
                s = self.head_idx * self.head_dim
                e = (self.head_idx + 1) * self.head_dim
                self.captured.append(x[0, 0, s:e].detach().cpu().float())

    def reset(self):
        self.captured = []


class ResidualStreamPatcher:
    """Patches post-attention, pre-MLP residual during generation.

    Modifies self_attn output so that (residual + modified_attn) = patched_residual.
    The residual (pre-attention input) is captured via input_layernorm.forward_pre_hook.
    """

    def __init__(self, layer: torch.nn.Module,
                 patch_fn: Callable[[torch.Tensor, int], torch.Tensor]):
        self.layer = layer
        self.patch_fn = patch_fn
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._residual_before: Optional[torch.Tensor] = None
        self._is_prefill = True
        self.step = 0

    def __enter__(self):
        h1 = self.layer.input_layernorm.register_forward_pre_hook(self._ln_pre)
        h2 = self.layer.self_attn.register_forward_hook(self._attn_post)
        self._handles = [h1, h2]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _ln_pre(self, module, args):
        if args and isinstance(args[0], torch.Tensor):
            x = args[0]
            self._is_prefill = x.shape[1] > 1
            if not self._is_prefill:
                self._residual_before = x

    def _attn_post(self, module, input, output):
        if self._is_prefill or self._residual_before is None:
            self._residual_before = None
            return
        attn_output = output[0] if isinstance(output, tuple) else output
        r_current = self._residual_before + attn_output
        r_patched = self.patch_fn(r_current, self.step)
        modified_attn = r_patched - self._residual_before
        self.step += 1
        self._residual_before = None
        if isinstance(output, tuple):
            return (modified_attn,) + output[1:]
        return modified_attn


# --------------------------------------------------------------------------- #
# Patch functions                                                              #
# --------------------------------------------------------------------------- #

def make_norm_preserve_fn(baseline_norms: List[float]):
    """3b-1: Keep zero(head) direction, restore baseline norm."""
    def fn(r_current: torch.Tensor, step: int) -> torch.Tensor:
        if step >= len(baseline_norms):
            return r_current
        target_norm = baseline_norms[step]
        current_norm = float(r_current.float().norm().item())
        if current_norm < 1e-8:
            return r_current
        scale = target_norm / current_norm
        return r_current * scale
    return fn


def make_dir_preserve_fn(
    baseline_dirs: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
    """3b-2: Keep zero(head) norm, restore baseline direction."""
    def fn(r_current: torch.Tensor, step: int) -> torch.Tensor:
        if step >= len(baseline_dirs):
            return r_current
        target_dir = baseline_dirs[step].to(device=device, dtype=dtype)
        current_norm = float(r_current.float().norm().item())
        return (target_dir * current_norm).unsqueeze(0).unsqueeze(0)
    return fn


def make_project_out_fn(
    direction: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    """3b-3: Project out a fixed direction from the baseline residual."""
    d_unit = direction.to(device=device, dtype=torch.float32)
    d_norm = d_unit.norm()
    if d_norm > 1e-8:
        d_unit = d_unit / d_norm

    def fn(r_current: torch.Tensor, step: int) -> torch.Tensor:
        r_flat = r_current.view(-1).float()
        proj_scalar = torch.dot(r_flat, d_unit)
        r_patched = r_flat - proj_scalar * d_unit
        return r_patched.to(dtype=dtype).view_as(r_current)
    return fn


# --------------------------------------------------------------------------- #
# Per-example analysis                                                         #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_pass1(
    backend,
    prompt_prefix: str,
    gen_config: GenerationConfig,
    target_head_label: str,
) -> Dict[str, Any]:
    """Pass 1: baseline + zero + norm-preserve + dir-preserve.

    Returns per-example results and L0H3 output vectors for global mean direction.
    """
    model = backend.model
    target_heads, attn_modules, _ = resolve_head_targets(model, [target_head_label])
    target = target_heads[0]
    layers, _ = get_decoder_layers(model)
    target_layer = layers[target.layer_idx]
    attn_mod = get_attention_module(target_layer)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    thresholds = LoopBenchThresholds()
    zero_ops = [(target, 0.0)]

    # --- Baseline: capture residuals + head outputs ---
    resid_cap = MidLayerResidualCapture(target_layer)
    head_cap = HeadOutputCapture(attn_mod, target.head_idx, target.head_dim)

    seed_everything(1234)
    with resid_cap, head_cap:
        bl_result = backend.generate(prompt_prefix, gen_config)
    bl_rep = analyze_loopbench_repetition(bl_result.continuation, thresholds=thresholds)
    bl_is_rep = bool(bl_rep["matched"])
    baseline_norms = resid_cap.captured_norms
    baseline_dirs = resid_cap.captured_dirs

    o_proj_w = attn_mod.o_proj.weight.detach().float()
    h_start = target.head_idx * target.head_dim
    h_end = (target.head_idx + 1) * target.head_dim
    head_outputs = [o_proj_w[:, h_start:h_end].cpu() @ h for h in head_cap.captured]

    # --- Zero(target) ---
    seed_everything(1234)
    with MultiLayerHeadIntervention(attn_modules, zero_ops):
        zero_result = backend.generate(prompt_prefix, gen_config)
    zero_rep = analyze_loopbench_repetition(zero_result.continuation, thresholds=thresholds)
    zero_is_rep = bool(zero_rep["matched"])

    # --- 3b-1: Norm-preserve (zero direction + baseline norm) ---
    patcher_np = ResidualStreamPatcher(
        target_layer, make_norm_preserve_fn(baseline_norms),
    )
    seed_everything(1234)
    with MultiLayerHeadIntervention(attn_modules, zero_ops), patcher_np:
        np_result = backend.generate(prompt_prefix, gen_config)
    np_rep = analyze_loopbench_repetition(np_result.continuation, thresholds=thresholds)
    np_is_rep = bool(np_rep["matched"])

    # --- 3b-2: Dir-preserve (baseline direction + zero norm) ---
    patcher_dp = ResidualStreamPatcher(
        target_layer, make_dir_preserve_fn(baseline_dirs, device, dtype),
    )
    seed_everything(1234)
    with MultiLayerHeadIntervention(attn_modules, zero_ops), patcher_dp:
        dp_result = backend.generate(prompt_prefix, gen_config)
    dp_rep = analyze_loopbench_repetition(dp_result.continuation, thresholds=thresholds)
    dp_is_rep = bool(dp_rep["matched"])

    if zero_is_rep and not bl_is_rep:
        group = "A"
    elif not zero_is_rep:
        group = "B"
    elif bl_is_rep and zero_is_rep:
        group = "both_rep"
    else:
        group = "neither"

    return {
        "group": group,
        "bl_is_rep": bl_is_rep,
        "zero_is_rep": zero_is_rep,
        "np_is_rep": np_is_rep,
        "dp_is_rep": dp_is_rep,
        "po_is_rep": None,
        "bl_tokens": bl_result.generated_tokens,
        "zero_tokens": zero_result.generated_tokens,
        "np_tokens": np_result.generated_tokens,
        "dp_tokens": dp_result.generated_tokens,
        "po_tokens": None,
        "n_baseline_residuals": len(baseline_norms),
        "_head_outputs": head_outputs,
    }


@torch.no_grad()
def run_project_out(
    backend,
    prompt_prefix: str,
    gen_config: GenerationConfig,
    target_head_label: str,
    mean_dir: torch.Tensor,
) -> Tuple[bool, int]:
    """Pass 2: project-out target mean direction from baseline residual."""
    model = backend.model
    target_heads, _, _ = resolve_head_targets(model, [target_head_label])
    target = target_heads[0]
    layers, _ = get_decoder_layers(model)
    target_layer = layers[target.layer_idx]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    thresholds = LoopBenchThresholds()

    patcher = ResidualStreamPatcher(
        target_layer, make_project_out_fn(mean_dir, device, dtype),
    )
    seed_everything(1234)
    with patcher:
        po_result = backend.generate(prompt_prefix, gen_config)
    po_rep = analyze_loopbench_repetition(po_result.continuation, thresholds=thresholds)
    return bool(po_rep["matched"]), po_result.generated_tokens


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", {"args": vars(args)})

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

    print(f"[Exp 3B] {len(rows)} examples, head={args.head_label}")

    backend = create_backend(
        BackendConfig(
            backend_type="hf",
            model_name_or_path=args.model_name_or_path,
            device_map={"": args.gpu_id},
            load_in_half=args.load_in_half,
            use_fast_tokenizer=False,
            use_safetensors=True,
            local_files_only=args.local_files_only,
        )
    )
    gen_config = GenerationConfig(
        system_prompt=args.system_prompt,
        assistant_prefix=args.assistant_prefix,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
        enable_thinking=args.enable_thinking,
    )

    # ===== Pass 1: baseline + zero + norm-preserve + dir-preserve =====
    results = []
    all_head_outputs: List[List[torch.Tensor]] = []
    prompts: List[str] = []

    for idx, row in enumerate(tqdm(rows, desc="Pass 1 (bl+zero+np+dp)")):
        eid = str(row.get("id", idx))
        try:
            prompt = resolve_prompt_prefix_from_row(row, backend, gen_config)
        except Exception as exc:
            print(f"[Skip] {eid}: {exc}")
            continue

        prompts.append(prompt)
        r = run_pass1(backend, prompt, gen_config, args.head_label)
        r["example_id"] = eid
        head_outs = r.pop("_head_outputs")
        all_head_outputs.append(head_outs)
        results.append(r)

        if (idx + 1) % 10 == 0:
            n_a = sum(1 for r in results if r["group"] == "A")
            n_b = sum(1 for r in results if r["group"] == "B")
            print(f"[Progress] {idx+1}/{len(rows)}: A={n_a}, B={n_b}")
            torch.cuda.empty_cache()

    # ===== Compute global mean output direction =====
    all_vecs = []
    for outs in all_head_outputs:
        all_vecs.extend(outs)
    if all_vecs:
        stacked = torch.stack(all_vecs)
        mean_dir = stacked.mean(dim=0)
        mean_norm = float(mean_dir.norm().item())
        if mean_norm > 1e-8:
            mean_dir = mean_dir / mean_norm
        print(f"[Info] Mean direction computed from {len(all_vecs)} vectors, "
              f"pre-norm magnitude: {mean_norm:.4f}")
    else:
        mean_dir = torch.zeros(1)
        print("[Warn] No head output vectors captured; skipping project-out")

    # ===== Pass 2: project-out =====
    if mean_dir.numel() > 1:
        for idx in tqdm(range(len(results)), desc="Pass 2 (project-out)"):
            if idx >= len(prompts):
                break
            po_is_rep, po_tokens = run_project_out(
                backend, prompts[idx], gen_config, args.head_label, mean_dir,
            )
            results[idx]["po_is_rep"] = po_is_rep
            results[idx]["po_tokens"] = po_tokens
            if (idx + 1) % 20 == 0:
                torch.cuda.empty_cache()

    # ===== Save results =====
    dump_jsonl(output_dir / "per_example.jsonl", results)

    # Group summary
    groups: Dict[str, List[Dict]] = {}
    for r in results:
        groups.setdefault(r["group"], []).append(r)

    def recovery_rate(group_results: List[Dict], field: str) -> Optional[float]:
        vals = [r[field] for r in group_results if r[field] is not None]
        if not vals:
            return None
        return 1.0 - sum(vals) / len(vals)

    summary_rows = []
    for gname in sorted(groups.keys()):
        gr = groups[gname]
        summary_rows.append({
            "group": gname,
            "count": len(gr),
            "zero_loop_rate": sum(r["zero_is_rep"] for r in gr) / len(gr),
            "np_recovery": recovery_rate(gr, "np_is_rep"),
            "dp_recovery": recovery_rate(gr, "dp_is_rep"),
            "po_loop_rate": (
                sum(1 for r in gr if r.get("po_is_rep")) / len(gr)
            ),
        })
    write_csv(output_dir / "group_summary.csv", summary_rows)

    # Judgment
    ga = groups.get("A", [])
    np_rec = recovery_rate(ga, "np_is_rep")
    dp_rec = recovery_rate(ga, "dp_is_rep")

    if np_rec is not None and dp_rec is not None:
        if np_rec > 0.6 and dp_rec < 0.3:
            judgment = "norm_dominant"
        elif dp_rec > 0.6 and np_rec < 0.3:
            judgment = "direction_dominant"
        elif np_rec > 0.3 and dp_rec > 0.3:
            judgment = "mixed_contribution"
        elif np_rec < 0.3 and dp_rec < 0.3:
            judgment = "neither_sufficient"
        else:
            judgment = "inconclusive"
    else:
        judgment = "insufficient_data"

    po_loop_all = (
        sum(1 for r in results if r.get("po_is_rep"))
        / max(len(results), 1)
    )
    zero_loop_all = (
        sum(1 for r in results if r.get("zero_is_rep"))
        / max(len(results), 1)
    )

    summary = {
        "model": args.model_name_or_path,
        "head": args.head_label,
        "n_total": len(results),
        "n_group_A": len(ga),
        "n_group_B": len(groups.get("B", [])),
        "n_both_rep": len(groups.get("both_rep", [])),
        "n_neither": len(groups.get("neither", [])),
        "zero_loop_rate_all": round(zero_loop_all, 4),
        "np_recovery_A": round(np_rec, 4) if np_rec is not None else None,
        "dp_recovery_A": round(dp_rec, 4) if dp_rec is not None else None,
        "po_loop_rate_all": round(po_loop_all, 4),
        "judgment": judgment,
        "mean_dir_magnitude": float(mean_dir.norm().item()) if mean_dir.numel() > 1 else None,
        "group_summary": summary_rows,
    }
    write_json(output_dir / "summary.json", summary)

    print(f"\n{'='*60}")
    print(f"Exp 3B Causal Mediation — {args.model_name_or_path} {args.head_label}")
    print(f"{'='*60}")
    print(f"Total: {len(results)}, Group A (ablation→loop): {len(ga)}, "
          f"Group B: {len(groups.get('B', []))}")
    if np_rec is not None:
        print(f"  3b-1 Norm-preserve recovery (A): {np_rec:.1%}")
    if dp_rec is not None:
        print(f"  3b-2 Dir-preserve recovery (A):  {dp_rec:.1%}")
    print(f"  3b-3 Project-out loop rate (all): {po_loop_all:.1%}")
    print(f"  Zero-ablation loop rate (all):    {zero_loop_all:.1%}")
    print(f"  Judgment: {judgment}")


if __name__ == "__main__":
    main()
