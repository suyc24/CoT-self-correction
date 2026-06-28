#!/usr/bin/env python3
from __future__ import annotations

"""Exp 4: L0H3 Output Vector Functional Characterization.

4a: Output vector → logit space (cross-validate with Exp 0 DLA)
4b: Position dependency + OV eigenspectrum (fixed bias vs context-dependent)
4c: Cross-sample consistency (pairwise cosine of head outputs)

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/run_exp4_output_characterization.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --gpu_id 0 \
        --max_examples 100 \
        --output_dir experiment_results/experiments/phase7_exp4/1p7b
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
from cot_research.head_intervention import resolve_head_targets
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.model_utils import (
    get_attention_module,
    get_decoder_layers,
    get_input_device_for_model,
    infer_attention_head_shape,
    parse_head_label,
)
from cot_research.row_utils import resolve_prompt_prefix_from_row
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 4: L0H3 Output Vector Characterization")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--control_heads", type=str, default="L0H0,L0H1,L0H5,L0H7,L0H10")
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
    parser.add_argument("--max_analysis_positions", type=int, default=512)
    parser.add_argument(
        "--system_prompt", type=str,
        default="Please reason step by step in <think>...</think>. Put your final answer within \\boxed{} after the reasoning.",
    )
    parser.add_argument("--assistant_prefix", type=str, default="<think>\n")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class HeadOutputCapture:
    """Captures per-head pre-o_proj slice during step-by-step generation."""
    def __init__(self, attn_module, head_idx, head_dim):
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


def extract_gen_head_outputs(captured: List[torch.Tensor], max_pos: int) -> List[torch.Tensor]:
    """Extract per-generated-token head output vectors from captured hook data."""
    outputs = []
    for i in range(1, min(len(captured), max_pos + 1)):
        outputs.append(captured[i][-1])  # last position of each step
    return outputs


@torch.no_grad()
def analyze_single_example(
    backend, prompt_prefix, gen_config, target_label, control_labels, max_pos,
) -> Optional[Dict[str, Any]]:
    model = backend.model
    tokenizer = backend.tokenizer

    target_heads, attn_modules, _ = resolve_head_targets(model, [target_label])
    target = target_heads[0]
    attn0 = get_attention_module(get_decoder_layers(model)[0][0])
    o_proj_w = attn0.o_proj.weight.detach().cpu().float()

    lm_head_w = None
    for path in ["lm_head.weight"]:
        parts = path.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if isinstance(obj, torch.Tensor):
            lm_head_w = obj.detach().cpu().float()
    if lm_head_w is None:
        raise ValueError("Cannot find lm_head weight")

    # Generate with hooks
    cap = HeadOutputCapture(attn0, target.head_idx, target.head_dim)
    seed_everything(1234)
    with cap:
        result = backend.generate(prompt_prefix, gen_config)

    gen_outputs = extract_gen_head_outputs(cap.captured, max_pos)
    if len(gen_outputs) < 5:
        return None

    prompt_ids = backend.encode(prompt_prefix)
    gen_ids = result.token_ids if hasattr(result, 'token_ids') and result.token_ids else []
    if not gen_ids:
        gen_ids = backend.encode(result.continuation)

    h_start = target.head_idx * target.head_dim
    h_end = (target.head_idx + 1) * target.head_dim

    # 4a: Output → logit space
    logit_diffs = []
    repeat_token_logits = []
    for i, h_out in enumerate(gen_outputs):
        h_contrib = o_proj_w[:, h_start:h_end] @ h_out.float()
        logit_contribution = lm_head_w @ h_contrib  # [vocab_size]

        # tok_repeat = previous token in generation
        if i == 0 and len(prompt_ids) > 0:
            tok_repeat_id = prompt_ids[-1]
        elif i > 0 and i - 1 < len(gen_ids):
            tok_repeat_id = gen_ids[i - 1]
        else:
            continue

        logit_repeat = float(logit_contribution[tok_repeat_id].item())
        logit_contribution_masked = logit_contribution.clone()
        logit_contribution_masked[tok_repeat_id] = float("-inf")
        tok_best_alt_id = int(logit_contribution_masked.argmax().item())
        logit_best_alt = float(logit_contribution[tok_best_alt_id].item())

        logit_diffs.append(logit_repeat - logit_best_alt)
        repeat_token_logits.append(logit_repeat)

    # 4b: Position dependency
    output_vectors = torch.stack([o_proj_w[:, h_start:h_end] @ h.float() for h in gen_outputs])  # [N, hidden]
    n_vecs = output_vectors.shape[0]

    # PCA
    centered = output_vectors - output_vectors.mean(dim=0, keepdim=True)
    if n_vecs > 2:
        U, S, V = torch.svd(centered)
        total_var = (S ** 2).sum().item()
        pc1_var = (S[0] ** 2).item() / max(total_var, 1e-12) * 100
        pc2_var = (S[1] ** 2).item() / max(total_var, 1e-12) * 100 if S.shape[0] > 1 else 0
    else:
        pc1_var = 0
        pc2_var = 0

    # 4c: Cross-position cosine similarity
    norms = output_vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = output_vectors / norms
    cos_matrix = (normalized @ normalized.T)  # [N, N]
    upper_tri = cos_matrix[torch.triu(torch.ones(n_vecs, n_vecs, dtype=torch.bool), diagonal=1)]
    mean_pairwise_cos = float(upper_tri.mean().item()) if upper_tri.numel() > 0 else 0.0
    std_pairwise_cos = float(upper_tri.std().item()) if upper_tri.numel() > 1 else 0.0

    # Position binning: early / mid / late
    third = max(n_vecs // 3, 1)
    bins = {
        "early": output_vectors[:third],
        "mid": output_vectors[third:2*third],
        "late": output_vectors[2*third:],
    }
    bin_means = {k: v.mean(dim=0) for k, v in bins.items() if v.shape[0] > 0}
    cross_bin_cos = {}
    for k1 in bin_means:
        for k2 in bin_means:
            if k1 < k2:
                cos = float(torch.nn.functional.cosine_similarity(
                    bin_means[k1].unsqueeze(0), bin_means[k2].unsqueeze(0),
                ).item())
                cross_bin_cos[f"{k1}_vs_{k2}"] = cos

    return {
        "n_positions": n_vecs,
        "generated_tokens": result.generated_tokens,
        # 4a
        "logit_diff_mean": float(np.mean(logit_diffs)) if logit_diffs else None,
        "logit_diff_std": float(np.std(logit_diffs)) if len(logit_diffs) > 1 else None,
        "repeat_logit_mean": float(np.mean(repeat_token_logits)) if repeat_token_logits else None,
        # 4b
        "pca_pc1_var_pct": pc1_var,
        "pca_pc2_var_pct": pc2_var,
        # 4c
        "mean_pairwise_cosine": mean_pairwise_cos,
        "std_pairwise_cosine": std_pairwise_cos,
        "cross_bin_cosine": cross_bin_cos,
    }


# --------------------------------------------------------------------------- #
# OV Eigenspectrum (weight-only)                                               #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def compute_ov_eigenspectrum(model, layer_idx, head_idx) -> Dict[str, Any]:
    layers, _ = get_decoder_layers(model)
    attn = get_attention_module(layers[layer_idx])
    num_heads, head_dim = infer_attention_head_shape(model, attn)

    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    q_per_kv = num_heads // num_kv_heads
    kv_head_idx = head_idx // q_per_kv

    v_proj_w = attn.v_proj.weight.detach().float()
    o_proj_w = attn.o_proj.weight.detach().float()

    v_start = kv_head_idx * head_dim
    v_end = (kv_head_idx + 1) * head_dim
    o_start = head_idx * head_dim
    o_end = (head_idx + 1) * head_dim

    W_V_slice = v_proj_w[v_start:v_end, :]  # [head_dim, hidden]
    W_O_slice = o_proj_w[:, o_start:o_end]   # [hidden, head_dim]
    W_OV = W_O_slice @ W_V_slice  # [hidden, hidden]

    eigenvalues = torch.linalg.eigvalsh(W_OV @ W_OV.T)
    eigenvalues = eigenvalues.flip(0)  # descending
    total = eigenvalues.sum().item()
    top_eigenvalues = eigenvalues[:20].tolist()
    cumulative_var = [(eigenvalues[:i+1].sum().item() / max(total, 1e-12) * 100)
                      for i in range(min(20, len(eigenvalues)))]

    return {
        "head_label": f"L{layer_idx}H{head_idx}",
        "ov_matrix_shape": list(W_OV.shape),
        "top_20_eigenvalues": top_eigenvalues,
        "cumulative_variance_pct": cumulative_var,
        "top1_var_pct": cumulative_var[0] if cumulative_var else 0,
        "top5_var_pct": cumulative_var[4] if len(cumulative_var) > 4 else 0,
        "effective_rank": sum(1 for ev in eigenvalues.tolist() if ev > eigenvalues[0].item() * 0.01),
    }


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

    # OV Eigenspectrum (weight-only, fast)
    li, hi = parse_head_label(args.head_label)
    print(f"[4b] OV eigenspectrum for {args.head_label}...")
    ov_result = compute_ov_eigenspectrum(backend.model, li, hi)
    write_json(output_dir / "ov_eigenspectrum.json", ov_result)

    # Control heads eigenspectrum
    control_labels = [l.strip() for l in args.control_heads.split(",") if l.strip()]
    control_ov = []
    for cl in control_labels:
        cli, chi = parse_head_label(cl)
        control_ov.append(compute_ov_eigenspectrum(backend.model, cli, chi))
    write_json(output_dir / "control_ov_eigenspectrum.json", control_ov)

    print(f"[4b] {args.head_label} OV: top1_var={ov_result['top1_var_pct']:.1f}%, "
          f"eff_rank={ov_result['effective_rank']}")

    # Per-example analysis (4a + 4b + 4c)
    print(f"[4a/c] Analyzing {len(rows)} examples...")
    results = []
    for idx, row in enumerate(tqdm(rows, desc="Exp4")):
        eid = str(row.get("id", idx))
        try:
            prompt = resolve_prompt_prefix_from_row(row, backend, gen_config)
        except Exception:
            continue

        r = analyze_single_example(
            backend, prompt, gen_config,
            args.head_label, control_labels, args.max_analysis_positions,
        )
        if r:
            r["example_id"] = eid
            results.append(r)

        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    dump_jsonl(output_dir / "per_example.jsonl", results)

    # Aggregate
    if results:
        agg = {
            "n_examples": len(results),
            "logit_diff_mean": float(np.mean([r["logit_diff_mean"] for r in results if r["logit_diff_mean"] is not None])),
            "pca_pc1_var_mean": float(np.mean([r["pca_pc1_var_pct"] for r in results])),
            "pca_pc1_var_std": float(np.std([r["pca_pc1_var_pct"] for r in results])),
            "mean_pairwise_cosine": float(np.mean([r["mean_pairwise_cosine"] for r in results])),
            "std_pairwise_cosine": float(np.mean([r["std_pairwise_cosine"] for r in results])),
        }

        # Fixed bias judgment
        high_pc1 = agg["pca_pc1_var_mean"] > 50
        high_cos = agg["mean_pairwise_cosine"] > 0.8
        agg["judgment"] = "fixed_bias" if high_pc1 and high_cos else "context_dependent"
        agg["fixed_bias_criteria"] = {
            "pca_pc1_gt_50": high_pc1,
            "mean_cosine_gt_0.8": high_cos,
        }

        write_json(output_dir / "summary.json", agg)
        write_csv(output_dir / "per_example_summary.csv", [
            {k: v for k, v in r.items() if k != "cross_bin_cosine"} for r in results
        ])

        print(f"\n{'='*60}")
        print(f"Exp 4 — {args.model_name_or_path} {args.head_label}")
        print(f"{'='*60}")
        print(f"4a: Mean logit diff (repeat-alt): {agg['logit_diff_mean']:.4f}")
        print(f"4b: PCA PC1 variance: {agg['pca_pc1_var_mean']:.1f}% ± {agg['pca_pc1_var_std']:.1f}%")
        print(f"    OV effective rank: {ov_result['effective_rank']}")
        print(f"4c: Mean pairwise cosine: {agg['mean_pairwise_cosine']:.4f}")
        print(f"    Judgment: {agg['judgment']}")


if __name__ == "__main__":
    main()
