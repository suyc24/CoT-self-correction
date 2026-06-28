#!/usr/bin/env python3
"""Experiment 1: Residual Stream Dynamics at Collapse Point.

For samples where zero(L0H3) causes repetition collapse, analyze:
1. L0H3 output vector norm at each generated position
2. Residual stream norm before/after L0H3 contribution
3. L0H3 output fractional contribution to residual
4. Cosine similarity between L0H3 output and residual stream
5. Compare collapse vs. non-collapse positions

Runs two conditions per example:
  - baseline: normal generation → identify positions
  - zero(L0H3): ablated generation → find collapse onset, measure residual dynamics

Additionally collects the same metrics for control heads (random Layer 0 heads)
to verify that L0H3's contribution is structurally special.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.generation import HFGenerationBackend
from cot_research.head_intervention import MultiLayerHeadIntervention, resolve_head_targets
from cot_research.io_utils import dump_jsonl, load_jsonl, write_csv, write_json
from cot_research.model_utils import get_input_device_for_model
from cot_research.repetition_analysis import RepetitionThresholds, analyze_repetition
from cot_research.runtime_utils import resolve_device_map, seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze residual stream at collapse point (Experiment 1)")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--head_label", type=str, default="L0H3")
    parser.add_argument("--control_heads", type=str, default="L0H0,L0H1,L0H5,L0H7,L0H10",
                        help="Comma-separated control head labels for comparison")
    parser.add_argument("--input_jsonl", type=str,
                        default=str(ROOT_DIR / "evaluation" / "data" / "gsm8k" / "test.jsonl"))
    parser.add_argument("--output_dir", type=str,
                        default=str(ROOT_DIR / "outputs" / "residual_at_collapse"))
    parser.add_argument("--max_examples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system_prompt", type=str,
                        default="Please reason step by step in <think>...</think>. "
                                "Put your final answer within \\boxed{} after the reasoning.")
    parser.add_argument("--context_window", type=int, default=20,
                        help="Number of tokens before/after collapse onset to analyze in detail")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hook to capture L0H3 output vector (pre o_proj) at every position
# ---------------------------------------------------------------------------

class HeadOutputCaptureHook:
    """Captures the per-head slice of the o_proj input at every forward step."""

    def __init__(self, attn_module: torch.nn.Module, head_idx: int, head_dim: int):
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.attn_module = attn_module
        self.captured: List[torch.Tensor] = []  # list of [seq_len, head_dim] tensors

    def __enter__(self):
        self.handle = self.attn_module.o_proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, args):
        if not args:
            return None
        x = args[0]  # [batch, seq, hidden]
        start = self.head_idx * self.head_dim
        end = (self.head_idx + 1) * self.head_dim
        # Capture the head's slice (detached, on CPU to save VRAM)
        self.captured.append(x[0, :, start:end].detach().cpu())
        return None

    def reset(self):
        self.captured = []


class ResidualCaptureHook:
    """Captures residual stream (hidden_states) after layer 0 at every forward step."""

    def __init__(self, layer_module: torch.nn.Module):
        self.layer_module = layer_module
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.captured: List[torch.Tensor] = []

    def __enter__(self):
        self.handle = self.layer_module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, input, output):
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.captured.append(hidden[0].detach().cpu())  # [seq, hidden]
        return None

    def reset(self):
        self.captured = []


def find_collapse_onset(token_ids: List[int], min_run: int = 8) -> Optional[int]:
    """Find the position where same-token collapse begins (run of min_run identical tokens)."""
    if len(token_ids) < min_run:
        return None
    cur_run = 1
    for i in range(1, len(token_ids)):
        if token_ids[i] == token_ids[i - 1]:
            cur_run += 1
            if cur_run >= min_run:
                return i - min_run + 1
        else:
            cur_run = 1
    return None


def compute_position_metrics(
    head_output: torch.Tensor,  # [head_dim]
    residual: torch.Tensor,     # [hidden_dim]
    o_proj_weight: torch.Tensor,  # [hidden_dim, hidden_dim] or relevant slice
    head_idx: int,
    head_dim: int,
) -> Dict[str, float]:
    """Compute metrics for a single position."""
    # L0H3 output through o_proj to get its contribution to residual stream
    start = head_idx * head_dim
    end = (head_idx + 1) * head_dim
    # The o_proj maps from [hidden_dim] -> [hidden_dim], but the input has
    # per-head slices concatenated. The contribution of head h is:
    # o_proj.weight[:, start:end] @ head_output
    head_contribution = o_proj_weight[:, start:end].float() @ head_output.float()

    residual_f = residual.float()
    head_norm = head_contribution.norm().item()
    residual_norm = residual_f.norm().item()

    # Cosine similarity between head contribution and residual
    cos_sim = 0.0
    if head_norm > 1e-8 and residual_norm > 1e-8:
        cos_sim = (torch.dot(head_contribution, residual_f) / (head_norm * residual_norm)).item()

    # Fractional contribution (ratio of norms)
    frac = head_norm / max(residual_norm, 1e-8)

    return {
        "head_output_norm": float(head_output.norm().item()),
        "head_contribution_norm": head_norm,
        "residual_norm": residual_norm,
        "cosine_sim": cos_sim,
        "fractional_contribution": frac,
    }


@torch.no_grad()
def run_single_example(
    backend: HFGenerationBackend,
    prompt_prefix: str,
    gen_config: GenerationConfig,
    target_head_label: str,
    control_head_labels: List[str],
    context_window: int,
) -> Optional[Dict[str, Any]]:
    """Run baseline + zero(target) and collect residual metrics."""
    model = backend.model
    tokenizer = backend.tokenizer

    # Resolve heads
    target_heads, attn_modules, _ = resolve_head_targets(model, [target_head_label])
    target = target_heads[0]

    from cot_research.model_utils import get_decoder_layers
    decoder_layers, _ = get_decoder_layers(model)
    layer0 = decoder_layers[0]
    attn0 = attn_modules[0]  # layer 0 attention module
    o_proj_weight = attn0.o_proj.weight.detach().cpu()  # [hidden, hidden]

    # --- Step 1: Baseline generation ---
    baseline_result = backend.generate(prompt_prefix, gen_config)
    baseline_rep = analyze_repetition(baseline_result.continuation, baseline_result.token_ids)

    # --- Step 2: Zero(target) generation with hooks ---
    head_capture = HeadOutputCaptureHook(attn0, target.head_idx, target.head_dim)
    residual_capture = ResidualCaptureHook(layer0)
    zero_ops = [(target, 0.0)]

    with MultiLayerHeadIntervention(attn_modules, zero_ops), \
         head_capture, residual_capture:
        zero_result = backend.generate(prompt_prefix, gen_config)

    zero_rep = analyze_repetition(zero_result.continuation, zero_result.token_ids)

    # Only keep examples where zero causes collapse but baseline doesn't
    if not zero_rep.get("matched", False) or baseline_rep.get("matched", False):
        return None

    collapse_onset = find_collapse_onset(zero_result.token_ids, min_run=8)
    if collapse_onset is None:
        return None

    # --- Step 3: Extract per-position metrics ---
    # The head_capture contains one entry per forward call.
    # For generation with KV cache, the first call processes the full prompt,
    # then each subsequent call processes 1 token.
    # We need to reconstruct the per-generated-token head outputs.
    prompt_ids = backend.encode(prompt_prefix)
    n_prompt = len(prompt_ids)
    n_gen = len(zero_result.token_ids)

    # Reconstruct: first capture is prompt (n_prompt positions),
    # then one capture per generated token (1 position each)
    head_outputs_per_gen_token = []
    residuals_per_gen_token = []

    if len(head_capture.captured) >= 1:
        # First call: [n_prompt, head_dim]
        first_head = head_capture.captured[0]
        first_resid = residual_capture.captured[0]
        # last position of prompt is the "0th prediction position"
        # Subsequent captures are one-by-one
        for i in range(1, min(len(head_capture.captured), n_gen + 1)):
            head_outputs_per_gen_token.append(head_capture.captured[i][-1])  # [head_dim]
            residuals_per_gen_token.append(residual_capture.captured[i][-1])  # [hidden]

    if len(head_outputs_per_gen_token) == 0:
        return None

    # Compute metrics around collapse onset
    window_start = max(0, collapse_onset - context_window)
    window_end = min(len(head_outputs_per_gen_token), collapse_onset + context_window)

    position_metrics = []
    for pos in range(window_start, window_end):
        if pos >= len(head_outputs_per_gen_token):
            break
        m = compute_position_metrics(
            head_outputs_per_gen_token[pos],
            residuals_per_gen_token[pos],
            o_proj_weight,
            target.head_idx,
            target.head_dim,
        )
        m["position"] = pos
        m["relative_to_collapse"] = pos - collapse_onset
        m["token_id"] = zero_result.token_ids[pos] if pos < n_gen else -1
        m["token_text"] = tokenizer.decode([m["token_id"]]) if m["token_id"] >= 0 else ""
        position_metrics.append(m)

    # Also compute global statistics (all generated positions)
    all_norms = []
    all_cos = []
    all_frac = []
    for pos in range(min(len(head_outputs_per_gen_token), n_gen)):
        m = compute_position_metrics(
            head_outputs_per_gen_token[pos],
            residuals_per_gen_token[pos],
            o_proj_weight,
            target.head_idx,
            target.head_dim,
        )
        all_norms.append(m["head_contribution_norm"])
        all_cos.append(m["cosine_sim"])
        all_frac.append(m["fractional_contribution"])

    # --- Step 4: Baseline residual metrics for comparison ---
    # Re-run baseline with capture hooks (no intervention)
    head_capture_bl = HeadOutputCaptureHook(attn0, target.head_idx, target.head_dim)
    residual_capture_bl = ResidualCaptureHook(layer0)
    with head_capture_bl, residual_capture_bl:
        _ = backend.generate(prompt_prefix, gen_config)

    baseline_norms = []
    for i in range(1, min(len(head_capture_bl.captured), len(baseline_result.token_ids) + 1)):
        ho = head_capture_bl.captured[i][-1]
        rd = residual_capture_bl.captured[i][-1]
        m = compute_position_metrics(ho, rd, o_proj_weight, target.head_idx, target.head_dim)
        baseline_norms.append(m["head_contribution_norm"])

    return {
        "collapse_onset": collapse_onset,
        "baseline_tokens": baseline_result.generated_tokens,
        "zero_tokens": zero_result.generated_tokens,
        "baseline_repetition": baseline_rep.get("matched", False),
        "zero_repetition": zero_rep.get("matched", False),
        "collapse_token_id": zero_result.token_ids[collapse_onset] if collapse_onset < n_gen else -1,
        "collapse_token_text": tokenizer.decode([zero_result.token_ids[collapse_onset]]) if collapse_onset < n_gen else "",
        "position_metrics": position_metrics,
        "global_mean_head_norm_zero": sum(all_norms) / max(len(all_norms), 1),
        "global_mean_cosine_zero": sum(all_cos) / max(len(all_cos), 1),
        "global_mean_frac_zero": sum(all_frac) / max(len(all_frac), 1),
        "global_mean_head_norm_baseline": sum(baseline_norms) / max(len(baseline_norms), 1),
        "n_positions_analyzed": len(position_metrics),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    write_json(output_dir / "run_config.json", vars(args))

    # Load data
    examples = load_jsonl(args.input_jsonl)
    if args.max_examples > 0:
        examples = examples[:args.max_examples]

    # Create backend
    device_map = resolve_device_map("auto", args.gpu_id)
    backend_config = BackendConfig(
        backend_type="hf",
        model_name_or_path=args.model_name_or_path,
        device_map=device_map,
        load_in_half=True,
        local_files_only=args.local_files_only,
    )
    backend = HFGenerationBackend(backend_config)

    gen_config = GenerationConfig(
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    control_heads = [h.strip() for h in args.control_heads.split(",") if h.strip()]

    # Run examples
    rows = []
    collapse_count = 0
    for idx, ex in enumerate(tqdm(examples, desc="Processing examples")):
        question = ex.get("question", ex.get("problem", ""))
        if not question:
            continue

        prompt_prefix = backend.build_prompt(question, gen_config)
        result = run_single_example(
            backend, prompt_prefix, gen_config,
            args.head_label, control_heads, args.context_window,
        )

        if result is not None:
            result["example_idx"] = idx
            result["example_id"] = ex.get("id", str(idx))
            rows.append(result)
            collapse_count += 1
            tqdm.write(f"  [collapse #{collapse_count}] onset={result['collapse_onset']}, "
                       f"token='{result['collapse_token_text']}', "
                       f"mean_head_norm_zero={result['global_mean_head_norm_zero']:.4f}, "
                       f"mean_head_norm_baseline={result['global_mean_head_norm_baseline']:.4f}")

    # Save results
    # Flatten position_metrics into separate file
    position_rows = []
    for row in rows:
        eid = row["example_id"]
        for pm in row.get("position_metrics", []):
            pm_row = {"example_id": eid, **pm}
            position_rows.append(pm_row)

    summary_rows = []
    for row in rows:
        sr = {k: v for k, v in row.items() if k != "position_metrics"}
        summary_rows.append(sr)

    dump_jsonl(output_dir / "rows.jsonl", summary_rows)
    write_csv(output_dir / "position_metrics.csv", position_rows)

    # Aggregate summary
    if rows:
        summary = {
            "total_examples": len(examples),
            "collapse_examples": collapse_count,
            "mean_collapse_onset": sum(r["collapse_onset"] for r in rows) / len(rows),
            "mean_head_norm_zero": sum(r["global_mean_head_norm_zero"] for r in rows) / len(rows),
            "mean_head_norm_baseline": sum(r["global_mean_head_norm_baseline"] for r in rows) / len(rows),
            "mean_cosine_zero": sum(r["global_mean_cosine_zero"] for r in rows) / len(rows),
            "mean_frac_zero": sum(r["global_mean_frac_zero"] for r in rows) / len(rows),
        }
    else:
        summary = {"total_examples": len(examples), "collapse_examples": 0}
    write_json(output_dir / "summary.json", summary)
    print(f"\nDone. {collapse_count}/{len(examples)} examples collapsed.")
    print(f"Results saved to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
