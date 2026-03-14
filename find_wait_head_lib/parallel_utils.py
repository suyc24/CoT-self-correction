from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List

import torch

from .ablation import SingleHeadAblationHook, list_all_heads
from .io_utils import (
    build_export_row,
    init_summary_entry,
    init_wait_logit_entry,
    load_prepared_examples,
    should_keep_filtered_row,
    update_wait_logit_stats,
    update_summary,
)
from .model_utils import generate_continuation, get_next_token_logit, load_model_with_retries
from .pipeline import analyze_generation


def run_ablation_worker(
    worker_id: int,
    gpu_id: int,
    head_labels: List[str],
    prepared_jsonl_path: str,
    args_dict: Dict[str, Any],
    keywords: List[str],
    worker_output_path: str,
    worker_wait_logit_path: str,
    wait_token_id: int,
    baseline_wait_logits: Dict[str, float],
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    args = argparse.Namespace(**args_dict)
    device_map_override = {"": gpu_id}
    model, tokenizer, _, _ = load_model_with_retries(
        args,
        device_map_override=device_map_override,
        log_prefix=f"[Worker {worker_id} GPU{gpu_id}]",
    )

    all_heads, attn_modules, _ = list_all_heads(model)
    head_lookup = {h.label: h for h in all_heads}
    prepared_examples = load_prepared_examples(prepared_jsonl_path)

    stats: Dict[str, Dict[str, Any]] = defaultdict(init_summary_entry)
    wait_logit_stats: Dict[str, Dict[str, Any]] = defaultdict(init_wait_logit_entry)
    kept_rows = 0

    with open(worker_output_path, "w", encoding="utf-8") as raw_f, open(
        worker_wait_logit_path, "w", encoding="utf-8"
    ) as wait_f:
        for head_label in head_labels:
            if head_label not in head_lookup:
                raise ValueError(f"Worker {worker_id}: head {head_label} not found.")
            head = head_lookup[head_label]
            attn_module = attn_modules[head.layer_idx]
            for ex, prep_info in prepared_examples:
                with SingleHeadAblationHook(
                    attn_module=attn_module,
                    head_idx=head.head_idx,
                    num_heads=head.num_heads,
                    head_dim=head.head_dim,
                ):
                    ablated_wait_logit = get_next_token_logit(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_prefix=prep_info["tampered_prefix"],
                        target_token_id=wait_token_id,
                    )
                with SingleHeadAblationHook(
                    attn_module=attn_module,
                    head_idx=head.head_idx,
                    num_heads=head.num_heads,
                    head_dim=head.head_dim,
                ) as hook:
                    continuation, full_text, gen_tokens = generate_continuation(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_prefix=prep_info["tampered_prefix"],
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )

                row = analyze_generation(
                    example=ex,
                    prep_info=prep_info,
                    condition="single_head_ablation",
                    continuation=continuation,
                    full_text=full_text,
                    generated_tokens=gen_tokens,
                    keywords=keywords,
                    head=head,
                    hook=hook,
                )
                update_summary(stats, row)
                baseline_wait_logit = float(baseline_wait_logits[str(ex["id"])])
                update_wait_logit_stats(wait_logit_stats, head.label, baseline_wait_logit, ablated_wait_logit)
                wait_f.write(
                    json.dumps(
                        {
                            "example_id": ex["id"],
                            "head_label": head.label,
                            "layer_idx": head.layer_idx,
                            "head_idx": head.head_idx,
                            "logitbaseline_wait_token": baseline_wait_logit,
                            "logitablated_wait_token": ablated_wait_logit,
                            "delta_ablated_minus_baseline": float(ablated_wait_logit - baseline_wait_logit),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if should_keep_filtered_row(row):
                    raw_f.write(json.dumps(build_export_row(row), ensure_ascii=False) + "\n")
                    kept_rows += 1

    return {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "head_count": len(head_labels),
        "stats": dict(stats),
        "wait_logit_stats": dict(wait_logit_stats),
        "kept_rows": kept_rows,
        "worker_output_path": worker_output_path,
        "worker_wait_logit_path": worker_wait_logit_path,
    }
