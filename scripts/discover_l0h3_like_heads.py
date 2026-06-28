#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.io_utils import load_jsonl, write_json
from cot_research.l0h3_like_discovery import (
    CandidateHeadScore,
    evaluate_l0h3_like_behavior,
    load_csv_rows,
    rank_l0h3_like_candidates,
    save_discovery_bundle,
    summarize_scale_validation,
    summarize_zero_validation,
    write_candidate_report,
    write_subset_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover and validate L0H3-like heads on a small slice.")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "self_correction_ablation" / "test_questions.jsonl"),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--wait_examples", type=int, default=12)
    parser.add_argument("--attention_examples", type=int, default=9)
    parser.add_argument("--validation_examples", type=int, default=40)
    parser.add_argument("--candidate_top_k", type=int, default=5)
    parser.add_argument("--validate_top_k", type=int, default=1)
    parser.add_argument("--boost_scale", type=float, default=1.4)
    parser.add_argument("--parallel_gpu_ids", type=str, default="")
    parser.add_argument("--parallel_workers", type=int, default=0)
    parser.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait_max_stage1_tokens", type=int, default=2048)
    parser.add_argument("--attention_max_stage1_tokens", type=int, default=4096)
    parser.add_argument("--validation_max_new_tokens", type=int, default=4096)
    return parser.parse_args()


def select_prefix_rows(rows: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if count <= 0:
        return list(rows)
    return list(rows[:count])


def run_stage(
    cmd: List[str],
    *,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":")
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("$ " + " ".join(cmd) + "\n\n")
        log_f.flush()
        subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )


def build_common_model_args(args: argparse.Namespace) -> List[str]:
    return [
        "--model_name_or_path",
        args.model_name_or_path,
        "--parallel_gpu_ids",
        args.parallel_gpu_ids,
        "--parallel_workers",
        str(args.parallel_workers),
        "--seed",
        str(args.seed),
        f"--{'load_in_half' if args.load_in_half else 'no-load_in_half'}",
        f"--{'use_fast_tokenizer' if args.use_fast_tokenizer else 'no-use_fast_tokenizer'}",
        f"--{'use_safetensors' if args.use_safetensors else 'no-use_safetensors'}",
        f"--{'local_files_only' if args.local_files_only else 'no-local_files_only'}",
    ]


def prefix_keys(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in payload.items()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets_dir = output_dir / "subsets"
    stages_dir = output_dir / "stages"
    logs_dir = output_dir / "logs"
    subsets_dir.mkdir(parents=True, exist_ok=True)
    stages_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_jsonl(args.input_jsonl)
    if not all_rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    wait_rows = select_prefix_rows(all_rows, args.wait_examples)
    attention_rows = select_prefix_rows(all_rows, args.attention_examples)
    validation_rows = select_prefix_rows(all_rows, args.validation_examples)

    wait_jsonl = subsets_dir / "wait_rows.jsonl"
    attention_jsonl = subsets_dir / "attention_rows.jsonl"
    validation_jsonl = subsets_dir / "validation_rows.jsonl"
    write_subset_jsonl(wait_jsonl, wait_rows)
    write_subset_jsonl(attention_jsonl, attention_rows)
    write_subset_jsonl(validation_jsonl, validation_rows)

    common_model_args = build_common_model_args(args)

    wait_stage_dir = stages_dir / "wait_ablation"
    wait_cmd = [
        sys.executable,
        "scripts/find_reflection_heads_by_wait_ablation.py",
        "--input_jsonl",
        str(wait_jsonl),
        "--output_dir",
        str(wait_stage_dir),
        "--max_examples",
        str(len(wait_rows)),
        "--max_stage1_tokens",
        str(args.wait_max_stage1_tokens),
        *common_model_args,
    ]
    run_stage(wait_cmd, log_path=logs_dir / "wait_ablation.log")

    attention_stage_dir = stages_dir / "prev1_attention"
    attention_cmd = [
        sys.executable,
        "scripts/analyze_local_attention_heads.py",
        "--input_jsonl",
        str(attention_jsonl),
        "--output_dir",
        str(attention_stage_dir),
        "--max_examples",
        str(len(attention_rows)),
        "--max_stage1_tokens",
        str(args.attention_max_stage1_tokens),
        *common_model_args,
    ]
    run_stage(attention_cmd, log_path=logs_dir / "prev1_attention.log")

    wait_summary_rows = load_csv_rows(wait_stage_dir / "head_wait_prefix_summary.csv")
    prev_summary_rows = load_csv_rows(attention_stage_dir / "head_prev_attention_summary.csv")
    candidates = rank_l0h3_like_candidates(
        wait_rows=wait_summary_rows,
        prev_rows=prev_summary_rows,
        layer_idx_filter=0,
    )
    if not candidates:
        raise ValueError("No layer-0 candidates were produced by the discovery stages.")

    top_candidates = list(candidates[: max(args.candidate_top_k, 1)])
    validate_candidates = list(candidates[: max(args.validate_top_k, 1)])

    validation_rows_out: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(validate_candidates, start=1):
        head_label = candidate.head_label

        zero_stage_dir = stages_dir / f"validate_zero_{head_label.lower()}"
        zero_cmd = [
            sys.executable,
            "scripts/test_head_boost_effects.py",
            "--input_source",
            "jsonl",
            "--input_jsonl",
            str(validation_jsonl),
            "--output_dir",
            str(zero_stage_dir),
            "--max_examples",
            "0",
            "--head_labels",
            head_label,
            "--intervention_kind",
            "zero",
            "--scale",
            "0.0",
            "--max_new_tokens",
            str(args.validation_max_new_tokens),
            *common_model_args,
        ]
        run_stage(zero_cmd, log_path=logs_dir / f"validate_zero_{head_label.lower()}.log")

        scale_stage_dir = stages_dir / f"validate_scale_{head_label.lower()}"
        scale_cmd = [
            sys.executable,
            "scripts/run_l0h3_scale_wait_length.py",
            "--input_source",
            "jsonl",
            "--input_jsonl",
            str(validation_jsonl),
            "--output_dir",
            str(scale_stage_dir),
            "--max_examples",
            "0",
            "--head_label",
            head_label,
            "--scales",
            f"1.0,{args.boost_scale}",
            "--max_new_tokens",
            str(args.validation_max_new_tokens),
            *common_model_args,
        ]
        run_stage(scale_cmd, log_path=logs_dir / f"validate_scale_{head_label.lower()}.log")

        zero_summary = json.loads((zero_stage_dir / "summary.json").read_text(encoding="utf-8"))
        scale_summary = json.loads((scale_stage_dir / "summary.json").read_text(encoding="utf-8"))
        zero_eval = summarize_zero_validation(zero_summary)
        scale_eval = summarize_scale_validation(scale_summary, boosted_scale=args.boost_scale)
        behavior_eval = evaluate_l0h3_like_behavior(
            zero_validation=zero_eval,
            scale_validation=scale_eval,
        )

        validation_rows_out.append(
            {
                "candidate_rank": idx,
                "head_label": head_label,
                "combined_rank_score": candidate.combined_rank_score,
                "wait_rank": candidate.wait_rank,
                "prev_rank": candidate.prev_rank,
                **prefix_keys("zero_", zero_eval),
                **prefix_keys("scale_", scale_eval),
                **behavior_eval,
            }
        )

        candidate_bundle_dir = output_dir / f"candidate_{idx}_{head_label.lower()}"
        save_discovery_bundle(
            output_dir=candidate_bundle_dir,
            candidates=top_candidates,
            zero_validation=zero_eval,
            scale_validation=scale_eval,
            behavior_eval=behavior_eval,
            metadata={
                "model_name_or_path": args.model_name_or_path,
                "head_label": head_label,
                "candidate_rank": idx,
                "boost_scale": args.boost_scale,
                "parallel_gpu_ids": args.parallel_gpu_ids,
                "wait_stage_dir": str(wait_stage_dir),
                "attention_stage_dir": str(attention_stage_dir),
                "zero_stage_dir": str(zero_stage_dir),
                "scale_stage_dir": str(scale_stage_dir),
            },
        )
        write_candidate_report(
            path=candidate_bundle_dir / "report.md",
            model_name_or_path=args.model_name_or_path,
            candidates=top_candidates,
            chosen_head=head_label,
            zero_validation=zero_eval,
            scale_validation=scale_eval,
            behavior_eval=behavior_eval,
        )

    write_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "wait_subset_size": len(wait_rows),
            "attention_subset_size": len(attention_rows),
            "validation_subset_size": len(validation_rows),
            "top_candidates": [item.to_dict() for item in top_candidates],
            "validated_candidates": validation_rows_out,
        },
    )
    from cot_research.io_utils import write_csv

    write_csv(output_dir / "validated_candidates.csv", validation_rows_out)
    write_csv(output_dir / "candidate_scores.csv", [item.to_dict() for item in top_candidates])

    if validation_rows_out:
        chosen = validation_rows_out[0]["head_label"]
    else:
        chosen = top_candidates[0].head_label

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("# L0H3-Like Discovery Pipeline\n\n")
        f.write(f"- model: `{args.model_name_or_path}`\n")
        f.write(f"- wait_subset_size: `{len(wait_rows)}`\n")
        f.write(f"- attention_subset_size: `{len(attention_rows)}`\n")
        f.write(f"- validation_subset_size: `{len(validation_rows)}`\n")
        f.write(f"- chosen_head: `{chosen}`\n\n")
        f.write("## Top Candidates\n\n")
        for item in top_candidates:
            f.write(
                "- "
                f"{item.head_label}: combined_rank_score={item.combined_rank_score:.3f}, "
                f"mean_no_wait_rate={item.mean_no_wait_rate:.6f}, "
                f"mean_wait_logit_drop={item.mean_wait_logit_drop:.6f}, "
                f"prev_1_top_nonself_rate={item.prev_1_top_nonself_rate:.6f}, "
                f"mean_prev_mass_w1={item.mean_prev_mass_w1:.6f}\n"
            )
        if validation_rows_out:
            f.write("\n## Validation\n\n")
            for row in validation_rows_out:
                f.write(
                    "- "
                f"{row['head_label']}: "
                    f"zero_repetition_delta={row['zero_repetition_delta']:.6f}, "
                    f"ablation_pass={row['ablation_repetition_pass']}, "
                    f"boost_tokens_delta={row['scale_mean_generated_tokens_delta']:.6f}, "
                    f"boost_reflection_delta={row['scale_mean_reflection_count_delta']:.6f}, "
                    f"boost_accuracy_delta={row['scale_accuracy_delta']:.6f}, "
                    f"overall_pass={row['overall_pass']}\n"
                )


if __name__ == "__main__":
    main()
