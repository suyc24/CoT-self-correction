#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import classify_outcome, extract_last_boxed
from cot_research.io_utils import load_jsonl, write_json


REFLECTION_KEYWORDS = [
    "wait",
    "hold on",
    "let me check",
    "let me think",
    "actually",
    "on second thought",
    "recheck",
    "check again",
    "mistake",
    "不对",
    "等等",
    "等一下",
    "检查",
    "重新",
    "错误",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one full Qwen3 thinking trajectory per example with vLLM."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--eval_start_idx", type=int, default=0)
    parser.add_argument("--eval_max_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--system_prompt",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    parser.add_argument("--print_prompt_preview", action="store_true")
    return parser.parse_args()


def row_id(row: Dict[str, Any], fallback: int) -> str:
    for key in ["id", "example_id", "unique_id", "idx"]:
        if row.get(key) is not None:
            return str(row[key])
    return str(fallback)


def row_question(row: Dict[str, Any]) -> str:
    for key in ["question", "problem", "prompt"]:
        if row.get(key):
            return str(row[key])
    return ""


def row_answer(row: Dict[str, Any]) -> Optional[str]:
    for key in ["correct_answer", "answer", "target"]:
        if row.get(key) is not None:
            return str(row[key])
    return None


def build_prompt(tokenizer: Any, question: str, system_prompt: str, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def find_think_sections(text: str) -> Tuple[str, str]:
    lower = text.lower()
    start = lower.find("<think>")
    end = lower.find("</think>")
    if start >= 0 and end >= start:
        return text[start + len("<think>") : end], text[end + len("</think>") :]
    if end >= 0:
        return text[:end], text[end + len("</think>") :]
    return text, ""


def count_reflection_keywords(text: str) -> Tuple[int, List[str], Optional[int], Optional[str]]:
    lower = text.lower()
    hits: List[str] = []
    first_pos: Optional[int] = None
    first_kw: Optional[str] = None
    total = 0
    for kw in REFLECTION_KEYWORDS:
        kw_l = kw.lower()
        count = lower.count(kw_l)
        if count:
            total += count
            hits.append(kw)
            pos = lower.find(kw_l)
            if pos >= 0 and (first_pos is None or pos < first_pos):
                first_pos = int(pos)
                first_kw = kw
    return total, hits, first_pos, first_kw


def analyze_text(text: str, correct_answer: Optional[str], token_ids: Sequence[int]) -> Dict[str, Any]:
    think_text, final_text = find_think_sections(text)
    reflection_count, hits, first_pos, first_kw = count_reflection_keywords(think_text)
    first_nonspace = ""
    stripped = text.lstrip()
    if stripped:
        first_nonspace = stripped[:32].split()[0] if stripped.split() else stripped[:32]
    final_box = extract_last_boxed(text)
    return {
        "generated_tokens": len(token_ids),
        "generated_chars": len(text),
        "has_think_end": "</think>" in text.lower(),
        "think_chars": len(think_text),
        "final_chars": len(final_text),
        "first_nonspace_text": first_nonspace,
        "has_reflection": bool(hits),
        "reflection_keyword_count": int(reflection_count),
        "matched_reflection_keywords": hits,
        "first_reflection_pos_in_think": first_pos,
        "first_reflection_keyword": first_kw,
        "final_boxed_answer": final_box,
        "outcome": classify_outcome(final_box, correct_answer, None),
        "completion_text": text,
    }


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key == "completion_text":
                continue
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def mean(values: Sequence[Any]) -> float:
    vals: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            vals.append(x)
    return sum(vals) / len(vals) if vals else float("nan")


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    outcomes = Counter(str(row.get("outcome")) for row in rows)
    first_reflections = Counter(str(row.get("first_reflection_keyword")) for row in rows if row.get("first_reflection_keyword"))
    return {
        "count": len(rows),
        "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in rows]),
        "has_think_end_rate": mean([1.0 if row.get("has_think_end") else 0.0 for row in rows]),
        "mean_generated_tokens": mean([row.get("generated_tokens") for row in rows]),
        "mean_think_chars": mean([row.get("think_chars") for row in rows]),
        "mean_final_chars": mean([row.get("final_chars") for row in rows]),
        "outcome_counts": dict(outcomes),
        "first_reflection_keyword_counts": dict(first_reflections),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "eval_rows.jsonl"
    csv_path = output_dir / "eval_rows.csv"
    summary_path = output_dir / "summary.json"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    all_rows = load_jsonl(args.input_jsonl)
    max_examples = args.eval_max_examples if args.eval_max_examples > 0 else len(all_rows)
    selected = all_rows[args.eval_start_idx : args.eval_start_idx + max_examples]

    prepared: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for local_idx, row in enumerate(selected):
        global_idx = args.eval_start_idx + local_idx
        question = row_question(row)
        if not question:
            skipped.append({"example_id": row_id(row, global_idx), "global_idx": global_idx, "reason": "missing_question"})
            continue
        prompt = build_prompt(tokenizer, question, args.system_prompt, args.enable_thinking)
        prepared.append(
            {
                "global_idx": global_idx,
                "example_id": row_id(row, global_idx),
                "question": question,
                "correct_answer": row_answer(row),
                "prompt": prompt,
                "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
            }
        )

    write_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "num_input_rows": len(all_rows),
            "num_selected_rows": len(selected),
            "num_prepared_rows": len(prepared),
            "num_skipped_before_generation": len(skipped),
        },
    )
    append_jsonl(output_dir / "skipped_rows.jsonl", skipped)

    if args.print_prompt_preview and prepared:
        print(prepared[0]["prompt"][:2000])

    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    if rows_path.exists():
        rows_path.unlink()
    all_eval_rows: List[Dict[str, Any]] = []
    for start in tqdm(range(0, len(prepared), args.batch_size), desc="vLLM thinking trajectories", dynamic_ncols=True):
        batch = prepared[start : start + args.batch_size]
        outputs = llm.generate([item["prompt"] for item in batch], sampling_params, use_tqdm=False)
        batch_rows: List[Dict[str, Any]] = []
        for item, request_output in zip(batch, outputs):
            completion = request_output.outputs[0]
            token_ids = list(completion.token_ids or [])
            text = completion.text or ""
            analysis = analyze_text(text, item["correct_answer"], token_ids)
            batch_rows.append(
                {
                    "global_idx": item["global_idx"],
                    "example_id": item["example_id"],
                    "question": item["question"],
                    "correct_answer": item["correct_answer"],
                    "prompt_tokens": item["prompt_tokens"],
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                    **analysis,
                }
            )
        append_jsonl(rows_path, batch_rows)
        all_eval_rows.extend(batch_rows)
        write_csv(csv_path, all_eval_rows)
        write_json(summary_path, summarize(all_eval_rows))

    print("[Done] vLLM thinking trajectory collection finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- eval_rows: {len(all_eval_rows)}")
    print(json.dumps(summarize(all_eval_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
