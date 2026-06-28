#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import socket
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT_DIR / "evaluation"
for p in (EVALUATION_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cot_research.answer_extraction import classify_outcome, extract_last_boxed
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from grader import math_equal
from parser import extract_answer as benchmark_extract_answer
from parser import strip_string as benchmark_strip_string


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
WAIT_FIRST_TOKENS = {" Wait", " wait", "Wait", "wait"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one vLLM condition for reflection-gate sweep.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--eval_start_idx", type=int, default=0)
    parser.add_argument("--eval_max_examples", type=int, default=0)
    parser.add_argument("--condition", choices=["baseline", "gate"], required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--direction_cache", default="")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--site", choices=["post_attn", "block_output"], required=True)
    parser.add_argument("--benchmark_data_name", default="math")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_tokens", type=int, default=38912)
    parser.add_argument("--max_model_len", type=int, default=40960)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--system_prompt",
        default="Please reason step by step, and put your final answer within \\boxed{}.",
    )
    parser.add_argument("--print_every", type=int, default=1)
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


def row_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    out = dict(metadata) if isinstance(metadata, dict) else {}
    for key in ["repeat_idx", "repeat_seed", "source_idx", "level45_index", "unique_id"]:
        if row.get(key) is not None and key not in out:
            out[key] = row[key]
    return out


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
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def benchmark_judge_answer(text: str, correct_answer: Optional[str], data_name: str) -> Dict[str, Any]:
    pred = benchmark_strip_string(benchmark_extract_answer(text, data_name), skip_unit=False)
    reference = None if correct_answer is None else benchmark_strip_string(str(correct_answer), skip_unit=False)
    correct = bool(reference is not None and math_equal(pred, reference))
    return {
        "benchmark_pred_answer": pred,
        "benchmark_reference_answer": reference,
        "benchmark_correct": correct,
    }


def analyze_generation(tokenizer: Any, token_ids: Sequence[int], text: str, correct_answer: Optional[str], data_name: str) -> Dict[str, Any]:
    first_id = int(token_ids[0]) if token_ids else None
    first_text = tokenizer.decode([first_id], skip_special_tokens=False) if first_id is not None else ""
    reflection_count, hits, first_pos, first_kw = count_reflection_keywords(text)
    final_box = extract_last_boxed(text)
    return {
        "first_generated_token_id": first_id,
        "first_generated_token_text": first_text,
        "first_wait": bool(first_text in WAIT_FIRST_TOKENS),
        "has_reflection": bool(hits),
        "reflection_keyword_count": int(reflection_count),
        "matched_reflection_keywords": hits,
        "first_reflection_pos": first_pos,
        "first_reflection_keyword": first_kw,
        "final_boxed_answer": final_box,
        "continuation_outcome": classify_outcome(final_box, correct_answer, None),
        **benchmark_judge_answer(text, correct_answer, data_name),
        "generated_tokens": len(token_ids),
        "continuation_text": text,
    }


def load_gate_add_vector(cache_path: Path, layer: int, site: str, alpha: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
    payload = torch.load(cache_path, map_location="cpu")
    for item in payload.get("directions", []):
        if item.get("direction_type") == "gate" and int(item.get("layer_idx")) == int(layer) and str(item.get("site")) == site:
            direction = item["direction"].detach().float().cpu()
            scale = float(item["scale"])
            vector = -direction * (float(alpha) * scale)
            return vector, {
                "scale": scale,
                "n_pairs": item.get("n_pairs"),
                "mean_diff_norm": item.get("mean_diff_norm"),
                "mean_resid_std": item.get("mean_resid_std"),
            }
    raise ValueError(f"No gate direction found in {cache_path} for layer={layer} site={site}")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key == "continuation_text":
                continue
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    outcomes = Counter(str(row.get("continuation_outcome")) for row in rows)
    correct = sum(1 for row in rows if row.get("benchmark_correct"))
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "benchmark_correct_rate": correct / len(rows) if rows else 0.0,
        "has_reflection_rate": mean([1.0 if row.get("has_reflection") else 0.0 for row in rows]),
        "mean_reflection_keyword_count": mean([row.get("reflection_keyword_count") for row in rows]),
        "mean_generated_tokens": mean([row.get("generated_tokens") for row in rows]),
        "first_wait_rate": mean([1.0 if row.get("first_wait") else 0.0 for row in rows]),
        "outcome_counts": dict(outcomes),
    }


def configure_steering_env(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    for key in [
        "VLLM_QWEN2_STEERING_ENABLED",
        "VLLM_QWEN2_STEERING_LAYER",
        "VLLM_QWEN2_STEERING_SITE",
        "VLLM_QWEN2_STEERING_VECTOR_PATH",
        "VLLM_QWEN2_STEERING_DECODE_ONLY",
    ]:
        os.environ.pop(key, None)
    if args.condition == "baseline":
        return {"add_norm": 0.0}
    if not args.direction_cache:
        raise ValueError("--direction_cache is required for gate condition.")
    vector, meta = load_gate_add_vector(Path(args.direction_cache), args.layer, args.site, args.alpha)
    vector_dir = output_dir / "steering_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_path = vector_dir / f"gate_L{args.layer}_{args.site}_alpha{args.alpha:g}.pt"
    torch.save({"vector": vector}, vector_path)
    os.environ["VLLM_QWEN2_STEERING_ENABLED"] = "1"
    os.environ["VLLM_QWEN2_STEERING_LAYER"] = str(int(args.layer))
    os.environ["VLLM_QWEN2_STEERING_SITE"] = str(args.site)
    os.environ["VLLM_QWEN2_STEERING_VECTOR_PATH"] = str(vector_path)
    os.environ["VLLM_QWEN2_STEERING_DECODE_ONLY"] = "1"
    return {"add_norm": float(vector.norm().item()), **meta}


def configure_vllm_port(args: argparse.Namespace) -> Optional[int]:
    if os.environ.get("VLLM_PORT"):
        return int(os.environ["VLLM_PORT"])

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        gpu_id = int(visible)
    except ValueError:
        gpu_id = sum(ord(ch) for ch in visible) % 100

    base = int(os.environ.get("VLLM_PORT_BASE", "42000"))
    preferred = base + gpu_id * 100 + (int(args.eval_start_idx) % 97)
    for offset in range(50):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
            except OSError:
                continue
        os.environ["VLLM_PORT"] = str(port)
        return port
    raise RuntimeError(f"Could not find a free VLLM_PORT near {preferred}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vllm_port = configure_vllm_port(args)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    steering_meta = configure_steering_env(args, output_dir)
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
                "metadata": row_metadata(row),
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
            "vllm_port": vllm_port,
            "steering_meta": steering_meta,
        },
    )
    dump_jsonl(output_dir / "skipped_rows.jsonl", skipped)

    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_log_stats=True,
        enable_chunked_prefill=False,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        skip_special_tokens=False,
    )

    rows_path = output_dir / "eval_rows.jsonl"
    csv_path = output_dir / "eval_rows.csv"
    summary_path = output_dir / "summary.json"
    if rows_path.exists():
        rows_path.unlink()
    eval_rows: List[Dict[str, Any]] = []
    condition = "baseline" if args.condition == "baseline" else "gate_off"
    direction_type = "none" if args.condition == "baseline" else "gate"
    layer_idx: Any = "" if args.condition == "baseline" else int(args.layer)
    site: Any = "" if args.condition == "baseline" else args.site
    alpha = 0.0 if args.condition == "baseline" else float(args.alpha)

    for start in tqdm(range(0, len(prepared), args.batch_size), desc=f"vLLM {condition} alpha={alpha:g}", dynamic_ncols=True):
        batch = prepared[start : start + args.batch_size]
        outputs = llm.generate([item["prompt"] for item in batch], sampling_params, use_tqdm=False)
        batch_rows: List[Dict[str, Any]] = []
        for item, request_output in zip(batch, outputs):
            completion = request_output.outputs[0]
            token_ids = [int(x) for x in list(completion.token_ids or [])]
            text = completion.text or ""
            analysis = analyze_generation(tokenizer, token_ids, text, item["correct_answer"], args.benchmark_data_name)
            batch_rows.append(
                {
                    "global_idx": item["global_idx"],
                    "example_id": item["example_id"],
                    "condition": condition,
                    "direction_type": direction_type,
                    "layer_idx": layer_idx,
                    "site": site,
                    "alpha": alpha,
                    "scale": steering_meta.get("scale", ""),
                    "add_norm": steering_meta.get("add_norm", 0.0),
                    "question": item["question"],
                    "correct_answer": item["correct_answer"],
                    "metadata": item.get("metadata") or {},
                    "repeat_idx": (item.get("metadata") or {}).get("repeat_idx", ""),
                    "source_idx": (item.get("metadata") or {}).get("source_idx", ""),
                    "level45_index": (item.get("metadata") or {}).get("level45_index", ""),
                    "prompt_tokens": item["prompt_tokens"],
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                    **analysis,
                }
            )
        append_jsonl(rows_path, batch_rows)
        eval_rows.extend(batch_rows)
        if args.print_every > 0 and (len(eval_rows) % max(args.print_every, 1) == 0):
            write_csv(csv_path, eval_rows)
            write_json(summary_path, summarize(eval_rows))

    write_csv(csv_path, eval_rows)
    write_json(summary_path, summarize(eval_rows))
    try:
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[Done] vLLM reflection-gate condition finished.")
    print(f"- output_dir: {output_dir}")
    print(json.dumps(summarize(eval_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
