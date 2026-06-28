#!/usr/bin/env python3
"""Experiment 2: Token-level perturbation at key reasoning positions.

For 5 medium-difficulty problems, generates a baseline greedy trace,
then injects random low-probability tokens at three types of positions:
  1. Mid-computation (during a calculation step)
  2. Reflection turning point (where model is about to reflect)
  3. Transitional text (summary/bridging phrases)

Observes whether the model recovers, explicitly corrects, or derails.

Usage:
  python scripts/run_exp2_perturbation.py \
    --model Qwen/Qwen3-4B \
    --output-dir outputs/exp2_perturbation \
    --gpu-ids 0
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cot_research.answer_extraction import answers_match, extract_last_boxed

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

PROBLEMS = [
    {"id": "perturb_01", "difficulty": "medium",
     "question": "Find the sum of all values of $x$ such that $|2x - 6| = 10$.",
     "answer": "6"},
    {"id": "perturb_02", "difficulty": "medium",
     "question": "How many positive divisors does 360 have?",
     "answer": "24"},
    {"id": "perturb_03", "difficulty": "medium",
     "question": "If $x + y = 10$ and $xy = 21$, find $x^2 + y^2$.",
     "answer": "58"},
    {"id": "perturb_04", "difficulty": "medium",
     "question": "What is the remainder when $3^{100}$ is divided by 5?",
     "answer": "1"},
    {"id": "perturb_05", "difficulty": "hard",
     "question": "In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Find the area of triangle $ABC$.",
     "answer": "84"},
]

REFLECTION_PATTERNS = [
    re.compile(r"(?i)(?<![a-z])wait(?![a-z])"),
    re.compile(r"(?i)hold on|actually|however"),
    re.compile(r"(?i)let me re(?:think|check|consider|calculate|do)"),
    re.compile(r"(?i)mistake|incorrect|not correct"),
    re.compile(r"(?i)(?<![a-z])hmm+(?![a-z])"),
]
TRANSITION_PATTERNS = [
    re.compile(r"(?i)\bso we have\b|\bso,?\s+we\b"),
    re.compile(r"(?i)\btherefore\b|\bthus\b|\bhence\b"),
    re.compile(r"(?i)\bnow\b,?\s"),
    re.compile(r"(?i)\bfinally\b"),
    re.compile(r"(?i)\bin summary\b|\bto summarize\b"),
]
COMPUTATION_PATTERNS = [
    re.compile(r"\d+\s*[+\-*/×÷=]\s*\d+"),
    re.compile(r"(?i)\\(?:frac|sqrt|cdot|times)\{"),
    re.compile(r"(?i)(?:multiply|divide|equals|compute|calculate)\b"),
]


def find_perturbation_positions(
    token_ids: List[int],
    token_texts: List[str],
    full_text: str,
) -> Dict[str, List[int]]:
    """Find candidate token positions for each perturbation type."""
    positions: Dict[str, List[int]] = {
        "mid_computation": [],
        "reflection_turn": [],
        "transition": [],
    }

    cumulative = []
    running = 0
    for t in token_texts:
        cumulative.append(running)
        running += len(t)

    prefix_len = 0
    for pat in COMPUTATION_PATTERNS:
        for m in pat.finditer(full_text):
            mid = (m.start() + m.end()) // 2
            for i, (start, text) in enumerate(zip(cumulative, token_texts)):
                if start <= mid < start + len(text) and 10 < i < len(token_ids) - 10:
                    positions["mid_computation"].append(i)
                    break

    for pat in REFLECTION_PATTERNS:
        for m in pat.finditer(full_text):
            char_pos = m.start()
            for i, (start, text) in enumerate(zip(cumulative, token_texts)):
                if start <= char_pos < start + len(text) and i > 5:
                    candidates = list(range(max(0, i - 3), i))
                    positions["reflection_turn"].extend(
                        [c for c in candidates if 5 < c < len(token_ids) - 5]
                    )
                    break

    for pat in TRANSITION_PATTERNS:
        for m in pat.finditer(full_text):
            mid = (m.start() + m.end()) // 2
            for i, (start, text) in enumerate(zip(cumulative, token_texts)):
                if start <= mid < start + len(text) and 10 < i < len(token_ids) - 10:
                    positions["transition"].append(i)
                    break

    for k in positions:
        positions[k] = sorted(set(positions[k]))

    return positions


def generate_baseline_greedy(model, tokenizer, question: str, max_tokens: int):
    """Generate a single greedy baseline trace, returning token-by-token info."""
    import torch

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    prompt_len = input_ids.shape[-1]
    new_ids = outputs.sequences[0][prompt_len:].tolist()
    new_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in new_ids]
    full_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # collect logits for perturbation sampling
    all_logits = []
    if hasattr(outputs, "scores") and outputs.scores:
        for score in outputs.scores:
            all_logits.append(score[0].cpu())

    return {
        "prompt": prompt,
        "prompt_ids": input_ids[0].tolist(),
        "token_ids": new_ids,
        "token_texts": new_texts,
        "full_text": full_text,
        "logits": all_logits,
    }


def perturb_and_continue(
    model,
    tokenizer,
    baseline: Dict[str, Any],
    perturb_pos: int,
    num_perturb_tokens: int = 4,
    k_low: int = 50,
    k_high: int = 200,
    max_tokens: int = 4096,
    seed: int = 42,
) -> Dict[str, Any]:
    """Insert random low-prob tokens at perturb_pos, then let model continue."""
    import torch

    rng = random.Random(seed)
    prompt_ids = baseline["prompt_ids"]
    baseline_ids = baseline["token_ids"]

    if perturb_pos >= len(baseline_ids) or perturb_pos < 0:
        return {"error": f"perturb_pos {perturb_pos} out of range"}

    # get logits at perturb position to sample low-prob tokens
    perturb_token_ids = []
    if perturb_pos < len(baseline.get("logits", [])):
        logits = baseline["logits"][perturb_pos]
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        low_prob_indices = sorted_indices[k_low:k_high].tolist()
        perturb_token_ids = [rng.choice(low_prob_indices) for _ in range(num_perturb_tokens)]
    else:
        vocab_size = model.config.vocab_size
        perturb_token_ids = [rng.randint(100, vocab_size - 1) for _ in range(num_perturb_tokens)]

    prefix_ids = prompt_ids + baseline_ids[:perturb_pos] + perturb_token_ids

    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)

    remaining_tokens = max_tokens - len(prefix_ids) + len(prompt_ids)
    remaining_tokens = max(remaining_tokens, 256)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=remaining_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    all_new_ids = outputs[0][len(prompt_ids):].tolist()
    full_text = tokenizer.decode(all_new_ids, skip_special_tokens=True)

    perturb_texts = [tokenizer.decode([tid], skip_special_tokens=False) for tid in perturb_token_ids]
    pre_text = tokenizer.decode(baseline_ids[:perturb_pos], skip_special_tokens=True)
    post_text = tokenizer.decode(all_new_ids[perturb_pos + num_perturb_tokens:], skip_special_tokens=True)

    # classify recovery
    final_boxed = extract_last_boxed(full_text)
    baseline_boxed = extract_last_boxed(baseline["full_text"])

    if final_boxed and baseline_boxed and answers_match(final_boxed, baseline_boxed):
        recovery = "full_recovery"
    elif final_boxed:
        recovery = "partial_recovery"
    else:
        recovery = "derailed"

    # check for explicit correction
    correction_patterns = [
        r"(?i)that doesn't make sense",
        r"(?i)ignore that|disregard",
        r"(?i)let me (?:start over|go back|reconsider)",
        r"(?i)(?:wait|actually),?\s*(?:that's|this is)\s*(?:wrong|incorrect|not)",
    ]
    explicit_correction = any(
        re.search(pat, post_text) for pat in correction_patterns
    )

    return {
        "perturb_pos": perturb_pos,
        "perturb_token_ids": perturb_token_ids,
        "perturb_texts": perturb_texts,
        "pre_perturb_text": pre_text[-200:] if len(pre_text) > 200 else pre_text,
        "full_text": full_text,
        "final_boxed": final_boxed,
        "baseline_boxed": baseline_boxed,
        "recovery": recovery,
        "explicit_correction": explicit_correction,
        "total_tokens": len(all_new_ids),
    }


def run_experiment(args):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} (HF backend for token-level control) ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    all_results = []

    for prob in PROBLEMS:
        print(f"\n{'='*60}")
        print(f"Problem: {prob['id']} - {prob['question'][:80]}...")
        print(f"{'='*60}")

        # generate baseline
        print("  Generating baseline (greedy)...")
        baseline = generate_baseline_greedy(model, tokenizer, prob["question"], args.max_tokens)
        baseline_correct = bool(
            extract_last_boxed(baseline["full_text"])
            and answers_match(extract_last_boxed(baseline["full_text"]), prob["answer"])
        )
        print(f"  Baseline: {len(baseline['token_ids'])} tokens, correct={baseline_correct}")
        print(f"  Answer: {extract_last_boxed(baseline['full_text'])}")

        # find perturbation positions
        positions = find_perturbation_positions(
            baseline["token_ids"], baseline["token_texts"], baseline["full_text"],
        )
        print(f"  Found positions: " + ", ".join(f"{k}={len(v)}" for k, v in positions.items()))

        prob_results = {
            "problem_id": prob["id"],
            "question": prob["question"],
            "correct_answer": prob["answer"],
            "baseline_text": baseline["full_text"],
            "baseline_correct": baseline_correct,
            "baseline_answer": extract_last_boxed(baseline["full_text"]),
            "baseline_tokens": len(baseline["token_ids"]),
            "perturbations": [],
        }

        for perturb_type, pos_list in positions.items():
            if not pos_list:
                print(f"  Skipping {perturb_type}: no positions found")
                continue

            selected = pos_list[len(pos_list) // 2]
            if len(pos_list) >= 3:
                selected = pos_list[len(pos_list) // 3]

            print(f"  Perturbing at {perturb_type}, pos={selected}...")
            result = perturb_and_continue(
                model, tokenizer, baseline, selected,
                num_perturb_tokens=args.num_perturb_tokens,
                max_tokens=args.max_tokens,
                seed=42 + hash(prob["id"]) % 1000,
            )
            result["perturb_type"] = perturb_type

            perturbed_correct = bool(
                result.get("final_boxed")
                and answers_match(result["final_boxed"], prob["answer"])
            )
            result["perturbed_correct"] = perturbed_correct

            print(f"    Recovery: {result['recovery']}, Explicit correction: {result['explicit_correction']}")
            print(f"    Answer: {result.get('final_boxed', 'N/A')}, Correct: {perturbed_correct}")
            prob_results["perturbations"].append(result)

        all_results.append(prob_results)

    # save results
    results_path = output_dir / "perturbation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # generate report
    report = generate_perturbation_report(all_results)
    report_path = output_dir / "perturbation_report.md"
    report_path.write_text(report, encoding="utf-8")

    config = {
        "model": args.model,
        "num_perturb_tokens": args.num_perturb_tokens,
        "max_tokens": args.max_tokens,
        "num_problems": len(PROBLEMS),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def generate_perturbation_report(results: List[Dict]) -> str:
    lines = ["# Experiment 2: Token Perturbation Report\n"]

    recovery_counts = defaultdict(int)
    correction_counts = defaultdict(int)

    for prob in results:
        lines.append(f"\n## Problem: {prob['problem_id']}")
        lines.append(f"- **Question**: {prob['question'][:150]}...")
        lines.append(f"- **Correct answer**: {prob['correct_answer']}")
        lines.append(f"- **Baseline answer**: {prob['baseline_answer']} (correct: {prob['baseline_correct']})")
        lines.append(f"- **Baseline tokens**: {prob['baseline_tokens']}\n")

        for p in prob["perturbations"]:
            ptype = p["perturb_type"]
            recovery_counts[f"{ptype}_{p['recovery']}"] += 1
            if p["explicit_correction"]:
                correction_counts[ptype] += 1

            lines.append(f"### Perturbation: {ptype} at position {p['perturb_pos']}")
            lines.append(f"- **Injected tokens**: {p['perturb_texts']}")
            lines.append(f"- **Recovery**: {p['recovery']}")
            lines.append(f"- **Explicit correction**: {p['explicit_correction']}")
            lines.append(f"- **Perturbed answer**: {p.get('final_boxed', 'N/A')} (correct: {p.get('perturbed_correct', False)})")

            # show context around perturbation
            lines.append(f"\n**Pre-perturbation context** (last 200 chars):")
            lines.append(f"```\n{p.get('pre_perturb_text', '')}\n```")

            perturbed_text = p.get("full_text", "")
            if len(perturbed_text) > 2000:
                perturbed_text = perturbed_text[:1000] + "\n[...truncated...]\n" + perturbed_text[-1000:]
            lines.append(f"\n<details><summary>Full perturbed trace</summary>\n")
            lines.append(f"```\n{perturbed_text}\n```\n</details>\n")

        lines.append(f"\n**Baseline trace** (for comparison):")
        bt = prob["baseline_text"]
        if len(bt) > 1500:
            bt = bt[:750] + "\n[...truncated...]\n" + bt[-750:]
        lines.append(f"<details><summary>Baseline trace</summary>\n\n```\n{bt}\n```\n</details>\n")

    lines.append("\n## Summary\n")
    lines.append("### Recovery by perturbation type:")
    for k, v in sorted(recovery_counts.items()):
        lines.append(f"- {k}: {v}")
    lines.append("\n### Explicit correction counts:")
    for k, v in sorted(correction_counts.items()):
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Exp2: Token perturbation at key positions")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs/exp2_perturbation")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-perturb-tokens", type=int, default=4)
    parser.add_argument("--gpu-ids", default="0")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
