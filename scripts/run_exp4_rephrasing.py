#!/usr/bin/env python3
"""Experiment 4: Problem rephrasing effects on reasoning strategy.

For 10 math problems, creates 4 variants of each:
  1. Original
  2. Simplified/colloquial
  3. With distractor information
  4. Mathematically equivalent but different framing

Generates one greedy trace per variant and compares reasoning strategies,
step counts, reflection behavior, and correctness.

Usage:
  python scripts/run_exp4_rephrasing.py \
    --model Qwen/Qwen3-4B \
    --output-dir outputs/exp4_rephrasing \
    --gpu-ids 0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cot_research.answer_extraction import answers_match, extract_last_boxed

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

REFLECTION_PATTERNS = [
    ("wait", re.compile(r"(?i)(?<![a-z])wait(?![a-z])")),
    ("actually", re.compile(r"(?i)(?<![a-z])actually(?![a-z])")),
    ("however", re.compile(r"(?i)(?<![a-z])however(?![a-z])")),
    ("reconsider", re.compile(r"(?i)reconsider|let me reconsider|on second thought")),
    ("check", re.compile(r"(?i)let me check|recheck|verify|double-check")),
    ("mistake", re.compile(r"(?i)mistake|incorrect|not correct|error")),
    ("hmm", re.compile(r"(?i)(?<![a-z])hmm+(?![a-z])")),
    ("let_me_re", re.compile(r"(?i)let me re(?:think|do|calculate|compute|start)")),
]

STRATEGY_PATTERNS = [
    ("geometry", re.compile(r"(?i)triangle|circle|angle|radius|area|perimeter|Heron|coordinate|slope|distance")),
    ("algebra", re.compile(r"(?i)equation|solve for|substitute|expand|factor|quadratic|linear|polynomial")),
    ("number_theory", re.compile(r"(?i)modulo|modular|divisible|prime|gcd|lcm|remainder|congruent|Euler")),
    ("combinatorics", re.compile(r"(?i)choose|permutation|combination|count|subset|arrange|\\binom")),
    ("arithmetic", re.compile(r"(?i)multiply|divide|add|subtract|total|sum|product|compute|calculate")),
    ("casework", re.compile(r"(?i)case 1|case 2|cases|split into|consider the case")),
    ("direct", re.compile(r"(?i)directly|straightforward|simply|obvious|clear")),
]

PROBLEMS_WITH_VARIANTS = [
    {
        "id": "reph_01", "answer": "72",
        "variants": {
            "original": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "simplified": "Someone sold 48 items in month 1, then half that in month 2. What's the total?",
            "distractor": "Natalia has a small business selling hair clips. She started in March but didn't sell any. In April, she sold clips to 48 friends. In May, sales dropped and she sold half as many. Her profit margin is 30%. How many clips did she sell in April and May combined?",
            "equivalent": "Let $a = 48$ be the number of clips sold in April. If May sales are $a/2$, compute $a + a/2$.",
        },
    },
    {
        "id": "reph_02", "answer": "58",
        "variants": {
            "original": "If $x + y = 10$ and $xy = 21$, find $x^2 + y^2$.",
            "simplified": "Two numbers add up to 10 and multiply to 21. What is the sum of their squares?",
            "distractor": "In a physics experiment, two measurements x and y satisfy x + y = 10 and xy = 21. The measurement uncertainty is ±0.1. Find $x^2 + y^2$.",
            "equivalent": "Given that the sum of two numbers is 10 and their product is 21, compute $(x+y)^2 - 2xy$.",
        },
    },
    {
        "id": "reph_03", "answer": "24",
        "variants": {
            "original": "How many positive divisors does 360 have?",
            "simplified": "Count all the numbers that divide evenly into 360.",
            "distractor": "The number 360 appears frequently in geometry (degrees in a circle) and in the Babylonian number system. How many positive divisors does 360 have?",
            "equivalent": "Given that $360 = 2^3 \\cdot 3^2 \\cdot 5$, use the divisor function formula to find $d(360)$.",
        },
    },
    {
        "id": "reph_04", "answer": "1",
        "variants": {
            "original": "What is the remainder when $3^{100}$ is divided by 5?",
            "simplified": "If you raise 3 to the 100th power and divide by 5, what's the remainder?",
            "distractor": "In cryptography, modular exponentiation is fundamental. The RSA algorithm uses similar computations. What is the remainder when $3^{100}$ is divided by 5?",
            "equivalent": "Compute $3^{100} \\pmod{5}$.",
        },
    },
    {
        "id": "reph_05", "answer": "84",
        "variants": {
            "original": "In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Find the area of triangle $ABC$.",
            "simplified": "A triangle has sides 13, 14, and 15. What is its area?",
            "distractor": "In triangle ABC, AB = 13, BC = 14, and CA = 15. The triangle is inscribed in a circle. The altitude from A to BC has length h. Find the area of triangle ABC.",
            "equivalent": "Using Heron's formula, compute the area of a triangle with side lengths $a=13$, $b=14$, $c=15$.",
        },
    },
    {
        "id": "reph_06", "answer": "120",
        "variants": {
            "original": "Compute $\\dbinom{10}{3}$.",
            "simplified": "How many ways can you choose 3 items from 10 different items?",
            "distractor": "A committee of 3 people must be chosen from 10 candidates. The candidates include 6 men and 4 women, but there are no gender restrictions. How many different committees can be formed?",
            "equivalent": "Evaluate $\\frac{10!}{3! \\cdot 7!}$.",
        },
    },
    {
        "id": "reph_07", "answer": "6",
        "variants": {
            "original": "Find the sum of all values of $x$ such that $|2x - 6| = 10$.",
            "simplified": "The absolute value of (2x - 6) equals 10. Add up all possible values of x.",
            "distractor": "In signal processing, the equation |2x - 6| = 10 models the distance of a signal from its baseline. The signal frequency is 440 Hz. Find the sum of all values of x satisfying the equation.",
            "equivalent": "Solve $|2x - 6| = 10$ and compute the sum of the solutions.",
        },
    },
    {
        "id": "reph_08", "answer": "15",
        "variants": {
            "original": "How many 4-element subsets of $\\{1, 2, 3, 4, 5, 6, 7, 8, 9\\}$ have the property that no two elements are consecutive?",
            "simplified": "From numbers 1 to 9, pick 4 numbers so that no two are next to each other. How many ways?",
            "distractor": "A security system uses 4-digit codes from digits 1-9 where no two adjacent digits in the original set can appear. The system was designed in 2015 and uses SHA-256 hashing. How many valid 4-element subsets of {1,2,...,9} have no two consecutive elements?",
            "equivalent": "Count the number of ways to choose 4 elements from $\\{1,...,9\\}$ such that if $a$ and $b$ are both chosen, then $|a-b| \\ge 2$.",
        },
    },
    {
        "id": "reph_09", "answer": "615",
        "variants": {
            "original": "Let $S = \\sum_{k=1}^{100} \\lfloor \\sqrt{k} \\rfloor$. Compute $S$.",
            "simplified": "For each number from 1 to 100, round down its square root. What's the total?",
            "distractor": "In numerical analysis, floor functions appear in discretization. The approximation error of $\\lfloor \\sqrt{k} \\rfloor$ relative to $\\sqrt{k}$ is at most 1. Compute $\\sum_{k=1}^{100} \\lfloor \\sqrt{k} \\rfloor$.",
            "equivalent": "For each integer $n \\ge 1$, the equation $\\lfloor \\sqrt{k} \\rfloor = n$ holds for $k = n^2, n^2+1, \\ldots, (n+1)^2-1$. Use this to evaluate $\\sum_{k=1}^{100} \\lfloor \\sqrt{k} \\rfloor$.",
        },
    },
    {
        "id": "reph_10", "answer": "3",
        "variants": {
            "original": "Let $f(x) = x^3 - 3x + 1$. Find the number of real roots of $f(x) = 0$.",
            "simplified": "How many real solutions does x³ - 3x + 1 = 0 have?",
            "distractor": "The polynomial $f(x) = x^3 - 3x + 1$ arises in the study of regular 9-gons and is related to the minimal polynomial of $2\\cos(2\\pi/9)$. Find the number of real roots of $f(x) = 0$.",
            "equivalent": "Determine how many times the graph of $y = x^3 - 3x + 1$ crosses the x-axis.",
        },
    },
]


def extract_think(text: str) -> Tuple[str, str]:
    lo = text.lower()
    s = lo.find("<think>")
    e = lo.find("</think>")
    if s >= 0 and e > s:
        return text[s + 7:e], text[e + 8:]
    if e >= 0:
        return text[:e], text[e + 8:]
    if s >= 0:
        return text[s + 7:], ""
    return text, ""


def count_reflections(text: str) -> Tuple[int, List[str]]:
    hits = []
    for kind, pat in REFLECTION_PATTERNS:
        for m in pat.finditer(text):
            hits.append(kind)
    return len(hits), sorted(set(hits))


def detect_strategy(text: str) -> Tuple[str, Dict[str, int]]:
    counts = {}
    for name, pat in STRATEGY_PATTERNS:
        c = len(pat.findall(text))
        if c > 0:
            counts[name] = c
    if not counts:
        return "unknown", {}
    return max(counts, key=counts.get), counts


def count_steps(text: str) -> int:
    step_markers = len(re.findall(r"(?i)step \d|^\d+[\.\)]", text, re.MULTILINE))
    line_count = len([l for l in text.strip().split("\n") if l.strip()])
    return max(step_markers, line_count // 3)


def analyze_variant_trace(text: str, correct_answer: str) -> Dict[str, Any]:
    think, final = extract_think(text)
    ref_count, ref_kinds = count_reflections(think)
    strategy, strategy_counts = detect_strategy(think)
    final_boxed = extract_last_boxed(text)
    correct = bool(final_boxed and answers_match(final_boxed, correct_answer))

    return {
        "think_length": len(think),
        "total_length": len(text),
        "step_count": count_steps(think),
        "has_reflection": ref_count > 0,
        "reflection_count": ref_count,
        "reflection_kinds": ref_kinds,
        "strategy": strategy,
        "strategy_counts": strategy_counts,
        "final_answer": final_boxed,
        "correct": correct,
    }


def generate_all_variants_vllm(
    model_name: str,
    problems: List[Dict[str, Any]],
    max_tokens: int,
    gpu_ids: str,
) -> Dict[str, Dict[str, str]]:
    """Generate greedy traces for all variants. Returns {prob_id: {variant_name: text}}."""
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_ids)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_tokens,
        temperature=0.0,
        skip_special_tokens=False,
    )

    # build all prompts
    all_prompts = []
    prompt_keys = []
    for prob in problems:
        for vname, vtext in prob["variants"].items():
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": vtext},
            ]
            try:
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            all_prompts.append(prompt)
            prompt_keys.append((prob["id"], vname))

    print(f"Generating {len(all_prompts)} traces (greedy)...")
    t0 = time.time()
    outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)
    print(f"Done in {time.time() - t0:.1f}s")

    results: Dict[str, Dict[str, str]] = defaultdict(dict)
    for output, (pid, vname) in zip(outputs, prompt_keys):
        results[pid][vname] = output.outputs[0].text

    return dict(results)


def run_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traces = generate_all_variants_vllm(
        args.model, PROBLEMS_WITH_VARIANTS, args.max_tokens, args.gpu_ids,
    )

    all_results = []
    for prob in PROBLEMS_WITH_VARIANTS:
        pid = prob["id"]
        prob_traces = traces.get(pid, {})
        prob_result = {
            "problem_id": pid,
            "correct_answer": prob["answer"],
            "variants": {},
        }
        for vname, vtext in prob["variants"].items():
            trace = prob_traces.get(vname, "")
            analysis = analyze_variant_trace(trace, prob["answer"])
            prob_result["variants"][vname] = {
                "question": vtext,
                "trace": trace,
                **analysis,
            }
        all_results.append(prob_result)

    # save results
    with open(output_dir / "rephrasing_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # save config
    with open(output_dir / "run_config.json", "w") as f:
        json.dump({
            "model": args.model, "max_tokens": args.max_tokens,
            "num_problems": len(PROBLEMS_WITH_VARIANTS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    # generate report
    report = generate_rephrasing_report(all_results)
    (output_dir / "rephrasing_report.md").write_text(report, encoding="utf-8")

    print(f"\nResults saved to {output_dir}")


def generate_rephrasing_report(results: List[Dict]) -> str:
    lines = ["# Experiment 4: Problem Rephrasing Report\n"]

    strategy_changes = []
    reflection_changes = []
    correctness_changes = []

    for prob in results:
        pid = prob["problem_id"]
        variants = prob["variants"]
        lines.append(f"\n## Problem: {pid} (answer: {prob['correct_answer']})\n")

        # comparison table
        lines.append("| Variant | Correct | Steps | Reflections | Strategy | Think Length |")
        lines.append("|---------|---------|-------|-------------|----------|-------------|")

        strategies = {}
        for vname in ["original", "simplified", "distractor", "equivalent"]:
            v = variants.get(vname, {})
            strategies[vname] = v.get("strategy", "?")
            lines.append(
                f"| {vname} | {v.get('correct', '?')} | {v.get('step_count', '?')} | "
                f"{v.get('reflection_count', 0)} ({', '.join(v.get('reflection_kinds', []))}) | "
                f"{v.get('strategy', '?')} | {v.get('think_length', '?')} |"
            )

        # detect interesting phenomena
        unique_strategies = set(strategies.values()) - {"unknown"}
        if len(unique_strategies) >= 2:
            strategy_changes.append(pid)
            lines.append(f"\n**Strategy change detected**: {strategies}\n")

        orig_ref = variants.get("original", {}).get("reflection_count", 0)
        for vname in ["simplified", "distractor", "equivalent"]:
            v_ref = variants.get(vname, {}).get("reflection_count", 0)
            if v_ref >= 3 and orig_ref == 0:
                reflection_changes.append((pid, vname))

        orig_correct = variants.get("original", {}).get("correct", False)
        for vname in ["simplified", "distractor", "equivalent"]:
            v_correct = variants.get(vname, {}).get("correct", False)
            if orig_correct and not v_correct:
                correctness_changes.append((pid, vname, "correct→wrong"))
            elif not orig_correct and v_correct:
                correctness_changes.append((pid, vname, "wrong→correct"))

        # show traces
        for vname in ["original", "simplified", "distractor", "equivalent"]:
            v = variants.get(vname, {})
            trace = v.get("trace", "")
            if len(trace) > 2000:
                trace = trace[:1000] + "\n[...truncated...]\n" + trace[-1000:]
            lines.append(f"\n<details><summary>{vname} trace</summary>\n")
            lines.append(f"```\n{trace}\n```\n</details>")

    # summary
    lines.append("\n## Summary of Interesting Phenomena\n")
    lines.append(f"### Strategy changes across variants: {len(strategy_changes)} problems")
    for pid in strategy_changes:
        lines.append(f"  - {pid}")
    lines.append(f"\n### Rephrasing triggered heavy reflection: {len(reflection_changes)} cases")
    for pid, vname in reflection_changes:
        lines.append(f"  - {pid}: variant '{vname}'")
    lines.append(f"\n### Correctness changed by rephrasing: {len(correctness_changes)} cases")
    for pid, vname, change in correctness_changes:
        lines.append(f"  - {pid}: variant '{vname}' ({change})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Exp4: Problem rephrasing effects")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs/exp4_rephrasing")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-ids", default="0")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
