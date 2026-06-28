#!/usr/bin/env python3
"""Experiment 1: Multi-sample reasoning trace collection and analysis.

For each of 20 math problems (easy/medium/hard), generates N reasoning
traces at temperature=0.7 using Qwen3 with thinking enabled via vLLM.
Analyzes reflection patterns, answer correctness, and identifies
interesting cases (wrong→right, right→wrong, etc.).

Usage:
  python scripts/run_exp1_reasoning_trace.py \
    --model Qwen/Qwen3-4B \
    --output-dir outputs/exp1_reasoning_trace \
    --num-samples 10 \
    --temperature 0.7 \
    --max-tokens 4096
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cot_research.answer_extraction import answers_match, extract_last_boxed

# ---------------------------------------------------------------------------
# 20 curated problems: 7 easy (GSM8K-style), 7 medium (MATH L2-4), 6 hard
# ---------------------------------------------------------------------------
PROBLEMS = [
    # --- Easy (GSM8K-style) ---
    {"id": "easy_01", "difficulty": "easy",
     "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "answer": "72"},
    {"id": "easy_02", "difficulty": "easy",
     "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "answer": "10"},
    {"id": "easy_03", "difficulty": "easy",
     "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make?",
     "answer": "5"},
    {"id": "easy_04", "difficulty": "easy",
     "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "answer": "3"},
    {"id": "easy_05", "difficulty": "easy",
     "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
     "answer": "624"},
    {"id": "easy_06", "difficulty": "easy",
     "question": "Every day, Wendi feeds each of her 6 chickens 3 cups of mixed bird feed. She gives the chickens their feed in 2 separate meals. In the morning, she gives them 15 cups and in the afternoon she gives them another amount. How many cups does she need to give in the afternoon?",
     "answer": "3"},
    {"id": "easy_07", "difficulty": "easy",
     "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "answer": "6"},
    # --- Medium (MATH Level 2-4) ---
    {"id": "med_01", "difficulty": "medium",
     "question": "Find the sum of all values of $x$ such that $|2x - 6| = 10$.",
     "answer": "6"},
    {"id": "med_02", "difficulty": "medium",
     "question": "How many positive divisors does 360 have?",
     "answer": "24"},
    {"id": "med_03", "difficulty": "medium",
     "question": "What is the remainder when $3^{100}$ is divided by 5?",
     "answer": "1"},
    {"id": "med_04", "difficulty": "medium",
     "question": "If $x + y = 10$ and $xy = 21$, find $x^2 + y^2$.",
     "answer": "58"},
    {"id": "med_05", "difficulty": "medium",
     "question": "Compute $\\dbinom{10}{3}$.",
     "answer": "120"},
    {"id": "med_06", "difficulty": "medium",
     "question": "What is the greatest common divisor of $5!$ and $\\frac{8!}{3!}$?",
     "answer": "120"},
    {"id": "med_07", "difficulty": "medium",
     "question": "Find the arithmetic mean of the prime numbers in the list: 21, 23, 25, 27, 29.",
     "answer": "26"},
    # --- Hard (MATH Level 5 / AIME-style) ---
    {"id": "hard_01", "difficulty": "hard",
     "question": "In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Find the area of triangle $ABC$.",
     "answer": "84"},
    {"id": "hard_02", "difficulty": "hard",
     "question": "How many 4-element subsets of $\\{1, 2, 3, 4, 5, 6, 7, 8, 9\\}$ have the property that no two elements are consecutive?",
     "answer": "15"},
    {"id": "hard_03", "difficulty": "hard",
     "question": "Let $S = \\sum_{k=1}^{100} \\lfloor \\sqrt{k} \\rfloor$. Compute $S$.",
     "answer": "615"},
    {"id": "hard_04", "difficulty": "hard",
     "question": "Find the number of positive integers $n \\le 1000$ such that $15n$ is a perfect square.",
     "answer": "8"},
    {"id": "hard_05", "difficulty": "hard",
     "question": "Let $f(x) = x^3 - 3x + 1$. Find the number of real roots of $f(x) = 0$.",
     "answer": "3"},
    {"id": "hard_06", "difficulty": "hard",
     "question": "Determine the number of ordered pairs of positive integers $(a, b)$ with $a + b \\le 100$ such that $\\frac{a}{b}$ is a positive integer.",
     "answer": "187"},
]

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# Reflection signal patterns
REFLECTION_PATTERNS = [
    ("wait", re.compile(r"(?i)(?<![a-z])wait(?![a-z])")),
    ("hold_on", re.compile(r"(?i)hold on|one moment|hang on")),
    ("actually", re.compile(r"(?i)(?<![a-z])actually(?![a-z])")),
    ("however", re.compile(r"(?i)(?<![a-z])however(?![a-z])")),
    ("reconsider", re.compile(r"(?i)reconsider|on second thought|let me reconsider")),
    ("check", re.compile(r"(?i)let me check|check again|recheck|verify this|double-check")),
    ("mistake", re.compile(r"(?i)mistake|incorrect|that's wrong|that is wrong|not correct|error")),
    ("hmm", re.compile(r"(?i)(?<![a-z])hmm+(?![a-z])")),
    ("no_wait", re.compile(r"(?i)no,?\s*wait")),
    ("let_me_re", re.compile(r"(?i)let me re(?:think|do|calculate|compute|consider|evaluate|start)")),
    ("cn_reflect", re.compile(r"不对|等等|等一下|重新|有误|矛盾")),
]

STRATEGY_PATTERNS = [
    ("geometry", re.compile(r"(?i)triangle|circle|angle|radius|area|perimeter|coordinate|slope|Heron")),
    ("algebra", re.compile(r"(?i)equation|solve for|substitute|expand|factor|quadratic|variable")),
    ("number_theory", re.compile(r"(?i)modulo|modular|divisible|prime|gcd|lcm|remainder|congruent")),
    ("combinatorics", re.compile(r"(?i)choose|permutation|combination|count|subset|arrange")),
    ("arithmetic", re.compile(r"(?i)multiply|divide|add|subtract|total|sum|product")),
    ("calculus", re.compile(r"(?i)derivative|integral|limit|continuous|differentiat")),
]


def extract_think_text(text: str) -> Tuple[str, str]:
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


def find_reflection_hits(text: str) -> List[Dict[str, Any]]:
    hits = []
    for kind, pat in REFLECTION_PATTERNS:
        for m in pat.finditer(text):
            hits.append({"kind": kind, "start": m.start(), "end": m.end(),
                         "text": m.group(0)})
    return sorted(hits, key=lambda x: (x["start"], x["end"]))


def find_boxed_answers_in_think(think_text: str) -> List[Dict[str, Any]]:
    out = []
    for m in re.finditer(r"\\boxed\{", think_text):
        depth = 1
        p = m.end()
        while p < len(think_text) and depth > 0:
            if think_text[p] == "{":
                depth += 1
            elif think_text[p] == "}":
                depth -= 1
            p += 1
        if depth == 0:
            out.append({"start": m.start(), "end": p,
                         "value": think_text[m.end():p - 1]})
    return out


def detect_strategy(text: str) -> str:
    counts = Counter()
    for name, pat in STRATEGY_PATTERNS:
        counts[name] = len(pat.findall(text))
    if not counts or counts.most_common(1)[0][1] == 0:
        return "unknown"
    return counts.most_common(1)[0][0]


def analyze_single_trace(text: str, correct_answer: str) -> Dict[str, Any]:
    think_text, final_text = extract_think_text(text)
    reflections = find_reflection_hits(think_text)
    boxed_in_think = find_boxed_answers_in_think(think_text)
    final_boxed = extract_last_boxed(text)
    final_correct = bool(final_boxed and answers_match(final_boxed, correct_answer))

    first_ref_pos = reflections[0]["start"] if reflections else None

    # answers before and after first reflection
    pre_ref_answer = None
    post_ref_answer = None
    if first_ref_pos is not None and boxed_in_think:
        pre = [b for b in boxed_in_think if b["end"] <= first_ref_pos]
        post = [b for b in boxed_in_think if b["start"] >= first_ref_pos]
        if pre:
            pre_ref_answer = pre[-1]["value"]
        if post:
            post_ref_answer = post[-1]["value"]

    pre_correct = bool(pre_ref_answer and answers_match(pre_ref_answer, correct_answer)) if pre_ref_answer else None
    post_correct = bool(post_ref_answer and answers_match(post_ref_answer, correct_answer)) if post_ref_answer else None

    # classify reflection transition
    if not reflections:
        transition = "no_reflection"
    elif pre_ref_answer is None:
        transition = "reflection_no_prior_answer"
    elif not pre_correct and final_correct:
        transition = "wrong_to_right"
    elif pre_correct and not final_correct:
        transition = "right_to_wrong"
    elif pre_ref_answer and post_ref_answer and not answers_match(pre_ref_answer, post_ref_answer):
        transition = "answer_changed_other"
    else:
        transition = "no_answer_change"

    # detect multiple reflection episodes
    episode_boundaries = []
    if reflections:
        current_start = reflections[0]["start"]
        for i in range(1, len(reflections)):
            gap = reflections[i]["start"] - reflections[i - 1]["end"]
            if gap > 200:
                episode_boundaries.append(current_start)
                current_start = reflections[i]["start"]
        episode_boundaries.append(current_start)

    # check if final answer loops back to first answer
    all_boxed = find_boxed_answers_in_think(think_text)
    loops_back = False
    if len(all_boxed) >= 3 and final_boxed:
        first_ans = all_boxed[0]["value"]
        if answers_match(final_boxed, first_ans) and not answers_match(
            all_boxed[len(all_boxed) // 2]["value"], first_ans
        ):
            loops_back = True

    return {
        "think_length": len(think_text),
        "final_length": len(final_text),
        "total_length": len(text),
        "has_reflection": bool(reflections),
        "reflection_count": len(reflections),
        "reflection_kinds": sorted({h["kind"] for h in reflections}),
        "reflection_positions": [h["start"] for h in reflections],
        "reflection_episodes": len(episode_boundaries),
        "first_reflection_pos": first_ref_pos,
        "first_reflection_kind": reflections[0]["kind"] if reflections else None,
        "boxed_count_in_think": len(boxed_in_think),
        "pre_reflection_answer": pre_ref_answer,
        "post_reflection_answer": post_ref_answer,
        "pre_reflection_correct": pre_correct,
        "final_boxed_answer": final_boxed,
        "final_correct": final_correct,
        "reflection_transition": transition,
        "loops_back_to_first": loops_back,
        "strategy": detect_strategy(think_text),
    }


def highlight_reflections_in_text(text: str, max_context: int = 300) -> str:
    hits = find_reflection_hits(text)
    if not hits:
        return text
    parts = []
    last = 0
    for h in hits:
        parts.append(text[last:h["start"]])
        parts.append(f">>>REFLECT[{h['kind']}]: {text[h['start']:h['end']]}<<<")
        last = h["end"]
    parts.append(text[last:])
    return "".join(parts)


def build_case_report(
    problem: Dict[str, Any],
    samples: List[Dict[str, Any]],
    traces: List[str],
) -> Dict[str, Any]:
    by_transition = defaultdict(list)
    for i, s in enumerate(samples):
        by_transition[s["reflection_transition"]].append(i)
    return {
        "problem_id": problem["id"],
        "difficulty": problem["difficulty"],
        "question": problem["question"],
        "correct_answer": problem["answer"],
        "num_samples": len(samples),
        "num_correct": sum(1 for s in samples if s["final_correct"]),
        "num_with_reflection": sum(1 for s in samples if s["has_reflection"]),
        "transition_counts": {k: len(v) for k, v in by_transition.items()},
        "mean_think_length": sum(s["think_length"] for s in samples) / max(len(samples), 1),
        "mean_reflection_count": sum(s["reflection_count"] for s in samples) / max(len(samples), 1),
    }


def generate_traces_vllm(
    model_name: str,
    problems: List[Dict[str, Any]],
    num_samples: int,
    temperature: float,
    max_tokens: int,
    gpu_ids: str,
) -> Dict[str, List[Tuple[str, List[int]]]]:
    """Generate reasoning traces using vLLM. Returns {problem_id: [(text, token_ids), ...]}."""
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_ids)

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left",
    )
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_num_seqs=256,
        enforce_eager=False,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        n=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        skip_special_tokens=False,
    )

    # build prompts
    prompts = []
    prompt_to_pid = []
    for prob in problems:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prob["question"]},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(prompt)
        prompt_to_pid.append(prob["id"])

    print(f"Generating {len(prompts)} prompts x {num_samples} samples = {len(prompts) * num_samples} total ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"Generation done in {elapsed:.1f}s")

    results: Dict[str, List[Tuple[str, List[int]]]] = defaultdict(list)
    for output, pid in zip(outputs, prompt_to_pid):
        for out in output.outputs:
            text = out.text
            token_ids = list(getattr(out, "token_ids", []) or [])
            results[pid].append((text, token_ids))

    return dict(results)


def run_analysis(
    problems: List[Dict[str, Any]],
    traces: Dict[str, List[Tuple[str, List[int]]]],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    case_reports = []
    interesting_cases = {
        "wrong_to_right": [],
        "right_to_wrong": [],
        "direct_correct_no_reflection": [],
        "loops_back": [],
        "many_reflections": [],
        "anomalous": [],
    }

    for prob in problems:
        pid = prob["id"]
        samples_for_problem = traces.get(pid, [])
        analyzed = []

        for sample_idx, (text, token_ids) in enumerate(samples_for_problem):
            analysis = analyze_single_trace(text, prob["answer"])
            row = {
                "problem_id": pid,
                "difficulty": prob["difficulty"],
                "question": prob["question"],
                "correct_answer": prob["answer"],
                "sample_idx": sample_idx,
                "generated_text": text,
                "token_count": len(token_ids),
                **analysis,
            }
            all_rows.append(row)
            analyzed.append(analysis)

            # classify interesting cases
            case_ref = {"problem_id": pid, "sample_idx": sample_idx}
            if analysis["reflection_transition"] == "wrong_to_right":
                interesting_cases["wrong_to_right"].append(case_ref)
            elif analysis["reflection_transition"] == "right_to_wrong":
                interesting_cases["right_to_wrong"].append(case_ref)
            elif not analysis["has_reflection"] and analysis["final_correct"]:
                interesting_cases["direct_correct_no_reflection"].append(case_ref)
            if analysis["loops_back_to_first"]:
                interesting_cases["loops_back"].append(case_ref)
            if analysis["reflection_count"] >= 5:
                interesting_cases["many_reflections"].append(case_ref)

        report = build_case_report(prob, analyzed, [t for t, _ in samples_for_problem])
        case_reports.append(report)

    # save raw data
    raw_path = output_dir / "raw_traces.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            r = {k: v for k, v in row.items() if k != "generated_text"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_rows)} analyzed rows to {raw_path}")

    # save full traces separately (large)
    full_path = output_dir / "full_traces.jsonl"
    with open(full_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps({
                "problem_id": row["problem_id"],
                "sample_idx": row["sample_idx"],
                "generated_text": row["generated_text"],
            }, ensure_ascii=False) + "\n")

    # save summary
    summary = {
        "total_samples": len(all_rows),
        "total_problems": len(problems),
        "overall_accuracy": sum(1 for r in all_rows if r["final_correct"]) / max(len(all_rows), 1),
        "reflection_rate": sum(1 for r in all_rows if r["has_reflection"]) / max(len(all_rows), 1),
        "mean_reflection_count": sum(r["reflection_count"] for r in all_rows) / max(len(all_rows), 1),
        "transition_counts": dict(Counter(r["reflection_transition"] for r in all_rows)),
        "by_difficulty": {},
        "interesting_case_counts": {k: len(v) for k, v in interesting_cases.items()},
        "per_problem_reports": case_reports,
    }
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in all_rows if r["difficulty"] == diff]
        if subset:
            summary["by_difficulty"][diff] = {
                "count": len(subset),
                "accuracy": sum(1 for r in subset if r["final_correct"]) / len(subset),
                "reflection_rate": sum(1 for r in subset if r["has_reflection"]) / len(subset),
                "mean_reflection_count": sum(r["reflection_count"] for r in subset) / len(subset),
                "transition_counts": dict(Counter(r["reflection_transition"] for r in subset)),
            }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")

    # generate case report markdown
    report_md = generate_case_report_md(all_rows, interesting_cases, summary)
    report_path = output_dir / "case_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Saved case report to {report_path}")

    return summary, interesting_cases


def generate_case_report_md(
    all_rows: List[Dict[str, Any]],
    interesting_cases: Dict[str, List[Dict[str, Any]]],
    summary: Dict[str, Any],
) -> str:
    rows_by_key = {}
    for r in all_rows:
        rows_by_key[(r["problem_id"], r["sample_idx"])] = r

    lines = ["# Experiment 1: Reasoning Trace Analysis Report\n"]

    # overall stats
    lines.append("## Overall Statistics\n")
    lines.append(f"- Total samples: {summary['total_samples']}")
    lines.append(f"- Overall accuracy: {summary['overall_accuracy']:.1%}")
    lines.append(f"- Reflection rate: {summary['reflection_rate']:.1%}")
    lines.append(f"- Mean reflections per trace: {summary['mean_reflection_count']:.2f}")
    lines.append(f"- Transition distribution: {json.dumps(summary['transition_counts'])}\n")

    for diff in ["easy", "medium", "hard"]:
        if diff in summary.get("by_difficulty", {}):
            d = summary["by_difficulty"][diff]
            lines.append(f"### {diff.title()}")
            lines.append(f"- Accuracy: {d['accuracy']:.1%}, Reflection rate: {d['reflection_rate']:.1%}")
            lines.append(f"- Transitions: {json.dumps(d['transition_counts'])}\n")

    # interesting cases
    def show_cases(title: str, case_list: List[Dict], max_show: int = 3):
        lines.append(f"\n## {title} ({len(case_list)} found)\n")
        if not case_list:
            lines.append("*No cases found.*\n")
            return
        for case_ref in case_list[:max_show]:
            r = rows_by_key.get((case_ref["problem_id"], case_ref["sample_idx"]))
            if not r:
                continue
            lines.append(f"### Problem: {r['problem_id']} | Sample #{r['sample_idx']}")
            lines.append(f"- **Question**: {r['question'][:200]}...")
            lines.append(f"- **Correct answer**: {r['correct_answer']}")
            lines.append(f"- **Model final answer**: {r.get('final_boxed_answer', 'N/A')}")
            lines.append(f"- **Final correct**: {r['final_correct']}")
            lines.append(f"- **Reflections**: {r['reflection_count']} ({', '.join(r['reflection_kinds'])})")
            lines.append(f"- **Pre-reflection answer**: {r.get('pre_reflection_answer', 'N/A')} (correct: {r.get('pre_reflection_correct', 'N/A')})")
            lines.append(f"- **Transition**: {r['reflection_transition']}")
            lines.append(f"- **Strategy**: {r.get('strategy', 'N/A')}")
            think, _ = extract_think_text(r["generated_text"])
            highlighted = highlight_reflections_in_text(think)
            if len(highlighted) > 3000:
                highlighted = highlighted[:1500] + "\n\n[...truncated...]\n\n" + highlighted[-1500:]
            lines.append(f"\n<details><summary>Full Reasoning Trace (click to expand)</summary>\n")
            lines.append(f"```\n{highlighted}\n```\n</details>\n")

    show_cases("Wrong → Right (reflection rescued)", interesting_cases["wrong_to_right"])
    show_cases("Right → Wrong (reflection harmed)", interesting_cases["right_to_wrong"])
    show_cases("Direct Correct (no reflection)", interesting_cases["direct_correct_no_reflection"])
    show_cases("Loops Back to First Answer", interesting_cases["loops_back"])
    show_cases("Many Reflections (≥5)", interesting_cases["many_reflections"])
    show_cases("Anomalous Cases", interesting_cases["anomalous"])

    # per-problem summary table
    lines.append("\n## Per-Problem Summary\n")
    lines.append("| Problem | Difficulty | Accuracy | Reflect% | Avg Reflections | Transitions |")
    lines.append("|---------|-----------|----------|----------|-----------------|-------------|")
    for report in summary.get("per_problem_reports", []):
        acc = report["num_correct"] / max(report["num_samples"], 1)
        ref = report["num_with_reflection"] / max(report["num_samples"], 1)
        lines.append(
            f"| {report['problem_id']} | {report['difficulty']} | {acc:.0%} | {ref:.0%} | "
            f"{report['mean_reflection_count']:.1f} | {json.dumps(report['transition_counts'])} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Exp1: Multi-sample reasoning trace analysis")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs/exp1_reasoning_trace")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--problems-file", default="", help="Override with custom JSONL problems file")
    parser.add_argument("--analysis-only", default="", help="Skip generation, analyze existing traces from this dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # load problems
    if args.problems_file:
        with open(args.problems_file, "r", encoding="utf-8") as f:
            problems = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(problems)} problems from {args.problems_file}")
    else:
        problems = PROBLEMS
        print(f"Using {len(problems)} built-in problems")

    if args.analysis_only:
        print(f"Analysis-only mode: loading traces from {args.analysis_only}")
        analysis_dir = Path(args.analysis_only)
        full_traces_path = analysis_dir / "full_traces.jsonl"
        traces: Dict[str, List[Tuple[str, List[int]]]] = defaultdict(list)
        with open(full_traces_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                traces[obj["problem_id"]].append((obj["generated_text"], []))
        summary, cases = run_analysis(problems, dict(traces), output_dir)
    else:
        # generate traces
        traces = generate_traces_vllm(
            model_name=args.model,
            problems=problems,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            gpu_ids=args.gpu_ids,
        )
        # save run config
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "model": args.model,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_problems": len(problems),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(output_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

        summary, cases = run_analysis(problems, traces, output_dir)

    # print highlights
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 60)
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%}")
    print(f"Reflection rate: {summary['reflection_rate']:.1%}")
    print(f"Transitions: {json.dumps(summary['transition_counts'], indent=2)}")
    print(f"Interesting cases: {json.dumps(summary['interesting_case_counts'], indent=2)}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
