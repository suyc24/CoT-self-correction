#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import answers_match, normalize_answer, parse_simple_number


ROOT = Path(
    "outputs/stateful_tamper_attention_20260510/"
    "reflection_gate_qwen3_4b_20260512/"
    "fulltraj_logit_orth_math45_all100_L22post_alpha1_sameprompt_20260513"
)

PREV_SUMMARY = Path(
    "outputs/stateful_tamper_attention_20260510/"
    "reflection_gate_qwen3_4b_20260512/"
    "fulltraj_logit_math45_all100_L22post_alpha1_20260512/"
    "merged/compare_all_conditions_summary.json"
)


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted((root / "shards").glob("gpu*_start*_n*/eval_rows.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: (str(r.get("example_id", "")), str(r.get("condition", ""))))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def strip_units(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", s)
    return s.replace(r"\%", "%")


def numeric_variants(text: Optional[str]) -> List[float]:
    s0 = strip_units(text)
    variants = {s0}
    no_words = re.sub(r"(?<!\\)[A-Za-z]+", " ", s0)
    variants.add(no_words)
    variants.add(s0.replace("%", ""))
    variants.add(no_words.replace("%", ""))
    values: List[float] = []
    for variant in variants:
        try:
            value = parse_simple_number(variant)
            if value is not None and math.isfinite(value):
                values.append(float(value))
        except Exception:
            pass
    for variant in variants:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", variant.replace(",", ""))
        if match:
            try:
                values.append(float(match.group(0)))
            except Exception:
                pass
    out: List[float] = []
    for value in values:
        if not any(abs(value - old) <= 1e-9 * max(1.0, abs(value), abs(old)) for old in out):
            out.append(value)
    return out


def nums_match(left: Optional[str], right: Optional[str]) -> bool:
    left_values = numeric_variants(left)
    right_values = numeric_variants(right)
    for left_value in left_values:
        for right_value in right_values:
            if abs(left_value - right_value) <= 1e-9 * max(1.0, abs(left_value), abs(right_value)):
                return True
    return False


def normalized_contains_answer(text: str, answer: Optional[str]) -> bool:
    if not text or answer is None:
        return False
    tail = str(text)[-2500:]
    answer_norm = normalize_answer(strip_units(answer))
    tail_norm = normalize_answer(strip_units(tail))
    if answer_norm and len(answer_norm) >= 2 and answer_norm in tail_norm:
        return True
    lower = tail.lower()
    cue = max(
        lower.rfind("answer"),
        lower.rfind("final"),
        lower.rfind("therefore"),
        lower.rfind("so "),
        lower.rfind("thus"),
    )
    scope = tail[cue:] if cue >= 0 else tail[-800:]
    return nums_match(scope, answer)


def semantic_correct(row: Dict[str, Any]) -> bool:
    final_boxed = row.get("final_boxed_answer")
    correct_answer = row.get("correct_answer")
    if row.get("continuation_outcome") == "corrected":
        return True
    if answers_match(final_boxed, correct_answer):
        return True
    if final_boxed and nums_match(final_boxed, correct_answer):
        return True
    return normalized_contains_answer(str(row.get("continuation_text") or ""), correct_answer)


def strict_late_reflect(text: Optional[str]) -> bool:
    lower = (text or "").lower()
    patterns = [
        "wait",
        "hold on",
        "actually",
        "mistake",
        "incorrect",
        "not right",
        "recheck",
        "double-check",
        "double check",
        "let me check",
        "check",
        "verify",
        "but",
        "however",
        "on second thought",
    ]
    return any(pattern in lower for pattern in patterns)


def check_late(text: Optional[str]) -> bool:
    lower = (text or "").lower()
    patterns = [
        "check",
        "verify",
        "ensure",
        "let's see",
        "let us",
        "recalculate",
        "again",
        "mistake",
        "wait",
        "actually",
    ]
    return any(pattern in lower for pattern in patterns)


def summarize(condition: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tokens = [int(row.get("generated_tokens") or 0) for row in rows]
    reflection_counts = [int(row.get("reflection_keyword_count") or 0) for row in rows]
    semantic_flags = [semantic_correct(row) for row in rows]
    strict = sum(1 for row in rows if row.get("continuation_outcome") == "corrected")
    length_caps = sum(1 for value in tokens if value >= 8192)
    return {
        "condition": condition,
        "n": len(rows),
        "outcomes": dict(Counter(row.get("continuation_outcome") for row in rows)),
        "strict_boxed_correct": strict,
        "strict_boxed_correct_rate": strict / len(rows),
        "heuristic_semantic_correct": sum(semantic_flags),
        "heuristic_semantic_correct_rate": sum(semantic_flags) / len(rows),
        "mean_generated_tokens": round(sum(tokens) / len(tokens), 2),
        "median_generated_tokens": statistics.median(tokens),
        "mean_reflection_keyword_count": round(sum(reflection_counts) / len(reflection_counts), 2),
        "has_reflection_rate": sum(1 for row in rows if row.get("has_reflection")) / len(rows),
        "strict_late_reflection_rate": sum(
            1 for row in rows if strict_late_reflect(row.get("continuation_text"))
        )
        / len(rows),
        "check_late_rate": sum(1 for row in rows if check_late(row.get("continuation_text"))) / len(rows),
        "length_cap": length_caps,
        "length_cap_rate": length_caps / len(rows),
    }


def main() -> None:
    rows = load_rows(ROOT)
    merged = ROOT / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    with (merged / "eval_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_rows: List[Dict[str, Any]] = []
    for row in rows:
        compact = {key: value for key, value in row.items() if key != "continuation_text"}
        compact["continuation_text_chars"] = len(row.get("continuation_text") or "")
        csv_rows.append(compact)
    write_csv(merged / "eval_rows.csv", csv_rows)

    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    summary = [
        summarize("logit_minus_alpha1.0_sameprompt", by_condition["logit_minus"]),
        summarize("gate_orth_logit_minus_alpha1.0", by_condition["gate_orth_logit_minus"]),
    ]

    paired: List[Dict[str, Any]] = []
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[str(row["example_id"])][str(row["condition"])] = row
    for example_id, pair in sorted(pairs.items()):
        if "logit_minus" not in pair or "gate_orth_logit_minus" not in pair:
            continue
        logit = pair["logit_minus"]
        orth = pair["gate_orth_logit_minus"]
        paired.append(
            {
                "example_id": example_id,
                "logit_outcome": logit.get("continuation_outcome"),
                "orth_outcome": orth.get("continuation_outcome"),
                "logit_tokens": logit.get("generated_tokens"),
                "orth_tokens": orth.get("generated_tokens"),
                "token_delta_orth_minus_logit": int(orth.get("generated_tokens") or 0)
                - int(logit.get("generated_tokens") or 0),
                "logit_refkw": logit.get("reflection_keyword_count"),
                "orth_refkw": orth.get("reflection_keyword_count"),
                "refkw_delta_orth_minus_logit": int(orth.get("reflection_keyword_count") or 0)
                - int(logit.get("reflection_keyword_count") or 0),
                "logit_semantic_correct": semantic_correct(logit),
                "orth_semantic_correct": semantic_correct(orth),
                "logit_final_boxed": logit.get("final_boxed_answer"),
                "orth_final_boxed": orth.get("final_boxed_answer"),
                "correct_answer": logit.get("correct_answer"),
            }
        )

    write_csv(merged / "paired_logit_vs_gate_orth.csv", paired)
    with (merged / "paired_logit_vs_gate_orth.jsonl").open("w", encoding="utf-8") as f:
        for row in paired:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    previous = json.loads(PREV_SUMMARY.read_text(encoding="utf-8")) if PREV_SUMMARY.exists() else []
    for row in previous:
        if row.get("condition") == "logit_minus_alpha1.0":
            row["condition"] = "logit_minus_alpha1.0_prompt_mismatch"
    combined = previous + summary
    fields = [
        "condition",
        "n",
        "strict_boxed_correct",
        "strict_boxed_correct_rate",
        "heuristic_semantic_correct",
        "heuristic_semantic_correct_rate",
        "mean_generated_tokens",
        "median_generated_tokens",
        "mean_reflection_keyword_count",
        "has_reflection_rate",
        "strict_late_reflection_rate",
        "check_late_rate",
        "length_cap",
        "length_cap_rate",
    ]
    with (merged / "compare_all_conditions_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in combined])
    with (merged / "compare_all_conditions_summary.json").open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    pair_summary = {
        "n": len(paired),
        "orth_shorter_than_logit": sum(1 for row in paired if row["token_delta_orth_minus_logit"] < 0),
        "orth_longer_than_logit": sum(1 for row in paired if row["token_delta_orth_minus_logit"] > 0),
        "mean_token_delta_orth_minus_logit": round(
            sum(row["token_delta_orth_minus_logit"] for row in paired) / len(paired), 2
        ),
        "median_token_delta_orth_minus_logit": statistics.median(
            [row["token_delta_orth_minus_logit"] for row in paired]
        ),
        "mean_refkw_delta_orth_minus_logit": round(
            sum(row["refkw_delta_orth_minus_logit"] for row in paired) / len(paired), 2
        ),
        "semantic_transitions": dict(
            Counter(
                f"{row['logit_semantic_correct']}->{row['orth_semantic_correct']}" for row in paired
            )
        ),
        "strict_outcome_transitions": dict(
            Counter(f"{row['logit_outcome']}->{row['orth_outcome']}" for row in paired)
        ),
    }
    output = {
        "root": str(ROOT),
        "merged_dir": str(merged),
        "row_count": len(rows),
        "unique_examples": len(pairs),
        "conditions": summary,
        "paired_logit_vs_gate_orth": pair_summary,
    }
    with (merged / "summary_logit_orth_alpha1_sameprompt.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
