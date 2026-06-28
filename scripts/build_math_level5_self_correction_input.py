#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import answers_match
from cot_research.io_utils import dump_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MATH Level 5 JSONL with synthetic wrong_answer fields for forced-box experiments."
    )
    parser.add_argument("--input_jsonl", default=str(ROOT_DIR / "evaluation" / "data" / "math" / "test.jsonl"))
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_examples", type=int, default=500)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--level", default="Level 5")
    parser.add_argument("--split_label", default="math_level5_first500")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def increment_int_text(match: re.Match[str]) -> str:
    text = match.group(0)
    sign = ""
    body = text
    if body.startswith("-"):
        sign = "-"
        body = body[1:]
    try:
        value = int(sign + body)
    except ValueError:
        return text
    return str(value + 1)


def simple_wrong_candidates(answer: str) -> Iterable[str]:
    ans = str(answer or "").strip()
    if not ans:
        yield "1"
        return

    frac_match = re.search(r"\\(?:dfrac|tfrac|frac)\{(-?\d+)\}\{(-?\d+)\}", ans)
    if frac_match:
        num = int(frac_match.group(1))
        den = frac_match.group(2)
        yield ans[: frac_match.start()] + f"\\frac{{{num + 1}}}{{{den}}}" + ans[frac_match.end() :]

    number_match = re.search(r"-?\d+", ans)
    if number_match:
        yield ans[: number_match.start()] + increment_int_text(number_match) + ans[number_match.end() :]

    decimal_match = re.search(r"-?\d+(?:\.\d+)", ans)
    if decimal_match:
        try:
            value = float(decimal_match.group(0))
            yield ans[: decimal_match.start()] + str(value + 1.0) + ans[decimal_match.end() :]
        except ValueError:
            pass

    yield "0"
    yield "1"
    yield f"{ans}+1"
    yield f"\\text{{wrong answer: {ans}}}"


def make_wrong_answer(answer: str) -> str:
    seen = set()
    for candidate in simple_wrong_candidates(answer):
        candidate = str(candidate).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if not answers_match(candidate, answer):
            return candidate
    return "0" if not answers_match("0", answer) else "1"


def convert_row(row: Dict[str, Any], *, selected_idx: int, split_label: str) -> Dict[str, Any]:
    answer = str(row.get("answer") or row.get("correct_answer") or "").strip()
    unique_id = str(row.get("unique_id") or row.get("id") or selected_idx)
    metadata = {
        "source": "evaluation/data/math/test.jsonl",
        "unique_id": unique_id,
        "level": row.get("level"),
        "subject": row.get("subject"),
        "level5_index": selected_idx,
        "split": split_label,
        "wrong_answer_strategy": "synthetic_answer_perturbation",
    }
    return {
        "id": unique_id,
        "question": str(row.get("problem") or row.get("question") or "").strip(),
        "correct_answer": answer,
        "wrong_answer": make_wrong_answer(answer),
        "metadata": metadata,
        "level5_index": selected_idx,
        "source_idx": row.get("source_idx", selected_idx),
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.input_jsonl))
    filtered = [row for row in rows if str(row.get("level", "")).strip() == str(args.level)]
    selected = filtered[int(args.start_idx) : int(args.start_idx) + int(args.max_examples)]
    out_rows = [convert_row(row, selected_idx=int(args.start_idx) + idx, split_label=str(args.split_label)) for idx, row in enumerate(selected)]
    dump_jsonl(args.output_jsonl, out_rows)
    write_json(
        Path(args.output_jsonl).with_suffix(".summary.json"),
        {
            "input_jsonl": str(args.input_jsonl),
            "output_jsonl": str(args.output_jsonl),
            "level": str(args.level),
            "available_level_rows": len(filtered),
            "start_idx": int(args.start_idx),
            "max_examples": int(args.max_examples),
            "written": len(out_rows),
            "split_label": str(args.split_label),
        },
    )
    print(f"[Done] wrote {len(out_rows)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
