from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import load_jsonl
from .summary_utils import (
    summarize_condition_rows,
    summarize_next_token_targets,
    summarize_token_targets,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CoT experiment rows.jsonl")
    parser.add_argument("--input", required=True, help="Path to rows.jsonl")
    parser.add_argument("--output_dir", default="", help="Optional output dir; defaults to input parent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_summary = summarize_condition_rows(rows)
    token_summary = summarize_token_targets(rows)
    next_token_summary = summarize_next_token_targets(rows)

    write_csv(output_dir / "summary.csv", condition_summary)
    write_csv(output_dir / "token_target_summary.csv", token_summary)
    write_csv(output_dir / "next_token_summary.csv", next_token_summary)
    write_json(
        output_dir / "summary.json",
        {
            "condition_summary": condition_summary,
            "token_target_summary": token_summary,
            "next_token_summary": next_token_summary,
        },
    )
    print("[Done] Summary outputs written:")
    print(f"- summary: {output_dir / 'summary.csv'}")
    print(f"- token_summary: {output_dir / 'token_target_summary.csv'}")
    print(f"- next_token_summary: {output_dir / 'next_token_summary.csv'}")
    print(f"- summary_json: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
