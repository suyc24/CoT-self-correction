#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


LOOPBENCH_MODULE = _load_module(
    "loopbench_dataset_local",
    ROOT_DIR / "cot_research" / "loopbench_dataset.py",
)


def dump_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a LoopBench-inspired synthetic dataset from public task specifications."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT_DIR / "evaluation" / "data" / "loopbench_inspired"),
    )
    parser.add_argument("--per_task", type=int, default=100)
    parser.add_argument("--smoke_per_task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = LOOPBENCH_MODULE.build_loopbench_inspired_dataset(
        per_task=args.per_task,
        seed=args.seed,
    )
    summary = LOOPBENCH_MODULE.build_loopbench_inspired_summary(rows)
    smoke_rows = LOOPBENCH_MODULE.select_smoke_rows(rows, smoke_per_task=args.smoke_per_task)

    dump_jsonl(output_dir / "test.jsonl", rows)
    dump_jsonl(output_dir / "smoke_questions.jsonl", smoke_rows)
    write_json(output_dir / "summary.json", summary)

    print(
        f"Built loopbench_inspired dataset: total_examples={len(rows)} "
        f"smoke_examples={len(smoke_rows)} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
