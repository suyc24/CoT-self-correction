from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import importlib.util
import json
from math import isqrt
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOOPBENCH_PAPER_ID = "2601.05693"
LOOPBENCH_PAPER_URL = f"https://arxiv.org/abs/{LOOPBENCH_PAPER_ID}"

# Publicly visible fragment from Figure 10 plus a conservative completion that
# stays close to the appendix description. The full figure text is not publicly
# recoverable from the HTML export, so we mark this as a reconstruction.
PUBLIC_SYSTEM_PROMPT_FRAGMENT = (
    "You are a meticulous, conscientious, and by-the-book {role} who must present, "
    "step by step, without skipping any steps or omitting any intermediate quantities, "
    "the derivation and update of each digit/step."
)

RECONSTRUCTED_SYSTEM_PROMPT = (
    "You are a meticulous, conscientious, and by-the-book reasoner who must present, "
    "step by step, without skipping any steps or omitting any intermediate quantities, "
    "the derivation and update of each digit, state transition, recursive move, or "
    "logical elimination before giving the final answer."
)

PAPER_BASELINE_DECODING = {
    "setting_name": "conservative",
    "temperature": 0.1,
    "top_k": 5,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
}


def _load_helper_module():
    helper_path = Path(__file__).resolve().with_name("loopbench_dataset.py")
    spec = importlib.util.spec_from_file_location("loopbench_dataset_helpers_local", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load helper module from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_helper_module()
_sqrt_decimal_string = _HELPERS._sqrt_decimal_string
_long_division_decimal_string = _HELPERS._long_division_decimal_string
_nth_root_decimal_string = _HELPERS._nth_root_decimal_string
_solve_truth_teller_counts = _HELPERS._solve_truth_teller_counts
_truth_assignment_to_string = _HELPERS._truth_assignment_to_string
_prime_pool = _HELPERS._prime_pool
_format_point = _HELPERS._format_point
_shortest_path_with_slit_walls = _HELPERS._shortest_path_with_slit_walls


@dataclass(frozen=True)
class LogicalParadoxSpec:
    square: Tuple[Tuple[int, ...], ...]
    round_count_before_knowledge: int
    knower_index: int
    candidate_count: int
    target_sum: int
    prime_corner_count: int
    even_center_count: int


@dataclass(frozen=True)
class PathPlanningSpec:
    dimension: int
    side: int
    wall_positions: Tuple[int, ...]
    slit_heights: Tuple[int, ...]
    shortest_path: Tuple[Tuple[int, ...], ...]


def build_loopbench_reconstructed_dataset(
    *,
    per_task: int = 100,
    seed: int = 1234,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []

    rows.extend(_build_square_root_rows(per_task=per_task, rng=rng))
    rows.extend(_build_long_division_rows(per_task=per_task, rng=rng))
    rows.extend(_build_newton_rows(per_task=per_task, rng=rng))
    rows.extend(_build_truth_teller_rows(per_task=per_task, rng=rng))
    rows.extend(_build_logical_paradox_rows(per_task=per_task, rng=rng))
    rows.extend(_build_tower_of_hanoi_rows(per_task=per_task, rng=rng))
    rows.extend(_build_path_planning_rows(per_task=per_task, rng=rng))

    return rows


def build_loopbench_reconstructed_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    category_counts: Dict[str, int] = defaultdict(int)
    subtask_counts: Dict[str, int] = defaultdict(int)
    solver_status_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        category_counts[str(metadata.get("task_category") or "unknown")] += 1
        subtask_counts[str(metadata.get("subtask") or "unknown")] += 1
        solver_status_counts[str(metadata.get("solver_status") or "validated")] += 1

    return {
        "benchmark_name": "loopbench_reconstructed",
        "construction_note": (
            "Paper-faithful reconstruction of LoopBench based on Table 4 task formulations, "
            "Table 5 representative instances, Appendix A filtering criteria, and the public "
            f"implementation details from arXiv:{LOOPBENCH_PAPER_ID}. The original 700 GPT-5-"
            "synthesized prompts are not public; this dataset reconstructs the same task families "
            "rather than claiming byte-level recovery of the authors' hidden prompts."
        ),
        "paper_reference": {
            "arxiv_id": LOOPBENCH_PAPER_ID,
            "url": LOOPBENCH_PAPER_URL,
        },
        "public_prompt_fragment": PUBLIC_SYSTEM_PROMPT_FRAGMENT,
        "reconstructed_system_prompt": RECONSTRUCTED_SYSTEM_PROMPT,
        "paper_baseline_decoding": dict(PAPER_BASELINE_DECODING),
        "total_examples": len(rows),
        "solver_status_counts": dict(sorted(solver_status_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "subtask_counts": dict(sorted(subtask_counts.items())),
        "taxonomy": {
            "high_precision_arithmetic": [
                "square_root",
                "long_division",
                "newtons_iteration",
            ],
            "complex_recursive_reasoning": [
                "truth_teller_puzzles",
                "logical_paradox",
                "tower_of_hanoi",
                "path_planning",
            ],
        },
    }


def select_smoke_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    smoke_per_task: int = 2,
) -> List[Dict[str, Any]]:
    by_subtask: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        by_subtask[str(metadata.get("subtask") or "unknown")].append(row)

    smoke_rows: List[Dict[str, Any]] = []
    for subtask in sorted(by_subtask):
        smoke_rows.extend(by_subtask[subtask][: max(smoke_per_task, 0)])
    return smoke_rows


def _base_metadata(
    *,
    task_category: str,
    subtask: str,
    anchor_type: str,
) -> Dict[str, Any]:
    return {
        "benchmark_name": "loopbench_reconstructed",
        "paper_reference": {
            "arxiv_id": LOOPBENCH_PAPER_ID,
            "url": LOOPBENCH_PAPER_URL,
        },
        "task_category": task_category,
        "subtask": subtask,
        "paper_alignment": {
            "anchor_type": anchor_type,
            "task_formulation_source": "Table 4",
            "representative_instance_source": "Table 5",
            "system_prompt_source": "Figure 10 (public fragment only; reconstructed completion)",
        },
        "recommended_system_prompt": RECONSTRUCTED_SYSTEM_PROMPT,
        "recommended_decoding": dict(PAPER_BASELINE_DECODING),
    }


def _build_record(
    *,
    example_id: str,
    question: str,
    metadata: Dict[str, Any],
    correct_answer: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": example_id,
        "question": question,
        "correct_answer": correct_answer,
        "wrong_answer": None,
        "metadata": metadata,
    }


def _build_square_root_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int]] = set()

    anchor_n = 99_980_001
    anchor_digits = 350
    rows.append(
        _make_square_root_row(
            row_idx=1,
            n=anchor_n,
            digits=anchor_digits,
            anchor_type="table5_anchor",
        )
    )
    seen.add((anchor_n, anchor_digits))

    precision_pool = [300, 350, 400, 450, 500]
    while len(rows) < per_task:
        digits = rng.choice(precision_pool)
        n = rng.randint(10_000_000, 999_999_999)
        if _is_perfect_square(n):
            continue
        key = (n, digits)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            _make_square_root_row(
                row_idx=len(rows) + 1,
                n=n,
                digits=digits,
                anchor_type="table4_family_reconstruction",
            )
        )
    return rows


def _make_square_root_row(*, row_idx: int, n: int, digits: int, anchor_type: str) -> Dict[str, Any]:
    question = (
        f"Compute sqrt({n}) to exactly {digits} digits after the decimal point via the standard "
        "digit-by-digit square-root extraction algorithm. After the integer part, continue the same "
        "digit-pair extraction process and explicitly record, at every step, the current partial root, "
        "the trial digit, the subtraction performed, the updated remainder, and the next digit pair "
        "brought down. Do not skip intermediate states, compress repeated phases, or switch to a closed-form approximation. "
        f"Finally report sqrt({n}) with exactly {digits} digits after the decimal point."
    )
    metadata = _base_metadata(
        task_category="high_precision_arithmetic",
        subtask="square_root",
        anchor_type=anchor_type,
    )
    metadata.update(
        {
            "radicand": n,
            "precision_digits": digits,
            "answer_decimal": _sqrt_decimal_string(n=n, digits=digits),
            "construction_constraints": {
                "precision_cap_digits": 500,
                "intended_reasoning_depth": "100+ explicit digit updates",
            },
        }
    )
    return _build_record(
        example_id=f"loopbench_reconstructed_square_root_{row_idx:04d}",
        question=question,
        metadata=metadata,
    )


def _build_long_division_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int]] = set()

    rows.append(
        _make_long_division_row(
            row_idx=1,
            numerator=1,
            denominator=13_631,
            digits=350,
            anchor_type="table5_anchor",
        )
    )
    seen.add((1, 13_631, 350))

    denominator_pool = _prime_pool(lower=10_000, upper=50_000)
    precision_pool = [300, 350, 400, 450, 500]
    while len(rows) < per_task:
        denominator = rng.choice(denominator_pool)
        numerator = rng.randint(1, denominator - 1)
        digits = rng.choice(precision_pool)
        key = (numerator, denominator, digits)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            _make_long_division_row(
                row_idx=len(rows) + 1,
                numerator=numerator,
                denominator=denominator,
                digits=digits,
                anchor_type="table4_family_reconstruction",
            )
        )
    return rows


def _make_long_division_row(
    *,
    row_idx: int,
    numerator: int,
    denominator: int,
    digits: int,
    anchor_type: str,
) -> Dict[str, Any]:
    question = (
        f"Use standard long division to compute {numerator} / {denominator} to exactly {digits} digits "
        "after the decimal point. At every step, explicitly record the current remainder, the next quotient "
        "digit chosen, the subtraction, and the updated remainder before moving on. Do not collapse recurring-looking "
        "phases into shorthand. Finally report the full decimal expansion with all requested digits."
    )
    metadata = _base_metadata(
        task_category="high_precision_arithmetic",
        subtask="long_division",
        anchor_type=anchor_type,
    )
    metadata.update(
        {
            "numerator": numerator,
            "denominator": denominator,
            "precision_digits": digits,
            "answer_decimal": _long_division_decimal_string(
                numerator=numerator,
                denominator=denominator,
                digits=digits,
            ),
            "construction_constraints": {
                "precision_cap_digits": 500,
                "intended_reasoning_depth": "100+ explicit remainder updates",
            },
        }
    )
    return _build_record(
        example_id=f"loopbench_reconstructed_long_division_{row_idx:04d}",
        question=question,
        metadata=metadata,
    )


def _build_newton_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int, int]] = set()

    rows.append(
        _make_newton_row(
            row_idx=1,
            n=590_255_551,
            degree=5,
            digits=500,
            initial_guess=50,
            anchor_type="table5_anchor",
        )
    )
    seen.add((590_255_551, 5, 500, 50))

    precision_pool = [300, 350, 400, 450, 500]
    degree_pool = [3, 4, 5, 6]
    while len(rows) < per_task:
        degree = rng.choice(degree_pool)
        n = rng.randint(100_000, 999_999_999)
        if _is_perfect_power(n, degree):
            continue
        approx = max(2, int(round(n ** (1.0 / degree))))
        initial_guess = max(2, approx + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))
        digits = rng.choice(precision_pool)
        key = (n, degree, digits, initial_guess)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            _make_newton_row(
                row_idx=len(rows) + 1,
                n=n,
                degree=degree,
                digits=digits,
                initial_guess=initial_guess,
                anchor_type="table4_family_reconstruction",
            )
        )
    return rows


def _make_newton_row(
    *,
    row_idx: int,
    n: int,
    degree: int,
    digits: int,
    initial_guess: int,
    anchor_type: str,
) -> Dict[str, Any]:
    question = (
        f"Approximate the real {degree}-th root of {n} via Newton's method, starting from x_0 = {initial_guess}. "
        f"Use the update x_(t+1) = (({degree}-1) * x_t + {n} / x_t^({degree}-1)) / {degree}, and derive every "
        "transition x_t -> x_(t+1) step by step. Carry enough precision throughout so that the final answer is accurate "
        f"to {digits} digits after the decimal point, and continue until the update is stable to that precision. "
        f"Finally report the {degree}-th root of {n} with exactly {digits} digits after the decimal point."
    )
    metadata = _base_metadata(
        task_category="high_precision_arithmetic",
        subtask="newtons_iteration",
        anchor_type=anchor_type,
    )
    metadata.update(
        {
            "target_value": n,
            "degree": degree,
            "precision_digits": digits,
            "initial_guess": initial_guess,
            "answer_decimal": _nth_root_decimal_string(
                n=n,
                degree=degree,
                digits=digits,
                initial_guess=initial_guess,
            ),
            "construction_constraints": {
                "precision_cap_digits": 500,
                "intended_reasoning_depth": "100+ explicit arithmetic updates",
            },
        }
    )
    return _build_record(
        example_id=f"loopbench_reconstructed_newtons_iteration_{row_idx:04d}",
        question=question,
        metadata=metadata,
    )


def _build_truth_teller_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, Tuple[int, ...]]] = set()

    anchor_counts = tuple([2] * 30)
    rows.append(
        _make_truth_teller_row(
            row_idx=1,
            counts=anchor_counts,
            anchor_type="table5_anchor",
            solutions=[],
        )
    )
    seen.add((30, anchor_counts))

    while len(rows) < per_task:
        n = rng.randint(24, 36)
        counts = tuple(rng.randint(0, 3) for _ in range(n))
        key = (n, counts)
        if key in seen:
            continue
        seen.add(key)
        solutions = _solve_truth_teller_counts(counts)
        if len(solutions) != 1:
            continue
        rows.append(
            _make_truth_teller_row(
                row_idx=len(rows) + 1,
                counts=counts,
                anchor_type="table4_family_reconstruction",
                solutions=solutions,
            )
        )
    return rows


def _make_truth_teller_row(
    *,
    row_idx: int,
    counts: Sequence[int],
    anchor_type: str,
    solutions: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    n = len(counts)
    lines = [
        f"There are {n} people standing in a circle, labeled P_1 through P_{n}.",
        "Each person is either a truth-teller (always truthful) or a liar (always false).",
        "Indices wrap around the circle modulo the population size.",
        "Their public statements are:",
    ]
    for idx, count in enumerate(counts, start=1):
        lines.append(
            f"- P_{idx} says: \"Exactly {count} of P_{((idx) % n) + 1}, P_{((idx + 1) % n) + 1}, "
            f"and P_{((idx + 2) % n) + 1} are liars.\""
        )
    lines.append(
        "Determine the unique truth-value assignment consistent with the entire circle, and justify the state of each person "
        "using the local neighbor constraints rather than a shortcut or unsupported guess."
    )
    metadata = _base_metadata(
        task_category="complex_recursive_reasoning",
        subtask="truth_teller_puzzles",
        anchor_type=anchor_type,
    )
    metadata.update(
        {
            "agent_count": n,
            "window_size": 3,
            "statement_counts": list(counts),
            "solution_count": len(solutions),
            "solutions": [_truth_assignment_to_string(solution) for solution in solutions],
            "construction_constraints": {
                "intended_reasoning_depth": "100+ local consistency checks",
            },
        }
    )
    return _build_record(
        example_id=f"loopbench_reconstructed_truth_teller_{row_idx:04d}",
        question="\n".join(lines),
        metadata=metadata,
    )


def _build_logical_paradox_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(_make_logical_anchor_row())

    generated = _generate_logical_paradox_specs()
    if generated:
        rng.shuffle(generated)
        for spec in generated:
            if len(rows) >= per_task:
                break
            rows.append(
                _make_logical_generated_row(
                    row_idx=len(rows) + 1,
                    spec=spec,
                )
            )
    if len(rows) < per_task:
        fallback_specs = _generate_magic_square_prompt_specs_fallback()
        rng.shuffle(fallback_specs)
        for item in fallback_specs:
            if len(rows) >= per_task:
                break
            rows.append(
                _make_logical_generated_row_fallback(
                    row_idx=len(rows) + 1,
                    square=item["square"],
                    round_count=int(item["round_count"]),
                    knower_label=str(item["knower_label"]),
                )
            )
    if len(rows) < per_task:
        raise ValueError(
            f"Not enough logical-paradox rows built: requested={per_task}, built={len(rows)}."
        )
    return rows


def _make_logical_anchor_row() -> Dict[str, Any]:
    question = (
        "There are 16 logicians arranged in a 4x4 grid and labeled L1 through L16 in row-major order. "
        "Each logician holds one hidden integer, all hidden integers are distinct, and every value lies between 1 and 20 inclusive. "
        "Each logician can see the values held by the other logicians in the same row and the same column, but not their own value. "
        "It is publicly known that every row, every column, and both main diagonals sum to 50. "
        "It is also publicly known that all four corner values are prime and that at least two of the center four values are even. "
        "At the start of the public reasoning game, all logicians know the full rule set and know that everyone reasons perfectly. "
        "For the first 35 rounds, every logician publicly says, \"I don't know my number.\" In round 36, L1 says, \"I know my number now.\" "
        "Reconstruct the public-elimination process that makes this possible and determine the full hidden grid."
    )
    metadata = _base_metadata(
        task_category="complex_recursive_reasoning",
        subtask="logical_paradox",
        anchor_type="table5_anchor",
    )
    metadata.update(
        {
            "grid_size": 4,
            "value_range": [1, 20],
            "round_count_before_knowledge": 35,
            "knower_label": "L1",
            "construction_constraints": {
                "intended_reasoning_depth": "multi-round epistemic elimination",
            },
            "solver_status": "prompt_family_reconstruction_only",
        }
    )
    return _build_record(
        example_id="loopbench_reconstructed_logical_paradox_0001",
        question=question,
        metadata=metadata,
    )


def _make_logical_generated_row(
    *,
    row_idx: int,
    spec: LogicalParadoxSpec,
) -> Dict[str, Any]:
    square = spec.square
    flat = [value for row in square for value in row]
    size = len(square)
    knower_label = f"L{spec.knower_index + 1}"
    question = (
        f"There are {size * size} logicians arranged in a {size}x{size} grid and labeled L1 through L{size * size} in row-major order. "
        f"Each logician holds one hidden integer, all hidden integers are distinct, and every value lies between {min(flat)} and {max(flat)} inclusive. "
        "Each logician can see the values held by the other logicians in the same row and the same column, but not their own value. "
        f"It is publicly known that every row, every column, and both main diagonals sum to {spec.target_sum}. "
        f"It is also publicly known that exactly {spec.prime_corner_count} of the four corner values are prime and that exactly {spec.even_center_count} "
        "of the center four values are even. Everyone knows the full rule set and knows that the others are perfectly rational. "
        f"Among all candidate grids consistent with those public facts, for the first {spec.round_count_before_knowledge} rounds every logician publicly says, "
        "\"I don't know my number.\" In round "
        f"{spec.round_count_before_knowledge + 1}, {knower_label} says, \"I know my number now.\" "
        "Reconstruct the public-elimination process and determine the hidden grid."
    )
    metadata = _base_metadata(
        task_category="complex_recursive_reasoning",
        subtask="logical_paradox",
        anchor_type="table4_family_reconstruction",
    )
    metadata.update(
        {
            "grid_size": size,
            "value_range": [min(flat), max(flat)],
            "target_sum": spec.target_sum,
            "round_count_before_knowledge": spec.round_count_before_knowledge,
            "knower_label": knower_label,
            "hidden_grid": [list(row) for row in square],
            "candidate_count": spec.candidate_count,
            "construction_constraints": {
                "intended_reasoning_depth": "multi-round epistemic elimination",
            },
            "solver_status": "epistemic_family_validated",
        }
    )
    return _build_record(
        example_id=f"loopbench_reconstructed_logical_paradox_{row_idx:04d}",
        question=question,
        metadata=metadata,
    )


def _make_logical_generated_row_fallback(
    *,
    row_idx: int,
    square: Sequence[Sequence[int]],
    round_count: int,
    knower_label: str,
) -> Dict[str, Any]:
    flat = [value for row in square for value in row]
    size = len(square)
    corners = [square[0][0], square[0][-1], square[-1][0], square[-1][-1]]
    center_values = [square[1][1], square[1][2], square[2][1], square[2][2]]
    prime_corner_count = sum(1 for value in corners if _is_prime(value))
    even_center_count = sum(1 for value in center_values if value % 2 == 0)
    target_sum = sum(square[0])
    question = (
        f"There are {size * size} logicians arranged in a {size}x{size} grid and labeled L1 through L{size * size} in row-major order. "
        f"Each logician holds one hidden integer, all hidden integers are distinct, and every value lies between {min(flat)} and {max(flat)} inclusive. "
        "Each logician can see the values held by the other logicians in the same row and the same column, but not their own value. "
        f"It is publicly known that every row, every column, and both main diagonals sum to {target_sum}. "
        f"It is also publicly known that exactly {prime_corner_count} of the four corner values are prime and that exactly {even_center_count} "
        "of the center four values are even. Everyone knows the full rule set and knows that the others are perfectly rational. "
        f"For the first {round_count} rounds, every logician publicly says, \"I don't know my number.\" "
        f"In round {round_count + 1}, {knower_label} says, \"I know my number now.\" "
        "Reconstruct the public-elimination process and determine the hidden grid."
    )
    metadata = _base_metadata(
        task_category="complex_recursive_reasoning",
        subtask="logical_paradox",
        anchor_type="table4_family_reconstruction",
    )
    metadata.update(
        {
            "grid_size": size,
            "value_range": [min(flat), max(flat)],
            "target_sum": target_sum,
            "round_count_before_knowledge": round_count,
            "knower_label": knower_label,
            "hidden_grid": [list(row) for row in square],
            "construction_constraints": {
                "intended_reasoning_depth": "multi-round epistemic elimination",
            },
            "solver_status": "prompt_family_reconstruction_only",
        }
    )
    return _build_record(
        example_id=f"loopbench_reconstructed_logical_paradox_{row_idx:04d}",
        question=question,
        metadata=metadata,
    )


def _build_tower_of_hanoi_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    disk_counts = [20] + [count for count in range(18, 18 + per_task + 8) if count != 20]
    for row_idx, disk_count in enumerate(disk_counts[:per_task], start=1):
        anchor_type = "table5_anchor" if row_idx == 1 else "table4_family_reconstruction"
        minimal_moves = (1 << disk_count) - 1
        question = (
            f"Consider the classical Tower of Hanoi puzzle with three rods A, B, and C and {disk_count} disks of distinct sizes. "
            "Initially all disks are stacked on rod A in decreasing size order, and the goal is to move the entire stack to rod C using rod B as the auxiliary rod. "
            "A legal move transfers exactly one top disk at a time, and a larger disk may never be placed on top of a smaller disk. "
            "Derive the optimal recursive solution in enough detail that the full move sequence is unambiguous, and make explicit why the total number of moves is minimal."
        )
        metadata = _base_metadata(
            task_category="complex_recursive_reasoning",
            subtask="tower_of_hanoi",
            anchor_type=anchor_type,
        )
        metadata.update(
            {
                "disk_count": disk_count,
                "start_rod": "A",
                "target_rod": "C",
                "auxiliary_rod": "B",
                "minimal_moves": minimal_moves,
                "construction_constraints": {
                    "intended_reasoning_depth": "long recursive expansion",
                },
            }
        )
        rows.append(
            _build_record(
                example_id=f"loopbench_reconstructed_tower_of_hanoi_{row_idx:04d}",
                question=question,
                metadata=metadata,
            )
        )
    return rows


def _generate_magic_square_family() -> List[Tuple[Tuple[int, ...], ...]]:
    base_squares = [
        (
            (16, 3, 2, 13),
            (5, 10, 11, 8),
            (9, 6, 7, 12),
            (4, 15, 14, 1),
        ),
        (
            (1, 15, 14, 4),
            (12, 6, 7, 9),
            (8, 10, 11, 5),
            (13, 3, 2, 16),
        ),
        (
            (1, 2, 15, 16),
            (12, 14, 3, 5),
            (13, 7, 10, 4),
            (8, 11, 6, 9),
        ),
        (
            (4, 14, 15, 1),
            (9, 7, 6, 12),
            (5, 11, 10, 8),
            (16, 2, 3, 13),
        ),
    ]
    symmetries = [
        lambda s: s,
        lambda s: tuple(tuple(row[::-1]) for row in s),
        lambda s: tuple(s[::-1]),
        lambda s: tuple(tuple(s[len(s) - 1 - c][len(s) - 1 - r] for c in range(len(s))) for r in range(len(s))),
        lambda s: tuple(tuple(s[len(s) - 1 - c][r] for c in range(len(s))) for r in range(len(s))),
        lambda s: tuple(tuple(s[c][len(s) - 1 - r] for c in range(len(s))) for r in range(len(s))),
        lambda s: tuple(tuple(s[c][r] for c in range(len(s))) for r in range(len(s))),
        lambda s: tuple(tuple(s[len(s) - 1 - r][len(s) - 1 - c] for c in range(len(s))) for r in range(len(s))),
    ]
    all_squares: List[Tuple[Tuple[int, ...], ...]] = []
    seen: set[Tuple[Tuple[int, ...], ...]] = set()
    for base in base_squares:
        for shift in range(0, 5):
            shifted = tuple(tuple(value + shift for value in row) for row in base)
            for transform in symmetries:
                square = transform(shifted)
                if square in seen:
                    continue
                seen.add(square)
                all_squares.append(square)
    return all_squares


def _logical_public_signature(square: Sequence[Sequence[int]]) -> Tuple[int, int, int, int, int]:
    flat = [value for row in square for value in row]
    corners = [square[0][0], square[0][-1], square[-1][0], square[-1][-1]]
    centers = [square[1][1], square[1][2], square[2][1], square[2][2]]
    return (
        min(flat),
        max(flat),
        sum(square[0]),
        sum(1 for value in corners if _is_prime(value)),
        sum(1 for value in centers if value % 2 == 0),
    )


def _logical_observation_signature(square: Sequence[Sequence[int]], agent_index: int) -> Tuple[int, ...]:
    size = len(square)
    row_idx = agent_index // size
    col_idx = agent_index % size
    row_view = tuple(square[row_idx][col] for col in range(size) if col != col_idx)
    col_view = tuple(square[row][col_idx] for row in range(size) if row != row_idx)
    return row_view + col_view


def _search_logical_specs_in_family(family: Sequence[Tuple[Tuple[int, ...], ...]]) -> List[LogicalParadoxSpec]:
    if len(family) < 2:
        return []
    candidate_indices = list(range(len(family)))
    size = len(family[0])
    target_sum = sum(family[0][0])
    prime_corner_count = _logical_public_signature(family[0])[3]
    even_center_count = _logical_public_signature(family[0])[4]

    specs: List[LogicalParadoxSpec] = []
    remaining = list(candidate_indices)
    round_count = 0
    while remaining and round_count <= 24:
        knowers: Dict[int, List[int]] = {}
        for idx in remaining:
            grid = family[idx]
            knowing_agents: List[int] = []
            for agent in range(size * size):
                obs = _logical_observation_signature(grid, agent)
                possible_values = {
                    family[other_idx][agent // size][agent % size]
                    for other_idx in remaining
                    if _logical_observation_signature(family[other_idx], agent) == obs
                }
                if len(possible_values) == 1:
                    knowing_agents.append(agent)
            knowers[idx] = knowing_agents

        current_specs: List[LogicalParadoxSpec] = []
        for idx, agents in knowers.items():
            if len(agents) != 1:
                continue
            current_specs.append(
                LogicalParadoxSpec(
                    square=family[idx],
                    round_count_before_knowledge=round_count,
                    knower_index=agents[0],
                    candidate_count=len(family),
                    target_sum=target_sum,
                    prime_corner_count=prime_corner_count,
                    even_center_count=even_center_count,
                )
            )
        if current_specs:
            specs.extend(current_specs)

        next_remaining = [idx for idx in remaining if not knowers[idx]]
        if len(next_remaining) == len(remaining):
            break
        remaining = next_remaining
        round_count += 1
    return specs


def _build_path_planning_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(_make_path_planning_anchor_row())

    seen: set[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]] = set()
    attempts = 0
    while len(rows) < per_task and attempts < 10000:
        attempts += 1
        dimension = rng.choice([3, 4])
        side = rng.choice([7, 8, 9])
        barrier_count = rng.choice([3, 4] if side >= 8 else [3])
        wall_positions = tuple(sorted(rng.sample(range(1, side), barrier_count)))
        slit_heights = tuple(rng.randint(1, side - 1) for _ in range(barrier_count))
        key = (dimension, side, wall_positions, slit_heights)
        if key in seen:
            continue
        seen.add(key)
        spec = _find_modular_path_spec(
            dimension=dimension,
            side=side,
            wall_positions=wall_positions,
            slit_heights=slit_heights,
        )
        if spec is None:
            continue
        rows.append(
            _make_path_planning_row(
                row_idx=len(rows) + 1,
                spec=spec,
                anchor_type="table4_family_reconstruction",
            )
        )
    if len(rows) < per_task:
        raise ValueError(f"Unable to construct enough path-planning rows: requested={per_task}, built={len(rows)}.")
    return rows


def _make_path_planning_anchor_row() -> Dict[str, Any]:
    question = (
        "Consider a 4-dimensional grid of integer lattice points (x_1, x_2, x_3, x_4) with 0 <= x_i <= 9 for every coordinate. "
        "You start at (0, 0, 0, 0) and want to reach (9, 9, 9, 9). In one move, you may change exactly one coordinate by +1 or -1 while staying within bounds. "
        "A lattice point is blocked if x_1 + x_2 + x_3 + x_4 is divisible by 3, except that the start and goal remain allowed. "
        "Find a shortest valid path from (0, 0, 0, 0) to (9, 9, 9, 9) and output the full coordinate sequence along that path."
    )
    metadata = _base_metadata(
        task_category="complex_recursive_reasoning",
        subtask="path_planning",
        anchor_type="table5_anchor",
    )
    metadata.update(
        {
            "dimension": 4,
            "side": 9,
            "modulus": 3,
            "forbidden_residue": 0,
            "offset": 0,
            "start": [0, 0, 0, 0],
            "goal": [9, 9, 9, 9],
            "shortest_path_length": None,
            "shortest_path": None,
            "construction_constraints": {
                "obstacle_family": "modular_arithmetic_barrier",
                "intended_reasoning_depth": "long combinatorial search under modular constraints",
            },
            "solver_status": "table5_anchor_prompt_only",
        }
    )
    return _build_record(
        example_id="loopbench_reconstructed_path_planning_0001",
        question=question,
        metadata=metadata,
    )


def _make_path_planning_row(
    *,
    row_idx: int,
    spec: PathPlanningSpec,
    anchor_type: str,
) -> Dict[str, Any]:
    start = tuple(0 for _ in range(spec.dimension))
    goal = tuple(spec.side for _ in range(spec.dimension))
    coord_vars = ", ".join(f"x_{idx + 1}" for idx in range(spec.dimension))
    obstacle_rule = " ".join(
        f"For the barrier hyperplane x_1 = {wall_x}, the only passable lattice points on that hyperplane are those with x_2 = {slit_y}."
        for wall_x, slit_y in zip(spec.wall_positions, spec.slit_heights)
    )
    question = (
        f"Consider a {spec.dimension}-dimensional grid of integer lattice points ({coord_vars}) with 0 <= x_i <= {spec.side} for every coordinate. "
        f"You start at {_format_point(start)} and want to reach {_format_point(goal)}. In one move, you may change exactly one coordinate by +1 or -1 while staying within bounds. "
        f"{obstacle_rule} Find a shortest valid path from {_format_point(start)} to {_format_point(goal)} and output the full coordinate sequence along that path."
    )
    metadata = _base_metadata(
        task_category="complex_recursive_reasoning",
        subtask="path_planning",
        anchor_type=anchor_type,
    )
    metadata.update(
        {
            "dimension": spec.dimension,
            "side": spec.side,
            "wall_positions": list(spec.wall_positions),
            "slit_heights": list(spec.slit_heights),
            "start": list(start),
            "goal": list(goal),
            "shortest_path_length": len(spec.shortest_path) - 1,
            "shortest_path": [list(point) for point in spec.shortest_path],
            "construction_constraints": {
                "obstacle_family": "solver_backed_slit_walls",
                "intended_reasoning_depth": "long combinatorial search under modular constraints",
            },
            "solver_status": "validated",
        }
    )
    return _build_record(
        example_id=f"loopbench_reconstructed_path_planning_{row_idx:04d}",
        question=question,
        metadata=metadata,
    )


def _is_perfect_square(value: int) -> bool:
    root = isqrt(value)
    return root * root == value


def _is_perfect_power(value: int, degree: int) -> bool:
    root = round(value ** (1.0 / degree))
    return root ** degree == value or max(root - 1, 0) ** degree == value or (root + 1) ** degree == value


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    limit = isqrt(value)
    for factor in range(3, limit + 1, 2):
        if value % factor == 0:
            return False
    return True


def _find_modular_path_spec(
    *,
    dimension: int,
    side: int,
    wall_positions: Sequence[int],
    slit_heights: Sequence[int],
) -> Optional[PathPlanningSpec]:
    path = _shortest_path_with_slit_walls(
        dimension=dimension,
        side=side,
        wall_positions=wall_positions,
        slit_heights=slit_heights,
    )
    if path is None:
        return None
    minimum_possible = dimension * side
    if len(path) - 1 <= minimum_possible + 1:
        return None
    return PathPlanningSpec(
        dimension=dimension,
        side=side,
        wall_positions=tuple(int(item) for item in wall_positions),
        slit_heights=tuple(int(item) for item in slit_heights),
        shortest_path=tuple(path),
    )


def _generate_logical_paradox_specs() -> List[LogicalParadoxSpec]:
    all_squares = _generate_magic_square_family()
    by_signature: Dict[Tuple[int, int, int, int, int], List[Tuple[Tuple[int, ...], ...]]] = defaultdict(list)
    for square in all_squares:
        by_signature[_logical_public_signature(square)].append(square)

    specs: List[LogicalParadoxSpec] = []
    for family in by_signature.values():
        family_specs = _search_logical_specs_in_family(family)
        specs.extend(family_specs)
    specs.sort(
        key=lambda item: (
            item.round_count_before_knowledge,
            item.candidate_count,
            item.knower_index,
            item.target_sum,
            item.square,
        )
    )
    unique_specs: List[LogicalParadoxSpec] = []
    seen: set[Tuple[Tuple[Tuple[int, ...], ...], int, int]] = set()
    for spec in specs:
        key = (spec.square, spec.round_count_before_knowledge, spec.knower_index)
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append(spec)
    return unique_specs


def _generate_magic_square_prompt_specs_fallback() -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for square in _generate_magic_square_family():
        flat = [value for row in square for value in row]
        span = max(flat) - min(flat)
        round_count = 10 + (sum(flat) + span) % 19
        knower_index = (sum(square[0]) + square[1][1] + square[2][2]) % 16
        specs.append(
            {
                "square": square,
                "round_count": round_count,
                "knower_label": f"L{knower_index + 1}",
            }
        )
    return specs
