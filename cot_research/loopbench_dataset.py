from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import isqrt
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOOPBENCH_PAPER_ID = "2601.05693"
LOOPBENCH_PAPER_URL = f"https://arxiv.org/abs/{LOOPBENCH_PAPER_ID}"

# Paraphrased from the paper's unified instruction. We keep it in metadata so
# later experiments can opt into a LoopBench-style prompt without hard-coding it
# into every row's question text.
RECOMMENDED_SYSTEM_PROMPT = (
    "You are a meticulous, by-the-book reasoner. Show every intermediate step, "
    "digit update, remainder, state transition, or recursive move explicitly "
    "before giving the final answer."
)


@dataclass(frozen=True)
class SumProductCandidate:
    a: int
    b: int
    max_value: int
    first_knowledge_round: int
    first_knowledge_holder: str
    elimination_trace: Tuple[Dict[str, Any], ...]


def build_loopbench_inspired_dataset(
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


def build_loopbench_inspired_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    category_counts: Dict[str, int] = defaultdict(int)
    subtask_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        category_counts[str(metadata.get("task_category") or "unknown")] += 1
        subtask_counts[str(metadata.get("subtask") or "unknown")] += 1

    return {
        "benchmark_name": "loopbench_inspired",
        "construction_note": (
            "This is a LoopBench-inspired dataset reconstructed from the public "
            "task taxonomy, representative prompts, and appendix constraints in "
            f"arXiv:{LOOPBENCH_PAPER_ID}. It is not the paper's original 700-sample "
            "GPT-5-synthesized benchmark."
        ),
        "paper_reference": {
            "arxiv_id": LOOPBENCH_PAPER_ID,
            "url": LOOPBENCH_PAPER_URL,
        },
        "recommended_system_prompt": RECOMMENDED_SYSTEM_PROMPT,
        "total_examples": len(rows),
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
    source_variant: str = "loopbench_inspired",
) -> Dict[str, Any]:
    return {
        "benchmark_name": "loopbench_inspired",
        "benchmark_origin": source_variant,
        "paper_reference": {
            "arxiv_id": LOOPBENCH_PAPER_ID,
            "url": LOOPBENCH_PAPER_URL,
        },
        "task_category": task_category,
        "subtask": subtask,
        "recommended_system_prompt": RECOMMENDED_SYSTEM_PROMPT,
    }


def _build_record(
    *,
    example_id: str,
    question: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": example_id,
        "question": question,
        "correct_answer": None,
        "wrong_answer": None,
        "metadata": metadata,
    }


def _build_square_root_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int]] = set()

    while len(rows) < per_task:
        precision = rng.choice([220, 250, 300, 350, 400, 450, 500])
        n = rng.randint(10_000_000, 999_999_999)
        if _is_perfect_square(n):
            continue
        key = (n, precision)
        if key in seen:
            continue
        seen.add(key)
        answer = _sqrt_decimal_string(n=n, digits=precision)
        example_id = f"loopbench_inspired_square_root_{len(rows) + 1:04d}"
        question = (
            f"Compute sqrt({n}) using the standard digit-by-digit square-root extraction algorithm. "
            f"After obtaining the integer part, continue the same process until you have produced exactly "
            f"{precision} digits after the decimal point. At every step, explicitly record the trial digit, "
            f"the updated remainder, and the next digit-pair brought down. Do not skip intermediate states. "
            f"Finally, report the decimal expansion of sqrt({n}) with exactly {precision} digits after the decimal point."
        )
        metadata = _base_metadata(
            task_category="high_precision_arithmetic",
            subtask="square_root",
        )
        metadata.update(
            {
                "radicand": n,
                "precision_digits": precision,
                "answer_decimal": answer,
                "construction_constraints": {
                    "precision_cap_digits": 500,
                    "intended_reasoning_depth": "100+ explicit digit updates",
                },
            }
        )
        rows.append(_build_record(example_id=example_id, question=question, metadata=metadata))
    return rows


def _build_long_division_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int]] = set()
    denominator_pool = _prime_pool(lower=10_000, upper=50_000)

    while len(rows) < per_task:
        precision = rng.choice([220, 250, 300, 350, 400, 450, 500])
        denominator = rng.choice(denominator_pool)
        numerator = rng.randint(1, denominator - 1)
        if numerator % denominator == 0:
            continue
        key = (numerator, denominator, precision)
        if key in seen:
            continue
        seen.add(key)
        decimal_expansion = _long_division_decimal_string(
            numerator=numerator,
            denominator=denominator,
            digits=precision,
        )
        example_id = f"loopbench_inspired_long_division_{len(rows) + 1:04d}"
        question = (
            f"Compute {numerator} ÷ {denominator} using the standard long-division algorithm. "
            f"Carry out the division until you obtain exactly {precision} digits after the decimal point. "
            f"At each step, explicitly record the current remainder, the next quotient digit, and the updated remainder. "
            f"Do not compress repeated phases. Finally, report the decimal expansion with all {precision} digits."
        )
        metadata = _base_metadata(
            task_category="high_precision_arithmetic",
            subtask="long_division",
        )
        metadata.update(
            {
                "numerator": numerator,
                "denominator": denominator,
                "precision_digits": precision,
                "answer_decimal": decimal_expansion,
                "construction_constraints": {
                    "precision_cap_digits": 500,
                    "intended_reasoning_depth": "100+ explicit remainder updates",
                },
            }
        )
        rows.append(_build_record(example_id=example_id, question=question, metadata=metadata))
    return rows


def _build_newton_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, int, int]] = set()

    while len(rows) < per_task:
        degree = rng.choice([3, 4, 5, 6])
        precision = rng.choice([220, 250, 300, 350, 400, 450, 500])
        n = rng.randint(100_000, 999_999_999)
        if _is_perfect_power(n, degree):
            continue
        initial_guess = max(2, int(round(n ** (1.0 / degree))))
        jitter = rng.choice([-3, -2, -1, 1, 2, 3])
        initial_guess = max(2, initial_guess + jitter)
        key = (n, degree, precision, initial_guess)
        if key in seen:
            continue
        seen.add(key)
        answer = _nth_root_decimal_string(
            n=n,
            degree=degree,
            digits=precision,
            initial_guess=initial_guess,
        )
        example_id = f"loopbench_inspired_newtons_iteration_{len(rows) + 1:04d}"
        question = (
            f"Use Newton's method to compute the real {degree}-th root of {n} to {precision} correct decimal places. "
            f"Start from x_0 = {initial_guess} and apply the update "
            f"x_(t+1) = (({degree}-1)·x_t + {n}/x_t^({degree}-1)) / {degree}. "
            f"For each iteration, explicitly derive x_t -> x_(t+1), carry enough precision to keep the final result "
            f"accurate to {precision} digits after the decimal point, and do not omit any intermediate iterate. "
            f"Finally, report the decimal expansion of the real {degree}-th root of {n} with exactly {precision} digits after the decimal point."
        )
        metadata = _base_metadata(
            task_category="high_precision_arithmetic",
            subtask="newtons_iteration",
        )
        metadata.update(
            {
                "target_value": n,
                "degree": degree,
                "precision_digits": precision,
                "initial_guess": initial_guess,
                "answer_decimal": answer,
                "construction_constraints": {
                    "precision_cap_digits": 500,
                    "intended_reasoning_depth": "100+ explicit arithmetic updates",
                },
            }
        )
        rows.append(_build_record(example_id=example_id, question=question, metadata=metadata))
    return rows


def _build_truth_teller_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, Tuple[int, ...]]] = set()

    while len(rows) < per_task:
        n = rng.randint(18, 28)
        counts = tuple(rng.randint(0, 3) for _ in range(n))
        key = (n, counts)
        if key in seen:
            continue
        seen.add(key)
        solutions = _solve_truth_teller_counts(counts)
        if not solutions or len(solutions) > 4:
            continue
        if all(all(bit == solution[0] for bit in solution) for solution in solutions):
            continue

        lines = [
            f"There are {n} agents, labeled C_1 through C_{n}, standing in a circle. "
            "Each agent is either a truth-teller or a liar.",
            "For every i, indices are taken modulo the circle.",
            "Their statements are:",
        ]
        for idx, count in enumerate(counts, start=1):
            lines.append(
                f"- C_{idx} says: \"Exactly {count} of C_{_wrap_idx(idx + 1, n)}, "
                f"C_{_wrap_idx(idx + 2, n)}, and C_{_wrap_idx(idx + 3, n)} are liars.\""
            )
        lines.append(
            "Determine all truth-value assignments consistent with the full system, and for each surviving assignment "
            "verify that every agent's statement matches the speaker's status."
        )
        question = "\n".join(lines)
        example_id = f"loopbench_inspired_truth_teller_{len(rows) + 1:04d}"
        metadata = _base_metadata(
            task_category="complex_recursive_reasoning",
            subtask="truth_teller_puzzles",
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
        rows.append(_build_record(example_id=example_id, question=question, metadata=metadata))
    return rows


def _build_logical_paradox_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidates = _enumerate_sum_product_candidates(max_value_min=24, max_value_max=120, max_rounds=12)
    candidates = [
        item
        for item in candidates
        if item.first_knowledge_round >= 4
    ]
    rng.shuffle(candidates)
    used: set[Tuple[int, int, int]] = set()

    for candidate in candidates:
        if len(rows) >= per_task:
            break
        key = (candidate.max_value, candidate.a, candidate.b)
        if key in used:
            continue
        used.add(key)
        transcript = _build_sum_product_transcript(candidate)
        holder_name = "Product-holder P" if candidate.first_knowledge_holder == "product" else "Sum-holder S"
        question = (
            f"Two hidden integers x and y satisfy 2 <= x < y <= {candidate.max_value}. "
            "Logician S knows only the sum x+y, while logician P knows only the product x·y. "
            "Both know the range, both know that the other is perfectly rational, and every public statement becomes common knowledge.\n\n"
            "Transcript:\n"
            f"{transcript}\n\n"
            f"In Round {candidate.first_knowledge_round}, {holder_name} is the first speaker to say "
            "\"Now I know the two integers.\" Determine the unique pair (x, y), and explicitly reconstruct the candidate-elimination chain."
        )
        example_id = f"loopbench_inspired_logical_paradox_{len(rows) + 1:04d}"
        metadata = _base_metadata(
            task_category="complex_recursive_reasoning",
            subtask="logical_paradox",
            source_variant="loopbench_inspired_sum_product_variant",
        )
        metadata.update(
            {
                "variant": "sum_product_public_announcement",
                "max_value": candidate.max_value,
                "solution_pair": [candidate.a, candidate.b],
                "first_knowledge_round": candidate.first_knowledge_round,
                "first_knowledge_holder": candidate.first_knowledge_holder,
                "elimination_trace": list(candidate.elimination_trace),
                "construction_constraints": {
                    "intended_reasoning_depth": "multi-round public-announcement elimination",
                },
                "reproduction_note": (
                    "The paper's logical-paradox family is described only at a high level. "
                    "This repo reconstructs it as a solver-backed public-announcement puzzle "
                    "instead of claiming exact sample-level reproduction."
                ),
            }
        )
        rows.append(_build_record(example_id=example_id, question=question, metadata=metadata))
    if len(rows) < per_task:
        raise ValueError(
            f"Not enough logical-paradox candidates were generated: requested={per_task}, built={len(rows)}."
        )
    return rows


def _build_tower_of_hanoi_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, str, str, str]] = set()
    rod_labels = ["A", "B", "C"]

    while len(rows) < per_task:
        # The classical 3-rod variant has only 6 start/target assignments, so
        # we need a wider disk-count range than [16, 22] to build 100 distinct
        # prompts without looping forever.
        disk_count = rng.randint(16, 96)
        start, target = rng.sample(rod_labels, 2)
        auxiliary = [rod for rod in rod_labels if rod not in {start, target}][0]
        key = (disk_count, start, target, auxiliary)
        if key in seen:
            continue
        seen.add(key)
        minimal_moves = (1 << disk_count) - 1
        example_id = f"loopbench_inspired_tower_of_hanoi_{len(rows) + 1:04d}"
        question = (
            f"Consider the classical Tower of Hanoi puzzle with rods {', '.join(rod_labels)} and {disk_count} disks of distinct sizes. "
            f"Initially all disks are stacked on rod {start} in decreasing size order, and the goal is to move the full stack to rod {target} "
            f"using rod {auxiliary} as the auxiliary rod. A legal move transfers exactly one top disk at a time, and a larger disk may never be placed on a smaller disk. "
            f"Describe the optimal recursive move pattern that achieves the transfer in the minimum possible number of moves, and make the recurrence explicit enough that a reader could reconstruct all {minimal_moves} moves without ambiguity."
        )
        metadata = _base_metadata(
            task_category="complex_recursive_reasoning",
            subtask="tower_of_hanoi",
        )
        metadata.update(
            {
                "disk_count": disk_count,
                "start_rod": start,
                "target_rod": target,
                "auxiliary_rod": auxiliary,
                "minimal_moves": minimal_moves,
                "construction_constraints": {
                    "intended_reasoning_depth": "long recursive expansion",
                },
            }
        )
        rows.append(_build_record(example_id=example_id, question=question, metadata=metadata))
    return rows


def _build_path_planning_rows(*, per_task: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]] = set()

    while len(rows) < per_task:
        dimension = rng.choice([3, 4])
        side = rng.choice([7, 8, 9])
        barrier_count = rng.choice([3, 4] if side >= 8 else [3])
        wall_positions = tuple(sorted(rng.sample(range(1, side), barrier_count)))
        slit_heights = tuple(rng.randint(1, side - 1) for _ in range(barrier_count))
        key = (dimension, side, wall_positions, slit_heights)
        if key in seen:
            continue
        seen.add(key)

        y_travel = abs(slit_heights[0]) + abs(side - slit_heights[-1])
        y_travel += sum(abs(slit_heights[idx + 1] - slit_heights[idx]) for idx in range(barrier_count - 1))
        if y_travel - side < 2:
            continue

        path = _shortest_path_with_slit_walls(
            dimension=dimension,
            side=side,
            wall_positions=wall_positions,
            slit_heights=slit_heights,
        )
        if path is None:
            continue

        start = tuple(0 for _ in range(dimension))
        goal = tuple(side for _ in range(dimension))
        coord_vars = ", ".join(f"x_{idx + 1}" for idx in range(dimension))
        start_text = _format_point(start)
        goal_text = _format_point(goal)
        example_id = f"loopbench_inspired_path_planning_{len(rows) + 1:04d}"
        barrier_rules = " ".join(
            f"For the wall x_1 = {wall_x}, the only passable points on that wall are those with x_2 = {slit_y}."
            for wall_x, slit_y in zip(wall_positions, slit_heights)
        )
        question = (
            f"Consider a {dimension}-dimensional grid of integer lattice points ({coord_vars}) with 0 <= x_i <= {side} for every coordinate. "
            f"You start at {start_text} and want to reach {goal_text}. In one move, you may change exactly one coordinate by ±1 while staying within bounds. "
            f"{barrier_rules} "
            f"The start and goal are always allowed even if they satisfy one of the blocking rules. "
            f"Find a shortest valid path from {start_text} to {goal_text}, and output the full coordinate sequence along that path."
        )
        metadata = _base_metadata(
            task_category="complex_recursive_reasoning",
            subtask="path_planning",
        )
        metadata.update(
            {
                "dimension": dimension,
                "side": side,
                "start": list(start),
                "goal": list(goal),
                "wall_positions": list(wall_positions),
                "slit_heights": list(slit_heights),
                "shortest_path_length": len(path) - 1,
                "shortest_path": [list(point) for point in path],
                "construction_constraints": {
                    "obstacle_family": "alternating_vertical_slit_walls",
                    "minimum_detour_over_monotone_shortest_path": 2,
                    "intended_reasoning_depth": "state expansion over long combinatorial search",
                },
            }
        )
        rows.append(_build_record(example_id=example_id, question=question, metadata=metadata))
    return rows


def _is_perfect_square(value: int) -> bool:
    root = isqrt(value)
    return root * root == value


def _is_perfect_power(value: int, degree: int) -> bool:
    root = round(value ** (1.0 / degree))
    return root ** degree == value or max(root - 1, 0) ** degree == value or (root + 1) ** degree == value


def _sqrt_decimal_string(*, n: int, digits: int) -> str:
    integer_digits = max(len(str(n)) // 2 + 2, 4)
    with localcontext() as ctx:
        ctx.prec = digits + integer_digits + 20
        value = Decimal(n).sqrt()
        return format(value, f".{digits}f")


def _long_division_decimal_string(*, numerator: int, denominator: int, digits: int) -> str:
    integer_part = numerator // denominator
    remainder = numerator % denominator
    pieces: List[str] = []
    for _ in range(digits):
        remainder *= 10
        pieces.append(str(remainder // denominator))
        remainder %= denominator
    return f"{integer_part}." + "".join(pieces)


def _nth_root_decimal_string(*, n: int, degree: int, digits: int, initial_guess: int) -> str:
    if degree <= 1:
        raise ValueError("degree must be >= 2")
    with localcontext() as ctx:
        ctx.prec = digits + 50
        target = Decimal(n)
        m = Decimal(degree)
        x = Decimal(initial_guess)
        tolerance = Decimal(1).scaleb(-(digits + 10))
        for _ in range(256):
            prev = x
            x = ((m - 1) * x + target / (x ** (degree - 1))) / m
            if abs(x - prev) < tolerance:
                break
        return format(x, f".{digits}f")


def _wrap_idx(idx: int, n: int) -> str:
    value = ((idx - 1) % n) + 1
    return f"C_{value}"


def _solve_truth_teller_counts(counts: Sequence[int]) -> List[Tuple[int, ...]]:
    n = len(counts)
    if n < 4:
        raise ValueError("truth-teller cycle needs at least 4 agents")
    solutions: List[Tuple[int, ...]] = []
    seen: set[Tuple[int, ...]] = set()

    for s0 in (0, 1):
        for s1 in (0, 1):
            for s2 in (0, 1):
                status: List[Optional[int]] = [None] * n
                status[0] = s0
                status[1] = s1
                status[2] = s2
                for idx in range(n - 1, 2, -1):
                    liar_count = sum(1 - int(status[(idx + delta) % n]) for delta in (1, 2, 3))
                    status[idx] = 1 if liar_count == counts[idx] else 0
                if any(item is None for item in status):
                    continue
                resolved = tuple(int(item) for item in status)
                if not _truth_assignment_is_consistent(resolved, counts):
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                solutions.append(resolved)

    solutions.sort()
    return solutions


def _truth_assignment_is_consistent(status: Sequence[int], counts: Sequence[int]) -> bool:
    n = len(status)
    for idx, claim in enumerate(counts):
        liar_count = sum(1 - status[(idx + delta) % n] for delta in (1, 2, 3))
        expected_status = 1 if liar_count == claim else 0
        if status[idx] != expected_status:
            return False
    return True


def _truth_assignment_to_string(status: Sequence[int]) -> str:
    return "".join("T" if value == 1 else "L" for value in status)


def _enumerate_sum_product_candidates(
    *,
    max_value_min: int,
    max_value_max: int,
    max_rounds: int,
) -> List[SumProductCandidate]:
    candidates: List[SumProductCandidate] = []
    for max_value in range(max_value_min, max_value_max + 1):
        universe = [(a, b) for a in range(2, max_value + 1) for b in range(a + 1, max_value + 1)]
        current = list(universe)
        first_knowledge: Dict[Tuple[int, int], Tuple[int, str, Tuple[Dict[str, Any], ...]]] = {}
        trace: List[Dict[str, Any]] = []
        for round_idx in range(1, max_rounds + 1):
            holder = "product" if round_idx % 2 == 1 else "sum"
            grouped: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            for pair in current:
                key = pair[0] * pair[1] if holder == "product" else pair[0] + pair[1]
                grouped[key].append(pair)
            know_pairs = {group[0] for group in grouped.values() if len(group) == 1}
            trace.append(
                {
                    "round": round_idx,
                    "holder": holder,
                    "candidate_count_before": len(current),
                    "knowledge_count": len(know_pairs),
                }
            )
            if know_pairs:
                frozen_trace = tuple(dict(item) for item in trace)
                for pair in know_pairs:
                    first_knowledge.setdefault(pair, (round_idx, holder, frozen_trace))
            current = [pair for pair in current if pair not in know_pairs]
            if not current:
                break
        for pair, (round_idx, holder, frozen_trace) in first_knowledge.items():
            candidates.append(
                SumProductCandidate(
                    a=pair[0],
                    b=pair[1],
                    max_value=max_value,
                    first_knowledge_round=round_idx,
                    first_knowledge_holder=holder,
                    elimination_trace=frozen_trace,
                )
            )
    candidates.sort(
        key=lambda item: (
            item.first_knowledge_round,
            item.max_value,
            item.a,
            item.b,
        )
    )
    return candidates


def _build_sum_product_transcript(candidate: SumProductCandidate) -> str:
    lines: List[str] = []
    for round_idx in range(1, candidate.first_knowledge_round):
        holder = "P" if round_idx % 2 == 1 else "S"
        lines.append(f"Round {round_idx}: {holder} says, \"I do not know the two integers.\"")
    final_holder = "P" if candidate.first_knowledge_holder == "product" else "S"
    lines.append(f"Round {candidate.first_knowledge_round}: {final_holder} says, \"Now I know the two integers.\"")
    return "\n".join(lines)


def _prime_pool(*, lower: int, upper: int) -> List[int]:
    if upper < 2 or upper < lower:
        return []
    sieve = [True] * (upper + 1)
    sieve[0] = False
    sieve[1] = False
    for value in range(2, isqrt(upper) + 1):
        if not sieve[value]:
            continue
        start = value * value
        sieve[start : upper + 1 : value] = [False] * (((upper - start) // value) + 1)
    return [value for value in range(max(2, lower), upper + 1) if sieve[value] and value not in (2, 5)]


def _shortest_path_with_slit_walls(
    *,
    dimension: int,
    side: int,
    wall_positions: Sequence[int],
    slit_heights: Sequence[int],
) -> Optional[List[Tuple[int, ...]]]:
    from collections import deque

    wall_to_slit = {int(wall_x): int(slit_y) for wall_x, slit_y in zip(wall_positions, slit_heights)}
    start = tuple(0 for _ in range(dimension))
    goal = tuple(side for _ in range(dimension))
    queue = deque([start])
    parent: Dict[Tuple[int, ...], Optional[Tuple[int, ...]]] = {start: None}

    while queue:
        point = queue.popleft()
        if point == goal:
            break
        for axis in range(dimension):
            for delta in (-1, 1):
                nxt = list(point)
                nxt[axis] += delta
                if nxt[axis] < 0 or nxt[axis] > side:
                    continue
                nxt_tuple = tuple(nxt)
                if nxt_tuple in parent:
                    continue
                if nxt_tuple not in {start, goal}:
                    wall_x = wall_to_slit.get(nxt_tuple[0])
                    if wall_x is not None and nxt_tuple[1] != wall_x:
                        continue
                parent[nxt_tuple] = point
                queue.append(nxt_tuple)

    if goal not in parent:
        return None

    path: List[Tuple[int, ...]] = []
    cursor: Optional[Tuple[int, ...]] = goal
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return path


def _format_point(point: Sequence[int]) -> str:
    return "(" + ", ".join(str(value) for value in point) + ")"
