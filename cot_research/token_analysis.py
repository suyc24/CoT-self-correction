from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .schemas import TokenCountResult
from .text_analysis import count_substring_occurrences


def count_token_sequence(token_ids: Sequence[int], needle: Sequence[int]) -> tuple[int, Optional[int]]:
    if not token_ids or not needle or len(needle) > len(token_ids):
        return 0, None
    count = 0
    first_index: Optional[int] = None
    max_start = len(token_ids) - len(needle)
    for start in range(max_start + 1):
        if list(token_ids[start : start + len(needle)]) == list(needle):
            if first_index is None:
                first_index = start
            count += 1
    return count, first_index


def analyze_token_targets(
    backend,
    full_text: str,
    think_text: str,
    generated_token_ids: Sequence[int],
    tracked_strings: Iterable[str],
    step_scores: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    full_token_ids = backend.encode(full_text)
    think_token_ids = backend.encode(think_text) if think_text else []
    for target in tracked_strings:
        if not target:
            continue
        tracked_token_ids = backend.encode(target)
        text_count_full, first_text_pos_full = count_substring_occurrences(full_text, target)
        text_count_think, first_text_pos_think = count_substring_occurrences(think_text, target)
        token_count_full, first_token_index_full = count_token_sequence(full_token_ids, tracked_token_ids)
        token_count_think, first_token_index_think = count_token_sequence(think_token_ids, tracked_token_ids)
        filtered_step_scores: List[Dict[str, Any]] = []
        if step_scores:
            for item in step_scores:
                candidate_stats = item.get("candidates", {})
                relevant = {str(token_id): candidate_stats[str(token_id)] for token_id in tracked_token_ids if str(token_id) in candidate_stats}
                if relevant:
                    filtered_step_scores.append(
                        {
                            "step": item.get("step"),
                            "generated_token_id": item.get("generated_token_id"),
                            "generated_text": item.get("generated_text"),
                            "candidates": relevant,
                        }
                    )
        result = TokenCountResult(
            target=target,
            text_count_full=text_count_full,
            text_count_think=text_count_think,
            first_text_pos_full=first_text_pos_full,
            first_text_pos_think=first_text_pos_think,
            token_count_full=token_count_full,
            token_count_think=token_count_think,
            first_token_index_full=first_token_index_full,
            first_token_index_think=first_token_index_think,
            tracked_token_ids=tracked_token_ids,
            step_scores=filtered_step_scores,
        )
        results.append(result.to_dict())
    return results
