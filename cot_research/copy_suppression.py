from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _dedup_preserve_order(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        token_id = int(value)
        if token_id in seen:
            continue
        seen.add(token_id)
        out.append(token_id)
    return out


def build_generated_copy_step_rows(
    generated_token_ids: Sequence[int],
    *,
    recent_window: int,
) -> List[Dict[str, Any]]:
    token_ids = [int(token_id) for token_id in generated_token_ids]
    rows: List[Dict[str, Any]] = []
    for step_idx, token_id in enumerate(token_ids):
        prefix_ids = token_ids[:step_idx]
        if recent_window > 0:
            recent_prefix = prefix_ids[max(0, step_idx - recent_window) :]
            older_prefix = prefix_ids[: max(0, step_idx - recent_window)]
        else:
            recent_prefix = []
            older_prefix = prefix_ids
        recent_token_ids = _dedup_preserve_order(recent_prefix)
        recent_set = set(recent_token_ids)
        older_seen_token_ids = [item for item in _dedup_preserve_order(older_prefix) if item not in recent_set]

        last_occurrence_distance: Optional[int] = None
        for prev_idx in range(step_idx - 1, -1, -1):
            if token_ids[prev_idx] == token_id:
                last_occurrence_distance = int(step_idx - prev_idx)
                break

        rows.append(
            {
                "step_idx": int(step_idx),
                "target_token_id": token_id,
                "target_is_recent_copy": bool(token_id in recent_set),
                "target_is_older_copy": bool(token_id in older_seen_token_ids),
                "target_is_any_copy": bool(token_id in prefix_ids),
                "target_equals_prev_token": bool(step_idx > 0 and token_ids[step_idx - 1] == token_id),
                "last_occurrence_distance": last_occurrence_distance,
                "recent_token_ids": recent_token_ids,
                "older_seen_token_ids": older_seen_token_ids,
                "recent_unique_count": int(len(recent_token_ids)),
                "older_seen_unique_count": int(len(older_seen_token_ids)),
                "prefix_generated_tokens": int(step_idx),
            }
        )
    return rows


def summarize_generated_copy_step_rows(step_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_steps = len(step_rows)
    eligible_recent_steps = sum(1 for row in step_rows if int(row.get("recent_unique_count", 0)) > 0)
    any_copy_rows = [row for row in step_rows if bool(row.get("target_is_any_copy"))]
    recent_copy_rows = [row for row in step_rows if bool(row.get("target_is_recent_copy"))]
    older_copy_rows = [row for row in step_rows if bool(row.get("target_is_older_copy"))]
    prev_copy_rows = [row for row in step_rows if bool(row.get("target_equals_prev_token"))]
    return {
        "total_generated_tokens": int(total_steps),
        "eligible_recent_context_steps": int(eligible_recent_steps),
        "any_copy_count": int(len(any_copy_rows)),
        "any_copy_rate": 0.0 if total_steps == 0 else round(len(any_copy_rows) / total_steps, 6),
        "recent_copy_count": int(len(recent_copy_rows)),
        "recent_copy_rate_over_all_steps": 0.0 if total_steps == 0 else round(len(recent_copy_rows) / total_steps, 6),
        "recent_copy_rate_over_eligible_steps": 0.0
        if eligible_recent_steps == 0
        else round(len(recent_copy_rows) / eligible_recent_steps, 6),
        "older_copy_count": int(len(older_copy_rows)),
        "older_copy_rate": 0.0 if total_steps == 0 else round(len(older_copy_rows) / total_steps, 6),
        "prev_token_copy_count": int(len(prev_copy_rows)),
        "prev_token_copy_rate": 0.0 if total_steps == 0 else round(len(prev_copy_rows) / total_steps, 6),
        "mean_last_occurrence_distance_any_copy": round(
            _mean(float(row["last_occurrence_distance"]) for row in any_copy_rows if row.get("last_occurrence_distance") is not None),
            6,
        ),
        "mean_last_occurrence_distance_recent_copy": round(
            _mean(float(row["last_occurrence_distance"]) for row in recent_copy_rows if row.get("last_occurrence_distance") is not None),
            6,
        ),
    }


def sample_analysis_step_indices(
    step_rows: Sequence[Dict[str, Any]],
    *,
    max_recent_copy_steps: int,
    max_control_steps: int,
    seed: int,
) -> List[int]:
    rng = random.Random(seed)
    recent_copy_indices = [
        int(row["step_idx"])
        for row in step_rows
        if bool(row.get("target_is_recent_copy"))
    ]
    control_indices = [
        int(row["step_idx"])
        for row in step_rows
        if int(row.get("recent_unique_count", 0)) > 0 and not bool(row.get("target_is_recent_copy"))
    ]
    if max_recent_copy_steps >= 0 and len(recent_copy_indices) > max_recent_copy_steps:
        recent_copy_indices = sorted(rng.sample(recent_copy_indices, max_recent_copy_steps))
    if max_control_steps >= 0 and len(control_indices) > max_control_steps:
        control_indices = sorted(rng.sample(control_indices, max_control_steps))
    return sorted(set(recent_copy_indices + control_indices))


def sample_random_control_token_ids(
    *,
    vocab_size: int,
    excluded_token_ids: Sequence[int],
    sample_size: int,
    seed: int,
) -> List[int]:
    if sample_size <= 0 or vocab_size <= 0:
        return []
    rng = random.Random(seed)
    excluded = {int(token_id) for token_id in excluded_token_ids}
    sample: List[int] = []
    while len(sample) < sample_size:
        token_id = int(rng.randrange(vocab_size))
        if token_id in excluded or token_id in sample:
            continue
        sample.append(token_id)
    return sample


def token_stats_from_logits(
    logits_row: torch.Tensor,
    token_id: int,
    *,
    log_norm: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if log_norm is None:
        log_norm = torch.logsumexp(logits_row, dim=-1)
    token_id = int(token_id)
    token_logit = logits_row[token_id]
    token_prob = torch.exp(token_logit - log_norm)
    rank = int(torch.count_nonzero(logits_row > token_logit).item()) + 1
    return {
        "token_id": token_id,
        "logit": float(token_logit.detach().item()),
        "prob": float(token_prob.detach().item()),
        "rank": int(rank),
    }


def prob_mass_for_token_ids(
    logits_row: torch.Tensor,
    token_ids: Sequence[int],
    *,
    log_norm: Optional[torch.Tensor] = None,
) -> float:
    if not token_ids:
        return 0.0
    if log_norm is None:
        log_norm = torch.logsumexp(logits_row, dim=-1)
    indices = torch.tensor([int(token_id) for token_id in token_ids], dtype=torch.long, device=logits_row.device)
    selected_logits = torch.index_select(logits_row, 0, indices)
    probs = torch.exp(selected_logits - log_norm)
    return float(probs.sum().detach().item())


def build_condition_step_metric(
    logits_row: torch.Tensor,
    step_row: Dict[str, Any],
    *,
    vocab_size: int,
    random_seed: int,
    fixed_candidate_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    logits_row = logits_row.float()
    log_norm = torch.logsumexp(logits_row, dim=-1)

    recent_token_ids = [int(token_id) for token_id in step_row.get("recent_token_ids") or []]
    older_seen_token_ids = [int(token_id) for token_id in step_row.get("older_seen_token_ids") or []]
    excluded_token_ids = list(recent_token_ids) + list(older_seen_token_ids)
    random_control_token_ids = sample_random_control_token_ids(
        vocab_size=vocab_size,
        excluded_token_ids=excluded_token_ids,
        sample_size=len(recent_token_ids),
        seed=random_seed,
    )

    candidate_token_id = None if fixed_candidate_token_id is None else int(fixed_candidate_token_id)
    if candidate_token_id is None and recent_token_ids:
        recent_indices = torch.tensor(recent_token_ids, dtype=torch.long, device=logits_row.device)
        selected_recent_logits = torch.index_select(logits_row, 0, recent_indices)
        best_idx = int(torch.argmax(selected_recent_logits).item())
        candidate_token_id = int(recent_token_ids[best_idx])

    actual_token_stats = token_stats_from_logits(
        logits_row,
        int(step_row["target_token_id"]),
        log_norm=log_norm,
    )
    candidate_stats = None
    if candidate_token_id is not None:
        candidate_stats = token_stats_from_logits(
            logits_row,
            candidate_token_id,
            log_norm=log_norm,
        )

    return {
        "recent_mass": prob_mass_for_token_ids(logits_row, recent_token_ids, log_norm=log_norm),
        "older_seen_mass": prob_mass_for_token_ids(logits_row, older_seen_token_ids, log_norm=log_norm),
        "random_control_mass": prob_mass_for_token_ids(logits_row, random_control_token_ids, log_norm=log_norm),
        "actual_token_id": int(step_row["target_token_id"]),
        "actual_token_logit": float(actual_token_stats["logit"]),
        "actual_token_prob": float(actual_token_stats["prob"]),
        "actual_token_rank": int(actual_token_stats["rank"]),
        "candidate_token_id": None if candidate_stats is None else int(candidate_stats["token_id"]),
        "candidate_token_logit": None if candidate_stats is None else float(candidate_stats["logit"]),
        "candidate_token_prob": None if candidate_stats is None else float(candidate_stats["prob"]),
        "candidate_token_rank": None if candidate_stats is None else int(candidate_stats["rank"]),
    }
