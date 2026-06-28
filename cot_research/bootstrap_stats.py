"""Prompt-level bootstrap CI + Benjamini-Hochberg FDR for head ablation screens.

All statistics follow the global specification from RESEARCH_BRIEF.md:
- Bootstrap: B=10000, prompt-level resampling (with replacement)
- Hypothesis: one-sided (H0: mean Δrep ≤ 0, H1: mean Δrep > 0)
- FDR: Benjamini-Hochberg, α=0.05
- Effect size: Cohen's d (paired)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_BOOTSTRAP_B = 10000
DEFAULT_FDR_ALPHA = 0.05


def compute_paired_delta_rep(
    baseline_rep: Sequence[bool],
    ablation_rep: Sequence[bool],
) -> np.ndarray:
    base = np.asarray(baseline_rep, dtype=np.float64)
    abl = np.asarray(ablation_rep, dtype=np.float64)
    if base.shape != abl.shape:
        raise ValueError(f"Shape mismatch: baseline {base.shape} vs ablation {abl.shape}")
    return abl - base


def bootstrap_mean_one_sided(
    delta: np.ndarray,
    B: int = DEFAULT_BOOTSTRAP_B,
    seed: int = 42,
) -> Dict[str, float]:
    n = len(delta)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_value": 1.0}

    rng = np.random.RandomState(seed)
    observed_mean = float(delta.mean())

    boot_means = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = delta[idx].mean()

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    p_value = float(np.mean(boot_means <= 0.0))

    return {
        "mean": observed_mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
    }


def cohens_d_paired(delta: np.ndarray) -> float:
    n = len(delta)
    if n < 2:
        return 0.0
    sd = float(delta.std(ddof=1))
    if sd < 1e-12:
        return 0.0 if abs(delta.mean()) < 1e-12 else float("inf")
    return float(delta.mean() / sd)


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = DEFAULT_FDR_ALPHA,
) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    q_values = [0.0] * m
    prev_q = 1.0
    for rank_from_end, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = m - rank_from_end
        q = min(p * m / rank, prev_q)
        q = min(q, 1.0)
        q_values[orig_idx] = q
        prev_q = q
    return q_values


def compute_head_ablation_stats(
    head_label: str,
    baseline_rep: Sequence[bool],
    ablation_rep: Sequence[bool],
    B: int = DEFAULT_BOOTSTRAP_B,
    seed: int = 42,
) -> Dict[str, Any]:
    delta = compute_paired_delta_rep(baseline_rep, ablation_rep)
    boot = bootstrap_mean_one_sided(delta, B=B, seed=seed)
    d = cohens_d_paired(delta)
    return {
        "head_label": head_label,
        "n_prompts": len(delta),
        "mean_delta_rep": boot["mean"],
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "p_value_raw": boot["p_value"],
        "cohens_d": d,
        "baseline_rep_rate": float(np.mean(baseline_rep)),
        "ablation_rep_rate": float(np.mean(ablation_rep)),
    }


def apply_fdr_to_stats(
    stats_list: List[Dict[str, Any]],
    alpha: float = DEFAULT_FDR_ALPHA,
) -> List[Dict[str, Any]]:
    p_values = [s["p_value_raw"] for s in stats_list]
    q_values = benjamini_hochberg(p_values, alpha=alpha)
    for s, q in zip(stats_list, q_values):
        s["q_value_bh"] = q
        s["significant_bh"] = q < alpha
    ranked = sorted(stats_list, key=lambda s: s["q_value_bh"])
    for rank, s in enumerate(ranked, start=1):
        s["rank_by_q"] = rank
    return stats_list


def recall_at_k(
    ranked_labels: Sequence[str],
    positive_labels: Sequence[str],
    k: int,
) -> float:
    pos_set = set(positive_labels)
    if not pos_set:
        return 0.0
    top_k = set(ranked_labels[:k])
    hits = len(pos_set & top_k)
    return hits / len(pos_set)
