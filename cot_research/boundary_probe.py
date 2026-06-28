from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .local_copy_temptation import (
    DEFAULT_K_VALUES,
    _decode_token,
    build_local_copy_cases,
    build_matched_control_token_ids,
    mean_logit_for_token_ids,
)
from .ov_circuit_analysis import extract_head_ov_components


STATE_ORDER = [
    "sharp_phrase_boundary",
    "semantic_wrong_tail_cot",
    "matched_local_copy_control",
]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_bool_mean(values: Sequence[Optional[bool]]) -> float:
    filtered = [1.0 if bool(item) else 0.0 for item in values if item is not None]
    if not filtered:
        return 0.0
    return round(sum(filtered) / len(filtered), 6)


def format_scale_label(scale: float) -> str:
    return f"scale_{float(scale):g}"


def parse_scale_values(text: str) -> List[float]:
    if not str(text).strip():
        return [1.2, 1.5]
    values: List[float] = []
    seen = set()
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if value <= 0.0:
            raise ValueError(f"All scale values must be positive, got {value}.")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("No valid scale values were parsed.")
    return values


def build_boundary_probe_cases(
    *,
    sharp_phrase_count: int,
    wrong_tail_count: int,
    control_count: int,
    cot_input_jsonl: str | Path,
    sharp_phrase_max_new_tokens: int,
    wrong_tail_max_new_tokens: int,
    control_max_new_tokens: int,
    seed: int,
) -> List[Dict[str, Any]]:
    base_phrase_cases = [
        case
        for case in build_local_copy_cases(
            token_count=0,
            phrase_count=max(int(sharp_phrase_count), int(control_count)),
            cot_count=0,
            cot_input_jsonl=cot_input_jsonl,
            token_max_new_tokens=0,
            phrase_max_new_tokens=max(int(sharp_phrase_max_new_tokens), int(control_max_new_tokens)),
            cot_max_new_tokens=0,
            seed=seed,
            variant="sharp_prev1",
        )
        if str(case.get("family") or "") == "phrase"
    ]
    wrong_tail_cases = [
        case
        for case in build_local_copy_cases(
            token_count=0,
            phrase_count=0,
            cot_count=wrong_tail_count,
            cot_input_jsonl=cot_input_jsonl,
            token_max_new_tokens=0,
            phrase_max_new_tokens=0,
            cot_max_new_tokens=wrong_tail_max_new_tokens,
            seed=seed + 37,
            variant="default",
        )
        if str(case.get("family") or "") == "cot"
    ]

    cases: List[Dict[str, Any]] = []

    for idx, case in enumerate(base_phrase_cases[: max(int(sharp_phrase_count), 0)]):
        derived = dict(case)
        metadata = dict(derived.get("metadata") or {})
        derived["family"] = "sharp_phrase_boundary"
        derived["state"] = "sharp_phrase_boundary"
        derived["pair_id"] = str(case.get("example_id") or f"phrase_pair_{idx:03d}")
        derived["example_id"] = f"{derived['pair_id']}_sharp"
        derived["max_new_tokens"] = int(sharp_phrase_max_new_tokens)
        metadata["probe_state"] = "sharp_phrase_boundary"
        derived["metadata"] = metadata
        cases.append(derived)

    for idx, case in enumerate(base_phrase_cases[: max(int(control_count), 0)]):
        derived = dict(case)
        metadata = dict(derived.get("metadata") or {})
        wrapper = str(metadata.get("wrapper") or "{tail}")
        matched_tail = str(metadata.get("matched_control_tail") or metadata.get("base_prefix") or "")
        derived["assistant_prefix"] = wrapper.format(tail=matched_tail)
        derived["family"] = "matched_local_copy_control"
        derived["state"] = "matched_local_copy_control"
        derived["pair_id"] = str(case.get("example_id") or f"phrase_pair_{idx:03d}")
        derived["example_id"] = f"{derived['pair_id']}_control"
        derived["max_new_tokens"] = int(control_max_new_tokens)
        metadata["probe_state"] = "matched_local_copy_control"
        derived["metadata"] = metadata
        cases.append(derived)

    for idx, case in enumerate(wrong_tail_cases[: max(int(wrong_tail_count), 0)]):
        derived = dict(case)
        metadata = dict(derived.get("metadata") or {})
        derived["family"] = "semantic_wrong_tail_cot"
        derived["state"] = "semantic_wrong_tail_cot"
        derived["pair_id"] = str(case.get("example_id") or f"wrong_tail_{idx:03d}")
        derived["example_id"] = f"{derived['pair_id']}_wrong_tail"
        derived["max_new_tokens"] = int(wrong_tail_max_new_tokens)
        metadata["probe_state"] = "semantic_wrong_tail_cot"
        derived["metadata"] = metadata
        cases.append(derived)

    return cases


class SingleHeadBoundaryCaptureHook:
    def __init__(self, attn_module: torch.nn.Module, target, scale: float) -> None:
        self.attn_module = attn_module
        self.target = target
        self.scale = float(scale)
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.captured_before: Optional[torch.Tensor] = None
        self.captured_after: Optional[torch.Tensor] = None

    def __enter__(self) -> "SingleHeadBoundaryCaptureHook":
        if not hasattr(self.attn_module, "o_proj"):
            raise ValueError("Attention module has no o_proj; cannot attach prompt-boundary capture hook.")
        self.handle = self.attn_module.o_proj.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _pre_hook(self, module: torch.nn.Module, args: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
        if not args:
            return None
        x = args[0]
        if not isinstance(x, torch.Tensor):
            return None
        start = int(self.target.head_idx * self.target.head_dim)
        end = int((self.target.head_idx + 1) * self.target.head_dim)
        x_out = x.clone()
        self.captured_before = x[..., start:end].detach()[0, -1].float().cpu()
        x_out[..., start:end] = x_out[..., start:end] * self.scale
        self.captured_after = x_out[..., start:end].detach()[0, -1].float().cpu()
        if len(args) == 1:
            return (x_out,)
        return (x_out, *args[1:])


@torch.no_grad()
def forward_prompt_with_capture(
    model: torch.nn.Module,
    prompt_token_ids: Sequence[int],
    *,
    attn_module: torch.nn.Module,
    target,
    scale: float,
    model_device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    input_ids = torch.tensor([list(prompt_token_ids)], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    with SingleHeadBoundaryCaptureHook(attn_module, target, scale) as hook:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    debug = {
        "capture_scale": float(scale),
        "captured_head_before_abs_mean": None
        if hook.captured_before is None
        else float(hook.captured_before.abs().mean().item()),
        "captured_head_after_abs_mean": None
        if hook.captured_after is None
        else float(hook.captured_after.abs().mean().item()),
        "captured_head_before": None
        if hook.captured_before is None
        else [float(value) for value in hook.captured_before.tolist()],
        "captured_head_after": None
        if hook.captured_after is None
        else [float(value) for value in hook.captured_after.tolist()],
    }
    return outputs.logits[0, -1].detach().float().cpu(), debug


def top_k_from_logits(logits_row: torch.Tensor, tokenizer, *, k: int) -> List[Dict[str, Any]]:
    probs = torch.softmax(logits_row.float(), dim=-1)
    top_values, top_indices = torch.topk(logits_row.float(), k=min(max(int(k), 1), int(logits_row.shape[-1])))
    rows: List[Dict[str, Any]] = []
    for rank_idx, (logit, token_id) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
        rows.append(
            {
                "rank": int(rank_idx),
                "token_id": int(token_id),
                "token_text": _decode_token(tokenizer, int(token_id)),
                "logit": float(logit),
                "prob": float(probs[int(token_id)].item()),
            }
        )
    return rows


def mean_prob_for_token_ids(
    logits_row: torch.Tensor,
    token_ids: Sequence[int],
    *,
    log_norm: Optional[torch.Tensor] = None,
) -> float:
    if not token_ids:
        return 0.0
    if log_norm is None:
        log_norm = torch.logsumexp(logits_row.float(), dim=-1)
    return _mean(float(torch.exp(logits_row[int(token_id)].float() - log_norm).item()) for token_id in token_ids)


def build_target_control_token_ids(
    *,
    prompt_token_ids: Sequence[int],
    target_token_id: int,
    vocab_size: int,
    control_sample_size: int,
    seed: int,
) -> List[int]:
    return build_matched_control_token_ids(
        prefix_token_ids=prompt_token_ids,
        recent_raw_token_ids=[int(target_token_id)],
        recent_token_ids=[int(target_token_id)],
        sample_size=max(int(control_sample_size), 1),
        vocab_size=vocab_size,
        seed=seed,
    )


def build_direct_write_components(model: torch.nn.Module, *, layer_idx: int, head_idx: int) -> Dict[str, Any]:
    ov = extract_head_ov_components(model, layer_idx=layer_idx, head_idx=head_idx)
    return {
        "o_proj_slice": ov["o_proj_slice"].detach().float().cpu(),
        "lm_head_weight": ov["lm_head_weight"].detach().float().cpu(),
    }


def compute_direct_write_stats(
    *,
    components: Dict[str, Any],
    head_vector_after: Sequence[float] | None,
    target_token_id: int,
    control_token_ids: Sequence[int],
) -> Dict[str, Optional[float]]:
    if head_vector_after is None:
        return {
            "direct_write_target_logit": None,
            "direct_write_control_logit_mean": None,
            "direct_write_target_minus_control_logit": None,
        }
    o_proj_slice = components["o_proj_slice"]
    lm_head_weight = components["lm_head_weight"]
    head_vec = torch.tensor(list(head_vector_after), dtype=torch.float32)
    head_residual = torch.matmul(o_proj_slice, head_vec)
    target_logit = float(torch.dot(lm_head_weight[int(target_token_id)], head_residual).item())
    control_logits = [
        float(torch.dot(lm_head_weight[int(token_id)], head_residual).item()) for token_id in control_token_ids
    ]
    control_mean = None if not control_logits else float(_mean(control_logits))
    return {
        "direct_write_target_logit": float(target_logit),
        "direct_write_control_logit_mean": control_mean,
        "direct_write_target_minus_control_logit": None
        if control_mean is None
        else float(target_logit - control_mean),
    }


def aggregate_boundary_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    k_values: Sequence[int],
    condition_order: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("family") or "")
        condition = str(row.get("condition") or "")
        grouped[(family, condition)].append(dict(row))
        grouped[("overall", condition)].append(dict(row))

    summary_rows: List[Dict[str, Any]] = []
    ordered_families = ["overall"] + [family for family in STATE_ORDER if any(key[0] == family for key in grouped)]
    for family in ordered_families:
        for condition in condition_order:
            family_rows = grouped.get((family, condition), [])
            if not family_rows:
                continue
            summary: Dict[str, Any] = {
                "family": family,
                "condition": condition,
                "examples": int(len(family_rows)),
                "target_prob_mean": round(
                    _mean(float(row.get("target_prob", 0.0)) for row in family_rows if row.get("target_prob") is not None),
                    6,
                ),
                "target_logit_mean": round(
                    _mean(float(row.get("target_logit", 0.0)) for row in family_rows if row.get("target_logit") is not None),
                    6,
                ),
                "target_rank_mean": round(
                    _mean(float(row.get("target_rank", 0.0)) for row in family_rows if row.get("target_rank") is not None),
                    6,
                ),
                "target_prob_delta_mean": round(
                    _mean(float(row.get("target_prob_delta", 0.0)) for row in family_rows if row.get("target_prob_delta") is not None),
                    6,
                ),
                "target_logit_delta_mean": round(
                    _mean(float(row.get("target_logit_delta", 0.0)) for row in family_rows if row.get("target_logit_delta") is not None),
                    6,
                ),
                "target_rank_delta_mean": round(
                    _mean(float(row.get("target_rank_delta", 0.0)) for row in family_rows if row.get("target_rank_delta") is not None),
                    6,
                ),
                "target_control_prob_gap_mean": round(
                    _mean(
                        float(row.get("target_control_prob_gap", 0.0))
                        for row in family_rows
                        if row.get("target_control_prob_gap") is not None
                    ),
                    6,
                ),
                "target_control_logit_gap_mean": round(
                    _mean(
                        float(row.get("target_control_logit_gap", 0.0))
                        for row in family_rows
                        if row.get("target_control_logit_gap") is not None
                    ),
                    6,
                ),
                "target_control_logit_gap_delta_mean": round(
                    _mean(
                        float(row.get("target_control_logit_gap_delta", 0.0))
                        for row in family_rows
                        if row.get("target_control_logit_gap_delta") is not None
                    ),
                    6,
                ),
                "eos_prob_mean": round(
                    _mean(float(row.get("eos_prob", 0.0)) for row in family_rows if row.get("eos_prob") is not None),
                    6,
                ),
                "eos_rank_mean": round(
                    _mean(float(row.get("eos_rank", 0.0)) for row in family_rows if row.get("eos_rank") is not None),
                    6,
                ),
                "eos_prob_delta_mean": round(
                    _mean(float(row.get("eos_prob_delta", 0.0)) for row in family_rows if row.get("eos_prob_delta") is not None),
                    6,
                ),
                "top1_is_target_rate": _safe_bool_mean([row.get("top1_is_target") for row in family_rows]),
            }
            for k in k_values:
                key = str(int(k))
                summary[f"recent_mass_mean_k{key}"] = round(
                    _mean(float(row.get(f"recent_mass_k{key}", 0.0)) for row in family_rows),
                    6,
                )
                summary[f"recent_mass_delta_mean_k{key}"] = round(
                    _mean(
                        float(row.get(f"recent_mass_delta_k{key}", 0.0))
                        for row in family_rows
                        if row.get(f"recent_mass_delta_k{key}") is not None
                    ),
                    6,
                )
                summary[f"recent_gap_mean_k{key}"] = round(
                    _mean(float(row.get(f"recent_gap_k{key}", 0.0)) for row in family_rows),
                    6,
                )
                summary[f"recent_gap_delta_mean_k{key}"] = round(
                    _mean(
                        float(row.get(f"recent_gap_delta_k{key}", 0.0))
                        for row in family_rows
                        if row.get(f"recent_gap_delta_k{key}") is not None
                    ),
                    6,
                )
            summary_rows.append(summary)
    return summary_rows


def build_boundary_summary_json(
    *,
    summary_rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    condition_order: Sequence[str],
) -> Dict[str, Any]:
    nested: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for row in summary_rows:
        nested[str(row.get("family") or "")][str(row.get("condition") or "")] = dict(row)
    return {
        "k_values": [int(k) for k in k_values],
        "condition_order": list(condition_order),
        "families": nested,
        "family_order": ["overall"] + [family for family in STATE_ORDER if family in nested],
    }


def _write_simple_svg_line_chart(
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    x_labels: Sequence[str],
    series_rows: Dict[str, Sequence[float]],
) -> str:
    width = 760
    height = 440
    left = 72
    right = 24
    top = 48
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    flat_values = [float(value) for values in series_rows.values() for value in values]
    y_min = min(flat_values) if flat_values else 0.0
    y_max = max(flat_values) if flat_values else 1.0
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5
    palette = ["#1b6ca8", "#c44536", "#3a7d44", "#7b2cbf", "#f08c00"]

    def _escape(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def x_to_svg(idx: int) -> float:
        if len(x_labels) <= 1:
            return left + plot_w / 2.0
        return left + idx * (plot_w / max(len(x_labels) - 1, 1))

    def y_to_svg(value: float) -> float:
        ratio = (float(value) - y_min) / max(y_max - y_min, 1e-9)
        return top + plot_h * (1.0 - ratio)

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
        f'<text x="{width/2:.1f}" y="24" font-size="18" text-anchor="middle" font-family="monospace">{_escape(title)}</text>',
        f'<text x="20" y="{height/2:.1f}" font-size="12" text-anchor="middle" transform="rotate(-90 20 {height/2:.1f})" font-family="monospace">{_escape(ylabel)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#444" stroke-width="1.5" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#444" stroke-width="1.5" />',
    ]
    for tick_idx in range(5):
        tick_value = y_min + (y_max - y_min) * tick_idx / 4.0
        y = y_to_svg(tick_value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1" />')
        lines.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" font-size="11" text-anchor="end" font-family="monospace">{tick_value:.3f}</text>'
        )
    for idx, label in enumerate(x_labels):
        x = x_to_svg(idx)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#444" stroke-width="1" />')
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 20}" font-size="11" text-anchor="middle" font-family="monospace">{_escape(label)}</text>'
        )
    legend_x = left + 8
    legend_y = height - 18
    for idx, (label, values) in enumerate(series_rows.items()):
        color = palette[idx % len(palette)]
        points = " ".join(f"{x_to_svg(i):.2f},{y_to_svg(float(v)):.2f}" for i, v in enumerate(values))
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />')
        for i, value in enumerate(values):
            lines.append(f'<circle cx="{x_to_svg(i):.2f}" cy="{y_to_svg(float(value)):.2f}" r="3.5" fill="{color}" />')
        tx = legend_x + idx * 138
        lines.append(f'<rect x="{tx}" y="{legend_y - 10}" width="14" height="3" fill="{color}" />')
        lines.append(f'<text x="{tx + 20}" y="{legend_y - 2}" font-size="11" font-family="monospace">{_escape(label)}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def maybe_write_boundary_plots(
    summary_rows: Sequence[Dict[str, Any]],
    *,
    plots_dir: str | Path,
    k_values: Sequence[int],
    condition_order: Sequence[str],
) -> List[str]:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    rows_by_family: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        rows_by_family[str(row.get("family") or "")][str(row.get("condition") or "")] = dict(row)

    plot_paths: List[str] = []
    for family in STATE_ORDER:
        family_rows = rows_by_family.get(family, {})
        if not family_rows:
            continue
        recent_series: Dict[str, List[float]] = {}
        for condition in condition_order:
            row = family_rows.get(condition)
            if not row:
                continue
            recent_series[condition] = [float(row.get(f"recent_mass_mean_k{int(k)}", 0.0)) for k in k_values]
        if recent_series:
            plot_paths.append(
                _write_simple_svg_line_chart(
                    plots_dir / f"{family}_recent_mass.svg",
                    title=f"{family}: recent-set mass",
                    ylabel="Mean probability mass",
                    x_labels=[str(int(k)) for k in k_values],
                    series_rows=recent_series,
                )
            )

        target_prob_series: Dict[str, List[float]] = {}
        target_gap_series: Dict[str, List[float]] = {}
        for condition in condition_order:
            row = family_rows.get(condition)
            if not row:
                continue
            target_prob_series[condition] = [float(row.get("target_prob_mean", 0.0))]
            target_gap_series[condition] = [float(row.get("target_control_logit_gap_mean", 0.0))]
        if target_prob_series:
            plot_paths.append(
                _write_simple_svg_line_chart(
                    plots_dir / f"{family}_target_prob.svg",
                    title=f"{family}: semantic target prob",
                    ylabel="Mean probability",
                    x_labels=[family],
                    series_rows=target_prob_series,
                )
            )
        if target_gap_series:
            plot_paths.append(
                _write_simple_svg_line_chart(
                    plots_dir / f"{family}_target_control_gap.svg",
                    title=f"{family}: target-control logit gap",
                    ylabel="Mean logit gap",
                    x_labels=[family],
                    series_rows=target_gap_series,
                )
            )
    return plot_paths


def write_boundary_report(
    path: str | Path,
    *,
    args: Dict[str, Any],
    summary_rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    condition_order: Sequence[str],
) -> None:
    rows_by_family: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        rows_by_family[str(row.get("family") or "")][str(row.get("condition") or "")] = dict(row)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Boundary Head Probe\n\n")
        f.write(f"- model: `{args.get('model_name_or_path')}`\n")
        f.write(f"- head_label: `{args.get('head_label')}`\n")
        f.write(f"- scale_values: `{args.get('scale_values')}`\n")
        f.write(f"- k_values: `{','.join(str(int(k)) for k in k_values)}`\n")
        f.write(f"- sharp_phrase_count: `{args.get('sharp_phrase_count')}`\n")
        f.write(f"- wrong_tail_count: `{args.get('wrong_tail_count')}`\n")
        f.write(f"- control_count: `{args.get('control_count')}`\n\n")
        ordered_families = ["overall"] + [family for family in STATE_ORDER if family in rows_by_family]
        for family in ordered_families:
            family_rows = rows_by_family.get(family, {})
            if not family_rows:
                continue
            f.write(f"## {family}\n\n")
            for condition in condition_order:
                row = family_rows.get(condition)
                if not row:
                    continue
                parts = [
                    f"condition=`{condition}`",
                    f"examples={int(row.get('examples', 0))}",
                    f"target_prob_mean={float(row.get('target_prob_mean', 0.0)):.4f}",
                    f"target_rank_mean={float(row.get('target_rank_mean', 0.0)):.2f}",
                    f"target_control_logit_gap_mean={float(row.get('target_control_logit_gap_mean', 0.0)):.4f}",
                    f"eos_prob_mean={float(row.get('eos_prob_mean', 0.0)):.4f}",
                    f"recent_mass_k1={float(row.get('recent_mass_mean_k1', 0.0)):.4f}",
                    f"recent_mass_k8={float(row.get('recent_mass_mean_k8', 0.0)):.4f}",
                ]
                if condition != "baseline":
                    parts.append(f"target_prob_delta_mean={float(row.get('target_prob_delta_mean', 0.0)):.4f}")
                    parts.append(
                        f"target_control_logit_gap_delta_mean={float(row.get('target_control_logit_gap_delta_mean', 0.0)):.4f}"
                    )
                    parts.append(f"eos_prob_delta_mean={float(row.get('eos_prob_delta_mean', 0.0)):.4f}")
                f.write("- " + ", ".join(parts) + "\n")
            f.write("\n")


def flatten_boundary_row_for_csv(
    row: Dict[str, Any],
    *,
    k_values: Sequence[int],
) -> Dict[str, Any]:
    flat = dict(row)
    top_k = flat.pop("top_k", [])
    debug = flat.pop("debug", {})
    flat["top_k"] = json.dumps(top_k, ensure_ascii=False)
    flat["debug"] = json.dumps(debug, ensure_ascii=False)
    flat["control_token_ids"] = json.dumps(flat.get("control_token_ids") or [], ensure_ascii=False)
    flat["control_token_texts"] = json.dumps(flat.get("control_token_texts") or [], ensure_ascii=False)
    for k in k_values:
        key = str(int(k))
        prompt_metrics = dict((row.get("prompt_metrics_by_k") or {}).get(key) or {})
        flat[f"recent_mass_k{key}"] = float(flat.get(f"recent_mass_k{key}", prompt_metrics.get("recent_mass", 0.0)))
        flat[f"control_mass_k{key}"] = float(flat.get(f"control_mass_k{key}", prompt_metrics.get("control_mass", 0.0)))
        flat[f"recent_gap_k{key}"] = float(
            flat.get(f"recent_gap_k{key}", prompt_metrics.get("recent_minus_control_gap", 0.0))
        )
    flat.pop("prompt_metrics_by_k", None)
    return flat
