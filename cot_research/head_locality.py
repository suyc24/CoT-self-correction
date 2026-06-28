from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


WINDOWS = [1, 2, 4, 8, 16, 32, 64]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def classify_head_locality(
    row: Dict[str, Any],
    *,
    local_near_window: int = 16,
    local_mass_threshold: float = 0.8,
    self_mass_threshold: float = 0.5,
    prev_local_threshold: float = 0.5,
    prev_thresholds: Sequence[float] = (0.25, 0.5, 0.8),
    criterion: str = "near_total",
) -> Dict[str, Any]:
    out = dict(row)
    self_mass = float(row.get("mean_bucket_self", 0.0))
    near_prev_mass = float(row.get(f"mean_prev_mass_w{int(local_near_window)}", 0.0))
    near_total_mass = self_mass + near_prev_mass
    out["near_total_mass"] = near_total_mass
    out["far_mass_after_near_window"] = max(0.0, 1.0 - near_total_mass)

    criterion_name = str(criterion).strip().lower() or "near_total"
    if criterion_name == "prev_only":
        classification_mass = near_prev_mass
    elif criterion_name == "near_total":
        classification_mass = near_total_mass
    else:
        raise ValueError(f"Unsupported locality criterion: {criterion}")
    out["classification_criterion"] = criterion_name
    out["classification_mass"] = float(classification_mass)
    out["classification_far_mass"] = max(0.0, 1.0 - float(classification_mass))
    out["locality_label"] = "local" if float(classification_mass) >= float(local_mass_threshold) else "global"

    if self_mass >= float(self_mass_threshold):
        subtype = "self_local"
    elif float(row.get("mean_prev_mass_w8", 0.0)) >= float(prev_local_threshold):
        subtype = "recent_local"
    elif out["locality_label"] == "local":
        subtype = "mixed_local"
    else:
        subtype = "global"
    out["locality_subtype"] = subtype

    for threshold in prev_thresholds:
        key = f"prev_window_reaching_{str(threshold).replace('.', 'p')}"
        reached_window = None
        for window in WINDOWS:
            if float(row.get(f"mean_prev_mass_w{window}", 0.0)) >= float(threshold):
                reached_window = int(window)
                break
        out[key] = reached_window

    for threshold in prev_thresholds:
        key = f"near_window_reaching_{str(threshold).replace('.', 'p')}"
        reached_window = None
        for window in WINDOWS:
            total_mass = self_mass + float(row.get(f"mean_prev_mass_w{window}", 0.0))
            if total_mass >= float(threshold):
                reached_window = int(window)
                break
        out[key] = reached_window
    return out


def summarize_locality_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    prev_thresholds: Sequence[float] = (0.25, 0.5, 0.8),
) -> Dict[str, Any]:
    label_counts = Counter(str(row.get("locality_label") or "") for row in rows)
    subtype_counts = Counter(str(row.get("locality_subtype") or "") for row in rows)
    summary: Dict[str, Any] = {
        "head_count": int(len(rows)),
        "label_counts": dict(label_counts),
        "subtype_counts": dict(subtype_counts),
        "mean_self_mass": round(_mean(float(row.get("mean_bucket_self", 0.0)) for row in rows), 6),
        "mean_prev_mass_w8": round(_mean(float(row.get("mean_prev_mass_w8", 0.0)) for row in rows), 6),
        "mean_prev_mass_w16": round(_mean(float(row.get("mean_prev_mass_w16", 0.0)) for row in rows), 6),
        "mean_prev_mass_w32": round(_mean(float(row.get("mean_prev_mass_w32", 0.0)) for row in rows), 6),
        "mean_near_total_mass": round(_mean(float(row.get("near_total_mass", 0.0)) for row in rows), 6),
    }

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("locality_label") or "")].append(dict(row))

    for label, label_rows in by_label.items():
        summary[f"{label}_heads"] = int(len(label_rows))
        summary[f"{label}_mean_self_mass"] = round(
            _mean(float(row.get("mean_bucket_self", 0.0)) for row in label_rows),
            6,
        )
        summary[f"{label}_mean_prev_mass_w16"] = round(
            _mean(float(row.get("mean_prev_mass_w16", 0.0)) for row in label_rows),
            6,
        )
        summary[f"{label}_mean_near_total_mass"] = round(
            _mean(float(row.get("near_total_mass", 0.0)) for row in label_rows),
            6,
        )

    for threshold in prev_thresholds:
        key = f"prev_window_reaching_{str(threshold).replace('.', 'p')}"
        counts = Counter(str(row.get(key)) for row in rows)
        summary[f"{key}_counts"] = dict(counts)
    return summary


def build_locality_report(
    *,
    model_name_or_path: str,
    source_summary_csv: str | Path,
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    local_near_window: int,
    local_mass_threshold: float,
    self_mass_threshold: float,
    prev_local_threshold: float,
    criterion: str = "near_total",
) -> str:
    label_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label_groups[str(row.get("locality_label") or "")].append(dict(row))

    lines: List[str] = []
    lines.append("# Head Locality Classification")
    lines.append("")
    lines.append(f"- model: `{model_name_or_path}`")
    lines.append(f"- source_summary_csv: `{source_summary_csv}`")
    lines.append(f"- local_near_window: `{local_near_window}`")
    lines.append(f"- criterion: `{criterion}`")
    lines.append(f"- local_mass_threshold: `{local_mass_threshold}`")
    lines.append(f"- self_mass_threshold: `{self_mass_threshold}`")
    lines.append(f"- prev_local_threshold: `{prev_local_threshold}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- head_count: `{summary.get('head_count', 0)}`")
    lines.append(f"- label_counts: `{summary.get('label_counts', {})}`")
    lines.append(f"- subtype_counts: `{summary.get('subtype_counts', {})}`")
    lines.append("")

    for label in ["local", "global"]:
        label_rows = sorted(
            label_groups.get(label, []),
            key=lambda item: (
                -float(item.get("near_total_mass", 0.0)),
                -float(item.get("mean_prev_mass_w16", 0.0)),
                -float(item.get("mean_bucket_self", 0.0)),
            ),
        )
        if not label_rows:
            continue
        lines.append(f"## {label.capitalize()} Heads")
        lines.append("")
        for row in label_rows[:20]:
            lines.append(
                "- "
                f"{row['head_label']}: subtype={row.get('locality_subtype')}, "
                f"self={float(row.get('mean_bucket_self', 0.0)):.4f}, "
                f"prev_w8={float(row.get('mean_prev_mass_w8', 0.0)):.4f}, "
                f"prev_w16={float(row.get('mean_prev_mass_w16', 0.0)):.4f}, "
                f"near_total={float(row.get('near_total_mass', 0.0)):.4f}, "
                f"prev@0.5={row.get('prev_window_reaching_0p5')}, "
                f"near@0.8={row.get('near_window_reaching_0p8')}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
