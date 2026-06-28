#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.io_utils import dump_jsonl, load_jsonl, write_json

try:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.linear_model import LogisticRegression, RidgeClassifier

    SKLEARN_AVAILABLE = True
except Exception:
    np = None  # type: ignore[assignment]
    AgglomerativeClustering = None  # type: ignore[assignment]
    KMeans = None  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    RidgeClassifier = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


DEFAULT_MARKER_KEYWORDS = [
    "wait",
    "actually",
    "however",
    "but",
    "check",
    "hold on",
    "let me check",
    "mistake",
    "incorrect",
    "wrong",
    "不对",
    "重新检查",
    "重新",
    "重算",
    "等等",
    "等一下",
]

BOOLEAN_FIELDS = [
    "surface_reflection",
    "has_reflection",
    "explicit_error_ack",
    "recompute_or_revise",
    "semantic_repair",
    "functional_repair",
    "silent_answer_correction",
    "final_matches_wrong",
    "new_answer_not_A_not_B",
    "reject_B",
    "uses_restored_anchor",
    "hit_max_new_tokens",
    "cap",
]

BEHAVIOR_NUMERIC_FIELDS = [
    "generated_tokens",
    "p0_reflect_vs_stop",
    "p0_marker_logit_margin",
    "p0_finalization_logit_margin",
]

LOGIT_NUMERIC_FIELDS = [
    "reflect_logsum",
    "stop_logsum",
    "reflect_vs_stop",
    "marker_logsum",
    "marker_vs_stop",
    "finalize_logsum",
    "finalize_vs_reflect",
    "wait_logsum",
    "check_logsum",
    "actually_logsum",
    "newline_logsum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract reflection events from saved hidden trajectory traces, then run "
            "natural-vs-forced probes and light clustering."
        )
    )
    parser.add_argument(
        "--root",
        nargs="+",
        required=True,
        help="One or more experiment roots containing activation_traces/*.pt and behavior_rows.jsonl.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", default="19-22")
    parser.add_argument("--sites", default="post_attn,block_output")
    parser.add_argument("--primary_layer", type=int, default=22)
    parser.add_argument("--primary_site", default="post_attn")
    parser.add_argument("--feature_kinds", default="h_pre,h_marker,h_post,delta_marker,delta_post")
    parser.add_argument("--cluster_feature_kinds", default="h_pre,h_marker,delta_marker,delta_post")
    parser.add_argument("--marker_keywords", default=",".join(DEFAULT_MARKER_KEYWORDS))
    parser.add_argument("--max_activation_files", type=int, default=0)
    parser.add_argument("--train_fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--permutation_primary_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pca_components", type=int, default=50)
    parser.add_argument("--cluster_ks", default="2-8")
    parser.add_argument("--kmeans_n_init", type=int, default=10)
    parser.add_argument("--kmeans_max_iter", type=int, default=100)
    parser.add_argument("--bootstrap_repeats", type=int, default=20)
    parser.add_argument("--bootstrap_fraction", type=float, default=0.8)
    parser.add_argument("--silhouette_max_points", type=int, default=500)
    parser.add_argument(
        "--min_task_examples",
        type=int,
        default=8,
        help="Minimum examples per binary task after intersecting with a feature set.",
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_int_ranges(text: str) -> List[int]:
    out: List[int] = []
    for item in parse_csv_list(text):
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            values = range(start, end + step, step)
        else:
            values = [int(item)]
        for value in values:
            if value not in out:
                out.append(value)
    return out


def parse_ks(text: str) -> List[int]:
    values = parse_int_ranges(text)
    return [k for k in values if k >= 2]


def parse_activation_key(key: str) -> Tuple[int, str]:
    layer_text, site = str(key).split("/", 1)
    return int(layer_text.lstrip("L")), site


def iter_activation_paths(root: Path) -> List[Path]:
    return sorted(root.glob("**/activation_traces/*.pt"))


def iter_jsonl(root: Path, name: str) -> List[Path]:
    return sorted(root.glob(f"**/{name}"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[Any]) -> float:
    xs: List[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            xs.append(x)
    return sum(xs) / len(xs) if xs else float("nan")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def tensor_normed(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vec = vec.detach().float().cpu()
    norm = float(vec.norm().item())
    if norm < eps:
        return torch.zeros_like(vec)
    return vec / norm


def load_behavior_and_logits(
    roots: Sequence[Path],
) -> Tuple[Dict[Tuple[str, str, str, str], Dict[str, Any]], Dict[Tuple[str, str, str, str, int], Dict[str, Any]], List[Dict[str, Any]]]:
    behavior_lookup: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    logit_lookup: Dict[Tuple[str, str, str, str, int], Dict[str, Any]] = {}
    all_behavior: List[Dict[str, Any]] = []
    for root in roots:
        root_label = str(root)
        for path in iter_jsonl(root, "behavior_rows.jsonl"):
            for row in load_jsonl(path):
                mode = str(row.get("mode", "free"))
                key = (root_label, str(row.get("example_id")), mode, str(row.get("condition")))
                behavior_lookup[key] = row
                all_behavior.append({"root": root_label, **row})
        for path in iter_jsonl(root, "logit_rows.jsonl"):
            for row in load_jsonl(path):
                pos = int(row.get("position_index", -10**9))
                mode = str(row.get("mode", "free"))
                key = (root_label, str(row.get("example_id")), mode, str(row.get("condition")), pos)
                logit_lookup[key] = row
    return behavior_lookup, logit_lookup, all_behavior


def keyword_hit(text: str, keywords: Sequence[str]) -> Optional[str]:
    lower = text.lower()
    for keyword in keywords:
        kw = str(keyword).strip().lower()
        if not kw:
            continue
        if any(ord(ch) > 127 for ch in kw) or " " in kw:
            if kw in lower:
                return keyword
            continue
        pattern = r"(?<![a-z0-9_])" + re.escape(kw) + r"(?![a-z0-9_])"
        if re.search(pattern, lower):
            return keyword
    return None


def marker_group(keyword: str, token_text: str) -> str:
    text = f"{keyword} {token_text}".lower()
    if "wait" in text or "等" in text:
        return "wait"
    if "actually" in text or "不对" in text:
        return "actually_or_wrong"
    if "however" in text or "but" in text:
        return "contrast"
    if "check" in text or "重新" in text or "重算" in text:
        return "check_or_recompute"
    if "mistake" in text or "incorrect" in text or "wrong" in text:
        return "error_ack"
    return "other_marker"


def find_first_marker_position(
    position_records: Sequence[Mapping[str, Any]],
    keywords: Sequence[str],
) -> Tuple[Optional[int], str, str]:
    cumulative = ""
    for rec in sorted(position_records, key=lambda row: int(row.get("position_index", 0))):
        pos = int(rec.get("position_index", 0))
        if pos <= 0:
            continue
        token_text = str(rec.get("token_text") or "")
        cumulative += token_text
        hit = keyword_hit(cumulative, keywords)
        if hit is not None:
            return pos, hit, token_text
    return None, "", ""


def is_forced_marker_condition(condition: str, behavior: Mapping[str, Any]) -> bool:
    lower = str(condition).lower()
    forced_prefix = str(behavior.get("forced_prefix_name", "")).lower()
    return "force_wait" in lower or forced_prefix == "wait"


def choose_event(
    *,
    condition: str,
    run: Mapping[str, Any],
    behavior: Mapping[str, Any],
    keywords: Sequence[str],
) -> Optional[Dict[str, Any]]:
    position_records = list(run.get("position_records") or [])
    generated_ids = list(run.get("generated_token_ids") or [])
    if not position_records and not generated_ids:
        return None
    generated_count = len(generated_ids)
    has_behavior = bool(behavior)
    has_reflection = bool_value(behavior.get("surface_reflection")) or bool_value(behavior.get("has_reflection"))
    silent = bool_value(behavior.get("silent_answer_correction"))
    forced_marker = is_forced_marker_condition(condition, behavior)
    marker_pos, keyword, token_text = find_first_marker_position(position_records, keywords)

    if forced_marker:
        event_type = "forced_marker"
        event_pos = marker_pos if marker_pos is not None else (1 if generated_count > 0 else 0)
        keyword = keyword or "wait"
        token_text = token_text or first_generated_token_text(position_records, behavior)
    elif marker_pos is not None and (has_reflection or not has_behavior):
        event_type = "natural_marker"
        event_pos = marker_pos
    elif silent:
        event_type = "silent_correction"
        event_pos = 1 if generated_count > 0 else 0
        keyword = ""
        token_text = first_generated_token_text(position_records, behavior)
    elif generated_count > 0 or position_records:
        event_type = "nonreflection_termination"
        event_pos = 1 if generated_count > 0 else 0
        keyword = ""
        token_text = first_generated_token_text(position_records, behavior)
    else:
        return None

    event_pos = max(int(event_pos), 0)
    return {
        "event_type": event_type,
        "marker_position_index": event_pos,
        "pre_position_index": max(event_pos - 1, 0),
        "post_position_index": event_pos + 1,
        "marker_keyword": str(keyword or ""),
        "marker_token_text": str(token_text or ""),
        "marker_group": marker_group(str(keyword or ""), str(token_text or "")) if event_type in {"natural_marker", "forced_marker"} else "none",
    }


def first_generated_token_text(position_records: Sequence[Mapping[str, Any]], behavior: Mapping[str, Any]) -> str:
    if behavior.get("first_generated_token_text") is not None:
        return str(behavior.get("first_generated_token_text"))
    for rec in position_records:
        if int(rec.get("position_index", 0)) == 1:
            return str(rec.get("token_text") or "")
    return ""


def feature_vectors(
    tensor: torch.Tensor,
    *,
    pre_pos: int,
    marker_pos: int,
    post_pos: int,
    feature_kinds: Sequence[str],
) -> Dict[str, torch.Tensor]:
    tensor = tensor.detach().float().cpu()
    n_pos = int(tensor.shape[0])
    vecs: Dict[str, torch.Tensor] = {}
    h_pre = tensor[int(pre_pos)] if 0 <= int(pre_pos) < n_pos else None
    h_marker = tensor[int(marker_pos)] if 0 <= int(marker_pos) < n_pos else None
    h_post = tensor[int(post_pos)] if 0 <= int(post_pos) < n_pos else None
    for kind in feature_kinds:
        if kind == "h_pre" and h_pre is not None:
            vecs[kind] = h_pre
        elif kind == "h_marker" and h_marker is not None:
            vecs[kind] = h_marker
        elif kind == "h_post" and h_post is not None:
            vecs[kind] = h_post
        elif kind == "delta_marker" and h_pre is not None and h_marker is not None:
            vecs[kind] = h_marker - h_pre
        elif kind == "delta_post" and h_marker is not None and h_post is not None:
            vecs[kind] = h_post - h_marker
    return vecs


def event_behavior_fields(behavior: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in BOOLEAN_FIELDS:
        out[field] = bool_value(behavior.get(field))
    for field in BEHAVIOR_NUMERIC_FIELDS:
        value = behavior.get(field)
        if value is not None:
            out[field] = finite_float(value)
    if "generated_tokens" not in out:
        out["generated_tokens"] = 0.0
    for key in [
        "condition_kind",
        "intervention_type",
        "gate_sign",
        "gate_mode",
        "forced_prefix_name",
        "patch_type",
        "patch_layer",
        "patch_site",
        "patch_timing",
        "base_condition",
        "source_condition",
    ]:
        if behavior.get(key) is not None:
            out[key] = behavior.get(key)
    return out


def add_logit_fields(
    row: Dict[str, Any],
    logit_lookup: Mapping[Tuple[str, str, str, str, int], Dict[str, Any]],
    *,
    root_label: str,
    example_id: str,
    mode: str,
    condition: str,
    marker_position_index: int,
) -> None:
    # Logit row p predicts generated token p+1, so the decision for marker p is row p-1.
    decision_pos = max(int(marker_position_index) - 1, 0)
    logit_row = logit_lookup.get((root_label, example_id, mode, condition, decision_pos), {})
    for field in LOGIT_NUMERIC_FIELDS:
        if field in logit_row:
            row[f"decision_{field}"] = finite_float(logit_row.get(field))


def extract_events(
    *,
    roots: Sequence[Path],
    selected_layers: Sequence[int],
    selected_sites: Sequence[str],
    feature_kinds: Sequence[str],
    keywords: Sequence[str],
    max_activation_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    behavior_lookup, logit_lookup, behavior_rows = load_behavior_and_logits(roots)
    activation_paths: List[Tuple[str, Path]] = []
    for root in roots:
        for path in iter_activation_paths(root):
            activation_paths.append((str(root), path))
    activation_paths.sort(key=lambda item: (item[0], str(item[1])))
    if int(max_activation_files) > 0:
        activation_paths = activation_paths[: int(max_activation_files)]

    events: List[Dict[str, Any]] = []
    feature_sets: Dict[str, Dict[str, Any]] = {}
    stats: Dict[str, Any] = {
        "roots": [str(root) for root in roots],
        "behavior_rows": len(behavior_rows),
        "activation_files_seen": len(activation_paths),
        "bad_activation_files": 0,
        "events": 0,
    }

    selected_layer_set = {int(x) for x in selected_layers}
    selected_site_set = {str(x) for x in selected_sites}

    for root_label, path in activation_paths:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            stats["bad_activation_files"] = int(stats.get("bad_activation_files", 0)) + 1
            stats.setdefault("bad_activation_errors", []).append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        example_id = str(payload.get("example_id"))
        global_idx = payload.get("global_idx")
        runs = payload.get("runs") or {}
        for mode, mode_runs in runs.items():
            if not isinstance(mode_runs, dict):
                continue
            for condition, run in mode_runs.items():
                if not isinstance(run, dict):
                    continue
                behavior = behavior_lookup.get((root_label, example_id, str(mode), str(condition)), {})
                event = choose_event(condition=str(condition), run=run, behavior=behavior, keywords=keywords)
                if event is None:
                    continue
                event_idx = len(events)
                row: Dict[str, Any] = {
                    "event_idx": event_idx,
                    "event_id": f"{Path(root_label).name}:{path.stem}:{mode}:{condition}:{event['event_type']}",
                    "root": root_label,
                    "activation_path": str(path),
                    "example_id": example_id,
                    "global_idx": global_idx,
                    "mode": str(mode),
                    "condition": str(condition),
                    "correct_answer": payload.get("correct_answer"),
                    "wrong_answer": payload.get("wrong_answer"),
                    **event,
                    **event_behavior_fields(behavior),
                }
                add_logit_fields(
                    row,
                    logit_lookup,
                    root_label=root_label,
                    example_id=example_id,
                    mode=str(mode),
                    condition=str(condition),
                    marker_position_index=int(event["marker_position_index"]),
                )
                events.append(row)

                acts = run.get("activations") or {}
                available = 0
                for act_key, tensor in acts.items():
                    layer_idx, site = parse_activation_key(act_key)
                    if int(layer_idx) not in selected_layer_set or str(site) not in selected_site_set:
                        continue
                    vecs = feature_vectors(
                        tensor,
                        pre_pos=int(event["pre_position_index"]),
                        marker_pos=int(event["marker_position_index"]),
                        post_pos=int(event["post_position_index"]),
                        feature_kinds=feature_kinds,
                    )
                    for kind, vec in vecs.items():
                        key = f"L{int(layer_idx)}/{site}/{kind}"
                        item = feature_sets.setdefault(
                            key,
                            {
                                "layer_idx": int(layer_idx),
                                "site": str(site),
                                "feature_kind": str(kind),
                                "event_indices": [],
                                "values": [],
                            },
                        )
                        item["event_indices"].append(int(event_idx))
                        item["values"].append(vec.detach().float().cpu())
                        available += 1
                row["available_feature_count"] = available

    for item in feature_sets.values():
        values = item["values"]
        item["event_indices"] = torch.tensor(item["event_indices"], dtype=torch.long)
        item["values"] = torch.stack(values, dim=0) if values else torch.empty((0, 0), dtype=torch.float32)
    stats["events"] = len(events)
    stats["feature_sets"] = len(feature_sets)
    stats["event_type_counts"] = dict(Counter(str(row.get("event_type")) for row in events))
    return events, feature_sets, stats


def grouped_train_test_split(
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    seed: int,
    train_fraction: float,
    attempts: int = 200,
) -> Optional[Tuple[List[int], List[int]]]:
    n = len(labels)
    if n < 4 or len(set(labels)) < 2:
        return None
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        return None
    rng = random.Random(seed)
    best: Optional[Tuple[List[int], List[int]]] = None
    for _ in range(attempts):
        shuffled = list(unique_groups)
        rng.shuffle(shuffled)
        n_train_groups = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(train_fraction)))))
        train_groups = set(shuffled[:n_train_groups])
        train_idx = [idx for idx, group in enumerate(groups) if group in train_groups]
        test_idx = [idx for idx, group in enumerate(groups) if group not in train_groups]
        if not train_idx or not test_idx:
            continue
        if len({labels[idx] for idx in train_idx}) < 2 or len({labels[idx] for idx in test_idx}) < 2:
            continue
        best = (train_idx, test_idx)
        break
    return best


def standardize_train_test(X: torch.Tensor, train_idx: Sequence[int], test_idx: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    X_train = X[torch.tensor(list(train_idx), dtype=torch.long)]
    X_test = X[torch.tensor(list(test_idx), dtype=torch.long)]
    mean_vec = X_train.mean(dim=0, keepdim=True)
    std_vec = X_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (X_train - mean_vec) / std_vec, (X_test - mean_vec) / std_vec


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [score for label, score in zip(labels, scores) if int(label) == 1]
    neg = [score for label, score in zip(labels, scores) if int(label) == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / max(total, 1)


def classification_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    preds = [1 if score >= 0 else 0 for score in scores]
    labels = [int(x) for x in labels]
    correct = sum(1 for y, pred in zip(labels, preds) if y == pred)
    tp = sum(1 for y, pred in zip(labels, preds) if y == 1 and pred == 1)
    fp = sum(1 for y, pred in zip(labels, preds) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(labels, preds) if y == 1 and pred == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": correct / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": binary_auc(labels, scores),
    }


def centroid_scores(X_train: torch.Tensor, y_train: Sequence[int], X_test: torch.Tensor) -> torch.Tensor:
    y = torch.tensor([int(v) for v in y_train], dtype=torch.long)
    pos = X_train[y == 1]
    neg = X_train[y == 0]
    if int(pos.shape[0]) == 0 or int(neg.shape[0]) == 0:
        return torch.zeros((int(X_test.shape[0]),), dtype=torch.float32)
    pos_mean = pos.mean(dim=0)
    neg_mean = neg.mean(dim=0)
    unit = tensor_normed(pos_mean - neg_mean)
    threshold = float(torch.dot(unit, (pos_mean + neg_mean) / 2.0).item())
    return X_test.float().matmul(unit.float()) - threshold


def balanced_class_weights(labels: Sequence[int]) -> torch.Tensor:
    y = torch.tensor([int(v) for v in labels], dtype=torch.float32)
    pos = float((y == 1).sum().item())
    neg = float((y == 0).sum().item())
    weights = torch.ones_like(y)
    if pos > 0:
        weights[y == 1] = float(y.numel()) / (2.0 * pos)
    if neg > 0:
        weights[y == 0] = float(y.numel()) / (2.0 * neg)
    return weights


def append_bias(X: torch.Tensor) -> torch.Tensor:
    return torch.cat([X.float(), torch.ones((int(X.shape[0]), 1), dtype=torch.float32)], dim=1)


def torch_ridge_scores(
    X_train: torch.Tensor,
    y_train: Sequence[int],
    X_test: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> torch.Tensor:
    y = torch.tensor([1.0 if int(v) == 1 else -1.0 for v in y_train], dtype=torch.float32)
    weights = balanced_class_weights(y_train)
    Xb = append_bias(X_train)
    Xtb = append_bias(X_test)
    sqrt_w = weights.clamp_min(1e-8).sqrt().unsqueeze(1)
    Xw = Xb * sqrt_w
    yw = y * sqrt_w.squeeze(1)
    reg = torch.eye(int(Xb.shape[1]), dtype=torch.float32) * float(alpha)
    reg[-1, -1] = 0.0
    lhs = Xw.T.matmul(Xw) + reg
    rhs = Xw.T.matmul(yw)
    try:
        coef = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coef = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution.squeeze(1)
    return Xtb.matmul(coef)


def torch_logistic_scores(
    X_train: torch.Tensor,
    y_train: Sequence[int],
    X_test: torch.Tensor,
    *,
    seed: int,
    lr: float = 0.05,
    epochs: int = 400,
    l2: float = 1e-3,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    Xb = append_bias(X_train)
    Xtb = append_bias(X_test)
    y = torch.tensor([float(int(v)) for v in y_train], dtype=torch.float32)
    weights = balanced_class_weights(y_train)
    coef = torch.zeros((int(Xb.shape[1]),), dtype=torch.float32)
    coef[:-1] = torch.randn((int(Xb.shape[1]) - 1,), generator=gen, dtype=torch.float32) * 1e-3
    for _ in range(max(1, int(epochs))):
        logits = Xb.matmul(coef)
        probs = torch.sigmoid(logits)
        grad = Xb.T.matmul((probs - y) * weights) / max(float(Xb.shape[0]), 1.0)
        grad[:-1] += float(l2) * coef[:-1]
        coef -= float(lr) * grad
    return Xtb.matmul(coef)


def fit_predict_scores(
    X_train: torch.Tensor,
    y_train: Sequence[int],
    X_test: torch.Tensor,
    *,
    method: str,
    seed: int,
) -> Optional[List[float]]:
    method = str(method)
    if method == "centroid":
        return [float(x) for x in centroid_scores(X_train, y_train, X_test).tolist()]
    if method == "logistic_torch":
        return [float(x) for x in torch_logistic_scores(X_train, y_train, X_test, seed=seed).tolist()]
    if method == "ridge_torch":
        return [float(x) for x in torch_ridge_scores(X_train, y_train, X_test).tolist()]
    if not SKLEARN_AVAILABLE:
        return None
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    y_train_np = np.asarray([int(x) for x in y_train])  # type: ignore[union-attr]
    if method == "logistic":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=int(seed))
        clf.fit(X_train_np, y_train_np)
        return [float(x) for x in clf.decision_function(X_test_np).tolist()]
    if method == "ridge":
        clf = RidgeClassifier(class_weight="balanced")
        clf.fit(X_train_np, y_train_np)
        return [float(x) for x in clf.decision_function(X_test_np).tolist()]
    return None


def run_binary_classifier(
    X: torch.Tensor,
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    method: str,
    seed: int,
    train_fraction: float,
    permutations: int,
) -> Dict[str, Any]:
    split = grouped_train_test_split(labels, groups, seed=seed, train_fraction=train_fraction)
    if split is None:
        return {"status": "skipped_split", "method": method}
    train_idx, test_idx = split
    X_train, X_test = standardize_train_test(X.float(), train_idx, test_idx)
    y_train = [int(labels[idx]) for idx in train_idx]
    y_test = [int(labels[idx]) for idx in test_idx]
    scores = fit_predict_scores(X_train, y_train, X_test, method=method, seed=seed)
    if scores is None:
        return {"status": "skipped_method", "method": method}
    metrics = classification_metrics(y_test, scores)
    out: Dict[str, Any] = {
        "status": "ok",
        "method": method,
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "train_positive": sum(y_train),
        "test_positive": sum(y_test),
        "group_leakage": bool(set(groups[idx] for idx in train_idx) & set(groups[idx] for idx in test_idx)),
        **metrics,
    }
    if int(permutations) > 0 and math.isfinite(metrics.get("auc", float("nan"))):
        rng = random.Random(seed + 99173)
        valid = 0
        ge_observed = 0
        observed = float(metrics["auc"])
        labels_list = [int(x) for x in labels]
        for _ in range(int(permutations)):
            shuffled = list(labels_list)
            rng.shuffle(shuffled)
            y_train_perm = [int(shuffled[idx]) for idx in train_idx]
            y_test_perm = [int(shuffled[idx]) for idx in test_idx]
            if len(set(y_train_perm)) < 2 or len(set(y_test_perm)) < 2:
                continue
            perm_scores = fit_predict_scores(X_train, y_train_perm, X_test, method=method, seed=seed)
            if perm_scores is None:
                continue
            perm_auc = binary_auc(y_test_perm, perm_scores)
            if not math.isfinite(perm_auc):
                continue
            valid += 1
            if perm_auc >= observed:
                ge_observed += 1
        out["permutation_count"] = valid
        out["permutation_p_value"] = (ge_observed + 1) / (valid + 1) if valid else float("nan")
    return out


def metadata_features(events: Sequence[Mapping[str, Any]], indices: Sequence[int], kind: str) -> torch.Tensor:
    rows = [events[idx] for idx in indices]
    if kind == "token":
        groups = sorted(set(str(row.get("marker_group", "none")) for row in rows))
        group_to_idx = {group: idx for idx, group in enumerate(groups)}
        X = torch.zeros((len(rows), len(groups) + 2), dtype=torch.float32)
        for i, row in enumerate(rows):
            X[i, group_to_idx[str(row.get("marker_group", "none"))]] = 1.0
            X[i, len(groups)] = 1.0 if str(row.get("marker_token_text", "")).strip() else 0.0
            X[i, len(groups) + 1] = float(len(str(row.get("marker_token_text", ""))))
        return X
    if kind == "token_norm":
        groups = sorted(set(str(row.get("marker_group", "none")) for row in rows))
        tokens = sorted(set(str(row.get("marker_token_text", "")).strip().lower() for row in rows))
        group_to_idx = {group: idx for idx, group in enumerate(groups)}
        token_to_idx = {token: idx for idx, token in enumerate(tokens)}
        X = torch.zeros((len(rows), len(groups) + len(tokens) + 1), dtype=torch.float32)
        for i, row in enumerate(rows):
            group = str(row.get("marker_group", "none"))
            token = str(row.get("marker_token_text", "")).strip().lower()
            X[i, group_to_idx[group]] = 1.0
            X[i, len(groups) + token_to_idx[token]] = 1.0
            X[i, len(groups) + len(tokens)] = float(len(token))
        return X
    if kind == "position":
        X = torch.zeros((len(rows), 3), dtype=torch.float32)
        for i, row in enumerate(rows):
            X[i, 0] = finite_float(row.get("marker_position_index"), 0.0)
            X[i, 1] = finite_float(row.get("pre_position_index"), 0.0)
            X[i, 2] = finite_float(row.get("generated_tokens"), 0.0)
        return X
    if kind == "logit":
        fields = [f"decision_{name}" for name in LOGIT_NUMERIC_FIELDS]
        behavior_fields = [
            "p0_reflect_vs_stop",
            "p0_marker_logit_margin",
            "p0_finalization_logit_margin",
        ]
        X = torch.zeros((len(rows), len(fields) + len(behavior_fields)), dtype=torch.float32)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                value = finite_float(row.get(field), 0.0)
                X[i, j] = 0.0 if not math.isfinite(value) else value
            for j, field in enumerate(behavior_fields, start=len(fields)):
                value = finite_float(row.get(field), 0.0)
                X[i, j] = 0.0 if not math.isfinite(value) else value
        return X
    raise ValueError(f"Unsupported metadata feature kind: {kind}")


def task_labels(events: Sequence[Mapping[str, Any]], task: str) -> Dict[int, int]:
    labels: Dict[int, int] = {}
    for idx, row in enumerate(events):
        event_type = str(row.get("event_type"))
        if task == "natural_vs_forced":
            if event_type == "natural_marker":
                labels[idx] = 1
            elif event_type == "forced_marker":
                labels[idx] = 0
        elif task == "natural_wait_vs_forced_wait":
            marker_group_name = str(row.get("marker_group", "none"))
            marker_token_norm = str(row.get("marker_token_text", "")).strip().lower()
            is_wait_like = marker_group_name == "wait" or marker_token_norm == "wait"
            if event_type == "natural_marker" and is_wait_like:
                labels[idx] = 1
            elif event_type == "forced_marker" and is_wait_like:
                labels[idx] = 0
        elif task == "marker_vs_nonreflection":
            if event_type in {"natural_marker", "forced_marker"}:
                labels[idx] = 1
            elif event_type == "nonreflection_termination":
                labels[idx] = 0
        elif task == "explicit_vs_silent":
            if event_type in {"natural_marker", "forced_marker"}:
                labels[idx] = 1
            elif event_type == "silent_correction":
                labels[idx] = 0
        else:
            raise ValueError(f"Unsupported task: {task}")
    return labels


def classifier_rows(
    *,
    events: Sequence[Mapping[str, Any]],
    feature_sets: Mapping[str, Mapping[str, Any]],
    primary_layer: int,
    primary_site: str,
    seed: int,
    train_fraction: float,
    min_task_examples: int,
    permutations: int,
    permutation_primary_only: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tasks = ["natural_vs_forced", "natural_wait_vs_forced_wait", "marker_vs_nonreflection", "explicit_vs_silent"]
    methods = ["logistic_torch", "ridge_torch", "centroid"]
    if SKLEARN_AVAILABLE:
        methods = ["logistic", "ridge", "centroid"]

    for task in tasks:
        label_lookup = task_labels(events, task)
        if len(label_lookup) < int(min_task_examples) or len(set(label_lookup.values())) < 2:
            rows.append({"task": task, "feature_source": "none", "status": "skipped_insufficient_labels", "count": len(label_lookup)})
            continue

        task_indices = sorted(label_lookup)
        task_labels_list = [label_lookup[idx] for idx in task_indices]
        task_groups = [str(events[idx].get("example_id")) for idx in task_indices]
        for baseline_kind in ["token", "token_norm", "position", "logit"]:
            X_base = metadata_features(events, task_indices, baseline_kind)
            for method in methods:
                result = run_binary_classifier(
                    X_base,
                    task_labels_list,
                    task_groups,
                    method=method,
                    seed=seed,
                    train_fraction=train_fraction,
                    permutations=permutations if task in {"natural_vs_forced", "natural_wait_vs_forced_wait"} else 0,
                )
                rows.append(
                    {
                        "task": task,
                        "feature_source": baseline_kind,
                        "layer_idx": "",
                        "site": "",
                        "feature_kind": "metadata",
                        "n_examples": len(task_indices),
                        "positive_count": sum(task_labels_list),
                        **result,
                    }
                )

        sorted_feature_items = sorted(
            feature_sets.items(),
            key=lambda kv: (
                0
                if int(kv[1]["layer_idx"]) == int(primary_layer)
                and str(kv[1]["site"]) == str(primary_site)
                else 1,
                str(kv[0]),
            ),
        )
        for feature_key, item in sorted_feature_items:
            event_indices = [int(x) for x in item["event_indices"].tolist()]
            idx_to_row = {event_idx: row_idx for row_idx, event_idx in enumerate(event_indices)}
            common = [event_idx for event_idx in task_indices if event_idx in idx_to_row]
            if len(common) < int(min_task_examples):
                continue
            labels = [label_lookup[idx] for idx in common]
            if len(set(labels)) < 2:
                continue
            X = item["values"][torch.tensor([idx_to_row[idx] for idx in common], dtype=torch.long)]
            groups = [str(events[idx].get("example_id")) for idx in common]
            is_primary = int(item["layer_idx"]) == int(primary_layer) and str(item["site"]) == str(primary_site)
            do_perm = (
                int(permutations)
                if task in {"natural_vs_forced", "natural_wait_vs_forced_wait"}
                and (is_primary or not permutation_primary_only)
                else 0
            )
            for method in methods:
                result = run_binary_classifier(
                    X,
                    labels,
                    groups,
                    method=method,
                    seed=seed,
                    train_fraction=train_fraction,
                    permutations=do_perm,
                )
                rows.append(
                    {
                        "task": task,
                        "feature_source": "hidden",
                        "feature_key": feature_key,
                        "layer_idx": int(item["layer_idx"]),
                        "site": str(item["site"]),
                        "feature_kind": str(item["feature_kind"]),
                        "n_examples": len(common),
                        "positive_count": sum(labels),
                        **result,
                    }
                )
    return rows


def standardize_all(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean_vec = X.float().mean(dim=0, keepdim=True)
    std_vec = X.float().std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (X.float() - mean_vec) / std_vec, mean_vec.squeeze(0), std_vec.squeeze(0)


def pca_reduce(X: torch.Tensor, max_components: int) -> Tuple[torch.Tensor, List[float]]:
    n, d = int(X.shape[0]), int(X.shape[1])
    if n < 2 or d < 1:
        return X.float(), []
    components = max(1, min(int(max_components), n - 1, d))
    centered = X.float() - X.float().mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    basis = Vh[:components].T
    reduced = centered.matmul(basis)
    denom = float((S**2).sum().item())
    explained = [float(((S[i] ** 2) / max(denom, 1e-12)).item()) for i in range(min(components, int(S.numel())))]
    return reduced, explained


def torch_kmeans(
    X: torch.Tensor,
    k: int,
    *,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[torch.Tensor, float]:
    n = int(X.shape[0])
    if k <= 1 or k > n:
        raise ValueError("k must be in [2, n].")
    best_labels: Optional[torch.Tensor] = None
    best_inertia = float("inf")
    for init_idx in range(max(1, int(n_init))):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + init_idx * 104729 + int(k))
        perm = torch.randperm(n, generator=gen)
        centers = X[perm[:k]].clone()
        labels = torch.zeros((n,), dtype=torch.long)
        for _ in range(max(1, int(max_iter))):
            distances = torch.cdist(X.float(), centers.float(), p=2.0)
            new_labels = distances.argmin(dim=1)
            if torch.equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for cluster_id in range(k):
                mask = labels == cluster_id
                if bool(mask.any()):
                    centers[cluster_id] = X[mask].mean(dim=0)
                else:
                    centers[cluster_id] = X[int(torch.randint(0, n, (1,), generator=gen).item())]
        inertia = float(((X - centers[labels]) ** 2).sum().item())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
    if best_labels is None:
        raise ValueError("k-means failed to initialize.")
    return best_labels, best_inertia


def cluster_labels(
    X: torch.Tensor,
    k: int,
    *,
    method: str,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[Optional[torch.Tensor], float]:
    if method == "kmeans_sklearn" and SKLEARN_AVAILABLE:
        model = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=int(seed), max_iter=int(max_iter))
        labels = model.fit_predict(X.numpy())
        return torch.tensor(labels, dtype=torch.long), float(model.inertia_)
    if method == "agglomerative_ward" and SKLEARN_AVAILABLE:
        model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
        labels = model.fit_predict(X.numpy())
        inertia = float("nan")
        return torch.tensor(labels, dtype=torch.long), inertia
    if method == "kmeans_torch":
        return torch_kmeans(X, int(k), seed=seed, n_init=n_init, max_iter=max_iter)
    return None, float("nan")


def silhouette_score_torch(X: torch.Tensor, labels: torch.Tensor, *, max_points: int, seed: int) -> float:
    n = int(X.shape[0])
    if n < 3 or len(set(int(x) for x in labels.tolist())) < 2:
        return float("nan")
    idx = list(range(n))
    if n > int(max_points):
        rng = random.Random(seed)
        idx = sorted(rng.sample(idx, int(max_points)))
    Xs = X[torch.tensor(idx, dtype=torch.long)]
    ys = labels[torch.tensor(idx, dtype=torch.long)]
    distances = torch.cdist(Xs.float(), Xs.float(), p=2.0)
    scores: List[float] = []
    clusters = sorted(set(int(x) for x in ys.tolist()))
    for i in range(int(Xs.shape[0])):
        own = int(ys[i].item())
        own_mask = ys == own
        other_mask = ~own_mask
        if int(own_mask.sum().item()) <= 1 or not bool(other_mask.any()):
            continue
        a = float(distances[i][own_mask].sum().item() / max(int(own_mask.sum().item()) - 1, 1))
        b = float("inf")
        for cluster_id in clusters:
            if cluster_id == own:
                continue
            mask = ys == cluster_id
            if bool(mask.any()):
                b = min(b, float(distances[i][mask].mean().item()))
        if math.isfinite(b):
            scores.append((b - a) / max(a, b, 1e-12))
    return sum(scores) / len(scores) if scores else float("nan")


def pair_agreement(labels_a: Sequence[int], labels_b: Sequence[int], *, max_pairs: int, seed: int) -> float:
    n = len(labels_a)
    if n < 2:
        return float("nan")
    pairs: List[Tuple[int, int]] = []
    if n * (n - 1) // 2 <= max_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:
        rng = random.Random(seed)
        seen = set()
        while len(pairs) < max_pairs:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (i, j) in seen:
                continue
            seen.add((i, j))
            pairs.append((i, j))
    same = 0
    for i, j in pairs:
        if (int(labels_a[i]) == int(labels_a[j])) == (int(labels_b[i]) == int(labels_b[j])):
            same += 1
    return same / max(len(pairs), 1)


def bootstrap_stability(
    X: torch.Tensor,
    full_labels: torch.Tensor,
    k: int,
    *,
    method: str,
    seed: int,
    repeats: int,
    fraction: float,
    n_init: int,
    max_iter: int,
) -> float:
    n = int(X.shape[0])
    if n < k or int(repeats) <= 0:
        return float("nan")
    rng = random.Random(seed + 31337 + k)
    subset_size = max(k, min(n, int(round(n * float(fraction)))))
    scores: List[float] = []
    for repeat_idx in range(int(repeats)):
        subset = sorted(rng.sample(list(range(n)), subset_size))
        labels, _ = cluster_labels(
            X[torch.tensor(subset, dtype=torch.long)],
            k,
            method=method,
            seed=seed + repeat_idx * 17,
            n_init=n_init,
            max_iter=max_iter,
        )
        if labels is None:
            continue
        full_subset = [int(full_labels[i].item()) for i in subset]
        scores.append(pair_agreement(full_subset, [int(x) for x in labels.tolist()], max_pairs=2000, seed=seed + repeat_idx))
    return sum(scores) / len(scores) if scores else float("nan")


def build_cluster_matrix(
    feature_sets: Mapping[str, Mapping[str, Any]],
    *,
    layer_idx: int,
    site: str,
    feature_kinds: Sequence[str],
) -> Tuple[List[int], torch.Tensor]:
    per_kind: Dict[str, Tuple[Dict[int, int], torch.Tensor]] = {}
    for kind in feature_kinds:
        key = f"L{int(layer_idx)}/{site}/{kind}"
        item = feature_sets.get(key)
        if item is None:
            continue
        event_indices = [int(x) for x in item["event_indices"].tolist()]
        per_kind[kind] = ({event_idx: row_idx for row_idx, event_idx in enumerate(event_indices)}, item["values"].float())
    if not per_kind:
        return [], torch.empty((0, 0), dtype=torch.float32)
    common = set(next(iter(per_kind.values()))[0].keys())
    for idx_map, _values in per_kind.values():
        common &= set(idx_map.keys())
    event_indices = sorted(common)
    if not event_indices:
        return [], torch.empty((0, 0), dtype=torch.float32)
    cols: List[torch.Tensor] = []
    for kind in feature_kinds:
        if kind not in per_kind:
            continue
        idx_map, values = per_kind[kind]
        rows = torch.tensor([idx_map[event_idx] for event_idx in event_indices], dtype=torch.long)
        cols.append(values[rows].float())
    return event_indices, torch.cat(cols, dim=1) if cols else torch.empty((0, 0), dtype=torch.float32)


def direction_projections(
    X: torch.Tensor,
    events: Sequence[Mapping[str, Any]],
    event_indices: Sequence[int],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    labels = [str(events[idx].get("event_type")) for idx in event_indices]
    token_counts = [finite_float(events[idx].get("generated_tokens"), 0.0) for idx in event_indices]

    def mean_for(predicate) -> Optional[torch.Tensor]:
        selected = [i for i, label in enumerate(labels) if predicate(label, token_counts[i])]
        if not selected:
            return None
        return X[torch.tensor(selected, dtype=torch.long)].mean(dim=0)

    axes_specs = {
        "forced_vs_natural": (
            lambda label, _tok: label == "forced_marker",
            lambda label, _tok: label == "natural_marker",
        ),
        "local_inconsistency_alarm": (
            lambda label, _tok: label == "natural_marker",
            lambda label, _tok: label == "nonreflection_termination",
        ),
        "lexical_marker_branch": (
            lambda label, _tok: label == "forced_marker",
            lambda label, _tok: label == "nonreflection_termination",
        ),
        "silent_correction": (
            lambda label, _tok: label == "silent_correction",
            lambda label, _tok: label == "nonreflection_termination",
        ),
        "explicit_vs_silent": (
            lambda label, _tok: label in {"natural_marker", "forced_marker"},
            lambda label, _tok: label == "silent_correction",
        ),
    }
    if token_counts:
        threshold = sorted(token_counts)[len(token_counts) // 2]
        axes_specs["long_deliberation"] = (
            lambda _label, tok: tok >= threshold,
            lambda _label, tok: tok < threshold,
        )

    axes: Dict[str, torch.Tensor] = {}
    projections: Dict[str, torch.Tensor] = {}
    for name, (pos_pred, neg_pred) in axes_specs.items():
        pos = mean_for(pos_pred)
        neg = mean_for(neg_pred)
        if pos is None or neg is None:
            continue
        axis = tensor_normed(pos - neg)
        if float(axis.norm().item()) <= 0:
            continue
        axes[name] = axis
        projections[name] = X.float().matmul(axis.float())
    return axes, projections


def summarize_cluster(
    *,
    events: Sequence[Mapping[str, Any]],
    event_indices: Sequence[int],
    labels: torch.Tensor,
    method: str,
    k: int,
    inertia: float,
    silhouette: float,
    stability: float,
    projections: Mapping[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cluster_id in range(int(k)):
        member_positions = [i for i, value in enumerate(labels.tolist()) if int(value) == int(cluster_id)]
        member_events = [events[event_indices[i]] for i in member_positions]
        type_counts = Counter(str(row.get("event_type")) for row in member_events)
        count = len(member_events)
        row: Dict[str, Any] = {
            "method": method,
            "k": int(k),
            "cluster_id": int(cluster_id),
            "count": count,
            "fraction": count / max(len(event_indices), 1),
            "dominant_event_type": type_counts.most_common(1)[0][0] if type_counts else "",
            "event_type_counts": json.dumps(dict(type_counts), ensure_ascii=False),
            "inertia": inertia,
            "silhouette": silhouette,
            "bootstrap_pair_agreement": stability,
            "surface_reflection_rate": mean(bool_value(row.get("surface_reflection")) for row in member_events),
            "explicit_error_ack_rate": mean(bool_value(row.get("explicit_error_ack")) for row in member_events),
            "semantic_repair_rate": mean(bool_value(row.get("semantic_repair")) for row in member_events),
            "functional_repair_rate": mean(bool_value(row.get("functional_repair")) for row in member_events),
            "silent_answer_correction_rate": mean(bool_value(row.get("silent_answer_correction")) for row in member_events),
            "mean_generated_tokens": mean(row.get("generated_tokens") for row in member_events),
        }
        for name, values in projections.items():
            selected = [float(values[i].item()) for i in member_positions]
            row[f"mean_proj_{name}"] = mean(selected)
        rows.append(row)
    return rows


def run_clustering(
    *,
    events: List[Dict[str, Any]],
    feature_sets: Mapping[str, Mapping[str, Any]],
    layer_idx: int,
    site: str,
    feature_kinds: Sequence[str],
    pca_components: int,
    ks: Sequence[int],
    seed: int,
    n_init: int,
    max_iter: int,
    bootstrap_repeats: int,
    bootstrap_fraction: float,
    silhouette_max_points: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    event_indices, X_raw = build_cluster_matrix(feature_sets, layer_idx=layer_idx, site=site, feature_kinds=feature_kinds)
    if not event_indices or int(X_raw.shape[0]) < 3:
        return [{"status": "skipped_no_cluster_matrix"}], {}
    X_std, mean_vec, std_vec = standardize_all(X_raw.float())
    X_pca, explained = pca_reduce(X_std, int(pca_components))
    axes, projections = direction_projections(X_std, events, event_indices)
    for local_pos, event_idx in enumerate(event_indices):
        for name, values in projections.items():
            events[event_idx][f"proj_{name}"] = float(values[local_pos].item())

    methods = ["kmeans_sklearn"] if SKLEARN_AVAILABLE else ["kmeans_torch"]
    if SKLEARN_AVAILABLE:
        methods.append("agglomerative_ward")

    rows: List[Dict[str, Any]] = []
    assignments: Dict[str, torch.Tensor] = {}
    for method in methods:
        for k in ks:
            if int(k) > int(X_pca.shape[0]):
                continue
            labels, inertia = cluster_labels(
                X_pca,
                int(k),
                method=method,
                seed=seed,
                n_init=n_init,
                max_iter=max_iter,
            )
            if labels is None:
                continue
            silhouette = silhouette_score_torch(X_pca, labels, max_points=silhouette_max_points, seed=seed)
            stability = bootstrap_stability(
                X_pca,
                labels,
                int(k),
                method=method,
                seed=seed,
                repeats=bootstrap_repeats,
                fraction=bootstrap_fraction,
                n_init=n_init,
                max_iter=max_iter,
            )
            assign_key = f"{method}_k{k}"
            assignments[assign_key] = labels
            for local_pos, event_idx in enumerate(event_indices):
                events[event_idx][f"cluster_{assign_key}"] = int(labels[local_pos].item())
            rows.extend(
                summarize_cluster(
                    events=events,
                    event_indices=event_indices,
                    labels=labels,
                    method=method,
                    k=int(k),
                    inertia=inertia,
                    silhouette=silhouette,
                    stability=stability,
                    projections=projections,
                )
            )

    payload = {
        "layer_idx": int(layer_idx),
        "site": str(site),
        "feature_kinds": list(feature_kinds),
        "event_indices": torch.tensor(event_indices, dtype=torch.long),
        "x_raw": X_raw.float(),
        "x_standardized": X_std.float(),
        "x_pca": X_pca.float(),
        "pca_explained_variance": explained,
        "standardize_mean": mean_vec.float(),
        "standardize_std": std_vec.float(),
        "axes": axes,
        "projections": projections,
        "assignments": assignments,
    }
    return rows, payload


def reproduce_gate_check(
    *,
    roots: Sequence[Path],
    layer_idx: int,
    site: str,
    max_activation_files: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    activation_paths: List[Path] = []
    for root in roots:
        activation_paths.extend(iter_activation_paths(root))
    activation_paths = sorted(activation_paths)
    if int(max_activation_files) > 0:
        activation_paths = activation_paths[: int(max_activation_files)]

    pairs = [("T", "C"), ("IW", "CW")]
    act_key = f"L{int(layer_idx)}/{site}"
    data: Dict[Tuple[str, str], Dict[str, List[torch.Tensor]]] = {
        pair: {"t": [], "c": [], "diff": []}
        for pair in pairs
    }
    for path in activation_paths:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            continue
        free = ((payload.get("runs") or {}).get("free") or {})
        for pair in pairs:
            t_condition, c_condition = pair
            t_tensor = ((free.get(t_condition) or {}).get("activations") or {}).get(act_key)
            c_tensor = ((free.get(c_condition) or {}).get("activations") or {}).get(act_key)
            if t_tensor is None or c_tensor is None:
                continue
            if int(t_tensor.shape[0]) < 1 or int(c_tensor.shape[0]) < 1:
                continue
            t_vec = t_tensor[0].detach().float().cpu()
            c_vec = c_tensor[0].detach().float().cpu()
            if tuple(t_vec.shape) != tuple(c_vec.shape):
                continue
            data[pair]["t"].append(t_vec)
            data[pair]["c"].append(c_vec)
            data[pair]["diff"].append(tensor_normed(t_vec - c_vec))

    rows: List[Dict[str, Any]] = []
    directions: Dict[str, torch.Tensor] = {}
    for pair, values in data.items():
        if not values["diff"]:
            continue
        direction = tensor_normed(torch.stack(values["diff"], dim=0).mean(dim=0))
        t_stack = torch.stack(values["t"], dim=0)
        c_stack = torch.stack(values["c"], dim=0)
        t_proj = t_stack.float().matmul(direction.float())
        c_proj = c_stack.float().matmul(direction.float())
        name = f"{pair[0]}_minus_{pair[1]}"
        directions[name] = direction
        rows.append(
            {
                "pair": name,
                "layer_idx": int(layer_idx),
                "site": str(site),
                "count": len(values["diff"]),
                "direction_norm": float(direction.norm().item()),
                "mean_t_projection": float(t_proj.mean().item()),
                "mean_c_projection": float(c_proj.mean().item()),
                "mean_t_minus_c_projection": float((t_proj - c_proj).mean().item()),
                "mean_diff_norm": float(torch.stack([t - c for t, c in zip(values["t"], values["c"])], dim=0).norm(dim=1).mean().item()),
            }
        )
    return rows, directions


def compact_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], max_rows: int = 12) -> str:
    rows = list(rows)[: int(max_rows)]
    if not rows:
        return "（空）"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}" if math.isfinite(value) else "")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def write_report(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    reproduce_summary: Sequence[Mapping[str, Any]],
    classifier_summary: Sequence[Mapping[str, Any]],
    cluster_summary: Sequence[Mapping[str, Any]],
    pca_explained: Sequence[float],
) -> None:
    top_hidden = [
        row
        for row in classifier_summary
        if row.get("status") == "ok" and row.get("feature_source") == "hidden" and row.get("task") == "natural_vs_forced"
    ]
    top_hidden = sorted(top_hidden, key=lambda row: finite_float(row.get("auc"), -1.0), reverse=True)
    top_wait_hidden = [
        row
        for row in classifier_summary
        if row.get("status") == "ok"
        and row.get("feature_source") == "hidden"
        and row.get("task") == "natural_wait_vs_forced_wait"
    ]
    top_wait_hidden = sorted(top_wait_hidden, key=lambda row: finite_float(row.get("auc"), -1.0), reverse=True)
    natural_baselines = [
        row
        for row in classifier_summary
        if row.get("status") == "ok"
        and row.get("task") == "natural_vs_forced"
        and row.get("feature_source") in {"token", "token_norm", "position", "logit"}
    ]
    natural_baselines = sorted(natural_baselines, key=lambda row: (str(row.get("feature_source")), str(row.get("method"))))
    wait_baselines = [
        row
        for row in classifier_summary
        if row.get("status") == "ok"
        and row.get("task") == "natural_wait_vs_forced_wait"
        and row.get("feature_source") in {"token", "token_norm", "position", "logit"}
    ]
    wait_baselines = sorted(wait_baselines, key=lambda row: (str(row.get("feature_source")), str(row.get("method"))))
    best_clusters = sorted(
        [row for row in cluster_summary if row.get("count")],
        key=lambda row: (finite_float(row.get("silhouette"), -1.0), finite_float(row.get("bootstrap_pair_agreement"), -1.0)),
        reverse=True,
    )
    lines = [
        "# 自然反思 vs 人为反思 Hidden-State 事件空间报告",
        "",
        "## 研究问题",
        "这份报告想回答一个很具体的问题：模型自己生成 `Wait/Actually/However/...` 这类反思 marker 时，",
        "对应 hidden state 是否和我们人为塞入 forced `Wait` 时的 hidden state 不一样。进一步地，",
        "如果把所有反思相关事件放进同一个表示空间里，能不能自然分出几类行为状态，例如局部不一致报警、",
        "语义修复、functional repair、silent correction、或者长思考/exploration。",
        "",
        "## 数据概览",
        f"- 输入目录：`{json.dumps(summary.get('roots', []), ensure_ascii=False)}`",
        f"- 事件总数：`{summary.get('events')}`；事件类型计数：`{json.dumps(summary.get('event_type_counts', {}), ensure_ascii=False)}`",
        f"- hidden feature 组数：`{summary.get('feature_sets')}`",
        f"- sklearn 是否可用：`{summary.get('sklearn_available')}`。若为 `False`，报告使用纯 torch 的 logistic/ridge/k-means fallback。",
        "",
        "事件类型含义：`natural_marker` 是模型自然生成反思 marker；`forced_marker` 是实验人为强制前缀触发的 marker；",
        "`nonreflection_termination` 是没有显式反思 marker 的终止/继续事件；`silent_correction` 是没有显式 marker 但答案发生修正的事件。",
        "",
        "## 文献定位",
        "- ReflCtrl 说明自然轨迹中的 reflection hidden direction 可以被抽出来控制反思频率。",
        "- Are Reflective Words ... 直接用 hidden state 区分 checkpoint 和 termination，并做 PCA、logistic probe 和 causal intervention。",
        "- Molecular Structure of Thought 把推理轨迹组织成 normal / deep reasoning / self-reflection / exploration 等行为拓扑；它启发这里的聚类解释，但不是现成的 forced-wait classifier。",
        "",
        "## 方向复现检查",
        "这里复现 L22 `post_attn` 上已有 T/C gate 方向的投影差。如果 `mean_t_minus_c_projection` 明显为正，",
        "说明这批 raw activation 至少能复现已有 Stage 1 方向，不是读错 layer/site 或 position。",
        compact_table(
            reproduce_summary,
            ["pair", "layer_idx", "site", "count", "mean_t_minus_c_projection", "mean_diff_norm"],
            4,
        ),
        "",
        "## 自然 vs 人为反思 Probe",
        "下面只看 hidden feature。`natural_vs_forced` 的正类是自然生成 marker，负类是 forced marker；",
        "训练/测试按 `example_id` 分组，避免同一道题同时出现在 train/test。AUC 越高表示越容易分开。",
        compact_table(
            top_hidden,
            ["task", "method", "layer_idx", "site", "feature_kind", "n_examples", "auc", "accuracy", "f1", "permutation_p_value"],
            12,
        ),
        "",
        "## 自然 vs 人为反思 Baseline",
        "这些 baseline 用来防止 probe 只是抓住表面信息。`token` 包含原始 marker token 文本，因此可能利用前导空格等细节；",
        "`token_norm` 会把 token strip/lower 后再编码，更接近“只知道这是 wait/actually 一类词”；`position` 只看位置/长度；",
        "`logit` 看 marker 前的决策 logit 统计。",
        compact_table(
            natural_baselines,
            ["task", "feature_source", "method", "n_examples", "auc", "accuracy", "f1", "permutation_p_value"],
            16,
        ),
        "",
        "## Wait 归一化 Probe",
        "这个任务更严格：只比较 wait-like 的自然 marker 和 forced wait，尽量排除 `but/however/incorrect` 等不同词本身带来的差异。",
        "如果这里 hidden 仍高，而 `token_norm` baseline 接近 chance，说明差异不只是“词类别不同”。",
        compact_table(
            top_wait_hidden,
            ["task", "method", "layer_idx", "site", "feature_kind", "n_examples", "auc", "accuracy", "f1", "permutation_p_value"],
            12,
        ),
        "",
        "## Wait 归一化 Baseline",
        "读这个表时要特别看 `token_norm`、`position`、`logit`。如果 position/logit 也很高，",
        "说明 hidden 可分性仍可能来自生成位置或前一 token 决策难度，而不能直接解释成“真实反思状态不同”。",
        compact_table(
            wait_baselines,
            ["task", "feature_source", "method", "n_examples", "auc", "accuracy", "f1", "permutation_p_value"],
            16,
        ),
        "",
        "## 聚类快照",
        f"- PCA 前 5 个分量解释方差：`{[round(float(x), 4) for x in list(pca_explained)[:5]]}`",
        "聚类使用主配置 L22 `post_attn` 的 `h_pre/h_marker/delta_marker/delta_post` 拼接后标准化，再 PCA 到指定维度。",
        "`silhouette` 越高表示分得越清楚；`bootstrap_pair_agreement` 越高表示 bootstrap 后同簇关系越稳定。",
        compact_table(
            best_clusters,
            [
                "method",
                "k",
                "cluster_id",
                "count",
                "dominant_event_type",
                "silhouette",
                "bootstrap_pair_agreement",
                "semantic_repair_rate",
                "mean_generated_tokens",
            ],
            16,
        ),
        "",
        "## 本次 pilot 的读法",
        "- hidden probe 很容易把 natural marker 和 forced marker 分开；但普通 `token` 与 `logit` baseline 也很强，所以普通 natural-vs-forced 不能单独证明 hidden 捕捉到了内在反思差异。",
        "- wait 归一化后，`token_norm` baseline 如果接近 chance，而 hidden 仍很高，说明至少在“同样是 wait-like marker”的条件下 hidden 里仍有可分信号。",
        "- 但如果 `position` 或 `logit` baseline 仍然很高，这个可分信号还可能是位置、生成长度或 marker 前决策 logit 的 proxy；下一步需要做位置匹配、logit 匹配或 residualization。",
        "- 聚类结果更像一个行为状态图谱：可以看是否出现 natural repair、forced repair、silent correction、short termination、long deliberation 等簇，而不是只盯着二分类 AUC。",
        "",
        "## 下一步建议",
        "- 扩大样本量，同时固定每个 example 的自然/forced 对照集合，减少 skipped example 带来的不平衡。",
        "- 对 `natural_wait_vs_forced_wait` 做 matched-position 子集，或先用 position/logit 回归掉 confound 后再 probe hidden residual。",
        "- 把这里的事件级簇和 D5 的 `d_inconsistency / d_marker / d_commit / d_silent_correction / d_length` 投影合并，给每个 cluster 打更稳定的行为标签。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(root) for root in args.root]
    selected_layers = parse_int_ranges(args.layers)
    selected_sites = parse_csv_list(args.sites)
    feature_kinds = parse_csv_list(args.feature_kinds)
    cluster_feature_kinds = parse_csv_list(args.cluster_feature_kinds)
    marker_keywords = parse_csv_list(args.marker_keywords)
    cluster_ks = parse_ks(args.cluster_ks)

    events, feature_sets, stats = extract_events(
        roots=roots,
        selected_layers=selected_layers,
        selected_sites=selected_sites,
        feature_kinds=feature_kinds,
        keywords=marker_keywords,
        max_activation_files=int(args.max_activation_files),
    )

    classifier_summary = classifier_rows(
        events=events,
        feature_sets=feature_sets,
        primary_layer=int(args.primary_layer),
        primary_site=str(args.primary_site),
        seed=int(args.seed),
        train_fraction=float(args.train_fraction),
        min_task_examples=int(args.min_task_examples),
        permutations=int(args.permutations),
        permutation_primary_only=bool(args.permutation_primary_only),
    )

    cluster_summary, cluster_payload = run_clustering(
        events=events,
        feature_sets=feature_sets,
        layer_idx=int(args.primary_layer),
        site=str(args.primary_site),
        feature_kinds=cluster_feature_kinds,
        pca_components=int(args.pca_components),
        ks=cluster_ks,
        seed=int(args.seed),
        n_init=int(args.kmeans_n_init),
        max_iter=int(args.kmeans_max_iter),
        bootstrap_repeats=int(args.bootstrap_repeats),
        bootstrap_fraction=float(args.bootstrap_fraction),
        silhouette_max_points=int(args.silhouette_max_points),
    )
    reproduce_summary, reproduce_directions = reproduce_gate_check(
        roots=roots,
        layer_idx=int(args.primary_layer),
        site=str(args.primary_site),
        max_activation_files=int(args.max_activation_files),
    )

    summary = {
        **stats,
        "selected_layers": selected_layers,
        "selected_sites": selected_sites,
        "feature_kinds": feature_kinds,
        "cluster_feature_kinds": cluster_feature_kinds,
        "cluster_ks": cluster_ks,
        "sklearn_available": SKLEARN_AVAILABLE,
        "classifier_rows": len(classifier_summary),
        "cluster_rows": len(cluster_summary),
        "reproduce_gate_rows": len(reproduce_summary),
    }

    dump_jsonl(output_dir / "event_rows.jsonl", events)
    write_csv(output_dir / "classifier_summary.csv", classifier_summary)
    write_csv(output_dir / "cluster_summary.csv", cluster_summary)
    write_csv(output_dir / "reproduce_gate_summary.csv", reproduce_summary)
    write_json(output_dir / "summary.json", summary)
    torch.save(
        {
            "events": events,
            "feature_sets": feature_sets,
            "cluster": cluster_payload,
            "reproduce_gate_directions": reproduce_directions,
            "summary": summary,
        },
        output_dir / "event_features.pt",
    )
    write_report(
        output_dir,
        summary=summary,
        reproduce_summary=reproduce_summary,
        classifier_summary=classifier_summary,
        cluster_summary=cluster_summary,
        pca_explained=(cluster_payload or {}).get("pca_explained_variance", []),
    )

    print("[Done] Reflection event-space analysis finished.")
    print(f"- output_dir: {output_dir}")
    print(f"- events: {len(events)}")
    print(f"- classifier_rows: {len(classifier_summary)}")
    print(f"- cluster_rows: {len(cluster_summary)}")


if __name__ == "__main__":
    main()
