from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TokenRegion:
    name: str
    token_indices: Tuple[int, ...]
    meta: Mapping[str, Any]


def _unique_sorted(values: Iterable[int], n_tokens: int) -> Tuple[int, ...]:
    return tuple(sorted({int(x) for x in values if 0 <= int(x) < int(n_tokens)}))


def _token_char_offsets(tokenizer, token_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
    pieces: List[str] = []
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        pieces.append(text)
        offsets.append((cursor, cursor + len(text)))
        cursor += len(text)
    return "".join(pieces), offsets


def _chars_to_tokens(
    offsets: Sequence[Tuple[int, int]],
    *,
    start_char: int,
    end_char: int,
    offset_token_start: int = 0,
) -> Tuple[int, ...]:
    out: List[int] = []
    for local_idx, (left, right) in enumerate(offsets):
        if right <= int(start_char) or left >= int(end_char):
            continue
        out.append(int(offset_token_start) + int(local_idx))
    return tuple(out)


def _find_answer_spans(text: str, answers: Sequence[str]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    seen = set()
    for answer in answers:
        answer = str(answer).strip()
        if not answer:
            continue
        start = 0
        while True:
            idx = text.find(answer, start)
            if idx < 0:
                break
            key = (idx, idx + len(answer), answer)
            if key not in seen:
                spans.append(key)
                seen.add(key)
            start = idx + max(len(answer), 1)
    spans.sort()
    return spans


def annotate_forced_box_regions(
    tokenizer,
    *,
    prepared: Mapping[str, Any],
    correct_answer: str,
    local_window_chars: int = 1200,
    prev_k: int = 4,
) -> Dict[str, TokenRegion]:
    """Annotate source regions in a prepared forced-box tamper transcript.

    Token indices are absolute positions in `prepared["tamper_full_ids"]`.
    The annotations are intentionally conservative approximations: answer anchors
    use decoded-text search, while the forced wrong box span is known exactly from
    the tamper construction.
    """

    full_ids = [int(x) for x in prepared["tamper_full_ids"]]
    prompt_len = int(len(prepared["prompt_ids"]))
    prefix_len = int(len(prepared["prompt_ids"]) + len(prepared["clean_generated_ids"][: prepared["box_span"]["answer_start"]]))
    wrong_len = int(prepared.get("wrong_answer_token_len") or 0)
    force_len = int(prepared.get("tamper_force_token_len") or 0)
    n_tokens = len(full_ids)

    prefix_ids = full_ids[:prefix_len]
    prefix_text, prefix_offsets = _token_char_offsets(tokenizer, prefix_ids)
    prefix_text_len = len(prefix_text)

    clean_answer = str(prepared.get("clean_box_answer") or "").strip()
    correct_answer = str(correct_answer).strip()
    anchors = _find_answer_spans(prefix_text, [clean_answer, correct_answer])
    nearest_anchor = anchors[-1:] if anchors else []
    prev_anchors = anchors[-int(prev_k) :] if anchors else []

    local_start = max(0, prefix_text_len - int(local_window_chars))
    local_tokens = _chars_to_tokens(prefix_offsets, start_char=local_start, end_char=prefix_text_len)
    nonlocal_tokens = _chars_to_tokens(prefix_offsets, start_char=0, end_char=local_start)
    nearest_tokens: Tuple[int, ...] = tuple()
    if nearest_anchor:
        left, right, _answer = nearest_anchor[-1]
        nearest_tokens = _chars_to_tokens(prefix_offsets, start_char=left, end_char=right)
    prev_anchor_tokens: List[int] = []
    for left, right, _answer in prev_anchors:
        prev_anchor_tokens.extend(_chars_to_tokens(prefix_offsets, start_char=left, end_char=right))

    box_wrong = tuple(range(prefix_len, min(prefix_len + wrong_len, n_tokens)))
    box_format_left = max(0, prefix_len - 8)
    box_format_right = min(prefix_len + force_len, n_tokens)
    box_format = tuple(range(box_format_left, box_format_right))
    prompt = tuple(range(0, min(prompt_len, n_tokens)))

    conflict_all = _unique_sorted(
        list(box_wrong) + list(nearest_tokens) + list(prev_anchor_tokens) + list(local_tokens),
        n_tokens,
    )
    inconsistency_anchors = _unique_sorted(list(box_wrong) + list(nearest_tokens) + list(prev_anchor_tokens), n_tokens)

    regions: Dict[str, TokenRegion] = {
        "box_wrong": TokenRegion("box_wrong", _unique_sorted(box_wrong, n_tokens), {"prefix_len": prefix_len, "wrong_len": wrong_len}),
        "box_format": TokenRegion(
            "box_format",
            _unique_sorted(box_format, n_tokens),
            {"left": box_format_left, "right": box_format_right},
        ),
        "nearest_clean_anchor": TokenRegion(
            "nearest_clean_anchor",
            _unique_sorted(nearest_tokens, n_tokens),
            {"anchor_count": len(anchors), "nearest_anchor": nearest_anchor[-1] if nearest_anchor else None},
        ),
        "prev_k_clean_anchors": TokenRegion(
            "prev_k_clean_anchors",
            _unique_sorted(prev_anchor_tokens, n_tokens),
            {"anchor_count": len(anchors), "k": int(prev_k), "anchors": prev_anchors},
        ),
        "local_evidence_window": TokenRegion(
            "local_evidence_window",
            _unique_sorted(local_tokens, n_tokens),
            {"window_chars": int(local_window_chars), "start_char": local_start, "end_char": prefix_text_len},
        ),
        "nonlocal_reasoning": TokenRegion(
            "nonlocal_reasoning",
            _unique_sorted(nonlocal_tokens, n_tokens),
            {"end_char": local_start},
        ),
        "prompt": TokenRegion("prompt", _unique_sorted(prompt, n_tokens), {"prompt_len": prompt_len}),
        "inconsistency_anchors": TokenRegion(
            "inconsistency_anchors",
            inconsistency_anchors,
            {"parts": ["box_wrong", "nearest_clean_anchor", "prev_k_clean_anchors"]},
        ),
        "conflict_all": TokenRegion(
            "conflict_all",
            conflict_all,
            {"parts": ["box_wrong", "nearest_clean_anchor", "prev_k_clean_anchors", "local_evidence_window"]},
        ),
    }
    return regions


def serialize_regions(regions: Mapping[str, TokenRegion]) -> Dict[str, Any]:
    return {
        name: {
            "n_tokens": len(region.token_indices),
            "token_indices": list(region.token_indices),
            "meta": dict(region.meta),
        }
        for name, region in sorted(regions.items())
    }
