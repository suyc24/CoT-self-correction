from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Sequence

from .row_utils import select_continuation_text


@dataclass
class RepetitionThresholds:
    same_token_run_threshold: int = 24
    tail_repeat_min_repeats: int = 4
    tail_repeat_max_ngram: int = 32
    tail_repeat_min_span: int = 0
    line_repeat_threshold: int = 3
    line_run_score_multiplier: int = 10
    word_tail_repeat_min_repeats: int = 4
    word_tail_repeat_max_ngram: int = 24
    word_tail_repeat_min_span: int = 0
    tail_word_requires_hard_signal: bool = False
    min_trigger_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoopBenchThresholds:
    numerical_loop_min_repeated_span: int = 500
    statement_loop_min_repeat_count: int = 4
    max_numerical_ngram: int = 128
    max_statement_ngram: int = 8
    numerical_same_digit_run_threshold: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def longest_same_token_run(token_ids: Sequence[int]) -> Dict[str, Any]:
    if not token_ids:
        return {"length": 0, "token_id": None, "start": None, "end": None}
    best_length = 1
    best_token = int(token_ids[0])
    best_start = 0
    cur_length = 1
    cur_start = 0
    for idx in range(1, len(token_ids)):
        if token_ids[idx] == token_ids[idx - 1]:
            cur_length += 1
        else:
            if cur_length > best_length:
                best_length = cur_length
                best_token = int(token_ids[idx - 1])
                best_start = cur_start
            cur_length = 1
            cur_start = idx
    if cur_length > best_length:
        best_length = cur_length
        best_token = int(token_ids[-1])
        best_start = cur_start
    return {
        "length": int(best_length),
        "token_id": best_token,
        "start": int(best_start),
        "end": int(best_start + best_length),
    }


def repeated_line_stats(text: str) -> Dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"line_count": 0, "unique_line_count": 0, "max_consecutive_same_line_run": 0}
    unique_line_count = len(set(lines))
    max_run = 1
    cur_run = 1
    for idx in range(1, len(lines)):
        if lines[idx] == lines[idx - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return {
        "line_count": len(lines),
        "unique_line_count": unique_line_count,
        "max_consecutive_same_line_run": max_run,
    }


def repeated_suffix_stats(items: Sequence[Any], n_values: Sequence[int]) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    values = list(items)
    for n in n_values:
        if len(values) < n * 2:
            continue
        pattern = values[-n:]
        repeats = 1
        cursor = len(values) - n
        while cursor - n >= 0 and values[cursor - n : cursor] == pattern:
            repeats += 1
            cursor -= n
        stats.append(
            {
                "ngram_size": int(n),
                "repeat_count_at_tail": int(repeats),
                "pattern": pattern,
            }
        )
    return stats


def build_repetition_profile(
    text: str,
    token_ids: Optional[Sequence[int]] = None,
    existing_repetition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    repetition = dict(existing_repetition or {})

    if not repetition.get("longest_same_token_run") and token_ids:
        repetition["longest_same_token_run"] = longest_same_token_run(token_ids)

    if not repetition.get("repeated_line_stats"):
        repetition["repeated_line_stats"] = repeated_line_stats(text)

    if not repetition.get("tail_repeated_suffix") and token_ids:
        repetition["tail_repeated_suffix"] = repeated_suffix_stats(token_ids, n_values=(1, 2, 4, 8, 16, 32))

    repetition["word_tail_repeated_suffix"] = repeated_suffix_stats(
        text.split(),
        n_values=(1, 2, 4, 8, 16, 24),
    )
    return repetition


def detect_repetition_from_profile(
    repetition: Dict[str, Any],
    thresholds: RepetitionThresholds,
) -> Dict[str, Any]:
    triggers: List[Dict[str, Any]] = []
    hard_trigger_count = 0

    same_run = repetition.get("longest_same_token_run") or {}
    same_run_length = int(same_run.get("length") or 0)
    if same_run_length >= thresholds.same_token_run_threshold:
        triggers.append(
            {
                "type": "same_token_run",
                "length": same_run_length,
                "token_id": same_run.get("token_id"),
                "severity": "hard",
            }
        )
        hard_trigger_count += 1

    strongest_tail_token_loop: Dict[str, Any] | None = None
    for item in repetition.get("tail_repeated_suffix") or []:
        repeats = int(item.get("repeat_count_at_tail") or 0)
        ngram_size = int(item.get("ngram_size") or 0)
        repeated_span = repeats * ngram_size
        if (
            repeats >= thresholds.tail_repeat_min_repeats
            and ngram_size <= thresholds.tail_repeat_max_ngram
            and repeated_span >= thresholds.tail_repeat_min_span
        ):
            candidate = {
                "type": "tail_token_loop",
                "ngram_size": ngram_size,
                "repeat_count": repeats,
                "repeated_span": repeated_span,
                "severity": "hard",
            }
            if strongest_tail_token_loop is None or repeated_span > int(strongest_tail_token_loop["repeated_span"]):
                strongest_tail_token_loop = candidate
    if strongest_tail_token_loop is not None:
        triggers.append(strongest_tail_token_loop)
        hard_trigger_count += 1

    line_stats = repetition.get("repeated_line_stats") or {}
    line_run = int(line_stats.get("max_consecutive_same_line_run") or 0)
    if line_run >= thresholds.line_repeat_threshold:
        triggers.append(
            {
                "type": "repeated_line_run",
                "line_run": line_run,
                "line_count": int(line_stats.get("line_count") or 0),
                "severity": "hard",
            }
        )
        hard_trigger_count += 1

    strongest_tail_word_loop: Dict[str, Any] | None = None
    for item in repetition.get("word_tail_repeated_suffix") or []:
        repeats = int(item.get("repeat_count_at_tail") or 0)
        ngram_size = int(item.get("ngram_size") or 0)
        repeated_span = repeats * ngram_size
        if (
            repeats >= thresholds.word_tail_repeat_min_repeats
            and ngram_size <= thresholds.word_tail_repeat_max_ngram
            and repeated_span >= thresholds.word_tail_repeat_min_span
        ):
            candidate = {
                "type": "tail_word_loop",
                "ngram_size": ngram_size,
                "repeat_count": repeats,
                "repeated_span": repeated_span,
                "severity": "soft",
            }
            if strongest_tail_word_loop is None or repeated_span > int(strongest_tail_word_loop["repeated_span"]):
                strongest_tail_word_loop = candidate
    if strongest_tail_word_loop is not None:
        if (not thresholds.tail_word_requires_hard_signal) or hard_trigger_count >= 1:
            triggers.append(strongest_tail_word_loop)
        else:
            strongest_tail_word_loop["suppressed"] = True

    trigger_types = list(dict.fromkeys(item["type"] for item in triggers))
    score = 0
    for item in triggers:
        if item["type"] == "same_token_run":
            score = max(score, int(item["length"]))
        elif item["type"] == "tail_token_loop":
            score = max(score, int(item["repeated_span"]))
        elif item["type"] == "repeated_line_run":
            score = max(score, int(item["line_run"]) * int(thresholds.line_run_score_multiplier))
        elif item["type"] == "tail_word_loop":
            score = max(score, int(item["repeated_span"]))

    return {
        "matched": hard_trigger_count >= thresholds.min_trigger_count,
        "score": int(score),
        "trigger_count": len(triggers),
        "hard_trigger_count": hard_trigger_count,
        "trigger_types": trigger_types,
        "triggers": triggers,
        "repetition": repetition,
    }


def analyze_repetition(
    text: str,
    token_ids: Optional[Sequence[int]] = None,
    existing_repetition: Optional[Dict[str, Any]] = None,
    thresholds: Optional[RepetitionThresholds] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or RepetitionThresholds()
    profile = build_repetition_profile(text, token_ids=token_ids, existing_repetition=existing_repetition)
    return detect_repetition_from_profile(profile, thresholds)


def _extract_digit_stream(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _split_sentences(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts: List[str] = []
    for piece in re.split(r"(?<=[.!?])\s+|\n+", normalized):
        clean = piece.strip()
        if clean:
            parts.append(clean)
    return parts


def _split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]


def _split_clauses(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts: List[str] = []
    for piece in re.split(r"(?<=[,;:])\s+|\n+", normalized):
        clean = piece.strip()
        if clean:
            parts.append(clean)
    return parts


def _tail_repeat_info(items: Sequence[Any], max_ngram: int) -> Dict[str, Any]:
    best: Dict[str, Any] = {
        "matched": False,
        "ngram_size": 0,
        "repeat_count": 0,
        "repeated_span": 0,
        "pattern": [],
    }
    max_ngram = max(1, min(int(max_ngram), len(items) // 2 if items else 1))
    for ngram_size in range(1, max_ngram + 1):
        stats = repeated_suffix_stats(items, n_values=(ngram_size,))
        if not stats:
            continue
        item = stats[0]
        repeat_count = int(item.get("repeat_count_at_tail") or 0)
        repeated_span = repeat_count * int(item.get("ngram_size") or 0)
        if repeated_span > int(best["repeated_span"]):
            best = {
                "matched": True,
                "ngram_size": int(item.get("ngram_size") or 0),
                "repeat_count": repeat_count,
                "repeated_span": repeated_span,
                "pattern": list(item.get("pattern") or []),
            }
    return best


def analyze_loopbench_repetition(
    text: str,
    *,
    thresholds: Optional[LoopBenchThresholds] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or LoopBenchThresholds()

    digit_stream = _extract_digit_stream(text)
    sentence_units = _split_sentences(text)
    line_units = _split_lines(text)
    clause_units = _split_clauses(text)

    numerical_tail = _tail_repeat_info(digit_stream, max_ngram=thresholds.max_numerical_ngram)
    sentence_tail = _tail_repeat_info(sentence_units, max_ngram=thresholds.max_statement_ngram)
    line_tail = _tail_repeat_info(line_units, max_ngram=thresholds.max_statement_ngram)
    clause_tail = _tail_repeat_info(clause_units, max_ngram=thresholds.max_statement_ngram)
    statement_tail = max(
        [sentence_tail, line_tail, clause_tail],
        key=lambda item: (int(item.get("repeat_count") or 0), int(item.get("repeated_span") or 0)),
    )
    same_digit_run = 0
    if digit_stream:
        prev = digit_stream[0]
        cur = 1
        same_digit_run = 1
        for ch in digit_stream[1:]:
            if ch == prev:
                cur += 1
                same_digit_run = max(same_digit_run, cur)
            else:
                prev = ch
                cur = 1

    numerical_loop = bool(
        (
            numerical_tail["matched"]
            and int(numerical_tail["repeated_span"]) > int(thresholds.numerical_loop_min_repeated_span)
        )
        or same_digit_run >= int(thresholds.numerical_same_digit_run_threshold)
    )
    statement_loop = bool(
        statement_tail["matched"]
        and int(statement_tail["repeat_count"]) >= int(thresholds.statement_loop_min_repeat_count)
    )

    return {
        "matched": bool(numerical_loop or statement_loop),
        "numerical_loop": numerical_loop,
        "statement_loop": statement_loop,
        "digit_stream_length": len(digit_stream),
        "sentence_count": len(sentence_units),
        "line_count": len(line_units),
        "clause_count": len(clause_units),
        "same_digit_run": same_digit_run,
        "numerical_tail": numerical_tail,
        "statement_tail": statement_tail,
        "sentence_tail": sentence_tail,
        "line_tail": line_tail,
        "clause_tail": clause_tail,
        "thresholds": thresholds.to_dict(),
    }


def analyze_row_repetition(
    row: Dict[str, Any],
    *,
    thresholds: Optional[RepetitionThresholds] = None,
    token_id_key: str = "generated_token_ids",
    repetition_key: str = "repetition",
    reuse_existing: bool = True,
) -> Dict[str, Any]:
    token_ids = row.get(token_id_key)
    if token_ids is not None:
        token_ids = [int(token_id) for token_id in token_ids]
    existing_repetition = row.get(repetition_key) if reuse_existing else None
    return analyze_repetition(
        select_continuation_text(row),
        token_ids=token_ids,
        existing_repetition=existing_repetition,
        thresholds=thresholds,
    )
