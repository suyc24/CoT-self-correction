from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schemas import KeywordHit


DEFAULT_REPAIR_LEXICON = [
    "wait",
    "no",
    "actually",
    "however",
    "mistake",
    "incorrect",
    "等一下",
    "不对",
    "重新检查",
    "重算",
]


def extract_think_segments(text: str) -> List[str]:
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    segments = [match.strip() for match in matches if match and match.strip()]
    if segments:
        return segments

    lower = text.lower()
    pos = lower.rfind("<think>")
    if pos >= 0:
        tail = text[pos + len("<think>") :].strip()
        if tail:
            return [tail]
    return []


def extract_continuation_think_text(text: str) -> str:
    if not text.strip():
        return ""
    lower = text.lower()
    close_pos = lower.find("</think>")
    think_text = text[:close_pos] if close_pos >= 0 else text
    if think_text.lower().startswith("<think>"):
        think_text = think_text[len("<think>") :]
    return think_text.strip()


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    lowered = keyword.lower().strip()
    if re.fullmatch(r"[a-z0-9 '\-]+", lowered):
        return re.compile(r"\b" + re.escape(lowered) + r"\b", flags=re.IGNORECASE)
    return re.compile(re.escape(lowered), flags=re.IGNORECASE)


def analyze_text_keywords(text: str, keywords: Sequence[str]) -> Dict[str, Any]:
    lowered = text.lower()
    hits: List[KeywordHit] = []
    matched_keywords: List[str] = []
    for keyword in keywords:
        kw = keyword.strip()
        if not kw:
            continue
        pattern = _keyword_pattern(kw)
        local_hits = [KeywordHit(keyword=kw, start=m.start(), end=m.end()) for m in pattern.finditer(lowered)]
        if local_hits:
            matched_keywords.append(kw)
            hits.extend(local_hits)
    hits.sort(key=lambda item: (item.start, item.end))
    return {
        "hit": bool(hits),
        "count": len(hits),
        "matched_keywords": matched_keywords,
        "positions": [item.to_dict() for item in hits],
    }


def analyze_repair_signals(text: str, lexicon: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    active_lexicon = list(lexicon or DEFAULT_REPAIR_LEXICON)
    keyword_stats = analyze_text_keywords(text, active_lexicon)
    unique_count = len(keyword_stats["matched_keywords"])
    total_count = keyword_stats["count"]
    token_count = max(len(text.split()), 1) if text.strip() else 1
    density = total_count / token_count
    score = round(total_count + 0.5 * unique_count + density, 6)
    strength = "none"
    if total_count >= 4 or score >= 5.0:
        strength = "strong"
    elif total_count >= 2 or score >= 2.0:
        strength = "moderate"
    elif total_count >= 1:
        strength = "weak"
    return {
        "score": score,
        "strength": strength,
        "total_hits": total_count,
        "unique_hits": unique_count,
        "matched_keywords": keyword_stats["matched_keywords"],
        "positions": keyword_stats["positions"],
    }


def count_substring_occurrences(text: str, needle: str) -> Tuple[int, Optional[int]]:
    if not text or not needle:
        return 0, None
    count = 0
    first_pos: Optional[int] = None
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos < 0:
            break
        if first_pos is None:
            first_pos = pos
        count += 1
        start = pos + max(len(needle), 1)
    return count, first_pos
