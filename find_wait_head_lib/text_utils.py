from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Optional, Tuple


def build_stage1_prompt(
    tokenizer,
    question: str,
    system_prompt: str,
    assistant_prefix: str,
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in sig.parameters:
            kwargs["enable_thinking"] = enable_thinking
    except (TypeError, ValueError):
        pass

    try:
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Backward compatibility for tokenizers without enable_thinking support.
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        # Fallback for tokenizers without chat template.
        prompt = (
            f"System: {system_prompt}\n"
            f"User: {question}\n"
            "Assistant:\n"
        )

    if assistant_prefix and not prompt.endswith(assistant_prefix):
        prompt = prompt + assistant_prefix
    return prompt


def find_first_think_span(text: str) -> Optional[Tuple[int, int]]:
    lower = text.lower()
    start = lower.find("<think>")
    if start < 0:
        return None
    end = lower.find("</think>", start + len("<think>"))
    if end < 0:
        return None
    return start + len("<think>"), end


def find_last_boxed_span(text: str) -> Optional[Tuple[int, int, str]]:
    marker = "\\boxed"
    search_end = len(text)
    while True:
        pos = text.rfind(marker, 0, search_end)
        if pos < 0:
            return None
        brace_pos = text.find("{", pos + len(marker))
        if brace_pos < 0:
            search_end = pos
            continue
        depth = 0
        for i in range(brace_pos, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return pos, i + 1, text[brace_pos + 1 : i]
        search_end = pos


def tamper_think_answer(
    stage1_full_text: str,
    correct_answer: Any,
    wrong_answer: Any,
    strict_tamper: bool = False,
) -> Dict[str, Any]:
    span = find_first_think_span(stage1_full_text)
    if span is None:
        if strict_tamper:
            return {
                "ok": False,
                "reason": "stage1 output has no complete <think>...</think> span",
            }
        lower = stage1_full_text.lower()
        tag = "<think>"
        pos = lower.rfind(tag)
        wrong_text = str(wrong_answer).strip()
        if pos >= 0:
            head = stage1_full_text[: pos + len(tag)]
            tail = stage1_full_text[pos + len(tag) :].rstrip()
            tampered_prefix = head + tail + f"\nInterim answer: \\boxed{{{wrong_text}}}."
        else:
            tampered_prefix = stage1_full_text.rstrip() + f"\n<think>\nInterim answer: \\boxed{{{wrong_text}}}."
        return {
            "ok": True,
            "think_content_before_tamper": "",
            "think_content_after_tamper": f"Interim answer: \\boxed{{{wrong_text}}}.",
            "tampered_prefix": tampered_prefix,
            "tamper_method": "force_append_wrong_boxed_no_complete_think",
            "tampered_from": "",
            "tampered_to": wrong_text,
        }

    think_start, think_end = span
    think_content = stage1_full_text[think_start:think_end]
    wrong_text = str(wrong_answer).strip()
    correct_text = str(correct_answer).strip()

    tampered_content = think_content
    tamper_method = "none"
    replaced_from = ""

    boxed_span = find_last_boxed_span(think_content)
    if boxed_span is not None:
        b_start, b_end, boxed_inner = boxed_span
        if answers_match(boxed_inner, correct_answer) or (not strict_tamper):
            tampered_content = think_content[:b_start] + f"\\boxed{{{wrong_text}}}" + think_content[b_end:]
            tamper_method = "replace_last_boxed" if answers_match(boxed_inner, correct_answer) else "replace_last_boxed_nonmatching"
            replaced_from = boxed_inner
        else:
            boxed_span = None

    if boxed_span is None:
        last_pos = think_content.rfind(correct_text) if correct_text else -1
        if last_pos >= 0:
            tampered_content = (
                think_content[:last_pos] + wrong_text + think_content[last_pos + len(correct_text) :]
            )
            tamper_method = "replace_last_correct_literal"
            replaced_from = correct_text
        else:
            if not strict_tamper:
                tampered_content = think_content.rstrip() + f"\nInterim answer: \\boxed{{{wrong_text}}}."
                tamper_method = "append_wrong_boxed_fallback"
                replaced_from = ""
            else:
                return {
                    "ok": False,
                    "reason": "stage1 think does not contain a detectable correct answer to tamper",
                    "think_content_before_tamper": think_content,
                }

    # Remove </think> so model continues from inside think after tampering.
    tampered_prefix = stage1_full_text[:think_start] + tampered_content

    return {
        "ok": True,
        "think_content_before_tamper": think_content,
        "think_content_after_tamper": tampered_content,
        "tampered_prefix": tampered_prefix,
        "tamper_method": tamper_method,
        "tampered_from": replaced_from,
        "tampered_to": wrong_text,
    }


def extract_think_segments(text: str) -> List[str]:
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    segments = [m.strip() for m in matches if m and m.strip()]
    if segments:
        return segments

    # If the continuation is still inside an unfinished <think> block, keep the tail.
    lower = text.lower()
    pos = lower.rfind("<think>")
    if pos >= 0:
        tail = text[pos + len("<think>") :].strip()
        if tail:
            return [tail]
    return []


def extract_last_boxed(text: str) -> Optional[str]:
    marker = "\\boxed"
    search_end = len(text)
    while True:
        pos = text.rfind(marker, 0, search_end)
        if pos < 0:
            return None
        brace_pos = text.find("{", pos + len(marker))
        if brace_pos < 0:
            search_end = pos
            continue

        depth = 0
        for i in range(brace_pos, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_pos + 1 : i].strip()

        # unmatched brace; try previous boxed occurrence
        search_end = pos


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = s.strip("$")

    boxed = extract_last_boxed(s)
    if boxed is not None:
        s = boxed

    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"(?i)^\s*(the\s+answer\s+is|answer\s+is|final\s+answer\s+is)\s*[:：]?\s*", "", s)
    s = s.strip().strip(".").strip("。")
    s = s.replace(",", "").replace("，", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def parse_simple_number(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"[-+]?\d+", s):
        return float(int(s))
    if re.fullmatch(r"[-+]?\d*\.\d+", s):
        return float(s)
    if re.fullmatch(r"[-+]?\d+/\d+", s):
        num, den = s.split("/")
        den_v = int(den)
        if den_v == 0:
            return None
        return float(int(num) / den_v)
    return None


def answers_match(a: Optional[str], b: Optional[str]) -> bool:
    na = normalize_answer(a)
    nb = normalize_answer(b)
    if na == nb:
        return True
    va = parse_simple_number(na)
    vb = parse_simple_number(nb)
    if va is None or vb is None:
        return False
    return abs(va - vb) <= 1e-9


def classify_outcome(final_boxed: Optional[str], correct_answer: Any, wrong_answer: Any) -> str:
    if not final_boxed:
        return "no_boxed"
    if answers_match(final_boxed, wrong_answer):
        return "keep_wrong"
    if answers_match(final_boxed, correct_answer):
        return "corrected"
    return "other_answer"


def is_wrong_final_answer(final_boxed: Optional[str], correct_answer: Any) -> bool:
    if not final_boxed:
        return False
    return not answers_match(final_boxed, correct_answer)


def parse_keywords(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def detect_self_correction_keywords(think_text: str, keywords: List[str]) -> Tuple[bool, List[str]]:
    if not think_text.strip():
        return False, []

    lowered = think_text.lower()
    matched: List[str] = []
    for kw in keywords:
        kw_norm = kw.strip().lower()
        if not kw_norm:
            continue

        if re.fullmatch(r"[a-z0-9 '\\-]+", kw_norm):
            pattern = r"\b" + re.escape(kw_norm) + r"\b"
            if re.search(pattern, lowered):
                matched.append(kw)
        else:
            if kw_norm in lowered:
                matched.append(kw)

    return len(matched) > 0, matched
