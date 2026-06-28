from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .answer_extraction import answers_match
from .registry import Registry
from .schemas import EditOperation


EDITOR_REGISTRY = Registry()


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
        for idx in range(brace_pos, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return pos, idx + 1, text[brace_pos + 1 : idx]
        search_end = pos


class ThinkEditorContext:
    def __init__(self, text: str):
        self.text = text
        span = find_first_think_span(text)
        self.think_start = span[0] if span else None
        self.think_end = span[1] if span else None
        self.think_text = text[self.think_start : self.think_end] if span else ""

    def replace_in_think(
        self,
        rel_start: int,
        rel_end: int,
        replacement_text: str,
        editor_name: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> EditOperation:
        if self.think_start is None or self.think_end is None:
            return EditOperation(
                editor_name=editor_name,
                success=False,
                input_text=self.text,
                output_text=self.text,
                details={"reason": "missing_complete_think_span", **(details or {})},
            )
        absolute_start = self.think_start + rel_start
        absolute_end = self.think_start + rel_end
        new_text = self.text[:absolute_start] + replacement_text + self.text[absolute_end:]
        return EditOperation(
            editor_name=editor_name,
            success=True,
            input_text=self.text,
            output_text=new_text,
            absolute_start=absolute_start,
            absolute_end=absolute_end,
            replacement_text=replacement_text,
            details=details or {},
        )


@EDITOR_REGISTRY.register("replace_last_boxed")
def edit_replace_last_boxed(text: str, params: Dict[str, Any]) -> EditOperation:
    ctx = ThinkEditorContext(text)
    boxed_span = find_last_boxed_span(ctx.think_text)
    replacement_text = str(params["replacement_text"])
    if boxed_span is None:
        return EditOperation(
            editor_name="replace_last_boxed",
            success=False,
            input_text=text,
            output_text=text,
            details={"reason": "no_boxed_found_in_think"},
        )
    start, end, previous = boxed_span
    return ctx.replace_in_think(
        start,
        end,
        f"\\boxed{{{replacement_text}}}",
        "replace_last_boxed",
        details={"previous_text": previous},
    )


@EDITOR_REGISTRY.register("replace_think_tail_answer")
def edit_replace_tail_answer(text: str, params: Dict[str, Any]) -> EditOperation:
    ctx = ThinkEditorContext(text)
    if not ctx.think_text:
        return EditOperation(
            editor_name="replace_think_tail_answer",
            success=False,
            input_text=text,
            output_text=text,
            details={"reason": "no_think_text"},
        )
    search_text = str(params.get("search_text", "") or "")
    if not search_text:
        search_text = str(params.get("correct_answer", "") or "")
    replacement_text = str(params["replacement_text"])
    last_pos = ctx.think_text.rfind(search_text) if search_text else -1
    if last_pos < 0:
        return EditOperation(
            editor_name="replace_think_tail_answer",
            success=False,
            input_text=text,
            output_text=text,
            details={"reason": "search_text_not_found", "search_text": search_text},
        )
    return ctx.replace_in_think(
        last_pos,
        last_pos + len(search_text),
        replacement_text,
        "replace_think_tail_answer",
        details={"search_text": search_text},
    )


@EDITOR_REGISTRY.register("replace_keyword")
def edit_replace_keyword(text: str, params: Dict[str, Any]) -> EditOperation:
    ctx = ThinkEditorContext(text)
    keyword = str(params["keyword"])
    replacement_text = str(params["replacement_text"])
    if not ctx.think_text:
        return EditOperation(
            editor_name="replace_keyword",
            success=False,
            input_text=text,
            output_text=text,
            details={"reason": "no_think_text", "keyword": keyword},
        )
    flags = re.IGNORECASE if params.get("ignore_case", True) else 0
    matches = list(re.finditer(re.escape(keyword), ctx.think_text, flags=flags))
    if not matches:
        return EditOperation(
            editor_name="replace_keyword",
            success=False,
            input_text=text,
            output_text=text,
            details={"reason": "keyword_not_found", "keyword": keyword},
        )
    index = int(params.get("match_index", -1))
    match = matches[index]
    return ctx.replace_in_think(
        match.start(),
        match.end(),
        replacement_text,
        "replace_keyword",
        details={"keyword": keyword, "match_index": index},
    )


@EDITOR_REGISTRY.register("replace_span")
def edit_replace_span(text: str, params: Dict[str, Any]) -> EditOperation:
    ctx = ThinkEditorContext(text)
    start = int(params["start"])
    end = int(params["end"])
    replacement_text = str(params["replacement_text"])
    return ctx.replace_in_think(
        start,
        end,
        replacement_text,
        "replace_span",
        details={"relative_start": start, "relative_end": end},
    )


@EDITOR_REGISTRY.register("append_wrong_boxed")
def edit_append_wrong_boxed(text: str, params: Dict[str, Any]) -> EditOperation:
    ctx = ThinkEditorContext(text)
    replacement_text = str(params["replacement_text"])
    append_text = str(params.get("template", "\nInterim answer: \\boxed{{{value}}}."))
    rendered = append_text.replace("{value}", replacement_text)
    if ctx.think_start is None:
        output_text = text.rstrip() + "\n<think>" + rendered
        return EditOperation(
            editor_name="append_wrong_boxed",
            success=True,
            input_text=text,
            output_text=output_text,
            absolute_start=len(text),
            absolute_end=len(text),
            replacement_text=rendered,
            details={"reason": "missing_complete_think_span_appended_new_block"},
        )
    return ctx.replace_in_think(
        len(ctx.think_text),
        len(ctx.think_text),
        rendered,
        "append_wrong_boxed",
    )


@EDITOR_REGISTRY.register("auto_tamper_answer")
def edit_auto_tamper_answer(text: str, params: Dict[str, Any]) -> EditOperation:
    correct_answer = params.get("correct_answer")
    wrong_answer = str(params["replacement_text"])
    op = edit_replace_last_boxed(text, {"replacement_text": wrong_answer})
    if op.success:
        previous_text = str(op.details.get("previous_text", ""))
        if correct_answer is None or answers_match(previous_text, str(correct_answer)):
            op.editor_name = "auto_tamper_answer"
            op.details["auto_strategy"] = "replace_last_boxed"
            return op
    op = edit_replace_tail_answer(
        text,
        {
            "search_text": str(correct_answer or ""),
            "correct_answer": str(correct_answer or ""),
            "replacement_text": wrong_answer,
        },
    )
    if op.success:
        op.editor_name = "auto_tamper_answer"
        op.details["auto_strategy"] = "replace_think_tail_answer"
        return op
    op = edit_append_wrong_boxed(text, {"replacement_text": wrong_answer})
    if op.success:
        op.editor_name = "auto_tamper_answer"
        op.details["auto_strategy"] = "append_wrong_boxed"
    return op


def apply_editor_chain(text: str, plan: Sequence[Dict[str, Any]]) -> Tuple[str, List[EditOperation]]:
    current = text
    operations: List[EditOperation] = []
    for step in plan:
        name = str(step["name"])
        params = dict(step.get("params", {}))
        fn = EDITOR_REGISTRY.get_required(name)
        op = fn(current, params)
        operations.append(op)
        if op.success:
            current = op.output_text
    return current, operations
