from __future__ import annotations

import ast
import math
import re
from typing import Optional


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
        for idx in range(brace_pos, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_pos + 1 : idx].strip()
        search_end = pos


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).strip().strip("$")
    boxed = extract_last_boxed(s)
    if boxed is not None:
        s = boxed
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "")
    s = s.replace("\u2212", "-")
    s = s.replace("−", "-")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"(?i)^\s*(the\s+answer\s+is|answer\s+is|final\s+answer\s+is|答案是)\s*[:：]?\s*", "", s)
    s = s.strip().strip(".").strip("。")
    s = s.replace(",", "").replace("，", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _strip_outer_pairs(text: str) -> str:
    s = text.strip()
    while len(s) >= 2:
        pair = (s[0], s[-1])
        if pair not in {("(", ")"), ("[", "]"), ("{", "}")}:
            break
        depth = 0
        balanced = True
        for idx, ch in enumerate(s):
            if ch == s[0]:
                depth += 1
            elif ch == s[-1]:
                depth -= 1
                if depth == 0 and idx != len(s) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        s = s[1:-1].strip()
    return s


def _read_group_or_token(text: str, start: int) -> tuple[Optional[str], int]:
    n = len(text)
    idx = start
    while idx < n and text[idx].isspace():
        idx += 1
    if idx >= n:
        return None, idx

    if text[idx] == "{":
        depth = 0
        for pos in range(idx, n):
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[idx + 1 : pos], pos + 1
        return None, n

    if text[idx] in "([":
        closing = ")" if text[idx] == "(" else "]"
        depth = 0
        for pos in range(idx, n):
            ch = text[pos]
            if ch == text[idx]:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[idx : pos + 1], pos + 1
        return None, n

    if text[idx] == "\\":
        end = idx + 1
        while end < n and text[end].isalpha():
            end += 1
        return text[idx:end], end

    if text[idx].isdigit() or text[idx] == ".":
        end = idx + 1
        while end < n and (text[end].isdigit() or text[end] == "."):
            end += 1
        return text[idx:end], end

    if text[idx].isalpha():
        end = idx + 1
        while end < n and text[end].isalpha():
            end += 1
        return text[idx:end], end

    return text[idx], idx + 1


def _replace_latex_frac(text: str) -> str:
    s = text
    for command in ("\\dfrac", "\\tfrac"):
        s = s.replace(command, "\\frac")

    while True:
        idx = s.find("\\frac")
        if idx < 0:
            break
        num, after_num = _read_group_or_token(s, idx + len("\\frac"))
        den, after_den = _read_group_or_token(s, after_num)
        if num is None or den is None:
            break
        replacement = f"(({num})/({den}))"
        s = s[:idx] + replacement + s[after_den:]
    return s


def _replace_latex_sqrt(text: str) -> str:
    s = text
    while True:
        idx = s.find("\\sqrt")
        if idx < 0:
            break
        arg, after_arg = _read_group_or_token(s, idx + len("\\sqrt"))
        if arg is None:
            break
        if arg.startswith("(") and arg.endswith(")"):
            replacement = f"sqrt{arg}"
        else:
            replacement = f"sqrt({arg})"
        s = s[:idx] + replacement + s[after_arg:]
    return s


def _to_eval_expr(text: str) -> Optional[str]:
    if not text:
        return None

    expr = normalize_answer(text)
    if not expr:
        return None

    if expr.count("=") == 1:
        lhs, rhs = expr.split("=", 1)
        if lhs and rhs and len(lhs) <= 3:
            expr = rhs

    expr = _strip_outer_pairs(expr)
    expr = expr.replace("%", "/100")
    expr = expr.replace("\\cdot", "*").replace("\\times", "*")
    expr = expr.replace("\\pi", "pi")
    expr = re.sub(r"(?<=\d)\\frac", r"+\\frac", expr)
    expr = _replace_latex_frac(expr)
    expr = _replace_latex_sqrt(expr)
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("[", "(").replace("]", ")")
    expr = expr.replace("^", "**")

    raw_tokens = re.findall(r"\d*\.\d+|\d+|\*\*|[a-zA-Z_]+|[()+\-*/,]", expr)
    if not raw_tokens:
        return None

    allowed_names = {"pi", "sqrt"}
    tokens: list[str] = []
    prev: Optional[str] = None

    for token in raw_tokens:
        if token.isalpha() and token not in allowed_names:
            return None

        current_is_operand_start = bool(re.fullmatch(r"\d*\.\d+|\d+|pi|sqrt|\(", token))
        prev_is_operand_end = bool(prev and re.fullmatch(r"\d*\.\d+|\d+|pi|\)", prev))

        if prev_is_operand_end and current_is_operand_start:
            if not (prev == "sqrt" and token == "("):
                tokens.append("*")

        tokens.append(token)
        prev = token

    return "".join(tokens)


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("unsupported constant")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left ** right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt":
        if len(node.args) != 1 or node.keywords:
            raise ValueError("invalid sqrt")
        return math.sqrt(_eval_ast(node.args[0]))
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    raise ValueError("unsupported expression")


def _eval_numeric_value(text: str) -> Optional[float]:
    expr = _to_eval_expr(text)
    if expr is None:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        return float(_eval_ast(tree))
    except Exception:
        return None


def parse_simple_number(text: str) -> Optional[float]:
    if not text:
        return None
    return _eval_numeric_value(text)


def answers_match(a: Optional[str], b: Optional[str]) -> bool:
    left = normalize_answer(a)
    right = normalize_answer(b)
    if left == right:
        return True

    # Cheap canonical fallback before numeric parsing.
    if _strip_outer_pairs(left) == _strip_outer_pairs(right):
        return True

    left_num = parse_simple_number(left)
    right_num = parse_simple_number(right)
    if left_num is None or right_num is None:
        return False
    tolerance = 1e-9 * max(1.0, abs(left_num), abs(right_num))
    return abs(left_num - right_num) <= tolerance


def classify_outcome(
    final_boxed: Optional[str],
    correct_answer: Optional[str],
    wrong_answer: Optional[str],
) -> str:
    if not final_boxed:
        return "no_boxed_answer"
    if wrong_answer is not None and answers_match(final_boxed, wrong_answer):
        return "keep_wrong"
    if correct_answer is not None and answers_match(final_boxed, correct_answer):
        return "corrected"
    return "other_answer"
