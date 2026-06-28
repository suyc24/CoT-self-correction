from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .schemas import GenerationConfig


def select_continuation_text(row: Dict[str, Any]) -> str:
    return str(
        row.get("generated_continuation")
        or row.get("cot_continuation")
        or row.get("continuation")
        or row.get("think_text")
        or row.get("full_text")
        or row.get("text")
        or row.get("output")
        or ""
    )


def resolve_row_identities(row: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ["example_id", "id"]:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return values


def build_problem_text(row: Dict[str, Any]) -> str:
    problem = row.get("problem")
    if isinstance(problem, str) and problem.strip():
        return problem.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if str(item.get("role", "")).lower() == "user":
                content = str(item.get("content", "")).strip()
                if content:
                    return content
    raise ValueError("Dataset row has neither non-empty `problem` nor a user message.")


def build_reference_solution(row: Dict[str, Any]) -> Optional[str]:
    solution = row.get("solution")
    if isinstance(solution, str) and solution.strip():
        return solution
    messages = row.get("messages")
    if isinstance(messages, list):
        assistant_contents = [
            str(item.get("content", ""))
            for item in messages
            if str(item.get("role", "")).lower() == "assistant"
        ]
        if assistant_contents:
            return "\n\n".join(assistant_contents)
    return None


def build_example_id(row: Dict[str, Any], dataset_local_index: int) -> str:
    source = str(row.get("source", "unknown")).strip() or "unknown"
    if "id" in row:
        return f"{source}:{row['id']}"
    if "idx" in row:
        return f"{source}:{row['idx']}"
    return f"{source}:{dataset_local_index}"


def resolve_prompt_prefix_from_row(
    row: Dict[str, Any],
    backend,
    generation_config: "GenerationConfig",
) -> str:
    prompt = str(row.get("prompt") or row.get("prompt_prefix") or "")
    if prompt.strip():
        return prompt

    continuation = select_continuation_text(row)
    full_text = str(row.get("full_text") or "")
    if full_text and continuation and full_text.endswith(continuation):
        return full_text[: -len(continuation)]

    question = str(row.get("problem") or row.get("question") or "")
    if question.strip():
        return backend.build_prompt(question, generation_config)

    raise ValueError("Cannot reconstruct prompt prefix for this row.")
