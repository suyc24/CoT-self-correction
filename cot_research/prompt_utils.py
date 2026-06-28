from __future__ import annotations

import inspect


def build_chat_prompt(
    tokenizer,
    *,
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
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        prompt = (
            f"System: {system_prompt}\n"
            f"User: {question}\n"
            "Assistant:\n"
        )

    if assistant_prefix and not prompt.endswith(assistant_prefix):
        prompt = prompt + assistant_prefix
    return prompt
