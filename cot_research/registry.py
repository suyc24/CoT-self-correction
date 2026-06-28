from __future__ import annotations

from typing import Any, Callable, Dict, Iterable


class Registry(dict):
    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        key = name.strip().lower()
        if not key:
            raise ValueError("Registry name must be non-empty.")

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if key in self:
                raise ValueError(f"Duplicate registry key: {name}")
            self[key] = fn
            return fn

        return decorator

    def get_required(self, name: str) -> Callable[..., Any]:
        key = name.strip().lower()
        if key not in self:
            known = ", ".join(sorted(self.keys()))
            raise KeyError(f"Unknown registry key '{name}'. Known: {known}")
        return self[key]

    def names(self) -> Iterable[str]:
        return sorted(self.keys())
