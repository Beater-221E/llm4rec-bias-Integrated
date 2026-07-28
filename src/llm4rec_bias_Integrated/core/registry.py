"""Generic name → factory registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from llm4rec_bias_Integrated.core.exceptions import ConfigurationError

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple string-keyed registry with decorator registration."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            key = name.strip().lower()
            if key in self._items:
                raise ConfigurationError(
                    f"{self.kind} '{key}' is already registered"
                )
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        key = name.strip().lower()
        if key not in self._items:
            known = ", ".join(sorted(self._items)) or "(none)"
            raise ConfigurationError(
                f"Unknown {self.kind} '{name}'. Known: {known}"
            )
        return self._items[key]

    def contains(self, name: str) -> bool:
        return name.strip().lower() in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def clear(self) -> None:
        """Test helper — clear registrations."""
        self._items.clear()
