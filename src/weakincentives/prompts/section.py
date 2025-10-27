from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any


class Section[T](ABC):
    """Abstract building block for prompt content."""

    def __init__(
        self,
        *,
        title: str,
        params: type[T],
        defaults: T | None = None,
        children: Sequence[Section[Any]] | None = None,
        enabled: Callable[[T], bool] | None = None,
    ) -> None:
        self.title = title
        self.params = params
        self.defaults = defaults
        normalized_children: list[Section[Any]] = []
        for child in children or ():
            if not isinstance(child, Section):
                raise TypeError("Section children must be Section instances.")
            normalized_children.append(child)
        self.children = tuple(normalized_children)
        self._enabled = enabled

    def is_enabled(self, params: T) -> bool:
        """Return True when the section should render for the given params."""

        if self._enabled is None:
            return True
        return bool(self._enabled(params))

    @abstractmethod
    def render(self, params: T, depth: int) -> str:
        """Produce markdown output for the section at the supplied depth."""

    def placeholder_names(self) -> set[str]:
        """Return placeholder identifiers used by the section template."""

        return set()

    def tools(self) -> tuple[Any, ...]:
        """Return the tools exposed by this section."""

        return ()


__all__ = ["Section"]
