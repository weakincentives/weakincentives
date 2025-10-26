from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Generic, TypeVar

_ParamsT = TypeVar("_ParamsT")


class Section(Generic[_ParamsT], ABC):
    """Abstract building block for prompt content."""

    def __init__(
        self,
        *,
        title: str,
        params: type[_ParamsT],
        defaults: _ParamsT | None = None,
        children: Sequence["Section[Any]"] | None = None,
        enabled: Callable[[_ParamsT], bool] | None = None,
    ) -> None:
        self.title = title
        self.params = params
        self.defaults = defaults
        self.children: tuple[Section[Any], ...] = tuple(children or ())
        self._enabled = enabled

    def is_enabled(self, params: _ParamsT) -> bool:
        """Return True when the section should render for the given params."""

        if self._enabled is None:
            return True
        return bool(self._enabled(params))

    @abstractmethod
    def render(self, params: _ParamsT, depth: int) -> str:
        """Produce markdown output for the section at the supplied depth."""

    def placeholder_names(self) -> set[str]:
        """Return placeholder identifiers used by the section template."""

        return set()

    def tools(self) -> tuple[Any, ...]:
        """Return the tools exposed by this section."""

        return ()


__all__ = ["Section", "_ParamsT"]
