"""Prompt module scaffolding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, Generic, TypeVar


SectionPath = tuple[str, ...]


def _normalize_section_path(section_path: Sequence[str] | None) -> SectionPath:
    if section_path is None:
        return ()
    return tuple(section_path)


class PromptError(Exception):
    """Base class for prompt-related failures providing structured context."""

    def __init__(
        self,
        message: str,
        *,
        section_path: Sequence[str] | None = None,
        dataclass_type: type | None = None,
        placeholder: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.section_path: SectionPath = _normalize_section_path(section_path)
        self.dataclass_type = dataclass_type
        self.placeholder = placeholder


class PromptValidationError(PromptError):
    """Raised when prompt construction validation fails."""


class PromptRenderError(PromptError):
    """Raised when rendering a prompt fails."""


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


__all__ = [
    "PromptError",
    "PromptValidationError",
    "PromptRenderError",
    "Section",
]
