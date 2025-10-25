"""Prompt module scaffolding."""

from __future__ import annotations

from collections.abc import Sequence


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


__all__ = [
    "PromptError",
    "PromptValidationError",
    "PromptRenderError",
]
