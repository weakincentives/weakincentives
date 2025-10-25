"""Prompt module scaffolding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence
from string import Template
from typing import Any, Callable, Generic, TypeVar

import textwrap


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


class TextSection(Section[_ParamsT]):
    """Render markdown text content using string.Template."""

    def __init__(
        self,
        *,
        title: str,
        body: str,
        params: type[_ParamsT],
        defaults: _ParamsT | None = None,
        children: Sequence[Section[Any]] | None = None,
        enabled: Callable[[_ParamsT], bool] | None = None,
    ) -> None:
        super().__init__(
            title=title,
            params=params,
            defaults=defaults,
            children=children,
            enabled=enabled,
        )
        self.body = body

    def render(self, params: _ParamsT, depth: int) -> str:
        heading_level = "#" * (depth + 2)
        heading = f"{heading_level} {self.title.strip()}"
        template = Template(textwrap.dedent(self.body).strip())
        rendered_body = template.safe_substitute(vars(params))
        if rendered_body:
            return f"{heading}\n\n{rendered_body.strip()}"
        return heading


@dataclass(frozen=True, slots=True)
class PromptSectionNode:
    """Flattened view of a section within a prompt."""

    section: Section[Any]
    depth: int
    path: SectionPath


class Prompt:
    """Coordinate prompt sections and their parameter bindings."""

    def __init__(self, *, root_sections: Sequence[Section[Any]] | None = None) -> None:
        self.root_sections: tuple[Section[Any], ...] = tuple(root_sections or ())
        self._section_nodes: list[PromptSectionNode] = []
        self._params_registry: dict[type[Any], PromptSectionNode] = {}
        self.defaults: dict[type[Any], Any] = {}
        self.placeholders: dict[SectionPath, set[str]] = {}

        for section in self.root_sections:
            self._register_section(section, path=(section.title,), depth=0)

    def _register_section(
        self,
        section: Section[Any],
        *,
        path: SectionPath,
        depth: int,
    ) -> None:
        params_type = section.params
        if params_type in self._params_registry:
            raise PromptValidationError(
                "Duplicate params dataclass registered for prompt section.",
                section_path=path,
                dataclass_type=params_type,
            )

        node = PromptSectionNode(section=section, depth=depth, path=path)
        self._section_nodes.append(node)
        self._params_registry[params_type] = node

        if section.defaults is not None:
            self.defaults[params_type] = section.defaults

        for child in section.children:
            child_path = path + (child.title,)
            self._register_section(child, path=child_path, depth=depth + 1)

    @property
    def sections(self) -> tuple[PromptSectionNode, ...]:
        return tuple(self._section_nodes)

    @property
    def params_types(self) -> set[type[Any]]:
        return set(self._params_registry.keys())


__all__ = [
    "PromptError",
    "PromptValidationError",
    "PromptRenderError",
    "Prompt",
    "Section",
    "TextSection",
]
