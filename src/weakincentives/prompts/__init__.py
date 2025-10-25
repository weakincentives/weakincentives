"""Prompt module scaffolding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
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

    def placeholder_names(self) -> set[str]:
        """Return placeholder identifiers used by the section template."""

        return set()


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
        try:
            rendered_body = template.safe_substitute(vars(params))
        except KeyError as error:  # pragma: no cover - handled at prompt level
            missing = error.args[0]
            raise PromptRenderError(
                "Missing placeholder during render.",
                placeholder=str(missing),
            ) from error
        if rendered_body:
            return f"{heading}\n\n{rendered_body.strip()}"
        return heading

    def placeholder_names(self) -> set[str]:
        template = Template(textwrap.dedent(self.body).strip())
        placeholders: set[str] = set()
        for match in template.pattern.finditer(template.template):
            named = match.group("named")
            if named:
                placeholders.add(named)
                continue
            braced = match.group("braced")
            if braced:
                placeholders.add(braced)
        return placeholders


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

    def render(self, *, params: Sequence[Any]) -> str:
        """Render the prompt using provided parameter dataclass instances."""

        param_lookup: dict[type[Any], Any] = {}
        for value in params:
            provided_type = value if isinstance(value, type) else type(value)
            if isinstance(value, type) or not is_dataclass(value):
                raise PromptValidationError(
                    "Prompt.render expects dataclass instances.",
                    dataclass_type=provided_type,
                )
            params_type = provided_type
            if params_type not in self._params_registry:
                raise PromptValidationError(
                    "Unexpected params type supplied to prompt render.",
                    dataclass_type=params_type,
                )
            param_lookup[params_type] = value

        rendered_sections: list[str] = []
        skip_depth: int | None = None

        for node in self._section_nodes:
            if skip_depth is not None:
                if node.depth > skip_depth:
                    continue
                skip_depth = None

            params_type = node.section.params
            section_params = param_lookup.get(params_type)

            if section_params is None:
                default_value = self.defaults.get(params_type)
                if default_value is not None:
                    section_params = replace(default_value)
                else:
                    try:
                        section_params = params_type()  # type: ignore[call-arg]
                    except TypeError as error:
                        raise PromptRenderError(
                            "Missing parameters for section.",
                            section_path=node.path,
                            dataclass_type=params_type,
                        ) from error

            if section_params is None:
                raise PromptRenderError(
                    "Missing parameters for section.",
                    section_path=node.path,
                    dataclass_type=params_type,
                )

            try:
                enabled = node.section.is_enabled(section_params)
            except Exception as error:  # pragma: no cover - defensive guard
                raise PromptRenderError(
                    "Section enabled predicate failed.",
                    section_path=node.path,
                    dataclass_type=params_type,
                ) from error

            if not enabled:
                skip_depth = node.depth
                continue

            try:
                rendered = node.section.render(section_params, node.depth)
            except PromptRenderError as error:
                if error.section_path and error.dataclass_type:
                    raise
                raise PromptRenderError(
                    error.message,
                    section_path=node.path,
                    dataclass_type=params_type,
                    placeholder=error.placeholder,
                ) from error
            except Exception as error:  # pragma: no cover - defensive guard
                raise PromptRenderError(
                    "Section rendering failed.",
                    section_path=node.path,
                    dataclass_type=params_type,
                ) from error

            if rendered:
                rendered_sections.append(rendered)

        return "\n\n".join(rendered_sections)

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
            default_value = section.defaults
            if isinstance(default_value, type) or not is_dataclass(default_value):
                raise PromptValidationError(
                    "Section defaults must be dataclass instances.",
                    section_path=path,
                    dataclass_type=params_type,
                )
            if type(default_value) is not params_type:
                raise PromptValidationError(
                    "Section defaults must match section params type.",
                    section_path=path,
                    dataclass_type=params_type,
                )
            self.defaults[params_type] = default_value

        if not is_dataclass(params_type):
            raise PromptValidationError(
                "Section params must be a dataclass.",
                section_path=path,
                dataclass_type=params_type,
            )

        section_placeholders = section.placeholder_names()
        self.placeholders[path] = set(section_placeholders)
        param_fields = {field.name for field in fields(params_type)}
        unknown_placeholders = section_placeholders - param_fields
        if unknown_placeholders:
            placeholder = sorted(unknown_placeholders)[0]
            raise PromptValidationError(
                "Template references unknown placeholder.",
                section_path=path,
                dataclass_type=params_type,
                placeholder=placeholder,
            )

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
