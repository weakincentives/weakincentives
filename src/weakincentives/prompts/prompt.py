from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Iterator, TYPE_CHECKING

from .errors import (
    PromptRenderError,
    PromptValidationError,
    SectionPath,
)
from .section import Section

if TYPE_CHECKING:
    from .tool import Tool


@dataclass(frozen=True, slots=True)
class PromptSectionNode:
    """Flattened view of a section within a prompt."""

    section: Section[Any]
    depth: int
    path: SectionPath


class Prompt:
    """Coordinate prompt sections and their parameter bindings."""

    def __init__(
        self,
        *,
        name: str | None = None,
        sections: Sequence[Section[Any]] | None = None,
    ) -> None:
        self.name = name
        self._sections: tuple[Section[Any], ...] = tuple(sections or ())
        self._section_nodes: list[PromptSectionNode] = []
        self._params_registry: dict[type[Any], PromptSectionNode] = {}
        self.defaults: dict[type[Any], Any] = {}
        self.placeholders: dict[SectionPath, set[str]] = {}

        for section in self._sections:
            self._register_section(section, path=(section.title,), depth=0)

    def render(self, *params: Any) -> str:
        """Render the prompt using provided parameter dataclass instances."""

        param_lookup = self._collect_param_lookup(params)
        rendered_sections: list[str] = []

        for node, section_params in self._iter_enabled_sections(param_lookup):
            params_type = node.section.params
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

    def tools(self, *params: Any) -> tuple["Tool[Any, Any]", ...]:
        """Return tools exposed by enabled sections in traversal order."""

        param_lookup = self._collect_param_lookup(params)
        collected: list["Tool[Any, Any]"] = []

        for node, _section_params in self._iter_enabled_sections(param_lookup):
            section_tools = node.section.tools()
            if section_tools:
                collected.extend(section_tools)

        return tuple(collected)

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

        if not is_dataclass(params_type):
            raise PromptValidationError(
                "Section params must be a dataclass.",
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

    def _collect_param_lookup(self, params: tuple[Any, ...]) -> dict[type[Any], Any]:
        lookup: dict[type[Any], Any] = {}
        for value in params:
            provided_type = value if isinstance(value, type) else type(value)
            if isinstance(value, type) or not is_dataclass(value):
                raise PromptValidationError(
                    "Prompt expects dataclass instances.",
                    dataclass_type=provided_type,
                )
            params_type = provided_type
            if params_type not in self._params_registry:
                raise PromptValidationError(
                    "Unexpected params type supplied to prompt.",
                    dataclass_type=params_type,
                )
            lookup[params_type] = value
        return lookup

    def _resolve_section_params(
        self,
        node: PromptSectionNode,
        param_lookup: dict[type[Any], Any],
    ) -> Any:
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

        return section_params

    def _iter_enabled_sections(
        self,
        param_lookup: dict[type[Any], Any],
    ) -> Iterator[tuple[PromptSectionNode, Any]]:
        skip_depth: int | None = None

        for node in self._section_nodes:
            if skip_depth is not None:
                if node.depth > skip_depth:
                    continue
                skip_depth = None

            section_params = self._resolve_section_params(node, param_lookup)

            try:
                enabled = node.section.is_enabled(section_params)
            except Exception as error:  # pragma: no cover - defensive guard
                raise PromptRenderError(
                    "Section enabled predicate failed.",
                    section_path=node.path,
                    dataclass_type=node.section.params,
                ) from error

            if not enabled:
                skip_depth = node.depth
                continue

            yield node, section_params


__all__ = ["Prompt", "PromptSectionNode"]
