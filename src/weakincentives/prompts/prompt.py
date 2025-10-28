from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, ClassVar, Literal, cast, get_args, get_origin

from .errors import (
    PromptRenderError,
    PromptValidationError,
    SectionPath,
)
from .section import Section
from .tool import Tool


@dataclass(frozen=True, slots=True)
class RenderedPrompt[OutputT = Any]:
    """Rendered prompt text paired with structured output metadata."""

    text: str
    output_type: type[Any] | None
    output_container: Literal["object", "array"] | None
    allow_extra_keys: bool | None
    _tools: tuple[Tool[Any, Any], ...] = field(default_factory=tuple)

    def __str__(self) -> str:  # pragma: no cover - convenience for logging
        return self.text

    @property
    def tools(self) -> tuple[Tool[Any, Any], ...]:
        """Tools contributed by enabled sections in traversal order."""

        return self._tools


def _clone_dataclass(instance: object) -> object:
    """Return a shallow copy of the provided dataclass instance."""

    return cast(object, replace(cast(Any, instance)))


def _format_specialization_argument(argument: object | None) -> str:
    if argument is None:  # pragma: no cover - defensive formatting
        return "?"
    if isinstance(argument, type):
        return argument.__name__
    return repr(argument)  # pragma: no cover - fallback for debugging


@dataclass(frozen=True, slots=True)
class PromptSectionNode:
    """Flattened view of a section within a prompt."""

    section: Section[Any]
    depth: int
    path: SectionPath


class Prompt[OutputT = Any]:
    """Coordinate prompt sections and their parameter bindings."""

    _output_container_spec: ClassVar[Literal["object", "array"] | None] = None
    _output_dataclass_candidate: ClassVar[Any] = None

    def __class_getitem__(cls, item: object) -> type[Prompt[Any]]:
        origin = get_origin(item)
        candidate = item
        container: Literal["object", "array"] | None = "object"

        if origin is list:
            args = get_args(item)
            candidate = args[0] if len(args) == 1 else None
            container = "array"
            label = f"list[{_format_specialization_argument(candidate)}]"
        else:
            container = "object"
            label = _format_specialization_argument(candidate)

        name = f"{cls.__name__}[{label}]"
        namespace = {
            "__module__": cls.__module__,
            "_output_container_spec": container if candidate is not None else None,
            "_output_dataclass_candidate": candidate,
        }
        return type(name, (cls,), namespace)

    def __init__(
        self,
        *,
        name: str | None = None,
        sections: Sequence[Section[Any]] | None = None,
        inject_output_instructions: bool = True,
        allow_extra_keys: bool = False,
    ) -> None:
        self.name = name
        self._sections: tuple[Section[Any], ...] = tuple(sections or ())
        self._section_nodes: list[PromptSectionNode] = []
        self._params_registry: dict[type[Any], list[PromptSectionNode]] = {}
        self._defaults_by_path: dict[SectionPath, object] = {}
        self._defaults_by_type: dict[type[Any], object] = {}
        self.placeholders: dict[SectionPath, set[str]] = {}
        self._tool_name_registry: dict[str, SectionPath] = {}

        self._output_type: type[Any] | None
        self._output_container: Literal["object", "array"] | None
        self._allow_extra_keys: bool | None
        (
            self._output_type,
            self._output_container,
            self._allow_extra_keys,
        ) = self._resolve_output_spec(allow_extra_keys)

        self.inject_output_instructions = inject_output_instructions

        for section in self._sections:
            self._register_section(section, path=(section.title,), depth=0)

    def render(self, *params: object) -> RenderedPrompt[OutputT]:
        """Render the prompt using provided parameter dataclass instances."""

        param_lookup = self._collect_param_lookup(params)
        rendered_sections: list[str] = []
        collected_tools: list[Tool[Any, Any]] = []

        for node, section_params in self._iter_enabled_sections(param_lookup):
            params_type = node.section.params
            try:
                rendered = node.section.render(cast(Any, section_params), node.depth)
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

            section_tools = node.section.tools()
            if section_tools:
                collected_tools.extend(section_tools)

            if rendered:
                rendered_sections.append(rendered)

        text = "\n\n".join(rendered_sections)

        if self._should_inject_response_format():
            instructions = self._build_response_format_section()
            text = f"{text}\n\n{instructions}" if text else instructions

        return RenderedPrompt(
            text=text,
            output_type=self._output_type,
            output_container=self._output_container,
            allow_extra_keys=self._allow_extra_keys,
            _tools=tuple(collected_tools),
        )

    def _register_section(
        self,
        section: Section[Any],
        *,
        path: SectionPath,
        depth: int,
    ) -> None:
        params_type = section.params
        if not is_dataclass(params_type):
            raise PromptValidationError(
                "Section params must be a dataclass.",
                section_path=path,
                dataclass_type=params_type,
            )

        node = PromptSectionNode(section=section, depth=depth, path=path)
        self._section_nodes.append(node)
        self._params_registry.setdefault(params_type, []).append(node)

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
            self._defaults_by_path[path] = default_value
            self._defaults_by_type.setdefault(params_type, default_value)

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

        self._register_section_tools(section, path)

        for child in section.children:
            child_path = path + (child.title,)
            self._register_section(child, path=child_path, depth=depth + 1)

    @property
    def sections(self) -> tuple[PromptSectionNode, ...]:
        return tuple(self._section_nodes)

    @property
    def params_types(self) -> set[type[Any]]:
        return set(self._params_registry.keys())

    def _resolve_output_spec(
        self, allow_extra_keys: bool
    ) -> tuple[type[Any] | None, Literal["object", "array"] | None, bool | None]:
        candidate = getattr(type(self), "_output_dataclass_candidate", None)
        container = cast(
            Literal["object", "array"] | None,
            getattr(type(self), "_output_container_spec", None),
        )

        if candidate is None or container is None:
            return None, None, None

        if not isinstance(candidate, type):  # pragma: no cover - defensive guard
            candidate_type = cast(type[Any], type(candidate))
            raise PromptValidationError(
                "Prompt output type must be a dataclass.",
                dataclass_type=candidate_type,
            )

        if not is_dataclass(candidate):
            bad_dataclass = cast(type[Any], candidate)
            raise PromptValidationError(
                "Prompt output type must be a dataclass.",
                dataclass_type=bad_dataclass,
            )

        dataclass_type = cast(type[Any], candidate)
        return dataclass_type, container, allow_extra_keys

    def _should_inject_response_format(self) -> bool:
        return (
            self._output_type is not None
            and self._output_container is not None
            and self.inject_output_instructions
        )

    def _build_response_format_section(self) -> str:
        container = self._output_container
        if container is None:  # pragma: no cover - defensive guard
            raise RuntimeError(
                "Output container missing during response format construction."
            )

        article = "an" if container.startswith(("a", "e", "i", "o", "u")) else "a"
        ending = ". Do not add extra keys." if not self._allow_extra_keys else "."
        template = """## Response Format

Return ONLY a single fenced JSON code block. Do not include any text
before or after the block.

The top-level JSON value MUST be {article} {container} that matches the fields
of the expected schema{ending}"""
        return template.format(
            article=article,
            container=container,
            ending=ending,
        )

    def _collect_param_lookup(
        self, params: tuple[object, ...]
    ) -> dict[type[Any], object]:
        lookup: dict[type[Any], object] = {}
        for value in params:
            provided_type = value if isinstance(value, type) else type(value)
            if isinstance(value, type) or not is_dataclass(value):
                raise PromptValidationError(
                    "Prompt expects dataclass instances.",
                    dataclass_type=provided_type,
                )
            params_type = provided_type
            if params_type in lookup:
                raise PromptValidationError(
                    "Duplicate params type supplied to prompt.",
                    dataclass_type=params_type,
                )
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
        param_lookup: dict[type[Any], object],
    ) -> object:
        params_type = node.section.params
        section_params = param_lookup.get(params_type)

        if section_params is None:
            default_value = self._defaults_by_path.get(node.path)
            if default_value is not None:
                section_params = _clone_dataclass(default_value)
            else:
                type_default = self._defaults_by_type.get(params_type)
                if type_default is not None:
                    section_params = _clone_dataclass(type_default)
                else:
                    try:
                        constructor = cast(Callable[[], object], params_type)
                        section_params = constructor()
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
        param_lookup: dict[type[Any], object],
    ) -> Iterator[tuple[PromptSectionNode, object]]:
        skip_depth: int | None = None

        for node in self._section_nodes:
            if skip_depth is not None:
                if node.depth > skip_depth:
                    continue
                skip_depth = None

            section_params = self._resolve_section_params(node, param_lookup)

            try:
                enabled = node.section.is_enabled(cast(Any, section_params))
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

    def _register_section_tools(
        self,
        section: Section[Any],
        path: SectionPath,
    ) -> None:
        section_tools = section.tools()
        if not section_tools:
            return

        tools_iterable: Sequence[object] = cast(Sequence[object], section_tools)
        for tool_candidate in tools_iterable:
            if not isinstance(tool_candidate, Tool):
                raise PromptValidationError(
                    "Section tools() must return Tool instances.",
                    section_path=path,
                    dataclass_type=section.params,
                )
            tool: Tool[Any, Any] = cast(Tool[Any, Any], tool_candidate)
            params_type = cast(type[Any] | None, getattr(tool, "params_type", None))
            if not isinstance(params_type, type) or not is_dataclass(params_type):
                raise PromptValidationError(
                    "Tool params_type must be a dataclass type.",
                    section_path=path,
                    dataclass_type=(
                        params_type
                        if isinstance(params_type, type)
                        else type(params_type)
                    ),
                )
            existing_path = self._tool_name_registry.get(tool.name)
            if existing_path is not None:
                raise PromptValidationError(
                    "Duplicate tool name registered for prompt.",
                    section_path=path,
                    dataclass_type=tool.params_type,
                )

            self._tool_name_registry[tool.name] = path


__all__ = ["Prompt", "PromptSectionNode", "RenderedPrompt"]
