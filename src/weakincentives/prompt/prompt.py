# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    cast,
    get_args,
    get_origin,
    override,
)

from ._types import SupportsDataclass
from .chapter import Chapter, ChaptersExpansionPolicy
from .errors import (
    PromptRenderError,
    PromptValidationError,
    SectionPath,
)
from .response_format import ResponseFormatParams, ResponseFormatSection
from .section import Section
from .tool import Tool

if TYPE_CHECKING:
    from .overrides import PromptLike, PromptOverridesStore, ToolOverride

_EMPTY_TOOL_PARAM_DESCRIPTIONS: Mapping[str, Mapping[str, str]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class RenderedPrompt[OutputT]:
    """Rendered prompt text paired with structured output metadata."""

    text: str
    output_type: type[Any] | None
    container: Literal["object", "array"] | None
    allow_extra_keys: bool | None
    _tools: tuple[Tool[SupportsDataclass, SupportsDataclass], ...] = field(
        default_factory=tuple
    )
    _tool_param_descriptions: Mapping[str, Mapping[str, str]] = field(
        default=_EMPTY_TOOL_PARAM_DESCRIPTIONS
    )

    @override
    def __str__(self) -> str:
        return self.text

    @property
    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]:
        """Tools contributed by enabled sections in traversal order."""

        return self._tools

    @property
    def tool_param_descriptions(
        self,
    ) -> Mapping[str, Mapping[str, str]]:
        """Description patches keyed by tool name."""

        return self._tool_param_descriptions


def _clone_dataclass(instance: SupportsDataclass) -> SupportsDataclass:
    """Return a shallow copy of the provided dataclass instance."""

    return cast(SupportsDataclass, replace(cast(Any, instance)))


def _format_specialization_argument(argument: object | None) -> str:
    if argument is None:
        return "?"
    if isinstance(argument, type):
        return argument.__name__
    return repr(argument)


@dataclass(frozen=True, slots=True)
class SectionNode[ParamsT: SupportsDataclass]:
    """Flattened view of a section within a prompt."""

    section: Section[ParamsT]
    depth: int
    path: SectionPath


class Prompt[OutputT]:
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
        ns: str,
        key: str,
        name: str | None = None,
        sections: Sequence[Section[Any]] | None = None,
        chapters: Sequence[Chapter[Any]] | None = None,
        inject_output_instructions: bool = True,
        allow_extra_keys: bool = False,
    ) -> None:
        super().__init__()
        stripped_ns = ns.strip()
        if not stripped_ns:
            raise PromptValidationError("Prompt namespace must be a non-empty string.")
        stripped_key = key.strip()
        if not stripped_key:
            raise PromptValidationError("Prompt key must be a non-empty string.")
        self.ns = stripped_ns
        self.key = stripped_key
        self.name = name
        base_sections: list[Section[SupportsDataclass]] = [
            cast(Section[SupportsDataclass], section) for section in sections or ()
        ]
        self._base_sections: tuple[Section[SupportsDataclass], ...] = tuple(
            base_sections
        )
        self._sections: tuple[Section[SupportsDataclass], ...] = tuple(base_sections)
        self._section_nodes: list[SectionNode[SupportsDataclass]] = []
        self._params_registry: dict[
            type[SupportsDataclass], list[SectionNode[SupportsDataclass]]
        ] = {}
        self._defaults_by_path: dict[SectionPath, SupportsDataclass] = {}
        self._defaults_by_type: dict[type[SupportsDataclass], SupportsDataclass] = {}
        self.placeholders: dict[SectionPath, set[str]] = {}
        self._tool_name_registry: dict[str, SectionPath] = {}
        self._allow_extra_keys_requested = allow_extra_keys

        normalized_chapters: list[Chapter[SupportsDataclass]] = []
        seen_chapter_keys: set[str] = set()
        for chapter in chapters or ():
            if not isinstance(chapter, Chapter):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise PromptValidationError(
                    "Prompt chapters must be Chapter instances.",
                    section_path=(getattr(chapter, "key", "?"),),
                )
            normalized = cast(Chapter[SupportsDataclass], chapter)
            if normalized.key in seen_chapter_keys:
                raise PromptValidationError(
                    "Prompt chapters must use unique keys.",
                    section_path=(normalized.key,),
                )
            seen_chapter_keys.add(normalized.key)
            normalized_chapters.append(normalized)
        self._chapters: tuple[Chapter[SupportsDataclass], ...] = tuple(
            normalized_chapters
        )
        self._chapter_key_registry: dict[str, Chapter[SupportsDataclass]] = {
            chapter.key: chapter for chapter in self._chapters
        }

        self._output_type: type[Any] | None
        self._output_container: Literal["object", "array"] | None
        self._allow_extra_keys: bool | None
        (
            self._output_type,
            self._output_container,
            self._allow_extra_keys,
        ) = self._resolve_output_spec(allow_extra_keys)

        self.inject_output_instructions = inject_output_instructions

        for section in base_sections:
            self._register_section(section, path=(section.key,), depth=0)

        self._response_section: ResponseFormatSection | None = None
        if self._output_type is not None and self._output_container is not None:
            response_params = self._build_response_format_params()
            response_section = ResponseFormatSection(
                params=response_params,
                enabled=lambda _params, prompt=self: prompt.inject_output_instructions,
            )
            self._response_section = response_section
            section_for_registry = cast(Section[SupportsDataclass], response_section)
            self._sections += (section_for_registry,)
            self._register_section(
                section_for_registry,
                path=(response_section.key,),
                depth=0,
            )

    def render(
        self,
        *params: SupportsDataclass,
        inject_output_instructions: bool | None = None,
    ) -> RenderedPrompt[OutputT]:
        """Render the prompt using provided parameter dataclass instances."""

        param_lookup = self._collect_param_lookup(params)
        return self._render_internal(
            param_lookup,
            inject_output_instructions=inject_output_instructions,
        )

    def render_with_overrides(
        self,
        *params: SupportsDataclass,
        overrides_store: PromptOverridesStore,
        tag: str = "latest",
        inject_output_instructions: bool | None = None,
    ) -> RenderedPrompt[OutputT]:
        """Render the prompt using overrides supplied by an overrides store."""

        from .overrides import PromptDescriptor

        descriptor = PromptDescriptor.from_prompt(cast("PromptLike", self))
        override = overrides_store.resolve(descriptor=descriptor, tag=tag)

        overrides: dict[SectionPath, str] = {}
        tool_overrides: dict[str, ToolOverride] = {}
        if (
            override is not None
            and override.ns == descriptor.ns
            and override.prompt_key == descriptor.key
        ):
            descriptor_index = {
                section.path: section.content_hash for section in descriptor.sections
            }
            for path, section_override in override.sections.items():
                descriptor_hash = descriptor_index.get(path)
                if (
                    descriptor_hash is not None
                    and section_override.expected_hash == descriptor_hash
                ):
                    overrides[path] = section_override.body
            if override.tool_overrides:
                descriptor_tool_index = {
                    tool.name: tool.contract_hash for tool in descriptor.tools
                }
                for name, tool_override in override.tool_overrides.items():
                    descriptor_hash = descriptor_tool_index.get(name)
                    if (
                        descriptor_hash is not None
                        and tool_override.expected_contract_hash == descriptor_hash
                    ):
                        tool_overrides[name] = tool_override

        param_lookup = self._collect_param_lookup(params)
        return self._render_internal(
            param_lookup,
            overrides,
            tool_overrides,
            inject_output_instructions=inject_output_instructions,
        )

    def _register_section(
        self,
        section: Section[SupportsDataclass],
        *,
        path: SectionPath,
        depth: int,
    ) -> None:
        params_type = section.param_type
        if params_type is not None and not is_dataclass(params_type):
            raise PromptValidationError(
                "Section params must be a dataclass.",
                section_path=path,
                dataclass_type=params_type,
            )

        node: SectionNode[SupportsDataclass] = SectionNode(
            section=section, depth=depth, path=path
        )
        self._section_nodes.append(node)
        if params_type is not None:
            self._params_registry.setdefault(params_type, []).append(node)

        if params_type is not None and section.default_params is not None:
            default_value = section.default_params
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
            _ = self._defaults_by_type.setdefault(params_type, default_value)

        section_placeholders = section.placeholder_names()
        self.placeholders[path] = set(section_placeholders)
        if params_type is None:
            if section_placeholders:
                placeholder = sorted(section_placeholders)[0]
                raise PromptValidationError(
                    "Section does not accept parameters but declares placeholders.",
                    section_path=path,
                    placeholder=placeholder,
                )
        else:
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
            child_path = (*path, child.key)
            self._register_section(child, path=child_path, depth=depth + 1)

    @property
    def sections(self) -> tuple[SectionNode[SupportsDataclass], ...]:
        return tuple(self._section_nodes)

    @property
    def param_types(self) -> set[type[SupportsDataclass]]:
        return set(self._params_registry.keys())

    @property
    def chapters(self) -> tuple[Chapter[SupportsDataclass], ...]:
        return self._chapters

    def expand_chapters(
        self,
        policy: ChaptersExpansionPolicy,
        *,
        chapter_params: Mapping[str, SupportsDataclass | None] | None = None,
    ) -> Prompt[OutputT]:
        """Return a prompt snapshot with chapters opened per the supplied policy."""

        if not self._chapters:
            return self

        if policy is ChaptersExpansionPolicy.ALL_INCLUDED:
            return self._expand_chapters_all_included(
                chapter_params=chapter_params or {}
            )

        raise NotImplementedError(
            f"Chapters expansion policy '{policy.value}' is not supported."
        )

    def _expand_chapters_all_included(
        self,
        *,
        chapter_params: Mapping[str, SupportsDataclass | None],
    ) -> Prompt[OutputT]:
        provided_lookup = dict(chapter_params)
        unknown_keys = set(provided_lookup) - set(self._chapter_key_registry.keys())
        if unknown_keys:
            unknown_key = sorted(unknown_keys)[0]
            raise PromptValidationError(
                "Chapter parameters reference unknown chapter key.",
                section_path=(unknown_key,),
            )

        open_sections: list[Section[SupportsDataclass]] = list(self._base_sections)

        for chapter in self._chapters:
            key_present = chapter.key in provided_lookup
            params = provided_lookup.get(chapter.key)
            if key_present:
                params = self._normalize_chapter_params(chapter, params)
            elif chapter.default_params is not None:
                params = _clone_dataclass(chapter.default_params)

            if chapter.enabled is not None:
                if params is None and chapter.param_type is not None:
                    raise PromptValidationError(
                        "Chapter requires parameters for enabled predicate.",
                        section_path=(chapter.key,),
                        dataclass_type=chapter.param_type,
                    )
                try:
                    enabled = chapter.is_enabled(cast(Any, params))
                except Exception as error:
                    raise PromptValidationError(
                        "Chapter enabled predicate failed.",
                        section_path=(chapter.key,),
                        dataclass_type=chapter.param_type,
                    ) from error
                if not enabled:
                    continue

            open_sections.extend(chapter.sections)

        prompt_cls = type(self)

        return prompt_cls(
            ns=self.ns,
            key=self.key,
            name=self.name,
            sections=open_sections,
            chapters=(),
            inject_output_instructions=self.inject_output_instructions,
            allow_extra_keys=self._allow_extra_keys_requested,
        )

    def _normalize_chapter_params(
        self,
        chapter: Chapter[SupportsDataclass],
        params: SupportsDataclass | None,
    ) -> SupportsDataclass | None:
        params_type = chapter.param_type
        if params_type is None:
            if params is not None:
                raise PromptValidationError(
                    "Chapter does not accept parameters.",
                    section_path=(chapter.key,),
                )
            return None

        if not isinstance(params, params_type) or not is_dataclass(params):
            raise PromptValidationError(
                "Chapter parameters must be instances of the declared dataclass.",
                section_path=(chapter.key,),
                dataclass_type=params_type,
            )
        return params

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

        if not isinstance(candidate, type):
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

    def _build_response_format_params(self) -> ResponseFormatParams:
        container = self._output_container
        if container is None:
            raise RuntimeError(
                "Output container missing during response format construction."
            )

        article: Literal["a", "an"] = (
            "an" if container.startswith(("a", "e", "i", "o", "u")) else "a"
        )
        extra_clause = ". Do not add extra keys." if not self._allow_extra_keys else "."
        return ResponseFormatParams(
            article=article,
            container=container,
            extra_clause=extra_clause,
        )

    def _collect_param_lookup(
        self, params: tuple[SupportsDataclass, ...]
    ) -> dict[type[SupportsDataclass], SupportsDataclass]:
        lookup: dict[type[SupportsDataclass], SupportsDataclass] = {}
        for value in params:
            if isinstance(value, type):
                provided_type: type[Any] = value
            else:
                provided_type = type(value)
            if isinstance(value, type) or not is_dataclass(value):
                raise PromptValidationError(
                    "Prompt expects dataclass instances.",
                    dataclass_type=provided_type,
                )
            params_type = cast(type[SupportsDataclass], provided_type)
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

    def _render_internal(
        self,
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
        overrides: Mapping[SectionPath, str] | None = None,
        tool_overrides: Mapping[str, ToolOverride] | None = None,
        *,
        inject_output_instructions: bool | None = None,
    ) -> RenderedPrompt[OutputT]:
        rendered_sections: list[str] = []
        collected_tools: list[Tool[SupportsDataclass, SupportsDataclass]] = []
        override_lookup = dict(overrides or {})
        tool_override_lookup = dict(tool_overrides or {})
        field_description_patches: dict[str, dict[str, str]] = {}

        for node, section_params in self._iter_enabled_sections(
            dict(param_lookup),
            inject_output_instructions=inject_output_instructions,
        ):
            override_body = (
                override_lookup.get(node.path)
                if getattr(node.section, "accepts_overrides", True)
                else None
            )
            rendered = self._render_section(node, section_params, override_body)

            section_tools = node.section.tools()
            if section_tools:
                for tool in section_tools:
                    override = (
                        tool_override_lookup.get(tool.name)
                        if tool.accepts_overrides
                        else None
                    )
                    patched_tool = tool
                    if override is not None:
                        if (
                            override.description is not None
                            and override.description != tool.description
                        ):
                            patched_tool = replace(
                                tool, description=override.description
                            )
                        if override.param_descriptions:
                            field_description_patches[tool.name] = dict(
                                override.param_descriptions
                            )
                    collected_tools.append(patched_tool)

            if rendered:
                rendered_sections.append(rendered)

        text = "\n\n".join(rendered_sections)

        return RenderedPrompt(
            text=text,
            output_type=self._output_type,
            container=self._output_container,
            allow_extra_keys=self._allow_extra_keys,
            _tools=tuple(collected_tools),
            _tool_param_descriptions=_freeze_tool_param_descriptions(
                field_description_patches
            ),
        )

    def _render_section(
        self,
        node: SectionNode[SupportsDataclass],
        section_params: SupportsDataclass | None,
        override_body: str | None,
    ) -> str:
        params_type = node.section.param_type
        try:
            render_override = getattr(node.section, "render_with_template", None)
            if override_body is not None and callable(render_override):
                override_renderer = cast(
                    Callable[[str, SupportsDataclass | None, int], str],
                    render_override,
                )
                rendered = override_renderer(override_body, section_params, node.depth)
            else:
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
        except Exception as error:
            raise PromptRenderError(
                "Section rendering failed.",
                section_path=node.path,
                dataclass_type=params_type,
            ) from error

        return rendered

    def _resolve_section_params(
        self,
        node: SectionNode[SupportsDataclass],
        param_lookup: dict[type[SupportsDataclass], SupportsDataclass],
    ) -> SupportsDataclass | None:
        params_type = node.section.param_type
        if params_type is None:
            return None

        section_params: SupportsDataclass | None = param_lookup.get(params_type)

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
                        constructor = cast(
                            Callable[[], SupportsDataclass | None], params_type
                        )
                        section_params = constructor()
                    except TypeError as error:
                        raise PromptRenderError(
                            "Missing parameters for section.",
                            section_path=node.path,
                            dataclass_type=params_type,
                        ) from error

        result: SupportsDataclass | None = section_params
        if result is None:
            raise PromptRenderError(
                "Section constructor must return a dataclass instance.",
                section_path=node.path,
                dataclass_type=params_type,
            )

        if not is_dataclass(result):
            raise PromptRenderError(
                "Section constructor must return a dataclass instance.",
                section_path=node.path,
                dataclass_type=params_type,
            )

        return result

    def _iter_enabled_sections(
        self,
        param_lookup: dict[type[SupportsDataclass], SupportsDataclass],
        *,
        inject_output_instructions: bool | None = None,
    ) -> Iterator[tuple[SectionNode[SupportsDataclass], SupportsDataclass | None]]:
        skip_depth: int | None = None

        for node in self._section_nodes:
            if skip_depth is not None:
                if node.depth > skip_depth:
                    continue
                skip_depth = None

            section_params = self._resolve_section_params(node, param_lookup)

            if node.section is self._response_section and (
                inject_output_instructions is not None
            ):
                enabled = inject_output_instructions
            else:
                try:
                    enabled = node.section.is_enabled(cast(Any, section_params))
                except Exception as error:
                    raise PromptRenderError(
                        "Section enabled predicate failed.",
                        section_path=node.path,
                        dataclass_type=node.section.param_type,
                    ) from error

            if not enabled:
                skip_depth = node.depth
                continue

            yield node, section_params

    def _register_section_tools(
        self,
        section: Section[SupportsDataclass],
        path: SectionPath,
    ) -> None:
        section_tools = section.tools()
        if not section_tools:
            return

        for tool in section_tools:
            params_type = cast(
                type[SupportsDataclass] | None, getattr(tool, "params_type", None)
            )
            if not isinstance(params_type, type) or not is_dataclass(params_type):
                dataclass_type = (
                    params_type if isinstance(params_type, type) else section.param_type
                )
                raise PromptValidationError(
                    "Tool params_type must be a dataclass type.",
                    section_path=path,
                    dataclass_type=dataclass_type,
                )
            existing_path = self._tool_name_registry.get(tool.name)
            if existing_path is not None:
                raise PromptValidationError(
                    "Duplicate tool name registered for prompt.",
                    section_path=path,
                    dataclass_type=tool.params_type,
                )

            self._tool_name_registry[tool.name] = path


__all__ = ["Prompt", "RenderedPrompt", "SectionNode"]


def _freeze_tool_param_descriptions(
    descriptions: Mapping[str, dict[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    if not descriptions:
        return MappingProxyType({})
    frozen: dict[str, Mapping[str, str]] = {}
    for name, field_mapping in descriptions.items():
        frozen[name] = MappingProxyType(dict(field_mapping))
    return MappingProxyType(frozen)
