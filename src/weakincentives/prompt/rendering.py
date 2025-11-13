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

"""Rendering helpers for :mod:`weakincentives.prompt`."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field, is_dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, override

from ..deadlines import Deadline
from ._types import SupportsDataclass
from .errors import PromptRenderError, PromptValidationError, SectionPath
from .registry import RegistrySnapshot, SectionNode
from .response_format import ResponseFormatSection
from .tool import Tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .overrides import ToolOverride


_EMPTY_TOOL_PARAM_DESCRIPTIONS: Mapping[str, Mapping[str, str]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class RenderedPrompt[OutputT]:
    """Rendered prompt text paired with structured output metadata."""

    text: str
    output_type: type[Any] | None
    container: Literal["object", "array"] | None
    allow_extra_keys: bool | None
    deadline: Deadline | None = None
    _tools: tuple[Tool[SupportsDataclass, SupportsDataclass], ...] = field(
        default_factory=tuple
    )
    _tool_param_descriptions: Mapping[str, Mapping[str, str]] = field(
        default=_EMPTY_TOOL_PARAM_DESCRIPTIONS
    )

    @override
    def __str__(self) -> str:  # pragma: no cover - delegated behaviour
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


def _freeze_tool_param_descriptions(
    descriptions: Mapping[str, dict[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    if not descriptions:
        return MappingProxyType({})
    frozen: dict[str, Mapping[str, str]] = {}
    for name, field_mapping in descriptions.items():
        frozen[name] = MappingProxyType(dict(field_mapping))
    return MappingProxyType(frozen)


class PromptRenderer:
    """Render prompts using a registry snapshot."""

    def __init__(
        self,
        *,
        registry: RegistrySnapshot,
        output_type: type[Any] | None,
        output_container: Literal["object", "array"] | None,
        allow_extra_keys: bool | None,
        response_section: ResponseFormatSection | None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._output_type: type[Any] | None = output_type
        self._output_container: Literal["object", "array"] | None = output_container
        self._allow_extra_keys: bool | None = allow_extra_keys
        self._response_section: ResponseFormatSection | None = response_section

    def build_param_lookup(
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
            if params_type not in self._registry.param_types:
                raise PromptValidationError(
                    "Unexpected params type supplied to prompt.",
                    dataclass_type=params_type,
                )
            lookup[params_type] = value
        return lookup

    def render(
        self,
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
        overrides: Mapping[SectionPath, str] | None = None,
        tool_overrides: Mapping[str, ToolOverride] | None = None,
        *,
        inject_output_instructions: bool | None = None,
    ) -> RenderedPrompt[Any]:
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

        return RenderedPrompt[Any](
            text=text,
            output_type=self._output_type,
            container=self._output_container,
            allow_extra_keys=self._allow_extra_keys,
            _tools=tuple(collected_tools),
            _tool_param_descriptions=_freeze_tool_param_descriptions(
                field_description_patches
            ),
        )

    def _iter_enabled_sections(
        self,
        param_lookup: MutableMapping[type[SupportsDataclass], SupportsDataclass],
        *,
        inject_output_instructions: bool | None = None,
    ) -> Iterator[tuple[SectionNode[SupportsDataclass], SupportsDataclass | None]]:
        skip_depth: int | None = None

        for node in self._registry.sections:
            if skip_depth is not None:
                if node.depth > skip_depth:
                    continue
                skip_depth = None

            section_params = self._registry.resolve_section_params(node, param_lookup)

            if node.section is self._response_section and (
                inject_output_instructions is not None
            ):
                enabled = inject_output_instructions
            else:
                try:
                    enabled = node.section.is_enabled(cast(Any, section_params))
                except Exception as error:  # pragma: no cover - defensive
                    raise PromptRenderError(
                        "Section enabled predicate failed.",
                        section_path=node.path,
                        dataclass_type=node.section.param_type,
                    ) from error

            if not enabled:
                skip_depth = node.depth
                continue

            yield node, section_params

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
        except Exception as error:  # pragma: no cover - defensive
            raise PromptRenderError(
                "Section rendering failed.",
                section_path=node.path,
                dataclass_type=params_type,
            ) from error

        return rendered


__all__ = ["PromptRenderer", "RenderedPrompt"]
