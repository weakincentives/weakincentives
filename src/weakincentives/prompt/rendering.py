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

"""Rendering helpers for :class:`~weakincentives.prompt.prompt.Prompt`."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, override

from ._types import SupportsDataclass
from .errors import PromptRenderError, PromptValidationError
from .registry import PromptRegistrySnapshot, SectionNode
from .tool import Tool

if TYPE_CHECKING:
    from .overrides import ToolOverride


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
        default=MappingProxyType({})
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


class PromptRenderer:
    """Render a prompt using registry metadata and optional overrides."""

    def __init__(
        self,
        *,
        registry: PromptRegistrySnapshot,
        output_type: type[Any] | None,
        output_container: Literal["object", "array"] | None,
        allow_extra_keys: bool | None,
        overrides: Mapping[tuple[str, ...], str] | None = None,
        tool_overrides: Mapping[str, ToolOverride] | None = None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._output_type: type[Any] | None = output_type
        self._output_container: Literal["object", "array"] | None = output_container
        self._allow_extra_keys: bool | None = allow_extra_keys
        self._overrides: dict[tuple[str, ...], str] = dict(overrides or {})
        self._tool_overrides: dict[str, ToolOverride] = dict(tool_overrides or {})

    def collect_param_lookup(
        self, params: Sequence[SupportsDataclass]
    ) -> dict[type[SupportsDataclass], SupportsDataclass]:
        """Normalize positional params into a lookup keyed by dataclass type."""

        lookup: dict[type[SupportsDataclass], SupportsDataclass] = {}
        valid_types = set(self._registry.params_registry.keys())
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
            if params_type not in valid_types:
                raise PromptValidationError(
                    "Unexpected params type supplied to prompt.",
                    dataclass_type=params_type,
                )
            lookup[params_type] = value
        return lookup

    def render(
        self,
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
        *,
        inject_output_instructions: bool | None = None,
    ) -> RenderedPrompt[Any]:
        """Render a prompt using a pre-normalized param lookup."""

        rendered_sections: list[str] = []
        collected_tools: list[Tool[SupportsDataclass, SupportsDataclass]] = []
        field_description_patches: dict[str, dict[str, str]] = {}

        for node, section_params in self._iter_enabled_sections(
            dict(param_lookup),
            inject_output_instructions=inject_output_instructions,
        ):
            override_body = (
                self._overrides.get(node.path)
                if getattr(node.section, "accepts_overrides", True)
                else None
            )
            rendered = self._render_section(node, section_params, override_body)

            section_tools = node.section.tools()
            if section_tools:
                for tool in section_tools:
                    override = (
                        self._tool_overrides.get(tool.name)
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
        param_lookup: dict[type[SupportsDataclass], SupportsDataclass],
        *,
        inject_output_instructions: bool | None = None,
    ) -> Iterator[tuple[SectionNode[SupportsDataclass], SupportsDataclass | None]]:
        skip_depth: int | None = None
        response_section = self._registry.response_section

        for node in self._registry.section_nodes:
            if skip_depth is not None:
                if node.depth > skip_depth:
                    continue
                skip_depth = None

            section_params = self._registry.resolve_section_params(node, param_lookup)

            if (
                node.section is response_section
                and inject_output_instructions is not None
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


def _freeze_tool_param_descriptions(
    descriptions: Mapping[str, dict[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    if not descriptions:
        return MappingProxyType({})
    frozen: dict[str, Mapping[str, str]] = {}
    for name, field_mapping in descriptions.items():
        frozen[name] = MappingProxyType(dict(field_mapping))
    return MappingProxyType(frozen)


__all__ = ["PromptRenderer", "RenderedPrompt"]
