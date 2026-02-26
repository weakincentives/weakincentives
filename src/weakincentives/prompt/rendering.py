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

from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field, is_dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, override

from ..dataclasses import FrozenDataclassMixin
from ..deadlines import Deadline
from ..runtime.logging import StructuredLogger, get_logger
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._visibility import SectionVisibility
from .errors import PromptRenderError, PromptValidationError, SectionPath
from .progressive_disclosure import (
    build_summary_suffix,
    compute_current_visibility,
    create_open_sections_handler,
    create_read_section_handler,
)
from .registry import RegistrySnapshot, SectionNode
from .section import Section
from .structured_output import StructuredOutputConfig
from .tool import Tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..runtime.session.protocols import SessionProtocol
    from ..skills import SkillMount
    from .overrides import PromptDescriptor, ToolOverride

logger: StructuredLogger = get_logger(__name__, context={"component": "prompt"})


_EMPTY_TOOL_PARAM_DESCRIPTIONS: Mapping[str, Mapping[str, str]] = MappingProxyType({})


@dataclass(slots=True, frozen=True)
class RenderedPrompt[OutputT_co](FrozenDataclassMixin):
    """Rendered prompt text paired with structured output metadata."""

    text: str
    structured_output: StructuredOutputConfig[SupportsDataclass] | None = None
    deadline: Deadline | None = None
    descriptor: PromptDescriptor | None = None
    _tools: tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...] = field(
        default_factory=tuple
    )
    _skills: tuple[SkillMount, ...] = field(default_factory=tuple)
    _tool_param_descriptions: Mapping[str, Mapping[str, str]] = field(
        default=_EMPTY_TOOL_PARAM_DESCRIPTIONS
    )

    @override
    def __str__(self) -> str:  # pragma: no cover - delegated behaviour
        return self.text

    @property
    def tools(self) -> tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]:
        """Tools contributed by enabled sections in traversal order."""

        return self._tools

    @property
    def skills(self) -> tuple[SkillMount, ...]:
        """Skills contributed by enabled sections in traversal order."""

        return self._skills

    @property
    def tool_param_descriptions(
        self,
    ) -> Mapping[str, Mapping[str, str]]:
        """Description patches keyed by tool name."""

        return self._tool_param_descriptions

    @property
    def output_type(self) -> type[SupportsDataclass] | None:
        """Return the declared dataclass type for structured output."""

        if self.structured_output is None:
            return None
        return self.structured_output.dataclass_type

    @property
    def container(self) -> Literal["object", "array"] | None:
        """Return the declared container for structured output."""

        if self.structured_output is None:
            return None
        return self.structured_output.container

    @property
    def allow_extra_keys(self) -> bool | None:
        """Return whether extra keys are allowed in structured output."""

        if self.structured_output is None:
            return None
        return self.structured_output.allow_extra_keys


def _freeze_tool_param_descriptions(
    descriptions: Mapping[str, dict[str, str]],
) -> Mapping[str, Mapping[str, str]]:
    if not descriptions:
        return MappingProxyType({})
    frozen: dict[str, Mapping[str, str]] = {}
    for name, field_mapping in descriptions.items():
        frozen[name] = MappingProxyType(dict(field_mapping))
    return MappingProxyType(frozen)


class PromptRenderer[OutputT]:
    """Render prompts using a registry snapshot."""

    def __init__(
        self,
        *,
        registry: RegistrySnapshot,
        structured_output: StructuredOutputConfig[SupportsDataclass] | None,
    ) -> None:
        super().__init__()
        self._registry = registry
        self._structured_output = structured_output

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
            if params_type not in self._registry.params_types:
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
        descriptor: PromptDescriptor | None = None,
        session: SessionProtocol | None = None,
    ) -> RenderedPrompt[OutputT]:
        logger.debug(
            "prompt.render.start",
            event="prompt.render.start",
            context={
                "descriptor": str(descriptor) if descriptor is not None else None,
                "param_types": [t.__qualname__ for t in param_lookup],
                "override_count": len(overrides) if overrides else 0,
                "tool_override_count": len(tool_overrides) if tool_overrides else 0,
            },
        )
        rendered_sections: list[str] = []
        collected_tools: list[Tool[SupportsDataclassOrNone, SupportsToolResult]] = []
        collected_skills: list[SkillMount] = []
        override_lookup = dict(overrides or {})
        tool_override_lookup = dict(tool_overrides or {})
        field_description_patches: dict[str, dict[str, str]] = {}
        summary_skip_depth: int | None = None
        has_summarized_with_tools = False
        has_summarized_without_tools = False

        for node, section_params in self._iter_enabled_sections(
            dict(param_lookup), session=session
        ):
            # Skip children of sections rendered with SUMMARY visibility
            if summary_skip_depth is not None:
                if node.depth > summary_skip_depth:
                    continue
                summary_skip_depth = None

            override_body = (
                override_lookup.get(node.path)
                if getattr(node.section, "accepts_overrides", True)
                else None
            )
            # Visibility overrides are now managed exclusively via session state.
            # effective_visibility checks session's VisibilityOverrides first.
            effective_visibility = node.section.effective_visibility(
                None, section_params, session=session, path=node.path
            )

            # When rendering with SUMMARY visibility, skip children
            if effective_visibility == SectionVisibility.SUMMARY:
                summary_skip_depth = node.depth
                has_tools_or_skills = self._subtree_has_tools_or_skills(node)
                if has_tools_or_skills:
                    has_summarized_with_tools = True
                else:
                    has_summarized_without_tools = True

            rendered = self._render_section(
                node, section_params, override_body, effective_visibility
            )
            rendered = self._append_summary_suffix(node, effective_visibility, rendered)

            # Don't collect tools/skills when rendering with SUMMARY visibility
            if effective_visibility != SectionVisibility.SUMMARY:
                self._collect_section_tools(
                    node.section,
                    tool_override_lookup,
                    collected_tools,
                    field_description_patches,
                )
                self._collect_section_skills(node.section, collected_skills)

            if rendered:
                rendered_sections.append(rendered)

        self._inject_disclosure_tools(
            param_lookup,
            collected_tools,
            has_summarized_with_tools=has_summarized_with_tools,
            has_summarized_without_tools=has_summarized_without_tools,
            session=session,
        )

        text = "\n\n".join(rendered_sections)

        logger.debug(
            "prompt.render.complete",
            event="prompt.render.complete",
            context={
                "descriptor": str(descriptor) if descriptor is not None else None,
                "section_count": len(rendered_sections),
                "tool_count": len(collected_tools),
                "skill_count": len(collected_skills),
                "text_length": len(text),
                "has_structured_output": self._structured_output is not None,
            },
        )

        return RenderedPrompt[OutputT](
            text=text,
            structured_output=self._structured_output,
            descriptor=descriptor,
            _tools=tuple(collected_tools),
            _skills=tuple(collected_skills),
            _tool_param_descriptions=_freeze_tool_param_descriptions(
                field_description_patches
            ),
        )

    def _subtree_has_tools_or_skills(
        self, node: SectionNode[SupportsDataclass]
    ) -> bool:
        """Check if section or descendants have tools or skills."""
        return self._section_or_descendants_have_tools(
            node
        ) or self._section_or_descendants_have_skills(node)

    def _append_summary_suffix(
        self,
        node: SectionNode[SupportsDataclass],
        effective_visibility: SectionVisibility,
        rendered: str,
    ) -> str:
        """Append summary suffix when rendering with SUMMARY visibility."""
        if (
            effective_visibility != SectionVisibility.SUMMARY
            or node.section.summary is None
            or not rendered
        ):
            return rendered
        section_key = ".".join(node.path)
        child_keys = self._collect_child_keys(node)
        has_tools_or_skills = self._subtree_has_tools_or_skills(node)
        suffix = build_summary_suffix(
            section_key, child_keys, has_tools=has_tools_or_skills
        )
        return rendered + suffix

    def _inject_disclosure_tools(
        self,
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
        collected_tools: list[Tool[SupportsDataclassOrNone, SupportsToolResult]],
        *,
        has_summarized_with_tools: bool,
        has_summarized_without_tools: bool,
        session: SessionProtocol | None,
    ) -> None:
        """Inject progressive disclosure tools when sections are summarized."""
        if not has_summarized_with_tools and not has_summarized_without_tools:
            return

        current_visibility = compute_current_visibility(
            self._registry,
            param_lookup,
            session=session,
        )

        if has_summarized_with_tools:
            open_sections_tool = create_open_sections_handler(
                registry=self._registry,
                current_visibility=current_visibility,
            )
            collected_tools.append(
                cast(
                    Tool[SupportsDataclassOrNone, SupportsToolResult],
                    open_sections_tool,
                )
            )

        if has_summarized_without_tools:
            read_section_tool = create_read_section_handler(
                registry=self._registry,
                current_visibility=current_visibility,
                param_lookup=param_lookup,
                session=session,
            )
            collected_tools.append(
                cast(
                    Tool[SupportsDataclassOrNone, SupportsToolResult],
                    read_section_tool,
                )
            )

    def _section_or_descendants_have_tools(
        self, parent_node: SectionNode[SupportsDataclass]
    ) -> bool:
        """Check if section or any descendant has tools attached.

        Uses precomputed subtree_has_tools index for O(1) lookup.
        """
        return self._registry.subtree_has_tools.get(parent_node.path, False)

    def _section_or_descendants_have_skills(
        self, parent_node: SectionNode[SupportsDataclass]
    ) -> bool:
        """Check if section or any descendant has skills attached.

        Uses precomputed subtree_has_skills index for O(1) lookup.
        """
        return self._registry.subtree_has_skills.get(parent_node.path, False)

    def _collect_child_keys(
        self, parent_node: SectionNode[SupportsDataclass]
    ) -> tuple[str, ...]:
        """Collect the keys of direct child sections for a parent node.

        Uses precomputed children_by_path index for O(1) lookup.
        """
        return self._registry.children_by_path.get(parent_node.path, ())

    def _collect_section_tools(
        self,
        section: Section[SupportsDataclass],
        tool_override_lookup: dict[str, ToolOverride],
        collected_tools: list[Tool[SupportsDataclassOrNone, SupportsToolResult]],
        field_description_patches: dict[str, dict[str, str]],
    ) -> None:
        section_tools = section.tools()
        if not section_tools:
            return

        for tool in section_tools:
            override = (
                tool_override_lookup.get(tool.name) if tool.accepts_overrides else None
            )
            patched_tool = tool
            if override is not None:
                if (
                    override.description is not None
                    and override.description != tool.description
                ):
                    patched_tool = replace(tool, description=override.description)
                if override.param_descriptions:
                    field_description_patches[tool.name] = dict(
                        override.param_descriptions
                    )
            collected_tools.append(patched_tool)

    def _collect_section_skills(
        self,
        section: Section[SupportsDataclass],
        collected_skills: list[SkillMount],
    ) -> None:
        """Collect skills from a section."""
        section_skills = section.skills()
        if not section_skills:
            return

        collected_skills.extend(section_skills)

    def _iter_enabled_sections(
        self,
        param_lookup: MutableMapping[type[SupportsDataclass], SupportsDataclass],
        *,
        session: SessionProtocol | None = None,
    ) -> Iterator[tuple[SectionNode[SupportsDataclass], SupportsDataclass | None]]:
        skip_depth: int | None = None

        for node in self._registry.sections:
            if skip_depth is not None:
                if node.depth > skip_depth:
                    continue
                skip_depth = None

            section_params = self._registry.resolve_section_params(node, param_lookup)

            try:
                enabled = node.section.is_enabled(section_params, session=session)
            except Exception as error:  # pragma: no cover - defensive
                raise PromptRenderError(
                    "Section enabled predicate failed.",
                    section_path=node.path,
                    dataclass_type=node.section.params_type,
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
        effective_visibility: SectionVisibility,
    ) -> str:
        params_type = node.section.params_type
        section_path = ".".join(node.path) if node.path else "(root)"
        logger.debug(
            "prompt.render.section",
            event="prompt.render.section",
            context={
                "section_path": section_path,
                "section_type": type(node.section).__qualname__,
                "visibility": effective_visibility.name,
                "has_override": override_body is not None,
                "depth": node.depth,
            },
        )
        try:
            if override_body is not None:
                rendered = node.section.render_override(
                    override_body, section_params, node.depth, node.number, node.path
                )
            else:
                rendered = node.section.render(
                    section_params,
                    node.depth,
                    node.number,
                    path=node.path,
                    visibility=effective_visibility,
                )
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
