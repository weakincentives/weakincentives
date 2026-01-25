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
from dataclasses import field, is_dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, override

from ..dataclasses import FrozenDataclass
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
    from .overrides import PromptDescriptor, ToolOverride

logger: StructuredLogger = get_logger(__name__, context={"component": "prompt"})


_EMPTY_TOOL_PARAM_DESCRIPTIONS: Mapping[str, Mapping[str, str]] = MappingProxyType({})


@FrozenDataclass()
class RenderedPrompt[OutputT_co]:
    """The result of rendering a prompt template with parameters.

    Contains the rendered text, collected tools from enabled sections,
    structured output configuration, and optional deadline constraints.

    This is an immutable dataclass returned by `PromptRenderer.render()`.

    Attributes:
        text: The fully rendered prompt text with all sections joined.
        structured_output: Configuration for structured output parsing, if any.
        deadline: Optional deadline constraint for the prompt execution.
        descriptor: Metadata identifying the prompt template that was rendered.

    Example:
        rendered = renderer.render(param_lookup)
        print(rendered.text)           # The prompt text
        print(rendered.tools)          # Tools contributed by sections
        print(rendered.output_type)    # Expected output dataclass type
    """

    text: str
    structured_output: StructuredOutputConfig[SupportsDataclass] | None = None
    deadline: Deadline | None = None
    descriptor: PromptDescriptor | None = None
    _tools: tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...] = field(
        default_factory=tuple
    )
    _tool_param_descriptions: Mapping[str, Mapping[str, str]] = field(
        default=_EMPTY_TOOL_PARAM_DESCRIPTIONS
    )

    @override
    def __str__(self) -> str:  # pragma: no cover - delegated behaviour
        return self.text

    @property
    def tools(self) -> tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]:
        """Tools contributed by enabled sections in traversal order.

        Returns all tools declared on sections that were rendered with
        non-SUMMARY visibility. Tools are collected during rendering and
        may include dynamically injected tools like `open_sections` or
        `read_section` for progressive disclosure.

        Returns:
            Tuple of Tool instances in the order they were collected.
        """

        return self._tools

    @property
    def tool_param_descriptions(
        self,
    ) -> Mapping[str, Mapping[str, str]]:
        """Parameter description overrides keyed by tool name.

        When tool overrides specify custom parameter descriptions, those
        patches are collected here. Adapters can use these to override
        the default parameter descriptions in the tool schema.

        Returns:
            Nested mapping of {tool_name: {param_name: description}}.
            Empty mapping if no overrides were applied.
        """

        return self._tool_param_descriptions

    @property
    def output_type(self) -> type[SupportsDataclass] | None:
        """The dataclass type expected for structured output parsing.

        When the prompt template declares a structured output type, this
        returns the dataclass class that responses should be parsed into.

        Returns:
            The dataclass type if structured output is configured, else None.
        """

        if self.structured_output is None:
            return None
        return self.structured_output.dataclass_type

    @property
    def container(self) -> Literal["object", "array"] | None:
        """The container type for structured output.

        Indicates whether the structured output should be parsed as a
        single object or an array of objects.

        Returns:
            "object" for single item, "array" for multiple items,
            or None if no structured output is configured.
        """

        if self.structured_output is None:
            return None
        return self.structured_output.container

    @property
    def allow_extra_keys(self) -> bool | None:
        """Whether extra keys are permitted in structured output.

        When True, the parser will ignore unrecognized keys in the response.
        When False, unrecognized keys will cause parsing to fail.

        Returns:
            True if extra keys are allowed, False if strict, or None
            if no structured output is configured.
        """

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
    """Renders prompt templates into final prompt text with tools.

    The renderer traverses sections from a registry snapshot, evaluates
    which sections are enabled based on parameters and session state,
    collects tools from visible sections, and produces a RenderedPrompt.

    The rendering process:
    1. Iterates through sections in registry order
    2. Skips disabled sections and their children
    3. Applies visibility rules (FULL, SUMMARY, HIDDEN)
    4. Collects tools from non-summarized sections
    5. Injects progressive disclosure tools when needed
    6. Joins rendered section text with double newlines

    Example:
        renderer = PromptRenderer(
            registry=snapshot,
            structured_output=config,
        )
        param_lookup = renderer.build_param_lookup((my_params,))
        rendered = renderer.render(param_lookup)
    """

    def __init__(
        self,
        *,
        registry: RegistrySnapshot,
        structured_output: StructuredOutputConfig[SupportsDataclass] | None,
    ) -> None:
        """Initialize the renderer with a registry snapshot.

        Args:
            registry: Snapshot of registered sections to render.
            structured_output: Optional configuration for structured output
                parsing. When set, the rendered prompt's output_type will
                be available for response parsing.
        """
        super().__init__()
        self._registry = registry
        self._structured_output = structured_output

    def build_param_lookup(
        self, params: tuple[SupportsDataclass, ...]
    ) -> dict[type[SupportsDataclass], SupportsDataclass]:
        """Build a type-to-instance mapping from parameter dataclasses.

        Validates that all provided parameters are dataclass instances
        (not types), checks for duplicates, and verifies each type is
        expected by sections in the registry.

        Args:
            params: Tuple of dataclass instances to use as section parameters.

        Returns:
            Dictionary mapping each dataclass type to its instance.

        Raises:
            PromptValidationError: If a parameter is not a dataclass instance,
                if duplicate types are provided, or if a type is not expected
                by any section in the registry.

        Example:
            lookup = renderer.build_param_lookup((UserContext(...), TaskParams(...)))
            # Returns {UserContext: instance, TaskParams: instance}
        """
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

    def render(  # noqa: C901
        self,
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
        overrides: Mapping[SectionPath, str] | None = None,
        tool_overrides: Mapping[str, ToolOverride] | None = None,
        *,
        descriptor: PromptDescriptor | None = None,
        session: SessionProtocol | None = None,
    ) -> RenderedPrompt[OutputT]:
        """Render the prompt with the given parameters and overrides.

        Traverses all registered sections, evaluates enabled state and
        visibility, renders each section's content, collects tools, and
        returns a complete RenderedPrompt.

        Args:
            param_lookup: Mapping from dataclass types to instances, as
                returned by `build_param_lookup()`.
            overrides: Optional mapping from section paths to override text.
                When a section path is present, the override text replaces
                the section's normal rendered output.
            tool_overrides: Optional mapping from tool names to override
                configurations. Allows customizing tool descriptions and
                parameter descriptions at render time.
            descriptor: Optional metadata identifying this prompt for
                logging and debugging purposes.
            session: Optional session for accessing state used by section
                enabled predicates and visibility rules.

        Returns:
            A RenderedPrompt containing the rendered text, collected tools,
            structured output configuration, and metadata.

        Raises:
            PromptRenderError: If any section fails to render or if an
                enabled predicate raises an exception.
        """
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
                # Track whether summarized section (or its descendants) has tools
                if self._section_or_descendants_have_tools(node):
                    has_summarized_with_tools = True
                else:
                    has_summarized_without_tools = True

            rendered = self._render_section(
                node, section_params, override_body, effective_visibility
            )

            # Append summary suffix for sections rendered with SUMMARY visibility
            if (
                effective_visibility == SectionVisibility.SUMMARY
                and node.section.summary is not None
                and rendered
            ):
                section_key = ".".join(node.path)
                child_keys = self._collect_child_keys(node)
                has_tools = self._section_or_descendants_have_tools(node)
                suffix = build_summary_suffix(
                    section_key, child_keys, has_tools=has_tools
                )
                rendered += suffix

            # Don't collect tools when rendering with SUMMARY visibility
            if effective_visibility != SectionVisibility.SUMMARY:
                self._collect_section_tools(
                    node.section,
                    tool_override_lookup,
                    collected_tools,
                    field_description_patches,
                )

            if rendered:
                rendered_sections.append(rendered)

        # Compute current visibility once for tool injection
        current_visibility: dict[SectionPath, SectionVisibility] | None = None
        if has_summarized_with_tools or has_summarized_without_tools:
            current_visibility = compute_current_visibility(
                self._registry,
                param_lookup,
                session=session,
            )

        # Inject open_sections tool when there are summarized sections with tools
        if has_summarized_with_tools and current_visibility is not None:
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

        # Inject read_section tool when there are summarized sections without tools
        if has_summarized_without_tools and current_visibility is not None:
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

        text = "\n\n".join(rendered_sections)

        logger.debug(
            "prompt.render.complete",
            event="prompt.render.complete",
            context={
                "descriptor": str(descriptor) if descriptor is not None else None,
                "section_count": len(rendered_sections),
                "tool_count": len(collected_tools),
                "text_length": len(text),
                "has_structured_output": self._structured_output is not None,
            },
        )

        return RenderedPrompt[OutputT](
            text=text,
            structured_output=self._structured_output,
            descriptor=descriptor,
            _tools=tuple(collected_tools),
            _tool_param_descriptions=_freeze_tool_param_descriptions(
                field_description_patches
            ),
        )

    def _section_or_descendants_have_tools(
        self, parent_node: SectionNode[SupportsDataclass]
    ) -> bool:
        """Check if section or any descendant has tools attached.

        Uses precomputed subtree_has_tools index for O(1) lookup.
        """
        return self._registry.subtree_has_tools.get(parent_node.path, False)

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
