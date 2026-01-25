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

"""Progressive disclosure for prompt sections via the open_sections tool.

This module implements the open_sections tool that enables progressive disclosure
of prompt content. When a prompt contains sections rendered with SUMMARY visibility,
the framework automatically registers this builtin tool so the model can request
expanded views of summarized content.

See specs/PROGRESSIVE_DISCLOSURE.md for the complete specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn

from ._visibility import SectionVisibility
from .errors import PromptValidationError, SectionPath, VisibilityExpansionRequired
from .tool import Tool, ToolContext
from .tool_result import ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Set

    from ..runtime.session.protocols import SessionProtocol
    from ._types import SupportsDataclass
    from .registry import RegistrySnapshot, SectionNode


@dataclass(slots=True, frozen=True)
class OpenSectionsParams:
    """Parameters for the open_sections tool.

    This dataclass defines the input schema for the builtin open_sections tool,
    which models use to request expansion of summarized prompt sections. When a
    model calls open_sections, the framework raises VisibilityExpansionRequired
    to signal the caller to re-render with expanded sections.

    Attributes:
        section_keys: Section keys to expand (dot-notation for nested sections).
        reason: Explanation of why the model needs the expanded content.

    Example:
        A model might call open_sections with::

            {"section_keys": ["tools.filesystem", "context.project"],
             "reason": "Need file operations and project structure details"}
    """

    section_keys: tuple[str, ...] = field(
        metadata={
            "description": (
                "Section keys to open. Each key identifies a summarized section. "
                "Use the keys shown in section summaries. Nested sections use "
                "dot notation (e.g., 'parent.child')."
            ),
        },
    )
    reason: str = field(
        metadata={
            "description": (
                "Brief explanation of why these sections need to be expanded. "
                "Helps the caller understand the model's information needs."
            ),
        },
    )


@dataclass(slots=True, frozen=True)
class ReadSectionParams:
    """Parameters for the read_section tool.

    This dataclass defines the input schema for the builtin read_section tool,
    which models use to retrieve full content of a summarized section without
    permanently expanding it. Unlike open_sections, this is a read-only operation
    that returns rendered markdown content directly.

    Attributes:
        section_key: The key of the section to read (dot-notation for nested).

    Example:
        A model might call read_section with::

            {"section_key": "guidelines.code-style"}
    """

    section_key: str = field(
        metadata={
            "description": (
                "The key of the summarized section to read. "
                "Use the key shown in the section summary. Nested sections use "
                "dot notation (e.g., 'parent.child')."
            ),
        },
    )


@dataclass(slots=True, frozen=True)
class ReadSectionResult:
    """Result returned by the read_section tool.

    Contains the fully rendered markdown content of a summarized section.
    Child sections within the result may still appear summarized depending
    on their own visibility settings.

    Attributes:
        content: The rendered markdown content of the section and its children.
    """

    content: str = field(
        metadata={
            "description": "The rendered markdown content of the section.",
        },
    )

    def render(self) -> str:
        """Return the canonical textual representation of this result.

        Called by the framework when serializing tool results for the model.
        Returns the raw markdown content without additional formatting.

        Returns:
            The section's rendered markdown content.
        """
        return self.content


def _parse_section_key(key: str) -> SectionPath:
    """Convert a dot-notation key to a section path tuple."""
    return tuple(key.split("."))


def _validate_section_keys(
    section_keys: tuple[str, ...],
    *,
    registry: RegistrySnapshot,
    current_visibility: Mapping[SectionPath, SectionVisibility],
) -> dict[SectionPath, SectionVisibility]:
    """Validate section keys and build the visibility override mapping.

    Sections that are already expanded are silently skipped. This allows
    models to request expansion of multiple sections without needing to
    track which ones are already expanded.

    Returns:
        Mapping from section paths to FULL visibility.

    Raises:
        PromptValidationError: If no keys provided, key is invalid, or all
            requested sections are already expanded.
    """
    if not section_keys:
        raise PromptValidationError("At least one section key must be provided.")

    requested_overrides: dict[SectionPath, SectionVisibility] = {}
    valid_paths: Set[SectionPath] = set(registry.section_paths)

    for key in section_keys:
        path = _parse_section_key(key)

        if path not in valid_paths:
            raise PromptValidationError(
                f"Section '{key}' does not exist in this prompt."
            )

        effective_visibility = current_visibility.get(path)
        if effective_visibility == SectionVisibility.FULL:
            # Skip already-expanded sections silently
            continue

        requested_overrides[path] = SectionVisibility.FULL

    # If all requested sections were already expanded, inform the model
    if not requested_overrides:
        if len(section_keys) == 1:
            raise PromptValidationError(
                f"Section '{section_keys[0]}' is already expanded."
            )
        keys_str = ", ".join(f"'{k}'" for k in section_keys)
        raise PromptValidationError(
            f"All requested sections are already expanded: {keys_str}"
        )

    return requested_overrides


def _raise_visibility_expansion(
    section_keys: tuple[str, ...],
    requested_overrides: dict[SectionPath, SectionVisibility],
    reason: str,
) -> NoReturn:
    """Raise VisibilityExpansionRequired with the computed overrides."""
    keys_str = ", ".join(section_keys)
    raise VisibilityExpansionRequired(
        f"Model requested expansion of sections: {keys_str}",
        requested_overrides=requested_overrides,
        reason=reason,
        section_keys=section_keys,
    )


def create_open_sections_handler(
    *,
    registry: RegistrySnapshot,
    current_visibility: Mapping[SectionPath, SectionVisibility],
) -> Tool[OpenSectionsParams, None]:
    """Create an open_sections tool bound to the current prompt state.

    The open_sections tool enables progressive disclosure by allowing models
    to request expansion of summarized sections. When invoked, the handler
    validates the requested section keys and raises VisibilityExpansionRequired,
    which signals the caller to re-render the prompt with expanded visibility.

    This tool is automatically registered when a prompt contains sections with
    SUMMARY visibility. Sections with tools attached should use open_sections
    (not read_section) so the tools become available after expansion.

    Args:
        registry: The prompt's section registry snapshot containing all
            registered sections and their metadata.
        current_visibility: Mapping from section paths to their current
            visibility state. Sections already at FULL are silently skipped.

    Returns:
        A Tool instance that raises VisibilityExpansionRequired on invocation.

    Raises:
        PromptValidationError: Via the tool handler if section keys are invalid
            or all requested sections are already expanded.
    """

    def handler(
        params: OpenSectionsParams, *, context: ToolContext
    ) -> ToolResult[None]:
        """Handle open_sections requests by raising VisibilityExpansionRequired."""
        requested_overrides = _validate_section_keys(
            params.section_keys,
            registry=registry,
            current_visibility=current_visibility,
        )
        _raise_visibility_expansion(
            params.section_keys,
            requested_overrides,
            params.reason,
        )

    return Tool[OpenSectionsParams, None](
        name="open_sections",
        description="Expand summarized sections to view their full content.",
        handler=handler,
        accepts_overrides=False,
    )


def create_read_section_handler(
    *,
    registry: RegistrySnapshot,
    current_visibility: Mapping[SectionPath, SectionVisibility],
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass] | None = None,
    session: SessionProtocol | None = None,
) -> Tool[ReadSectionParams, ReadSectionResult]:
    """Create a read_section tool bound to the current prompt state.

    The read_section tool allows models to retrieve the full markdown content
    of a summarized section without permanently changing visibility state.
    Unlike open_sections, this is a read-only operation - the section remains
    summarized in subsequent turns, and no VisibilityExpansionRequired is raised.

    Use read_section for sections that have no tools attached, or when you want
    the model to peek at content without committing to expansion. Use open_sections
    when the section has tools that should become available after expansion.

    Args:
        registry: The prompt's section registry snapshot containing all
            registered sections and their metadata.
        current_visibility: Mapping from section paths to their current
            visibility state. Reading an already-expanded section raises an error.
        param_lookup: Optional mapping from parameter types to parameter instances,
            used to resolve section parameters during rendering.
        session: Optional session for visibility callables that inspect state
            and for checking section enablement conditions.

    Returns:
        A Tool instance that returns ReadSectionResult with rendered markdown.

    Raises:
        PromptValidationError: Via the tool handler if the section key is invalid
            or the section is not currently summarized.
    """
    section_params_lookup = dict(param_lookup or {})

    def handler(
        params: ReadSectionParams, *, context: ToolContext
    ) -> ToolResult[ReadSectionResult]:
        """Handle read_section requests by rendering and returning section content."""
        section_key = params.section_key
        path = _parse_section_key(section_key)

        # Find the section node using O(1) lookup
        target_node = registry.node_by_path.get(path)
        if target_node is None:
            raise PromptValidationError(
                f"Section '{section_key}' does not exist in this prompt."
            )

        # Check section is actually summarized
        effective_visibility = current_visibility.get(path)
        if effective_visibility == SectionVisibility.FULL:
            raise PromptValidationError(f"Section '{section_key}' is not summarized.")

        # Resolve section parameters
        section_params = registry.resolve_section_params(
            target_node, dict(section_params_lookup)
        )

        # Render the section with FULL visibility (but children keep their visibility)
        rendered = target_node.section.render(
            section_params,
            target_node.depth,
            target_node.number,
            path=target_node.path,
            visibility=SectionVisibility.FULL,
        )

        # If the section has children, render them with their current visibility
        child_content = _render_children_for_read(
            target_node,
            registry,
            section_params_lookup,
            current_visibility,
            session,
        )

        if child_content:
            rendered = f"{rendered}\n\n{child_content}"

        return ToolResult(
            message=f"Content of section '{section_key}':",
            value=ReadSectionResult(content=rendered),
            success=True,
        )

    return Tool[ReadSectionParams, ReadSectionResult](
        name="read_section",
        description=(
            "Read the full content of a summarized section without expanding it. "
            "Returns the section's markdown content. Child sections may still be "
            "summarized in the output."
        ),
        handler=handler,
        accepts_overrides=False,
    )


def _render_child_section(
    node: SectionNode[SupportsDataclass],
    registry: RegistrySnapshot,
    section_params: SupportsDataclass,
    effective: SectionVisibility,
) -> tuple[str, bool]:
    """Render a child section and return (rendered_text, should_skip_children)."""
    rendered = node.section.render(
        section_params,
        node.depth,
        node.number,
        path=node.path,
        visibility=effective,
    )

    should_skip = False
    if (
        effective == SectionVisibility.SUMMARY
        and node.section.summary is not None
        and rendered
    ):
        section_key = ".".join(node.path)
        child_keys = _collect_child_keys_for_node(node, registry)
        has_tools = bool(node.section.tools())
        suffix = build_summary_suffix(section_key, child_keys, has_tools=has_tools)
        rendered += suffix
        should_skip = True

    return rendered, should_skip


def _should_skip_node(
    node: SectionNode[SupportsDataclass], skip_depth: int | None
) -> tuple[bool, int | None]:
    """Check if node should be skipped based on skip_depth."""
    if skip_depth is None:
        return False, None
    if node.depth > skip_depth:
        return True, skip_depth
    return False, None


def _iterate_child_nodes(
    parent_node: SectionNode[SupportsDataclass],
    registry: RegistrySnapshot,
) -> Iterator[SectionNode[SupportsDataclass]]:
    """Iterate over child nodes of a parent, stopping when leaving subtree."""
    parent_depth = parent_node.depth
    in_children = False
    for node in registry.sections:
        if node is parent_node:
            in_children = True
            continue
        if not in_children:
            continue
        if node.depth <= parent_depth:
            break
        yield node


def _render_children_for_read(
    parent_node: SectionNode[SupportsDataclass],
    registry: RegistrySnapshot,
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
    current_visibility: Mapping[SectionPath, SectionVisibility],
    session: SessionProtocol | None,
) -> str:
    """Render child sections for read_section, respecting their visibility."""
    rendered_children: list[str] = []
    skip_depth: int | None = None

    for node in _iterate_child_nodes(parent_node, registry):
        should_skip, skip_depth = _should_skip_node(node, skip_depth)
        if should_skip:
            continue

        section_params = registry.resolve_section_params(node, dict(param_lookup))
        if (
            section_params is None
        ):  # pragma: no cover - defensive skip for unresolved params
            skip_depth = node.depth
            continue
        if not node.section.is_enabled(section_params, session=session):
            skip_depth = node.depth
            continue

        effective = current_visibility.get(node.path, SectionVisibility.FULL)
        rendered, should_skip_children = _render_child_section(
            node, registry, section_params, effective
        )
        if should_skip_children:
            skip_depth = node.depth

        # render() always returns at least a heading; no need to check for empty
        rendered_children.append(rendered)

    return "\n\n".join(rendered_children)


def _collect_child_keys_for_node(
    parent_node: SectionNode[SupportsDataclass],
    registry: RegistrySnapshot,
) -> tuple[str, ...]:
    """Collect the keys of direct child sections for a parent node.

    Uses precomputed children_by_path index for O(1) lookup.
    """
    return registry.children_by_path.get(parent_node.path, ())


def build_summary_suffix(
    section_key: str,
    child_keys: tuple[str, ...],
    *,
    has_tools: bool = True,
) -> str:
    """Build the instruction suffix appended to summarized sections.

    Creates the bracketed instruction text that tells models how to expand
    a summarized section. The suffix is appended after a horizontal rule
    following the section's summary content.

    Args:
        section_key: The dot-notation key for the summarized section
            (e.g., "tools.filesystem").
        child_keys: Keys of direct child sections that would be revealed
            on expansion. Included in the instruction if non-empty.
        has_tools: Whether the section has tools attached. When True, the suffix
            directs the model to use open_sections (which triggers re-rendering).
            When False, directs to read_section (read-only, returns content).

    Returns:
        Formatted instruction text starting with newlines, horizontal rule,
        and bracketed expansion instructions.

    Example:
        >>> build_summary_suffix("tools", ("tools.fs", "tools.git"), has_tools=True)
        '\\n\\n---\\n[This section is summarized. Call `open_sections` with key "tools" to view full content including subsections: tools.fs, tools.git.]'
    """
    if has_tools:
        # Section has tools - use open_sections to expand and access tools
        base_instruction = (
            f"[This section is summarized. To view full content, "
            f'call `open_sections` with key "{section_key}".]'
        )

        if child_keys:
            children_str = ", ".join(child_keys)
            base_instruction = (
                f"[This section is summarized. Call `open_sections` with key "
                f'"{section_key}" to view full content including subsections: {children_str}.]'
            )
    else:
        # Section has no tools - use read_section for read-only access
        base_instruction = (
            f"[This section is summarized. To view full content, "
            f'call `read_section` with key "{section_key}".]'
        )

        if child_keys:
            children_str = ", ".join(child_keys)
            base_instruction = (
                f"[This section is summarized. Call `read_section` with key "
                f'"{section_key}" to view full content including subsections: {children_str}.]'
            )

    return f"\n\n---\n{base_instruction}"


def has_summarized_sections(
    registry: RegistrySnapshot,
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass] | None = None,
    *,
    session: SessionProtocol | None = None,
) -> bool:
    """Check if the prompt contains any sections that will render as SUMMARY.

    This function determines whether the progressive disclosure tools should be
    registered. It evaluates the effective visibility of each section, taking into
    account base visibility settings, visibility callables, and any overrides stored
    in session state.

    A section is considered summarized only if:
    1. Its effective visibility is SUMMARY
    2. It has a summary defined (summary is not None)

    Visibility overrides are managed exclusively via Session state using the
    VisibilityOverrides state slice.

    Args:
        registry: The prompt's section registry snapshot containing all
            registered sections and their metadata.
        param_lookup: Optional mapping from parameter types to parameter instances,
            used to evaluate visibility selectors that depend on section parameters.
        session: Optional session for visibility callables that inspect state.
            Also used to query VisibilityOverrides from session state.

    Returns:
        True if at least one enabled section renders with SUMMARY visibility
        and has a summary defined. False otherwise.
    """
    params = dict(param_lookup or {})

    for node in registry.sections:
        section_params = registry.resolve_section_params(node, dict(params))
        effective = node.section.effective_visibility(
            None,
            section_params,
            session=session,
            path=node.path,
        )
        if effective == SectionVisibility.SUMMARY and node.section.summary is not None:
            return True

    return False


def compute_current_visibility(
    registry: RegistrySnapshot,
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass] | None = None,
    *,
    session: SessionProtocol | None = None,
) -> dict[SectionPath, SectionVisibility]:
    """Compute the effective visibility for all sections in a prompt.

    Iterates over all registered sections and determines their effective visibility
    by evaluating base visibility settings, visibility callables, and any overrides
    stored in session state. The result is used by progressive disclosure tools to
    track which sections are summarized vs expanded.

    Visibility overrides are managed exclusively via Session state using the
    VisibilityOverrides state slice. Direct visibility_overrides parameters are
    no longer supported.

    Args:
        registry: The prompt's section registry snapshot containing all
            registered sections and their metadata.
        param_lookup: Optional mapping from parameter types to parameter instances,
            used to evaluate visibility selectors that depend on section parameters.
        session: Optional session for visibility callables that inspect state.
            Also used to query VisibilityOverrides from session state.

    Returns:
        Dictionary mapping section paths (tuples of key segments) to their
        effective SectionVisibility (FULL, SUMMARY, or HIDDEN).

    Example:
        >>> visibility = compute_current_visibility(registry, session=session)
        >>> visibility[("tools", "filesystem")]
        <SectionVisibility.SUMMARY: 'summary'>
    """
    params = dict(param_lookup or {})
    result: dict[SectionPath, SectionVisibility] = {}

    for node in registry.sections:
        section_params = registry.resolve_section_params(node, dict(params))
        effective = node.section.effective_visibility(
            None,
            section_params,
            session=session,
            path=node.path,
        )
        result[node.path] = effective

    return result


__all__ = [
    "OpenSectionsParams",
    "ReadSectionParams",
    "ReadSectionResult",
    "VisibilityExpansionRequired",
    "build_summary_suffix",
    "compute_current_visibility",
    "create_open_sections_handler",
    "create_read_section_handler",
    "has_summarized_sections",
]
