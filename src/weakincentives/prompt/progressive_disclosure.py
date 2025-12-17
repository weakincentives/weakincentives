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

The tool branches based on section content:

- **Tool-bearing sections**: Raises ``VisibilityExpansionRequired`` for re-render
- **Content-only sections**: Writes to ``context/*.md`` and returns file paths

See specs/PROGRESSIVE_DISCLOSURE.md for the complete specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, NoReturn

from ._visibility import SectionVisibility
from .errors import PromptValidationError, SectionPath, VisibilityExpansionRequired
from .tool import Tool, ToolContext
from .tool_result import ToolResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Set

    from ..contrib.tools.filesystem import Filesystem
    from ..runtime.session.protocols import SessionProtocol
    from ..types.dataclass import SupportsDataclass
    from .registry import RegistrySnapshot, SectionNode

#: Folder where context files are written for content-only sections.
CONTEXT_FOLDER: Final[str] = "context"


@dataclass(slots=True, frozen=True)
class OpenSectionsParams:
    """Parameters for the open_sections tool."""

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
class OpenSectionsResult:
    """Result from opening content-only sections."""

    written_files: tuple[str, ...] = field(
        metadata={
            "description": (
                "Paths to context files written. Read these files to view "
                "the expanded section content."
            ),
        },
    )


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

    Returns:
        Mapping from section paths to FULL visibility.

    Raises:
        PromptValidationError: If any key is invalid or already expanded.
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
            raise PromptValidationError(f"Section '{key}' is already expanded.")

        requested_overrides[path] = SectionVisibility.FULL

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


def _is_descendant(candidate: SectionPath, parent: SectionPath) -> bool:
    """Check if candidate path is a descendant of parent path."""
    return len(candidate) > len(parent) and candidate[: len(parent)] == parent


def section_has_tools(
    section_path: SectionPath,
    registry: RegistrySnapshot,
) -> bool:
    """Check if a section or any of its descendants exposes tools.

    Used by rendering to determine appropriate summary suffix messaging.

    Args:
        section_path: Path tuple identifying the section.
        registry: Registry snapshot to query.

    Returns:
        True if the section or any descendant has tools.
    """
    for node in registry.sections:
        # Check if this node is the target or a descendant and has tools
        is_target_or_descendant = node.path == section_path or _is_descendant(
            node.path, section_path
        )
        if is_target_or_descendant and node.section.tools():
            return True
    return False


def _find_section_node(
    section_path: SectionPath,
    registry: RegistrySnapshot,
) -> SectionNode[SupportsDataclass] | None:
    """Find a section node by its path.

    Args:
        section_path: Path tuple identifying the section.
        registry: Registry snapshot to search.

    Returns:
        The matching section node, or None if not found.
    """
    for node in registry.sections:
        if node.path == section_path:
            return node
    return None


def _write_section_context(
    section_path: SectionPath,
    registry: RegistrySnapshot,
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
    filesystem: Filesystem,
) -> str:
    """Render section content and write to context folder.

    Args:
        section_path: Path tuple identifying the section.
        registry: Registry snapshot with section tree.
        param_lookup: Parameters for rendering.
        filesystem: Filesystem to write to.

    Returns:
        The file path where content was written.

    Raises:
        PromptValidationError: If section is not found.
    """
    node = _find_section_node(section_path, registry)
    if node is None:
        key_str = ".".join(section_path)
        raise PromptValidationError(f"Section '{key_str}' not found.")

    # Resolve params for this section
    section_params = registry.resolve_section_params(node, dict(param_lookup))

    # Render section content with FULL visibility, standalone formatting
    content = node.section.render(
        section_params,
        depth=0,  # Top-level heading for standalone file
        number="",  # No numbering for standalone file
        path=node.path,
        visibility=SectionVisibility.FULL,
    )

    # Build file path: context/<dot-notation-key>.md
    file_key = ".".join(section_path)
    file_path = f"{CONTEXT_FOLDER}/{file_key}.md"

    # Ensure context directory exists and write
    filesystem.mkdir(CONTEXT_FOLDER, parents=True, exist_ok=True)
    _ = filesystem.write(file_path, content, mode="overwrite")

    return file_path


def create_open_sections_handler(
    *,
    registry: RegistrySnapshot,
    current_visibility: Mapping[SectionPath, SectionVisibility],
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass] | None = None,
) -> Tool[OpenSectionsParams, OpenSectionsResult | None]:
    """Create an open_sections tool bound to the current prompt state.

    The handler branches based on whether requested sections contain tools:

    - **Tool-bearing sections**: Raises ``VisibilityExpansionRequired`` for re-render
    - **Content-only sections**: Writes to ``context/*.md`` and returns file paths

    If a request contains both tool-bearing and content-only sections, the
    exception path is used for all sections to ensure consistent state.

    Args:
        registry: The prompt's section registry snapshot.
        current_visibility: Current visibility state for all sections.
        param_lookup: Parameters for rendering sections to context files.

    Returns:
        A Tool instance configured for progressive disclosure.
    """
    # Freeze param_lookup for closure
    frozen_params: dict[type[SupportsDataclass], SupportsDataclass] = dict(
        param_lookup or {}
    )

    def handler(
        params: OpenSectionsParams, *, context: ToolContext
    ) -> ToolResult[OpenSectionsResult | None]:
        """Handle open_sections requests.

        Branches based on tool presence:
        - Sections with tools raise VisibilityExpansionRequired
        - Content-only sections write to context/*.md
        """
        # Validate all section keys first
        requested_overrides = _validate_section_keys(
            params.section_keys,
            registry=registry,
            current_visibility=current_visibility,
        )

        # Partition sections: those with tools vs content-only
        sections_with_tools: dict[SectionPath, SectionVisibility] = {}
        content_only_sections: list[SectionPath] = []

        for path, visibility in requested_overrides.items():
            if section_has_tools(path, registry):
                sections_with_tools[path] = visibility
            else:
                content_only_sections.append(path)

        # If any section has tools, raise for re-render (all sections)
        # This ensures consistent state and avoids partial writes
        if sections_with_tools:
            _raise_visibility_expansion(
                params.section_keys,
                requested_overrides,
                params.reason,
            )

        # All sections are content-only: write to context files
        if context.filesystem is None:
            return ToolResult(
                message="Cannot write context files: no filesystem available.",
                value=None,
                success=False,
            )

        written_files: list[str] = []
        for path in content_only_sections:
            try:
                file_path = _write_section_context(
                    path,
                    registry,
                    frozen_params,
                    context.filesystem,
                )
                written_files.append(file_path)
            except Exception as exc:
                key_str = ".".join(path)
                return ToolResult(
                    message=f"Failed to write context for '{key_str}': {exc}",
                    value=None,
                    success=False,
                )

        # Return success with file paths
        files_list = ", ".join(written_files)
        return ToolResult(
            message=(
                f"Section content written to: {files_list}. "
                "Read these files to view the expanded content."
            ),
            value=OpenSectionsResult(written_files=tuple(written_files)),
            success=True,
        )

    return Tool[OpenSectionsParams, OpenSectionsResult | None](
        name="open_sections",
        description="Expand summarized sections to view their full content.",
        handler=handler,
        accepts_overrides=False,
    )


def build_summary_suffix(
    section_key: str,
    child_keys: tuple[str, ...],
    *,
    has_tools: bool = False,
) -> str:
    """Build the instruction suffix appended to summarized sections.

    The suffix message varies based on whether the section has tools:

    - **Tool-bearing sections**: Indicate that tools will become available
    - **Content-only sections**: Indicate that content will be written to a file

    Args:
        section_key: The dot-notation key for the summarized section.
        child_keys: Keys of child sections that would be revealed on expansion.
        has_tools: Whether this section or its children have tools.

    Returns:
        Formatted instruction text for the model.
    """
    if has_tools and child_keys:
        # Tool-bearing section with children
        children_str = ", ".join(child_keys)
        base_instruction = (
            f"[This section is summarized. Call `open_sections` with key "
            f'"{section_key}" to view full content including subsections: '
            f"{children_str}. Additional tools may become available.]"
        )
    elif has_tools:
        # Tool-bearing section without children
        base_instruction = (
            f"[This section is summarized. To view full content and access "
            f'additional tools, call `open_sections` with key "{section_key}".]'
        )
    elif child_keys:
        # Content-only section with children
        children_str = ", ".join(child_keys)
        base_instruction = (
            f"[This section is summarized. Call `open_sections` with key "
            f'"{section_key}" to write content (including subsections: '
            f"{children_str}) to {CONTEXT_FOLDER}/{section_key}.md.]"
        )
    else:
        # Content-only section without children
        base_instruction = (
            f"[This section is summarized. To view full content, call "
            f'`open_sections` with key "{section_key}". The content will be '
            f"written to {CONTEXT_FOLDER}/{section_key}.md for you to read.]"
        )

    return f"\n\n---\n{base_instruction}"


def has_summarized_sections(
    registry: RegistrySnapshot,
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass] | None = None,
    *,
    session: SessionProtocol | None = None,
) -> bool:
    """Check if the prompt contains any sections that will render as SUMMARY.

    Visibility overrides are managed exclusively via Session state using the
    VisibilityOverrides state slice.

    Args:
        registry: The prompt's section registry snapshot.
        param_lookup: Optional parameter lookup used to evaluate visibility
            selectors that depend on section parameters.
        session: Optional session for visibility callables that inspect state.
            Also used to query VisibilityOverrides from session state.

    Returns:
        True if at least one section renders with SUMMARY visibility.
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
    """Compute the effective visibility for all sections.

    Visibility overrides are managed exclusively via Session state using the
    VisibilityOverrides state slice.

    Args:
        registry: The prompt's section registry snapshot.
        param_lookup: Optional parameter lookup used to evaluate visibility
            selectors that depend on section parameters.
        session: Optional session for visibility callables that inspect state.
            Also used to query VisibilityOverrides from session state.

    Returns:
        Mapping from section paths to their effective visibility.
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
    "CONTEXT_FOLDER",
    "OpenSectionsParams",
    "OpenSectionsResult",
    "VisibilityExpansionRequired",
    "build_summary_suffix",
    "compute_current_visibility",
    "create_open_sections_handler",
    "has_summarized_sections",
    "section_has_tools",
]
