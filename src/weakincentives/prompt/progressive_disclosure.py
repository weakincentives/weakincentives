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

For sections that don't register tools, the framework can optionally auto-render
their full content to the VFS at ``/context/{section-key}.md``, avoiding the
``open_sections`` round-trip.

See specs/PROMPTS.md for the complete specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, runtime_checkable

from ._visibility import SectionVisibility
from .errors import PromptValidationError, SectionPath, VisibilityExpansionRequired
from .section import Section
from .tool import Tool, ToolContext
from .tool_result import ToolResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Set

    from ._types import SupportsDataclass
    from .registry import RegistrySnapshot


@runtime_checkable
class WorkspaceSection(Protocol):
    """Protocol for sections that provide a workspace session.

    Both ``VfsToolsSection`` and ``PodmanSandboxSection`` implement this protocol,
    allowing the rendering system to detect and use the session for auto-rendering
    summarized content to the filesystem.
    """

    @property
    def session(self) -> Any:  # noqa: ANN401
        """Return the session bound to this workspace section.

        Returns a ``weakincentives.runtime.Session`` instance. We use ``Any``
        as the return type to avoid circular imports between prompt and runtime.
        """
        ...


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


def create_open_sections_handler(
    *,
    registry: RegistrySnapshot,
    current_visibility: Mapping[SectionPath, SectionVisibility],
) -> Tool[OpenSectionsParams, None]:
    """Create an open_sections tool bound to the current prompt state.

    Args:
        registry: The prompt's section registry snapshot.
        current_visibility: Current visibility state for all sections.

    Returns:
        A Tool instance configured for progressive disclosure.
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


def build_summary_suffix(section_key: str, child_keys: tuple[str, ...]) -> str:
    """Build the instruction suffix appended to summarized sections.

    Args:
        section_key: The dot-notation key for the summarized section.
        child_keys: Keys of child sections that would be revealed on expansion.

    Returns:
        Formatted instruction text for the model.
    """
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

    return f"\n\n---\n{base_instruction}"


def has_summarized_sections(
    registry: RegistrySnapshot,
    visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass] | None = None,
) -> bool:
    """Check if the prompt contains any sections that will render as SUMMARY.

    Args:
        registry: The prompt's section registry snapshot.
        visibility_overrides: Optional visibility overrides applied at render time.
        param_lookup: Optional parameter lookup used to evaluate visibility
            selectors that depend on section parameters.

    Returns:
        True if at least one section renders with SUMMARY visibility.
    """
    overrides = visibility_overrides or {}
    params = dict(param_lookup or {})

    for node in registry.sections:
        section_params = registry.resolve_section_params(node, dict(params))
        effective = node.section.effective_visibility(
            overrides.get(node.path),
            section_params,
        )
        if effective == SectionVisibility.SUMMARY and node.section.summary is not None:
            return True

    return False


def compute_current_visibility(
    registry: RegistrySnapshot,
    visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass] | None = None,
) -> dict[SectionPath, SectionVisibility]:
    """Compute the effective visibility for all sections.

    Args:
        registry: The prompt's section registry snapshot.
        visibility_overrides: Optional visibility overrides.
        param_lookup: Optional parameter lookup used to evaluate visibility
            selectors that depend on section parameters.

    Returns:
        Mapping from section paths to their effective visibility.
    """
    overrides = visibility_overrides or {}
    params = dict(param_lookup or {})
    result: dict[SectionPath, SectionVisibility] = {}

    for node in registry.sections:
        section_params = registry.resolve_section_params(node, dict(params))
        effective = node.section.effective_visibility(
            overrides.get(node.path),
            section_params,
        )
        result[node.path] = effective

    return result


def section_subtree_has_tools(
    section: Section[SupportsDataclass],
) -> bool:
    """Check if a section or any of its descendants register tools.

    Args:
        section: The root section to check.

    Returns:
        True if the section or any descendant has tools registered.
    """
    if section.tools():
        return True
    return any(section_subtree_has_tools(child) for child in section.children)


def compute_vfs_context_path(section_path: SectionPath) -> str:
    """Compute the VFS path for auto-rendered section content.

    Args:
        section_path: The section path tuple (e.g., ("parent", "child")).

    Returns:
        The VFS path string (e.g., "/context/parent.child.md").
    """
    key = ".".join(section_path)
    return f"/context/{key}.md"


def build_vfs_summary_suffix(section_path: SectionPath) -> str:
    """Build the instruction suffix for sections auto-rendered to VFS.

    Args:
        section_path: The section path tuple.

    Returns:
        Formatted instruction text directing the model to the VFS.
    """
    vfs_path = compute_vfs_context_path(section_path)
    return f"\n\n---\n[This section is summarized. Full content is available at `{vfs_path}`.]"


def find_workspace_section(
    registry: RegistrySnapshot,
) -> WorkspaceSection | None:
    """Find a workspace section in the registry.

    Searches for a section implementing the ``WorkspaceSection`` protocol
    (e.g., ``VfsToolsSection`` or ``PodmanSandboxSection``).

    Args:
        registry: The prompt's section registry snapshot.

    Returns:
        The first workspace section found, or None if none exists.
    """
    for node in registry.sections:
        if isinstance(node.section, WorkspaceSection):
            return node.section
    return None


__all__ = [
    "OpenSectionsParams",
    "VisibilityExpansionRequired",
    "WorkspaceSection",
    "build_summary_suffix",
    "build_vfs_summary_suffix",
    "compute_current_visibility",
    "compute_vfs_context_path",
    "create_open_sections_handler",
    "find_workspace_section",
    "has_summarized_sections",
    "section_subtree_has_tools",
]
