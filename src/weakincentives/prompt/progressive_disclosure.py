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
    from collections.abc import Mapping, Set

    from ..runtime.session.protocols import SessionProtocol
    from ..types import SupportsDataclass
    from .registry import RegistrySnapshot


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
    "OpenSectionsParams",
    "VisibilityExpansionRequired",
    "build_summary_suffix",
    "compute_current_visibility",
    "create_open_sections_handler",
    "has_summarized_sections",
]
