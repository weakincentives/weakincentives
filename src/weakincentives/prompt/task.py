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

"""Task abstraction for representing user intent in prompts.

This module provides a canonical way to represent user intent via the ``Task``
base dataclass and ``TaskSection`` component. When progressive disclosure
triggers a re-render, expansion instructions flow through the task object
itself via rebinding.

See specs/TASK_SECTION.md for the complete specification.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import field
from typing import TypeVar, override

from ..dataclasses import FrozenDataclass
from ._types import SupportsDataclass
from .markdown import MarkdownSection
from .section import Section, SectionVisibility, VisibilitySelector


@FrozenDataclass()
class Task:
    """Base dataclass for capturing user intent.

    Subclass this to define domain-specific task types with additional fields.

    Attributes:
        request: The user's request in natural language.
        background: Additional context or background for the request.
        expansion_instructions: Instructions injected after sections are expanded
            via progressive disclosure. Guides the model on how to proceed with
            newly visible content.
    """

    request: str = field(
        metadata={"description": "The user's request in natural language."},
    )
    background: str | None = field(
        default=None,
        metadata={"description": "Additional context or background for the request."},
    )
    expansion_instructions: str | None = field(
        default=None,
        metadata={
            "description": (
                "Instructions injected after sections are expanded via progressive "
                "disclosure. Guides the model on how to proceed with newly visible "
                "content."
            ),
        },
    )


TaskT = TypeVar("TaskT", bound=Task)


@FrozenDataclass()
class _ComputedTaskParams:
    """Internal dataclass holding computed template parameters."""

    request: str
    background: str | None
    expansion_instructions: str | None
    _expansion_block: str
    _background_block: str


def _build_computed_params(
    params: SupportsDataclass | None,
) -> SupportsDataclass | None:
    """Build computed parameters including expansion and background blocks."""
    if params is None:
        return None

    # Extract task fields
    request = getattr(params, "request", "")
    background = getattr(params, "background", None)
    expansion_instructions = getattr(params, "expansion_instructions", None)

    # Build expansion block
    expansion_block = ""
    if expansion_instructions:
        expansion_block = f"**Expansion Context:** {expansion_instructions}\n\n---\n\n"

    # Build background block
    background_block = ""
    if background:
        background_block = f"\n\n**Background:** {background}"

    # Create a params-like object with computed fields
    return _ComputedTaskParams(
        request=request,
        background=background,
        expansion_instructions=expansion_instructions,
        _expansion_block=expansion_block,
        _background_block=background_block,
    )


class TaskSection(MarkdownSection[TaskT]):
    """Section that renders a Task with optional background and expansion context.

    The TaskSection provides a standard way to render user intent with support for:
    - The primary request from the user
    - Optional background context
    - Expansion instructions injected after progressive disclosure

    Example rendered output (initial):

        ## 5. Task

        Review the authentication module for security vulnerabilities.

        **Background:** Follow-up to the Q4 security audit findings.

    Example rendered output (after expansion):

        ## 5. Task

        **Expansion Context:** Sections expanded: `reference-docs`. Reason: Need
        security guidelines. Continue with your task using the newly visible content.

        ---

        Review the authentication module for security vulnerabilities.

        **Background:** Follow-up to the Q4 security audit findings.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        key: str = "task",
        title: str = "Task",
        default_params: TaskT | None = None,
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: Callable[[SupportsDataclass], bool] | None = None,
        tools: Sequence[object] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        """Initialize a TaskSection.

        Args:
            key: Section key for prompt composition. Defaults to "task".
            title: Section title rendered in the prompt. Defaults to "Task".
            default_params: Default Task instance for this section.
            children: Child sections nested under this section.
            enabled: Predicate controlling whether this section renders.
            tools: Tools available within this section.
            accepts_overrides: Whether this section accepts overrides.
            summary: Summary text for progressive disclosure.
            visibility: Visibility selector controlling full/summary rendering.
        """
        # The template uses internal placeholders that are computed during render
        template = "${_expansion_block}${request}${_background_block}"
        super().__init__(
            title=title,
            key=key,
            template=template,
            default_params=default_params,
            children=children,
            enabled=enabled,
            tools=tools,
            accepts_overrides=accepts_overrides,
            summary=summary,
            visibility=visibility,
        )

    @override
    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        visibility: SectionVisibility | None = None,
    ) -> str:
        """Render the task section with expansion and background blocks.

        This override computes the expansion block and background block based on
        the Task's fields before delegating to the parent render method.
        """
        # Build computed fields for the template
        computed_params = _build_computed_params(params)
        return super().render(computed_params, depth, number, visibility=visibility)

    @override
    def placeholder_names(self) -> set[str]:
        """Return the set of placeholder names in the template.

        This override returns the user-facing placeholders, not the internal
        computed ones, for validation purposes.
        """
        return {"request", "background", "expansion_instructions"}


def build_expansion_instructions(
    section_keys: tuple[str, ...],
    reason: str,
) -> str:
    """Build guidance for continuing after expansion.

    Args:
        section_keys: Keys of the sections that were expanded.
        reason: The reason provided by the model for requesting expansion.

    Returns:
        Formatted instructions for the model to continue with the expanded content.
    """
    keys = ", ".join(f"`{k}`" for k in section_keys)
    return (
        f"Sections expanded: {keys}. "
        f"Reason: {reason}. "
        "Continue with your task using the newly visible content."
    )


__all__ = [
    "Task",
    "TaskSection",
    "build_expansion_instructions",
]
