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

"""Task history section for displaying plan state and visibility transitions.

This module provides a section type that automatically renders the current plan
and visibility override state at prompt rendering time. When the prompt includes
summarized sections with tools, it also includes guidance for providing rationale
when expanding sections.
"""

from __future__ import annotations

import textwrap
from dataclasses import field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, override

from ..dataclasses import FrozenDataclass
from ._types import SupportsDataclass
from ._visibility import SectionVisibility
from .errors import SectionPath
from .section import Section

if TYPE_CHECKING:
    from ..runtime.session import Session
    from ..runtime.session.protocols import SessionProtocol


@FrozenDataclass()
class VisibilityTransition:
    """Record of a visibility state transition for a section."""

    section_path: SectionPath
    reason: str
    expanded_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def render(self) -> str:
        """Render the transition as a human-readable string."""
        path_str = ".".join(self.section_path)
        return f"- {path_str}: {self.reason}"


@FrozenDataclass()
class SummarizedSectionWithTools:
    """Information about a summarized section that has tools attached."""

    section_path: SectionPath
    section_title: str
    tool_names: tuple[str, ...]
    is_currently_summarized: bool


def _empty_visibility_overrides() -> dict[SectionPath, SectionVisibility]:
    return {}


@FrozenDataclass()
class TaskHistoryContext:
    """Context for task history tracking stored in session.

    This dataclass captures the visibility override state and transition history
    for progressive disclosure tracking. It is stored in the session and updated
    when sections are expanded.
    """

    visibility_overrides: dict[SectionPath, SectionVisibility] = field(
        default_factory=_empty_visibility_overrides
    )
    transitions: tuple[VisibilityTransition, ...] = field(default_factory=tuple)
    sections_with_tools: tuple[SummarizedSectionWithTools, ...] = field(
        default_factory=tuple
    )


def set_task_history_context(
    session: SessionProtocol,
    *,
    visibility_overrides: dict[SectionPath, SectionVisibility] | None = None,
    sections_with_tools: tuple[SummarizedSectionWithTools, ...] | None = None,
) -> TaskHistoryContext:
    """Update the task history context in the session.

    This function creates or updates the TaskHistoryContext stored in the session.
    It preserves the transition history while allowing updates to the current
    visibility state and sections with tools information.

    Args:
        session: The session to store the context in.
        visibility_overrides: Current visibility overrides for sections.
        sections_with_tools: Information about sections that have tools
            (directly or via children) and can be summarized.

    Returns:
        The updated TaskHistoryContext.
    """
    existing = latest_task_history_context(session)

    new_context = TaskHistoryContext(
        visibility_overrides=(
            visibility_overrides
            if visibility_overrides is not None
            else (existing.visibility_overrides if existing else {})
        ),
        transitions=existing.transitions if existing else (),
        sections_with_tools=(
            sections_with_tools
            if sections_with_tools is not None
            else (existing.sections_with_tools if existing else ())
        ),
    )
    session.mutate(TaskHistoryContext).seed(new_context)
    return new_context


def record_visibility_transition(
    session: SessionProtocol,
    *,
    section_path: SectionPath,
    reason: str,
) -> TaskHistoryContext:
    """Record a visibility transition in the task history.

    This function adds a new transition record when a section is expanded
    from SUMMARY to FULL visibility.

    Args:
        session: The session storing the context.
        section_path: The path of the section being expanded.
        reason: The rationale provided for why expansion was needed.

    Returns:
        The updated TaskHistoryContext.
    """
    existing = latest_task_history_context(session)

    transition = VisibilityTransition(
        section_path=section_path,
        reason=reason,
    )

    new_context = TaskHistoryContext(
        visibility_overrides=existing.visibility_overrides if existing else {},
        transitions=(
            (*existing.transitions, transition) if existing else (transition,)
        ),
        sections_with_tools=existing.sections_with_tools if existing else (),
    )
    session.mutate(TaskHistoryContext).seed(new_context)
    return new_context


def latest_task_history_context(
    session: SessionProtocol,
) -> TaskHistoryContext | None:
    """Return the most recent task history context from the session."""
    return session.query(TaskHistoryContext).latest()


def clear_task_history_context(session: SessionProtocol) -> None:
    """Clear the task history context from the session."""
    session.mutate(TaskHistoryContext).clear(lambda _: True)


_DEFAULT_TITLE = "Task History"
_DEFAULT_KEY = "task-history"

_VISIBILITY_GUIDANCE = textwrap.dedent(
    """
    ### Section Visibility State

    Some sections in this prompt are summarized to reduce context length.
    You can expand them using the `open_sections` tool when you need the full content.

    **Current Summarized Sections with Tools:**
    {summarized_sections}

    **Important:** When calling `open_sections`, you MUST provide a clear rationale
    in the `reason` parameter explaining:
    - Why you need the expanded content
    - What specific information you're looking for
    - How it relates to your current task

    This helps maintain a clear audit trail of state transitions.
    """
).strip()

_TRANSITIONS_HEADER = textwrap.dedent(
    """
    ### Previous Section Expansions

    The following sections were expanded during this session:
    """
).strip()


class TaskHistorySection(Section[SupportsDataclass]):
    """Section that renders current plan state and visibility transitions.

    This section automatically includes at render time:
    - The current plan (objective, steps, status) from the session
    - When summarized sections with tools exist: visibility override state
      and guidance for providing rationale when calling open_sections

    The section queries the session for:
    - Plan: The current execution plan with objective and steps
    - TaskHistoryContext: Visibility override state and transition history

    Example:
        >>> from weakincentives.prompt import TaskHistorySection
        >>> from weakincentives.runtime import Session, InProcessEventBus
        >>>
        >>> bus = InProcessEventBus()
        >>> session = Session(bus=bus)
        >>> section = TaskHistorySection(session=session)
    """

    def __init__(
        self,
        *,
        session: Session,
        title: str = _DEFAULT_TITLE,
        key: str = _DEFAULT_KEY,
        include_visibility_guidance: bool = True,
    ) -> None:
        """Initialize the TaskHistorySection.

        Args:
            session: The session to query for plan and context state.
            title: The section title. Defaults to "Task History".
            key: The section key. Defaults to "task-history".
            include_visibility_guidance: Whether to include visibility
                override guidance when summarized sections with tools exist.
                Defaults to True.
        """
        self._session = session
        self._include_visibility_guidance = include_visibility_guidance
        super().__init__(title=title, key=key, accepts_overrides=False)
        self.params_type = None
        self.param_type = None

    @property
    def session(self) -> Session:
        """Return the session associated with this section."""
        return self._session

    @override
    def original_body_template(self) -> str | None:
        return None

    @override
    def render(
        self,
        params: SupportsDataclass | None,
        depth: int,
        number: str,
        *,
        path: tuple[str, ...] = (),
        visibility: SectionVisibility | None = None,
    ) -> str:
        del params, visibility
        body = self._render_body()
        return self._render_block(body, depth, number, path)

    @override
    def clone(self, **kwargs: object) -> TaskHistorySection:
        from ..runtime.session import Session

        session = kwargs.get("session")
        if not isinstance(session, Session):
            msg = "session is required to clone TaskHistorySection."
            raise TypeError(msg)
        return TaskHistorySection(
            session=session,
            title=self.title,
            key=self.key,
            include_visibility_guidance=self._include_visibility_guidance,
        )

    def _render_body(self) -> str:
        """Render the section body with plan and visibility state."""
        parts: list[str] = []

        # Render plan
        plan_text = self._render_plan()
        if plan_text:
            parts.append(plan_text)

        # Render visibility guidance and state if applicable
        if self._include_visibility_guidance:
            visibility_text = self._render_visibility_state()
            if visibility_text:
                parts.append(visibility_text)

        if not parts:
            return "No task history recorded."

        return "\n\n".join(parts)

    def _render_plan(self) -> str:
        """Render the current plan from the session."""
        # Import here to avoid circular imports
        from ..tools.planning import Plan

        plan = self._session.query(Plan).latest()
        if plan is None:
            return ""

        return plan.render()

    def _render_visibility_state(self) -> str:
        """Render visibility override state and guidance."""
        context = latest_task_history_context(self._session)
        if context is None:
            return ""

        parts: list[str] = []

        # Check if there are summarized sections with tools
        summarized_with_tools = [
            s for s in context.sections_with_tools if s.is_currently_summarized
        ]

        if summarized_with_tools:
            # Render guidance
            summarized_list = "\n".join(
                f"- **{s.section_title}** ({'.'.join(s.section_path)}): tools: {', '.join(s.tool_names)}"
                for s in summarized_with_tools
            )
            guidance = _VISIBILITY_GUIDANCE.format(summarized_sections=summarized_list)
            parts.append(guidance)

        # Render transition history if any
        if context.transitions:
            transitions_text = _TRANSITIONS_HEADER + "\n"
            transitions_text += "\n".join(t.render() for t in context.transitions)
            parts.append(transitions_text)

        return "\n\n".join(parts)

    def _render_block(
        self, body: str, depth: int, number: str, path: tuple[str, ...] = ()
    ) -> str:
        """Render the section with heading."""
        heading_level = "#" * (depth + 2)
        normalized_number = number.rstrip(".")
        path_str = ".".join(path) if path else ""
        title_with_path = (
            f"{self.title.strip()} ({path_str})" if path_str else self.title.strip()
        )
        heading = f"{heading_level} {normalized_number}. {title_with_path}"
        if body:
            return f"{heading}\n\n{body.strip()}"
        return heading


__all__ = [
    "SummarizedSectionWithTools",
    "TaskHistoryContext",
    "TaskHistorySection",
    "VisibilityTransition",
    "clear_task_history_context",
    "latest_task_history_context",
    "record_visibility_transition",
    "set_task_history_context",
]
