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

"""Tests for TaskHistorySection and related helpers."""

from __future__ import annotations

import pytest

from weakincentives.prompt import SectionVisibility
from weakincentives.prompt.task_history import (
    SummarizedSectionWithTools,
    TaskHistoryContext,
    TaskHistorySection,
    VisibilityTransition,
    clear_task_history_context,
    latest_task_history_context,
    record_visibility_transition,
    set_task_history_context,
)
from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import Session
from weakincentives.tools.planning import Plan, PlanStep


def _make_session() -> Session:
    bus = InProcessEventBus()
    return Session(bus=bus)


def _make_section(
    *,
    session: Session | None = None,
    title: str = "Task History",
    key: str = "task-history",
    include_visibility_guidance: bool = True,
) -> TaskHistorySection:
    if session is None:
        session = _make_session()
    return TaskHistorySection(
        session=session,
        title=title,
        key=key,
        include_visibility_guidance=include_visibility_guidance,
    )


class TestTaskHistorySection:
    """Tests for TaskHistorySection rendering."""

    def test_renders_empty_state(self) -> None:
        section = _make_section()
        rendered = section.render(None, depth=0, number="1")

        assert "## 1. Task History" in rendered
        assert "No task history recorded" in rendered

    def test_renders_plan_when_present(self) -> None:
        session = _make_session()
        session.mutate(Plan).seed(
            Plan(
                objective="Complete the project",
                status="active",
                steps=(
                    PlanStep(step_id=1, title="Research", status="done"),
                    PlanStep(step_id=2, title="Implement", status="in_progress"),
                    PlanStep(step_id=3, title="Test", status="pending"),
                ),
            )
        )
        section = _make_section(session=session)

        rendered = section.render(None, depth=0, number="1")

        assert "Complete the project" in rendered
        assert "active" in rendered
        assert "Research" in rendered
        assert "Implement" in rendered
        assert "Test" in rendered
        assert "[done]" in rendered
        assert "[in_progress]" in rendered
        assert "[pending]" in rendered

    def test_renders_plan_without_steps(self) -> None:
        session = _make_session()
        session.mutate(Plan).seed(
            Plan(
                objective="Start something",
                status="active",
                steps=(),
            )
        )
        section = _make_section(session=session)

        rendered = section.render(None, depth=0, number="1")

        assert "Start something" in rendered
        assert "<no steps>" in rendered

    def test_custom_title_and_key(self) -> None:
        section = _make_section(title="My History", key="my-history")

        assert section.title == "My History"
        assert section.key == "my-history"

        rendered = section.render(None, depth=0, number="1")
        assert "My History" in rendered

    def test_renders_at_different_depths(self) -> None:
        section = _make_section()

        depth_0 = section.render(None, depth=0, number="1")
        depth_1 = section.render(None, depth=1, number="1.1")
        depth_2 = section.render(None, depth=2, number="1.1.1")

        assert "## 1." in depth_0
        assert "### 1.1." in depth_1
        assert "#### 1.1.1." in depth_2

    def test_includes_path_in_heading(self) -> None:
        section = _make_section()

        rendered = section.render(
            None, depth=0, number="1", path=("parent", "task-history")
        )

        assert "Task History (parent.task-history)" in rendered

    def test_session_property(self) -> None:
        session = _make_session()
        section = _make_section(session=session)

        assert section.session is session

    def test_original_body_template_returns_none(self) -> None:
        section = _make_section()
        assert section.original_body_template() is None

    def test_render_block_with_empty_body(self) -> None:
        """Test that _render_block handles empty body correctly."""
        section = _make_section()
        # Access the private method to test the edge case where body is empty
        result = section._render_block("", depth=0, number="1", path=())

        # Should return just the heading without extra newlines
        assert result == "## 1. Task History"
        assert "\n\n" not in result


class TestTaskHistoryVisibilityState:
    """Tests for visibility state rendering."""

    def test_renders_visibility_guidance_when_summarized_sections_exist(self) -> None:
        session = _make_session()
        set_task_history_context(
            session,
            sections_with_tools=(
                SummarizedSectionWithTools(
                    section_path=("tools", "planning"),
                    section_title="Planning Tools",
                    tool_names=("setup_plan", "add_step"),
                    is_currently_summarized=True,
                ),
            ),
        )
        section = _make_section(session=session)

        rendered = section.render(None, depth=0, number="1")

        assert "Section Visibility State" in rendered
        assert "Planning Tools" in rendered
        assert "tools.planning" in rendered
        assert "setup_plan" in rendered
        assert "open_sections" in rendered
        assert "rationale" in rendered or "reason" in rendered

    def test_no_visibility_guidance_when_no_summarized_sections(self) -> None:
        session = _make_session()
        set_task_history_context(
            session,
            sections_with_tools=(
                SummarizedSectionWithTools(
                    section_path=("tools",),
                    section_title="Tools",
                    tool_names=("tool1",),
                    is_currently_summarized=False,
                ),
            ),
        )
        section = _make_section(session=session)

        rendered = section.render(None, depth=0, number="1")

        assert "Section Visibility State" not in rendered

    def test_visibility_guidance_can_be_disabled(self) -> None:
        session = _make_session()
        set_task_history_context(
            session,
            sections_with_tools=(
                SummarizedSectionWithTools(
                    section_path=("tools",),
                    section_title="Tools",
                    tool_names=("tool1",),
                    is_currently_summarized=True,
                ),
            ),
        )
        section = _make_section(session=session, include_visibility_guidance=False)

        rendered = section.render(None, depth=0, number="1")

        assert "Section Visibility State" not in rendered

    def test_renders_transition_history(self) -> None:
        session = _make_session()
        record_visibility_transition(
            session,
            section_path=("context", "details"),
            reason="Need detailed information about the context",
        )
        record_visibility_transition(
            session,
            section_path=("tools", "vfs"),
            reason="Looking for file operations",
        )
        section = _make_section(session=session)

        rendered = section.render(None, depth=0, number="1")

        assert "Previous Section Expansions" in rendered
        assert "context.details" in rendered
        assert "Need detailed information" in rendered
        assert "tools.vfs" in rendered
        assert "Looking for file operations" in rendered

    def test_combines_plan_and_visibility_state(self) -> None:
        session = _make_session()
        session.mutate(Plan).seed(
            Plan(
                objective="Review documents",
                status="active",
                steps=(PlanStep(step_id=1, title="Read files", status="pending"),),
            )
        )
        set_task_history_context(
            session,
            sections_with_tools=(
                SummarizedSectionWithTools(
                    section_path=("vfs",),
                    section_title="VFS Tools",
                    tool_names=("read_file",),
                    is_currently_summarized=True,
                ),
            ),
        )
        section = _make_section(session=session)

        rendered = section.render(None, depth=0, number="1")

        assert "Review documents" in rendered
        assert "Read files" in rendered
        assert "VFS Tools" in rendered
        assert "Section Visibility State" in rendered


class TestTaskHistoryContextHelpers:
    """Tests for task history context helper functions."""

    def test_set_context_creates_new_context(self) -> None:
        session = _make_session()

        context = set_task_history_context(
            session,
            visibility_overrides={("section",): SectionVisibility.FULL},
            sections_with_tools=(
                SummarizedSectionWithTools(
                    section_path=("tools",),
                    section_title="Tools",
                    tool_names=("t1",),
                    is_currently_summarized=True,
                ),
            ),
        )

        assert context.visibility_overrides == {("section",): SectionVisibility.FULL}
        assert len(context.sections_with_tools) == 1

    def test_set_context_preserves_transitions(self) -> None:
        session = _make_session()
        record_visibility_transition(session, section_path=("a",), reason="reason a")

        set_task_history_context(
            session,
            visibility_overrides={("b",): SectionVisibility.FULL},
        )

        context = latest_task_history_context(session)
        assert context is not None
        assert len(context.transitions) == 1
        assert context.transitions[0].section_path == ("a",)

    def test_latest_context_returns_none_when_empty(self) -> None:
        session = _make_session()

        context = latest_task_history_context(session)

        assert context is None

    def test_latest_context_returns_stored_context(self) -> None:
        session = _make_session()
        set_task_history_context(
            session, visibility_overrides={("x",): SectionVisibility.SUMMARY}
        )

        context = latest_task_history_context(session)

        assert context is not None
        assert ("x",) in context.visibility_overrides

    def test_record_transition_adds_to_history(self) -> None:
        session = _make_session()

        record_visibility_transition(
            session, section_path=("first",), reason="first reason"
        )
        record_visibility_transition(
            session, section_path=("second",), reason="second reason"
        )

        context = latest_task_history_context(session)
        assert context is not None
        assert len(context.transitions) == 2
        assert context.transitions[0].section_path == ("first",)
        assert context.transitions[1].section_path == ("second",)

    def test_clear_context_removes_all_data(self) -> None:
        session = _make_session()
        set_task_history_context(
            session, visibility_overrides={("x",): SectionVisibility.FULL}
        )
        record_visibility_transition(session, section_path=("y",), reason="reason")

        clear_task_history_context(session)

        context = latest_task_history_context(session)
        assert context is None


class TestVisibilityTransition:
    """Tests for VisibilityTransition dataclass."""

    def test_transition_renders_correctly(self) -> None:
        transition = VisibilityTransition(
            section_path=("context", "background"),
            reason="Need background information",
        )

        rendered = transition.render()

        assert "context.background" in rendered
        assert "Need background information" in rendered

    def test_transition_has_timestamp(self) -> None:
        transition = VisibilityTransition(
            section_path=("test",),
            reason="test",
        )

        assert transition.expanded_at is not None


class TestSummarizedSectionWithTools:
    """Tests for SummarizedSectionWithTools dataclass."""

    def test_stores_section_info(self) -> None:
        info = SummarizedSectionWithTools(
            section_path=("planning", "tools"),
            section_title="Planning Tools",
            tool_names=("setup_plan", "add_step", "update_step"),
            is_currently_summarized=True,
        )

        assert info.section_path == ("planning", "tools")
        assert info.section_title == "Planning Tools"
        assert info.tool_names == ("setup_plan", "add_step", "update_step")
        assert info.is_currently_summarized is True


class TestTaskHistorySectionClone:
    """Tests for TaskHistorySection cloning."""

    def test_clone_requires_session(self) -> None:
        section = _make_section()

        with pytest.raises(TypeError, match="session is required"):
            section.clone()

    def test_clone_with_new_session(self) -> None:
        original_session = _make_session()
        new_session = _make_session()
        section = _make_section(
            session=original_session,
            title="Custom Title",
            key="custom-key",
            include_visibility_guidance=False,
        )

        cloned = section.clone(session=new_session)

        assert cloned.session is new_session
        assert cloned.title == "Custom Title"
        assert cloned.key == "custom-key"
        assert cloned._include_visibility_guidance is False

    def test_clone_preserves_behavior(self) -> None:
        original_session = _make_session()
        original_session.mutate(Plan).seed(
            Plan(objective="Test", status="active", steps=())
        )
        new_session = _make_session()
        new_session.mutate(Plan).seed(
            Plan(objective="Different", status="completed", steps=())
        )
        section = _make_section(session=original_session)

        cloned = section.clone(session=new_session)
        original_rendered = section.render(None, depth=0, number="1")
        cloned_rendered = cloned.render(None, depth=0, number="1")

        assert "Test" in original_rendered
        assert "Different" in cloned_rendered


class TestTaskHistoryContextDataclass:
    """Tests for TaskHistoryContext dataclass."""

    def test_default_values(self) -> None:
        context = TaskHistoryContext()

        assert context.visibility_overrides == {}
        assert context.transitions == ()
        assert context.sections_with_tools == ()

    def test_custom_values(self) -> None:
        transition = VisibilityTransition(section_path=("x",), reason="r")
        section_info = SummarizedSectionWithTools(
            section_path=("y",),
            section_title="Y",
            tool_names=("t",),
            is_currently_summarized=True,
        )
        context = TaskHistoryContext(
            visibility_overrides={("a",): SectionVisibility.FULL},
            transitions=(transition,),
            sections_with_tools=(section_info,),
        )

        assert ("a",) in context.visibility_overrides
        assert len(context.transitions) == 1
        assert len(context.sections_with_tools) == 1
