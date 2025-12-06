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

"""Tests for Task dataclass and TaskSection component."""

from __future__ import annotations

from dataclasses import dataclass, replace

from weakincentives.dataclasses import FrozenDataclass
from weakincentives.prompt import (
    PromptTemplate,
    SectionVisibility,
    Task,
    TaskSection,
    VisibilityExpansionRequired,
    build_expansion_instructions,
)

# Tests for Task dataclass


def test_task_request_field_required() -> None:
    """Task requires a request field."""
    task = Task(request="Review the code")
    assert task.request == "Review the code"


def test_task_background_defaults_to_none() -> None:
    """Background field defaults to None."""
    task = Task(request="Review")
    assert task.background is None


def test_task_expansion_instructions_defaults_to_none() -> None:
    """Expansion instructions defaults to None."""
    task = Task(request="Review")
    assert task.expansion_instructions is None


def test_task_with_all_fields() -> None:
    """Task accepts all fields."""
    task = Task(
        request="Review the auth module",
        background="Follow-up to Q4 audit",
        expansion_instructions="Sections expanded: `docs`. Continue.",
    )
    assert task.request == "Review the auth module"
    assert task.background == "Follow-up to Q4 audit"
    assert task.expansion_instructions == "Sections expanded: `docs`. Continue."


def test_task_update_with_replace() -> None:
    """dataclasses.replace works with Task."""
    task = Task(request="Original")
    updated = replace(task, request="Modified")

    assert task.request == "Original"
    assert updated.request == "Modified"
    assert task is not updated


def test_task_replace_with_expansion_instructions() -> None:
    """dataclasses.replace works with Task for expansion instructions."""
    task = Task(request="Review code")
    updated = replace(task, expansion_instructions="Sections expanded.")

    assert task.expansion_instructions is None
    assert updated.expansion_instructions == "Sections expanded."


# Tests for domain-specific Task subclass


@FrozenDataclass()
class CodeReviewTask(Task):
    """Test subclass for code review tasks."""

    files: tuple[str, ...] | None = None
    focus: str | None = None


def test_task_subclass_inherits_fields() -> None:
    """Task subclass inherits base fields."""
    task = CodeReviewTask(request="Review PR #123")
    assert task.request == "Review PR #123"
    assert task.background is None
    assert task.expansion_instructions is None


def test_task_subclass_accepts_custom_fields() -> None:
    """Task subclass accepts custom fields."""
    task = CodeReviewTask(
        request="Review",
        files=("auth.py", "login.py"),
        focus="security",
    )
    assert task.files == ("auth.py", "login.py")
    assert task.focus == "security"


def test_task_subclass_replace_preserves_type() -> None:
    """Replace on subclass preserves custom fields."""
    task = CodeReviewTask(request="Review", files=("main.py",))
    updated = replace(task, expansion_instructions="Continue.")

    assert isinstance(updated, CodeReviewTask)
    assert updated.files == ("main.py",)
    assert updated.expansion_instructions == "Continue."


# Tests for TaskSection


def test_task_section_renders_request_only() -> None:
    """TaskSection renders just the request when no background/expansion."""
    section = TaskSection[Task](title="Task", key="task")
    task = Task(request="Review the authentication module.")

    rendered = section.render(task, depth=0, number="1")

    assert "## 1. Task" in rendered
    assert "Review the authentication module." in rendered
    assert "**Background:**" not in rendered
    assert "**Expansion Context:**" not in rendered


def test_task_section_renders_with_background() -> None:
    """TaskSection renders background when provided."""
    section = TaskSection[Task](title="Task", key="task")
    task = Task(
        request="Review the code.",
        background="Follow-up to security audit.",
    )

    rendered = section.render(task, depth=0, number="1")

    assert "Review the code." in rendered
    assert "**Background:** Follow-up to security audit." in rendered


def test_task_section_renders_with_expansion_instructions() -> None:
    """TaskSection renders expansion context when provided."""
    section = TaskSection[Task](title="Task", key="task")
    task = Task(
        request="Review the code.",
        expansion_instructions="Sections expanded: `docs`. Reason: Need guidelines. Continue with your task using the newly visible content.",
    )

    rendered = section.render(task, depth=0, number="1")

    assert "**Expansion Context:**" in rendered
    assert "Sections expanded: `docs`" in rendered
    assert "---" in rendered
    assert "Review the code." in rendered


def test_task_section_renders_all_fields() -> None:
    """TaskSection renders all fields together."""
    section = TaskSection[Task](title="Task", key="task")
    task = Task(
        request="Review the auth module.",
        background="Q4 audit follow-up.",
        expansion_instructions="Sections expanded: `reference-docs`. Continue.",
    )

    rendered = section.render(task, depth=0, number="1")

    assert (
        "**Expansion Context:** Sections expanded: `reference-docs`. Continue."
        in rendered
    )
    assert "---" in rendered
    assert "Review the auth module." in rendered
    assert "**Background:** Q4 audit follow-up." in rendered


def test_task_section_custom_title_and_key() -> None:
    """TaskSection accepts custom title and key."""
    section = TaskSection[Task](title="Review Task", key="review-task")

    assert section.title == "Review Task"
    assert section.key == "review-task"


def test_task_section_integrates_with_prompt_template() -> None:
    """TaskSection works within a PromptTemplate."""

    @dataclass
    class IntroParams:
        intro: str

    from weakincentives.prompt import MarkdownSection

    intro = MarkdownSection[IntroParams](
        title="Introduction",
        template="${intro}",
        key="intro",
    )
    task_section = TaskSection[Task](title="Task", key="task")

    prompt = PromptTemplate(
        ns="test",
        key="task-integration",
        sections=(intro, task_section),
    )

    rendered = prompt.render(
        IntroParams(intro="Welcome"),
        Task(request="Review the code."),
    )

    assert "## 1. Introduction" in rendered.text
    assert "Welcome" in rendered.text
    assert "## 2. Task" in rendered.text
    assert "Review the code." in rendered.text


def test_task_section_with_subclass() -> None:
    """TaskSection works with Task subclasses."""
    section = TaskSection[CodeReviewTask](title="Review Task", key="review-task")
    task = CodeReviewTask(
        request="Review PR #42",
        background="Security audit",
        files=("auth.py",),
        focus="security",
    )

    rendered = section.render(task, depth=0, number="5")

    assert "## 5. Review Task" in rendered
    assert "Review PR #42" in rendered
    assert "**Background:** Security audit" in rendered


# Tests for build_expansion_instructions


def test_build_expansion_instructions_single_key() -> None:
    """Build expansion instructions for a single section key."""
    instructions = build_expansion_instructions(("docs",), "Need documentation")

    assert "Sections expanded: `docs`" in instructions
    assert "Reason: Need documentation" in instructions
    assert "Continue with your task" in instructions


def test_build_expansion_instructions_multiple_keys() -> None:
    """Build expansion instructions for multiple section keys."""
    instructions = build_expansion_instructions(
        ("reference-docs", "api-spec"),
        "Need detailed information",
    )

    assert "`reference-docs`" in instructions
    assert "`api-spec`" in instructions
    assert "Reason: Need detailed information" in instructions


# Tests for VisibilityExpansionRequired with expansion_instructions


def test_visibility_expansion_required_has_expansion_instructions() -> None:
    """VisibilityExpansionRequired includes expansion_instructions attribute."""
    exc = VisibilityExpansionRequired(
        "Expansion needed",
        requested_overrides={("docs",): SectionVisibility.FULL},
        reason="Need details",
        section_keys=("docs",),
        expansion_instructions="Sections expanded: `docs`. Continue.",
    )

    assert exc.expansion_instructions == "Sections expanded: `docs`. Continue."


def test_visibility_expansion_required_expansion_instructions_defaults_none() -> None:
    """VisibilityExpansionRequired defaults expansion_instructions to None."""
    exc = VisibilityExpansionRequired(
        "Expansion needed",
        requested_overrides={("docs",): SectionVisibility.FULL},
        reason="Need details",
        section_keys=("docs",),
    )

    assert exc.expansion_instructions is None


# Tests for integration with evaluation loop pattern


def test_task_rebind_after_expansion() -> None:
    """Task can be rebinded with expansion instructions after VisibilityExpansionRequired."""
    task = CodeReviewTask(request="Review the code")

    # Simulate catching VisibilityExpansionRequired
    expansion_instructions = build_expansion_instructions(
        ("reference-docs",),
        "Need security guidelines",
    )

    # Rebind task with expansion instructions
    updated_task = replace(task, expansion_instructions=expansion_instructions)

    assert updated_task.request == "Review the code"
    assert "Sections expanded: `reference-docs`" in (
        updated_task.expansion_instructions or ""
    )
    assert "Reason: Need security guidelines" in (
        updated_task.expansion_instructions or ""
    )


def test_task_section_render_with_none_params() -> None:
    """TaskSection handles None params gracefully."""
    from weakincentives.prompt.task import _build_computed_params

    # When params is None, _build_computed_params returns None
    result = _build_computed_params(None)
    assert result is None
