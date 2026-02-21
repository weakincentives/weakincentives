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

"""Unit tests for TaskExamplesSection and rendering."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptValidationError,
    SectionVisibility,
    TaskExample,
    TaskExamplesSection,
    TaskStep,
    ToolExample,
)

from .conftest import (
    AnalysisOutput,
    ReadParams,
    ReadResult,
    SearchParams,
    SearchResult,
)

# --- TaskExamplesSection tests ---


class TestTaskExamplesSection:
    def test_taskexamplessection_basic_construction(self) -> None:
        example = TaskExample(
            key="example-1",
            objective="Test objective",
            outcome="Test outcome",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read",
                        input=ReadParams(path="x"),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        section = TaskExamplesSection(examples=[example])

        assert section.key == "task-examples"
        assert section.title == "Task Examples"
        assert len(section.children) == 1
        assert section.children[0] is example

    def test_taskexamplessection_custom_key_and_title(self) -> None:
        example = TaskExample(
            key="example-1",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="test",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="x"),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        section = TaskExamplesSection(
            key="custom-examples",
            title="Custom Examples",
            examples=[example],
        )

        assert section.key == "custom-examples"
        assert section.title == "Custom Examples"

    def test_taskexamplessection_validates_empty_examples(self) -> None:
        with pytest.raises(PromptValidationError) as exc:
            TaskExamplesSection(examples=[])

        assert "requires at least one example" in str(exc.value)

    def test_taskexamplessection_validates_example_type(self) -> None:
        # Cast to bypass static type checking for this runtime validation test
        from typing import Any, cast

        bad_example = MarkdownSection(key="wrong", title="Wrong", template="text")
        bad_examples = cast("list[TaskExample[Any]]", [bad_example])

        with pytest.raises(PromptValidationError) as exc:
            TaskExamplesSection(examples=bad_examples)

        assert "must be TaskExample instances" in str(exc.value)
        assert "MarkdownSection" in str(exc.value)

    def test_taskexamplessection_multiple_examples(self) -> None:
        example1 = TaskExample(
            key="example-1",
            objective="First",
            outcome="Done 1",
            steps=[
                TaskStep(
                    tool_name="tool_a",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="a"),
                        output=ReadResult(content="a"),
                    ),
                ),
            ],
        )
        example2 = TaskExample(
            key="example-2",
            objective="Second",
            outcome="Done 2",
            steps=[
                TaskStep(
                    tool_name="tool_b",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="b"),
                        output=ReadResult(content="b"),
                    ),
                ),
            ],
        )

        section = TaskExamplesSection(examples=[example1, example2])

        assert len(section.children) == 2
        assert section.children == (example1, example2)

    def test_taskexamplessection_clone(self) -> None:
        example = TaskExample(
            key="example",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="test",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="x"),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        section = TaskExamplesSection(examples=[example])
        cloned = section.clone()

        assert cloned is not section
        assert cloned.key == section.key
        assert cloned.title == section.title
        assert len(cloned.children) == 1
        assert cloned.children[0] is not section.children[0]


# --- Rendering tests ---


class TestNullRendering:
    """Test rendering with None values."""

    def test_render_step_with_none_input(self) -> None:
        """Test rendering step with None input shows 'null'."""

        @dataclass(slots=True, frozen=True)
        class SimpleResult:
            value: str

        example = TaskExample(
            key="none-render",
            objective="Test null rendering",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="test_tool",
                    example=ToolExample(
                        description="Step with null input",
                        input=None,
                        output=SimpleResult(value="ok"),
                    ),
                ),
            ],
        )

        rendered = example.render(None, depth=0, number="1")

        assert "null" in rendered
        assert '{"value": "ok"}' in rendered


class TestTaskExampleRendering:
    def test_taskexample_render_basic(self) -> None:
        example = TaskExample(
            key="render-test",
            objective="Review code for security issues",
            outcome="Found 2 vulnerabilities",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read the source file",
                        input=ReadParams(path="src/auth.py"),
                        output=ReadResult(content="def authenticate(): ..."),
                    ),
                ),
            ],
        )

        rendered = example.render(None, depth=0, number="1")

        assert "## 1. Review code for security issues" in rendered
        assert "**Objective:** Review code for security issues" in rendered
        assert "**Steps:**" in rendered
        assert "1. **read_file** - Read the source file" in rendered
        assert '{"path": "src/auth.py"}' in rendered
        assert '{"content": "def authenticate(): ..."}' in rendered
        assert "**Outcome:** Found 2 vulnerabilities" in rendered

    def test_taskexample_render_dataclass_outcome(self) -> None:
        """Test that dataclass outcomes render as JSON."""
        example = TaskExample(
            key="dataclass-outcome",
            objective="Test dataclass outcome rendering",
            outcome=AnalysisOutput(summary="Test complete", issues_found=5),
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read the file",
                        input=ReadParams(path="src/main.py"),
                        output=ReadResult(content="content"),
                    ),
                ),
            ],
        )

        rendered = example.render(None, depth=0, number="1")

        assert "**Outcome:**" in rendered
        assert '"summary": "Test complete"' in rendered
        assert '"issues_found": 5' in rendered

    def test_taskexample_render_multiple_steps(self) -> None:
        example = TaskExample(
            key="multi-step",
            objective="Multi-step workflow",
            outcome="Complete",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Step 1",
                        input=ReadParams(path="a.py"),
                        output=ReadResult(content="content a"),
                    ),
                ),
                TaskStep(
                    tool_name="search",
                    example=ToolExample(
                        description="Step 2",
                        input=SearchParams(pattern="test", path="src/"),
                        output=SearchResult(matches=["src/test.py"]),
                    ),
                ),
            ],
        )

        rendered = example.render(None, depth=0, number="1")

        assert "1. **read_file** - Step 1" in rendered
        assert "2. **search** - Step 2" in rendered

    def test_taskexample_render_with_summary_visibility(self) -> None:
        example = TaskExample(
            key="summary-test",
            objective="Test with summary",
            outcome="Done",
            summary="Brief workflow description",
            steps=[
                TaskStep(
                    tool_name="test",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="x"),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        rendered = example.render(
            None, depth=0, number="1", visibility=SectionVisibility.SUMMARY
        )

        assert "Brief workflow description" in rendered
        assert "**Steps:**" not in rendered

    def test_taskexamplessection_render_heading_only(self) -> None:
        example = TaskExample(
            key="example",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="test",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="x"),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        section = TaskExamplesSection(examples=[example])
        rendered = section.render(None, depth=0, number="1")

        assert "## 1. Task Examples" in rendered
