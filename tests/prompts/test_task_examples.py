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

"""Unit tests for TaskExamplesSection, TaskExample, and TaskStep."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    PromptValidationError,
    SectionVisibility,
    Tool,
    ToolContext,
    ToolExample,
    ToolResult,
)
from weakincentives.prompt.task_examples import (
    TaskExample,
    TaskExamplesSection,
    TaskStep,
)

# --- Test fixtures and helper types ---


@dataclass(slots=True, frozen=True)
class ReadParams:
    path: str = field(metadata={"description": "File path to read"})


@dataclass(slots=True, frozen=True)
class ReadResult:
    content: str


@dataclass(slots=True, frozen=True)
class SearchParams:
    pattern: str = field(metadata={"description": "Search pattern"})
    path: str = field(metadata={"description": "Directory to search"})


@dataclass(slots=True, frozen=True)
class SearchResult:
    matches: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ReportParams:
    severity: str = field(metadata={"description": "Issue severity"})
    title: str = field(metadata={"description": "Issue title"})


@dataclass(slots=True, frozen=True)
class ReportResult:
    issue_id: str


@dataclass(slots=True, frozen=True)
class AnalysisOutput:
    """Structured output type for testing typed outcomes."""

    summary: str
    issues_found: int


def read_handler(params: ReadParams, *, context: ToolContext) -> ToolResult[ReadResult]:
    del context
    return ToolResult(
        message="Read file", value=ReadResult(content="..."), success=True
    )


def search_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[SearchResult]:
    del context
    return ToolResult(message="Searched", value=SearchResult(matches=[]), success=True)


def report_handler(
    params: ReportParams, *, context: ToolContext
) -> ToolResult[ReportResult]:
    del context
    return ToolResult(
        message="Reported", value=ReportResult(issue_id="SEC-1"), success=True
    )


@pytest.fixture
def read_tool() -> Tool[ReadParams, ReadResult]:
    return Tool[ReadParams, ReadResult](
        name="read_file",
        description="Read a file from the workspace.",
        handler=read_handler,
    )


@pytest.fixture
def search_tool() -> Tool[SearchParams, SearchResult]:
    return Tool[SearchParams, SearchResult](
        name="search",
        description="Search for patterns in files.",
        handler=search_handler,
    )


@pytest.fixture
def report_tool() -> Tool[ReportParams, ReportResult]:
    return Tool[ReportParams, ReportResult](
        name="report_issue",
        description="Report a security issue.",
        handler=report_handler,
    )


# --- TaskStep tests ---


class TestTaskStep:
    def test_taskstep_is_frozen_dataclass(self) -> None:
        step = TaskStep[ReadParams, ReadResult](
            tool_name="read_file",
            example=ToolExample(
                description="Read a file",
                input=ReadParams(path="test.py"),
                output=ReadResult(content="content"),
            ),
        )

        assert step.tool_name == "read_file"
        assert step.example.description == "Read a file"
        assert step.example.input == ReadParams(path="test.py")
        assert step.example.output == ReadResult(content="content")

    def test_taskstep_preserves_generic_types(self) -> None:
        step = TaskStep[SearchParams, SearchResult](
            tool_name="search",
            example=ToolExample(
                description="Search files",
                input=SearchParams(pattern="test", path="src/"),
                output=SearchResult(matches=["src/test.py"]),
            ),
        )

        assert isinstance(step.example.input, SearchParams)
        assert isinstance(step.example.output, SearchResult)


# --- TaskExample tests ---


class TestTaskExample:
    def test_taskexample_basic_construction(self) -> None:
        example = TaskExample(
            key="test-example",
            objective="Test the authentication module",
            outcome="All tests passed",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read auth module",
                        input=ReadParams(path="src/auth.py"),
                        output=ReadResult(content="def authenticate(): ..."),
                    ),
                ),
            ],
        )

        assert example.key == "test-example"
        assert example.objective == "Test the authentication module"
        assert example.outcome == "All tests passed"
        assert len(example.steps) == 1
        assert example.title == "Test the authentication module"

    def test_taskexample_custom_title(self) -> None:
        example = TaskExample(
            key="custom-title",
            objective="A very long objective that would be truncated",
            outcome="Done",
            title="Short Title",
            steps=[
                TaskStep(
                    tool_name="test",
                    example=ToolExample(
                        description="Test step",
                        input=ReadParams(path="x"),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        assert example.title == "Short Title"

    def test_taskexample_title_truncation(self) -> None:
        long_objective = "A" * 100
        example = TaskExample(
            key="truncate",
            objective=long_objective,
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

        assert len(example.title) == 60
        assert example.title.endswith("...")

    def test_taskexample_validates_empty_objective(self) -> None:
        with pytest.raises(PromptValidationError) as exc:
            TaskExample(
                key="bad",
                objective="",
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

        assert "objective must not be empty" in str(exc.value)

    def test_taskexample_validates_objective_too_long(self) -> None:
        with pytest.raises(PromptValidationError) as exc:
            TaskExample(
                key="bad",
                objective="A" * 501,
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

        assert "objective must be <= 500 characters" in str(exc.value)

    def test_taskexample_validates_non_ascii_objective(self) -> None:
        with pytest.raises(PromptValidationError) as exc:
            TaskExample(
                key="bad",
                objective="Test with emoji 🔥",
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

        assert "ASCII" in str(exc.value)

    def test_taskexample_validates_empty_steps(self) -> None:
        with pytest.raises(PromptValidationError) as exc:
            TaskExample(
                key="bad",
                objective="Valid",
                outcome="Done",
                steps=[],
            )

        assert "steps must not be empty" in str(exc.value)

    def test_taskexample_validates_invalid_tool_name_format(self) -> None:
        with pytest.raises(PromptValidationError) as exc:
            TaskExample(
                key="bad",
                objective="Valid",
                outcome="Done",
                steps=[
                    TaskStep(
                        tool_name="Invalid Tool Name",  # Spaces not allowed
                        example=ToolExample(
                            description="Step",
                            input=ReadParams(path="x"),
                            output=ReadResult(content="y"),
                        ),
                    ),
                ],
            )

        assert "Invalid tool name format" in str(exc.value)

    def test_taskexample_validates_step_type(self) -> None:
        """Test that non-TaskStep objects in steps raise an error."""
        # Cast to bypass static type checking for this runtime validation test
        from typing import Any, cast

        bad_step: Any = {"tool_name": "test", "example": {}}
        bad_steps = cast("list[TaskStep[Any, Any]]", [bad_step])

        with pytest.raises(PromptValidationError) as exc:
            TaskExample(
                key="bad-step",
                objective="Valid objective",
                outcome="Done",
                steps=bad_steps,
            )

        assert "must be a TaskStep instance" in str(exc.value)

    def test_taskexample_children_is_empty(self) -> None:
        example = TaskExample(
            key="no-children",
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

        assert example.children == ()

    def test_taskexample_clone(self) -> None:
        original = TaskExample(
            key="clone-test",
            objective="Test cloning",
            outcome="Clone successful",
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

        cloned = original.clone()

        assert cloned is not original
        assert cloned.key == original.key
        assert cloned.objective == original.objective
        assert cloned.outcome == original.outcome
        assert cloned.steps == original.steps


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


# --- Integration with PromptTemplate tests ---


class TestTaskExamplesIntegration:
    def test_prompt_with_task_examples(
        self,
        read_tool: Tool[ReadParams, ReadResult],
        search_tool: Tool[SearchParams, SearchResult],
    ) -> None:
        tools_section = MarkdownSection(
            key="tools",
            title="Available Tools",
            template="Use these tools:",
            tools=[read_tool, search_tool],
        )

        example = TaskExample(
            key="example-1",
            objective="Find patterns in code",
            outcome="Found patterns",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read source",
                        input=ReadParams(path="src/main.py"),
                        output=ReadResult(content="def main(): ..."),
                    ),
                ),
                TaskStep(
                    tool_name="search",
                    example=ToolExample(
                        description="Search for patterns",
                        input=SearchParams(pattern="TODO", path="src/"),
                        output=SearchResult(matches=["src/main.py"]),
                    ),
                ),
            ],
        )

        examples_section = TaskExamplesSection(examples=[example])

        template = PromptTemplate(
            ns="test",
            key="with-examples",
            sections=[tools_section, examples_section],
        )

        rendered = Prompt(template).render()

        assert "Available Tools" in rendered.text
        assert "Task Examples" in rendered.text
        assert "Find patterns in code" in rendered.text

    def test_prompt_validates_unknown_tool(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools available:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="bad-example",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="unknown_tool",  # Not registered
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="x"),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate(
                ns="test",
                key="bad",
                sections=[
                    tools_section,
                    TaskExamplesSection(examples=[example]),
                ],
            )

        assert 'Unknown tool "unknown_tool"' in str(exc.value)
        assert "Available tools:" in str(exc.value)

    def test_prompt_validates_input_type_mismatch(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="type-mismatch",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Step",
                        input=SearchParams(  # Wrong type - should be ReadParams
                            pattern="test", path="src/"
                        ),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate(
                ns="test",
                key="bad",
                sections=[
                    tools_section,
                    TaskExamplesSection(examples=[example]),
                ],
            )

        assert "input type mismatch" in str(exc.value)
        assert "read_file" in str(exc.value)
        assert "ReadParams" in str(exc.value)
        assert "SearchParams" in str(exc.value)

    def test_prompt_validates_output_type_mismatch(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="type-mismatch",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="x"),
                        output=SearchResult(  # Wrong type - should be ReadResult
                            matches=["a.py"]
                        ),
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate(
                ns="test",
                key="bad",
                sections=[
                    tools_section,
                    TaskExamplesSection(examples=[example]),
                ],
            )

        assert "output type mismatch" in str(exc.value)
        assert "read_file" in str(exc.value)

    def test_prompt_validates_input_none_when_expected(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        """Test error when input is None but tool expects params."""
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="none-input",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Step",
                        input=None,
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate(
                ns="test",
                key="bad",
                sections=[
                    tools_section,
                    TaskExamplesSection(examples=[example]),
                ],
            )

        assert "input type mismatch" in str(exc.value)
        assert "Expected: ReadParams" in str(exc.value)
        assert "got: None" in str(exc.value)

    def test_prompt_validates_output_none_when_expected(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        """Test error when output is None but tool expects result."""
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="none-output",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="x"),
                        output=None,
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate(
                ns="test",
                key="bad",
                sections=[
                    tools_section,
                    TaskExamplesSection(examples=[example]),
                ],
            )

        assert "output type mismatch" in str(exc.value)
        assert "Expected: ReadResult" in str(exc.value)
        assert "got: None" in str(exc.value)

    def test_tools_registered_after_examples_still_validate(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        """Test that tool order doesn't matter - validation happens after all registered."""
        example = TaskExample(
            key="example",
            objective="Test",
            outcome="Done",
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

        examples_section = TaskExamplesSection(examples=[example])

        # Tools come AFTER examples section
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        # Should succeed - validation happens after all sections registered
        template = PromptTemplate(
            ns="test",
            key="order-test",
            sections=[examples_section, tools_section],
        )

        rendered = Prompt(template).render()
        assert "Test" in rendered.text

    def test_prompt_string_outcome_for_unstructured_output(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        """String outcome is valid for prompts without structured output."""
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="string-outcome",
            objective="Test string outcome",
            outcome="Task completed successfully",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read file",
                        input=ReadParams(path="test.py"),
                        output=ReadResult(content="content"),
                    ),
                ),
            ],
        )

        # Should succeed - string outcome for unstructured prompt
        template = PromptTemplate(
            ns="test",
            key="string-outcome-test",
            sections=[tools_section, TaskExamplesSection(examples=[example])],
        )

        rendered = Prompt(template).render()
        assert "Task completed successfully" in rendered.text

    def test_prompt_dataclass_outcome_for_unstructured_output_fails(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        """Dataclass outcome is invalid for prompts without structured output."""
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="bad-outcome",
            objective="Test bad outcome",
            outcome=AnalysisOutput(summary="test", issues_found=0),
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read file",
                        input=ReadParams(path="test.py"),
                        output=ReadResult(content="content"),
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate(
                ns="test",
                key="bad-outcome-test",
                sections=[tools_section, TaskExamplesSection(examples=[example])],
            )

        assert "outcome must be a string" in str(exc.value)
        assert "no structured output" in str(exc.value)

    def test_prompt_dataclass_outcome_for_structured_output(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        """Dataclass outcome matching output type is valid for structured prompts."""
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="typed-outcome",
            objective="Test typed outcome",
            outcome=AnalysisOutput(summary="Analysis complete", issues_found=3),
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read file",
                        input=ReadParams(path="test.py"),
                        output=ReadResult(content="content"),
                    ),
                ),
            ],
        )

        # Should succeed - dataclass outcome matches prompt output type
        template = PromptTemplate[AnalysisOutput](
            ns="test",
            key="typed-outcome-test",
            sections=[tools_section, TaskExamplesSection(examples=[example])],
        )

        rendered = Prompt(template).render()
        assert "Analysis complete" in rendered.text
        assert '"issues_found": 3' in rendered.text

    def test_prompt_string_outcome_for_structured_output_fails(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        """String outcome is invalid for prompts with structured output."""
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="bad-string-outcome",
            objective="Test bad string outcome",
            outcome="This should be AnalysisOutput",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read file",
                        input=ReadParams(path="test.py"),
                        output=ReadResult(content="content"),
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate[AnalysisOutput](
                ns="test",
                key="bad-string-outcome-test",
                sections=[tools_section, TaskExamplesSection(examples=[example])],
            )

        assert "outcome type mismatch" in str(exc.value)
        assert "AnalysisOutput" in str(exc.value)

    def test_prompt_wrong_dataclass_outcome_fails(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        """Wrong dataclass type for outcome fails validation."""
        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        example = TaskExample(
            key="wrong-type-outcome",
            objective="Test wrong type outcome",
            outcome=ReadResult(content="wrong type"),  # Should be AnalysisOutput
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Read file",
                        input=ReadParams(path="test.py"),
                        output=ReadResult(content="content"),
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate[AnalysisOutput](
                ns="test",
                key="wrong-type-outcome-test",
                sections=[tools_section, TaskExamplesSection(examples=[example])],
            )

        assert "outcome type mismatch" in str(exc.value)
        assert "AnalysisOutput" in str(exc.value)
        assert "ReadResult" in str(exc.value)


@dataclass(slots=True, frozen=True)
class ArrayItem:
    """Array item for testing array output validation."""

    value: str


def _array_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[list[ArrayItem]]:
    del params, context
    return ToolResult(
        message="Done",
        value=[ArrayItem(value="a")],
        success=True,
    )


class TestArrayOutputValidation:
    """Test array result type validation."""

    def test_array_output_not_sequence(self) -> None:
        """Test error when array output is not a sequence."""
        tool: Tool[SearchParams, list[ArrayItem]] = Tool[SearchParams, list[ArrayItem]](
            name="array_tool",
            description="Tool returning array",
            handler=_array_handler,
        )

        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[tool],
        )

        example = TaskExample(
            key="array-bad",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="array_tool",
                    example=ToolExample(
                        description="Step",
                        input=SearchParams(pattern="x", path="y"),
                        output=ArrayItem(value="not a list"),
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate(
                ns="test",
                key="bad",
                sections=[
                    tools_section,
                    TaskExamplesSection(examples=[example]),
                ],
            )

        assert "output type mismatch" in str(exc.value)
        assert "sequence of ArrayItem" in str(exc.value)

    def test_array_output_wrong_element_type(self) -> None:
        """Test error when array output has wrong element type."""
        tool: Tool[SearchParams, list[ArrayItem]] = Tool[SearchParams, list[ArrayItem]](
            name="array_tool",
            description="Tool returning array",
            handler=_array_handler,
        )

        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[tool],
        )

        example = TaskExample(
            key="array-bad-elem",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="array_tool",
                    example=ToolExample(
                        description="Step",
                        input=SearchParams(pattern="x", path="y"),
                        output=[ReadResult(content="wrong type")],
                    ),
                ),
            ],
        )

        with pytest.raises(PromptValidationError) as exc:
            PromptTemplate(
                ns="test",
                key="bad",
                sections=[
                    tools_section,
                    TaskExamplesSection(examples=[example]),
                ],
            )

        assert "output type mismatch" in str(exc.value)
        assert "sequence of ArrayItem" in str(exc.value)
        assert "got item of type" in str(exc.value)


# --- Progressive disclosure tests ---


class TestTaskExamplesVisibility:
    def test_taskexample_summary_visibility(self) -> None:
        example = TaskExample(
            key="visibility-test",
            objective="Test visibility",
            outcome="Done",
            summary="Brief summary of the workflow",
            visibility=SectionVisibility.SUMMARY,
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

        assert example.visibility == SectionVisibility.SUMMARY
        assert example.summary == "Brief summary of the workflow"

    def test_taskexamplessection_summary_visibility(self) -> None:
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

        section = TaskExamplesSection(
            examples=[example],
            summary="Examples available for reference",
            visibility=SectionVisibility.SUMMARY,
        )

        rendered = section.render(
            None, depth=0, number="1", visibility=SectionVisibility.SUMMARY
        )

        assert "Examples available for reference" in rendered


# --- Conditional rendering tests ---


class TestTaskExamplesConditionalRendering:
    def test_taskexample_enabled_predicate(self) -> None:
        @dataclass
        class ExampleParams:
            show_advanced: bool = False

        example = TaskExample[ExampleParams](
            key="conditional",
            objective="Advanced workflow",
            outcome="Done",
            enabled=lambda params: params.show_advanced,
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

        assert example.is_enabled(ExampleParams(show_advanced=True)) is True
        assert example.is_enabled(ExampleParams(show_advanced=False)) is False


# --- Section hierarchy tests ---


class TestSectionHierarchy:
    def test_taskexample_is_child_of_container(self) -> None:
        example1 = TaskExample(
            key="auth-review",
            objective="Review authentication",
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
        example2 = TaskExample(
            key="perf-audit",
            objective="Audit performance",
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
            key="examples",
            title="Workflow Examples",
            examples=[example1, example2],
        )

        assert section.children == (example1, example2)
        assert section.children[0].key == "auth-review"
        assert section.children[1].key == "perf-audit"

    def test_section_paths_in_prompt(
        self,
        read_tool: Tool[ReadParams, ReadResult],
    ) -> None:
        example = TaskExample(
            key="example-1",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="read_file",
                    example=ToolExample(
                        description="Step",
                        input=ReadParams(path="x"),
                        output=ReadResult(content="y"),
                    ),
                ),
            ],
        )

        tools_section = MarkdownSection(
            key="tools",
            title="Tools",
            template="Tools:",
            tools=[read_tool],
        )

        examples_section = TaskExamplesSection(
            key="workflow-examples",
            examples=[example],
        )

        template = PromptTemplate(
            ns="test",
            key="paths-test",
            sections=[tools_section, examples_section],
        )

        # Verify section paths are registered correctly
        snapshot = template._snapshot
        assert snapshot is not None

        section_paths = snapshot.section_paths
        assert ("tools",) in section_paths
        assert ("workflow-examples",) in section_paths
        assert ("workflow-examples", "example-1") in section_paths
