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
    PromptValidationError,
    TaskExample,
    TaskExamplesSection,
    TaskStep,
    Tool,
    ToolContext,
    ToolExample,
    ToolResult,
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
    return ToolResult.ok(ReadResult(content="..."), message="Read file")


def search_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[SearchResult]:
    del context
    return ToolResult.ok(SearchResult(matches=[]), message="Searched")


def report_handler(
    params: ReportParams, *, context: ToolContext
) -> ToolResult[ReportResult]:
    del context
    return ToolResult.ok(ReportResult(issue_id="SEC-1"), message="Reported")


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
