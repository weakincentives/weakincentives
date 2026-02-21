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

"""Unit tests for TaskStep and TaskExample construction and validation."""

from __future__ import annotations

import pytest

from weakincentives.prompt import (
    PromptValidationError,
    TaskExample,
    TaskStep,
    ToolExample,
)

from .conftest import (
    ReadParams,
    ReadResult,
    SearchParams,
    SearchResult,
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
