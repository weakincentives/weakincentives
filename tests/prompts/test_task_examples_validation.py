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

"""Tests for array output validation, visibility, conditional rendering, and hierarchy."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    PromptTemplate,
    PromptValidationError,
    SectionVisibility,
    TaskExample,
    TaskExamplesSection,
    TaskStep,
    Tool,
    ToolContext,
    ToolExample,
    ToolResult,
)

from .conftest import (
    ReadParams,
    ReadResult,
    SearchParams,
)


@dataclass(slots=True, frozen=True)
class ArrayItem:
    """Array item for testing array output validation."""

    value: str


def _array_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[list[ArrayItem]]:
    del params, context
    return ToolResult.ok([ArrayItem(value="a")], message="Done")


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


class TestArrayOutputValidationEdgeCases:
    """Additional edge case tests for array output validation."""

    def test_array_output_empty_sequence(self) -> None:
        """Test validation succeeds with empty array output."""
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
            key="array-empty",
            objective="Test empty array",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="array_tool",
                    example=ToolExample(
                        description="Step with empty output",
                        input=SearchParams(pattern="x", path="y"),
                        output=[],  # Empty array
                    ),
                ),
            ],
        )

        # Should succeed - empty arrays are valid (validation only, not rendering)
        template = PromptTemplate(
            ns="test",
            key="empty-array",
            sections=[
                tools_section,
                TaskExamplesSection(examples=[example]),
            ],
        )

        # Validation passed - template created successfully
        assert template is not None

    def test_array_output_correct_types_all_match(self) -> None:
        """Test validation succeeds when all items match expected type."""
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
            key="array-good",
            objective="Test",
            outcome="Done",
            steps=[
                TaskStep(
                    tool_name="array_tool",
                    example=ToolExample(
                        description="Step",
                        input=SearchParams(pattern="x", path="y"),
                        output=[ArrayItem(value="a"), ArrayItem(value="b")],
                    ),
                ),
            ],
        )

        # Should succeed - all items match (validation only, not rendering)
        template = PromptTemplate(
            ns="test",
            key="good-array",
            sections=[
                tools_section,
                TaskExamplesSection(examples=[example]),
            ],
        )

        # Validation passed - template created successfully
        assert template is not None
