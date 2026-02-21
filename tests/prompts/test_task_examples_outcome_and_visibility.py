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

"""Outcome validation, array output, visibility, conditional rendering, and hierarchy tests
for TaskExamplesSection, TaskExample, and TaskStep."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
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


@pytest.fixture
def read_tool() -> Tool[ReadParams, ReadResult]:
    return Tool[ReadParams, ReadResult](
        name="read_file",
        description="Read a file from the workspace.",
        handler=read_handler,
    )


# --- Outcome type validation tests ---


class TestTaskExamplesOutcomeValidation:
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


# --- Array output validation tests ---


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
