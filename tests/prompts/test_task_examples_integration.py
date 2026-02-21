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

"""Integration tests for TaskExamplesSection with PromptTemplate."""

from __future__ import annotations

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    PromptValidationError,
    TaskExample,
    TaskExamplesSection,
    TaskStep,
    Tool,
    ToolExample,
)

from .conftest import (
    AnalysisOutput,
    ReadParams,
    ReadResult,
    SearchParams,
    SearchResult,
)


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
