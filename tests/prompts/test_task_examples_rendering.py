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

"""Rendering and integration tests for TaskExamplesSection, TaskExample, and TaskStep."""

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
