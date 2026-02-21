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

"""Shared fixtures and helper types for prompt tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import MagicMock

import pytest

from weakincentives.prompt import (
    MarkdownSection,
    SectionVisibility,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.prompt.registry import PromptRegistry
from weakincentives.prompt.section import Section
from weakincentives.types.dataclass import SupportsDataclass

# --- Helper dataclass types ---


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


# --- Handler functions ---


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


# --- Fixtures ---


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


# --- Progressive disclosure helpers ---


@dataclass
class PDTestParams:
    """Shared parameter type for progressive disclosure tests."""

    name: str = "test"


def make_pd_section(
    *,
    key: str = "test-section",
    summary: str | None = None,
    visibility: SectionVisibility
    | Callable[[PDTestParams], SectionVisibility]
    | Callable[[], SectionVisibility] = SectionVisibility.FULL,
) -> MarkdownSection[PDTestParams]:
    """Create a MarkdownSection for progressive disclosure tests."""
    return MarkdownSection[PDTestParams](
        title="Test Section",
        template="Content: ${name}",
        key=key,
        summary=summary,
        visibility=visibility,
        default_params=PDTestParams(),
    )


def make_pd_registry(
    sections: tuple[MarkdownSection[PDTestParams], ...],
) -> PromptRegistry:
    """Create a PromptRegistry from sections for progressive disclosure tests."""
    registry = PromptRegistry()
    for section in sections:
        registry.register_section(
            cast(Section[SupportsDataclass], section),
            path=(section.key,),
            depth=0,
        )
    return registry


def make_pd_tool_context() -> ToolContext:
    """Create a mock ToolContext for progressive disclosure tests."""
    return ToolContext(
        prompt=MagicMock(),
        rendered_prompt=None,
        adapter=MagicMock(),
        session=MagicMock(),
    )
