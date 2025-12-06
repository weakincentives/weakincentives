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

from __future__ import annotations

from dataclasses import dataclass, field

from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolExample,
    ToolResult,
)


@dataclass
class GuidanceParams:
    primary_tool: str


@dataclass
class ToolDescriptionParams:
    primary_tool: str = "lookup_entity"


@dataclass
class LookupParams:
    entity_id: str
    include_related: bool = field(default=False)


@dataclass
class LookupResult:
    entity_id: str
    document_url: str


@dataclass
class LookupResultWithOptional:
    entity_id: str
    document_url: str
    note: str | None = None


@dataclass
class RenderedLookupResult:
    entity_id: str
    description: str

    def render(self) -> str:
        return f"{self.entity_id}: {self.description}"


def _lookup_handler(
    params: LookupParams, *, context: ToolContext
) -> ToolResult[LookupResult]:
    del context
    result = LookupResult(
        entity_id=params.entity_id,
        document_url="https://example.com",
    )
    message = f"Fetched entity {result.entity_id}."
    return ToolResult(message=message, value=result)


def _lookup_handler_with_optional_list(
    params: LookupParams, *, context: ToolContext
) -> ToolResult[list[LookupResultWithOptional]]:
    del context
    result = LookupResultWithOptional(
        entity_id=params.entity_id,
        document_url="https://example.com",
    )
    message = f"Fetched entity {result.entity_id}."
    return ToolResult(message=message, value=[result])


def _lookup_handler_with_render(
    params: LookupParams, *, context: ToolContext
) -> ToolResult[RenderedLookupResult]:
    del context
    result = RenderedLookupResult(
        entity_id=params.entity_id,
        description="Rendered value from handler.",
    )
    return ToolResult(message="Rendered output", value=result)


def test_prompt_tools_integration_example() -> None:
    lookup_tool = Tool[LookupParams, LookupResult](
        name="lookup_entity",
        description="Fetch structured information for a given entity id.",
        handler=_lookup_handler,
    )

    tools_section = MarkdownSection[ToolDescriptionParams](
        title="Available Tools",
        template="Invoke ${primary_tool} whenever you need fresh entity context.",
        key="available-tools",
        tools=[lookup_tool],
        default_params=ToolDescriptionParams(),
    )

    guidance = MarkdownSection[GuidanceParams](
        title="Guidance",
        template=(
            "Use tools when you need up-to-date context. "
            "Prefer ${primary_tool} for critical lookups."
        ),
        key="guidance",
        children=[tools_section],
    )

    prompt_template = PromptTemplate(
        ns="tests/prompts",
        key="tools-overview",
        name="tools_overview",
        sections=[guidance],
    )

    rendered = (
        Prompt(prompt_template)
        .bind(GuidanceParams(primary_tool="lookup_entity"))
        .render()
    )
    markdown = rendered.text

    assert markdown == (
        "## 1. Guidance\n\nUse tools when you need up-to-date context. "
        "Prefer lookup_entity for critical lookups.\n\n"
        "### 1.1. Available Tools\n\nInvoke lookup_entity whenever you need fresh entity context."
    )

    tools = rendered.tools

    assert tools == (lookup_tool,)


def test_prompt_renders_tool_examples_with_rendered_output() -> None:
    lookup_tool = Tool[LookupParams, RenderedLookupResult](
        name="lookup_entity",
        description="Fetch structured information for a given entity id.",
        handler=_lookup_handler_with_render,
        examples=(
            ToolExample[LookupParams, RenderedLookupResult](
                description="Rendered lookup",
                input=LookupParams(entity_id="abc-123", include_related=False),
                output=RenderedLookupResult(
                    entity_id="abc-123",
                    description="Rendered value from handler.",
                ),
            ),
        ),
    )

    tools_section = MarkdownSection[ToolDescriptionParams](
        title="Available Tools",
        template="Invoke ${primary_tool} whenever you need fresh entity context.",
        key="available-tools",
        tools=[lookup_tool],
        default_params=ToolDescriptionParams(),
    )

    template = PromptTemplate(
        ns="tests/prompts",
        key="tools-rendered-output",
        name="tools_rendered_output",
        sections=[tools_section],
    )

    rendered = Prompt(template).render()

    assert rendered.text == (
        "## 1. Available Tools\n\n"
        "Invoke lookup_entity whenever you need fresh entity context.\n\n"
        "Tools:\n"
        "- lookup_entity: Fetch structured information for a given entity id.\n"
        "  - lookup_entity examples:\n"
        "    - description: Rendered lookup\n"
        "      input:\n"
        "        ```json\n"
        '        {"entity_id": "abc-123", "include_related": false}\n'
        "        ```\n"
        "      output:\n"
        "        ```\n"
        "        abc-123: Rendered value from handler.\n"
        "        ```"
    )
    tools = rendered.tools

    assert tools == (lookup_tool,)
    assert tools[0].handler is _lookup_handler_with_render
    assert tools[0].result_type is RenderedLookupResult


def test_prompt_renders_tool_examples_inline() -> None:
    lookup_tool = Tool[LookupParams, list[LookupResultWithOptional]](
        name="lookup_entity",
        description="Fetch structured information for a given entity id.",
        handler=_lookup_handler_with_optional_list,
        examples=(
            ToolExample[LookupParams, list[LookupResultWithOptional]](
                description="Direct lookup with related entities",
                input=LookupParams(entity_id="abc-123", include_related=True),
                output=[
                    LookupResultWithOptional(
                        entity_id="abc-123",
                        document_url="https://example.com/entities/abc-123",
                    )
                ],
            ),
        ),
    )

    tools_section = MarkdownSection[ToolDescriptionParams](
        title="Available Tools",
        template="Invoke ${primary_tool} whenever you need fresh entity context.",
        key="available-tools",
        tools=[lookup_tool],
        default_params=ToolDescriptionParams(),
    )

    guidance = MarkdownSection[GuidanceParams](
        title="Guidance",
        template=(
            "Use tools when you need up-to-date context. "
            "Prefer ${primary_tool} for critical lookups."
        ),
        key="guidance",
        children=[tools_section],
    )

    prompt_template = PromptTemplate(
        ns="tests/prompts",
        key="tools-overview",
        name="tools_overview",
        sections=[guidance],
    )

    rendered = (
        Prompt(prompt_template)
        .bind(GuidanceParams(primary_tool="lookup_entity"))
        .render()
    )

    assert rendered.text == (
        "## 1. Guidance\n\nUse tools when you need up-to-date context. "
        "Prefer lookup_entity for critical lookups.\n\n"
        "### 1.1. Available Tools\n\n"
        "Invoke lookup_entity whenever you need fresh entity context.\n\n"
        "Tools:\n"
        "- lookup_entity: Fetch structured information for a given entity id.\n"
        "  - lookup_entity examples:\n"
        "    - description: Direct lookup with related entities\n"
        "      input:\n"
        "        ```json\n"
        '        {"entity_id": "abc-123", "include_related": true}\n'
        "        ```\n"
        "      output:\n"
        "        ```\n"
        '        [{"entity_id": "abc-123", "document_url": "https://example.com/entities/abc-123"}]\n'
        "        ```"
    )

    tools = rendered.tools

    assert tools == (lookup_tool,)
