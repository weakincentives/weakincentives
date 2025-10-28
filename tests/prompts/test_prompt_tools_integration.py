from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from weakincentives.prompts import (
    Prompt,
    Section,
    SupportsDataclass,
    TextSection,
    Tool,
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


def _lookup_handler(params: LookupParams) -> ToolResult[LookupResult]:
    result = LookupResult(
        entity_id=params.entity_id,
        document_url="https://example.com",
    )
    message = f"Fetched entity {result.entity_id}."
    return ToolResult(message=message, payload=result)


def test_prompt_tools_integration_example() -> None:
    lookup_tool = Tool[LookupParams, LookupResult](
        name="lookup_entity",
        description="Fetch structured information for a given entity id.",
        handler=_lookup_handler,
    )

    tools_section = TextSection[ToolDescriptionParams](
        title="Available Tools",
        body="Invoke ${primary_tool} whenever you need fresh entity context.",
        tools=cast(
            list[Tool[SupportsDataclass, SupportsDataclass]],
            [lookup_tool],
        ),
        defaults=ToolDescriptionParams(),
    )

    guidance = TextSection[GuidanceParams](
        title="Guidance",
        body=(
            "Use tools when you need up-to-date context. "
            "Prefer ${primary_tool} for critical lookups."
        ),
        children=cast(
            list[Section[SupportsDataclass]],
            [tools_section],
        ),
    )

    prompt = Prompt(
        name="tools_overview",
        sections=cast(list[Section[SupportsDataclass]], [guidance]),
    )

    markdown = prompt.render(GuidanceParams(primary_tool="lookup_entity"))

    assert markdown == (
        "## Guidance\n\nUse tools when you need up-to-date context. "
        "Prefer lookup_entity for critical lookups.\n\n"
        "### Available Tools\n\nInvoke lookup_entity whenever you need fresh entity context."
    )

    tools = prompt.tools(GuidanceParams(primary_tool="lookup_entity"))

    assert tools == (lookup_tool,)
    assert tools[0].handler is _lookup_handler
    assert tools[0].result_type is LookupResult
