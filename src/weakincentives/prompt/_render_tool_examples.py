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

"""Tool examples rendering for section output."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

from ..serde import dump
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from .tool_result import render_tool_payload

if TYPE_CHECKING:
    from .tool import Tool


def render_tool_examples_block(
    tools: Sequence[Tool[SupportsDataclassOrNone, SupportsToolResult]],
) -> str:
    """Render tool examples as a markdown block.

    Args:
        tools: Sequence of tools to render examples for.

    Returns:
        Rendered markdown string with tool examples, or empty string if no examples.
    """
    rendered: list[str] = []
    for tool in tools:
        if not tool.examples:
            continue

        examples_block = _render_examples_for_tool(tool)
        rendered.append(
            "\n".join(
                [
                    f"- {tool.name}: {tool.description}",
                    textwrap.indent(examples_block, "  "),
                ]
            )
        )

    if not rendered:
        return ""

    return "\n".join(["Tools:", *rendered])


def _render_examples_for_tool(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
) -> str:
    """Render examples for a single tool."""
    lines: list[str] = [f"- {tool.name} examples:"]
    for example in tool.examples:
        rendered_output = _render_example_output(
            example.output, container=tool.result_container
        )
        lines.extend(
            [
                f"  - description: {example.description}",
                "    input:",
                *_render_fenced_block(
                    _render_example_value(example.input),
                    indent="      ",
                    language="json",
                ),
                "    output:",
                *_render_fenced_block(rendered_output, indent="      ", language=None),
            ]
        )
    return "\n".join(lines)


def _render_example_value(value: SupportsDataclass | None) -> str:
    """Serialize example input/output to JSON string."""
    if value is None:
        return "null"

    serialized_value = dump(value, exclude_none=True)

    return json.dumps(serialized_value, ensure_ascii=False)


def _render_example_output(
    value: SupportsToolResult | None,
    *,
    container: Literal["object", "array"],
) -> str:
    """Render example output based on container type."""
    if container == "array":
        sequence_value = cast(Sequence[object], value or [])
        return render_tool_payload(list(sequence_value))

    return render_tool_payload(value)


def _render_fenced_block(
    content: str, *, indent: str, language: str | None
) -> list[str]:
    """Render content inside a fenced code block."""
    fence = "```" if language is None else f"```{language}"
    indented_lines = content.splitlines() or [""]
    return [
        f"{indent}{fence}",
        *[f"{indent}{line}" for line in indented_lines],
        f"{indent}```",
    ]


__all__ = ["render_tool_examples_block"]
