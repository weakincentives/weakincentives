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

"""Schema transforms for the Codex App Server adapter."""

from __future__ import annotations

from typing import Any, cast

from ...prompt import RenderedPrompt
from ...serde import schema
from .._shared._bridge import BridgedTool


def bridged_tools_to_dynamic_specs(
    tools: tuple[BridgedTool, ...],
) -> list[dict[str, Any]]:
    """Convert BridgedTool list to Codex DynamicToolSpec format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }
        for tool in tools
    ]


def openai_strict_schema(s: dict[str, Any]) -> dict[str, Any]:
    """Adapt a WINK serde schema for OpenAI/Codex structured output.

    OpenAI's structured output requires:
    - ``additionalProperties: false`` on all object types
    - All properties listed in ``required`` when additionalProperties is false

    WINK's ``serde.schema()`` emits ``additionalProperties: true`` by
    default and only marks fields without defaults as required.
    """
    out = dict(s)
    if out.get("type") == "object":
        out["additionalProperties"] = False
        props: dict[str, Any] | None = out.get("properties")
        if isinstance(props, dict):
            out["properties"] = {
                k: openai_strict_schema(cast(dict[str, Any], v))
                if isinstance(v, dict)
                else v
                for k, v in props.items()
            }
            # OpenAI requires all properties in required when
            # additionalProperties is false.
            out["required"] = list(props.keys())

    # Recurse into array items
    if "items" in out and isinstance(out["items"], dict):
        out["items"] = openai_strict_schema(cast(dict[str, Any], out["items"]))

    # Recurse into combinators
    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in out and isinstance(out[combinator], list):
            out[combinator] = [
                openai_strict_schema(cast(dict[str, Any], entry))
                if isinstance(entry, dict)
                else entry
                for entry in out[combinator]
            ]

    # Recurse into schema definitions
    for defs_key in ("$defs", "definitions"):
        if defs_key in out and isinstance(out[defs_key], dict):
            out[defs_key] = {
                k: openai_strict_schema(cast(dict[str, Any], v))
                if isinstance(v, dict)
                else v
                for k, v in out[defs_key].items()
            }

    return out


def build_output_schema(rendered: RenderedPrompt[Any]) -> dict[str, Any] | None:
    """Build the output schema for the Codex API from a rendered prompt."""
    if rendered.output_type is None:
        return None
    element_schema = openai_strict_schema(schema(rendered.output_type))
    if rendered.container == "array":
        return openai_strict_schema(
            {
                "type": "object",
                "properties": {"items": {"type": "array", "items": element_schema}},
                "required": ["items"],
            }
        )
    return element_schema
