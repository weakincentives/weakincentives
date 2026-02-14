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

"""Structured output MCP tool for ACP adapter."""

from __future__ import annotations

import json
import threading
from typing import Any

from ...runtime.logging import StructuredLogger, get_logger
from ...serde import schema as generate_schema

__all__ = [
    "STRUCTURED_OUTPUT_TOOL_NAME",
    "StructuredOutputCapture",
    "StructuredOutputTool",
    "create_structured_output_tool",
]

logger: StructuredLogger = get_logger(
    __name__, context={"component": "acp_structured_output"}
)

STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"


class StructuredOutputCapture:
    """Thread-safe capture for structured output data.

    The MCP tool handler runs in a thread pool (via ``asyncio.to_thread`` in
    ``create_mcp_tool_server``) while the adapter reads from the main asyncio
    thread.  A lock serialises access to ``_data`` and ``_called``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Any = None
        self._called = False

    @property
    def called(self) -> bool:
        """Whether ``store`` has been called."""
        with self._lock:
            return self._called

    @property
    def data(self) -> Any:
        """The stored data, or ``None`` if not yet called."""
        with self._lock:
            return self._data

    def store(self, data: Any) -> None:
        """Store structured output data."""
        with self._lock:
            self._data = data
            self._called = True


class StructuredOutputTool:
    """MCP-compatible tool for capturing structured output.

    Duck-typed to be compatible with ``create_mcp_tool_server()`` --- has ``.name``,
    ``.description``, ``.input_schema``, and ``__call__``.
    """

    def __init__(
        self,
        *,
        json_schema: dict[str, Any],
        capture: StructuredOutputCapture,
    ) -> None:
        self.name = STRUCTURED_OUTPUT_TOOL_NAME
        schema_str = json.dumps(json_schema, indent=2)
        self.description = (
            "Call this tool to submit your final structured output.\n"
            "The data must conform to the following JSON schema:\n"
            f"{schema_str}"
        )
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "data": json_schema,
            },
            "required": ["data"],
        }
        self._capture = capture
        self._json_schema = json_schema

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute the structured output tool."""
        data = args.get("data")
        if data is None:
            return {
                "content": [{"type": "text", "text": "Missing 'data' field"}],
                "isError": True,
            }

        # Validate against schema before storing (best-effort).
        try:
            import jsonschema

            jsonschema.validate(data, self._json_schema)
        except ImportError:
            pass  # jsonschema not available — skip validation
        except jsonschema.ValidationError as exc:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Schema validation failed: {exc.message}",
                    }
                ],
                "isError": True,
            }

        self._capture.store(data)

        return {
            "content": [
                {"type": "text", "text": "Structured output received successfully."}
            ],
            "isError": False,
        }


def create_structured_output_tool(
    output_type: type[Any],
    *,
    container: str = "object",
) -> tuple[StructuredOutputTool, StructuredOutputCapture]:
    """Create a structured output tool and its capture.

    Returns ``(tool, capture)`` tuple.  The tool is compatible with
    ``create_mcp_tool_server()``.
    """
    element_schema = generate_schema(output_type)

    if container == "array":
        json_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": element_schema},
            },
            "required": ["items"],
        }
    else:
        json_schema = element_schema

    capture = StructuredOutputCapture()
    tool = StructuredOutputTool(json_schema=json_schema, capture=capture)
    return tool, capture
