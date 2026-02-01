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

"""Structured output MCP tool for OpenCode ACP adapter.

This module provides the ``structured_output`` MCP tool that the model calls
to finalize structured output. Unlike provider-native structured output, ACP
uses a tool-based approach where the model explicitly calls this tool with
the output data.

The tool validates the data against the prompt's output schema and stores
it for retrieval after the session completes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from ...dataclasses import FrozenDataclass
from ...prompt.tool import ToolContext, ToolResult
from ...serde import parse

if TYPE_CHECKING:
    from ...prompt.protocols import RenderedPromptProtocol

__all__ = [
    "StructuredOutputParams",
    "StructuredOutputSignal",
    "create_structured_output_tool_spec",
    "structured_output_handler",
]


@FrozenDataclass()
class StructuredOutputParams:
    """Parameters for the structured_output tool.

    Attributes:
        data: The structured output data to validate and store. Accepts
            any JSON value (object or array) to support array-root schemas.
    """

    data: Any


@dataclass(slots=True)
class StructuredOutputSignal:
    """Thread-safe container for structured output data.

    When the model calls the ``structured_output`` tool, the handler stores
    the validated output in this signal. After the session completes, the
    adapter retrieves the output from this signal.

    Thread Safety:
        All operations are protected by a lock. Only the first successful
        call stores data (first-one-wins semantics).
    """

    _data: object | None = field(default=None)
    _error: str | None = field(default=None)
    _lock: Lock = field(default_factory=Lock)

    def set(self, data: object) -> None:
        """Store validated structured output data.

        Only the first call stores data; subsequent calls are ignored.

        Args:
            data: The validated structured output data.
        """
        with self._lock:
            if self._data is None and self._error is None:
                self._data = data

    def set_error(self, error: str) -> None:
        """Store a validation error.

        Only the first call stores an error; subsequent calls are ignored.

        Args:
            error: The validation error message.
        """
        with self._lock:
            if self._data is None and self._error is None:
                self._error = error

    def get(self) -> tuple[object | None, str | None]:
        """Get the stored data and error.

        Returns:
            Tuple of (data, error). At most one will be non-None.
        """
        with self._lock:
            return self._data, self._error

    def is_set(self) -> bool:
        """Check if data or error is stored.

        Returns:
            True if data or error is stored, False otherwise.
        """
        with self._lock:
            return self._data is not None or self._error is not None


def create_structured_output_tool_spec(
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Create the MCP tool specification for structured_output.

    Args:
        json_schema: The JSON schema for the expected output.

    Returns:
        MCP tool specification dict with name, description, and input schema.
    """
    schema_str = json.dumps(json_schema, indent=2)
    return {
        "name": "structured_output",
        "description": (
            "Call this tool to submit your final structured output.\n"
            "The data must conform to the following JSON schema:\n"
            f"{schema_str}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data": {
                    "description": "The structured output data conforming to the schema.",
                }
            },
            "required": ["data"],
        },
    }


def structured_output_handler(
    params: StructuredOutputParams,
    *,
    context: ToolContext,
    signal: StructuredOutputSignal,
    rendered: RenderedPromptProtocol[Any],
) -> ToolResult[None]:
    """Handle the structured_output tool call.

    Validates the provided data against the prompt's output schema and stores
    it in the signal for later retrieval.

    Args:
        params: The tool parameters containing the data.
        context: Tool execution context.
        signal: Signal to store the validated output.
        rendered: Rendered prompt with output type for validation.

    Returns:
        ToolResult indicating success or validation failure.
    """
    output_type = rendered.output_type
    if output_type is None or output_type is type(None):
        return ToolResult.error("No structured output type declared for this prompt.")

    try:
        # Parse and validate against the output type
        validated = parse(output_type, params.data, extra="ignore")
        signal.set(validated)
        return ToolResult.ok(None, message="Structured output accepted.")
    except (TypeError, ValueError) as error:
        error_msg = f"Validation error: {error}"
        signal.set_error(error_msg)
        return ToolResult.error(error_msg)
