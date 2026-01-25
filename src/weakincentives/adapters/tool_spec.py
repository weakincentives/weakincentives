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

"""Tool specification building and serialization for provider adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from ..serde import schema
from ..types.dataclass import (
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._provider_protocols import ProviderToolCall
from .core import (
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
)

if TYPE_CHECKING:
    from ..prompt.tool import Tool


ToolChoice = str | Mapping[str, Any] | None
"""Tool choice directive controlling how providers select tools.

Supported values:
    - ``None``: Provider decides whether to call tools (default behavior).
    - ``"auto"``: Provider may call zero or more tools as needed.
    - ``"required"``: Provider must call at least one tool.
    - ``"none"``: Provider must not call any tools.
    - ``{"type": "function", "function": {"name": "tool_name"}}``: Force
      the provider to call a specific tool by name.

The exact interpretation depends on the provider adapter being used.
"""


class ToolArgumentsParser(Protocol):
    """Protocol for parsing tool call arguments from provider JSON payloads.

    Provider adapters may supply custom parsers to handle provider-specific
    argument formats or to apply transformations during parsing.

    The default implementation is :func:`parse_tool_arguments`.
    """

    def __call__(
        self,
        arguments_json: str | None,
        *,
        prompt_name: str,
        provider_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Parse JSON-encoded tool arguments into a dictionary.

        Args:
            arguments_json: JSON string containing tool arguments, or ``None``
                if no arguments were provided by the model.
            prompt_name: Name of the prompt being evaluated, used for error
                context in diagnostics.
            provider_payload: Raw provider response payload for error context.

        Returns:
            Dictionary of parsed arguments ready to be passed to the tool
            handler. Returns an empty dict if ``arguments_json`` is empty.

        Raises:
            PromptEvaluationError: If the JSON is malformed or not a valid
                object with string keys.
        """
        ...


_EMPTY_TOOL_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def tool_to_spec(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
) -> dict[str, Any]:
    """Convert a Tool instance to a provider-agnostic function specification.

    Generates the JSON Schema representation of the tool suitable for
    inclusion in provider API requests. The schema follows the OpenAI
    function calling format, which is supported by most providers.

    Args:
        tool: The Tool instance to convert. The tool's ``params_type``
            determines the generated parameter schema.

    Returns:
        A dictionary with the following structure::

            {
                "type": "function",
                "function": {
                    "name": "<tool.name>",
                    "description": "<tool.description>",
                    "parameters": <JSON Schema for params_type>
                }
            }

        If the tool accepts no parameters (``params_type`` is ``None``),
        the parameters schema allows an empty object only.
    """

    if tool.params_type is type(None):
        parameters_schema = dict(_EMPTY_TOOL_PARAMETERS_SCHEMA)
    else:
        parameters_schema = schema(tool.params_type, extra="forbid")
        _ = parameters_schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema,
        },
    }


def serialize_tool_call(tool_call: ProviderToolCall) -> dict[str, Any]:
    """Serialize a provider tool call for inclusion in message history.

    Converts a tool call from a provider response into the format expected
    when sending message history back to the provider. This is used to
    reconstruct assistant messages that contained tool calls.

    Args:
        tool_call: The tool call object from the provider response,
            conforming to :class:`ProviderToolCall`.

    Returns:
        A dictionary with the following structure::

            {
                "id": "<tool_call.id>",
                "type": "function",
                "function": {
                    "name": "<function.name>",
                    "arguments": "<function.arguments or '{}'>"
                }
            }

        The arguments field defaults to ``"{}"`` if the original was empty.
    """

    function = tool_call.function
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": function.name,
            "arguments": function.arguments or "{}",
        },
    }


def parse_tool_arguments(
    arguments_json: str | None,
    *,
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Parse JSON-encoded tool arguments from a provider response.

    This is the default implementation of :class:`ToolArgumentsParser`.
    It validates that the JSON represents an object with string keys,
    which is required for tool parameter binding.

    Args:
        arguments_json: JSON string containing the tool arguments as
            returned by the provider, or ``None``/empty string if the
            model provided no arguments.
        prompt_name: Name of the prompt being evaluated, included in
            error messages for debugging.
        provider_payload: The complete provider response payload, attached
            to errors for diagnostic purposes.

    Returns:
        A dictionary mapping argument names to their values. Returns an
        empty dictionary if ``arguments_json`` is ``None`` or empty.

    Raises:
        PromptEvaluationError: If JSON parsing fails, if the parsed value
            is not an object, or if any object key is not a string.
    """

    if not arguments_json:
        return {}
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError as error:
        raise PromptEvaluationError(
            "Failed to decode tool call arguments.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=provider_payload,
        ) from error
    if not isinstance(parsed, Mapping):
        raise PromptEvaluationError(
            "Tool call arguments must be a JSON object.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=provider_payload,
        )
    parsed_mapping = cast(Mapping[Any, Any], parsed)
    arguments: dict[str, Any] = {}
    for key, value in parsed_mapping.items():
        if not isinstance(key, str):
            raise PromptEvaluationError(
                "Tool call arguments must use string keys.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_TOOL,
                provider_payload=provider_payload,
            )
        arguments[key] = value
    return arguments


__all__ = [
    "ToolArgumentsParser",
    "ToolChoice",
    "parse_tool_arguments",
    "serialize_tool_call",
    "tool_to_spec",
]
