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

"""Tests for tool specification building and serialization."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.adapters._provider_protocols import (
    ProviderFunctionCallData,
    ProviderToolCallData,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.adapters.tool_spec import (
    parse_tool_arguments,
    serialize_tool_call,
    tool_to_spec,
)
from weakincentives.prompt import Tool, ToolResult


@dataclass(frozen=True)
class FileParams:
    """Test parameter dataclass."""

    path: str
    mode: str = "r"


def _dummy_handler(params: FileParams) -> ToolResult[str]:
    """Dummy handler for testing."""
    return ToolResult.ok(params.path)


def _none_handler(params: None) -> ToolResult[None]:
    """Dummy handler for testing tools with no params."""
    del params
    return ToolResult.ok(None)


class TestToolToSpec:
    """Tests for tool_to_spec function."""

    def test_tool_with_params(self) -> None:
        """Tool with params generates schema from dataclass."""
        tool = Tool[FileParams, str](
            name="read_file",
            description="Read a file from disk.",
            handler=_dummy_handler,
        )

        spec = tool_to_spec(tool)

        assert spec["type"] == "function"
        assert spec["function"]["name"] == "read_file"
        assert spec["function"]["description"] == "Read a file from disk."
        assert "properties" in spec["function"]["parameters"]
        assert "path" in spec["function"]["parameters"]["properties"]

    def test_tool_without_params(self) -> None:
        """Tool with None params uses empty schema."""
        tool = Tool[None, None](
            name="get_time",
            description="Get the current time.",
            handler=_none_handler,
        )

        spec = tool_to_spec(tool)

        assert spec["type"] == "function"
        assert spec["function"]["name"] == "get_time"
        assert spec["function"]["parameters"]["type"] == "object"
        assert spec["function"]["parameters"]["properties"] == {}
        assert spec["function"]["parameters"]["additionalProperties"] is False


class TestSerializeToolCall:
    """Tests for serialize_tool_call function."""

    def test_serializes_complete_tool_call(self) -> None:
        """Full tool call is serialized correctly."""
        function = ProviderFunctionCallData(
            name="read_file",
            arguments='{"path": "/tmp/test.txt"}',
        )
        tool_call = ProviderToolCallData(id="call_123", function=function)

        result = serialize_tool_call(tool_call)

        assert result == {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/tmp/test.txt"}',
            },
        }

    def test_handles_none_arguments(self) -> None:
        """Tool call with None arguments uses empty object."""
        function = ProviderFunctionCallData(name="get_time", arguments=None)
        tool_call = ProviderToolCallData(id="call_456", function=function)

        result = serialize_tool_call(tool_call)

        assert result["function"]["arguments"] == "{}"


class TestParseToolArguments:
    """Tests for parse_tool_arguments function."""

    def test_parses_valid_json(self) -> None:
        """Valid JSON is parsed correctly."""
        args = parse_tool_arguments(
            '{"path": "/tmp/test.txt", "mode": "w"}',
            prompt_name="test",
            provider_payload=None,
        )

        assert args == {"path": "/tmp/test.txt", "mode": "w"}

    def test_empty_string_returns_empty_dict(self) -> None:
        """Empty string returns empty dict."""
        args = parse_tool_arguments(
            "",
            prompt_name="test",
            provider_payload=None,
        )

        assert args == {}

    def test_none_returns_empty_dict(self) -> None:
        """None returns empty dict."""
        args = parse_tool_arguments(
            None,
            prompt_name="test",
            provider_payload=None,
        )

        assert args == {}

    def test_invalid_json_raises_error(self) -> None:
        """Invalid JSON raises PromptEvaluationError."""
        with pytest.raises(PromptEvaluationError) as excinfo:
            parse_tool_arguments(
                "{invalid json}",
                prompt_name="test_prompt",
                provider_payload={"raw": "data"},
            )

        assert "decode" in str(excinfo.value).lower()

    def test_non_object_raises_error(self) -> None:
        """Non-object JSON raises PromptEvaluationError."""
        with pytest.raises(PromptEvaluationError) as excinfo:
            parse_tool_arguments(
                '["array", "not", "object"]',
                prompt_name="test_prompt",
                provider_payload=None,
            )

        assert "object" in str(excinfo.value).lower()
