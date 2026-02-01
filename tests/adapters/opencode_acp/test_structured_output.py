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

"""Tests for OpenCode ACP structured output tool."""

from __future__ import annotations

from dataclasses import dataclass

from weakincentives.adapters.opencode_acp._structured_output import (
    StructuredOutputParams,
    StructuredOutputSignal,
    create_structured_output_tool_spec,
)


class TestStructuredOutputParams:
    def test_with_dict_data(self) -> None:
        params = StructuredOutputParams(data={"key": "value"})
        assert params.data == {"key": "value"}

    def test_with_list_data(self) -> None:
        params = StructuredOutputParams(data=[1, 2, 3])
        assert params.data == [1, 2, 3]

    def test_with_nested_data(self) -> None:
        params = StructuredOutputParams(data={"items": [{"name": "a"}, {"name": "b"}]})
        assert params.data == {"items": [{"name": "a"}, {"name": "b"}]}


class TestStructuredOutputSignal:
    def test_initial_state(self) -> None:
        signal = StructuredOutputSignal()
        assert not signal.is_set()
        data, error = signal.get()
        assert data is None
        assert error is None

    def test_set_data(self) -> None:
        signal = StructuredOutputSignal()
        signal.set({"key": "value"})
        assert signal.is_set()
        data, error = signal.get()
        assert data == {"key": "value"}
        assert error is None

    def test_set_error(self) -> None:
        signal = StructuredOutputSignal()
        signal.set_error("Validation failed")
        assert signal.is_set()
        data, error = signal.get()
        assert data is None
        assert error == "Validation failed"

    def test_first_set_wins(self) -> None:
        signal = StructuredOutputSignal()
        signal.set({"first": "value"})
        signal.set({"second": "value"})  # Should be ignored
        data, _ = signal.get()
        assert data == {"first": "value"}

    def test_data_blocks_error(self) -> None:
        signal = StructuredOutputSignal()
        signal.set({"data": "value"})
        signal.set_error("Should be ignored")
        data, error = signal.get()
        assert data == {"data": "value"}
        assert error is None

    def test_error_blocks_data(self) -> None:
        signal = StructuredOutputSignal()
        signal.set_error("First error")
        signal.set({"should": "be ignored"})
        data, error = signal.get()
        assert data is None
        assert error == "First error"


class TestCreateStructuredOutputToolSpec:
    def test_generates_tool_spec(self) -> None:
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
            "required": ["name", "value"],
        }

        spec = create_structured_output_tool_spec(json_schema)

        assert spec["name"] == "structured_output"
        assert "structured output" in spec["description"].lower()
        assert "input_schema" in spec
        assert spec["input_schema"]["properties"]["data"] is not None
        assert "data" in spec["input_schema"]["required"]

    def test_includes_schema_in_description(self) -> None:
        json_schema = {"type": "object", "properties": {"test": {"type": "string"}}}

        spec = create_structured_output_tool_spec(json_schema)

        # Schema should be included in description
        assert '"type": "object"' in spec["description"]
        assert '"test"' in spec["description"]

    def test_handles_array_schema(self) -> None:
        json_schema = {
            "type": "array",
            "items": {"type": "string"},
        }

        spec = create_structured_output_tool_spec(json_schema)

        assert spec["name"] == "structured_output"
        assert '"type": "array"' in spec["description"]


@dataclass
class SampleOutput:
    name: str
    count: int


class TestStructuredOutputHandler:
    """Tests for structured_output_handler function."""

    def test_validates_and_stores_valid_data(self) -> None:
        from unittest.mock import MagicMock

        from weakincentives.adapters.opencode_acp._structured_output import (
            structured_output_handler,
        )
        from weakincentives.prompt.tool import ToolContext

        signal = StructuredOutputSignal()

        # Create mock rendered prompt with output type
        rendered = MagicMock()
        rendered.output_type = SampleOutput

        # Create mock context
        context = MagicMock(spec=ToolContext)

        params = StructuredOutputParams(data={"name": "test", "count": 42})
        result = structured_output_handler(
            params,
            context=context,
            signal=signal,
            rendered=rendered,
        )

        assert result.success
        data, error = signal.get()
        assert isinstance(data, SampleOutput)
        assert data.name == "test"
        assert data.count == 42
        assert error is None

    def test_returns_error_for_invalid_data(self) -> None:
        from unittest.mock import MagicMock

        from weakincentives.adapters.opencode_acp._structured_output import (
            structured_output_handler,
        )
        from weakincentives.prompt.tool import ToolContext

        signal = StructuredOutputSignal()

        # Create mock rendered prompt with output type
        rendered = MagicMock()
        rendered.output_type = SampleOutput

        # Create mock context
        context = MagicMock(spec=ToolContext)

        # Invalid data - missing required field
        params = StructuredOutputParams(data={"name": "test"})  # missing count
        result = structured_output_handler(
            params,
            context=context,
            signal=signal,
            rendered=rendered,
        )

        assert not result.success
        data, error = signal.get()
        assert data is None
        assert error is not None
        assert "Validation error" in error

    def test_returns_error_for_no_output_type(self) -> None:
        from unittest.mock import MagicMock

        from weakincentives.adapters.opencode_acp._structured_output import (
            structured_output_handler,
        )
        from weakincentives.prompt.tool import ToolContext

        signal = StructuredOutputSignal()

        # Create mock rendered prompt with no output type
        rendered = MagicMock()
        rendered.output_type = None

        # Create mock context
        context = MagicMock(spec=ToolContext)

        params = StructuredOutputParams(data={"key": "value"})
        result = structured_output_handler(
            params,
            context=context,
            signal=signal,
            rendered=rendered,
        )

        assert not result.success
        assert "No structured output type" in result.message
