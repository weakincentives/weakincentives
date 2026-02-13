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

"""Tests for ACP structured output tool."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from unittest.mock import patch

from weakincentives.adapters.acp._structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputCapture,
    StructuredOutputTool,
    create_structured_output_tool,
)


class TestStructuredOutputCapture:
    def test_initial_state(self) -> None:
        capture = StructuredOutputCapture()
        assert not capture.called
        assert capture.data is None

    def test_store_sets_data_and_called(self) -> None:
        capture = StructuredOutputCapture()
        capture.store({"key": "value"})
        assert capture.called
        assert capture.data == {"key": "value"}

    def test_store_overwrites(self) -> None:
        capture = StructuredOutputCapture()
        capture.store("first")
        capture.store("second")
        assert capture.data == "second"
        assert capture.called

    def test_store_none_value(self) -> None:
        """Storing None explicitly still marks as called."""
        capture = StructuredOutputCapture()
        capture.store(None)
        assert capture.called
        assert capture.data is None

    def test_concurrent_store_and_read(self) -> None:
        """Concurrent store/read from multiple threads must not corrupt state."""
        capture = StructuredOutputCapture()
        barrier = threading.Barrier(3)
        errors: list[str] = []

        def writer() -> None:
            barrier.wait()
            for i in range(200):
                capture.store({"v": i})

        def reader() -> None:
            barrier.wait()
            for _ in range(200):
                called = capture.called
                data = capture.data
                # Once called is True, data must be a dict
                if called and not isinstance(data, dict):
                    errors.append(f"called={called} but data={data!r}")

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        barrier.wait()
        t1.join()
        t2.join()
        assert not errors, errors


class TestStructuredOutputTool:
    def test_name(self) -> None:
        capture = StructuredOutputCapture()
        tool = StructuredOutputTool(
            json_schema={"type": "object"},
            capture=capture,
        )
        assert tool.name == STRUCTURED_OUTPUT_TOOL_NAME

    def test_description_contains_schema(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        capture = StructuredOutputCapture()
        tool = StructuredOutputTool(json_schema=schema, capture=capture)
        assert "x" in tool.description
        assert "integer" in tool.description
        assert "Call this tool" in tool.description

    def test_input_schema_wraps_in_data(self) -> None:
        inner = {"type": "object", "properties": {"x": {"type": "integer"}}}
        capture = StructuredOutputCapture()
        tool = StructuredOutputTool(json_schema=inner, capture=capture)
        assert tool.input_schema["type"] == "object"
        assert "data" in tool.input_schema["properties"]
        assert tool.input_schema["properties"]["data"] == inner
        assert tool.input_schema["required"] == ["data"]

    def test_call_with_valid_data(self) -> None:
        capture = StructuredOutputCapture()
        tool = StructuredOutputTool(
            json_schema={"type": "object"},
            capture=capture,
        )
        result = tool({"data": {"key": "value"}})
        assert not result["isError"]
        assert capture.called
        assert capture.data == {"key": "value"}

    def test_call_with_missing_data(self) -> None:
        capture = StructuredOutputCapture()
        tool = StructuredOutputTool(
            json_schema={"type": "object"},
            capture=capture,
        )
        result = tool({})
        assert result["isError"]
        assert not capture.called

    def test_call_result_format(self) -> None:
        capture = StructuredOutputCapture()
        tool = StructuredOutputTool(
            json_schema={"type": "object"},
            capture=capture,
        )
        result = tool({"data": {"key": "value"}})
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "received" in result["content"][0]["text"]

    def test_call_with_schema_violation(self) -> None:
        """Data that violates the JSON schema returns isError without storing."""
        capture = StructuredOutputCapture()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        tool = StructuredOutputTool(json_schema=schema, capture=capture)
        result = tool({"data": {"name": 42}})
        assert result["isError"]
        assert "Schema validation failed" in result["content"][0]["text"]
        assert not capture.called

    def test_call_with_valid_data_passes_schema(self) -> None:
        """Data matching the JSON schema stores successfully."""
        capture = StructuredOutputCapture()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        tool = StructuredOutputTool(json_schema=schema, capture=capture)
        result = tool({"data": {"name": "Alice"}})
        assert not result["isError"]
        assert capture.called
        assert capture.data == {"name": "Alice"}

    def test_call_graceful_degradation_without_jsonschema(self) -> None:
        """When jsonschema is unavailable, validation is skipped."""
        import builtins

        original_import = builtins.__import__

        def _block_jsonschema(name: str, *args: object, **kwargs: object) -> object:
            if name == "jsonschema":
                raise ImportError("no jsonschema")
            return original_import(name, *args, **kwargs)

        capture = StructuredOutputCapture()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        tool = StructuredOutputTool(json_schema=schema, capture=capture)

        with patch("builtins.__import__", side_effect=_block_jsonschema):
            result = tool({"data": {"name": 42}})

        # Should succeed (skip validation) — existing behavior preserved
        assert not result["isError"]
        assert capture.called


class TestCreateStructuredOutputTool:
    def test_object_container(self) -> None:
        @dataclass(frozen=True)
        class Item:
            name: str
            count: int

        tool, capture = create_structured_output_tool(Item)
        assert tool.name == STRUCTURED_OUTPUT_TOOL_NAME
        assert not capture.called

        # Schema should be for the dataclass itself
        assert "properties" in tool._json_schema
        assert "name" in tool._json_schema["properties"]
        assert "count" in tool._json_schema["properties"]

    def test_array_container(self) -> None:
        @dataclass(frozen=True)
        class Item:
            name: str

        tool, capture = create_structured_output_tool(Item, container="array")
        assert not capture.called

        # Schema should have items wrapper
        assert "properties" in tool._json_schema
        assert "items" in tool._json_schema["properties"]
        items_prop = tool._json_schema["properties"]["items"]
        assert items_prop["type"] == "array"
        assert "items" in items_prop  # the element schema
        assert "properties" in items_prop["items"]
        assert "name" in items_prop["items"]["properties"]

    def test_tool_callable(self) -> None:
        @dataclass(frozen=True)
        class Item:
            name: str

        tool, capture = create_structured_output_tool(Item)
        result = tool({"data": {"name": "test"}})
        assert not result["isError"]
        assert capture.called
        assert capture.data == {"name": "test"}
