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

import json
import logging
from dataclasses import dataclass

import pytest

from weakincentives.prompt.tool_result import ToolResult, render_tool_payload


@dataclass(slots=True)
class RenderableData:
    value: str

    def render(self) -> str:
        return f"rendered:{self.value}"


@dataclass(slots=True)
class NonStringRenderableData:
    value: str

    def render(self) -> object:
        return [self.value]


@dataclass(slots=True)
class PlainData:
    value: str


@dataclass(slots=True)
class FailingRenderableData:
    value: str

    def render(self) -> str:
        raise ValueError(f"boom:{self.value}")


def test_render_tool_payload_prefers_custom_render() -> None:
    payload = render_tool_payload(RenderableData(value="example"))
    assert payload == "rendered:example"


def test_render_tool_payload_casts_non_string_render() -> None:
    payload = render_tool_payload(NonStringRenderableData(value="value"))
    assert payload == "['value']"


def test_render_tool_payload_falls_back_to_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = PlainData(value="data")

    def failing_dump(*_: object, **__: object) -> dict[str, object]:
        raise TypeError("cannot serialize")

    monkeypatch.setattr("weakincentives.prompt.tool_result.dump", failing_dump)
    payload = render_tool_payload(data)
    assert "PlainData" in payload


def test_render_tool_payload_handles_bytes_and_sequences() -> None:
    payload = render_tool_payload(b"text")
    assert payload == "text"

    json_sequence = render_tool_payload(
        [PlainData(value="one"), PlainData(value="two")]
    )
    assert json.loads(json_sequence) == [
        {"value": "one"},
        {"value": "two"},
    ]

    fallback_sequence = render_tool_payload([PlainData(value="one"), "raw"])
    assert "raw" in fallback_sequence


def test_render_tool_payload_renders_nested_mappings() -> None:
    nested = {
        "data": RenderableData(value="nested"),
        "bytes": b"blob",
        "items": [RenderableData(value="alpha")],
        "other": object(),
    }

    payload = render_tool_payload(nested)
    decoded = json.loads(payload)
    assert decoded["bytes"] == "blob"
    assert decoded["data"] == {"value": "nested"}
    assert decoded["items"][0] == {"value": "alpha"}
    assert decoded["other"].startswith("<object")


def test_render_tool_payload_handles_render_exceptions() -> None:
    payload = render_tool_payload(FailingRenderableData(value="fallback"))
    decoded = json.loads(payload)
    assert decoded == {"value": "fallback"}


def test_render_tool_payload_handles_unserializable_mapping_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_dump(*_: object, **__: object) -> dict[str, object]:
        raise TypeError("fail")

    monkeypatch.setattr("weakincentives.prompt.tool_result.dump", failing_dump)
    payload = render_tool_payload({"data": PlainData(value="fallback")})
    decoded = json.loads(payload)
    assert decoded["data"].startswith("PlainData(")


def test_render_tool_payload_warns_when_render_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        payload = render_tool_payload(PlainData(value="data"))

    decoded = json.loads(payload)
    assert decoded == {"value": "data"}

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "render() implementation" in record.getMessage()
    assert getattr(record, "event", None) == "tool_result.render.missing"
    assert getattr(record, "dataclass", None) == "PlainData"


def test_render_tool_payload_normalizes_primitive_values_in_mapping() -> None:
    """Test that primitive values in mappings are preserved correctly."""
    payload = render_tool_payload(
        {
            "string": "text",
            "integer": 42,
            "floating": 3.14,
            "boolean": True,
            "null_val": None,
            "nested": {"inner_key": "inner_value", "inner_num": 123},
        }
    )
    decoded = json.loads(payload)
    assert decoded["string"] == "text"
    assert decoded["integer"] == 42
    assert decoded["floating"] == 3.14
    assert decoded["boolean"] is True
    assert decoded["null_val"] is None
    assert decoded["nested"] == {"inner_key": "inner_value", "inner_num": 123}


class TestToolResultConvenienceConstructors:
    """Tests for ToolResult.ok() and ToolResult.error() class methods."""

    def test_ok_with_value_and_default_message(self) -> None:
        result = ToolResult.ok(PlainData(value="test"))

        assert result.success is True
        assert result.message == "OK"
        assert result.value == PlainData(value="test")
        assert result.exclude_value_from_context is False

    def test_ok_with_value_and_custom_message(self) -> None:
        result = ToolResult.ok(PlainData(value="data"), message="File written")

        assert result.success is True
        assert result.message == "File written"
        assert result.value == PlainData(value="data")

    def test_ok_preserves_type(self) -> None:
        result: ToolResult[RenderableData] = ToolResult.ok(
            RenderableData(value="typed")
        )

        assert isinstance(result.value, RenderableData)
        assert result.value.value == "typed"

    def test_error_with_message(self) -> None:
        result = ToolResult.error("File not found")

        assert result.success is False
        assert result.message == "File not found"
        assert result.value is None
        assert result.exclude_value_from_context is False

    def test_error_returns_none_typed_result(self) -> None:
        result: ToolResult[None] = ToolResult.error("Something went wrong")

        assert result.value is None
        assert result.success is False
