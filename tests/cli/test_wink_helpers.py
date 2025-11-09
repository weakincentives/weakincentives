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

"""Tests for wink CLI helper utilities."""

from __future__ import annotations

import json
from pathlib import Path

import mcp.types as mcp_types
import pytest
from pydantic import BaseModel, Field

from weakincentives.cli import wink
from weakincentives.prompt.overrides.inspection import OverrideFileMetadata


class _DemoModel(BaseModel):
    """Simple model used to exercise schema helpers."""

    foo: int = Field(..., description="An integer field.")


def test_model_schema_returns_json_schema() -> None:
    schema = wink._model_schema(_DemoModel)
    assert schema["properties"]["foo"]["type"] == "integer"


def test_normalise_payload_supports_pydantic_models() -> None:
    model_payload = _DemoModel(foo=3)
    assert wink._normalise_payload(model_payload) == {"foo": 3}
    mapping_payload = {"bar": "baz"}
    assert wink._normalise_payload(mapping_payload) == mapping_payload


def test_build_call_result_formats_message_and_payload() -> None:
    result = wink._ToolExecutionResult(
        message="hello",
        payload=_DemoModel(foo=9),
    )
    call_result = wink._build_call_result(result)
    assert isinstance(call_result, mcp_types.CallToolResult)
    assert call_result.content[0].text == "hello"
    assert call_result.structuredContent == {"foo": 9}


def test_mcp_error_wraps_error_data() -> None:
    error = wink._mcp_error(code=42, message="failure", data={"detail": "nope"})
    assert isinstance(error, wink.McpError)
    assert error.error.code == 42
    assert error.error.message == "failure"
    assert error.error.data == {"detail": "nope"}


def test_override_identity_loads_metadata(tmp_path: Path) -> None:
    override_path = tmp_path / "demo.json"
    override_path.write_text(
        json.dumps({"ns": "demo", "prompt_key": "example", "tag": "latest"}),
        encoding="utf-8",
    )
    identity = wink._override_identity(override_path)
    assert identity == ("demo", "example", "latest")


def test_override_identity_rejects_non_mapping_payload(tmp_path: Path) -> None:
    override_path = tmp_path / "invalid.json"
    override_path.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(wink.PromptOverridesError):
        wink._override_identity(override_path)


def test_override_identity_requires_metadata_fields(tmp_path: Path) -> None:
    override_path = tmp_path / "missing.json"
    override_path.write_text(json.dumps({"ns": "demo"}), encoding="utf-8")
    with pytest.raises(wink.PromptOverridesError):
        wink._override_identity(override_path)


def test_build_override_entry_populates_fields(tmp_path: Path) -> None:
    file_path = tmp_path / "demo" / "override.json"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("{}", encoding="utf-8")
    metadata = OverrideFileMetadata(
        path=file_path,
        relative_segments=("demo", "override.json"),
        modified_time=1700000000.0,
        content_hash="deadbeef",
        section_count=2,
        tool_count=1,
    )
    entry = wink._build_override_entry(
        metadata=metadata,
        ns="demo",
        prompt="example",
        tag="latest",
    )
    assert entry.backing_file_path == str(file_path)
    assert entry.relative_path == "demo/override.json"
    assert entry.section_count == 2
    assert entry.tool_count == 1


@pytest.mark.parametrize(
    ("host", "expected"),
    [("0.0.0.0", "127.0.0.1"), ("::", "127.0.0.1"), ("example.com", "example.com")],
)
def test_format_host_for_display_normalises_bind_all(host: str, expected: str) -> None:
    assert wink._format_host_for_display(host) == expected


@pytest.mark.parametrize(
    ("host", "expected_authority"),
    [
        ("127.0.0.1", "127.0.0.1:8080"),
        ("2001:db8::1", "[2001:db8::1]:8080"),
    ],
)
def test_format_base_url_handles_ipv4_and_ipv6(
    host: str, expected_authority: str
) -> None:
    base, sse, post = wink._format_base_url(host, 8080, sse_path="/sse")
    assert base == f"http://{expected_authority}"
    assert sse == f"{base}/sse"
    assert post == f"{base}/messages"


def test_connection_instructions_renders_expected_lines() -> None:
    instructions = wink._connection_instructions("127.0.0.1", 8123, sse_path="/sse")
    lines = instructions.splitlines()
    assert "wink MCP server ready at http://127.0.0.1:8123/sse" in lines[0]
    assert "Claude Desktop" in lines[1]
    assert "Codex CLI" in lines[2]
    assert lines[-1].endswith("/messages")
