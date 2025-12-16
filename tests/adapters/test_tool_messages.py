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
from typing import cast

from weakincentives.adapters._tool_messages import serialize_tool_message
from weakincentives.prompt import ToolResult
from weakincentives.types import SupportsToolResult

try:
    from tests.adapters._test_stubs import ToolPayload
except ModuleNotFoundError:  # pragma: no cover - fallback for direct invocation
    from ._test_stubs import ToolPayload


def test_serialize_tool_message_overrides_mapping_payload() -> None:
    result = ToolResult(message="ok", value=None)

    serialized = serialize_tool_message(
        cast(ToolResult[SupportsToolResult], result),
        payload={"group": {"value": 1}},
    )

    assert serialized == "ok\n\n" + json.dumps(
        {"group": {"value": 1}}, ensure_ascii=False
    )


def test_serialize_tool_message_overrides_sequence_payload() -> None:
    result = ToolResult(message="ok", value=None)

    payload = [ToolPayload(answer="first"), ToolPayload(answer="second")]
    serialized = serialize_tool_message(
        cast(ToolResult[SupportsToolResult], result),
        payload=payload,
    )

    assert serialized == "ok\n\n" + json.dumps(
        [
            {"answer": "first"},
            {"answer": "second"},
        ],
        ensure_ascii=False,
    )


def test_serialize_tool_message_skips_payload_when_excluded() -> None:
    result = ToolResult(
        message="ok",
        value=ToolPayload(answer="secret"),
        exclude_value_from_context=True,
    )

    serialized = serialize_tool_message(cast(ToolResult[SupportsToolResult], result))
    assert serialized == "ok"


def test_serialize_tool_message_skips_override_payload_when_excluded() -> None:
    result = ToolResult(message="ok", value=None, exclude_value_from_context=True)

    serialized = serialize_tool_message(
        cast(ToolResult[SupportsToolResult], result),
        payload={"answer": "hidden"},
    )
    assert serialized == "ok"


def test_serialize_tool_message_falls_back_to_stringification() -> None:
    class UnknownPayload:
        def __str__(self) -> str:
            return "payload"

    result = ToolResult(message="ok", value=None)

    serialized = serialize_tool_message(
        cast(ToolResult[SupportsToolResult], result),
        payload=UnknownPayload(),
    )
    assert serialized == "ok\n\npayload"


def test_serialize_tool_message_without_message() -> None:
    payload = ToolPayload(answer="value")
    result = ToolResult(message="", value=payload)

    serialized = serialize_tool_message(cast(ToolResult[SupportsToolResult], result))

    assert serialized == json.dumps({"answer": "value"}, ensure_ascii=False)


def test_serialize_tool_message_without_payload_defaults_to_message() -> None:
    result = ToolResult(message="fallback", value=None)

    serialized = serialize_tool_message(cast(ToolResult[SupportsToolResult], result))

    assert serialized == "fallback"
