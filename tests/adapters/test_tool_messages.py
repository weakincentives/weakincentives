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
from weakincentives.prompt import SupportsDataclass, ToolResult

try:
    from tests.adapters._test_stubs import ToolPayload
except ModuleNotFoundError:  # pragma: no cover - fallback for direct invocation
    from ._test_stubs import ToolPayload


def test_serialize_tool_message_overrides_mapping_payload() -> None:
    result = ToolResult(message="ok", value=None)

    serialized = serialize_tool_message(
        cast(ToolResult[SupportsDataclass], result),
        payload={"group": {"value": 1}},
    )

    decoded = json.loads(serialized)
    assert decoded["message"] == "ok"
    assert decoded["success"] is True
    assert decoded["payload"] == {"group": {"value": 1}}


def test_serialize_tool_message_overrides_sequence_payload() -> None:
    result = ToolResult(message="ok", value=None)

    payload = [ToolPayload(answer="first"), ToolPayload(answer="second")]
    serialized = serialize_tool_message(
        cast(ToolResult[SupportsDataclass], result),
        payload=payload,
    )

    decoded = json.loads(serialized)
    assert decoded["payload"] == [
        {"answer": "first"},
        {"answer": "second"},
    ]


def test_serialize_tool_message_skips_payload_when_excluded() -> None:
    result = ToolResult(
        message="ok",
        value=ToolPayload(answer="secret"),
        exclude_value_from_context=True,
    )

    decoded = json.loads(
        serialize_tool_message(cast(ToolResult[SupportsDataclass], result))
    )
    assert decoded == {"message": "ok", "success": True}


def test_serialize_tool_message_skips_override_payload_when_excluded() -> None:
    result = ToolResult(message="ok", value=None, exclude_value_from_context=True)

    decoded = json.loads(
        serialize_tool_message(
            cast(ToolResult[SupportsDataclass], result),
            payload={"answer": "hidden"},
        )
    )
    assert decoded == {"message": "ok", "success": True}
