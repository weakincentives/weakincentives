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

"""Shared helpers for serialising tool results into provider messages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass
from typing import Final, cast

from ..prompt._types import SupportsToolResult
from ..prompt.tool import ToolResult
from ..serde import dump

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

_UNSET: Final = object()


def serialize_tool_message(
    result: ToolResult[SupportsToolResult], *, payload: object = _UNSET
) -> str:
    """Return a JSON string summarising a tool invocation for provider APIs."""

    message_payload: dict[str, JsonValue | bool | str] = {
        "message": result.message,
        "success": result.success,
    }

    if not result.exclude_value_from_context:
        value = result.value if payload is _UNSET else payload
        if value is not None:
            message_payload["payload"] = _serialize_value(value)

    return json.dumps(message_payload, ensure_ascii=False)


def _serialize_value(value: object) -> JsonValue:
    """Convert tool result payloads to JSON-compatible structures."""

    if is_dataclass(value):
        return _serialize_str_mapping(dump(value, exclude_none=True))

    if isinstance(value, Mapping):
        return _serialize_mapping(cast(Mapping[object, object], value))

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence_items = cast(Sequence[object], value)
        return [_serialize_value(item) for item in sequence_items]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _serialize_mapping(mapping: Mapping[object, object]) -> dict[str, JsonValue]:
    serialized: dict[str, JsonValue] = {}
    for key, item in mapping.items():
        serialized[str(key)] = _serialize_value(item)
    return serialized


def _serialize_str_mapping(mapping: Mapping[str, object]) -> dict[str, JsonValue]:
    return {key: _serialize_value(item) for key, item in mapping.items()}


__all__ = ["serialize_tool_message"]
