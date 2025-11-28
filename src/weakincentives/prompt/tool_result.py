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

"""Shared result container returned by tool handlers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from typing import Any, cast

from ..serde import dump
from ._types import SupportsDataclass

_SEQUENCE_EXCLUSIONS = (str, bytes, bytearray)


@dataclass(slots=True)
class ToolResult[ResultValueT]:
    """Structured response emitted by a tool handler."""

    message: str
    value: ResultValueT | None
    success: bool = True
    exclude_value_from_context: bool = False

    def render(self) -> str:
        """Return the canonical textual representation of the payload."""

        return render_tool_payload(self.value)


def render_tool_payload(value: object) -> str:
    """Convert tool payloads into textual output for adapters and telemetry."""

    rendered = ""
    if value is None:
        return rendered

    if _is_dataclass_instance(value):
        rendered = _render_dataclass(cast(SupportsDataclass, value))
    elif isinstance(value, Mapping):
        typed_mapping = cast(Mapping[Any, object], value)
        rendered = _render_mapping(typed_mapping)
    elif isinstance(value, Sequence) and not isinstance(value, _SEQUENCE_EXCLUSIONS):
        sequence_items = cast(Sequence[object], value)
        rendered = _render_sequence(sequence_items)
    elif isinstance(value, bytes):
        rendered = value.decode("utf-8", errors="replace")
    else:
        rendered = str(value)

    return rendered


def _is_dataclass_instance(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _render_dataclass(value: SupportsDataclass) -> str:
    render_method = cast(Callable[[], object] | None, getattr(value, "render", None))
    render_result: object | None = None
    if callable(render_method):
        try:
            render_result = render_method()
        except Exception:
            render_result = None

        if render_result is not None:
            return (
                render_result if isinstance(render_result, str) else str(render_result)
            )

    try:
        payload = dump(value, exclude_none=True)
    except TypeError:
        rendered = repr(value)
    else:
        rendered = json.dumps(payload, ensure_ascii=False)

    return rendered


def _render_mapping(mapping: Mapping[Any, object]) -> str:
    serialized: dict[str, Any] = {}
    for key, item in mapping.items():
        serialized[str(key)] = _normalize_mapping_value(item)
    return json.dumps(serialized, ensure_ascii=False)


def _normalize_mapping_value(value: object) -> object:
    normalized: object
    if _is_dataclass_instance(value):
        try:
            normalized = dump(value, exclude_none=True)
        except TypeError:
            normalized = repr(value)
    elif isinstance(value, Mapping):
        typed_mapping = cast(Mapping[Any, object], value)
        normalized = {
            str(key): _normalize_mapping_value(item)
            for key, item in typed_mapping.items()
        }
    elif isinstance(value, Sequence) and not isinstance(value, _SEQUENCE_EXCLUSIONS):
        sequence_values = cast(Sequence[object], value)
        normalized = [_normalize_mapping_value(item) for item in sequence_values]
    elif isinstance(value, (str, int, float, bool)) or value is None:
        normalized = value
    elif isinstance(value, bytes):
        normalized = value.decode("utf-8", errors="replace")
    else:
        normalized = str(value)

    return normalized


def _render_sequence(items: Sequence[object]) -> str:
    rendered_sequence = "[]"
    if not items:
        return rendered_sequence

    rendered_items = [render_tool_payload(item) for item in items]
    try:
        normalized = [json.loads(item) for item in rendered_items]
    except (TypeError, ValueError):
        trimmed = [item for item in rendered_items if item]
        rendered_sequence = "\n\n".join(trimmed) if trimmed else rendered_sequence
    else:
        rendered_sequence = json.dumps(normalized, ensure_ascii=False)

    return rendered_sequence


__all__ = ["ToolResult", "render_tool_payload"]
