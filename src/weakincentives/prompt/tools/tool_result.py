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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from typing import Any, cast

from ...serde import dump
from .._types import SupportsDataclass

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

    if value is None:
        return ""

    if _is_dataclass_instance(value):
        return _render_dataclass(cast(SupportsDataclass, value))

    if isinstance(value, Mapping):
        typed_mapping = cast(Mapping[Any, object], value)
        return _render_mapping(typed_mapping)

    if isinstance(value, Sequence) and not isinstance(value, _SEQUENCE_EXCLUSIONS):
        sequence_items = cast(Sequence[object], value)
        return _render_sequence(sequence_items)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def _is_dataclass_instance(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _render_dataclass(value: SupportsDataclass) -> str:
    render_method = getattr(value, "render", None)
    if callable(render_method):
        try:
            rendered = render_method()
        except Exception:
            rendered = None
        else:
            if not isinstance(rendered, str):
                rendered = str(rendered)
        if rendered is not None:
            return rendered

    try:
        payload = dump(value, exclude_none=True)
    except TypeError:
        return repr(value)
    return json.dumps(payload, ensure_ascii=False)


def _render_mapping(mapping: Mapping[Any, object]) -> str:
    serialized: dict[str, Any] = {}
    for key, item in mapping.items():
        serialized[str(key)] = _normalize_mapping_value(item)
    return json.dumps(serialized, ensure_ascii=False)


def _normalize_mapping_value(value: object) -> object:
    if _is_dataclass_instance(value):
        try:
            return dump(value, exclude_none=True)
        except TypeError:
            return repr(value)
    if isinstance(value, Mapping):
        typed_mapping = cast(Mapping[Any, object], value)
        return {
            str(key): _normalize_mapping_value(item)
            for key, item in typed_mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, _SEQUENCE_EXCLUSIONS):
        sequence_values = cast(Sequence[object], value)
        return [_normalize_mapping_value(item) for item in sequence_values]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _render_sequence(items: Sequence[object]) -> str:
    if not items:
        return "[]"

    rendered_items = [render_tool_payload(item) for item in items]
    try:
        normalized = [json.loads(item) for item in rendered_items]
    except (TypeError, ValueError):
        trimmed = [item for item in rendered_items if item]
        return "\n\n".join(trimmed) if trimmed else "[]"
    return json.dumps(normalized, ensure_ascii=False)


__all__ = ["ToolResult", "render_tool_payload"]
