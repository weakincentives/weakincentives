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
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ..serde import dump
from ..types.dataclass import SupportsDataclass, is_dataclass_instance

_LOGGER = logging.getLogger(__name__)

_SEQUENCE_EXCLUSIONS = (str, bytes, bytearray)


@dataclass(slots=True)
class ToolResult[ResultValueT]:
    """Structured response returned by tool handlers to the runtime.

    Tool handlers return ToolResult to communicate both a typed value for
    programmatic access and a human-readable message for the LLM context.
    Use the class methods `ok()` and `error()` to construct results.

    Attributes:
        message: Human-readable text included in the LLM conversation context.
            Should describe the outcome clearly for the model.
        value: Typed payload accessible programmatically. May be None on error
            or when no return value is meaningful.
        success: Whether the tool invocation succeeded. Failed results trigger
            automatic rollback in transactional contexts.
        exclude_value_from_context: If True, the rendered value is omitted from
            the LLM context while still being available programmatically. Useful
            for large payloads that would consume excessive tokens.

    Example:
        >>> def read_file(params: ReadParams, *, context: ToolContext) -> ToolResult[str]:
        ...     content = Path(params.path).read_text()
        ...     return ToolResult.ok(content, message=f"Read {len(content)} bytes")
        ...
        >>> def validate(params: Params, *, context: ToolContext) -> ToolResult[None]:
        ...     if not params.valid:
        ...         return ToolResult.error("Validation failed: missing required field")
        ...     return ToolResult.ok(None, message="Validation passed")
    """

    message: str
    value: ResultValueT | None
    success: bool = True
    exclude_value_from_context: bool = False

    @classmethod
    def ok(cls, value: ResultValueT, message: str = "OK") -> ToolResult[ResultValueT]:
        """Create a successful result with the given value.

        Args:
            value: The typed payload to include in the result.
            message: Human-readable message for the LLM (default: "OK").

        Returns:
            A ToolResult with success=True.
        """
        return cls(message=message, value=value, success=True)

    @classmethod
    def error(cls, message: str) -> ToolResult[None]:
        """Create a failed result with no value.

        Args:
            message: Error message describing the failure.

        Returns:
            A ToolResult with success=False and value=None.
        """
        return ToolResult(message=message, value=None, success=False)

    def render(self) -> str:
        """Convert the result value to a string for LLM context inclusion.

        Delegates to `render_tool_payload()` to produce a text representation
        suitable for including in the conversation. Dataclasses with a custom
        `render()` method use that; otherwise values are serialized to JSON.

        Returns:
            String representation of `self.value`. Returns empty string if
            value is None.
        """
        return render_tool_payload(self.value)


def render_tool_payload(value: object) -> str:
    """Convert a tool result payload to a string for LLM context and telemetry.

    Transforms arbitrary Python values into text suitable for including in
    conversation context. The rendering strategy depends on the value type:

    - **None**: Returns empty string.
    - **Dataclass with render()**: Calls the custom `render()` method. If the
      dataclass lacks `render()`, logs a warning and falls back to JSON.
    - **Mapping (dict, etc.)**: Serializes to JSON with nested dataclasses
      converted recursively.
    - **Sequence (list, tuple)**: Renders each item and combines as JSON array,
      or newline-separated text if JSON parsing fails.
    - **bytes**: Decodes as UTF-8 with replacement for invalid characters.
    - **Other types**: Converts via `str()`.

    Args:
        value: The payload to render. Accepts any object type.

    Returns:
        String representation suitable for LLM consumption. Empty string for None.

    Note:
        Dataclass results should implement a `render()` method returning a string
        to control their text representation. Without it, the dataclass is
        serialized to JSON, which may be verbose or expose internal structure.
    """
    if value is None:
        return ""

    if is_dataclass_instance(value):
        # ty doesn't recognize TypeGuard narrowing; cast required for ty
        return _render_dataclass(cast(SupportsDataclass, value))  # pyright: ignore[reportUnnecessaryCast]

    if isinstance(value, Mapping):
        typed_mapping = cast(Mapping[Any, object], value)
        return _render_mapping(typed_mapping)

    if isinstance(value, Sequence) and not isinstance(value, _SEQUENCE_EXCLUSIONS):
        sequence_items = cast(Sequence[object], value)
        return _render_sequence(sequence_items)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


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
    else:
        _LOGGER.warning(
            "Tool result dataclass %s is missing a render() implementation; falling back to serialization.",
            type(value).__name__,
            extra={
                "event": "tool_result.render.missing",
                "dataclass": type(value).__name__,
            },
        )

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
    if is_dataclass_instance(value):
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
