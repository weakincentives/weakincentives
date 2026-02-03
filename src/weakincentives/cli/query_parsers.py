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

"""Parsing helpers for query-related logs and transcripts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from ..dbc import pure
from ..types import JSONValue


@pure
def is_tool_event(event: str) -> bool:
    """Check if an event string represents a tool call event.

    Matches events like:
    - tool.execution.start, tool.execution.complete (actual log events)
    - tool.execute.*, tool.call.*, tool.result.* (alternative formats)
    """
    event_lower = event.lower()
    return "tool" in event_lower and (
        "call" in event_lower
        or "result" in event_lower
        or "execut" in event_lower  # matches both "execute" and "execution"
    )


@pure
def safe_json_dumps(value: object) -> str:
    """Serialize value to JSON, falling back to str on failure."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


@pure
def stringify_transcript_tool_use(mapping: Mapping[str, object]) -> str:
    name = mapping.get("name")
    tool_id = mapping.get("id") or mapping.get("tool_use_id")
    input_val = mapping.get("input")
    parts: list[str] = []
    if name:
        parts.append(str(name))
    if tool_id:
        parts.append(str(tool_id))
    if input_val is not None:
        parts.append(safe_json_dumps(input_val))
    detail = " ".join(parts).strip()
    return f"[tool_use] {detail}".strip()


@pure
def stringify_transcript_mapping(mapping: Mapping[str, object]) -> str:
    for key in ("text", "content", "thinking", "summary"):
        if key not in mapping:
            continue
        extracted = stringify_transcript_content(mapping.get(key))
        if extracted:
            return extracted
    if mapping.get("type") == "tool_use":
        return stringify_transcript_tool_use(mapping)
    return safe_json_dumps(mapping)


@pure
def stringify_transcript_content(value: object) -> str:
    """Extract a readable string from transcript content blocks."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        items = cast("list[object]", value)
        parts = (stringify_transcript_content(item) for item in items)
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        return stringify_transcript_mapping(cast("Mapping[str, object]", value))
    return str(value)


@pure
def extract_tool_use_from_content(content: object) -> tuple[str, str]:
    """Extract tool name and ID from a message content block."""
    candidates: list[Mapping[str, object]] = []
    if isinstance(content, Mapping):
        candidates.append(cast("Mapping[str, object]", content))
    elif isinstance(content, list):
        items = cast("list[object]", content)
        candidates.extend(
            cast("Mapping[str, object]", item)
            for item in items
            if isinstance(item, Mapping)
        )

    for candidate in candidates:
        if candidate.get("type") == "tool_use":
            name = candidate.get("name")
            tool_id = candidate.get("id") or candidate.get("tool_use_id")
            return (
                str(name) if name is not None else "",
                str(tool_id) if tool_id is not None else "",
            )
    return "", ""


@pure
def extract_transcript_details(
    parsed: Mapping[str, object],
    entry_type: str,
) -> tuple[str, str, str, str]:
    """Extract role, content, tool_name, tool_use_id from parsed transcript."""
    role, content, tool_name, tool_use_id = extract_transcript_message_details(parsed)
    if entry_type == "tool_result":
        content, tool_name, tool_use_id = apply_tool_result_details(
            parsed,
            content=content,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
        )
    content = apply_transcript_content_fallbacks(parsed, entry_type, content)
    return role, content, tool_name, tool_use_id


@pure
def extract_transcript_message_details(
    parsed: Mapping[str, object],
) -> tuple[str, str, str, str]:
    role = ""
    content = ""
    tool_name = ""
    tool_use_id = ""

    message_raw = parsed.get("message")
    if not isinstance(message_raw, Mapping):
        return role, content, tool_name, tool_use_id

    message = cast("Mapping[str, object]", message_raw)
    role_val = message.get("role")
    if isinstance(role_val, str):
        role = role_val
    content = stringify_transcript_content(message.get("content"))
    tool_name, tool_use_id = extract_tool_use_from_content(message.get("content"))
    return role, content, tool_name, tool_use_id


@pure
def apply_tool_result_details(
    parsed: Mapping[str, object],
    *,
    content: str,
    tool_name: str,
    tool_use_id: str,
) -> tuple[str, str, str]:
    resolved_tool_use_id = tool_use_id
    tool_id_val = parsed.get("tool_use_id")
    if isinstance(tool_id_val, str):
        resolved_tool_use_id = resolved_tool_use_id or tool_id_val

    resolved_tool_name = tool_name
    name_val = parsed.get("tool_name")
    if isinstance(name_val, str):
        resolved_tool_name = resolved_tool_name or name_val

    resolved_content = content
    if not resolved_content:
        resolved_content = stringify_transcript_content(parsed.get("content"))

    return resolved_content, resolved_tool_name, resolved_tool_use_id


@pure
def apply_transcript_content_fallbacks(
    parsed: Mapping[str, object],
    entry_type: str,
    content: str,
) -> str:
    if content:
        return content

    if entry_type == "thinking":
        content = stringify_transcript_content(parsed.get("thinking"))
    elif entry_type == "summary":
        content = stringify_transcript_content(parsed.get("summary"))
    elif entry_type == "system":
        content = stringify_transcript_content(
            parsed.get("details") or parsed.get("event")
        )
    else:
        content = stringify_transcript_content(parsed.get("content"))

    if not content:
        content = stringify_transcript_content(parsed)
    return content


@pure
def extract_transcript_parsed_obj(
    context: Mapping[str, object],
    raw_json: str | None,
) -> Mapping[str, object] | None:
    parsed_raw = context.get("parsed")
    if isinstance(parsed_raw, Mapping):
        return cast("Mapping[str, object]", parsed_raw)
    if raw_json is None:
        return None
    try:
        parsed_candidate = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed_candidate, Mapping):
        return cast("Mapping[str, object]", parsed_candidate)
    return None


@pure
def _coerce_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


@pure
def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


@pure
def _extract_transcript_details_tuple(
    parsed: Mapping[str, object] | None,
    entry_type: str,
) -> tuple[str, str, str, str, str, str | None]:
    if parsed is None:
        return entry_type, "", "", "", "", None
    resolved_entry_type = str(parsed.get("type") or entry_type)
    role, content, tool_name, tool_use_id = extract_transcript_details(
        parsed,
        resolved_entry_type,
    )
    return (
        resolved_entry_type,
        role,
        content,
        tool_name,
        tool_use_id,
        safe_json_dumps(parsed),
    )


@pure
def extract_transcript_row(
    entry: Mapping[str, object],
) -> (
    tuple[str, str, str, int | None, str, str, str, str, str, str | None, str | None]
    | None
):
    """Extract a row for the transcript table from a log entry."""
    if entry.get("event") != "transcript.collector.entry":
        return None

    ctx_raw = entry.get("context")
    if not isinstance(ctx_raw, Mapping):
        return None
    ctx = cast("Mapping[str, object]", ctx_raw)

    prompt_name = str(ctx.get("prompt_name") or "")
    transcript_source = str(ctx.get("transcript_source") or "")
    entry_type = str(ctx.get("entry_type") or "unknown")

    sequence_number = _coerce_int(ctx.get("sequence_number"))
    raw_json = _coerce_str(ctx.get("raw_json"))
    parsed_obj = extract_transcript_parsed_obj(ctx, raw_json)

    resolved_entry_type, role, content, tool_name, tool_use_id, parsed_json = (
        _extract_transcript_details_tuple(parsed_obj, entry_type)
    )
    if not content:
        content = raw_json or ""

    timestamp = str(entry.get("timestamp") or "")

    return (
        timestamp,
        prompt_name,
        transcript_source,
        sequence_number,
        resolved_entry_type,
        role,
        content,
        tool_name,
        tool_use_id,
        raw_json,
        parsed_json,
    )


def extract_tool_call_from_entry(
    entry: dict[str, Any],
) -> tuple[str, str, str, str, int, str, float] | None:
    """Extract tool call data from a log entry.

    Returns tuple of (timestamp, tool_name, params, result, success, error_code,
    duration_ms) or None if not a tool call.
    """
    context_raw: Any = entry.get("context", {})
    if not isinstance(context_raw, dict):
        return None

    ctx = cast("dict[str, Any]", context_raw)
    tool_name: str = str(ctx.get("tool_name") or ctx.get("tool") or "")
    if not tool_name:
        return None

    # Use explicit success field if present (from tool.execution.complete logs)
    # Fall back to inferring from error presence for compatibility
    success_val: Any = ctx.get("success")
    if success_val is not None:
        success = 1 if success_val else 0
    else:
        # Legacy fallback: infer from error text presence
        context_str = str(ctx)
        success = 0 if "error" in context_str.lower() else 1

    error_code = ""
    if success == 0:
        err_code: Any = ctx.get("error_code") or ctx.get("error") or ctx.get("message")
        error_code = str(err_code) if err_code else ""

    duration: Any = ctx.get("duration_ms")
    duration_ms: float = float(duration) if duration is not None else 0.0

    # Read tool arguments: current logs use "arguments", legacy may use "params"
    params: Any = ctx.get("arguments") or ctx.get("params") or {}

    # Read tool result: current logs use "value"/"message", legacy may use "result"
    result_val: Any = ctx.get("value")
    if result_val is None:
        result_val = ctx.get("result") or {}
    result: Any = result_val

    return (
        str(entry.get("timestamp", "")),
        tool_name,
        json.dumps(params),
        json.dumps(result),
        success,
        error_code,
        duration_ms,
    )


@pure
def extract_slices_from_snapshot(
    entry_raw: Mapping[str, object],
) -> list[tuple[str, Mapping[str, JSONValue]]]:
    """Extract slice type and items from a session snapshot entry.

    Returns list of (slice_type, item_entry) tuples.
    """
    result: list[tuple[str, Mapping[str, JSONValue]]] = []
    slices_list: object = entry_raw.get("slices", [])
    if not isinstance(slices_list, list):
        return result

    slices_list_typed = cast("list[object]", slices_list)
    for slice_obj_raw in slices_list_typed:
        if not isinstance(slice_obj_raw, Mapping):
            continue
        slice_mapping = cast("Mapping[str, object]", slice_obj_raw)
        slice_dict: dict[str, object] = dict(slice_mapping)
        items: object = slice_dict.get("items", [])
        slice_type_val: object = slice_dict.get("slice_type", "unknown")
        slice_type = str(slice_type_val)
        if not isinstance(items, list):
            continue

        items_list = cast("list[object]", items)
        for item_raw in items_list:
            if isinstance(item_raw, Mapping):
                item_entry = cast("Mapping[str, JSONValue]", item_raw)
                result.append((slice_type, item_entry))

    return result


@pure
def extract_session_slices_from_line(
    line: str,
) -> list[tuple[str, Mapping[str, JSONValue]]]:
    """Extract session slices from a JSONL line."""
    try:
        entry_raw: Any = json.loads(line)
    except json.JSONDecodeError:
        return []

    if not isinstance(entry_raw, Mapping):
        return []

    entry_mapping = cast("Mapping[str, object]", entry_raw)

    if "slices" in entry_mapping:
        return extract_slices_from_snapshot(entry_mapping)

    entry = cast("Mapping[str, JSONValue]", entry_mapping)
    type_val: Any = entry.get("__type__")
    slice_type = str(type_val) if type_val is not None else "unknown"
    return [(slice_type, entry)]


__all__ = [
    "apply_tool_result_details",
    "apply_transcript_content_fallbacks",
    "extract_session_slices_from_line",
    "extract_slices_from_snapshot",
    "extract_tool_call_from_entry",
    "extract_tool_use_from_content",
    "extract_transcript_details",
    "extract_transcript_message_details",
    "extract_transcript_parsed_obj",
    "extract_transcript_row",
    "is_tool_event",
    "safe_json_dumps",
    "stringify_transcript_content",
    "stringify_transcript_mapping",
    "stringify_transcript_tool_use",
]
