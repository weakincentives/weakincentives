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

"""Transcript extraction helpers for the query module."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from ._query_helpers import _safe_json_dumps

__all__ = [
    "_TRANSCRIPT_INSERT_SQL",
    "_apply_bridged_tool_details",
    "_apply_notification_item_details",
    "_apply_split_block_details",
    "_apply_tool_result_details",
    "_apply_transcript_content_fallbacks",
    "_extract_tool_use_from_content",
    "_extract_transcript_details",
    "_extract_transcript_message_details",
    "_extract_transcript_parsed_obj",
    "_extract_transcript_row",
    "_stringify_transcript_content",
    "_stringify_transcript_mapping",
    "_stringify_transcript_tool_use",
]

_TRANSCRIPT_INSERT_SQL = """
    INSERT INTO transcript (
        timestamp, prompt_name, transcript_source, sequence_number,
        entry_type, role, content, tool_name, tool_use_id, raw_json, parsed
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _stringify_transcript_tool_use(mapping: Mapping[str, object]) -> str:
    name = mapping.get("name")
    tool_id = mapping.get("id") or mapping.get("tool_use_id")
    input_val = mapping.get("input")
    parts: list[str] = []
    if name:
        parts.append(str(name))
    if tool_id:
        parts.append(str(tool_id))
    if input_val is not None:
        parts.append(_safe_json_dumps(input_val))
    detail = " ".join(parts).strip()
    return f"[tool_use] {detail}".strip()


def _stringify_transcript_mapping(mapping: Mapping[str, object]) -> str:
    for key in ("text", "content", "thinking", "summary"):
        if key not in mapping:
            continue
        extracted = _stringify_transcript_content(mapping.get(key))
        if extracted:
            return extracted
    if mapping.get("type") == "tool_use":
        return _stringify_transcript_tool_use(mapping)
    return _safe_json_dumps(mapping)


def _stringify_transcript_content(value: object) -> str:
    """Extract a readable string from transcript content blocks."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        items = cast("list[object]", value)
        parts = (_stringify_transcript_content(item) for item in items)
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        return _stringify_transcript_mapping(cast("Mapping[str, object]", value))
    return str(value)


def _extract_tool_use_from_content(content: object) -> tuple[str, str]:
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
            tool_id = (
                candidate.get("id")
                or candidate.get("tool_use_id")
                or candidate.get("tool_call_id")
            )
            return (
                str(name) if name is not None else "",
                str(tool_id) if tool_id is not None else "",
            )
    return "", ""


def _apply_split_block_details(
    parsed: Mapping[str, object],
    entry_type: str,
    *,
    content: str,
    tool_name: str,
    tool_use_id: str,
) -> tuple[str, str, str]:
    """Extract details from split tool_use/tool_result blocks."""
    if entry_type == "tool_use" and not tool_name and parsed.get("type") == "tool_use":
        name_val = parsed.get("name")
        if isinstance(name_val, str):
            tool_name = name_val
        id_val = (
            parsed.get("id") or parsed.get("tool_use_id") or parsed.get("tool_call_id")
        )
        if isinstance(id_val, str):
            tool_use_id = id_val
    if (
        entry_type == "tool_result"
        and not tool_use_id
        and parsed.get("type") == "tool_result"
    ):
        id_val = parsed.get("tool_use_id") or parsed.get("tool_call_id")
        if isinstance(id_val, str):
            tool_use_id = id_val
        content_val = parsed.get("content")
        if not content and isinstance(content_val, str):
            content = content_val
    return content, tool_name, tool_use_id


def _extract_transcript_message_details(
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
    content = _stringify_transcript_content(message.get("content"))
    tool_name, tool_use_id = _extract_tool_use_from_content(message.get("content"))
    return role, content, tool_name, tool_use_id


def _apply_tool_result_details(
    parsed: Mapping[str, object],
    *,
    content: str,
    tool_name: str,
    tool_use_id: str,
) -> tuple[str, str, str]:
    resolved_tool_use_id = tool_use_id
    tool_id_val = parsed.get("tool_use_id") or parsed.get("tool_call_id")
    if isinstance(tool_id_val, str):
        resolved_tool_use_id = resolved_tool_use_id or tool_id_val

    resolved_tool_name = tool_name
    name_val = parsed.get("tool_name")
    if isinstance(name_val, str):
        resolved_tool_name = resolved_tool_name or name_val

    resolved_content = content
    if not resolved_content:
        resolved_content = _stringify_transcript_content(parsed.get("content"))

    return resolved_content, resolved_tool_name, resolved_tool_use_id


def _get_notification_item(
    parsed: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return ``notification.item`` if both are mappings, else None."""
    notif_raw = parsed.get("notification")
    if not isinstance(notif_raw, Mapping):
        return None
    item_raw = cast("Mapping[str, object]", notif_raw).get("item")
    if not isinstance(item_raw, Mapping):
        return None
    return cast("Mapping[str, object]", item_raw)


def _apply_bridged_tool_details(
    parsed: Mapping[str, object],
    notification: Mapping[str, object],
    *,
    tool_use_id: str,
    tool_name: str,
    content: str,
) -> tuple[str, str, str]:
    """Extract tool metadata from bridged WINK tool events."""
    resolved_id = tool_use_id
    call_id = notification.get("callId")
    if isinstance(call_id, str) and not resolved_id:
        resolved_id = call_id

    resolved_name = tool_name
    tool_val = notification.get("tool")
    if isinstance(tool_val, str) and not resolved_name:
        resolved_name = tool_val

    resolved_content = content
    if not resolved_content:
        result_val = parsed.get("result")
        if isinstance(result_val, str) and result_val:
            resolved_content = result_val
        elif result_val is not None:
            resolved_content = _safe_json_dumps(result_val)

    return resolved_id, resolved_name, resolved_content


def _apply_notification_item_details(
    parsed: Mapping[str, object],
    *,
    tool_use_id: str,
    tool_name: str,
    content: str,
) -> tuple[str, str, str]:
    """Extract tool_use_id, tool_name, content from Codex notification.item."""
    item = _get_notification_item(parsed)
    if item is None:
        notif_raw = parsed.get("notification")
        if isinstance(notif_raw, Mapping):
            return _apply_bridged_tool_details(
                parsed,
                cast("Mapping[str, object]", notif_raw),
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                content=content,
            )
        return tool_use_id, tool_name, content

    resolved_id = tool_use_id
    item_id = item.get("id")
    if isinstance(item_id, str) and not resolved_id:
        resolved_id = item_id

    resolved_name = tool_name
    if not resolved_name:
        cmd = item.get("command")
        resolved_name = cmd if isinstance(cmd, str) else str(item.get("type", ""))

    resolved_content = content
    if not resolved_content:
        output = item.get("aggregatedOutput")
        if isinstance(output, str) and output:
            resolved_content = output

    return resolved_id, resolved_name, resolved_content


def _get_notification_item_text(parsed: Mapping[str, object]) -> str:
    """Return ``notification.item.text`` if present, else empty string."""
    item = _get_notification_item(parsed)
    if item is None:
        return ""
    text_val = item.get("text")
    return text_val if isinstance(text_val, str) else ""


def _apply_transcript_content_fallbacks(
    parsed: Mapping[str, object],
    entry_type: str,
    content: str,
) -> str:
    if content:
        return content

    if entry_type == "thinking":
        content = _stringify_transcript_content(parsed.get("thinking"))
    elif entry_type == "summary":
        content = _stringify_transcript_content(parsed.get("summary"))
    elif entry_type == "system":
        content = _stringify_transcript_content(
            parsed.get("details") or parsed.get("event")
        )
    else:
        content = _stringify_transcript_content(parsed.get("content"))

    if not content:
        content = _get_notification_item_text(parsed)

    if not content:
        content = _stringify_transcript_content(parsed)
    return content


def _extract_transcript_details(
    parsed: Mapping[str, object],
    entry_type: str,
) -> tuple[str, str, str, str]:
    """Extract role, content, tool_name, tool_use_id from parsed transcript."""
    role, content, tool_name, tool_use_id = _extract_transcript_message_details(parsed)
    if entry_type == "tool_result":
        content, tool_name, tool_use_id = _apply_tool_result_details(
            parsed,
            content=content,
            tool_name=tool_name,
            tool_use_id=tool_use_id,
        )
    if not tool_use_id and entry_type in {"tool_use", "tool_result"}:
        tool_use_id, tool_name, content = _apply_notification_item_details(
            parsed,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            content=content,
        )
    content, tool_name, tool_use_id = _apply_split_block_details(
        parsed,
        entry_type,
        content=content,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
    )
    # Fallback: ACP adapter uses "tool_call_id" and direct "tool_name" keys
    # in its detail payload instead of the Anthropic-style "tool_use_id"/"id".
    if not tool_use_id:
        call_id = parsed.get("tool_call_id")
        if isinstance(call_id, str):
            tool_use_id = call_id
    if not tool_name:
        name_val = parsed.get("tool_name")
        if isinstance(name_val, str):
            tool_name = name_val
    content = _apply_transcript_content_fallbacks(parsed, entry_type, content)
    return role, content, tool_name, tool_use_id


def _extract_transcript_parsed_obj(
    context: Mapping[str, object],
    raw_json: str | None,
) -> Mapping[str, object] | None:
    detail_raw = context.get("detail")
    if isinstance(detail_raw, Mapping):
        detail = cast("Mapping[str, object]", detail_raw)
        sdk_entry = detail.get("sdk_entry")
        if isinstance(sdk_entry, Mapping):
            return cast("Mapping[str, object]", sdk_entry)
        return detail

    if raw_json is None:
        return None
    try:
        parsed_candidate = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed_candidate, Mapping):
        return cast("Mapping[str, object]", parsed_candidate)
    return None


def _coerce_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _extract_transcript_details_tuple(
    parsed: Mapping[str, object] | None,
    entry_type: str,
) -> tuple[str, str, str, str, str, str | None]:
    if parsed is None:
        return entry_type, "", "", "", "", None
    role, content, tool_name, tool_use_id = _extract_transcript_details(
        parsed,
        entry_type,
    )
    return (
        entry_type,
        role,
        content,
        tool_name,
        tool_use_id,
        _safe_json_dumps(parsed),
    )


def _extract_transcript_row(
    entry: Mapping[str, object],
) -> (
    tuple[str, str, str, int | None, str, str, str, str, str, str | None, str | None]
    | None
):
    """Extract a row for the transcript table from a log entry."""
    if entry.get("event") != "transcript.entry":
        return None

    ctx_raw = entry.get("context")
    if not isinstance(ctx_raw, Mapping):
        return None
    ctx = cast("Mapping[str, object]", ctx_raw)

    prompt_name = str(ctx.get("prompt_name") or "")
    transcript_source = str(ctx.get("source") or "")
    entry_type = str(ctx.get("entry_type") or "unknown")

    sequence_number = _coerce_int(ctx.get("sequence_number"))
    raw_json = _coerce_str(ctx.get("raw"))
    parsed_obj = _extract_transcript_parsed_obj(ctx, raw_json)

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
