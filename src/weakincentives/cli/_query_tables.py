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

"""Table/slice creation and tool extraction for the query module."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ..types import JSONValue
from ._query_helpers import (
    _infer_sqlite_type,
    _json_to_sql_value,
    _normalize_slice_type,
)

__all__ = [
    "_VIEW_DEFINITIONS",
    "_create_dynamic_slice_table",
    "_extract_slices_from_snapshot",
    "_extract_tool_call_from_entry",
    "_insert_error_from_log",
    "_insert_errors_from_tool_calls",
    "_is_tool_event",
    "_process_session_line",
]


def _is_tool_event(event: str) -> bool:
    """Check if an event string represents a tool call event."""
    event_lower = event.lower()
    return "tool" in event_lower and (
        "call" in event_lower or "result" in event_lower or "execut" in event_lower
    )


def _extract_tool_call_from_entry(
    entry: dict[str, Any],
) -> tuple[str, str, str, str, int, str, float] | None:
    """Extract tool call data from a log entry."""
    context_raw: Any = entry.get("context", {})
    if not isinstance(context_raw, dict):
        return None

    ctx = cast("dict[str, Any]", context_raw)
    tool_name: str = str(ctx.get("tool_name") or ctx.get("tool") or "")
    if not tool_name:
        return None

    success_val: Any = ctx.get("success")
    if success_val is not None:
        success = 1 if success_val else 0
    else:
        context_str = str(ctx)
        success = 0 if "error" in context_str.lower() else 1

    error_code = ""
    if success == 0:
        err_code: Any = ctx.get("error_code") or ctx.get("error") or ctx.get("message")
        error_code = str(err_code) if err_code else ""

    duration: Any = ctx.get("duration_ms")
    duration_ms: float = float(duration) if duration is not None else 0.0

    params: Any = ctx.get("arguments") or ctx.get("params") or {}

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


def _create_dynamic_slice_table(
    conn: sqlite3.Connection,
    slice_type: str,
    slices: Sequence[Mapping[str, JSONValue]],
) -> None:
    """Create a typed table for a specific slice type."""
    table_name = _normalize_slice_type(slice_type)

    columns: dict[str, str] = {"rowid": "INTEGER PRIMARY KEY"}
    for slice_data in slices:
        for key, value in slice_data.items():
            if key == "__type__":
                continue
            if key not in columns:
                columns[key] = _infer_sqlite_type(value)

    col_defs = ", ".join(
        f'"{col}" {typ}' for col, typ in columns.items() if col != "rowid"
    )
    if col_defs:
        col_defs = "rowid INTEGER PRIMARY KEY, " + col_defs
    else:
        col_defs = "rowid INTEGER PRIMARY KEY"

    safe_name = re.sub(r"[^a-z0-9_]", "_", table_name.lower())
    _ = conn.execute(f"CREATE TABLE IF NOT EXISTS {safe_name} ({col_defs})")  # nosec B608

    col_names = [c for c in columns if c != "rowid"]
    if col_names:
        placeholders = ", ".join("?" for _ in col_names)
        col_list = ", ".join(f'"{c}"' for c in col_names)
        for slice_data in slices:
            values = [_json_to_sql_value(slice_data.get(c)) for c in col_names]
            _ = conn.execute(
                f"INSERT INTO {safe_name} ({col_list}) VALUES ({placeholders})",  # nosec B608
                values,
            )


def _insert_error_from_log(conn: sqlite3.Connection, entry: dict[str, Any]) -> None:
    """Insert an error row from a log entry with level=ERROR."""
    context_raw: Any = entry.get("context", {})
    tb = ""
    if isinstance(context_raw, dict):
        ctx = cast("dict[str, Any]", context_raw)
        tb_val: Any = ctx.get("traceback", "")
        tb = str(tb_val)
    _ = conn.execute(
        """
        INSERT INTO errors (source, error_type, message, traceback)
        VALUES (?, ?, ?, ?)
    """,
        (
            "log",
            entry.get("event", ""),
            entry.get("message", ""),
            tb,
        ),
    )


def _insert_errors_from_tool_calls(conn: sqlite3.Connection) -> None:
    """Insert errors from failed tool calls."""
    cursor = conn.execute("SELECT * FROM tool_calls WHERE success = 0")
    for row in cursor.fetchall():
        _ = conn.execute(
            """
            INSERT INTO errors (source, error_type, message, traceback)
            VALUES (?, ?, ?, ?)
        """,
            (
                "tool_call",
                f"ToolError:{row['tool_name']}",
                row["error_code"],
                "",
            ),
        )


def _extract_slices_from_snapshot(
    entry_raw: Mapping[str, object],
) -> list[tuple[str, Mapping[str, JSONValue]]]:
    """Extract slice type and items from a session snapshot entry."""
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


def _insert_session_slice(
    conn: sqlite3.Connection,
    slice_type: str,
    item: Mapping[str, JSONValue],
    slices_by_type: dict[str, list[Mapping[str, JSONValue]]],
) -> None:
    """Insert a session slice into the database and tracking dict."""
    _ = conn.execute(
        """
        INSERT INTO session_slices (slice_type, data)
        VALUES (?, ?)
    """,
        (slice_type, json.dumps(item)),
    )
    if slice_type not in slices_by_type:
        slices_by_type[slice_type] = []
    slices_by_type[slice_type].append(item)


def _process_session_line(
    conn: sqlite3.Connection,
    line: str,
    slices_by_type: dict[str, list[Mapping[str, JSONValue]]],
) -> None:
    """Process a single JSONL line from session data."""
    try:
        entry_raw: Any = json.loads(line)
    except json.JSONDecodeError:
        return

    if not isinstance(entry_raw, Mapping):
        return

    entry_mapping = cast("Mapping[str, object]", entry_raw)

    if "slices" in entry_mapping:
        for slice_type, item_entry in _extract_slices_from_snapshot(entry_mapping):
            _insert_session_slice(conn, slice_type, item_entry, slices_by_type)
    else:
        entry = cast("Mapping[str, JSONValue]", entry_mapping)
        type_val: Any = entry.get("__type__")
        slice_type = str(type_val) if type_val is not None else "unknown"
        _insert_session_slice(conn, slice_type, entry, slices_by_type)


# SQL view definitions extracted for readability.
_VIEW_DEFINITIONS: tuple[str, ...] = (
    """
    CREATE VIEW IF NOT EXISTS tool_timeline AS
    SELECT
        rowid,
        timestamp,
        tool_name,
        json_extract(params, '$.command') as command,
        success,
        duration_ms,
        CASE WHEN success = 0 THEN error_code ELSE NULL END as error
    FROM tool_calls
    ORDER BY timestamp
    """,
    """
    CREATE VIEW IF NOT EXISTS native_tool_calls AS
    SELECT
        seq as sequence_number,
        timestamp,
        'log_aggregator' as source,
        NULL as tool_name,
        NULL as tool_use_id,
        json_extract(context, '$.file') as source_file,
        json_extract(context, '$.content') as content,
        json_extract(context, '$.content') as raw_json
    FROM logs
    WHERE event = 'log_aggregator.log_line'
      AND json_extract(context, '$.content') LIKE '%"type":"tool_%'
    UNION ALL
    SELECT
        sequence_number,
        timestamp,
        transcript_source as source,
        tool_name,
        tool_use_id,
        NULL as source_file,
        content,
        raw_json
    FROM transcript
    WHERE tool_name IS NOT NULL AND tool_name != ''
    """,
    """
    CREATE VIEW IF NOT EXISTS transcript_entries AS
    SELECT
        rowid,
        timestamp,
        prompt_name,
        transcript_source,
        sequence_number,
        entry_type,
        role,
        content,
        tool_name,
        tool_use_id,
        raw_json,
        parsed
    FROM transcript
    """,
    """
    CREATE VIEW IF NOT EXISTS transcript_flow AS
    SELECT
        sequence_number,
        transcript_source,
        entry_type,
        role,
        CASE
            WHEN entry_type IN ('user_message', 'assistant_message') THEN
                CASE
                    WHEN LENGTH(content) > 100 THEN SUBSTR(content, 1, 97) || '...'
                    ELSE content
                END
            WHEN entry_type = 'thinking' THEN '[THINKING]'
            WHEN entry_type = 'tool_result' AND tool_name IS NOT NULL THEN
                '[TOOL RESULT: ' || tool_name || ']'
            WHEN entry_type = 'tool_result' THEN '[TOOL RESULT]'
            ELSE '[' || entry_type || ']'
        END as message_preview,
        timestamp
    FROM transcript
    ORDER BY transcript_source, sequence_number
    """,
    """
    CREATE VIEW IF NOT EXISTS transcript_tools AS
    SELECT
        t1.sequence_number as call_seq,
        t2.sequence_number as result_seq,
        t1.transcript_source,
        t1.tool_name,
        t1.tool_use_id,
        t1.content as tool_params,
        t2.content as tool_result,
        t1.timestamp as call_time,
        t2.timestamp as result_time
    FROM transcript t1
    LEFT JOIN transcript t2
        ON t1.tool_use_id = t2.tool_use_id
        AND t2.entry_type = 'tool_result'
        AND t1.transcript_source = t2.transcript_source
    WHERE t1.entry_type IN ('assistant_message', 'tool_use')
        AND t1.tool_name IS NOT NULL
        AND t1.tool_name != ''
    ORDER BY t1.sequence_number
    """,
    """
    CREATE VIEW IF NOT EXISTS transcript_thinking AS
    SELECT
        sequence_number,
        transcript_source,
        CASE
            WHEN LENGTH(content) > 200 THEN SUBSTR(content, 1, 197) || '...'
            ELSE content
        END as thinking_preview,
        LENGTH(content) as thinking_length,
        timestamp
    FROM transcript
    WHERE entry_type = 'thinking'
        AND content IS NOT NULL
        AND content != ''
    ORDER BY sequence_number
    """,
    """
    CREATE VIEW IF NOT EXISTS transcript_agents AS
    SELECT
        transcript_source,
        CASE
            WHEN transcript_source LIKE 'subagent:%' THEN
                SUBSTR(transcript_source, 10)
            ELSE NULL
        END as agent_id,
        MIN(sequence_number) as first_entry,
        MAX(sequence_number) as last_entry,
        COUNT(*) as total_entries,
        COUNT(CASE WHEN entry_type = 'user_message' THEN 1 END) as user_messages,
        COUNT(CASE WHEN entry_type = 'assistant_message' THEN 1 END) as assistant_messages,
        COUNT(CASE WHEN entry_type = 'thinking' THEN 1 END) as thinking_blocks,
        COUNT(DISTINCT CASE WHEN tool_name IS NOT NULL AND tool_name != '' THEN tool_name END) as unique_tools,
        COUNT(CASE WHEN tool_name IS NOT NULL AND tool_name != '' THEN 1 END) as total_tool_calls
    FROM transcript
    GROUP BY transcript_source
    ORDER BY MIN(sequence_number)
    """,
    """
    CREATE VIEW IF NOT EXISTS error_summary AS
    SELECT
        source,
        error_type,
        message,
        SUBSTR(traceback, 1, 200) as traceback_head
    FROM errors
    """,
)
