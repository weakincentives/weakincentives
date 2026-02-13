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

"""Table builder functions for the query module.

Each function creates one SQLite table from debug bundle data.
Called by ``QueryDatabase.build()`` during database construction.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any, cast

from ..debug import BundleValidationError, DebugBundle
from ..types import JSONValue
from ._query_helpers import _flatten_json
from ._query_tables import (
    _VIEW_DEFINITIONS,
    _create_dynamic_slice_table,
    _extract_tool_call_from_entry,
    _insert_error_from_log,
    _insert_errors_from_tool_calls,
    _is_tool_event,
    _process_session_line,
)
from ._query_transcript import (
    _TRANSCRIPT_INSERT_SQL,
    _extract_transcript_row,
)

__all__ = [
    "_build_config_table",
    "_build_errors_table",
    "_build_eval_table",
    "_build_files_table",
    "_build_logs_table",
    "_build_manifest_table",
    "_build_metrics_table",
    "_build_prompt_overrides_table",
    "_build_run_context_table",
    "_build_session_slices_table",
    "_build_tool_calls_table",
    "_build_transcript_table",
    "_build_views",
]


def _build_manifest_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build manifest table from manifest.json."""
    manifest = bundle.manifest
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS manifest (
            bundle_id TEXT, format_version TEXT, created_at TEXT,
            status TEXT, request_id TEXT, session_id TEXT,
            started_at TEXT, ended_at TEXT, capture_mode TEXT,
            capture_trigger TEXT, prompt_ns TEXT, prompt_key TEXT,
            prompt_adapter TEXT
        )
    """)
    _ = conn.execute(
        "INSERT INTO manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            manifest.bundle_id,
            manifest.format_version,
            manifest.created_at,
            manifest.request.status,
            manifest.request.request_id,
            manifest.request.session_id,
            manifest.request.started_at,
            manifest.request.ended_at,
            manifest.capture.mode,
            manifest.capture.trigger,
            manifest.prompt.ns,
            manifest.prompt.key,
            manifest.prompt.adapter,
        ),
    )


def _build_logs_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build logs table from logs/app.jsonl."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            rowid INTEGER PRIMARY KEY, timestamp TEXT, level TEXT,
            logger TEXT, event TEXT, message TEXT, context TEXT,
            seq INTEGER
        )
    """)
    logs_content = bundle.logs
    if not logs_content:
        return
    for line in logs_content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry_raw: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry_raw, dict):
            continue
        entry = cast("dict[str, Any]", entry_raw)
        seq: int | None = None
        ctx_raw = entry.get("context", {})
        if isinstance(ctx_raw, dict):
            ctx = cast("dict[str, Any]", ctx_raw)
            seq_val: Any = ctx.get("sequence_number")
            if isinstance(seq_val, int):
                seq = seq_val
        _ = conn.execute(
            """INSERT INTO logs (timestamp, level, logger, event, message, context, seq)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.get("timestamp", ""),
                entry.get("level", ""),
                entry.get("logger", ""),
                entry.get("event", ""),
                entry.get("message", ""),
                json.dumps(entry.get("context", {})),
                seq,
            ),
        )


def _build_transcript_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build transcript table from transcript entries."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS transcript (
            rowid INTEGER PRIMARY KEY, timestamp TEXT,
            prompt_name TEXT, transcript_source TEXT,
            sequence_number INTEGER, entry_type TEXT, role TEXT,
            content TEXT, tool_name TEXT, tool_use_id TEXT,
            raw_json TEXT, parsed TEXT
        )
    """)
    inserted = _insert_transcript_from_artifact(conn, bundle)
    if not inserted:
        _insert_transcript_from_logs(conn, bundle)


def _insert_transcript_from_artifact(
    conn: sqlite3.Connection, bundle: DebugBundle
) -> bool:
    """Try loading transcript entries from transcript.jsonl artifact."""
    transcript_entries = bundle.transcript
    if not transcript_entries:
        return False
    count = 0
    for entry in transcript_entries:
        row = _extract_transcript_row(entry)
        if row is None:
            continue
        _ = conn.execute(_TRANSCRIPT_INSERT_SQL, row)
        count += 1
    return count > 0


def _insert_transcript_from_logs(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Fall back to scanning logs/app.jsonl for transcript entries."""
    logs_content = bundle.logs
    if not logs_content:
        return
    for line in logs_content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry_raw: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry_raw, Mapping):
            continue
        entry = cast("Mapping[str, object]", entry_raw)
        row = _extract_transcript_row(entry)
        if row is None:
            continue
        _ = conn.execute(_TRANSCRIPT_INSERT_SQL, row)


def _build_tool_calls_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build tool_calls table derived from logs."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_calls (
            rowid INTEGER PRIMARY KEY, timestamp TEXT,
            tool_name TEXT, params TEXT, result TEXT,
            success INTEGER, error_code TEXT, duration_ms REAL
        )
    """)
    logs_content = bundle.logs
    if not logs_content:
        return
    for line in logs_content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry_raw: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry_raw, dict):
            continue
        entry = cast("dict[str, Any]", entry_raw)
        event = entry.get("event", "")
        if not _is_tool_event(event):
            continue
        tool_data = _extract_tool_call_from_entry(entry)
        if tool_data is None:
            continue
        _ = conn.execute(
            """INSERT INTO tool_calls
                (timestamp, tool_name, params, result, success,
                 error_code, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            tool_data,
        )


def _build_errors_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build errors table aggregating errors from multiple sources."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            rowid INTEGER PRIMARY KEY, source TEXT,
            error_type TEXT, message TEXT, traceback TEXT
        )
    """)
    _insert_errors_from_error_json(conn, bundle)
    _insert_errors_from_logs(conn, bundle)
    _insert_errors_from_tool_calls(conn)


def _insert_errors_from_error_json(
    conn: sqlite3.Connection, bundle: DebugBundle
) -> None:
    """Insert errors from error.json if present."""
    error_data = bundle.error
    if not error_data or not isinstance(error_data, Mapping):
        return
    err = cast("dict[str, Any]", dict(error_data))
    tb_val: Any = err.get("traceback", [])
    tb_str: str
    if isinstance(tb_val, list):
        tb_list = cast("list[Any]", tb_val)
        tb_str = "".join(str(item) for item in tb_list)
    else:
        tb_str = str(tb_val)
    _ = conn.execute(
        """INSERT INTO errors (source, error_type, message, traceback)
        VALUES (?, ?, ?, ?)""",
        (
            "error.json",
            str(err.get("type", "")),
            str(err.get("message", "")),
            tb_str,
        ),
    )


def _insert_errors_from_logs(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Insert errors from log entries with level=ERROR."""
    logs_content = bundle.logs
    if not logs_content:
        return
    for line in logs_content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry_raw: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry_raw, dict):
            continue
        entry = cast("dict[str, Any]", entry_raw)
        if entry.get("level") == "ERROR":
            _insert_error_from_log(conn, entry)


def _build_session_slices_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build session_slices table and dynamic slice tables."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS session_slices (
            rowid INTEGER PRIMARY KEY, slice_type TEXT, data TEXT
        )
    """)
    session_content = bundle.session_after
    if not session_content:
        return
    slices_by_type: dict[str, list[Mapping[str, JSONValue]]] = {}
    for line in session_content.strip().split("\n"):
        if not line.strip():
            continue
        _process_session_line(conn, line, slices_by_type)
    for slice_type, slices in slices_by_type.items():
        _create_dynamic_slice_table(conn, slice_type, slices)


def _build_files_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build files table from filesystem/ directory."""
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            rowid INTEGER PRIMARY KEY, path TEXT,
            content TEXT, size_bytes INTEGER
        )
    """)
    all_files = bundle.list_files()
    for file_path in all_files:
        if file_path.startswith("filesystem/"):
            try:
                content = bundle.read_file(file_path)
                rel_path = file_path[len("filesystem/") :]
                try:
                    text_content = content.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = content.hex()
                _ = conn.execute(
                    "INSERT INTO files (path, content, size_bytes) VALUES (?, ?, ?)",
                    (rel_path, text_content, len(content)),
                )
            except BundleValidationError:
                continue


def _build_config_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build config table from config.json (flattened)."""
    config_data = bundle.config
    if not config_data or not isinstance(config_data, Mapping):
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
            )
        """)
        return
    flattened = _flatten_json(config_data)
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS config (
            rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
        )
    """)
    for key, value in flattened.items():
        _ = conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?)",
            (key, str(value) if value is not None else None),
        )


def _build_metrics_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build metrics table from metrics.json."""
    metrics_data = bundle.metrics
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
        )
    """)
    if metrics_data and isinstance(metrics_data, Mapping):
        flattened = _flatten_json(metrics_data)
        for key, value in flattened.items():
            _ = conn.execute(
                "INSERT INTO metrics (key, value) VALUES (?, ?)",
                (key, str(value) if value is not None else None),
            )


def _build_run_context_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build run_context table from run_context.json."""
    run_context = bundle.run_context
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS run_context (
            rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
        )
    """)
    if run_context and isinstance(run_context, Mapping):
        flattened = _flatten_json(run_context)
        for key, value in flattened.items():
            _ = conn.execute(
                "INSERT INTO run_context (key, value) VALUES (?, ?)",
                (key, str(value) if value is not None else None),
            )


def _build_prompt_overrides_table(
    conn: sqlite3.Connection, bundle: DebugBundle
) -> None:
    """Build prompt_overrides table if file exists."""
    overrides = bundle.prompt_overrides
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_overrides (
            rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
        )
    """)
    if overrides and isinstance(overrides, Mapping):
        flattened = _flatten_json(overrides)
        for key, value in flattened.items():
            _ = conn.execute(
                "INSERT INTO prompt_overrides (key, value) VALUES (?, ?)",
                (key, str(value) if value is not None else None),
            )


def _build_eval_table(conn: sqlite3.Connection, bundle: DebugBundle) -> None:
    """Build eval table if file exists."""
    eval_data = bundle.eval
    _ = conn.execute("""
        CREATE TABLE IF NOT EXISTS eval (
            rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
        )
    """)
    if eval_data and isinstance(eval_data, Mapping):
        flattened = _flatten_json(eval_data)
        for key, value in flattened.items():
            _ = conn.execute(
                "INSERT INTO eval (key, value) VALUES (?, ?)",
                (key, str(value) if value is not None else None),
            )


def _build_views(conn: sqlite3.Connection) -> None:
    """Create SQL views for common query patterns."""
    for sql in _VIEW_DEFINITIONS:
        _ = conn.execute(sql)
