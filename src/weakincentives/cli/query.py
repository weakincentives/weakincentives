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

"""SQL query interface for debug bundles.

Enables querying debug bundle contents via SQL by loading bundle artifacts
into a cached SQLite database.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import Any, cast, override

from ..dataclasses import FrozenDataclass
from ..dbc import pure
from ..debug.bundle import BundleValidationError, DebugBundle
from ..errors import WinkError
from ..resources.protocols import Closeable
from ..types import JSONValue


class QueryError(WinkError, RuntimeError):
    """Raised when a query operation fails."""


# Maximum column width for ASCII table output
_MAX_COLUMN_WIDTH = 50

# Schema version for cache invalidation - increment when schema changes
_SCHEMA_VERSION = 6  # v6: unified transcript (transcript.entry + source/raw/detail)

_TRANSCRIPT_INSERT_SQL = """
    INSERT INTO transcript (
        timestamp, prompt_name, transcript_source, sequence_number,
        entry_type, role, content, tool_name, tool_use_id, raw_json, parsed
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@FrozenDataclass()
class ColumnInfo:
    """Column metadata for schema output."""

    name: str
    type: str
    description: str = ""


@FrozenDataclass()
class TableInfo:
    """Table metadata for schema output."""

    name: str
    description: str
    row_count: int
    columns: tuple[ColumnInfo, ...] = ()


@FrozenDataclass()
class SchemaHints:
    """Hints for querying the database effectively."""

    json_extraction: tuple[str, ...] = ()
    common_queries: dict[str, str] = field(default_factory=lambda: {})


@FrozenDataclass()
class SchemaOutput:
    """Schema output structure."""

    bundle_id: str
    status: str
    created_at: str
    tables: tuple[TableInfo, ...] = field(default_factory=tuple)
    hints: SchemaHints | None = None

    def to_json(self) -> str:
        """Serialize schema to JSON string."""
        from ..serde import dump

        return json.dumps(dump(self), indent=2)


@pure
def _normalize_slice_type(type_name: str) -> str:
    """Normalize a slice type name to a valid table name.

    Example: 'myapp.state:AgentPlan' -> 'slice_agentplan'
    """
    # Extract class name after colon if present
    if ":" in type_name:
        type_name = type_name.split(":")[-1]
    # Extract class name after last dot
    if "." in type_name:
        type_name = type_name.rsplit(".", 1)[-1]
    # Lowercase and prefix
    return f"slice_{type_name.lower()}"


@pure
def _flatten_json(
    obj: JSONValue, prefix: str = "", sep: str = "_"
) -> dict[str, JSONValue]:
    """Flatten nested JSON object into flat key-value pairs."""
    result: dict[str, JSONValue] = {}

    if isinstance(obj, Mapping):
        # Cast to proper type for iteration
        mapping = cast("Mapping[str, JSONValue]", obj)
        for key, value in mapping.items():
            new_key = f"{prefix}{sep}{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                nested = cast(JSONValue, value)
                result.update(_flatten_json(nested, new_key, sep))
            elif isinstance(value, list):
                # Store lists as JSON strings
                result[new_key] = json.dumps(value)
            else:
                result[new_key] = value
    elif isinstance(obj, list):
        result[prefix] = json.dumps(obj)
    else:
        result[prefix] = obj

    return result


@pure
def _infer_sqlite_type(value: object) -> str:
    """Infer SQLite type from Python value."""
    if value is None:
        return "TEXT"
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


@pure
def _json_to_sql_value(value: JSONValue) -> object:
    """Convert JSON value to SQL-compatible value."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float | str):
        return value
    # Lists and dicts become JSON strings
    return json.dumps(value)


@pure
def _is_tool_event(event: str) -> bool:
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
def _safe_json_dumps(value: object) -> str:
    """Serialize value to JSON, falling back to str on failure."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


@pure
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


@pure
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


@pure
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


@pure
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
            tool_id = candidate.get("id") or candidate.get("tool_use_id")
            return (
                str(name) if name is not None else "",
                str(tool_id) if tool_id is not None else "",
            )
    return "", ""


@pure
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
    content = _apply_transcript_content_fallbacks(parsed, entry_type, content)
    return role, content, tool_name, tool_use_id


@pure
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


@pure
def _apply_tool_result_details(
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
        resolved_content = _stringify_transcript_content(parsed.get("content"))

    return resolved_content, resolved_tool_name, resolved_tool_use_id


@pure
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
        content = _stringify_transcript_content(parsed)
    return content


@pure
def _extract_transcript_parsed_obj(
    context: Mapping[str, object],
    raw_json: str | None,
) -> Mapping[str, object] | None:
    # Unified format: context.detail (may contain sdk_entry or notification)
    detail_raw = context.get("detail")
    if isinstance(detail_raw, Mapping):
        detail = cast("Mapping[str, object]", detail_raw)
        # Claude SDK wraps in sdk_entry; unwrap so downstream sees message/type
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
    role, content, tool_name, tool_use_id = _extract_transcript_details(
        parsed,
        resolved_entry_type,
    )
    return (
        resolved_entry_type,
        role,
        content,
        tool_name,
        tool_use_id,
        _safe_json_dumps(parsed),
    )


@pure
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


def _extract_tool_call_from_entry(
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


def _create_dynamic_slice_table(
    conn: sqlite3.Connection,
    slice_type: str,
    slices: Sequence[Mapping[str, JSONValue]],
) -> None:
    """Create a typed table for a specific slice type."""
    table_name = _normalize_slice_type(slice_type)

    # Collect all column names and infer types from first non-null value
    columns: dict[str, str] = {"rowid": "INTEGER PRIMARY KEY"}
    for slice_data in slices:
        for key, value in slice_data.items():
            if key == "__type__":
                continue
            if key not in columns:
                columns[key] = _infer_sqlite_type(value)

    # Create table with discovered columns
    col_defs = ", ".join(
        f'"{col}" {typ}' for col, typ in columns.items() if col != "rowid"
    )
    if col_defs:
        col_defs = "rowid INTEGER PRIMARY KEY, " + col_defs
    else:
        col_defs = "rowid INTEGER PRIMARY KEY"

    # Sanitize table name (alphanumeric and underscore only)
    safe_name = re.sub(r"[^a-z0-9_]", "_", table_name.lower())
    # safe_name is sanitized via regex; col_defs uses quoted identifiers
    _ = conn.execute(f"CREATE TABLE IF NOT EXISTS {safe_name} ({col_defs})")  # nosec B608

    # Insert data
    col_names = [c for c in columns if c != "rowid"]
    if col_names:
        placeholders = ", ".join("?" for _ in col_names)
        col_list = ", ".join(f'"{c}"' for c in col_names)
        for slice_data in slices:
            values = [_json_to_sql_value(slice_data.get(c)) for c in col_names]
            # safe_name is sanitized via regex; col_names come from internal slice data
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

    # Cast to Mapping for proper typing
    entry_mapping = cast("Mapping[str, object]", entry_raw)

    # Check if this is a session snapshot (has "slices" field)
    if "slices" in entry_mapping:
        # Session snapshot format - extract items from slices
        for slice_type, item_entry in _extract_slices_from_snapshot(entry_mapping):
            _insert_session_slice(conn, slice_type, item_entry, slices_by_type)
    else:
        # Direct slice format (has __type__ field)
        entry = cast("Mapping[str, JSONValue]", entry_mapping)
        type_val: Any = entry.get("__type__")
        slice_type = str(type_val) if type_val is not None else "unknown"
        _insert_session_slice(conn, slice_type, entry, slices_by_type)


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


class QueryDatabase(Closeable):
    """Builds and manages SQLite database from a debug bundle.

    This class implements the Closeable protocol for proper resource management.

    Note: For very large bundles (100MB+), all data is loaded into memory during
    build. Consider this limitation when working with large debug captures.

    The database is opened in read-only mode after building to prevent
    accidental modification of the cached data.

    Thread safety: Uses a lock to ensure only one thread accesses the
    connection at a time, making it safe for use with FastAPI's thread pool.
    """

    _bundle: DebugBundle
    _bundle_path: Path
    _db_path: Path
    _conn: sqlite3.Connection | None
    _built: bool
    _lock: threading.Lock

    def __init__(self, bundle: DebugBundle, db_path: Path) -> None:
        """Initialize query database.

        Args:
            bundle: Debug bundle to load.
            db_path: Path for SQLite database file.
        """
        super().__init__()
        self._bundle = bundle
        # Derive bundle path from db_path (remove .sqlite suffix)
        self._bundle_path = Path(str(db_path).removesuffix(".sqlite"))
        self._db_path = db_path
        self._conn = None
        self._built = False
        self._lock = threading.Lock()

    @property
    def bundle(self) -> DebugBundle:
        """The underlying debug bundle."""
        return self._bundle

    @property
    def bundle_path(self) -> Path:
        """Path to the bundle zip file."""
        return self._bundle_path

    @property
    def connection(self) -> sqlite3.Connection:
        """Get database connection, opening if needed.

        Opens in read-only mode if database is already built.
        """
        if self._conn is None:
            if self._built and self._db_path.exists():
                # Open in read-only mode after building
                uri = f"file:{self._db_path}?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            else:
                self._conn = sqlite3.connect(
                    str(self._db_path), check_same_thread=False
                )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def execute(
        self, query: str, params: Sequence[Any] | None = None
    ) -> list[sqlite3.Row]:
        """Execute a query with thread safety.

        Args:
            query: SQL query to execute.
            params: Optional query parameters.

        Returns:
            List of result rows.
        """
        with self._lock:
            conn = self.connection
            cursor = conn.execute(query, params or [])
            return cursor.fetchall()

    @override
    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def mark_built(self) -> None:
        """Mark database as built for read-only mode on next connection."""
        self._built = True

    def build(self) -> None:
        """Build SQLite database from bundle contents.

        Note: Loads all bundle data into memory. For bundles over 100MB,
        this may consume significant memory.
        """
        conn = self.connection

        # Store schema version for cache invalidation
        _ = conn.execute(
            "CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER PRIMARY KEY)"
        )
        _ = conn.execute(
            "INSERT INTO _schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
        )

        # Core tables
        self._build_manifest_table(conn)
        self._build_logs_table(conn)
        self._build_transcript_table(conn)
        self._build_tool_calls_table(conn)
        self._build_errors_table(conn)
        self._build_session_slices_table(conn)
        self._build_files_table(conn)
        self._build_config_table(conn)
        self._build_metrics_table(conn)
        self._build_run_context_table(conn)

        # Optional tables
        self._build_prompt_overrides_table(conn)
        self._build_eval_table(conn)
        self._build_environment_tables(conn)

        # Views for common query patterns
        self._build_views(conn)

        conn.commit()

        # Mark as built and reopen in read-only mode
        self._built = True
        self.close()  # Close so next access reopens in read-only mode

    def _build_manifest_table(self, conn: sqlite3.Connection) -> None:
        """Build manifest table from manifest.json."""
        manifest = self._bundle.manifest
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest (
                bundle_id TEXT,
                format_version TEXT,
                created_at TEXT,
                status TEXT,
                request_id TEXT,
                session_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                capture_mode TEXT,
                capture_trigger TEXT,
                prompt_ns TEXT,
                prompt_key TEXT,
                prompt_adapter TEXT
            )
        """)
        _ = conn.execute(
            """
            INSERT INTO manifest VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
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

    def _build_logs_table(self, conn: sqlite3.Connection) -> None:
        """Build logs table from logs/app.jsonl."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                rowid INTEGER PRIMARY KEY,
                timestamp TEXT,
                level TEXT,
                logger TEXT,
                event TEXT,
                message TEXT,
                context TEXT,
                seq INTEGER
            )
        """)

        logs_content = self._bundle.logs
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

            # Extract sequence number from events that include it in context
            seq: int | None = None
            ctx_raw = entry.get("context", {})
            if isinstance(ctx_raw, dict):
                ctx = cast("dict[str, Any]", ctx_raw)
                seq_val: Any = ctx.get("sequence_number")
                if isinstance(seq_val, int):
                    seq = seq_val

            _ = conn.execute(
                """
                INSERT INTO logs (timestamp, level, logger, event, message, context, seq)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
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

    def _build_transcript_table(self, conn: sqlite3.Connection) -> None:
        """Build transcript table from transcript entries.

        Sources (in priority order):
        1. ``transcript.jsonl`` artifact (new unified format)
        2. Fall back to scanning ``logs/app.jsonl`` for transcript events
        """
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript (
                rowid INTEGER PRIMARY KEY,
                timestamp TEXT,
                prompt_name TEXT,
                transcript_source TEXT,
                sequence_number INTEGER,
                entry_type TEXT,
                role TEXT,
                content TEXT,
                tool_name TEXT,
                tool_use_id TEXT,
                raw_json TEXT,
                parsed TEXT
            )
        """)

        inserted = self._insert_transcript_from_artifact(conn)
        if not inserted:
            self._insert_transcript_from_logs(conn)

    def _insert_transcript_from_artifact(self, conn: sqlite3.Connection) -> bool:
        """Try loading transcript entries from transcript.jsonl artifact.

        Returns True if any rows were inserted.
        """
        transcript_entries = self._bundle.transcript
        if not transcript_entries:
            return False

        count = 0
        for entry in transcript_entries:
            row = _extract_transcript_row(entry)
            if row is None:
                continue
            _ = conn.execute(
                _TRANSCRIPT_INSERT_SQL,
                row,
            )
            count += 1
        return count > 0

    def _insert_transcript_from_logs(self, conn: sqlite3.Connection) -> None:
        """Fall back to scanning logs/app.jsonl for transcript entries."""
        logs_content = self._bundle.logs
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

            _ = conn.execute(
                _TRANSCRIPT_INSERT_SQL,
                row,
            )

    def _build_tool_calls_table(self, conn: sqlite3.Connection) -> None:
        """Build tool_calls table derived from logs."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                rowid INTEGER PRIMARY KEY,
                timestamp TEXT,
                tool_name TEXT,
                params TEXT,
                result TEXT,
                success INTEGER,
                error_code TEXT,
                duration_ms REAL
            )
        """)

        logs_content = self._bundle.logs
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
                """
                INSERT INTO tool_calls
                    (timestamp, tool_name, params, result, success,
                     error_code, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                tool_data,
            )

    def _build_errors_table(self, conn: sqlite3.Connection) -> None:
        """Build errors table aggregating errors from multiple sources."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                rowid INTEGER PRIMARY KEY,
                source TEXT,
                error_type TEXT,
                message TEXT,
                traceback TEXT
            )
        """)

        self._insert_errors_from_error_json(conn)
        self._insert_errors_from_logs(conn)
        _insert_errors_from_tool_calls(conn)

    def _insert_errors_from_error_json(self, conn: sqlite3.Connection) -> None:
        """Insert errors from error.json if present."""
        error_data = self._bundle.error
        if not error_data or not isinstance(error_data, Mapping):
            return

        # Convert to dict for proper type access
        err = cast("dict[str, Any]", dict(error_data))
        tb_val: Any = err.get("traceback", [])
        tb_str: str
        if isinstance(tb_val, list):
            tb_list = cast("list[Any]", tb_val)
            tb_str = "".join(str(item) for item in tb_list)
        else:
            tb_str = str(tb_val)
        _ = conn.execute(
            """
            INSERT INTO errors (source, error_type, message, traceback)
            VALUES (?, ?, ?, ?)
        """,
            (
                "error.json",
                str(err.get("type", "")),
                str(err.get("message", "")),
                tb_str,
            ),
        )

    def _insert_errors_from_logs(self, conn: sqlite3.Connection) -> None:
        """Insert errors from log entries with level=ERROR."""
        logs_content = self._bundle.logs
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

    def _build_session_slices_table(self, conn: sqlite3.Connection) -> None:
        """Build session_slices table and dynamic slice tables."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS session_slices (
                rowid INTEGER PRIMARY KEY,
                slice_type TEXT,
                data TEXT
            )
        """)

        session_content = self._bundle.session_after
        if not session_content:
            return

        # Group slices by type for dynamic table creation
        slices_by_type: dict[str, list[Mapping[str, JSONValue]]] = {}

        for line in session_content.strip().split("\n"):
            if not line.strip():
                continue
            _process_session_line(conn, line, slices_by_type)

        # Create dynamic slice tables
        for slice_type, slices in slices_by_type.items():
            _create_dynamic_slice_table(conn, slice_type, slices)

    def _build_files_table(self, conn: sqlite3.Connection) -> None:
        """Build files table from filesystem/ directory."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                rowid INTEGER PRIMARY KEY,
                path TEXT,
                content TEXT,
                size_bytes INTEGER
            )
        """)

        # List files in bundle
        all_files = self._bundle.list_files()
        for file_path in all_files:
            if file_path.startswith("filesystem/"):
                try:
                    content = self._bundle.read_file(file_path)
                    # Store relative path within filesystem/
                    rel_path = file_path[len("filesystem/") :]
                    # Try to decode as text, fall back to hex for binary
                    try:
                        text_content = content.decode("utf-8")
                    except UnicodeDecodeError:
                        text_content = content.hex()

                    _ = conn.execute(
                        """
                        INSERT INTO files (path, content, size_bytes)
                        VALUES (?, ?, ?)
                    """,
                        (rel_path, text_content, len(content)),
                    )
                except BundleValidationError:
                    continue

    def _build_config_table(self, conn: sqlite3.Connection) -> None:
        """Build config table from config.json (flattened)."""
        config_data = self._bundle.config
        if not config_data or not isinstance(config_data, Mapping):
            _ = conn.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    rowid INTEGER PRIMARY KEY,
                    key TEXT,
                    value TEXT
                )
            """)
            return

        # Flatten the config
        flattened = _flatten_json(config_data)

        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                rowid INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT
            )
        """)

        for key, value in flattened.items():
            _ = conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?)",
                (key, str(value) if value is not None else None),
            )

    def _build_metrics_table(self, conn: sqlite3.Connection) -> None:
        """Build metrics table from metrics.json."""
        metrics_data = self._bundle.metrics

        # Flatten metrics for flexible schema
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                rowid INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT
            )
        """)

        if metrics_data and isinstance(metrics_data, Mapping):
            flattened = _flatten_json(metrics_data)
            for key, value in flattened.items():
                _ = conn.execute(
                    "INSERT INTO metrics (key, value) VALUES (?, ?)",
                    (key, str(value) if value is not None else None),
                )

    def _build_run_context_table(self, conn: sqlite3.Connection) -> None:
        """Build run_context table from run_context.json."""
        run_context = self._bundle.run_context

        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS run_context (
                rowid INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT
            )
        """)

        if run_context and isinstance(run_context, Mapping):
            flattened = _flatten_json(run_context)
            for key, value in flattened.items():
                _ = conn.execute(
                    "INSERT INTO run_context (key, value) VALUES (?, ?)",
                    (key, str(value) if value is not None else None),
                )

    def _build_prompt_overrides_table(self, conn: sqlite3.Connection) -> None:
        """Build prompt_overrides table if file exists."""
        overrides = self._bundle.prompt_overrides

        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_overrides (
                rowid INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT
            )
        """)

        if overrides and isinstance(overrides, Mapping):
            flattened = _flatten_json(overrides)
            for key, value in flattened.items():
                _ = conn.execute(
                    "INSERT INTO prompt_overrides (key, value) VALUES (?, ?)",
                    (key, str(value) if value is not None else None),
                )

    def _build_eval_table(self, conn: sqlite3.Connection) -> None:
        """Build eval table if file exists."""
        eval_data = self._bundle.eval

        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS eval (
                rowid INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT
            )
        """)

        if eval_data and isinstance(eval_data, Mapping):
            flattened = _flatten_json(eval_data)
            for key, value in flattened.items():
                _ = conn.execute(
                    "INSERT INTO eval (key, value) VALUES (?, ?)",
                    (key, str(value) if value is not None else None),
                )

    def _build_environment_tables(self, conn: sqlite3.Connection) -> None:
        """Build environment tables from environment/ directory."""
        env_data = self._bundle.environment
        if not env_data:
            # Create empty tables for consistency
            self._create_empty_environment_tables(conn)
            return

        # Build env_system table
        self._build_env_system_table(conn, env_data.get("system"))

        # Build env_python table
        self._build_env_python_table(conn, env_data.get("python"))

        # Build env_git table
        self._build_env_git_table(conn, env_data.get("git"))

        # Build env_container table
        self._build_env_container_table(conn, env_data.get("container"))

        # Build env_vars table (key-value pairs)
        self._build_env_vars_table(conn, env_data.get("env_vars"))

        # Build environment table (flat key-value for all data)
        self._build_environment_flat_table(conn, env_data)

    @staticmethod
    def _create_empty_environment_tables(conn: sqlite3.Connection) -> None:
        """Create empty environment tables when no environment data exists."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_system (
                os_name TEXT,
                os_release TEXT,
                kernel_version TEXT,
                architecture TEXT,
                processor TEXT,
                cpu_count INTEGER,
                memory_total_bytes INTEGER,
                hostname TEXT
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_python (
                version TEXT,
                version_info TEXT,
                implementation TEXT,
                executable TEXT,
                prefix TEXT,
                base_prefix TEXT,
                is_virtualenv INTEGER
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_git (
                repo_root TEXT,
                commit_sha TEXT,
                commit_short TEXT,
                branch TEXT,
                is_dirty INTEGER,
                remotes TEXT,
                tags TEXT
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_container (
                runtime TEXT,
                container_id TEXT,
                image TEXT,
                image_digest TEXT,
                cgroup_path TEXT,
                is_containerized INTEGER
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_vars (
                rowid INTEGER PRIMARY KEY,
                name TEXT,
                value TEXT
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS environment (
                rowid INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT
            )
        """)

    @staticmethod
    def _build_env_system_table(
        conn: sqlite3.Connection, data: JSONValue | None
    ) -> None:
        """Build env_system table from system.json."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_system (
                os_name TEXT,
                os_release TEXT,
                kernel_version TEXT,
                architecture TEXT,
                processor TEXT,
                cpu_count INTEGER,
                memory_total_bytes INTEGER,
                hostname TEXT
            )
        """)
        if not data or not isinstance(data, Mapping):
            return

        system = cast("Mapping[str, JSONValue]", data)
        _ = conn.execute(
            """
            INSERT INTO env_system (
                os_name, os_release, kernel_version, architecture,
                processor, cpu_count, memory_total_bytes, hostname
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                system.get("os_name"),
                system.get("os_release"),
                system.get("kernel_version"),
                system.get("architecture"),
                system.get("processor"),
                system.get("cpu_count"),
                system.get("memory_total_bytes"),
                system.get("hostname"),
            ),
        )

    @staticmethod
    def _build_env_python_table(
        conn: sqlite3.Connection, data: JSONValue | None
    ) -> None:
        """Build env_python table from python.json."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_python (
                version TEXT,
                version_info TEXT,
                implementation TEXT,
                executable TEXT,
                prefix TEXT,
                base_prefix TEXT,
                is_virtualenv INTEGER
            )
        """)
        if not data or not isinstance(data, Mapping):
            return

        python = cast("Mapping[str, JSONValue]", data)
        version_info = python.get("version_info")
        version_info_str = (
            json.dumps(version_info) if version_info is not None else None
        )
        is_venv = python.get("is_virtualenv")
        _ = conn.execute(
            """
            INSERT INTO env_python (
                version, version_info, implementation, executable,
                prefix, base_prefix, is_virtualenv
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                python.get("version"),
                version_info_str,
                python.get("implementation"),
                python.get("executable"),
                python.get("prefix"),
                python.get("base_prefix"),
                1 if is_venv else 0 if is_venv is not None else None,
            ),
        )

    @staticmethod
    def _build_env_git_table(conn: sqlite3.Connection, data: JSONValue | None) -> None:
        """Build env_git table from git.json."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_git (
                repo_root TEXT,
                commit_sha TEXT,
                commit_short TEXT,
                branch TEXT,
                is_dirty INTEGER,
                remotes TEXT,
                tags TEXT
            )
        """)
        if not data or not isinstance(data, Mapping):
            return

        git = cast("Mapping[str, JSONValue]", data)
        remotes = git.get("remotes")
        remotes_str = json.dumps(remotes) if remotes is not None else None
        tags = git.get("tags")
        tags_str = json.dumps(tags) if tags is not None else None
        is_dirty = git.get("is_dirty")
        _ = conn.execute(
            """
            INSERT INTO env_git (
                repo_root, commit_sha, commit_short, branch,
                is_dirty, remotes, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                git.get("repo_root"),
                git.get("commit_sha"),
                git.get("commit_short"),
                git.get("branch"),
                1 if is_dirty else 0 if is_dirty is not None else None,
                remotes_str,
                tags_str,
            ),
        )

    @staticmethod
    def _build_env_container_table(
        conn: sqlite3.Connection, data: JSONValue | None
    ) -> None:
        """Build env_container table from container.json."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_container (
                runtime TEXT,
                container_id TEXT,
                image TEXT,
                image_digest TEXT,
                cgroup_path TEXT,
                is_containerized INTEGER
            )
        """)
        if not data or not isinstance(data, Mapping):
            return

        container = cast("Mapping[str, JSONValue]", data)
        is_containerized = container.get("is_containerized")
        _ = conn.execute(
            """
            INSERT INTO env_container (
                runtime, container_id, image, image_digest,
                cgroup_path, is_containerized
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                container.get("runtime"),
                container.get("container_id"),
                container.get("image"),
                container.get("image_digest"),
                container.get("cgroup_path"),
                1 if is_containerized else 0 if is_containerized is not None else None,
            ),
        )

    @staticmethod
    def _build_env_vars_table(conn: sqlite3.Connection, data: JSONValue | None) -> None:
        """Build env_vars table from env_vars.json."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_vars (
                rowid INTEGER PRIMARY KEY,
                name TEXT,
                value TEXT
            )
        """)
        if not data or not isinstance(data, Mapping):
            return

        env_vars = cast("Mapping[str, JSONValue]", data)
        for name, value in env_vars.items():
            _ = conn.execute(
                "INSERT INTO env_vars (name, value) VALUES (?, ?)",
                (name, str(value) if value is not None else None),
            )

    @staticmethod
    def _build_environment_flat_table(
        conn: sqlite3.Connection, env_data: Mapping[str, JSONValue | str | None]
    ) -> None:
        """Build flat environment table with all data as key-value pairs."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS environment (
                rowid INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT
            )
        """)

        # Flatten system, python, git, container data
        for category in ("system", "python", "git", "container"):
            category_data = env_data.get(category)
            if category_data and isinstance(category_data, Mapping):
                flattened = _flatten_json(category_data, prefix=category)
                for key, value in flattened.items():
                    _ = conn.execute(
                        "INSERT INTO environment (key, value) VALUES (?, ?)",
                        (key, str(value) if value is not None else None),
                    )

        # Add packages as a single entry
        packages = env_data.get("packages")
        if packages:
            _ = conn.execute(
                "INSERT INTO environment (key, value) VALUES (?, ?)",
                ("packages", str(packages)),
            )

        # Add command as a single entry
        command = env_data.get("command")
        if command:
            _ = conn.execute(
                "INSERT INTO environment (key, value) VALUES (?, ?)",
                ("command", str(command)),
            )

        # Add git_diff as a single entry (if present)
        git_diff = env_data.get("git_diff")
        if git_diff:
            _ = conn.execute(
                "INSERT INTO environment (key, value) VALUES (?, ?)",
                ("git_diff", str(git_diff)),
            )

    @staticmethod
    def _build_views(conn: sqlite3.Connection) -> None:
        """Create SQL views for common query patterns."""
        # Tool execution timeline
        _ = conn.execute("""
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
        """)

        # Native tool calls from transcripts (and legacy log_aggregator events)
        _ = conn.execute("""
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
        """)

        # Transcript entries derived from TranscriptCollector structured logs
        _ = conn.execute("""
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
        """)

        # Transcript conversation flow view
        _ = conn.execute("""
            CREATE VIEW IF NOT EXISTS transcript_flow AS
            SELECT
                sequence_number,
                transcript_source,
                entry_type,
                role,
                CASE
                    WHEN entry_type IN ('user', 'assistant') THEN
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
        """)

        # Transcript tool usage analysis view
        _ = conn.execute("""
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
            WHERE t1.entry_type = 'assistant'
                AND t1.tool_name IS NOT NULL
                AND t1.tool_name != ''
            ORDER BY t1.sequence_number
        """)

        # Transcript thinking blocks view
        _ = conn.execute("""
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
        """)

        # Transcript agents hierarchy and metrics view
        _ = conn.execute("""
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
                COUNT(CASE WHEN entry_type = 'user' THEN 1 END) as user_messages,
                COUNT(CASE WHEN entry_type = 'assistant' THEN 1 END) as assistant_messages,
                COUNT(CASE WHEN entry_type = 'thinking' THEN 1 END) as thinking_blocks,
                COUNT(DISTINCT CASE WHEN tool_name IS NOT NULL AND tool_name != '' THEN tool_name END) as unique_tools,
                COUNT(CASE WHEN tool_name IS NOT NULL AND tool_name != '' THEN 1 END) as total_tool_calls
            FROM transcript
            GROUP BY transcript_source
            ORDER BY MIN(sequence_number)
        """)

        # Error summary with truncated traceback
        _ = conn.execute("""
            CREATE VIEW IF NOT EXISTS error_summary AS
            SELECT
                source,
                error_type,
                message,
                SUBSTR(traceback, 1, 200) as traceback_head
            FROM errors
        """)

    def get_schema(self) -> SchemaOutput:
        """Get schema information for all tables and views."""
        manifest = self._bundle.manifest
        conn = self.connection

        tables: list[TableInfo] = []

        # Get all tables and views
        cursor = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type DESC, name"
        )
        for table_name, obj_type in cursor.fetchall():
            # table_name comes from sqlite_master, which is internal SQLite metadata
            # Get column info
            col_cursor = conn.execute(f"PRAGMA table_info({table_name})")  # nosec B608
            columns = tuple(
                ColumnInfo(
                    name=row[1],
                    type=row[2],
                    description="",
                )
                for row in col_cursor.fetchall()
            )

            # Get row count
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")  # nosec B608
            row_count = count_cursor.fetchone()[0]

            tables.append(
                TableInfo(
                    name=table_name,
                    description=_get_table_description(
                        table_name, is_view=obj_type == "view"
                    ),
                    row_count=row_count,
                    columns=columns,
                )
            )

        hints = SchemaHints(
            json_extraction=(
                "json_extract(context, '$.tool_name')",
                "json_extract(params, '$.command')",
                "json_extract(context, '$.content')",
                "json_extract(parsed, '$.message.content')",
                "json_extract(parsed, '$.tool_use_id')",
            ),
            common_queries={
                "native_tools": "SELECT * FROM native_tool_calls ORDER BY timestamp",
                "transcript": (
                    "SELECT transcript_source, sequence_number, entry_type, role, content "
                    "FROM transcript ORDER BY rowid"
                ),
                "conversation_flow": (
                    "SELECT * FROM transcript_flow "
                    "WHERE transcript_source = 'main' "
                    "ORDER BY sequence_number DESC LIMIT 50"
                ),
                "tool_usage_summary": (
                    "SELECT tool_name, COUNT(*) as usage_count "
                    "FROM transcript_tools "
                    "GROUP BY tool_name ORDER BY usage_count DESC"
                ),
                "thinking_blocks": (
                    "SELECT * FROM transcript_thinking WHERE thinking_length > 1000"
                ),
                "subagent_activity": (
                    "SELECT * FROM transcript_agents WHERE transcript_source != 'main'"
                ),
                "tool_timeline": (
                    "SELECT * FROM tool_timeline WHERE duration_ms > 1000"
                ),
                "error_context": (
                    "SELECT timestamp, message, context FROM logs WHERE level = 'ERROR'"
                ),
                "system_info": "SELECT * FROM env_system",
                "python_info": "SELECT * FROM env_python",
                "git_info": "SELECT * FROM env_git",
                "env_vars": "SELECT name, value FROM env_vars ORDER BY name",
            },
        )

        return SchemaOutput(
            bundle_id=manifest.bundle_id,
            status=manifest.request.status,
            created_at=manifest.created_at,
            tables=tuple(tables),
            hints=hints,
        )

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        """Execute SQL query and return results as list of dicts."""
        conn = self.connection
        try:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description or []]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise QueryError(f"SQL error: {e}") from e


@pure
def _get_table_description(table_name: str, *, is_view: bool = False) -> str:
    """Get description for a table or view by name."""
    descriptions = {
        "manifest": "Bundle metadata",
        "logs": "Log entries (seq extracted from context.sequence_number when present)",
        "transcript": "Transcript entries extracted from logs",
        "tool_calls": "Tool invocations",
        "errors": "Aggregated errors",
        "session_slices": "Session state items",
        "files": "Workspace files",
        "config": "Flattened configuration",
        "metrics": "Token usage and timing",
        "run_context": "Execution IDs",
        "prompt_overrides": "Visibility overrides",
        "eval": "Eval metadata",
        # Environment tables
        "environment": "Flattened environment data (key-value)",
        "env_system": "System/OS info (architecture, CPU, memory)",
        "env_python": "Python runtime (version, venv, executable)",
        "env_git": "Git repository state (commit, branch, remotes)",
        "env_container": "Container runtime info (Docker/K8s)",
        "env_vars": "Filtered environment variables",
        # Views
        "tool_timeline": "View: Tool calls ordered by timestamp",
        "native_tool_calls": "View: Native tool calls from transcripts or legacy logs",
        "transcript_entries": "View: Transcript entries (alias of transcript table)",
        "transcript_flow": "View: Conversation flow with message previews",
        "transcript_tools": "View: Tool usage analysis with paired calls and results",
        "transcript_thinking": "View: Thinking blocks with preview and length",
        "transcript_agents": "View: Agent hierarchy and activity metrics",
        "error_summary": "View: Errors with truncated traceback",
    }
    if table_name.startswith("slice_"):
        return f"Session slice: {table_name[6:]}"
    desc = descriptions.get(table_name, "")
    if is_view and not desc.startswith("View:"):
        desc = f"View: {desc}" if desc else "View"
    return desc


def _is_cache_valid(bundle_path: Path, cache_path: Path) -> bool:
    """Check if cache is still valid based on mtime and schema version."""
    if not cache_path.exists():
        return False
    if cache_path.stat().st_mtime < bundle_path.stat().st_mtime:
        return False
    # Check schema version
    try:
        conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
        try:
            cursor = conn.execute("SELECT version FROM _schema_version LIMIT 1")
            row = cursor.fetchone()
            if row is None or row[0] != _SCHEMA_VERSION:
                return False
        except sqlite3.OperationalError:
            # Table doesn't exist (old cache) or other error
            return False
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return True


def resolve_bundle_path(path: Path) -> Path:
    """Resolve a path to a bundle file.

    If path is a directory, returns the most recently modified .zip file.
    If path is a file, returns it directly.

    Raises:
        QueryError: If path is a directory with no bundles.
    """
    if path.is_dir():
        candidates = sorted(
            (p for p in path.glob("*.zip") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            msg = f"No bundles found in directory: {path}"
            raise QueryError(msg)
        return candidates[0]
    return path


def iter_bundle_files(root: Path) -> list[Path]:
    """Find all bundle zip files in a directory."""
    return [p for p in root.glob("*.zip") if p.is_file()]


def open_query_database(bundle_path: Path) -> QueryDatabase:
    """Open or create a query database for a bundle.

    Implements caching: creates bundle.zip.sqlite alongside the bundle,
    reusing if valid (cache mtime >= bundle mtime).

    Args:
        bundle_path: Path to the debug bundle zip file or directory.
            If a directory, uses the most recently modified .zip file.

    Returns:
        QueryDatabase instance ready for queries.

    Raises:
        QueryError: If bundle cannot be loaded.
    """
    resolved_path = resolve_bundle_path(bundle_path)
    cache_path = resolved_path.with_suffix(resolved_path.suffix + ".sqlite")

    try:
        bundle = DebugBundle.load(resolved_path)
    except BundleValidationError as e:
        raise QueryError(f"Failed to load bundle: {e}") from e

    db = QueryDatabase(bundle, cache_path)

    if not _is_cache_valid(resolved_path, cache_path):
        # Remove stale cache if exists
        if cache_path.exists():
            cache_path.unlink()
        db.build()
    else:
        # Cache is valid, mark as built for read-only mode
        db.mark_built()

    return db


@pure
def format_as_table(rows: Sequence[Mapping[str, Any]], *, truncate: bool = True) -> str:
    """Format query results as ASCII table.

    Args:
        rows: Query results to format.
        truncate: If True, truncate long values to _MAX_COLUMN_WIDTH.
    """
    if not rows:
        return "(no results)"

    max_width = _MAX_COLUMN_WIDTH if truncate else None

    # Get columns from first row
    columns = list(rows[0].keys())

    # Calculate column widths
    widths: dict[str, int] = {}
    for col in columns:
        col_max = len(col)
        for row in rows:
            val = str(row.get(col, ""))
            if max_width is not None and len(val) > max_width:
                val = val[: max_width - 3] + "..."
            col_max = max(col_max, len(val))
        widths[col] = col_max if max_width is None else min(col_max, max_width)

    # Build header
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)

    # Build rows
    lines: list[str] = [header, separator]
    for row in rows:
        cells: list[str] = []
        for col in columns:
            val = str(row.get(col, ""))
            if max_width is not None and len(val) > max_width:
                val = val[: max_width - 3] + "..."
            cells.append(val.ljust(widths[col]))
        lines.append(" | ".join(cells))

    return "\n".join(lines)


@pure
def format_as_json(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format query results as JSON."""
    # Convert to list of plain dicts for JSON serialization
    result = [dict(row) for row in rows]
    return json.dumps(result, indent=2)


def export_jsonl(bundle: DebugBundle, source: str) -> str | None:
    """Export raw JSONL content from bundle.

    Args:
        bundle: Debug bundle to export from.
        source: Either "logs" for logs/app.jsonl or "session" for session/after.jsonl.

    Returns:
        Raw JSONL content, or None if not present.
    """
    if source == "logs":
        return bundle.logs
    if source == "session":
        return bundle.session_after
    return None


__all__ = [
    "ColumnInfo",
    "QueryDatabase",
    "QueryError",
    "SchemaHints",
    "SchemaOutput",
    "TableInfo",
    "export_jsonl",
    "format_as_json",
    "format_as_table",
    "iter_bundle_files",
    "open_query_database",
    "resolve_bundle_path",
]
