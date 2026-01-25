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
_SCHEMA_VERSION = 4  # v4: tool_calls from ToolInvoked session slices


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


def _is_tool_invoked_slice(slice_entry: object) -> bool:
    """Check if a slice entry is a ToolInvoked slice."""
    if not isinstance(slice_entry, dict):
        return False
    entry = cast("dict[str, Any]", slice_entry)
    slice_type: object = entry.get("slice_type", "")
    return isinstance(slice_type, str) and "ToolInvoked" in slice_type


def _extract_items_from_slice(slice_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract items from a slice entry."""
    items: Any = slice_entry.get("items", [])
    if not isinstance(items, list):
        return []
    items_list = cast("list[object]", items)
    return [
        cast("dict[str, Any]", item) for item in items_list if isinstance(item, dict)
    ]


def _extract_tool_invoked_items(session_content: str) -> list[dict[str, Any]]:
    """Extract ToolInvoked items from session snapshot.

    Parses the session/after.jsonl snapshot and returns all ToolInvoked
    slice items as dicts.
    """
    result: list[dict[str, Any]] = []

    for line in session_content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data: Any = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, dict):
            continue

        data_dict = cast("dict[str, Any]", data)
        slices_raw: Any = data_dict.get("slices", [])
        if not isinstance(slices_raw, list):
            continue

        slices = cast("list[object]", slices_raw)
        for slice_entry in slices:
            if _is_tool_invoked_slice(slice_entry):
                entry = cast("dict[str, Any]", slice_entry)
                result.extend(_extract_items_from_slice(entry))

    return result


def _check_sdk_error_indicators(result_dict: dict[str, Any]) -> str | None:
    """Check for error indicators in native SDK tool responses.

    Returns error message if found, None otherwise.
    """
    # Check isError/is_error flag (MCP content format)
    if result_dict.get("isError") or result_dict.get("is_error"):
        return "Tool returned error"

    # Check stderr (Bash tool format)
    stderr: object = result_dict.get("stderr")
    if stderr and isinstance(stderr, str) and stderr.strip():
        return stderr.strip()[:200]

    # Check content for error text (Claude Agent SDK format)
    content: object = result_dict.get("content")
    if isinstance(content, list) and content:
        content_list = cast("list[object]", content)
        first_item: object = content_list[0]
        if isinstance(first_item, dict):
            first_dict = cast("dict[str, Any]", first_item)
            text: object = first_dict.get("text", "")
            if isinstance(text, str):
                text_lower = text.lower()
                if text_lower.startswith("error:") or text_lower.startswith("error -"):
                    return text[:200]

    return None


def _extract_result_fields(
    result_raw: object,
) -> tuple[object, object, object]:
    """Extract value, success, message from a result object.

    Handles both MCP/BridgedTool results (ToolResult with value/success/message)
    and native SDK tool results (raw dict with stdout/stderr or isError).
    """
    if not isinstance(result_raw, dict):
        return (result_raw, True, "")

    result_dict = cast("dict[str, Any]", result_raw)

    # MCP/BridgedTool format: has explicit success field
    if "success" in result_dict:
        return (
            result_dict.get("value", result_raw),
            result_dict.get("success", True),
            result_dict.get("message", ""),
        )

    # Native SDK format: check for error indicators
    error_msg = _check_sdk_error_indicators(result_dict)
    if error_msg:
        return (result_dict, False, error_msg)

    return (result_dict, True, "")


def _tool_invoked_to_row(
    item: dict[str, Any],
) -> tuple[str, str, str, str, int, str, str | None, float | None]:
    """Convert a ToolInvoked item to a tool_calls row.

    Returns (timestamp, tool_name, params, result, success, error_msg, call_id, duration_ms).
    """
    created_at = item.get("created_at", "")
    name = item.get("name", "")
    params = item.get("params", {})
    call_id = item.get("call_id")
    duration_ms = item.get("duration_ms")

    result_value, result_success, result_message = _extract_result_fields(
        item.get("result", {})
    )

    return (
        str(created_at) if created_at else "",
        str(name) if name else "",
        json.dumps(params) if params else "{}",
        json.dumps(result_value),
        1 if result_success else 0,
        str(result_message) if not result_success and result_message else "",
        str(call_id) if call_id else None,
        float(duration_ms) if duration_ms is not None else None,
    )


def _insert_tool_invoked_items(
    conn: sqlite3.Connection,
    session_content: str,
) -> None:
    """Insert tool calls from ToolInvoked session slices."""
    items = _extract_tool_invoked_items(session_content)
    seen_ids: set[str] = set()

    for item in items:
        row = _tool_invoked_to_row(item)
        (
            timestamp,
            tool_name,
            params,
            result,
            success,
            error_msg,
            call_id,
            duration_ms,
        ) = row

        # Skip if no tool name
        if not tool_name:
            continue

        # Deduplicate by call_id if present
        if call_id:
            if call_id in seen_ids:
                continue
            seen_ids.add(call_id)

        # Determine source from adapter field
        # native = SDK-native tools (Bash, Read, etc.)
        # custom = weakincentives tools bridged via MCP
        adapter: Any = item.get("adapter", "")
        source = "native" if adapter == "claude_agent_sdk" else "custom"

        _ = conn.execute(
            """
            INSERT INTO tool_calls
                (timestamp, tool_name, params, result, success,
                 error_code, duration_ms, source, seq, tool_use_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
        """,
            (
                timestamp,
                tool_name,
                params,
                result,
                success,
                error_msg,
                duration_ms,
                source,
                call_id,
            ),
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
                entry: dict[str, Any] = json.loads(line)
                # Extract sequence number from log_aggregator events
                seq: int | None = None
                ctx_raw = entry.get("context", {})
                if (
                    isinstance(ctx_raw, dict)
                    and entry.get("event") == "log_aggregator.log_line"
                ):
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
            except json.JSONDecodeError:
                continue

    def _build_tool_calls_table(self, conn: sqlite3.Connection) -> None:
        """Build tool_calls table from ToolInvoked session slices.

        ToolInvoked events are the authoritative source for tool execution data.
        They are dispatched by adapters after each tool call completes and
        captured in the session snapshot.
        """
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                rowid INTEGER PRIMARY KEY,
                timestamp TEXT,
                tool_name TEXT,
                params TEXT,
                result TEXT,
                success INTEGER,
                error_code TEXT,
                duration_ms REAL,
                source TEXT,
                seq INTEGER,
                tool_use_id TEXT
            )
        """)

        session_content = self._bundle.session_after
        if not session_content:
            return

        _insert_tool_invoked_items(conn, session_content)

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
                entry: dict[str, Any] = json.loads(line)
                if entry.get("level") == "ERROR":
                    _insert_error_from_log(conn, entry)
            except json.JSONDecodeError:
                continue

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

    @staticmethod
    def _build_views(conn: sqlite3.Connection) -> None:
        """Create SQL views for common query patterns."""
        # Tool execution timeline - unified view of all tools (native + custom)
        # Ordered by timestamp to properly interleave native and custom tools
        _ = conn.execute("""
            CREATE VIEW IF NOT EXISTS tool_timeline AS
            SELECT
                rowid,
                timestamp,
                tool_name,
                json_extract(params, '$.command') as command,
                success,
                duration_ms,
                source,
                seq,
                CASE WHEN success = 0 THEN error_code ELSE NULL END as error
            FROM tool_calls
            ORDER BY timestamp, rowid
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
                "json_extract(params, '$.path')",
            ),
            common_queries={
                "all_tools": (
                    "SELECT tool_name, source, success "
                    "FROM tool_calls ORDER BY timestamp"
                ),
                "tool_timeline": "SELECT * FROM tool_timeline",
                "failed_tools": (
                    "SELECT tool_name, error_code FROM tool_calls WHERE success = 0"
                ),
                "error_context": (
                    "SELECT timestamp, message, context FROM logs WHERE level = 'ERROR'"
                ),
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
        "logs": "Log entries (seq column for log_aggregator events)",
        "tool_calls": "Tool invocations from ToolInvoked session events",
        "errors": "Aggregated errors",
        "session_slices": "Session state items",
        "files": "Workspace files",
        "config": "Flattened configuration",
        "metrics": "Token usage and timing",
        "run_context": "Execution IDs",
        "prompt_overrides": "Visibility overrides",
        "eval": "Eval metadata",
        # Views
        "tool_timeline": "View: All tools ordered by execution sequence",
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
