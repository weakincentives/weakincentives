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

# pyright: reportPrivateUsage=false
# Re-exports private helpers from sub-modules for backward compatibility.

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import Any, cast, override

from ..dataclasses import FrozenDataclass
from ..debug.bundle import BundleValidationError, DebugBundle
from ..errors import WinkError
from ..resources.protocols import Closeable
from ..types import JSONValue
from ._query_helpers import (
    _MAX_COLUMN_WIDTH,
    _SCHEMA_VERSION,
    _flatten_json as _flatten_json,
    _get_table_description as _get_table_description,
    _infer_sqlite_type as _infer_sqlite_type,
    _is_cache_valid as _is_cache_valid,
    _json_to_sql_value as _json_to_sql_value,
    _normalize_slice_type as _normalize_slice_type,
    _safe_json_dumps as _safe_json_dumps,
)
from ._query_tables import (
    _VIEW_DEFINITIONS,
    _create_dynamic_slice_table as _create_dynamic_slice_table,
    _extract_slices_from_snapshot as _extract_slices_from_snapshot,
    _extract_tool_call_from_entry as _extract_tool_call_from_entry,
    _insert_error_from_log,
    _insert_errors_from_tool_calls,
    _is_tool_event as _is_tool_event,
    _process_session_line as _process_session_line,
)
from ._query_transcript import (
    _TRANSCRIPT_INSERT_SQL,
    _apply_bridged_tool_details as _apply_bridged_tool_details,
    _apply_notification_item_details as _apply_notification_item_details,
    _apply_split_block_details as _apply_split_block_details,
    _apply_tool_result_details as _apply_tool_result_details,
    _apply_transcript_content_fallbacks as _apply_transcript_content_fallbacks,
    _extract_tool_use_from_content as _extract_tool_use_from_content,
    _extract_transcript_details as _extract_transcript_details,
    _extract_transcript_message_details as _extract_transcript_message_details,
    _extract_transcript_parsed_obj as _extract_transcript_parsed_obj,
    _extract_transcript_row as _extract_transcript_row,
    _stringify_transcript_content as _stringify_transcript_content,
    _stringify_transcript_mapping as _stringify_transcript_mapping,
    _stringify_transcript_tool_use as _stringify_transcript_tool_use,
)


class QueryError(WinkError, RuntimeError):
    """Raised when a query operation fails."""


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
        """Initialize query database."""
        super().__init__()
        self._bundle = bundle
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
        """Get database connection, opening if needed."""
        if self._conn is None:
            if self._built and self._db_path.exists():
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
        """Execute a query with thread safety."""
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
        """Build SQLite database from bundle contents."""
        conn = self.connection
        _ = conn.execute(
            "CREATE TABLE IF NOT EXISTS _schema_version (version INTEGER PRIMARY KEY)"
        )
        _ = conn.execute(
            "INSERT INTO _schema_version (version) VALUES (?)", (_SCHEMA_VERSION,)
        )
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
        self._build_prompt_overrides_table(conn)
        self._build_eval_table(conn)
        self._build_environment_tables(conn)
        self._build_views(conn)
        conn.commit()
        self._built = True
        self.close()

    def _build_manifest_table(self, conn: sqlite3.Connection) -> None:
        """Build manifest table from manifest.json."""
        manifest = self._bundle.manifest
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

    def _build_logs_table(self, conn: sqlite3.Connection) -> None:
        """Build logs table from logs/app.jsonl."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                rowid INTEGER PRIMARY KEY, timestamp TEXT, level TEXT,
                logger TEXT, event TEXT, message TEXT, context TEXT,
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

    def _build_transcript_table(self, conn: sqlite3.Connection) -> None:
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
        inserted = self._insert_transcript_from_artifact(conn)
        if not inserted:
            self._insert_transcript_from_logs(conn)

    def _insert_transcript_from_artifact(self, conn: sqlite3.Connection) -> bool:
        """Try loading transcript entries from transcript.jsonl artifact."""
        transcript_entries = self._bundle.transcript
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
            _ = conn.execute(_TRANSCRIPT_INSERT_SQL, row)

    def _build_tool_calls_table(self, conn: sqlite3.Connection) -> None:
        """Build tool_calls table derived from logs."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                rowid INTEGER PRIMARY KEY, timestamp TEXT,
                tool_name TEXT, params TEXT, result TEXT,
                success INTEGER, error_code TEXT, duration_ms REAL
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
                """INSERT INTO tool_calls
                    (timestamp, tool_name, params, result, success,
                     error_code, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                tool_data,
            )

    def _build_errors_table(self, conn: sqlite3.Connection) -> None:
        """Build errors table aggregating errors from multiple sources."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                rowid INTEGER PRIMARY KEY, source TEXT,
                error_type TEXT, message TEXT, traceback TEXT
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
                rowid INTEGER PRIMARY KEY, slice_type TEXT, data TEXT
            )
        """)
        session_content = self._bundle.session_after
        if not session_content:
            return
        slices_by_type: dict[str, list[Mapping[str, JSONValue]]] = {}
        for line in session_content.strip().split("\n"):
            if not line.strip():
                continue
            _process_session_line(conn, line, slices_by_type)
        for slice_type, slices in slices_by_type.items():
            _create_dynamic_slice_table(conn, slice_type, slices)

    def _build_files_table(self, conn: sqlite3.Connection) -> None:
        """Build files table from filesystem/ directory."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                rowid INTEGER PRIMARY KEY, path TEXT,
                content TEXT, size_bytes INTEGER
            )
        """)
        all_files = self._bundle.list_files()
        for file_path in all_files:
            if file_path.startswith("filesystem/"):
                try:
                    content = self._bundle.read_file(file_path)
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

    def _build_config_table(self, conn: sqlite3.Connection) -> None:
        """Build config table from config.json (flattened)."""
        config_data = self._bundle.config
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

    def _build_metrics_table(self, conn: sqlite3.Connection) -> None:
        """Build metrics table from metrics.json."""
        metrics_data = self._bundle.metrics
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

    def _build_run_context_table(self, conn: sqlite3.Connection) -> None:
        """Build run_context table from run_context.json."""
        run_context = self._bundle.run_context
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

    def _build_prompt_overrides_table(self, conn: sqlite3.Connection) -> None:
        """Build prompt_overrides table if file exists."""
        overrides = self._bundle.prompt_overrides
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

    def _build_eval_table(self, conn: sqlite3.Connection) -> None:
        """Build eval table if file exists."""
        eval_data = self._bundle.eval
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

    def _build_environment_tables(self, conn: sqlite3.Connection) -> None:
        """Build environment tables from environment/ directory."""
        env_data = self._bundle.environment
        if not env_data:
            self._create_empty_environment_tables(conn)
            return
        self._build_env_system_table(conn, env_data.get("system"))
        self._build_env_python_table(conn, env_data.get("python"))
        self._build_env_git_table(conn, env_data.get("git"))
        self._build_env_container_table(conn, env_data.get("container"))
        self._build_env_vars_table(conn, env_data.get("env_vars"))
        self._build_environment_flat_table(conn, env_data)

    @staticmethod
    def _create_empty_environment_tables(conn: sqlite3.Connection) -> None:
        """Create empty environment tables when no environment data exists."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_system (
                os_name TEXT, os_release TEXT, kernel_version TEXT,
                architecture TEXT, processor TEXT, cpu_count INTEGER,
                memory_total_bytes INTEGER, hostname TEXT
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_python (
                version TEXT, version_info TEXT, implementation TEXT,
                executable TEXT, prefix TEXT, base_prefix TEXT,
                is_virtualenv INTEGER
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_git (
                repo_root TEXT, commit_sha TEXT, commit_short TEXT,
                branch TEXT, is_dirty INTEGER, remotes TEXT, tags TEXT
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_container (
                runtime TEXT, container_id TEXT, image TEXT,
                image_digest TEXT, cgroup_path TEXT, is_containerized INTEGER
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_vars (
                rowid INTEGER PRIMARY KEY, name TEXT, value TEXT
            )
        """)
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS environment (
                rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
            )
        """)

    @staticmethod
    def _build_env_system_table(
        conn: sqlite3.Connection, data: JSONValue | None
    ) -> None:
        """Build env_system table from system.json."""
        _ = conn.execute("""
            CREATE TABLE IF NOT EXISTS env_system (
                os_name TEXT, os_release TEXT, kernel_version TEXT,
                architecture TEXT, processor TEXT, cpu_count INTEGER,
                memory_total_bytes INTEGER, hostname TEXT
            )
        """)
        if not data or not isinstance(data, Mapping):
            return
        system = cast("Mapping[str, JSONValue]", data)
        _ = conn.execute(
            """INSERT INTO env_system (
                os_name, os_release, kernel_version, architecture,
                processor, cpu_count, memory_total_bytes, hostname
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
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
                version TEXT, version_info TEXT, implementation TEXT,
                executable TEXT, prefix TEXT, base_prefix TEXT,
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
            """INSERT INTO env_python (
                version, version_info, implementation, executable,
                prefix, base_prefix, is_virtualenv
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
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
                repo_root TEXT, commit_sha TEXT, commit_short TEXT,
                branch TEXT, is_dirty INTEGER, remotes TEXT, tags TEXT
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
            """INSERT INTO env_git (
                repo_root, commit_sha, commit_short, branch,
                is_dirty, remotes, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
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
                runtime TEXT, container_id TEXT, image TEXT,
                image_digest TEXT, cgroup_path TEXT, is_containerized INTEGER
            )
        """)
        if not data or not isinstance(data, Mapping):
            return
        container = cast("Mapping[str, JSONValue]", data)
        is_containerized = container.get("is_containerized")
        _ = conn.execute(
            """INSERT INTO env_container (
                runtime, container_id, image, image_digest,
                cgroup_path, is_containerized
            ) VALUES (?, ?, ?, ?, ?, ?)""",
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
                rowid INTEGER PRIMARY KEY, name TEXT, value TEXT
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
                rowid INTEGER PRIMARY KEY, key TEXT, value TEXT
            )
        """)
        for category in ("system", "python", "git", "container"):
            category_data = env_data.get(category)
            if category_data and isinstance(category_data, Mapping):
                flattened = _flatten_json(category_data, prefix=category)
                for key, value in flattened.items():
                    _ = conn.execute(
                        "INSERT INTO environment (key, value) VALUES (?, ?)",
                        (key, str(value) if value is not None else None),
                    )
        packages = env_data.get("packages")
        if packages:
            _ = conn.execute(
                "INSERT INTO environment (key, value) VALUES (?, ?)",
                ("packages", str(packages)),
            )
        command = env_data.get("command")
        if command:
            _ = conn.execute(
                "INSERT INTO environment (key, value) VALUES (?, ?)",
                ("command", str(command)),
            )
        git_diff = env_data.get("git_diff")
        if git_diff:
            _ = conn.execute(
                "INSERT INTO environment (key, value) VALUES (?, ?)",
                ("git_diff", str(git_diff)),
            )

    @staticmethod
    def _build_views(conn: sqlite3.Connection) -> None:
        """Create SQL views for common query patterns."""
        for sql in _VIEW_DEFINITIONS:
            _ = conn.execute(sql)

    def get_schema(self) -> SchemaOutput:
        """Get schema information for all tables and views."""
        manifest = self._bundle.manifest
        conn = self.connection
        tables: list[TableInfo] = []
        cursor = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY type DESC, name"
        )
        for table_name, obj_type in cursor.fetchall():
            col_cursor = conn.execute(f"PRAGMA table_info({table_name})")  # nosec B608
            columns = tuple(
                ColumnInfo(name=row[1], type=row[2], description="")
                for row in col_cursor.fetchall()
            )
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


def resolve_bundle_path(path: Path) -> Path:
    """Resolve a path to a bundle file."""
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
    """Open or create a query database for a bundle."""
    resolved_path = resolve_bundle_path(bundle_path)
    cache_path = resolved_path.with_suffix(resolved_path.suffix + ".sqlite")
    try:
        bundle = DebugBundle.load(resolved_path)
    except BundleValidationError as e:
        raise QueryError(f"Failed to load bundle: {e}") from e
    db = QueryDatabase(bundle, cache_path)
    if not _is_cache_valid(resolved_path, cache_path):
        if cache_path.exists():
            cache_path.unlink()
        db.build()
    else:
        db.mark_built()
    return db


def format_as_table(rows: Sequence[Mapping[str, Any]], *, truncate: bool = True) -> str:
    """Format query results as ASCII table."""
    if not rows:
        return "(no results)"
    max_width = _MAX_COLUMN_WIDTH if truncate else None
    columns = list(rows[0].keys())
    widths: dict[str, int] = {}
    for col in columns:
        col_max = len(col)
        for row in rows:
            val = str(row.get(col, ""))
            if max_width is not None and len(val) > max_width:
                val = val[: max_width - 3] + "..."
            col_max = max(col_max, len(val))
        widths[col] = col_max if max_width is None else min(col_max, max_width)
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)
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


def format_as_json(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format query results as JSON."""
    result = [dict(row) for row in rows]
    return json.dumps(result, indent=2)


def export_jsonl(bundle: DebugBundle, source: str) -> str | None:
    """Export raw JSONL content from bundle."""
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
