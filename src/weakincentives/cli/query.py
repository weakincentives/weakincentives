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

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override

from ..dataclasses import FrozenDataclassMixin
from ..debug import BundleValidationError, DebugBundle
from ..errors import WinkError
from ..resources.protocols import Closeable
from ._query_builders import build_all_tables
from ._query_formatters import (
    export_jsonl as export_jsonl,
    format_as_json as format_as_json,
    format_as_table as format_as_table,
)
from ._query_helpers import (
    _SCHEMA_VERSION,
    _get_table_description,
    _is_cache_valid,
)


class QueryError(WinkError, RuntimeError):
    """Raised when a query operation fails."""


@dataclass(slots=True, frozen=True)
class ColumnInfo(FrozenDataclassMixin):
    """Column metadata for schema output."""

    name: str
    type: str
    description: str = ""


@dataclass(slots=True, frozen=True)
class TableInfo(FrozenDataclassMixin):
    """Table metadata for schema output."""

    name: str
    description: str
    row_count: int
    columns: tuple[ColumnInfo, ...] = ()


@dataclass(slots=True, frozen=True)
class SchemaHints(FrozenDataclassMixin):
    """Hints for querying the database effectively."""

    json_extraction: tuple[str, ...] = ()
    common_queries: dict[str, str] = field(default_factory=lambda: {})


@dataclass(slots=True, frozen=True)
class SchemaOutput(FrozenDataclassMixin):
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
        build_all_tables(conn, self._bundle)
        conn.commit()
        self._built = True
        self.close()

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
