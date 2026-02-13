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

"""Bundle storage and query helpers for the debug app."""
# pyright: reportImportCycles=false

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from markdown_it import MarkdownIt

from ..debug import DebugBundle
from ..errors import WinkError
from ..runtime.logging import StructuredLogger, get_logger
from ..types import JSONValue
from .query import (
    QueryError,
    open_query_database,
    resolve_bundle_path,
)

_MARKDOWN_WRAPPER_KEY = "__markdown__"
_MARKDOWN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|\n)#{1,6}\s"),
    re.compile(r"(^|\n)[-*+]\s"),
    re.compile(r"(^|\n)\d+\.\s"),
    re.compile(r"`{1,3}[^`]+`{1,3}"),
    re.compile(r"\[.+?\]\(.+?\)"),
    re.compile(r"\n\n"),
    re.compile(r"\*\*[^\s].+?\*\*"),
)
_MIN_MARKDOWN_LENGTH = 16
_markdown = MarkdownIt("commonmark", {"linkify": True, "html": False})


def _looks_like_markdown(text: str) -> bool:
    candidate = text.strip()
    if len(candidate) < _MIN_MARKDOWN_LENGTH:
        return False
    return any(pattern.search(candidate) for pattern in _MARKDOWN_PATTERNS)


def _render_markdown(text: str) -> Mapping[str, str]:
    return {
        "text": text,
        "html": _markdown.render(text),
    }


def _render_markdown_values(value: JSONValue) -> JSONValue:
    if isinstance(value, str):
        if _looks_like_markdown(value):
            return {_MARKDOWN_WRAPPER_KEY: _render_markdown(value)}
        return value

    if isinstance(value, Mapping):
        if _MARKDOWN_WRAPPER_KEY in value:
            return value
        mapping_value = cast(Mapping[str, JSONValue], value)
        normalized: dict[str, JSONValue] = {}
        for key, item in mapping_value.items():
            normalized[str(key)] = _render_markdown_values(item)
        return normalized

    if isinstance(value, list):
        return [_render_markdown_values(item) for item in value]

    return value


def _class_name(type_identifier: str) -> str:
    """Extract the class name from a fully qualified type identifier."""
    return type_identifier.rsplit(".", 1)[-1]


def _split_csv(value: str | None) -> list[str]:
    """Split comma-separated string into list of stripped values."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _add_in_filter(
    conditions: list[str],
    params: list[Any],
    column: str,
    values: list[str],
    *,
    upper: bool = False,
) -> None:
    """Add IN filter condition."""
    if not values:
        return
    placeholders = ",".join("?" for _ in values)
    col_expr = f"UPPER({column})" if upper else column
    vals = [v.upper() for v in values] if upper else values
    conditions.append(f"{col_expr} IN ({placeholders})")
    params.extend(vals)


def _add_not_in_filter(
    conditions: list[str],
    params: list[Any],
    column: str,
    values: list[str],
) -> None:
    """Add NOT IN filter condition (with NULL handling)."""
    if not values:
        return
    placeholders = ",".join("?" for _ in values)
    conditions.append(f"({column} IS NULL OR {column} NOT IN ({placeholders}))")
    params.extend(values)


def _build_log_filters(filters: Mapping[str, str | None]) -> tuple[str, list[Any]]:
    """Build WHERE clause and params for log filtering."""
    conditions: list[str] = []
    params: list[Any] = []

    _add_in_filter(
        conditions, params, "level", _split_csv(filters.get("level")), upper=True
    )
    _add_in_filter(conditions, params, "logger", _split_csv(filters.get("logger")))
    _add_in_filter(conditions, params, "event", _split_csv(filters.get("event")))
    _add_not_in_filter(
        conditions, params, "logger", _split_csv(filters.get("exclude_logger"))
    )
    _add_not_in_filter(
        conditions, params, "event", _split_csv(filters.get("exclude_event"))
    )

    search = filters.get("search")
    if search:
        conditions.append("(message LIKE ? OR context LIKE ?)")
        pattern = f"%{search}%"
        params.extend([pattern, pattern])

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params


def _parse_log_row(row: sqlite3.Row) -> Mapping[str, JSONValue]:
    """Parse a log row into a dictionary."""
    context_str = row[5] or "{}"
    try:
        context: JSONValue = json.loads(context_str)
    except json.JSONDecodeError:  # pragma: no cover
        context = {}
    return {
        "timestamp": row[0],
        "level": row[1],
        "logger": row[2],
        "event": row[3],
        "message": row[4],
        "context": context,
    }


def _build_transcript_filters(
    filters: Mapping[str, str | None],
) -> tuple[str, list[Any]]:
    """Build WHERE clause and params for transcript filtering."""
    conditions: list[str] = []
    params: list[Any] = []

    _add_in_filter(
        conditions,
        params,
        "transcript_source",
        _split_csv(filters.get("source")),
    )
    _add_in_filter(
        conditions,
        params,
        "entry_type",
        _split_csv(filters.get("entry_type")),
    )
    _add_not_in_filter(
        conditions,
        params,
        "transcript_source",
        _split_csv(filters.get("exclude_source")),
    )
    _add_not_in_filter(
        conditions,
        params,
        "entry_type",
        _split_csv(filters.get("exclude_entry_type")),
    )

    search = filters.get("search")
    if search:
        conditions.append("(content LIKE ? OR raw_json LIKE ? OR parsed LIKE ?)")
        pattern = f"%{search}%"
        params.extend([pattern, pattern, pattern])

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where_clause, params


def _parse_transcript_row(row: sqlite3.Row) -> Mapping[str, JSONValue]:
    """Parse a transcript row into a dictionary."""
    content = row[7]
    entry: dict[str, JSONValue] = {
        "rowid": row[0],
        "timestamp": row[1],
        "prompt_name": row[2],
        "transcript_source": row[3],
        "sequence_number": row[4],
        "entry_type": row[5],
        "role": row[6],
        "content": content,
        "tool_name": row[8],
        "tool_use_id": row[9],
        "raw_json": row[10],
        "parsed": row[11],
    }
    if isinstance(content, str) and _looks_like_markdown(content):
        entry["content_html"] = _markdown.render(content)
    return entry


class BundleLoadError(WinkError, RuntimeError):
    """Raised when a bundle cannot be loaded or validated."""


class BundleStore:
    """Manages the active bundle via SQLite database."""

    def __init__(
        self,
        path: Path,
        *,
        logger: StructuredLogger | None = None,
    ) -> None:
        super().__init__()
        resolved = path.resolve()
        self._root, self._path = self._normalize_path(resolved)
        self._logger = logger or get_logger(__name__)
        self._db = self._open_database(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def bundle(self) -> DebugBundle:
        """Access the underlying DebugBundle for raw file operations."""
        return self._db.bundle

    @staticmethod
    def _open_database(bundle_path: Path):  # noqa: ANN205
        """Open SQLite database for a bundle."""
        try:
            return open_query_database(bundle_path)
        except QueryError as e:
            raise BundleLoadError(str(e)) from e

    def get_meta(self) -> Mapping[str, JSONValue]:
        """Get bundle metadata from SQLite."""
        rows = self._db.execute_query("SELECT * FROM manifest LIMIT 1")
        if not rows:  # pragma: no cover - manifest always exists
            return {}

        manifest = rows[0]

        # Get slice information
        slice_rows = self._db.execute_query("""
            SELECT slice_type, COUNT(*) as count
            FROM session_slices
            GROUP BY slice_type
        """)

        slices: list[Mapping[str, JSONValue]] = []
        for row in slice_rows:
            slice_type = str(row["slice_type"])
            slices.append(
                {
                    "slice_type": slice_type,
                    "item_type": slice_type,
                    "display_name": _class_name(slice_type),
                    "item_display_name": _class_name(slice_type),
                    "count": row["count"],
                }
            )

        # Check whether the bundle has any transcript entries.
        transcript_count_rows = self._db.execute("SELECT COUNT(*) FROM transcript")
        has_transcript = transcript_count_rows[0][0] > 0

        return {
            "bundle_id": manifest.get("bundle_id", ""),
            "created_at": manifest.get("created_at", ""),
            "has_transcript": has_transcript,
            "path": str(self._path),
            "request_id": manifest.get("request_id", ""),
            "session_id": manifest.get("session_id"),
            "status": manifest.get("status", ""),
            "validation_error": None,
            "slices": slices,
        }

    def get_slice_items(
        self, slice_type: str, *, offset: int = 0, limit: int | None = None
    ) -> Mapping[str, JSONValue]:
        """Get items for a specific slice type."""
        # Query session_slices table
        query = "SELECT data FROM session_slices WHERE slice_type = ?"
        params: list[Any] = [slice_type]

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset > 0:
            query += " LIMIT -1 OFFSET ?"
            params.append(offset)

        # Use thread-safe execute
        rows = self._db.execute(query, params)

        if not rows and offset == 0:
            # Check if slice type exists at all
            count_rows = self._db.execute(
                "SELECT COUNT(*) FROM session_slices WHERE slice_type = ?",
                [slice_type],
            )
            if count_rows[0][0] == 0:  # pragma: no branch
                raise KeyError(f"Unknown slice type: {slice_type}")

        items: list[Mapping[str, JSONValue]] = []
        for row in rows:
            data_str = row[0]
            try:
                item = json.loads(data_str)
                if isinstance(
                    item, Mapping
                ):  # pragma: no branch - data is always Mapping
                    items.append(cast(Mapping[str, JSONValue], item))
            except (
                json.JSONDecodeError
            ):  # pragma: no cover - data is validated during build
                continue

        return {
            "slice_type": slice_type,
            "item_type": slice_type,
            "display_name": _class_name(slice_type),
            "item_display_name": _class_name(slice_type),
            "items": [_render_markdown_values(item) for item in items],
        }

    def get_logs(  # noqa: PLR0913
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        level: str | None = None,
        logger: str | None = None,
        event: str | None = None,
        search: str | None = None,
        exclude_logger: str | None = None,
        exclude_event: str | None = None,
    ) -> Mapping[str, JSONValue]:
        """Get log entries with filtering."""
        filters = {
            "level": level,
            "logger": logger,
            "event": event,
            "search": search,
            "exclude_logger": exclude_logger,
            "exclude_event": exclude_event,
        }
        where_clause, params = _build_log_filters(filters)
        return self._execute_log_query(where_clause, params, offset, limit)

    def get_transcript(  # noqa: PLR0913
        self,
        *,
        offset: int = 0,
        limit: int | None = None,
        source: str | None = None,
        entry_type: str | None = None,
        search: str | None = None,
        exclude_source: str | None = None,
        exclude_entry_type: str | None = None,
    ) -> Mapping[str, JSONValue]:
        """Get transcript entries with filtering."""
        filters = {
            "source": source,
            "entry_type": entry_type,
            "search": search,
            "exclude_source": exclude_source,
            "exclude_entry_type": exclude_entry_type,
        }
        where_clause, params = _build_transcript_filters(filters)
        return self._execute_transcript_query(where_clause, params, offset, limit)

    def _execute_log_query(
        self,
        where_clause: str,
        params: list[Any],
        offset: int,
        limit: int | None,
    ) -> Mapping[str, JSONValue]:
        """Execute log query with pagination."""
        # Get total count (where_clause built from safe patterns)
        count_query = f"SELECT COUNT(*) FROM logs {where_clause}"  # nosec B608
        count_rows = self._db.execute(count_query, list(params))
        total = count_rows[0][0]

        # Build select query (where_clause built from safe patterns)
        query = f"SELECT timestamp, level, logger, event, message, context FROM logs {where_clause}"  # nosec B608
        query_params = list(params)

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            query_params.extend([limit, offset])
        elif offset > 0:
            query += " LIMIT -1 OFFSET ?"
            query_params.append(offset)

        rows = self._db.execute(query, query_params)
        entries = [_parse_log_row(row) for row in rows]
        return {"entries": entries, "total": total}

    def _execute_transcript_query(
        self,
        where_clause: str,
        params: list[Any],
        offset: int,
        limit: int | None,
    ) -> Mapping[str, JSONValue]:
        count_query = f"SELECT COUNT(*) FROM transcript {where_clause}"  # nosec B608
        count_rows = self._db.execute(count_query, list(params))
        total = count_rows[0][0]

        query = (
            "SELECT rowid, timestamp, prompt_name, transcript_source, "
            "sequence_number, entry_type, role, content, tool_name, tool_use_id, "
            "raw_json, parsed "
            f"FROM transcript {where_clause} "  # nosec B608
            "ORDER BY rowid"
        )
        query_params = list(params)

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            query_params.extend([limit, offset])
        elif offset > 0:
            query += " LIMIT -1 OFFSET ?"
            query_params.append(offset)

        rows = self._db.execute(query, query_params)
        entries = [_parse_transcript_row(row) for row in rows]
        return {"entries": entries, "total": total}

    def get_log_facets(self) -> Mapping[str, JSONValue]:
        """Get unique loggers and events for filter suggestions."""
        logger_rows = self._db.execute("""
            SELECT logger, COUNT(*) as count
            FROM logs
            WHERE logger IS NOT NULL AND logger != ''
            GROUP BY logger
            ORDER BY count DESC
        """)
        loggers: list[Mapping[str, JSONValue]] = [
            {"name": row[0], "count": row[1]} for row in logger_rows
        ]

        event_rows = self._db.execute("""
            SELECT event, COUNT(*) as count
            FROM logs
            WHERE event IS NOT NULL AND event != ''
            GROUP BY event
            ORDER BY count DESC
        """)
        events: list[Mapping[str, JSONValue]] = [
            {"name": row[0], "count": row[1]} for row in event_rows
        ]

        level_rows = self._db.execute("""
            SELECT UPPER(level) as level, COUNT(*) as count
            FROM logs
            WHERE level IS NOT NULL AND level != ''
            GROUP BY UPPER(level)
            ORDER BY count DESC
        """)
        levels: list[Mapping[str, JSONValue]] = [
            {"name": row[0], "count": row[1]} for row in level_rows
        ]

        return {"loggers": loggers, "events": events, "levels": levels}

    def get_transcript_facets(self) -> Mapping[str, JSONValue]:
        """Get unique transcript sources and entry types for filter suggestions."""
        source_rows = self._db.execute("""
            SELECT transcript_source, COUNT(*) as count
            FROM transcript
            WHERE transcript_source IS NOT NULL AND transcript_source != ''
            GROUP BY transcript_source
            ORDER BY count DESC
        """)
        sources: list[Mapping[str, JSONValue]] = [
            {"name": row[0], "count": row[1]} for row in source_rows
        ]

        entry_type_rows = self._db.execute("""
            SELECT entry_type, COUNT(*) as count
            FROM transcript
            WHERE entry_type IS NOT NULL AND entry_type != ''
            GROUP BY entry_type
            ORDER BY count DESC
        """)
        entry_types: list[Mapping[str, JSONValue]] = [
            {"name": row[0], "count": row[1]} for row in entry_type_rows
        ]

        return {"sources": sources, "entry_types": entry_types}

    def get_environment(self) -> Mapping[str, JSONValue]:
        """Get environment data (system, python, git, container, env_vars)."""
        result: dict[str, JSONValue] = {}

        system_rows = self._db.execute("SELECT * FROM env_system LIMIT 1")
        if system_rows:
            row = system_rows[0]
            result["system"] = {
                "os_name": row["os_name"],
                "os_release": row["os_release"],
                "kernel_version": row["kernel_version"],
                "architecture": row["architecture"],
                "processor": row["processor"],
                "cpu_count": row["cpu_count"],
                "memory_total_bytes": row["memory_total_bytes"],
                "hostname": row["hostname"],
            }
        else:
            result["system"] = None

        python_rows = self._db.execute("SELECT * FROM env_python LIMIT 1")
        if python_rows:
            row = python_rows[0]
            version_info = row["version_info"]
            result["python"] = {
                "version": row["version"],
                "version_info": json.loads(version_info) if version_info else None,
                "implementation": row["implementation"],
                "executable": row["executable"],
                "prefix": row["prefix"],
                "base_prefix": row["base_prefix"],
                "is_virtualenv": bool(row["is_virtualenv"]),
            }
        else:
            result["python"] = None

        git_rows = self._db.execute("SELECT * FROM env_git LIMIT 1")
        if git_rows:
            row = git_rows[0]
            remotes = row["remotes"]
            tags = row["tags"]
            result["git"] = {
                "repo_root": row["repo_root"],
                "commit_sha": row["commit_sha"],
                "commit_short": row["commit_short"],
                "branch": row["branch"],
                "is_dirty": bool(row["is_dirty"])
                if row["is_dirty"] is not None
                else None,
                "remotes": json.loads(remotes) if remotes else None,
                "tags": json.loads(tags) if tags else None,
            }
        else:
            result["git"] = None

        container_rows = self._db.execute("SELECT * FROM env_container LIMIT 1")
        if container_rows:
            row = container_rows[0]
            is_containerized = row["is_containerized"]
            if is_containerized is not None and is_containerized:
                result["container"] = {
                    "runtime": row["runtime"],
                    "container_id": row["container_id"],
                    "image": row["image"],
                    "image_digest": row["image_digest"],
                    "cgroup_path": row["cgroup_path"],
                    "is_containerized": bool(is_containerized),
                }
            else:
                result["container"] = None
        else:
            result["container"] = None

        env_var_rows = self._db.execute(
            "SELECT name, value FROM env_vars ORDER BY name"
        )
        result["env_vars"] = {row["name"]: row["value"] for row in env_var_rows}

        pkg_rows = self._db.execute(
            "SELECT value FROM environment WHERE key = 'packages' LIMIT 1"
        )
        result["packages"] = pkg_rows[0]["value"] if pkg_rows else None

        cmd_rows = self._db.execute(
            "SELECT value FROM environment WHERE key = 'command' LIMIT 1"
        )
        result["command"] = cmd_rows[0]["value"] if cmd_rows else None

        diff_rows = self._db.execute(
            "SELECT value FROM environment WHERE key = 'git_diff' LIMIT 1"
        )
        result["git_diff"] = diff_rows[0]["value"] if diff_rows else None

        return result

    def list_bundles(self) -> list[Mapping[str, JSONValue]]:
        """List all bundles in the root directory."""
        # Late import: tests monkeypatch ``debug_app.iter_bundle_files`` so
        # we must resolve through that module to honour the patch.
        from . import debug_app as _host

        bundles: list[tuple[float, Path]] = []
        for candidate in sorted(_host.iter_bundle_files(self._root)):
            try:
                stats = candidate.stat()
                created_at = max(stats.st_ctime, stats.st_mtime)
            except OSError:
                continue
            bundles.append((created_at, candidate))

        entries: list[Mapping[str, JSONValue]] = []
        for created_at, candidate in sorted(
            bundles, key=lambda entry: entry[0], reverse=True
        ):
            created_iso = datetime.fromtimestamp(created_at, tz=UTC).isoformat()
            entries.append(
                {
                    "path": str(candidate),
                    "name": candidate.name,
                    "created_at": created_iso,
                    "selected": candidate == self._path,
                }
            )
        return entries

    def reload(self) -> Mapping[str, JSONValue]:
        """Reload the current bundle from disk."""
        # Delete cache to force rebuild
        cache_path = self._path.with_suffix(self._path.suffix + ".sqlite")
        if cache_path.exists():
            cache_path.unlink()
        # Open new database before closing old to preserve state on failure
        new_db = self._open_database(self._path)
        self._db.close()
        self._db = new_db
        self._logger.info(
            "Bundle reloaded",
            event="debug.server.reload",
            context={"path": str(self._path)},
        )
        return self.get_meta()

    def switch(self, path: Path) -> Mapping[str, JSONValue]:
        """Switch to a different bundle."""
        resolved = path.resolve()
        root, target = self._normalize_path(resolved)
        if root != self._root:
            msg = f"Bundle must live under {self._root}"
            raise BundleLoadError(msg)

        # Open new database before closing old to preserve state on failure
        new_db = self._open_database(target)
        self._db.close()
        self._root = root
        self._path = target
        self._db = new_db
        self._logger.info(
            "Bundle switched",
            event="debug.server.switch",
            context={"path": str(self._path)},
        )
        return self.get_meta()

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    @staticmethod
    def _normalize_path(path: Path) -> tuple[Path, Path]:
        if path.is_dir():
            root = path
            try:
                target = resolve_bundle_path(path)
            except QueryError:
                msg = f"No bundles found under {path}"
                raise BundleLoadError(msg) from None
        else:
            root = path.parent
            target = path
        return root, target
