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

"""FastAPI app for exploring debug bundle zip files.

This module provides a web UI for inspecting debug bundles, powered by
SQLite for fast querying. It reuses the same database caching logic as
the `wink query` command.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import webbrowser
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from ..debug.bundle import DebugBundle
from ..errors import WinkError
from ..runtime.logging import StructuredLogger, get_logger
from ..types import JSONValue
from .query import (
    QueryDatabase,
    QueryError,
    iter_bundle_files,
    open_query_database,
    resolve_bundle_path,
)

# Module-level logger keeps loader warnings consistent with the debug server.
logger: StructuredLogger = get_logger(__name__)

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
_markdown = MarkdownIt("commonmark", {"linkify": True})

# pyright: reportUnusedFunction=false


class BundleLoadError(WinkError, RuntimeError):
    """Raised when a bundle cannot be loaded or validated."""


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
    """Build WHERE clause and params for log filtering.

    Args:
        filters: Dict with keys: level, logger, event, search, exclude_logger, exclude_event
    """
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


class BundleStore:
    """Manages the active bundle via SQLite database.

    Wraps QueryDatabase to provide bundle management (switching, reloading)
    and access to bundle data through SQL queries.
    """

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
        self._db: QueryDatabase = self._open_database(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def bundle(self) -> DebugBundle:
        """Access the underlying DebugBundle for raw file operations."""
        return self._db.bundle

    @staticmethod
    def _open_database(bundle_path: Path) -> QueryDatabase:
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

        return {
            "bundle_id": manifest.get("bundle_id", ""),
            "created_at": manifest.get("created_at", ""),
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

        # Use raw connection for parameterized queries
        conn = self._db.connection
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        if not rows and offset == 0:
            # Check if slice type exists at all
            count_cursor = conn.execute(
                "SELECT COUNT(*) FROM session_slices WHERE slice_type = ?",
                [slice_type],
            )
            if count_cursor.fetchone()[0] == 0:  # pragma: no branch
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

    def _execute_log_query(
        self,
        where_clause: str,
        params: list[Any],
        offset: int,
        limit: int | None,
    ) -> Mapping[str, JSONValue]:
        """Execute log query with pagination."""
        conn = self._db.connection

        # Get total count (where_clause built from safe patterns)
        count_query = f"SELECT COUNT(*) FROM logs {where_clause}"  # nosec B608
        total = conn.execute(count_query, list(params)).fetchone()[0]

        # Build select query (where_clause built from safe patterns)
        query = f"SELECT timestamp, level, logger, event, message, context FROM logs {where_clause}"  # nosec B608
        query_params = list(params)

        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            query_params.extend([limit, offset])
        elif offset > 0:
            query += " LIMIT -1 OFFSET ?"
            query_params.append(offset)

        cursor = conn.execute(query, query_params)
        entries = [_parse_log_row(row) for row in cursor.fetchall()]
        return {"entries": entries, "total": total}

    def get_log_facets(self) -> Mapping[str, JSONValue]:
        """Get unique loggers and events for filter suggestions."""
        conn = self._db.connection

        # Get unique loggers with counts
        loggers_cursor = conn.execute("""
            SELECT logger, COUNT(*) as count
            FROM logs
            WHERE logger IS NOT NULL AND logger != ''
            GROUP BY logger
            ORDER BY count DESC
        """)
        loggers: list[Mapping[str, JSONValue]] = [
            {"name": row[0], "count": row[1]} for row in loggers_cursor.fetchall()
        ]

        # Get unique events with counts
        events_cursor = conn.execute("""
            SELECT event, COUNT(*) as count
            FROM logs
            WHERE event IS NOT NULL AND event != ''
            GROUP BY event
            ORDER BY count DESC
        """)
        events: list[Mapping[str, JSONValue]] = [
            {"name": row[0], "count": row[1]} for row in events_cursor.fetchall()
        ]

        # Get level counts
        levels_cursor = conn.execute("""
            SELECT UPPER(level) as level, COUNT(*) as count
            FROM logs
            WHERE level IS NOT NULL AND level != ''
            GROUP BY UPPER(level)
            ORDER BY count DESC
        """)
        levels: list[Mapping[str, JSONValue]] = [
            {"name": row[0], "count": row[1]} for row in levels_cursor.fetchall()
        ]

        return {"loggers": loggers, "events": events, "levels": levels}

    def list_bundles(self) -> list[Mapping[str, JSONValue]]:
        """List all bundles in the root directory."""
        bundles: list[tuple[float, Path]] = []
        for candidate in sorted(iter_bundle_files(self._root)):
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


class _DebugAppHandlers:
    def __init__(
        self, *, store: BundleStore, logger: StructuredLogger, static_dir: Path
    ) -> None:
        super().__init__()
        self._store = store
        self._logger = logger
        self._static_dir = static_dir

    def index(self) -> str:
        index_path = self._static_dir / "index.html"
        return index_path.read_text()

    def get_meta(self) -> Mapping[str, JSONValue]:
        return self._store.get_meta()

    def get_manifest(self) -> JSONResponse:
        """Return the raw bundle manifest."""
        manifest = self._store.bundle.manifest
        return JSONResponse(json.loads(manifest.to_json()))

    def get_slice(
        self,
        encoded_slice_type: str,
        *,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int | None, Query(ge=0)] = None,
    ) -> Mapping[str, JSONValue]:
        from urllib.parse import unquote

        slice_type = unquote(encoded_slice_type)
        try:
            return self._store.get_slice_items(slice_type, offset=offset, limit=limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    def get_logs(  # noqa: PLR0913
        self,
        *,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int | None, Query(ge=0)] = None,
        level: Annotated[str | None, Query()] = None,
        logger: Annotated[str | None, Query()] = None,
        event: Annotated[str | None, Query()] = None,
        search: Annotated[str | None, Query()] = None,
        exclude_logger: Annotated[str | None, Query()] = None,
        exclude_event: Annotated[str | None, Query()] = None,
    ) -> Mapping[str, JSONValue]:
        """Return log entries from the bundle."""
        return self._store.get_logs(
            offset=offset,
            limit=limit,
            level=level,
            logger=logger,
            event=event,
            search=search,
            exclude_logger=exclude_logger,
            exclude_event=exclude_event,
        )

    def get_log_facets(self) -> Mapping[str, JSONValue]:
        """Return unique loggers, events, and levels for filter UI."""
        return self._store.get_log_facets()

    def get_config(self) -> JSONResponse:
        """Return the bundle config."""
        config = self._store.bundle.config
        if config is None:
            raise HTTPException(status_code=404, detail="Config not found in bundle")
        return JSONResponse(config)

    def get_metrics(self) -> JSONResponse:
        """Return the bundle metrics."""
        metrics = self._store.bundle.metrics
        if metrics is None:
            raise HTTPException(status_code=404, detail="Metrics not found in bundle")
        return JSONResponse(metrics)

    def get_error(self) -> JSONResponse:
        """Return error details if present."""
        error = self._store.bundle.error
        if error is None:
            raise HTTPException(status_code=404, detail="No error in bundle")
        return JSONResponse(error)

    def get_request_input(self) -> JSONResponse:
        """Return the request input."""
        return JSONResponse(self._store.bundle.request_input)

    def get_request_output(self) -> JSONResponse:
        """Return the request output."""
        return JSONResponse(self._store.bundle.request_output)

    def list_files(self) -> list[str]:
        """List files in the bundle."""
        return self._store.bundle.list_files()

    def get_file(self, file_path: str) -> JSONResponse:
        """Get content of a specific file in the bundle."""
        from ..debug.bundle import BundleValidationError

        try:
            content = self._store.bundle.read_file(file_path)
            # Try to parse as JSON
            try:
                parsed = json.loads(content.decode("utf-8"))
                return JSONResponse({"content": parsed, "type": "json"})
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Return as text or indicate binary
                try:
                    text = content.decode("utf-8")
                    return JSONResponse({"content": text, "type": "text"})
                except UnicodeDecodeError:
                    return JSONResponse({"content": None, "type": "binary"})
        except BundleValidationError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    def reload(self) -> Mapping[str, JSONValue]:
        try:
            return self._store.reload()
        except BundleLoadError as error:
            self._logger.warning(
                "Bundle reload failed",
                event="debug.server.reload_failed",
                context={"path": str(self._store.path), "error": str(error)},
            )
            raise HTTPException(status_code=400, detail=str(error)) from error

    def list_bundles(self) -> list[Mapping[str, JSONValue]]:
        return self._store.list_bundles()

    def switch(self, payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        path = self._parse_switch_payload(payload)
        try:
            return self._store.switch(path)
        except BundleLoadError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @staticmethod
    def _parse_switch_payload(payload: Mapping[str, JSONValue]) -> Path:
        path_value = payload.get("path")
        if not isinstance(path_value, str):
            raise HTTPException(status_code=400, detail="path is required")
        return Path(path_value)


def build_debug_app(store: BundleStore, logger: StructuredLogger) -> FastAPI:
    """Construct the FastAPI application for inspecting debug bundles."""

    static_dir = Path(str(files(__package__).joinpath("static")))
    handlers = _DebugAppHandlers(store=store, logger=logger, static_dir=static_dir)

    app = FastAPI(title="wink debug bundle server")
    app.state.bundle_store = store
    app.state.logger = logger
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _ = app.get("/", response_class=HTMLResponse)(handlers.index)
    _ = app.get("/api/meta")(handlers.get_meta)
    _ = app.get("/api/manifest")(handlers.get_manifest)
    _ = app.get("/api/slices/{encoded_slice_type}")(handlers.get_slice)
    _ = app.get("/api/logs")(handlers.get_logs)
    _ = app.get("/api/logs/facets")(handlers.get_log_facets)
    _ = app.get("/api/config")(handlers.get_config)
    _ = app.get("/api/metrics")(handlers.get_metrics)
    _ = app.get("/api/error")(handlers.get_error)
    _ = app.get("/api/request/input")(handlers.get_request_input)
    _ = app.get("/api/request/output")(handlers.get_request_output)
    _ = app.get("/api/files")(handlers.list_files)
    _ = app.get("/api/files/{file_path:path}")(handlers.get_file)
    _ = app.post("/api/reload")(handlers.reload)
    _ = app.get("/api/bundles")(handlers.list_bundles)
    _ = app.post("/api/switch")(handlers.switch)

    return app


def run_debug_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
    open_browser: bool,
    logger: StructuredLogger,
) -> int:
    """Run the uvicorn server for the supplied FastAPI app."""

    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.2, _open_browser, args=(url, logger)).start()

    logger.info(
        "Starting wink debug server",
        event="debug.server.start",
        context={"url": url},
    )

    try:
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_config=None,
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as error:  # pragma: no cover - exercised in wink tests
        logger.exception(
            "Failed to start wink debug server",
            event="debug.server.error",
            context={"url": url, "error": repr(error)},
        )
        return 3
    return 0


def _open_browser(url: str, logger: StructuredLogger) -> None:
    try:
        _ = webbrowser.open(url)
    except Exception as error:  # pragma: no cover - best effort logging
        logger.warning(
            "Unable to open browser",
            event="debug.server.browser",
            context={"url": url, "error": repr(error)},
        )
