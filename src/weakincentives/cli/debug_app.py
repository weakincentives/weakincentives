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

"""FastAPI app for exploring debug bundle zip files."""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Annotated, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from ..dataclasses import FrozenDataclass
from ..debug.bundle import BundleValidationError, DebugBundle
from ..errors import WinkError
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session.snapshots import (
    Snapshot,
    SnapshotPayload,
    SnapshotRestoreError,
    SnapshotSlicePayload,
)
from ..types import JSONValue

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


@FrozenDataclass()
class SliceSummary:
    slice_type: str
    item_type: str
    count: int


@FrozenDataclass()
class BundleMeta:
    """Metadata about a loaded bundle for the debug UI."""

    bundle_id: str
    created_at: str
    path: str
    request_id: str
    session_id: str | None
    status: str
    slices: tuple[SliceSummary, ...]
    validation_error: str | None = None


@FrozenDataclass()
class LoadedBundle:
    """A loaded debug bundle with parsed session data."""

    meta: BundleMeta
    bundle: DebugBundle
    session_slices: Mapping[str, SnapshotSlicePayload]
    path: Path


def load_bundle(bundle_path: Path) -> LoadedBundle:
    """Load and validate a debug bundle from disk."""
    if not bundle_path.exists():
        msg = f"Bundle file not found: {bundle_path}"
        raise BundleLoadError(msg)

    try:
        bundle = DebugBundle.load(bundle_path)
    except BundleValidationError as error:
        msg = f"Invalid bundle: {error}"
        raise BundleLoadError(msg) from error

    # Parse session data to get slices
    session_slices: dict[str, SnapshotSlicePayload] = {}
    slice_summaries: list[SliceSummary] = []
    validation_error: str | None = None

    session_content = bundle.session_after
    if session_content:
        # Session is JSONL - parse the first line
        lines = [line.strip() for line in session_content.splitlines() if line.strip()]
        if lines:
            try:
                payload = SnapshotPayload.from_json(lines[0])
                session_slices = {entry.slice_type: entry for entry in payload.slices}
                slice_summaries = [
                    SliceSummary(
                        slice_type=entry.slice_type,
                        item_type=entry.item_type,
                        count=len(entry.items),
                    )
                    for entry in payload.slices
                ]

                # Try full validation
                try:
                    _ = Snapshot.from_json(lines[0])
                except SnapshotRestoreError as error:
                    validation_error = str(error)
                    logger.warning(
                        "Session validation failed",
                        event="wink.debug.session_error",
                        context={
                            "path": str(bundle_path),
                            "error": validation_error,
                        },
                    )
            except SnapshotRestoreError as error:
                validation_error = str(error)
                logger.warning(
                    "Failed to parse session data",
                    event="wink.debug.session_parse_error",
                    context={
                        "path": str(bundle_path),
                        "error": validation_error,
                    },
                )

    manifest = bundle.manifest
    meta = BundleMeta(
        bundle_id=manifest.bundle_id,
        created_at=manifest.created_at,
        path=str(bundle_path),
        request_id=manifest.request.request_id,
        session_id=manifest.request.session_id,
        status=manifest.request.status,
        slices=tuple(slice_summaries),
        validation_error=validation_error,
    )

    return LoadedBundle(
        meta=meta,
        bundle=bundle,
        session_slices=session_slices,
        path=bundle_path,
    )


class BundleStore:
    """In-memory store for the active bundle and reload handling."""

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
        self._bundle: LoadedBundle = load_bundle(self._path)

    @property
    def meta(self) -> BundleMeta:
        return self._bundle.meta

    @property
    def bundle(self) -> DebugBundle:
        return self._bundle.bundle

    @property
    def path(self) -> Path:
        return self._path

    def list_bundles(self) -> list[Mapping[str, JSONValue]]:
        """List all bundles in the root directory."""
        bundles: list[tuple[float, Path]] = []
        for candidate in sorted(self._iter_bundle_files(self._root)):
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

    def slice_items(self, slice_type: str) -> SnapshotSlicePayload:
        try:
            return self._bundle.session_slices[slice_type]
        except KeyError as error:
            raise KeyError(f"Unknown slice type: {slice_type}") from error

    def reload(self) -> BundleMeta:
        """Reload the current bundle from disk."""
        self._bundle = load_bundle(self._path)
        self._logger.info(
            "Bundle reloaded",
            event="debug.server.reload",
            context={"path": str(self._path)},
        )
        return self.meta

    def switch(self, path: Path) -> BundleMeta:
        """Switch to a different bundle."""
        resolved = path.resolve()
        root, target = self._normalize_path(resolved)
        if root != self._root:
            msg = f"Bundle must live under {self._root}"
            raise BundleLoadError(msg)

        self._root = root
        self._path = target
        self._bundle = load_bundle(target)
        self._logger.info(
            "Bundle switched",
            event="debug.server.switch",
            context={"path": str(self._path)},
        )
        return self.meta

    def _normalize_path(self, path: Path) -> tuple[Path, Path]:
        if path.is_dir():
            root = path
            candidates = sorted(
                self._iter_bundle_files(root),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                msg = f"No bundles found under {root}"
                raise BundleLoadError(msg)
            target = candidates[0]
        else:
            root = path.parent
            target = path
        return root, target

    @staticmethod
    def _iter_bundle_files(root: Path) -> list[Path]:
        """Find all bundle zip files in a directory."""
        return [p for p in root.glob("*.zip") if p.is_file()]


def _meta_response(meta: BundleMeta) -> Mapping[str, JSONValue]:
    return {
        "bundle_id": meta.bundle_id,
        "created_at": meta.created_at,
        "path": meta.path,
        "request_id": meta.request_id,
        "session_id": meta.session_id,
        "status": meta.status,
        "validation_error": meta.validation_error,
        "slices": [
            {
                "slice_type": entry.slice_type,
                "item_type": entry.item_type,
                "display_name": _class_name(entry.slice_type),
                "item_display_name": _class_name(entry.item_type),
                "count": entry.count,
            }
            for entry in meta.slices
        ],
    }


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
        return _meta_response(self._store.meta)

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
        slice_items = self._slice_items(encoded_slice_type)
        items = self._paginate_items(
            list(slice_items.items), offset=offset, limit=limit
        )
        rendered_items = [_render_markdown_values(item) for item in items]
        return {
            "slice_type": slice_items.slice_type,
            "item_type": slice_items.item_type,
            "display_name": _class_name(slice_items.slice_type),
            "item_display_name": _class_name(slice_items.item_type),
            "items": rendered_items,
        }

    def get_logs(
        self,
        *,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int | None, Query(ge=0)] = None,
        level: Annotated[str | None, Query()] = None,
    ) -> Mapping[str, JSONValue]:
        """Return log entries from the bundle."""
        logs_content = self._store.bundle.logs
        if logs_content is None:
            return {"entries": [], "total": 0}

        entries = self._parse_log_entries(logs_content, level)
        total = len(entries)
        entries = self._paginate_items(entries, offset=offset, limit=limit)
        return {"entries": entries, "total": total}

    @staticmethod
    def _parse_log_entries(
        logs_content: str, level_filter: str | None
    ) -> list[Mapping[str, JSONValue]]:
        """Parse JSONL log content, optionally filtering by level."""
        entries: list[Mapping[str, JSONValue]] = []
        for line in logs_content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed: JSONValue = json.loads(stripped)
                if not isinstance(parsed, dict):
                    continue
                entry: Mapping[str, JSONValue] = cast(Mapping[str, JSONValue], parsed)
                if level_filter:
                    entry_level: JSONValue = entry.get("level", "")
                    if (
                        isinstance(entry_level, str)
                        and entry_level.upper() != level_filter.upper()
                    ):
                        continue
                    if not isinstance(entry_level, str):
                        continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue
        return entries

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

    def list_files_with_info(self) -> list[dict[str, str | int]]:
        """List files in the bundle with size info."""
        return self._store.bundle._list_files_with_info()  # pyright: ignore[reportPrivateUsage]

    def get_file(self, file_path: str) -> JSONResponse:
        """Get content of a specific file in the bundle."""
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

    def get_file_chunk(
        self,
        file_path: str,
        *,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int | None, Query(ge=1)] = None,
    ) -> JSONResponse:
        """Get a chunk of a text file by line numbers."""
        try:
            content, total_lines, has_more = self._store.bundle._read_file_chunk(  # pyright: ignore[reportPrivateUsage]
                file_path, offset=offset, limit=limit
            )
            return JSONResponse(
                {
                    "content": content,
                    "offset": offset,
                    "total_lines": total_lines,
                    "has_more": has_more,
                    "type": "text",
                }
            )
        except BundleValidationError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    def reload(self) -> Mapping[str, JSONValue]:
        return _meta_response(self._execute_reload())

    def list_bundles(self) -> list[Mapping[str, JSONValue]]:
        return self._store.list_bundles()

    def switch(self, payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        path = self._parse_switch_payload(payload)
        meta = self._execute_bundle_command(lambda: self._store.switch(path))
        return _meta_response(meta)

    def _slice_items(self, encoded_slice_type: str) -> SnapshotSlicePayload:
        from urllib.parse import unquote

        slice_type = unquote(encoded_slice_type)
        try:
            return self._store.slice_items(slice_type)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @staticmethod
    def _paginate_items(
        items: list[Mapping[str, JSONValue]], *, offset: int, limit: int | None
    ) -> list[Mapping[str, JSONValue]]:
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def _execute_reload(self) -> BundleMeta:
        try:
            return self._store.reload()
        except BundleLoadError as error:
            self._logger.warning(
                "Bundle reload failed",
                event="debug.server.reload_failed",
                context={"path": self._store.meta.path, "error": str(error)},
            )
            raise self._translate_bundle_error(error) from error

    @staticmethod
    def _translate_bundle_error(error: BundleLoadError) -> HTTPException:
        return HTTPException(status_code=400, detail=str(error))

    def _execute_bundle_command(self, command: Callable[[], BundleMeta]) -> BundleMeta:
        try:
            return command()
        except BundleLoadError as error:
            raise self._translate_bundle_error(error) from error

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
    _ = app.get("/api/config")(handlers.get_config)
    _ = app.get("/api/metrics")(handlers.get_metrics)
    _ = app.get("/api/error")(handlers.get_error)
    _ = app.get("/api/request/input")(handlers.get_request_input)
    _ = app.get("/api/request/output")(handlers.get_request_output)
    _ = app.get("/api/files")(handlers.list_files)
    _ = app.get("/api/files-info")(handlers.list_files_with_info)
    _ = app.get("/api/files/{file_path:path}")(handlers.get_file)
    _ = app.get("/api/files-chunk/{file_path:path}")(handlers.get_file_chunk)
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


# Backwards compatibility aliases (deprecated)
SnapshotLoadError = BundleLoadError
SnapshotStore = BundleStore
load_snapshot = load_bundle
