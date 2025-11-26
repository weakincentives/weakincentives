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

"""FastAPI app for exploring session snapshot JSONL files."""

from __future__ import annotations

import json
import re
import threading
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from types import MappingProxyType
from typing import cast
from urllib.parse import unquote

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session.snapshots import Snapshot, SnapshotPayload, SnapshotRestoreError
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
_markdown = MarkdownIt("commonmark", {"linkify": True})

# pyright: reportUnusedFunction=false


class SnapshotLoadError(RuntimeError):
    """Raised when a snapshot cannot be loaded or validated."""


def _looks_like_markdown(text: str) -> bool:
    candidate = text.strip()
    if len(candidate) < 16:
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


@dataclass(slots=True, frozen=True)
class SliceSummary:
    slice_type: str
    item_type: str
    count: int


@dataclass(slots=True, frozen=True)
class SnapshotMeta:
    version: str
    created_at: str
    path: str
    session_id: str
    line_number: int
    slices: tuple[SliceSummary, ...]
    tags: Mapping[str, str]
    validation_error: str | None = None


@dataclass(slots=True, frozen=True)
class SliceItems:
    slice_type: str
    item_type: str
    items: tuple[Mapping[str, JSONValue], ...]


@dataclass(slots=True, frozen=True)
class LoadedSnapshot:
    meta: SnapshotMeta
    slices: Mapping[str, SliceItems]
    raw_payload: Mapping[str, JSONValue]
    raw_text: str
    path: Path


SnapshotLoader = Callable[[Path], tuple[LoadedSnapshot, ...]]


def load_snapshot(snapshot_path: Path) -> tuple[LoadedSnapshot, ...]:
    """Load and validate one or more snapshots from disk."""

    if not snapshot_path.exists():
        msg = f"Snapshot file not found: {snapshot_path}"
        raise SnapshotLoadError(msg)

    try:
        raw_text = snapshot_path.read_text()
    except OSError as error:  # pragma: no cover - filesystem failures are unlikely
        msg = f"Snapshot file cannot be read: {snapshot_path}"
        raise SnapshotLoadError(msg) from error

    entries: list[LoadedSnapshot] = []
    for line_number, line in _extract_snapshot_lines(raw_text):
        entries.append(_load_snapshot_line(line, line_number, snapshot_path))

    if not entries:
        msg = f"Snapshot file contained no entries: {snapshot_path}"
        raise SnapshotLoadError(msg)

    return tuple(entries)


def _extract_snapshot_lines(raw_text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for index, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        lines.append((index, stripped))
    return lines


def _load_snapshot_line(
    line: str,
    line_number: int,
    snapshot_path: Path,
) -> LoadedSnapshot:
    try:
        payload = SnapshotPayload.from_json(line)
    except SnapshotRestoreError as error:
        msg = f"Invalid snapshot at line {line_number}: {error}"
        raise SnapshotLoadError(msg) from error

    validation_error: str | None = None
    try:
        _ = Snapshot.from_json(line)
    except SnapshotRestoreError as error:
        validation_error = str(error)
        logger.warning(
            "Snapshot validation failed",
            event="wink.debug.snapshot_error",
            context={
                "path": str(snapshot_path),
                "line_number": line_number,
                "error": validation_error,
            },
        )

    raw_payload = MappingProxyType(
        json.loads(line, object_pairs_hook=dict),
    )

    slices: dict[str, SliceItems] = {}
    summaries: list[SliceSummary] = []
    for entry in payload.slices:
        items = tuple(entry.items)
        slices[entry.slice_type] = SliceItems(
            slice_type=entry.slice_type,
            item_type=entry.item_type,
            items=items,
        )
        summaries.append(
            SliceSummary(
                slice_type=entry.slice_type,
                item_type=entry.item_type,
                count=len(items),
            )
        )

    session_id = payload.tags.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        msg = f"Snapshot is missing a session_id tag at line {line_number}"
        raise SnapshotLoadError(msg)

    meta = SnapshotMeta(
        version=payload.version,
        created_at=payload.created_at,
        path=str(snapshot_path),
        session_id=session_id,
        line_number=line_number,
        slices=tuple(summaries),
        tags=payload.tags,
        validation_error=validation_error,
    )

    return LoadedSnapshot(
        meta=meta,
        slices=MappingProxyType(slices),
        raw_payload=raw_payload,
        raw_text=line,
        path=snapshot_path,
    )


class SnapshotStore:
    """In-memory store for the active snapshot and reload handling."""

    def __init__(
        self,
        path: Path,
        *,
        loader: SnapshotLoader,
        logger: StructuredLogger | None = None,
    ) -> None:
        super().__init__()
        resolved = path.resolve()
        self._root, self._path = self._normalize_path(resolved)
        self._loader = loader
        self._logger = logger or get_logger(__name__)
        self._entries: tuple[LoadedSnapshot, ...] = self._load_entries(self._path)
        self._index = 0

    @property
    def meta(self) -> SnapshotMeta:
        return self._current.meta

    @property
    def raw_payload(self) -> Mapping[str, JSONValue]:
        return self._current.raw_payload

    @property
    def raw_text(self) -> str:
        return self._current.raw_text

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> tuple[LoadedSnapshot, ...]:
        return self._entries

    def list_snapshots(self) -> list[Mapping[str, JSONValue]]:
        snapshots: list[tuple[float, Path]] = []
        for candidate in sorted(self._iter_snapshot_files(self._root)):
            try:
                stats = candidate.stat()
                created_at = max(stats.st_ctime, stats.st_mtime)
            except OSError:
                continue
            snapshots.append((created_at, candidate))

        entries: list[Mapping[str, JSONValue]] = []
        for created_at, candidate in sorted(
            snapshots, key=lambda entry: entry[0], reverse=True
        ):
            created_iso = datetime.fromtimestamp(created_at, tz=UTC).isoformat()
            entries.append(
                {
                    "path": str(candidate),
                    "name": candidate.name,
                    "created_at": created_iso,
                }
            )
        return entries

    def list_entries(self) -> list[Mapping[str, JSONValue]]:
        entries: list[Mapping[str, JSONValue]] = []
        for entry in self._entries:
            meta = entry.meta
            entries.append(
                {
                    "session_id": meta.session_id,
                    "name": f"{meta.session_id} (line {meta.line_number})",
                    "path": meta.path,
                    "line_number": meta.line_number,
                    "created_at": meta.created_at,
                    "tags": dict(meta.tags),
                    "selected": meta.session_id == self.meta.session_id,
                }
            )
        return entries

    def slice_items(self, slice_type: str) -> SliceItems:
        try:
            return self._current.slices[slice_type]
        except KeyError as error:
            raise KeyError(f"Unknown slice type: {slice_type}") from error

    def reload(self) -> SnapshotMeta:
        current_session_id = self.meta.session_id
        self._entries = self._load_entries(self._path)
        try:
            self._index = self._select_index(session_id=current_session_id)
        except SnapshotLoadError:
            self._index = 0
        self._logger.info(
            "Snapshot reloaded",
            event="debug.server.reload",
            context={"path": str(self._path)},
        )
        return self.meta

    def select(
        self, *, session_id: str | None = None, line_number: int | None = None
    ) -> SnapshotMeta:
        self._index = self._select_index(
            session_id=session_id,
            line_number=line_number,
        )
        return self.meta

    def switch(
        self,
        path: Path,
        *,
        session_id: str | None = None,
        line_number: int | None = None,
    ) -> SnapshotMeta:
        resolved = path.resolve()
        root, target = self._normalize_path(resolved)
        if root != self._root:
            msg = f"Snapshot must live under {self._root}"
            raise SnapshotLoadError(msg)

        self._root = root
        self._path = target
        self._entries = self._load_entries(target)
        self._index = self._select_index(
            session_id=session_id,
            line_number=line_number,
        )
        self._logger.info(
            "Snapshot switched",
            event="debug.server.switch",
            context={"path": str(self._path)},
        )
        return self.meta

    def _normalize_path(self, path: Path) -> tuple[Path, Path]:
        if path.is_dir():
            root = path
            candidates = sorted(
                self._iter_snapshot_files(root),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                msg = f"No snapshots found under {root}"
                raise SnapshotLoadError(msg)
            target = candidates[0]
        else:
            root = path.parent
            target = path
        return root, target

    def _select_index(
        self, *, session_id: str | None = None, line_number: int | None = None
    ) -> int:
        if session_id is not None:
            for index, entry in enumerate(self._entries):
                if entry.meta.session_id == session_id:
                    return index
            msg = f"Unknown session_id: {session_id}"
            raise SnapshotLoadError(msg)

        if line_number is not None:
            for index, entry in enumerate(self._entries):
                if entry.meta.line_number == line_number:
                    return index
            msg = f"Unknown line_number: {line_number}"
            raise SnapshotLoadError(msg)

        return 0

    def _load_entries(self, path: Path) -> tuple[LoadedSnapshot, ...]:
        entries = self._loader(path)
        if not entries:
            msg = f"No snapshots found under {path}"
            raise SnapshotLoadError(msg)
        return entries

    @staticmethod
    def _iter_snapshot_files(root: Path) -> list[Path]:
        candidates: list[Path] = []
        for pattern in ("*.jsonl", "*.json"):
            candidates.extend(p for p in root.glob(pattern) if p.is_file())
        return candidates

    @property
    def _current(self) -> LoadedSnapshot:
        return self._entries[self._index]


def build_debug_app(store: SnapshotStore, logger: StructuredLogger) -> FastAPI:
    """Construct the FastAPI application for inspecting snapshots."""

    static_dir = files(__package__).joinpath("static")
    app = _create_app()
    _configure_dependencies(app, store, logger)
    _configure_middleware(app)
    _mount_static_files(app, static_dir)
    _register_routes(app, store, logger, static_dir)
    return app


def _create_app() -> FastAPI:
    return FastAPI(title="wink snapshot debug server")


def _configure_dependencies(
    app: FastAPI, store: SnapshotStore, logger: StructuredLogger
) -> None:
    app.state.snapshot_store = store
    app.state.logger = logger


def _configure_middleware(app: FastAPI) -> None:
    del app  # Middleware hook placeholder for future instrumentation.


def _mount_static_files(app: FastAPI, static_dir: Traversable) -> None:
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _register_routes(
    app: FastAPI,
    store: SnapshotStore,
    logger: StructuredLogger,
    static_dir: Traversable,
) -> None:
    router = APIRouter()
    router.add_api_route(
        "/",
        _build_index_handler(static_dir),
        methods=["GET"],
        response_class=HTMLResponse,
    )
    router.add_api_route(
        "/api/meta",
        _build_meta_handler(store),
        methods=["GET"],
    )
    router.add_api_route(
        "/api/entries",
        _build_entries_handler(store),
        methods=["GET"],
    )
    router.add_api_route(
        "/api/slices/{encoded_slice_type}",
        _build_slice_handler(store),
        methods=["GET"],
    )
    router.add_api_route(
        "/api/raw",
        _build_raw_handler(store),
        methods=["GET"],
    )
    router.add_api_route(
        "/api/reload",
        _build_reload_handler(store, logger),
        methods=["POST"],
    )
    router.add_api_route(
        "/api/snapshots",
        _build_snapshots_handler(store),
        methods=["GET"],
    )
    router.add_api_route(
        "/api/select",
        _build_select_handler(store),
        methods=["POST"],
    )
    router.add_api_route(
        "/api/switch",
        _build_switch_handler(store),
        methods=["POST"],
    )

    app.include_router(router)


def _build_index_handler(static_dir: Traversable) -> Callable[[], str]:
    def index() -> str:
        index_path = static_dir / "index.html"
        return index_path.read_text()

    return index


def _build_meta_handler(store: SnapshotStore) -> Callable[[], Mapping[str, JSONValue]]:
    def get_meta() -> Mapping[str, JSONValue]:
        return _meta_response(store.meta)

    return get_meta


def _build_entries_handler(
    store: SnapshotStore,
) -> Callable[[], list[Mapping[str, JSONValue]]]:
    def list_entries() -> list[Mapping[str, JSONValue]]:
        return store.list_entries()

    return list_entries


def _build_slice_handler(
    store: SnapshotStore,
) -> Callable[..., Mapping[str, JSONValue]]:
    def get_slice(
        encoded_slice_type: str,
        *,
        offset: int = Query(0, ge=0),  # pyright: ignore[reportCallInDefaultInitializer]
        limit: int | None = Query(None, ge=0),  # pyright: ignore[reportCallInDefaultInitializer]
    ) -> Mapping[str, JSONValue]:
        slice_type = unquote(encoded_slice_type)
        try:
            slice_items = store.slice_items(slice_type)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        items = list(slice_items.items)
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]

        rendered_items = [_render_markdown_values(item) for item in items]

        return {
            "slice_type": slice_items.slice_type,
            "item_type": slice_items.item_type,
            "items": rendered_items,
        }

    return get_slice


def _build_raw_handler(store: SnapshotStore) -> Callable[[], JSONResponse]:
    def get_raw() -> JSONResponse:
        return JSONResponse(json.loads(store.raw_text))

    return get_raw


def _build_reload_handler(
    store: SnapshotStore, logger: StructuredLogger
) -> Callable[[], Mapping[str, JSONValue]]:
    def reload() -> Mapping[str, JSONValue]:
        try:
            meta = store.reload()
            return _meta_response(meta)
        except SnapshotLoadError as error:
            logger.warning(
                "Snapshot reload failed",
                event="debug.server.reload_failed",
                context={"path": store.meta.path, "error": str(error)},
            )
            raise HTTPException(status_code=400, detail=str(error)) from error

    return reload


def _build_snapshots_handler(
    store: SnapshotStore,
) -> Callable[[], list[Mapping[str, JSONValue]]]:
    def list_snapshots() -> list[Mapping[str, JSONValue]]:
        return store.list_snapshots()

    return list_snapshots


def _build_select_handler(
    store: SnapshotStore,
) -> Callable[[dict[str, JSONValue]], Mapping[str, JSONValue]]:
    def select(payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        session_value = payload.get("session_id")
        line_value = payload.get("line_number")

        if session_value is None and line_value is None:
            raise HTTPException(
                status_code=400,
                detail="session_id or line_number is required",
            )

        if session_value is not None and not isinstance(session_value, str):
            raise HTTPException(status_code=400, detail="session_id must be a string")

        if line_value is not None and not isinstance(line_value, int):
            raise HTTPException(
                status_code=400,
                detail="line_number must be an integer",
            )

        try:
            meta = store.select(session_id=session_value, line_number=line_value)
        except SnapshotLoadError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return _meta_response(meta)

    return select


def _build_switch_handler(
    store: SnapshotStore,
) -> Callable[[dict[str, JSONValue]], Mapping[str, JSONValue]]:
    def switch(payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        path_value = payload.get("path")
        if not isinstance(path_value, str):
            raise HTTPException(status_code=400, detail="path is required")

        session_value = payload.get("session_id")
        if session_value is not None and not isinstance(session_value, str):
            raise HTTPException(status_code=400, detail="session_id must be a string")

        line_value = payload.get("line_number")
        if line_value is not None and not isinstance(line_value, int):
            raise HTTPException(
                status_code=400,
                detail="line_number must be an integer",
            )

        try:
            meta = store.switch(
                Path(path_value),
                session_id=session_value,
                line_number=line_value,
            )
        except SnapshotLoadError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return _meta_response(meta)

    return switch


def _meta_response(meta: SnapshotMeta) -> Mapping[str, JSONValue]:
    return {
        "version": meta.version,
        "created_at": meta.created_at,
        "path": meta.path,
        "session_id": meta.session_id,
        "line_number": meta.line_number,
        "tags": dict(meta.tags),
        "validation_error": meta.validation_error,
        "slices": [
            {
                "slice_type": entry.slice_type,
                "item_type": entry.item_type,
                "count": entry.count,
            }
            for entry in meta.slices
        ],
    }


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
