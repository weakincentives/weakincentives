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
from pathlib import Path
from types import MappingProxyType
from typing import cast
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt

from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session.snapshots import (
    Snapshot,
    SnapshotDocument,
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


class SnapshotLoadError(RuntimeError):
    """Raised when a snapshot cannot be loaded or validated."""


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
class LoadedSnapshot:
    meta: SnapshotMeta
    slices: Mapping[str, SnapshotSlicePayload]
    raw_payload: Mapping[str, JSONValue]
    raw_text: str
    path: Path
    document: SnapshotDocument

    def restore(self) -> Snapshot:
        return self.document.restore()


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


def _slice_lookup(
    slices: tuple[SnapshotSlicePayload, ...],
) -> Mapping[str, SnapshotSlicePayload]:
    return MappingProxyType({entry.slice_type: entry for entry in slices})


def _summaries_from_slices(
    slices: tuple[SnapshotSlicePayload, ...],
) -> tuple[SliceSummary, ...]:
    return tuple(
        SliceSummary(
            slice_type=entry.slice_type,
            item_type=entry.item_type,
            count=len(entry.items),
        )
        for entry in slices
    )


def _load_snapshot_line(
    line: str,
    line_number: int,
    snapshot_path: Path,
) -> LoadedSnapshot:
    try:
        document = SnapshotDocument.from_json(line)
    except SnapshotRestoreError as error:
        msg = f"Invalid snapshot at line {line_number}: {error}"
        raise SnapshotLoadError(msg) from error

    raw_payload = MappingProxyType(
        json.loads(line, object_pairs_hook=dict),
    )

    slices = _slice_lookup(document.payload.slices)
    summaries = _summaries_from_slices(document.payload.slices)

    session_id = document.payload.tags.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        msg = f"Snapshot is missing a session_id tag at line {line_number}"
        raise SnapshotLoadError(msg)

    meta = SnapshotMeta(
        version=document.payload.version,
        created_at=document.payload.created_at,
        path=str(snapshot_path),
        session_id=session_id,
        line_number=line_number,
        slices=tuple(summaries),
        tags=document.payload.tags,
    )

    return LoadedSnapshot(
        meta=meta,
        slices=slices,
        raw_payload=raw_payload,
        raw_text=line,
        path=snapshot_path,
        document=document,
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

    def snapshot(self) -> Snapshot:
        return self._current.restore()

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

    def slice_items(self, slice_type: str) -> SnapshotSlicePayload:
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


class _DebugAppHandlers:
    def __init__(
        self, *, store: SnapshotStore, logger: StructuredLogger, static_dir: Path
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

    def list_entries(self) -> list[Mapping[str, JSONValue]]:
        return self._store.list_entries()

    def get_slice(
        self,
        encoded_slice_type: str,
        *,
        offset: int = Query(0, ge=0),  # pyright: ignore[reportCallInDefaultInitializer]
        limit: int | None = Query(None, ge=0),  # pyright: ignore[reportCallInDefaultInitializer]
    ) -> Mapping[str, JSONValue]:
        slice_items = self._slice_items(encoded_slice_type)
        items = self._paginate_items(
            list(slice_items.items), offset=offset, limit=limit
        )
        rendered_items = [_render_markdown_values(item) for item in items]
        return {
            "slice_type": slice_items.slice_type,
            "item_type": slice_items.item_type,
            "items": rendered_items,
        }

    def get_raw(self) -> JSONResponse:
        return JSONResponse(json.loads(self._store.raw_text))

    def reload(self) -> Mapping[str, JSONValue]:
        return _meta_response(self._execute_reload())

    def list_snapshots(self) -> list[Mapping[str, JSONValue]]:
        return self._store.list_snapshots()

    def select(self, payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        session_id, line_number = self._parse_select_payload(payload)
        meta = self._execute_snapshot_command(
            lambda: self._store.select(session_id=session_id, line_number=line_number)
        )
        return _meta_response(meta)

    def switch(self, payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        path, session_id, line_number = self._parse_switch_payload(payload)
        meta = self._execute_snapshot_command(
            lambda: self._store.switch(
                path,
                session_id=session_id,
                line_number=line_number,
            )
        )
        return _meta_response(meta)

    def _slice_items(self, encoded_slice_type: str) -> SnapshotSlicePayload:
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

    def _execute_reload(self) -> SnapshotMeta:
        try:
            return self._store.reload()
        except SnapshotLoadError as error:
            self._logger.warning(
                "Snapshot reload failed",
                event="debug.server.reload_failed",
                context={"path": self._store.meta.path, "error": str(error)},
            )
            raise self._translate_snapshot_error(error) from error

    @staticmethod
    def _translate_snapshot_error(error: SnapshotLoadError) -> HTTPException:
        return HTTPException(status_code=400, detail=str(error))

    def _execute_snapshot_command(
        self, command: Callable[[], SnapshotMeta]
    ) -> SnapshotMeta:
        try:
            return command()
        except SnapshotLoadError as error:
            raise self._translate_snapshot_error(error) from error

    @staticmethod
    def _parse_select_payload(
        payload: Mapping[str, JSONValue],
    ) -> tuple[str | None, int | None]:
        session_value = payload.get("session_id")
        line_value = payload.get("line_number")

        if session_value is None and line_value is None:
            raise HTTPException(
                status_code=400,
                detail="session_id or line_number is required",
            )

        _validate_optional_session_id(session_value)
        _validate_optional_line_number(line_value)
        return cast(str | None, session_value), cast(int | None, line_value)

    @staticmethod
    def _parse_switch_payload(
        payload: Mapping[str, JSONValue],
    ) -> tuple[Path, str | None, int | None]:
        path_value = payload.get("path")
        if not isinstance(path_value, str):
            raise HTTPException(status_code=400, detail="path is required")

        session_value = payload.get("session_id")
        _validate_optional_session_id(session_value)

        line_value = payload.get("line_number")
        _validate_optional_line_number(line_value)

        return (
            Path(path_value),
            cast(str | None, session_value),
            cast(int | None, line_value),
        )


def _validate_optional_session_id(value: JSONValue | None) -> None:
    if value is not None and not isinstance(value, str):
        raise HTTPException(status_code=400, detail="session_id must be a string")


def _validate_optional_line_number(value: JSONValue | None) -> None:
    if value is not None and not isinstance(value, int):
        raise HTTPException(status_code=400, detail="line_number must be an integer")


def build_debug_app(store: SnapshotStore, logger: StructuredLogger) -> FastAPI:
    """Construct the FastAPI application for inspecting snapshots."""

    static_dir = Path(str(files(__package__).joinpath("static")))
    handlers = _DebugAppHandlers(store=store, logger=logger, static_dir=static_dir)

    app = FastAPI(title="wink snapshot debug server")
    app.state.snapshot_store = store
    app.state.logger = logger
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _ = app.get("/", response_class=HTMLResponse)(handlers.index)
    _ = app.get("/api/meta")(handlers.get_meta)
    _ = app.get("/api/entries")(handlers.list_entries)
    _ = app.get("/api/slices/{encoded_slice_type}")(handlers.get_slice)
    _ = app.get("/api/raw")(handlers.get_raw)
    _ = app.post("/api/reload")(handlers.reload)
    _ = app.get("/api/snapshots")(handlers.list_snapshots)
    _ = app.post("/api/select")(handlers.select)
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
