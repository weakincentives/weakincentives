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

"""FastAPI app for exploring session snapshot JSON files."""

from __future__ import annotations

import json
import threading
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session.snapshots import Snapshot, SnapshotPayload, SnapshotRestoreError
from ..types import JSONValue

# pyright: reportUnusedFunction=false


class SnapshotLoadError(RuntimeError):
    """Raised when a snapshot cannot be loaded or validated."""


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
    slices: tuple[SliceSummary, ...]


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


SnapshotLoader = Callable[[Path], LoadedSnapshot]


def load_snapshot(snapshot_path: Path) -> LoadedSnapshot:
    """Load and validate a snapshot from disk."""

    if not snapshot_path.exists():
        msg = f"Snapshot file not found: {snapshot_path}"
        raise SnapshotLoadError(msg)

    try:
        raw_text = snapshot_path.read_text()
    except OSError as error:  # pragma: no cover - filesystem failures are unlikely
        msg = f"Snapshot file cannot be read: {snapshot_path}"
        raise SnapshotLoadError(msg) from error

    try:
        payload = SnapshotPayload.from_json(raw_text)
        _ = Snapshot.from_json(raw_text)
    except SnapshotRestoreError as error:
        msg = f"Invalid snapshot: {error}"
        raise SnapshotLoadError(msg) from error

    raw_payload = MappingProxyType(
        json.loads(raw_text, object_pairs_hook=dict),
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

    meta = SnapshotMeta(
        version=payload.version,
        created_at=payload.created_at,
        path=str(snapshot_path),
        slices=tuple(summaries),
    )

    return LoadedSnapshot(
        meta=meta,
        slices=MappingProxyType(slices),
        raw_payload=raw_payload,
        raw_text=raw_text,
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
        self._current = self._loader(self._path)

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

    def list_snapshots(self) -> list[Mapping[str, JSONValue]]:
        snapshots: list[tuple[float, Path]] = []
        for candidate in sorted(self._root.glob("*.json")):
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

    def slice_items(self, slice_type: str) -> SliceItems:
        try:
            return self._current.slices[slice_type]
        except KeyError as error:
            raise KeyError(f"Unknown slice type: {slice_type}") from error

    def reload(self) -> SnapshotMeta:
        loaded = self._loader(self._path)
        self._current = loaded
        self._logger.info(
            "Snapshot reloaded",
            event="debug.server.reload",
            context={"path": str(self._path)},
        )
        return self._current.meta

    def switch(self, path: Path) -> SnapshotMeta:
        resolved = path.resolve()
        root, target = self._normalize_path(resolved)
        if root != self._root:
            msg = f"Snapshot must live under {self._root}"
            raise SnapshotLoadError(msg)

        loaded = self._loader(target)
        self._root = root
        self._path = target
        self._current = loaded
        self._logger.info(
            "Snapshot switched",
            event="debug.server.switch",
            context={"path": str(self._path)},
        )
        return self._current.meta

    def _normalize_path(self, path: Path) -> tuple[Path, Path]:
        if path.is_dir():
            root = path
            candidates = sorted(
                (p for p in root.glob("*.json") if p.is_file()),
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


def build_debug_app(store: SnapshotStore, logger: StructuredLogger) -> FastAPI:
    """Construct the FastAPI application for inspecting snapshots."""

    static_dir = files(__package__).joinpath("static")
    app = FastAPI(title="wink snapshot debug server")
    app.state.snapshot_store = store
    app.state.logger = logger
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_path = static_dir / "index.html"
        return index_path.read_text()

    @app.get("/api/meta")
    def get_meta() -> Mapping[str, JSONValue]:
        meta = store.meta
        return {
            "version": meta.version,
            "created_at": meta.created_at,
            "path": meta.path,
            "slices": [
                {
                    "slice_type": entry.slice_type,
                    "item_type": entry.item_type,
                    "count": entry.count,
                }
                for entry in meta.slices
            ],
        }

    @app.get("/api/slices/{encoded_slice_type}")
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

        return {
            "slice_type": slice_items.slice_type,
            "item_type": slice_items.item_type,
            "items": items,
        }

    @app.get("/api/raw")
    def get_raw() -> JSONResponse:
        return JSONResponse(json.loads(store.raw_text))

    @app.post("/api/reload")
    def reload() -> Mapping[str, JSONValue]:
        try:
            return {
                "version": store.reload().version,
                "created_at": store.meta.created_at,
                "path": store.meta.path,
                "slices": [
                    {
                        "slice_type": entry.slice_type,
                        "item_type": entry.item_type,
                        "count": entry.count,
                    }
                    for entry in store.meta.slices
                ],
            }
        except SnapshotLoadError as error:
            logger.warning(
                "Snapshot reload failed",
                event="debug.server.reload_failed",
                context={"path": store.meta.path, "error": str(error)},
            )
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/snapshots")
    def list_snapshots() -> list[Mapping[str, JSONValue]]:
        return store.list_snapshots()

    @app.post("/api/switch")
    def switch(payload: dict[str, JSONValue]) -> Mapping[str, JSONValue]:
        path_value = payload.get("path")
        if not isinstance(path_value, str):
            raise HTTPException(status_code=400, detail="path is required")

        try:
            meta = store.switch(Path(path_value))
        except SnapshotLoadError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {
            "version": meta.version,
            "created_at": meta.created_at,
            "path": meta.path,
            "slices": [
                {
                    "slice_type": entry.slice_type,
                    "item_type": entry.item_type,
                    "count": entry.count,
                }
                for entry in meta.slices
            ],
        }

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
