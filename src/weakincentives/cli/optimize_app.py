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

"""FastAPI app for editing prompt overrides captured in snapshots."""

from __future__ import annotations

import json
import threading
import webbrowser
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..prompt.overrides.versioning import PromptDescriptor, hash_json
from ..runtime.events import PromptRendered
from ..runtime.logging import StructuredLogger
from ..runtime.session.snapshots import Snapshot, SnapshotPayload, SnapshotRestoreError
from ..types import JSONValue

_ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "temperature",
        "system",
        "tools",
        "tools_enabled",
        "tone",
        "tool_choice",
        "max_tokens",
        "top_p",
    }
)


class SnapshotLoadError(RuntimeError):
    """Raised when an optimize snapshot cannot be loaded."""


@dataclass(slots=True, frozen=True)
class PromptOverrideSnapshotEntry:
    """Persist overrides for a specific prompt execution."""

    prompt_id: str
    overrides: dict[str, object]


@dataclass(slots=True, frozen=True)
class LoadedOptimizeSnapshot:
    """Container for a parsed snapshot entry."""

    snapshot: Snapshot
    payload: SnapshotPayload
    raw_text: str
    path: Path
    line_number: int


@dataclass(slots=True)
class PromptEntry:
    """Editable prompt metadata with overrides and provenance."""

    prompt_id: str
    descriptor: PromptDescriptor
    index: int
    overrides: MutableMapping[str, JSONValue]
    original_overrides: Mapping[str, JSONValue]

    def allowed_fields(self) -> frozenset[str]:
        keys = set(self.original_overrides)
        keys.update(self.overrides)
        return frozenset(keys | set(_ALLOWED_FIELDS))

    def has_unsaved_changes(self) -> bool:
        return dict(self.original_overrides) != dict(self.overrides)


def _extract_snapshot_lines(raw_text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for index, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        lines.append((index, stripped))
    return lines


def _resolve_snapshot_path(path: Path) -> Path:
    if path.is_file():
        return path

    if not path.exists():
        msg = f"Snapshot file not found: {path}"
        raise SnapshotLoadError(msg)

    if not path.is_dir():
        msg = f"Snapshot path must be a file or directory: {path}"
        raise SnapshotLoadError(msg)

    candidates: list[Path] = []
    for pattern in ("*.jsonl", "*.json"):
        candidates.extend(p for p in path.glob(pattern) if p.is_file())

    if not candidates:
        msg = f"No snapshots found under {path}"
        raise SnapshotLoadError(msg)

    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def _descriptor_version(descriptor: PromptDescriptor) -> str:
    canonical = asdict(descriptor)
    return str(hash_json(canonical))


def _build_prompt_id(descriptor: PromptDescriptor, index: int) -> str:
    version = _descriptor_version(descriptor)
    return f"{descriptor.ns}:{descriptor.key}:v{version}:{index}"


def load_snapshot(snapshot_path: Path) -> LoadedOptimizeSnapshot:
    target = _resolve_snapshot_path(snapshot_path)

    try:
        raw_text = target.read_text()
    except OSError as error:  # pragma: no cover - filesystem failure is rare
        msg = f"Snapshot file cannot be read: {target}"
        raise SnapshotLoadError(msg) from error

    lines = _extract_snapshot_lines(raw_text)
    if not lines:
        msg = f"Snapshot file contained no entries: {target}"
        raise SnapshotLoadError(msg)

    line_number, line = lines[0]
    try:
        payload = SnapshotPayload.from_json(line)
        snapshot = Snapshot.from_json(line)
    except SnapshotRestoreError as error:
        msg = f"Invalid snapshot at line {line_number}: {error}"
        raise SnapshotLoadError(msg) from error

    return LoadedOptimizeSnapshot(
        snapshot=snapshot,
        payload=payload,
        raw_text=line,
        path=target,
        line_number=line_number,
    )


class OptimizeStore:
    """In-memory override store backed by a snapshot on disk."""

    def __init__(
        self, loaded: LoadedOptimizeSnapshot, logger: StructuredLogger
    ) -> None:
        super().__init__()
        self._loaded = loaded
        self._logger = logger
        self._lock = threading.RLock()
        self._prompts: list[PromptEntry] = []
        self._build_prompts()

    @property
    def prompts(self) -> list[PromptEntry]:
        with self._lock:
            return list(self._prompts)

    def get_prompt(self, prompt_id: str) -> PromptEntry:
        with self._lock:
            for entry in self._prompts:
                if entry.prompt_id == prompt_id:
                    return entry
        msg = f"Unknown prompt_id: {prompt_id}"
        raise HTTPException(status_code=404, detail=msg)

    def update_overrides(
        self, prompt_id: str, updates: Mapping[str, JSONValue]
    ) -> PromptEntry:
        prompt = self.get_prompt(prompt_id)
        allowed = prompt.allowed_fields()
        for field in updates:
            if field not in allowed:
                raise HTTPException(
                    status_code=400, detail=f"Unknown override field: {field}"
                )

        with self._lock:
            prompt.overrides.update(updates)
            return replace(prompt, overrides=dict(prompt.overrides))

    def save(self) -> None:
        with self._lock:
            snapshot = self._loaded.snapshot
            overrides = tuple(
                PromptOverrideSnapshotEntry(
                    prompt_id=entry.prompt_id,
                    overrides=dict(entry.overrides),
                )
                for entry in self._prompts
            )

            slices = dict(snapshot.slices)
            slices[PromptOverrideSnapshotEntry] = overrides

            updated = Snapshot(
                created_at=snapshot.created_at,
                parent_id=snapshot.parent_id,
                children_ids=snapshot.children_ids,
                slices=slices,
                tags=snapshot.tags,
            )

            try:
                payload = updated.to_json()
                _ = self._loaded.path.write_text(payload + "\n", encoding="utf-8")
            except Exception as error:  # pragma: no cover - defensive
                msg = "Failed to persist snapshot"
                raise SnapshotLoadError(msg) from error

        self._logger.info(
            "Snapshot saved with overrides",
            event="optimize.server.save",
            context={"path": str(self._loaded.path)},
        )

    def reset(self) -> None:
        refreshed = load_snapshot(self._loaded.path)
        with self._lock:
            self._loaded = refreshed
            self._build_prompts()
        self._logger.info(
            "Snapshot reloaded",
            event="optimize.server.reset",
            context={"path": str(self._loaded.path)},
        )

    def _build_prompts(self) -> None:
        snapshot = self._loaded.snapshot
        prompt_slice = snapshot.slices.get(PromptRendered, ())
        overrides_slice: tuple[PromptOverrideSnapshotEntry, ...] = tuple(
            entry
            for entry in snapshot.slices.get(PromptOverrideSnapshotEntry, ())
            if isinstance(entry, PromptOverrideSnapshotEntry)
        )
        overrides_index = {
            entry.prompt_id: entry.overrides for entry in overrides_slice
        }

        entries: list[PromptEntry] = []
        for index, value in enumerate(prompt_slice):
            if not isinstance(value, PromptRendered):
                continue
            descriptor = value.descriptor
            if not isinstance(descriptor, PromptDescriptor):
                continue

            prompt_id = _build_prompt_id(descriptor, index)
            stored_overrides = {
                key: cast(JSONValue, value)
                for key, value in overrides_index.get(prompt_id, {}).items()
            }
            entries.append(
                PromptEntry(
                    prompt_id=prompt_id,
                    descriptor=descriptor,
                    index=index,
                    overrides=stored_overrides,
                    original_overrides=dict(stored_overrides),
                )
            )

        self._prompts = entries

    @property
    def path(self) -> Path:
        return self._loaded.path


def _serialize_descriptor(descriptor: PromptDescriptor) -> Mapping[str, JSONValue]:
    return cast(Mapping[str, JSONValue], json.loads(json.dumps(asdict(descriptor))))


class _OptimizeHandlers:
    def __init__(self, store: OptimizeStore, logger: StructuredLogger) -> None:
        super().__init__()
        self._store = store
        self._logger = logger

    @staticmethod
    def index() -> HTMLResponse:
        html = """
        <!doctype html>
        <html>
            <head>
                <title>wink optimize</title>
                <link rel="stylesheet" href="/static/style.css" />
            </head>
            <body>
                <h1>wink optimize</h1>
                <p>Use the API endpoints to list and update prompt overrides.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=html)

    def list_prompts(self) -> list[Mapping[str, JSONValue]]:
        return [
            {
                "prompt_id": entry.prompt_id,
                "descriptor": _serialize_descriptor(entry.descriptor),
                "overrides": dict(entry.overrides),
                "original_overrides": dict(entry.original_overrides),
                "has_unsaved_changes": entry.has_unsaved_changes(),
            }
            for entry in self._store.prompts
        ]

    def get_prompt(self, prompt_id: str) -> Mapping[str, JSONValue]:
        entry = self._store.get_prompt(prompt_id)
        return {
            "prompt_id": entry.prompt_id,
            "descriptor": _serialize_descriptor(entry.descriptor),
            "overrides": dict(entry.overrides),
            "original_overrides": dict(entry.original_overrides),
            "has_unsaved_changes": entry.has_unsaved_changes(),
        }

    def update_overrides(
        self, prompt_id: str, payload: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        entry = self._store.update_overrides(prompt_id, payload)
        return {
            "prompt_id": entry.prompt_id,
            "overrides": dict(entry.overrides),
            "has_unsaved_changes": entry.has_unsaved_changes(),
        }

    def save(self) -> Mapping[str, JSONValue]:
        try:
            self._store.save()
        except SnapshotLoadError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"status": "ok", "path": str(self._store.path)}

    def reset(self) -> list[Mapping[str, JSONValue]]:
        try:
            self._store.reset()
        except SnapshotLoadError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return self.list_prompts()


def build_optimize_app(store: OptimizeStore, logger: StructuredLogger) -> FastAPI:
    app = FastAPI(title="wink optimize server")
    handlers = _OptimizeHandlers(store=store, logger=logger)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    _ = app.get("/", response_class=HTMLResponse)(handlers.index)
    _ = app.get("/api/prompts")(handlers.list_prompts)
    _ = app.get("/api/prompts/{prompt_id}")(handlers.get_prompt)
    _ = app.post("/api/prompts/{prompt_id}/overrides")(handlers.update_overrides)
    _ = app.post("/api/save")(handlers.save)
    _ = app.post("/api/reset")(handlers.reset)
    return app


def run_optimize_server(
    app: FastAPI,
    *,
    host: str,
    port: int,
    open_browser: bool,
    logger: StructuredLogger,
) -> int:
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.2, _open_browser, args=(url, logger)).start()

    logger.info(
        "Starting wink optimize server",
        event="optimize.server.start",
        context={"url": url},
    )

    try:
        config = uvicorn.Config(app, host=host, port=port, log_config=None)
        server = uvicorn.Server(config)
        server.run()
    except Exception as error:  # pragma: no cover - defensive guard
        logger.exception(
            "Failed to start wink optimize server",
            event="optimize.server.error",
            context={"url": url, "error": repr(error)},
        )
        return 3
    return 0


def _open_browser(url: str, logger: StructuredLogger) -> None:
    try:
        _ = webbrowser.open(url)
    except Exception as error:  # pragma: no cover - best effort
        logger.warning(
            "Unable to open browser",
            event="optimize.server.browser",
            context={"url": url, "error": repr(error)},
        )
