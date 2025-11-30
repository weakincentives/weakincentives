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

"""FastAPI app for editing prompt overrides captured in a snapshot."""

from __future__ import annotations

import threading
import webbrowser
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import Mapping, cast
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..prompt.overrides import PromptDescriptor, PromptOverride, SectionOverride, ToolOverride
from ..runtime.events import PromptRendered
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session.snapshots import Snapshot, SnapshotRestoreError
from ..types import JSONValue

logger: StructuredLogger = get_logger(__name__)


class SnapshotLoadError(RuntimeError):
    """Raised when a snapshot cannot be loaded or validated."""


@dataclass(frozen=True, slots=True)
class OptimizableSnapshot:
    """Container for a snapshot ready to be optimized."""

    path: Path
    snapshot: Snapshot
    raw_text: str


def load_snapshot(path: Path) -> OptimizableSnapshot:
    """Load a single snapshot entry from ``path``."""

    if not path.exists():
        msg = f"Snapshot file not found: {path}"
        raise SnapshotLoadError(msg)

    try:
        raw_text = path.read_text()
    except OSError as error:  # pragma: no cover - filesystem failures are unlikely
        msg = f"Snapshot file cannot be read: {path}"
        raise SnapshotLoadError(msg) from error

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        msg = f"Snapshot file contained no entries: {path}"
        raise SnapshotLoadError(msg)

    first_line = lines[0]
    try:
        snapshot = Snapshot.from_json(first_line)
    except SnapshotRestoreError as error:
        msg = f"Invalid snapshot: {error}"
        raise SnapshotLoadError(msg) from error

    return OptimizableSnapshot(path=path, snapshot=snapshot, raw_text=first_line)


def _descriptor_to_json(descriptor: PromptDescriptor) -> Mapping[str, JSONValue]:
    return {
        "ns": descriptor.ns,
        "key": descriptor.key,
        "sections": [
            {
                "path": "/".join(section.path),
                "content_hash": str(section.content_hash),
                "number": section.number,
            }
            for section in descriptor.sections
        ],
        "tools": [
            {
                "path": "/".join(tool.path),
                "name": tool.name,
                "contract_hash": str(tool.contract_hash),
            }
            for tool in descriptor.tools
        ],
    }


def _override_to_json(override: PromptOverride) -> Mapping[str, JSONValue]:
    return {
        "ns": override.ns,
        "prompt_key": override.prompt_key,
        "tag": override.tag,
        "sections": {
            "/".join(path) if isinstance(path, tuple) else str(path): {
                "expected_hash": str(section.expected_hash),
                "body": section.body,
            }
            for path, section in override.sections.items()
        },
        "tools": {
            name: {
                "expected_contract_hash": str(tool.expected_contract_hash),
                "description": tool.description,
                "param_descriptions": dict(tool.param_descriptions),
            }
            for name, tool in override.tool_overrides.items()
        },
    }


def _empty_override(descriptor: PromptDescriptor) -> PromptOverride:
    sections = {
        section.path: SectionOverride(expected_hash=section.content_hash, body="")
        for section in descriptor.sections
    }
    tools = {
        tool.name: ToolOverride(
            name=tool.name,
            expected_contract_hash=tool.contract_hash,
            description=None,
            param_descriptions={},
        )
        for tool in descriptor.tools
    }
    return PromptOverride(
        ns=descriptor.ns,
        prompt_key=descriptor.key,
        tag="latest",
        sections=sections,
        tool_overrides=tools,
    )


def _normalize_override(
    override: PromptOverride, descriptor: PromptDescriptor
) -> PromptOverride:
    normalized_sections: dict[tuple[str, ...], SectionOverride] = {}
    for path_key, section in override.sections.items():
        normalized_path = (
            tuple(path_key)
            if isinstance(path_key, tuple)
            else tuple(str(path_key).split("/"))
        )
        normalized_sections[normalized_path] = SectionOverride(
            expected_hash=section.expected_hash,
            body=section.body,
        )

    normalized_tools = dict(override.tool_overrides)

    if normalized_sections or normalized_tools:
        return replace(
            override,
            sections=normalized_sections,
            tool_overrides=normalized_tools,
        )
    return override


def _normalize_prompt_id(descriptor: PromptDescriptor, index: int) -> str:
    return f"{descriptor.ns}:{descriptor.key}:{index}"


class _OverrideStore:
    def __init__(self, snapshot: Snapshot) -> None:
        super().__init__()
        prompts: tuple[PromptRendered, ...] = snapshot.slices.get(
            PromptRendered, ()
        )
        self._prompts: list[PromptRendered] = list(prompts)
        overrides_slice: tuple[PromptOverride, ...] = snapshot.slices.get(
            PromptOverride, ()
        )
        override_lookup: dict[tuple[str, str], PromptOverride] = {
            (override.ns, override.prompt_key): override
            for override in overrides_slice
        }

        self._descriptors: dict[str, PromptDescriptor] = {}
        self._overrides: dict[str, PromptOverride] = {}
        self._original: dict[str, PromptOverride] = {}

        for index, prompt in enumerate(self._prompts):
            if prompt.descriptor is None:
                continue
            descriptor = prompt.descriptor
            prompt_id = _normalize_prompt_id(descriptor, index)
            self._descriptors[prompt_id] = descriptor
            override = override_lookup.get((descriptor.ns, descriptor.key))
            if override is None:
                override = _empty_override(descriptor)
            else:
                override = _normalize_override(override, descriptor)
            self._overrides[prompt_id] = override
            self._original[prompt_id] = override

    def list_prompts(self) -> list[Mapping[str, JSONValue]]:
        entries: list[Mapping[str, JSONValue]] = []
        for index, prompt in enumerate(self._prompts):
            descriptor = prompt.descriptor
            if descriptor is None:
                continue
            prompt_id = _normalize_prompt_id(descriptor, index)
            entries.append(
                {
                    "id": prompt_id,
                    "descriptor": _descriptor_to_json(descriptor),
                    "overrides": _override_to_json(self._overrides[prompt_id]),
                }
            )
        return entries

    def prompt_detail(self, prompt_id: str) -> Mapping[str, JSONValue]:
        descriptor = self._descriptors.get(prompt_id)
        if descriptor is None:
            raise HTTPException(status_code=404, detail="prompt not found")
        override = self._overrides[prompt_id]
        original = self._original[prompt_id]
        return {
            "id": prompt_id,
            "descriptor": _descriptor_to_json(descriptor),
            "overrides": _override_to_json(override),
            "original_overrides": _override_to_json(original),
        }

    def update_overrides(self, prompt_id: str, payload: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
        descriptor = self._descriptors.get(prompt_id)
        if descriptor is None:
            raise HTTPException(status_code=404, detail="prompt not found")
        override = self._overrides[prompt_id]

        allowed_keys = {"sections", "tools"}
        unknown = [key for key in payload if key not in allowed_keys]
        if unknown:
            msg = f"Unknown override fields: {', '.join(sorted(unknown))}"
            raise HTTPException(status_code=400, detail=msg)

        updated_sections = dict(override.sections)
        sections_payload = payload.get("sections", {})
        if not isinstance(sections_payload, Mapping):
            raise HTTPException(status_code=400, detail="sections must be an object")
        for path_str, entry in sections_payload.items():
            if not isinstance(path_str, str):
                raise HTTPException(status_code=400, detail="section keys must be strings")
            path = tuple(unquote(path_str).split("/"))
            descriptor_section = next(
                (
                    section
                    for section in descriptor.sections
                    if tuple(section.path) == path
                ),
                None,
            )
            if descriptor_section is None:
                raise HTTPException(status_code=400, detail=f"Unknown section: {path_str}")
            if not isinstance(entry, Mapping):
                raise HTTPException(status_code=400, detail="section payload must be an object")
            body = entry.get("body", "")
            if not isinstance(body, str):
                raise HTTPException(status_code=400, detail="section body must be a string")
            updated_sections[path] = SectionOverride(
                expected_hash=descriptor_section.content_hash,
                body=body,
            )

        updated_tools = dict(override.tool_overrides)
        tools_payload = payload.get("tools", {})
        if not isinstance(tools_payload, Mapping):
            raise HTTPException(status_code=400, detail="tools must be an object")
        for name, entry in tools_payload.items():
            if not isinstance(name, str):
                raise HTTPException(status_code=400, detail="tool names must be strings")
            descriptor_tool = next(
                (tool for tool in descriptor.tools if tool.name == name), None
            )
            if descriptor_tool is None:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {name}")
            if not isinstance(entry, Mapping):
                raise HTTPException(status_code=400, detail="tool payload must be an object")
            description = entry.get("description")
            if description is not None and not isinstance(description, str):
                raise HTTPException(status_code=400, detail="tool description must be a string or null")
            param_descriptions_raw = entry.get("param_descriptions", {})
            if not isinstance(param_descriptions_raw, Mapping):
                raise HTTPException(
                    status_code=400, detail="param_descriptions must be an object"
                )
            param_descriptions: dict[str, str] = {}
            for field_name, field_value in param_descriptions_raw.items():
                if not isinstance(field_name, str) or not isinstance(field_value, str):
                    raise HTTPException(
                        status_code=400,
                        detail="param_descriptions keys and values must be strings",
                    )
                param_descriptions[field_name] = field_value

            updated_tools[name] = ToolOverride(
                name=name,
                expected_contract_hash=descriptor_tool.contract_hash,
                description=description,
                param_descriptions=param_descriptions,
            )

        updated_override = replace(
            override,
            sections=updated_sections,
            tool_overrides=updated_tools,
        )
        self._overrides[prompt_id] = updated_override
        return _override_to_json(updated_override)

    def snapshot(self, base_snapshot: Snapshot) -> Snapshot:
        slices = dict(base_snapshot.slices)
        ordered_overrides: list[PromptOverride] = []
        for index, prompt in enumerate(self._prompts):
            descriptor = prompt.descriptor
            if descriptor is None:
                continue
            prompt_id = _normalize_prompt_id(descriptor, index)
            override = self._overrides[prompt_id]
            serialized_override = replace(
                override,
                sections={
                    "/".join(path): section for path, section in override.sections.items()
                },
            )
            ordered_overrides.append(serialized_override)
        slices[PromptOverride] = tuple(ordered_overrides)
        return Snapshot(
            created_at=base_snapshot.created_at,
            parent_id=base_snapshot.parent_id,
            children_ids=base_snapshot.children_ids,
            slices=slices,
            tags=base_snapshot.tags,
        )


class OptimizedSnapshotStore:
    """In-memory store for prompt overrides and snapshot reload handling."""

    def __init__(
        self,
        path: Path,
        *,
        loader: callable[[Path], OptimizableSnapshot],
        logger: StructuredLogger | None = None,
    ) -> None:
        super().__init__()
        self._path = path.resolve()
        self._loader = loader
        self._logger = logger or get_logger(__name__)
        self._snapshot: OptimizableSnapshot = self._loader(self._path)
        self._store = _OverrideStore(self._snapshot.snapshot)

    @property
    def snapshot_path(self) -> Path:
        return self._path

    def list_prompts(self) -> list[Mapping[str, JSONValue]]:
        return self._store.list_prompts()

    def get_prompt(self, prompt_id: str) -> Mapping[str, JSONValue]:
        return self._store.prompt_detail(prompt_id)

    def update_overrides(
        self, prompt_id: str, payload: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        return self._store.update_overrides(prompt_id, payload)

    def save(self) -> Mapping[str, JSONValue]:
        updated_snapshot = self._store.snapshot(self._snapshot.snapshot)
        payload = updated_snapshot.to_json()
        try:
            self._path.write_text(payload)
        except OSError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        self._logger.info(
            "Snapshot saved with overrides",
            event="optimize.server.save",
            context={"path": str(self._path)},
        )
        self._snapshot = OptimizableSnapshot(
            path=self._path, snapshot=updated_snapshot, raw_text=payload
        )
        return {"path": str(self._path), "bytes": len(payload.encode("utf-8"))}

    def reset(self) -> list[Mapping[str, JSONValue]]:
        self._snapshot = self._loader(self._path)
        self._store = _OverrideStore(self._snapshot.snapshot)
        self._logger.info(
            "Snapshot reset from disk",
            event="optimize.server.reset",
            context={"path": str(self._path)},
        )
        return self._store.list_prompts()


class _OptimizeHandlers:
    def __init__(self, *, store: OptimizedSnapshotStore, logger: StructuredLogger, static_dir: Path) -> None:
        super().__init__()
        self._store = store
        self._logger = logger
        self._static_dir = static_dir

    def index(self) -> str:
        index_path = self._static_dir / "optimize.html"
        return index_path.read_text()

    def list_prompts(self) -> list[Mapping[str, JSONValue]]:
        return self._store.list_prompts()

    def get_prompt(self, prompt_id: str) -> Mapping[str, JSONValue]:
        return self._store.get_prompt(prompt_id)

    def update_overrides(
        self, prompt_id: str, payload: Mapping[str, JSONValue]
    ) -> Mapping[str, JSONValue]:
        updated = self._store.update_overrides(prompt_id, payload)
        return {"id": prompt_id, "overrides": updated}

    def save(self) -> Mapping[str, JSONValue]:
        return self._store.save()

    def reset(self) -> list[Mapping[str, JSONValue]]:
        return self._store.reset()


def build_optimize_app(store: OptimizedSnapshotStore, logger: StructuredLogger) -> FastAPI:
    static_dir = Path(str(files(__package__).joinpath("static")))
    handlers = _OptimizeHandlers(store=store, logger=logger, static_dir=static_dir)

    app = FastAPI(title="wink optimize server")
    app.state.snapshot_store = store
    app.state.logger = logger
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
            "Failed to start wink optimize server",
            event="optimize.server.error",
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
            event="optimize.server.browser",
            context={"url": url, "error": repr(error)},
        )
