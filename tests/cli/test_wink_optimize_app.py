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

"""Tests for the wink optimize server."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from weakincentives.cli import optimize_app
from weakincentives.runtime.events import PromptRendered
from weakincentives.runtime.logging import StructuredLogger, get_logger
from weakincentives.runtime.session.snapshots import Snapshot


def _build_snapshot(tmp_path: Path) -> tuple[Path, str]:
    descriptor = optimize_app.PromptDescriptor(
        ns="demo", key="hello", sections=[], tools=[], chapters=[]
    )
    prompt = PromptRendered(
        prompt_ns="demo",
        prompt_key="hello",
        prompt_name="hello",
        adapter="test",
        session_id=None,
        render_inputs=(),
        rendered_prompt="hello",
        created_at=datetime.now(UTC),
        descriptor=descriptor,
    )

    prompt_id = optimize_app._build_prompt_id(descriptor, 0)
    override = optimize_app.PromptOverrideSnapshotEntry(
        prompt_id=prompt_id,
        overrides={"model": "gpt-4"},
    )

    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            PromptRendered: (prompt,),
            optimize_app.PromptOverrideSnapshotEntry: (override,),
        },
        tags={"session_id": "demo-session"},
    )

    path = tmp_path / "snapshot.jsonl"
    path.write_text(snapshot.to_json() + "\n", encoding="utf-8")
    return path, prompt_id


def test_optimize_store_builds_prompts(tmp_path: Path) -> None:
    snapshot_path, prompt_id = _build_snapshot(tmp_path)
    loaded = optimize_app.load_snapshot(snapshot_path)
    store = optimize_app.OptimizeStore(loaded, logger=_build_logger())

    prompts = store.prompts
    assert len(prompts) == 1
    assert prompts[0].prompt_id == prompt_id
    assert prompts[0].overrides == {"model": "gpt-4"}


def test_optimize_app_routes(tmp_path: Path) -> None:
    snapshot_path, prompt_id = _build_snapshot(tmp_path)
    loaded = optimize_app.load_snapshot(snapshot_path)
    store = optimize_app.OptimizeStore(loaded, logger=_build_logger())
    app = optimize_app.build_optimize_app(store, logger=_build_logger())
    client = TestClient(app)

    index = client.get("/")
    assert index.status_code == 200
    assert "wink optimize" in index.text

    response = client.get("/api/prompts")
    assert response.status_code == 200
    prompts = response.json()
    assert prompts[0]["prompt_id"] == prompt_id

    prompt_response = client.get(f"/api/prompts/{prompt_id}")
    assert prompt_response.status_code == 200
    assert prompt_response.json()["prompt_id"] == prompt_id

    update = client.post(
        f"/api/prompts/{prompt_id}/overrides",
        json={"temperature": 0.4},
    )
    assert update.status_code == 200
    assert update.json()["overrides"]["temperature"] == 0.4

    bad = client.post(
        f"/api/prompts/{prompt_id}/overrides",
        json={"unknown": True},
    )
    assert bad.status_code == 400

    save = client.post("/api/save")
    assert save.status_code == 200

    reloaded = Snapshot.from_json(snapshot_path.read_text().strip())
    saved_overrides = cast(
        optimize_app.PromptOverrideSnapshotEntry,
        reloaded.slices[optimize_app.PromptOverrideSnapshotEntry][0],
    )
    assert saved_overrides.overrides["temperature"] == 0.4

    # mutate the file and ensure reset reloads the latest content
    replacement_override = optimize_app.PromptOverrideSnapshotEntry(
        prompt_id=prompt_id,
        overrides={"model": "gpt-3.5"},
    )
    replacement_snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            PromptRendered: reloaded.slices[PromptRendered],
            optimize_app.PromptOverrideSnapshotEntry: (replacement_override,),
        },
        tags={"session_id": "demo-session"},
    )
    snapshot_path.write_text(replacement_snapshot.to_json() + "\n", encoding="utf-8")

    reset = client.post("/api/reset")
    assert reset.status_code == 200
    refreshed = reset.json()[0]
    assert refreshed["overrides"]["model"] == "gpt-3.5"


def test_optimize_app_handlers_error_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_path, prompt_id = _build_snapshot(tmp_path)
    loaded = optimize_app.load_snapshot(snapshot_path)
    store = optimize_app.OptimizeStore(loaded, logger=_build_logger())
    app = optimize_app.build_optimize_app(store, logger=_build_logger())
    client = TestClient(app)

    monkeypatch.setattr(
        store,
        "save",
        lambda: (_ for _ in ()).throw(optimize_app.SnapshotLoadError("boom")),
    )
    save_error = client.post("/api/save")
    assert save_error.status_code == 400

    monkeypatch.setattr(
        store,
        "reset",
        lambda: (_ for _ in ()).throw(optimize_app.SnapshotLoadError("boom")),
    )
    reset_error = client.post("/api/reset")
    assert reset_error.status_code == 400

    missing = client.get("/api/prompts/unknown")
    assert missing.status_code == 404
    assert missing.json()["detail"].startswith("Unknown prompt_id")

    update_error = client.post(
        f"/api/prompts/{prompt_id}/overrides", json={"unknown": True}
    )
    assert update_error.status_code == 400


def test_snapshot_parsing_errors(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("\n", encoding="utf-8")
    with pytest.raises(optimize_app.SnapshotLoadError):
        optimize_app.load_snapshot(empty_path)

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(optimize_app.SnapshotLoadError):
        optimize_app.load_snapshot(invalid_path)

    lines = optimize_app._extract_snapshot_lines("\nfirst\n\nsecond\n")
    assert lines == [(2, "first"), (4, "second")]


def test_resolve_snapshot_path_variants(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(optimize_app.SnapshotLoadError):
        optimize_app._resolve_snapshot_path(missing)

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(optimize_app.SnapshotLoadError):
        optimize_app._resolve_snapshot_path(fifo)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(optimize_app.SnapshotLoadError):
        optimize_app._resolve_snapshot_path(empty_dir)

    older = empty_dir / "older.jsonl"
    older.write_text("older", encoding="utf-8")
    newer = empty_dir / "newer.json"
    newer.write_text("newer", encoding="utf-8")
    os.utime(older, (time.time() - 10, time.time() - 10))
    resolved = optimize_app._resolve_snapshot_path(empty_dir)
    assert resolved == newer


def test_optimize_store_get_prompt_unknown(tmp_path: Path) -> None:
    snapshot_path, _ = _build_snapshot(tmp_path)
    loaded = optimize_app.load_snapshot(snapshot_path)
    store = optimize_app.OptimizeStore(loaded, logger=_build_logger())

    with pytest.raises(optimize_app.HTTPException):
        store.get_prompt("does-not-exist")


def test_optimize_store_filters_invalid_entries(tmp_path: Path) -> None:
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            optimize_app.PromptOverrideSnapshotEntry: (
                optimize_app.PromptOverrideSnapshotEntry(
                    prompt_id="demo:hello:v0:0", overrides={"model": "gpt-4"}
                ),
            ),
            PromptRendered: (
                PromptRendered(
                    prompt_ns="demo",
                    prompt_key="hello",
                    prompt_name="hello",
                    adapter="test",
                    session_id=None,
                    render_inputs=(),
                    rendered_prompt="hello",
                    created_at=datetime.now(UTC),
                    descriptor=None,
                ),
            ),
        },
        tags={"session_id": "demo-session"},
    )
    loaded = optimize_app.LoadedOptimizeSnapshot(
        snapshot=snapshot,
        payload=optimize_app.SnapshotPayload.from_json(snapshot.to_json()),
        raw_text=snapshot.to_json(),
        path=tmp_path / "snapshot.jsonl",
        line_number=1,
    )
    store = optimize_app.OptimizeStore(loaded, logger=_build_logger())
    assert store._loaded.snapshot.slices[PromptRendered]
    assert store.prompts == []


def test_optimize_store_skips_non_prompt_rendered(tmp_path: Path) -> None:
    snapshot_path, _ = _build_snapshot(tmp_path)
    loaded = optimize_app.load_snapshot(snapshot_path)
    dummy_snapshot = SimpleNamespace(slices={PromptRendered: (object(),)})
    injected = replace(loaded, snapshot=dummy_snapshot)

    store = optimize_app.OptimizeStore(injected, logger=_build_logger())
    assert store.prompts == []


def test_run_optimize_server_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class DummyConfig:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("config")

    class DummyServer:
        def __init__(self, config: object) -> None:  # pragma: no cover - simple stub
            calls.append("server")

        def run(self) -> None:
            calls.append("run")

    class DummyTimer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls.append("timer")

        def start(self) -> None:
            calls.append("timer_start")

    monkeypatch.setattr(optimize_app.uvicorn, "Config", DummyConfig)
    monkeypatch.setattr(optimize_app.uvicorn, "Server", DummyServer)
    monkeypatch.setattr(optimize_app.threading, "Timer", DummyTimer)

    result = optimize_app.run_optimize_server(
        optimize_app.FastAPI(),
        host="127.0.0.1",
        port=8000,
        open_browser=True,
        logger=_build_logger(),
    )
    assert result == 0
    assert calls == ["timer", "timer_start", "config", "server", "run"]

    class ErrorServer(DummyServer):
        def run(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(optimize_app.uvicorn, "Server", ErrorServer)
    calls.clear()
    result = optimize_app.run_optimize_server(
        optimize_app.FastAPI(),
        host="127.0.0.1",
        port=8001,
        open_browser=False,
        logger=_build_logger(),
    )
    assert result == 3


def test_open_browser_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    def _fail(url: str) -> None:
        opened.append(url)
        raise RuntimeError("no browser")

    monkeypatch.setattr(optimize_app.webbrowser, "open", _fail)
    optimize_app._open_browser("http://example.com", _build_logger())
    assert opened == ["http://example.com"]


def _build_logger() -> StructuredLogger:
    return get_logger("tests.weakincentives.optimize_app")
