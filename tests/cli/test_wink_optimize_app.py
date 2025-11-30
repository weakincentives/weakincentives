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

"""Tests for the wink optimize FastAPI application."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from weakincentives.cli import optimize_app
from weakincentives.prompt.overrides import PromptOverride, SectionOverride
from weakincentives.prompt.overrides.versioning import (
    SectionDescriptor,
    ToolDescriptor,
    hash_text,
)
from weakincentives.runtime.events import PromptRendered
from weakincentives.runtime.session.snapshots import Snapshot


def _build_snapshot(path: Path, section_body: str = "body") -> tuple[Snapshot, str]:
    content_hash = hash_text("section-template")
    descriptor = optimize_app.PromptDescriptor(
        ns="demo",
        key="example",
        sections=[SectionDescriptor(("intro",), content_hash, 0)],
        tools=[],
    )
    prompt = PromptRendered(
        prompt_ns="demo",
        prompt_key="example",
        prompt_name="Demo",
        adapter="openai",
        session_id=None,
        render_inputs=(),
        rendered_prompt="hello",
        descriptor=descriptor,
        created_at=datetime.now(UTC),
    )
    override = PromptOverride(
        ns="demo",
        prompt_key="example",
        tag="latest",
        sections={
            "intro": SectionOverride(
                expected_hash=content_hash,
                body=section_body,
            )
        },
        tool_overrides={},
    )
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={
            PromptRendered: (prompt,),
            optimize_app.PromptOverride: (override,),
        },
        tags={"session_id": "demo"},
    )
    payload = snapshot.to_json()
    path.write_text(payload)
    return snapshot, payload


def _build_snapshot_without_overrides(path: Path) -> optimize_app.OptimizableSnapshot:
    content_hash = hash_text("section-template")
    descriptor = optimize_app.PromptDescriptor(
        ns="demo",
        key="missing",
        sections=[SectionDescriptor(("intro",), content_hash, 0)],
        tools=[],
    )
    prompt = PromptRendered(
        prompt_ns="demo",
        prompt_key="missing",
        prompt_name="Demo",
        adapter="openai",
        session_id=None,
        render_inputs=(),
        rendered_prompt="hello",
        descriptor=descriptor,
        created_at=datetime.now(UTC),
    )
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={PromptRendered: (prompt,)},
        tags={"session_id": "demo"},
    )
    payload = snapshot.to_json()
    path.write_text(payload)
    return optimize_app.OptimizableSnapshot(path=path, snapshot=snapshot, raw_text=payload)


def test_load_snapshot_errors(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "missing.jsonl"

    with pytest.raises(optimize_app.SnapshotLoadError):
        optimize_app.load_snapshot(snapshot_path)

    snapshot_path.write_text("")
    with pytest.raises(optimize_app.SnapshotLoadError):
        optimize_app.load_snapshot(snapshot_path)

    snapshot_path.write_text("{")
    with pytest.raises(optimize_app.SnapshotLoadError):
        optimize_app.load_snapshot(snapshot_path)


def test_optimize_routes_update_and_save(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot, _ = _build_snapshot(snapshot_path)
    logger = optimize_app.get_logger("test.optimize")
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path, loader=optimize_app.load_snapshot, logger=logger
    )
    app = optimize_app.build_optimize_app(store, logger=logger)
    client = TestClient(app)

    list_response = client.get("/api/prompts")
    assert list_response.status_code == 200
    prompts = list_response.json()
    assert len(prompts) == 1
    prompt_id = prompts[0]["id"]

    detail = client.get(f"/api/prompts/{prompt_id}").json()
    assert detail["descriptor"]["ns"] == "demo"
    assert detail["overrides"]["sections"]["intro"]["body"] == "body"

    update_response = client.post(
        f"/api/prompts/{prompt_id}/overrides",
        json={"sections": {"intro": {"body": "updated"}}},
    )
    assert update_response.status_code == 200
    assert update_response.json()["overrides"]["sections"]["intro"]["body"] == "updated"

    save_response = client.post("/api/save")
    assert save_response.status_code == 200
    saved = Snapshot.from_json(snapshot_path.read_text())
    overrides_slice = saved.slices[optimize_app.PromptOverride]
    assert overrides_slice[0].sections[("intro",)].body == "updated"
    assert saved.slices[PromptRendered][0].rendered_prompt == snapshot.slices[PromptRendered][0].rendered_prompt


def test_reset_reload_restores_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _, original_payload = _build_snapshot(snapshot_path, section_body="initial")
    logger = optimize_app.get_logger("test.reset")
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path, loader=optimize_app.load_snapshot, logger=logger
    )
    app = optimize_app.build_optimize_app(store, logger=logger)
    client = TestClient(app)

    client.post("/api/prompts/demo:example:0/overrides", json={"sections": {"intro": {"body": "changed"}}})
    snapshot_path.write_text(original_payload)

    reset_response = client.post("/api/reset")
    assert reset_response.status_code == 200
    reset_prompts = reset_response.json()
    assert reset_prompts[0]["overrides"]["sections"]["intro"]["body"] == "initial"

    assert Snapshot.from_json(snapshot_path.read_text()).slices[PromptRendered][0].rendered_prompt == "hello"


def test_store_seeds_overrides_when_missing_slice(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "no_overrides.json"
    seeded = _build_snapshot_without_overrides(snapshot_path)
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path, loader=lambda _: seeded, logger=optimize_app.get_logger("seed")
    )

    prompts = store.list_prompts()
    assert prompts[0]["overrides"]["sections"]["intro"]["body"] == ""

    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides("unknown", {})


def test_store_skips_prompts_without_descriptor(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "missing_descriptor.json"
    descriptorless = PromptRendered(
        prompt_ns="demo",
        prompt_key="none",
        prompt_name="None",
        adapter="openai",
        session_id=None,
        render_inputs=(),
        rendered_prompt="hello",
        descriptor=None,
        created_at=datetime.now(UTC),
    )
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={PromptRendered: (descriptorless,)},
    )
    payload = snapshot.to_json()
    snapshot_path.write_text(payload)
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path,
        loader=lambda path: optimize_app.OptimizableSnapshot(
            path=path, snapshot=snapshot, raw_text=payload
        ),
        logger=optimize_app.get_logger("skip"),
    )

    assert store.list_prompts() == []
    with pytest.raises(optimize_app.HTTPException):
        store.get_prompt("demo:none:0")


def test_update_overrides_validation_errors(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "validation.json"
    tool_hash = hash_text("tool-contract")
    descriptor = optimize_app.PromptDescriptor(
        ns="demo",
        key="example",
        sections=[SectionDescriptor(("intro",), hash_text("section"), 0)],
        tools=[ToolDescriptor(("intro",), "tool", tool_hash)],
    )
    prompt = PromptRendered(
        prompt_ns="demo",
        prompt_key="example",
        prompt_name="Demo",
        adapter="openai",
        session_id=None,
        render_inputs=(),
        rendered_prompt="hello",
        descriptor=descriptor,
        created_at=datetime.now(UTC),
    )
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={PromptRendered: (prompt,)},
    )
    payload = snapshot.to_json()
    snapshot_path.write_text(payload)
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path,
        loader=lambda path: optimize_app.OptimizableSnapshot(
            path=path, snapshot=snapshot, raw_text=payload
        ),
        logger=optimize_app.get_logger("validation"),
    )

    prompt_id = store.list_prompts()[0]["id"]
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"unknown": "field"})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"sections": []})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"sections": {1: {"body": "x"}}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"sections": {"missing": {"body": "x"}}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"sections": {"intro": "bad"}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"sections": {"intro": {"body": 1}}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"tools": []})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"tools": {1: {"description": "x"}}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"tools": {"missing": {"description": "x"}}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"tools": {"tool": "bad"}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"tools": {"tool": {"description": 1}}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(prompt_id, {"tools": {"tool": {"param_descriptions": "bad"}}})
    with pytest.raises(optimize_app.HTTPException):
        store.update_overrides(
            prompt_id,
            {"tools": {"tool": {"param_descriptions": {"field": 1}}}},
        )


def test_update_overrides_allows_tool_params(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "tool_update.json"
    tool_hash = hash_text("tool-contract")
    descriptor = optimize_app.PromptDescriptor(
        ns="demo",
        key="example",
        sections=[SectionDescriptor(("intro",), hash_text("section"), 0)],
        tools=[ToolDescriptor(("intro",), "tool", tool_hash)],
    )
    prompt = PromptRendered(
        prompt_ns="demo",
        prompt_key="example",
        prompt_name="Demo",
        adapter="openai",
        session_id=None,
        render_inputs=(),
        rendered_prompt="hello",
        descriptor=descriptor,
        created_at=datetime.now(UTC),
    )
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={PromptRendered: (prompt,)},
    )
    payload = snapshot.to_json()
    snapshot_path.write_text(payload)
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path,
        loader=lambda path: optimize_app.OptimizableSnapshot(
            path=path, snapshot=snapshot, raw_text=payload
        ),
        logger=optimize_app.get_logger("tool-update"),
    )

    prompt_id = store.list_prompts()[0]["id"]
    updated = store.update_overrides(
        prompt_id,
        {"tools": {"tool": {"description": "patched", "param_descriptions": {"field": "desc"}}}},
    )
    assert updated["tools"]["tool"]["param_descriptions"]["field"] == "desc"


def test_save_handles_write_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot, _ = _build_snapshot(snapshot_path)
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path, loader=optimize_app.load_snapshot, logger=optimize_app.get_logger("save")
    )

    class BrokenPath(Path):
        _flavour = type(Path())._flavour  # pragma: no cover - Path subclassing boilerplate

    def broken_write(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise OSError("unwritable")

    monkeypatch.setattr(store.snapshot_path.__class__, "write_text", broken_write)

    with pytest.raises(optimize_app.HTTPException):
        store.save()


def test_snapshot_skips_descriptorless_entries(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    descriptor = optimize_app.PromptDescriptor(
        ns="demo",
        key="example",
        sections=[SectionDescriptor(("intro",), hash_text("section"), 0)],
        tools=[],
    )
    prompt = PromptRendered(
        prompt_ns="demo",
        prompt_key="example",
        prompt_name="Demo",
        adapter="openai",
        session_id=None,
        render_inputs=(),
        rendered_prompt="hello",
        descriptor=descriptor,
        created_at=datetime.now(UTC),
    )
    descriptorless = PromptRendered(
        prompt_ns="demo",
        prompt_key="none",
        prompt_name="None",
        adapter="openai",
        session_id=None,
        render_inputs=(),
        rendered_prompt="other",
        descriptor=None,
        created_at=datetime.now(UTC),
    )
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={PromptRendered: (descriptorless, prompt)},
        tags={"session_id": "demo"},
    )
    payload = snapshot.to_json()
    snapshot_path.write_text(payload)
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path,
        loader=lambda path: optimize_app.OptimizableSnapshot(
            path=path, snapshot=snapshot, raw_text=payload
        ),
        logger=optimize_app.get_logger("snapshot"),
    )

    updated = store._store.snapshot(snapshot)
    overrides = updated.slices[optimize_app.PromptOverride]
    assert len(overrides) == 1


def test_normalize_override_handles_empty_payload(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "empty_override.json"
    descriptor = optimize_app.PromptDescriptor(
        ns="demo",
        key="example",
        sections=[SectionDescriptor(("intro",), hash_text("section"), 0)],
        tools=[],
    )
    prompt = PromptRendered(
        prompt_ns="demo",
        prompt_key="example",
        prompt_name="Demo",
        adapter="openai",
        session_id=None,
        render_inputs=(),
        rendered_prompt="hello",
        descriptor=descriptor,
        created_at=datetime.now(UTC),
    )
    override = optimize_app.PromptOverride(
        ns="demo",
        prompt_key="example",
        tag="latest",
        sections={},
        tool_overrides={},
    )
    snapshot = Snapshot(
        created_at=datetime.now(UTC),
        slices={PromptRendered: (prompt,), optimize_app.PromptOverride: (override,)},
        tags={"session_id": "demo"},
    )
    payload = snapshot.to_json()
    snapshot_path.write_text(payload)
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path,
        loader=lambda path: optimize_app.OptimizableSnapshot(
            path=path, snapshot=snapshot, raw_text=payload
        ),
        logger=optimize_app.get_logger("normalize"),
    )

    prompts = store.list_prompts()
    assert prompts[0]["overrides"]["sections"] == {}


def test_build_app_serves_index(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.jsonl"
    _build_snapshot(snapshot_path)
    store = optimize_app.OptimizedSnapshotStore(
        snapshot_path, loader=optimize_app.load_snapshot, logger=optimize_app.get_logger("index")
    )
    app = optimize_app.build_optimize_app(store, logger=optimize_app.get_logger("index"))
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "wink optimize" in response.text


def test_run_optimize_server_handles_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config: object) -> None:
            captured["config"] = config

        def run(self) -> None:
            captured["ran"] = True

    class FakeConfig:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["config_args"] = {"args": args, "kwargs": kwargs}

    class FakeTimer:
        def __init__(self, interval: float, func: object, args: tuple[object, ...]) -> None:
            captured["timer"] = {"interval": interval, "func": func, "args": args}

        def start(self) -> "FakeTimer":
            captured["timer_started"] = True
            return self

    monkeypatch.setattr(optimize_app.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(optimize_app.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(optimize_app.threading, "Timer", FakeTimer)

    result = optimize_app.run_optimize_server(
        object(), host="127.0.0.1", port=1, open_browser=True, logger=optimize_app.get_logger("server")
    )

    assert result == 0
    assert captured["ran"] is True
    assert captured["timer_started"] is True

    def raising_run(_: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(FakeServer, "run", raising_run)
    failure = optimize_app.run_optimize_server(
        object(), host="127.0.0.1", port=1, open_browser=False, logger=optimize_app.get_logger("server")
    )
    assert failure == 3


def test_open_browser_logs_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: dict[str, object] = {}

    class FakeLogger:
        def warning(self, message: str, *, event: str, context: object | None = None) -> None:
            messages["message"] = message
            messages["event"] = event
            messages["context"] = context

    def raising_open(url: str) -> None:
        raise RuntimeError(f"bad {url}")

    monkeypatch.setattr(optimize_app.webbrowser, "open", raising_open)

    optimize_app._open_browser("http://localhost", FakeLogger())

    assert messages["event"] == "optimize.server.browser"
