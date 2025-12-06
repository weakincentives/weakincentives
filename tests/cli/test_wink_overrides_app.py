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

"""Tests for the wink overrides FastAPI application."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from weakincentives.cli import overrides_app
from weakincentives.cli.debug_app import (
    LoadedSnapshot,
    SliceSummary,
    SnapshotMeta,
)
from weakincentives.prompt.overrides import (
    HexDigest,
    PromptDescriptor,
)
from weakincentives.runtime.session.snapshots import SnapshotSlicePayload

_MOCK_HASH_1 = "a" * 64  # Valid 64-char hex string
_MOCK_HASH_2 = "b" * 64
_MOCK_HASH_3 = "c" * 64


def _make_seed_file_content(
    ns: str = "test-ns",
    key: str = "test-key",
    tag: str = "latest",
    sections: dict[str, dict[str, str]] | None = None,
    tools: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Create a properly formatted override file content."""
    return json.dumps(
        {
            "version": 1,
            "ns": ns,
            "prompt_key": key,
            "tag": tag,
            "sections": sections or {},
            "tools": tools or {},
        }
    )


def _make_prompt_rendered_event(
    *,
    ns: str = "test-ns",
    key: str = "test-key",
    name: str | None = "Test Prompt",
    rendered_text: str = "# Hello\n\nThis is a test prompt.",
    sections: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    created_at: datetime | None = None,
    event_id: UUID | None = None,
) -> dict[str, Any]:
    """Create a mock PromptRendered event dict."""
    if sections is None:
        sections = [
            {
                "path": ["root", "instructions"],
                "content_hash": _MOCK_HASH_1,
                "number": "1",
            },
            {"path": ["root", "examples"], "content_hash": _MOCK_HASH_2, "number": "2"},
        ]
    if tools is None:
        tools = [
            {"path": ["root"], "name": "search", "contract_hash": _MOCK_HASH_3},
        ]

    return {
        "prompt_ns": ns,
        "prompt_key": key,
        "prompt_name": name,
        "adapter": "openai",
        "session_id": str(uuid4()),
        "render_inputs": [],
        "rendered_prompt": rendered_text,
        "created_at": (created_at or datetime.now(UTC)).isoformat(),
        "descriptor": {
            "ns": ns,
            "key": key,
            "sections": sections,
            "tools": tools,
        },
        "event_id": str(event_id or uuid4()),
    }


def _make_loaded_snapshot(
    events: list[dict[str, Any]],
    path: Path,
) -> LoadedSnapshot:
    """Create a LoadedSnapshot with PromptRendered events."""
    slice_type = "weakincentives.runtime.events:PromptRendered"
    slices = {
        slice_type: SnapshotSlicePayload(
            slice_type=slice_type,
            item_type=slice_type,
            items=tuple(events),
        )
    }
    meta = SnapshotMeta(
        version="1",
        created_at=datetime.now(UTC).isoformat(),
        slices=(
            SliceSummary(
                slice_type=slice_type,
                item_type=slice_type,
                count=len(events),
            ),
        ),
        tags={"session_id": "test-session"},
        path=str(path),
        line_number=1,
        session_id="test-session",
        validation_error=None,
    )
    return LoadedSnapshot(
        meta=meta,
        slices=slices,
        raw_payload={},
        raw_text="{}",
        path=path,
    )


def _make_store(
    tmp_path: Path,
    events: list[dict[str, Any]] | None = None,
    tag: str = "latest",
    store_root: Path | None = None,
) -> overrides_app.OverridesStore:
    """Create an OverridesStore with mock events."""
    if events is None:
        events = [_make_prompt_rendered_event()]

    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot_path.write_text("{}")

    def mock_loader(path: Path) -> tuple[LoadedSnapshot, ...]:
        return (_make_loaded_snapshot(events, path),)

    log = overrides_app.get_logger("test.overrides")
    return overrides_app.OverridesStore(
        snapshot_path,
        tag=tag,
        store_root=store_root or tmp_path,
        loader=mock_loader,
        log=log,
    )


# --- Data Class Tests ---


def test_extracted_prompt_frozen() -> None:
    prompt = overrides_app.ExtractedPrompt(
        ns="ns",
        key="key",
        name="Test",
        descriptor=PromptDescriptor(ns="ns", key="key", sections=[], tools=[]),
        rendered_text="hello",
        created_at=datetime.now(UTC),
        event_id=uuid4(),
    )
    assert prompt.ns == "ns"
    with pytest.raises(AttributeError):
        prompt.ns = "other"


def test_section_state_frozen() -> None:
    state = overrides_app.SectionState(
        path=("root", "section"),
        number="1",
        original_hash=HexDigest(_MOCK_HASH_1),
        current_body=None,
        is_overridden=False,
        is_stale=False,
    )
    assert state.path == ("root", "section")


def test_tool_state_frozen() -> None:
    state = overrides_app.ToolState(
        name="tool",
        path=("root",),
        original_contract_hash=HexDigest(_MOCK_HASH_1),
        current_description=None,
        current_param_descriptions={},
        is_overridden=False,
        is_stale=False,
    )
    assert state.name == "tool"


# --- Error Type Tests ---


def test_error_hierarchy() -> None:
    assert issubclass(overrides_app.OverridesEditorError, RuntimeError)
    assert issubclass(
        overrides_app.PromptNotFoundError, overrides_app.OverridesEditorError
    )
    assert issubclass(
        overrides_app.SectionNotFoundError, overrides_app.OverridesEditorError
    )
    assert issubclass(
        overrides_app.ToolNotFoundError, overrides_app.OverridesEditorError
    )
    assert issubclass(
        overrides_app.PromptNotSeededError, overrides_app.OverridesEditorError
    )
    assert issubclass(
        overrides_app.HashMismatchError, overrides_app.OverridesEditorError
    )


# --- OverridesStore Tests ---


def test_overrides_store_extracts_prompts(tmp_path: Path) -> None:
    events = [
        _make_prompt_rendered_event(ns="ns1", key="key1"),
        _make_prompt_rendered_event(ns="ns2", key="key2"),
    ]
    store = _make_store(tmp_path, events)

    assert len(store.prompts) == 2
    assert store.tag == "latest"


def test_overrides_store_deduplicates_prompts(tmp_path: Path) -> None:
    old_time = datetime(2024, 1, 1, tzinfo=UTC)
    new_time = datetime(2024, 12, 1, tzinfo=UTC)

    events = [
        _make_prompt_rendered_event(ns="ns", key="key", created_at=old_time),
        _make_prompt_rendered_event(
            ns="ns", key="key", created_at=new_time, name="Newer"
        ),
    ]
    store = _make_store(tmp_path, events)

    assert len(store.prompts) == 1
    assert store.prompts[0].name == "Newer"


def test_overrides_store_get_prompt_state(tmp_path: Path) -> None:
    events = [_make_prompt_rendered_event()]
    store = _make_store(tmp_path, events)

    state = store.get_prompt_state("test-ns", "test-key")
    assert state is not None
    assert state.prompt.ns == "test-ns"
    assert len(state.sections) == 2
    assert len(state.tools) == 1
    assert state.is_seeded is False

    none_state = store.get_prompt_state("missing", "key")
    assert none_state is None


def test_overrides_store_update_section_requires_seed(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(overrides_app.PromptNotSeededError):
        store.update_section(
            "test-ns", "test-key", ("root", "instructions"), "new body"
        )


def test_overrides_store_update_section_missing_prompt(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    with pytest.raises(overrides_app.PromptNotFoundError):
        store.update_section("missing", "key", ("root",), "body")


def test_overrides_store_update_section_missing_section(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_path = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_path.mkdir(parents=True)
    seed_file = seed_path / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    with pytest.raises(overrides_app.SectionNotFoundError):
        store.update_section("test-ns", "test-key", ("missing", "path"), "body")


def test_overrides_store_update_section_success(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    section = store.update_section(
        "test-ns", "test-key", ("root", "instructions"), "new body"
    )

    assert section.is_overridden is True
    assert section.current_body == "new body"


def test_overrides_store_delete_section(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file with existing override
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(
        _make_seed_file_content(
            sections={
                "root/instructions": {"expected_hash": _MOCK_HASH_1, "body": "old"}
            }
        )
    )

    section = store.delete_section("test-ns", "test-key", ("root", "instructions"))

    assert section.is_overridden is False
    assert section.current_body is None


def test_overrides_store_update_tool(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    tool = store.update_tool(
        "test-ns", "test-key", "search", "New description", {"query": "The query"}
    )

    assert tool.is_overridden is True
    assert tool.current_description == "New description"
    assert tool.current_param_descriptions == {"query": "The query"}


def test_overrides_store_delete_tool(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file with existing tool override
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(
        _make_seed_file_content(
            tools={
                "search": {
                    "name": "search",
                    "expected_contract_hash": _MOCK_HASH_3,
                    "description": "old",
                    "param_descriptions": {},
                }
            }
        )
    )

    tool = store.delete_tool("test-ns", "test-key", "search")

    assert tool.is_overridden is False


def test_overrides_store_delete_prompt_overrides(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    store.delete_prompt_overrides("test-ns", "test-key")

    assert not seed_file.exists()


def test_overrides_store_reload(tmp_path: Path) -> None:
    events = [_make_prompt_rendered_event()]
    store = _make_store(tmp_path, events)

    store.reload()

    assert len(store.prompts) == 1


def test_overrides_store_handles_no_prompts(tmp_path: Path) -> None:
    store = _make_store(tmp_path, events=[])

    assert len(store.prompts) == 0


def test_overrides_store_tool_not_found(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    with pytest.raises(overrides_app.ToolNotFoundError):
        store.update_tool("test-ns", "test-key", "missing-tool", "desc", {})


# --- API Route Tests ---


def test_api_list_prompts(tmp_path: Path) -> None:
    events = [
        _make_prompt_rendered_event(ns="ns1", key="key1", name="First"),
        _make_prompt_rendered_event(ns="ns2", key="key2", name="Second"),
    ]
    store = _make_store(tmp_path, events)
    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.get("/api/prompts")
    assert response.status_code == 200

    prompts = response.json()
    assert len(prompts) == 2
    assert any(p["name"] == "First" for p in prompts)
    assert any(p["name"] == "Second" for p in prompts)


def test_api_get_prompt(tmp_path: Path) -> None:
    events = [_make_prompt_rendered_event()]
    store = _make_store(tmp_path, events)
    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.get(f"/api/prompts/{quote('test-ns')}/test-key")
    assert response.status_code == 200

    detail = response.json()
    assert detail["ns"] == "test-ns"
    assert detail["key"] == "test-key"
    assert detail["name"] == "Test Prompt"
    assert len(detail["sections"]) == 2
    assert len(detail["tools"]) == 1
    assert "rendered_prompt" in detail
    assert "html" in detail["rendered_prompt"]


def test_api_get_prompt_not_found(tmp_path: Path) -> None:
    store = _make_store(tmp_path, events=[_make_prompt_rendered_event()])
    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.get("/api/prompts/missing/key")
    assert response.status_code == 404


def test_api_update_section(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.put(
        f"/api/prompts/{quote('test-ns')}/test-key/sections/{'root/instructions'}",
        json={"body": "new content"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["section"]["is_overridden"] is True


def test_api_update_section_missing_body(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file to allow update_section to pass seeding check
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.put(
        f"/api/prompts/{quote('test-ns')}/test-key/sections/{'root/instructions'}",
        json={},
    )

    assert response.status_code == 400


def test_api_delete_section(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file with override
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(
        _make_seed_file_content(
            sections={
                "root/instructions": {"expected_hash": _MOCK_HASH_1, "body": "old"}
            }
        )
    )

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.delete(
        f"/api/prompts/{quote('test-ns')}/test-key/sections/{'root/instructions'}"
    )

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["section"]["is_overridden"] is False


def test_api_update_tool(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.put(
        f"/api/prompts/{quote('test-ns')}/test-key/tools/search",
        json={"description": "New desc", "param_descriptions": {"q": "query"}},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["tool"]["is_overridden"] is True


def test_api_delete_tool(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file with tool override
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(
        _make_seed_file_content(
            tools={
                "search": {
                    "name": "search",
                    "expected_contract_hash": _MOCK_HASH_3,
                    "description": "old",
                    "param_descriptions": {},
                }
            }
        )
    )

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.delete(f"/api/prompts/{quote('test-ns')}/test-key/tools/search")

    assert response.status_code == 200
    result = response.json()
    assert result["success"] is True
    assert result["tool"]["is_overridden"] is False


def test_api_delete_prompt(tmp_path: Path) -> None:
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.delete(f"/api/prompts/{quote('test-ns')}/test-key")

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_api_get_config(tmp_path: Path) -> None:
    store = _make_store(tmp_path, tag="test-tag")
    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.get("/api/config")
    assert response.status_code == 200

    config = response.json()
    assert config["tag"] == "test-tag"
    assert "snapshot_path" in config


def test_api_reload(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.post("/api/reload")
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_api_index_returns_html(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "<!doctype html>" in response.text.lower()


# --- Server Runner Tests ---


def test_run_overrides_server_opens_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeTimer:
        def __init__(
            self,
            interval: float,
            function: Callable[..., None],
            args: tuple[object, ...] | None = None,
            kwargs: dict[str, object] | None = None,
        ) -> None:
            calls["timer_interval"] = interval
            self._function = function
            self._args: tuple[object, ...] = tuple(args or ())
            self._kwargs: dict[str, object] = dict(kwargs or {})

        def start(self) -> None:
            calls["timer_started"] = True
            self._function(*self._args, **self._kwargs)

    monkeypatch.setattr(overrides_app.threading, "Timer", FakeTimer)

    def fake_webbrowser_open(url: str) -> bool:
        calls["browser_url"] = url
        return True

    monkeypatch.setattr(overrides_app.webbrowser, "open", fake_webbrowser_open)

    class FakeConfig:
        def __init__(
            self, app: object, host: str, port: int, log_config: object
        ) -> None:
            calls["config"] = {
                "app": app,
                "host": host,
                "port": port,
                "log_config": log_config,
            }

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            calls["server_config"] = config

        @staticmethod
        def run() -> None:
            calls["run_called"] = True

    monkeypatch.setattr(overrides_app.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(overrides_app.uvicorn, "Server", FakeServer)

    logger = overrides_app.get_logger("test.run")
    infos: list[dict[str, object]] = []

    def capture_info(message: str, *, event: str, context: dict[str, object]) -> None:
        infos.append({"message": message, "event": event, "context": context})

    monkeypatch.setattr(logger, "info", capture_info)

    app = FastAPI()
    app.state.overrides_store = type("Store", (), {"snapshot_path": Path("/test")})()

    exit_code = overrides_app.run_overrides_server(
        app=app,
        host="0.0.0.0",
        port=8001,
        open_browser=True,
        log=logger,
    )

    assert exit_code == 0
    assert calls["timer_started"] is True
    assert calls["browser_url"] == "http://0.0.0.0:8001/"
    assert calls["run_called"] is True


# --- CLI Integration Tests ---


class _FakeLogger:
    """Fake logger for CLI tests."""

    def __init__(self) -> None:
        self.logs: list[tuple[str, str]] = []

    def info(self, message: str, *, event: str, context: object | None = None) -> None:
        self.logs.append((event, message))

    def warning(
        self, message: str, *, event: str, context: object | None = None
    ) -> None:
        self.logs.append((event, message))

    def error(self, message: str, *, event: str, context: object | None = None) -> None:
        self.logs.append((event, message))

    def exception(
        self, message: str, *, event: str, context: object | None = None
    ) -> None:
        self.logs.append((event, message))


def test_wink_overrides_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from weakincentives.cli import wink

    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot_path.write_text("{}")
    events = [_make_prompt_rendered_event()]

    def mock_loader(path: Path) -> tuple[LoadedSnapshot, ...]:
        return (_make_loaded_snapshot(events, path),)

    calls: dict[str, Any] = {}
    fake_logger = _FakeLogger()

    def fake_configure_logging(*, level: object, json_mode: object) -> None:
        calls["configure"] = {"level": level, "json_mode": json_mode}

    def fake_get_logger(_: str) -> _FakeLogger:
        return fake_logger

    def fake_build_app(store: object, *, log: object) -> str:
        return "app"

    def fake_run_server(
        app: object, *, host: str, port: int, open_browser: bool, log: object
    ) -> int:
        calls["run"] = {"host": host, "port": port, "open_browser": open_browser}
        return 0

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.overrides_app, "load_snapshot", mock_loader)
    monkeypatch.setattr(wink.overrides_app, "build_overrides_app", fake_build_app)
    monkeypatch.setattr(wink.overrides_app, "run_overrides_server", fake_run_server)

    exit_code = wink.main(
        [
            "overrides",
            str(snapshot_path),
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--tag",
            "test-tag",
            "--no-open-browser",
        ]
    )

    assert exit_code == 0
    assert calls["run"]["host"] == "127.0.0.1"
    assert calls["run"]["port"] == 9001
    assert calls["run"]["open_browser"] is False


def test_wink_overrides_no_prompts_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from weakincentives.cli import wink

    snapshot_path = tmp_path / "snapshot.jsonl"
    snapshot_path.write_text("{}")

    def mock_loader(path: Path) -> tuple[LoadedSnapshot, ...]:
        return (_make_loaded_snapshot([], path),)

    calls: dict[str, Any] = {}

    class ErrorTrackingLogger(_FakeLogger):
        def error(
            self, message: str, *, event: str, context: object | None = None
        ) -> None:
            super().error(message, event=event, context=context)
            calls["error"] = event

    def fake_configure_logging(*, level: object, json_mode: object) -> None:
        pass

    def fake_get_logger(_: str) -> ErrorTrackingLogger:
        return ErrorTrackingLogger()

    monkeypatch.setattr(wink, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(wink, "get_logger", fake_get_logger)
    monkeypatch.setattr(wink.overrides_app, "load_snapshot", mock_loader)

    exit_code = wink.main(["overrides", str(snapshot_path)])

    assert exit_code == 4
    assert calls["error"] == "wink.overrides.no_prompts"


# --- Edge Case Tests ---


def test_parse_datetime_with_datetime_input() -> None:
    """Test _parse_datetime when input is already a datetime."""
    from weakincentives.cli.overrides_app import _parse_datetime

    dt = datetime.now(UTC)
    assert _parse_datetime(dt) == dt


def test_parse_datetime_with_invalid_input() -> None:
    """Test _parse_datetime with invalid input type."""
    from weakincentives.cli.overrides_app import _parse_datetime

    with pytest.raises(TypeError, match="Cannot parse datetime"):
        _parse_datetime(12345)


def test_parse_uuid_with_uuid_input() -> None:
    """Test _parse_uuid when input is already a UUID."""
    from weakincentives.cli.overrides_app import _parse_uuid

    uid = uuid4()
    assert _parse_uuid(uid) == uid


def test_parse_uuid_with_invalid_input() -> None:
    """Test _parse_uuid with invalid input type."""
    from weakincentives.cli.overrides_app import _parse_uuid

    with pytest.raises(TypeError, match="Cannot parse UUID"):
        _parse_uuid(12345)


def test_extraction_handles_invalid_event(tmp_path: Path) -> None:
    """Test that extraction handles malformed events gracefully."""
    # Create event with invalid descriptor
    invalid_events = [{"prompt_ns": "ns", "prompt_key": "key", "descriptor": None}]

    slice_type = "weakincentives.runtime.events:PromptRendered"
    slices = {
        slice_type: SnapshotSlicePayload(
            slice_type=slice_type,
            item_type=slice_type,
            items=tuple(invalid_events),
        )
    }
    meta = SnapshotMeta(
        version="1",
        created_at=datetime.now(UTC).isoformat(),
        slices=(
            SliceSummary(
                slice_type=slice_type,
                item_type=slice_type,
                count=1,
            ),
        ),
        tags={"session_id": "test-session"},
        path=str(tmp_path / "snapshot.jsonl"),
        line_number=1,
        session_id="test-session",
        validation_error=None,
    )
    snapshot = LoadedSnapshot(
        meta=meta,
        slices=slices,
        raw_payload={},
        raw_text="{}",
        path=tmp_path / "snapshot.jsonl",
    )

    from weakincentives.cli.overrides_app import _extract_prompts_from_snapshot

    # Should not raise, just return empty list
    result = _extract_prompts_from_snapshot(snapshot)
    assert result == []


def test_deduplicate_prompts_hash_drift(tmp_path: Path) -> None:
    """Test that hash drift warning is logged when descriptors differ."""
    from weakincentives.cli.overrides_app import (
        ExtractedPrompt,
        _deduplicate_prompts,
    )

    # Create two prompts with same ns/key but different descriptors
    desc1 = PromptDescriptor(
        ns="ns",
        key="key",
        sections=[
            overrides_app.SectionDescriptor(
                path=("root",),
                content_hash=HexDigest(_MOCK_HASH_1),
                number="1",
            )
        ],
        tools=[],
    )
    desc2 = PromptDescriptor(
        ns="ns",
        key="key",
        sections=[
            overrides_app.SectionDescriptor(
                path=("root",),
                content_hash=HexDigest(_MOCK_HASH_2),  # Different hash
                number="1",
            )
        ],
        tools=[],
    )

    earlier = datetime(2024, 1, 1, tzinfo=UTC)
    later = datetime(2024, 1, 2, tzinfo=UTC)

    prompt1 = ExtractedPrompt(
        ns="ns",
        key="key",
        name="test",
        descriptor=desc1,
        rendered_text="text1",
        created_at=earlier,
        event_id=uuid4(),
    )
    prompt2 = ExtractedPrompt(
        ns="ns",
        key="key",
        name="test",
        descriptor=desc2,
        rendered_text="text2",
        created_at=later,
        event_id=uuid4(),
    )

    # Should keep the later one despite hash drift
    result = _deduplicate_prompts([prompt1, prompt2])
    assert len(result) == 1
    assert result["ns", "key"].created_at == later


def test_api_section_not_found_error(tmp_path: Path) -> None:
    """Test that SectionNotFoundError returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    # Try to update a non-existent section
    response = client.put(
        f"/api/prompts/{quote('test-ns')}/test-key/sections/nonexistent/section",
        json={"body": "content"},
    )

    assert response.status_code == 404


def test_api_tool_not_found_error(tmp_path: Path) -> None:
    """Test that ToolNotFoundError returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    # Try to update a non-existent tool
    response = client.put(
        f"/api/prompts/{quote('test-ns')}/test-key/tools/nonexistent",
        json={"description": "desc"},
    )

    assert response.status_code == 404


def test_api_prompt_not_seeded_error(tmp_path: Path) -> None:
    """Test that PromptNotSeededError returns 400."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Don't create seed file

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    # Try to update section without seed
    response = client.put(
        f"/api/prompts/{quote('test-ns')}/test-key/sections/root/instructions",
        json={"body": "content"},
    )

    assert response.status_code == 400


def test_api_delete_section_prompt_not_found(tmp_path: Path) -> None:
    """Test delete_section with non-existent prompt returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.delete(
        f"/api/prompts/{quote('missing')}/key/sections/root/instructions"
    )

    assert response.status_code == 404


def test_api_delete_tool_prompt_not_found(tmp_path: Path) -> None:
    """Test delete_tool with non-existent prompt returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.delete(f"/api/prompts/{quote('missing')}/key/tools/some_tool")

    assert response.status_code == 404


def test_api_delete_section_not_found(tmp_path: Path) -> None:
    """Test delete_section with non-existent section returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.delete(
        f"/api/prompts/{quote('test-ns')}/test-key/sections/nonexistent/section"
    )

    assert response.status_code == 404


def test_api_delete_tool_not_found(tmp_path: Path) -> None:
    """Test delete_tool with non-existent tool returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.delete(
        f"/api/prompts/{quote('test-ns')}/test-key/tools/nonexistent"
    )

    assert response.status_code == 404


def test_api_update_tool_prompt_not_found(tmp_path: Path) -> None:
    """Test update_tool with non-existent prompt returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.put(
        f"/api/prompts/{quote('missing')}/key/tools/some_tool",
        json={"description": "desc"},
    )

    assert response.status_code == 404


def test_api_update_tool_not_seeded(tmp_path: Path) -> None:
    """Test update_tool without seed returns 400."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Don't create seed file

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.put(
        f"/api/prompts/{quote('test-ns')}/test-key/tools/search",
        json={"description": "desc"},
    )

    assert response.status_code == 400


def test_api_update_tool_invalid_param_descriptions(tmp_path: Path) -> None:
    """Test update_tool with invalid param_descriptions returns 400."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(_make_seed_file_content())

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.put(
        f"/api/prompts/{quote('test-ns')}/test-key/tools/search",
        json={"description": "desc", "param_descriptions": "invalid"},
    )

    assert response.status_code == 400


def test_extraction_with_no_prompts_slice(tmp_path: Path) -> None:
    """Test extraction returns empty list when no PromptRendered slice exists."""
    # Create snapshot with a different slice type
    slices: dict[str, SnapshotSlicePayload] = {}
    meta = SnapshotMeta(
        version="1",
        created_at=datetime.now(UTC).isoformat(),
        slices=(),
        tags={"session_id": "test-session"},
        path=str(tmp_path / "snapshot.jsonl"),
        line_number=1,
        session_id="test-session",
        validation_error=None,
    )
    snapshot = LoadedSnapshot(
        meta=meta,
        slices=slices,
        raw_payload={},
        raw_text="{}",
        path=tmp_path / "snapshot.jsonl",
    )

    from weakincentives.cli.overrides_app import _extract_prompts_from_snapshot

    result = _extract_prompts_from_snapshot(snapshot)
    assert result == []


def test_extraction_with_malformed_event_missing_key(tmp_path: Path) -> None:
    """Test extraction handles events with missing required keys."""
    # Create event missing required 'prompt_key' field
    malformed_events = [
        {
            "prompt_ns": "ns",
            # Missing 'prompt_key'
            "descriptor": {"ns": "ns", "key": "key", "sections": [], "tools": []},
            "rendered_prompt": "text",
            "created_at": datetime.now(UTC).isoformat(),
            "event_id": str(uuid4()),
        }
    ]

    slice_type = "weakincentives.runtime.events:PromptRendered"
    slices = {
        slice_type: SnapshotSlicePayload(
            slice_type=slice_type,
            item_type=slice_type,
            items=tuple(malformed_events),
        )
    }
    meta = SnapshotMeta(
        version="1",
        created_at=datetime.now(UTC).isoformat(),
        slices=(
            SliceSummary(
                slice_type=slice_type,
                item_type=slice_type,
                count=1,
            ),
        ),
        tags={"session_id": "test-session"},
        path=str(tmp_path / "snapshot.jsonl"),
        line_number=1,
        session_id="test-session",
        validation_error=None,
    )
    snapshot = LoadedSnapshot(
        meta=meta,
        slices=slices,
        raw_payload={},
        raw_text="{}",
        path=tmp_path / "snapshot.jsonl",
    )

    from weakincentives.cli.overrides_app import _extract_prompts_from_snapshot

    result = _extract_prompts_from_snapshot(snapshot)
    assert result == []


def test_get_prompt_state_with_stale_section(tmp_path: Path) -> None:
    """Test get_prompt_state detects stale sections."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file with override that has a different hash
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    # Use _MOCK_HASH_2 for the section when the actual section has _MOCK_HASH_1
    seed_file.write_text(
        _make_seed_file_content(
            sections={
                "root/instructions": {"expected_hash": _MOCK_HASH_2, "body": "old body"}
            }
        )
    )

    state = store.get_prompt_state("test-ns", "test-key")
    assert state is not None
    section = next(s for s in state.sections if s.path == ("root", "instructions"))
    assert section.is_overridden is True
    assert section.is_stale is True
    assert section.current_body == "old body"


def test_api_update_section_prompt_not_found(tmp_path: Path) -> None:
    """Test update_section with non-existent prompt returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.put(
        f"/api/prompts/{quote('missing')}/key/sections/root/instructions",
        json={"body": "content"},
    )

    assert response.status_code == 404


def test_api_delete_prompt_prompt_not_found(tmp_path: Path) -> None:
    """Test delete_prompt with non-existent prompt returns 404."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    logger = overrides_app.get_logger("test.api")
    app = overrides_app.build_overrides_app(store, log=logger)
    client = TestClient(app)

    response = client.delete(f"/api/prompts/{quote('missing')}/key")

    assert response.status_code == 404


def test_get_prompt_state_with_stale_tool(tmp_path: Path) -> None:
    """Test get_prompt_state detects stale tool overrides."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file with tool override that has a different hash
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    # Use _MOCK_HASH_2 for the tool when the actual tool has _MOCK_HASH_3
    seed_file.write_text(
        _make_seed_file_content(
            tools={
                "search": {
                    "expected_contract_hash": _MOCK_HASH_2,
                    "description": "old desc",
                    "param_descriptions": {"query": "old param"},
                }
            }
        )
    )

    state = store.get_prompt_state("test-ns", "test-key")
    assert state is not None
    tool = next(t for t in state.tools if t.name == "search")
    assert tool.is_overridden is True
    assert tool.is_stale is True
    assert tool.current_description == "old desc"
    assert tool.current_param_descriptions == {"query": "old param"}


def test_delete_tool_override_when_exists(tmp_path: Path) -> None:
    """Test deleting an existing tool override."""
    store_root = tmp_path / "overrides"
    store_root.mkdir(parents=True)
    store = _make_store(tmp_path, store_root=store_root)

    # Create seed file with tool override (matching hash)
    seed_dir = (
        store_root
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "test-ns"
        / "test-key"
    )
    seed_dir.mkdir(parents=True)
    seed_file = seed_dir / "latest.json"
    seed_file.write_text(
        _make_seed_file_content(
            sections={
                "root/instructions": {"expected_hash": _MOCK_HASH_1, "body": "new body"}
            },
            tools={
                "search": {
                    "expected_contract_hash": _MOCK_HASH_3,
                    "description": "custom desc",
                }
            },
        )
    )

    # Delete the tool override
    result = store.delete_tool("test-ns", "test-key", "search")

    # Tool should no longer be overridden
    assert result.is_overridden is False

    # Section should still be overridden
    state = store.get_prompt_state("test-ns", "test-key")
    assert state is not None
    section = next(s for s in state.sections if s.path == ("root", "instructions"))
    assert section.is_overridden is True
