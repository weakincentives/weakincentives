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

from datetime import UTC, datetime
from pathlib import Path
from fastapi.testclient import TestClient

from weakincentives.cli import optimize_app
from weakincentives.runtime.events import PromptRendered
from weakincentives.runtime.session.snapshots import Snapshot


def _build_snapshot(tmp_path: Path) -> tuple[Path, str]:
    descriptor = optimize_app.PromptDescriptor(ns="demo", key="hello", sections=[], tools=[], chapters=[])
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
    store = optimize_app.OptimizeStore(loaded, logger=_FakeLogger())

    prompts = store.prompts
    assert len(prompts) == 1
    assert prompts[0].prompt_id == prompt_id
    assert prompts[0].overrides == {"model": "gpt-4"}


def test_optimize_app_routes(tmp_path: Path) -> None:
    snapshot_path, prompt_id = _build_snapshot(tmp_path)
    loaded = optimize_app.load_snapshot(snapshot_path)
    store = optimize_app.OptimizeStore(loaded, logger=_FakeLogger())
    app = optimize_app.build_optimize_app(store, logger=_FakeLogger())
    client = TestClient(app)

    response = client.get("/api/prompts")
    assert response.status_code == 200
    prompts = response.json()
    assert prompts[0]["prompt_id"] == prompt_id

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
    saved_overrides = reloaded.slices[optimize_app.PromptOverrideSnapshotEntry][0]
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


class _FakeLogger:
    def info(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - test helper
        return None

    def warning(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - test helper
        return None

    def exception(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - test helper
        return None
