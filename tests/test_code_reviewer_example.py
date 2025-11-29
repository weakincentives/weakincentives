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

"""Regression tests for the interactive code reviewer example."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from pytest import CaptureFixture

import code_reviewer_example as reviewer_example
from code_reviewer_example import (
    CodeReviewApp,
    ReviewResponse,
    ReviewTurnParams,
    build_task_prompt,
    initialize_code_reviewer_runtime,
)
from tests.helpers.adapters import UNIT_TEST_ADAPTER_NAME
from weakincentives.adapters.core import PromptResponse
from weakincentives.adapters.core import (
    OptimizationResult,
    OptimizationScope,
    ProviderAdapter,
)
from weakincentives.debug import dump_session
from weakincentives.prompt import Prompt, SupportsDataclass
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    PromptDescriptor,
)
from weakincentives.runtime.events import InProcessEventBus, PromptRendered
from weakincentives.runtime.session import Session
from weakincentives.tools.digests import (
    latest_workspace_digest,
    set_workspace_digest,
)


class _RepositoryOptimizationAdapter:
    """Stub adapter that emits canned repository instructions."""

    def __init__(self, instructions: str) -> None:
        self.instructions = instructions
        self.calls: list[str] = []
        self.optimization_sessions: list[Session] = []

    def evaluate(
        self,
        prompt: Prompt[SupportsDataclass],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: InProcessEventBus | None = None,
        session: Session | None = None,
        deadline: object | None = None,
        overrides_store: LocalPromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[Any]:
        del params, parse_output, bus, session, deadline, overrides_store, overrides_tag
        self.calls.append(prompt.key)
        return PromptResponse(
            prompt_name=prompt.name or prompt.key,
            text="",
            output=None,
        )

    def optimize(
        self,
        prompt: Prompt[SupportsDataclass],
        *,
        store_scope: OptimizationScope = OptimizationScope.SESSION,
        overrides_store: LocalPromptOverridesStore | None = None,
        overrides_tag: str | None = None,
        session: Session | None = None,
        optimization_session: Session | None = None,
    ) -> OptimizationResult:
        assert session is not None
        assert optimization_session is not None
        self.optimization_sessions.append(optimization_session)
        assert optimization_session is not session
        optimization_event = PromptRendered(
            prompt_ns=prompt.ns,
            prompt_key=prompt.key,
            prompt_name=prompt.name,
            adapter=UNIT_TEST_ADAPTER_NAME,
            session_id=optimization_session.session_id,
            render_inputs=(),
            rendered_prompt="<optimize prompt>",
            created_at=datetime.now(UTC),
            event_id=uuid4(),
        )
        publish_result = optimization_session.event_bus.publish(optimization_event)
        assert publish_result.handled_count >= 1
        self.calls.append(f"optimize:{prompt.key}")
        set_workspace_digest(session, "workspace-digest", self.instructions)
        if overrides_store is not None and overrides_tag is not None:
            digest_node = next(
                node
                for node in prompt.sections
                if node.section.key == "workspace-digest"
            )
            overrides_store.set_section_override(
                prompt,
                tag=overrides_tag,
                path=digest_node.path,
                body=self.instructions,
            )
        response = PromptResponse(
            prompt_name=prompt.name or prompt.key,
            text=self.instructions,
            output=None,
        )
        return OptimizationResult(
            response=response,
            digest=self.instructions,
            scope=store_scope,
            section_key="workspace-digest",
        )


def test_prompt_render_reducer_prints_full_prompt(
    capsys: CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    (
        _prompt,
        session,
        bus,
        _store,
        _tag,
    ) = initialize_code_reviewer_runtime(
        overrides_store=overrides_store,
        override_tag="prompt-log",
    )

    event = PromptRendered(
        prompt_ns="example",
        prompt_key="sunfish",
        prompt_name="sunfish_code_review_agent",
        adapter=UNIT_TEST_ADAPTER_NAME,
        session_id=session.session_id,
        render_inputs=(),
        rendered_prompt="<prompt body>",
        created_at=datetime.now(UTC),
        event_id=uuid4(),
    )

    publish_result = bus.publish(event)
    assert publish_result.handled_count >= 1

    captured = capsys.readouterr()
    assert "Rendered prompt" in captured.out
    assert "<prompt body>" in captured.out

    stored_events = session.select_all(PromptRendered)
    assert stored_events
    assert stored_events[-1].rendered_prompt == "<prompt body>"


def test_workspace_digest_section_empty_by_default() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    prompt = build_task_prompt(session=session)

    rendered = prompt.render(ReviewTurnParams(request="demo request"))

    assert "## 2. Workspace Digest" in rendered.text
    post_section = rendered.text.split("## 2. Workspace Digest", 1)[1]
    section_body = post_section.split("\n## ", 1)[0]
    assert "Workspace digest unavailable" in section_body


def test_workspace_digest_override_applied_when_no_session_digest(
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    bus = InProcessEventBus()
    session = Session(bus=bus)
    prompt = build_task_prompt(session=session)

    digest_node = next(
        node for node in prompt.sections if node.section.key == "workspace-digest"
    )
    overrides_store.set_section_override(
        prompt,
        tag="seed",
        path=digest_node.path,
        body="- Override digest",
    )

    rendered = prompt.render(
        ReviewTurnParams(request="demo request"),
        overrides_store=overrides_store,
        tag="seed",
    )
    assert "Override digest" in rendered.text


def test_workspace_digest_prefers_session_snapshot_over_override(
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    session = Session()
    prompt = build_task_prompt(session=session)
    digest_node = next(
        node for node in prompt.sections if node.section.key == "workspace-digest"
    )

    overrides_store.set_section_override(
        prompt,
        tag="seed",
        path=digest_node.path,
        body="- Override digest",
    )
    set_workspace_digest(session, "workspace-digest", "- Session digest")

    rendered = prompt.render(
        ReviewTurnParams(request="demo request"),
        overrides_store=overrides_store,
        tag="seed",
    )
    assert "Session digest" in rendered.text


def test_optimize_command_persists_override(tmp_path: Path) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    adapter = _RepositoryOptimizationAdapter("- Repo instructions from stub")
    app = CodeReviewApp(
        cast(ProviderAdapter[ReviewResponse], adapter),
        overrides_store=overrides_store,
    )

    assert app.override_tag == "latest"

    app._handle_optimize_command()

    descriptor = PromptDescriptor.from_prompt(app.prompt)
    override = overrides_store.resolve(descriptor=descriptor, tag=app.override_tag)
    assert override is not None
    placeholder_body = override.sections["workspace-digest",].body
    assert "Workspace digest unavailable" in placeholder_body
    session_digest = latest_workspace_digest(app.session, "workspace-digest")
    assert session_digest is not None
    assert session_digest.body == "- Repo instructions from stub"


@pytest.fixture(autouse=True)
def _redirect_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    snapshot_dir = tmp_path_factory.mktemp("snapshots")
    monkeypatch.setattr(reviewer_example, "SNAPSHOT_DIR", snapshot_dir, raising=False)
    return snapshot_dir


def test_dump_session_logs_success(
    caplog: pytest.LogCaptureFixture,
    _redirect_snapshots: Path,
) -> None:
    caplog.set_level(logging.INFO, logger="weakincentives.debug")
    session = Session()
    set_workspace_digest(session, "workspace-digest", "body")

    snapshot_path = dump_session(session, _redirect_snapshots)

    assert snapshot_path == _redirect_snapshots / f"{session.session_id}.jsonl"
    assert snapshot_path is not None and snapshot_path.exists()
    content = snapshot_path.read_text().splitlines()
    assert len(content) == 1
    record = next(
        rec
        for rec in caplog.records
        if getattr(rec, "session_id", None) == str(session.session_id)
    )
    assert record.levelno == logging.INFO
    assert "Session snapshots persisted" in record.getMessage()
    assert record.snapshot_path == str(snapshot_path)
    assert record.snapshot_count == 1


def test_dump_session_skips_empty_session(
    caplog: pytest.LogCaptureFixture,
    _redirect_snapshots: Path,
) -> None:
    caplog.set_level(logging.INFO, logger="weakincentives.debug")
    session = Session()

    snapshot_path = dump_session(session, _redirect_snapshots)

    assert snapshot_path is None
    assert not any(_redirect_snapshots.iterdir())
    record = next(
        rec
        for rec in caplog.records
        if getattr(rec, "session_id", None) == str(session.session_id)
    )
    assert "skipped; no slices" in record.getMessage()
