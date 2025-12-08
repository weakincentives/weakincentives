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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from pytest import CaptureFixture

import code_reviewer_example as reviewer_example
from code_reviewer_example import (
    CodeReviewLoop,
    ReviewResponse,
    ReviewTurnParams,
    build_task_prompt,
)
from examples.logging import attach_logging_subscribers
from tests.helpers.adapters import UNIT_TEST_ADAPTER_NAME
from weakincentives.adapters import PromptResponse
from weakincentives.adapters.core import ProviderAdapter
from weakincentives.deadlines import Deadline
from weakincentives.debug import dump_session
from weakincentives.prompt import Prompt, SupportsDataclass
from weakincentives.prompt.overrides import LocalPromptOverridesStore
from weakincentives.runtime.events import InProcessEventBus, PromptRendered
from weakincentives.runtime.session import Session
from weakincentives.tools.digests import (
    latest_workspace_digest,
    set_workspace_digest,
)


@dataclass(slots=True, frozen=True)
class _StubDigestOutput:
    """Stub output for optimization prompt."""

    digest: str


class _RepositoryOptimizationAdapter:
    """Stub adapter that emits canned repository instructions."""

    def __init__(self, instructions: str) -> None:
        self.instructions = instructions
        self.calls: list[str] = []
        self.optimization_sessions: list[Session] = []

    def evaluate(
        self,
        prompt: Prompt[SupportsDataclass],
        *,
        bus: InProcessEventBus | None = None,
        session: Session | None = None,
        deadline: object | None = None,
        visibility_overrides: object | None = None,
        budget: object | None = None,
        budget_tracker: object | None = None,
    ) -> PromptResponse[Any]:
        del deadline, visibility_overrides, budget, budget_tracker
        self.calls.append(prompt.key)

        if "workspace-digest" in prompt.key:
            assert session is not None
            self.optimization_sessions.append(session)
            if bus is not None:
                optimization_event = PromptRendered(
                    prompt_ns=prompt.ns,
                    prompt_key=prompt.key,
                    prompt_name=prompt.name,
                    adapter=UNIT_TEST_ADAPTER_NAME,
                    session_id=session.session_id,
                    render_inputs=(),
                    rendered_prompt="<optimize prompt>",
                    created_at=datetime.now(UTC),
                    event_id=uuid4(),
                )
                bus.publish(optimization_event)
            return PromptResponse(
                prompt_name=prompt.name or prompt.key,
                text=self.instructions,
                output=_StubDigestOutput(digest=self.instructions),
            )

        return PromptResponse(
            prompt_name=prompt.name or prompt.key,
            text="",
            output=None,
        )


class _RecordingDeadlineAdapter:
    """Stub adapter that records deadlines passed to evaluate."""

    def __init__(self) -> None:
        self.deadlines: list[Deadline | None] = []

    def evaluate(
        self,
        prompt: Prompt[SupportsDataclass],
        *,
        bus: InProcessEventBus | None = None,
        session: Session | None = None,
        deadline: Deadline | None = None,
        visibility_overrides: object | None = None,
        budget: object | None = None,
        budget_tracker: object | None = None,
    ) -> PromptResponse[Any]:
        del bus, session, visibility_overrides, budget, budget_tracker
        self.deadlines.append(deadline)
        return PromptResponse(
            prompt_name=prompt.name or prompt.key,
            text="",
            output=None,
        )


def test_prompt_render_reducer_prints_full_prompt(
    capsys: CaptureFixture[str],
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    bus = InProcessEventBus()
    attach_logging_subscribers(bus)
    session = Session(bus=bus)

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
    del overrides_store  # Not needed for this test

    captured = capsys.readouterr()
    assert "Rendered prompt" in captured.out
    assert "<prompt body>" in captured.out

    stored_events = session.query(PromptRendered).all()
    assert stored_events
    assert stored_events[-1].rendered_prompt == "<prompt body>"


def test_workspace_digest_section_empty_by_default() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    template = build_task_prompt(session=session)

    rendered = Prompt(template).bind(ReviewTurnParams(request="demo request")).render()

    assert "## 3. Workspace Digest (workspace-digest)" in rendered.text
    post_section = rendered.text.split("## 3. Workspace Digest (workspace-digest)", 1)[
        1
    ]
    section_body = post_section.split("\n## ", 1)[0]
    assert "Workspace digest unavailable" in section_body


def test_workspace_digest_override_applied_when_no_session_digest(
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    bus = InProcessEventBus()
    session = Session(bus=bus)
    template = build_task_prompt(session=session)

    digest_node = next(
        node
        for node in template.sections
        if node.section.key == "workspace-digest"  # type: ignore[union-attr]
    )
    overrides_store.set_section_override(
        template,
        tag="seed",
        path=digest_node.path,
        body="- Override digest",
    )

    rendered = (
        Prompt(
            template,
            overrides_store=overrides_store,
            overrides_tag="seed",
        )
        .bind(ReviewTurnParams(request="demo request"))
        .render()
    )
    assert "Override digest" in rendered.text


def test_workspace_digest_prefers_session_snapshot_over_override(
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    session = Session()
    template = build_task_prompt(session=session)
    digest_node = next(
        node
        for node in template.sections
        if node.section.key == "workspace-digest"  # type: ignore[union-attr]
    )

    overrides_store.set_section_override(
        template,
        tag="seed",
        path=digest_node.path,
        body="- Override digest",
    )
    set_workspace_digest(session, "workspace-digest", "- Session digest")

    rendered = (
        Prompt(
            template,
            overrides_store=overrides_store,
            overrides_tag="seed",
        )
        .bind(ReviewTurnParams(request="demo request"))
        .render()
    )
    assert "Session digest" in rendered.text


def test_auto_optimization_runs_on_first_execute(tmp_path: Path) -> None:
    """Auto-optimization runs when execute() is called without existing digest."""
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    adapter = _RepositoryOptimizationAdapter("- Repo instructions from stub")
    bus = InProcessEventBus()
    attach_logging_subscribers(bus)
    loop = CodeReviewLoop(
        adapter=cast(ProviderAdapter[ReviewResponse], adapter),
        bus=bus,
        overrides_store=overrides_store,
    )

    assert loop.override_tag == "latest"

    # Execute triggers auto-optimization since no digest exists
    loop.execute(ReviewTurnParams(request="test request"))

    # Verify optimization was called (adapter recorded the call)
    assert any("workspace-digest" in call for call in adapter.calls)

    # Verify digest was persisted to session
    session_digest = latest_workspace_digest(loop.session, "workspace-digest")
    assert session_digest is not None
    assert session_digest.body == "- Repo instructions from stub"


def test_default_deadline_refreshed_per_execute(tmp_path: Path) -> None:
    """Each execute() call builds a fresh default deadline."""

    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    adapter = cast(ProviderAdapter[ReviewResponse], _RecordingDeadlineAdapter())
    bus = InProcessEventBus()
    loop = CodeReviewLoop(
        adapter=adapter,
        bus=bus,
        overrides_store=overrides_store,
    )

    set_workspace_digest(loop.session, "workspace-digest", "- existing digest")

    loop.execute(ReviewTurnParams(request="first"))
    loop.execute(ReviewTurnParams(request="second"))

    assert len(adapter.deadlines) == 2
    first_deadline, second_deadline = adapter.deadlines
    assert isinstance(first_deadline, Deadline)
    assert isinstance(second_deadline, Deadline)
    assert first_deadline is not second_deadline

    now = datetime.now(UTC)
    assert first_deadline.expires_at > now
    assert second_deadline.expires_at > now


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
