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
from weakincentives.contrib.tools.digests import (
    latest_workspace_digest,
    set_workspace_digest,
)
from weakincentives.deadlines import Deadline
from weakincentives.debug import dump_session
from weakincentives.prompt import Prompt
from weakincentives.prompt.overrides import (
    LocalPromptOverridesStore,
    SectionOverride,
    descriptor_for_prompt,
)
from weakincentives.runtime import (
    InMemoryMailbox,
    MainLoopRequest,
    MainLoopResult,
)
from weakincentives.runtime.events import InProcessDispatcher, PromptRendered
from weakincentives.runtime.session import Session
from weakincentives.types import SupportsDataclass


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
        bus: InProcessDispatcher | None = None,
        session: Session | None = None,
        deadline: object | None = None,
        budget: object | None = None,
        budget_tracker: object | None = None,
        resources: object | None = None,
    ) -> PromptResponse[Any]:
        del deadline, budget, budget_tracker, resources
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
                bus.dispatch(optimization_event)
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
        bus: InProcessDispatcher | None = None,
        session: Session | None = None,
        deadline: Deadline | None = None,
        budget: object | None = None,
        budget_tracker: object | None = None,
        resources: object | None = None,
    ) -> PromptResponse[Any]:
        del bus, session, budget, budget_tracker, resources
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
    bus = InProcessDispatcher()
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

    publish_result = bus.dispatch(event)
    assert publish_result.handled_count >= 1
    del overrides_store  # Not needed for this test

    captured = capsys.readouterr()
    assert "Rendered prompt" in captured.out
    assert "<prompt body>" in captured.out

    stored_events = session[PromptRendered].all()
    assert stored_events
    assert stored_events[-1].rendered_prompt == "<prompt body>"


def test_workspace_digest_section_empty_by_default() -> None:
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    template = build_task_prompt(session=session)

    rendered = Prompt(template).bind(ReviewTurnParams(request="demo request")).render()

    assert "## 2. Workspace Digest (workspace-digest)" in rendered.text
    post_section = rendered.text.split("## 2. Workspace Digest (workspace-digest)", 1)[
        1
    ]
    section_body = post_section.split("\n## ", 1)[0]
    assert "Workspace digest unavailable" in section_body


def test_workspace_digest_override_applied_when_no_session_digest(
    tmp_path: Path,
) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    bus = InProcessDispatcher()
    session = Session(bus=bus)
    template = build_task_prompt(session=session)

    descriptor = descriptor_for_prompt(template)
    digest_section = next(
        s for s in descriptor.sections if s.path == ("workspace-digest",)
    )
    override = SectionOverride(
        path=digest_section.path,
        expected_hash=digest_section.content_hash,
        body="- Override digest",
    )
    overrides_store.store(descriptor, override, tag="seed")

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
    descriptor = descriptor_for_prompt(template)
    digest_section = next(
        s for s in descriptor.sections if s.path == ("workspace-digest",)
    )
    override = SectionOverride(
        path=digest_section.path,
        expected_hash=digest_section.content_hash,
        body="- Override digest",
    )
    overrides_store.store(descriptor, override, tag="seed")
    # When a digest exists, visibility is SUMMARY by default, so we provide
    # an explicit summary to verify it's used instead of the override body.
    set_workspace_digest(
        session,
        "workspace-digest",
        "- Session digest body",
        summary="Session digest summary",
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
    # With SUMMARY visibility, the session digest summary is shown (not the
    # override body or the session body)
    assert "Session digest summary" in rendered.text
    assert "Override digest" not in rendered.text


def test_auto_optimization_runs_on_first_request(tmp_path: Path) -> None:
    """Auto-optimization runs when first request is processed (if enabled)."""
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    adapter = _RepositoryOptimizationAdapter("- Repo instructions from stub")
    responses: InMemoryMailbox[MainLoopResult[ReviewResponse], None] = InMemoryMailbox(
        name="responses"
    )
    requests: InMemoryMailbox[
        MainLoopRequest[ReviewTurnParams], MainLoopResult[ReviewResponse]
    ] = InMemoryMailbox(name="requests")
    try:
        loop = CodeReviewLoop(
            adapter=cast(ProviderAdapter[ReviewResponse], adapter),
            requests=requests,
            overrides_store=overrides_store,
            enable_optimization=True,
        )

        assert loop.override_tag == "latest"

        # Send request via mailbox with reply_to
        request_event = MainLoopRequest(
            request=ReviewTurnParams(request="test request")
        )
        requests.send(request_event, reply_to=responses)

        # Process one iteration
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Verify optimization was called (adapter recorded the call)
        assert any("workspace-digest" in call for call in adapter.calls)

        # Verify digest was persisted to session
        session_digest = latest_workspace_digest(loop.session, "workspace-digest")
        assert session_digest is not None
        assert session_digest.body == "- Repo instructions from stub"
    finally:
        requests.close()
        responses.close()


def test_deadline_passed_per_request(tmp_path: Path) -> None:
    """Each request gets a deadline from MainLoopRequest."""
    from datetime import timedelta

    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    adapter = cast(ProviderAdapter[ReviewResponse], _RecordingDeadlineAdapter())
    responses: InMemoryMailbox[MainLoopResult[ReviewResponse], None] = InMemoryMailbox(
        name="responses"
    )
    requests: InMemoryMailbox[
        MainLoopRequest[ReviewTurnParams], MainLoopResult[ReviewResponse]
    ] = InMemoryMailbox(name="requests")
    try:
        loop = CodeReviewLoop(
            adapter=adapter,
            requests=requests,
            overrides_store=overrides_store,
        )

        set_workspace_digest(loop.session, "workspace-digest", "- existing digest")

        # Send requests with different deadlines
        first_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
        second_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=10))

        requests.send(
            MainLoopRequest(
                request=ReviewTurnParams(request="first"),
                deadline=first_deadline,
            ),
            reply_to=responses,
        )
        loop.run(max_iterations=1, wait_time_seconds=0)

        requests.send(
            MainLoopRequest(
                request=ReviewTurnParams(request="second"),
                deadline=second_deadline,
            ),
            reply_to=responses,
        )
        loop.run(max_iterations=1, wait_time_seconds=0)

        assert len(adapter.deadlines) == 2
        recorded_first, recorded_second = adapter.deadlines
        assert recorded_first is first_deadline
        assert recorded_second is second_deadline
    finally:
        requests.close()
        responses.close()


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
    """Sessions with no data (only empty slices) skip snapshot dump."""
    caplog.set_level(logging.INFO, logger="weakincentives.debug")
    session = Session()

    # Sessions have builtin reducers registered but no data
    # Empty slices are excluded from snapshots, so this should be skipped
    snapshot_path = dump_session(session, _redirect_snapshots)

    assert snapshot_path is None
    assert not any(_redirect_snapshots.iterdir())
    record = next(
        rec
        for rec in caplog.records
        if getattr(rec, "session_id", None) == str(session.session_id)
    )
    assert "skipped; no slices" in record.getMessage()
