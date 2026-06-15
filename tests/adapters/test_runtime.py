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

"""Tests for Runtime: the bound (adapter, prompt, sandbox) triple."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.sandbox import make_memory_sandbox
from weakincentives.adapters.core import (
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
    RuntimeReleasedError,
)
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    Prompt,
    PromptTemplate,
    ToolContext,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session.protocols import SessionProtocol
from weakincentives.sandbox import (
    HostMount,
    Sandbox,
    SandboxProvider,
    WorkspaceConfig,
)


class _RecordingAdapter(ProviderAdapter[Any]):
    """Minimal adapter recording what its core contract receives."""

    def __init__(self, *, sandbox_provider: SandboxProvider | None = None) -> None:
        super().__init__(sandbox_provider=sandbox_provider)
        self.seen: list[Sandbox] = []
        self.trackers: list[BudgetTracker | None] = []
        self.deadlines: list[Deadline | None] = []

    def _evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: object = None,
        run_context: RunContext | None = None,
        sandbox: Sandbox,
    ) -> PromptResponse[OutputT]:
        del session, heartbeat, run_context
        self.seen.append(sandbox)
        self.trackers.append(budget_tracker)
        self.deadlines.append(deadline)
        _ = sandbox.filesystem.write(f"round-{len(self.seen)}.txt", "x")
        return PromptResponse(prompt_name=prompt.key, text="ok", output=None)


def _plain_prompt(key: str = "plain") -> Prompt[Any]:
    return Prompt(PromptTemplate.create(ns="t", key=key))


class TestRuntimePairing:
    def test_runtime_materializes_template_config(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hello")
        prompt: Prompt[Any] = Prompt(
            PromptTemplate.create(
                ns="t",
                key="with-mounts",
                workspace=WorkspaceConfig(
                    mounts=(HostMount(host_path=str(tmp_path), mount_path="src"),)
                ),
            )
        )
        adapter = _RecordingAdapter()

        with adapter.runtime(prompt) as rt:
            assert rt.adapter is adapter
            assert rt.prompt is prompt
            assert rt.sandbox.filesystem.exists("src/README.md")
            assert rt.sandbox.filesystem.root == rt.sandbox.root
            root = Path(rt.sandbox.root)
            assert root.is_dir()
            assert rt.released is False

        # Lease released on exit: locally provisioned sandboxes are removed.
        assert rt.released is True
        assert not root.exists()

    def test_runtime_defaults_to_empty_config(self) -> None:
        opened: list[WorkspaceConfig] = []

        class _SpyProvider:
            def open(self, config: WorkspaceConfig) -> Sandbox:
                opened.append(config)
                return make_memory_sandbox()

        adapter = _RecordingAdapter(sandbox_provider=_SpyProvider())

        with adapter.runtime(_plain_prompt()):
            pass

        assert opened == [WorkspaceConfig()]


class TestRuntimeEvaluate:
    def test_sandbox_spans_multiple_evaluations(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="spans")

        with adapter.runtime(prompt) as rt:
            _ = rt.evaluate(session=None)  # type: ignore[arg-type]
            _ = rt.evaluate(session=None)  # type: ignore[arg-type]

            # Same environment both rounds; round-1 output visible in round 2.
            assert adapter.seen == [rt.sandbox, rt.sandbox]
            assert rt.sandbox.filesystem.exists("round-1.txt")
            assert rt.sandbox.filesystem.exists("round-2.txt")

    def test_release_is_idempotent(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="idempotent")

        with adapter.runtime(prompt) as rt:
            pass

        # A second release (e.g. defensive teardown) is a no-op.
        rt._release()
        assert rt.released is True

    def test_evaluate_after_release_raises(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="released")

        with adapter.runtime(prompt) as rt:
            pass

        with pytest.raises(RuntimeReleasedError, match="released"):
            _ = rt.evaluate(session=None)  # type: ignore[arg-type]

    def test_promotes_budget_to_tracker(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="budget")
        budget = Budget(
            deadline=Deadline.create(datetime.now(UTC) + timedelta(minutes=5))
        )

        with adapter.runtime(prompt) as rt:
            _ = rt.evaluate(session=None, budget=budget)  # type: ignore[arg-type]

        tracker = adapter.trackers[0]
        assert tracker is not None
        assert tracker.budget is budget
        # Effective deadline derived from the budget.
        assert adapter.deadlines[0] is budget.deadline

    def test_passes_existing_tracker_through(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="tracker")
        budget = Budget(
            deadline=Deadline.create(datetime.now(UTC) + timedelta(minutes=5))
        )
        tracker = BudgetTracker(budget)

        with adapter.runtime(prompt) as rt:
            _ = rt.evaluate(  # type: ignore[arg-type]
                session=None, budget=budget, budget_tracker=tracker
            )

        assert adapter.trackers[0] is tracker

    def test_expired_deadline_fails_before_adapter_call(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="expired")
        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline.create(
            expires_at=anchor + timedelta(seconds=5), clock=clock
        )
        clock.advance(10)

        with adapter.runtime(prompt) as rt:
            with pytest.raises(PromptEvaluationError, match="Deadline expired"):
                _ = rt.evaluate(session=None, deadline=deadline)  # type: ignore[arg-type]

        assert adapter.seen == []


class TestWorkspacePreview:
    def test_render_resolves_preview_from_sandbox(self) -> None:
        prompt: Prompt[Any] = Prompt(
            PromptTemplate.create(ns="t", key="preview", workspace=WorkspaceConfig())
        )
        adapter = _RecordingAdapter()

        with adapter.runtime(prompt) as rt:
            _ = rt.evaluate(session=None)  # type: ignore[arg-type]
            # Render-time resolution: the listing reflects the live sandbox.
            assert "- round-1.txt" in prompt.render(sandbox=rt.sandbox).text

            _ = rt.evaluate(session=None)  # type: ignore[arg-type]
            assert "- round-2.txt" in prompt.render(sandbox=rt.sandbox).text

        # No run state is stored on the prompt itself.
        assert prompt.params == ()
        assert "round-1.txt" not in prompt.render().text

    def test_render_without_sandbox_shows_placeholder(self) -> None:
        prompt: Prompt[Any] = Prompt(
            PromptTemplate.create(
                ns="t", key="placeholder", workspace=WorkspaceConfig()
            )
        )

        assert "not yet materialized" in prompt.render().text

    def test_no_preview_without_template_sandbox(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt(key="no-preview")

        with adapter.runtime(prompt) as rt:
            _ = rt.evaluate(session=None)  # type: ignore[arg-type]
            # No preview section: rendering with the sandbox adds nothing.
            assert "round-1.txt" not in prompt.render(sandbox=rt.sandbox).text

        assert prompt.params == ()


class TestEvaluateSugar:
    def test_one_shot_opens_and_releases(self) -> None:
        adapter = _RecordingAdapter()
        prompt = _plain_prompt()

        response = adapter.evaluate(prompt, session=None)  # type: ignore[arg-type]

        assert response.text == "ok"
        assert len(adapter.seen) == 1
        # The call-scoped runtime released its lease before returning.
        assert not Path(adapter.seen[0].root).exists()


class TestToolContextShell:
    def test_facets_exposed_from_sandbox(self) -> None:
        sandbox = make_memory_sandbox()
        context = ToolContext(
            prompt=Prompt(PromptTemplate.create(ns="t", key="shell")),  # type: ignore[arg-type]
            rendered_prompt=None,
            adapter=None,  # type: ignore[arg-type]
            session=None,  # type: ignore[arg-type]
            sandbox=sandbox,
        )

        assert context.shell is sandbox.shell
        assert context.filesystem is sandbox.filesystem
