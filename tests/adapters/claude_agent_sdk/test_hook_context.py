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

"""Tests for HookContext construction and behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from weakincentives.adapters.claude_agent_sdk._hooks import HookConstraints, HookContext
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.session import Session

from ._hook_helpers import _make_prompt


class TestHookContext:
    def test_basic_construction(self, session: Session) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )
        assert context.session is session
        assert context.adapter_name == "test_adapter"
        assert context.prompt_name == "test_prompt"
        assert context.deadline is None
        assert context.budget_tracker is None
        assert context.stop_reason is None
        assert context._tool_count == 0

    def test_with_deadline_and_budget(self, session: Session) -> None:
        from weakincentives.clock import FakeClock

        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        deadline = Deadline(anchor + timedelta(minutes=5), clock=clock)
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget)

        constraints = HookConstraints(deadline=deadline, budget_tracker=tracker)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )
        assert context.deadline is deadline
        assert context.budget_tracker is tracker

    def test_beat_with_heartbeat_configured(self, session: Session) -> None:
        from weakincentives.runtime.watchdog import Heartbeat

        beat_count = 0

        def on_beat() -> None:
            nonlocal beat_count
            beat_count += 1

        heartbeat = Heartbeat()
        heartbeat.add_callback(on_beat)

        constraints = HookConstraints(heartbeat=heartbeat)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )

        assert context.heartbeat is heartbeat
        context.beat()
        assert beat_count == 1

    def test_beat_without_heartbeat_is_noop(self, session: Session) -> None:
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        assert context.heartbeat is None
        # Should not raise
        context.beat()
