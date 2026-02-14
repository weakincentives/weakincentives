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

"""Tests for task completion stop hook, PlanBasedChecker, and CompositeChecker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import cast

from weakincentives.adapters.claude_agent_sdk._hooks import (
    HookConstraints,
    HookContext,
    create_task_completion_stop_hook,
)
from weakincentives.adapters.claude_agent_sdk._task_completion import PlanBasedChecker
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt.protocols import PromptProtocol
from weakincentives.runtime.events.types import TokenUsage
from weakincentives.runtime.session import Session

from ._hook_helpers import Plan, PlanStep, _initialize_plan_session, _make_prompt


class TestTaskCompletionStopHook:
    """Tests for task completion stop hook functionality."""

    def test_allows_stop_when_no_plan_slice(self, hook_context: HookContext) -> None:
        """Stop is allowed when Plan slice is not installed in session."""
        hook = create_task_completion_stop_hook(
            hook_context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}
        assert hook_context.stop_reason == "end_turn"

    def test_allows_stop_when_no_plan_initialized(self, session: Session) -> None:
        """Stop is allowed when Plan slice exists but no plan is initialized."""
        # Install Plan slice but don't initialize a plan
        _initialize_plan_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_allows_stop_when_all_tasks_complete(self, session: Session) -> None:
        """Stop is allowed when all plan tasks are marked as done."""
        _initialize_plan_session(session)

        # Create a plan with all steps done
        completed_plan = Plan(
            objective="Test objective",
            status="completed",
            steps=(
                PlanStep(step_id=1, title="Task 1", status="done"),
                PlanStep(step_id=2, title="Task 2", status="done"),
            ),
        )
        session.dispatch(completed_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_signals_continue_when_tasks_incomplete(self, session: Session) -> None:
        """Stop hook signals continuation when tasks are incomplete."""
        _initialize_plan_session(session)

        # Create a plan with incomplete steps
        incomplete_plan = Plan(
            objective="Test objective",
            status="active",
            steps=(
                PlanStep(step_id=1, title="Done task", status="done"),
                PlanStep(step_id=2, title="Pending task", status="pending"),
                PlanStep(step_id=3, title="In progress task", status="in_progress"),
            ),
        )
        session.dispatch(incomplete_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        # Should return continue_: True to force continuation (SDK converts to "continue")
        assert result.get("continue_") is True
        assert "2 incomplete task(s)" in result.get("reason", "")
        assert "Pending task" in result.get("reason", "")
        assert context.stop_reason == "end_turn"

    def test_records_stop_reason(self, session: Session) -> None:
        """Stop hook records the stop reason regardless of task state."""
        _initialize_plan_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        # SDK StopHookInput doesn't have stopReason field; hook defaults to end_turn
        input_data = {"hook_event_name": "Stop", "stop_hook_active": False}

        assert context.stop_reason is None
        asyncio.run(hook(input_data, None, {"signal": None}))
        # StopHookInput doesn't have stopReason; hook always sets end_turn
        assert context.stop_reason == "end_turn"

    def test_defaults_stop_reason_to_end_turn(self, session: Session) -> None:
        """Stop hook defaults stopReason to end_turn when not provided."""
        _initialize_plan_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )

        asyncio.run(hook({"hook_event_name": "Stop"}, None, {"signal": None}))

        assert context.stop_reason == "end_turn"

    def test_handles_empty_steps(self, session: Session) -> None:
        """Stop is allowed when plan has no steps."""
        _initialize_plan_session(session)

        # Create a plan with no steps
        empty_plan = Plan(
            objective="Test objective",
            status="active",
            steps=(),
        )
        session.dispatch(empty_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_reminder_message_truncates_long_task_list(self, session: Session) -> None:
        """Reminder message truncates task titles when there are many incomplete."""
        _initialize_plan_session(session)

        # Create a plan with many incomplete steps
        many_steps_plan = Plan(
            objective="Test objective",
            status="active",
            steps=tuple(
                PlanStep(step_id=i, title=f"Task {i}", status="pending")
                for i in range(1, 10)
            ),
        )
        session.dispatch(many_steps_plan)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        result = asyncio.run(hook(input_data, None, {"signal": None}))

        # Should return continue_: True to force continuation (SDK converts to "continue")
        assert result.get("continue_") is True
        reason = result.get("reason", "")
        # Should contain first 3 tasks and ellipsis
        assert "Task 1" in reason
        assert "Task 2" in reason
        assert "Task 3" in reason
        assert "..." in reason
        # Should not list all tasks
        assert "Task 9" not in reason

    def test_checker_protocol_with_plan_based_checker(self, session: Session) -> None:
        """Hook accepts any TaskCompletionChecker implementation."""
        _initialize_plan_session(session)

        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
        )

        # Create hook with PlanBasedChecker - verifies protocol compatibility
        checker = PlanBasedChecker(plan_type=Plan)
        hook = create_task_completion_stop_hook(context, checker=checker)
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        # Should work with checker protocol
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        # No plan initialized, so stop is allowed
        assert result == {}
        assert context.stop_reason == "end_turn"

    def test_skips_check_when_deadline_exceeded(self, session: Session) -> None:
        """Stop hook skips task completion check when deadline expired."""
        from weakincentives.clock import FakeClock

        _initialize_plan_session(session)
        # Create incomplete plan
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        # Create expired deadline using fake clock
        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        expired_deadline = Deadline(anchor + timedelta(seconds=5), clock=clock)
        clock.advance(10)  # Now expired

        constraints = HookConstraints(deadline=expired_deadline)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        # Even with incomplete tasks, stop is allowed due to expired deadline
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}

    def test_skips_check_when_budget_exhausted(self, session: Session) -> None:
        """Stop hook skips task completion check when budget exhausted."""
        _initialize_plan_session(session)
        # Create incomplete plan
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        # Create exhausted budget tracker
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget=budget)
        tracker.record_cumulative(
            "eval-1", TokenUsage(input_tokens=500, output_tokens=600)
        )  # Over budget

        constraints = HookConstraints(budget_tracker=tracker)
        context = HookContext(
            session=session,
            prompt=cast("PromptProtocol[object]", _make_prompt()),
            adapter_name="test_adapter",
            prompt_name="test_prompt",
            constraints=constraints,
        )

        hook = create_task_completion_stop_hook(
            context, checker=PlanBasedChecker(plan_type=Plan)
        )
        input_data = {"hook_event_name": "Stop", "stopReason": "end_turn"}

        # Even with incomplete tasks, stop is allowed due to exhausted budget
        result = asyncio.run(hook(input_data, None, {"signal": None}))

        assert result == {}


class TestPlanBasedCheckerNoPlanType:
    """Tests for PlanBasedChecker with plan_type=None."""

    def test_no_plan_type_returns_ok(self, session: Session) -> None:
        """Checker returns ok when plan_type is None."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            PlanBasedChecker,
            TaskCompletionContext,
        )

        checker = PlanBasedChecker(plan_type=None)
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is True
        assert "No planning tools available" in result.feedback


class TestCompositeChecker:
    """Tests for CompositeChecker."""

    def test_empty_checkers_returns_ok(self, session: Session) -> None:
        """Empty composite checker returns ok."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            TaskCompletionContext,
        )

        checker = CompositeChecker(checkers=())
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is True
        assert "No checkers configured" in result.feedback

    def test_all_must_pass_short_circuits_on_failure(self, session: Session) -> None:
        """Composite checker short-circuits on first failure when all_must_pass=True."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            PlanBasedChecker,
            TaskCompletionContext,
        )

        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        checker = CompositeChecker(
            checkers=(
                PlanBasedChecker(plan_type=Plan),
                PlanBasedChecker(plan_type=None),  # Would pass but won't be reached
            ),
            all_must_pass=True,
        )
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is False
        assert "incomplete" in result.feedback.lower()

    def test_all_must_pass_all_succeed(self, session: Session) -> None:
        """Composite checker returns ok when all checkers pass."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            PlanBasedChecker,
            TaskCompletionContext,
        )

        checker = CompositeChecker(
            checkers=(
                PlanBasedChecker(plan_type=None),
                PlanBasedChecker(plan_type=None),
            ),
            all_must_pass=True,
        )
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is True

    def test_any_pass_short_circuits_on_success(self, session: Session) -> None:
        """Composite checker short-circuits on first success when all_must_pass=False."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            PlanBasedChecker,
            TaskCompletionContext,
        )

        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        checker = CompositeChecker(
            checkers=(
                PlanBasedChecker(plan_type=None),  # Passes
                PlanBasedChecker(plan_type=Plan),  # Would fail but won't be reached
            ),
            all_must_pass=False,
        )
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is True

    def test_any_pass_all_fail(self, session: Session) -> None:
        """Composite checker returns incomplete when all checkers fail in any mode."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            CompositeChecker,
            PlanBasedChecker,
            TaskCompletionContext,
        )

        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Incomplete", status="pending"),),
            )
        )

        checker = CompositeChecker(
            checkers=(
                PlanBasedChecker(plan_type=Plan),
                PlanBasedChecker(plan_type=Plan),
            ),
            all_must_pass=False,
        )
        context = TaskCompletionContext(
            session=session,
            tentative_output=None,
            stop_reason="end_turn",
        )

        result = checker.check(context)

        assert result.complete is False
