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

"""Tests for task completion checker abstractions."""

from __future__ import annotations

from typing import cast

import pytest

from weakincentives.adapters.claude_agent_sdk._task_completion import (
    CompositeChecker,
    LLMJudgeChecker,
    PlanBasedChecker,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)
from weakincentives.adapters.core import ProviderAdapter
from weakincentives.contrib.tools.planning import Plan, PlanningToolsSection, PlanStep
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session


@pytest.fixture
def session() -> Session:
    bus = InProcessDispatcher()
    return Session(bus=bus)


class TestTaskCompletionResult:
    def test_ok_without_feedback(self) -> None:
        result = TaskCompletionResult.ok()
        assert result.complete is True
        assert result.feedback is None

    def test_ok_with_feedback(self) -> None:
        result = TaskCompletionResult.ok("All tasks done!")
        assert result.complete is True
        assert result.feedback == "All tasks done!"

    def test_incomplete_requires_feedback(self) -> None:
        result = TaskCompletionResult.incomplete("2 tasks remaining")
        assert result.complete is False
        assert result.feedback == "2 tasks remaining"

    def test_direct_construction(self) -> None:
        result = TaskCompletionResult(complete=True, feedback="Done")
        assert result.complete is True
        assert result.feedback == "Done"

    def test_is_frozen(self) -> None:
        result = TaskCompletionResult.ok()
        with pytest.raises(AttributeError):
            result.complete = False  # type: ignore[misc]


class TestTaskCompletionContext:
    def test_basic_construction(self, session: Session) -> None:
        context = TaskCompletionContext(session=session)
        assert context.session is session
        assert context.tentative_output is None
        assert context.filesystem is None
        assert context.adapter is None
        assert context.stop_reason is None

    def test_with_all_fields(self, session: Session) -> None:
        mock_adapter = cast(ProviderAdapter, object())

        context = TaskCompletionContext(
            session=session,
            tentative_output={"key": "value"},
            filesystem=None,
            adapter=mock_adapter,
            stop_reason="end_turn",
        )

        assert context.session is session
        assert context.tentative_output == {"key": "value"}
        assert context.adapter is mock_adapter
        assert context.stop_reason == "end_turn"


class TestPlanBasedChecker:
    def test_complete_when_no_plan_type_configured(self, session: Session) -> None:
        """Checker returns complete when plan_type is None (no-op mode)."""
        checker = PlanBasedChecker(plan_type=None)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True
        assert result.feedback is not None
        assert "No planning tools available" in result.feedback

    def test_complete_when_no_plan_slice(self, session: Session) -> None:
        """Checker returns complete when Plan slice isn't registered or empty."""
        # Use a fresh session without Plan slice to test the no-plan case
        fresh_bus = InProcessDispatcher()
        fresh_session = Session(bus=fresh_bus)

        checker = PlanBasedChecker(plan_type=Plan)
        context = TaskCompletionContext(session=fresh_session)

        result = checker.check(context)

        # Should be complete regardless of which code path (no slice vs empty slice)
        assert result.complete is True
        # Feedback should indicate no plan work to do
        assert result.feedback is not None
        assert "plan" in result.feedback.lower() or "No" in result.feedback

    def test_complete_when_no_plan_initialized(self, session: Session) -> None:
        """Checker returns complete when Plan slice exists but is empty."""
        PlanningToolsSection._initialize_session(session)

        checker = PlanBasedChecker(plan_type=Plan)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True
        assert "No plan has been created" in str(result.feedback)

    def test_complete_when_all_tasks_done(self, session: Session) -> None:
        """Checker returns complete when all plan steps are done."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test objective",
                status="completed",
                steps=(
                    PlanStep(step_id=1, title="Task 1", status="done"),
                    PlanStep(step_id=2, title="Task 2", status="done"),
                ),
            )
        )

        checker = PlanBasedChecker(plan_type=Plan)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True
        assert "2 task(s)" in str(result.feedback)

    def test_incomplete_when_tasks_pending(self, session: Session) -> None:
        """Checker returns incomplete when tasks are pending."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test objective",
                status="active",
                steps=(
                    PlanStep(step_id=1, title="Done task", status="done"),
                    PlanStep(step_id=2, title="Pending task", status="pending"),
                ),
            )
        )

        checker = PlanBasedChecker(plan_type=Plan)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is False
        assert "1 incomplete task(s)" in str(result.feedback)
        assert "Pending task" in str(result.feedback)

    def test_incomplete_when_tasks_in_progress(self, session: Session) -> None:
        """Checker returns incomplete when tasks are in_progress."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test objective",
                status="active",
                steps=(
                    PlanStep(step_id=1, title="In progress task", status="in_progress"),
                ),
            )
        )

        checker = PlanBasedChecker(plan_type=Plan)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is False
        assert "1 incomplete task(s)" in str(result.feedback)

    def test_truncates_long_task_list(self, session: Session) -> None:
        """Feedback truncates task titles when many are incomplete."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test objective",
                status="active",
                steps=tuple(
                    PlanStep(step_id=i, title=f"Task {i}", status="pending")
                    for i in range(1, 10)
                ),
            )
        )

        checker = PlanBasedChecker(plan_type=Plan)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is False
        assert "Task 1" in str(result.feedback)
        assert "Task 2" in str(result.feedback)
        assert "Task 3" in str(result.feedback)
        assert "..." in str(result.feedback)
        assert "Task 9" not in str(result.feedback)

    def test_with_explicit_plan_type(self, session: Session) -> None:
        """Checker works with explicitly provided plan type."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="completed",
                steps=(PlanStep(step_id=1, title="Task", status="done"),),
            )
        )

        checker = PlanBasedChecker(plan_type=Plan)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True

    def test_complete_when_empty_steps(self, session: Session) -> None:
        """Checker returns complete when plan has no steps."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(Plan(objective="Test", status="active", steps=()))

        checker = PlanBasedChecker(plan_type=Plan)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True


class TestLLMJudgeChecker:
    def test_emits_placeholder_warning(self, session: Session) -> None:
        """LLMJudgeChecker emits a warning about being a placeholder."""
        with pytest.warns(UserWarning, match="placeholder implementation"):
            LLMJudgeChecker()

    def test_incomplete_without_adapter_when_required(self, session: Session) -> None:
        """Checker returns incomplete when adapter required but missing."""
        with pytest.warns(UserWarning):
            checker = LLMJudgeChecker(require_adapter=True)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is False
        assert "no adapter available" in str(result.feedback).lower()

    def test_complete_without_adapter_when_not_required(self, session: Session) -> None:
        """Checker returns complete when adapter not required and missing."""
        with pytest.warns(UserWarning):
            checker = LLMJudgeChecker(require_adapter=False)
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True

    def test_with_adapter_returns_placeholder(self, session: Session) -> None:
        """Checker with adapter returns placeholder response."""
        mock_adapter = cast(ProviderAdapter, object())
        with pytest.warns(UserWarning):
            checker = LLMJudgeChecker()
        context = TaskCompletionContext(session=session, adapter=mock_adapter)

        result = checker.check(context)

        # Current implementation returns a placeholder
        assert result.complete is True
        assert "implementation pending" in str(result.feedback).lower()

    def test_custom_criteria(self, session: Session) -> None:
        """Checker accepts custom evaluation criteria."""
        with pytest.warns(UserWarning):
            checker = LLMJudgeChecker(
                criteria="Verify all tests pass and coverage is 100%."
            )
        context = TaskCompletionContext(session=session)

        # Without adapter, it should just return incomplete
        result = checker.check(context)

        assert result.complete is False

    def test_build_verification_prompt_includes_output(self, session: Session) -> None:
        """Verification prompt includes tentative output."""
        with pytest.warns(UserWarning):
            checker = LLMJudgeChecker()
        context = TaskCompletionContext(
            session=session,
            tentative_output={"summary": "All done"},
            stop_reason="end_turn",
        )

        prompt = checker._build_verification_prompt(context)

        assert "Tentative Output" in prompt
        assert "All done" in prompt
        assert "Stop Reason" in prompt
        assert "end_turn" in prompt


class TestCompositeChecker:
    def test_empty_checkers_returns_complete(self, session: Session) -> None:
        """Empty composite returns complete."""
        checker = CompositeChecker(checkers=())
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True
        assert "No checkers configured" in str(result.feedback)

    def test_all_must_pass_returns_first_failure(self, session: Session) -> None:
        """With all_must_pass, first failure stops evaluation."""

        class FailingChecker(TaskCompletionChecker):
            def check(self, ctx: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.incomplete("Task A incomplete")

        class PassingChecker(TaskCompletionChecker):
            def check(self, ctx: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.ok("Task B complete")

        checker = CompositeChecker(
            checkers=(FailingChecker(), PassingChecker()),
            all_must_pass=True,
        )
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is False
        assert "Task A incomplete" in str(result.feedback)

    def test_all_must_pass_returns_combined_on_success(self, session: Session) -> None:
        """With all_must_pass, all passing combines feedback."""

        class PassingCheckerA(TaskCompletionChecker):
            def check(self, ctx: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.ok("A passed")

        class PassingCheckerB(TaskCompletionChecker):
            def check(self, ctx: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.ok("B passed")

        checker = CompositeChecker(
            checkers=(PassingCheckerA(), PassingCheckerB()),
            all_must_pass=True,
        )
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True
        assert "A passed" in str(result.feedback)
        assert "B passed" in str(result.feedback)

    def test_any_pass_returns_first_success(self, session: Session) -> None:
        """With all_must_pass=False, first success stops evaluation."""

        class FailingChecker(TaskCompletionChecker):
            def check(self, ctx: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.incomplete("Failed")

        class PassingChecker(TaskCompletionChecker):
            def check(self, ctx: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.ok("Passed")

        checker = CompositeChecker(
            checkers=(FailingChecker(), PassingChecker()),
            all_must_pass=False,
        )
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True
        assert "Passed" in str(result.feedback)

    def test_any_pass_returns_incomplete_when_all_fail(self, session: Session) -> None:
        """With all_must_pass=False, all failing combines feedback."""

        class FailingCheckerA(TaskCompletionChecker):
            def check(self, ctx: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.incomplete("A failed")

        class FailingCheckerB(TaskCompletionChecker):
            def check(self, ctx: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.incomplete("B failed")

        checker = CompositeChecker(
            checkers=(FailingCheckerA(), FailingCheckerB()),
            all_must_pass=False,
        )
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is False
        assert "A failed" in str(result.feedback)
        assert "B failed" in str(result.feedback)

    def test_with_plan_based_checker(self, session: Session) -> None:
        """Composite works with PlanBasedChecker."""
        PlanningToolsSection._initialize_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="completed",
                steps=(PlanStep(step_id=1, title="Task", status="done"),),
            )
        )

        checker = CompositeChecker(
            checkers=(PlanBasedChecker(plan_type=Plan),),
            all_must_pass=True,
        )
        context = TaskCompletionContext(session=session)

        result = checker.check(context)

        assert result.complete is True


class TestTaskCompletionCheckerProtocol:
    def test_plan_based_checker_satisfies_protocol(self) -> None:
        """PlanBasedChecker is a TaskCompletionChecker."""
        checker = PlanBasedChecker(plan_type=Plan)
        assert isinstance(checker, TaskCompletionChecker)

    def test_llm_judge_checker_satisfies_protocol(self) -> None:
        """LLMJudgeChecker is a TaskCompletionChecker."""
        with pytest.warns(UserWarning):
            checker = LLMJudgeChecker()
        assert isinstance(checker, TaskCompletionChecker)

    def test_composite_checker_satisfies_protocol(self) -> None:
        """CompositeChecker is a TaskCompletionChecker."""
        checker = CompositeChecker(checkers=())
        assert isinstance(checker, TaskCompletionChecker)

    def test_custom_checker_satisfies_protocol(self) -> None:
        """Custom implementations satisfy the protocol."""

        class CustomChecker:
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.ok("Custom check passed")

        checker = CustomChecker()
        assert isinstance(checker, TaskCompletionChecker)

    def test_custom_checker_can_be_used_in_composite(self, session: Session) -> None:
        """Custom checkers can be composed."""

        class AlwaysPassChecker:
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult.ok("Always pass")

        composite = CompositeChecker(
            checkers=(AlwaysPassChecker(), PlanBasedChecker(plan_type=Plan)),  # type: ignore[arg-type]
            all_must_pass=True,
        )
        context = TaskCompletionContext(session=session)

        result = composite.check(context)

        assert result.complete is True
