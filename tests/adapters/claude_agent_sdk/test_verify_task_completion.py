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

"""Tests for verify_task_completion and check_task_completion functions."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)
from weakincentives.adapters.claude_agent_sdk._task_completion import PlanBasedChecker
from weakincentives.prompt import Prompt, PromptTemplate
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.session import Session

from ._hook_helpers import Plan, PlanStep, _initialize_plan_session


class TestVerifyTaskCompletion:
    """Tests for verify_task_completion function."""

    @pytest.fixture
    def session(self) -> Session:
        return Session(dispatcher=InProcessDispatcher())

    @pytest.fixture
    def adapter(self) -> ClaudeAgentSDKAdapter:
        return ClaudeAgentSDKAdapter()

    @staticmethod
    def _call_verify(adapter: ClaudeAgentSDKAdapter, **kwargs: Any) -> None:
        from weakincentives.adapters.claude_agent_sdk._result_extraction import (
            verify_task_completion,
        )

        verify_task_completion(
            **kwargs, client_config=adapter._client_config, adapter=adapter
        )

    def test_no_checker_configured_does_nothing(
        self, adapter: ClaudeAgentSDKAdapter, session: Session
    ) -> None:
        """When no checker is configured, verification passes."""
        self._call_verify(
            adapter,
            output={"key": "value"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test",
        )

    def test_no_output_does_nothing(self, session: Session) -> None:
        """When output is None, verification passes."""
        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )
        self._call_verify(
            adapter,
            output=None,
            session=session,
            stop_reason="structured_output",
            prompt_name="test",
        )

    def test_logs_warning_when_tasks_incomplete(
        self, session: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When tasks are incomplete, logs warning but doesn't raise error."""
        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(
                    PlanStep(step_id=1, title="Done", status="done"),
                    PlanStep(step_id=2, title="Pending", status="pending"),
                ),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        caplog.set_level(logging.WARNING)

        self._call_verify(
            adapter,
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
        )

        assert any("incomplete_tasks" in record.message for record in caplog.records), (
            "Should log warning about incomplete tasks"
        )
        warning_logged = False
        for record in caplog.records:
            if "incomplete_tasks" in record.message:
                warning_logged = True
                break
        assert warning_logged, "Should have logged incomplete tasks warning"

    def test_passes_when_tasks_complete(self, session: Session) -> None:
        """When all tasks are complete, verification passes."""
        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="completed",
                steps=(
                    PlanStep(step_id=1, title="Task 1", status="done"),
                    PlanStep(step_id=2, title="Task 2", status="done"),
                ),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        self._call_verify(
            adapter,
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
        )

    def test_skips_when_deadline_exceeded(self, session: Session) -> None:
        """When deadline is exceeded, verification is skipped."""
        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Pending", status="pending"),),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        exceeded_deadline = MagicMock()
        exceeded_deadline.remaining.return_value = timedelta(seconds=-1)

        self._call_verify(
            adapter,
            output={"summary": "partial"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            deadline=exceeded_deadline,
        )

    def test_skips_when_budget_exhausted(self, session: Session) -> None:
        """When budget is exhausted, verification is skipped."""
        from weakincentives.budget import Budget, BudgetTracker
        from weakincentives.runtime.events.types import TokenUsage

        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Pending", status="pending"),),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        budget = Budget(max_total_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative("test", TokenUsage(input_tokens=50, output_tokens=50))

        self._call_verify(
            adapter,
            output={"summary": "partial"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            budget_tracker=tracker,
        )

    def test_passes_filesystem_and_adapter_to_context(self, session: Session) -> None:
        """Filesystem and adapter are passed to TaskCompletionContext."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            TaskCompletionChecker,
            TaskCompletionContext,
            TaskCompletionResult,
        )
        from weakincentives.filesystem import Filesystem

        captured_context: list[TaskCompletionContext] = []

        class CapturingChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                captured_context.append(context)
                return TaskCompletionResult.ok()

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=CapturingChecker(),
            ),
        )

        mock_filesystem = MagicMock(spec=Filesystem)
        mock_resources = MagicMock()
        mock_resources.get.return_value = mock_filesystem
        mock_prompt = MagicMock()
        mock_prompt.resources = mock_resources

        self._call_verify(
            adapter,
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            prompt=mock_prompt,
        )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.filesystem is mock_filesystem
        assert ctx.adapter is adapter

    def test_handles_filesystem_lookup_failure(self, session: Session) -> None:
        """When filesystem lookup fails, context still gets adapter but no filesystem."""
        from weakincentives.adapters.claude_agent_sdk._task_completion import (
            TaskCompletionChecker,
            TaskCompletionContext,
            TaskCompletionResult,
        )
        from weakincentives.resources.errors import UnboundResourceError

        captured_context: list[TaskCompletionContext] = []

        class CapturingChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                captured_context.append(context)
                return TaskCompletionResult.ok()

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=CapturingChecker(),
            ),
        )

        mock_resources = MagicMock()
        mock_resources.get.side_effect = UnboundResourceError(object)
        mock_prompt = MagicMock()
        mock_prompt.resources = mock_resources

        self._call_verify(
            adapter,
            output={"summary": "done"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            prompt=mock_prompt,
        )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.filesystem is None
        assert ctx.adapter is adapter

    def test_logs_warning_when_budget_not_exhausted(
        self, session: Session, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When budget_tracker is provided but not exhausted, logs warning."""
        from weakincentives.budget import Budget, BudgetTracker
        from weakincentives.runtime.events.types import TokenUsage

        _initialize_plan_session(session)
        session.dispatch(
            Plan(
                objective="Test",
                status="active",
                steps=(PlanStep(step_id=1, title="Pending", status="pending"),),
            )
        )

        adapter = ClaudeAgentSDKAdapter(
            client_config=ClaudeAgentSDKClientConfig(
                task_completion_checker=PlanBasedChecker(plan_type=Plan),
            ),
        )

        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative("test", TokenUsage(input_tokens=50, output_tokens=50))

        caplog.set_level(logging.WARNING)

        self._call_verify(
            adapter,
            output={"summary": "partial"},
            session=session,
            stop_reason="structured_output",
            prompt_name="test_prompt",
            budget_tracker=tracker,
        )

        assert any("incomplete_tasks" in record.message for record in caplog.records), (
            "Should log warning about incomplete tasks when budget remains"
        )


class TestCheckTaskCompletion:
    """Tests for check_task_completion function."""

    def test_returns_false_none_for_empty_messages(self, session: Session) -> None:
        """check_task_completion returns (False, None) for empty message list."""
        from weakincentives.adapters.claude_agent_sdk._hooks import (
            HookConstraints,
            HookContext,
        )
        from weakincentives.adapters.claude_agent_sdk._sdk_execution import (
            check_task_completion,
        )

        checker = MagicMock()
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        prompt: Prompt[None] = Prompt(template)
        constraints = HookConstraints()
        hook_context = HookContext(
            prompt=prompt,
            session=session,
            adapter_name="test",
            prompt_name="test",
            constraints=constraints,
        )

        result = check_task_completion(checker, [], hook_context)

        assert result == (False, None)
        checker.check.assert_not_called()
