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

"""Tests for ACP adapter guardrails: feedback, task completion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from weakincentives.adapters.acp._guardrails import (
    accumulate_usage,
    append_feedback,
    check_task_completion,
    resolve_filesystem,
)
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.prompt.task_completion import (
    TaskCompletionChecker,
    TaskCompletionResult,
)
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.runtime.events.types import TokenUsage
from weakincentives.runtime.session import Session

# ---- Helpers ----


def _make_session() -> tuple[Session, InProcessDispatcher]:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher, tags={"suite": "tests"})
    return session, dispatcher


# ---- TestAppendFeedback ----


class TestAppendFeedback:
    """Tests for append_feedback."""

    def test_feedback_appended_on_success(self) -> None:
        """Feedback is appended as additional content item after successful tool call."""
        content: list[dict[str, str]] = [{"type": "text", "text": "tool output"}]
        session, _ = _make_session()
        mock_prompt = MagicMock()

        with patch(
            "weakincentives.adapters.acp._guardrails.collect_feedback",
            return_value="<feedback>test feedback</feedback>",
        ):
            append_feedback(
                content,
                is_error=False,
                prompt=mock_prompt,
                session=session,
                deadline=None,
            )

        assert len(content) == 2
        assert content[1]["text"] == "<feedback>test feedback</feedback>"

    def test_no_feedback_when_none_returned(self) -> None:
        """No extra content item when collect_feedback returns None."""
        content: list[dict[str, str]] = [{"type": "text", "text": "output"}]
        session, _ = _make_session()
        mock_prompt = MagicMock()

        with patch(
            "weakincentives.adapters.acp._guardrails.collect_feedback",
            return_value=None,
        ):
            append_feedback(
                content,
                is_error=False,
                prompt=mock_prompt,
                session=session,
                deadline=None,
            )

        assert len(content) == 1

    def test_no_feedback_on_error(self) -> None:
        """Feedback is not collected when is_error is True."""
        content: list[dict[str, str]] = [{"type": "text", "text": "error msg"}]
        session, _ = _make_session()
        mock_prompt = MagicMock()

        with patch(
            "weakincentives.adapters.acp._guardrails.collect_feedback",
        ) as mock_collect:
            append_feedback(
                content,
                is_error=True,
                prompt=mock_prompt,
                session=session,
                deadline=None,
            )
            mock_collect.assert_not_called()

    def test_no_feedback_when_no_prompt(self) -> None:
        """Feedback collection is skipped when prompt is None."""
        content: list[dict[str, str]] = [{"type": "text", "text": "output"}]

        with patch(
            "weakincentives.adapters.acp._guardrails.collect_feedback",
        ) as mock_collect:
            append_feedback(
                content,
                is_error=False,
                prompt=None,
                session=MagicMock(),
                deadline=None,
            )
            mock_collect.assert_not_called()

    def test_no_feedback_when_no_session(self) -> None:
        """Feedback collection is skipped when session is None."""
        content: list[dict[str, str]] = [{"type": "text", "text": "output"}]
        mock_prompt = MagicMock()

        with patch(
            "weakincentives.adapters.acp._guardrails.collect_feedback",
        ) as mock_collect:
            append_feedback(
                content,
                is_error=False,
                prompt=mock_prompt,
                session=None,
                deadline=None,
            )
            mock_collect.assert_not_called()

    def test_feedback_with_deadline(self) -> None:
        """Deadline is passed through to collect_feedback."""
        content: list[dict[str, str]] = [{"type": "text", "text": "output"}]
        session, _ = _make_session()
        mock_prompt = MagicMock()
        clock = FakeClock()
        clock.set_wall(datetime(2024, 1, 1, 12, 0, tzinfo=UTC))
        deadline = Deadline(
            expires_at=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
            clock=clock,
        )

        with patch(
            "weakincentives.adapters.acp._guardrails.collect_feedback",
            return_value=None,
        ) as mock_collect:
            append_feedback(
                content,
                is_error=False,
                prompt=mock_prompt,
                session=session,
                deadline=deadline,
            )
            mock_collect.assert_called_once_with(
                prompt=mock_prompt,
                session=session,
                deadline=deadline,
            )


# ---- TestResolveFilesystem ----


class TestResolveFilesystem:
    """Tests for resolve_filesystem."""

    def test_none_prompt(self) -> None:
        assert resolve_filesystem(None) is None

    def test_prompt_with_filesystem(self) -> None:
        mock_fs = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.resources.get_optional.return_value = mock_fs

        result = resolve_filesystem(mock_prompt)
        assert result is mock_fs

    def test_prompt_without_filesystem(self) -> None:
        mock_prompt = MagicMock()
        mock_prompt.resources.get_optional.return_value = None

        result = resolve_filesystem(mock_prompt)
        assert result is None

    def test_prompt_resources_raises(self) -> None:
        mock_prompt = MagicMock()
        mock_prompt.resources.get_optional.side_effect = RuntimeError("no context")

        result = resolve_filesystem(mock_prompt)
        assert result is None


# ---- TestCheckTaskCompletion ----


class TestCheckTaskCompletion:
    """Tests for check_task_completion."""

    def test_no_prompt(self) -> None:
        session, _ = _make_session()
        should_continue, feedback = check_task_completion(
            prompt=None,
            session=session,
            accumulated_text="text",
            deadline=None,
            budget_tracker=None,
        )
        assert should_continue is False
        assert feedback is None

    def test_no_checker(self) -> None:
        session, _ = _make_session()
        mock_prompt = MagicMock()
        mock_prompt.task_completion_checker = None

        should_continue, feedback = check_task_completion(
            prompt=mock_prompt,
            session=session,
            accumulated_text="text",
            deadline=None,
            budget_tracker=None,
        )
        assert should_continue is False
        assert feedback is None

    def test_complete(self) -> None:
        session, _ = _make_session()
        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.return_value = TaskCompletionResult.ok("All done.")

        mock_prompt = MagicMock()
        mock_prompt.task_completion_checker = mock_checker

        with patch(
            "weakincentives.adapters.acp._guardrails.resolve_filesystem",
            return_value=None,
        ):
            should_continue, feedback = check_task_completion(
                prompt=mock_prompt,
                session=session,
                accumulated_text="output",
                deadline=None,
                budget_tracker=None,
            )
        assert should_continue is False
        assert feedback is None

    def test_incomplete_with_feedback(self) -> None:
        session, _ = _make_session()
        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.return_value = TaskCompletionResult.incomplete(
            "Missing report.md"
        )

        mock_prompt = MagicMock()
        mock_prompt.task_completion_checker = mock_checker

        with patch(
            "weakincentives.adapters.acp._guardrails.resolve_filesystem",
            return_value=None,
        ):
            should_continue, feedback = check_task_completion(
                prompt=mock_prompt,
                session=session,
                accumulated_text="output",
                deadline=None,
                budget_tracker=None,
            )
        assert should_continue is True
        assert feedback == "Missing report.md"

    def test_incomplete_without_feedback(self) -> None:
        """Incomplete with None feedback returns should_continue=False."""
        session, _ = _make_session()
        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.return_value = TaskCompletionResult(
            complete=False, feedback=None
        )

        mock_prompt = MagicMock()
        mock_prompt.task_completion_checker = mock_checker

        with patch(
            "weakincentives.adapters.acp._guardrails.resolve_filesystem",
            return_value=None,
        ):
            should_continue, feedback = check_task_completion(
                prompt=mock_prompt,
                session=session,
                accumulated_text="output",
                deadline=None,
                budget_tracker=None,
            )
        assert should_continue is False
        assert feedback is None

    def test_deadline_exhausted(self) -> None:
        session, _ = _make_session()
        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_prompt = MagicMock()
        mock_prompt.task_completion_checker = mock_checker

        clock = FakeClock()
        anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        clock.set_wall(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=10), clock=clock)
        clock.set_wall(anchor + timedelta(seconds=20))

        should_continue, feedback = check_task_completion(
            prompt=mock_prompt,
            session=session,
            accumulated_text="output",
            deadline=deadline,
            budget_tracker=None,
        )
        assert should_continue is False
        assert feedback is None
        mock_checker.check.assert_not_called()

    def test_budget_exhausted(self) -> None:
        session, _ = _make_session()
        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_prompt = MagicMock()
        mock_prompt.task_completion_checker = mock_checker

        budget = Budget(max_input_tokens=100)
        tracker = BudgetTracker(budget)
        tracker.record_cumulative(
            "eval-1", TokenUsage(input_tokens=200, output_tokens=0)
        )

        should_continue, feedback = check_task_completion(
            prompt=mock_prompt,
            session=session,
            accumulated_text="output",
            deadline=None,
            budget_tracker=tracker,
        )
        assert should_continue is False
        assert feedback is None
        mock_checker.check.assert_not_called()

    def test_filesystem_passed_to_context(self) -> None:
        """Filesystem resolved from prompt is passed to TaskCompletionContext."""
        session, _ = _make_session()
        mock_fs = MagicMock()
        mock_checker = MagicMock(spec=TaskCompletionChecker)
        mock_checker.check.return_value = TaskCompletionResult.ok()

        mock_prompt = MagicMock()
        mock_prompt.task_completion_checker = mock_checker

        with patch(
            "weakincentives.adapters.acp._guardrails.resolve_filesystem",
            return_value=mock_fs,
        ):
            check_task_completion(
                prompt=mock_prompt,
                session=session,
                accumulated_text="output",
                deadline=None,
                budget_tracker=None,
            )

        ctx = mock_checker.check.call_args[0][0]
        assert ctx.filesystem is mock_fs
        assert ctx.session is session
        assert ctx.tentative_output == "output"


# ---- TestAccumulateUsage ----


class TestAccumulateUsage:
    """Tests for accumulate_usage helper."""

    def test_none_current(self) -> None:
        new = TokenUsage(input_tokens=10, output_tokens=5)
        result = accumulate_usage(None, new)
        assert result is new

    def test_sums_all_fields(self) -> None:
        current = TokenUsage(input_tokens=10, output_tokens=5, cached_tokens=2)
        new = TokenUsage(input_tokens=30, output_tokens=15, cached_tokens=3)
        result = accumulate_usage(current, new)
        assert result.input_tokens == 40
        assert result.output_tokens == 20
        assert result.cached_tokens == 5

    def test_none_fields_treated_as_zero(self) -> None:
        current = TokenUsage(input_tokens=None, output_tokens=5)
        new = TokenUsage(input_tokens=10, output_tokens=None)
        result = accumulate_usage(current, new)
        assert result.input_tokens == 10
        assert result.output_tokens == 5
