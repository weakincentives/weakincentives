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

"""Tests for completion handler types and integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from tests.helpers import FrozenUtcNow
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    CompletionContext,
    CompletionResult,
    MarkdownSection,
    Prompt,
    PromptTemplate,
)
from weakincentives.runtime.events._types import TokenUsage

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ReviewResult:
    """Sample output type for testing."""

    summary: str
    findings: list[str]
    has_security_section: bool = False


class TestCompletionResult:
    """Tests for CompletionResult dataclass."""

    def test_complete_result(self) -> None:
        """Test creating a complete result."""
        result = CompletionResult(complete=True)
        assert result.complete is True
        assert result.reason is None

    def test_incomplete_result_with_reason(self) -> None:
        """Test creating an incomplete result with a reason."""
        result = CompletionResult(complete=False, reason="Missing required section")
        assert result.complete is False
        assert result.reason == "Missing required section"

    def test_result_is_frozen(self) -> None:
        """Test that CompletionResult is immutable."""
        result = CompletionResult(complete=True)
        with pytest.raises(AttributeError):
            result.complete = False  # type: ignore[misc]


class TestCompletionContext:
    """Tests for CompletionContext dataclass."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock session."""
        return MagicMock()

    @pytest.fixture
    def mock_prompt(self) -> MagicMock:
        """Create a mock prompt."""
        return MagicMock()

    def test_context_creation(
        self, mock_session: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Test creating a completion context."""
        context = CompletionContext(
            prompt=mock_prompt,
            rendered_prompt=None,
            session=mock_session,
            stop_reason="end_turn",
        )
        assert context.prompt is mock_prompt
        assert context.session is mock_session
        assert context.stop_reason == "end_turn"
        assert context.deadline is None
        assert context.budget_tracker is None

    def test_has_remaining_budget_no_constraints(
        self, mock_session: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Test has_remaining_budget with no deadline or budget."""
        context = CompletionContext(
            prompt=mock_prompt,
            rendered_prompt=None,
            session=mock_session,
            stop_reason=None,
        )
        assert context.has_remaining_budget() is True

    def test_has_remaining_budget_deadline_expired(
        self,
        mock_session: MagicMock,
        mock_prompt: MagicMock,
        frozen_utcnow: FrozenUtcNow,
    ) -> None:
        """Test has_remaining_budget with expired deadline."""
        anchor = datetime.now(UTC)
        frozen_utcnow.set(anchor)
        deadline = Deadline(expires_at=anchor + timedelta(seconds=5))
        frozen_utcnow.advance(timedelta(seconds=10))  # Now expired
        context = CompletionContext(
            prompt=mock_prompt,
            rendered_prompt=None,
            session=mock_session,
            stop_reason=None,
            deadline=deadline,
        )
        assert context.has_remaining_budget() is False

    def test_has_remaining_budget_deadline_active(
        self, mock_session: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Test has_remaining_budget with active deadline."""
        active = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))
        context = CompletionContext(
            prompt=mock_prompt,
            rendered_prompt=None,
            session=mock_session,
            stop_reason=None,
            deadline=active,
        )
        assert context.has_remaining_budget() is True

    def test_has_remaining_budget_tokens_exhausted(
        self, mock_session: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Test has_remaining_budget with exhausted token budget."""
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget=budget)
        # Simulate consuming all tokens
        tracker.record_cumulative(
            "test",
            TokenUsage(input_tokens=600, output_tokens=400, cached_tokens=None),
        )
        context = CompletionContext(
            prompt=mock_prompt,
            rendered_prompt=None,
            session=mock_session,
            stop_reason=None,
            budget_tracker=tracker,
        )
        assert context.has_remaining_budget() is False

    def test_has_remaining_budget_tokens_available(
        self, mock_session: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Test has_remaining_budget with remaining token budget."""
        budget = Budget(max_total_tokens=1000)
        tracker = BudgetTracker(budget=budget)
        # Simulate consuming some tokens
        tracker.record_cumulative(
            "test",
            TokenUsage(input_tokens=300, output_tokens=200, cached_tokens=None),
        )
        context = CompletionContext(
            prompt=mock_prompt,
            rendered_prompt=None,
            session=mock_session,
            stop_reason=None,
            budget_tracker=tracker,
        )
        assert context.has_remaining_budget() is True

    def test_context_is_frozen(
        self, mock_session: MagicMock, mock_prompt: MagicMock
    ) -> None:
        """Test that CompletionContext is immutable."""
        context = CompletionContext(
            prompt=mock_prompt,
            rendered_prompt=None,
            session=mock_session,
            stop_reason=None,
        )
        with pytest.raises(AttributeError):
            context.stop_reason = "modified"  # type: ignore[misc]


class TestPromptCompletionHandler:
    """Tests for Prompt completion handler integration."""

    @pytest.fixture
    def template(self) -> PromptTemplate[ReviewResult]:
        """Create a sample prompt template."""
        return PromptTemplate[ReviewResult](
            ns="test",
            key="review",
            sections=[
                MarkdownSection(
                    title="Task",
                    key="task",
                    template="Review the code.",
                ),
            ],
        )

    def test_prompt_no_completion_handler(
        self, template: PromptTemplate[ReviewResult]
    ) -> None:
        """Test that prompt has no completion handler by default."""
        prompt = Prompt(template)
        assert prompt.completion_handler is None

    def test_with_completion_handler(
        self, template: PromptTemplate[ReviewResult]
    ) -> None:
        """Test setting a completion handler."""

        def handler(
            output: ReviewResult, *, context: CompletionContext
        ) -> CompletionResult:
            return CompletionResult(complete=True)

        prompt = Prompt(template).with_completion_handler(handler)
        assert prompt.completion_handler is handler

    def test_with_completion_handler_returns_self(
        self, template: PromptTemplate[ReviewResult]
    ) -> None:
        """Test that with_completion_handler returns self for chaining."""

        def handler(
            output: ReviewResult, *, context: CompletionContext
        ) -> CompletionResult:
            return CompletionResult(complete=True)

        prompt = Prompt(template)
        result = prompt.with_completion_handler(handler)
        assert result is prompt

    def test_check_completion_no_handler(
        self, template: PromptTemplate[ReviewResult]
    ) -> None:
        """Test check_completion returns complete when no handler."""
        prompt = Prompt(template)
        mock_context = MagicMock(spec=CompletionContext)
        output = ReviewResult(summary="test", findings=["finding1"])

        result = prompt.check_completion(output, context=mock_context)
        assert result.complete is True

    def test_check_completion_with_handler_complete(
        self, template: PromptTemplate[ReviewResult]
    ) -> None:
        """Test check_completion calls handler and returns result."""

        def handler(
            output: ReviewResult, *, context: CompletionContext
        ) -> CompletionResult:
            return CompletionResult(complete=True)

        prompt = Prompt(template).with_completion_handler(handler)
        mock_context = MagicMock(spec=CompletionContext)
        output = ReviewResult(summary="test", findings=["finding1"])

        result = prompt.check_completion(output, context=mock_context)
        assert result.complete is True

    def test_check_completion_with_handler_incomplete(
        self, template: PromptTemplate[ReviewResult]
    ) -> None:
        """Test check_completion returns incomplete result."""

        def handler(
            output: ReviewResult, *, context: CompletionContext
        ) -> CompletionResult:
            if not output.has_security_section:
                return CompletionResult(
                    complete=False, reason="Missing security section"
                )
            return CompletionResult(complete=True)

        prompt = Prompt(template).with_completion_handler(handler)
        mock_context = MagicMock(spec=CompletionContext)
        output = ReviewResult(
            summary="test", findings=["finding1"], has_security_section=False
        )

        result = prompt.check_completion(output, context=mock_context)
        assert result.complete is False
        assert result.reason == "Missing security section"

    def test_method_chaining(self, template: PromptTemplate[ReviewResult]) -> None:
        """Test that bind and with_completion_handler can be chained."""

        @dataclass(frozen=True)
        class ReviewParams:
            target: str

        def handler(
            output: ReviewResult, *, context: CompletionContext
        ) -> CompletionResult:
            return CompletionResult(complete=True)

        prompt = (
            Prompt(template)
            .bind(ReviewParams(target="module"))
            .with_completion_handler(handler)
        )

        assert prompt.completion_handler is handler
        assert len(prompt.params) == 1
