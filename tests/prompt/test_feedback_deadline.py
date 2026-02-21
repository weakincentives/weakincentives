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

"""Tests for DeadlineFeedback provider and _format_duration helper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    DeadlineFeedback,
    FeedbackContext,
    Prompt,
    PromptTemplate,
)
from weakincentives.runtime import InProcessDispatcher, Session


def make_session() -> Session:
    """Create a session for testing."""
    return Session(dispatcher=InProcessDispatcher())


def make_prompt() -> Prompt[None]:
    """Create a minimal prompt for testing."""
    template: PromptTemplate[None] = PromptTemplate(
        ns="test", key="test-prompt", name="test"
    )
    return Prompt(template)


# =============================================================================
# DeadlineFeedback Tests
# =============================================================================


class TestDeadlineFeedback:
    """Tests for the built-in DeadlineFeedback provider."""

    def _make_context(self, deadline: Deadline | None = None) -> FeedbackContext:
        session = make_session()
        prompt = make_prompt()
        return FeedbackContext(session=session, prompt=prompt, deadline=deadline)

    def test_name_property(self) -> None:
        provider = DeadlineFeedback()
        assert provider.name == "Deadline"

    def test_should_run_returns_false_without_deadline(self) -> None:
        provider = DeadlineFeedback()
        context = self._make_context(deadline=None)

        assert provider.should_run(context=context) is False

    def test_should_run_returns_true_with_deadline(self) -> None:
        provider = DeadlineFeedback()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))
        context = self._make_context(deadline=deadline)

        assert provider.should_run(context=context) is True

    def test_provide_returns_info_when_plenty_of_time(self) -> None:
        provider = DeadlineFeedback(warning_threshold_seconds=120)
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=2))
        context = self._make_context(deadline=deadline)

        feedback = provider.provide(context=context)

        assert feedback.provider_name == "Deadline"
        assert feedback.severity == "info"
        assert feedback.suggestions == ()
        # Check for both elapsed and remaining time in message
        assert "the work so far took" in feedback.summary.lower()
        assert "remaining to complete" in feedback.summary.lower()
        assert "hour" in feedback.summary.lower()

    def test_provide_returns_warning_when_low_time(self) -> None:
        provider = DeadlineFeedback(warning_threshold_seconds=120)
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=90))
        context = self._make_context(deadline=deadline)

        feedback = provider.provide(context=context)

        assert feedback.severity == "warning"
        assert len(feedback.suggestions) > 0
        assert "the work so far took" in feedback.summary.lower()
        assert "remaining to complete" in feedback.summary.lower()

    def test_provide_returns_warning_when_deadline_passed(self) -> None:
        provider = DeadlineFeedback()
        session = make_session()
        prompt = make_prompt()
        # Mock deadline with negative remaining time and elapsed time
        mock_deadline = type(
            "MockDeadline",
            (),
            {
                "remaining": lambda self: timedelta(seconds=-1),
                "elapsed": lambda self: timedelta(minutes=20),
            },
        )()
        mock_context = FeedbackContext(
            session=session,
            prompt=prompt,
            deadline=mock_deadline,  # type: ignore[arg-type]
        )

        feedback = provider.provide(context=mock_context)

        assert feedback.severity == "warning"
        assert "deadline" in feedback.summary.lower()
        assert "20 minutes" in feedback.summary.lower()

    def test_custom_warning_threshold(self) -> None:
        provider = DeadlineFeedback(warning_threshold_seconds=300)
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=240))
        context = self._make_context(deadline=deadline)

        feedback = provider.provide(context=context)

        assert feedback.severity == "warning"

    def test_provide_without_deadline_raises(self) -> None:
        provider = DeadlineFeedback()
        context = self._make_context(deadline=None)

        with pytest.raises(ValueError, match="requires a deadline"):
            provider.provide(context=context)


class TestFormatDuration:
    """Tests for the _format_duration helper."""

    def test_format_seconds(self) -> None:
        from weakincentives.prompt.feedback_providers import _format_duration

        assert _format_duration(45) == "45 seconds"

    def test_format_minute_singular(self) -> None:
        from weakincentives.prompt.feedback_providers import _format_duration

        assert _format_duration(60) == "1 minute"

    def test_format_minutes_plural(self) -> None:
        from weakincentives.prompt.feedback_providers import _format_duration

        assert _format_duration(300) == "5 minutes"

    def test_format_hours(self) -> None:
        from weakincentives.prompt.feedback_providers import _format_duration

        assert "hour" in _format_duration(7200)
