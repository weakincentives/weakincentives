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

"""Tests for FeedbackTrigger conditions and _should_trigger logic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from weakincentives.prompt import (
    Feedback,
    FeedbackContext,
    FeedbackTrigger,
    Prompt,
    PromptTemplate,
)
from weakincentives.prompt.feedback import _should_trigger
from weakincentives.runtime import InProcessDispatcher, Session
from weakincentives.runtime.events import ToolInvoked


def make_session() -> Session:
    """Create a session for testing."""
    return Session(dispatcher=InProcessDispatcher())


def make_prompt() -> Prompt[None]:
    """Create a minimal prompt for testing."""
    template: PromptTemplate[None] = PromptTemplate(
        ns="test", key="test-prompt", name="test"
    )
    return Prompt(template)


def make_tool_invoked(name: str = "test_tool") -> ToolInvoked:
    """Create a ToolInvoked event for testing."""
    return ToolInvoked(
        prompt_name="test",
        adapter="test",
        name=name,
        params=None,
        result=None,
        session_id=None,
        created_at=datetime.now(UTC),
    )


# =============================================================================
# FeedbackTrigger Tests
# =============================================================================


class TestFeedbackTrigger:
    """Tests for FeedbackTrigger configuration."""

    def test_creates_trigger_with_call_count(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=10)

        assert trigger.every_n_calls == 10
        assert trigger.every_n_seconds is None

    def test_creates_trigger_with_time_interval(self) -> None:
        trigger = FeedbackTrigger(every_n_seconds=30.0)

        assert trigger.every_n_calls is None
        assert trigger.every_n_seconds == 30.0

    def test_creates_trigger_with_both_conditions(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=10, every_n_seconds=60.0)

        assert trigger.every_n_calls == 10
        assert trigger.every_n_seconds == 60.0


class TestShouldTrigger:
    """Tests for the _should_trigger helper function."""

    def _make_context(
        self,
        tool_calls: int = 0,
        last_feedback_call_index: int | None = None,
        last_feedback_timestamp: datetime | None = None,
    ) -> FeedbackContext:
        session = make_session()
        prompt = make_prompt()

        for i in range(tool_calls):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        if last_feedback_call_index is not None:
            ts = last_feedback_timestamp or datetime.now(UTC)
            session[Feedback].append(
                Feedback(
                    provider_name="A",
                    summary="Test",
                    call_index=last_feedback_call_index,
                    timestamp=ts,
                    prompt_name="test",  # Match make_prompt's name
                )
            )

        return FeedbackContext(session=session, prompt=prompt)

    def test_returns_false_for_empty_trigger(self) -> None:
        trigger = FeedbackTrigger()
        context = self._make_context(tool_calls=10)

        assert _should_trigger(trigger, context, "A") is False

    def test_call_count_trigger_fires_when_threshold_met(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=5)
        context = self._make_context(tool_calls=5)

        assert _should_trigger(trigger, context, "A") is True

    def test_call_count_trigger_does_not_fire_below_threshold(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=5)
        context = self._make_context(tool_calls=3)

        assert _should_trigger(trigger, context, "A") is False

    def test_call_count_trigger_counts_since_last_feedback(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=3)
        # 5 total, last at 3, so only 2 since
        context = self._make_context(tool_calls=5, last_feedback_call_index=3)

        assert _should_trigger(trigger, context, "A") is False

    def test_time_trigger_fires_when_no_previous_feedback(self) -> None:
        trigger = FeedbackTrigger(every_n_seconds=30)
        context = self._make_context(tool_calls=1)

        assert _should_trigger(trigger, context, "A") is True

    def test_time_trigger_fires_when_time_elapsed(self) -> None:
        trigger = FeedbackTrigger(every_n_seconds=30)
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        context = self._make_context(
            tool_calls=1,
            last_feedback_call_index=0,
            last_feedback_timestamp=old_time,
        )

        assert _should_trigger(trigger, context, "A") is True

    def test_time_trigger_does_not_fire_when_too_recent(self) -> None:
        trigger = FeedbackTrigger(every_n_seconds=30)
        recent_time = datetime.now(UTC) - timedelta(seconds=10)
        context = self._make_context(
            tool_calls=1,
            last_feedback_call_index=0,
            last_feedback_timestamp=recent_time,
        )

        assert _should_trigger(trigger, context, "A") is False

    def test_or_logic_fires_on_call_count_when_time_not_met(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=3, every_n_seconds=300)
        recent_time = datetime.now(UTC) - timedelta(seconds=10)
        context = self._make_context(
            tool_calls=5,
            last_feedback_call_index=1,
            last_feedback_timestamp=recent_time,
        )
        # 5 - 1 = 4 calls since last >= 3 threshold

        assert _should_trigger(trigger, context, "A") is True

    def test_or_logic_fires_on_time_when_calls_not_met(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=10, every_n_seconds=30)
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        context = self._make_context(
            tool_calls=3,
            last_feedback_call_index=2,
            last_feedback_timestamp=old_time,
        )
        # Only 1 call since, but time elapsed

        assert _should_trigger(trigger, context, "A") is True

    def test_trigger_scoped_to_provider(self) -> None:
        """Trigger state is independent per provider."""
        trigger = FeedbackTrigger(every_n_calls=3)
        # Feedback from provider "A" at call 3
        context = self._make_context(tool_calls=5, last_feedback_call_index=3)

        # Provider "A" should NOT trigger (5 - 3 = 2 calls since)
        assert _should_trigger(trigger, context, "A") is False
        # Provider "B" should trigger (no feedback from B, so 5 calls since start)
        assert _should_trigger(trigger, context, "B") is True
