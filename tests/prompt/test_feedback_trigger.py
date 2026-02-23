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

"""Tests for FeedbackTrigger, run_feedback_providers, and DeadlineFeedback.

This module complements test_feedback.py and tests:
- FeedbackTrigger conditions (_should_trigger helper)
- run_feedback_providers runner logic
- Built-in DeadlineFeedback provider
- PromptTemplate/Prompt integration
- Feedback session storage

FileCreatedTrigger and StaticFeedbackProvider tests are in
test_feedback_file_created.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    DeadlineFeedback,
    Feedback,
    FeedbackContext,
    FeedbackProviderConfig,
    FeedbackTrigger,
    Prompt,
    PromptTemplate,
    run_feedback_providers,
)
from weakincentives.prompt.feedback import _should_trigger
from weakincentives.runtime import InProcessDispatcher, Session
from weakincentives.runtime.events import ToolInvoked

# =============================================================================
# Shared helpers (duplicated from test_feedback.py for module independence)
# =============================================================================


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


@dataclass(frozen=True)
class MockFeedbackProvider:
    """Test provider with configurable behavior."""

    should_run_return: bool = True
    feedback_summary: str = "Mock feedback"
    provider_name: str = "Mock"

    @property
    def name(self) -> str:
        return self.provider_name

    def should_run(self, *, context: FeedbackContext) -> bool:
        return self.should_run_return

    def provide(self, *, context: FeedbackContext) -> Feedback:
        return Feedback(provider_name=self.name, summary=self.feedback_summary)


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


# =============================================================================
# run_feedback_providers Tests
# =============================================================================


class TestRunFeedbackProviders:
    """Tests for the run_feedback_providers runner function."""

    def _make_context(self, tool_calls: int = 0) -> FeedbackContext:
        session = make_session()
        prompt = make_prompt()

        for i in range(tool_calls):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        return FeedbackContext(session=session, prompt=prompt)

    def test_returns_none_when_no_providers(self) -> None:
        context = self._make_context(tool_calls=5)

        result = run_feedback_providers(providers=(), context=context)

        assert result is None

    def test_returns_none_when_trigger_not_met(self) -> None:
        context = self._make_context(tool_calls=2)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(),
            trigger=FeedbackTrigger(every_n_calls=10),
        )

        result = run_feedback_providers(providers=(config,), context=context)

        assert result is None

    def test_returns_none_when_should_run_false(self) -> None:
        context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(should_run_return=False),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        result = run_feedback_providers(providers=(config,), context=context)

        assert result is None

    def test_returns_rendered_feedback_when_triggered(self) -> None:
        context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(feedback_summary="Test feedback"),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        result = run_feedback_providers(providers=(config,), context=context)

        assert result is not None
        assert "Test feedback" in result
        assert "<feedback provider='Mock'>" in result

    def test_stores_feedback_in_session(self) -> None:
        context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        run_feedback_providers(providers=(config,), context=context)

        latest = context.session[Feedback].latest()
        assert latest is not None
        assert latest.provider_name == "Mock"
        assert latest.call_index == 5  # Updated to current count

    def test_all_matching_providers_collected(self) -> None:
        context = self._make_context(tool_calls=5)
        configs = (
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="First"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="Second"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
        )

        result = run_feedback_providers(providers=configs, context=context)

        assert result is not None
        assert "First" in result
        assert "Second" in result
        # Verify both are rendered as separate blocks
        assert result.count("<feedback provider='Mock'>") == 2
        assert result.count("</feedback>") == 2

    def test_all_matching_feedback_stored_in_session(self) -> None:
        context = self._make_context(tool_calls=5)
        configs = (
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="First"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="Second"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
        )

        run_feedback_providers(providers=configs, context=context)

        all_feedback = context.session[Feedback].all()
        assert len(all_feedback) == 2
        summaries = {fb.summary for fb in all_feedback}
        assert "First" in summaries
        assert "Second" in summaries

    def test_only_matching_providers_included(self) -> None:
        context = self._make_context(tool_calls=5)
        configs = (
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="Matches"),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(
                    feedback_summary="Skipped", should_run_return=False
                ),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(feedback_summary="No trigger"),
                trigger=FeedbackTrigger(every_n_calls=100),  # Won't trigger
            ),
        )

        result = run_feedback_providers(providers=configs, context=context)

        assert result is not None
        assert "Matches" in result
        assert "Skipped" not in result
        assert "No trigger" not in result

    def test_providers_maintain_independent_trigger_cadences(self) -> None:
        """Each provider tracks its own feedback history for triggers."""
        session = make_session()
        prompt = make_prompt()

        # Provider A triggers every 3 calls, Provider B triggers every 5 calls
        configs = (
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(
                    provider_name="ProviderA", feedback_summary="From A"
                ),
                trigger=FeedbackTrigger(every_n_calls=3),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(
                    provider_name="ProviderB", feedback_summary="From B"
                ),
                trigger=FeedbackTrigger(every_n_calls=5),
            ),
        )

        # Simulate tool calls and run feedback collection
        def run_at_call_count(n: int) -> str | None:
            # Add tool calls to reach count n
            while len(session[ToolInvoked].all()) < n:
                session.dispatcher.dispatch(make_tool_invoked("tool"))
            context = FeedbackContext(session=session, prompt=prompt)
            return run_feedback_providers(providers=configs, context=context)

        # At call 3: A triggers (3 calls since start), B doesn't (only 3, needs 5)
        result = run_at_call_count(3)
        assert result is not None
        assert "From A" in result
        assert "From B" not in result

        # At call 5: B triggers (5 calls since start), A doesn't (only 2 since last)
        result = run_at_call_count(5)
        assert result is not None
        assert "From A" not in result
        assert "From B" in result

        # At call 6: A triggers (3 calls since last at 3), B doesn't (1 since last)
        result = run_at_call_count(6)
        assert result is not None
        assert "From A" in result
        assert "From B" not in result

        # At call 9: A triggers (3 calls since last at 6), B doesn't (4 since last)
        result = run_at_call_count(9)
        assert result is not None
        assert "From A" in result
        assert "From B" not in result

        # At call 10: B triggers (5 calls since last at 5), A doesn't (1 since last)
        result = run_at_call_count(10)
        assert result is not None
        assert "From A" not in result
        assert "From B" in result


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


# =============================================================================
# PromptTemplate Integration Tests
# =============================================================================


class TestPromptTemplateFeedbackProviders:
    """Tests for feedback providers on PromptTemplate."""

    def test_creates_template_without_providers(self) -> None:
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        assert template.feedback_providers == ()

    def test_creates_template_with_providers(self) -> None:
        provider = DeadlineFeedback()
        config = FeedbackProviderConfig(
            provider=provider, trigger=FeedbackTrigger(every_n_seconds=30)
        )
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", feedback_providers=[config]
        )

        assert len(template.feedback_providers) == 1
        assert template.feedback_providers[0].provider is provider

    def test_providers_converted_to_tuple(self) -> None:
        configs = [
            FeedbackProviderConfig(
                provider=DeadlineFeedback(),
                trigger=FeedbackTrigger(every_n_seconds=30),
            ),
            FeedbackProviderConfig(
                provider=MockFeedbackProvider(),
                trigger=FeedbackTrigger(every_n_calls=10),
            ),
        ]
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", feedback_providers=configs
        )

        assert isinstance(template.feedback_providers, tuple)
        assert len(template.feedback_providers) == 2


class TestPromptFeedbackProviders:
    """Tests for feedback providers on Prompt."""

    def test_prompt_exposes_template_providers(self) -> None:
        provider = DeadlineFeedback()
        config = FeedbackProviderConfig(
            provider=provider, trigger=FeedbackTrigger(every_n_seconds=30)
        )
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", feedback_providers=[config]
        )
        prompt: Prompt[None] = Prompt(template)

        assert len(prompt.feedback_providers) == 1
        assert prompt.feedback_providers[0].provider is provider

    def test_prompt_providers_returns_tuple(self) -> None:
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        prompt: Prompt[None] = Prompt(template)

        assert isinstance(prompt.feedback_providers, tuple)


# =============================================================================
# Feedback Session Storage Tests
# =============================================================================


class TestFeedbackSessionStorage:
    """Tests for Feedback storage in session slices."""

    def test_feedback_stored_in_session_slice(self) -> None:
        session = make_session()

        session[Feedback].append(Feedback(provider_name="A", summary="First"))

        stored = session[Feedback].latest()
        assert stored is not None
        assert stored.provider_name == "A"

    def test_multiple_feedback_stored(self) -> None:
        session = make_session()

        session[Feedback].append(Feedback(provider_name="A", summary="First"))
        session[Feedback].append(Feedback(provider_name="B", summary="Second"))

        all_feedback = session[Feedback].all()
        assert len(all_feedback) == 2
