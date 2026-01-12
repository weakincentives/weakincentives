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

"""Tests for feedback providers.

This module tests the feedback provider system including:
- Core data types (Observation, Feedback)
- FeedbackContext and its helper methods
- FeedbackTrigger conditions
- Provider runner logic
- Built-in DeadlineFeedback provider
- PromptTemplate integration
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
    Observation,
    Prompt,
    PromptResources,
    PromptTemplate,
    run_feedback_providers,
)
from weakincentives.prompt.feedback import _should_trigger
from weakincentives.runtime import InProcessDispatcher, Session
from weakincentives.runtime.events import ToolInvoked

# =============================================================================
# Test Fixtures
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

    @property
    def name(self) -> str:
        return "Mock"

    def should_run(self, session: Session, *, context: FeedbackContext) -> bool:
        return self.should_run_return

    def provide(self, session: Session, *, context: FeedbackContext) -> Feedback:
        return Feedback(provider_name=self.name, summary=self.feedback_summary)


# =============================================================================
# Observation Tests
# =============================================================================


class TestObservation:
    """Tests for the Observation data type."""

    def test_creates_basic_observation(self) -> None:
        obs = Observation(category="Pattern", description="Repeated failures")

        assert obs.category == "Pattern"
        assert obs.description == "Repeated failures"
        assert obs.evidence is None

    def test_creates_observation_with_evidence(self) -> None:
        obs = Observation(
            category="Loop",
            description="Same file read 3 times",
            evidence="file.txt",
        )

        assert obs.evidence == "file.txt"

    def test_observation_is_frozen(self) -> None:
        obs = Observation(category="Test", description="Test")

        with pytest.raises(AttributeError):
            obs.category = "Changed"  # type: ignore[misc]


# =============================================================================
# Feedback Tests
# =============================================================================


class TestFeedback:
    """Tests for the Feedback data type."""

    def test_creates_basic_feedback(self) -> None:
        feedback = Feedback(provider_name="Test", summary="All good")

        assert feedback.provider_name == "Test"
        assert feedback.summary == "All good"
        assert feedback.observations == ()
        assert feedback.suggestions == ()
        assert feedback.severity == "info"
        assert feedback.call_index == 0

    def test_creates_feedback_with_observations(self) -> None:
        obs = Observation(category="Pattern", description="Loop detected")
        feedback = Feedback(
            provider_name="Loop",
            summary="Possible loop",
            observations=(obs,),
        )

        assert len(feedback.observations) == 1
        assert feedback.observations[0].category == "Pattern"

    def test_creates_feedback_with_suggestions(self) -> None:
        feedback = Feedback(
            provider_name="Time",
            summary="Running low on time",
            suggestions=("Wrap up soon", "Summarize progress"),
        )

        assert len(feedback.suggestions) == 2

    def test_severity_levels(self) -> None:
        info = Feedback(provider_name="A", summary="Info", severity="info")
        caution = Feedback(provider_name="B", summary="Caution", severity="caution")
        warning = Feedback(provider_name="C", summary="Warning", severity="warning")

        assert info.severity == "info"
        assert caution.severity == "caution"
        assert warning.severity == "warning"


class TestFeedbackRender:
    """Tests for Feedback.render() method."""

    def test_render_basic_feedback(self) -> None:
        feedback = Feedback(provider_name="Test", summary="Status check")
        rendered = feedback.render()

        assert "[Feedback - Test]" in rendered
        assert "Status check" in rendered

    def test_render_with_observations(self) -> None:
        obs = Observation(category="Pattern", description="Loop detected")
        feedback = Feedback(
            provider_name="Loop",
            summary="Possible loop",
            observations=(obs,),
        )
        rendered = feedback.render()

        assert "• Pattern: Loop detected" in rendered

    def test_render_with_suggestions(self) -> None:
        feedback = Feedback(
            provider_name="Time",
            summary="Low time",
            suggestions=("Wrap up", "Summarize"),
        )
        rendered = feedback.render()

        assert "→ Wrap up" in rendered
        assert "→ Summarize" in rendered

    def test_render_full_feedback(self) -> None:
        obs1 = Observation(category="Files", description="10 files read")
        obs2 = Observation(category="Time", description="5 minutes elapsed")
        feedback = Feedback(
            provider_name="Progress",
            summary="Making progress",
            observations=(obs1, obs2),
            suggestions=("Continue current approach",),
            severity="info",
        )
        rendered = feedback.render()

        assert "[Feedback - Progress]" in rendered
        assert "Making progress" in rendered
        assert "• Files: 10 files read" in rendered
        assert "• Time: 5 minutes elapsed" in rendered
        assert "→ Continue current approach" in rendered


# =============================================================================
# FeedbackContext Tests
# =============================================================================


class TestFeedbackContext:
    """Tests for the FeedbackContext class."""

    def test_creates_context_with_basic_fields(self) -> None:
        session = make_session()
        prompt = make_prompt()

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.session is session
        assert context.prompt is prompt
        assert context.deadline is None

    def test_creates_context_with_deadline(self) -> None:
        session = make_session()
        prompt = make_prompt()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))

        context = FeedbackContext(session=session, prompt=prompt, deadline=deadline)

        assert context.deadline is deadline

    def test_resources_returns_prompt_resources(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert isinstance(context.resources, PromptResources)

    def test_filesystem_returns_none_when_not_registered(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        with prompt.resources:
            assert context.filesystem is None

    def test_last_feedback_returns_none_when_empty(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert context.last_feedback is None

    def test_last_feedback_returns_most_recent(self) -> None:
        session = make_session()
        prompt = make_prompt()

        session[Feedback].append(Feedback(provider_name="A", summary="First"))
        session[Feedback].append(Feedback(provider_name="B", summary="Second"))

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.last_feedback is not None
        assert context.last_feedback.summary == "Second"

    def test_tool_call_count_returns_zero_when_empty(self) -> None:
        session = make_session()
        prompt = make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_call_count == 0

    def test_tool_call_count_returns_correct_count(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(3):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_call_count == 3

    def test_tool_calls_since_last_feedback_all_when_no_feedback(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(5):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_calls_since_last_feedback() == 5

    def test_tool_calls_since_last_feedback_counts_after_last(self) -> None:
        session = make_session()
        prompt = make_prompt()

        # 3 tool calls, then feedback, then 2 more
        for i in range(3):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))
        session[Feedback].append(
            Feedback(provider_name="A", summary="Test", call_index=3)
        )
        for i in range(2):
            session.dispatcher.dispatch(make_tool_invoked(f"more_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)

        # 5 total - 3 at last feedback = 2 since
        assert context.tool_calls_since_last_feedback() == 2

    def test_recent_tool_calls_returns_last_n(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(5):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(3)

        assert len(recent) == 3
        assert recent[0].name == "tool_2"
        assert recent[1].name == "tool_3"
        assert recent[2].name == "tool_4"

    def test_recent_tool_calls_returns_all_when_fewer_than_n(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(2):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(5)

        assert len(recent) == 2

    def test_recent_tool_calls_returns_empty_when_n_is_zero(self) -> None:
        session = make_session()
        prompt = make_prompt()

        for i in range(3):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(0)

        assert len(recent) == 0

    def test_recent_tool_calls_returns_empty_when_no_tool_invoked_events(self) -> None:
        session = make_session()
        prompt = make_prompt()

        context = FeedbackContext(session=session, prompt=prompt)
        recent = context.recent_tool_calls(5)

        assert len(recent) == 0


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
                )
            )

        return FeedbackContext(session=session, prompt=prompt)

    def test_returns_false_for_empty_trigger(self) -> None:
        trigger = FeedbackTrigger()
        context = self._make_context(tool_calls=10)

        assert _should_trigger(trigger, context) is False

    def test_call_count_trigger_fires_when_threshold_met(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=5)
        context = self._make_context(tool_calls=5)

        assert _should_trigger(trigger, context) is True

    def test_call_count_trigger_does_not_fire_below_threshold(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=5)
        context = self._make_context(tool_calls=3)

        assert _should_trigger(trigger, context) is False

    def test_call_count_trigger_counts_since_last_feedback(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=3)
        # 5 total, last at 3, so only 2 since
        context = self._make_context(tool_calls=5, last_feedback_call_index=3)

        assert _should_trigger(trigger, context) is False

    def test_time_trigger_fires_when_no_previous_feedback(self) -> None:
        trigger = FeedbackTrigger(every_n_seconds=30)
        context = self._make_context(tool_calls=1)

        assert _should_trigger(trigger, context) is True

    def test_time_trigger_fires_when_time_elapsed(self) -> None:
        trigger = FeedbackTrigger(every_n_seconds=30)
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        context = self._make_context(
            tool_calls=1,
            last_feedback_call_index=0,
            last_feedback_timestamp=old_time,
        )

        assert _should_trigger(trigger, context) is True

    def test_time_trigger_does_not_fire_when_too_recent(self) -> None:
        trigger = FeedbackTrigger(every_n_seconds=30)
        recent_time = datetime.now(UTC) - timedelta(seconds=10)
        context = self._make_context(
            tool_calls=1,
            last_feedback_call_index=0,
            last_feedback_timestamp=recent_time,
        )

        assert _should_trigger(trigger, context) is False

    def test_or_logic_fires_on_call_count_when_time_not_met(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=3, every_n_seconds=300)
        recent_time = datetime.now(UTC) - timedelta(seconds=10)
        context = self._make_context(
            tool_calls=5,
            last_feedback_call_index=1,
            last_feedback_timestamp=recent_time,
        )
        # 5 - 1 = 4 calls since last >= 3 threshold

        assert _should_trigger(trigger, context) is True

    def test_or_logic_fires_on_time_when_calls_not_met(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=10, every_n_seconds=30)
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        context = self._make_context(
            tool_calls=3,
            last_feedback_call_index=2,
            last_feedback_timestamp=old_time,
        )
        # Only 1 call since, but time elapsed

        assert _should_trigger(trigger, context) is True


# =============================================================================
# run_feedback_providers Tests
# =============================================================================


class TestRunFeedbackProviders:
    """Tests for the run_feedback_providers runner function."""

    def _make_context(self, tool_calls: int = 0) -> tuple[Session, FeedbackContext]:
        session = make_session()
        prompt = make_prompt()

        for i in range(tool_calls):
            session.dispatcher.dispatch(make_tool_invoked(f"tool_{i}"))

        return session, FeedbackContext(session=session, prompt=prompt)

    def test_returns_none_when_no_providers(self) -> None:
        session, context = self._make_context(tool_calls=5)

        result = run_feedback_providers(providers=(), context=context, session=session)

        assert result is None

    def test_returns_none_when_trigger_not_met(self) -> None:
        session, context = self._make_context(tool_calls=2)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(),
            trigger=FeedbackTrigger(every_n_calls=10),
        )

        result = run_feedback_providers(
            providers=(config,), context=context, session=session
        )

        assert result is None

    def test_returns_none_when_should_run_false(self) -> None:
        session, context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(should_run_return=False),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        result = run_feedback_providers(
            providers=(config,), context=context, session=session
        )

        assert result is None

    def test_returns_rendered_feedback_when_triggered(self) -> None:
        session, context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(feedback_summary="Test feedback"),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        result = run_feedback_providers(
            providers=(config,), context=context, session=session
        )

        assert result is not None
        assert "Test feedback" in result
        assert "[Feedback - Mock]" in result

    def test_stores_feedback_in_session(self) -> None:
        session, context = self._make_context(tool_calls=5)
        config = FeedbackProviderConfig(
            provider=MockFeedbackProvider(),
            trigger=FeedbackTrigger(every_n_calls=3),
        )

        run_feedback_providers(providers=(config,), context=context, session=session)

        latest = session[Feedback].latest()
        assert latest is not None
        assert latest.provider_name == "Mock"
        assert latest.call_index == 5  # Updated to current count

    def test_first_matching_provider_wins(self) -> None:
        session, context = self._make_context(tool_calls=5)
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

        result = run_feedback_providers(
            providers=configs, context=context, session=session
        )

        assert result is not None
        assert "First" in result
        assert "Second" not in result


# =============================================================================
# DeadlineFeedback Tests
# =============================================================================


class TestDeadlineFeedback:
    """Tests for the built-in DeadlineFeedback provider."""

    def _make_context(
        self, deadline: Deadline | None = None
    ) -> tuple[Session, FeedbackContext]:
        session = make_session()
        prompt = make_prompt()
        return session, FeedbackContext(
            session=session, prompt=prompt, deadline=deadline
        )

    def test_name_property(self) -> None:
        provider = DeadlineFeedback()
        assert provider.name == "Deadline"

    def test_should_run_returns_false_without_deadline(self) -> None:
        provider = DeadlineFeedback()
        session, context = self._make_context(deadline=None)

        assert provider.should_run(session, context=context) is False

    def test_should_run_returns_true_with_deadline(self) -> None:
        provider = DeadlineFeedback()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))
        session, context = self._make_context(deadline=deadline)

        assert provider.should_run(session, context=context) is True

    def test_provide_returns_info_when_plenty_of_time(self) -> None:
        provider = DeadlineFeedback(warning_threshold_seconds=120)
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=2))
        session, context = self._make_context(deadline=deadline)

        feedback = provider.provide(session, context=context)

        assert feedback.provider_name == "Deadline"
        assert feedback.severity == "info"
        assert feedback.suggestions == ()
        assert "hour" in feedback.summary.lower()

    def test_provide_returns_warning_when_low_time(self) -> None:
        provider = DeadlineFeedback(warning_threshold_seconds=120)
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=90))
        session, context = self._make_context(deadline=deadline)

        feedback = provider.provide(session, context=context)

        assert feedback.severity == "warning"
        assert len(feedback.suggestions) > 0
        assert "remaining" in feedback.summary.lower()

    def test_provide_returns_warning_when_deadline_passed(self) -> None:
        provider = DeadlineFeedback()
        # Mock deadline with negative remaining time
        mock_deadline = type(
            "MockDeadline", (), {"remaining": lambda self: timedelta(seconds=-1)}
        )()
        session, context = self._make_context(deadline=None)
        mock_context = FeedbackContext(
            session=session,
            prompt=context.prompt,
            deadline=mock_deadline,  # type: ignore[arg-type]
        )

        feedback = provider.provide(session, context=mock_context)

        assert feedback.severity == "warning"
        assert "deadline" in feedback.summary.lower()

    def test_provide_without_deadline_returns_info(self) -> None:
        provider = DeadlineFeedback()
        session, context = self._make_context(deadline=None)

        feedback = provider.provide(session, context=context)

        assert feedback.severity == "info"
        assert "no deadline" in feedback.summary.lower()

    def test_custom_warning_threshold(self) -> None:
        provider = DeadlineFeedback(warning_threshold_seconds=300)
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=240))
        session, context = self._make_context(deadline=deadline)

        feedback = provider.provide(session, context=context)

        assert feedback.severity == "warning"


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
