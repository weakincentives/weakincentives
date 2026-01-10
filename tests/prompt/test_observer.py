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

"""Tests for feedback providers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

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
    PromptTemplate,
    RecordFeedback,
    run_feedback_providers,
)
from weakincentives.prompt.observer import _should_trigger
from weakincentives.runtime import InProcessDispatcher, Session
from weakincentives.runtime.events import ToolInvoked

if TYPE_CHECKING:
    pass


# --- Observation Tests ---


class TestObservation:
    def test_creates_basic_observation(self) -> None:
        obs = Observation(category="Pattern", description="Repeated failures")
        assert obs.category == "Pattern"
        assert obs.description == "Repeated failures"
        assert obs.evidence is None

    def test_creates_observation_with_evidence(self) -> None:
        obs = Observation(
            category="Loop", description="Same file read 3 times", evidence="file.txt"
        )
        assert obs.category == "Loop"
        assert obs.description == "Same file read 3 times"
        assert obs.evidence == "file.txt"

    def test_observation_is_frozen(self) -> None:
        obs = Observation(category="Test", description="Test")
        with pytest.raises(AttributeError):
            obs.category = "Changed"  # type: ignore[misc]


# --- Feedback Tests ---


class TestFeedback:
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

    def test_feedback_severity_levels(self) -> None:
        info = Feedback(provider_name="A", summary="Info", severity="info")
        caution = Feedback(provider_name="B", summary="Caution", severity="caution")
        warning = Feedback(provider_name="C", summary="Warning", severity="warning")

        assert info.severity == "info"
        assert caution.severity == "caution"
        assert warning.severity == "warning"

    def test_render_basic_feedback(self) -> None:
        feedback = Feedback(provider_name="Test", summary="Status check")
        rendered = feedback.render()

        assert "[Feedback - Test]" in rendered
        assert "Status check" in rendered

    def test_render_feedback_with_observations(self) -> None:
        obs = Observation(category="Pattern", description="Loop detected")
        feedback = Feedback(
            provider_name="Loop",
            summary="Possible loop",
            observations=(obs,),
        )
        rendered = feedback.render()

        assert "• Pattern: Loop detected" in rendered

    def test_render_feedback_with_suggestions(self) -> None:
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


# --- FeedbackContext Tests ---


class TestFeedbackContext:
    def _make_session(self) -> Session:
        bus = InProcessDispatcher()
        return Session(bus=bus)

    def _make_prompt(self) -> Prompt[None]:
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test-prompt", name="test"
        )
        return Prompt(template)

    def test_creates_context_with_basic_fields(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.session is session
        assert context.prompt is prompt
        assert context.deadline is None

    def test_creates_context_with_deadline(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))

        context = FeedbackContext(session=session, prompt=prompt, deadline=deadline)

        assert context.deadline is deadline

    def test_resources_returns_prompt_resources(self) -> None:
        from weakincentives.prompt import PromptResources

        session = self._make_session()
        prompt = self._make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        # Resources should be accessible and of the right type
        assert isinstance(context.resources, PromptResources)

    def test_filesystem_returns_none_when_not_registered(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        # Need to enter resource context to access filesystem
        with prompt.resources:
            # No filesystem registered, should return None
            assert context.filesystem is None

    def test_last_feedback_returns_none_when_empty(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert context.last_feedback is None

    def test_last_feedback_returns_most_recent(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()

        # Add feedback to session
        feedback1 = Feedback(provider_name="A", summary="First", call_index=1)
        feedback2 = Feedback(provider_name="B", summary="Second", call_index=2)
        session[Feedback].append(feedback1)
        session[Feedback].append(feedback2)

        context = FeedbackContext(session=session, prompt=prompt)

        last = context.last_feedback
        assert last is not None
        assert last.summary == "Second"

    def test_tool_call_count_returns_zero_when_empty(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()
        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_call_count == 0

    def test_tool_call_count_returns_correct_count(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()

        # Add tool invocations
        for i in range(3):
            event = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_call_count == 3

    def test_tool_calls_since_last_feedback_all_when_no_feedback(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()

        # Add tool invocations
        for i in range(5):
            event = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        context = FeedbackContext(session=session, prompt=prompt)

        assert context.tool_calls_since_last_feedback() == 5

    def test_tool_calls_since_last_feedback_counts_after_last(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()

        # Add 3 tool invocations
        for i in range(3):
            event = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        # Add feedback at call_index=3
        feedback = Feedback(provider_name="A", summary="Test", call_index=3)
        session[Feedback].append(feedback)

        # Add 2 more tool invocations
        for i in range(2):
            event = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"more_tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        context = FeedbackContext(session=session, prompt=prompt)

        # Total is 5, last feedback at 3, so 2 since
        assert context.tool_calls_since_last_feedback() == 2

    def test_recent_tool_calls_returns_last_n(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()

        # Add 5 tool invocations
        for i in range(5):
            event = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        context = FeedbackContext(session=session, prompt=prompt)

        recent = context.recent_tool_calls(3)
        assert len(recent) == 3
        assert recent[0].name == "tool_2"
        assert recent[1].name == "tool_3"
        assert recent[2].name == "tool_4"

    def test_recent_tool_calls_returns_all_when_fewer_than_n(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()

        # Add 2 tool invocations
        for i in range(2):
            event = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        context = FeedbackContext(session=session, prompt=prompt)

        recent = context.recent_tool_calls(5)
        assert len(recent) == 2


# --- FeedbackTrigger Tests ---


class TestFeedbackTrigger:
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


# --- _should_trigger Tests ---


class TestShouldTrigger:
    def _make_context(
        self,
        tool_calls: int = 0,
        last_feedback_call_index: int | None = None,
        last_feedback_timestamp: datetime | None = None,
    ) -> FeedbackContext:
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test-prompt", name="test"
        )
        prompt: Prompt[None] = Prompt(template)

        # Add tool invocations
        for i in range(tool_calls):
            event = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        # Add feedback if specified
        if last_feedback_call_index is not None:
            ts = last_feedback_timestamp or datetime.now(UTC)
            feedback = Feedback(
                provider_name="A",
                summary="Test",
                call_index=last_feedback_call_index,
                timestamp=ts,
            )
            session[Feedback].append(feedback)

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
        # 5 total calls, last feedback at index 3, so 2 calls since
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
        # 5 - 1 = 4 calls since last feedback >= 3

        assert _should_trigger(trigger, context) is True

    def test_or_logic_fires_on_time_when_calls_not_met(self) -> None:
        trigger = FeedbackTrigger(every_n_calls=10, every_n_seconds=30)
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        context = self._make_context(
            tool_calls=3,
            last_feedback_call_index=2,
            last_feedback_timestamp=old_time,
        )
        # Only 1 call since last feedback, but time elapsed

        assert _should_trigger(trigger, context) is True


# --- FeedbackProviderConfig Tests ---


class TestFeedbackProviderConfig:
    def test_creates_config_with_provider_and_trigger(self) -> None:
        provider = DeadlineFeedback()
        trigger = FeedbackTrigger(every_n_seconds=30)

        config = FeedbackProviderConfig(provider=provider, trigger=trigger)

        assert config.provider is provider
        assert config.trigger is trigger


# --- DeadlineFeedback Tests ---


class TestDeadlineFeedback:
    def _make_context(
        self,
        deadline: Deadline | None = None,
    ) -> tuple[Session, FeedbackContext]:
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test-prompt", name="test"
        )
        prompt: Prompt[None] = Prompt(template)

        context = FeedbackContext(session=session, prompt=prompt, deadline=deadline)
        return session, context

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
        # Use hours=2 to ensure we always get "hour" in the output
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
        # Test the case where deadline has passed (remaining <= 0)
        provider = DeadlineFeedback()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=2))
        session, context = self._make_context(deadline=deadline)

        # Create a mock deadline that returns negative remaining time
        mock_deadline = type(
            "MockDeadline", (), {"remaining": lambda self: timedelta(seconds=-1)}
        )()

        # Create a new context with the mock deadline
        mock_context = FeedbackContext(
            session=context.session,
            prompt=context.prompt,
            deadline=mock_deadline,  # type: ignore[arg-type]
        )

        feedback = provider.provide(session, context=mock_context)

        assert feedback.severity == "warning"
        assert (
            "reached" in feedback.summary.lower()
            or "deadline" in feedback.summary.lower()
        )

    def test_provide_without_deadline_returns_info(self) -> None:
        provider = DeadlineFeedback()
        session, context = self._make_context(deadline=None)

        feedback = provider.provide(session, context=context)

        assert feedback.severity == "info"
        assert "no deadline" in feedback.summary.lower()

    def test_format_duration_seconds(self) -> None:
        from weakincentives.prompt.observers import _format_duration

        assert _format_duration(45) == "45 seconds"

    def test_format_duration_minute_singular(self) -> None:
        from weakincentives.prompt.observers import _format_duration

        assert _format_duration(60) == "1 minute"

    def test_format_duration_minutes_plural(self) -> None:
        from weakincentives.prompt.observers import _format_duration

        assert _format_duration(300) == "5 minutes"

    def test_format_duration_hours(self) -> None:
        from weakincentives.prompt.observers import _format_duration

        assert "hour" in _format_duration(7200)

    def test_custom_warning_threshold(self) -> None:
        provider = DeadlineFeedback(warning_threshold_seconds=300)  # 5 minutes
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=240))
        session, context = self._make_context(deadline=deadline)

        feedback = provider.provide(session, context=context)

        assert feedback.severity == "warning"


# --- run_feedback_providers Tests ---


@dataclass(frozen=True)
class MockFeedbackProvider:
    """Test provider that always runs and returns fixed feedback."""

    should_run_return: bool = True
    feedback_summary: str = "Mock feedback"

    @property
    def name(self) -> str:
        return "Mock"

    def should_run(self, session: Session, *, context: FeedbackContext) -> bool:
        return self.should_run_return

    def provide(self, session: Session, *, context: FeedbackContext) -> Feedback:
        return Feedback(
            provider_name=self.name,
            summary=self.feedback_summary,
        )


class TestRunFeedbackProviders:
    def _make_context(self, tool_calls: int = 0) -> tuple[Session, FeedbackContext]:
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test-prompt", name="test"
        )
        prompt: Prompt[None] = Prompt(template)

        # Add tool invocations to satisfy triggers
        for i in range(tool_calls):
            event = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session.dispatcher.dispatch(event)

        context = FeedbackContext(session=session, prompt=prompt)
        return session, context

    def test_returns_none_when_no_providers(self) -> None:
        session, context = self._make_context(tool_calls=5)

        result = run_feedback_providers(providers=(), context=context, session=session)

        assert result is None

    def test_returns_none_when_trigger_not_met(self) -> None:
        session, context = self._make_context(tool_calls=2)
        provider = MockFeedbackProvider()
        config = FeedbackProviderConfig(
            provider=provider, trigger=FeedbackTrigger(every_n_calls=10)
        )

        result = run_feedback_providers(
            providers=(config,), context=context, session=session
        )

        assert result is None

    def test_returns_none_when_should_run_false(self) -> None:
        session, context = self._make_context(tool_calls=5)
        provider = MockFeedbackProvider(should_run_return=False)
        config = FeedbackProviderConfig(
            provider=provider, trigger=FeedbackTrigger(every_n_calls=3)
        )

        result = run_feedback_providers(
            providers=(config,), context=context, session=session
        )

        assert result is None

    def test_returns_rendered_feedback_when_triggered(self) -> None:
        session, context = self._make_context(tool_calls=5)
        provider = MockFeedbackProvider(feedback_summary="Test feedback")
        config = FeedbackProviderConfig(
            provider=provider, trigger=FeedbackTrigger(every_n_calls=3)
        )

        result = run_feedback_providers(
            providers=(config,), context=context, session=session
        )

        assert result is not None
        assert "Test feedback" in result
        assert "[Feedback - Mock]" in result

    def test_stores_feedback_in_session(self) -> None:
        session, context = self._make_context(tool_calls=5)
        provider = MockFeedbackProvider()
        config = FeedbackProviderConfig(
            provider=provider, trigger=FeedbackTrigger(every_n_calls=3)
        )

        run_feedback_providers(providers=(config,), context=context, session=session)

        # Check that feedback was recorded in session
        latest = session[Feedback].latest()
        assert latest is not None
        assert latest.provider_name == "Mock"
        assert latest.call_index == 5  # Updated to current tool call count

    def test_first_matching_provider_wins(self) -> None:
        session, context = self._make_context(tool_calls=5)

        provider1 = MockFeedbackProvider(feedback_summary="First")
        provider2 = MockFeedbackProvider(feedback_summary="Second")

        configs = (
            FeedbackProviderConfig(
                provider=provider1, trigger=FeedbackTrigger(every_n_calls=3)
            ),
            FeedbackProviderConfig(
                provider=provider2, trigger=FeedbackTrigger(every_n_calls=3)
            ),
        )

        result = run_feedback_providers(
            providers=configs, context=context, session=session
        )

        assert result is not None
        assert "First" in result
        assert "Second" not in result


# --- PromptTemplate FeedbackProvider Integration Tests ---


class TestPromptTemplateFeedbackProviders:
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


# --- RecordFeedback Event Tests ---


class TestRecordFeedback:
    def test_creates_record_feedback_event(self) -> None:
        feedback = Feedback(provider_name="Test", summary="Test summary")
        event = RecordFeedback(feedback=feedback)

        assert event.feedback is feedback
        assert event.feedback.provider_name == "Test"

    def test_feedback_stored_in_session_slice(self) -> None:
        bus = InProcessDispatcher()
        session = Session(bus=bus)

        feedback = Feedback(provider_name="A", summary="First")
        session[Feedback].append(feedback)

        stored = session[Feedback].latest()
        assert stored is not None
        assert stored.provider_name == "A"

    def test_multiple_feedback_stored(self) -> None:
        bus = InProcessDispatcher()
        session = Session(bus=bus)

        feedback1 = Feedback(provider_name="A", summary="First")
        feedback2 = Feedback(provider_name="B", summary="Second")

        session[Feedback].append(feedback1)
        session[Feedback].append(feedback2)

        all_feedback = session[Feedback].all()
        assert len(all_feedback) == 2
