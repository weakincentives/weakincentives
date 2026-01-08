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

"""Tests for trajectory observers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    Assessment,
    DeadlineObserver,
    Observation,
    ObserverConfig,
    ObserverContext,
    ObserverTrigger,
    Prompt,
    PromptTemplate,
    RecordAssessment,
    run_observers,
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


# --- Assessment Tests ---


class TestAssessment:
    def test_creates_basic_assessment(self) -> None:
        assessment = Assessment(observer_name="Test", summary="All good")
        assert assessment.observer_name == "Test"
        assert assessment.summary == "All good"
        assert assessment.observations == ()
        assert assessment.suggestions == ()
        assert assessment.severity == "info"
        assert assessment.call_index == 0

    def test_creates_assessment_with_observations(self) -> None:
        obs = Observation(category="Pattern", description="Loop detected")
        assessment = Assessment(
            observer_name="Loop",
            summary="Possible loop",
            observations=(obs,),
        )
        assert len(assessment.observations) == 1
        assert assessment.observations[0].category == "Pattern"

    def test_creates_assessment_with_suggestions(self) -> None:
        assessment = Assessment(
            observer_name="Time",
            summary="Running low on time",
            suggestions=("Wrap up soon", "Summarize progress"),
        )
        assert len(assessment.suggestions) == 2

    def test_assessment_severity_levels(self) -> None:
        info = Assessment(observer_name="A", summary="Info", severity="info")
        caution = Assessment(observer_name="B", summary="Caution", severity="caution")
        warning = Assessment(observer_name="C", summary="Warning", severity="warning")

        assert info.severity == "info"
        assert caution.severity == "caution"
        assert warning.severity == "warning"

    def test_render_basic_assessment(self) -> None:
        assessment = Assessment(observer_name="Test", summary="Status check")
        rendered = assessment.render()

        assert "[Trajectory Assessment - Test]" in rendered
        assert "Status check" in rendered

    def test_render_assessment_with_observations(self) -> None:
        obs = Observation(category="Pattern", description="Loop detected")
        assessment = Assessment(
            observer_name="Loop",
            summary="Possible loop",
            observations=(obs,),
        )
        rendered = assessment.render()

        assert "• Pattern: Loop detected" in rendered

    def test_render_assessment_with_suggestions(self) -> None:
        assessment = Assessment(
            observer_name="Time",
            summary="Low time",
            suggestions=("Wrap up", "Summarize"),
        )
        rendered = assessment.render()

        assert "→ Wrap up" in rendered
        assert "→ Summarize" in rendered

    def test_render_full_assessment(self) -> None:
        obs1 = Observation(category="Files", description="10 files read")
        obs2 = Observation(category="Time", description="5 minutes elapsed")
        assessment = Assessment(
            observer_name="Progress",
            summary="Making progress",
            observations=(obs1, obs2),
            suggestions=("Continue current approach",),
            severity="info",
        )
        rendered = assessment.render()

        assert "[Trajectory Assessment - Progress]" in rendered
        assert "Making progress" in rendered
        assert "• Files: 10 files read" in rendered
        assert "• Time: 5 minutes elapsed" in rendered
        assert "→ Continue current approach" in rendered


# --- ObserverContext Tests ---


class TestObserverContext:
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

        context = ObserverContext(session=session, prompt=prompt)

        assert context.session is session
        assert context.prompt is prompt
        assert context.deadline is None

    def test_creates_context_with_deadline(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))

        context = ObserverContext(session=session, prompt=prompt, deadline=deadline)

        assert context.deadline is deadline

    def test_resources_returns_prompt_resources(self) -> None:
        from weakincentives.prompt import PromptResources

        session = self._make_session()
        prompt = self._make_prompt()
        context = ObserverContext(session=session, prompt=prompt)

        # Resources should be accessible and of the right type
        assert isinstance(context.resources, PromptResources)

    def test_filesystem_returns_none_when_not_registered(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()
        context = ObserverContext(session=session, prompt=prompt)

        # Need to enter resource context to access filesystem
        with prompt.resources:
            # No filesystem registered, should return None
            assert context.filesystem is None

    def test_last_assessment_returns_none_when_empty(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()
        context = ObserverContext(session=session, prompt=prompt)

        assert context.last_assessment is None

    def test_last_assessment_returns_most_recent(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()

        # Add assessments to session
        assessment1 = Assessment(observer_name="A", summary="First", call_index=1)
        assessment2 = Assessment(observer_name="B", summary="Second", call_index=2)
        session[Assessment].append(assessment1)
        session[Assessment].append(assessment2)

        context = ObserverContext(session=session, prompt=prompt)

        last = context.last_assessment
        assert last is not None
        assert last.summary == "Second"

    def test_tool_call_count_returns_zero_when_empty(self) -> None:
        session = self._make_session()
        prompt = self._make_prompt()
        context = ObserverContext(session=session, prompt=prompt)

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

        context = ObserverContext(session=session, prompt=prompt)

        assert context.tool_call_count == 3

    def test_tool_calls_since_last_assessment_all_when_no_assessment(self) -> None:
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

        context = ObserverContext(session=session, prompt=prompt)

        assert context.tool_calls_since_last_assessment() == 5

    def test_tool_calls_since_last_assessment_counts_after_last(self) -> None:
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

        # Add assessment at call_index=3
        assessment = Assessment(observer_name="A", summary="Test", call_index=3)
        session[Assessment].append(assessment)

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

        context = ObserverContext(session=session, prompt=prompt)

        # Total is 5, last assessment at 3, so 2 since
        assert context.tool_calls_since_last_assessment() == 2

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

        context = ObserverContext(session=session, prompt=prompt)

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

        context = ObserverContext(session=session, prompt=prompt)

        recent = context.recent_tool_calls(5)
        assert len(recent) == 2


# --- ObserverTrigger Tests ---


class TestObserverTrigger:
    def test_creates_trigger_with_call_count(self) -> None:
        trigger = ObserverTrigger(every_n_calls=10)
        assert trigger.every_n_calls == 10
        assert trigger.every_n_seconds is None

    def test_creates_trigger_with_time_interval(self) -> None:
        trigger = ObserverTrigger(every_n_seconds=30.0)
        assert trigger.every_n_calls is None
        assert trigger.every_n_seconds == 30.0

    def test_creates_trigger_with_both_conditions(self) -> None:
        trigger = ObserverTrigger(every_n_calls=10, every_n_seconds=60.0)
        assert trigger.every_n_calls == 10
        assert trigger.every_n_seconds == 60.0


# --- _should_trigger Tests ---


class TestShouldTrigger:
    def _make_context(
        self,
        tool_calls: int = 0,
        last_assessment_call_index: int | None = None,
        last_assessment_timestamp: datetime | None = None,
    ) -> ObserverContext:
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

        # Add assessment if specified
        if last_assessment_call_index is not None:
            ts = last_assessment_timestamp or datetime.now(UTC)
            assessment = Assessment(
                observer_name="A",
                summary="Test",
                call_index=last_assessment_call_index,
                timestamp=ts,
            )
            session[Assessment].append(assessment)

        return ObserverContext(session=session, prompt=prompt)

    def test_returns_false_for_empty_trigger(self) -> None:
        trigger = ObserverTrigger()
        context = self._make_context(tool_calls=10)

        assert _should_trigger(trigger, context) is False

    def test_call_count_trigger_fires_when_threshold_met(self) -> None:
        trigger = ObserverTrigger(every_n_calls=5)
        context = self._make_context(tool_calls=5)

        assert _should_trigger(trigger, context) is True

    def test_call_count_trigger_does_not_fire_below_threshold(self) -> None:
        trigger = ObserverTrigger(every_n_calls=5)
        context = self._make_context(tool_calls=3)

        assert _should_trigger(trigger, context) is False

    def test_call_count_trigger_counts_since_last_assessment(self) -> None:
        trigger = ObserverTrigger(every_n_calls=3)
        # 5 total calls, last assessment at index 3, so 2 calls since
        context = self._make_context(tool_calls=5, last_assessment_call_index=3)

        assert _should_trigger(trigger, context) is False

    def test_time_trigger_fires_when_no_previous_assessment(self) -> None:
        trigger = ObserverTrigger(every_n_seconds=30)
        context = self._make_context(tool_calls=1)

        assert _should_trigger(trigger, context) is True

    def test_time_trigger_fires_when_time_elapsed(self) -> None:
        trigger = ObserverTrigger(every_n_seconds=30)
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        context = self._make_context(
            tool_calls=1,
            last_assessment_call_index=0,
            last_assessment_timestamp=old_time,
        )

        assert _should_trigger(trigger, context) is True

    def test_time_trigger_does_not_fire_when_too_recent(self) -> None:
        trigger = ObserverTrigger(every_n_seconds=30)
        recent_time = datetime.now(UTC) - timedelta(seconds=10)
        context = self._make_context(
            tool_calls=1,
            last_assessment_call_index=0,
            last_assessment_timestamp=recent_time,
        )

        assert _should_trigger(trigger, context) is False

    def test_or_logic_fires_on_call_count_when_time_not_met(self) -> None:
        trigger = ObserverTrigger(every_n_calls=3, every_n_seconds=300)
        recent_time = datetime.now(UTC) - timedelta(seconds=10)
        context = self._make_context(
            tool_calls=5,
            last_assessment_call_index=1,
            last_assessment_timestamp=recent_time,
        )
        # 5 - 1 = 4 calls since last assessment >= 3

        assert _should_trigger(trigger, context) is True

    def test_or_logic_fires_on_time_when_calls_not_met(self) -> None:
        trigger = ObserverTrigger(every_n_calls=10, every_n_seconds=30)
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        context = self._make_context(
            tool_calls=3,
            last_assessment_call_index=2,
            last_assessment_timestamp=old_time,
        )
        # Only 1 call since last assessment, but time elapsed

        assert _should_trigger(trigger, context) is True


# --- ObserverConfig Tests ---


class TestObserverConfig:
    def test_creates_config_with_observer_and_trigger(self) -> None:
        observer = DeadlineObserver()
        trigger = ObserverTrigger(every_n_seconds=30)

        config = ObserverConfig(observer=observer, trigger=trigger)

        assert config.observer is observer
        assert config.trigger is trigger


# --- DeadlineObserver Tests ---


class TestDeadlineObserver:
    def _make_context(
        self,
        deadline: Deadline | None = None,
    ) -> tuple[Session, ObserverContext]:
        bus = InProcessDispatcher()
        session = Session(bus=bus)
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test-prompt", name="test"
        )
        prompt: Prompt[None] = Prompt(template)

        context = ObserverContext(session=session, prompt=prompt, deadline=deadline)
        return session, context

    def test_name_property(self) -> None:
        observer = DeadlineObserver()
        assert observer.name == "Deadline"

    def test_should_run_returns_false_without_deadline(self) -> None:
        observer = DeadlineObserver()
        session, context = self._make_context(deadline=None)

        assert observer.should_run(session, context=context) is False

    def test_should_run_returns_true_with_deadline(self) -> None:
        observer = DeadlineObserver()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=1))
        session, context = self._make_context(deadline=deadline)

        assert observer.should_run(session, context=context) is True

    def test_observe_returns_info_when_plenty_of_time(self) -> None:
        observer = DeadlineObserver(warning_threshold_seconds=120)
        # Use hours=2 to ensure we always get "hour" in the output
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=2))
        session, context = self._make_context(deadline=deadline)

        assessment = observer.observe(session, context=context)

        assert assessment.observer_name == "Deadline"
        assert assessment.severity == "info"
        assert assessment.suggestions == ()
        assert "hour" in assessment.summary.lower()

    def test_observe_returns_warning_when_low_time(self) -> None:
        observer = DeadlineObserver(warning_threshold_seconds=120)
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=90))
        session, context = self._make_context(deadline=deadline)

        assessment = observer.observe(session, context=context)

        assert assessment.severity == "warning"
        assert len(assessment.suggestions) > 0
        assert "remaining" in assessment.summary.lower()

    def test_observe_returns_warning_when_deadline_passed(self) -> None:
        # Test the case where deadline has passed (remaining <= 0)
        observer = DeadlineObserver()
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=2))
        session, context = self._make_context(deadline=deadline)

        # Create a mock deadline that returns negative remaining time
        mock_deadline = type(
            "MockDeadline", (), {"remaining": lambda self: timedelta(seconds=-1)}
        )()

        # Create a new context with the mock deadline
        mock_context = ObserverContext(
            session=context.session,
            prompt=context.prompt,
            deadline=mock_deadline,  # type: ignore[arg-type]
        )

        assessment = observer.observe(session, context=mock_context)

        assert assessment.severity == "warning"
        assert (
            "reached" in assessment.summary.lower()
            or "deadline" in assessment.summary.lower()
        )

    def test_observe_without_deadline_returns_info(self) -> None:
        observer = DeadlineObserver()
        session, context = self._make_context(deadline=None)

        assessment = observer.observe(session, context=context)

        assert assessment.severity == "info"
        assert "no deadline" in assessment.summary.lower()

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
        observer = DeadlineObserver(warning_threshold_seconds=300)  # 5 minutes
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=240))
        session, context = self._make_context(deadline=deadline)

        assessment = observer.observe(session, context=context)

        assert assessment.severity == "warning"


# --- run_observers Tests ---


@dataclass(frozen=True)
class MockObserver:
    """Test observer that always runs and returns a fixed assessment."""

    should_run_return: bool = True
    assessment_summary: str = "Mock assessment"

    @property
    def name(self) -> str:
        return "Mock"

    def should_run(self, session: Session, *, context: ObserverContext) -> bool:
        return self.should_run_return

    def observe(self, session: Session, *, context: ObserverContext) -> Assessment:
        return Assessment(
            observer_name=self.name,
            summary=self.assessment_summary,
        )


class TestRunObservers:
    def _make_context(self, tool_calls: int = 0) -> tuple[Session, ObserverContext]:
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

        context = ObserverContext(session=session, prompt=prompt)
        return session, context

    def test_returns_none_when_no_observers(self) -> None:
        session, context = self._make_context(tool_calls=5)

        result = run_observers(observers=(), context=context, session=session)

        assert result is None

    def test_returns_none_when_trigger_not_met(self) -> None:
        session, context = self._make_context(tool_calls=2)
        observer = MockObserver()
        config = ObserverConfig(
            observer=observer, trigger=ObserverTrigger(every_n_calls=10)
        )

        result = run_observers(observers=(config,), context=context, session=session)

        assert result is None

    def test_returns_none_when_should_run_false(self) -> None:
        session, context = self._make_context(tool_calls=5)
        observer = MockObserver(should_run_return=False)
        config = ObserverConfig(
            observer=observer, trigger=ObserverTrigger(every_n_calls=3)
        )

        result = run_observers(observers=(config,), context=context, session=session)

        assert result is None

    def test_returns_rendered_assessment_when_triggered(self) -> None:
        session, context = self._make_context(tool_calls=5)
        observer = MockObserver(assessment_summary="Test feedback")
        config = ObserverConfig(
            observer=observer, trigger=ObserverTrigger(every_n_calls=3)
        )

        result = run_observers(observers=(config,), context=context, session=session)

        assert result is not None
        assert "Test feedback" in result
        assert "[Trajectory Assessment - Mock]" in result

    def test_dispatches_record_assessment_event(self) -> None:
        session, context = self._make_context(tool_calls=5)
        observer = MockObserver()
        config = ObserverConfig(
            observer=observer, trigger=ObserverTrigger(every_n_calls=3)
        )

        run_observers(observers=(config,), context=context, session=session)

        # Check that assessment was recorded in session
        latest = session[Assessment].latest()
        assert latest is not None
        assert latest.observer_name == "Mock"
        assert latest.call_index == 5  # Updated to current tool call count

    def test_first_matching_observer_wins(self) -> None:
        session, context = self._make_context(tool_calls=5)

        observer1 = MockObserver(assessment_summary="First")
        observer2 = MockObserver(assessment_summary="Second")

        configs = (
            ObserverConfig(
                observer=observer1, trigger=ObserverTrigger(every_n_calls=3)
            ),
            ObserverConfig(
                observer=observer2, trigger=ObserverTrigger(every_n_calls=3)
            ),
        )

        result = run_observers(observers=configs, context=context, session=session)

        assert result is not None
        assert "First" in result
        assert "Second" not in result


# --- PromptTemplate Observer Integration Tests ---


class TestPromptTemplateObservers:
    def test_creates_template_without_observers(self) -> None:
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        assert template.observers == ()

    def test_creates_template_with_observers(self) -> None:
        observer = DeadlineObserver()
        config = ObserverConfig(
            observer=observer, trigger=ObserverTrigger(every_n_seconds=30)
        )

        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", observers=[config]
        )

        assert len(template.observers) == 1
        assert template.observers[0].observer is observer

    def test_observers_converted_to_tuple(self) -> None:
        configs = [
            ObserverConfig(
                observer=DeadlineObserver(),
                trigger=ObserverTrigger(every_n_seconds=30),
            ),
            ObserverConfig(
                observer=MockObserver(),
                trigger=ObserverTrigger(every_n_calls=10),
            ),
        ]

        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", observers=configs
        )

        assert isinstance(template.observers, tuple)
        assert len(template.observers) == 2


class TestPromptObservers:
    def test_prompt_exposes_template_observers(self) -> None:
        observer = DeadlineObserver()
        config = ObserverConfig(
            observer=observer, trigger=ObserverTrigger(every_n_seconds=30)
        )

        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test", observers=[config]
        )
        prompt: Prompt[None] = Prompt(template)

        assert len(prompt.observers) == 1
        assert prompt.observers[0].observer is observer

    def test_prompt_observers_returns_tuple(self) -> None:
        template: PromptTemplate[None] = PromptTemplate(
            ns="test", key="test", name="test"
        )
        prompt: Prompt[None] = Prompt(template)

        assert isinstance(prompt.observers, tuple)


# --- RecordAssessment Event Tests ---


class TestRecordAssessment:
    def test_creates_record_assessment_event(self) -> None:
        assessment = Assessment(observer_name="Test", summary="Test summary")
        event = RecordAssessment(assessment=assessment)

        assert event.assessment is assessment
        assert event.assessment.observer_name == "Test"

    def test_assessment_stored_in_session_slice(self) -> None:
        bus = InProcessDispatcher()
        session = Session(bus=bus)

        assessment = Assessment(observer_name="A", summary="First")
        session[Assessment].append(assessment)

        stored = session[Assessment].latest()
        assert stored is not None
        assert stored.observer_name == "A"

    def test_multiple_assessments_stored(self) -> None:
        bus = InProcessDispatcher()
        session = Session(bus=bus)

        assessment1 = Assessment(observer_name="A", summary="First")
        assessment2 = Assessment(observer_name="B", summary="Second")

        session[Assessment].append(assessment1)
        session[Assessment].append(assessment2)

        all_assessments = session[Assessment].all()
        assert len(all_assessments) == 2
