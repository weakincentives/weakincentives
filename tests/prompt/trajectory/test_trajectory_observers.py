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

"""Tests for trajectory observer functionality."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    Assessment,
    DeadlineObserver,
    MarkdownSection,
    Observation,
    ObserverConfig,
    ObserverContext,
    ObserverTrigger,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
    run_observers,
)
from weakincentives.runtime import InProcessDispatcher, Session
from weakincentives.runtime.events import ToolInvoked

if TYPE_CHECKING:
    from weakincentives.runtime.session.protocols import SessionProtocol


# --- Test Data ---


@dataclass(frozen=True)
class SampleParams:
    value: str


def _test_handler(params: SampleParams, *, context: ToolContext) -> ToolResult[None]:
    """Test handler."""
    return ToolResult.ok(None, message="done")


test_tool = Tool[SampleParams, None](
    name="test_tool",
    description="A test tool",
    handler=_test_handler,
)


def _create_session() -> Session:
    """Create a session for testing."""
    return Session(bus=InProcessDispatcher())


def _create_prompt_with_observers(
    observers: list[ObserverConfig],
) -> Prompt[None]:
    """Create a prompt with observers configured."""
    template = PromptTemplate[None](
        ns="test",
        key="trajectory",
        sections=[
            MarkdownSection(
                title="Test",
                template="Test prompt",
                key="test",
                tools=(test_tool,),
            ),
        ],
        observers=observers,
    )
    return Prompt(template)


# --- Observation Tests ---


class TestObservation:
    def test_observation_basic_properties(self) -> None:
        obs = Observation(category="test", description="Test observation")
        assert obs.category == "test"
        assert obs.description == "Test observation"
        assert obs.evidence is None

    def test_observation_with_evidence(self) -> None:
        obs = Observation(
            category="error", description="Failed test", evidence="Stack trace here"
        )
        assert obs.evidence == "Stack trace here"

    def test_observation_is_frozen(self) -> None:
        obs = Observation(category="test", description="desc")
        with pytest.raises(AttributeError):
            obs.category = "changed"  # type: ignore[misc]


# --- Assessment Tests ---


class TestAssessment:
    def test_assessment_default_values(self) -> None:
        assessment = Assessment(observer_name="Test", summary="Test summary")
        assert assessment.observer_name == "Test"
        assert assessment.summary == "Test summary"
        assert assessment.observations == ()
        assert assessment.suggestions == ()
        assert assessment.severity == "info"
        assert assessment.call_index == 0
        assert assessment.timestamp is not None

    def test_assessment_with_observations(self) -> None:
        obs = Observation(category="cat", description="desc")
        assessment = Assessment(
            observer_name="Test", summary="Summary", observations=(obs,)
        )
        assert len(assessment.observations) == 1
        assert assessment.observations[0].category == "cat"

    def test_assessment_with_suggestions(self) -> None:
        assessment = Assessment(
            observer_name="Test",
            summary="Summary",
            suggestions=("Do this", "Do that"),
        )
        assert len(assessment.suggestions) == 2

    def test_assessment_severity_values(self) -> None:
        for severity in ("info", "caution", "warning"):
            assessment = Assessment(
                observer_name="Test",
                summary="Summary",
                severity=severity,  # type: ignore[arg-type]
            )
            assert assessment.severity == severity

    def test_assessment_render_basic(self) -> None:
        assessment = Assessment(observer_name="Deadline", summary="5 minutes remaining")
        rendered = assessment.render()
        assert "[Trajectory Assessment - Deadline]" in rendered
        assert "5 minutes remaining" in rendered

    def test_assessment_render_with_observations(self) -> None:
        obs = Observation(category="Pattern", description="Repeated read calls")
        assessment = Assessment(
            observer_name="Stall",
            summary="Potential stall detected",
            observations=(obs,),
        )
        rendered = assessment.render()
        assert "• Pattern: Repeated read calls" in rendered

    def test_assessment_render_with_suggestions(self) -> None:
        assessment = Assessment(
            observer_name="Deadline",
            summary="2 minutes remaining",
            suggestions=("Prioritize critical work.", "Summarize progress."),
        )
        rendered = assessment.render()
        assert "→ Prioritize critical work." in rendered
        assert "→ Summarize progress." in rendered

    def test_assessment_is_frozen(self) -> None:
        assessment = Assessment(observer_name="Test", summary="Summary")
        with pytest.raises(AttributeError):
            assessment.summary = "changed"  # type: ignore[misc]


# --- ObserverTrigger Tests ---


class TestObserverTrigger:
    def test_trigger_default_values(self) -> None:
        trigger = ObserverTrigger()
        assert trigger.every_n_calls is None
        assert trigger.every_n_seconds is None

    def test_trigger_with_call_count(self) -> None:
        trigger = ObserverTrigger(every_n_calls=10)
        assert trigger.every_n_calls == 10
        assert trigger.every_n_seconds is None

    def test_trigger_with_time_interval(self) -> None:
        trigger = ObserverTrigger(every_n_seconds=30.0)
        assert trigger.every_n_calls is None
        assert trigger.every_n_seconds == 30.0

    def test_trigger_with_both_conditions(self) -> None:
        trigger = ObserverTrigger(every_n_calls=10, every_n_seconds=60.0)
        assert trigger.every_n_calls == 10
        assert trigger.every_n_seconds == 60.0

    def test_trigger_is_frozen(self) -> None:
        trigger = ObserverTrigger(every_n_calls=5)
        with pytest.raises(AttributeError):
            trigger.every_n_calls = 10  # type: ignore[misc]


# --- ObserverContext Tests ---


class TestObserverContext:
    def test_context_basic_properties(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        assert context.session is session
        assert context.prompt is prompt
        assert context.deadline is None

    def test_context_resources(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        with prompt.resources:
            context = ObserverContext(session=session, prompt=prompt, deadline=None)
            # resources should return a PromptResources instance
            resources = context.resources
            assert resources is not None

    def test_context_filesystem_none(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        with prompt.resources:
            context = ObserverContext(session=session, prompt=prompt, deadline=None)
            # No filesystem registered, should return None
            assert context.filesystem is None

    def test_context_with_deadline(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))

        context = ObserverContext(session=session, prompt=prompt, deadline=deadline)
        assert context.deadline is deadline

    def test_context_tool_call_count_empty(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        assert context.tool_call_count == 0

    def test_context_tool_call_count_with_invocations(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Seed some tool invocations
        invocation = ToolInvoked(
            prompt_name="test",
            adapter="test",
            name="test_tool",
            params=None,
            result=None,
            session_id=None,
            created_at=datetime.now(UTC),
        )
        session[ToolInvoked].append(invocation)
        session[ToolInvoked].append(invocation)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        assert context.tool_call_count == 2

    def test_context_last_assessment_none(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        assert context.last_assessment is None

    def test_context_last_assessment_with_history(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Seed some assessments
        assessment1 = Assessment(observer_name="Test1", summary="First", call_index=5)
        assessment2 = Assessment(observer_name="Test2", summary="Second", call_index=10)
        session[Assessment].append(assessment1)
        session[Assessment].append(assessment2)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        last = context.last_assessment
        assert last is not None
        assert last.observer_name == "Test2"
        assert last.call_index == 10

    def test_context_tool_calls_since_last_assessment_no_history(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        invocation = ToolInvoked(
            prompt_name="test",
            adapter="test",
            name="test_tool",
            params=None,
            result=None,
            session_id=None,
            created_at=datetime.now(UTC),
        )
        session[ToolInvoked].append(invocation)
        session[ToolInvoked].append(invocation)
        session[ToolInvoked].append(invocation)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        assert context.tool_calls_since_last_assessment() == 3

    def test_context_tool_calls_since_last_assessment_with_history(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Add 5 tool invocations
        invocation = ToolInvoked(
            prompt_name="test",
            adapter="test",
            name="test_tool",
            params=None,
            result=None,
            session_id=None,
            created_at=datetime.now(UTC),
        )
        for _ in range(5):
            session[ToolInvoked].append(invocation)

        # Add assessment at call_index 3
        assessment = Assessment(observer_name="Test", summary="Test", call_index=3)
        session[Assessment].append(assessment)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        assert context.tool_calls_since_last_assessment() == 2  # 5 - 3 = 2

    def test_context_recent_tool_calls(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Add tool invocations with different names
        for i in range(5):
            invocation = ToolInvoked(
                prompt_name="test",
                adapter="test",
                name=f"tool_{i}",
                params=None,
                result=None,
                session_id=None,
                created_at=datetime.now(UTC),
            )
            session[ToolInvoked].append(invocation)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        recent = context.recent_tool_calls(3)
        assert len(recent) == 3
        assert recent[0].name == "tool_2"
        assert recent[1].name == "tool_3"
        assert recent[2].name == "tool_4"

    def test_context_recent_tool_calls_fewer_than_requested(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        invocation = ToolInvoked(
            prompt_name="test",
            adapter="test",
            name="tool",
            params=None,
            result=None,
            session_id=None,
            created_at=datetime.now(UTC),
        )
        session[ToolInvoked].append(invocation)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)
        recent = context.recent_tool_calls(5)
        assert len(recent) == 1


# --- DeadlineObserver Tests ---


class TestDeadlineObserver:
    def test_observer_name(self) -> None:
        observer = DeadlineObserver()
        assert observer.name == "Deadline"

    def test_observer_default_warning_threshold(self) -> None:
        observer = DeadlineObserver()
        assert observer.warning_threshold_seconds == 120

    def test_observer_custom_warning_threshold(self) -> None:
        observer = DeadlineObserver(warning_threshold_seconds=60)
        assert observer.warning_threshold_seconds == 60

    def test_should_run_with_deadline(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
        context = ObserverContext(session=session, prompt=prompt, deadline=deadline)

        observer = DeadlineObserver()
        assert observer.should_run(session, context=context) is True

    def test_should_run_without_deadline(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = DeadlineObserver()
        assert observer.should_run(session, context=context) is False

    def test_observe_with_ample_time(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        # Use 2 hours to avoid edge case where 1 hour rounds to 59 minutes
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(hours=2))
        context = ObserverContext(session=session, prompt=prompt, deadline=deadline)

        observer = DeadlineObserver()
        assessment = observer.observe(session, context=context)

        assert assessment.observer_name == "Deadline"
        assert "hours remaining" in assessment.summary
        assert assessment.severity == "info"
        assert assessment.suggestions == ()

    def test_observe_with_minutes_remaining(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=10))
        context = ObserverContext(session=session, prompt=prompt, deadline=deadline)

        observer = DeadlineObserver()
        assessment = observer.observe(session, context=context)

        assert (
            "minutes remaining" in assessment.summary or "minute" in assessment.summary
        )
        assert assessment.severity == "info"

    def test_observe_warning_threshold(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        # 90 seconds remaining (below 120s threshold)
        deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=90))
        context = ObserverContext(session=session, prompt=prompt, deadline=deadline)

        observer = DeadlineObserver()
        assessment = observer.observe(session, context=context)

        assert assessment.severity == "warning"
        assert len(assessment.suggestions) == 2
        assert "Prioritize" in assessment.suggestions[0]

    def test_observe_no_deadline(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = DeadlineObserver()
        assessment = observer.observe(session, context=context)

        assert "No deadline" in assessment.summary

    def test_observe_deadline_expired(self) -> None:
        from unittest.mock import MagicMock

        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Create a mock deadline that returns negative remaining time
        mock_deadline = MagicMock()
        mock_deadline.remaining.return_value = timedelta(seconds=-10)

        context = ObserverContext(
            session=session, prompt=prompt, deadline=mock_deadline
        )

        observer = DeadlineObserver()
        assessment = observer.observe(session, context=context)

        assert "reached the time deadline" in assessment.summary
        assert assessment.severity == "warning"
        assert "Wrap up immediately" in assessment.suggestions[0]

    def test_format_duration_seconds(self) -> None:
        observer = DeadlineObserver()
        assert observer._format_duration(45) == "45 seconds"

    def test_format_duration_minutes_singular(self) -> None:
        observer = DeadlineObserver()
        assert observer._format_duration(60) == "1 minute"

    def test_format_duration_minutes_plural(self) -> None:
        observer = DeadlineObserver()
        assert observer._format_duration(300) == "5 minutes"

    def test_format_duration_hours(self) -> None:
        observer = DeadlineObserver()
        assert observer._format_duration(3600) == "1.0 hours"
        assert observer._format_duration(7200) == "2.0 hours"


# --- ObserverConfig Tests ---


class TestObserverConfig:
    def test_config_basic_creation(self) -> None:
        observer = DeadlineObserver()
        trigger = ObserverTrigger(every_n_calls=10)
        config = ObserverConfig(observer=observer, trigger=trigger)

        assert config.observer is observer
        assert config.trigger is trigger

    def test_config_is_frozen(self) -> None:
        observer = DeadlineObserver()
        trigger = ObserverTrigger()
        config = ObserverConfig(observer=observer, trigger=trigger)

        with pytest.raises(AttributeError):
            config.observer = DeadlineObserver()  # type: ignore[misc]


# --- Custom Observer for Testing ---


class CountingObserver:
    """A simple observer that counts calls for testing."""

    def __init__(self, name: str = "Counter") -> None:
        self._name = name
        self.should_run_count = 0
        self.observe_count = 0
        self._should_run_result = True

    @property
    def name(self) -> str:
        return self._name

    def should_run(
        self,
        session: SessionProtocol,
        *,
        context: ObserverContext,
    ) -> bool:
        self.should_run_count += 1
        return self._should_run_result

    def observe(
        self,
        session: SessionProtocol,
        *,
        context: ObserverContext,
    ) -> Assessment:
        self.observe_count += 1
        return Assessment(
            observer_name=self.name,
            summary=f"Observation #{self.observe_count}",
        )


# --- run_observers Tests ---


class TestRunObservers:
    def test_run_observers_empty_list(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        result = run_observers(observers=[], context=context, session=session)
        assert result is None

    def test_run_observers_no_trigger(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = CountingObserver()
        config = ObserverConfig(
            observer=observer,
            trigger=ObserverTrigger(),  # No triggers set
        )

        result = run_observers(observers=[config], context=context, session=session)
        assert result is None
        assert observer.should_run_count == 0

    def test_run_observers_trigger_by_call_count(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Add tool invocations
        invocation = ToolInvoked(
            prompt_name="test",
            adapter="test",
            name="test_tool",
            params=None,
            result=None,
            session_id=None,
            created_at=datetime.now(UTC),
        )
        for _ in range(5):
            session[ToolInvoked].append(invocation)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = CountingObserver()
        config = ObserverConfig(
            observer=observer,
            trigger=ObserverTrigger(every_n_calls=5),
        )

        result = run_observers(observers=[config], context=context, session=session)
        assert result is not None
        assert "Counter" in result
        assert observer.should_run_count == 1
        assert observer.observe_count == 1

    def test_run_observers_trigger_by_time(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])
        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = CountingObserver()
        config = ObserverConfig(
            observer=observer,
            trigger=ObserverTrigger(every_n_seconds=30.0),
        )

        # No previous assessment, should trigger immediately
        result = run_observers(observers=[config], context=context, session=session)
        assert result is not None

    def test_run_observers_trigger_by_time_not_enough_elapsed(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Add a recent assessment (just now)
        recent_assessment = Assessment(
            observer_name="Test",
            summary="Recent",
            timestamp=datetime.now(UTC),  # Just now
            call_index=0,
        )
        session[Assessment].append(recent_assessment)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = CountingObserver()
        config = ObserverConfig(
            observer=observer,
            trigger=ObserverTrigger(every_n_seconds=30.0),  # 30 seconds threshold
        )

        # Recent assessment exists, not enough time has passed
        result = run_observers(observers=[config], context=context, session=session)
        assert result is None  # Should not trigger
        assert observer.should_run_count == 0

    def test_run_observers_trigger_by_time_elapsed(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Add an old assessment (over 30 seconds ago)
        old_assessment = Assessment(
            observer_name="Test",
            summary="Old",
            timestamp=datetime.now(UTC) - timedelta(seconds=60),  # 60 seconds ago
            call_index=0,
        )
        session[Assessment].append(old_assessment)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = CountingObserver()
        config = ObserverConfig(
            observer=observer,
            trigger=ObserverTrigger(every_n_seconds=30.0),  # 30 seconds threshold
        )

        # Old assessment exists, enough time has passed
        result = run_observers(observers=[config], context=context, session=session)
        assert result is not None

    def test_run_observers_should_run_false(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Add tool invocations to trigger
        invocation = ToolInvoked(
            prompt_name="test",
            adapter="test",
            name="test_tool",
            params=None,
            result=None,
            session_id=None,
            created_at=datetime.now(UTC),
        )
        for _ in range(5):
            session[ToolInvoked].append(invocation)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = CountingObserver()
        observer._should_run_result = False
        config = ObserverConfig(
            observer=observer,
            trigger=ObserverTrigger(every_n_calls=5),
        )

        result = run_observers(observers=[config], context=context, session=session)
        assert result is None
        assert observer.should_run_count == 1
        assert observer.observe_count == 0

    def test_run_observers_stores_assessment(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Add tool invocations
        invocation = ToolInvoked(
            prompt_name="test",
            adapter="test",
            name="test_tool",
            params=None,
            result=None,
            session_id=None,
            created_at=datetime.now(UTC),
        )
        for _ in range(5):
            session[ToolInvoked].append(invocation)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer = CountingObserver()
        config = ObserverConfig(
            observer=observer,
            trigger=ObserverTrigger(every_n_calls=5),
        )

        # Verify no assessments initially
        assert session[Assessment].latest() is None

        run_observers(observers=[config], context=context, session=session)

        # Verify assessment was stored
        stored = session[Assessment].latest()
        assert stored is not None
        assert stored.observer_name == "Counter"
        assert stored.call_index == 5

    def test_run_observers_first_match_wins(self) -> None:
        session = _create_session()
        prompt = _create_prompt_with_observers([])

        # Add tool invocations
        invocation = ToolInvoked(
            prompt_name="test",
            adapter="test",
            name="test_tool",
            params=None,
            result=None,
            session_id=None,
            created_at=datetime.now(UTC),
        )
        for _ in range(5):
            session[ToolInvoked].append(invocation)

        context = ObserverContext(session=session, prompt=prompt, deadline=None)

        observer1 = CountingObserver("First")
        observer2 = CountingObserver("Second")
        configs = [
            ObserverConfig(
                observer=observer1,
                trigger=ObserverTrigger(every_n_calls=5),
            ),
            ObserverConfig(
                observer=observer2,
                trigger=ObserverTrigger(every_n_calls=5),
            ),
        ]

        result = run_observers(observers=configs, context=context, session=session)
        assert result is not None
        assert "First" in result
        assert observer1.observe_count == 1
        assert observer2.observe_count == 0


# --- PromptTemplate Integration Tests ---


class TestPromptTemplateIntegration:
    def test_prompt_template_default_observers_empty(self) -> None:
        template = PromptTemplate[None](
            ns="test",
            key="test",
            sections=[
                MarkdownSection(title="Test", template="Test", key="test"),
            ],
        )
        assert template.observers == ()

    def test_prompt_template_with_observers(self) -> None:
        observer = DeadlineObserver()
        trigger = ObserverTrigger(every_n_seconds=30)
        config = ObserverConfig(observer=observer, trigger=trigger)

        template = PromptTemplate[None](
            ns="test",
            key="test",
            sections=[
                MarkdownSection(title="Test", template="Test", key="test"),
            ],
            observers=[config],
        )
        assert len(template.observers) == 1
        assert template.observers[0] is config

    def test_prompt_observers_property(self) -> None:
        observer = DeadlineObserver()
        trigger = ObserverTrigger(every_n_seconds=30)
        config = ObserverConfig(observer=observer, trigger=trigger)

        template = PromptTemplate[None](
            ns="test",
            key="test",
            sections=[
                MarkdownSection(title="Test", template="Test", key="test"),
            ],
            observers=[config],
        )
        prompt = Prompt(template)

        observers = prompt.observers
        assert len(observers) == 1
        assert observers[0] is config
