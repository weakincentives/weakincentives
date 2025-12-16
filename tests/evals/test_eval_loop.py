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

"""Tests for EvalLoop event-driven orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch
from uuid import UUID

import pytest

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.evals import (
    EvalLoop,
    EvalLoopCompleted,
    EvalLoopConfig,
    EvalLoopFailed,
    EvalLoopRequest,
    EvalReport,
    Sample,
    SampleEvaluated,
    Score,
)
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.prompt.prompt import PromptTemplate
from weakincentives.runtime import InProcessEventBus, MainLoop, Session
from weakincentives.runtime.session.protocols import SessionProtocol


@dataclass(slots=True, frozen=True)
class _Output:
    """Test output type."""

    answer: str


@dataclass(slots=True, frozen=True)
class _Params:
    """Test params type."""

    question: str


def _output_match(output: _Output, expected: _Output) -> Score:
    """Evaluator that compares Output objects."""
    passed = output.answer == expected.answer
    return Score(value=1.0 if passed else 0.0, passed=passed)


class _MockAdapter(ProviderAdapter[_Output]):
    """Mock adapter that returns fixed responses."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self._call_count = 0

    def evaluate(
        self,
        prompt: Prompt[_Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[_Output]:
        del session, deadline, budget, budget_tracker
        self._call_count += 1
        question = ""
        for param in prompt.params:
            if isinstance(param, _Params):
                question = param.question
                break
        answer = self._responses.get(question, "unknown")
        return PromptResponse(
            prompt_name="test",
            text=answer,
            output=_Output(answer=answer),
        )


class _ErrorAdapter(ProviderAdapter[_Output]):
    """Mock adapter that raises errors."""

    def __init__(self, error_samples: set[str]) -> None:
        self._error_samples = error_samples

    def evaluate(
        self,
        prompt: Prompt[_Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[_Output]:
        del session, deadline, budget, budget_tracker
        question = ""
        for param in prompt.params:
            if isinstance(param, _Params):
                question = param.question
                break
        if question in self._error_samples:
            raise RuntimeError(f"Error processing: {question}")
        return PromptResponse(
            prompt_name="test", text="ok", output=_Output(answer="ok")
        )


class _NoneOutputAdapter(ProviderAdapter[_Output]):
    """Mock adapter that returns None output."""

    def evaluate(
        self,
        prompt: Prompt[_Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[_Output]:
        del prompt, session, deadline, budget, budget_tracker
        return PromptResponse(prompt_name="test", text=None, output=None)


class _TestLoop(MainLoop[str, _Output]):
    """Test loop implementation."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        bus: InProcessEventBus,
    ) -> None:
        super().__init__(adapter=adapter, bus=bus)
        self._template = PromptTemplate[_Output](
            ns="test",
            key="qa",
            sections=[
                MarkdownSection[_Params](
                    title="Question",
                    template="$question",
                    key="question",
                ),
            ],
        )

    def create_prompt(self, request: str) -> Prompt[_Output]:
        return Prompt(self._template).bind(_Params(question=request))

    def create_session(self) -> Session:
        return Session(bus=self._bus)


# =============================================================================
# EvalLoopConfig Tests
# =============================================================================


def test_eval_loop_config_default() -> None:
    """EvalLoopConfig creates with defaults."""
    config = EvalLoopConfig()
    assert config is not None


# =============================================================================
# EvalLoopRequest Tests
# =============================================================================


def test_eval_loop_request_default_fields() -> None:
    """EvalLoopRequest auto-generates request_id and created_at."""
    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)
    request = EvalLoopRequest(dataset=dataset)

    assert isinstance(request.request_id, UUID)
    assert request.created_at is not None
    assert request.dataset == dataset


def test_eval_loop_request_unique_ids() -> None:
    """Each EvalLoopRequest gets a unique request_id."""
    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)
    request1 = EvalLoopRequest(dataset=dataset)
    request2 = EvalLoopRequest(dataset=dataset)

    assert request1.request_id != request2.request_id


# =============================================================================
# EvalLoopCompleted Tests
# =============================================================================


def test_eval_loop_completed_fields() -> None:
    """EvalLoopCompleted stores request_id and report."""
    from uuid import uuid4

    request_id = uuid4()
    report = EvalReport(results=())
    completed = EvalLoopCompleted(request_id=request_id, report=report)

    assert completed.request_id == request_id
    assert completed.report is report
    assert completed.completed_at is not None


# =============================================================================
# EvalLoopFailed Tests
# =============================================================================


def test_eval_loop_failed_fields() -> None:
    """EvalLoopFailed stores error and optional partial_report."""
    from uuid import uuid4

    request_id = uuid4()
    error = RuntimeError("test error")
    failed = EvalLoopFailed(request_id=request_id, error=error, partial_report=None)

    assert failed.request_id == request_id
    assert failed.error is error
    assert failed.partial_report is None
    assert failed.failed_at is not None


def test_eval_loop_failed_with_partial_report() -> None:
    """EvalLoopFailed can include a partial report."""
    from uuid import uuid4

    request_id = uuid4()
    error = RuntimeError("test error")
    partial = EvalReport(results=())
    failed = EvalLoopFailed(request_id=request_id, error=error, partial_report=partial)

    assert failed.partial_report is partial


# =============================================================================
# EvalLoop.execute() Tests
# =============================================================================


def test_eval_loop_execute_all_pass() -> None:
    """EvalLoop.execute handles all passing samples."""
    bus = InProcessEventBus()
    responses = {"What is 2+2?": "4", "Capital of France?": "Paris"}
    adapter = _MockAdapter(responses)
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (
        Sample(id="1", input="What is 2+2?", expected=_Output(answer="4")),
        Sample(id="2", input="Capital of France?", expected=_Output(answer="Paris")),
    )

    report = eval_loop.execute(dataset)

    assert isinstance(report, EvalReport)
    assert report.total == 2
    assert report.successful == 2
    assert report.pass_rate == 1.0


def test_eval_loop_execute_mixed_results() -> None:
    """EvalLoop.execute handles mixed pass/fail results."""
    bus = InProcessEventBus()
    responses = {"What is 2+2?": "4", "Capital of France?": "London"}
    adapter = _MockAdapter(responses)
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (
        Sample(id="1", input="What is 2+2?", expected=_Output(answer="4")),
        Sample(id="2", input="Capital of France?", expected=_Output(answer="Paris")),
    )

    report = eval_loop.execute(dataset)

    assert report.total == 2
    assert report.pass_rate == 0.5


def test_eval_loop_execute_empty_dataset() -> None:
    """EvalLoop.execute handles empty dataset."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({})
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset: tuple[Sample[str, _Output], ...] = ()

    report = eval_loop.execute(dataset)

    assert report.total == 0
    assert report.pass_rate == 0.0


def test_eval_loop_execute_handles_errors() -> None:
    """EvalLoop.execute catches and records errors."""
    bus = InProcessEventBus()
    adapter = _ErrorAdapter({"Q1"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (
        Sample(id="1", input="Q1", expected=_Output(answer="answer")),
        Sample(id="2", input="Q2", expected=_Output(answer="ok")),
    )

    report = eval_loop.execute(dataset)

    assert report.total == 2
    assert report.successful == 1  # Only Q2 succeeded

    error_result = next(r for r in report.results if r.sample_id == "1")
    assert error_result.error is not None
    assert "Error processing" in error_result.error


def test_eval_loop_execute_none_output() -> None:
    """EvalLoop.execute records error when adapter returns None output."""
    bus = InProcessEventBus()
    adapter = _NoneOutputAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)

    report = eval_loop.execute(dataset)

    assert report.total == 1
    assert report.successful == 0
    result = report.results[0]
    assert result.error == "No output from loop"


def test_eval_loop_execute_publishes_sample_events() -> None:
    """EvalLoop.execute publishes SampleEvaluated events."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({"Q1": "A1", "Q2": "A2"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (
        Sample(id="1", input="Q1", expected=_Output(answer="A1")),
        Sample(id="2", input="Q2", expected=_Output(answer="A2")),
    )

    events: list[SampleEvaluated] = []
    bus.subscribe(SampleEvaluated, lambda e: events.append(e))

    eval_loop.execute(dataset)

    assert len(events) == 2
    assert events[0].sample_id == "1"
    assert events[1].sample_id == "2"


def test_eval_loop_execute_with_custom_config() -> None:
    """EvalLoop accepts custom config."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({"Q": "A"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    config = EvalLoopConfig()  # Future: add config options
    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus, config=config)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)

    report = eval_loop.execute(dataset)
    assert report.total == 1


# =============================================================================
# EvalLoop.handle_request() Tests
# =============================================================================


def test_eval_loop_handle_request_success() -> None:
    """EvalLoop.handle_request publishes EvalLoopCompleted on success."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({"Q": "A"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)
    request = EvalLoopRequest(dataset=dataset)

    completed_events: list[EvalLoopCompleted] = []
    bus.subscribe(EvalLoopCompleted, lambda e: completed_events.append(e))

    eval_loop.handle_request(request)

    assert len(completed_events) == 1
    assert completed_events[0].request_id == request.request_id
    assert completed_events[0].report.total == 1


def test_eval_loop_handle_request_evaluator_error() -> None:
    """EvalLoop.handle_request records evaluator errors in results (not re-raised)."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({})
    loop = _TestLoop(adapter=adapter, bus=bus)

    # Evaluator that always raises
    def failing_evaluator(output: _Output, expected: _Output) -> Score:
        raise RuntimeError("Evaluator failure")

    eval_loop = EvalLoop(loop=loop, evaluator=failing_evaluator, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)
    request = EvalLoopRequest(dataset=dataset)

    completed_events: list[EvalLoopCompleted] = []
    bus.subscribe(EvalLoopCompleted, lambda e: completed_events.append(e))

    # Per-sample evaluator errors are captured, not re-raised
    eval_loop.handle_request(request)

    # Should complete (not fail) with error recorded in result
    assert len(completed_events) == 1
    report = completed_events[0].report
    assert report.total == 1
    assert report.successful == 0  # Error means not successful
    assert "Evaluator failure" in (report.results[0].error or "")


# =============================================================================
# Event Bus Integration Tests
# =============================================================================


def test_eval_loop_auto_subscribes() -> None:
    """EvalLoop auto-subscribes to EvalLoopRequest on init."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({"Q": "A"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    # Creating EvalLoop auto-subscribes
    _ = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)

    completed_events: list[EvalLoopCompleted] = []
    bus.subscribe(EvalLoopCompleted, lambda e: completed_events.append(e))

    # Publishing request should trigger evaluation
    request = EvalLoopRequest(dataset=dataset)
    bus.publish(request)

    assert len(completed_events) == 1
    assert completed_events[0].request_id == request.request_id


def test_eval_loop_end_to_end_event_driven() -> None:
    """End-to-end test of event-driven EvalLoop workflow."""
    bus = InProcessEventBus()
    responses = {"What is 1+1?": "2", "What is 2+2?": "4", "What is 3+3?": "6"}
    adapter = _MockAdapter(responses)
    loop = _TestLoop(adapter=adapter, bus=bus)

    # Create EvalLoop (auto-subscribes)
    _ = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (
        Sample(id="1", input="What is 1+1?", expected=_Output(answer="2")),
        Sample(id="2", input="What is 2+2?", expected=_Output(answer="4")),
        Sample(id="3", input="What is 3+3?", expected=_Output(answer="6")),
    )

    # Track all events
    sample_events: list[SampleEvaluated] = []
    completed_events: list[EvalLoopCompleted] = []
    bus.subscribe(SampleEvaluated, lambda e: sample_events.append(e))
    bus.subscribe(EvalLoopCompleted, lambda e: completed_events.append(e))

    # Publish request
    request = EvalLoopRequest(dataset=dataset)
    bus.publish(request)

    # Verify sample events
    assert len(sample_events) == 3
    assert all(e.result.score.passed for e in sample_events)

    # Verify completion
    assert len(completed_events) == 1
    report = completed_events[0].report
    assert report.total == 3
    assert report.pass_rate == 1.0


def test_eval_loop_records_latency() -> None:
    """EvalLoop records latency for each sample."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({"Q": "A"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)

    report = eval_loop.execute(dataset)

    assert report.results[0].latency_ms >= 0
    assert report.mean_latency_ms >= 0


def test_eval_loop_handle_request_fatal_error() -> None:
    """EvalLoop.handle_request publishes EvalLoopFailed on fatal errors."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({})
    loop = _TestLoop(adapter=adapter, bus=bus)

    eval_loop = EvalLoop(loop=loop, evaluator=_output_match, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)
    request = EvalLoopRequest(dataset=dataset)

    failed_events: list[EvalLoopFailed] = []
    bus.subscribe(EvalLoopFailed, lambda e: failed_events.append(e))

    # Mock execute() to raise a fatal error
    with patch.object(eval_loop, "execute", side_effect=RuntimeError("Fatal error")):
        with pytest.raises(RuntimeError, match="Fatal error"):
            eval_loop.handle_request(request)

    assert len(failed_events) == 1
    assert failed_events[0].request_id == request.request_id
    assert isinstance(failed_events[0].error, RuntimeError)
    assert str(failed_events[0].error) == "Fatal error"
    assert failed_events[0].partial_report is None
