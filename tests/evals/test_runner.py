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

"""Tests for run_eval function."""

from __future__ import annotations

from dataclasses import dataclass

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.evals import (
    EvalReport,
    Sample,
    SampleEvaluated,
    Score,
    run_eval,
)
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.prompt.prompt import PromptTemplate
from weakincentives.runtime import InProcessEventBus, MainLoop, Session
from weakincentives.runtime.events._types import EventBus
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
        # Extract question from params
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
        bus: EventBus,
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
# run_eval Basic Tests
# =============================================================================


def test_run_eval_all_pass() -> None:
    """run_eval handles all passing samples."""
    bus = InProcessEventBus()
    responses = {"What is 2+2?": "4", "Capital of France?": "Paris"}
    adapter = _MockAdapter(responses)
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (
        Sample(id="1", input="What is 2+2?", expected=_Output(answer="4")),
        Sample(id="2", input="Capital of France?", expected=_Output(answer="Paris")),
    )

    report = run_eval(loop, dataset, _output_match)

    assert isinstance(report, EvalReport)
    assert report.total == 2
    assert report.successful == 2
    assert report.pass_rate == 1.0
    assert report.mean_score == 1.0


def test_run_eval_mixed_results() -> None:
    """run_eval handles mixed pass/fail results."""
    bus = InProcessEventBus()
    responses = {"What is 2+2?": "4", "Capital of France?": "London"}
    adapter = _MockAdapter(responses)
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (
        Sample(id="1", input="What is 2+2?", expected=_Output(answer="4")),
        Sample(id="2", input="Capital of France?", expected=_Output(answer="Paris")),
    )

    report = run_eval(loop, dataset, _output_match)

    assert report.total == 2
    assert report.pass_rate == 0.5


def test_run_eval_all_fail() -> None:
    """run_eval handles all failing samples."""
    bus = InProcessEventBus()
    responses = {"Q1": "wrong", "Q2": "wrong"}
    adapter = _MockAdapter(responses)
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (
        Sample(id="1", input="Q1", expected=_Output(answer="right")),
        Sample(id="2", input="Q2", expected=_Output(answer="right")),
    )

    report = run_eval(loop, dataset, _output_match)

    assert report.pass_rate == 0.0
    assert len(report.failed_samples()) == 2


def test_run_eval_empty_dataset() -> None:
    """run_eval handles empty dataset."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({})
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset: tuple[Sample[str, _Output], ...] = ()

    report = run_eval(loop, dataset, _output_match)

    assert report.total == 0
    assert report.pass_rate == 0.0


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_run_eval_handles_errors() -> None:
    """run_eval catches and records errors."""
    bus = InProcessEventBus()
    adapter = _ErrorAdapter({"Q1"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (
        Sample(id="1", input="Q1", expected=_Output(answer="answer")),
        Sample(id="2", input="Q2", expected=_Output(answer="ok")),
    )

    report = run_eval(loop, dataset, _output_match)

    assert report.total == 2
    assert report.successful == 1  # Only Q2 succeeded

    # Check error result
    error_result = next(r for r in report.results if r.sample_id == "1")
    assert error_result.error is not None
    assert "Error processing" in error_result.error
    assert error_result.success is False


def test_run_eval_error_score() -> None:
    """run_eval records zero score for errors."""
    bus = InProcessEventBus()
    adapter = _ErrorAdapter({"Q1"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (Sample(id="1", input="Q1", expected=_Output(answer="answer")),)

    report = run_eval(loop, dataset, _output_match)

    error_result = report.results[0]
    assert error_result.score.value == 0.0
    assert error_result.score.passed is False


# =============================================================================
# Latency Tracking Tests
# =============================================================================


def test_run_eval_records_latency() -> None:
    """run_eval records latency for each sample."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({"Q": "A"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)

    report = run_eval(loop, dataset, _output_match)

    # Latency should be recorded (>= 0)
    assert report.results[0].latency_ms >= 0
    assert report.mean_latency_ms >= 0


# =============================================================================
# Event Bus Integration Tests
# =============================================================================


def test_run_eval_publishes_events() -> None:
    """run_eval publishes SampleEvaluated events."""
    control_bus = InProcessEventBus()
    event_bus = InProcessEventBus()
    adapter = _MockAdapter({"Q1": "A1", "Q2": "A2"})
    loop = _TestLoop(adapter=adapter, bus=control_bus)

    dataset = (
        Sample(id="1", input="Q1", expected=_Output(answer="A1")),
        Sample(id="2", input="Q2", expected=_Output(answer="A2")),
    )

    events: list[SampleEvaluated] = []
    event_bus.subscribe(SampleEvaluated, lambda e: events.append(e))

    run_eval(loop, dataset, _output_match, bus=event_bus)

    assert len(events) == 2
    assert events[0].sample_id == "1"
    assert events[1].sample_id == "2"


def test_run_eval_no_bus() -> None:
    """run_eval works without event bus."""
    control_bus = InProcessEventBus()
    adapter = _MockAdapter({"Q": "A"})
    loop = _TestLoop(adapter=adapter, bus=control_bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)

    # Should not raise
    report = run_eval(loop, dataset, _output_match, bus=None)
    assert report.total == 1


# =============================================================================
# Custom Evaluator Tests
# =============================================================================


def test_run_eval_custom_evaluator() -> None:
    """run_eval works with custom evaluators."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({"Q": "The answer is 42"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="42")),)

    def length_check(output: _Output, expected: _Output) -> Score:
        passed = len(output.answer) > len(expected.answer)
        return Score(value=1.0 if passed else 0.0, passed=passed)

    report = run_eval(loop, dataset, length_check)

    assert report.pass_rate == 1.0


def test_run_eval_evaluator_receives_correct_args() -> None:
    """run_eval passes output and expected to evaluator."""
    bus = InProcessEventBus()
    adapter = _MockAdapter({"input_q": "output_val"})
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (
        Sample(id="1", input="input_q", expected=_Output(answer="expected_val")),
    )

    received: list[tuple[_Output, _Output]] = []

    def capturing_evaluator(output: _Output, expected: _Output) -> Score:
        received.append((output, expected))
        return Score(value=1.0, passed=True)

    run_eval(loop, dataset, capturing_evaluator)

    assert len(received) == 1
    assert received[0] == (_Output(answer="output_val"), _Output(answer="expected_val"))


# =============================================================================
# None Output Tests
# =============================================================================


def test_run_eval_handles_none_output() -> None:
    """run_eval records error when adapter returns None output."""
    bus = InProcessEventBus()
    adapter = _NoneOutputAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus)

    dataset = (Sample(id="1", input="Q", expected=_Output(answer="A")),)

    report = run_eval(loop, dataset, _output_match)

    assert report.total == 1
    assert report.successful == 0
    result = report.results[0]
    assert result.error == "No output from loop"
    assert result.score.passed is False
    assert result.score.value == 0.0
