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

"""Tests for EvalLoop and helper functions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.evals import (
    Dataset,
    EvalLoop,
    EvalRequest,
    EvalResult,
    Sample,
    Score,
    collect_results,
    exact_match,
    submit_dataset,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.prompt.tool import ResourceRegistry
from weakincentives.runtime import InMemoryMailbox, MainLoop, Session
from weakincentives.runtime.session import SessionProtocol

# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass(slots=True, frozen=True)
class _Params:
    """Test prompt params."""

    content: str


@dataclass(slots=True, frozen=True)
class _Output:
    """Test output type."""

    result: str


class _MockAdapter(ProviderAdapter[_Output]):
    """Mock adapter for testing."""

    def __init__(
        self,
        *,
        result: str = "success",
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.call_count = 0

    def evaluate(
        self,
        prompt: Prompt[_Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        resources: ResourceRegistry | None = None,
    ) -> PromptResponse[_Output]:
        del prompt, session, deadline, budget, budget_tracker, resources
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return PromptResponse(
            prompt_name="test",
            text=self._result,
            output=_Output(result=self._result),
        )


class _TestLoop(MainLoop[str, _Output]):
    """Test MainLoop implementation."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[object],
        responses: InMemoryMailbox[object],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, responses=responses)  # type: ignore[arg-type]
        self._template = PromptTemplate[_Output](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[_Params](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )

    def initialize(self, request: str) -> tuple[Prompt[_Output], Session]:
        prompt = Prompt(self._template).bind(_Params(content=request))
        session = Session(tags={"loop": "test"})
        return prompt, session


def _create_test_loop(
    *,
    result: str = "success",
    error: Exception | None = None,
) -> _TestLoop:
    """Create a test MainLoop with mock adapter."""
    adapter = _MockAdapter(result=result, error=error)
    # EvalLoop doesn't use MainLoop's mailboxes, but MainLoop requires them
    requests: InMemoryMailbox[object] = InMemoryMailbox(name="dummy-requests")
    responses: InMemoryMailbox[object] = InMemoryMailbox(name="dummy-responses")
    return _TestLoop(adapter=adapter, requests=requests, responses=responses)


def _output_to_str(output: _Output, expected: str) -> Score:
    """Convert _Output to string for evaluation."""
    return exact_match(output.result, expected)


# =============================================================================
# EvalLoop Tests
# =============================================================================


def test_eval_loop_processes_sample() -> None:
    """EvalLoop processes a sample and produces result."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        main_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
            results=results,
        )

        # Submit a sample
        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample))

        # Run single iteration
        eval_loop.run(max_iterations=1)

        # Check result
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "1"
        assert result.score.passed is True
        assert result.score.value == 1.0
        assert result.latency_ms >= 0
        assert result.error is None
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_handles_failure() -> None:
    """EvalLoop handles evaluation failure gracefully."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        main_loop = _create_test_loop(error=RuntimeError("test error"))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
            results=results,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample))

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "1"
        assert result.error == "test error"
        assert result.score.passed is False
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_respects_max_iterations() -> None:
    """EvalLoop respects max_iterations limit."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        main_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
            results=results,
        )

        # Submit multiple samples
        for i in range(5):
            sample = Sample(id=str(i), input=f"input-{i}", expected="success")
            requests.send(EvalRequest(sample=sample))

        # Run only 2 iterations
        eval_loop.run(max_iterations=2)

        # Should have processed some samples
        assert results.approximate_count() >= 1
    finally:
        requests.close()
        results.close()


def test_eval_loop_failing_score() -> None:
    """EvalLoop correctly reports failing evaluations."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        main_loop = _create_test_loop(result="wrong")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
            results=results,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample))

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.score.passed is False
        assert result.score.value == 0.0
        assert result.error is None  # No error, just failed evaluation
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


# =============================================================================
# submit_dataset Tests
# =============================================================================


def test_submit_dataset() -> None:
    """submit_dataset sends all samples to mailbox."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        samples = (
            Sample(id="1", input="a", expected="b"),
            Sample(id="2", input="c", expected="d"),
            Sample(id="3", input="e", expected="f"),
        )
        dataset = Dataset(samples=samples)

        submit_dataset(dataset, requests)

        assert requests.approximate_count() == 3

        # Verify samples were submitted correctly
        msgs = requests.receive(max_messages=3)
        assert len(msgs) == 3
        ids = {msg.body.sample.id for msg in msgs}
        assert ids == {"1", "2", "3"}
        for msg in msgs:
            msg.acknowledge()
    finally:
        requests.close()


def test_submit_dataset_empty() -> None:
    """submit_dataset handles empty dataset."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        dataset: Dataset[str, str] = Dataset(samples=())
        submit_dataset(dataset, requests)
        assert requests.approximate_count() == 0
    finally:
        requests.close()


# =============================================================================
# collect_results Tests
# =============================================================================


def test_collect_results() -> None:
    """collect_results gathers results into report."""
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        # Send some results
        for i in range(3):
            results.send(
                EvalResult(
                    sample_id=str(i),
                    score=Score(value=1.0 if i < 2 else 0.5, passed=i < 2),
                    latency_ms=100 + i * 50,
                )
            )

        report = collect_results(results, expected_count=3, timeout_seconds=5)

        assert report.total == 3
        assert report.pass_rate == pytest.approx(2 / 3)
    finally:
        results.close()


def test_collect_results_timeout() -> None:
    """collect_results respects timeout."""
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        # Send only 1 result but expect 2
        results.send(
            EvalResult(
                sample_id="1",
                score=Score(value=1.0, passed=True),
                latency_ms=100,
            )
        )

        # Short timeout - should return partial results
        report = collect_results(results, expected_count=2, timeout_seconds=0.1)

        # Should have collected what was available
        assert report.total == 1
    finally:
        results.close()


def test_collect_results_empty() -> None:
    """collect_results handles empty results."""
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        report = collect_results(results, expected_count=0, timeout_seconds=0.1)
        assert report.total == 0
        assert report.pass_rate == 0.0
    finally:
        results.close()


# =============================================================================
# End-to-End Tests
# =============================================================================


def test_end_to_end_evaluation() -> None:
    """Full evaluation flow from dataset to report."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        # Create dataset
        samples = (
            Sample(id="1", input="a", expected="success"),
            Sample(id="2", input="b", expected="success"),
            Sample(id="3", input="c", expected="wrong"),  # This one should fail
        )
        dataset = Dataset(samples=samples)

        # Create loop
        main_loop = _create_test_loop(result="success")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
            results=results,
        )

        # Submit and evaluate
        submit_dataset(dataset, requests)
        eval_loop.run(max_iterations=5)

        # Collect results
        report = collect_results(results, expected_count=3, timeout_seconds=5)

        # Verify
        assert report.total == 3
        assert report.successful == 3
        assert report.pass_rate == pytest.approx(2 / 3)  # 2 of 3 passed
        assert len(report.failed_samples()) == 1
        assert report.failed_samples()[0].sample_id == "3"
    finally:
        requests.close()
        results.close()


# =============================================================================
# Edge Case Tests for Coverage
# =============================================================================


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
        resources: ResourceRegistry | None = None,
    ) -> PromptResponse[_Output]:
        del prompt, session, deadline, budget, budget_tracker, resources
        return PromptResponse(
            prompt_name="test",
            text="no structured output",
            output=None,
        )


class _NoneOutputLoop(MainLoop[str, _Output]):
    """MainLoop that returns None output."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[object],
        responses: InMemoryMailbox[object],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, responses=responses)  # type: ignore[arg-type]
        self._template = PromptTemplate[_Output](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[_Params](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )

    def initialize(self, request: str) -> tuple[Prompt[_Output], Session]:
        prompt = Prompt(self._template).bind(_Params(content=request))
        session = Session(tags={"loop": "test"})
        return prompt, session


def test_eval_loop_none_output() -> None:
    """EvalLoop handles None output from adapter."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )
    results: InMemoryMailbox[EvalResult] = InMemoryMailbox(name="eval-results")

    try:
        adapter = _NoneOutputAdapter()
        dummy_requests: InMemoryMailbox[object] = InMemoryMailbox(name="dummy-requests")
        dummy_responses: InMemoryMailbox[object] = InMemoryMailbox(
            name="dummy-responses"
        )
        main_loop = _NoneOutputLoop(
            adapter=adapter, requests=dummy_requests, responses=dummy_responses
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
            results=results,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample))

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "1"
        assert result.score.passed is False
        assert result.score.value == 0.0
        assert result.score.reason == "No output from MainLoop"
        assert result.error == "No output from MainLoop"
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dummy_requests.close()
        dummy_responses.close()


class _FailingMailbox(InMemoryMailbox[EvalResult]):
    """Mailbox that fails on send for testing."""

    def __init__(self, *, fail_on_send: bool = False) -> None:
        super().__init__(name="failing-results")
        self._fail_on_send = fail_on_send

    def send(self, message: EvalResult) -> str:
        if self._fail_on_send:
            msg = "Simulated send failure"
            raise RuntimeError(msg)
        return super().send(message)


def test_eval_loop_handle_failure_nack() -> None:
    """EvalLoop nacks message when result send fails."""
    requests: InMemoryMailbox[EvalRequest[str, str]] = InMemoryMailbox(
        name="eval-requests"
    )
    # Create a mailbox that fails on send
    failing_results = _FailingMailbox(fail_on_send=True)

    try:
        # Create loop that will throw an exception during evaluation
        main_loop = _create_test_loop(error=RuntimeError("eval error"))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
            results=failing_results,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample))

        # Run - this should trigger _handle_failure, which will try to send to
        # results mailbox and fail, causing a nack
        eval_loop.run(max_iterations=1)

        # The message should have been nacked (not acknowledged), so it should
        # still be in the queue after visibility timeout expires
        # Since we can't easily verify nack was called, we check the request
        # mailbox still has the message after clearing visibility
        assert requests.approximate_count() == 1
    finally:
        requests.close()
        failing_results.close()
