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
from pathlib import Path

import pytest

from tests.helpers.time import ControllableClock
from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.evals import (
    BASELINE,
    CONTROL,
    Dataset,
    EvalLoop,
    EvalRequest,
    EvalResult,
    Sample,
    Score,
    collect_results,
    exact_match,
    submit_dataset,
    submit_experiments,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import AgentLoop, InMemoryMailbox, Session
from weakincentives.runtime.agent_loop import AgentLoopRequest, AgentLoopResult
from weakincentives.runtime.dlq import DeadLetter, DLQPolicy
from weakincentives.runtime.mailbox import (
    Mailbox,
    ReceiptHandleExpiredError,
)
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
        heartbeat: object = None,
        run_context: object = None,
    ) -> PromptResponse[_Output]:
        del prompt, session, deadline, budget, budget_tracker, heartbeat, run_context
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return PromptResponse(
            prompt_name="test",
            text=self._result,
            output=_Output(result=self._result),
        )


class _TestLoop(AgentLoop[str, _Output]):
    """Test AgentLoop implementation."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[AgentLoopRequest[str], AgentLoopResult[_Output]],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests)
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

    def prepare(
        self,
        request: str,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[_Output], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(_Params(content=request))
        session = Session(tags={"loop": "test"})
        return prompt, session


def _create_test_loop(
    *,
    result: str = "success",
    error: Exception | None = None,
) -> _TestLoop:
    """Create a test AgentLoop with mock adapter."""
    adapter = _MockAdapter(result=result, error=error)
    # EvalLoop doesn't use AgentLoop's mailboxes directly, but AgentLoop requires one
    requests: InMemoryMailbox[AgentLoopRequest[str], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="dummy-requests")
    )
    return _TestLoop(adapter=adapter, requests=requests)


def _output_to_str(output: _Output, expected: str) -> Score:
    """Convert _Output to string for evaluation."""
    return exact_match(output.result, expected)


# =============================================================================
# EvalLoop Tests
# =============================================================================


def test_eval_loop_processes_sample() -> None:
    """EvalLoop processes a sample and produces result."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        # Submit a sample with reply_to
        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

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
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(error=RuntimeError("test error"))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

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
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        # Submit multiple samples
        for i in range(5):
            sample = Sample(id=str(i), input=f"input-{i}", expected="success")
            requests.send(
                EvalRequest(sample=sample, experiment=BASELINE), reply_to=results
            )

        # Run only 2 iterations
        eval_loop.run(max_iterations=2)

        # Should have processed some samples
        assert results.approximate_count() >= 1
    finally:
        requests.close()
        results.close()


def test_eval_loop_failing_score() -> None:
    """EvalLoop correctly reports failing evaluations."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="wrong")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

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
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        samples = (
            Sample(id="1", input="a", expected="b"),
            Sample(id="2", input="c", expected="d"),
            Sample(id="3", input="e", expected="f"),
        )
        dataset = Dataset(samples=samples)

        submit_dataset(dataset, BASELINE, requests)

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
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        dataset: Dataset[str, str] = Dataset(samples=())
        submit_dataset(dataset, BASELINE, requests)
        assert requests.approximate_count() == 0
    finally:
        requests.close()


# =============================================================================
# submit_experiments Tests
# =============================================================================


def test_submit_experiments() -> None:
    """submit_experiments sends all samples under each experiment."""
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        samples = (
            Sample(id="1", input="a", expected="b"),
            Sample(id="2", input="c", expected="d"),
        )
        dataset = Dataset(samples=samples)
        experiments = [BASELINE, CONTROL]

        count = submit_experiments(dataset, experiments, requests)

        # 2 samples * 2 experiments = 4 requests
        assert count == 4
        assert requests.approximate_count() == 4

        # Verify experiment distribution
        msgs = requests.receive(max_messages=4)
        experiments_sent = [msg.body.experiment.name for msg in msgs]
        assert experiments_sent.count("baseline") == 2
        assert experiments_sent.count("control") == 2
        for msg in msgs:
            msg.acknowledge()
    finally:
        requests.close()


# =============================================================================
# collect_results Tests
# =============================================================================


def test_collect_results() -> None:
    """collect_results gathers results into report."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")

    try:
        # Send some results
        for i in range(3):
            results.send(
                EvalResult(
                    sample_id=str(i),
                    experiment_name="baseline",
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
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")

    try:
        # Send only 1 result but expect 2
        results.send(
            EvalResult(
                sample_id="1",
                experiment_name="baseline",
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
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")

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
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Create dataset
        samples = (
            Sample(id="1", input="a", expected="success"),
            Sample(id="2", input="b", expected="success"),
            Sample(id="3", input="c", expected="wrong"),  # This one should fail
        )
        dataset = Dataset(samples=samples)

        # Create loop
        agent_loop = _create_test_loop(result="success")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        # Submit with reply_to and evaluate
        for sample in dataset:
            requests.send(
                EvalRequest(sample=sample, experiment=BASELINE), reply_to=results
            )
        eval_loop.run(max_iterations=5, wait_time_seconds=0)

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
        heartbeat: object = None,
        run_context: object = None,
    ) -> PromptResponse[_Output]:
        del prompt, session, deadline, budget, budget_tracker, heartbeat, run_context
        return PromptResponse(
            prompt_name="test",
            text="no structured output",
            output=None,
        )


class _NoneOutputLoop(AgentLoop[str, _Output]):
    """AgentLoop that returns None output."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[AgentLoopRequest[str], AgentLoopResult[_Output]],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests)
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

    def prepare(
        self,
        request: str,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[_Output], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(_Params(content=request))
        session = Session(tags={"loop": "test"})
        return prompt, session


def test_eval_loop_none_output() -> None:
    """EvalLoop handles None output from adapter."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        adapter = _NoneOutputAdapter()
        dummy_requests: InMemoryMailbox[
            AgentLoopRequest[str], AgentLoopResult[_Output]
        ] = InMemoryMailbox(name="dummy-requests")
        agent_loop = _NoneOutputLoop(adapter=adapter, requests=dummy_requests)
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "1"
        assert result.score.passed is False
        assert result.score.value == 0.0
        assert result.score.reason == "No output from AgentLoop"
        assert result.error == "No output from AgentLoop"
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dummy_requests.close()


class _FailingMailbox(InMemoryMailbox[EvalResult, None]):
    """Mailbox that fails on send for testing."""

    send_attempts: int = 0

    def __init__(self, *, fail_on_send: bool = False) -> None:
        super().__init__(name="failing-results")
        self._fail_on_send = fail_on_send
        self.send_attempts = 0

    def send(
        self, message: EvalResult, *, reply_to: Mailbox[None, None] | None = None
    ) -> str:
        _ = reply_to
        self.send_attempts += 1
        if self._fail_on_send:
            msg = "Simulated send failure"
            raise RuntimeError(msg)
        return super().send(message)


def test_eval_loop_nacks_on_send_failure() -> None:
    """EvalLoop nacks message when result send fails (not acknowledges)."""
    failing_mailbox = _FailingMailbox(fail_on_send=True)
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Create a successful loop - the key point is that evaluation succeeds
        # but send fails, so we should nack (not fabricate an error result)
        agent_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(
            EvalRequest(sample=sample, experiment=BASELINE), reply_to=failing_mailbox
        )

        # Run - evaluation succeeds, but send fails, should nack
        eval_loop.run(max_iterations=1)

        # The message should have been nacked (not acknowledged), so it should
        # still be in the queue after visibility timeout expires
        assert requests.approximate_count() == 1
        # Verify send was attempted
        assert failing_mailbox.send_attempts == 1
    finally:
        requests.close()
        failing_mailbox.close()


def test_eval_loop_nacks_on_send_failure_after_eval_error() -> None:
    """EvalLoop nacks message when send fails even after evaluation error."""
    failing_mailbox = _FailingMailbox(fail_on_send=True)
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Create loop that will throw an exception during evaluation
        agent_loop = _create_test_loop(error=RuntimeError("eval error"))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(
            EvalRequest(sample=sample, experiment=BASELINE), reply_to=failing_mailbox
        )

        # Run - evaluation fails, send also fails, should nack
        eval_loop.run(max_iterations=1)

        # The message should have been nacked (not acknowledged)
        assert requests.approximate_count() == 1
        assert failing_mailbox.send_attempts == 1
    finally:
        requests.close()
        failing_mailbox.close()


class _ExpiredHandleMailbox(InMemoryMailbox[EvalResult, None]):
    """Mailbox that raises ReceiptHandleExpiredError on send."""

    def send(
        self, message: EvalResult, *, reply_to: Mailbox[None, None] | None = None
    ) -> str:
        _ = (message, reply_to)
        raise ReceiptHandleExpiredError("Handle expired")


def test_eval_loop_handles_expired_receipt_on_send() -> None:
    """EvalLoop handles ReceiptHandleExpiredError on send gracefully."""
    expired_mailbox = _ExpiredHandleMailbox(name="expired-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(
            EvalRequest(sample=sample, experiment=BASELINE), reply_to=expired_mailbox
        )

        # Run - should handle ReceiptHandleExpiredError gracefully (pass, not raise)
        eval_loop.run(max_iterations=1)

        # Message was processed (expired handle means message already requeued)
        # The mailbox should be empty since we don't ack or nack on expired handle
        assert requests.approximate_count() == 1
    finally:
        requests.close()
        expired_mailbox.close()


class _NackExpiresRequestMailbox(InMemoryMailbox[EvalRequest[str, str], EvalResult]):
    """Request mailbox where nack raises ReceiptHandleExpiredError."""

    def receive(
        self,
        *,
        max_messages: int = 10,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> list[object]:
        msgs = super().receive(
            max_messages=max_messages,
            visibility_timeout=visibility_timeout,
            wait_time_seconds=wait_time_seconds,
        )
        # Wrap each message's nack method to raise ReceiptHandleExpiredError
        return [_NackExpiresMessage(msg) for msg in msgs]


class _NackExpiresMessage:
    """Message wrapper that raises ReceiptHandleExpiredError on nack."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    @property
    def body(self) -> object:
        return self._inner.body  # type: ignore[attr-defined]

    @property
    def id(self) -> str:
        return self._inner.id  # type: ignore[attr-defined, no-any-return]

    @property
    def reply_to(self) -> Mailbox[object, None] | None:
        return self._inner.reply_to  # type: ignore[attr-defined, no-any-return]

    @property
    def delivery_count(self) -> int:
        return self._inner.delivery_count  # type: ignore[attr-defined,no-any-return]

    def reply(self, body: object) -> str:
        return self._inner.reply(body)  # type: ignore[attr-defined, no-any-return]

    def acknowledge(self) -> None:
        self._inner.acknowledge()  # type: ignore[attr-defined]

    def nack(self, *, visibility_timeout: int = 0) -> None:
        _ = visibility_timeout
        raise ReceiptHandleExpiredError("Handle expired on nack")


def test_eval_loop_handles_expired_receipt_on_nack() -> None:
    """EvalLoop handles ReceiptHandleExpiredError on nack gracefully."""
    failing_mailbox = _FailingMailbox(fail_on_send=True)
    requests: _NackExpiresRequestMailbox = _NackExpiresRequestMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,  # type: ignore[arg-type]
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(
            EvalRequest(sample=sample, experiment=BASELINE), reply_to=failing_mailbox
        )

        # Run - send fails, nack raises ReceiptHandleExpiredError, should handle gracefully
        eval_loop.run(max_iterations=1, wait_time_seconds=0)

        # Should not raise, just pass
    finally:
        requests.close()
        failing_mailbox.close()


def test_eval_loop_exits_when_mailbox_closed() -> None:
    """EvalLoop exits cleanly when requests mailbox is closed."""
    import threading

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        # Close the mailbox before running
        requests.close()

        # Run with no max_iterations - should exit immediately due to closed mailbox
        # If the closed check is missing, this would spin forever
        loop_completed = threading.Event()

        def run_loop() -> None:
            eval_loop.run(max_iterations=None)
            loop_completed.set()

        thread = threading.Thread(target=run_loop)
        thread.start()

        # Wait up to 1 second for the loop to exit
        completed = loop_completed.wait(timeout=1.0)
        assert completed, "EvalLoop did not exit when mailbox was closed"

        thread.join(timeout=1.0)
        assert not thread.is_alive(), "EvalLoop thread still running"
    finally:
        results.close()


def test_eval_loop_handles_no_reply_to() -> None:
    """EvalLoop logs warning and acks when message has no reply_to."""
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        # Send without reply_to
        requests.send(EvalRequest(sample=sample, experiment=BASELINE))

        # Run - should handle missing reply_to gracefully (log warning, ack)
        eval_loop.run(max_iterations=1)

        # Message should have been acknowledged
        assert requests.approximate_count() == 0
    finally:
        requests.close()


def _session_aware_evaluator(
    output: object,
    expected: object,
    session: SessionProtocol,
) -> Score:
    """Session-aware evaluator for testing the 3-param path."""
    _ = session  # Use session parameter to mark as session-aware
    if isinstance(output, _Output) and isinstance(expected, str):
        return exact_match(output.result, expected)
    return Score(value=0.0, passed=False, reason="Type mismatch")


def test_eval_loop_with_session_aware_evaluator() -> None:
    """EvalLoop correctly passes session to session-aware evaluators."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_session_aware_evaluator,
            requests=requests,
        )

        # Submit a sample with reply_to
        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

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


# =============================================================================
# EvalLoop DLQ Integration Tests
# =============================================================================


def test_eval_loop_error_reply_without_dlq() -> None:
    """EvalLoop sends error reply without DLQ configured (original behavior)."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(error=RuntimeError("failure"))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

        eval_loop.run(max_iterations=1)

        # Message should be acknowledged (removed from queue)
        assert requests.approximate_count() == 0

        # Error reply should be sent
        assert results.approximate_count() == 1
        msgs = results.receive(max_messages=1)
        assert not msgs[0].body.score.passed
        assert "failure" in (msgs[0].body.error or "")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_nacks_with_dlq_before_threshold() -> None:
    """EvalLoop nacks failed messages with DLQ before threshold is reached."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[EvalRequest[str, str]], None] = (
        InMemoryMailbox(name="eval-requests-dlq")
    )

    try:
        agent_loop = _create_test_loop(error=RuntimeError("failure"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox, max_delivery_count=5
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            dlq=dlq,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

        # Run once - should nack for retry (below threshold)
        eval_loop.run(max_iterations=1)

        # Message should be nacked (still in queue)
        assert requests.approximate_count() == 1

        # No error reply sent on retry path
        assert results.approximate_count() == 0

        # Not dead-lettered yet
        assert dlq_mailbox.approximate_count() == 0
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_eval_loop_sends_to_dlq_after_threshold() -> None:
    """EvalLoop sends to DLQ when delivery count equals threshold."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[EvalRequest[str, str]], None] = (
        InMemoryMailbox(name="eval-requests-dlq")
    )

    try:
        agent_loop = _create_test_loop(error=RuntimeError("persistent failure"))
        # Use max_delivery_count=1 to trigger DLQ on first failure
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox, max_delivery_count=1
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            dlq=dlq,
        )

        sample = Sample(id="sample-1", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        # Run once - should dead-letter immediately (delivery_count=1 >= max=1)
        eval_loop.run(max_iterations=1)

        # Message should be dead-lettered
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1

        # Check dead letter content
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        dead_letter = dlq_msgs[0].body
        assert dead_letter.body.sample.id == "sample-1"
        assert dead_letter.source_mailbox == "eval-requests"
        assert dead_letter.delivery_count == 1
        assert "persistent failure" in dead_letter.last_error
        dlq_msgs[0].acknowledge()

        # Error reply should be sent
        result_msgs = results.receive(max_messages=1)
        assert len(result_msgs) == 1
        assert not result_msgs[0].body.score.passed
        assert "Dead-lettered" in (result_msgs[0].body.error or "")
        result_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_eval_loop_immediate_dlq_for_included_error() -> None:
    """EvalLoop immediately dead-letters included error types."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[EvalRequest[str, str]], None] = (
        InMemoryMailbox(name="eval-requests-dlq")
    )

    try:
        agent_loop = _create_test_loop(error=ValueError("validation error"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=5,
            include_errors=frozenset({ValueError}),
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            dlq=dlq,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

        # Run once - should immediately dead-letter
        eval_loop.run(max_iterations=1)

        # Message should be dead-lettered on first attempt
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1

        # Check delivery count is 1 (immediate dead-letter)
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert dlq_msgs[0].body.delivery_count == 1
        dlq_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_eval_loop_never_dlq_for_excluded_error() -> None:
    """EvalLoop never dead-letters excluded error types."""
    clock = ControllableClock()
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(
        name="eval-results", clock=clock
    )
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests", clock=clock
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[EvalRequest[str, str]], None] = (
        InMemoryMailbox(name="eval-requests-dlq", clock=clock)
    )

    try:
        agent_loop = _create_test_loop(error=TimeoutError("transient timeout"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=2,
            exclude_errors=frozenset({TimeoutError}),
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            dlq=dlq,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

        # Run many times - should never dead-letter
        # After each run, advance clock past backoff and trigger reap synchronously
        for _ in range(5):
            eval_loop.run(max_iterations=1, wait_time_seconds=0)
            # Advance clock past the backoff (min(60 * delivery_count, 900))
            clock.advance(1000)
            requests._reap_expired()  # Deterministic requeue

        # Message should still be in queue (nacked, not dead-lettered)
        assert requests.approximate_count() == 1
        assert dlq_mailbox.approximate_count() == 0
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_eval_loop_dlq_handles_reply_error() -> None:
    """EvalLoop DLQ handles errors when sending reply."""
    from weakincentives.runtime.mailbox import FakeMailbox, MailboxConnectionError

    results: FakeMailbox[EvalResult, None] = FakeMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[EvalRequest[str, str]], None] = (
        InMemoryMailbox(name="eval-requests-dlq")
    )

    try:
        agent_loop = _create_test_loop(error=RuntimeError("failure"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox, max_delivery_count=1
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            dlq=dlq,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

        # Make reply send fail
        results.set_connection_error(MailboxConnectionError("connection lost"))

        eval_loop.run(max_iterations=1)

        # Message should still be dead-lettered despite reply failure
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1
    finally:
        requests.close()
        dlq_mailbox.close()


def test_eval_loop_dlq_without_reply_to() -> None:
    """EvalLoop DLQ handles messages without reply_to."""
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[EvalRequest[str, str]], None] = (
        InMemoryMailbox(name="eval-requests-dlq")
    )

    try:
        agent_loop = _create_test_loop(error=RuntimeError("failure"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox, max_delivery_count=1
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            dlq=dlq,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        # Send without reply_to
        requests.send(EvalRequest(sample=sample, experiment=BASELINE))

        eval_loop.run(max_iterations=1)

        # Message should be dead-lettered
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1

        # reply_to should be None in dead letter
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert dlq_msgs[0].body.reply_to is None
        dlq_msgs[0].acknowledge()
    finally:
        requests.close()
        dlq_mailbox.close()


# =============================================================================
# EvalLoop Debug Bundle Tests
# =============================================================================


def test_eval_loop_creates_debug_bundle(tmp_path: Path) -> None:
    """EvalLoop creates debug bundle when debug_bundle is configured."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-1", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        # Check result
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "sample-1"
        assert result.score.passed is True
        assert result.bundle_path is not None

        # Verify bundle exists and is valid
        assert result.bundle_path.exists()
        bundle = DebugBundle.load(result.bundle_path)

        # Verify bundle contains expected artifacts
        files = bundle.list_files()
        assert "manifest.json" in files
        assert "request/input.json" in files
        assert "request/output.json" in files
        assert "logs/app.jsonl" in files
        assert "metrics.json" in files
        assert "eval.json" in files
        # session/after.jsonl is only written if session has slices

        # Verify eval.json content
        eval_data = bundle.eval
        assert eval_data is not None
        assert eval_data["sample_id"] == "sample-1"
        assert eval_data["experiment_name"] == "baseline"
        assert eval_data["score"]["passed"] is True
        assert eval_data["score"]["value"] == 1.0
        assert eval_data["latency_ms"] >= 0

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_contains_request_id_directory(tmp_path: Path) -> None:
    """EvalLoop creates bundle in request-specific directory."""
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-1", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.bundle_path is not None

        # Bundle should be in a request_id subdirectory
        assert result.bundle_path.parent.parent == tmp_path
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_captures_failed_evaluation(tmp_path: Path) -> None:
    """EvalLoop bundle captures evaluation failures."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Create loop that returns "wrong" - will fail exact_match
        agent_loop = _create_test_loop(result="wrong")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-fail", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.score.passed is False
        assert result.bundle_path is not None

        # Verify bundle captures the failed evaluation
        bundle = DebugBundle.load(result.bundle_path)
        eval_data = bundle.eval
        assert eval_data is not None
        assert eval_data["score"]["passed"] is False
        assert eval_data["score"]["value"] == 0.0

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_captures_none_output(tmp_path: Path) -> None:
    """EvalLoop bundle captures None output scenario."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Use the NoneOutputAdapter
        adapter = _NoneOutputAdapter()
        dummy_requests: InMemoryMailbox[
            AgentLoopRequest[str], AgentLoopResult[_Output]
        ] = InMemoryMailbox(name="dummy-requests")
        agent_loop = _NoneOutputLoop(adapter=adapter, requests=dummy_requests)
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-none", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.score.passed is False
        assert result.error == "No output from AgentLoop"
        assert result.bundle_path is not None

        # Verify bundle captures the error
        bundle = DebugBundle.load(result.bundle_path)
        eval_data = bundle.eval
        assert eval_data is not None
        assert eval_data["error"] == "No output from AgentLoop"
        assert eval_data["score"]["reason"] == "No output from AgentLoop"

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dummy_requests.close()


def test_eval_loop_no_bundle_without_config() -> None:
    """EvalLoop does not create bundle when debug_bundle is not set."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        # No debug_bundle configured
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample, experiment=BASELINE), reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.bundle_path is None
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_with_session_aware_evaluator(tmp_path: Path) -> None:
    """EvalLoop bundle works with session-aware evaluators."""
    from weakincentives.debug import DebugBundle
    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_session_aware_evaluator,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-session", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.score.passed is True
        assert result.bundle_path is not None

        # Verify bundle is valid
        bundle = DebugBundle.load(result.bundle_path)
        eval_data = bundle.eval
        assert eval_data is not None
        assert eval_data["score"]["passed"] is True

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_fallback_on_error(tmp_path: Path) -> None:
    """EvalLoop falls back to non-bundled path when bundle creation fails."""
    from unittest.mock import patch

    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-fail-bundle", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        # Mock BundleWriter to raise an exception
        with patch(
            "weakincentives.debug._bundle_writer.BundleWriter.__enter__",
            side_effect=RuntimeError("Simulated bundle creation failure"),
        ):
            eval_loop.run(max_iterations=1)

        # Result should still be successful (just without bundle)
        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.sample_id == "sample-fail-bundle"
        assert result.score.passed is True
        assert result.bundle_path is None  # Bundle creation failed
        assert result.error is None  # But evaluation succeeded

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_no_reexecution_on_finalization_error(tmp_path: Path) -> None:
    """EvalLoop does NOT re-execute when bundle finalization fails after execution.

    This test verifies the fix for the data consistency bug where a post-execution
    failure (e.g., in bundle writing) would cause the sample to be re-executed,
    potentially returning different results and creating inconsistency.
    """
    import contextlib
    from collections.abc import Iterator
    from unittest.mock import patch

    from weakincentives.debug.bundle import BundleConfig
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    execution_count = 0

    try:
        agent_loop = _create_test_loop(result="correct")
        config = EvalLoopConfig(debug_bundle=BundleConfig(target=tmp_path))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(
            id="sample-finalize-fail", input="test input", expected="correct"
        )
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        # Mock the context manager's __exit__ to raise after execution completes
        original_execute_with_bundle = agent_loop.execute_with_bundle

        @contextlib.contextmanager
        def failing_bundle_context(*args: object, **kwargs: object) -> Iterator[object]:
            nonlocal execution_count
            with original_execute_with_bundle(*args, **kwargs) as ctx:  # type: ignore[arg-type]
                execution_count += 1
                yield ctx
            # Raise during finalization (after yield returns)
            raise RuntimeError("Simulated finalization failure")

        with patch.object(agent_loop, "execute_with_bundle", failing_bundle_context):
            eval_loop.run(max_iterations=1)

        # Verify execution happened exactly once (no re-execution)
        assert execution_count == 1, f"Expected 1 execution, got {execution_count}"

        # Result should still be successful
        msgs = results.receive(max_messages=1)
        result = msgs[0].body
        assert result.sample_id == "sample-finalize-fail"
        assert result.score.passed is True
        assert result.error is None  # Evaluation succeeded

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_invokes_storage_handler(tmp_path: Path) -> None:
    """EvalLoop invokes storage_handler when bundle is finalized."""
    from weakincentives.debug.bundle import BundleConfig, BundleManifest
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    # Track storage handler invocations
    stored_bundles: list[tuple[Path, BundleManifest]] = []

    class TestStorageHandler:
        """Test storage handler that records invocations."""

        def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
            stored_bundles.append((bundle_path, manifest))

    try:
        agent_loop = _create_test_loop(result="correct")
        storage_handler = TestStorageHandler()
        config = EvalLoopConfig(
            debug_bundle=BundleConfig(target=tmp_path, storage_handler=storage_handler)
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-upload", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        # Check result
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "sample-upload"
        assert result.score.passed is True
        assert result.bundle_path is not None

        # Verify storage handler was invoked
        assert len(stored_bundles) == 1
        stored_path, stored_manifest = stored_bundles[0]
        assert stored_path == result.bundle_path
        assert stored_manifest.bundle_id is not None
        assert stored_manifest.request.status == "success"

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_eval_loop_bundle_storage_handler_error_does_not_fail_eval(
    tmp_path: Path,
) -> None:
    """EvalLoop evaluation succeeds even if storage handler fails."""
    from weakincentives.debug.bundle import BundleConfig, BundleManifest
    from weakincentives.evals import EvalLoopConfig

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    class FailingStorageHandler:
        """Storage handler that always fails."""

        def store_bundle(self, bundle_path: Path, manifest: BundleManifest) -> None:
            raise RuntimeError("Simulated upload failure")

    try:
        agent_loop = _create_test_loop(result="correct")
        config = EvalLoopConfig(
            debug_bundle=BundleConfig(
                target=tmp_path, storage_handler=FailingStorageHandler()
            )
        )
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=_output_to_str,
            requests=requests,
            config=config,
        )

        sample = Sample(id="sample-fail-upload", input="test input", expected="correct")
        request = EvalRequest(sample=sample, experiment=BASELINE)
        requests.send(request, reply_to=results)

        eval_loop.run(max_iterations=1)

        # Evaluation should still succeed
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.sample_id == "sample-fail-upload"
        assert result.score.passed is True
        assert result.error is None  # No eval error despite storage failure
        assert result.bundle_path is not None  # Bundle was still created locally

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
