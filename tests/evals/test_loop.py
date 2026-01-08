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
from weakincentives.resources import ResourceRegistry
from weakincentives.runtime import InMemoryMailbox, MainLoop, Session
from weakincentives.runtime.mailbox import (
    Mailbox,
    ReceiptHandleExpiredError,
)
from weakincentives.runtime.main_loop import MainLoopRequest, MainLoopResult
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
        requests: InMemoryMailbox[MainLoopRequest[str], MainLoopResult[_Output]],
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

    def prepare(self, request: str) -> tuple[Prompt[_Output], Session]:
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
    # EvalLoop doesn't use MainLoop's mailboxes directly, but MainLoop requires one
    requests: InMemoryMailbox[MainLoopRequest[str], MainLoopResult[_Output]] = (
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
        main_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        # Submit a sample with reply_to
        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=results)

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
        main_loop = _create_test_loop(error=RuntimeError("test error"))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=results)

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
        main_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        # Submit multiple samples
        for i in range(5):
            sample = Sample(id=str(i), input=f"input-{i}", expected="success")
            requests.send(EvalRequest(sample=sample), reply_to=results)

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
        main_loop = _create_test_loop(result="wrong")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=results)

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
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
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
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")

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
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")

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
        main_loop = _create_test_loop(result="success")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        # Submit with reply_to and evaluate
        for sample in dataset:
            requests.send(EvalRequest(sample=sample), reply_to=results)
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
        requests: InMemoryMailbox[MainLoopRequest[str], MainLoopResult[_Output]],
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

    def prepare(self, request: str) -> tuple[Prompt[_Output], Session]:
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
            MainLoopRequest[str], MainLoopResult[_Output]
        ] = InMemoryMailbox(name="dummy-requests")
        main_loop = _NoneOutputLoop(adapter=adapter, requests=dummy_requests)
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=results)

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
        main_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=failing_mailbox)

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
        main_loop = _create_test_loop(error=RuntimeError("eval error"))
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=failing_mailbox)

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
        main_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=expired_mailbox)

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
    def reply_to(self) -> str | None:
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
        main_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,  # type: ignore[arg-type]
        )

        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=failing_mailbox)

        # Run - send fails, nack raises ReceiptHandleExpiredError, should handle gracefully
        eval_loop.run(max_iterations=1)

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
        main_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
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
        main_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_output_to_str,
            requests=requests,
        )

        sample = Sample(id="1", input="test input", expected="correct")
        # Send without reply_to
        requests.send(EvalRequest(sample=sample))

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
        main_loop = _create_test_loop(result="correct")
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=_session_aware_evaluator,
            requests=requests,
        )

        # Submit a sample with reply_to
        sample = Sample(id="1", input="test input", expected="correct")
        requests.send(EvalRequest(sample=sample), reply_to=results)

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
