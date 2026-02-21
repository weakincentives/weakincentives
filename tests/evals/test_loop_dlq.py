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

"""Tests for EvalLoop DLQ integration and edge cases."""

from __future__ import annotations

from dataclasses import dataclass

from tests.helpers.time import ControllableClock
from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.evals import (
    BASELINE,
    EvalLoop,
    EvalRequest,
    EvalResult,
    Sample,
    Score,
    exact_match,
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
# Shared Test Fixtures (duplicated from test_loop.py)
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


# =============================================================================
# Edge Case Tests (continued from test_loop.py)
# =============================================================================


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
