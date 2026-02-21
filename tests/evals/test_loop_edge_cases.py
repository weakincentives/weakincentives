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

"""Tests for EvalLoop edge cases and error handling."""

from __future__ import annotations

from tests.evals.conftest import (
    NoneOutputAdapter,
    NoneOutputLoop,
    Output,
    create_test_loop,
    output_to_str,
    session_aware_evaluator,
)
from weakincentives.evals import (
    BASELINE,
    EvalLoop,
    EvalRequest,
    EvalResult,
    Sample,
)
from weakincentives.runtime import InMemoryMailbox
from weakincentives.runtime.agent_loop import AgentLoopRequest, AgentLoopResult
from weakincentives.runtime.mailbox import (
    Mailbox,
    ReceiptHandleExpiredError,
)

# =============================================================================
# Edge Case Helper Classes
# =============================================================================


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


class _ExpiredHandleMailbox(InMemoryMailbox[EvalResult, None]):
    """Mailbox that raises ReceiptHandleExpiredError on send."""

    def send(
        self, message: EvalResult, *, reply_to: Mailbox[None, None] | None = None
    ) -> str:
        _ = (message, reply_to)
        raise ReceiptHandleExpiredError("Handle expired")


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


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_eval_loop_none_output() -> None:
    """EvalLoop handles None output from adapter."""
    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        adapter = NoneOutputAdapter()
        dummy_requests: InMemoryMailbox[
            AgentLoopRequest[str], AgentLoopResult[Output]
        ] = InMemoryMailbox(name="dummy-requests")
        agent_loop = NoneOutputLoop(adapter=adapter, requests=dummy_requests)
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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


def test_eval_loop_nacks_on_send_failure() -> None:
    """EvalLoop nacks message when result send fails (not acknowledges)."""
    failing_mailbox = _FailingMailbox(fail_on_send=True)
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Create a successful loop - the key point is that evaluation succeeds
        # but send fails, so we should nack (not fabricate an error result)
        agent_loop = create_test_loop(result="correct")
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(error=RuntimeError("eval error"))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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


def test_eval_loop_handles_expired_receipt_on_send() -> None:
    """EvalLoop handles ReceiptHandleExpiredError on send gracefully."""
    expired_mailbox = _ExpiredHandleMailbox(name="expired-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = create_test_loop(result="correct")
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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


def test_eval_loop_handles_expired_receipt_on_nack() -> None:
    """EvalLoop handles ReceiptHandleExpiredError on nack gracefully."""
    failing_mailbox = _FailingMailbox(fail_on_send=True)
    requests: _NackExpiresRequestMailbox = _NackExpiresRequestMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = create_test_loop(result="correct")
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(result="correct")
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(result="correct")
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(result="correct")
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=session_aware_evaluator,
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
