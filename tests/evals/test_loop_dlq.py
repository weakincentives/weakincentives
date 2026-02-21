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

"""Tests for EvalLoop DLQ (Dead Letter Queue) integration."""

from __future__ import annotations

from tests.evals.conftest import Output, create_test_loop, output_to_str
from tests.helpers.time import ControllableClock
from weakincentives.evals import (
    BASELINE,
    EvalLoop,
    EvalRequest,
    EvalResult,
    Sample,
)
from weakincentives.runtime import InMemoryMailbox
from weakincentives.runtime.dlq import DeadLetter, DLQPolicy

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
        agent_loop = create_test_loop(error=RuntimeError("failure"))
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(error=RuntimeError("failure"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox, max_delivery_count=5
        )
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(error=RuntimeError("persistent failure"))
        # Use max_delivery_count=1 to trigger DLQ on first failure
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox, max_delivery_count=1
        )
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(error=ValueError("validation error"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=5,
            include_errors=frozenset({ValueError}),
        )
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(error=TimeoutError("transient timeout"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=2,
            exclude_errors=frozenset({TimeoutError}),
        )
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(error=RuntimeError("failure"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox, max_delivery_count=1
        )
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
        agent_loop = create_test_loop(error=RuntimeError("failure"))
        dlq: DLQPolicy[EvalRequest[str, str], EvalResult] = DLQPolicy(
            mailbox=dlq_mailbox, max_delivery_count=1
        )
        eval_loop: EvalLoop[str, Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=output_to_str,
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
