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

"""Tests for AgentLoop DLQ integration."""

from __future__ import annotations

from dataclasses import dataclass

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.dlq import DeadLetter, DLQPolicy
from weakincentives.runtime.mailbox import (
    FakeMailbox,
    InMemoryMailbox,
    Mailbox,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.protocols import SessionProtocol
from weakincentives.runtime.watchdog import Heartbeat


@dataclass(slots=True, frozen=True)
class _Request:
    """Sample request type for testing."""

    message: str


@dataclass(slots=True, frozen=True)
class _Output:
    """Sample output type for testing."""

    result: str


@dataclass(slots=True, frozen=True)
class _Params:
    """Sample params type for testing."""

    content: str


class _MockAdapter(ProviderAdapter[_Output]):
    """Mock adapter that can be configured to fail."""

    def __init__(
        self,
        *,
        response: PromptResponse[_Output] | None = None,
        error: Exception | None = None,
        fail_count: int = 0,
    ) -> None:
        self._response = response or PromptResponse(
            prompt_name="test",
            text="test output",
            output=_Output(result="success"),
        )
        self._error = error
        self._fail_count = fail_count
        self._call_count = 0

    def evaluate(
        self,
        prompt: Prompt[_Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[_Output]:
        del prompt, session, deadline, budget, budget_tracker, heartbeat, run_context
        self._call_count += 1
        if self._error is not None:
            raise self._error
        if self._call_count <= self._fail_count:
            raise RuntimeError(f"Simulated failure {self._call_count}")
        return self._response


class _TestLoop(AgentLoop[_Request, _Output]):
    """Test implementation of AgentLoop."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: Mailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]],
        config: AgentLoopConfig | None = None,
        dlq: DLQPolicy[AgentLoopRequest[_Request], AgentLoopResult[_Output]]
        | None = None,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, config=config, dlq=dlq)
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
        self, request: _Request, *, experiment: object = None
    ) -> tuple[Prompt[_Output], Session]:
        del experiment  # Unused in tests
        prompt = Prompt(self._template).bind(_Params(content=request.message))
        session = Session(tags={"loop": "test"})
        return prompt, session


# =============================================================================
# AgentLoop DLQ Integration Tests
# =============================================================================


def test_agentloop_error_reply_without_dlq() -> None:
    """AgentLoop sends error reply without DLQ configured (original behavior)."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should be acknowledged (removed from queue)
        assert requests.approximate_count() == 0

        # Error reply should be sent
        assert results.approximate_count() == 1
        msgs = results.receive(max_messages=1)
        assert not msgs[0].body.success
        assert "failure" in (msgs[0].body.error or "")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_agentloop_nacks_with_dlq_before_threshold() -> None:
    """AgentLoop nacks failed messages with DLQ before threshold is reached."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=5)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run once - should nack for retry (below threshold)
        loop.run(max_iterations=1, wait_time_seconds=0)

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


def test_agentloop_sends_to_dlq_after_threshold() -> None:
    """AgentLoop sends to DLQ when delivery count equals threshold."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("persistent failure"))
        # Use max_delivery_count=1 to trigger DLQ on first failure
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run once - should dead-letter immediately (delivery_count=1 >= max=1)
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should be dead-lettered
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1

        # Check dead letter content
        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        dead_letter = dlq_msgs[0].body
        assert dead_letter.message_id is not None
        assert dead_letter.body.request_id == request.request_id
        assert dead_letter.source_mailbox == "requests"
        assert dead_letter.delivery_count == 1
        assert "persistent failure" in dead_letter.last_error
        assert dead_letter.last_error_type == "builtins.RuntimeError"
        dlq_msgs[0].acknowledge()

        # Error reply should be sent
        result_msgs = results.receive(max_messages=1)
        assert len(result_msgs) == 1
        assert not result_msgs[0].body.success
        assert "Dead-lettered" in (result_msgs[0].body.error or "")
        result_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_agentloop_immediate_dlq_for_included_error() -> None:
    """AgentLoop immediately dead-letters included error types."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=ValueError("validation error"))
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=5,
            include_errors=frozenset({ValueError}),
        )
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run once - should immediately dead-letter
        loop.run(max_iterations=1, wait_time_seconds=0)

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


def test_agentloop_never_dlq_for_excluded_error() -> None:
    """AgentLoop never dead-letters excluded error types."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=TimeoutError("transient timeout"))
        dlq = DLQPolicy(
            mailbox=dlq_mailbox,
            max_delivery_count=2,
            exclude_errors=frozenset({TimeoutError}),
        )
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run many times - should never dead-letter
        for _ in range(5):
            loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should still be in queue (nacked, not dead-lettered)
        assert requests.approximate_count() == 1
        assert dlq_mailbox.approximate_count() == 0
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_agentloop_dlq_preserves_request_id() -> None:
    """AgentLoop DLQ preserves request ID in dead letter."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0].body.request_id == request.request_id
        dlq_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_agentloop_dlq_preserves_reply_to() -> None:
    """AgentLoop DLQ preserves reply_to mailbox name in dead letter."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="my-results-queue"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        dlq_msgs = dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0].body.reply_to == "my-results-queue"
        dlq_msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
        dlq_mailbox.close()


def test_agentloop_dlq_without_reply_to() -> None:
    """AgentLoop DLQ handles messages without reply_to."""
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        # Send without reply_to
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

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


def test_agentloop_dlq_handles_reply_error() -> None:
    """AgentLoop DLQ handles errors when sending reply."""
    from weakincentives.runtime.mailbox import MailboxConnectionError

    results: FakeMailbox[AgentLoopResult[_Output], None] = FakeMailbox(name="results")
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = (
        InMemoryMailbox(name="requests-dlq")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Make reply send fail
        results.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Message should still be dead-lettered despite reply failure
        assert requests.approximate_count() == 0
        assert dlq_mailbox.approximate_count() == 1
    finally:
        requests.close()
        dlq_mailbox.close()
