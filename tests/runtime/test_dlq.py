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

"""Tests for Dead Letter Queue functionality."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

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
    Message,
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
        self._template = PromptTemplate[_Output].create(
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
# Fixtures
# =============================================================================


@pytest.fixture()
def dlq_mailbox() -> Generator[InMemoryMailbox[DeadLetter[str], None]]:
    """DLQ mailbox for DLQPolicy unit tests."""
    mb: InMemoryMailbox[DeadLetter[str], None] = InMemoryMailbox(name="dlq")
    yield mb
    mb.close()


@pytest.fixture()
def results_mailbox() -> Generator[InMemoryMailbox[AgentLoopResult[_Output], None]]:
    mb: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    yield mb
    mb.close()


@pytest.fixture()
def requests_mailbox() -> Generator[
    InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]]
]:
    mb: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    yield mb
    mb.close()


@pytest.fixture()
def agent_dlq_mailbox() -> Generator[
    InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None]
]:
    mb: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None] = InMemoryMailbox(
        name="requests-dlq"
    )
    yield mb
    mb.close()


def _make_message(body: str = "test", delivery_count: int = 1) -> Message[str, Any]:
    return Message(
        id="msg-123",
        body=body,
        receipt_handle="handle",
        delivery_count=delivery_count,
        enqueued_at=datetime.now(UTC),
    )


# =============================================================================
# DeadLetter Tests
# =============================================================================


def test_dead_letter_creation_captures_metadata() -> None:
    """DeadLetter captures all required metadata."""
    body = AgentLoopRequest(request=_Request(message="test"))
    dead_letter: DeadLetter[AgentLoopRequest[_Request]] = DeadLetter(
        message_id="msg-123",
        body=body,
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
        request_id=body.request_id,
        reply_to="results",
        trace_id="trace-abc",
    )

    assert dead_letter.message_id == "msg-123"
    assert dead_letter.body is body
    assert dead_letter.source_mailbox == "requests"
    assert dead_letter.delivery_count == 5
    assert dead_letter.last_error == "Test error"
    assert dead_letter.last_error_type == "builtins.RuntimeError"
    assert dead_letter.request_id == body.request_id
    assert dead_letter.reply_to == "results"
    assert dead_letter.trace_id == "trace-abc"


def test_dead_letter_optional_fields_default_to_none() -> None:
    """DeadLetter has sensible defaults for optional fields."""
    dead_letter: DeadLetter[str] = DeadLetter(
        message_id="msg-123",
        body="test message",
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
    )

    assert dead_letter.request_id is None
    assert dead_letter.reply_to is None
    assert dead_letter.trace_id is None


def test_dead_letter_is_frozen() -> None:
    """DeadLetter is immutable."""
    dead_letter: DeadLetter[str] = DeadLetter(
        message_id="msg-123",
        body="test message",
        source_mailbox="requests",
        delivery_count=5,
        last_error="Test error",
        last_error_type="builtins.RuntimeError",
        dead_lettered_at=datetime.now(UTC),
        first_received_at=datetime.now(UTC),
    )

    with pytest.raises(AttributeError):
        dead_letter.message_id = "changed"  # type: ignore[misc]


# =============================================================================
# DLQPolicy Tests
# =============================================================================


def test_dlq_policy_default_values(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """DLQPolicy has sensible defaults."""
    policy: DLQPolicy[str, None] = DLQPolicy(mailbox=dlq_mailbox)

    assert policy.mailbox is dlq_mailbox
    assert policy.max_delivery_count == 5
    assert policy.include_errors is None
    assert policy.exclude_errors is None


def test_dlq_policy_below_threshold_no_dead_letter(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """DLQPolicy does not dead-letter when below delivery count threshold."""
    policy: DLQPolicy[str, None] = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=3)
    msg = _make_message(delivery_count=2)

    assert not policy.should_dead_letter(msg, RuntimeError("test error"))


def test_dlq_policy_at_threshold_dead_letters(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """DLQPolicy dead-letters when delivery count reaches threshold."""
    policy: DLQPolicy[str, None] = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=3)
    msg = _make_message(delivery_count=3)

    assert policy.should_dead_letter(msg, RuntimeError("test error"))


def test_dlq_policy_above_threshold_dead_letters(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """DLQPolicy dead-letters when delivery count exceeds threshold."""
    policy: DLQPolicy[str, None] = DLQPolicy(mailbox=dlq_mailbox, max_delivery_count=3)
    msg = _make_message(delivery_count=5)

    assert policy.should_dead_letter(msg, RuntimeError("test error"))


def test_dlq_policy_include_errors_immediate_dead_letter(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """DLQPolicy immediately dead-letters included error types."""
    policy: DLQPolicy[str, None] = DLQPolicy(
        mailbox=dlq_mailbox,
        max_delivery_count=5,
        include_errors=frozenset({ValueError, TypeError}),
    )
    msg = _make_message(delivery_count=1)

    assert policy.should_dead_letter(msg, ValueError("bad value"))
    assert policy.should_dead_letter(msg, TypeError("bad type"))


def test_dlq_policy_include_errors_skips_other_types(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """DLQPolicy does not dead-letter non-included errors on first attempt."""
    policy: DLQPolicy[str, None] = DLQPolicy(
        mailbox=dlq_mailbox,
        max_delivery_count=5,
        include_errors=frozenset({ValueError, TypeError}),
    )
    msg = _make_message(delivery_count=1)

    assert not policy.should_dead_letter(msg, RuntimeError("other error"))


def test_dlq_policy_exclude_errors_never_dead_letters(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """DLQPolicy never dead-letters excluded error types."""
    policy: DLQPolicy[str, None] = DLQPolicy(
        mailbox=dlq_mailbox,
        max_delivery_count=2,
        exclude_errors=frozenset({TimeoutError}),
    )
    msg = _make_message(delivery_count=10)

    assert not policy.should_dead_letter(msg, TimeoutError("timeout"))


def test_dlq_policy_exclude_errors_allows_other_types(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """DLQPolicy dead-letters non-excluded errors above threshold."""
    policy: DLQPolicy[str, None] = DLQPolicy(
        mailbox=dlq_mailbox,
        max_delivery_count=2,
        exclude_errors=frozenset({TimeoutError}),
    )
    msg = _make_message(delivery_count=10)

    assert policy.should_dead_letter(msg, RuntimeError("other error"))


def test_dlq_policy_exclude_takes_precedence_over_include(
    dlq_mailbox: InMemoryMailbox[DeadLetter[str], None],
) -> None:
    """Exclude errors take precedence over include errors."""
    policy: DLQPolicy[str, None] = DLQPolicy(
        mailbox=dlq_mailbox,
        max_delivery_count=5,
        include_errors=frozenset({ValueError}),
        exclude_errors=frozenset({ValueError}),
    )
    msg = _make_message(delivery_count=1)

    assert not policy.should_dead_letter(msg, ValueError("test"))


# =============================================================================
# AgentLoop DLQ Integration Tests
# =============================================================================


def test_agentloop_error_reply_acks_message(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
) -> None:
    """AgentLoop without DLQ acknowledges failed messages."""
    adapter = _MockAdapter(error=RuntimeError("failure"))
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    loop.run(max_iterations=1, wait_time_seconds=0)

    assert requests_mailbox.approximate_count() == 0


def test_agentloop_error_reply_sends_error_result(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
) -> None:
    """AgentLoop without DLQ sends error reply."""
    adapter = _MockAdapter(error=RuntimeError("failure"))
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    loop.run(max_iterations=1, wait_time_seconds=0)

    assert results_mailbox.approximate_count() == 1
    msgs = results_mailbox.receive(max_messages=1)
    assert not msgs[0].body.success
    assert "failure" in (msgs[0].body.error or "")
    msgs[0].acknowledge()


def test_agentloop_nacks_below_dlq_threshold(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop nacks failed messages with DLQ before threshold is reached."""
    adapter = _MockAdapter(error=RuntimeError("failure"))
    dlq = DLQPolicy(mailbox=agent_dlq_mailbox, max_delivery_count=5)
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    loop.run(max_iterations=1, wait_time_seconds=0)

    assert requests_mailbox.approximate_count() == 1
    assert results_mailbox.approximate_count() == 0
    assert agent_dlq_mailbox.approximate_count() == 0


def test_agentloop_dead_letters_at_threshold(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop sends to DLQ when delivery count reaches threshold."""
    adapter = _MockAdapter(error=RuntimeError("persistent failure"))
    dlq = DLQPolicy(mailbox=agent_dlq_mailbox, max_delivery_count=1)
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    loop.run(max_iterations=1, wait_time_seconds=0)

    assert requests_mailbox.approximate_count() == 0
    assert agent_dlq_mailbox.approximate_count() == 1


def test_agentloop_dead_letter_content(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """Dead letter captures correct metadata from the failed message."""
    adapter = _MockAdapter(error=RuntimeError("persistent failure"))
    dlq = DLQPolicy(mailbox=agent_dlq_mailbox, max_delivery_count=1)
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    loop.run(max_iterations=1, wait_time_seconds=0)

    dlq_msgs = agent_dlq_mailbox.receive(max_messages=1)
    assert len(dlq_msgs) == 1
    dead_letter = dlq_msgs[0].body
    assert dead_letter.message_id is not None
    assert dead_letter.body.request_id == request.request_id
    assert dead_letter.source_mailbox == "requests"
    assert dead_letter.delivery_count == 1
    assert "persistent failure" in dead_letter.last_error
    assert dead_letter.last_error_type == "builtins.RuntimeError"
    dlq_msgs[0].acknowledge()


def test_agentloop_dead_letter_sends_error_reply(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop sends Dead-lettered error reply when DLQ threshold hit."""
    adapter = _MockAdapter(error=RuntimeError("persistent failure"))
    dlq = DLQPolicy(mailbox=agent_dlq_mailbox, max_delivery_count=1)
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    loop.run(max_iterations=1, wait_time_seconds=0)

    result_msgs = results_mailbox.receive(max_messages=1)
    assert len(result_msgs) == 1
    assert not result_msgs[0].body.success
    assert "Dead-lettered" in (result_msgs[0].body.error or "")
    result_msgs[0].acknowledge()


def test_agentloop_immediate_dlq_for_included_error(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop immediately dead-letters included error types."""
    adapter = _MockAdapter(error=ValueError("validation error"))
    dlq = DLQPolicy(
        mailbox=agent_dlq_mailbox,
        max_delivery_count=5,
        include_errors=frozenset({ValueError}),
    )
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    loop.run(max_iterations=1, wait_time_seconds=0)

    assert requests_mailbox.approximate_count() == 0
    assert agent_dlq_mailbox.approximate_count() == 1
    dlq_msgs = agent_dlq_mailbox.receive(max_messages=1)
    assert dlq_msgs[0].body.delivery_count == 1
    dlq_msgs[0].acknowledge()


def test_agentloop_never_dlq_for_excluded_error(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop never dead-letters excluded error types."""
    adapter = _MockAdapter(error=TimeoutError("transient timeout"))
    dlq = DLQPolicy(
        mailbox=agent_dlq_mailbox,
        max_delivery_count=2,
        exclude_errors=frozenset({TimeoutError}),
    )
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    for _ in range(5):
        loop.run(max_iterations=1, wait_time_seconds=0)

    assert requests_mailbox.approximate_count() == 1
    assert agent_dlq_mailbox.approximate_count() == 0


def test_agentloop_dlq_preserves_request_id(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    results_mailbox: InMemoryMailbox[AgentLoopResult[_Output], None],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop DLQ preserves request ID in dead letter."""
    adapter = _MockAdapter(error=RuntimeError("failure"))
    dlq = DLQPolicy(mailbox=agent_dlq_mailbox, max_delivery_count=1)
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results_mailbox)
    loop.run(max_iterations=1, wait_time_seconds=0)

    dlq_msgs = agent_dlq_mailbox.receive(max_messages=1)
    assert len(dlq_msgs) == 1
    assert dlq_msgs[0].body.request_id == request.request_id
    dlq_msgs[0].acknowledge()


def test_agentloop_dlq_preserves_reply_to(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop DLQ preserves reply_to mailbox name in dead letter."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="my-results-queue"
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("failure"))
        dlq = DLQPolicy(mailbox=agent_dlq_mailbox, max_delivery_count=1)
        loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests_mailbox.send(request, reply_to=results)
        loop.run(max_iterations=1, wait_time_seconds=0)

        dlq_msgs = agent_dlq_mailbox.receive(max_messages=1)
        assert len(dlq_msgs) == 1
        assert dlq_msgs[0].body.reply_to == "my-results-queue"
        dlq_msgs[0].acknowledge()
    finally:
        results.close()


def test_agentloop_dlq_without_reply_to(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop DLQ handles messages without reply_to."""
    adapter = _MockAdapter(error=RuntimeError("failure"))
    dlq = DLQPolicy(mailbox=agent_dlq_mailbox, max_delivery_count=1)
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request)
    loop.run(max_iterations=1, wait_time_seconds=0)

    assert requests_mailbox.approximate_count() == 0
    assert agent_dlq_mailbox.approximate_count() == 1
    dlq_msgs = agent_dlq_mailbox.receive(max_messages=1)
    assert dlq_msgs[0].body.reply_to is None
    dlq_msgs[0].acknowledge()


def test_agentloop_dlq_handles_reply_error(
    requests_mailbox: InMemoryMailbox[
        AgentLoopRequest[_Request], AgentLoopResult[_Output]
    ],
    agent_dlq_mailbox: InMemoryMailbox[DeadLetter[AgentLoopRequest[_Request]], None],
) -> None:
    """AgentLoop DLQ handles errors when sending reply."""
    from weakincentives.runtime.mailbox import MailboxConnectionError

    results: FakeMailbox[AgentLoopResult[_Output], None] = FakeMailbox(name="results")
    adapter = _MockAdapter(error=RuntimeError("failure"))
    dlq = DLQPolicy(mailbox=agent_dlq_mailbox, max_delivery_count=1)
    loop = _TestLoop(adapter=adapter, requests=requests_mailbox, dlq=dlq)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests_mailbox.send(request, reply_to=results)
    results.set_connection_error(MailboxConnectionError("connection lost"))
    loop.run(max_iterations=1, wait_time_seconds=0)

    assert requests_mailbox.approximate_count() == 0
    assert agent_dlq_mailbox.approximate_count() == 1
