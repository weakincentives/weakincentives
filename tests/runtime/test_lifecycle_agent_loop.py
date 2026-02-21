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

"""Tests for AgentLoop shutdown behavior and Runnable protocol."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

from tests.runtime.conftest import (
    LifecycleMockAdapter,
    LifecycleOutput,
    LifecycleRequest,
    LifecycleTestLoop,
)
from weakincentives.adapters.core import PromptResponse
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt
from weakincentives.runtime import (
    AgentLoop,
    AgentLoopRequest,
    AgentLoopResult,
    InMemoryMailbox,
    Runnable,
)
from weakincentives.runtime.mailbox import Message, ReceiptHandleExpiredError
from weakincentives.runtime.session.protocols import SessionProtocol

# =============================================================================
# AgentLoop Shutdown Tests
# =============================================================================


def test_agent_loop_shutdown_stops_loop() -> None:
    """AgentLoop.shutdown() stops the run loop."""
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        adapter = LifecycleMockAdapter()
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        thread = threading.Thread(
            target=loop.run, kwargs={"wait_time_seconds": 1, "max_iterations": None}
        )
        thread.start()

        time.sleep(0.1)
        assert loop.running

        result = loop.shutdown(timeout=2.0)
        thread.join(timeout=2.0)

        assert result is True
        assert not loop.running
        assert not thread.is_alive()
    finally:
        requests.close()


def test_agent_loop_shutdown_completes_in_flight() -> None:
    """AgentLoop.shutdown() waits for in-flight message to complete."""
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        # Adapter with delay to simulate in-flight work
        adapter = LifecycleMockAdapter(delay=0.2)
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        # Send a request
        requests.send(AgentLoopRequest(request=LifecycleRequest(message="test")))

        thread = threading.Thread(
            target=loop.run, kwargs={"wait_time_seconds": 0, "max_iterations": 1}
        )
        thread.start()

        # Wait for processing to start
        time.sleep(0.05)

        # Shutdown should wait for completion
        result = loop.shutdown(timeout=2.0)
        thread.join(timeout=2.0)

        assert result is True
        assert adapter.call_count == 1
    finally:
        requests.close()


def test_agent_loop_shutdown_nacks_unprocessed_messages() -> None:
    """AgentLoop.shutdown() nacks unprocessed messages from batch."""
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        # Slow adapter to allow shutdown during batch processing
        adapter = LifecycleMockAdapter(delay=0.1)
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        # Send multiple requests
        for i in range(3):
            requests.send(
                AgentLoopRequest(request=LifecycleRequest(message=f"msg-{i}"))
            )

        thread = threading.Thread(
            target=loop.run, kwargs={"wait_time_seconds": 0, "max_iterations": None}
        )
        thread.start()

        # Wait for first message to start processing
        time.sleep(0.05)

        # Shutdown during processing
        loop.shutdown(timeout=2.0)
        thread.join(timeout=2.0)

        # Some messages should have been processed, some nacked
        assert adapter.call_count >= 1
    finally:
        requests.close()


def test_agent_loop_running_property() -> None:
    """AgentLoop.running property reflects loop state."""
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        adapter = LifecycleMockAdapter()
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        assert not loop.running

        thread = threading.Thread(
            target=loop.run, kwargs={"wait_time_seconds": 0, "max_iterations": 1}
        )
        thread.start()

        time.sleep(0.05)
        # Might still be running or finished depending on timing

        loop.shutdown(timeout=1.0)
        thread.join(timeout=1.0)

        assert not loop.running
    finally:
        requests.close()


def test_agent_loop_context_manager() -> None:
    """AgentLoop supports context manager protocol."""
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        adapter = LifecycleMockAdapter()
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        with loop:
            thread = threading.Thread(
                target=loop.run,
                kwargs={"wait_time_seconds": 1, "max_iterations": None},
            )
            thread.start()
            time.sleep(0.05)

        # Context exit should trigger shutdown
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        requests.close()


def test_agent_loop_shutdown_timeout_returns_false() -> None:
    """AgentLoop.shutdown() returns False when timeout expires."""
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        # Very slow adapter
        adapter = LifecycleMockAdapter(delay=5.0)
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        requests.send(AgentLoopRequest(request=LifecycleRequest(message="slow")))

        thread = threading.Thread(
            target=loop.run, kwargs={"wait_time_seconds": 0, "max_iterations": 1}
        )
        thread.start()

        time.sleep(0.1)

        # Short timeout - should return False
        result = loop.shutdown(timeout=0.1)

        # Clean up - let the thread finish
        requests.close()
        thread.join(timeout=6.0)

        assert result is False
    finally:
        pass


def test_agent_loop_can_restart_after_shutdown() -> None:
    """AgentLoop can run again after shutdown."""
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        adapter = LifecycleMockAdapter()
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        # First run
        requests.send(AgentLoopRequest(request=LifecycleRequest(message="first")))
        loop.run(max_iterations=1, wait_time_seconds=0)
        assert adapter.call_count == 1

        # Second run
        requests.send(AgentLoopRequest(request=LifecycleRequest(message="second")))
        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter.call_count == 2
    finally:
        requests.close()


def test_agent_loop_nacks_remaining_messages_on_shutdown() -> None:
    """AgentLoop nacks remaining messages in batch when shutdown is triggered mid-batch."""

    class _ShutdownTriggeringAdapter(LifecycleMockAdapter):
        """Adapter that triggers loop shutdown after first call."""

        def __init__(self, loop: AgentLoop[LifecycleRequest, LifecycleOutput]) -> None:
            super().__init__()
            self._loop = loop

        def evaluate(
            self,
            prompt: Prompt[LifecycleOutput],
            *,
            session: SessionProtocol,
            deadline: Deadline | None = None,
            budget: Budget | None = None,
            budget_tracker: BudgetTracker | None = None,
            heartbeat: object = None,
            run_context: object = None,
        ) -> PromptResponse[LifecycleOutput]:
            result = super().evaluate(
                prompt,
                session=session,
                deadline=deadline,
                budget=budget,
                budget_tracker=budget_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
            )
            # Trigger shutdown after first message
            if self.call_count == 1:
                self._loop._shutdown_event.set()
            return result

    class _MultiMessageMailbox(
        InMemoryMailbox[
            AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
        ]
    ):
        """Mailbox that returns all messages in a single receive call."""

        def receive(
            self,
            *,
            max_messages: int = 10,
            visibility_timeout: int = 30,
            wait_time_seconds: int = 0,
        ) -> Sequence[
            Message[
                AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
            ]
        ]:
            # Override to return up to 10 messages at once
            return super().receive(
                max_messages=10,
                visibility_timeout=visibility_timeout,
                wait_time_seconds=wait_time_seconds,
            )

    requests = _MultiMessageMailbox(name="requests")

    try:
        # Create a loop with a temporary adapter, then replace
        temp_adapter = LifecycleMockAdapter()
        loop = LifecycleTestLoop(
            adapter=temp_adapter,
            requests=requests,  # type: ignore[arg-type]
        )

        # Now create the shutdown-triggering adapter
        adapter = _ShutdownTriggeringAdapter(loop)
        loop._adapter = adapter

        # Send multiple messages
        for i in range(3):
            requests.send(
                AgentLoopRequest(request=LifecycleRequest(message=f"msg-{i}"))
            )

        # Run - first message will trigger shutdown, remaining should be nacked
        loop.run(max_iterations=1, wait_time_seconds=0)

        # First message should be processed
        assert adapter.call_count == 1
    finally:
        requests.close()


def test_agent_loop_nacks_with_expired_receipt_handle() -> None:
    """AgentLoop handles expired receipt handle during shutdown nack."""

    class _ShutdownAfterFirstAdapter(LifecycleMockAdapter):
        """Adapter that sets shutdown after first call."""

        def __init__(self, loop: AgentLoop[LifecycleRequest, LifecycleOutput]) -> None:
            super().__init__()
            self._loop = loop

        def evaluate(
            self,
            prompt: Prompt[LifecycleOutput],
            *,
            session: SessionProtocol,
            deadline: Deadline | None = None,
            budget: Budget | None = None,
            budget_tracker: BudgetTracker | None = None,
            heartbeat: object = None,
            run_context: object = None,
        ) -> PromptResponse[LifecycleOutput]:
            result = super().evaluate(
                prompt,
                session=session,
                deadline=deadline,
                budget=budget,
                budget_tracker=budget_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
            )
            # Trigger shutdown after first message
            if self.call_count == 1:
                self._loop._shutdown_event.set()
            return result

    class _ExpiredNackMailbox(
        InMemoryMailbox[
            AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
        ]
    ):
        """Mailbox that returns multiple messages and raises ReceiptHandleExpiredError on nack."""

        def receive(
            self,
            *,
            max_messages: int = 10,
            visibility_timeout: int = 30,
            wait_time_seconds: int = 0,
        ) -> list[
            Message[
                AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
            ]
        ]:
            msgs = super().receive(
                max_messages=10,  # Return multiple messages
                visibility_timeout=visibility_timeout,
                wait_time_seconds=wait_time_seconds,
            )
            # Wrap messages to raise on nack
            return [_ExpiredNackMessage(msg) for msg in msgs]  # type: ignore[misc]

    class _ExpiredNackMessage:
        """Message that raises ReceiptHandleExpiredError on nack."""

        def __init__(
            self,
            inner: Message[
                AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
            ],
        ) -> None:
            self._inner = inner

        @property
        def id(self) -> str:
            return self._inner.id

        @property
        def body(self) -> AgentLoopRequest[LifecycleRequest]:
            return self._inner.body

        @property
        def delivery_count(self) -> int:
            return self._inner.delivery_count

        def acknowledge(self) -> None:
            self._inner.acknowledge()

        def nack(self, *, visibility_timeout: int = 0) -> None:
            raise ReceiptHandleExpiredError("Handle expired")

    requests = _ExpiredNackMailbox(name="requests")

    try:
        temp_adapter = LifecycleMockAdapter()
        loop = LifecycleTestLoop(
            adapter=temp_adapter,
            requests=requests,  # type: ignore[arg-type]
        )

        # Create adapter that triggers shutdown after first message
        adapter = _ShutdownAfterFirstAdapter(loop)
        loop._adapter = adapter

        # Send multiple messages
        for i in range(3):
            requests.send(
                AgentLoopRequest(request=LifecycleRequest(message=f"msg-{i}"))
            )

        # Run - should handle ReceiptHandleExpiredError gracefully during nack
        loop.run(max_iterations=1, wait_time_seconds=0)

        # First message should have been processed
        assert adapter.call_count == 1
    finally:
        requests.close()


# =============================================================================
# Runnable Protocol Tests
# =============================================================================


def test_agent_loop_implements_runnable() -> None:
    """AgentLoop conforms to Runnable protocol."""
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        adapter = LifecycleMockAdapter()
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        # Type check - AgentLoop should be usable where Runnable is expected
        runnable: Runnable = loop
        assert hasattr(runnable, "run")
        assert hasattr(runnable, "shutdown")
        assert hasattr(runnable, "running")
        assert hasattr(runnable, "__enter__")
        assert hasattr(runnable, "__exit__")
    finally:
        requests.close()


def test_agent_loop_has_heartbeat_property() -> None:
    """AgentLoop exposes heartbeat property for watchdog monitoring."""
    from weakincentives.runtime.watchdog import Heartbeat

    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="requests")

    try:
        adapter = LifecycleMockAdapter()
        loop = LifecycleTestLoop(adapter=adapter, requests=requests)

        # AgentLoop should have heartbeat property
        assert hasattr(loop, "heartbeat")
        hb = loop.heartbeat
        assert isinstance(hb, Heartbeat)
        assert hb.elapsed() < 1.0  # Recently created
    finally:
        requests.close()
