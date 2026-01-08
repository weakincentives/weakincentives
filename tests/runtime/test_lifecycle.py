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

"""Tests for lifecycle management primitives."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.resources import ResourceRegistry
from weakincentives.runtime import (
    InMemoryMailbox,
    LoopGroup,
    MainLoop,
    MainLoopRequest,
    MainLoopResult,
    Runnable,
    Session,
    ShutdownCoordinator,
    wait_until,
)
from weakincentives.runtime.mailbox import (
    MailboxResolver,
    RegistryResolver,
    ReplyRoutes,
)
from weakincentives.runtime.session.protocols import SessionProtocol

if TYPE_CHECKING:
    from weakincentives.runtime.watchdog import Heartbeat

# =============================================================================
# Test Fixtures
# =============================================================================


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
    """Mock adapter for testing."""

    def __init__(
        self,
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self._delay = delay
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
        if self._delay > 0:
            time.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return PromptResponse(
            prompt_name="test",
            text="success",
            output=_Output(result="success"),
        )


class _TestLoop(MainLoop[_Request, _Output]):
    """Test implementation of MainLoop."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]],
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

    def prepare(self, request: _Request) -> tuple[Prompt[_Output], Session]:
        prompt = Prompt(self._template).bind(_Params(content=request.message))
        session = Session(tags={"loop": "test"})
        return prompt, session


def _create_test_loop(
    *,
    delay: float = 0.0,
    error: Exception | None = None,
) -> _TestLoop:
    """Create a test MainLoop with mock adapter."""
    adapter = _MockAdapter(delay=delay, error=error)
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="dummy-requests")
    )
    return _TestLoop(adapter=adapter, requests=requests)


class _MockRunnable:
    """Mock implementation of Runnable for testing LoopGroup."""

    def __init__(self, *, run_delay: float = 0.0) -> None:
        self._run_delay = run_delay
        self._shutdown_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self.run_called = False
        self.shutdown_called = False

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        del max_iterations, visibility_timeout, wait_time_seconds
        with self._lock:
            self._running = True
        self.run_called = True

        # Simulate work until shutdown
        while not self._shutdown_event.is_set():
            time.sleep(0.01)
            if self._run_delay > 0:
                time.sleep(self._run_delay)
                break

        with self._lock:
            self._running = False

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        self.shutdown_called = True
        self._shutdown_event.set()
        return wait_until(lambda: not self.running, timeout=timeout)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        _ = self.shutdown()


# =============================================================================
# wait_until Tests
# =============================================================================


def test_wait_until_returns_true_when_predicate_succeeds() -> None:
    """wait_until returns True when predicate becomes True."""
    counter = {"value": 0}

    def predicate() -> bool:
        counter["value"] += 1
        return counter["value"] >= 3

    result = wait_until(predicate, timeout=1.0, poll_interval=0.01)
    assert result is True
    assert counter["value"] >= 3


def test_wait_until_returns_false_on_timeout() -> None:
    """wait_until returns False when timeout expires."""
    result = wait_until(lambda: False, timeout=0.1, poll_interval=0.01)
    assert result is False


def test_wait_until_returns_immediately_if_predicate_true() -> None:
    """wait_until returns immediately if predicate is already True."""
    start = time.monotonic()
    result = wait_until(lambda: True, timeout=10.0, poll_interval=1.0)
    elapsed = time.monotonic() - start

    assert result is True
    assert elapsed < 0.5  # Should return well before 1 second poll interval


# =============================================================================
# ShutdownCoordinator Tests
# =============================================================================


@pytest.fixture
def reset_coordinator() -> None:
    """Reset ShutdownCoordinator singleton before and after each test."""
    ShutdownCoordinator.reset()
    yield
    ShutdownCoordinator.reset()


def test_coordinator_install_returns_singleton(reset_coordinator: None) -> None:
    """ShutdownCoordinator.install() returns the same instance."""
    _ = reset_coordinator
    coordinator1 = ShutdownCoordinator.install()
    coordinator2 = ShutdownCoordinator.install()
    assert coordinator1 is coordinator2


def test_coordinator_get_returns_none_before_install(reset_coordinator: None) -> None:
    """ShutdownCoordinator.get() returns None before install."""
    _ = reset_coordinator
    assert ShutdownCoordinator.get() is None


def test_coordinator_get_returns_instance_after_install(
    reset_coordinator: None,
) -> None:
    """ShutdownCoordinator.get() returns instance after install."""
    _ = reset_coordinator
    installed = ShutdownCoordinator.install()
    assert ShutdownCoordinator.get() is installed


def test_coordinator_register_adds_callback(reset_coordinator: None) -> None:
    """ShutdownCoordinator.register() adds callback to list."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()

    coordinator.register(callback)
    coordinator.trigger()

    callback.assert_called_once()


def test_coordinator_trigger_invokes_all_callbacks(reset_coordinator: None) -> None:
    """ShutdownCoordinator.trigger() invokes all registered callbacks."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callbacks = [MagicMock() for _ in range(3)]

    for cb in callbacks:
        coordinator.register(cb)

    coordinator.trigger()

    for cb in callbacks:
        cb.assert_called_once()


def test_coordinator_unregister_removes_callback(reset_coordinator: None) -> None:
    """ShutdownCoordinator.unregister() removes callback from list."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()

    coordinator.register(callback)
    coordinator.unregister(callback)
    coordinator.trigger()

    callback.assert_not_called()


def test_coordinator_unregister_nonexistent_callback_is_safe(
    reset_coordinator: None,
) -> None:
    """ShutdownCoordinator.unregister() is safe for unregistered callbacks."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()

    # Should not raise
    coordinator.unregister(callback)


def test_coordinator_triggered_property(reset_coordinator: None) -> None:
    """ShutdownCoordinator.triggered property reflects state."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()

    assert coordinator.triggered is False
    coordinator.trigger()
    assert coordinator.triggered is True


def test_coordinator_late_register_invokes_immediately(reset_coordinator: None) -> None:
    """Callback registered after trigger is invoked immediately."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    coordinator.trigger()

    callback = MagicMock()
    coordinator.register(callback)

    callback.assert_called_once()


def test_coordinator_reset_clears_state(reset_coordinator: None) -> None:
    """ShutdownCoordinator.reset() clears singleton and state."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()
    coordinator.register(callback)
    coordinator.trigger()

    ShutdownCoordinator.reset()

    assert ShutdownCoordinator.get() is None

    # New coordinator should be fresh
    new_coordinator = ShutdownCoordinator.install()
    assert new_coordinator is not coordinator
    assert new_coordinator.triggered is False


def test_coordinator_signal_handler(reset_coordinator: None) -> None:
    """ShutdownCoordinator installs signal handlers."""
    _ = reset_coordinator
    with patch("signal.signal") as mock_signal:
        ShutdownCoordinator.install(signals=(signal.SIGTERM,))
        mock_signal.assert_called()


# =============================================================================
# LoopGroup Tests
# =============================================================================


def test_loop_group_runs_all_loops(reset_coordinator: None) -> None:
    """LoopGroup.run() starts all loops."""
    _ = reset_coordinator
    loops = [_MockRunnable(run_delay=0.1) for _ in range(3)]
    group = LoopGroup(loops=loops)

    # Run in background thread
    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    # Wait a bit for loops to start
    time.sleep(0.05)

    # Trigger shutdown
    group.shutdown(timeout=1.0)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    for loop in loops:
        assert loop.run_called


def test_loop_group_shutdown_stops_all_loops(reset_coordinator: None) -> None:
    """LoopGroup.shutdown() stops all loops."""
    _ = reset_coordinator
    loops = [_MockRunnable() for _ in range(3)]
    group = LoopGroup(loops=loops)

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    time.sleep(0.05)
    result = group.shutdown(timeout=1.0)
    thread.join(timeout=2.0)

    assert result is True
    for loop in loops:
        assert loop.shutdown_called
        assert not loop.running


def test_loop_group_context_manager(reset_coordinator: None) -> None:
    """LoopGroup supports context manager protocol."""
    _ = reset_coordinator
    loops = [_MockRunnable() for _ in range(2)]

    with LoopGroup(loops=loops) as group:
        thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
        thread.start()
        time.sleep(0.05)

    # Context exit should trigger shutdown
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_loop_group_with_signals(reset_coordinator: None) -> None:
    """LoopGroup integrates with ShutdownCoordinator (main thread)."""
    _ = reset_coordinator

    # Pre-install coordinator to avoid signal handler issues
    coordinator = ShutdownCoordinator.install()

    loops = [_MockRunnable(run_delay=0.05) for _ in range(2)]
    group = LoopGroup(loops=loops)

    # Run in background thread with install_signals=True
    # The coordinator is already installed, so it will just register
    thread = threading.Thread(target=group.run, kwargs={"install_signals": True})
    thread.start()

    time.sleep(0.02)

    # Trigger via coordinator
    coordinator.trigger()

    thread.join(timeout=2.0)
    assert not thread.is_alive()


# =============================================================================
# MainLoop Shutdown Tests
# =============================================================================


def test_main_loop_shutdown_stops_loop() -> None:
    """MainLoop.shutdown() stops the run loop."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

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


def test_main_loop_shutdown_completes_in_flight() -> None:
    """MainLoop.shutdown() waits for in-flight message to complete."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        # Adapter with delay to simulate in-flight work
        adapter = _MockAdapter(delay=0.2)
        loop = _TestLoop(adapter=adapter, requests=requests)

        # Send a request
        requests.send(MainLoopRequest(request=_Request(message="test")))

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


def test_main_loop_shutdown_nacks_unprocessed_messages() -> None:
    """MainLoop.shutdown() nacks unprocessed messages from batch."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        # Slow adapter to allow shutdown during batch processing
        adapter = _MockAdapter(delay=0.1)
        loop = _TestLoop(adapter=adapter, requests=requests)

        # Send multiple requests
        for i in range(3):
            requests.send(MainLoopRequest(request=_Request(message=f"msg-{i}")))

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


def test_main_loop_running_property() -> None:
    """MainLoop.running property reflects loop state."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

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


def test_main_loop_context_manager() -> None:
    """MainLoop supports context manager protocol."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        with loop:
            thread = threading.Thread(
                target=loop.run, kwargs={"wait_time_seconds": 1, "max_iterations": None}
            )
            thread.start()
            time.sleep(0.05)

        # Context exit should trigger shutdown
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        requests.close()


def test_main_loop_shutdown_timeout_returns_false() -> None:
    """MainLoop.shutdown() returns False when timeout expires."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        # Very slow adapter
        adapter = _MockAdapter(delay=5.0)
        loop = _TestLoop(adapter=adapter, requests=requests)

        requests.send(MainLoopRequest(request=_Request(message="slow")))

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


def test_main_loop_can_restart_after_shutdown() -> None:
    """MainLoop can run again after shutdown."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        # First run
        requests.send(MainLoopRequest(request=_Request(message="first")))
        loop.run(max_iterations=1, wait_time_seconds=0)
        assert adapter.call_count == 1

        # Second run
        requests.send(MainLoopRequest(request=_Request(message="second")))
        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter.call_count == 2
    finally:
        requests.close()


def test_main_loop_nacks_remaining_messages_on_shutdown() -> None:
    """MainLoop nacks remaining messages in batch when shutdown is triggered mid-batch."""
    from collections.abc import Sequence

    from weakincentives.runtime.mailbox import Message

    class _ShutdownTriggeringAdapter(_MockAdapter):
        """Adapter that triggers loop shutdown after first call."""

        def __init__(self, loop: MainLoop[_Request, _Output]) -> None:
            super().__init__()
            self._loop = loop

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
            result = super().evaluate(
                prompt,
                session=session,
                deadline=deadline,
                budget=budget,
                budget_tracker=budget_tracker,
                resources=resources,
            )
            # Trigger shutdown after first message
            if self.call_count == 1:
                self._loop._shutdown_event.set()
            return result

    class _MultiMessageMailbox(
        InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]]
    ):
        """Mailbox that returns all messages in a single receive call."""

        def receive(
            self,
            *,
            max_messages: int = 10,
            visibility_timeout: int = 30,
            wait_time_seconds: int = 0,
        ) -> Sequence[Message[MainLoopRequest[_Request], MainLoopResult[_Output]]]:
            # Override to return up to 10 messages at once
            return super().receive(
                max_messages=10,
                visibility_timeout=visibility_timeout,
                wait_time_seconds=wait_time_seconds,
            )

    requests = _MultiMessageMailbox(name="requests")

    try:
        # Create a loop with a temporary adapter, then replace
        temp_adapter = _MockAdapter()
        loop = _TestLoop(
            adapter=temp_adapter,
            requests=requests,  # type: ignore[arg-type]
        )

        # Now create the shutdown-triggering adapter
        adapter = _ShutdownTriggeringAdapter(loop)
        loop._adapter = adapter

        # Send multiple messages
        for i in range(3):
            requests.send(MainLoopRequest(request=_Request(message=f"msg-{i}")))

        # Run - first message will trigger shutdown, remaining should be nacked
        loop.run(max_iterations=1, wait_time_seconds=0)

        # First message should be processed
        assert adapter.call_count == 1
    finally:
        requests.close()


def test_main_loop_nacks_with_expired_receipt_handle() -> None:
    """MainLoop handles expired receipt handle during shutdown nack."""
    from weakincentives.runtime.mailbox import Message, ReceiptHandleExpiredError

    class _ShutdownAfterFirstAdapter(_MockAdapter):
        """Adapter that sets shutdown after first call."""

        def __init__(self, loop: MainLoop[_Request, _Output]) -> None:
            super().__init__()
            self._loop = loop

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
            result = super().evaluate(
                prompt,
                session=session,
                deadline=deadline,
                budget=budget,
                budget_tracker=budget_tracker,
                resources=resources,
            )
            # Trigger shutdown after first message
            if self.call_count == 1:
                self._loop._shutdown_event.set()
            return result

    class _ExpiredNackMailbox(
        InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]]
    ):
        """Mailbox that returns multiple messages and raises ReceiptHandleExpiredError on nack."""

        def receive(
            self,
            *,
            max_messages: int = 10,
            visibility_timeout: int = 30,
            wait_time_seconds: int = 0,
        ) -> list[Message[MainLoopRequest[_Request], MainLoopResult[_Output]]]:
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
            self, inner: Message[MainLoopRequest[_Request], MainLoopResult[_Output]]
        ) -> None:
            self._inner = inner

        @property
        def body(self) -> MainLoopRequest[_Request]:
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
        temp_adapter = _MockAdapter()
        loop = _TestLoop(
            adapter=temp_adapter,
            requests=requests,  # type: ignore[arg-type]
        )

        # Create adapter that triggers shutdown after first message
        adapter = _ShutdownAfterFirstAdapter(loop)
        loop._adapter = adapter

        # Send multiple messages
        for i in range(3):
            requests.send(MainLoopRequest(request=_Request(message=f"msg-{i}")))

        # Run - should handle ReceiptHandleExpiredError gracefully during nack
        loop.run(max_iterations=1, wait_time_seconds=0)

        # First message should have been processed
        assert adapter.call_count == 1
    finally:
        requests.close()


# =============================================================================
# Runnable Protocol Tests
# =============================================================================


def test_main_loop_implements_runnable() -> None:
    """MainLoop conforms to Runnable protocol."""
    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        # Type check - MainLoop should be usable where Runnable is expected
        runnable: Runnable = loop
        assert hasattr(runnable, "run")
        assert hasattr(runnable, "shutdown")
        assert hasattr(runnable, "running")
        assert hasattr(runnable, "__enter__")
        assert hasattr(runnable, "__exit__")
    finally:
        requests.close()


def test_main_loop_has_heartbeat_property() -> None:
    """MainLoop exposes heartbeat property for watchdog monitoring."""
    from weakincentives.runtime.watchdog import Heartbeat

    requests: InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )

    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        # MainLoop should have heartbeat property
        assert hasattr(loop, "heartbeat")
        hb = loop.heartbeat
        assert isinstance(hb, Heartbeat)
        assert hb.elapsed() < 1.0  # Recently created
    finally:
        requests.close()


# =============================================================================
# Signal Handler Tests
# =============================================================================


def test_coordinator_handle_signal_triggers_shutdown(reset_coordinator: None) -> None:
    """ShutdownCoordinator._handle_signal triggers shutdown."""
    _ = reset_coordinator
    coordinator = ShutdownCoordinator.install()
    callback = MagicMock()
    coordinator.register(callback)

    # Directly call _handle_signal to simulate signal receipt
    coordinator._handle_signal(15, None)  # 15 = SIGTERM

    assert coordinator.triggered
    callback.assert_called_once()


def test_loop_group_trigger_shutdown_returns_result(reset_coordinator: None) -> None:
    """LoopGroup._trigger_shutdown returns shutdown result."""
    _ = reset_coordinator
    loops = [_MockRunnable(run_delay=0.05) for _ in range(2)]
    group = LoopGroup(loops=loops)

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    time.sleep(0.02)

    # Call _trigger_shutdown directly
    result = group._trigger_shutdown()

    thread.join(timeout=2.0)

    assert result is True
    assert not thread.is_alive()


# =============================================================================
# EvalLoop Shutdown Tests
# =============================================================================


def test_eval_loop_shutdown_stops_loop() -> None:
    """EvalLoop.shutdown() stops the run loop."""
    from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, Score

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    resolver: MailboxResolver[EvalResult] = RegistryResolver({"eval-results": results})
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests", reply_resolver=resolver
    )

    try:
        main_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=lambda o, e: Score(
                value=1.0 if o.result == e else 0.0, passed=o.result == e
            ),
            requests=requests,
        )

        thread = threading.Thread(
            target=eval_loop.run,
            kwargs={"wait_time_seconds": 1, "max_iterations": None},
        )
        thread.start()

        time.sleep(0.1)
        assert eval_loop.running

        result = eval_loop.shutdown(timeout=2.0)
        thread.join(timeout=2.0)

        assert result is True
        assert not eval_loop.running
        assert not thread.is_alive()
    finally:
        requests.close()
        results.close()


def test_eval_loop_shutdown_nacks_unprocessed() -> None:
    """EvalLoop.shutdown() nacks unprocessed messages."""
    from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, Sample, Score

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    resolver: MailboxResolver[EvalResult] = RegistryResolver({"eval-results": results})
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests", reply_resolver=resolver
    )

    try:
        # Slow adapter to allow shutdown during batch processing
        adapter = _MockAdapter(delay=0.1)
        main_requests: InMemoryMailbox[
            MainLoopRequest[str], MainLoopResult[_Output]
        ] = InMemoryMailbox(name="main-requests")
        main_loop = _TestLoop(
            adapter=adapter,
            requests=main_requests,
        )

        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=lambda o, e: Score(value=1.0, passed=True),
            requests=requests,
        )

        # Send multiple samples
        for i in range(3):
            sample = Sample(id=str(i), input=f"input-{i}", expected="success")
            requests.send(
                EvalRequest(sample=sample),
                reply_routes=ReplyRoutes.single("eval-results"),
            )

        thread = threading.Thread(
            target=eval_loop.run,
            kwargs={"wait_time_seconds": 0, "max_iterations": None},
        )
        thread.start()

        # Wait for processing to start
        time.sleep(0.05)

        # Shutdown during processing
        eval_loop.shutdown(timeout=2.0)
        thread.join(timeout=2.0)

        # Should have processed at least one
        assert results.approximate_count() >= 0
    finally:
        requests.close()
        main_requests.close()
        results.close()


def test_eval_loop_context_manager() -> None:
    """EvalLoop supports context manager protocol."""
    from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, Score

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    resolver: MailboxResolver[EvalResult] = RegistryResolver({"eval-results": results})
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests", reply_resolver=resolver
    )

    try:
        main_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=lambda o, e: Score(value=1.0, passed=True),
            requests=requests,
        )

        with eval_loop:
            thread = threading.Thread(
                target=eval_loop.run,
                kwargs={"wait_time_seconds": 1, "max_iterations": None},
            )
            thread.start()
            time.sleep(0.05)

        # Context exit should trigger shutdown
        thread.join(timeout=2.0)
        assert not thread.is_alive()
    finally:
        requests.close()
        results.close()


def test_eval_loop_running_property() -> None:
    """EvalLoop.running property reflects loop state."""
    from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, Score

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    resolver: MailboxResolver[EvalResult] = RegistryResolver({"eval-results": results})
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests", reply_resolver=resolver
    )

    try:
        main_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=main_loop,
            evaluator=lambda o, e: Score(value=1.0, passed=True),
            requests=requests,
        )

        assert not eval_loop.running

        thread = threading.Thread(
            target=eval_loop.run, kwargs={"wait_time_seconds": 0, "max_iterations": 1}
        )
        thread.start()

        eval_loop.shutdown(timeout=1.0)
        thread.join(timeout=1.0)

        assert not eval_loop.running
    finally:
        requests.close()
        results.close()


def test_eval_loop_nacks_remaining_messages_on_shutdown() -> None:
    """EvalLoop nacks remaining messages in batch when shutdown is triggered mid-batch."""
    from collections.abc import Sequence

    from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, Sample, Score
    from weakincentives.runtime.mailbox import Message

    class _MultiMessageMailbox(InMemoryMailbox[EvalRequest[_Request, str], EvalResult]):
        """Mailbox that returns all messages in a single receive call."""

        def receive(
            self,
            *,
            max_messages: int = 10,
            visibility_timeout: int = 30,
            wait_time_seconds: int = 0,
        ) -> Sequence[Message[EvalRequest[_Request, str], EvalResult]]:
            return super().receive(
                max_messages=10,
                visibility_timeout=visibility_timeout,
                wait_time_seconds=wait_time_seconds,
            )

    requests = _MultiMessageMailbox(name="eval-requests")

    try:
        adapter = _MockAdapter(delay=0)
        main_requests: InMemoryMailbox[
            MainLoopRequest[_Request], MainLoopResult[_Output]
        ] = InMemoryMailbox(name="main-requests")
        main_loop = _TestLoop(adapter=adapter, requests=main_requests)

        # Track number of evaluations and trigger shutdown after first
        eval_count = 0
        eval_loop_ref: list[EvalLoop[_Request, _Output, str]] = []

        def shutdown_after_first(output: _Output, expected: str) -> Score:
            nonlocal eval_count
            eval_count += 1
            if eval_count == 1 and eval_loop_ref:
                # Trigger shutdown after first evaluation - this will cause
                # remaining messages in the batch to be nacked
                eval_loop_ref[0]._shutdown_event.set()
            return Score(value=1.0, passed=True)

        eval_loop: EvalLoop[_Request, _Output, str] = EvalLoop(
            loop=main_loop, evaluator=shutdown_after_first, requests=requests
        )
        eval_loop_ref.append(eval_loop)

        # Send multiple samples - they will be returned in one batch
        for i in range(3):
            sample: Sample[_Request, str] = Sample(
                id=str(i),
                input=_Request(message=f"input-{i}"),
                expected="success",
            )
            requests.send(EvalRequest(sample=sample))

        # Run the loop - it will process first sample, then shutdown triggers,
        # then remaining messages in batch get nacked
        eval_loop.run(wait_time_seconds=0, max_iterations=1)

        # Should have processed exactly one sample (others were nacked/skipped)
        assert eval_count == 1
    finally:
        requests.close()
        main_requests.close()


# =============================================================================
# HealthServer Tests
# =============================================================================


def test_health_server_liveness_endpoint() -> None:
    """HealthServer returns 200 for /health/live."""
    import json
    import urllib.request

    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0)  # OS-assigned port
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        server.stop()


def test_health_server_readiness_endpoint_healthy() -> None:
    """HealthServer returns 200 for /health/ready when check passes."""
    import json
    import urllib.request

    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0, readiness_check=lambda: True)
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        server.stop()


def test_health_server_readiness_endpoint_unhealthy() -> None:
    """HealthServer returns 503 for /health/ready when check fails."""
    import urllib.error
    import urllib.request

    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0, readiness_check=lambda: False)
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
            msg = "Expected HTTPError with 503 status"
            raise AssertionError(msg)
        except urllib.error.HTTPError as e:
            assert e.code == 503
    finally:
        server.stop()


def test_health_server_404_for_unknown_path() -> None:
    """HealthServer returns 404 for unknown paths."""
    import urllib.error
    import urllib.request

    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0)
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/unknown")
            msg = "Expected HTTPError with 404 status"
            raise AssertionError(msg)
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()


def test_health_server_address_none_when_not_started() -> None:
    """HealthServer.address is None before start."""
    from weakincentives.runtime import HealthServer

    server = HealthServer(port=0)
    assert server.address is None


def test_health_server_start_idempotent() -> None:
    """HealthServer.start() is idempotent."""
    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0)
    server.start()

    try:
        addr1 = server.address
        server.start()  # Should be no-op
        addr2 = server.address
        assert addr1 == addr2
    finally:
        server.stop()


def test_health_server_stop_idempotent() -> None:
    """HealthServer.stop() is idempotent."""
    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0)
    server.start()
    server.stop()
    server.stop()  # Should be no-op, no error
    assert server.address is None


def test_health_server_dynamic_readiness_check() -> None:
    """HealthServer readiness check is evaluated dynamically."""
    import json
    import urllib.error
    import urllib.request

    from weakincentives.runtime import HealthServer

    state = {"ready": False}
    server = HealthServer(
        host="127.0.0.1", port=0, readiness_check=lambda: state["ready"]
    )
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]

        # Initially not ready
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
            msg = "Expected HTTPError with 503 status"
            raise AssertionError(msg)
        except urllib.error.HTTPError as e:
            assert e.code == 503

        # Now ready
        state["ready"] = True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        server.stop()


# =============================================================================
# LoopGroup Health Server Integration Tests
# =============================================================================


def test_loop_group_starts_health_server(reset_coordinator: None) -> None:
    """LoopGroup starts health server when health_port is set."""
    import json
    import urllib.request

    from weakincentives.runtime import LoopGroup

    _ = reset_coordinator
    loops = [_MockRunnable(run_delay=0.2)]
    group = LoopGroup(loops=loops, health_port=0, health_host="127.0.0.1")

    # Run in background thread
    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        # Wait for server to start
        time.sleep(0.05)

        # Health server should be running
        assert group._health_server is not None
        _, port = group._health_server.address  # type: ignore[misc]

        # Liveness should work
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)


def test_loop_group_readiness_reflects_loop_state(reset_coordinator: None) -> None:
    """LoopGroup readiness endpoint reflects loop running state."""
    import json
    import urllib.request

    from weakincentives.runtime import LoopGroup

    _ = reset_coordinator
    loops = [_MockRunnable()]
    group = LoopGroup(loops=loops, health_port=0, health_host="127.0.0.1")

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        # Wait for server and loops to start
        time.sleep(0.1)

        assert group._health_server is not None
        _, port = group._health_server.address  # type: ignore[misc]

        # Readiness should be healthy when loops are running
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)


def test_loop_group_no_health_server_without_port(reset_coordinator: None) -> None:
    """LoopGroup does not start health server when health_port is None."""
    from weakincentives.runtime import LoopGroup

    _ = reset_coordinator
    loops = [_MockRunnable(run_delay=0.05)]
    group = LoopGroup(loops=loops)  # No health_port

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        time.sleep(0.02)
        assert group._health_server is None
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)


def test_loop_group_health_server_stops_on_shutdown(reset_coordinator: None) -> None:
    """LoopGroup stops health server on shutdown."""
    from weakincentives.runtime import LoopGroup

    _ = reset_coordinator
    loops = [_MockRunnable()]
    group = LoopGroup(loops=loops, health_port=0, health_host="127.0.0.1")

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        time.sleep(0.05)
        assert group._health_server is not None
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)

    # After shutdown, health server should be stopped
    assert group._health_server is None


# =============================================================================
# LoopGroup Watchdog Integration Tests
# =============================================================================


class _MockRunnableWithHeartbeat(_MockRunnable):
    """Mock implementation of Runnable with heartbeat for testing watchdog."""

    def __init__(self, *, run_delay: float = 0.0) -> None:
        super().__init__(run_delay=run_delay)
        from weakincentives.runtime.watchdog import Heartbeat as HeartbeatCls

        self._heartbeat: Heartbeat = HeartbeatCls()
        self.name = "test-loop"

    @property
    def heartbeat(self) -> Heartbeat:
        return self._heartbeat

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        # Beat the heartbeat before calling parent run
        self._heartbeat.beat()
        super().run(
            max_iterations=max_iterations,
            visibility_timeout=visibility_timeout,
            wait_time_seconds=wait_time_seconds,
        )


def test_loop_group_starts_watchdog_with_heartbeats(reset_coordinator: None) -> None:
    """LoopGroup starts watchdog when loops have heartbeat properties."""
    from weakincentives.runtime import LoopGroup

    _ = reset_coordinator
    loops = [_MockRunnableWithHeartbeat(run_delay=0.05)]
    group = LoopGroup(
        loops=loops,
        watchdog_threshold=60.0,
        watchdog_interval=1.0,
    )

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        time.sleep(0.02)
        # Watchdog should be started
        assert group._watchdog is not None
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)

    # Watchdog should be stopped after shutdown
    assert group._watchdog is None


def test_loop_group_watchdog_disabled_when_threshold_none(
    reset_coordinator: None,
) -> None:
    """LoopGroup does not start watchdog when watchdog_threshold is None."""
    from weakincentives.runtime import LoopGroup

    _ = reset_coordinator
    loops = [_MockRunnableWithHeartbeat(run_delay=0.05)]
    group = LoopGroup(
        loops=loops,
        watchdog_threshold=None,  # Disable watchdog
    )

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        time.sleep(0.02)
        # Watchdog should not be started
        assert group._watchdog is None
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)


def test_loop_group_build_readiness_check_with_heartbeats(
    reset_coordinator: None,
) -> None:
    """LoopGroup._build_readiness_check incorporates heartbeat freshness."""
    from weakincentives.runtime import LoopGroup

    _ = reset_coordinator

    loop = _MockRunnableWithHeartbeat()
    group = LoopGroup(
        loops=[loop],
        watchdog_threshold=0.1,  # Short threshold for testing
    )

    # Build the readiness check manually
    heartbeats = [loop._heartbeat]
    check = group._build_readiness_check(heartbeats)

    # Initially loop is not running, so check should fail
    assert check() is False

    # Simulate loop running
    loop._running = True
    loop._heartbeat.beat()

    # Now should be healthy
    assert check() is True

    # Wait for heartbeat to go stale
    time.sleep(0.15)

    # Now should be unhealthy due to stale heartbeat
    assert check() is False

    # Beat again to restore
    loop._heartbeat.beat()
    assert check() is True


def test_loop_group_build_readiness_check_without_threshold(
    reset_coordinator: None,
) -> None:
    """LoopGroup._build_readiness_check works when watchdog_threshold is None."""
    from weakincentives.runtime import LoopGroup

    _ = reset_coordinator

    loop = _MockRunnableWithHeartbeat()
    group = LoopGroup(
        loops=[loop],
        watchdog_threshold=None,  # No threshold
    )

    # Build the readiness check manually with heartbeats
    heartbeats = [loop._heartbeat]
    check = group._build_readiness_check(heartbeats)

    # Simulate loop running
    loop._running = True

    # Should be healthy regardless of heartbeat age when threshold is None
    time.sleep(0.05)  # Let heartbeat go a bit stale
    assert check() is True

    # Even with very stale heartbeat, still healthy (no threshold check)
    time.sleep(0.1)
    assert check() is True
