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
from weakincentives.runtime import (
    AgentLoop,
    AgentLoopRequest,
    AgentLoopResult,
    InMemoryMailbox,
    LoopGroup,
    Session,
    ShutdownCoordinator,
    wait_until,
)
from weakincentives.runtime.session.protocols import SessionProtocol

if TYPE_CHECKING:
    pass

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
        heartbeat: object = None,
        run_context: object = None,
    ) -> PromptResponse[_Output]:
        del prompt, session, deadline, budget, budget_tracker, heartbeat, run_context
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


class _TestLoop(AgentLoop[_Request, _Output]):
    """Test implementation of AgentLoop."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]],
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
        request: _Request,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[_Output], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(_Params(content=request.message))
        session = Session(tags={"loop": "test"})
        return prompt, session


def _create_test_loop(
    *,
    delay: float = 0.0,
    error: Exception | None = None,
) -> _TestLoop:
    """Create a test AgentLoop with mock adapter."""
    adapter = _MockAdapter(delay=delay, error=error)
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
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
