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

"""Shared test helpers for runtime tests."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

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
    Session,
    ShutdownCoordinator,
    wait_until,
)
from weakincentives.runtime.session.protocols import SessionProtocol

if TYPE_CHECKING:
    from weakincentives.runtime.watchdog import Heartbeat

# =============================================================================
# Existing helpers for mailbox tests
# =============================================================================


@dataclass(slots=True, frozen=True)
class SampleEvent:
    """Sample event type for testing."""

    data: str


@dataclass(slots=True, frozen=True)
class SampleResult:
    """Sample result type for testing."""

    status: str


# =============================================================================
# Lifecycle test data types
# =============================================================================


@dataclass(slots=True, frozen=True)
class LifecycleRequest:
    """Sample request type for lifecycle testing."""

    message: str


@dataclass(slots=True, frozen=True)
class LifecycleOutput:
    """Sample output type for lifecycle testing."""

    result: str


@dataclass(slots=True, frozen=True)
class LifecycleParams:
    """Sample params type for lifecycle testing."""

    content: str


# =============================================================================
# Mock Adapter
# =============================================================================


class LifecycleMockAdapter(ProviderAdapter[LifecycleOutput]):
    """Mock adapter for lifecycle testing."""

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
        prompt: Prompt[LifecycleOutput],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: object = None,
        run_context: object = None,
    ) -> PromptResponse[LifecycleOutput]:
        del prompt, session, deadline, budget, budget_tracker, heartbeat, run_context
        self.call_count += 1
        if self._delay > 0:
            time.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return PromptResponse(
            prompt_name="test",
            text="success",
            output=LifecycleOutput(result="success"),
        )


# =============================================================================
# Test AgentLoop
# =============================================================================


class LifecycleTestLoop(AgentLoop[LifecycleRequest, LifecycleOutput]):
    """Test implementation of AgentLoop for lifecycle testing."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[LifecycleOutput],
        requests: InMemoryMailbox[
            AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
        ],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests)
        self._template = PromptTemplate[LifecycleOutput](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[LifecycleParams](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )

    def prepare(
        self,
        request: LifecycleRequest,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[LifecycleOutput], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(LifecycleParams(content=request.message))
        session = Session(tags={"loop": "test"})
        return prompt, session


# =============================================================================
# Factory Helpers
# =============================================================================


def create_lifecycle_test_loop(
    *,
    delay: float = 0.0,
    error: Exception | None = None,
) -> LifecycleTestLoop:
    """Create a test AgentLoop with mock adapter."""
    adapter = LifecycleMockAdapter(delay=delay, error=error)
    requests: InMemoryMailbox[
        AgentLoopRequest[LifecycleRequest], AgentLoopResult[LifecycleOutput]
    ] = InMemoryMailbox(name="dummy-requests")
    return LifecycleTestLoop(adapter=adapter, requests=requests)


# =============================================================================
# Mock Runnable
# =============================================================================


class LifecycleMockRunnable:
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
# Mock Runnable with Heartbeat
# =============================================================================


class LifecycleMockRunnableWithHeartbeat(LifecycleMockRunnable):
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


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def reset_coordinator() -> None:
    """Reset ShutdownCoordinator singleton before and after each test."""
    ShutdownCoordinator.reset()
    yield  # type: ignore[misc]
    ShutdownCoordinator.reset()
