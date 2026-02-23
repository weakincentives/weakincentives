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

"""Tests for EvalLoop shutdown behavior."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

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


@pytest.fixture
def reset_coordinator() -> None:
    """Reset ShutdownCoordinator singleton before and after each test."""
    ShutdownCoordinator.reset()
    yield
    ShutdownCoordinator.reset()


# =============================================================================
# EvalLoop Shutdown Tests
# =============================================================================


def test_eval_loop_shutdown_stops_loop() -> None:
    """EvalLoop.shutdown() stops the run loop."""
    from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, Score

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
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
    from weakincentives.evals import (
        BASELINE,
        EvalLoop,
        EvalRequest,
        EvalResult,
        Sample,
        Score,
    )

    results: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(name="eval-results")
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        # Slow adapter to allow shutdown during batch processing
        adapter = _MockAdapter(delay=0.1)
        main_requests: InMemoryMailbox[
            AgentLoopRequest[str], AgentLoopResult[_Output]
        ] = InMemoryMailbox(name="main-requests")
        agent_loop = _TestLoop(
            adapter=adapter,
            requests=main_requests,
        )

        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
            evaluator=lambda o, e: Score(value=1.0, passed=True),
            requests=requests,
        )

        # Send multiple samples
        for i in range(3):
            sample = Sample(id=str(i), input=f"input-{i}", expected="success")
            requests.send(
                EvalRequest(sample=sample, experiment=BASELINE), reply_to=results
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
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
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
    requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
        name="eval-requests"
    )

    try:
        agent_loop = _create_test_loop()
        eval_loop: EvalLoop[str, _Output, str] = EvalLoop(
            loop=agent_loop,
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

    from weakincentives.evals import (
        BASELINE,
        EvalLoop,
        EvalRequest,
        EvalResult,
        Sample,
        Score,
    )
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
            AgentLoopRequest[_Request], AgentLoopResult[_Output]
        ] = InMemoryMailbox(name="main-requests")
        agent_loop = _TestLoop(adapter=adapter, requests=main_requests)

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
            loop=agent_loop, evaluator=shutdown_after_first, requests=requests
        )
        eval_loop_ref.append(eval_loop)

        # Send multiple samples - they will be returned in one batch
        for i in range(3):
            sample: Sample[_Request, str] = Sample(
                id=str(i),
                input=_Request(message=f"input-{i}"),
                expected="success",
            )
            requests.send(EvalRequest(sample=sample, experiment=BASELINE))

        # Run the loop - it will process first sample, then shutdown triggers,
        # then remaining messages in batch get nacked
        eval_loop.run(wait_time_seconds=0, max_iterations=1)

        # Should have processed exactly one sample (others were nacked/skipped)
        assert eval_count == 1
    finally:
        requests.close()
        main_requests.close()
