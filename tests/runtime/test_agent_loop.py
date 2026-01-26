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

"""Tests for AgentLoop orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

if TYPE_CHECKING:
    pass

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    SectionPath,
    SectionVisibility,
    VisibilityExpansionRequired,
)
from weakincentives.runtime.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.mailbox import (
    FakeMailbox,
    InMemoryMailbox,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.runtime.session import Session, VisibilityOverrides
from weakincentives.runtime.session.protocols import SessionProtocol


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
    """Mock adapter for testing AgentLoop behavior."""

    def __init__(
        self,
        *,
        response: PromptResponse[_Output] | None = None,
        error: Exception | None = None,
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]]
        | None = None,
    ) -> None:
        self._response = response or PromptResponse(
            prompt_name="test",
            text="test output",
            output=_Output(result="success"),
        )
        self._error = error
        self._visibility_requests = list(visibility_requests or [])
        self._call_count = 0
        self._last_budget_tracker: BudgetTracker | None = None
        self._budget_trackers: list[BudgetTracker | None] = []
        self._last_deadline: Deadline | None = None
        self._last_session: SessionProtocol | None = None
        # Track prompts to verify resources through prompt.resources
        self._last_prompt: Prompt[_Output] | None = None
        self._prompts: list[Prompt[_Output]] = []
        # Track resources captured during evaluate (while context is active)
        self._last_custom_resource: _CustomResource | None = None
        self._custom_resources: list[_CustomResource | None] = []
        # Track run_context passed during evaluate
        self._last_run_context: RunContext | None = None
        self._run_contexts: list[RunContext | None] = []

    def evaluate(
        self,
        prompt: Prompt[_Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: object = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[_Output]:
        del budget, heartbeat
        self._call_count += 1
        self._last_run_context = run_context
        self._run_contexts.append(run_context)
        self._last_budget_tracker = budget_tracker
        self._budget_trackers.append(budget_tracker)
        self._last_deadline = deadline
        self._last_session = session
        # Capture prompt to verify resources via prompt.resources
        self._last_prompt = prompt
        self._prompts.append(prompt)

        # Enter resource context (like real adapters do)
        with prompt.resources:
            # Capture resource during evaluate while context is active
            try:
                self._last_custom_resource = prompt.resources.get(_CustomResource)
            except Exception:
                # UnboundResourceError: resource type not registered
                self._last_custom_resource = None
            self._custom_resources.append(self._last_custom_resource)

            # If there are visibility requests remaining, raise the exception
            if self._visibility_requests:
                overrides = self._visibility_requests.pop(0)
                raise VisibilityExpansionRequired(
                    "Expansion required",
                    requested_overrides=overrides,
                    reason="test",
                    section_keys=tuple(k[0] for k in overrides),
                )

            if self._error is not None:
                raise self._error

            return self._response


class _TestLoop(AgentLoop[_Request, _Output]):
    """Test implementation of AgentLoop."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]]
        | FakeMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]],
        config: AgentLoopConfig | None = None,
        worker_id: str = "",
    ) -> None:
        super().__init__(
            adapter=adapter, requests=requests, config=config, worker_id=worker_id
        )
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
        self.session_created: Session | None = None
        self.finalize_called = False

    def prepare(
        self,
        request: _Request,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[_Output], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(_Params(content=request.message))
        session = Session(tags={"loop": "test"})
        self.session_created = session
        return prompt, session

    def finalize(self, prompt: Prompt[_Output], session: Session) -> None:
        del prompt
        self.finalize_called = True
        _ = session


# =============================================================================
# AgentLoopConfig Tests
# =============================================================================


def test_config_default_values() -> None:
    """AgentLoopConfig has sensible defaults."""
    config = AgentLoopConfig()
    assert config.deadline is None
    assert config.budget is None


def test_config_custom_values() -> None:
    """AgentLoopConfig accepts custom values."""
    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    budget = Budget(max_total_tokens=1000)
    config = AgentLoopConfig(
        deadline=deadline,
        budget=budget,
    )
    assert config.deadline is deadline
    assert config.budget is budget


# =============================================================================
# AgentLoopRequest Tests
# =============================================================================


def test_request_default_values() -> None:
    """AgentLoopRequest has sensible defaults."""
    request = AgentLoopRequest(request=_Request(message="hello"))
    assert request.request == _Request(message="hello")
    assert request.budget is None
    assert request.deadline is None
    assert isinstance(request.request_id, UUID)
    assert request.created_at.tzinfo == UTC


def test_request_custom_values() -> None:
    """AgentLoopRequest accepts custom values."""
    budget = Budget(max_total_tokens=1000)
    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    request = AgentLoopRequest(
        request=_Request(message="hello"),
        budget=budget,
        deadline=deadline,
    )
    assert request.budget is budget
    assert request.deadline is deadline


# =============================================================================
# AgentLoopResult Tests
# =============================================================================


def test_result_success_case() -> None:
    """AgentLoopResult represents successful completion."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    session_id = UUID("87654321-4321-8765-4321-876543218765")
    output = _Output(result="success")
    result: AgentLoopResult[_Output] = AgentLoopResult(
        request_id=request_id,
        output=output,
        session_id=session_id,
    )
    assert result.request_id == request_id
    assert result.output == output
    assert result.error is None
    assert result.session_id == session_id
    assert result.success is True
    assert result.completed_at.tzinfo == UTC


def test_result_error_case() -> None:
    """AgentLoopResult represents failure."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    result: AgentLoopResult[_Output] = AgentLoopResult(
        request_id=request_id,
        error="adapter failure",
    )
    assert result.request_id == request_id
    assert result.output is None
    assert result.error == "adapter failure"
    assert result.session_id is None
    assert result.success is False


def test_result_is_frozen() -> None:
    """AgentLoopResult is immutable."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    result: AgentLoopResult[_Output] = AgentLoopResult(
        request_id=request_id,
        output=_Output(result="success"),
    )
    with pytest.raises(AttributeError):
        result.output = None  # type: ignore[misc]


# =============================================================================
# AgentLoop Tests
# =============================================================================


def test_loop_processes_request() -> None:
    """AgentLoop processes request from mailbox."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        # Send request with reply_to
        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run single iteration
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check response
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.request_id == request.request_id
        assert msgs[0].body.output == _Output(result="success")
        assert msgs[0].body.error is None
        assert msgs[0].body.success is True
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_sends_error_on_failure() -> None:
    """AgentLoop sends error result on adapter failure."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("adapter failure"))
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.request_id == request.request_id
        assert msgs[0].body.output is None
        assert msgs[0].body.error == "adapter failure"
        assert msgs[0].body.success is False
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_acknowledges_request() -> None:
    """AgentLoop acknowledges processed request."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be acknowledged (gone from queue)
        assert requests.approximate_count() == 0
    finally:
        requests.close()
        results.close()


def test_loop_calls_finalize() -> None:
    """AgentLoop calls finalize after successful processing."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert loop.finalize_called
    finally:
        requests.close()
        results.close()


def test_loop_respects_max_iterations() -> None:
    """AgentLoop respects max_iterations limit."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        for i in range(5):
            requests.send(
                AgentLoopRequest(request=_Request(message=f"msg-{i}")),
                reply_to=results,
            )

        # Only run 2 iterations
        loop.run(max_iterations=2, wait_time_seconds=0)

        # Some requests may still be pending (depending on batch size)
        # At least we should have some responses
        assert results.approximate_count() >= 1
    finally:
        requests.close()
        results.close()


def test_loop_handles_visibility_expansion() -> None:
    """AgentLoop handles visibility expansion correctly."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Should succeed after visibility expansion
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert adapter._call_count == 2  # 1 visibility expansion + 1 success
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_uses_config_budget() -> None:
    """AgentLoop uses budget from config."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        budget = Budget(max_total_tokens=1000)
        config = AgentLoopConfig(budget=budget)
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_budget_tracker is not None
        assert adapter._last_budget_tracker.budget is budget
    finally:
        requests.close()
        results.close()


def test_loop_request_overrides_config() -> None:
    """AgentLoop uses request budget/deadline over config."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        config_budget = Budget(max_total_tokens=1000)
        config = AgentLoopConfig(budget=config_budget)
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        override_budget = Budget(max_total_tokens=2000)
        request = AgentLoopRequest(
            request=_Request(message="hello"),
            budget=override_budget,
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_budget_tracker is not None
        assert adapter._last_budget_tracker.budget is override_budget
    finally:
        requests.close()
        results.close()


def test_loop_nacks_on_response_send_failure() -> None:
    """AgentLoop nacks request when response send fails."""
    results: FakeMailbox[AgentLoopResult[_Output], None] = FakeMailbox(name="results")
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Make response send fail
        from weakincentives.runtime.mailbox import MailboxConnectionError

        results.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be nacked (still in queue for retry)
        # Note: it's in invisible state after receive, so we need to wait
        # or check approximate_count
        assert requests.approximate_count() == 1
    finally:
        requests.close()


def test_loop_nacks_on_error_response_send_failure() -> None:
    """AgentLoop nacks request when error response send fails."""
    results: FakeMailbox[AgentLoopResult[_Output], None] = FakeMailbox(name="results")
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        # Adapter that fails
        adapter = _MockAdapter(error=RuntimeError("adapter failure"))
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Make error response send fail too
        from weakincentives.runtime.mailbox import MailboxConnectionError

        results.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be nacked (still in queue for retry)
        assert requests.approximate_count() == 1
    finally:
        requests.close()


class _TestLoopNoFinalizeOverride(AgentLoop[_Request, _Output]):
    """Test implementation that doesn't override finalize."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]]
        | FakeMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]],
        config: AgentLoopConfig | None = None,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, config=config)
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


def test_loop_default_finalize() -> None:
    """AgentLoop default finalize does nothing."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoopNoFinalizeOverride(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        # Run should succeed even without finalize override
        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


# =============================================================================
# Postprocess Response Tests
# =============================================================================


class _TestLoopWithPostprocess(AgentLoop[_Request, _Output]):
    """Test implementation that tracks postprocess_response calls."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]]
        | FakeMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]],
        config: AgentLoopConfig | None = None,
        modify_output: bool = False,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, config=config)
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
        self.postprocess_response_called = False
        self.postprocess_response_input: _Output | None = None
        self._modify_output = modify_output

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

    def postprocess_response(
        self,
        output: _Output | None,
        prompt: Prompt[_Output],
        session: Session,
    ) -> _Output | None:
        del prompt, session
        self.postprocess_response_called = True
        self.postprocess_response_input = output
        if self._modify_output and output is not None:
            return _Output(result=f"modified: {output.result}")
        return output


def test_loop_calls_postprocess_response() -> None:
    """AgentLoop calls postprocess_response after successful processing."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoopWithPostprocess(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert loop.postprocess_response_called
        assert loop.postprocess_response_input is not None
        assert loop.postprocess_response_input == _Output(result="success")
    finally:
        requests.close()
        results.close()


def test_loop_postprocess_response_can_modify_output() -> None:
    """AgentLoop postprocess_response can modify the output."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoopWithPostprocess(
            adapter=adapter, requests=requests, modify_output=True
        )

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        # Output should be modified by postprocess_response
        assert msgs[0].body.output == _Output(result="modified: success")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_default_postprocess_response() -> None:
    """AgentLoop default postprocess_response returns response unchanged."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        # Use _TestLoopNoFinalizeOverride which doesn't override postprocess_response
        loop = _TestLoopNoFinalizeOverride(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        # Output should be unchanged (default postprocess_response returns as-is)
        assert msgs[0].body.output == _Output(result="success")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_postprocess_response_called_after_visibility_expansion() -> None:
    """AgentLoop calls postprocess_response after visibility expansion retries."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoopWithPostprocess(
            adapter=adapter, requests=requests, modify_output=True
        )

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Should succeed after visibility expansion
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        # postprocess_response should have been called (only once, after final success)
        assert loop.postprocess_response_called
        # Output should be modified
        assert msgs[0].body.output == _Output(result="modified: success")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


# =============================================================================
# Resource Injection Tests
# =============================================================================


@dataclass(slots=True, frozen=True)
class _CustomResource:
    """Custom resource for testing resource injection."""

    name: str


def test_config_accepts_resources() -> None:
    """AgentLoopConfig accepts resources parameter."""
    resource = _CustomResource(name="config-resource")
    resources: dict[type[object], object] = {_CustomResource: resource}
    config = AgentLoopConfig(resources=resources)
    assert config.resources is resources


def test_request_accepts_resources() -> None:
    """AgentLoopRequest accepts resources parameter."""
    resource = _CustomResource(name="request-resource")
    resources: dict[type[object], object] = {_CustomResource: resource}
    request = AgentLoopRequest(
        request=_Request(message="hello"),
        resources=resources,
    )
    assert request.resources is resources


def test_loop_passes_resources_from_config() -> None:
    """AgentLoop binds config resources to prompt."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        resource = _CustomResource(name="config-resource")
        config = AgentLoopConfig(resources={_CustomResource: resource})
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Resources are now bound to prompt and accessible via prompt.resources
        # (captured during evaluate while context is active)
        assert adapter._last_custom_resource is resource
    finally:
        requests.close()
        results.close()


def test_loop_request_resources_overrides_config() -> None:
    """AgentLoop request resources override config resources."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        config_resource = _CustomResource(name="config-resource")
        config = AgentLoopConfig(resources={_CustomResource: config_resource})
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        override_resource = _CustomResource(name="override-resource")
        request = AgentLoopRequest(
            request=_Request(message="hello"),
            resources={_CustomResource: override_resource},
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Override resources are bound to prompt, overriding config resources
        # (captured during evaluate while context is active)
        assert adapter._last_custom_resource is override_resource
    finally:
        requests.close()
        results.close()


def test_same_resources_used_across_visibility_retries() -> None:
    """Same resources are used across visibility expansion retries."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        resource = _CustomResource(name="persistent-resource")
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(
            request=_Request(message="hello"),
            resources={_CustomResource: resource},
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Called 3 times: 2 visibility expansions + 1 success
        assert adapter._call_count == 3
        # Same resource should be used for all calls
        # (captured during evaluate while context is active)
        assert len(adapter._custom_resources) == 3
        assert all(r is resource for r in adapter._custom_resources)
    finally:
        requests.close()
        results.close()


def test_no_resources_when_not_set() -> None:
    """Prompt has empty resource context when no resources configured."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # When no resources configured, the custom resource is None
        # (captured during evaluate while context is active)
        assert adapter._last_custom_resource is None
    finally:
        requests.close()
        results.close()


# =============================================================================
# Visibility Override Accumulation Tests
# =============================================================================


def test_visibility_overrides_accumulate_in_session() -> None:
    """AgentLoop accumulates visibility overrides in session state."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check session state has accumulated overrides
        assert loop.session_created is not None
        overrides = loop.session_created[VisibilityOverrides].latest()
        assert overrides is not None
        assert overrides.get(("section1",)) == SectionVisibility.FULL
        assert overrides.get(("section2",)) == SectionVisibility.FULL
    finally:
        requests.close()
        results.close()


def test_same_budget_tracker_used_across_visibility_retries() -> None:
    """Same BudgetTracker is used across visibility expansion retries."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        budget = Budget(max_total_tokens=1000)
        config = AgentLoopConfig(budget=budget)
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Called 3 times: 2 visibility expansions + 1 success
        assert adapter._call_count == 3
        # Same BudgetTracker should be used for all calls
        assert len(adapter._budget_trackers) == 3
        assert all(t is adapter._budget_trackers[0] for t in adapter._budget_trackers)
        # And it should have the correct budget
        assert adapter._budget_trackers[0] is not None
        assert adapter._budget_trackers[0].budget is budget
    finally:
        requests.close()
        results.close()


def test_no_budget_tracker_when_no_budget() -> None:
    """No BudgetTracker is created when no budget is set."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_budget_tracker is None
    finally:
        requests.close()
        results.close()


# =============================================================================
# Mailbox Close Tests
# =============================================================================


def test_loop_handles_expired_receipt_handle_on_ack() -> None:
    """AgentLoop continues when receipt handle expires during processing."""
    results: FakeMailbox[AgentLoopResult[_Output], None] = FakeMailbox(name="results")
    requests: FakeMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        FakeMailbox(name="requests")
    )

    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=requests)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests.send(request, reply_to=results)

    # Receive the message to get the handle, then expire it
    msgs = requests.receive(max_messages=1)
    assert len(msgs) == 1
    msg = msgs[0]

    # Expire the handle to simulate slow processing
    requests.expire_handle(msg.receipt_handle)

    # Create a result to send
    result: AgentLoopResult[_Output] = AgentLoopResult(
        request_id=request.request_id,
        output=_Output(result="success"),
    )

    # Call _reply_and_ack directly - should handle expired handle gracefully
    loop._reply_and_ack(msg, result)

    # Should not raise - the expired handle is handled gracefully
    # Response should still be sent
    assert results.approximate_count() == 1


def test_loop_handles_expired_receipt_handle_on_nack() -> None:
    """AgentLoop continues when receipt handle expires during nack after send failure."""
    from weakincentives.runtime.mailbox import MailboxConnectionError

    results: FakeMailbox[AgentLoopResult[_Output], None] = FakeMailbox(name="results")
    requests: FakeMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        FakeMailbox(name="requests")
    )

    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=requests)

    request = AgentLoopRequest(request=_Request(message="hello"))
    requests.send(request, reply_to=results)

    # Receive the message to get the handle
    msgs = requests.receive(max_messages=1)
    assert len(msgs) == 1
    msg = msgs[0]

    # Expire the handle AND make send fail
    # This simulates: processing took too long, handle expired,
    # AND the response queue is also having issues
    requests.expire_handle(msg.receipt_handle)
    results.set_connection_error(MailboxConnectionError("connection lost"))

    # Create a result to send
    result: AgentLoopResult[_Output] = AgentLoopResult(
        request_id=request.request_id,
        output=_Output(result="success"),
    )

    # Call _reply_and_ack directly - should handle both failures gracefully
    loop._reply_and_ack(msg, result)

    # Should not raise - both failures are handled gracefully


def test_loop_exits_when_mailbox_closed() -> None:
    """AgentLoop.run() exits when requests mailbox is closed."""
    import threading

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=requests)

    exited = []

    def run_loop() -> None:
        # Would run forever with max_iterations=None
        loop.run(max_iterations=None, wait_time_seconds=1)
        exited.append(True)

    thread = threading.Thread(target=run_loop)
    thread.start()

    # Close the mailbox - should cause loop to exit
    requests.close()
    results.close()

    # Thread should exit quickly
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(exited) == 1


# =============================================================================
# Worker ID Tests
# =============================================================================


def test_loop_worker_id_property() -> None:
    """AgentLoop exposes worker_id property."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, worker_id="test-worker-42")
        assert loop.worker_id == "test-worker-42"
    finally:
        requests.close()
        results.close()


# =============================================================================
# RunContext Tests
# =============================================================================


def test_loop_includes_run_context_in_result() -> None:
    """AgentLoop result includes RunContext."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, worker_id="worker-1")

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.run_context is not None
        assert msgs[0].body.run_context.request_id == request.request_id
        assert msgs[0].body.run_context.worker_id == "worker-1"
        assert msgs[0].body.run_context.attempt == 1
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_preserves_run_context_from_request() -> None:
    """AgentLoop preserves trace_id and span_id from request RunContext."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, worker_id="worker-2")

        # Include run_context with trace/span IDs in request
        input_run_ctx = RunContext(
            trace_id="trace-abc-123",
            span_id="span-xyz-456",
        )
        request = AgentLoopRequest(
            request=_Request(message="hello"),
            run_context=input_run_ctx,
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result_ctx = msgs[0].body.run_context
        assert result_ctx is not None
        # run_id is generated fresh per execution
        assert result_ctx.run_id != input_run_ctx.run_id
        # request_id comes from AgentLoopRequest.request_id (for correlation)
        assert result_ctx.request_id == request.request_id
        # trace/span IDs are preserved from input run_context
        assert result_ctx.trace_id == "trace-abc-123"
        assert result_ctx.span_id == "span-xyz-456"
        # worker_id comes from the loop
        assert result_ctx.worker_id == "worker-2"
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_run_id_matches_during_and_after_execution() -> None:
    """RunContext.run_id during execution matches run_id in result.

    This verifies that the run_id is generated once and preserved (via replace())
    rather than regenerated, which would break telemetry correlation.
    """
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Get run_context that was passed to adapter during execution
        assert adapter._last_run_context is not None
        execution_run_id = adapter._last_run_context.run_id

        # Get result
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result_ctx = msgs[0].body.run_context
        assert result_ctx is not None

        # CRITICAL: run_id must be the same
        assert result_ctx.run_id == execution_run_id

        # session_id should be populated in result (via replace())
        assert result_ctx.session_id is not None
        assert result_ctx.session_id == msgs[0].body.session_id

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_includes_run_context_on_error() -> None:
    """AgentLoop includes RunContext in error result."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("adapter failure"))
        loop = _TestLoop(adapter=adapter, requests=requests, worker_id="worker-err")

        request = AgentLoopRequest(request=_Request(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        assert msgs[0].body.run_context is not None
        assert msgs[0].body.run_context.worker_id == "worker-err"
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


# =============================================================================
# Debug Bundle Integration Tests
# =============================================================================


def test_loop_with_debug_bundle_enabled(tmp_path: Path) -> None:
    """AgentLoop creates debug bundle when enabled in config."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello with bundle"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True

        # Bundle path should be set
        assert result.bundle_path is not None
        assert isinstance(result.bundle_path, Path)
        assert result.bundle_path.exists()

        # Bundle should be loadable and contain request data
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.manifest is not None
        assert bundle.request_input is not None
        assert bundle.request_output is not None
        assert bundle.run_context is not None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_debug_bundle_includes_filesystem(tmp_path: Path) -> None:
    """AgentLoop includes filesystem snapshot in debug bundle when provided."""
    from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
    from weakincentives.debug.bundle import BundleConfig, DebugBundle
    from weakincentives.filesystem import Filesystem

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        # Create filesystem with some files
        fs = InMemoryFilesystem()
        _ = fs.write("/test.txt", "Hello, World!")
        _ = fs.write("/subdir/nested.txt", "Nested content")

        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(
            debug_bundle=bundle_config,
            resources={Filesystem: fs},
        )
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello with fs"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should contain filesystem files
        bundle = DebugBundle.load(result.bundle_path)
        files = bundle.list_files()
        filesystem_files = [f for f in files if f.startswith("filesystem/")]
        assert len(filesystem_files) > 0
        assert any("test.txt" in f for f in filesystem_files)
        assert any("nested.txt" in f for f in filesystem_files)

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_debug_bundle_no_filesystem_in_resources(tmp_path: Path) -> None:
    """AgentLoop handles debug bundle when resources exist but no Filesystem."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        # Provide resources but without Filesystem
        class DummyResource:
            pass

        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(
            debug_bundle=bundle_config,
            resources={DummyResource: DummyResource()},  # Resources but no Filesystem
        )
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello without fs"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should NOT contain filesystem files
        bundle = DebugBundle.load(result.bundle_path)
        files = bundle.list_files()
        filesystem_files = [f for f in files if f.startswith("filesystem/")]
        assert len(filesystem_files) == 0

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_without_debug_bundle() -> None:
    """AgentLoop works normally without debug bundle."""
    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        # No debug_bundle in config
        config = AgentLoopConfig()
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello without bundle"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        # Bundle path should be None when not enabled
        assert result.bundle_path is None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_bundle_config_no_target() -> None:
    """AgentLoop falls back to unbundled when bundle target is None."""
    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        # BundleConfig with target=None
        bundle_config = BundleConfig(target=None)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello with no target"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        # Bundle path should be None when target is not set
        assert result.bundle_path is None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_bundle_failure_uses_handle_failure(tmp_path: Path) -> None:
    """AgentLoop uses handle_failure when bundle creation fails."""
    from unittest.mock import patch

    from weakincentives.debug.bundle import BundleConfig

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello bundle fail"))
        requests.send(request, reply_to=results)

        # Make BundleWriter raise an exception on enter
        def failing_enter(self: object) -> object:
            raise RuntimeError("Bundle creation failed")

        with patch("weakincentives.debug.bundle.BundleWriter.__enter__", failing_enter):
            loop.run(max_iterations=1, wait_time_seconds=0)

        # The error should have been handled via handle_failure path
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        # Result should indicate failure
        assert result.success is False
        assert result.error is not None
        assert "Bundle creation failed" in result.error

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_execution_failure_but_bundle_created(tmp_path: Path) -> None:
    """AgentLoop includes bundle_path in error response when execution fails."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        # Adapter that raises an error during evaluate (after bundle is entered)
        adapter = _MockAdapter(error=RuntimeError("Execution failed"))
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello execution fail"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # The error should have been handled but bundle_path should still be set
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        # Result should indicate failure
        assert result.success is False
        assert result.error is not None
        assert "Execution failed" in result.error

        # Bundle path should be set even though execution failed
        assert result.bundle_path is not None
        assert isinstance(result.bundle_path, Path)
        assert result.bundle_path.exists()

        # Bundle should be loadable and contain error info
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.manifest is not None
        assert bundle.manifest.request.status == "error"
        assert bundle.request_input is not None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_calls_session_methods(tmp_path: Path) -> None:
    """AgentLoop attempts to write session snapshots to debug bundle."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello session methods"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should be loadable and have standard files
        # Note: session files may be empty for fresh sessions with no state,
        # so we just verify the bundle was created successfully
        bundle = DebugBundle.load(result.bundle_path)
        files = bundle.list_files()
        # At minimum, the bundle should contain manifest, readme, and request files
        assert any("manifest.json" in f for f in files)
        assert any("request/input.json" in f for f in files)

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_includes_metrics(tmp_path: Path) -> None:
    """AgentLoop writes metrics.json to debug bundle with timing and token info."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello metrics"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should contain metrics
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.metrics is not None
        assert "timing" in bundle.metrics
        assert "token_usage" in bundle.metrics
        assert "duration_ms" in bundle.metrics["timing"]

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_metrics_include_budget(tmp_path: Path) -> None:
    """AgentLoop writes budget info to metrics.json when budget tracking is enabled."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        budget = Budget(max_total_tokens=1000)
        config = AgentLoopConfig(debug_bundle=bundle_config, budget=budget)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello budget metrics"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle metrics should include budget info
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.metrics is not None
        assert "budget" in bundle.metrics
        assert "consumed" in bundle.metrics["budget"]
        assert "limits" in bundle.metrics["budget"]

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_includes_prompt_info(tmp_path: Path) -> None:
    """AgentLoop writes prompt info (ns, key, adapter) to bundle manifest."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello prompt info"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle manifest should include prompt info
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.manifest is not None
        assert bundle.manifest.prompt.ns == "test"
        assert bundle.manifest.prompt.key == "test-prompt"
        # Adapter name is from the mock adapter class name since it's not a known adapter
        assert bundle.manifest.prompt.adapter is not None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_run_context_has_session_id(tmp_path: Path) -> None:
    """AgentLoop writes run_context.json with session_id after execution."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello run context"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle run_context should have session_id
        bundle = DebugBundle.load(result.bundle_path)
        assert bundle.run_context is not None
        assert "session_id" in bundle.run_context
        # session_id should match result
        assert bundle.run_context["session_id"] == str(result.session_id)

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_includes_prompt_overrides(tmp_path: Path) -> None:
    """AgentLoop writes prompt_overrides.json when visibility overrides are set."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        # Request visibility expansion to trigger visibility overrides
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello overrides"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True
        assert result.bundle_path is not None

        # Bundle should contain prompt overrides
        bundle = DebugBundle.load(result.bundle_path)
        # prompt_overrides should have overrides dict with section key
        # Note: the file list should include prompt_overrides.json
        files = bundle.list_files()
        assert any("prompt_overrides.json" in f for f in files)
        assert bundle.prompt_overrides is not None
        assert "overrides" in bundle.prompt_overrides
        assert "section1" in bundle.prompt_overrides["overrides"]

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_debug_bundle_per_request_override(tmp_path: Path) -> None:
    """AgentLoop uses per-request debug_bundle config override."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        # No debug_bundle in config
        config = AgentLoopConfig()
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        # But include debug_bundle in request
        bundle_config = BundleConfig(target=tmp_path)
        request = AgentLoopRequest(
            request=_Request(message="hello request bundle"),
            debug_bundle=bundle_config,
        )
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        result = msgs[0].body
        assert result.success is True

        # Bundle should be created even though config didn't have it
        assert result.bundle_path is not None
        assert result.bundle_path.exists()

        bundle = DebugBundle.load(result.bundle_path)
        # Trigger should be "request" since it came from per-request config
        assert bundle.manifest.capture.trigger == "request"

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_handles_visibility_expansion_with_bundle(tmp_path: Path) -> None:
    """AgentLoop handles visibility expansion correctly with bundling enabled."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello expansion"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Should succeed after visibility expansion
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert adapter._call_count == 2  # 1 visibility expansion + 1 success
        assert msgs[0].body.bundle_path is not None

        # Bundle should have visibility overrides recorded in prompt_overrides.json
        bundle = DebugBundle.load(msgs[0].body.bundle_path)
        files = bundle.list_files()
        assert any("prompt_overrides.json" in f for f in files)
        assert bundle.prompt_overrides is not None

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_with_debug_bundle_includes_environment(tmp_path: Path) -> None:
    """AgentLoop includes environment capture in debug bundle."""
    from weakincentives.debug.bundle import BundleConfig, DebugBundle

    results: InMemoryMailbox[AgentLoopResult[_Output], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[AgentLoopRequest[_Request], AgentLoopResult[_Output]] = (
        InMemoryMailbox(name="requests")
    )
    try:
        adapter = _MockAdapter()
        bundle_config = BundleConfig(target=tmp_path)
        config = AgentLoopConfig(debug_bundle=bundle_config)
        loop = _TestLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=_Request(message="hello environment"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert msgs[0].body.bundle_path is not None

        # Bundle should have environment files
        bundle = DebugBundle.load(msgs[0].body.bundle_path)
        files = bundle.list_files()
        assert any("environment/system.json" in f for f in files)
        assert any("environment/python.json" in f for f in files)
        assert any("environment/env_vars.json" in f for f in files)
        assert any("environment/command.txt" in f for f in files)

        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
