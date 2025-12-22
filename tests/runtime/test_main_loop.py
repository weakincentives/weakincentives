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

"""Tests for MainLoop orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

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
    VisibilityOverrides,
)
from weakincentives.prompt.tool import ResourceRegistry
from weakincentives.runtime import InMemoryMailbox, InProcessDispatcher, Session
from weakincentives.runtime.main_loop import (
    MainLoop,
    MainLoopConfig,
    MainLoopRequest,
    MainLoopResult,
)
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
    """Mock adapter for testing MainLoop behavior."""

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
        self._last_resources: ResourceRegistry | None = None
        self._resources_list: list[ResourceRegistry | None] = []

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
        del prompt, budget
        self._call_count += 1
        self._last_budget_tracker = budget_tracker
        self._budget_trackers.append(budget_tracker)
        self._last_deadline = deadline
        self._last_session = session
        self._last_resources = resources
        self._resources_list.append(resources)

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


# Type alias for the test mailbox
_TestMailbox = InMemoryMailbox[MainLoopRequest[_Request], MainLoopResult[_Output]]


class _TestLoop(MainLoop[_Request, _Output]):
    """Test implementation of MainLoop."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: _TestMailbox,
        config: MainLoopConfig | None = None,
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
        self.session_created: Session | None = None
        self._bus = InProcessDispatcher()

    def create_prompt(self, request: _Request) -> Prompt[_Output]:
        return Prompt(self._template).bind(_Params(content=request.message))

    def create_session(self) -> Session:
        session = Session(bus=self._bus, tags={"loop": "test"})
        self.session_created = session
        return session


# =============================================================================
# MainLoopConfig Tests
# =============================================================================


def test_config_default_values() -> None:
    """MainLoopConfig has sensible defaults."""
    config = MainLoopConfig()
    assert config.deadline is None
    assert config.budget is None
    assert config.visibility_timeout == 300
    assert config.wait_time_seconds == 20


def test_config_custom_values() -> None:
    """MainLoopConfig accepts custom values."""
    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(
        deadline=deadline,
        budget=budget,
        visibility_timeout=60,
        wait_time_seconds=5,
    )
    assert config.deadline is deadline
    assert config.budget is budget
    assert config.visibility_timeout == 60
    assert config.wait_time_seconds == 5


# =============================================================================
# MainLoopRequest Tests
# =============================================================================


def test_request_default_values() -> None:
    """MainLoopRequest has sensible defaults."""
    request = MainLoopRequest(request=_Request(message="hello"))
    assert request.request == _Request(message="hello")
    assert request.budget is None
    assert request.deadline is None
    assert isinstance(request.request_id, UUID)
    assert request.created_at.tzinfo == UTC


def test_request_custom_values() -> None:
    """MainLoopRequest accepts custom values."""
    budget = Budget(max_total_tokens=1000)
    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    request = MainLoopRequest(
        request=_Request(message="hello"),
        budget=budget,
        deadline=deadline,
    )
    assert request.budget is budget
    assert request.deadline is deadline


# =============================================================================
# MainLoopResult Tests
# =============================================================================


def test_result_success() -> None:
    """MainLoopResult success property returns True for successful execution."""
    response = PromptResponse(
        prompt_name="test",
        text="result",
        output=_Output(result="success"),
    )
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    session_id = UUID("87654321-4321-8765-4321-876543218765")
    result = MainLoopResult(
        request_id=request_id,
        response=response,
        error=None,
        session_id=session_id,
    )
    assert result.success is True
    assert result.request_id == request_id
    assert result.response is response
    assert result.session_id == session_id
    assert result.completed_at.tzinfo == UTC


def test_result_failure() -> None:
    """MainLoopResult success property returns False for failed execution."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    session_id = UUID("87654321-4321-8765-4321-876543218765")
    error = ValueError("test error")
    result = MainLoopResult(
        request_id=request_id,
        response=None,
        error=error,
        session_id=session_id,
    )
    assert result.success is False
    assert result.error is error


# =============================================================================
# MainLoop.execute() Tests
# =============================================================================


def test_execute_successful_execution() -> None:
    """MainLoop.execute returns response on success."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    response, session = loop.execute(_Request(message="hello"))

    assert response.output == _Output(result="success")
    assert adapter._call_count == 1
    assert loop.session_created is not None
    assert session is loop.session_created


def test_execute_passes_budget_from_config() -> None:
    """MainLoop.execute creates BudgetTracker with config budget."""
    mailbox: _TestMailbox = InMemoryMailbox()
    budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(budget=budget)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    loop.execute(_Request(message="hello"))

    assert adapter._last_budget_tracker is not None
    assert adapter._last_budget_tracker.budget is budget


def test_execute_passes_deadline_from_config() -> None:
    """MainLoop.execute passes config deadline to adapter."""
    mailbox: _TestMailbox = InMemoryMailbox()
    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    config = MainLoopConfig(deadline=deadline)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    loop.execute(_Request(message="hello"))

    assert adapter._last_deadline is deadline


def test_execute_budget_overrides_config() -> None:
    """MainLoop.execute budget parameter overrides config."""
    mailbox: _TestMailbox = InMemoryMailbox()
    config_budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(budget=config_budget)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    override_budget = Budget(max_total_tokens=2000)
    loop.execute(_Request(message="hello"), budget=override_budget)

    assert adapter._last_budget_tracker is not None
    assert adapter._last_budget_tracker.budget is override_budget


def test_execute_deadline_overrides_config() -> None:
    """MainLoop.execute deadline parameter overrides config."""
    mailbox: _TestMailbox = InMemoryMailbox()
    config_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    config = MainLoopConfig(deadline=config_deadline)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    override_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=10))
    loop.execute(_Request(message="hello"), deadline=override_deadline)

    assert adapter._last_deadline is override_deadline


def test_execute_handles_visibility_expansion() -> None:
    """MainLoop.execute accumulates visibility overrides in session state and retries."""
    mailbox: _TestMailbox = InMemoryMailbox()
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("section1",): SectionVisibility.FULL},
        {("section2",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    response, session = loop.execute(_Request(message="hello"))

    assert response.output == _Output(result="success")
    # Called 3 times: 2 visibility expansions + 1 success
    assert adapter._call_count == 3
    # Final overrides should be in session state
    overrides = session[VisibilityOverrides].latest()
    assert overrides is not None
    assert overrides.get(("section1",)) == SectionVisibility.FULL
    assert overrides.get(("section2",)) == SectionVisibility.FULL


def test_execute_propagates_adapter_error() -> None:
    """MainLoop.execute propagates adapter exceptions."""
    mailbox: _TestMailbox = InMemoryMailbox()
    error = RuntimeError("adapter failure")
    adapter = _MockAdapter(error=error)
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    with pytest.raises(RuntimeError, match="adapter failure"):
        loop.execute(_Request(message="hello"))


# =============================================================================
# MainLoop.run() Tests
# =============================================================================


def test_run_processes_single_message() -> None:
    """MainLoop.run processes a single message from mailbox."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    # Send a message
    request = MainLoopRequest(request=_Request(message="hello"))
    reply = mailbox.send_expecting_reply(request)

    # Process it
    loop.run(max_iterations=1)

    # Get result
    result = reply.wait(timeout=1)
    assert result.success is True
    assert result.response is not None
    assert result.response.output == _Output(result="success")


def test_run_processes_multiple_messages() -> None:
    """MainLoop.run processes multiple messages from mailbox."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    # Send multiple messages
    replies = []
    for i in range(3):
        request = MainLoopRequest(request=_Request(message=f"message-{i}"))
        replies.append(mailbox.send_expecting_reply(request))

    # Process all
    loop.run(max_iterations=3)

    # Verify all completed
    for reply in replies:
        result = reply.wait(timeout=1)
        assert result.success is True


def test_run_stops_at_max_iterations() -> None:
    """MainLoop.run respects max_iterations limit."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    # Send more messages than we'll process
    for i in range(5):
        request = MainLoopRequest(request=_Request(message=f"message-{i}"))
        _ = mailbox.send(request)

    # Process only 2
    loop.run(max_iterations=2)

    # Should have 3 remaining
    assert mailbox.approximate_count() == 3


def test_run_stops_when_mailbox_empty() -> None:
    """MainLoop.run stops when mailbox is empty."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    # Empty mailbox - run should return immediately
    loop.run(max_iterations=None)

    assert adapter._call_count == 0


def test_run_handles_error_in_result() -> None:
    """MainLoop.run captures errors in result instead of propagating."""
    mailbox: _TestMailbox = InMemoryMailbox()
    error = RuntimeError("adapter failure")
    adapter = _MockAdapter(error=error)
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    request = MainLoopRequest(request=_Request(message="hello"))
    reply = mailbox.send_expecting_reply(request)

    # Run should not raise
    loop.run(max_iterations=1)

    # Error should be in result
    result = reply.wait(timeout=1)
    assert result.success is False
    assert result.error is error


def test_run_fire_and_forget_message() -> None:
    """MainLoop.run handles fire-and-forget messages."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    # Send without expecting reply
    request = MainLoopRequest(request=_Request(message="hello"))
    _ = mailbox.send(request)

    # Process it
    loop.run(max_iterations=1)

    # Should be processed
    assert adapter._call_count == 1
    assert mailbox.approximate_count() == 0


# =============================================================================
# Request-Reply Pattern Tests
# =============================================================================


def test_request_reply_pattern() -> None:
    """MainLoop supports request-reply pattern via mailbox."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    # Client sends request expecting reply
    request = MainLoopRequest(request=_Request(message="hello"))
    reply = mailbox.send_expecting_reply(request)

    # Loop processes
    loop.run(max_iterations=1)

    # Client gets result
    result = reply.wait(timeout=1)
    assert result.success is True
    assert result.request_id == request.request_id


def test_request_reply_with_budget_override() -> None:
    """MainLoopRequest budget is passed through to result."""
    mailbox: _TestMailbox = InMemoryMailbox()
    config_budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(budget=config_budget)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    override_budget = Budget(max_total_tokens=2000)
    request = MainLoopRequest(
        request=_Request(message="hello"),
        budget=override_budget,
    )
    reply = mailbox.send_expecting_reply(request)

    loop.run(max_iterations=1)
    _ = reply.wait(timeout=1)

    assert adapter._last_budget_tracker is not None
    assert adapter._last_budget_tracker.budget is override_budget


def test_request_reply_with_deadline_override() -> None:
    """MainLoopRequest deadline is passed through."""
    mailbox: _TestMailbox = InMemoryMailbox()
    config_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    config = MainLoopConfig(deadline=config_deadline)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    override_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=10))
    request = MainLoopRequest(
        request=_Request(message="hello"),
        deadline=override_deadline,
    )
    reply = mailbox.send_expecting_reply(request)

    loop.run(max_iterations=1)
    _ = reply.wait(timeout=1)

    assert adapter._last_deadline is override_deadline


# =============================================================================
# Session and Budget Persistence Tests
# =============================================================================


def test_same_budget_tracker_used_across_visibility_retries() -> None:
    """Same BudgetTracker is used across visibility expansion retries."""
    mailbox: _TestMailbox = InMemoryMailbox()
    budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(budget=budget)
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("section1",): SectionVisibility.FULL},
        {("section2",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    loop.execute(_Request(message="hello"))

    # Called 3 times: 2 visibility expansions + 1 success
    assert adapter._call_count == 3
    # Same BudgetTracker should be used for all calls
    assert len(adapter._budget_trackers) == 3
    assert all(t is adapter._budget_trackers[0] for t in adapter._budget_trackers)
    # And it should have the correct budget
    assert adapter._budget_trackers[0] is not None
    assert adapter._budget_trackers[0].budget is budget


def test_no_budget_tracker_when_no_budget() -> None:
    """No BudgetTracker is created when no budget is set."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    loop.execute(_Request(message="hello"))

    assert adapter._last_budget_tracker is None


# =============================================================================
# Resource Injection Tests
# =============================================================================


@dataclass(slots=True, frozen=True)
class _CustomResource:
    """Custom resource for testing resource injection."""

    name: str


def test_config_accepts_resources() -> None:
    """MainLoopConfig accepts resources parameter."""
    resource = _CustomResource(name="config-resource")
    resources = ResourceRegistry.build({_CustomResource: resource})
    config = MainLoopConfig(resources=resources)
    assert config.resources is resources


def test_request_accepts_resources() -> None:
    """MainLoopRequest accepts resources parameter."""
    resource = _CustomResource(name="request-resource")
    resources = ResourceRegistry.build({_CustomResource: resource})
    request = MainLoopRequest(
        request=_Request(message="hello"),
        resources=resources,
    )
    assert request.resources is resources


def test_execute_passes_resources_from_config() -> None:
    """MainLoop.execute passes config resources to adapter."""
    mailbox: _TestMailbox = InMemoryMailbox()
    resource = _CustomResource(name="config-resource")
    resources = ResourceRegistry.build({_CustomResource: resource})
    config = MainLoopConfig(resources=resources)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    loop.execute(_Request(message="hello"))

    assert adapter._last_resources is resources


def test_execute_resources_overrides_config() -> None:
    """MainLoop.execute resources parameter overrides config."""
    mailbox: _TestMailbox = InMemoryMailbox()
    config_resource = _CustomResource(name="config-resource")
    config_resources = ResourceRegistry.build({_CustomResource: config_resource})
    config = MainLoopConfig(resources=config_resources)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    override_resource = _CustomResource(name="override-resource")
    override_resources = ResourceRegistry.build({_CustomResource: override_resource})
    loop.execute(_Request(message="hello"), resources=override_resources)

    assert adapter._last_resources is override_resources


def test_run_passes_resources() -> None:
    """MainLoopRequest resources are passed to adapter via run()."""
    mailbox: _TestMailbox = InMemoryMailbox()
    resource = _CustomResource(name="request-resource")
    resources = ResourceRegistry.build({_CustomResource: resource})
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    request = MainLoopRequest(
        request=_Request(message="hello"),
        resources=resources,
    )
    reply = mailbox.send_expecting_reply(request)

    loop.run(max_iterations=1)
    _ = reply.wait(timeout=1)

    assert adapter._last_resources is resources


def test_run_resources_overrides_config() -> None:
    """MainLoopRequest resources override config resources via run()."""
    mailbox: _TestMailbox = InMemoryMailbox()
    config_resource = _CustomResource(name="config-resource")
    config_resources = ResourceRegistry.build({_CustomResource: config_resource})
    config = MainLoopConfig(resources=config_resources)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox, config=config)

    override_resource = _CustomResource(name="override-resource")
    override_resources = ResourceRegistry.build({_CustomResource: override_resource})
    request = MainLoopRequest(
        request=_Request(message="hello"),
        resources=override_resources,
    )
    reply = mailbox.send_expecting_reply(request)

    loop.run(max_iterations=1)
    _ = reply.wait(timeout=1)

    assert adapter._last_resources is override_resources


def test_same_resources_used_across_visibility_retries() -> None:
    """Same resources are passed across visibility expansion retries."""
    mailbox: _TestMailbox = InMemoryMailbox()
    resource = _CustomResource(name="persistent-resource")
    resources = ResourceRegistry.build({_CustomResource: resource})
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("section1",): SectionVisibility.FULL},
        {("section2",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    loop.execute(_Request(message="hello"), resources=resources)

    # Called 3 times: 2 visibility expansions + 1 success
    assert adapter._call_count == 3
    # Same resources should be used for all calls
    assert len(adapter._resources_list) == 3
    assert all(r is resources for r in adapter._resources_list)


def test_no_resources_when_not_set() -> None:
    """No resources are passed when not configured."""
    mailbox: _TestMailbox = InMemoryMailbox()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=mailbox)

    loop.execute(_Request(message="hello"))

    assert adapter._last_resources is None
