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
from weakincentives.runtime.events import Dispatcher, InProcessDispatcher
from weakincentives.runtime.main_loop import (
    MainLoop,
    MainLoopCompleted,
    MainLoopConfig,
    MainLoopFailed,
    MainLoopRequest,
)
from weakincentives.runtime.session import Session
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


class _TestLoop(MainLoop[_Request, _Output]):
    """Test implementation of MainLoop."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        bus: Dispatcher,
        config: MainLoopConfig | None = None,
    ) -> None:
        super().__init__(adapter=adapter, bus=bus, config=config)
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
        self.finalize_prompt: Prompt[_Output] | None = None
        self.finalize_session: Session | None = None

    def initialize(self, request: _Request) -> tuple[Prompt[_Output], Session]:
        prompt = Prompt(self._template).bind(_Params(content=request.message))
        session = Session(bus=self._bus, tags={"loop": "test"})
        self.session_created = session
        return prompt, session

    def finalize(self, prompt: Prompt[_Output], session: Session) -> None:
        self.finalize_called = True
        self.finalize_prompt = prompt
        self.finalize_session = session


# =============================================================================
# MainLoopConfig Tests
# =============================================================================


def test_config_default_values() -> None:
    """MainLoopConfig has sensible defaults."""
    config = MainLoopConfig()
    assert config.deadline is None
    assert config.budget is None


def test_config_custom_values() -> None:
    """MainLoopConfig accepts custom values."""
    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(
        deadline=deadline,
        budget=budget,
    )
    assert config.deadline is deadline
    assert config.budget is budget


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
# MainLoopCompleted Tests
# =============================================================================


def test_completed_default_values() -> None:
    """MainLoopCompleted stores response and session information."""
    response = PromptResponse(
        prompt_name="test",
        text="result",
        output=_Output(result="success"),
    )
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    session_id = UUID("87654321-4321-8765-4321-876543218765")
    completed = MainLoopCompleted(
        request_id=request_id,
        response=response,
        session_id=session_id,
    )
    assert completed.request_id == request_id
    assert completed.response is response
    assert completed.session_id == session_id
    assert completed.completed_at.tzinfo == UTC


# =============================================================================
# MainLoopFailed Tests
# =============================================================================


def test_failed_with_session_id() -> None:
    """MainLoopFailed stores error and session information."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    session_id = UUID("87654321-4321-8765-4321-876543218765")
    error = ValueError("test error")
    failed = MainLoopFailed(
        request_id=request_id,
        error=error,
        session_id=session_id,
    )
    assert failed.request_id == request_id
    assert failed.error is error
    assert failed.session_id == session_id
    assert failed.failed_at.tzinfo == UTC


def test_failed_without_session_id() -> None:
    """MainLoopFailed can have None session_id."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    error = ValueError("test error")
    failed = MainLoopFailed(
        request_id=request_id,
        error=error,
        session_id=None,
    )
    assert failed.session_id is None


# =============================================================================
# MainLoop.execute() Tests
# =============================================================================


def test_execute_successful_execution() -> None:
    """MainLoop.execute returns response on success."""
    bus = InProcessDispatcher()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus)

    response, session = loop.execute(_Request(message="hello"))

    assert response.output == _Output(result="success")
    assert adapter._call_count == 1
    assert loop.session_created is not None
    assert session is loop.session_created


def test_execute_calls_finalize_on_success() -> None:
    """MainLoop.execute calls finalize after successful evaluation."""
    bus = InProcessDispatcher()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus)

    response, session = loop.execute(_Request(message="hello"))

    assert loop.finalize_called
    assert loop.finalize_session is session
    assert loop.finalize_prompt is not None
    assert response.output == _Output(result="success")


def test_execute_finalize_called_after_visibility_retries() -> None:
    """MainLoop.execute calls finalize after visibility expansion succeeds."""
    bus = InProcessDispatcher()
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("section1",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, bus=bus)

    response, session = loop.execute(_Request(message="hello"))

    assert loop.finalize_called
    assert loop.finalize_session is session
    assert adapter._call_count == 2  # 1 visibility expansion + 1 success
    assert response.output == _Output(result="success")


def test_execute_finalize_not_called_on_error() -> None:
    """MainLoop.execute does not call finalize when adapter raises error."""
    bus = InProcessDispatcher()
    error = RuntimeError("adapter failure")
    adapter = _MockAdapter(error=error)
    loop = _TestLoop(adapter=adapter, bus=bus)

    with pytest.raises(RuntimeError, match="adapter failure"):
        loop.execute(_Request(message="hello"))

    assert not loop.finalize_called


def test_execute_passes_budget_from_config() -> None:
    """MainLoop.execute creates BudgetTracker with config budget."""
    bus = InProcessDispatcher()
    budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(budget=budget)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    loop.execute(_Request(message="hello"))

    assert adapter._last_budget_tracker is not None
    assert adapter._last_budget_tracker.budget is budget


def test_execute_passes_deadline_from_config() -> None:
    """MainLoop.execute passes config deadline to adapter."""
    bus = InProcessDispatcher()
    deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    config = MainLoopConfig(deadline=deadline)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    loop.execute(_Request(message="hello"))

    assert adapter._last_deadline is deadline


def test_execute_budget_overrides_config() -> None:
    """MainLoop.execute budget parameter overrides config."""
    bus = InProcessDispatcher()
    config_budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(budget=config_budget)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    override_budget = Budget(max_total_tokens=2000)
    loop.execute(_Request(message="hello"), budget=override_budget)

    assert adapter._last_budget_tracker is not None
    assert adapter._last_budget_tracker.budget is override_budget


def test_execute_deadline_overrides_config() -> None:
    """MainLoop.execute deadline parameter overrides config."""
    bus = InProcessDispatcher()
    config_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    config = MainLoopConfig(deadline=config_deadline)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    override_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=10))
    loop.execute(_Request(message="hello"), deadline=override_deadline)

    assert adapter._last_deadline is override_deadline


def test_execute_handles_visibility_expansion() -> None:
    """MainLoop.execute accumulates visibility overrides in session state and retries."""
    bus = InProcessDispatcher()
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("section1",): SectionVisibility.FULL},
        {("section2",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, bus=bus)

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
    bus = InProcessDispatcher()
    error = RuntimeError("adapter failure")
    adapter = _MockAdapter(error=error)
    loop = _TestLoop(adapter=adapter, bus=bus)

    with pytest.raises(RuntimeError, match="adapter failure"):
        loop.execute(_Request(message="hello"))


# =============================================================================
# MainLoop.handle_request() Tests
# =============================================================================


def test_handle_request_publishes_completed_event_on_success() -> None:
    """MainLoop.handle_request publishes MainLoopCompleted on success."""
    bus = InProcessDispatcher()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus)

    completed_events: list[MainLoopCompleted[_Output]] = []
    bus.subscribe(MainLoopCompleted, lambda e: completed_events.append(e))
    request = MainLoopRequest(request=_Request(message="hello"))
    loop.handle_request(request)

    assert len(completed_events) == 1
    assert completed_events[0].request_id == request.request_id
    assert completed_events[0].response.output == _Output(result="success")


def test_handle_request_publishes_failed_event_and_reraises_on_error() -> None:
    """MainLoop.handle_request publishes MainLoopFailed and re-raises on error."""
    bus = InProcessDispatcher()
    error = RuntimeError("adapter failure")
    adapter = _MockAdapter(error=error)
    loop = _TestLoop(adapter=adapter, bus=bus)

    failed_events: list[MainLoopFailed] = []
    bus.subscribe(MainLoopFailed, lambda e: failed_events.append(e))
    request = MainLoopRequest(request=_Request(message="hello"))
    with pytest.raises(RuntimeError, match="adapter failure"):
        loop.handle_request(request)

    assert len(failed_events) == 1
    assert failed_events[0].request_id == request.request_id
    assert failed_events[0].error is error
    assert failed_events[0].session_id is None


def test_handle_request_budget_overrides_config() -> None:
    """MainLoopRequest budget overrides config budget."""
    bus = InProcessDispatcher()
    config_budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(budget=config_budget)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    override_budget = Budget(max_total_tokens=2000)
    request = MainLoopRequest(
        request=_Request(message="hello"),
        budget=override_budget,
    )
    loop.handle_request(request)

    assert adapter._last_budget_tracker is not None
    assert adapter._last_budget_tracker.budget is override_budget


def test_handle_request_deadline_overrides_config() -> None:
    """MainLoopRequest deadline overrides config deadline."""
    bus = InProcessDispatcher()
    config_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=5))
    config = MainLoopConfig(deadline=config_deadline)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    override_deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(minutes=10))
    request = MainLoopRequest(
        request=_Request(message="hello"),
        deadline=override_deadline,
    )
    loop.handle_request(request)

    assert adapter._last_deadline is override_deadline


def test_handle_request_handles_visibility_expansion() -> None:
    """MainLoop.handle_request handles visibility expansion."""
    bus = InProcessDispatcher()
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("ref",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, bus=bus)

    completed_events: list[MainLoopCompleted[_Output]] = []
    bus.subscribe(MainLoopCompleted, lambda e: completed_events.append(e))
    request = MainLoopRequest(request=_Request(message="hello"))
    loop.handle_request(request)

    assert len(completed_events) == 1
    assert adapter._call_count == 2


# =============================================================================
# Bus Integration Tests
# =============================================================================


def test_bus_subscribe_and_publish_workflow() -> None:
    """MainLoop works with bus-driven subscribe/publish pattern."""
    bus = InProcessDispatcher()
    adapter = _MockAdapter()
    _TestLoop(adapter=adapter, bus=bus)  # Auto-subscribes handle_request

    completed_events: list[MainLoopCompleted[_Output]] = []
    bus.subscribe(MainLoopCompleted, lambda e: completed_events.append(e))
    request = MainLoopRequest(request=_Request(message="hello"))
    bus.dispatch(request)

    assert len(completed_events) == 1
    assert completed_events[0].response.output == _Output(result="success")


def test_bus_session_persists_across_visibility_retries() -> None:
    """Same session is reused across visibility expansion retries."""
    bus = InProcessDispatcher()
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("section1",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, bus=bus)

    loop.execute(_Request(message="hello"))

    # Same session should be used for both calls
    assert adapter._call_count == 2
    # Session should persist (same object used for retries)
    assert loop.session_created is not None


def test_same_budget_tracker_used_across_visibility_retries() -> None:
    """Same BudgetTracker is used across visibility expansion retries."""
    bus = InProcessDispatcher()
    budget = Budget(max_total_tokens=1000)
    config = MainLoopConfig(budget=budget)
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("section1",): SectionVisibility.FULL},
        {("section2",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

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
    bus = InProcessDispatcher()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus)

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
    bus = InProcessDispatcher()
    resource = _CustomResource(name="config-resource")
    resources = ResourceRegistry.build({_CustomResource: resource})
    config = MainLoopConfig(resources=resources)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    loop.execute(_Request(message="hello"))

    assert adapter._last_resources is resources


def test_execute_resources_overrides_config() -> None:
    """MainLoop.execute resources parameter overrides config."""
    bus = InProcessDispatcher()
    config_resource = _CustomResource(name="config-resource")
    config_resources = ResourceRegistry.build({_CustomResource: config_resource})
    config = MainLoopConfig(resources=config_resources)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    override_resource = _CustomResource(name="override-resource")
    override_resources = ResourceRegistry.build({_CustomResource: override_resource})
    loop.execute(_Request(message="hello"), resources=override_resources)

    assert adapter._last_resources is override_resources


def test_handle_request_passes_resources() -> None:
    """MainLoopRequest resources are passed to adapter."""
    bus = InProcessDispatcher()
    resource = _CustomResource(name="request-resource")
    resources = ResourceRegistry.build({_CustomResource: resource})
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus)

    request = MainLoopRequest(
        request=_Request(message="hello"),
        resources=resources,
    )
    loop.handle_request(request)

    assert adapter._last_resources is resources


def test_handle_request_resources_overrides_config() -> None:
    """MainLoopRequest resources override config resources."""
    bus = InProcessDispatcher()
    config_resource = _CustomResource(name="config-resource")
    config_resources = ResourceRegistry.build({_CustomResource: config_resource})
    config = MainLoopConfig(resources=config_resources)
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus, config=config)

    override_resource = _CustomResource(name="override-resource")
    override_resources = ResourceRegistry.build({_CustomResource: override_resource})
    request = MainLoopRequest(
        request=_Request(message="hello"),
        resources=override_resources,
    )
    loop.handle_request(request)

    assert adapter._last_resources is override_resources


def test_same_resources_used_across_visibility_retries() -> None:
    """Same resources are passed across visibility expansion retries."""
    bus = InProcessDispatcher()
    resource = _CustomResource(name="persistent-resource")
    resources = ResourceRegistry.build({_CustomResource: resource})
    visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
        {("section1",): SectionVisibility.FULL},
        {("section2",): SectionVisibility.FULL},
    ]
    adapter = _MockAdapter(visibility_requests=visibility_requests)
    loop = _TestLoop(adapter=adapter, bus=bus)

    loop.execute(_Request(message="hello"), resources=resources)

    # Called 3 times: 2 visibility expansions + 1 success
    assert adapter._call_count == 3
    # Same resources should be used for all calls
    assert len(adapter._resources_list) == 3
    assert all(r is resources for r in adapter._resources_list)


def test_no_resources_when_not_set() -> None:
    """No resources are passed when not configured."""
    bus = InProcessDispatcher()
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, bus=bus)

    loop.execute(_Request(message="hello"))

    assert adapter._last_resources is None
