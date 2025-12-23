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
)
from weakincentives.prompt.tool import ResourceRegistry
from weakincentives.runtime.mailbox import FakeMailbox, InMemoryMailbox
from weakincentives.runtime.main_loop import (
    MainLoop,
    MainLoopConfig,
    MainLoopRequest,
    MainLoopResult,
)
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
        requests: InMemoryMailbox[MainLoopRequest[_Request]]
        | FakeMailbox[MainLoopRequest[_Request]],
        responses: InMemoryMailbox[MainLoopResult[_Output]]
        | FakeMailbox[MainLoopResult[_Output]],
        config: MainLoopConfig | None = None,
    ) -> None:
        super().__init__(
            adapter=adapter, requests=requests, responses=responses, config=config
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

    def initialize(self, request: _Request) -> tuple[Prompt[_Output], Session]:
        prompt = Prompt(self._template).bind(_Params(content=request.message))
        session = Session(tags={"loop": "test"})
        self.session_created = session
        return prompt, session

    def finalize(self, prompt: Prompt[_Output], session: Session) -> None:
        del prompt
        self.finalize_called = True
        _ = session


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
# MainLoopResult Tests
# =============================================================================


def test_result_success_case() -> None:
    """MainLoopResult represents successful completion."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    session_id = UUID("87654321-4321-8765-4321-876543218765")
    output = _Output(result="success")
    result: MainLoopResult[_Output] = MainLoopResult(
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
    """MainLoopResult represents failure."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    result: MainLoopResult[_Output] = MainLoopResult(
        request_id=request_id,
        error="adapter failure",
    )
    assert result.request_id == request_id
    assert result.output is None
    assert result.error == "adapter failure"
    assert result.session_id is None
    assert result.success is False


def test_result_is_frozen() -> None:
    """MainLoopResult is immutable."""
    request_id = UUID("12345678-1234-5678-1234-567812345678")
    result: MainLoopResult[_Output] = MainLoopResult(
        request_id=request_id,
        output=_Output(result="success"),
    )
    with pytest.raises(AttributeError):
        result.output = None  # type: ignore[misc]


# =============================================================================
# MainLoop Tests
# =============================================================================


def test_loop_processes_request() -> None:
    """MainLoop processes request from mailbox."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        # Send request
        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        # Run single iteration
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check response
        msgs = responses.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.request_id == request.request_id
        assert msgs[0].body.output == _Output(result="success")
        assert msgs[0].body.error is None
        assert msgs[0].body.success is True
        msgs[0].acknowledge()
    finally:
        requests.close()
        responses.close()


def test_loop_sends_error_on_failure() -> None:
    """MainLoop sends error result on adapter failure."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        adapter = _MockAdapter(error=RuntimeError("adapter failure"))
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = responses.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.request_id == request.request_id
        assert msgs[0].body.output is None
        assert msgs[0].body.error == "adapter failure"
        assert msgs[0].body.success is False
        msgs[0].acknowledge()
    finally:
        requests.close()
        responses.close()


def test_loop_acknowledges_request() -> None:
    """MainLoop acknowledges processed request."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be acknowledged (gone from queue)
        assert requests.approximate_count() == 0
    finally:
        requests.close()
        responses.close()


def test_loop_calls_finalize() -> None:
    """MainLoop calls finalize after successful processing."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert loop.finalize_called
    finally:
        requests.close()
        responses.close()


def test_loop_respects_max_iterations() -> None:
    """MainLoop respects max_iterations limit."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        for i in range(5):
            requests.send(MainLoopRequest(request=_Request(message=f"msg-{i}")))

        # Only run 2 iterations
        loop.run(max_iterations=2, wait_time_seconds=0)

        # Some requests may still be pending (depending on batch size)
        # At least we should have some responses
        assert responses.approximate_count() >= 1
    finally:
        requests.close()
        responses.close()


def test_loop_handles_visibility_expansion() -> None:
    """MainLoop handles visibility expansion correctly."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Should succeed after visibility expansion
        msgs = responses.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        assert adapter._call_count == 2  # 1 visibility expansion + 1 success
        msgs[0].acknowledge()
    finally:
        requests.close()
        responses.close()


def test_loop_uses_config_budget() -> None:
    """MainLoop uses budget from config."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        budget = Budget(max_total_tokens=1000)
        config = MainLoopConfig(budget=budget)
        adapter = _MockAdapter()
        loop = _TestLoop(
            adapter=adapter, requests=requests, responses=responses, config=config
        )

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_budget_tracker is not None
        assert adapter._last_budget_tracker.budget is budget
    finally:
        requests.close()
        responses.close()


def test_loop_request_overrides_config() -> None:
    """MainLoop uses request budget/deadline over config."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        config_budget = Budget(max_total_tokens=1000)
        config = MainLoopConfig(budget=config_budget)
        adapter = _MockAdapter()
        loop = _TestLoop(
            adapter=adapter, requests=requests, responses=responses, config=config
        )

        override_budget = Budget(max_total_tokens=2000)
        request = MainLoopRequest(
            request=_Request(message="hello"),
            budget=override_budget,
        )
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_budget_tracker is not None
        assert adapter._last_budget_tracker.budget is override_budget
    finally:
        requests.close()
        responses.close()


def test_loop_nacks_on_response_send_failure() -> None:
    """MainLoop nacks request when response send fails."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: FakeMailbox[MainLoopResult[_Output]] = FakeMailbox(name="responses")
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        # Make response send fail
        from weakincentives.runtime.mailbox import MailboxConnectionError

        responses.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be nacked (still in queue for retry)
        # Note: it's in invisible state after receive, so we need to wait
        # or check approximate_count
        assert requests.approximate_count() == 1
    finally:
        requests.close()


def test_loop_nacks_on_error_response_send_failure() -> None:
    """MainLoop nacks request when error response send fails."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: FakeMailbox[MainLoopResult[_Output]] = FakeMailbox(name="responses")
    try:
        # Adapter that fails
        adapter = _MockAdapter(error=RuntimeError("adapter failure"))
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        # Make error response send fail too
        from weakincentives.runtime.mailbox import MailboxConnectionError

        responses.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be nacked (still in queue for retry)
        assert requests.approximate_count() == 1
    finally:
        requests.close()


class _TestLoopNoFinalizeOverride(MainLoop[_Request, _Output]):
    """Test implementation that doesn't override finalize."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[_Output],
        requests: InMemoryMailbox[MainLoopRequest[_Request]]
        | FakeMailbox[MainLoopRequest[_Request]],
        responses: InMemoryMailbox[MainLoopResult[_Output]]
        | FakeMailbox[MainLoopResult[_Output]],
        config: MainLoopConfig | None = None,
    ) -> None:
        super().__init__(
            adapter=adapter, requests=requests, responses=responses, config=config
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

    def initialize(self, request: _Request) -> tuple[Prompt[_Output], Session]:
        prompt = Prompt(self._template).bind(_Params(content=request.message))
        session = Session(tags={"loop": "test"})
        return prompt, session


def test_loop_default_finalize() -> None:
    """MainLoop default finalize does nothing."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoopNoFinalizeOverride(
            adapter=adapter, requests=requests, responses=responses
        )

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        # Run should succeed even without finalize override
        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = responses.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is True
        msgs[0].acknowledge()
    finally:
        requests.close()
        responses.close()


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


def test_loop_passes_resources_from_config() -> None:
    """MainLoop passes config resources to adapter."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        resource = _CustomResource(name="config-resource")
        resources = ResourceRegistry.build({_CustomResource: resource})
        config = MainLoopConfig(resources=resources)
        adapter = _MockAdapter()
        loop = _TestLoop(
            adapter=adapter, requests=requests, responses=responses, config=config
        )

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_resources is resources
    finally:
        requests.close()
        responses.close()


def test_loop_request_resources_overrides_config() -> None:
    """MainLoop request resources override config resources."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        config_resource = _CustomResource(name="config-resource")
        config_resources = ResourceRegistry.build({_CustomResource: config_resource})
        config = MainLoopConfig(resources=config_resources)
        adapter = _MockAdapter()
        loop = _TestLoop(
            adapter=adapter, requests=requests, responses=responses, config=config
        )

        override_resource = _CustomResource(name="override-resource")
        override_resources = ResourceRegistry.build(
            {_CustomResource: override_resource}
        )
        request = MainLoopRequest(
            request=_Request(message="hello"),
            resources=override_resources,
        )
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_resources is override_resources
    finally:
        requests.close()
        responses.close()


def test_same_resources_used_across_visibility_retries() -> None:
    """Same resources are passed across visibility expansion retries."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        resource = _CustomResource(name="persistent-resource")
        resources = ResourceRegistry.build({_CustomResource: resource})
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(
            request=_Request(message="hello"),
            resources=resources,
        )
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Called 3 times: 2 visibility expansions + 1 success
        assert adapter._call_count == 3
        # Same resources should be used for all calls
        assert len(adapter._resources_list) == 3
        assert all(r is resources for r in adapter._resources_list)
    finally:
        requests.close()
        responses.close()


def test_no_resources_when_not_set() -> None:
    """No resources are passed when not configured."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_resources is None
    finally:
        requests.close()
        responses.close()


# =============================================================================
# Visibility Override Accumulation Tests
# =============================================================================


def test_visibility_overrides_accumulate_in_session() -> None:
    """MainLoop accumulates visibility overrides in session state."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check session state has accumulated overrides
        assert loop.session_created is not None
        overrides = loop.session_created[VisibilityOverrides].latest()
        assert overrides is not None
        assert overrides.get(("section1",)) == SectionVisibility.FULL
        assert overrides.get(("section2",)) == SectionVisibility.FULL
    finally:
        requests.close()
        responses.close()


def test_same_budget_tracker_used_across_visibility_retries() -> None:
    """Same BudgetTracker is used across visibility expansion retries."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        budget = Budget(max_total_tokens=1000)
        config = MainLoopConfig(budget=budget)
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
            {("section2",): SectionVisibility.FULL},
        ]
        adapter = _MockAdapter(visibility_requests=visibility_requests)
        loop = _TestLoop(
            adapter=adapter, requests=requests, responses=responses, config=config
        )

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

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
        responses.close()


def test_no_budget_tracker_when_no_budget() -> None:
    """No BudgetTracker is created when no budget is set."""
    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    try:
        adapter = _MockAdapter()
        loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

        request = MainLoopRequest(request=_Request(message="hello"))
        requests.send(request)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_budget_tracker is None
    finally:
        requests.close()
        responses.close()


# =============================================================================
# Mailbox Close Tests
# =============================================================================


def test_loop_exits_when_mailbox_closed() -> None:
    """MainLoop.run() exits when requests mailbox is closed."""
    import threading

    requests: InMemoryMailbox[MainLoopRequest[_Request]] = InMemoryMailbox(
        name="requests"
    )
    responses: InMemoryMailbox[MainLoopResult[_Output]] = InMemoryMailbox(
        name="responses"
    )
    adapter = _MockAdapter()
    loop = _TestLoop(adapter=adapter, requests=requests, responses=responses)

    exited = []

    def run_loop() -> None:
        # Would run forever with max_iterations=None
        loop.run(max_iterations=None, wait_time_seconds=1)
        exited.append(True)

    thread = threading.Thread(target=run_loop)
    thread.start()

    # Close the mailbox - should cause loop to exit
    requests.close()
    responses.close()

    # Thread should exit quickly
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(exited) == 1
