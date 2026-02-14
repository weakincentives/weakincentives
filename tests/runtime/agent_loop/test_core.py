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

"""Tests for core AgentLoop processing behavior."""

from __future__ import annotations

from collections.abc import Mapping

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.budget import Budget
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    SectionPath,
    SectionVisibility,
)
from weakincentives.runtime.agent_loop import (
    _MAX_VISIBILITY_RETRIES,
    AgentLoop,
    AgentLoopConfig,
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.mailbox import (
    FakeMailbox,
    InMemoryMailbox,
)
from weakincentives.runtime.session import Session

from .conftest import (
    MockAdapter,
    SampleLoop,
    SampleOutput,
    SampleParams,
    SampleRequest,
)


class _TransformingSampleLoop(AgentLoop[SampleRequest, SampleOutput]):
    """Test implementation that transforms output in finalize."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[SampleOutput],
        requests: InMemoryMailbox[
            AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
        ]
        | FakeMailbox[AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]],
        config: AgentLoopConfig | None = None,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, config=config)
        self._template = PromptTemplate[SampleOutput](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[SampleParams](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )

    def prepare(
        self,
        request: SampleRequest,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[SampleOutput], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(SampleParams(content=request.message))
        session = Session(tags={"loop": "test"})
        return prompt, session

    def finalize(
        self,
        prompt: Prompt[SampleOutput],
        session: Session,
        output: SampleOutput | None,
    ) -> SampleOutput | None:
        del prompt, session
        if output is None:
            return None
        # Transform the output by appending "-transformed" to result
        return SampleOutput(result=f"{output.result}-transformed")


class _SampleLoopNoFinalizeOverride(AgentLoop[SampleRequest, SampleOutput]):
    """Test implementation that doesn't override finalize."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[SampleOutput],
        requests: InMemoryMailbox[
            AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
        ]
        | FakeMailbox[AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]],
        config: AgentLoopConfig | None = None,
    ) -> None:
        super().__init__(adapter=adapter, requests=requests, config=config)
        self._template = PromptTemplate[SampleOutput](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[SampleParams](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )

    def prepare(
        self,
        request: SampleRequest,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[SampleOutput], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(SampleParams(content=request.message))
        session = Session(tags={"loop": "test"})
        return prompt, session


# =============================================================================
# AgentLoop Tests
# =============================================================================


def test_loop_processes_request() -> None:
    """AgentLoop processes request from mailbox."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        # Send request with reply_to
        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        # Run single iteration
        loop.run(max_iterations=1, wait_time_seconds=0)

        # Check response
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.request_id == request.request_id
        assert msgs[0].body.output == SampleOutput(result="success")
        assert msgs[0].body.error is None
        assert msgs[0].body.success is True
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_sends_error_on_failure() -> None:
    """AgentLoop sends error result on adapter failure."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter(error=RuntimeError("adapter failure"))
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
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
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be acknowledged (gone from queue)
        assert requests.approximate_count() == 0
    finally:
        requests.close()
        results.close()


def test_loop_calls_finalize() -> None:
    """AgentLoop calls finalize after successful processing."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert loop.finalize_called
    finally:
        requests.close()
        results.close()


def test_loop_finalize_transforms_output() -> None:
    """AgentLoop uses transformed output from finalize."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = _TransformingSampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        # Output should be transformed by finalize
        assert msgs[0].body.output == SampleOutput(result="success-transformed")
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()


def test_loop_respects_max_iterations() -> None:
    """AgentLoop respects max_iterations limit."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        for i in range(5):
            requests.send(
                AgentLoopRequest(request=SampleRequest(message=f"msg-{i}")),
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
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL},
        ]
        adapter = MockAdapter(visibility_requests=visibility_requests)
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
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
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        budget = Budget(max_total_tokens=1000)
        config = AgentLoopConfig(budget=budget)
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        assert adapter._last_budget_tracker is not None
        assert adapter._last_budget_tracker.budget is budget
    finally:
        requests.close()
        results.close()


def test_loop_request_overrides_config() -> None:
    """AgentLoop uses request budget/deadline over config."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        config_budget = Budget(max_total_tokens=1000)
        config = AgentLoopConfig(budget=config_budget)
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests, config=config)

        override_budget = Budget(max_total_tokens=2000)
        request = AgentLoopRequest(
            request=SampleRequest(message="hello"),
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
    results: FakeMailbox[AgentLoopResult[SampleOutput], None] = FakeMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
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
    results: FakeMailbox[AgentLoopResult[SampleOutput], None] = FakeMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        # Adapter that fails
        adapter = MockAdapter(error=RuntimeError("adapter failure"))
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        # Make error response send fail too
        from weakincentives.runtime.mailbox import MailboxConnectionError

        results.set_connection_error(MailboxConnectionError("connection lost"))

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Request should be nacked (still in queue for retry)
        assert requests.approximate_count() == 1
    finally:
        requests.close()


def test_loop_default_finalize() -> None:
    """AgentLoop default finalize does nothing."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = _SampleLoopNoFinalizeOverride(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
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


def test_visibility_expansion_retry_cap() -> None:
    """Exceeding _MAX_VISIBILITY_RETRIES raises PromptEvaluationError."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        # Create more visibility requests than the retry cap allows
        visibility_requests: list[Mapping[SectionPath, SectionVisibility]] = [
            {("section1",): SectionVisibility.FULL}
            for _ in range(_MAX_VISIBILITY_RETRIES + 1)
        ]
        adapter = MockAdapter(visibility_requests=visibility_requests)
        loop = SampleLoop(adapter=adapter, requests=requests)

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
        requests.send(request, reply_to=results)

        loop.run(max_iterations=1, wait_time_seconds=0)

        # Should fail with error message about retry cap
        msgs = results.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0].body.success is False
        assert msgs[0].body.error is not None
        assert "Visibility expansion retries exceeded" in msgs[0].body.error
        assert str(_MAX_VISIBILITY_RETRIES) in msgs[0].body.error
        msgs[0].acknowledge()
    finally:
        requests.close()
        results.close()
