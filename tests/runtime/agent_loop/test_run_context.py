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

"""Tests for AgentLoop RunContext and worker ID behavior."""

from __future__ import annotations

from weakincentives.runtime.agent_loop import (
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.mailbox import InMemoryMailbox
from weakincentives.runtime.run_context import RunContext

from .conftest import (
    MockAdapter,
    SampleLoop,
    SampleOutput,
    SampleRequest,
)

# =============================================================================
# Worker ID Tests
# =============================================================================


def test_loop_worker_id_property() -> None:
    """AgentLoop exposes worker_id property."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(
            adapter=adapter, requests=requests, worker_id="test-worker-42"
        )
        assert loop.worker_id == "test-worker-42"
    finally:
        requests.close()
        results.close()


# =============================================================================
# RunContext Tests
# =============================================================================


def test_loop_includes_run_context_in_result() -> None:
    """AgentLoop result includes RunContext."""
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests, worker_id="worker-1")

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
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
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter()
        loop = SampleLoop(adapter=adapter, requests=requests, worker_id="worker-2")

        # Include run_context with trace/span IDs in request
        input_run_ctx = RunContext(
            trace_id="trace-abc-123",
            span_id="span-xyz-456",
        )
        request = AgentLoopRequest(
            request=SampleRequest(message="hello"),
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
    results: InMemoryMailbox[AgentLoopResult[SampleOutput], None] = InMemoryMailbox(
        name="results"
    )
    requests: InMemoryMailbox[
        AgentLoopRequest[SampleRequest], AgentLoopResult[SampleOutput]
    ] = InMemoryMailbox(name="requests")
    try:
        adapter = MockAdapter(error=RuntimeError("adapter failure"))
        loop = SampleLoop(adapter=adapter, requests=requests, worker_id="worker-err")

        request = AgentLoopRequest(request=SampleRequest(message="hello"))
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
