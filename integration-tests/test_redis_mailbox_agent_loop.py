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

"""Integration tests for RedisMailbox with AgentLoop request/response flow.

These tests verify that the complete serialization/deserialization flow works
correctly for typical AgentLoop request/response patterns using RedisMailbox
with generic type parameters for deserialization.

Tests cover:
1. AgentLoopRequest[UserRequestT] serialization/deserialization
2. AgentLoopResult[OutputT] serialization/deserialization
3. Full request/response round-trip through Redis
4. Nested dataclass serialization
5. Various field types (UUID, datetime, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from redis_utils import is_redis_available, redis_standalone, skip_if_no_redis

from weakincentives.contrib.mailbox import RedisMailbox
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.runtime.agent_loop_types import AgentLoopRequest, AgentLoopResult
from weakincentives.runtime.run_context import RunContext

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Domain Types
# =============================================================================


@dataclass(slots=True, frozen=True)
class TaskRequest:
    """Sample request type for testing AgentLoop flow."""

    task_id: str
    payload: str
    priority: int = 0


@dataclass(slots=True, frozen=True)
class TaskResult:
    """Sample result type for testing AgentLoop flow."""

    task_id: str
    status: str
    output: str | None = None


@FrozenDataclass()
class ComplexRequest:
    """Complex request with nested types to test deep serialization."""

    name: str
    items: list[str]
    metadata: dict[str, str] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class ComplexResult:
    """Complex result with nested types to test deep serialization."""

    request_name: str
    processed_items: list[str]
    summary: dict[str, int] | None = None


# Skip all tests in this module if Redis is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.redis,
    pytest.mark.skipif(not is_redis_available(), reason=skip_if_no_redis()),
    pytest.mark.timeout(60),
]


# =============================================================================
# AgentLoopRequest Serialization Tests
# =============================================================================


@pytest.mark.redis_standalone
class TestAgentLoopRequestSerialization:
    """Tests for AgentLoopRequest serialization/deserialization via RedisMailbox."""

    def test_simple_request_round_trip(self) -> None:
        """AgentLoopRequest with simple UserRequestT serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-agent-request",
                client=client,
            )
            try:
                # Create request with all fields
                original = AgentLoopRequest(
                    request=TaskRequest(task_id="task-123", payload="test data"),
                )

                # Send through Redis (serializes)
                mailbox.send(original)

                # Receive from Redis (deserializes)
                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 1
                received = messages[0].body

                # Verify all fields
                assert received.request.task_id == original.request.task_id
                assert received.request.payload == original.request.payload
                assert received.request.priority == original.request.priority
                assert received.request_id == original.request_id
                assert received.budget is None
                assert received.deadline is None

                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_request_with_priority_field(self) -> None:
        """AgentLoopRequest with non-default fields serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-priority-request",
                client=client,
            )
            try:
                original = AgentLoopRequest(
                    request=TaskRequest(
                        task_id="high-priority",
                        payload="urgent",
                        priority=10,
                    ),
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.request.priority == 10
                assert received.request.task_id == "high-priority"
                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_request_with_run_context(self) -> None:
        """AgentLoopRequest with RunContext serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-run-context-request",
                client=client,
            )
            try:
                run_context = RunContext(
                    trace_id=uuid4(),
                    span_id=uuid4(),
                )
                original = AgentLoopRequest(
                    request=TaskRequest(task_id="ctx-task", payload="with context"),
                    run_context=run_context,
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.run_context is not None
                assert received.run_context.trace_id == run_context.trace_id
                assert received.run_context.span_id == run_context.span_id
                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_complex_request_with_nested_types(self) -> None:
        """AgentLoopRequest with complex nested UserRequestT serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[ComplexRequest], None](
                name="test-complex-request",
                client=client,
            )
            try:
                original = AgentLoopRequest(
                    request=ComplexRequest(
                        name="batch-job",
                        items=["item1", "item2", "item3"],
                        metadata={"source": "test", "version": "1.0"},
                    ),
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.request.name == "batch-job"
                assert received.request.items == ["item1", "item2", "item3"]
                assert received.request.metadata == {"source": "test", "version": "1.0"}
                messages[0].acknowledge()
            finally:
                mailbox.close()


# =============================================================================
# AgentLoopResult Serialization Tests
# =============================================================================


@pytest.mark.redis_standalone
class TestAgentLoopResultSerialization:
    """Tests for AgentLoopResult serialization/deserialization via RedisMailbox."""

    def test_success_result_round_trip(self) -> None:
        """AgentLoopResult success case serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopResult[TaskResult], None](
                name="test-agent-result",
                client=client,
            )
            try:
                request_id = uuid4()
                session_id = uuid4()
                original: AgentLoopResult[TaskResult] = AgentLoopResult(
                    request_id=request_id,
                    output=TaskResult(
                        task_id="task-123",
                        status="completed",
                        output="success output",
                    ),
                    session_id=session_id,
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.request_id == request_id
                assert received.session_id == session_id
                assert received.error is None
                assert received.success is True
                assert received.output is not None
                assert received.output.task_id == "task-123"
                assert received.output.status == "completed"
                assert received.output.output == "success output"
                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_error_result_round_trip(self) -> None:
        """AgentLoopResult error case serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopResult[TaskResult], None](
                name="test-agent-error-result",
                client=client,
            )
            try:
                request_id = uuid4()
                original: AgentLoopResult[TaskResult] = AgentLoopResult(
                    request_id=request_id,
                    error="Processing failed: timeout",
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.request_id == request_id
                assert received.output is None
                assert received.error == "Processing failed: timeout"
                assert received.success is False
                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_result_with_run_context(self) -> None:
        """AgentLoopResult with RunContext serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopResult[TaskResult], None](
                name="test-result-run-context",
                client=client,
            )
            try:
                request_id = uuid4()
                session_id = uuid4()
                run_context = RunContext(
                    trace_id=uuid4(),
                    span_id=uuid4(),
                    session_id=session_id,
                )
                original: AgentLoopResult[TaskResult] = AgentLoopResult(
                    request_id=request_id,
                    output=TaskResult(task_id="ctx-task", status="done"),
                    session_id=session_id,
                    run_context=run_context,
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.run_context is not None
                assert received.run_context.trace_id == run_context.trace_id
                assert received.run_context.session_id == session_id
                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_complex_result_with_nested_types(self) -> None:
        """AgentLoopResult with complex nested OutputT serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopResult[ComplexResult], None](
                name="test-complex-result",
                client=client,
            )
            try:
                request_id = uuid4()
                original: AgentLoopResult[ComplexResult] = AgentLoopResult(
                    request_id=request_id,
                    output=ComplexResult(
                        request_name="batch-job",
                        processed_items=["result1", "result2"],
                        summary={"total": 2, "success": 2},
                    ),
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.output is not None
                assert received.output.request_name == "batch-job"
                assert received.output.processed_items == ["result1", "result2"]
                assert received.output.summary == {"total": 2, "success": 2}
                messages[0].acknowledge()
            finally:
                mailbox.close()


# =============================================================================
# Full Request/Response Flow Tests
# =============================================================================


@pytest.mark.redis_standalone
class TestRequestResponseFlow:
    """Tests for complete request/response flow through Redis."""

    def test_request_response_round_trip(self) -> None:
        """Full request/response flow works with typed mailboxes."""
        with redis_standalone() as client:
            # Create typed mailboxes for requests and responses
            requests = RedisMailbox[
                AgentLoopRequest[TaskRequest], AgentLoopResult[TaskResult]
            ](
                name="test-flow-requests",
                client=client,
            )
            responses = RedisMailbox[AgentLoopResult[TaskResult], None](
                name="test-flow-responses",
                client=client,
            )

            try:
                # Client sends request with reply_to
                request = AgentLoopRequest(
                    request=TaskRequest(
                        task_id="flow-task",
                        payload="process this",
                    ),
                )
                requests.send(request, reply_to=responses)

                # Worker receives request
                msgs = requests.receive(max_messages=1)
                assert len(msgs) == 1
                received_request = msgs[0].body
                assert received_request.request.task_id == "flow-task"
                assert received_request.request.payload == "process this"

                # Worker sends response via reply
                result: AgentLoopResult[TaskResult] = AgentLoopResult(
                    request_id=received_request.request_id,
                    output=TaskResult(
                        task_id=received_request.request.task_id,
                        status="completed",
                        output="processed result",
                    ),
                )
                msgs[0].reply(result)
                msgs[0].acknowledge()

                # Client receives response
                response_msgs = responses.receive(max_messages=1)
                assert len(response_msgs) == 1
                received_result = response_msgs[0].body
                assert received_result.request_id == request.request_id
                assert received_result.output is not None
                assert received_result.output.status == "completed"
                response_msgs[0].acknowledge()

            finally:
                requests.close()
                responses.close()

    def test_multiple_requests_fifo_order(self) -> None:
        """Multiple requests are processed in FIFO order."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-fifo-requests",
                client=client,
            )

            try:
                # Send multiple requests
                request_ids = []
                for i in range(3):
                    req = AgentLoopRequest(
                        request=TaskRequest(task_id=f"task-{i}", payload=f"data-{i}"),
                    )
                    request_ids.append(req.request_id)
                    mailbox.send(req)

                # Receive all
                msgs = mailbox.receive(max_messages=3)
                assert len(msgs) == 3

                # Verify FIFO order
                for i, msg in enumerate(msgs):
                    assert msg.body.request.task_id == f"task-{i}"
                    assert msg.body.request_id == request_ids[i]
                    msg.acknowledge()

            finally:
                mailbox.close()

    def test_error_response_flow(self) -> None:
        """Error responses are serialized and deserialized correctly."""
        with redis_standalone() as client:
            requests = RedisMailbox[
                AgentLoopRequest[TaskRequest], AgentLoopResult[TaskResult]
            ](
                name="test-error-flow-requests",
                client=client,
            )
            responses = RedisMailbox[AgentLoopResult[TaskResult], None](
                name="test-error-flow-responses",
                client=client,
            )

            try:
                request = AgentLoopRequest(
                    request=TaskRequest(task_id="fail-task", payload="will fail"),
                )
                requests.send(request, reply_to=responses)

                # Worker receives and sends error response
                msgs = requests.receive(max_messages=1)
                error_result: AgentLoopResult[TaskResult] = AgentLoopResult(
                    request_id=msgs[0].body.request_id,
                    error="Task processing failed: validation error",
                )
                msgs[0].reply(error_result)
                msgs[0].acknowledge()

                # Client receives error response
                response_msgs = responses.receive(max_messages=1)
                received = response_msgs[0].body
                assert received.success is False
                assert received.error == "Task processing failed: validation error"
                assert received.output is None
                response_msgs[0].acknowledge()

            finally:
                requests.close()
                responses.close()


# =============================================================================
# Generic Type Parameter Tests
# =============================================================================


@pytest.mark.redis_standalone
class TestGenericTypeExtraction:
    """Tests verifying generic type parameters drive deserialization correctly."""

    def test_generic_type_extracts_at_runtime(self) -> None:
        """RedisMailbox[T, R] extracts T for deserialization."""
        with redis_standalone() as client:
            # Use generic syntax - this should extract AgentLoopRequest[TaskRequest]
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-generic-extract",
                client=client,
            )

            try:
                original = AgentLoopRequest(
                    request=TaskRequest(task_id="generic-test", payload="test"),
                )

                mailbox.send(original)
                msgs = mailbox.receive(max_messages=1)

                # Body should be correctly typed via generic type extraction
                received = msgs[0].body
                assert isinstance(received, AgentLoopRequest)
                assert isinstance(received.request, TaskRequest)
                assert received.request.task_id == "generic-test"
                msgs[0].acknowledge()

            finally:
                mailbox.close()

    def test_datetime_fields_serialize_correctly(self) -> None:
        """datetime fields are serialized as ISO format and deserialized back."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-datetime-serde",
                client=client,
            )

            try:
                # created_at is auto-set with UTC timezone
                original = AgentLoopRequest(
                    request=TaskRequest(task_id="dt-test", payload="test"),
                )

                mailbox.send(original)
                msgs = mailbox.receive(max_messages=1)
                received = msgs[0].body

                # datetime should be preserved
                assert received.created_at.tzinfo == UTC
                # Allow small time delta for test execution
                delta = abs((received.created_at - original.created_at).total_seconds())
                assert delta < 1.0
                msgs[0].acknowledge()

            finally:
                mailbox.close()

    def test_uuid_fields_serialize_correctly(self) -> None:
        """UUID fields are serialized and deserialized correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopResult[TaskResult], None](
                name="test-uuid-serde",
                client=client,
            )

            try:
                request_id = UUID("12345678-1234-5678-1234-567812345678")
                session_id = UUID("87654321-4321-8765-4321-876543218765")

                original: AgentLoopResult[TaskResult] = AgentLoopResult(
                    request_id=request_id,
                    session_id=session_id,
                    output=TaskResult(task_id="uuid-test", status="ok"),
                )

                mailbox.send(original)
                msgs = mailbox.receive(max_messages=1)
                received = msgs[0].body

                assert received.request_id == request_id
                assert received.session_id == session_id
                assert isinstance(received.request_id, UUID)
                assert isinstance(received.session_id, UUID)
                msgs[0].acknowledge()

            finally:
                mailbox.close()
