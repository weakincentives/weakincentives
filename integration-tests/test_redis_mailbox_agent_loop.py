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
# Test Domain Types - Realistic nested dataclass structures
# =============================================================================


@dataclass(slots=True, frozen=True)
class UserContext:
    """User context embedded in requests."""

    user_id: str
    tenant_id: str
    roles: list[str]
    session_token: str | None = None


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """LLM model configuration embedded in requests."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 4096
    stop_sequences: list[str] | None = None


@dataclass(slots=True, frozen=True)
class InputDocument:
    """Document to be processed, embedded in requests."""

    document_id: str
    content: str
    content_type: str = "text/plain"
    metadata: dict[str, str] | None = None


@dataclass(slots=True, frozen=True)
class TaskRequest:
    """Realistic request type with nested dataclasses.

    Represents a typical agent task request with user context,
    model configuration, and input documents.
    """

    task_id: str
    user_context: UserContext
    model_config: ModelConfig
    documents: list[InputDocument]
    instructions: str
    priority: int = 0
    tags: dict[str, str] | None = None


@dataclass(slots=True, frozen=True)
class TokenUsage:
    """Token usage statistics embedded in results."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(slots=True, frozen=True)
class ProcessedChunk:
    """A processed chunk of output embedded in results."""

    chunk_id: str
    content: str
    confidence: float
    source_document_id: str


@dataclass(slots=True, frozen=True)
class TaskResult:
    """Realistic result type with nested dataclasses.

    Represents a typical agent task result with token usage,
    processed chunks, and execution metadata.
    """

    task_id: str
    status: str
    chunks: list[ProcessedChunk]
    token_usage: TokenUsage | None = None
    error_message: str | None = None
    execution_time_ms: int = 0
    metadata: dict[str, str] | None = None


@FrozenDataclass()
class NestedConfig:
    """Deeply nested configuration for stress testing serialization."""

    name: str
    settings: dict[str, str]


@FrozenDataclass()
class ComplexRequest:
    """Complex request with multiple levels of nesting."""

    name: str
    configs: list[NestedConfig]
    items: list[str]
    metadata: dict[str, str] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@FrozenDataclass()
class ComplexResult:
    """Complex result with nested types to test deep serialization."""

    request_name: str
    processed_configs: list[NestedConfig]
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


def _make_task_request(
    task_id: str = "task-123",
    *,
    priority: int = 0,
    tags: dict[str, str] | None = None,
) -> TaskRequest:
    """Create a realistic TaskRequest with nested dataclasses."""
    return TaskRequest(
        task_id=task_id,
        user_context=UserContext(
            user_id="user-456",
            tenant_id="tenant-789",
            roles=["analyst", "viewer"],
            session_token="token-abc123",
        ),
        model_config=ModelConfig(
            model_name="claude-3-opus",
            temperature=0.5,
            max_tokens=8192,
            stop_sequences=["END", "STOP"],
        ),
        documents=[
            InputDocument(
                document_id="doc-001",
                content="This is the first document content.",
                content_type="text/plain",
                metadata={"source": "upload", "page": "1"},
            ),
            InputDocument(
                document_id="doc-002",
                content="This is the second document with more content.",
                content_type="text/markdown",
            ),
        ],
        instructions="Analyze these documents and provide a summary.",
        priority=priority,
        tags=tags,
    )


def _make_task_result(
    task_id: str = "task-123",
    *,
    status: str = "completed",
    error_message: str | None = None,
) -> TaskResult:
    """Create a realistic TaskResult with nested dataclasses."""
    return TaskResult(
        task_id=task_id,
        status=status,
        chunks=[
            ProcessedChunk(
                chunk_id="chunk-001",
                content="Summary of document 1: Key findings include...",
                confidence=0.95,
                source_document_id="doc-001",
            ),
            ProcessedChunk(
                chunk_id="chunk-002",
                content="Summary of document 2: Additional analysis shows...",
                confidence=0.87,
                source_document_id="doc-002",
            ),
        ],
        token_usage=TokenUsage(
            prompt_tokens=1500,
            completion_tokens=500,
            total_tokens=2000,
        ),
        error_message=error_message,
        execution_time_ms=3500,
        metadata={"model_version": "v1.2", "cache_hit": "false"},
    )


@pytest.mark.redis_standalone
class TestAgentLoopRequestSerialization:
    """Tests for AgentLoopRequest serialization/deserialization via RedisMailbox."""

    def test_nested_request_round_trip(self) -> None:
        """AgentLoopRequest with nested dataclasses serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-agent-request",
                client=client,
            )
            try:
                # Create request with deeply nested structure
                original = AgentLoopRequest(request=_make_task_request())

                # Send through Redis (serializes)
                mailbox.send(original)

                # Receive from Redis (deserializes)
                messages = mailbox.receive(max_messages=1)
                assert len(messages) == 1
                received = messages[0].body

                # Verify top-level fields
                assert received.request.task_id == original.request.task_id
                assert received.request.instructions == original.request.instructions
                assert received.request.priority == original.request.priority
                assert received.request_id == original.request_id

                # Verify nested UserContext
                assert received.request.user_context.user_id == "user-456"
                assert received.request.user_context.tenant_id == "tenant-789"
                assert received.request.user_context.roles == ["analyst", "viewer"]
                assert received.request.user_context.session_token == "token-abc123"

                # Verify nested ModelConfig
                assert received.request.model_config.model_name == "claude-3-opus"
                assert received.request.model_config.temperature == 0.5
                assert received.request.model_config.max_tokens == 8192
                assert received.request.model_config.stop_sequences == ["END", "STOP"]

                # Verify nested list of InputDocuments
                assert len(received.request.documents) == 2
                assert received.request.documents[0].document_id == "doc-001"
                assert received.request.documents[0].content_type == "text/plain"
                assert received.request.documents[0].metadata == {
                    "source": "upload",
                    "page": "1",
                }
                assert received.request.documents[1].document_id == "doc-002"
                assert received.request.documents[1].content_type == "text/markdown"
                assert received.request.documents[1].metadata is None

                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_request_with_priority_and_tags(self) -> None:
        """AgentLoopRequest with non-default fields serializes correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-priority-request",
                client=client,
            )
            try:
                original = AgentLoopRequest(
                    request=_make_task_request(
                        task_id="high-priority",
                        priority=10,
                        tags={"urgency": "high", "department": "engineering"},
                    ),
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.request.priority == 10
                assert received.request.task_id == "high-priority"
                assert received.request.tags == {
                    "urgency": "high",
                    "department": "engineering",
                }
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
                    request=_make_task_request(task_id="ctx-task"),
                    run_context=run_context,
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.run_context is not None
                assert received.run_context.trace_id == run_context.trace_id
                assert received.run_context.span_id == run_context.span_id
                # Also verify nested structure preserved
                assert received.request.user_context.user_id == "user-456"
                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_complex_request_with_nested_configs(self) -> None:
        """AgentLoopRequest with ComplexRequest containing nested configs."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[ComplexRequest], None](
                name="test-complex-request",
                client=client,
            )
            try:
                original = AgentLoopRequest(
                    request=ComplexRequest(
                        name="batch-job",
                        configs=[
                            NestedConfig(
                                name="config-1",
                                settings={"key1": "value1", "key2": "value2"},
                            ),
                            NestedConfig(
                                name="config-2",
                                settings={"key3": "value3"},
                            ),
                        ],
                        items=["item1", "item2", "item3"],
                        metadata={"source": "test", "version": "1.0"},
                    ),
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.request.name == "batch-job"
                assert len(received.request.configs) == 2
                assert received.request.configs[0].name == "config-1"
                assert received.request.configs[0].settings == {
                    "key1": "value1",
                    "key2": "value2",
                }
                assert received.request.configs[1].name == "config-2"
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

    def test_nested_result_round_trip(self) -> None:
        """AgentLoopResult with nested dataclasses serializes correctly."""
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
                    output=_make_task_result(),
                    session_id=session_id,
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                # Verify top-level result fields
                assert received.request_id == request_id
                assert received.session_id == session_id
                assert received.error is None
                assert received.success is True
                assert received.output is not None

                # Verify nested TaskResult
                assert received.output.task_id == "task-123"
                assert received.output.status == "completed"
                assert received.output.execution_time_ms == 3500
                assert received.output.metadata == {
                    "model_version": "v1.2",
                    "cache_hit": "false",
                }

                # Verify nested TokenUsage
                assert received.output.token_usage is not None
                assert received.output.token_usage.prompt_tokens == 1500
                assert received.output.token_usage.completion_tokens == 500
                assert received.output.token_usage.total_tokens == 2000

                # Verify nested list of ProcessedChunks
                assert len(received.output.chunks) == 2
                assert received.output.chunks[0].chunk_id == "chunk-001"
                assert received.output.chunks[0].confidence == 0.95
                assert received.output.chunks[0].source_document_id == "doc-001"
                assert received.output.chunks[1].chunk_id == "chunk-002"
                assert received.output.chunks[1].confidence == 0.87

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

    def test_result_with_run_context_and_nested_output(self) -> None:
        """AgentLoopResult with RunContext and nested output serializes correctly."""
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
                    output=_make_task_result(task_id="ctx-task", status="done"),
                    session_id=session_id,
                    run_context=run_context,
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                # Verify RunContext
                assert received.run_context is not None
                assert received.run_context.trace_id == run_context.trace_id
                assert received.run_context.session_id == session_id

                # Verify nested output preserved
                assert received.output is not None
                assert received.output.task_id == "ctx-task"
                assert len(received.output.chunks) == 2
                messages[0].acknowledge()
            finally:
                mailbox.close()

    def test_complex_result_with_nested_configs(self) -> None:
        """AgentLoopResult with ComplexResult containing nested configs."""
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
                        processed_configs=[
                            NestedConfig(
                                name="processed-config-1",
                                settings={"result": "success"},
                            ),
                        ],
                        processed_items=["result1", "result2"],
                        summary={"total": 2, "success": 2},
                    ),
                )

                mailbox.send(original)
                messages = mailbox.receive(max_messages=1)
                received = messages[0].body

                assert received.output is not None
                assert received.output.request_name == "batch-job"
                assert len(received.output.processed_configs) == 1
                assert received.output.processed_configs[0].name == "processed-config-1"
                assert received.output.processed_configs[0].settings == {
                    "result": "success"
                }
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

    def test_request_response_round_trip_with_nested_types(self) -> None:
        """Full request/response flow with deeply nested types."""
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
                # Client sends request with deeply nested structure
                task_request = _make_task_request(task_id="flow-task")
                request = AgentLoopRequest(request=task_request)
                requests.send(request, reply_to=responses)

                # Worker receives request
                msgs = requests.receive(max_messages=1)
                assert len(msgs) == 1
                received_request = msgs[0].body

                # Verify nested structure was preserved
                assert received_request.request.task_id == "flow-task"
                assert received_request.request.user_context.user_id == "user-456"
                assert (
                    received_request.request.model_config.model_name == "claude-3-opus"
                )
                assert len(received_request.request.documents) == 2

                # Worker creates response with nested structure
                result: AgentLoopResult[TaskResult] = AgentLoopResult(
                    request_id=received_request.request_id,
                    output=_make_task_result(
                        task_id=received_request.request.task_id,
                        status="completed",
                    ),
                )
                msgs[0].reply(result)
                msgs[0].acknowledge()

                # Client receives response
                response_msgs = responses.receive(max_messages=1)
                assert len(response_msgs) == 1
                received_result = response_msgs[0].body

                # Verify nested response structure
                assert received_result.request_id == request.request_id
                assert received_result.output is not None
                assert received_result.output.status == "completed"
                assert received_result.output.token_usage is not None
                assert received_result.output.token_usage.total_tokens == 2000
                assert len(received_result.output.chunks) == 2
                response_msgs[0].acknowledge()

            finally:
                requests.close()
                responses.close()

    def test_multiple_requests_fifo_order(self) -> None:
        """Multiple requests with nested types are processed in FIFO order."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-fifo-requests",
                client=client,
            )

            try:
                # Send multiple requests with nested structures
                request_ids = []
                for i in range(3):
                    req = AgentLoopRequest(
                        request=_make_task_request(task_id=f"task-{i}"),
                    )
                    request_ids.append(req.request_id)
                    mailbox.send(req)

                # Receive all
                msgs = mailbox.receive(max_messages=3)
                assert len(msgs) == 3

                # Verify FIFO order and nested structure
                for i, msg in enumerate(msgs):
                    assert msg.body.request.task_id == f"task-{i}"
                    assert msg.body.request_id == request_ids[i]
                    # Verify nested structures preserved
                    assert msg.body.request.user_context.user_id == "user-456"
                    assert len(msg.body.request.documents) == 2
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
                    request=_make_task_request(task_id="fail-task"),
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

    def test_generic_type_extracts_nested_types_at_runtime(self) -> None:
        """RedisMailbox[T, R] extracts T and deserializes nested types correctly."""
        with redis_standalone() as client:
            # Use generic syntax - this should extract AgentLoopRequest[TaskRequest]
            mailbox = RedisMailbox[AgentLoopRequest[TaskRequest], None](
                name="test-generic-extract",
                client=client,
            )

            try:
                original = AgentLoopRequest(
                    request=_make_task_request(task_id="generic-test"),
                )

                mailbox.send(original)
                msgs = mailbox.receive(max_messages=1)

                # Body should be correctly typed via generic type extraction
                received = msgs[0].body
                assert isinstance(received, AgentLoopRequest)
                assert isinstance(received.request, TaskRequest)
                assert received.request.task_id == "generic-test"

                # Verify nested types are correctly deserialized
                assert isinstance(received.request.user_context, UserContext)
                assert isinstance(received.request.model_config, ModelConfig)
                assert all(
                    isinstance(doc, InputDocument) for doc in received.request.documents
                )
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
                    request=_make_task_request(task_id="dt-test"),
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
                    output=_make_task_result(task_id="uuid-test", status="ok"),
                )

                mailbox.send(original)
                msgs = mailbox.receive(max_messages=1)
                received = msgs[0].body

                assert received.request_id == request_id
                assert received.session_id == session_id
                assert isinstance(received.request_id, UUID)
                assert isinstance(received.session_id, UUID)

                # Verify nested types in output
                assert received.output is not None
                assert isinstance(received.output.token_usage, TokenUsage)
                assert all(
                    isinstance(chunk, ProcessedChunk)
                    for chunk in received.output.chunks
                )
                msgs[0].acknowledge()

            finally:
                mailbox.close()

    def test_float_fields_serialize_correctly(self) -> None:
        """float fields in nested types are serialized/deserialized correctly."""
        with redis_standalone() as client:
            mailbox = RedisMailbox[AgentLoopResult[TaskResult], None](
                name="test-float-serde",
                client=client,
            )

            try:
                original: AgentLoopResult[TaskResult] = AgentLoopResult(
                    request_id=uuid4(),
                    output=_make_task_result(),
                )

                mailbox.send(original)
                msgs = mailbox.receive(max_messages=1)
                received = msgs[0].body

                # Verify float fields in nested ProcessedChunk
                assert received.output is not None
                assert received.output.chunks[0].confidence == 0.95
                assert received.output.chunks[1].confidence == 0.87

                # Verify float field in nested ModelConfig (if we had a request)
                msgs[0].acknowledge()

            finally:
                mailbox.close()
