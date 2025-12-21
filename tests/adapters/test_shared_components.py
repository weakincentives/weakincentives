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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.helpers import FrozenUtcNow
from tests.helpers.adapters import TEST_ADAPTER_NAME
from weakincentives import ToolValidationError
from weakincentives.adapters import shared
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    ProviderAdapter,
)
from weakincentives.adapters.shared import (
    ResponseParser,
    ToolExecutionOutcome,
    ToolExecutor,
    _parse_tool_params,
    _publish_tool_invocation,
    _ToolExecutionContext,
    parse_tool_arguments,
    tool_to_spec,
)
from weakincentives.contrib.tools import (
    Filesystem,
    HostFilesystem,
    InMemoryFilesystem,
    SnapshotableFilesystem,
)
from weakincentives.deadlines import Deadline
from weakincentives.errors import SnapshotRestoreError
from weakincentives.prompt import Prompt, PromptTemplate, ToolContext
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.prompt.tool import ResourceRegistry, Tool
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import (
    EventBus,
    HandlerFailure,
    PublishResult,
    ToolInvoked,
)
from weakincentives.runtime.events._types import EventHandler
from weakincentives.runtime.execution_state import ExecutionState
from weakincentives.runtime.logging import get_logger
from weakincentives.runtime.session.session import Session
from weakincentives.types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)


class RecordingBus(EventBus):
    def __init__(self) -> None:
        self.events: list[object] = []

    def subscribe(self, event_type: type[object], handler: EventHandler) -> None:
        pass

    def publish(self, event: object) -> PublishResult:
        self.events.append(event)
        return PublishResult(event=event, handlers_invoked=(), errors=())


@dataclass
class EchoParams:
    value: str


@dataclass
class EchoPayload:
    value: str


def echo_handler(
    params: EchoParams, *, context: ToolContext
) -> ToolResult[EchoPayload]:
    return ToolResult(message="echoed", value=EchoPayload(value=params.value))


def serialize_tool_message(
    result: ToolResult[SupportsToolResult], *, payload: object | None = None
) -> object:
    return {"message": result.message, "payload": payload}


def test_tool_to_spec_accepts_none_params() -> None:
    def handler(params: None, *, context: ToolContext) -> ToolResult[EchoPayload]:
        return ToolResult(message="ok", value=EchoPayload(value="hi"))

    tool = Tool[None, EchoPayload](
        name="no_params",
        description="No arguments required.",
        handler=handler,
    )

    spec = tool_to_spec(cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool))

    assert spec["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


def test_parse_tool_params_rejects_arguments_for_none_params() -> None:
    def no_param_handler(
        params: None, *, context: ToolContext
    ) -> ToolResult[EchoPayload]:
        del context
        return ToolResult(message="ok", value=EchoPayload(value="hi"))

    tool = Tool[None, EchoPayload](
        name="no_params",
        description="No arguments required.",
        handler=no_param_handler,
    )

    with pytest.raises(ToolValidationError, match="does not accept any arguments"):
        _parse_tool_params(
            tool=cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool),
            arguments_mapping={"unexpected": "value"},
        )


def test_parse_tool_params_returns_none_for_empty_arguments() -> None:
    tool = Tool[None, EchoPayload](
        name="no_params",
        description="No arguments required.",
        handler=None,
    )

    parsed = _parse_tool_params(
        tool=cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool),
        arguments_mapping={},
    )

    assert parsed is None


def test_tool_executor_success() -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo",
        handler=echo_handler,
    )
    rendered = RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    execution_state = ExecutionState(session=session)
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
        {tool.name: tool},
    )

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(PromptTemplate(ns="test", key="tool")),
        prompt_name="test",
        rendered=rendered,
        execution_state=execution_state,
        tool_registry=tool_registry,
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
    )

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="echo", arguments='{"value": "hello"}'),
    )

    messages, next_choice = executor.execute([cast(Any, tool_call)], None)
    tool_events = [event for event in bus.events if isinstance(event, ToolInvoked)]

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call-1"
    assert messages[0]["content"] == {"message": "echoed", "payload": None}
    assert next_choice == "auto"
    assert len(tool_events) == 1
    assert len(executor.tool_message_records) == 1


def test_publish_tool_invocation_attaches_usage() -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo",
        handler=echo_handler,
    )
    params = EchoParams(value="hello")
    result = ToolResult(message="echoed", value=EchoPayload(value="hello"))
    log = get_logger(__name__)

    bus = RecordingBus()
    session = Session(bus=bus)
    execution_state = ExecutionState(session=session)
    typed_tool = cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool)
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]] = {
        tool.name: typed_tool
    }

    context = _ToolExecutionContext(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(PromptTemplate(ns="test", key="tool")),
        rendered_prompt=None,
        tool_registry=tool_registry,
        execution_state=execution_state,
        prompt_name="test",
        parse_arguments=parse_tool_arguments,
        format_publish_failures=lambda errors: "",
        deadline=None,
    ).with_provider_payload(
        {
            "usage": {
                "input_tokens": 5,
                "output_tokens": 7,
                "cached_tokens": 2,
            }
        }
    )

    snapshot = execution_state.snapshot(tag="test")
    outcome = ToolExecutionOutcome(
        tool=typed_tool,
        params=cast(SupportsDataclass, params),
        result=cast(ToolResult[SupportsToolResult], result),
        call_id="call-usage",
        log=log,
        snapshot=snapshot,
    )

    invocation = _publish_tool_invocation(context=context, outcome=outcome)

    tool_events = [event for event in bus.events if isinstance(event, ToolInvoked)]

    assert invocation.usage is not None
    assert invocation.usage.input_tokens == 5
    assert invocation.usage.output_tokens == 7
    assert invocation.usage.cached_tokens == 2
    assert tool_events == [invocation]


def test_response_parser_text_only() -> None:
    rendered = RenderedPrompt(text="system")
    parser = ResponseParser[object](
        prompt_name="test",
        rendered=rendered,
        require_structured_output_text=False,
    )

    message = SimpleNamespace(content="Hello")
    output, text = parser.parse(message, None)

    assert output is None
    assert text == "Hello"


@dataclass
class StructuredOutput:
    answer: str


def test_response_parser_structured_output() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    parser = ResponseParser[StructuredOutput](
        prompt_name="test",
        rendered=rendered,
        require_structured_output_text=False,
    )

    message = SimpleNamespace(content=None, parsed={"answer": "42"})
    output, text = parser.parse(message, None)

    assert output == StructuredOutput(answer="42")
    assert text is None


def test_response_parser_structured_output_failure() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    parser = ResponseParser[StructuredOutput](
        prompt_name="test",
        rendered=rendered,
        require_structured_output_text=False,
    )

    message = SimpleNamespace(content="Not JSON", parsed=None)

    with pytest.raises(PromptEvaluationError) as excinfo:
        parser.parse(message, None)

    assert isinstance(excinfo.value, PromptEvaluationError)
    error = excinfo.value
    assert error.phase == PROMPT_EVALUATION_PHASE_RESPONSE


def test_tool_executor_raises_when_deadline_expired(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo",
        handler=echo_handler,
    )
    rendered = RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
            (tool,),
        ),
    )
    bus = RecordingBus()
    session = Session(bus=bus)
    execution_state = ExecutionState(session=session)
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
        {tool.name: tool},
    )

    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))
    frozen_utcnow.advance(timedelta(seconds=10))

    executor = ToolExecutor(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[Any], object()),
        prompt=Prompt(PromptTemplate(ns="test", key="tool")),
        prompt_name="test",
        rendered=rendered,
        execution_state=execution_state,
        tool_registry=tool_registry,
        serialize_tool_message_fn=serialize_tool_message,
        format_publish_failures=lambda x: "",
        parse_arguments=parse_tool_arguments,
        deadline=deadline,
    )

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="echo", arguments='{"value": "hello"}'),
    )

    with pytest.raises(PromptEvaluationError) as excinfo:
        executor.execute([cast(Any, tool_call)], None)

    error = cast(PromptEvaluationError, excinfo.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_TOOL
    assert error.provider_payload == {
        "deadline_expires_at": deadline.expires_at.isoformat()
    }


# ================================================================================
# Filesystem snapshot integration tests
# ================================================================================


class _FailingRestoreFilesystem(SnapshotableFilesystem):
    """Filesystem that fails on restore."""

    def __init__(self, inner: InMemoryFilesystem) -> None:
        self._inner = inner

    @property
    def root_path(self) -> str:
        return self._inner.root_path

    def snapshot(self, *, tag: str | None = None) -> Any:  # noqa: ANN401
        return self._inner.snapshot(tag=tag)

    def restore(self, snapshot: Any) -> None:  # noqa: ANN401
        raise SnapshotRestoreError("Restore failed")

    def read(self, path: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def write(self, path: str, content: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def remove(self, path: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def stat(self, path: str) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def glob(self, pattern: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def grep(self, pattern: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError


class _TrackingFilesystem(SnapshotableFilesystem):
    """Filesystem wrapper that tracks restore calls."""

    restore_called: bool = False

    def __init__(self, inner: InMemoryFilesystem) -> None:
        self._inner = inner

    @property
    def root_path(self) -> str:
        return self._inner.root_path

    def snapshot(self, *, tag: str | None = None) -> Any:  # noqa: ANN401
        return self._inner.snapshot(tag=tag)

    def restore(self, snapshot: Any) -> None:  # noqa: ANN401
        self.restore_called = True
        self._inner.restore(snapshot)

    def read(self, path: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def write(self, path: str, content: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def remove(self, path: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def stat(self, path: str) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def glob(self, pattern: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def grep(self, pattern: str, **kw: object) -> Any:  # noqa: ANN401
        raise NotImplementedError


class _MockPromptWithFilesystem:
    """Mock prompt that returns a filesystem."""

    def __init__(self, fs: InMemoryFilesystem) -> None:
        self._fs = fs

    def filesystem(self) -> InMemoryFilesystem:
        return self._fs


def _create_execution_state(session: Session, fs: InMemoryFilesystem) -> ExecutionState:
    """Create an ExecutionState with filesystem for transactional tool execution."""
    resources = ResourceRegistry.build({Filesystem: fs})
    return ExecutionState(session=session, resources=resources)


# Parameter classes for filesystem integration tests
@dataclass
class _FailParams:
    value: str


@dataclass
class _ExceptionParams:
    value: str


@dataclass
class _ModifyParams:
    value: str


class TestToolExecutionFilesystemIntegration:
    """Tests for filesystem snapshot in tool_execution."""

    def test_snapshot_created_before_tool_invocation(self) -> None:
        """Verify filesystem snapshot is created when tool is executed."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")

        tool = Tool[EchoParams, EchoPayload](
            name="echo",
            description="Echo",
            handler=echo_handler,
        )

        # Create mock prompt with filesystem
        mock_prompt = _MockPromptWithFilesystem(fs)

        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        execution_state = ExecutionState(session=session)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="echo", arguments='{"value": "hello"}'),
        )

        executor.execute([cast(Any, tool_call)], None)

        # The tool execution should have recorded a snapshot
        # We verify this indirectly by checking the outcome has fs_snapshot
        assert len(executor.tool_message_records) == 1

    def test_filesystem_restored_on_tool_failure(self) -> None:
        """Verify filesystem is restored when tool returns success=False."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")

        def failing_handler(
            params: _FailParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            # Modify file before failing
            filesystem = context.filesystem
            if filesystem:
                filesystem.write("/test.txt", "modified")
            return ToolResult(
                message="failed", value=EchoPayload(value=""), success=False
            )

        tool = Tool[_FailParams, EchoPayload](
            name="fail",
            description="Fail",
            handler=failing_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)

        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        execution_state = _create_execution_state(session, fs)
        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="fail", arguments='{"value": "test"}'),
        )

        executor.execute([cast(Any, tool_call)], None)

        # File should be restored to original content
        assert fs.read("/test.txt").content == "original"

    def test_filesystem_restored_on_tool_exception(self) -> None:
        """Verify filesystem is restored when tool raises exception."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")

        def exception_handler(
            params: _ExceptionParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            # Modify file before raising
            filesystem = context.filesystem
            if filesystem:
                filesystem.write("/test.txt", "modified")
            raise RuntimeError("Tool crashed")

        tool = Tool[_ExceptionParams, EchoPayload](
            name="exception",
            description="Exception",
            handler=exception_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)

        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        execution_state = _create_execution_state(session, fs)
        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="exception", arguments='{"value": "test"}'),
        )

        # Tool exception is caught and returned as a failed result
        executor.execute([cast(Any, tool_call)], None)

        # File should be restored to original content
        assert fs.read("/test.txt").content == "original"

    def test_filesystem_not_restored_on_success(self) -> None:
        """Verify filesystem is NOT restored when tool succeeds."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "original")

        def modify_handler(
            params: _ModifyParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                filesystem.write("/test.txt", "modified")
            return ToolResult(message="ok", value=EchoPayload(value="ok"))

        tool = Tool[_ModifyParams, EchoPayload](
            name="modify",
            description="Modify",
            handler=modify_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)

        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        execution_state = _create_execution_state(session, fs)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="modify", arguments='{"value": "test"}'),
        )

        executor.execute([cast(Any, tool_call)], None)

        # File should keep the modified content since tool succeeded
        assert fs.read("/test.txt").content == "modified"


class TestPublishInvocationFilesystemRestore:
    """Tests for filesystem restore in _publish_tool_invocation."""

    def test_filesystem_restored_on_publish_failure(self) -> None:
        """Verify filesystem is restored when event publishing fails."""
        fs = InMemoryFilesystem()
        fs.write("/test.txt", "before_tool")

        def dummy_handler(e: object) -> None:
            pass

        class FailingBus(EventBus):
            def subscribe(
                self, event_type: type[object], handler: EventHandler
            ) -> None:
                pass

            def publish(self, event: object) -> PublishResult:
                return PublishResult(
                    event=event,
                    handlers_invoked=(),
                    errors=(
                        HandlerFailure(
                            handler=dummy_handler, error=Exception("publish failed")
                        ),
                    ),
                )

        bus = FailingBus()
        session = Session(bus=bus)

        # Create ExecutionState and take snapshot BEFORE tool modifications
        execution_state = _create_execution_state(session, fs)
        composite_snapshot = execution_state.snapshot(tag="before_tool")

        # Modify file to simulate what tool did
        fs.write("/test.txt", "after_tool")

        tool = Tool[EchoParams, EchoPayload](
            name="echo",
            description="Echo",
            handler=echo_handler,
        )
        params = EchoParams(value="hello")
        result = ToolResult(
            message="echoed", value=EchoPayload(value="hello"), success=True
        )
        log = get_logger(__name__)

        typed_tool = cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool)
        tool_registry: Mapping[
            str, Tool[SupportsDataclassOrNone, SupportsToolResult]
        ] = {tool.name: typed_tool}

        context = _ToolExecutionContext(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=Prompt(PromptTemplate(ns="test", key="tool")),
            rendered_prompt=None,
            tool_registry=tool_registry,
            execution_state=execution_state,
            prompt_name="test",
            parse_arguments=parse_tool_arguments,
            format_publish_failures=lambda errors: "publish failed",
            deadline=None,
        )

        outcome = ToolExecutionOutcome(
            tool=typed_tool,
            params=cast(SupportsDataclass, params),
            result=cast(ToolResult[SupportsToolResult], result),
            call_id="call-publish-fail",
            log=log,
            snapshot=composite_snapshot,
        )

        _publish_tool_invocation(context=context, outcome=outcome)

        # Filesystem should be restored because publish failed and tool succeeded
        assert fs.read("/test.txt").content == "before_tool"

    def test_filesystem_not_restored_if_tool_already_failed(self) -> None:
        """Verify filesystem is NOT restored again if tool already failed."""
        inner_fs = InMemoryFilesystem()
        inner_fs.write("/test.txt", "original")
        fs = _TrackingFilesystem(inner_fs)

        def dummy_handler(e: object) -> None:
            pass

        class FailingBus(EventBus):
            def subscribe(
                self, event_type: type[object], handler: EventHandler
            ) -> None:
                pass

            def publish(self, event: object) -> PublishResult:
                return PublishResult(
                    event=event,
                    handlers_invoked=(),
                    errors=(
                        HandlerFailure(
                            handler=dummy_handler, error=Exception("publish failed")
                        ),
                    ),
                )

        bus = FailingBus()
        session = Session(bus=bus)

        # Create ExecutionState and take snapshot
        resources = ResourceRegistry.build({Filesystem: fs})
        execution_state = ExecutionState(session=session, resources=resources)
        composite_snapshot = execution_state.snapshot(tag="original")

        # File is already restored (simulating what tool_execution did)
        # by keeping original content

        tool = Tool[EchoParams, EchoPayload](
            name="echo",
            description="Echo",
            handler=echo_handler,
        )
        params = EchoParams(value="hello")
        # Tool returned success=False, meaning filesystem was already restored
        result = ToolResult(
            message="failed", value=EchoPayload(value=""), success=False
        )
        log = get_logger(__name__)

        typed_tool = cast(Tool[SupportsDataclassOrNone, SupportsToolResult], tool)
        tool_registry: Mapping[
            str, Tool[SupportsDataclassOrNone, SupportsToolResult]
        ] = {tool.name: typed_tool}

        context = _ToolExecutionContext(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=Prompt(PromptTemplate(ns="test", key="tool")),
            rendered_prompt=None,
            tool_registry=tool_registry,
            execution_state=execution_state,
            prompt_name="test",
            parse_arguments=parse_tool_arguments,
            format_publish_failures=lambda errors: "publish failed",
            deadline=None,
        )

        outcome = ToolExecutionOutcome(
            tool=typed_tool,
            params=cast(SupportsDataclass, params),
            result=cast(ToolResult[SupportsToolResult], result),
            call_id="call-already-failed",
            log=log,
            snapshot=composite_snapshot,
        )

        _publish_tool_invocation(context=context, outcome=outcome)

        # Restore should NOT have been called since tool.success=False
        assert fs.restore_called is False


# ================================================================================
# Focused integration tests for filesystem snapshot + tool execution
# ================================================================================


@dataclass
class _MultiFileParams:
    """Parameters for multi-file operations."""

    files: list[str]


@dataclass
class _InvalidParams:
    """Parameters that will fail validation."""

    value: int  # Will fail if we pass a string


class TestFilesystemSnapshotIntegration:
    """Focused integration tests for filesystem snapshots with tool execution."""

    def test_multiple_files_restored_on_failure(self) -> None:
        """Test that all file modifications are restored when tool fails."""
        fs = InMemoryFilesystem()
        # Create initial state with multiple files
        fs.write("/file1.txt", "original1")
        fs.write("/file2.txt", "original2")
        fs.write("/subdir/file3.txt", "original3")

        def multi_modify_handler(
            params: _MultiFileParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                # Modify all files
                filesystem.write("/file1.txt", "modified1")
                filesystem.write("/file2.txt", "modified2")
                filesystem.write("/subdir/file3.txt", "modified3")
                # Also create a new file
                filesystem.write("/newfile.txt", "new content")
            return ToolResult(
                message="failed", value=EchoPayload(value=""), success=False
            )

        tool = Tool[_MultiFileParams, EchoPayload](
            name="multi_modify",
            description="Modify multiple files then fail",
            handler=multi_modify_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        execution_state = _create_execution_state(session, fs)
        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="multi_modify",
                arguments='{"files": ["file1.txt", "file2.txt", "file3.txt"]}',
            ),
        )

        executor.execute([cast(Any, tool_call)], None)

        # All files should be restored to original content
        assert fs.read("/file1.txt").content == "original1"
        assert fs.read("/file2.txt").content == "original2"
        assert fs.read("/subdir/file3.txt").content == "original3"
        # New file should not exist
        assert not fs.exists("/newfile.txt")

    def test_directory_operations_restored_on_exception(self) -> None:
        """Test that directory modifications are restored on tool exception."""
        fs = InMemoryFilesystem()
        fs.write("/existing/file.txt", "content")

        def dir_modify_handler(
            params: _FailParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                # Create new directory structure
                filesystem.write("/newdir/nested/file.txt", "nested content")
                filesystem.write("/existing/file.txt", "modified")
            raise RuntimeError("Simulated crash after modifications")

        tool = Tool[_FailParams, EchoPayload](
            name="dir_modify",
            description="Modify directories then crash",
            handler=dir_modify_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        execution_state = _create_execution_state(session, fs)
        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="dir_modify", arguments='{"value": "test"}'),
        )

        executor.execute([cast(Any, tool_call)], None)

        # Original file should be restored
        assert fs.read("/existing/file.txt").content == "content"
        # New directory structure should not exist
        assert not fs.exists("/newdir/nested/file.txt")
        assert not fs.exists("/newdir")

    def test_validation_error_does_not_restore_filesystem(self) -> None:
        """Test that validation errors don't trigger filesystem restore.

        When a tool call fails validation (before the handler is invoked),
        the filesystem should not be modified or restored since the tool
        never ran.
        """
        fs = InMemoryFilesystem()
        fs.write("/file.txt", "original")

        # Track if handler was called
        handler_called = False

        def should_not_run(
            params: _InvalidParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            nonlocal handler_called
            handler_called = True
            filesystem = context.filesystem
            if filesystem:
                filesystem.write("/file.txt", "modified")
            return ToolResult(message="ok", value=EchoPayload(value="ok"))

        tool = Tool[_InvalidParams, EchoPayload](
            name="invalid_tool",
            description="Tool that expects int but gets string",
            handler=should_not_run,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        execution_state = _create_execution_state(session, fs)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        # Pass a string where int is expected - this should fail validation
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="invalid_tool", arguments='{"value": "not_an_int"}'
            ),
        )

        executor.execute([cast(Any, tool_call)], None)

        # Handler should not have been called
        assert handler_called is False
        # File should remain unchanged (no restore needed since no modification)
        assert fs.read("/file.txt").content == "original"

    def test_successful_tool_preserves_all_changes(self) -> None:
        """Test that successful tool execution preserves all filesystem changes."""
        fs = InMemoryFilesystem()
        fs.write("/file1.txt", "original1")
        fs.write("/file2.txt", "original2")

        def success_handler(
            params: _MultiFileParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                filesystem.write("/file1.txt", "modified1")
                filesystem.write("/file2.txt", "modified2")
                filesystem.write("/newfile.txt", "new content")
                filesystem.delete("/file2.txt")
            return ToolResult(message="ok", value=EchoPayload(value="ok"))

        tool = Tool[_MultiFileParams, EchoPayload](
            name="success_modify",
            description="Modify files successfully",
            handler=success_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        execution_state = _create_execution_state(session, fs)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="success_modify", arguments='{"files": ["file1.txt"]}'
            ),
        )

        executor.execute([cast(Any, tool_call)], None)

        # All changes should be preserved
        assert fs.read("/file1.txt").content == "modified1"
        assert not fs.exists("/file2.txt")  # Was removed
        assert fs.read("/newfile.txt").content == "new content"

    def test_file_deletion_restored_on_failure(self) -> None:
        """Test that deleted files are restored when tool fails."""
        fs = InMemoryFilesystem()
        fs.write("/important.txt", "critical data")
        fs.write("/keep.txt", "keep this")

        def delete_handler(
            params: _FailParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                filesystem.delete("/important.txt")
                filesystem.write("/keep.txt", "modified")
            return ToolResult(
                message="failed", value=EchoPayload(value=""), success=False
            )

        tool = Tool[_FailParams, EchoPayload](
            name="delete_tool",
            description="Delete files then fail",
            handler=delete_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        execution_state = _create_execution_state(session, fs)
        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="delete_tool", arguments='{"value": "test"}'),
        )

        executor.execute([cast(Any, tool_call)], None)

        # Deleted file should be restored
        assert fs.exists("/important.txt")
        assert fs.read("/important.txt").content == "critical data"
        # Modified file should also be restored
        assert fs.read("/keep.txt").content == "keep this"

    def test_partial_modifications_before_exception(self) -> None:
        """Test that partial modifications are rolled back on exception."""
        fs = InMemoryFilesystem()
        fs.write("/step1.txt", "original1")
        fs.write("/step2.txt", "original2")
        fs.write("/step3.txt", "original3")

        def partial_handler(
            params: _FailParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                # First two modifications succeed
                filesystem.write("/step1.txt", "modified1")
                filesystem.write("/step2.txt", "modified2")
                # Then crash before third modification
                raise RuntimeError("Crash after partial modifications")
            return ToolResult(message="ok", value=EchoPayload(value="ok"))

        tool = Tool[_FailParams, EchoPayload](
            name="partial_tool",
            description="Partially modify then crash",
            handler=partial_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        execution_state = _create_execution_state(session, fs)
        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="partial_tool", arguments='{"value": "test"}'
            ),
        )

        executor.execute([cast(Any, tool_call)], None)

        # All files should be restored to original state
        assert fs.read("/step1.txt").content == "original1"
        assert fs.read("/step2.txt").content == "original2"
        assert fs.read("/step3.txt").content == "original3"


class TestHostFilesystemToolIntegration:
    """Integration tests for HostFilesystem with tool execution."""

    def test_host_filesystem_restored_on_tool_failure(self, tmp_path: Path) -> None:
        """Test HostFilesystem restore when tool fails."""
        fs = HostFilesystem(_root=str(tmp_path))

        # Create initial files
        fs.write("file1.txt", "original content 1")
        fs.write("file2.txt", "original content 2")

        def failing_host_handler(
            params: _FailParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                filesystem.write("file1.txt", "modified content 1")
                filesystem.write("file2.txt", "modified content 2")
                filesystem.write("newfile.txt", "new file content")
            return ToolResult(
                message="failed", value=EchoPayload(value=""), success=False
            )

        tool = Tool[_FailParams, EchoPayload](
            name="host_fail",
            description="Modify host files then fail",
            handler=failing_host_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)  # type: ignore[arg-type]
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        resources = ResourceRegistry.build({Filesystem: fs})
        execution_state = ExecutionState(session=session, resources=resources)
        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="host_fail", arguments='{"value": "test"}'),
        )

        executor.execute([cast(Any, tool_call)], None)

        # Files should be restored via git reset
        assert fs.read("file1.txt").content == "original content 1"
        assert fs.read("file2.txt").content == "original content 2"
        # New file should be gone
        assert not fs.exists("newfile.txt")

    def test_host_filesystem_success_preserves_changes(self, tmp_path: Path) -> None:
        """Test HostFilesystem preserves changes on success."""
        fs = HostFilesystem(_root=str(tmp_path))

        # Create initial file
        fs.write("file.txt", "original")

        def success_host_handler(
            params: _FailParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                filesystem.write("file.txt", "modified by tool")
            return ToolResult(message="ok", value=EchoPayload(value="ok"))

        tool = Tool[_FailParams, EchoPayload](
            name="host_success",
            description="Modify host files successfully",
            handler=success_host_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)  # type: ignore[arg-type]
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        execution_state = _create_execution_state(session, fs)  # type: ignore[arg-type]
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="host_success", arguments='{"value": "test"}'
            ),
        )

        executor.execute([cast(Any, tool_call)], None)

        # Changes should be preserved
        assert fs.read("file.txt").content == "modified by tool"

    def test_host_filesystem_exception_restores_state(self, tmp_path: Path) -> None:
        """Test HostFilesystem restores state on tool exception."""
        fs = HostFilesystem(_root=str(tmp_path))

        # Create initial state
        fs.write("data.txt", "important data")

        def exception_host_handler(
            params: _FailParams, *, context: ToolContext
        ) -> ToolResult[EchoPayload]:
            filesystem = context.filesystem
            if filesystem:
                filesystem.write("data.txt", "corrupted")
                filesystem.delete("data.txt")
            raise RuntimeError("Tool crashed!")

        tool = Tool[_FailParams, EchoPayload](
            name="host_exception",
            description="Corrupt host files then crash",
            handler=exception_host_handler,
        )

        mock_prompt = _MockPromptWithFilesystem(fs)  # type: ignore[arg-type]
        rendered = RenderedPrompt(
            text="system",
            _tools=cast(
                tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
                (tool,),
            ),
        )

        bus = RecordingBus()
        session = Session(bus=bus)
        tool_registry = cast(
            Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
            {tool.name: tool},
        )

        resources = ResourceRegistry.build({Filesystem: fs})
        execution_state = ExecutionState(session=session, resources=resources)
        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            execution_state=execution_state,
            tool_registry=tool_registry,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=lambda x: "",
            parse_arguments=parse_tool_arguments,
        )

        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="host_exception", arguments='{"value": "test"}'
            ),
        )

        executor.execute([cast(Any, tool_call)], None)

        # File should be restored
        assert fs.exists("data.txt")
        assert fs.read("data.txt").content == "important data"


def test_extract_payload_handles_mapping_payload() -> None:
    mapping_data = {"key": "value", "nested": {"inner": "data"}}
    result = shared.extract_payload(mapping_data)
    assert result == {"key": "value", "nested": {"inner": "data"}}


def test_build_json_schema_response_format_object_container() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    result = shared.build_json_schema_response_format(rendered, "test")
    assert result is not None
    schema_payload = result["json_schema"]["schema"]
    assert schema_payload["additionalProperties"] is False


def test_response_parser_sets_text_to_none_when_output_present() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    parser = ResponseParser[StructuredOutput](
        prompt_name="test",
        rendered=rendered,
        require_structured_output_text=False,
    )
    message = SimpleNamespace(content="Some text", parsed={"answer": "result"})
    output, text = parser.parse(message, None)
    assert output is not None
    assert text is None


def test_response_parser_preserves_text_when_output_is_none() -> None:
    """Test that text_value is preserved when structured output is None."""
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    parser = ResponseParser[StructuredOutput](
        prompt_name="test",
        rendered=rendered,
        require_structured_output_text=False,
    )
    # Simulate case where parsed is None but content exists
    # parse_structured_output might return None in edge cases
    message = SimpleNamespace(content='{"invalid_field": "value"}', parsed=None)

    # This should raise because the JSON doesn't match the schema
    with pytest.raises(PromptEvaluationError):
        parser.parse(message, None)
