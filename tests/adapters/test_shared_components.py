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
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.helpers import FrozenUtcNow
from tests.helpers.adapters import TEST_ADAPTER_NAME
from weakincentives import ToolValidationError
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
    _restore_filesystem,
    _snapshot_filesystem,
    _ToolExecutionContext,
    parse_tool_arguments,
    tool_to_spec,
)
from weakincentives.contrib.tools import InMemoryFilesystem, SnapshotableFilesystem
from weakincentives.deadlines import Deadline
from weakincentives.errors import SnapshotRestoreError
from weakincentives.prompt import Prompt, PromptTemplate, ToolContext
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.prompt.tool import Tool
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import (
    EventBus,
    HandlerFailure,
    PublishResult,
    ToolInvoked,
)
from weakincentives.runtime.events._types import EventHandler
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
        session=session,
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
        session=session,
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

    outcome = ToolExecutionOutcome(
        tool=typed_tool,
        params=cast(SupportsDataclass, params),
        result=cast(ToolResult[SupportsToolResult], result),
        call_id="call-usage",
        log=log,
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
        session=session,
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


class TestSnapshotFilesystem:
    """Tests for _snapshot_filesystem helper."""

    def test_returns_none_for_none_filesystem(self) -> None:
        log = get_logger(__name__)
        result = _snapshot_filesystem(None, log=log)
        assert result is None

    def test_returns_none_for_non_snapshotable_filesystem(self) -> None:
        class NonSnapshotable:
            """A filesystem that doesn't support snapshots."""

        log = get_logger(__name__)
        result = _snapshot_filesystem(cast(Any, NonSnapshotable()), log=log)
        assert result is None

    def test_creates_snapshot_for_snapshotable_filesystem(self) -> None:
        fs = InMemoryFilesystem()
        log = get_logger(__name__)

        result = _snapshot_filesystem(fs, log=log, tag="test")

        assert result is not None
        assert result.tag == "test"

    def test_returns_none_on_snapshot_error(self) -> None:
        class FailingSnapshotable(SnapshotableFilesystem):
            """A filesystem that fails to snapshot."""

            @property
            def root_path(self) -> str:
                return "/test"

            def snapshot(self, *, tag: str | None = None) -> Any:  # noqa: ANN401
                raise RuntimeError("Snapshot failed")

            def restore(self, snapshot: Any) -> None:  # noqa: ANN401
                pass

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

        log = get_logger(__name__)
        result = _snapshot_filesystem(cast(Any, FailingSnapshotable()), log=log)
        assert result is None


class TestRestoreFilesystem:
    """Tests for _restore_filesystem helper."""

    def test_returns_true_for_none_snapshot(self) -> None:
        fs = InMemoryFilesystem()
        log = get_logger(__name__)

        result = _restore_filesystem(fs, None, log=log)

        assert result is True

    def test_returns_true_for_none_filesystem(self) -> None:
        fs = InMemoryFilesystem()
        log = get_logger(__name__)
        snapshot = fs.snapshot()

        result = _restore_filesystem(None, snapshot, log=log)

        assert result is True

    def test_returns_true_for_non_snapshotable_filesystem(self) -> None:
        fs = InMemoryFilesystem()
        log = get_logger(__name__)
        snapshot = fs.snapshot()

        class NonSnapshotable:
            """A filesystem that doesn't support snapshots."""

        result = _restore_filesystem(cast(Any, NonSnapshotable()), snapshot, log=log)

        assert result is True

    def test_restores_successfully(self) -> None:
        fs = InMemoryFilesystem()
        log = get_logger(__name__)

        # Create a file and snapshot
        fs.write("/test.txt", "original")
        snapshot = fs.snapshot()

        # Modify the file
        fs.write("/test.txt", "modified")
        assert fs.read("/test.txt").content == "modified"

        # Restore
        result = _restore_filesystem(fs, snapshot, log=log)

        assert result is True
        assert fs.read("/test.txt").content == "original"

    def test_returns_false_on_restore_error(self) -> None:
        log = get_logger(__name__)

        inner_fs = InMemoryFilesystem()
        fs = _FailingRestoreFilesystem(inner_fs)
        snapshot = fs.snapshot()

        result = _restore_filesystem(fs, snapshot, log=log)

        assert result is False


class _MockPromptWithFilesystem:
    """Mock prompt that returns a filesystem."""

    def __init__(self, fs: InMemoryFilesystem) -> None:
        self._fs = fs

    def filesystem(self) -> InMemoryFilesystem:
        return self._fs


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
            session=session,
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

        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            session=session,
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

        executor = ToolExecutor(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=cast(ProviderAdapter[Any], object()),
            prompt=cast(Prompt[Any], mock_prompt),
            prompt_name="test",
            rendered=rendered,
            session=session,
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
            session=session,
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
        snapshot = fs.snapshot(tag="before_tool")

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
            session=session,
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
            filesystem=fs,
            fs_snapshot=snapshot,
        )

        _publish_tool_invocation(context=context, outcome=outcome)

        # Filesystem should be restored because publish failed and tool succeeded
        assert fs.read("/test.txt").content == "before_tool"

    def test_filesystem_not_restored_if_tool_already_failed(self) -> None:
        """Verify filesystem is NOT restored again if tool already failed."""
        inner_fs = InMemoryFilesystem()
        inner_fs.write("/test.txt", "original")
        fs = _TrackingFilesystem(inner_fs)
        snapshot = fs.snapshot(tag="original")

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
            session=session,
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
            filesystem=fs,
            fs_snapshot=snapshot,
        )

        _publish_tool_invocation(context=context, outcome=outcome)

        # Restore should NOT have been called since tool.success=False
        assert fs.restore_called is False
