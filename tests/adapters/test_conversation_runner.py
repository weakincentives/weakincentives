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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from tests.helpers import (
    FrozenUtcNow,
    frozen_utcnow as _frozen_utcnow,  # noqa: F401
)
from tests.helpers.adapters import DUMMY_ADAPTER_NAME
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_BUDGET,
    PROMPT_EVALUATION_PHASE_REQUEST,
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from weakincentives.adapters.inner_loop import (
    InnerLoop,
    InnerLoopConfig,
    InnerLoopInputs,
    run_inner_loop,
)
from weakincentives.adapters.throttle import ThrottlePolicy, new_throttle_policy
from weakincentives.adapters.utilities import ToolChoice, token_usage_from_payload
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt, PromptTemplate, ToolContext
from weakincentives.prompt.overrides import PromptDescriptor
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import StructuredOutputConfig
from weakincentives.prompt.tool import Tool
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import (
    Dispatcher,
    DispatchResult,
    HandlerFailure,
    PromptExecuted,
    PromptRendered,
    TokenUsage,
    ToolInvoked,
)
from weakincentives.runtime.events._types import EventHandler
from weakincentives.runtime.session import (
    DEFAULT_SNAPSHOT_POLICIES,
    SessionProtocol,
    SlicePolicy,
)
from weakincentives.runtime.session.protocols import SnapshotProtocol
from weakincentives.runtime.session.session import Session
from weakincentives.runtime.session.snapshots import Snapshot
from weakincentives.types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)

from ._test_stubs import DummyChoice, DummyMessage, DummyResponse, DummyToolCall


class DummyAdapter(ProviderAdapter[object]):
    def evaluate(
        self,
        prompt: Prompt[object],
        *,
        bus: Dispatcher,
        session: SessionProtocol | None = None,
        deadline: Deadline | None = None,
    ) -> PromptResponse[object]:
        raise NotImplementedError


class ProviderStub:
    def __init__(self, responses: Sequence[DummyResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tool_specs: Sequence[Mapping[str, Any]],
        tool_choice: ToolChoice | None,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tool_specs": list(tool_specs),
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


def serialize_tool_message(
    result: ToolResult[SupportsToolResult], *, payload: object | None = None
) -> object:
    return {"message": result.message, "payload": payload}


class RecordingBus(Dispatcher):
    def __init__(
        self,
        *,
        fail_rendered: bool = False,
        fail_tool: bool = False,
        fail_prompt: bool = False,
    ) -> None:
        self.events: list[object] = []
        self.fail_rendered = fail_rendered
        self.fail_tool = fail_tool
        self.fail_prompt = fail_prompt
        self.subscriptions: list[tuple[type[object], EventHandler]] = []

    def subscribe(
        self, event_type: type[object], handler: EventHandler
    ) -> None:  # pragma: no cover - unused
        self.subscriptions.append((event_type, handler))

    def dispatch(self, event: object) -> DispatchResult:
        self.events.append(event)
        if self.fail_rendered and isinstance(event, PromptRendered):
            return self._failure_result(event, "prompt rendered dispatch failure")
        if self.fail_tool and isinstance(event, ToolInvoked):
            return self._failure_result(event, "reducer failure")
        if self.fail_prompt and isinstance(event, PromptExecuted):
            return self._failure_result(event, "prompt dispatch failure")
        return DispatchResult(event=event, handlers_invoked=(), errors=())

    @staticmethod
    def _failure_result(event: object, message: str) -> DispatchResult:
        def failure_handler(_event: object) -> None:  # pragma: no cover - defensive
            return None

        failure = HandlerFailure(handler=failure_handler, error=RuntimeError(message))
        return DispatchResult(
            event=event,
            handlers_invoked=(failure_handler,),
            errors=(failure,),
        )


def build_inner_loop(
    *,
    rendered: RenderedPrompt[object],
    provider: ProviderStub,
    session: SessionProtocol,
    tool_choice: ToolChoice = "auto",
    response_format: Mapping[str, Any] | None = None,
    render_inputs: tuple[SupportsDataclass, ...] | None = None,
    throttle_policy: ThrottlePolicy | None = None,
    budget_tracker: BudgetTracker | None = None,
    deadline: Deadline | None = None,
) -> InnerLoop[object]:
    """Build an InnerLoop instance using the new API."""
    template = PromptTemplate(ns="tests", key="example")
    prompt = Prompt(template).bind(*(render_inputs or ()))
    # Enter prompt context for resource lifecycle
    prompt.resources.__enter__()

    inputs = InnerLoopInputs[object](
        adapter_name=DUMMY_ADAPTER_NAME,
        adapter=DummyAdapter(),
        prompt=prompt,
        prompt_name="example",
        rendered=rendered,
        render_inputs=prompt.params,
        initial_messages=[{"role": "system", "content": rendered.text}],
    )

    config = InnerLoopConfig(
        session=session,
        tool_choice=tool_choice,
        response_format=response_format,
        require_structured_output_text=False,
        call_provider=provider,
        select_choice=lambda response: response.choices[0],
        serialize_tool_message_fn=serialize_tool_message,
        throttle_policy=throttle_policy or new_throttle_policy(),
        budget_tracker=budget_tracker,
        deadline=deadline,
    )
    return InnerLoop[object](inputs=inputs, config=config)


def test_token_usage_from_payload_handles_missing_counts() -> None:
    payload = {
        "usage": {
            "input_tokens": None,
            "output_tokens": "unknown",
            "cached_tokens": False,
        }
    }

    assert token_usage_from_payload(payload) is None


@dataclass
class EchoParams:
    value: str


@dataclass
class EchoPayload:
    value: str


def echo_handler(
    params: EchoParams, *, context: ToolContext
) -> ToolResult[EchoPayload]:
    del context
    return ToolResult.ok(EchoPayload(value=params.value), message="initial")


def tool_rendered_prompt(tool: Tool[EchoParams, EchoPayload]) -> RenderedPrompt[object]:
    return RenderedPrompt(
        text="system",
        _tools=cast(
            tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...],
            (tool,),
        ),
    )


def build_tool_responses() -> Sequence[DummyResponse]:
    first = DummyResponse(
        [
            DummyChoice(
                DummyMessage(
                    content="",
                    tool_calls=(DummyToolCall("call-1", "echo", '{"value": "hello"}'),),
                )
            )
        ]
    )
    second = DummyResponse([DummyChoice(DummyMessage(content="All done"))])
    return (first, second)


@dataclass
class StructuredOutput:
    answer: str


class SessionStub(SessionProtocol):
    def __init__(self, *, dispatcher: Dispatcher | None = None) -> None:
        self.snapshots: list[SnapshotProtocol] = []
        self.restores: list[SnapshotProtocol] = []
        self._dispatcher = dispatcher or RecordingBus()

    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    def snapshot(
        self,
        *,
        tag: str | None = None,
        policies: frozenset[SlicePolicy] = DEFAULT_SNAPSHOT_POLICIES,
        include_all: bool = False,
    ) -> SnapshotProtocol:
        del tag, policies, include_all
        snapshot = Snapshot(created_at=datetime.now(UTC))
        self.snapshots.append(snapshot)
        return snapshot

    def restore(
        self, snapshot: SnapshotProtocol, *, preserve_logs: bool = True
    ) -> None:
        del preserve_logs
        self.restores.append(snapshot)

    def reset(self) -> None:
        pass


# Tests for the InnerLoop API


def test_inner_loop_success() -> None:
    """Test that InnerLoop produces the expected result."""
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)

    loop = build_inner_loop(rendered=rendered, provider=provider, session=session)
    response = loop.run()

    assert response.text == "Hello"
    assert response.output is None
    assert isinstance(bus.events[-1], PromptExecuted)
    assert provider.calls[0]["messages"][0]["content"] == "system"


def test_inner_loop_includes_usage_in_event() -> None:
    """Test that InnerLoop records token usage in events."""
    rendered = RenderedPrompt(text="system")
    responses = [
        DummyResponse(
            [DummyChoice(DummyMessage(content="Hello"))],
            usage={"input_tokens": 12, "output_tokens": 5, "cached_tokens": 3},
        )
    ]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)

    loop = build_inner_loop(rendered=rendered, provider=provider, session=session)
    _ = loop.run()

    prompt_event = cast(PromptExecuted, bus.events[-1])
    assert prompt_event.usage == TokenUsage(
        input_tokens=12, output_tokens=5, cached_tokens=3
    )


def test_inner_loop_publishes_prompt_rendered_event() -> None:
    """Test that InnerLoop publishes PromptRendered event with render inputs."""
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)
    params = EchoParams(value="hello")

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,
        session=session,
        render_inputs=(params,),
    )
    loop.run()

    assert provider.calls
    assert bus.events and isinstance(bus.events[0], PromptRendered)
    event = cast(PromptRendered, bus.events[0])
    assert event.rendered_prompt == "system"
    assert event.render_inputs == (params,)
    assert event.prompt_ns == "tests"
    assert event.prompt_key == "example"


def test_inner_loop_parses_structured_output() -> None:
    """Test that InnerLoop correctly parses structured output."""
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    responses = [
        DummyResponse(
            [
                DummyChoice(
                    DummyMessage(
                        content=None,
                        parsed={"answer": "42"},
                    )
                )
            ]
        )
    ]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,
        session=session,
    )
    response = loop.run()

    assert response.text is None
    assert response.output == StructuredOutput(answer="42")


def test_inner_loop_records_usage_to_budget_tracker() -> None:
    """Test that InnerLoop records token usage to budget tracker."""
    rendered = RenderedPrompt(text="system")
    responses = [
        DummyResponse(
            [DummyChoice(DummyMessage(content="Hello"))],
            usage={"input_tokens": 100, "output_tokens": 50, "cached_tokens": 10},
        )
    ]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)
    budget = Budget(max_total_tokens=1000)
    tracker = BudgetTracker(budget=budget)

    loop = build_inner_loop(
        rendered=rendered, provider=provider, session=session, budget_tracker=tracker
    )
    loop.run()

    consumed = tracker.consumed
    assert consumed.input_tokens == 100
    assert consumed.output_tokens == 50
    assert consumed.cached_tokens == 10


def test_run_inner_loop_function() -> None:
    """Test the run_inner_loop convenience function."""
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)

    template = PromptTemplate(ns="tests", key="example")
    prompt = Prompt(template)
    # Enter prompt context for resource lifecycle
    prompt.resources.__enter__()
    session: SessionProtocol = Session(bus=bus)

    inputs = InnerLoopInputs[object](
        adapter_name=DUMMY_ADAPTER_NAME,
        adapter=DummyAdapter(),
        prompt=prompt,
        prompt_name="example",
        rendered=rendered,
        render_inputs=prompt.params,
        initial_messages=[{"role": "system", "content": rendered.text}],
    )
    config = InnerLoopConfig(
        session=session,
        tool_choice="auto",
        response_format=None,
        require_structured_output_text=False,
        call_provider=provider,
        select_choice=lambda response: response.choices[0],
        serialize_tool_message_fn=serialize_tool_message,
    )

    response = run_inner_loop(inputs=inputs, config=config)

    assert response.text == "Hello"
    assert isinstance(bus.events[-1], PromptExecuted)


def test_inner_loop_raises_on_budget_exceeded() -> None:
    """Test that InnerLoop raises when budget is exceeded."""
    rendered = RenderedPrompt(text="system")
    responses = [
        DummyResponse(
            [DummyChoice(DummyMessage(content="Hello"))],
            usage={"input_tokens": 600, "output_tokens": 500, "cached_tokens": 0},
        )
    ]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)
    budget = Budget(max_total_tokens=500)
    tracker = BudgetTracker(budget=budget)

    loop = build_inner_loop(
        rendered=rendered, provider=provider, session=session, budget_tracker=tracker
    )

    with pytest.raises(PromptEvaluationError) as exc_info:
        loop.run()

    error = cast(PromptEvaluationError, exc_info.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_BUDGET
    assert "Budget exceeded" in str(error)


def test_inner_loop_requires_message_payload() -> None:
    """Test that InnerLoop raises when message is missing."""
    rendered = RenderedPrompt(text="system")

    class MissingMessageResponse(DummyResponse):
        def __init__(self) -> None:
            super().__init__(choices=[DummyChoice(DummyMessage(content=None))])
            self.choices[0].message = None

    provider = ProviderStub([MissingMessageResponse()])
    bus = RecordingBus()
    session = Session(bus=bus)
    loop = build_inner_loop(rendered=rendered, provider=provider, session=session)

    with pytest.raises(PromptEvaluationError):
        loop.run()


def test_inner_loop_executes_tool_calls() -> None:
    """Test that InnerLoop executes tool calls correctly."""
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo the provided value.",
        handler=echo_handler,
    )
    rendered = tool_rendered_prompt(tool)
    provider = ProviderStub(build_tool_responses())
    bus = RecordingBus()
    session = Session(bus=bus)

    loop = build_inner_loop(rendered=rendered, provider=provider, session=session)
    response = loop.run()

    assert response.text == "All done"
    tool_event = next(event for event in bus.events if isinstance(event, ToolInvoked))
    assert tool_event.name == "echo"


def test_inner_loop_continues_on_prompt_rendered_publish_failure() -> None:
    """Test that InnerLoop continues when PromptRendered publish fails."""
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus(fail_rendered=True)
    session = Session(bus=bus)
    params = EchoParams(value="blocked")

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,
        session=session,
        render_inputs=(params,),
    )
    response = loop.run()

    assert response.text == "Hello"
    assert provider.calls
    assert bus.events and isinstance(bus.events[0], PromptRendered)
    assert isinstance(bus.events[-1], PromptExecuted)


def test_inner_loop_raises_on_prompt_publish_failure() -> None:
    """Test that InnerLoop raises when PromptExecuted publish fails."""
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus(fail_prompt=True)
    session = Session(bus=bus)

    loop = build_inner_loop(rendered=rendered, provider=provider, session=session)

    with pytest.raises(ExceptionGroup) as exc_info:
        loop.run()

    assert "prompt dispatch failure" in str(exc_info.value)
    assert isinstance(bus.events[-1], PromptExecuted)
    assert provider.calls[0]["messages"][0]["content"] == "system"


def test_inner_loop_formats_tool_dispatch_failures() -> None:
    """Test that InnerLoop formats dispatch failures for tools."""
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo the provided value.",
        handler=echo_handler,
    )
    rendered = tool_rendered_prompt(tool)
    provider = ProviderStub(build_tool_responses())
    bus = RecordingBus(fail_tool=True)
    session = Session(bus=bus)

    loop = build_inner_loop(rendered=rendered, provider=provider, session=session)
    response = loop.run()

    assert response.text == "All done"
    tool_event = next(event for event in bus.events if isinstance(event, ToolInvoked))
    failure_message = tool_event.result.message
    assert "Reducer errors prevented applying tool result" in failure_message


def test_inner_loop_rolls_back_on_tool_dispatch_failure() -> None:
    """Test that InnerLoop rolls back session on tool dispatch failure."""
    tool = Tool[EchoParams, EchoPayload](
        name="echo",
        description="Echo the provided value.",
        handler=echo_handler,
    )
    rendered = tool_rendered_prompt(tool)
    provider = ProviderStub(build_tool_responses())
    session = SessionStub(dispatcher=RecordingBus(fail_tool=True))

    loop = build_inner_loop(
        rendered=rendered,
        provider=provider,
        session=session,
    )
    response = loop.run()

    assert response.text == "All done"
    assert session.snapshots and session.restores
    assert session.restores == session.snapshots


def test_inner_loop_includes_prompt_descriptor_in_event() -> None:
    """Test that InnerLoop includes descriptor in PromptRendered event."""
    descriptor = PromptDescriptor(ns="tests", key="example", sections=[], tools=[])
    rendered = RenderedPrompt(text="system", descriptor=descriptor)
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)

    loop = build_inner_loop(rendered=rendered, provider=provider, session=session)
    loop.run()

    rendered_event = next(
        event for event in bus.events if isinstance(event, PromptRendered)
    )
    assert rendered_event.descriptor is descriptor


def test_inner_loop_ensure_deadline_remaining_no_deadline() -> None:
    """Test that InnerLoop deadline check passes when no deadline is set."""
    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)

    loop = build_inner_loop(rendered=rendered, provider=provider, session=session)
    # This should not raise since there's no deadline
    loop._ensure_deadline_remaining("test", phase=PROMPT_EVALUATION_PHASE_REQUEST)


def test_inner_loop_raise_deadline_error() -> None:
    """Test that InnerLoop raises deadline error correctly."""
    from datetime import UTC, timedelta

    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))

    loop = build_inner_loop(
        rendered=rendered, provider=provider, session=session, deadline=deadline
    )

    with pytest.raises(PromptEvaluationError) as exc_info:
        loop._raise_deadline_error("test", phase=PROMPT_EVALUATION_PHASE_REQUEST)

    error = cast(PromptEvaluationError, exc_info.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_REQUEST


def test_inner_loop_ensure_deadline_remaining_expired(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    """Test that InnerLoop raises when deadline is expired."""
    from datetime import timedelta

    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))

    rendered = RenderedPrompt(text="system")
    responses = [DummyResponse([DummyChoice(DummyMessage(content="Hello"))])]
    provider = ProviderStub(responses)
    bus = RecordingBus()
    session = Session(bus=bus)

    loop = build_inner_loop(
        rendered=rendered, provider=provider, session=session, deadline=deadline
    )

    # Advance time past the deadline
    frozen_utcnow.advance(timedelta(seconds=10))

    with pytest.raises(PromptEvaluationError) as exc_info:
        loop._ensure_deadline_remaining("test", phase=PROMPT_EVALUATION_PHASE_REQUEST)

    error = cast(PromptEvaluationError, exc_info.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_REQUEST
