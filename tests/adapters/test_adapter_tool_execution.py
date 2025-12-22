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

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MethodType
from typing import Any, cast

import pytest

from tests.helpers import FrozenUtcNow

try:
    from tests.adapters._test_stubs import (
        DummyChoice,
        DummyMessage,
        DummyOpenAIClient,
        DummyResponse,
        DummyToolCall,
        RecordingCompletion,
        ResponseType,
        ToolParams,
        ToolPayload,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct invocation
    from ._test_stubs import (
        DummyChoice,
        DummyMessage,
        DummyOpenAIClient,
        DummyResponse,
        DummyToolCall,
        RecordingCompletion,
        ResponseType,
        ToolParams,
        ToolPayload,
    )

from weakincentives import DeadlineExceededError, ToolValidationError
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolHandler,
    ToolResult,
)
from weakincentives.prompt._visibility import SectionVisibility
from weakincentives.prompt.errors import VisibilityExpansionRequired
from weakincentives.runtime.events import InProcessDispatcher, ToolInvoked
from weakincentives.runtime.session import (
    ReducerEvent,
    Session,
    SessionProtocol,
    replace_latest,
)
from weakincentives.types.dataclass import SupportsDataclass


def _split_tool_message_content(content: str) -> tuple[str, str | None]:
    if "\n\n" in content:
        head, tail = content.split("\n\n", 1)
        return head, tail or None
    return content, None


@dataclass
class AdapterHarness:
    """Factory wrapper that builds adapters and exposes recorded requests."""

    name: str
    build: Callable[[Sequence[ResponseType]], tuple[Any, list[dict[str, object]]]]


@pytest.fixture(params=("openai", "litellm"))
def adapter_harness(request: pytest.FixtureRequest) -> AdapterHarness:
    if request.param == "openai":
        from weakincentives.adapters import openai as openai_module

        def build(
            responses: Sequence[ResponseType],
        ) -> tuple[Any, list[dict[str, object]]]:
            client = DummyOpenAIClient(responses)
            adapter = openai_module.OpenAIAdapter(
                model="gpt-test", client=cast(Any, client)
            )
            return adapter, client.responses.requests

        return AdapterHarness(name="openai", build=build)

    from weakincentives.adapters import litellm as litellm_module

    def build(responses: Sequence[ResponseType]) -> tuple[Any, list[dict[str, object]]]:
        completion = RecordingCompletion(responses)
        adapter = litellm_module.LiteLLMAdapter(
            model="gpt-test", completion=cast(Any, completion)
        )
        return adapter, completion.requests

    return AdapterHarness(name="litellm", build=build)


def _build_prompt(
    harness: AdapterHarness, tool: Tool[ToolParams, ToolPayload]
) -> PromptTemplate[Any]:
    return PromptTemplate(
        ns=f"tests/adapters/{harness.name}",
        key=f"{harness.name}-shared-tool-execution",
        name="search",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
                tools=[tool],
            )
        ],
    )


def _build_responses(
    *,
    tool_call: DummyToolCall,
    final_message: DummyMessage,
) -> list[DummyResponse]:
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second = DummyResponse([DummyChoice(final_message)])
    return [first, second]


def _evaluate_prompt(
    adapter: ProviderAdapter[Any],
    prompt_template: PromptTemplate[Any],
    *params: SupportsDataclass,
    **kwargs: object,
) -> PromptResponse[Any]:
    bound_prompt = Prompt(prompt_template).bind(*params)
    return adapter.evaluate(bound_prompt, **kwargs)


def _record_tool_events(bus: InProcessDispatcher) -> list[ToolInvoked]:
    events: list[ToolInvoked] = []

    def capture(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        events.append(event)

    bus.subscribe(ToolInvoked, capture)
    return events


def _second_tool_message(requests: list[dict[str, object]]) -> dict[str, object]:
    assert len(requests) >= 2
    payload = requests[1]
    message_list = cast(
        list[dict[str, object]],
        payload.get("input") or payload.get("messages"),
    )
    messages = message_list
    return messages[-1]


def _tool_message_parts(
    tool_message: Mapping[str, object],
) -> tuple[str, str | None]:
    raw = tool_message.get("output") or tool_message.get("content") or ""
    return _split_tool_message_content(cast(str, raw))


def test_adapter_tool_execution_success(adapter_harness: AdapterHarness) -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context
        return ToolResult(
            message="completed",
            value=ToolPayload(answer=params.query),
        )

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="All done", tool_calls=None),
    )
    adapter, requests = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    events = _record_tool_events(bus)

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="policies"))

    adapter.evaluate(
        bound_prompt,
        session=cast(SessionProtocol, session),
    )

    assert len(events) == 1
    invocation = events[0]
    assert invocation.result.message == "completed"
    assert invocation.result.success is True
    assert invocation.result.value == ToolPayload(answer="policies")

    tool_message = _second_tool_message(requests)
    message_text, rendered_text = _tool_message_parts(tool_message)
    assert message_text == "completed"
    assert rendered_text is not None
    assert json.loads(rendered_text) == {"answer": "policies"}


def test_adapter_tool_context_receives_deadline(
    adapter_harness: AdapterHarness,
) -> None:
    captured: list[Deadline | None] = []

    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        captured.append(context.deadline)
        return ToolResult(message="ok", value=ToolPayload(answer=params.query))

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)
    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="All done", tool_calls=None),
    )
    adapter, _ = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="policies"))

    adapter.evaluate(
        bound_prompt,
        session=cast(SessionProtocol, session),
        deadline=deadline,
    )

    assert captured == [deadline]


def test_adapter_tool_deadline_exceeded(
    adapter_harness: AdapterHarness,
) -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del params, context
        raise DeadlineExceededError("deadline hit")

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)
    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="All done", tool_calls=None),
    )
    adapter, _ = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="policies"))

    with pytest.raises(PromptEvaluationError) as excinfo:
        adapter.evaluate(
            bound_prompt,
            session=cast(SessionProtocol, session),
            deadline=deadline,
        )

    error = cast(PromptEvaluationError, excinfo.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_TOOL
    payload = error.provider_payload
    assert isinstance(payload, dict)
    assert payload.get("deadline_expires_at") == deadline.expires_at.isoformat()


def test_adapter_deadline_preflight_rejection(
    adapter_harness: AdapterHarness, frozen_utcnow: FrozenUtcNow
) -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del params, context
        return ToolResult(message="ok", value=ToolPayload(answer="policies"))

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)
    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="All done", tool_calls=None),
    )
    adapter, _ = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))
    frozen_utcnow.advance(timedelta(seconds=10))

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="policies"))

    with pytest.raises(PromptEvaluationError) as excinfo:
        adapter.evaluate(
            bound_prompt,
            session=cast(SessionProtocol, session),
            deadline=deadline,
        )

    error = cast(PromptEvaluationError, excinfo.value)
    assert error.phase == PROMPT_EVALUATION_PHASE_REQUEST
    assert error.provider_payload == {
        "deadline_expires_at": deadline.expires_at.isoformat()
    }


def test_adapter_tool_execution_validation_error(
    adapter_harness: AdapterHarness,
) -> None:
    def handler(_: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context
        raise ToolValidationError("invalid query")

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "invalid"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(
            content="Please provide a different query.", tool_calls=None
        ),
    )
    adapter, requests = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    events = _record_tool_events(bus)

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="invalid"))

    adapter.evaluate(
        bound_prompt,
        session=cast(SessionProtocol, session),
    )

    assert len(events) == 1
    invocation = events[0]
    assert invocation.result.message == "Tool validation failed: invalid query"
    assert invocation.result.success is False
    assert invocation.result.value is None

    tool_message = _second_tool_message(requests)
    message_text, rendered_text = _tool_message_parts(tool_message)
    assert message_text == "Tool validation failed: invalid query"
    assert rendered_text is None


def test_adapter_tool_execution_rejects_extra_arguments(
    adapter_harness: AdapterHarness,
) -> None:
    invoked = False

    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context, params
        nonlocal invoked
        invoked = True
        return ToolResult(
            message="completed",
            value=ToolPayload(answer="should not run"),
        )

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies", "extra": True}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(
            content="Please adjust the payload.", tool_calls=None
        ),
    )
    adapter, requests = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    events = _record_tool_events(bus)

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="policies"))

    adapter.evaluate(
        bound_prompt,
        session=cast(SessionProtocol, session),
    )

    assert invoked is False
    assert len(events) == 1
    invocation = events[0]
    assert invocation.result.success is False
    assert invocation.result.value is None
    assert "Extra keys not permitted" in invocation.result.message

    tool_message = _second_tool_message(requests)
    message_text, rendered_text = _tool_message_parts(tool_message)
    assert "Extra keys not permitted" in message_text
    assert rendered_text is None


def test_adapter_tool_execution_rejects_type_errors(
    adapter_harness: AdapterHarness,
) -> None:
    invoked = False

    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context, params
        nonlocal invoked
        invoked = True
        return ToolResult(
            message="completed",
            value=ToolPayload(answer="should not run"),
        )

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": None}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(
            content="Please adjust the payload.", tool_calls=None
        ),
    )
    adapter, requests = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    events = _record_tool_events(bus)

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="policies"))

    adapter.evaluate(
        bound_prompt,
        session=cast(SessionProtocol, session),
    )

    assert invoked is False
    assert len(events) == 1
    invocation = events[0]
    assert invocation.result.success is False
    assert invocation.result.value is None
    assert (
        invocation.result.message
        == "Tool validation failed: query: value cannot be None"
    )

    tool_message = _second_tool_message(requests)
    message_text, rendered_text = _tool_message_parts(tool_message)
    assert message_text == "Tool validation failed: query: value cannot be None"
    assert rendered_text is None


def test_adapter_tool_execution_unexpected_exception(
    adapter_harness: AdapterHarness,
) -> None:
    def handler(_: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context
        raise RuntimeError("handler crash")

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="Unable to use the tool.", tool_calls=None),
    )
    adapter, requests = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    events = _record_tool_events(bus)

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="policies"))

    adapter.evaluate(
        bound_prompt,
        session=cast(SessionProtocol, session),
    )

    assert len(events) == 1
    invocation = events[0]
    assert (
        invocation.result.message
        == "Tool 'search_notes' execution failed: handler crash"
    )
    assert invocation.result.success is False
    assert invocation.result.value is None

    tool_message = _second_tool_message(requests)
    message_text, rendered_text = _tool_message_parts(tool_message)
    assert message_text == "Tool 'search_notes' execution failed: handler crash"
    assert rendered_text is None


def test_adapter_tool_execution_rolls_back_session(
    adapter_harness: AdapterHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context
        return ToolResult(
            message="completed",
            value=ToolPayload(answer=params.query),
        )

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="All done", tool_calls=None),
    )
    adapter, requests = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)
    session[ToolPayload].register(ToolPayload, replace_latest)
    session[ToolPayload].seed((ToolPayload(answer="baseline"),))

    original_dispatch = session._dispatch_data_event

    def failing_dispatch(
        self: Session,
        data_type: type[SupportsDataclass],
        event: ReducerEvent,
    ) -> None:
        if data_type is not ToolPayload:
            original_dispatch(data_type, event)
            return
        original_dispatch(data_type, event)
        raise RuntimeError("Reducer crashed")

    monkeypatch.setattr(
        session,
        "_dispatch_data_event",
        MethodType(failing_dispatch, session),
    )

    events = _record_tool_events(bus)

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="policies"))

    adapter.evaluate(
        bound_prompt,
        session=session,
    )

    assert len(events) == 1
    invocation = events[0]
    assert invocation.result.message.startswith(
        "Reducer errors prevented applying tool result:"
    )
    assert invocation.result.success is True
    assert invocation.result.value == ToolPayload(answer="policies")

    latest_payload = session[ToolPayload].latest()
    assert latest_payload == ToolPayload(answer="baseline")

    tool_message = _second_tool_message(requests)
    message_text, rendered_text = _tool_message_parts(tool_message)
    assert message_text == invocation.result.message
    assert rendered_text is not None
    assert json.loads(rendered_text) == {"answer": "policies"}


def test_adapter_tool_visibility_expansion_propagates(
    adapter_harness: AdapterHarness,
) -> None:
    """VisibilityExpansionRequired is propagated to caller (not wrapped as failed tool)."""

    def handler(_: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context
        raise VisibilityExpansionRequired(
            "Section expansion requested",
            requested_overrides={("docs",): SectionVisibility.FULL},
            reason="Need documentation details",
            section_keys=("docs",),
        )

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )
    prompt_template = _build_prompt(adapter_harness, tool)
    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "docs"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="Done", tool_calls=None),
    )
    adapter, _ = adapter_harness.build(responses)

    bus = InProcessDispatcher()
    session = Session(bus=bus)

    bound_prompt = Prompt(prompt_template).bind(ToolParams(query="docs"))

    with pytest.raises(VisibilityExpansionRequired) as exc_info:
        adapter.evaluate(
            bound_prompt,
            session=cast(SessionProtocol, session),
        )

    exc = cast(VisibilityExpansionRequired, exc_info.value)
    assert exc.section_keys == ("docs",)
    assert exc.reason == "Need documentation details"
    assert ("docs",) in exc.requested_overrides


def test_tool_receives_filesystem_from_workspace_section(
    adapter_harness: AdapterHarness,
) -> None:
    """Test that tool handlers receive filesystem from workspace sections."""
    from weakincentives.contrib.tools.filesystem_memory import InMemoryFilesystem
    from weakincentives.contrib.tools.vfs import VfsToolsSection
    from weakincentives.filesystem import Filesystem

    captured_filesystem: list[Filesystem | None] = []

    def tool_handler(
        params: ToolParams, *, context: ToolContext
    ) -> ToolResult[ToolPayload]:
        captured_filesystem.append(context.filesystem)
        return ToolResult(
            message=f"Searched for {params.query}",
            value=ToolPayload(url="http://example.com", snippet="Found it"),
        )

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )

    bus = InProcessDispatcher()
    session = Session(bus=bus)

    # Create a workspace section with VFS filesystem
    workspace_section = VfsToolsSection(session=session)

    prompt_template = PromptTemplate(
        ns=f"test/{adapter_harness.name}",
        key="test-filesystem-pass-through",
        sections=[
            workspace_section,
            MarkdownSection(
                title="Search",
                key="search",
                template="Use the search tool.",
                tools=[tool],
            ),
        ],
    )

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "docs"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="Done", tool_calls=None),
    )
    adapter, _ = adapter_harness.build(responses)

    prompt = Prompt(prompt_template)

    _ = adapter.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
    )

    assert len(captured_filesystem) == 1
    assert captured_filesystem[0] is not None
    assert isinstance(captured_filesystem[0], InMemoryFilesystem)


def test_budget_tracker_passed_to_tool_context_via_resources(
    adapter_harness: AdapterHarness,
) -> None:
    """Verify budget_tracker is accessible via context.budget_tracker in tool handlers."""
    from weakincentives.budget import Budget, BudgetTracker

    captured_tracker: list[BudgetTracker | None] = []

    def tool_handler(
        params: ToolParams, *, context: ToolContext
    ) -> ToolResult[ToolPayload]:
        captured_tracker.append(context.budget_tracker)
        return ToolResult(
            message=f"Searched for {params.query}",
            value=ToolPayload(url="http://example.com", snippet="Found it"),
        )

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )

    bus = InProcessDispatcher()
    session = Session(bus=bus)

    prompt_template = PromptTemplate(
        ns=f"test/{adapter_harness.name}",
        key="test-budget-tracker-pass-through",
        sections=[
            MarkdownSection(
                title="Search",
                key="search",
                template="Use the search tool.",
                tools=[tool],
            ),
        ],
    )

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "docs"}),
    )
    responses = _build_responses(
        tool_call=tool_call,
        final_message=DummyMessage(content="Done", tool_calls=None),
    )
    adapter, _ = adapter_harness.build(responses)

    prompt = Prompt(prompt_template)
    budget = Budget(max_total_tokens=10000)
    tracker = BudgetTracker(budget=budget)

    _ = adapter.evaluate(
        prompt,
        session=cast(SessionProtocol, session),
        budget_tracker=tracker,
    )

    assert len(captured_tracker) == 1
    assert captured_tracker[0] is tracker
