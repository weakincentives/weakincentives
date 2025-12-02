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

import importlib
import json
import sys
import types
from collections.abc import Mapping
from importlib import import_module as std_import_module
from types import MethodType
from typing import Any, Literal, TypeVar, cast

import pytest

from weakincentives.adapters import (
    LITELLM_ADAPTER_NAME,
    PromptEvaluationError,
    PromptResponse,
    shared,
)
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.prompt.structured_output import (
    ARRAY_WRAPPER_KEY,
    StructuredOutputConfig,
)

try:
    from tests.adapters._test_stubs import (
        DummyChoice,
        DummyMessage,
        DummyResponse,
        DummyToolCall,
        GreetingParams,
        MappingResponse,
        OptionalParams,
        OptionalPayload,
        RecordingCompletion,
        SimpleResponse,
        StructuredAnswer,
        ToolParams,
        ToolPayload,
        WeirdResponse,
        simple_handler,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct invocation
    from ._test_stubs import (
        DummyChoice,
        DummyMessage,
        DummyResponse,
        DummyToolCall,
        GreetingParams,
        MappingResponse,
        OptionalParams,
        OptionalPayload,
        RecordingCompletion,
        SimpleResponse,
        StructuredAnswer,
        ToolParams,
        ToolPayload,
        WeirdResponse,
        simple_handler,
    )
from tests.helpers.events import NullEventBus
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    SupportsDataclass,
    Tool,
    ToolContext,
    ToolHandler,
    ToolResult,
)
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.runtime.events import (
    EventBus,
    HandlerFailure,
    InProcessEventBus,
    PromptExecuted,
    ToolInvoked,
)
from weakincentives.runtime.session import (
    ReducerEvent,
    Session,
    replace_latest,
    select_latest,
)
from weakincentives.tools import ToolValidationError

MODULE_PATH = "weakincentives.adapters.litellm"
PROMPT_NS = "tests/adapters/litellm"


def _split_tool_message_content(content: str) -> tuple[str, str | None]:
    if "\n\n" in content:
        message, remainder = content.split("\n\n", 1)
        return message, remainder or None
    return content, None


def _reload_module() -> types.ModuleType:
    return importlib.reload(std_import_module(MODULE_PATH))


OutputT = TypeVar("OutputT")


def _evaluate_with_bus(
    adapter: ProviderAdapter[OutputT],
    prompt: Prompt[OutputT],
    *params: SupportsDataclass,
    bus: EventBus | None = None,
    session: SessionProtocol | None = None,
) -> PromptResponse[OutputT]:
    target_bus = bus or NullEventBus()
    target_session = (
        session
        if session is not None
        else cast(SessionProtocol, Session(bus=target_bus))
    )
    return adapter.evaluate(
        prompt,
        *params,
        bus=target_bus,
        session=target_session,
    )


def test_create_litellm_completion_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    def fail_import(name: str, package: str | None = None) -> types.ModuleType:
        if name == "litellm":
            raise ModuleNotFoundError("No module named 'litellm'")
        return std_import_module(name, package)

    monkeypatch.setattr(module, "import_module", fail_import)

    with pytest.raises(RuntimeError) as err:
        module.create_litellm_completion()

    message = str(err.value)
    assert "uv sync --extra litellm" in message
    assert "pip install weakincentives[litellm]" in message


def test_create_litellm_completion_wraps_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    captured_kwargs: list[dict[str, object]] = []

    class DummyLiteLLM(types.SimpleNamespace):
        def __init__(self) -> None:
            super().__init__()
            self.captured_kwargs = captured_kwargs

        def completion(self, **kwargs: object) -> DummyResponse:
            self.captured_kwargs.append(dict(kwargs))
            message = DummyMessage(content="Hello", tool_calls=None)
            return DummyResponse([DummyChoice(message)])

    dummy_module = cast(Any, DummyLiteLLM())
    monkeypatch.setitem(sys.modules, "litellm", dummy_module)

    completion = module.create_litellm_completion(api_key="secret")
    completion(model="gpt", messages=[{"role": "system", "content": "hi"}])

    assert captured_kwargs == [
        {
            "api_key": "secret",
            "model": "gpt",
            "messages": [{"role": "system", "content": "hi"}],
        }
    ]


def test_create_litellm_completion_returns_direct_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    class DummyLiteLLM(types.SimpleNamespace):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def completion(self, **kwargs: object) -> DummyResponse:
            self.calls.append(dict(kwargs))
            message = DummyMessage(content="Hi", tool_calls=None)
            return DummyResponse([DummyChoice(message)])

    dummy_module = cast(Any, DummyLiteLLM())
    monkeypatch.setitem(sys.modules, "litellm", dummy_module)

    completion = module.create_litellm_completion()
    result = completion(model="gpt", messages=[{"role": "system", "content": "hi"}])

    assert isinstance(result, DummyResponse)
    assert dummy_module.calls == [
        {"model": "gpt", "messages": [{"role": "system", "content": "hi"}]}
    ]


def test_litellm_adapter_constructs_completion_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-greeting",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyResponse(
        [DummyChoice(DummyMessage(content="Hello!", tool_calls=None))]
    )
    completion = RecordingCompletion([response])
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> RecordingCompletion:
        captured_kwargs.append(dict(kwargs))
        return completion

    monkeypatch.setattr(module, "create_litellm_completion", fake_factory)

    adapter = module.LiteLLMAdapter(
        model="gpt-test",
        completion_kwargs={"api_key": "secret-key"},
    )

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.text == "Hello!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_litellm_adapter_supports_custom_completion_factory() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-greeting",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyResponse(
        [DummyChoice(DummyMessage(content="Hello again!", tool_calls=None))]
    )
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> RecordingCompletion:
        captured_kwargs.append(dict(kwargs))
        return RecordingCompletion([response])

    adapter = module.LiteLLMAdapter(
        model="gpt-test",
        completion_factory=fake_factory,
        completion_kwargs={"api_key": "secret-key"},
    )

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.text == "Hello again!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_litellm_adapter_rejects_completion_kwargs_with_explicit_completion() -> None:
    module = cast(Any, _reload_module())
    completion = RecordingCompletion([])

    with pytest.raises(ValueError):
        module.LiteLLMAdapter(
            model="gpt-test",
            completion=completion,
            completion_kwargs={"api_key": "secret"},
        )


def test_litellm_adapter_rejects_completion_factory_with_explicit_completion() -> None:
    module = cast(Any, _reload_module())
    completion = RecordingCompletion([])

    with pytest.raises(ValueError):
        module.LiteLLMAdapter(
            model="gpt-test",
            completion=completion,
            completion_factory=lambda **_: completion,
        )


def test_litellm_adapter_returns_plain_text_response() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-plain",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyResponse(
        [DummyChoice(DummyMessage(content="Hello, Sam!", tool_calls=None))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.prompt_name == "greeting"
    assert result.text == "Hello, Sam!"
    assert result.output is None

    request = completion.requests[0]
    messages = cast(list[dict[str, Any]], request["messages"])
    assert messages[0]["role"] == "system"
    assert str(messages[0]["content"]).startswith("## 1. Greeting")
    assert "tools" not in request


def test_litellm_adapter_executes_tools_and_parses_output() -> None:
    module = cast(Any, _reload_module())

    calls: list[str] = []

    def handler(params: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context
        calls.append(params.query)
        payload = ToolPayload(answer=f"Result for {params.query}")
        return ToolResult(message="completed", value=payload)

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-structured-success",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second_message = DummyMessage(
        content=json.dumps({"answer": "Policy summary"}), tool_calls=None
    )
    second = DummyResponse([DummyChoice(second_message)])
    completion = RecordingCompletion([first, second])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Policy summary")
    assert calls == ["policies"]

    first_request = completion.requests[0]
    assert "response_format" in first_request
    tools = cast(list[dict[str, Any]], first_request["tools"])
    function_spec = cast(dict[str, Any], tools[0]["function"])
    assert function_spec["name"] == "search_notes"
    assert first_request.get("tool_choice") == "auto"

    second_request = completion.requests[1]
    second_messages = cast(list[dict[str, Any]], second_request["messages"])
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    message_text, rendered_text = _split_tool_message_content(tool_message["content"])
    assert message_text == "completed"
    assert rendered_text is not None
    assert json.loads(rendered_text) == {"answer": "Policy summary"}


def test_litellm_adapter_rolls_back_session_on_publish_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-session-rollback",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second_message = DummyMessage(
        content=json.dumps({"answer": "Policy summary"}), tool_calls=None
    )
    second = DummyResponse([DummyChoice(second_message)])
    completion = RecordingCompletion([first, second])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    session.register_reducer(ToolPayload, replace_latest)
    session.seed_slice(ToolPayload, (ToolPayload(answer="baseline"),))

    tool_events: list[ToolInvoked] = []
    prompt_events: list[PromptExecuted] = []

    def record_tool_event(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    def record_prompt_event(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        prompt_events.append(event)

    bus.subscribe(ToolInvoked, record_tool_event)
    bus.subscribe(PromptExecuted, record_prompt_event)

    original_dispatch = session._dispatch_data_event

    def failing_dispatch(
        self: Session,
        data_type: type[SupportsDataclass],
        event: ReducerEvent,
    ) -> None:
        original_dispatch(data_type, event)
        raise RuntimeError("Reducer crashed")

    monkeypatch.setattr(
        session,
        "_dispatch_data_event",
        MethodType(failing_dispatch, session),
    )

    with pytest.raises(ExceptionGroup) as exc_info:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
            bus=bus,
            session=session,
        )

    assert "Reducer crashed" in str(exc_info.value)

    assert tool_events
    tool_event = tool_events[0]
    assert tool_event.result.message.startswith(
        "Reducer errors prevented applying tool result:"
    )
    assert "Reducer crashed" in tool_event.result.message

    latest_payload = select_latest(session, ToolPayload)
    assert latest_payload == ToolPayload(answer="baseline")

    assert prompt_events
    prompt_result = prompt_events[0].result
    assert prompt_result.output == StructuredAnswer(answer="Policy summary")


def test_litellm_format_publish_failures_handles_defaults() -> None:
    module = cast(Any, _reload_module())

    failure = HandlerFailure(handler=lambda _: None, error=RuntimeError(""))
    message = module.format_publish_failures((failure,))
    assert message == "Reducer errors prevented applying tool result: RuntimeError"
    assert (
        module.format_publish_failures(())
        == "Reducer errors prevented applying tool result."
    )


def test_litellm_adapter_uses_parsed_payload_when_available() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-structured-parsed",
        name="structured",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return the structured result only.",
            )
        ],
    )

    message = DummyMessage(content=None, tool_calls=None, parsed={"answer": "Parsed"})
    response = DummyResponse([DummyChoice(message)])
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Parsed")

    request = completion.requests[0]
    assert "response_format" in request


def test_litellm_adapter_includes_response_format_for_array_outputs() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[list[StructuredAnswer]](
        ns=PROMPT_NS,
        key="litellm-structured-schema-array",
        name="structured_list",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return a list of answers for ${query}.",
            )
        ],
    )

    payload = [{"answer": "First"}, {"answer": "Second"}]
    message = DummyMessage(
        content=json.dumps(payload),
        tool_calls=None,
    )
    response = DummyResponse([DummyChoice(message)])
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert isinstance(result.output, list)
    assert [item.answer for item in result.output] == ["First", "Second"]

    request = completion.requests[0]
    response_format = cast(dict[str, Any], request["response_format"])
    json_schema = cast(dict[str, Any], response_format["json_schema"])
    schema_payload = cast(dict[str, Any], json_schema["schema"])
    properties = cast(dict[str, Any], schema_payload["properties"])
    assert ARRAY_WRAPPER_KEY in properties
    items_schema = cast(dict[str, Any], properties[ARRAY_WRAPPER_KEY])
    assert items_schema.get("type") == "array"
    assert items_schema.get("items", {}).get("type") == "object"


def test_litellm_adapter_relaxes_forced_tool_choice_after_first_call() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-tools-relaxed",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    final_message = DummyMessage(content="All done", tool_calls=None)
    second = DummyResponse([DummyChoice(final_message)])
    completion = RecordingCompletion([first, second])

    forced_choice: Mapping[str, object] = {
        "type": "function",
        "function": {"name": tool.name},
    }
    adapter = module.LiteLLMAdapter(
        model="gpt-test",
        completion=completion,
        tool_choice=forced_choice,
    )

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text == "All done"
    assert len(completion.requests) == 2
    assert completion.requests[0].get("tool_choice") == forced_choice
    assert completion.requests[1].get("tool_choice") == "auto"


def test_litellm_adapter_handles_tool_call_without_arguments() -> None:
    module = cast(Any, _reload_module())

    recorded: list[str] = []

    def handler(
        params: OptionalParams, *, context: ToolContext
    ) -> ToolResult[OptionalPayload]:
        del context
        recorded.append(params.query)
        payload = OptionalPayload(value=params.query)
        return ToolResult(message="used default", value=payload)

    tool_handler: ToolHandler[OptionalParams, OptionalPayload] = handler

    tool = Tool[OptionalParams, OptionalPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-tool-no-args",
        name="search",
        sections=[
            MarkdownSection[OptionalParams](
                title="Task",
                key="task",
                template="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = DummyToolCall(call_id="call_1", name="search_notes", arguments=None)
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    final_message = DummyMessage(content="All done", tool_calls=None)
    second = DummyResponse([DummyChoice(final_message)])
    completion = RecordingCompletion([first, second])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        OptionalParams(),
        bus=NullEventBus(),
    )

    assert result.text == "All done"
    assert recorded == ["default"]


def test_litellm_adapter_surfaces_tool_validation_errors() -> None:
    module = cast(Any, _reload_module())

    def handler(_: ToolParams, *, context: ToolContext) -> ToolResult[ToolPayload]:
        del context
        raise ToolValidationError("invalid query")

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-tool-validation",
        name="search-validation",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "invalid"}),
    )
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second = DummyResponse(
        [
            DummyChoice(
                DummyMessage(
                    content="Please provide a different query.", tool_calls=None
                )
            )
        ]
    )
    completion = RecordingCompletion([first, second])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    bus = InProcessEventBus()
    tool_events: list[ToolInvoked] = []

    def record(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="invalid"),
        bus=bus,
    )

    assert result.text == "Please provide a different query."
    assert result.output is None
    assert len(tool_events) == 1
    event = tool_events[0]
    assert event.result.message == "Tool validation failed: invalid query"
    assert event.result.success is False
    assert event.result.value is None
    assert event.call_id == "call_1"

    second_request = completion.requests[1]
    second_messages = cast(list[dict[str, Any]], second_request["messages"])
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    message_text, rendered_text = _split_tool_message_content(tool_message["content"])
    assert message_text == "Tool validation failed: invalid query"
    assert rendered_text is None


def test_litellm_adapter_surfaces_tool_type_errors() -> None:
    module = cast(Any, _reload_module())

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

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-tool-type-error",
        name="search-type-error",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": None}),
    )
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second = DummyResponse(
        [
            DummyChoice(
                DummyMessage(content="Please adjust the payload.", tool_calls=None)
            )
        ]
    )
    completion = RecordingCompletion([first, second])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    bus = InProcessEventBus()
    tool_events: list[ToolInvoked] = []

    def record(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=bus,
    )

    assert result.text == "Please adjust the payload."
    assert result.output is None
    assert invoked is False
    assert len(tool_events) == 1
    event = tool_events[0]
    assert event.result.message == "Tool validation failed: query: value cannot be None"
    assert event.result.success is False
    assert event.result.value is None
    assert event.call_id == "call_1"

    second_request = completion.requests[1]
    second_messages = cast(list[dict[str, Any]], second_request["messages"])
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    message_text, rendered_text = _split_tool_message_content(tool_message["content"])
    assert message_text == "Tool validation failed: query: value cannot be None"
    assert rendered_text is None


def test_litellm_adapter_reads_output_json_content_blocks() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-structured-json-block",
        name="structured",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return the structured result only.",
            )
        ],
    )

    class JsonBlock:
        def __init__(self, payload: dict[str, object]) -> None:
            self.type = "output_json"
            self.json = payload

    content_blocks = [
        {"type": "output_json", "json": {"answer": "Block"}},
        JsonBlock({"answer": "Attribute"}),
    ]
    message = DummyMessage(content=content_blocks, tool_calls=None)
    response = DummyResponse([DummyChoice(message)])
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Block")


def test_litellm_adapter_emits_events_during_evaluation() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-structured-events",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second_message = DummyMessage(
        content=json.dumps({"answer": "Policy summary"}), tool_calls=None
    )
    second = DummyResponse([DummyChoice(second_message)])
    completion = RecordingCompletion([first, second])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    bus = InProcessEventBus()
    tool_events: list[ToolInvoked] = []
    prompt_events: list[PromptExecuted] = []

    def record_tool_event(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    def record_prompt_event(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        prompt_events.append(event)

    bus.subscribe(ToolInvoked, record_tool_event)
    bus.subscribe(PromptExecuted, record_prompt_event)
    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=bus,
    )

    assert len(tool_events) == 1
    tool_event = tool_events[0]
    assert tool_event.prompt_name == "search"
    assert tool_event.adapter == "litellm"
    assert tool_event.name == "search_notes"
    assert tool_event.call_id == "call_1"

    assert len(prompt_events) == 1
    prompt_event = prompt_events[0]
    assert prompt_event.prompt_name == "search"
    assert prompt_event.adapter == "litellm"
    assert prompt_event.result is result


def test_litellm_adapter_raises_when_tool_handler_missing() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=None,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-handler-missing",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    response = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == PROMPT_EVALUATION_PHASE_TOOL


def test_litellm_adapter_raises_when_tool_not_registered() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-missing-tool",
        name="search",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
                tools=[],
            )
        ],
    )

    tool_call = DummyToolCall(
        call_id="call_1",
        name="missing_tool",
        arguments=json.dumps({"query": "policies"}),
    )
    response = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == PROMPT_EVALUATION_PHASE_TOOL


def test_litellm_adapter_handles_invalid_tool_params() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-invalid-tool-params",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({}),
    )
    responses = [
        DummyResponse(
            [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
        ),
        DummyResponse([DummyChoice(DummyMessage(content="Try again"))]),
    ]
    completion = RecordingCompletion(responses)
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    bus = InProcessEventBus()
    tool_events: list[ToolInvoked] = []
    bus.subscribe(
        ToolInvoked, lambda event: tool_events.append(cast(ToolInvoked, event))
    )
    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=bus,
    )

    assert result.text == "Try again"
    assert len(tool_events) == 1
    invocation = tool_events[0]
    assert invocation.result.success is False
    assert invocation.result.value is None
    assert "Missing required field" in invocation.result.message


def test_litellm_adapter_records_handler_failures() -> None:
    module = cast(Any, _reload_module())

    def failing_handler(
        params: ToolParams, *, context: ToolContext
    ) -> ToolResult[ToolPayload]:
        del context
        raise RuntimeError("boom")

    tool_handler = cast(ToolHandler[ToolParams, ToolPayload], failing_handler)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=tool_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-handler-failure",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second = DummyResponse(
        [
            DummyChoice(
                DummyMessage(
                    content="Please provide a different approach.", tool_calls=None
                )
            )
        ]
    )
    completion = RecordingCompletion([first, second])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    bus = InProcessEventBus()
    tool_events: list[ToolInvoked] = []

    def record(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
        bus=bus,
    )

    assert result.text == "Please provide a different approach."
    assert result.output is None
    assert len(tool_events) == 1
    event = tool_events[0]
    assert event.result.success is False
    assert event.result.value is None
    assert "execution failed: boom" in event.result.message

    second_request = completion.requests[1]
    second_messages = cast(list[dict[str, Any]], second_request["messages"])
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    message_text, rendered_text = _split_tool_message_content(tool_message["content"])
    assert message_text.endswith("execution failed: boom")
    assert rendered_text is None


def test_litellm_adapter_records_provider_payload_from_mapping() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-provider-payload",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    mapping_response = MappingResponse(
        [DummyChoice(DummyMessage(content="Hello!", tool_calls=None))]
    )
    completion = RecordingCompletion([mapping_response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert not hasattr(result, "provider_payload")


def test_litellm_adapter_ignores_non_mapping_model_dump() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-weird-dump",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = WeirdResponse(
        [DummyChoice(DummyMessage(content="Hello!", tool_calls=None))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert not hasattr(result, "provider_payload")


def test_litellm_adapter_handles_response_without_model_dump() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-simple-response",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = SimpleResponse(
        [DummyChoice(DummyMessage(content="Hello!", tool_calls=None))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert not hasattr(result, "provider_payload")


@pytest.mark.parametrize(
    "arguments_json",
    ["{", json.dumps("not a dict")],
)
def test_litellm_adapter_rejects_bad_tool_arguments(arguments_json: str) -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-bad-tool-arguments",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=arguments_json,
    )
    response = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == PROMPT_EVALUATION_PHASE_TOOL


def test_litellm_adapter_propagates_parse_errors_for_structured_output() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-structured-error",
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

    tool_call = DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    final_message = DummyMessage(content="not json", tool_calls=None)
    second = DummyResponse([DummyChoice(final_message)])
    completion = RecordingCompletion([first, second])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == PROMPT_EVALUATION_PHASE_RESPONSE


def test_litellm_adapter_raises_when_structured_output_missing() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-structured-missing",
        name="structured",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return the structured result only.",
            )
        ],
    )

    message = DummyMessage(content="", tool_calls=None)
    response = DummyResponse([DummyChoice(message)])
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    exc = err.value
    assert isinstance(exc, PromptEvaluationError)
    assert exc.phase == PROMPT_EVALUATION_PHASE_RESPONSE
    assert "structured output" in str(exc)


def test_litellm_adapter_raises_on_invalid_parsed_payload() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-structured-parsed-error",
        name="structured",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return the structured result only.",
            )
        ],
    )

    message = DummyMessage(content=None, tool_calls=None, parsed="not-a-mapping")
    response = DummyResponse([DummyChoice(message)])
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    exc = err.value
    assert isinstance(exc, PromptEvaluationError)
    assert exc.phase == PROMPT_EVALUATION_PHASE_RESPONSE


def test_litellm_message_text_content_handles_structured_parts() -> None:
    module = cast(Any, _reload_module())

    mapping_parts = [{"type": "output_text", "text": "Hello"}]
    assert module.message_text_content(mapping_parts) == "Hello"

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    assert module.message_text_content([TextBlock("World")]) == "World"
    assert module.message_text_content(123) == "123"
    assert shared._content_part_text(None) == ""
    assert shared._content_part_text({"type": "output_text", "text": 123}) == ""

    class BadTextBlock:
        def __init__(self) -> None:
            self.type = "text"
            self.text = 123

    assert shared._content_part_text(BadTextBlock()) == ""


def test_litellm_extract_parsed_content_handles_attribute_blocks() -> None:
    module = cast(Any, _reload_module())

    class JsonBlock:
        def __init__(self, payload: dict[str, object]) -> None:
            self.type = "output_json"
            self.json = payload

    block = JsonBlock({"answer": "attribute"})
    message = DummyMessage(content=[block], tool_calls=None)

    parsed = module.extract_parsed_content(message)

    assert parsed == {"answer": "attribute"}
    assert shared._parsed_payload_from_part({"type": "other"}) is None

    class OtherBlock:
        def __init__(self) -> None:
            self.type = "other"
            self.json = {"answer": "ignored"}

    assert shared._parsed_payload_from_part(OtherBlock()) is None


def test_litellm_parse_schema_constrained_payload_unwraps_wrapped_array() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[list[StructuredAnswer]](
        ns=PROMPT_NS,
        key="litellm-structured-schema-array-wrapped",
        name="structured_list",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return a list of answers for ${query}.",
            )
        ],
    )

    rendered = prompt.render(ToolParams(query="policies"))

    payload = {ARRAY_WRAPPER_KEY: [{"answer": "Ready"}]}

    parsed = module.parse_schema_constrained_payload(payload, rendered)

    assert isinstance(parsed, list)
    assert parsed[0].answer == "Ready"

    with pytest.raises(TypeError):
        module.parse_schema_constrained_payload({"wrong": []}, rendered)

    with pytest.raises(TypeError):
        module.parse_schema_constrained_payload(["oops"], rendered)


def test_litellm_parse_schema_constrained_payload_handles_object_container() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-structured-schema",
        name="structured",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Summarize ${query} as JSON.",
            )
        ],
    )

    rendered = prompt.render(ToolParams(query="policies"))

    parsed = module.parse_schema_constrained_payload({"answer": "Ready"}, rendered)

    assert parsed.answer == "Ready"

    with pytest.raises(TypeError):
        module.parse_schema_constrained_payload("oops", rendered)


def test_litellm_build_json_schema_response_format_returns_none_for_plain_prompt() -> (
    None
):
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="litellm-plain",
        name="plain",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Say hello to ${query}.",
            )
        ],
    )

    rendered = prompt.render(ToolParams(query="world"))

    response_format = module.build_json_schema_response_format(rendered, "plain")

    assert response_format is None


def test_litellm_parse_schema_constrained_payload_requires_structured_prompt() -> None:
    module = cast(Any, _reload_module())

    rendered = RenderedPrompt(text="")

    with pytest.raises(TypeError):
        module.parse_schema_constrained_payload({}, rendered)


def test_litellm_parse_schema_constrained_payload_rejects_non_sequence_arrays() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[list[StructuredAnswer]](
        ns=PROMPT_NS,
        key="litellm-structured-schema-array-non-seq",
        name="structured_list",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return a list of answers for ${query}.",
            )
        ],
    )

    rendered = prompt.render(ToolParams(query="policies"))

    with pytest.raises(TypeError):
        module.parse_schema_constrained_payload("oops", rendered)


def test_litellm_parse_schema_constrained_payload_rejects_unknown_container() -> None:
    module = cast(Any, _reload_module())

    rendered = RenderedPrompt(
        text="",
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredAnswer,
            container=cast(Literal["object", "array"], "invalid"),
            allow_extra_keys=False,
        ),
    )

    with pytest.raises(TypeError):
        module.parse_schema_constrained_payload({}, rendered)


def test_litellm_adapter_delegates_to_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="litellm-shared-runner",
        name="shared-runner",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    sentinel = PromptResponse(
        prompt_name="shared-runner",
        text="sentinel",
        output=None,
    )

    captured: dict[str, Any] = {}

    def fake_run_conversation(**kwargs: object) -> PromptResponse[StructuredAnswer]:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "run_conversation", fake_run_conversation)

    message = DummyMessage(content="hi")
    completion = RecordingCompletion([DummyResponse([DummyChoice(message)])])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    params = GreetingParams(user="Ari")
    result = _evaluate_with_bus(adapter, prompt, params)

    assert result is sentinel
    assert captured["adapter_name"] == LITELLM_ADAPTER_NAME
    assert captured["prompt_name"] == "shared-runner"

    expected_rendered = prompt.render(params, inject_output_instructions=False)
    assert captured["rendered"] == expected_rendered
    assert captured["render_inputs"] == (params,)
    assert captured["initial_messages"] == [
        {"role": "system", "content": expected_rendered.text}
    ]

    expected_response_format = module.build_json_schema_response_format(
        expected_rendered, "shared-runner"
    )
    config = captured["config"]
    assert isinstance(config, module.ConversationConfig)
    assert config.response_format == expected_response_format
    assert config.require_structured_output_text is True
    assert config.serialize_tool_message_fn is module.serialize_tool_message

    call_provider = config.call_provider
    select_choice = config.select_choice
    assert callable(call_provider)
    assert callable(select_choice)

    response = call_provider(
        [{"role": "system", "content": "hi"}],
        [],
        None,
        expected_response_format,
    )
    request_payload = cast(dict[str, Any], completion.requests[-1])
    assert request_payload["model"] == "gpt-test"
    messages = cast(list[dict[str, Any]], request_payload["messages"])
    assert messages[0]["content"] == "hi"
    assert request_payload["response_format"] == expected_response_format

    choice = select_choice(response)
    assert choice.message is message
