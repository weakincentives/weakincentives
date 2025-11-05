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
from collections.abc import Mapping, Sequence
from importlib import import_module as std_import_module
from types import MethodType
from typing import Any, TypeVar, cast

import pytest

from weakincentives.adapters import shared
from weakincentives.adapters.core import (
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.prompt.structured_output import ARRAY_WRAPPER_KEY

try:
    from tests.adapters._test_stubs import (
        DummyChoice,
        DummyMessage,
        DummyOpenAIClient,
        DummyResponse,
        DummyResponseFunctionToolCall,
        DummyResponseOutputMessage,
        DummyResponseOutputText,
        DummyToolCall,
        GreetingParams,
        MappingResponse,
        OptionalParams,
        OptionalPayload,
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
        DummyOpenAIClient,
        DummyResponse,
        DummyResponseFunctionToolCall,
        DummyResponseOutputMessage,
        DummyResponseOutputText,
        DummyToolCall,
        GreetingParams,
        MappingResponse,
        OptionalParams,
        OptionalPayload,
        SimpleResponse,
        StructuredAnswer,
        ToolParams,
        ToolPayload,
        WeirdResponse,
        simple_handler,
    )
from weakincentives.adapters import PromptEvaluationError
from weakincentives.events import (
    EventBus,
    HandlerFailure,
    InProcessEventBus,
    NullEventBus,
    PromptExecuted,
    ToolInvoked,
)
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
from weakincentives.session import ReducerEvent, Session, replace_latest, select_latest
from weakincentives.tools import ToolValidationError

MODULE_PATH = "weakincentives.adapters.openai"
PROMPT_NS = "tests/adapters/openai"


def _reload_module() -> types.ModuleType:
    return importlib.reload(std_import_module(MODULE_PATH))


OutputT = TypeVar("OutputT")


def _evaluate_with_bus(
    adapter: ProviderAdapter[OutputT],
    prompt: Prompt[OutputT],
    *params: SupportsDataclass,
    bus: EventBus | None = None,
) -> PromptResponse[OutputT]:
    target_bus = bus or NullEventBus()
    session: SessionProtocol = cast(SessionProtocol, Session(bus=target_bus))
    return adapter.evaluate(
        prompt,
        *params,
        bus=target_bus,
        session=session,
    )


def _make_text_message(
    text: str,
    *,
    parsed: object | None = None,
) -> DummyResponseOutputMessage:
    return DummyResponseOutputMessage([DummyResponseOutputText(text, parsed=parsed)])


def _make_response(
    text: str,
    *,
    tool_calls: Sequence[DummyResponseFunctionToolCall] = (),
    parsed: object | None = None,
) -> DummyResponse:
    output: list[object] = [_make_text_message(text, parsed=parsed)]
    output.extend(tool_calls)
    return DummyResponse(output)


def _tool_call_from_dummy(tool_call: DummyToolCall) -> DummyResponseFunctionToolCall:
    return DummyResponseFunctionToolCall(
        call_id=tool_call.id,
        tool_id=tool_call.id,
        name=tool_call.function.name,
        arguments=tool_call.function.arguments,
    )


def test_create_openai_client_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    def fail_import(name: str, package: str | None = None) -> types.ModuleType:
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return std_import_module(name, package)

    monkeypatch.setattr(module, "import_module", fail_import)

    with pytest.raises(RuntimeError) as err:
        module.create_openai_client()

    message = str(err.value)
    assert "uv sync --extra openai" in message
    assert "pip install weakincentives[openai]" in message


def test_create_openai_client_returns_openai_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    class DummyOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    dummy_module = cast(Any, types.ModuleType("openai"))
    dummy_module.OpenAI = DummyOpenAI

    monkeypatch.setitem(sys.modules, "openai", dummy_module)

    client = module.create_openai_client(api_key="secret-key")

    assert isinstance(client, DummyOpenAI)
    assert client.kwargs == {"api_key": "secret-key"}


def test_openai_adapter_constructs_client_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-greeting",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = _make_response("Hello, Sam!")
    client = DummyOpenAIClient([response])
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> DummyOpenAIClient:
        captured_kwargs.append(dict(kwargs))
        return client

    monkeypatch.setattr(module, "create_openai_client", fake_factory)

    adapter = module.OpenAIAdapter(
        model="gpt-test",
        client_kwargs={"api_key": "secret-key"},
    )

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.text == "Hello, Sam!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_openai_adapter_supports_custom_client_factory() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-greeting",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = _make_response("Hello again!")
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> DummyOpenAIClient:
        captured_kwargs.append(dict(kwargs))
        return DummyOpenAIClient([response])

    adapter = module.OpenAIAdapter(
        model="gpt-test",
        client_factory=fake_factory,
        client_kwargs={"api_key": "secret-key"},
    )

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.text == "Hello again!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_openai_adapter_rejects_client_kwargs_with_explicit_client() -> None:
    module = cast(Any, _reload_module())
    client = DummyOpenAIClient([])

    with pytest.raises(ValueError):
        module.OpenAIAdapter(
            model="gpt-test",
            client=client,
            client_kwargs={"api_key": "secret"},
        )


def test_openai_adapter_rejects_client_factory_with_explicit_client() -> None:
    module = cast(Any, _reload_module())
    client = DummyOpenAIClient([])

    with pytest.raises(ValueError):
        module.OpenAIAdapter(
            model="gpt-test",
            client=client,
            client_factory=lambda **_: client,
        )


def test_openai_adapter_returns_plain_text_response() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-plain",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = _make_response("Hello, Sam!")
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.prompt_name == "greeting"
    assert result.text == "Hello, Sam!"
    assert result.output is None
    assert result.tool_results == ()

    request = cast(dict[str, Any], client.responses.requests[0])
    input_items = cast(list[dict[str, Any]], request["input"])
    assert input_items[0]["role"] == "system"
    system_content = cast(list[dict[str, Any]], input_items[0]["content"])
    assert system_content[0]["text"].startswith("## Greeting")
    assert "tools" not in request


def test_openai_adapter_executes_tools_and_parses_output() -> None:
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
        key="openai-structured-success",
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
    first = _make_response("thinking", tool_calls=[_tool_call_from_dummy(tool_call)])
    second = _make_response(json.dumps({"answer": "Policy summary"}))
    client = DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
    )

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Policy summary")
    assert len(result.tool_results) == 1
    record = result.tool_results[0]
    assert record.name == "search_notes"
    assert isinstance(record.result.value, ToolPayload)
    assert record.call_id == "call_1"
    assert calls == ["policies"]

    first_request = cast(dict[str, Any], client.responses.requests[0])
    tools = cast(list[dict[str, Any]], first_request["tools"])
    function_spec = cast(dict[str, Any], tools[0]["function"])
    assert function_spec["name"] == "search_notes"
    assert first_request.get("tool_choice") == "auto"

    second_request = cast(dict[str, Any], client.responses.requests[1])
    second_input = cast(list[dict[str, Any]], second_request["input"])
    tool_message = second_input[-1]
    assert tool_message["type"] == "function_call_output"
    assert tool_message["call_id"] == "call_1"
    serialized = json.loads(tool_message["output"])
    assert serialized["message"] == "completed"
    assert serialized["success"] is True
    assert serialized["payload"] == {"answer": "Result for policies"}


def test_openai_adapter_rolls_back_session_on_publish_failure(
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
        key="openai-session-rollback",
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
    first = _make_response("thinking", tool_calls=[_tool_call_from_dummy(tool_call)])
    second = _make_response(json.dumps({"answer": "Policy summary"}))
    client = DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    session.register_reducer(ToolPayload, replace_latest)
    session.seed_slice(ToolPayload, (ToolPayload(answer="baseline"),))

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

    result = adapter.evaluate(
        prompt,
        ToolParams(query="policies"),
        bus=bus,
        session=session,
    )

    tool_event = result.tool_results[0]
    assert tool_event.result.message.startswith(
        "Reducer errors prevented applying tool result:"
    )
    assert "Reducer crashed" in tool_event.result.message

    latest_payload = select_latest(session, ToolPayload)
    assert latest_payload == ToolPayload(answer="baseline")
    assert result.output == StructuredAnswer(answer="Policy summary")


def test_openai_format_publish_failures_handles_defaults() -> None:
    module = cast(Any, _reload_module())

    failure = HandlerFailure(handler=lambda _: None, error=RuntimeError(""))
    message = module.format_publish_failures((failure,))
    assert message == "Reducer errors prevented applying tool result: RuntimeError"
    assert (
        module.format_publish_failures(())
        == "Reducer errors prevented applying tool result."
    )


def test_openai_adapter_surfaces_tool_validation_errors() -> None:
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
        key="openai-tool-validation",
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
        [DummyChoice(DummyMessage(content="", tool_calls=[tool_call]))]
    )
    second = DummyResponse(
        [DummyChoice(DummyMessage(content="Please provide a different query."))]
    )
    client = DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    tool_events: list[ToolInvoked] = []

    def record_tool_event(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record_tool_event)

    result = adapter.evaluate(
        prompt,
        ToolParams(query="invalid"),
        bus=bus,
        session=cast(SessionProtocol, session),
    )

    assert result.text == "Please provide a different query."
    assert result.output is None
    assert len(tool_events) == 1
    event = tool_events[0]
    assert event.name == "search_notes"
    assert event.result.message == "Tool validation failed: invalid query"
    assert event.result.success is False
    assert event.result.value is None
    assert event.call_id == "call_1"

    first_request = cast(dict[str, Any], client.responses.requests[0])
    first_input = cast(list[dict[str, Any]], first_request["input"])
    assert first_input[0]["role"] == "system"

    second_request = cast(dict[str, Any], client.responses.requests[1])
    second_input = cast(list[dict[str, Any]], second_request["input"])
    tool_message = second_input[-1]
    assert tool_message["type"] == "function_call_output"
    serialized = json.loads(tool_message["output"])
    assert serialized["message"] == "Tool validation failed: invalid query"
    assert serialized["success"] is False
    assert "payload" not in serialized


def test_openai_adapter_includes_response_format_for_array_outputs() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[list[StructuredAnswer]](
        ns=PROMPT_NS,
        key="openai-structured-schema-array",
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
    response = _make_response(json.dumps(payload))
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
    )

    assert isinstance(result.output, list)
    assert [item.answer for item in result.output] == ["First", "Second"]

    request = cast(dict[str, Any], client.responses.requests[0])
    response_format = cast(dict[str, Any], request["response_format"])
    json_schema = cast(dict[str, Any], response_format["json_schema"])
    schema_payload = cast(dict[str, Any], json_schema["schema"])
    properties = cast(dict[str, Any], schema_payload["properties"])
    assert ARRAY_WRAPPER_KEY in properties
    items_schema = cast(dict[str, Any], properties[ARRAY_WRAPPER_KEY])
    assert items_schema.get("type") == "array"
    assert items_schema.get("items", {}).get("type") == "object"


def test_openai_adapter_relaxes_forced_tool_choice_after_first_call() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-tools-relaxed",
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
    first = _make_response("thinking", tool_calls=[_tool_call_from_dummy(tool_call)])
    second = _make_response("All done")
    client = DummyOpenAIClient([first, second])

    forced_choice: Mapping[str, object] = {
        "type": "function",
        "function": {"name": tool.name},
    }
    adapter = module.OpenAIAdapter(
        model="gpt-test",
        client=client,
        tool_choice=forced_choice,
    )

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
    )

    assert result.text == "All done"

    assert len(client.responses.requests) == 2
    first_request = cast(dict[str, Any], client.responses.requests[0])
    second_request = cast(dict[str, Any], client.responses.requests[1])
    assert first_request.get("tool_choice") == forced_choice
    assert second_request.get("tool_choice") == "auto"


def test_openai_adapter_emits_events_during_evaluation() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="openai-structured-events",
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
    first = _make_response("thinking", tool_calls=[_tool_call_from_dummy(tool_call)])
    second = _make_response(json.dumps({"answer": "Policy summary"}))
    client = DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

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
    assert tool_event.adapter == "openai"
    assert tool_event.name == "search_notes"
    assert tool_event.call_id == "call_1"
    assert tool_event is result.tool_results[0]

    assert len(prompt_events) == 1
    prompt_event = prompt_events[0]
    assert prompt_event.prompt_name == "search"
    assert prompt_event.adapter == "openai"
    assert prompt_event.result is result


def test_openai_adapter_raises_when_tool_handler_missing() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=None,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-tools-missing-handler",
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
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == "tool"


def test_openai_adapter_handles_tool_call_without_arguments() -> None:
    module = cast(Any, _reload_module())

    def optional_handler(
        params: OptionalParams, *, context: ToolContext
    ) -> ToolResult[OptionalPayload]:
        del context
        return ToolResult(message="done", value=OptionalPayload(value=params.query))

    tool_handler: ToolHandler[OptionalParams, OptionalPayload] = optional_handler

    tool = Tool[OptionalParams, OptionalPayload](
        name="optional_tool",
        description="Uses defaults when args are missing.",
        handler=tool_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-optional-tool",
        name="optional",
        sections=[
            MarkdownSection[OptionalParams](
                title="Task",
                key="task",
                template="Provide data",
                tools=[tool],
            )
        ],
    )

    tool_call = DummyToolCall(
        call_id="call_1",
        name="optional_tool",
        arguments=None,
    )
    response_with_tool = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    final_response = DummyResponse(
        [DummyChoice(DummyMessage(content="All done", tool_calls=None))]
    )
    client = DummyOpenAIClient([response_with_tool, final_response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        OptionalParams(),
    )

    assert result.text == "All done"
    record = result.tool_results[0]
    params = cast(OptionalParams, record.params)
    assert params.query == "default"


def test_openai_adapter_reads_output_json_content_blocks() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="openai-structured-json-block",
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
    response = DummyResponse([DummyResponseOutputMessage(content_blocks)])
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        ToolParams(query="policies"),
    )

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Block")


def test_openai_adapter_raises_when_structured_output_missing_json() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="openai-structured-missing-json",
        name="search",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
            )
        ],
    )

    response = _make_response("no-json")
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == "response"


def test_openai_adapter_raises_on_invalid_parsed_payload() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="openai-structured-parsed-error",
        name="structured",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return the structured result only.",
            )
        ],
    )

    response = _make_response("", parsed="not-a-mapping")
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    exc = err.value
    assert isinstance(exc, PromptEvaluationError)
    assert exc.phase == "response"


def test_openai_message_text_content_handles_structured_parts() -> None:
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


def test_openai_extract_parsed_content_handles_attribute_blocks() -> None:
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


def test_openai_parse_schema_constrained_payload_unwraps_wrapped_array() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[list[StructuredAnswer]](
        ns=PROMPT_NS,
        key="openai-structured-schema-array-wrapped",
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


def test_openai_parse_schema_constrained_payload_handles_object_container() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="openai-structured-schema",
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


def test_openai_build_json_schema_response_format_returns_none_for_plain_prompt() -> (
    None
):
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-plain",
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


def test_openai_parse_schema_constrained_payload_requires_structured_prompt() -> None:
    module = cast(Any, _reload_module())

    rendered = RenderedPrompt(
        text="",
        output_type=None,
        container=None,
        allow_extra_keys=None,
    )

    with pytest.raises(TypeError):
        module.parse_schema_constrained_payload({}, rendered)


def test_openai_parse_schema_constrained_payload_rejects_non_sequence_arrays() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[list[StructuredAnswer]](
        ns=PROMPT_NS,
        key="openai-structured-schema-array-non-seq",
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


def test_openai_parse_schema_constrained_payload_rejects_unknown_container() -> None:
    module = cast(Any, _reload_module())

    rendered = RenderedPrompt(
        text="",
        output_type=StructuredAnswer,
        container="invalid",  # type: ignore[arg-type]
        allow_extra_keys=False,
    )

    with pytest.raises(TypeError):
        module.parse_schema_constrained_payload({}, rendered)


def test_openai_adapter_raises_for_unknown_tool() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-unknown-tool",
        name="search",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
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
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == "tool"


def test_openai_adapter_raises_when_tool_params_invalid() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-invalid-tool-params",
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
    response = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == "tool"


def test_openai_adapter_records_handler_failures() -> None:
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
        key="openai-handler-failure",
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
    client = DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    tool_events: list[ToolInvoked] = []

    def record(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record)

    result = adapter.evaluate(
        prompt,
        ToolParams(query="policies"),
        bus=bus,
        session=cast(SessionProtocol, session),
    )

    assert result.text == "Please provide a different approach."
    assert result.output is None
    assert len(tool_events) == 1
    event = tool_events[0]
    assert event.result.success is False
    assert event.result.value is None
    assert "execution failed: boom" in event.result.message

    second_request = cast(dict[str, Any], client.responses.requests[1])
    second_input = cast(list[dict[str, Any]], second_request["input"])
    tool_message = second_input[-1]
    assert tool_message["type"] == "function_call_output"
    serialized = json.loads(tool_message["output"])
    assert serialized["success"] is False
    assert serialized["message"].endswith("execution failed: boom")


def test_openai_adapter_records_provider_payload_from_mapping() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-provider-payload",
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
    client = DummyOpenAIClient([mapping_response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.provider_payload == {"meta": "value"}


def test_openai_adapter_ignores_non_mapping_model_dump() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-weird-dump",
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
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.provider_payload is None


def test_openai_adapter_handles_response_without_model_dump() -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-simple-response",
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
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = _evaluate_with_bus(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.provider_payload is None


@pytest.mark.parametrize(
    "arguments_json",
    ["{", json.dumps("not a dict")],
)
def test_openai_adapter_rejects_bad_tool_arguments(arguments_json: str) -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        ns=PROMPT_NS,
        key="openai-bad-tool-arguments",
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
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_bus(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == "tool"


def test_openai_adapter_delegates_to_shared_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="openai-shared-runner",
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
        tool_results=(),
        provider_payload=None,
    )

    captured: dict[str, Any] = {}

    def fake_run_conversation(**kwargs: object) -> PromptResponse[StructuredAnswer]:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "run_conversation", fake_run_conversation)

    client = DummyOpenAIClient([_make_response("hi")])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    params = GreetingParams(user="Ari")
    result = _evaluate_with_bus(adapter, prompt, params)

    assert result is sentinel
    assert captured["adapter_name"] == "openai"
    assert captured["prompt_name"] == "shared-runner"

    expected_rendered = prompt.render(params, inject_output_instructions=False)
    assert captured["rendered"] == expected_rendered
    assert captured["initial_messages"] == [
        {"role": "system", "content": expected_rendered.text}
    ]

    expected_response_format = module.build_json_schema_response_format(
        expected_rendered, "shared-runner"
    )
    assert captured["response_format"] == expected_response_format
    assert captured["require_structured_output_text"] is False
    assert captured["serialize_tool_message_fn"] is module.serialize_tool_message

    call_provider = captured["call_provider"]
    select_choice = captured["select_choice"]
    assert callable(call_provider)
    assert callable(select_choice)

    response = call_provider(
        [{"role": "system", "content": "hi"}],
        [],
        None,
        expected_response_format,
    )
    request_payload = cast(dict[str, Any], client.responses.requests[-1])
    assert request_payload["model"] == "gpt-test"
    input_items = cast(list[dict[str, Any]], request_payload["input"])
    assert input_items[0]["content"][0]["text"] == "hi"
    assert request_payload["response_format"] == expected_response_format

    choice = select_choice(response)
    content = getattr(choice.message, "content", ())
    assert content and getattr(content[0], "text", None) == "hi"


def test_openai_convert_messages_to_input_handles_edge_cases() -> None:
    module = cast(Any, _reload_module())

    messages = [
        {
            "role": "assistant",
            "content": "Hello there",
            "tool_calls": [
                "invalid",
                {"call_id": 123, "function": "unsupported"},
                {"id": "call_2", "function": {"name": "search", "arguments": None}},
            ],
        },
        {"role": "tool", "content": "ignored"},
        {"role": "tool", "tool_call_id": "call_2", "content": "result"},
    ]

    input_items = module._convert_messages_to_input(messages)

    assert input_items == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "Hello there"}],
        },
        {
            "type": "function_call",
            "call_id": "123",
            "name": None,
            "arguments": "{}",
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "search",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": "result",
        },
    ]


class _OtherResponseItem:
    def __init__(self) -> None:
        self.type = "other"

    def model_dump(self) -> dict[str, str]:
        return {"type": "other", "value": "kept"}


class _ScalarMessage:
    class _Content:
        def model_dump(self) -> str:
            return "raw text"

    def __init__(self) -> None:
        self.type = "message"
        self.content = self._Content()


def test_openai_wrap_response_choice_converts_items() -> None:
    module = cast(Any, _reload_module())

    text_part = DummyResponseOutputText("Hello", parsed={"answer": 1})
    message_item = DummyResponseOutputMessage([text_part])
    function_call = DummyResponseFunctionToolCall(
        call_id="call_1",
        name="lookup",
        arguments='{"query": "value"}',
        tool_id=None,
    )
    scalar_message = _ScalarMessage()
    other_item = _OtherResponseItem()

    response = types.SimpleNamespace(
        output=[message_item, function_call, scalar_message, other_item],
        parsed=None,
    )

    choice = module._wrap_response_choice(response, "prompt-name")
    message = choice.message

    payload = message.model_dump()
    assert payload["content"][0]["text"] == "Hello"
    assert payload["content"][1] == "raw text"
    assert payload["content"][2] == {"type": "other", "value": "kept"}
    assert payload["tool_calls"][0]["id"] == "call_1"
    assert payload["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"query": "value"}',
    }
    assert payload["parsed"] == {"answer": 1}
    assert choice.model_dump() == {"message": payload}
    assert module._model_dump_value({1: "one"}) == {"1": "one"}
    assert module._model_dump_value(5) == 5


def test_openai_wrap_response_choice_requires_sequence() -> None:
    module = cast(Any, _reload_module())

    response = types.SimpleNamespace(output=None)

    with pytest.raises(PromptEvaluationError):
        module._wrap_response_choice(response, "prompt-name")
