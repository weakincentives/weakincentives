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
from typing import Any, Literal, TypeVar, cast

import pytest

from weakincentives.adapters import openai as openai_adapter, shared
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.adapters.shared import OPENAI_ADAPTER_NAME
from weakincentives.prompt.structured_output import (
    ARRAY_WRAPPER_KEY,
    StructuredOutputConfig,
)

try:
    from tests.adapters._test_stubs import (
        DummyChoice,
        DummyMessage,
        DummyOpenAIClient,
        DummyResponse,
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
from tests.helpers.events import NullEventBus
from weakincentives.adapters import PromptEvaluationError
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

MODULE_PATH = "weakincentives.adapters.openai"
PROMPT_NS = "tests/adapters/openai"


def _split_tool_message_content(content: str) -> tuple[str, str | None]:
    if "\n\n" in content:
        message, remainder = content.split("\n\n", 1)
        return message, remainder or None
    return content, None


def _reload_module() -> types.ModuleType:
    return importlib.reload(std_import_module(MODULE_PATH))


def _text_config_from_json_schema(
    response_format: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if response_format is None:
        return None

    json_schema = cast(dict[str, Any], response_format["json_schema"])
    text_config: dict[str, Any] = {
        "format": {
            "type": "json_schema",
            "name": json_schema["name"],
            "schema": json_schema["schema"],
        }
    }
    description = json_schema.get("description")
    if isinstance(description, str):
        text_config["format"]["description"] = description
    strict = json_schema.get("strict")
    if isinstance(strict, bool):
        text_config["format"]["strict"] = strict
    return text_config


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

    message = DummyMessage(content="Hello, Sam!", tool_calls=None)
    response = DummyResponse([DummyChoice(message)])
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

    message = DummyMessage(content="Hello again!", tool_calls=None)
    response = DummyResponse([DummyChoice(message)])
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

    message = DummyMessage(content="Hello, Sam!", tool_calls=None)
    response = DummyResponse([DummyChoice(message)])
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

    request = cast(dict[str, Any], client.responses.requests[0])
    messages = cast(list[dict[str, Any]], request["input"])
    assert messages[0]["role"] == "system"
    assert str(messages[0]["content"]).startswith("## 1. Greeting")
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
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second_message = DummyMessage(content=json.dumps({"answer": "Policy summary"}))
    second = DummyResponse([DummyChoice(second_message)])
    client = DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

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

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Policy summary")
    assert len(tool_events) == 1
    record = tool_events[0]
    assert record.name == "search_notes"
    assert isinstance(record.result.value, ToolPayload)
    assert record.call_id == "call_1"
    assert calls == ["policies"]

    first_request = cast(dict[str, Any], client.responses.requests[0])
    tools = cast(list[dict[str, Any]], first_request["tools"])
    tool_spec = tools[0]
    assert tool_spec["type"] == "function"
    assert tool_spec["name"] == "search_notes"
    assert tool_spec["parameters"]["type"] == "object"
    assert first_request.get("tool_choice") == "auto"

    second_request = cast(dict[str, Any], client.responses.requests[1])
    second_messages = cast(list[dict[str, Any]], second_request["input"])
    tool_message = second_messages[-1]
    assert tool_message["type"] == "function_call_output"
    assert tool_message["call_id"] == "call_1"
    message_text, rendered_text = _split_tool_message_content(tool_message["output"])
    assert message_text == "completed"
    assert rendered_text is not None
    assert json.loads(rendered_text) == {"answer": "Policy summary"}


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
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second_message = DummyMessage(content=json.dumps({"answer": "Policy summary"}))
    second = DummyResponse([DummyChoice(second_message)])
    client = DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

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
        adapter.evaluate(
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
    first_messages = cast(list[dict[str, Any]], first_request["input"])
    assert first_messages[0]["role"] == "system"

    second_request = cast(dict[str, Any], client.responses.requests[1])
    second_messages = cast(list[dict[str, Any]], second_request["input"])
    tool_message = second_messages[-1]
    assert tool_message["type"] == "function_call_output"
    assert tool_message["call_id"] == "call_1"
    message_text, rendered_text = _split_tool_message_content(tool_message["output"])
    assert message_text == "Tool validation failed: invalid query"
    assert rendered_text is None


def test_openai_adapter_surfaces_tool_type_errors() -> None:
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
        key="openai-tool-type-error",
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
        [DummyChoice(DummyMessage(content="", tool_calls=[tool_call]))]
    )
    second = DummyResponse(
        [DummyChoice(DummyMessage(content="Please adjust the payload."))]
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
        ToolParams(query="policies"),
        bus=bus,
        session=cast(SessionProtocol, session),
    )

    assert result.text == "Please adjust the payload."
    assert result.output is None
    assert invoked is False
    assert len(tool_events) == 1
    event = tool_events[0]
    assert event.name == "search_notes"
    assert event.result.message == "Tool validation failed: query: value cannot be None"
    assert event.result.success is False
    assert event.result.value is None
    assert event.call_id == "call_1"

    second_request = cast(dict[str, Any], client.responses.requests[1])
    second_messages = cast(list[dict[str, Any]], second_request["input"])
    tool_message = second_messages[-1]
    assert tool_message["type"] == "function_call_output"
    assert tool_message["call_id"] == "call_1"
    message_text, rendered_text = _split_tool_message_content(tool_message["output"])
    assert message_text == "Tool validation failed: query: value cannot be None"
    assert rendered_text is None


def test_openai_adapter_includes_text_config_for_array_outputs() -> None:
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
    message = DummyMessage(
        content=json.dumps(payload),
        tool_calls=None,
    )
    response = DummyResponse([DummyChoice(message)])
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
    text_config = cast(dict[str, Any], request["text"])
    response_format = cast(dict[str, Any], text_config["format"])
    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "structured_list_schema"
    schema_payload = cast(dict[str, Any], response_format["schema"])
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
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    final_message = DummyMessage(content="All done")
    second = DummyResponse([DummyChoice(final_message)])
    client = DummyOpenAIClient([first, second])

    forced_choice: Mapping[str, object] = {
        "type": "function",
        "function": {"name": tool.name},
    }
    expected_tool_choice = {"type": "function", "name": tool.name}
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
    assert first_request.get("tool_choice") == expected_tool_choice
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
    first = DummyResponse(
        [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second_message = DummyMessage(content=json.dumps({"answer": "Policy summary"}))
    second = DummyResponse([DummyChoice(second_message)])
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
    assert err.value.phase == PROMPT_EVALUATION_PHASE_TOOL


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

    bus = InProcessEventBus()
    tool_events: list[ToolInvoked] = []
    bus.subscribe(
        ToolInvoked, lambda event: tool_events.append(cast(ToolInvoked, event))
    )
    result = _evaluate_with_bus(
        adapter,
        prompt,
        OptionalParams(),
        bus=bus,
    )

    assert result.text == "All done"
    assert tool_events
    record = tool_events[0]
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
    message = DummyMessage(content=content_blocks, tool_calls=None)
    response = DummyResponse([DummyChoice(message)])
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

    response = DummyResponse(
        [DummyChoice(DummyMessage(content="no-json", tool_calls=None))]
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
    assert err.value.phase == PROMPT_EVALUATION_PHASE_RESPONSE


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

    message = DummyMessage(content=None, tool_calls=None, parsed="not-a-mapping")
    response = DummyResponse([DummyChoice(message)])
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
    assert exc.phase == PROMPT_EVALUATION_PHASE_RESPONSE


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

    rendered = RenderedPrompt(text="")

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
        structured_output=StructuredOutputConfig(
            dataclass_type=StructuredAnswer,
            container=cast(Literal["object", "array"], "invalid"),
            allow_extra_keys=False,
        ),
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
    assert err.value.phase == PROMPT_EVALUATION_PHASE_TOOL


def test_openai_adapter_handles_invalid_tool_params() -> None:
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
    responses = [
        DummyResponse(
            [DummyChoice(DummyMessage(content="thinking", tool_calls=[tool_call]))]
        ),
        DummyResponse([DummyChoice(DummyMessage(content="Try again"))]),
    ]
    client = DummyOpenAIClient(responses)
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

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
    second_messages = cast(list[dict[str, Any]], second_request["input"])
    tool_message = second_messages[-1]
    assert tool_message["type"] == "function_call_output"
    assert tool_message["call_id"] == "call_1"
    message_text, rendered_text = _split_tool_message_content(tool_message["output"])
    assert message_text.endswith("execution failed: boom")
    assert rendered_text is None


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

    assert not hasattr(result, "provider_payload")


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

    assert not hasattr(result, "provider_payload")


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

    assert not hasattr(result, "provider_payload")


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
    assert err.value.phase == PROMPT_EVALUATION_PHASE_TOOL


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
    )

    captured: dict[str, Any] = {}

    def fake_run_conversation(**kwargs: object) -> PromptResponse[StructuredAnswer]:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "run_conversation", fake_run_conversation)

    message = DummyMessage(content="hi")
    client = DummyOpenAIClient([DummyResponse([DummyChoice(message)])])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    params = GreetingParams(user="Ari")
    result = _evaluate_with_bus(adapter, prompt, params)

    assert result is sentinel
    assert captured["adapter_name"] == OPENAI_ADAPTER_NAME
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
    expected_text_config = _text_config_from_json_schema(expected_response_format)
    config = captured["config"]
    assert isinstance(config, module.ConversationConfig)
    assert config.response_format == expected_text_config
    assert config.require_structured_output_text is False
    assert config.serialize_tool_message_fn is module.serialize_tool_message

    call_provider = config.call_provider
    select_choice = config.select_choice
    assert callable(call_provider)
    assert callable(select_choice)

    response = call_provider(
        [{"role": "system", "content": "hi"}],
        [],
        None,
        expected_text_config,
    )
    request_payload = cast(dict[str, Any], client.responses.requests[-1])
    assert request_payload["model"] == "gpt-test"
    messages = cast(list[dict[str, Any]], request_payload["input"])
    assert messages[0]["content"] == "hi"
    assert request_payload["text"] == expected_text_config
    assert config.response_format == expected_text_config

    choice = select_choice(response)
    content_parts = cast(Sequence[object], getattr(choice.message, "content", ()))
    first_part = content_parts[0]
    if isinstance(first_part, Mapping):
        mapping_part = cast(Mapping[str, object], first_part)
        assert mapping_part.get("text") == "hi"
    else:
        assert getattr(first_part, "text", None) == "hi"


def test_openai_normalizes_tool_call_mapping_arguments() -> None:
    module = cast(Any, _reload_module())

    tool_call = {
        "id": "call_1",
        "function": {"name": "do_it", "arguments": {"value": 1}},
    }

    normalized = module._normalize_tool_call(tool_call)

    assert normalized.id == "call_1"
    assert normalized.function.name == "do_it"
    assert normalized.function.arguments == json.dumps({"value": 1})


def test_openai_normalizes_unserializable_arguments() -> None:
    module = cast(Any, _reload_module())

    class Unserializable:
        pass

    arguments = module._normalize_tool_arguments(Unserializable())

    assert arguments is not None
    assert "Unserializable" in arguments


def test_openai_choice_requires_output_and_content() -> None:
    module = cast(Any, _reload_module())

    class MissingOutput:
        output = None

    with pytest.raises(PromptEvaluationError):
        module._choice_from_response(MissingOutput(), prompt_name="missing-output")

    class MissingContent:
        output = (object(),)

    with pytest.raises(PromptEvaluationError):
        module._choice_from_response(MissingContent(), prompt_name="missing-content")


def test_openai_choice_handles_tool_call_output() -> None:
    module = cast(Any, _reload_module())

    class ToolOutput:
        def __init__(self) -> None:
            self.name = "search"
            self.arguments = "{}"
            self.type = "function_call"
            self.call_id = "call_1"

    class Response:
        def __init__(self) -> None:
            self.output = (ToolOutput(),)

    choice = module._choice_from_response(Response(), prompt_name="tool-output")
    assert choice.message.tool_calls
    call = choice.message.tool_calls[0]
    assert call.id == "call_1"
    assert call.function.name == "search"


def test_openai_choice_skips_reasoning_output() -> None:
    module = cast(Any, _reload_module())

    class ReasoningOutput:
        def __init__(self) -> None:
            self.type = "reasoning"
            self.content = None
            self.name = None
            self.arguments = None

    class ToolOutput:
        def __init__(self) -> None:
            self.name = "search"
            self.arguments = "{}"
            self.type = "function_call"
            self.call_id = "call_1"

    class Response:
        def __init__(self) -> None:
            self.output = (ReasoningOutput(), ToolOutput())

    choice = module._choice_from_response(Response(), prompt_name="mixed-output")
    assert choice.message.tool_calls
    call = choice.message.tool_calls[0]
    assert call.id == "call_1"
    assert call.function.name == "search"


def test_openai_choice_raises_when_only_reasoning_output_present() -> None:
    module = cast(Any, _reload_module())

    class ReasoningOutput:
        def __init__(self) -> None:
            self.type = "reasoning"
            self.content = None
            self.name = None
            self.arguments = None

    class Response:
        def __init__(self) -> None:
            self.output = (ReasoningOutput(),)

    with pytest.raises(PromptEvaluationError):
        module._choice_from_response(Response(), prompt_name="reasoning-only")


def test_openai_tool_call_from_output_rejects_non_function_type() -> None:
    module = cast(Any, _reload_module())

    class WrongType:
        name = "tool"
        arguments = "{}"
        type = "other"

    assert module._tool_call_from_output(WrongType()) is None


def test_openai_responses_tool_spec_requires_function_payload() -> None:
    module = cast(Any, _reload_module())

    with pytest.raises(PromptEvaluationError):
        module._responses_tool_spec({"type": "non-function"}, prompt_name="prompt")

    with pytest.raises(PromptEvaluationError):
        module._responses_tool_spec({"type": "function"}, prompt_name="prompt")

    with pytest.raises(PromptEvaluationError):
        module._responses_tool_spec(
            {"type": "function", "function": {"description": "missing name"}},
            prompt_name="prompt",
        )


def test_openai_responses_tool_spec_preserves_strict() -> None:
    module = cast(Any, _reload_module())
    normalized = module._responses_tool_spec(
        {
            "type": "function",
            "function": {"name": "do_it", "parameters": {}, "strict": True},
        },
        prompt_name="prompt",
    )
    assert normalized["strict"] is True


def test_openai_responses_tool_choice_requires_name() -> None:
    module = cast(Any, _reload_module())
    with pytest.raises(PromptEvaluationError):
        module._responses_tool_choice(
            {"type": "function", "function": {}}, prompt_name="prompt"
        )


def test_openai_responses_tool_choice_supports_top_level_name() -> None:
    module = cast(Any, _reload_module())
    tool_choice = module._responses_tool_choice(
        {"type": "function", "name": "do_it"}, prompt_name="prompt"
    )
    assert tool_choice == {"type": "function", "name": "do_it"}


def test_openai_responses_tool_choice_rejects_unknown_type() -> None:
    module = cast(Any, _reload_module())
    with pytest.raises(PromptEvaluationError):
        module._responses_tool_choice({"type": "other"}, prompt_name="prompt")


def test_openai_normalize_input_messages_requires_tool_call_fields() -> None:
    module = cast(Any, _reload_module())

    with pytest.raises(PromptEvaluationError):
        module._normalize_input_messages(
            [{"role": "assistant", "tool_calls": [{"function": {"name": 1}}]}],
            prompt_name="prompt",
        )

    with pytest.raises(PromptEvaluationError):
        module._normalize_input_messages(
            [{"role": "tool", "content": "result"}],
            prompt_name="prompt",
        )


def test_openai_normalize_input_messages_allows_passthrough() -> None:
    module = cast(Any, _reload_module())
    messages = module._normalize_input_messages(
        [{"role": "unknown", "content": "raw"}],
        prompt_name="prompt",
    )
    assert messages[-1]["content"] == "raw"


def test_normalize_output_mapping_handles_mappings() -> None:
    assert openai_adapter._normalize_output_mapping({"type": "web_search_call"}) == {
        "type": "web_search_call"
    }

    class Dumpable:
        def model_dump(self) -> Mapping[str, Any]:
            return {"type": "web_search_call", "id": 1}

    assert openai_adapter._normalize_output_mapping(Dumpable()) == {
        "type": "web_search_call",
        "id": 1,
    }
    assert openai_adapter._normalize_output_mapping(object()) is None


def test_native_tool_calls_from_response_extracts_web_search_call() -> None:
    output_item = {
        "type": "web_search_call",
        "id": "abc",
        "status": "failed",
        "action": {"type": "search", "query": "latest"},
    }
    response = types.SimpleNamespace(output=[output_item])

    calls = openai_adapter._native_tool_calls_from_response(
        response, None, prompt_name="test"
    )

    assert len(calls) == 1
    assert calls[0].call_id == "abc"
    assert calls[0].success is False

    calls_from_payload = openai_adapter._native_tool_calls_from_response(
        response, {"output": [output_item]}, prompt_name="test"
    )
    assert calls_from_payload


def test_native_tool_calls_from_response_filters_invalid_payloads() -> None:
    response = types.SimpleNamespace(output=123)

    assert (
        openai_adapter._native_tool_calls_from_response(
            response, None, prompt_name="test"
        )
        == []
    )

    class Dumpable:
        def model_dump(self) -> str:
            return "not-a-mapping"

    response.output = [Dumpable()]
    assert (
        openai_adapter._native_tool_calls_from_response(
            response, None, prompt_name="test"
        )
        == []
    )


def test_responses_tool_spec_allows_web_search() -> None:
    module = cast(Any, _reload_module())

    spec = module._responses_tool_spec(
        {"type": "web_search", "web_search": {"search_context_size": "low"}},
        prompt_name="prompt",
    )

    assert spec == {"type": "web_search", "web_search": {"search_context_size": "low"}}


def test_responses_tool_choice_allows_web_search() -> None:
    module = cast(Any, _reload_module())

    choice = module._responses_tool_choice({"type": "web_search"}, prompt_name="prompt")

    assert choice == {"type": "web_search"}
