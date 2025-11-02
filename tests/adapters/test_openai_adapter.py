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
from typing import Any, cast

import pytest

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
from weakincentives.adapters import PromptEvaluationError
from weakincentives.events import (
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
    ToolResult,
)
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.session import DataEvent, Session, replace_latest, select_latest
from weakincentives.tools import ToolValidationError

MODULE_PATH = "weakincentives.adapters.openai"
PROMPT_NS = "tests/adapters/openai"


def _reload_module() -> types.ModuleType:
    return importlib.reload(std_import_module(MODULE_PATH))


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

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
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

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
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

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.prompt_name == "greeting"
    assert result.text == "Hello, Sam!"
    assert result.output is None
    assert result.tool_results == ()

    request = cast(dict[str, Any], client.completions.requests[0])
    messages = cast(list[dict[str, Any]], request["messages"])
    assert messages[0]["role"] == "system"
    assert str(messages[0]["content"]).startswith("## Greeting")
    assert "tools" not in request


def test_openai_adapter_executes_tools_and_parses_output() -> None:
    module = cast(Any, _reload_module())

    calls: list[str] = []

    def handler(params: ToolParams) -> ToolResult[ToolPayload]:
        calls.append(params.query)
        payload = ToolPayload(answer=f"Result for {params.query}")
        return ToolResult(message="completed", value=payload)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=handler,
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

    result = adapter.evaluate(
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Policy summary")
    assert len(result.tool_results) == 1
    record = result.tool_results[0]
    assert record.name == "search_notes"
    assert isinstance(record.result.value, ToolPayload)
    assert record.call_id == "call_1"
    assert calls == ["policies"]

    first_request = cast(dict[str, Any], client.completions.requests[0])
    tools = cast(list[dict[str, Any]], first_request["tools"])
    function_spec = cast(dict[str, Any], tools[0]["function"])
    assert function_spec["name"] == "search_notes"
    assert first_request.get("tool_choice") == "auto"

    second_request = cast(dict[str, Any], client.completions.requests[1])
    second_messages = cast(list[dict[str, Any]], second_request["messages"])
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "completed"
    assert "payload" not in tool_message


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

    original_dispatch = session._dispatch_data_event

    def failing_dispatch(
        self: Session,
        data_type: type[SupportsDataclass],
        event: DataEvent,
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
    message = module._format_publish_failures((failure,))
    assert message == "Reducer errors prevented applying tool result: RuntimeError"
    assert (
        module._format_publish_failures(())
        == "Reducer errors prevented applying tool result."
    )


def test_openai_adapter_surfaces_tool_validation_errors() -> None:
    module = cast(Any, _reload_module())

    def handler(_: ToolParams) -> ToolResult[ToolPayload]:
        raise ToolValidationError("invalid query")

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=handler,
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
    tool_events: list[ToolInvoked] = []

    def record_tool_event(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record_tool_event)

    result = adapter.evaluate(
        prompt,
        ToolParams(query="invalid"),
        bus=bus,
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

    first_request = cast(dict[str, Any], client.completions.requests[0])
    first_messages = cast(list[dict[str, Any]], first_request["messages"])
    assert first_messages[0]["role"] == "system"

    second_request = cast(dict[str, Any], client.completions.requests[1])
    second_messages = cast(list[dict[str, Any]], second_request["messages"])
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "Tool validation failed: invalid query"


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
    message = DummyMessage(
        content=json.dumps(payload),
        tool_calls=None,
    )
    response = DummyResponse([DummyChoice(message)])
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert isinstance(result.output, list)
    assert [item.answer for item in result.output] == ["First", "Second"]

    request = cast(dict[str, Any], client.completions.requests[0])
    response_format = cast(dict[str, Any], request["response_format"])
    json_schema = cast(dict[str, Any], response_format["json_schema"])
    schema_payload = cast(dict[str, Any], json_schema["schema"])
    properties = cast(dict[str, Any], schema_payload["properties"])
    assert module.ARRAY_WRAPPER_KEY in properties
    items_schema = cast(dict[str, Any], properties[module.ARRAY_WRAPPER_KEY])
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
    adapter = module.OpenAIAdapter(
        model="gpt-test",
        client=client,
        tool_choice=forced_choice,
    )

    result = adapter.evaluate(
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text == "All done"

    assert len(client.completions.requests) == 2
    first_request = cast(dict[str, Any], client.completions.requests[0])
    second_request = cast(dict[str, Any], client.completions.requests[1])
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
    result = adapter.evaluate(
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == "tool"


def test_openai_adapter_handles_tool_call_without_arguments() -> None:
    module = cast(Any, _reload_module())

    def optional_handler(params: OptionalParams) -> ToolResult[OptionalPayload]:
        return ToolResult(message="done", value=OptionalPayload(value=params.query))

    tool = Tool[OptionalParams, OptionalPayload](
        name="optional_tool",
        description="Uses defaults when args are missing.",
        handler=optional_handler,
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

    result = adapter.evaluate(
        prompt,
        OptionalParams(),
        bus=NullEventBus(),
    )

    assert result.text == "All done"
    assert result.tool_results[0].params.query == "default"


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

    result = adapter.evaluate(
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
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

    message = DummyMessage(content=None, tool_calls=None, parsed="not-a-mapping")
    response = DummyResponse([DummyChoice(message)])
    client = DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    exc = err.value
    assert isinstance(exc, PromptEvaluationError)
    assert exc.phase == "response"


def test_openai_message_text_content_handles_structured_parts() -> None:
    module = cast(Any, _reload_module())

    mapping_parts = [{"type": "output_text", "text": "Hello"}]
    assert module._message_text_content(mapping_parts) == "Hello"

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.type = "text"
            self.text = text

    assert module._message_text_content([TextBlock("World")]) == "World"
    assert module._message_text_content(123) == "123"
    assert module._content_part_text(None) == ""
    assert module._content_part_text({"type": "output_text", "text": 123}) == ""

    class BadTextBlock:
        def __init__(self) -> None:
            self.type = "text"
            self.text = 123

    assert module._content_part_text(BadTextBlock()) == ""


def test_openai_extract_parsed_content_handles_attribute_blocks() -> None:
    module = cast(Any, _reload_module())

    class JsonBlock:
        def __init__(self, payload: dict[str, object]) -> None:
            self.type = "output_json"
            self.json = payload

    block = JsonBlock({"answer": "attribute"})
    message = DummyMessage(content=[block], tool_calls=None)

    parsed = module._extract_parsed_content(message)

    assert parsed == {"answer": "attribute"}
    assert module._parsed_payload_from_part({"type": "other"}) is None

    class OtherBlock:
        def __init__(self) -> None:
            self.type = "other"
            self.json = {"answer": "ignored"}

    assert module._parsed_payload_from_part(OtherBlock()) is None


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

    payload = {module.ARRAY_WRAPPER_KEY: [{"answer": "Ready"}]}

    parsed = module._parse_schema_constrained_payload(payload, rendered)

    assert isinstance(parsed, list)
    assert parsed[0].answer == "Ready"

    with pytest.raises(TypeError):
        module._parse_schema_constrained_payload({"wrong": []}, rendered)

    with pytest.raises(TypeError):
        module._parse_schema_constrained_payload(["oops"], rendered)


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

    parsed = module._parse_schema_constrained_payload({"answer": "Ready"}, rendered)

    assert parsed.answer == "Ready"

    with pytest.raises(TypeError):
        module._parse_schema_constrained_payload("oops", rendered)


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

    response_format = module._build_json_schema_response_format(rendered, "plain")

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
        module._parse_schema_constrained_payload({}, rendered)


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
        module._parse_schema_constrained_payload("oops", rendered)


def test_openai_parse_schema_constrained_payload_rejects_unknown_container() -> None:
    module = cast(Any, _reload_module())

    rendered = RenderedPrompt(
        text="",
        output_type=StructuredAnswer,
        container="invalid",  # type: ignore[arg-type]
        allow_extra_keys=False,
    )

    with pytest.raises(TypeError):
        module._parse_schema_constrained_payload({}, rendered)


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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == "tool"


def test_openai_adapter_records_handler_failures() -> None:
    module = cast(Any, _reload_module())

    def failing_handler(params: ToolParams) -> ToolResult[ToolPayload]:
        raise RuntimeError("boom")

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=failing_handler,
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
    tool_events: list[ToolInvoked] = []

    def record(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record)

    result = adapter.evaluate(
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

    second_request = cast(dict[str, Any], client.completions.requests[1])
    second_messages = cast(list[dict[str, Any]], second_request["messages"])
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    assert "execution failed: boom" in tool_message["content"]


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

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
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

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
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

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == "tool"
