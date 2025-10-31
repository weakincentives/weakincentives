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
from dataclasses import dataclass
from importlib import import_module as std_import_module
from typing import Any, cast

import pytest

from weakincentives.adapters import PromptEvaluationError
from weakincentives.events import (
    InProcessEventBus,
    NullEventBus,
    PromptExecuted,
    ToolInvoked,
)
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult
from weakincentives.prompts.prompt import RenderedPrompt
from weakincentives.prompts.structured import ARRAY_RESULT_KEY

MODULE_PATH = "weakincentives.adapters.openai"


def _reload_module():
    return importlib.reload(std_import_module(MODULE_PATH))


def test_create_openai_client_requires_optional_dependency(monkeypatch):
    module = _reload_module()

    def fail_import(name: str, package: str | None = None):
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return std_import_module(name, package)

    monkeypatch.setattr(module, "import_module", fail_import)

    with pytest.raises(RuntimeError) as err:
        module.create_openai_client()

    message = str(err.value)
    assert "uv sync --extra openai" in message
    assert "pip install weakincentives[openai]" in message


def test_create_openai_client_returns_openai_instance(monkeypatch):
    module = _reload_module()

    class DummyOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    dummy_module = cast(module._OpenAIModule, types.ModuleType("openai"))
    dummy_module.OpenAI = DummyOpenAI

    monkeypatch.setitem(sys.modules, "openai", dummy_module)

    client = module.create_openai_client(api_key="secret-key")

    assert isinstance(client, DummyOpenAI)
    assert client.kwargs == {"api_key": "secret-key"}


def test_openai_adapter_constructs_client_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module()

    prompt = Prompt(
        key="openai-greeting",
        name="greeting",
        sections=[
            TextSection[_GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    message = _DummyMessage(content="Hello, Sam!", tool_calls=None)
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> _DummyOpenAIClient:
        captured_kwargs.append(dict(kwargs))
        return client

    monkeypatch.setattr(module, "create_openai_client", fake_factory)

    adapter = module.OpenAIAdapter(
        model="gpt-test",
        client_kwargs={"api_key": "secret-key"},
    )

    result = adapter.evaluate(
        prompt,
        _GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.text == "Hello, Sam!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_openai_adapter_supports_custom_client_factory() -> None:
    module = _reload_module()

    prompt = Prompt(
        key="openai-greeting",
        name="greeting",
        sections=[
            TextSection[_GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    message = _DummyMessage(content="Hello again!", tool_calls=None)
    response = _DummyResponse([_DummyChoice(message)])
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> _DummyOpenAIClient:
        captured_kwargs.append(dict(kwargs))
        return _DummyOpenAIClient([response])

    adapter = module.OpenAIAdapter(
        model="gpt-test",
        client_factory=fake_factory,
        client_kwargs={"api_key": "secret-key"},
    )

    result = adapter.evaluate(
        prompt,
        _GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.text == "Hello again!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_openai_adapter_rejects_client_kwargs_with_explicit_client() -> None:
    module = _reload_module()
    client = _DummyOpenAIClient([])

    with pytest.raises(ValueError):
        module.OpenAIAdapter(
            model="gpt-test",
            client=client,
            client_kwargs={"api_key": "secret"},
        )


def test_openai_adapter_rejects_client_factory_with_explicit_client() -> None:
    module = _reload_module()
    client = _DummyOpenAIClient([])

    with pytest.raises(ValueError):
        module.OpenAIAdapter(
            model="gpt-test",
            client=client,
            client_factory=lambda **_: client,
        )


@dataclass
class _GreetingParams:
    user: str


@dataclass(slots=True)
class _DummyFunctionCall:
    name: str
    arguments: str | None


class _DummyToolCall:
    def __init__(self, call_id: str, name: str, arguments: str | None) -> None:
        self.id = call_id
        self.function = _DummyFunctionCall(name=name, arguments=arguments)

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _DummyMessage:
    def __init__(
        self,
        *,
        content: str | Sequence[object] | None,
        tool_calls: Sequence[_DummyToolCall] | None = None,
        parsed: object | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tuple(tool_calls) if tool_calls else None
        self.parsed = parsed

    def model_dump(self) -> dict[str, Any]:
        if isinstance(self.content, Sequence) and not isinstance(
            self.content, (str, bytes, bytearray)
        ):
            payload_content: object = list(self.content)
        else:
            payload_content = self.content

        payload: dict[str, Any] = {"content": payload_content}
        if self.tool_calls is not None:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.parsed is not None:
            payload["parsed"] = self.parsed
        return payload


class _DummyChoice:
    def __init__(self, message: _DummyMessage) -> None:
        self.message = message

    def model_dump(self) -> dict[str, Any]:
        return {"message": self.message.model_dump()}


class _DummyResponse:
    def __init__(self, choices: Sequence[_DummyChoice]) -> None:
        self.choices = list(choices)

    def model_dump(self) -> dict[str, Any]:
        return {"choices": [choice.model_dump() for choice in self.choices]}


class _MappingResponse(dict):
    def __init__(self, choices: Sequence[_DummyChoice]) -> None:
        super().__init__({"meta": "value"})
        self.choices = list(choices)


class _WeirdResponse:
    def __init__(self, choices: Sequence[_DummyChoice]) -> None:
        self.choices = list(choices)

    def model_dump(self) -> list[object]:
        return ["unexpected"]


class _SimpleResponse:
    def __init__(self, choices: Sequence[_DummyChoice]) -> None:
        self.choices = list(choices)


_ResponseType = _DummyResponse | _MappingResponse | _WeirdResponse | _SimpleResponse


class _DummyCompletionsAPI:
    def __init__(self, responses: Sequence[_ResponseType]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _ResponseType:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


@dataclass(slots=True)
class _DummyChatAPI:
    completions: _DummyCompletionsAPI


class _DummyOpenAIClient:
    def __init__(self, responses: Sequence[_ResponseType]) -> None:
        completions = _DummyCompletionsAPI(responses)
        self.chat = _DummyChatAPI(completions)
        self.completions = completions


def test_openai_adapter_returns_plain_text_response():
    module = _reload_module()

    prompt = Prompt(
        key="openai-plain",
        name="greeting",
        sections=[
            TextSection[_GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    message = _DummyMessage(content="Hello, Sam!", tool_calls=None)
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _GreetingParams(user="Sam"),
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


@dataclass
class _ToolParams:
    query: str


@dataclass
class _ToolPayload:
    answer: str


@dataclass
class _StructuredAnswer:
    answer: str


@dataclass
class _OptionalParams:
    query: str = "default"


@dataclass
class _OptionalPayload:
    value: str


def _simple_handler(params: _ToolParams) -> ToolResult[_ToolPayload]:
    return ToolResult(message="ok", payload=_ToolPayload(answer=params.query))


def test_openai_adapter_executes_tools_and_parses_output():
    module = _reload_module()

    calls: list[str] = []

    def handler(params: _ToolParams) -> ToolResult[_ToolPayload]:
        calls.append(params.query)
        payload = _ToolPayload(answer=f"Result for {params.query}")
        return ToolResult(message="completed", payload=payload)

    tool = Tool[_ToolParams, _ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=handler,
    )

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-success",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second_message = _DummyMessage(content=json.dumps({"answer": "Policy summary"}))
    second = _DummyResponse([_DummyChoice(second_message)])
    client = _DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text is None
    assert result.output == _StructuredAnswer(answer="Policy summary")
    assert len(result.tool_results) == 1
    record = result.tool_results[0]
    assert record.name == "search_notes"
    assert isinstance(record.result.payload, _ToolPayload)
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


def test_openai_adapter_includes_response_format_for_structured_prompts():
    module = _reload_module()

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-schema",
        name="structured",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Summarize ${query} as JSON.",
            )
        ],
    )

    message = _DummyMessage(
        content=json.dumps({"answer": "Ready"}),
        tool_calls=None,
    )
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.output == _StructuredAnswer(answer="Ready")

    request = cast(dict[str, Any], client.completions.requests[0])
    assert "response_format" in request
    response_format = cast(dict[str, Any], request["response_format"])
    assert response_format["type"] == "json_schema"
    json_schema = cast(dict[str, Any], response_format["json_schema"])
    assert "name" in json_schema
    schema_payload = cast(dict[str, Any], json_schema["schema"])
    assert schema_payload.get("type") == "object"


def test_openai_adapter_uses_parsed_payload_when_available():
    module = _reload_module()

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-parsed",
        name="structured",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Return the structured result only.",
            )
        ],
    )

    message = _DummyMessage(
        content=None,
        tool_calls=None,
        parsed={"answer": "Parsed"},
    )
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text is None
    assert result.output == _StructuredAnswer(answer="Parsed")


def test_openai_adapter_omits_response_instructions_with_native_schema():
    module = _reload_module()

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-native-instructions",
        name="structured",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Summarize ${query} as JSON.",
            )
        ],
    )

    message = _DummyMessage(
        content=json.dumps({"answer": "Ready"}),
        tool_calls=None,
    )
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    request = cast(dict[str, Any], client.completions.requests[0])
    system_message = cast(dict[str, Any], request["messages"][0])
    system_text = cast(str, system_message["content"])
    assert "Response Format" not in system_text
    assert "Return ONLY a single fenced JSON code block" not in system_text


def test_openai_adapter_reads_output_json_content_blocks():
    module = _reload_module()

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-json-block",
        name="structured",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Return the structured result only.",
            )
        ],
    )

    content_blocks = [
        {"type": "output_json", "json": {"answer": "Block"}},
    ]
    message = _DummyMessage(
        content=content_blocks,
        tool_calls=None,
    )
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text is None
    assert result.output == _StructuredAnswer(answer="Block")


def test_openai_adapter_includes_response_format_for_array_outputs():
    module = _reload_module()

    prompt = Prompt[list[_StructuredAnswer]](
        key="openai-structured-schema-array",
        name="structured_list",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Return a list of answers for ${query}.",
            )
        ],
    )

    payload = [{"answer": "First"}, {"answer": "Second"}]
    message = _DummyMessage(
        content=json.dumps(payload),
        tool_calls=None,
    )
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert isinstance(result.output, list)
    assert [item.answer for item in result.output] == ["First", "Second"]

    request = cast(dict[str, Any], client.completions.requests[0])
    response_format = cast(dict[str, Any], request["response_format"])
    json_schema = cast(dict[str, Any], response_format["json_schema"])
    schema_payload = cast(dict[str, Any], json_schema["schema"])
    assert schema_payload.get("type") == "object"
    properties = cast(dict[str, Any], schema_payload.get("properties"))
    assert ARRAY_RESULT_KEY in properties
    items_schema = cast(dict[str, Any], properties[ARRAY_RESULT_KEY])
    assert items_schema.get("type") == "array"
    assert items_schema.get("items", {}).get("type") == "object"


def test_openai_adapter_skips_response_format_when_parse_output_disabled():
    module = _reload_module()

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-schema-disabled",
        name="structured",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Summarize ${query} as JSON.",
            )
        ],
    )

    payload = {"answer": "Ready"}
    message = _DummyMessage(
        content=json.dumps(payload),
        tool_calls=None,
    )
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
        parse_output=False,
        bus=NullEventBus(),
    )

    assert result.output is None
    assert result.text == json.dumps(payload)

    request = cast(dict[str, Any], client.completions.requests[0])
    assert "response_format" not in request


def test_openai_adapter_supports_instruction_based_structured_output():
    module = _reload_module()

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-schema-instructions",
        name="structured",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Summarize ${query} as JSON.",
            )
        ],
    )

    payload = {"answer": "Ready"}
    message = _DummyMessage(
        content=json.dumps(payload),
        tool_calls=None,
    )
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(
        model="gpt-test", client=client, use_native_response_format=False
    )

    result = adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.output == _StructuredAnswer(answer="Ready")

    request = cast(dict[str, Any], client.completions.requests[0])
    assert "response_format" not in request
    system_message = cast(dict[str, Any], request["messages"][0])
    system_text = cast(str, system_message["content"])
    assert "Response Format" in system_text
    assert "Return ONLY a single fenced JSON code block" in system_text


def test_openai_adapter_builds_response_format_only_for_structured_prompts():
    module = _reload_module()

    prompt = Prompt(
        key="openai-plain",
        name="plain",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Say hello to ${query}.",
            )
        ],
    )

    rendered = prompt.render(_ToolParams(query="world"))

    response_format = module._build_response_format(rendered, "plain")

    assert response_format is None


def test_openai_adapter_parse_provider_payload_requires_wrapped_array_key():
    module = _reload_module()

    prompt = Prompt[list[_StructuredAnswer]](
        key="openai-structured-schema-array-missing",
        name="structured_list",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Return a list of answers for ${query}.",
            )
        ],
    )

    rendered = prompt.render(_ToolParams(query="policies"))

    with pytest.raises(TypeError) as exc:
        module._parse_provider_payload({"wrong": []}, rendered)

    assert "Expected provider payload to be a JSON array." in str(exc.value)


def test_openai_adapter_parse_provider_payload_unwraps_wrapped_array():
    module = _reload_module()

    prompt = Prompt[list[_StructuredAnswer]](
        key="openai-structured-schema-array-wrapped",
        name="structured_list",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Return a list of answers for ${query}.",
            )
        ],
    )

    rendered = prompt.render(_ToolParams(query="policies"))

    payload = {"items": [{"answer": "Ready"}]}

    parsed = module._parse_provider_payload(payload, rendered)

    assert isinstance(parsed, list)
    assert parsed[0].answer == "Ready"


def test_openai_adapter_raises_on_invalid_parsed_payload():
    module = _reload_module()

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-parsed-error",
        name="structured",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Return the structured result only.",
            )
        ],
    )

    message = _DummyMessage(
        content=None,
        tool_calls=None,
        parsed="not-a-mapping",
    )
    response = _DummyResponse([_DummyChoice(message)])
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            _ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "response"


def test_openai_adapter_relaxes_forced_tool_choice_after_first_call():
    module = _reload_module()

    tool = Tool[_ToolParams, _ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=_simple_handler,
    )

    prompt = Prompt(
        key="openai-tools-relaxed",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    final_message = _DummyMessage(content="All done")
    second = _DummyResponse([_DummyChoice(final_message)])
    client = _DummyOpenAIClient([first, second])

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
        _ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text == "All done"

    assert len(client.completions.requests) == 2
    first_request = cast(dict[str, Any], client.completions.requests[0])
    second_request = cast(dict[str, Any], client.completions.requests[1])
    assert first_request.get("tool_choice") == forced_choice
    assert second_request.get("tool_choice") == "auto"


def test_openai_adapter_emits_events_during_evaluation() -> None:
    module = _reload_module()

    tool = Tool[_ToolParams, _ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=_simple_handler,
    )

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-events",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    first = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    second_message = _DummyMessage(content=json.dumps({"answer": "Policy summary"}))
    second = _DummyResponse([_DummyChoice(second_message)])
    client = _DummyOpenAIClient([first, second])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    bus = InProcessEventBus()
    tool_events: list[ToolInvoked] = []
    prompt_events: list[PromptExecuted] = []
    bus.subscribe(ToolInvoked, tool_events.append)
    bus.subscribe(PromptExecuted, prompt_events.append)
    result = adapter.evaluate(
        prompt,
        _ToolParams(query="policies"),
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
    assert prompt_event.response is result


def test_openai_adapter_raises_when_tool_handler_missing():
    module = _reload_module()

    tool = Tool[_ToolParams, _ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=None,
    )

    prompt = Prompt(
        key="openai-tools-missing-handler",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    response = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            _ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_openai_adapter_handles_tool_call_without_arguments():
    module = _reload_module()

    def optional_handler(params: _OptionalParams) -> ToolResult[_OptionalPayload]:
        return ToolResult(message="done", payload=_OptionalPayload(value=params.query))

    tool = Tool[_OptionalParams, _OptionalPayload](
        name="optional_tool",
        description="Uses defaults when args are missing.",
        handler=optional_handler,
    )

    prompt = Prompt(
        key="openai-optional-tool",
        name="optional",
        sections=[
            TextSection[_OptionalParams](
                title="Task",
                body="Provide data",
                tools=[tool],
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="optional_tool",
        arguments=None,
    )
    response_with_tool = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    final_response = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="All done", tool_calls=None))]
    )
    client = _DummyOpenAIClient([response_with_tool, final_response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _OptionalParams(),
        bus=NullEventBus(),
    )

    assert result.text == "All done"
    assert result.tool_results[0].params.query == "default"


def test_openai_adapter_raises_when_structured_output_missing_json():
    module = _reload_module()

    prompt = Prompt[_StructuredAnswer](
        key="openai-structured-missing-json",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
            )
        ],
    )

    response = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="no-json", tool_calls=None))]
    )
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            _ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "response"


def test_openai_adapter_raises_for_unknown_tool():
    module = _reload_module()

    prompt = Prompt(
        key="openai-unknown-tool",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="missing_tool",
        arguments=json.dumps({"query": "policies"}),
    )
    response = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            _ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_openai_adapter_raises_when_tool_params_invalid():
    module = _reload_module()

    tool = Tool[_ToolParams, _ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=_simple_handler,
    )

    prompt = Prompt(
        key="openai-invalid-tool-params",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({}),
    )
    response = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            _ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_openai_adapter_raises_when_handler_fails():
    module = _reload_module()

    def failing_handler(params: _ToolParams) -> ToolResult[_ToolPayload]:
        raise RuntimeError("boom")

    tool = Tool[_ToolParams, _ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=failing_handler,
    )

    prompt = Prompt(
        key="openai-handler-failure",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=json.dumps({"query": "policies"}),
    )
    response = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            _ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_openai_adapter_records_provider_payload_from_mapping():
    module = _reload_module()

    prompt = Prompt(
        key="openai-provider-payload",
        name="greeting",
        sections=[
            TextSection[_GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    mapping_response = _MappingResponse(
        [_DummyChoice(_DummyMessage(content="Hello!", tool_calls=None))]
    )
    client = _DummyOpenAIClient([mapping_response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.provider_payload == {"meta": "value"}


def test_openai_adapter_ignores_non_mapping_model_dump():
    module = _reload_module()

    prompt = Prompt(
        key="openai-weird-dump",
        name="greeting",
        sections=[
            TextSection[_GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    response = _WeirdResponse(
        [_DummyChoice(_DummyMessage(content="Hello!", tool_calls=None))]
    )
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.provider_payload is None


def test_openai_adapter_handles_response_without_model_dump():
    module = _reload_module()

    prompt = Prompt(
        key="openai-simple-response",
        name="greeting",
        sections=[
            TextSection[_GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    response = _SimpleResponse(
        [_DummyChoice(_DummyMessage(content="Hello!", tool_calls=None))]
    )
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    result = adapter.evaluate(
        prompt,
        _GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.provider_payload is None


@pytest.mark.parametrize(
    "arguments_json",
    ["{", json.dumps("not a dict")],
)
def test_openai_adapter_rejects_bad_tool_arguments(arguments_json: str) -> None:
    module = _reload_module()

    tool = Tool[_ToolParams, _ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=_simple_handler,
    )

    prompt = Prompt(
        key="openai-bad-tool-arguments",
        name="search",
        sections=[
            TextSection[_ToolParams](
                title="Task",
                body="Look up ${query}",
                tools=[tool],
            )
        ],
    )

    tool_call = _DummyToolCall(
        call_id="call_1",
        name="search_notes",
        arguments=arguments_json,
    )
    response = _DummyResponse(
        [_DummyChoice(_DummyMessage(content="thinking", tool_calls=[tool_call]))]
    )
    client = _DummyOpenAIClient([response])
    adapter = module.OpenAIAdapter(model="gpt-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            _ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_message_text_content_handles_structured_parts():
    module = _reload_module()

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


def test_extract_parsed_content_handles_attribute_blocks():
    module = _reload_module()

    class JsonBlock:
        def __init__(self, payload: dict[str, object]) -> None:
            self.type = "output_json"
            self.json = payload

    block = JsonBlock({"answer": "attribute"})
    message = _DummyMessage(content=[block], tool_calls=None)

    parsed = module._extract_parsed_content(message)

    assert parsed == {"answer": "attribute"}
    assert module._parsed_payload_from_part({"type": "other"}) is None

    class OtherBlock:
        def __init__(self) -> None:
            self.type = "other"
            self.json = {"answer": "ignored"}

    assert module._parsed_payload_from_part(OtherBlock()) is None


def test_parse_provider_payload_handles_array_and_unknown_containers():
    module = _reload_module()

    rendered_array = RenderedPrompt[list[_StructuredAnswer]](
        text="",
        output_type=_StructuredAnswer,
        output_container="array",
        allow_extra_keys=False,
    )
    parsed = module._parse_provider_payload(
        [{"answer": "First"}, {"answer": "Second"}],
        rendered_array,
    )
    assert [item.answer for item in parsed] == ["First", "Second"]

    with pytest.raises(TypeError):
        module._parse_provider_payload("oops", rendered_array)
    with pytest.raises(TypeError):
        module._parse_provider_payload(["oops"], rendered_array)

    rendered_invalid = RenderedPrompt[_StructuredAnswer](
        text="",
        output_type=_StructuredAnswer,
        output_container="invalid",  # type: ignore[arg-type]
        allow_extra_keys=False,
    )
    with pytest.raises(TypeError):
        module._parse_provider_payload({"answer": "value"}, rendered_invalid)

    rendered_missing = RenderedPrompt(
        text="",
        output_type=None,
        output_container=None,
        allow_extra_keys=None,
    )
    with pytest.raises(TypeError):
        module._parse_provider_payload({}, rendered_missing)
