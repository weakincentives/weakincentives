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
from typing import Any, cast

import pytest

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
from weakincentives.adapters import PromptEvaluationError
from weakincentives.events import (
    InProcessEventBus,
    NullEventBus,
    PromptExecuted,
    ToolInvoked,
)
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult

MODULE_PATH = "weakincentives.adapters.litellm"


def _reload_module():
    return importlib.reload(std_import_module(MODULE_PATH))


def test_create_litellm_completion_requires_optional_dependency(monkeypatch):
    module = _reload_module()

    def fail_import(name: str, package: str | None = None):
        if name == "litellm":
            raise ModuleNotFoundError("No module named 'litellm'")
        return std_import_module(name, package)

    monkeypatch.setattr(module, "import_module", fail_import)

    with pytest.raises(RuntimeError) as err:
        module.create_litellm_completion()

    message = str(err.value)
    assert "uv sync --extra litellm" in message
    assert "pip install weakincentives[litellm]" in message


def test_create_litellm_completion_wraps_kwargs(monkeypatch):
    module = _reload_module()

    captured_kwargs: list[dict[str, object]] = []

    class DummyLiteLLM(types.SimpleNamespace):
        def completion(self, **kwargs: object) -> DummyResponse:
            captured_kwargs.append(dict(kwargs))
            message = DummyMessage(content="Hello", tool_calls=None)
            return DummyResponse([DummyChoice(message)])

    dummy_module = cast(module._LiteLLMModule, DummyLiteLLM())
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


def test_create_litellm_completion_returns_direct_callable(monkeypatch):
    module = _reload_module()

    class DummyLiteLLM(types.SimpleNamespace):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, object]] = []

        def completion(self, **kwargs: object) -> DummyResponse:
            self.calls.append(dict(kwargs))
            message = DummyMessage(content="Hi", tool_calls=None)
            return DummyResponse([DummyChoice(message)])

    dummy_module = cast(module._LiteLLMModule, DummyLiteLLM())
    monkeypatch.setitem(sys.modules, "litellm", dummy_module)

    completion = module.create_litellm_completion()
    result = completion(model="gpt", messages=[{"role": "system", "content": "hi"}])

    assert isinstance(result, DummyResponse)
    assert dummy_module.calls == [
        {"model": "gpt", "messages": [{"role": "system", "content": "hi"}]}
    ]


def test_litellm_adapter_constructs_completion_when_not_provided(monkeypatch):
    module = _reload_module()

    prompt = Prompt(
        key="litellm-greeting",
        name="greeting",
        sections=[
            TextSection[GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
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

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.text == "Hello!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_litellm_adapter_supports_custom_completion_factory():
    module = _reload_module()

    prompt = Prompt(
        key="litellm-greeting",
        name="greeting",
        sections=[
            TextSection[GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
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

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.text == "Hello again!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_litellm_adapter_rejects_completion_kwargs_with_explicit_completion() -> None:
    module = _reload_module()
    completion = RecordingCompletion([])

    with pytest.raises(ValueError):
        module.LiteLLMAdapter(
            model="gpt-test",
            completion=completion,
            completion_kwargs={"api_key": "secret"},
        )


def test_litellm_adapter_rejects_completion_factory_with_explicit_completion() -> None:
    module = _reload_module()
    completion = RecordingCompletion([])

    with pytest.raises(ValueError):
        module.LiteLLMAdapter(
            model="gpt-test",
            completion=completion,
            completion_factory=lambda **_: completion,
        )


def test_litellm_adapter_returns_plain_text_response():
    module = _reload_module()

    prompt = Prompt(
        key="litellm-plain",
        name="greeting",
        sections=[
            TextSection[GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    response = DummyResponse(
        [DummyChoice(DummyMessage(content="Hello, Sam!", tool_calls=None))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.prompt_name == "greeting"
    assert result.text == "Hello, Sam!"
    assert result.output is None
    assert result.tool_results == ()

    request = completion.requests[0]
    messages = cast(list[dict[str, Any]], request["messages"])
    assert messages[0]["role"] == "system"
    assert str(messages[0]["content"]).startswith("## Greeting")
    assert "tools" not in request


def test_litellm_adapter_executes_tools_and_parses_output():
    module = _reload_module()

    calls: list[str] = []

    def handler(params: ToolParams) -> ToolResult[ToolPayload]:
        calls.append(params.query)
        payload = ToolPayload(answer=f"Result for {params.query}")
        return ToolResult(message="completed", payload=payload)

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=handler,
    )

    prompt = Prompt[StructuredAnswer](
        key="litellm-structured-success",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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

    result = adapter.evaluate(
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Policy summary")
    assert calls == ["policies"]

    first_request = completion.requests[0]
    tools = cast(list[dict[str, Any]], first_request["tools"])
    function_spec = cast(dict[str, Any], tools[0]["function"])
    assert function_spec["name"] == "search_notes"
    assert first_request.get("tool_choice") == "auto"

    second_request = completion.requests[1]
    second_messages = cast(list[dict[str, Any]], second_request["messages"])
    tool_message = second_messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == "completed"
    assert "payload" not in tool_message


def test_litellm_adapter_relaxes_forced_tool_choice_after_first_call():
    module = _reload_module()

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        key="litellm-tools-relaxed",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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

    result = adapter.evaluate(
        prompt,
        ToolParams(query="policies"),
        bus=NullEventBus(),
    )

    assert result.text == "All done"
    assert len(completion.requests) == 2
    assert completion.requests[0].get("tool_choice") == forced_choice
    assert completion.requests[1].get("tool_choice") == "auto"


def test_litellm_adapter_handles_tool_call_without_arguments() -> None:
    module = _reload_module()

    recorded: list[str] = []

    def handler(params: OptionalParams) -> ToolResult[OptionalPayload]:
        recorded.append(params.query)
        payload = OptionalPayload(value=params.query)
        return ToolResult(message="used default", payload=payload)

    tool = Tool[OptionalParams, OptionalPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=handler,
    )

    prompt = Prompt(
        key="litellm-tool-no-args",
        name="search",
        sections=[
            TextSection[OptionalParams](
                title="Task",
                body="Look up ${query}",
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

    result = adapter.evaluate(
        prompt,
        OptionalParams(),
        bus=NullEventBus(),
    )

    assert result.text == "All done"
    assert recorded == ["default"]


def test_litellm_adapter_emits_events_during_evaluation() -> None:
    module = _reload_module()

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt[StructuredAnswer](
        key="litellm-structured-events",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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
    bus.subscribe(ToolInvoked, tool_events.append)
    bus.subscribe(PromptExecuted, prompt_events.append)
    result = adapter.evaluate(
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
    assert tool_event is result.tool_results[0]

    assert len(prompt_events) == 1
    prompt_event = prompt_events[0]
    assert prompt_event.prompt_name == "search"
    assert prompt_event.adapter == "litellm"
    assert prompt_event.response is result


def test_litellm_adapter_raises_when_tool_handler_missing():
    module = _reload_module()

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=None,
    )

    prompt = Prompt(
        key="litellm-handler-missing",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_litellm_adapter_raises_when_tool_not_registered():
    module = _reload_module()

    prompt = Prompt(
        key="litellm-missing-tool",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_litellm_adapter_raises_when_tool_params_invalid():
    module = _reload_module()

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        key="litellm-invalid-tool-params",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_litellm_adapter_raises_when_handler_fails():
    module = _reload_module()

    def failing_handler(params: ToolParams) -> ToolResult[ToolPayload]:
        raise RuntimeError("boom")

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=failing_handler,
    )

    prompt = Prompt(
        key="litellm-handler-failure",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_litellm_adapter_records_provider_payload_from_mapping():
    module = _reload_module()

    prompt = Prompt(
        key="litellm-provider-payload",
        name="greeting",
        sections=[
            TextSection[GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    mapping_response = MappingResponse(
        [DummyChoice(DummyMessage(content="Hello!", tool_calls=None))]
    )
    completion = RecordingCompletion([mapping_response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.provider_payload == {"meta": "value"}


def test_litellm_adapter_ignores_non_mapping_model_dump():
    module = _reload_module()

    prompt = Prompt(
        key="litellm-weird-dump",
        name="greeting",
        sections=[
            TextSection[GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    response = WeirdResponse(
        [DummyChoice(DummyMessage(content="Hello!", tool_calls=None))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        bus=NullEventBus(),
    )

    assert result.provider_payload is None


def test_litellm_adapter_handles_response_without_model_dump():
    module = _reload_module()

    prompt = Prompt(
        key="litellm-simple-response",
        name="greeting",
        sections=[
            TextSection[GreetingParams](
                title="Greeting",
                body="Say hello to ${user}.",
            )
        ],
    )

    response = SimpleResponse(
        [DummyChoice(DummyMessage(content="Hello!", tool_calls=None))]
    )
    completion = RecordingCompletion([response])
    adapter = module.LiteLLMAdapter(model="gpt-test", completion=completion)

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
def test_litellm_adapter_rejects_bad_tool_arguments(arguments_json: str) -> None:
    module = _reload_module()

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt(
        key="litellm-bad-tool-arguments",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "tool"


def test_litellm_adapter_propagates_parse_errors_for_structured_output():
    module = _reload_module()

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = Prompt[StructuredAnswer](
        key="litellm-structured-error",
        name="search",
        sections=[
            TextSection[ToolParams](
                title="Task",
                body="Look up ${query}",
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
        adapter.evaluate(
            prompt,
            ToolParams(query="policies"),
            bus=NullEventBus(),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.stage == "response"
