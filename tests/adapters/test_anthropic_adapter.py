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
from collections.abc import Sequence
from importlib import import_module as std_import_module
from typing import Any, TypeVar, cast

import pytest

from weakincentives.adapters import (
    AnthropicClientConfig,
    AnthropicModelConfig,
)
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)

try:
    from tests.adapters._test_stubs import (
        GreetingParams,
        OptionalParams,
        OptionalPayload,
        StructuredAnswer,
        ToolParams,
        ToolPayload,
        simple_handler,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct invocation
    from ._test_stubs import (
        GreetingParams,
        OptionalParams,
        OptionalPayload,
        StructuredAnswer,
        ToolParams,
        ToolPayload,
        simple_handler,
    )
from tests.helpers.events import NullEventBus
from weakincentives.adapters import PromptEvaluationError
from weakincentives.budget import Budget
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    SupportsDataclass,
    Tool,
    ToolContext,
    ToolHandler,
    ToolResult,
)
from weakincentives.runtime.events import (
    InProcessEventBus,
    PromptExecuted,
    ToolInvoked,
)
from weakincentives.runtime.session import Session
from weakincentives.tools import ToolValidationError

MODULE_PATH = "weakincentives.adapters.anthropic"
PROMPT_NS = "tests/adapters/anthropic"


def _reload_module() -> types.ModuleType:
    return importlib.reload(std_import_module(MODULE_PATH))


OutputT = TypeVar("OutputT")


def _evaluate(
    adapter: ProviderAdapter[OutputT],
    prompt: PromptTemplate[OutputT],
    *params: SupportsDataclass,
    **kwargs: object,
) -> PromptResponse[OutputT]:
    bound_prompt = Prompt(prompt).bind(*params)
    return adapter.evaluate(bound_prompt, **kwargs)


def _evaluate_with_session(
    adapter: ProviderAdapter[OutputT],
    prompt: PromptTemplate[OutputT],
    *params: SupportsDataclass,
    session: SessionProtocol | None = None,
) -> PromptResponse[OutputT]:
    target_session = (
        session
        if session is not None
        else cast(SessionProtocol, Session(bus=NullEventBus()))
    )
    return _evaluate(adapter, prompt, *params, session=target_session)


# Anthropic-specific test stubs


class DummyTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class DummyToolUseBlock:
    def __init__(self, call_id: str, name: str, input_data: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.id = call_id
        self.name = name
        self.input = input_data


class DummyAnthropicResponse:
    def __init__(
        self,
        content: Sequence[object],
        *,
        stop_reason: str = "end_turn",
        usage: dict[str, int] | None = None,
    ) -> None:
        self.content = list(content)
        self.stop_reason = stop_reason
        self.usage = usage or {"input_tokens": 10, "output_tokens": 5}

    def model_dump(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
        }


class DummyAnthropicMessagesAPI:
    def __init__(self, responses: Sequence[DummyAnthropicResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> DummyAnthropicResponse:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


class DummyAnthropicBetaAPI:
    def __init__(self, messages: DummyAnthropicMessagesAPI) -> None:
        self._messages = messages

    @property
    def messages(self) -> DummyAnthropicMessagesAPI:
        return self._messages


class DummyAnthropicClient:
    def __init__(self, responses: Sequence[DummyAnthropicResponse]) -> None:
        self._beta = DummyAnthropicBetaAPI(DummyAnthropicMessagesAPI(responses))

    @property
    def beta(self) -> DummyAnthropicBetaAPI:
        return self._beta


def test_create_anthropic_client_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    def fail_import(name: str, package: str | None = None) -> types.ModuleType:
        if name == "anthropic":
            raise ModuleNotFoundError("No module named 'anthropic'")
        return std_import_module(name, package)

    monkeypatch.setattr(module, "import_module", fail_import)

    with pytest.raises(RuntimeError) as err:
        module.create_anthropic_client()

    message = str(err.value)
    assert "uv sync --extra anthropic" in message
    assert "pip install weakincentives[anthropic]" in message


def test_create_anthropic_client_returns_anthropic_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    class DummyAnthropic:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    dummy_module = cast(Any, types.ModuleType("anthropic"))
    dummy_module.Anthropic = DummyAnthropic

    monkeypatch.setitem(sys.modules, "anthropic", dummy_module)

    client = module.create_anthropic_client(api_key="secret-key")

    assert isinstance(client, DummyAnthropic)
    assert client.kwargs == {"api_key": "secret-key"}


def test_anthropic_adapter_constructs_client_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-greeting",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicResponse([DummyTextBlock("Hello, Sam!")])
    client = DummyAnthropicClient([response])
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> DummyAnthropicClient:
        captured_kwargs.append(dict(kwargs))
        return client

    monkeypatch.setattr(module, "create_anthropic_client", fake_factory)

    adapter = module.AnthropicAdapter(
        model="claude-test",
        client_config=AnthropicClientConfig(api_key="secret-key"),
    )

    result = _evaluate_with_session(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.text == "Hello, Sam!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_anthropic_adapter_rejects_client_config_with_explicit_client() -> None:
    module = cast(Any, _reload_module())
    client = DummyAnthropicClient([])

    with pytest.raises(ValueError):
        module.AnthropicAdapter(
            model="claude-test",
            client=client,
            client_config=AnthropicClientConfig(api_key="secret"),
        )


def test_anthropic_adapter_uses_model_config() -> None:
    module = cast(Any, _reload_module())

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-config",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicResponse([DummyTextBlock("Hello with temp!")])
    client = DummyAnthropicClient([response])

    adapter = module.AnthropicAdapter(
        model="claude-test",
        client=client,
        model_config=AnthropicModelConfig(temperature=0.5, max_tokens=100, top_k=40),
    )

    result = _evaluate_with_session(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.text == "Hello with temp!"
    # Verify model_config params were included in request
    request = client.beta.messages.requests[0]
    assert request["temperature"] == 0.5
    assert request["max_tokens"] == 100
    assert request["top_k"] == 40


def test_anthropic_adapter_returns_plain_text_response() -> None:
    module = cast(Any, _reload_module())

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-plain",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicResponse([DummyTextBlock("Hello, Sam!")])
    client = DummyAnthropicClient([response])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    result = _evaluate_with_session(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.prompt_name == "greeting"
    assert result.text == "Hello, Sam!"
    assert result.output is None

    request = cast(dict[str, Any], client.beta.messages.requests[0])
    assert "system" in request
    assert str(request["system"]).startswith("## 1. Greeting")
    assert "tools" not in request


def test_anthropic_adapter_executes_tools_and_parses_output() -> None:
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

    prompt = PromptTemplate[StructuredAnswer](
        ns=PROMPT_NS,
        key="anthropic-structured-success",
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

    tool_use = DummyToolUseBlock(
        call_id="call_1",
        name="search_notes",
        input_data={"query": "policies"},
    )
    first = DummyAnthropicResponse(
        [DummyTextBlock("thinking"), tool_use], stop_reason="tool_use"
    )
    second = DummyAnthropicResponse(
        [DummyTextBlock(json.dumps({"answer": "Policy summary"}))]
    )
    client = DummyAnthropicClient([first, second])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    tool_events: list[ToolInvoked] = []
    bus.subscribe(
        ToolInvoked, lambda event: tool_events.append(cast(ToolInvoked, event))
    )
    result = _evaluate_with_session(
        adapter,
        prompt,
        ToolParams(query="policies"),
        session=session,
    )

    assert result.text is None
    assert result.output == StructuredAnswer(answer="Policy summary")
    assert len(tool_events) == 1
    record = tool_events[0]
    assert record.name == "search_notes"
    assert isinstance(record.result.value, ToolPayload)
    assert record.call_id == "call_1"
    assert calls == ["policies"]

    first_request = cast(dict[str, Any], client.beta.messages.requests[0])
    tools = cast(list[dict[str, Any]], first_request["tools"])
    tool_spec = tools[0]
    assert tool_spec["name"] == "search_notes"
    assert tool_spec["input_schema"]["type"] == "object"


def test_anthropic_adapter_surfaces_tool_validation_errors() -> None:
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

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-tool-validation",
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

    tool_use = DummyToolUseBlock(
        call_id="call_1",
        name="search_notes",
        input_data={"query": "invalid"},
    )
    first = DummyAnthropicResponse([tool_use], stop_reason="tool_use")
    second = DummyAnthropicResponse(
        [DummyTextBlock("Please provide a different query.")]
    )
    client = DummyAnthropicClient([first, second])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    tool_events: list[ToolInvoked] = []

    def record_tool_event(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record_tool_event)

    result = _evaluate(
        adapter,
        prompt,
        ToolParams(query="invalid"),
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


def test_anthropic_adapter_raises_when_tool_handler_missing() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=None,
    )

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-tools-missing-handler",
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

    tool_use = DummyToolUseBlock(
        call_id="call_1",
        name="search_notes",
        input_data={"query": "policies"},
    )
    response = DummyAnthropicResponse([tool_use], stop_reason="tool_use")
    client = DummyAnthropicClient([response])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_session(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == PROMPT_EVALUATION_PHASE_TOOL


def test_anthropic_adapter_handles_tool_call_without_arguments() -> None:
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

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-optional-tool",
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

    tool_use = DummyToolUseBlock(
        call_id="call_1",
        name="optional_tool",
        input_data={},
    )
    first = DummyAnthropicResponse([tool_use], stop_reason="tool_use")
    second = DummyAnthropicResponse([DummyTextBlock("All done")])
    client = DummyAnthropicClient([first, second])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    tool_events: list[ToolInvoked] = []
    bus.subscribe(
        ToolInvoked, lambda event: tool_events.append(cast(ToolInvoked, event))
    )
    result = _evaluate_with_session(
        adapter,
        prompt,
        OptionalParams(),
        session=session,
    )

    assert result.text == "All done"
    assert tool_events
    record = tool_events[0]
    params = cast(OptionalParams, record.params)
    assert params.query == "default"


def test_anthropic_adapter_emits_events_during_evaluation() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    prompt = PromptTemplate[StructuredAnswer](
        ns=PROMPT_NS,
        key="anthropic-structured-events",
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

    tool_use = DummyToolUseBlock(
        call_id="call_1",
        name="search_notes",
        input_data={"query": "policies"},
    )
    first = DummyAnthropicResponse(
        [DummyTextBlock("thinking"), tool_use], stop_reason="tool_use"
    )
    second = DummyAnthropicResponse(
        [DummyTextBlock(json.dumps({"answer": "Policy summary"}))]
    )
    client = DummyAnthropicClient([first, second])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    bus = InProcessEventBus()
    session = Session(bus=bus)
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
    result = _evaluate_with_session(
        adapter,
        prompt,
        ToolParams(query="policies"),
        session=session,
    )

    assert len(tool_events) == 1
    tool_event = tool_events[0]
    assert tool_event.prompt_name == "search"
    assert tool_event.adapter == "anthropic"
    assert tool_event.name == "search_notes"
    assert tool_event.call_id == "call_1"

    assert len(prompt_events) == 1
    prompt_event = prompt_events[0]
    assert prompt_event.prompt_name == "search"
    assert prompt_event.adapter == "anthropic"
    assert prompt_event.result is result


def test_anthropic_adapter_raises_on_invalid_parsed_payload() -> None:
    module = cast(Any, _reload_module())

    prompt = PromptTemplate[StructuredAnswer](
        ns=PROMPT_NS,
        key="anthropic-structured-missing-json",
        name="search",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
            )
        ],
    )

    response = DummyAnthropicResponse([DummyTextBlock("no-json")])
    client = DummyAnthropicClient([response])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_session(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == PROMPT_EVALUATION_PHASE_RESPONSE


def test_anthropic_adapter_raises_for_unknown_tool() -> None:
    module = cast(Any, _reload_module())

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-unknown-tool",
        name="search",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Look up ${query}",
            )
        ],
    )

    tool_use = DummyToolUseBlock(
        call_id="call_1",
        name="missing_tool",
        input_data={"query": "policies"},
    )
    response = DummyAnthropicResponse([tool_use], stop_reason="tool_use")
    client = DummyAnthropicClient([response])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    with pytest.raises(PromptEvaluationError) as err:
        _evaluate_with_session(
            adapter,
            prompt,
            ToolParams(query="policies"),
        )

    assert isinstance(err.value, PromptEvaluationError)
    assert err.value.phase == PROMPT_EVALUATION_PHASE_TOOL


def test_anthropic_adapter_creates_budget_tracker_when_budget_provided() -> None:
    module = cast(Any, _reload_module())

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-budget-test",
        name="budget_test",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicResponse([DummyTextBlock("Hello!")])
    client = DummyAnthropicClient([response])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    budget = Budget(max_total_tokens=1000)
    bus = InProcessEventBus()
    session = Session(bus=bus)

    result = adapter.evaluate(
        Prompt(prompt).bind(GreetingParams(user="Test")),
        session=session,
        budget=budget,
    )

    assert result.text == "Hello!"


def test_anthropic_adapter_normalizes_messages_for_anthropic() -> None:
    module = cast(Any, _reload_module())

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Result here"},
    ]

    system_prompt, normalized = module._normalize_messages_for_anthropic(messages)

    assert system_prompt == "You are helpful."
    assert len(normalized) == 4  # user, assistant, assistant+tool, tool_result

    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"] == "Hello"

    assert normalized[1]["role"] == "assistant"
    assert normalized[1]["content"] == "Hi there!"

    # Assistant with tool calls
    assert normalized[2]["role"] == "assistant"
    content = normalized[2]["content"]
    assert len(content) == 2  # text block + tool_use block
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Let me check"
    assert content[1]["type"] == "tool_use"
    assert content[1]["id"] == "call_1"
    assert content[1]["name"] == "search"

    # Tool result
    assert normalized[3]["role"] == "user"
    tool_result_content = normalized[3]["content"]
    assert len(tool_result_content) == 1
    assert tool_result_content[0]["type"] == "tool_result"
    assert tool_result_content[0]["tool_use_id"] == "call_1"


def test_anthropic_tool_choice_conversion() -> None:
    module = cast(Any, _reload_module())

    assert module._anthropic_tool_choice(None) is None
    assert module._anthropic_tool_choice("auto") == {"type": "auto"}

    # Function-style tool choice
    tool_choice = {"type": "function", "function": {"name": "search"}}
    assert module._anthropic_tool_choice(tool_choice) == {
        "type": "tool",
        "name": "search",
    }

    # Top-level name
    tool_choice = {"type": "function", "name": "search"}
    assert module._anthropic_tool_choice(tool_choice) == {
        "type": "tool",
        "name": "search",
    }

    # Unknown mapping type returns auto
    unknown_choice: dict[str, str] = {"type": "unknown"}
    assert module._anthropic_tool_choice(unknown_choice) == {"type": "auto"}


def test_anthropic_model_config_rejects_unsupported_params() -> None:
    with pytest.raises(ValueError) as err:
        AnthropicModelConfig(seed=42)

    assert "seed" in str(err.value)

    with pytest.raises(ValueError) as err:
        AnthropicModelConfig(presence_penalty=0.5)

    assert "presence_penalty" in str(err.value)

    with pytest.raises(ValueError) as err:
        AnthropicModelConfig(frequency_penalty=0.5)

    assert "frequency_penalty" in str(err.value)


def test_anthropic_model_config_to_request_params() -> None:
    config = AnthropicModelConfig(
        temperature=0.7,
        max_tokens=4096,
        top_p=0.9,
        stop=("STOP", "END"),
        top_k=40,
        metadata={"user_id": "test123"},
    )

    params = config.to_request_params()

    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 4096
    assert params["top_p"] == 0.9
    assert params["stop_sequences"] == ["STOP", "END"]
    assert params["top_k"] == 40
    assert params["metadata"] == {"user_id": "test123"}


def test_anthropic_client_config_to_client_kwargs() -> None:
    config = AnthropicClientConfig(
        api_key="sk-test",
        base_url="https://api.example.com",
        timeout=30.0,
        max_retries=3,
    )

    kwargs = config.to_client_kwargs()

    assert kwargs["api_key"] == "sk-test"
    assert kwargs["base_url"] == "https://api.example.com"
    assert kwargs["timeout"] == 30.0
    assert kwargs["max_retries"] == 3


def test_anthropic_adapter_uses_default_model() -> None:
    module = cast(Any, _reload_module())

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-default-model",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicResponse([DummyTextBlock("Hello!")])
    client = DummyAnthropicClient([response])
    adapter = module.AnthropicAdapter(client=client)  # No model specified

    result = _evaluate_with_session(
        adapter,
        prompt,
        GreetingParams(user="Sam"),
    )

    assert result.text == "Hello!"
    request = client.beta.messages.requests[0]
    assert request["model"] == "claude-opus-4-5-20250929"


def test_anthropic_adapter_includes_structured_output_beta() -> None:
    module = cast(Any, _reload_module())

    prompt = PromptTemplate[StructuredAnswer](
        ns=PROMPT_NS,
        key="anthropic-structured-beta",
        name="structured",
        sections=[
            MarkdownSection[ToolParams](
                title="Task",
                key="task",
                template="Return structured JSON for ${query}.",
            )
        ],
    )

    response = DummyAnthropicResponse(
        [DummyTextBlock(json.dumps({"answer": "Structured"}))]
    )
    client = DummyAnthropicClient([response])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    result = _evaluate_with_session(
        adapter,
        prompt,
        ToolParams(query="test"),
    )

    assert result.output == StructuredAnswer(answer="Structured")

    request = cast(dict[str, Any], client.beta.messages.requests[0])
    assert "betas" in request
    betas = cast(list[str], request["betas"])
    assert "structured-outputs-2025-11-13" in betas
    assert "output_format" in request


def test_anthropic_tool_to_anthropic_spec() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    spec = module.tool_to_anthropic_spec(tool)

    assert spec["name"] == "search_notes"
    assert spec["description"] == "Search stored notes."
    assert "input_schema" in spec
    assert spec["input_schema"]["type"] == "object"


def test_anthropic_tool_to_anthropic_spec_with_strict() -> None:
    module = cast(Any, _reload_module())

    tool = Tool[ToolParams, ToolPayload](
        name="search_notes",
        description="Search stored notes.",
        handler=simple_handler,
    )

    spec = module.tool_to_anthropic_spec(tool, strict=True)

    assert spec["strict"] is True


def test_anthropic_extract_content() -> None:
    module = cast(Any, _reload_module())

    class TextBlock:
        def __init__(self) -> None:
            self.type = "text"
            self.text = "Hello"

    class ToolBlock:
        def __init__(self) -> None:
            self.type = "tool_use"
            self.id = "call_1"
            self.name = "search"
            self.input = {"q": "test"}

    class Response:
        def __init__(self) -> None:
            self.content = [TextBlock(), ToolBlock()]

    text, tool_uses = module._extract_anthropic_content(Response())

    assert text == "Hello"
    assert len(tool_uses) == 1


def test_anthropic_coerce_retry_after() -> None:
    module = cast(Any, _reload_module())
    from datetime import timedelta

    # None returns None
    assert module._coerce_retry_after(None) is None

    # timedelta returns itself if positive
    assert module._coerce_retry_after(timedelta(seconds=10)) == timedelta(seconds=10)
    assert module._coerce_retry_after(timedelta(seconds=0)) is None
    assert module._coerce_retry_after(timedelta(seconds=-5)) is None

    # Numeric values
    assert module._coerce_retry_after(30) == timedelta(seconds=30)
    assert module._coerce_retry_after(30.5) == timedelta(seconds=30.5)
    assert module._coerce_retry_after(-10) is None

    # String digit values
    assert module._coerce_retry_after("60") == timedelta(seconds=60)

    # Non-digit string returns None
    assert module._coerce_retry_after("abc") is None


def test_anthropic_retry_after_from_headers() -> None:
    module = cast(Any, _reload_module())
    from datetime import timedelta

    # None headers
    assert module._retry_after_from_headers(None) is None

    # Header with retry-after
    headers = {"retry-after": "30"}
    assert module._retry_after_from_headers(headers) == timedelta(seconds=30)

    # Header without retry-after
    assert module._retry_after_from_headers({}) is None


def test_anthropic_retry_after_from_error() -> None:
    module = cast(Any, _reload_module())
    from datetime import timedelta

    class ErrorWithRetryAfter:
        def __init__(self) -> None:
            self.retry_after = 45

    assert module._retry_after_from_error(ErrorWithRetryAfter()) == timedelta(
        seconds=45
    )

    class ErrorWithHeaders:
        def __init__(self) -> None:
            self.retry_after = None
            self.headers = {"retry-after": "60"}

    assert module._retry_after_from_error(ErrorWithHeaders()) == timedelta(seconds=60)

    class ErrorWithResponse:
        def __init__(self) -> None:
            self.retry_after = None
            self.headers = None
            self.response = {"retry_after": 90}

    assert module._retry_after_from_error(ErrorWithResponse()) == timedelta(seconds=90)

    class ErrorWithResponseHeaders:
        def __init__(self) -> None:
            self.retry_after = None
            self.headers = None
            self.response = {"headers": {"retry-after": "120"}}

    assert module._retry_after_from_error(ErrorWithResponseHeaders()) == timedelta(
        seconds=120
    )

    class EmptyError:
        pass

    assert module._retry_after_from_error(EmptyError()) is None


def test_anthropic_error_payload() -> None:
    module = cast(Any, _reload_module())

    class ErrorWithResponse:
        def __init__(self) -> None:
            self.response = {"error": "something went wrong", "code": 500}

    payload = module._error_payload(ErrorWithResponse())
    assert payload == {"error": "something went wrong", "code": 500}

    class ErrorWithBody:
        def __init__(self) -> None:
            self.response = None
            self.body = {"message": "error body"}

    payload = module._error_payload(ErrorWithBody())
    assert payload == {"message": "error body"}

    class EmptyError:
        pass

    assert module._error_payload(EmptyError()) is None


def test_anthropic_normalize_throttle_rate_limit() -> None:
    module = cast(Any, _reload_module())

    class RateLimitError(Exception):
        status_code = 429

    error = RateLimitError("Rate limit exceeded")
    throttle = module._normalize_anthropic_throttle(error, prompt_name="test")

    assert throttle is not None
    assert throttle.details.kind == "rate_limit"


def test_anthropic_normalize_throttle_overloaded() -> None:
    module = cast(Any, _reload_module())

    class OverloadedError(Exception):
        status_code = 529

    error = OverloadedError("API overloaded")
    throttle = module._normalize_anthropic_throttle(error, prompt_name="test")

    assert throttle is not None
    assert throttle.details.kind == "rate_limit"


def test_anthropic_normalize_throttle_timeout() -> None:
    module = cast(Any, _reload_module())

    class ConnectionTimeoutError(Exception):
        pass

    error = ConnectionTimeoutError("Connection timeout")
    throttle = module._normalize_anthropic_throttle(error, prompt_name="test")

    assert throttle is not None
    assert throttle.details.kind == "timeout"


def test_anthropic_normalize_throttle_no_match() -> None:
    module = cast(Any, _reload_module())

    class OtherError(Exception):
        status_code = 500

    error = OtherError("Internal server error")
    throttle = module._normalize_anthropic_throttle(error, prompt_name="test")

    assert throttle is None


def test_anthropic_normalize_messages_json_decode_error() -> None:
    module = cast(Any, _reload_module())

    messages = [
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "search", "arguments": "{invalid json}"},
                }
            ],
        },
    ]

    _, normalized = module._normalize_messages_for_anthropic(messages)

    assert len(normalized) == 1
    content = normalized[0]["content"]
    assert len(content) == 2  # text + tool_use
    tool_use = content[1]
    assert tool_use["type"] == "tool_use"
    assert tool_use["input"] == {}  # JSON decode error fallback


def test_anthropic_normalize_messages_dict_arguments() -> None:
    module = cast(Any, _reload_module())

    messages = [
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "search", "arguments": {"query": "test"}},
                }
            ],
        },
    ]

    _, normalized = module._normalize_messages_for_anthropic(messages)

    assert len(normalized) == 1
    content = normalized[0]["content"]
    tool_use = content[1]
    assert tool_use["input"] == {"query": "test"}


def test_anthropic_normalize_messages_unknown_role() -> None:
    module = cast(Any, _reload_module())

    messages = [
        {"role": "custom_role", "content": "custom content", "extra_field": "value"},
    ]

    _, normalized = module._normalize_messages_for_anthropic(messages)

    assert len(normalized) == 1
    assert normalized[0]["role"] == "custom_role"
    assert normalized[0]["extra_field"] == "value"


def test_anthropic_tool_to_spec_with_none_params() -> None:
    from typing import Literal

    module = cast(Any, _reload_module())

    def handler(
        params: Literal[None], *, context: ToolContext
    ) -> ToolResult[ToolPayload]:
        return ToolResult(message="ok", value=ToolPayload(answer="result"))

    tool = Tool[None, ToolPayload](
        name="no_params_tool",
        description="Tool with no params.",
        handler=handler,
    )

    spec = module.tool_to_anthropic_spec(tool)

    assert spec["name"] == "no_params_tool"
    assert spec["input_schema"] == {"type": "object", "properties": {}}


def test_anthropic_extract_content_no_sequence() -> None:
    module = cast(Any, _reload_module())

    class Response:
        def __init__(self) -> None:
            self.content = 12345  # Integer, not a sequence

    text, tool_uses = module._extract_anthropic_content(Response())

    assert text == ""
    assert tool_uses == []


def test_anthropic_tool_calls_string_input() -> None:
    module = cast(Any, _reload_module())

    class ToolUse:
        def __init__(self) -> None:
            self.type = "tool_use"
            self.id = "call_1"
            self.name = "search"
            self.input = '{"query": "test"}'  # String input

    tool_calls = module._tool_calls_from_anthropic([ToolUse()])

    assert len(tool_calls) == 1
    assert tool_calls[0].function.arguments == '{"query": "test"}'


def test_anthropic_tool_calls_other_input_type() -> None:
    module = cast(Any, _reload_module())

    class ToolUse:
        def __init__(self) -> None:
            self.type = "tool_use"
            self.id = "call_1"
            self.name = "search"
            self.input = ["list", "input"]  # Not str or Mapping

    tool_calls = module._tool_calls_from_anthropic([ToolUse()])

    assert len(tool_calls) == 1
    assert tool_calls[0].function.arguments == '["list", "input"]'


def test_anthropic_choice_from_response_tool_use_without_calls() -> None:
    module = cast(Any, _reload_module())

    class Response:
        def __init__(self) -> None:
            self.content = []  # Empty content
            self.stop_reason = "tool_use"

    with pytest.raises(PromptEvaluationError) as err:
        module._choice_from_anthropic_response(Response(), prompt_name="test")

    assert "tool_use but no tool calls" in str(err.value)


def test_anthropic_build_output_format_returns_none() -> None:
    module = cast(Any, _reload_module())

    # Use a plain prompt without structured output
    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-no-structured",
        name="plain",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    rendered = Prompt(prompt).bind(GreetingParams(user="Sam")).render()
    result = module._build_anthropic_output_format(rendered, "test")

    assert result is None


def test_anthropic_adapter_with_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC, datetime, timedelta

    from weakincentives.deadlines import Deadline

    module = cast(Any, _reload_module())

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-deadline",
        name="deadline",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicResponse([DummyTextBlock("Hello!")])
    client = DummyAnthropicClient([response])

    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> DummyAnthropicClient:
        captured_kwargs.append(dict(kwargs))
        return client

    monkeypatch.setattr(module, "create_anthropic_client", fake_factory)

    adapter = module.AnthropicAdapter(model="claude-test")

    # Create a valid future deadline
    expires = datetime.now(UTC) + timedelta(minutes=1)
    deadline = Deadline(expires_at=expires)
    bus = InProcessEventBus()
    session = Session(bus=bus)

    result = adapter.evaluate(
        Prompt(prompt).bind(GreetingParams(user="Sam")),
        session=session,
        deadline=deadline,
    )

    assert result.text == "Hello!"


def test_anthropic_adapter_expired_deadline_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import timedelta
    from unittest.mock import Mock

    module = cast(Any, _reload_module())

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-expired-deadline",
        name="expired_deadline",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )

    client = DummyAnthropicClient([])

    def fake_factory(**kwargs: object) -> DummyAnthropicClient:
        return client

    monkeypatch.setattr(module, "create_anthropic_client", fake_factory)

    adapter = module.AnthropicAdapter(model="claude-test")

    # Create a mock deadline that returns negative remaining time
    mock_deadline = Mock()
    mock_deadline.remaining.return_value = timedelta(seconds=-1)

    bus = InProcessEventBus()
    session = Session(bus=bus)

    with pytest.raises(PromptEvaluationError) as err:
        adapter.evaluate(
            Prompt(prompt).bind(GreetingParams(user="Sam")),
            session=session,
            deadline=mock_deadline,
        )

    assert "Deadline expired" in str(err.value)


def test_anthropic_adapter_handles_handler_failures() -> None:
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

    prompt = PromptTemplate(
        ns=PROMPT_NS,
        key="anthropic-handler-failure",
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

    tool_use = DummyToolUseBlock(
        call_id="call_1",
        name="search_notes",
        input_data={"query": "policies"},
    )
    first = DummyAnthropicResponse([tool_use], stop_reason="tool_use")
    second = DummyAnthropicResponse(
        [DummyTextBlock("Please provide a different approach.")]
    )
    client = DummyAnthropicClient([first, second])
    adapter = module.AnthropicAdapter(model="claude-test", client=client)

    bus = InProcessEventBus()
    session = Session(bus=bus)
    tool_events: list[ToolInvoked] = []

    def record(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        tool_events.append(event)

    bus.subscribe(ToolInvoked, record)

    result = _evaluate(
        adapter,
        prompt,
        ToolParams(query="policies"),
        session=cast(SessionProtocol, session),
    )

    assert result.text == "Please provide a different approach."
    assert result.output is None
    assert len(tool_events) == 1
    event = tool_events[0]
    assert event.result.success is False
    assert event.result.value is None
    assert "execution failed: boom" in event.result.message
