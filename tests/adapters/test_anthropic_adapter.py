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

import importlib
import json
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module as std_import_module
from typing import Any, cast

import pytest

from weakincentives.adapters import PromptEvaluationError

try:  # pragma: no cover - fallback for direct invocation
    from tests.adapters._test_stubs import (
        DummyAnthropicClient,
        DummyAnthropicMessage,
        DummyAnthropicTextBlock,
        DummyAnthropicToolUseBlock,
        GreetingParams,
        StructuredAnswer,
        ToolParams,
        ToolPayload,
        simple_handler,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for pytest -k
    from ._test_stubs import (
        DummyAnthropicClient,
        DummyAnthropicMessage,
        DummyAnthropicTextBlock,
        DummyAnthropicToolUseBlock,
        GreetingParams,
        StructuredAnswer,
        ToolParams,
        ToolPayload,
        simple_handler,
    )
from weakincentives.adapters.anthropic import AnthropicAdapter, AnthropicProtocol
from weakincentives.events import InProcessEventBus, NullEventBus, PromptExecuted
from weakincentives.prompt import MarkdownSection, Prompt, Tool
from weakincentives.session import Session

MODULE_PATH = "weakincentives.adapters.anthropic"
PROMPT_NS = "tests/adapters/anthropic"


def _reload_module() -> types.ModuleType:
    return importlib.reload(std_import_module(MODULE_PATH))


class MappingAnthropicResponse:
    def __init__(self, content: list[object]) -> None:
        self.content = content

    def model_dump(self) -> dict[str, object]:
        serialised: list[dict[str, object]] = []
        for block in self.content:
            if isinstance(block, Mapping):
                serialised.append({str(key): value for key, value in block.items()})
            else:
                serialised.append({"type": "text", "text": str(block)})
        return {"content": serialised, "meta": "value"}


@dataclass(slots=True)
class SimpleAnthropicResponse:
    content: list[object]


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


def test_create_anthropic_client_returns_client_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    class DummyAnthropic:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    dummy_module = cast(Any, types.ModuleType("anthropic"))
    dummy_module.Anthropic = DummyAnthropic

    monkeypatch.setitem(sys.modules, "anthropic", dummy_module)

    client = module.create_anthropic_client(api_key="secret")

    assert isinstance(client, DummyAnthropic)
    assert client.kwargs == {"api_key": "secret"}


def test_anthropic_adapter_constructs_client_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    prompt = Prompt(
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

    response = DummyAnthropicMessage([DummyAnthropicTextBlock(text="Hello!")])
    dummy_client = DummyAnthropicClient([response])
    client = cast(AnthropicProtocol, dummy_client)
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> AnthropicProtocol:
        captured_kwargs.append(dict(kwargs))
        return client

    monkeypatch.setattr(module, "create_anthropic_client", fake_factory)

    adapter = module.AnthropicAdapter(
        model="claude-test",
        client_kwargs={"api_key": "secret"},
    )

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        parse_output=False,
        bus=NullEventBus(),
    )

    assert result.text == "Hello!"
    assert captured_kwargs == [{"api_key": "secret"}]


def test_anthropic_adapter_rejects_mutually_exclusive_client_args() -> None:
    module = cast(Any, _reload_module())

    response = DummyAnthropicMessage([DummyAnthropicTextBlock(text="ok")])
    dummy_client = DummyAnthropicClient([response])
    client = cast(AnthropicProtocol, dummy_client)

    with pytest.raises(ValueError):
        module.AnthropicAdapter(
            model="claude-test", client=client, client_factory=lambda **_: client
        )

    with pytest.raises(ValueError):
        module.AnthropicAdapter(
            model="claude-test", client=client, client_kwargs={"api_key": "secret"}
        )


def test_anthropic_adapter_formats_requests() -> None:
    prompt = Prompt(
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

    response = DummyAnthropicMessage([DummyAnthropicTextBlock(text="Hello there!")])
    dummy_client = DummyAnthropicClient([response])
    client = cast(AnthropicProtocol, dummy_client)

    adapter = AnthropicAdapter(
        model="claude-test",
        client=client,
        max_output_tokens=321,
    )

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Riley"),
        parse_output=False,
        bus=NullEventBus(),
    )

    assert result.text == "Hello there!"
    assert dummy_client.messages.requests == [
        {
            "model": "claude-test",
            "system": "Say hello to ${user}.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Say hello to Riley.",
                        }
                    ],
                }
            ],
            "max_tokens": 321,
        }
    ]


def test_anthropic_adapter_processes_tool_invocation() -> None:
    tool = Tool[ToolParams, ToolPayload](
        name="echo_tool",
        description="Echo the provided query.",
        handler=simple_handler,
    )
    prompt = Prompt(
        ns=PROMPT_NS,
        key="anthropic-tool",
        name="anthropic_tool",
        sections=[
            MarkdownSection[ToolParams](
                title="Instruction",
                key="instruction",
                template="Call the echo_tool with query ${query}.",
                tools=(tool,),
            )
        ],
    )

    tool_request = DummyAnthropicMessage(
        [
            DummyAnthropicToolUseBlock(
                id="call_1",
                name="echo_tool",
                input={"query": "ping"},
            )
        ]
    )
    final_response = DummyAnthropicMessage(
        [DummyAnthropicTextBlock(text="Tool replied with ping")]
    )
    dummy_client = DummyAnthropicClient([tool_request, final_response])
    client = cast(AnthropicProtocol, dummy_client)

    adapter = AnthropicAdapter(model="claude-test", client=client)

    result = adapter.evaluate(
        prompt,
        ToolParams(query="ping"),
        parse_output=False,
        bus=NullEventBus(),
    )

    assert result.text == "Tool replied with ping"
    assert len(result.tool_results) == 1
    tool_result = result.tool_results[0]
    assert tool_result.name == "echo_tool"
    assert tool_result.result.success

    first_request, second_request = dummy_client.messages.requests

    assert first_request["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Call the echo_tool with query ping.",
                }
            ],
        }
    ]
    second_messages = cast(list[dict[str, object]], second_request["messages"])
    assert second_messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": json.dumps(
                    {
                        "message": "ok",
                        "success": True,
                        "payload": {"answer": "ping"},
                    }
                ),
            }
        ],
    }


def test_anthropic_adapter_honors_tool_choice_directive() -> None:
    prompt = Prompt(
        ns=PROMPT_NS,
        key="anthropic-choice",
        name="anthropic_choice",
        sections=[
            MarkdownSection[GreetingParams](
                title="Instruction",
                key="instruction",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicMessage([DummyAnthropicTextBlock(text="Hi!")])
    dummy_client = DummyAnthropicClient([response])
    client = cast(AnthropicProtocol, dummy_client)

    adapter = AnthropicAdapter(
        model="claude-test",
        client=client,
        tool_choice={"type": "function", "function": {"name": "echo_tool"}},
    )

    adapter.evaluate(
        prompt,
        GreetingParams(user="Morgan"),
        parse_output=False,
        bus=NullEventBus(),
    )

    assert dummy_client.messages.requests[0]["tool_choice"] == {
        "type": "tool",
        "name": "echo_tool",
    }


def test_anthropic_adapter_handles_structured_output() -> None:
    prompt = Prompt[StructuredAnswer](
        ns=PROMPT_NS,
        key="anthropic-structured",
        name="anthropic_structured",
        sections=[
            MarkdownSection[GreetingParams](
                title="Instruction",
                key="instruction",
                template="Provide a JSON object with answer for ${user}.",
            )
        ],
    )

    response_payload = json.dumps({"answer": "structured"})
    response = DummyAnthropicMessage([DummyAnthropicTextBlock(text=response_payload)])
    dummy_client = DummyAnthropicClient([response])
    client = cast(AnthropicProtocol, dummy_client)

    adapter = AnthropicAdapter(model="claude-test", client=client)

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Taylor"),
        bus=NullEventBus(),
    )

    assert isinstance(result.output, StructuredAnswer)
    assert result.output.answer == "structured"


def test_anthropic_adapter_rejects_unsupported_tool_choice() -> None:
    prompt = Prompt(
        ns=PROMPT_NS,
        key="anthropic-choice",
        name="anthropic_choice",
        sections=[
            MarkdownSection[GreetingParams](
                title="Instruction",
                key="instruction",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicMessage([DummyAnthropicTextBlock(text="Hi!")])
    dummy_client = DummyAnthropicClient([response])
    client = cast(AnthropicProtocol, dummy_client)

    adapter = AnthropicAdapter(
        model="claude-test",
        client=client,
        tool_choice={"type": "unsupported"},
    )

    with pytest.raises(PromptEvaluationError):
        adapter.evaluate(
            prompt,
            GreetingParams(user="Morgan"),
            parse_output=False,
            bus=NullEventBus(),
        )


def test_anthropic_adapter_publishes_events() -> None:
    prompt = Prompt(
        ns=PROMPT_NS,
        key="anthropic-events",
        name="anthropic_events",
        sections=[
            MarkdownSection[GreetingParams](
                title="Instruction",
                key="instruction",
                template="Say hello to ${user}.",
            )
        ],
    )

    response = DummyAnthropicMessage([DummyAnthropicTextBlock(text="Hi!")])
    dummy_client = DummyAnthropicClient([response])
    client = cast(AnthropicProtocol, dummy_client)

    bus = InProcessEventBus()

    events: list[PromptExecuted] = []
    bus.subscribe(PromptExecuted, events.append)

    adapter = AnthropicAdapter(model="claude-test", client=client)

    adapter.evaluate(
        prompt,
        GreetingParams(user="Kai"),
        parse_output=False,
        bus=bus,
    )

    assert len(events) == 1
    executed = events[0]
    assert executed.prompt_name == "anthropic_events"
    assert executed.adapter == "anthropic"


def test_anthropic_adapter_interacts_with_session() -> None:
    prompt = Prompt(
        ns=PROMPT_NS,
        key="anthropic-session",
        name="anthropic_session",
        sections=[
            MarkdownSection[GreetingParams](
                title="Instruction",
                key="instruction",
                template="Use the session to greet ${user}.",
            )
        ],
    )

    response = DummyAnthropicMessage([DummyAnthropicTextBlock(text="Hi from session!")])
    dummy_client = DummyAnthropicClient([response])
    client = cast(AnthropicProtocol, dummy_client)

    bus = InProcessEventBus()
    session = Session(bus=bus)

    adapter = AnthropicAdapter(model="claude-test", client=client)

    adapter.evaluate(
        prompt,
        GreetingParams(user="Avery"),
        parse_output=False,
        bus=bus,
        session=session,
    )

    session.snapshot()


def test_anthropic_adapter_accepts_mapping_response() -> None:
    prompt = Prompt(
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

    response = MappingAnthropicResponse([{"type": "text", "text": "Hello!"}])
    client = cast(AnthropicProtocol, DummyAnthropicClient([response]))

    adapter = AnthropicAdapter(model="claude-test", client=client)

    adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        parse_output=False,
        bus=NullEventBus(),
    )


def test_anthropic_adapter_handles_simple_response() -> None:
    prompt = Prompt(
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

    response = SimpleAnthropicResponse([{"type": "text", "text": "Hello!"}])
    client = cast(AnthropicProtocol, DummyAnthropicClient([response]))

    adapter = AnthropicAdapter(model="claude-test", client=client)

    adapter.evaluate(
        prompt,
        GreetingParams(user="Sam"),
        parse_output=False,
        bus=NullEventBus(),
    )


def test_strip_markdown_headings_handles_heading_only() -> None:
    module = cast(Any, _reload_module())

    assert module._strip_markdown_headings("## Heading") == ""
    mixed = "Intro\n\n## Heading\nDetails"
    assert module._strip_markdown_headings(mixed) == "Intro\n\nDetails"


@dataclass
class PlaceholderProbe:
    value: str
    metadata: str = field(init=False, default="meta")


def test_placeholder_dataclass_handles_edge_cases() -> None:
    module = cast(Any, _reload_module())

    assert module._placeholder_dataclass("value") == "value"

    probe = PlaceholderProbe(value="test")
    # Fields with init=False trigger dataclasses.replace TypeError.
    assert module._placeholder_dataclass(probe) is probe


def test_wrap_response_validates_content_sequence() -> None:
    module = cast(Any, _reload_module())

    bad_response = types.SimpleNamespace(content="oops")

    with pytest.raises(PromptEvaluationError):
        module._wrap_response(bad_response, "anthropic_bad")


def test_wrap_response_validates_tool_blocks() -> None:
    module = cast(Any, _reload_module())

    response_missing_name = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="tool_use", input={})]
    )

    with pytest.raises(PromptEvaluationError):
        module._wrap_response(response_missing_name, "anthropic_missing_name")

    response_bad_input = types.SimpleNamespace(
        content=[
            types.SimpleNamespace(
                type="tool_use", name="tool", input={"value": object()}, id="call"
            )
        ]
    )

    with pytest.raises(PromptEvaluationError):
        module._wrap_response(response_bad_input, "anthropic_bad_input")


def test_wrap_response_assigns_tool_use_identifiers() -> None:
    module = cast(Any, _reload_module())

    class ToolUseBlock:
        def __init__(self, block_id: object) -> None:
            self.type = "tool_use"
            self.id = block_id
            self.name = "tool"
            self.input = {"value": "ok"}

    content = [
        ToolUseBlock(block_id=None),
        ToolUseBlock(block_id=42),
        types.SimpleNamespace(type="text", text="Done"),
    ]

    response = types.SimpleNamespace(content=content)
    wrapper = module._wrap_response(response, "anthropic_tool_ids")

    choice = wrapper.choices[0]
    tool_calls = list(choice.message.tool_calls or ())
    assert len(tool_calls) == 2
    assert tool_calls[0].id == "tool_use_0"
    assert tool_calls[1].id == "42"


def test_convert_messages_validates_inputs() -> None:
    module = cast(Any, _reload_module())

    with pytest.raises(PromptEvaluationError):
        module._convert_messages(["bad"], "anthropic_convert")

    with pytest.raises(PromptEvaluationError):
        module._convert_messages([{"role": "tool", "content": ""}], "anthropic_convert")

    with pytest.raises(PromptEvaluationError):
        module._convert_messages(
            [{"role": "unknown", "content": ""}], "anthropic_convert"
        )


def test_convert_messages_handles_user_messages() -> None:
    module = cast(Any, _reload_module())

    system = {"role": "system", "content": "## Title\nBody"}
    user = {"role": "user", "content": "Hello"}
    system_payload, messages = module._convert_messages(
        [system, user], "anthropic_convert"
    )

    assert system_payload == "Body"
    assert messages[0]["content"][0]["text"] == "Body"
    assert messages[1]["content"][0]["text"] == "Hello"

    _, user_only = module._convert_messages([user], "anthropic_convert")
    assert user_only[0]["content"][0]["text"] == "Hello"


def test_convert_messages_with_no_system_returns_none() -> None:
    module = cast(Any, _reload_module())

    system_payload, messages = module._convert_messages(
        [{"role": "user", "content": "Hi"}], "anthropic_convert"
    )

    assert system_payload is None
    assert messages[0]["content"][0]["text"] == "Hi"


def test_normalise_system_content_handles_sequences() -> None:
    module = cast(Any, _reload_module())

    content = [
        {"type": "text", "text": "## Title"},
        {"type": "text", "text": "Body"},
    ]
    assert module._normalise_system_content(content) == "Body"
    assert module._normalise_system_content(123) == "123"


def test_normalise_text_content_handles_sequences() -> None:
    module = cast(Any, _reload_module())

    content = [
        {"type": "text", "text": "## Title"},
        {"type": "text", "text": "Body"},
    ]
    blocks = module._normalise_text_content(content)
    assert blocks[0]["text"] == ""
    assert blocks[1]["text"] == "Body"
    assert module._normalise_text_content(None) == [{"type": "text", "text": ""}]


def test_normalise_assistant_content_validates_tool_calls() -> None:
    module = cast(Any, _reload_module())

    with pytest.raises(PromptEvaluationError):
        module._normalise_assistant_content("", ["bad"], "anthropic")

    with pytest.raises(PromptEvaluationError):
        module._normalise_assistant_content(
            "",
            [{"id": "call"}],
            "anthropic",
        )

    with pytest.raises(PromptEvaluationError):
        module._normalise_assistant_content(
            "",
            [{"id": "call", "function": {}}],
            "anthropic",
        )

    with pytest.raises(PromptEvaluationError):
        module._normalise_assistant_content(
            "",
            [{"id": "call", "function": {"name": "tool", "arguments": 1}}],
            "anthropic",
        )

    with pytest.raises(PromptEvaluationError):
        module._normalise_assistant_content(
            "",
            [
                {
                    "id": "call",
                    "function": {"name": "tool", "arguments": "{invalid}"},
                }
            ],
            "anthropic",
        )

    with pytest.raises(PromptEvaluationError):
        module._normalise_assistant_content(
            "",
            [
                {
                    "id": "call",
                    "function": {"name": "tool", "arguments": "[]"},
                }
            ],
            "anthropic",
        )


def test_normalise_assistant_content_generates_tool_use_blocks() -> None:
    module = cast(Any, _reload_module())

    calls = [
        {
            "id": None,
            "function": {"name": "tool", "arguments": json.dumps({"a": 1})},
        },
        {
            "id": 5,
            "function": {"name": "tool", "arguments": json.dumps({"b": 2})},
        },
        {
            "id": "explicit",
            "function": {"name": "tool"},
        },
    ]
    blocks = module._normalise_assistant_content("", calls, "anthropic")
    tool_blocks = [block for block in blocks if block["type"] == "tool_use"]
    assert tool_blocks[0]["id"] == "tool_call_0"
    assert tool_blocks[1]["id"] == "5"
    assert tool_blocks[2]["id"] == "explicit"

    assert module._normalise_assistant_content(None, None, "anthropic") == [
        {"type": "text", "text": ""}
    ]


def test_normalise_content_variants() -> None:
    module = cast(Any, _reload_module())

    assert module._normalise_content(None) == []
    assert module._normalise_content("hi") == [{"type": "text", "text": "hi"}]
    sequence = module._normalise_content([{"type": "text", "text": "hi"}])
    assert sequence[0]["text"] == "hi"
    assert module._normalise_content(123) == [{"type": "text", "text": "123"}]


def test_normalise_content_block_variants() -> None:
    module = cast(Any, _reload_module())

    mapping_block = module._normalise_content_block({"text": "## Heading"})
    assert mapping_block["text"] == ""

    text_block = module._normalise_content_block(
        types.SimpleNamespace(type="text", text="Hello")
    )
    assert text_block == {"type": "text", "text": "Hello"}

    tool_use_block = module._normalise_content_block(
        types.SimpleNamespace(type="tool_use", id="1", name="tool", input={})
    )
    assert tool_use_block["type"] == "tool_use"

    tool_result_block = module._normalise_content_block(
        types.SimpleNamespace(type="tool_result", tool_use_id="1", content="value")
    )
    assert tool_result_block["tool_use_id"] == "1"

    fallback_block = module._normalise_content_block(types.SimpleNamespace(value=1))
    assert fallback_block["type"] == "text"

    custom_block = module._normalise_content_block(
        types.SimpleNamespace(type="custom", text="value", extra="data")
    )
    assert custom_block["text"] == "value"


def test_convert_tools_validates_payloads() -> None:
    module = cast(Any, _reload_module())

    with pytest.raises(PromptEvaluationError):
        module._convert_tools(["bad"], "anthropic")

    with pytest.raises(PromptEvaluationError):
        module._convert_tools([{"type": "unsupported"}], "anthropic")

    with pytest.raises(PromptEvaluationError):
        module._convert_tools([{"type": "function"}], "anthropic")

    with pytest.raises(PromptEvaluationError):
        module._convert_tools([{"type": "function", "function": {}}], "anthropic")

    with pytest.raises(PromptEvaluationError):
        module._convert_tools(
            [
                {
                    "type": "function",
                    "function": {"description": "d", "parameters": {}},
                }
            ],
            "anthropic",
        )

    with pytest.raises(PromptEvaluationError):
        module._convert_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "n", "parameters": {}},
                }
            ],
            "anthropic",
        )

    with pytest.raises(PromptEvaluationError):
        module._convert_tools(
            [
                {
                    "type": "function",
                    "function": {"name": "n", "description": "d", "parameters": []},
                }
            ],
            "anthropic",
        )

    converted = module._convert_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "tool",
                    "description": "desc",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "anthropic",
    )
    assert converted[0]["name"] == "tool"


def test_convert_tool_choice_variants() -> None:
    module = cast(Any, _reload_module())

    assert module._convert_tool_choice(None, "anthropic") is None
    assert module._convert_tool_choice("auto", "anthropic") == {"type": "auto"}

    payload = module._convert_tool_choice(
        {
            "type": "function",
            "function": {"name": "tool", "disable_parallel_tool_use": True},
        },
        "anthropic",
    )
    assert payload == {
        "type": "tool",
        "name": "tool",
        "disable_parallel_tool_use": True,
    }

    mapping_choice = module._convert_tool_choice(
        {"type": "tool", "name": "tool"},
        "anthropic",
    )
    assert mapping_choice["type"] == "tool"

    with pytest.raises(PromptEvaluationError):
        module._convert_tool_choice(
            {"type": "function", "function": "invalid"},
            "anthropic",
        )
