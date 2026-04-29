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

"""Tests for ``weakincentives.adapters.openai_compatible``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

import pytest

from weakincentives.adapters import ChatClient, OpenAICompatibleAdapter
from weakincentives.clock import Deadline, FakeClock
from weakincentives.core import (
    MarkdownSection,
    Prompt,
    Session,
    Tool,
    ToolContext,
    ToolResult,
)


def _noop(params: object, context: ToolContext) -> ToolResult[None]:
    del params, context
    return ToolResult.ok(None)


def _make_prompt() -> Prompt:
    section = MarkdownSection[None](
        title="System",
        key="system",
        template="You are helpful.",
        tools=(Tool(name="echo", description="echo back", handler=_noop),),
    )
    return Prompt(ns="ns", key="k", sections=(section,))


# ---------------------------------------------------------------------------
# Fakes mirroring the OpenAI shape
# ---------------------------------------------------------------------------


@dataclass
class _ToolFunction:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _ToolFunction


@dataclass
class _Message:
    content: str | None
    tool_calls: list[_ToolCall] | None = None


@dataclass
class _Choice:
    message: _Message
    finish_reason: str | None = "stop"


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _Completion:
    choices: list[_Choice]
    usage: _Usage | None = None


@dataclass
class _Completions:
    response: _Completion
    captured: list[Mapping[str, object]] = field(default_factory=lambda: [])

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None = None,
        timeout: float | None = None,
    ) -> _Completion:
        self.captured.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": list(tools) if tools is not None else None,
                "timeout": timeout,
            }
        )
        return self.response


@dataclass
class _Chat:
    completions: _Completions


@dataclass
class _Client:
    chat: _Chat


def _client_with(completion: _Completion) -> ChatClient:
    return cast(
        "ChatClient",
        _Client(chat=_Chat(completions=_Completions(response=completion))),
    )


def _build_capturing_client(
    completion: _Completion,
) -> tuple[ChatClient, _Completions]:
    completions = _Completions(response=completion)
    client = cast("ChatClient", _Client(chat=_Chat(completions=completions)))
    return client, completions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evaluate_returns_text_only_response() -> None:
    completion = _Completion(
        choices=[_Choice(message=_Message(content="hello"))],
        usage=_Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )
    client = _client_with(completion)
    adapter = OpenAICompatibleAdapter(client=client, model="gpt-x")
    response = adapter.evaluate(_make_prompt().render(), Session())
    assert response.text == "hello"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 6
    assert response.tool_calls == ()


def test_evaluate_translates_tool_calls() -> None:
    completion = _Completion(
        choices=[
            _Choice(
                message=_Message(
                    content=None,
                    tool_calls=[
                        _ToolCall(
                            id="call_1",
                            function=_ToolFunction(
                                name="echo",
                                arguments='{"text": "hi"}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    client = _client_with(completion)
    adapter = OpenAICompatibleAdapter(client=client, model="gpt-x")
    response = adapter.evaluate(_make_prompt().render(), Session())
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "echo"
    assert call.arguments == {"text": "hi"}
    assert call.call_id == "call_1"


def test_evaluate_handles_invalid_tool_arguments_json() -> None:
    completion = _Completion(
        choices=[
            _Choice(
                message=_Message(
                    content=None,
                    tool_calls=[
                        _ToolCall(
                            id="c",
                            function=_ToolFunction(
                                name="echo",
                                arguments="not json",
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )
    adapter = OpenAICompatibleAdapter(client=_client_with(completion), model="gpt-x")
    response = adapter.evaluate(_make_prompt().render(), Session())
    assert response.tool_calls[0].arguments == {"_raw": "not json"}


def test_evaluate_handles_non_dict_tool_arguments_after_parse() -> None:
    completion = _Completion(
        choices=[
            _Choice(
                message=_Message(
                    content=None,
                    tool_calls=[
                        _ToolCall(
                            id="c",
                            function=_ToolFunction(
                                name="echo",
                                arguments='["just", "an", "array"]',
                            ),
                        )
                    ],
                )
            )
        ]
    )
    adapter = OpenAICompatibleAdapter(client=_client_with(completion), model="gpt-x")
    response = adapter.evaluate(_make_prompt().render(), Session())
    assert response.tool_calls[0].arguments == {"_value": ["just", "an", "array"]}


def test_evaluate_passes_tool_payload_and_messages() -> None:
    completion = _Completion(choices=[_Choice(message=_Message(content="ok"))])
    client, completions = _build_capturing_client(completion)
    adapter = OpenAICompatibleAdapter(
        client=client,
        model="gpt-x",
        extra_messages=({"role": "user", "content": "ping"},),
    )
    adapter.evaluate(_make_prompt().render(), Session())
    captured = completions.captured[0]
    assert captured["model"] == "gpt-x"
    messages = cast("list[dict[str, object]]", captured["messages"])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == ("## 1. System\n\nYou are helpful.")
    assert messages[1] == {"role": "user", "content": "ping"}
    tools = cast("list[dict[str, object]] | None", captured["tools"])
    assert tools is not None
    function = cast("dict[str, object]", tools[0]["function"])
    assert function["name"] == "echo"


def test_evaluate_omits_tools_when_prompt_has_none() -> None:
    section = MarkdownSection[None](title="S", key="s", template="hi")
    prompt = Prompt(ns="ns", key="k", sections=(section,))
    completion = _Completion(choices=[_Choice(message=_Message(content="ok"))])
    client, completions = _build_capturing_client(completion)
    OpenAICompatibleAdapter(client=client, model="gpt-x").evaluate(
        prompt.render(), Session()
    )
    assert completions.captured[0]["tools"] is None


def test_evaluate_returns_empty_response_when_no_choices() -> None:
    completion = _Completion(choices=[])
    adapter = OpenAICompatibleAdapter(client=_client_with(completion), model="gpt-x")
    response = adapter.evaluate(_make_prompt().render(), Session())
    assert response.text == ""
    assert response.finish_reason == "stop"


def test_evaluate_propagates_deadline_as_timeout() -> None:
    clock = FakeClock()
    deadline = Deadline.create(
        expires_at=clock.utcnow() + timedelta(seconds=10),
        clock=clock,
    )
    completion = _Completion(choices=[_Choice(message=_Message(content="ok"))])
    client, completions = _build_capturing_client(completion)
    OpenAICompatibleAdapter(client=client, model="gpt-x").evaluate(
        _make_prompt().render(), Session(), deadline=deadline
    )
    assert completions.captured[0]["timeout"] == 10.0


def test_evaluate_clamps_negative_timeout_to_zero() -> None:
    clock = FakeClock()
    deadline = Deadline.create(
        expires_at=clock.utcnow() + timedelta(seconds=5),
        clock=clock,
    )
    clock.advance(10)  # deadline now in the past
    completion = _Completion(choices=[_Choice(message=_Message(content="ok"))])
    client, completions = _build_capturing_client(completion)
    OpenAICompatibleAdapter(client=client, model="gpt-x").evaluate(
        _make_prompt().render(), Session(), deadline=deadline
    )
    assert completions.captured[0]["timeout"] == 0.0


def test_evaluate_rejects_non_rendered_input() -> None:
    completion = _Completion(choices=[_Choice(message=_Message(content="ok"))])
    adapter = OpenAICompatibleAdapter(client=_client_with(completion), model="gpt-x")
    with pytest.raises(TypeError, match="RenderedPrompt"):
        adapter.evaluate("not rendered", Session())
