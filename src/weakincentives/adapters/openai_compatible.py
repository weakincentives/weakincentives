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

"""Adapter for OpenAI-shaped chat-completion clients.

The adapter does not import the ``openai`` package directly; instead it
talks to a small structural protocol (:class:`ChatClient`) that the real
``openai.OpenAI`` instance, the ``Mistral``/``Together``/``Fireworks``
clients, and ad-hoc mocks all satisfy. Pip extras provide the actual
SDKs as needed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, final, runtime_checkable

from weakincentives.core import (
    Deadline,
    PromptResponse,
    RenderedPrompt,
    Tool,
    ToolCall,
    Usage,
)

__all__ = [
    "ChatClient",
    "ChatCompletion",
    "ChatCompletions",
    "ChatToolCall",
    "ChatToolFunction",
    "ChatUsage",
    "OpenAICompatibleAdapter",
]


# =============================================================================
# Structural client protocols
# =============================================================================


@runtime_checkable
class ChatToolFunction(Protocol):
    """Function payload of a model-emitted tool call (OpenAI shape)."""

    name: str
    arguments: str


@runtime_checkable
class ChatToolCall(Protocol):
    """Single tool call entry on a chat message."""

    id: str
    function: ChatToolFunction


@runtime_checkable
class ChatMessage(Protocol):
    """Assistant message returned by a chat completion."""

    content: str | None
    tool_calls: Sequence[ChatToolCall] | None


@runtime_checkable
class ChatChoice(Protocol):
    """One choice from a chat completion response."""

    message: ChatMessage
    finish_reason: str | None


@runtime_checkable
class ChatUsage(Protocol):
    """Token accounting on a chat completion response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@runtime_checkable
class ChatCompletion(Protocol):
    """Top-level chat completion response."""

    choices: Sequence[ChatChoice]
    usage: ChatUsage | None


@runtime_checkable
class ChatCompletions(Protocol):
    """``client.chat.completions`` namespace."""

    def create(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None = None,
        timeout: float | None = None,
    ) -> ChatCompletion: ...


@runtime_checkable
class ChatNamespace(Protocol):
    """``client.chat`` namespace."""

    completions: ChatCompletions


@runtime_checkable
class ChatClient(Protocol):
    """Top-level OpenAI-style client."""

    chat: ChatNamespace


# =============================================================================
# Adapter
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class _AdapterConfig:
    model: str
    system_role: str = "system"
    user_role: str = "user"
    extra_messages: tuple[Mapping[str, object], ...] = ()


@final
class OpenAICompatibleAdapter:
    """``ProviderAdapter`` for any OpenAI-shaped chat-completion client.

    Parameters:
        client: object satisfying :class:`ChatClient` (real SDK, mock, …).
        model: chat model identifier passed to ``client.chat.completions.create``.
        system_role / user_role: role names for the rendered prompt and
            optional caller-supplied messages. Defaults match OpenAI.
        extra_messages: additional messages prepended after the rendered
            prompt; useful for in-context demonstrations.
    """

    def __init__(
        self,
        *,
        client: ChatClient,
        model: str,
        system_role: str = "system",
        user_role: str = "user",
        extra_messages: Sequence[Mapping[str, object]] = (),
    ) -> None:
        super().__init__()
        del user_role  # reserved for future role-routing customizations
        self._client = client
        self._config = _AdapterConfig(
            model=model,
            system_role=system_role,
            extra_messages=tuple(extra_messages),
        )

    def evaluate(
        self,
        rendered: object,
        session: object,
        *,
        deadline: Deadline | None = None,
    ) -> PromptResponse:
        del session
        if not isinstance(rendered, RenderedPrompt):
            raise TypeError(
                f"OpenAICompatibleAdapter expected RenderedPrompt, "
                f"got {type(rendered).__name__}"
            )
        messages = self._build_messages(rendered)
        tools = _tools_payload(rendered.tools)
        timeout = _deadline_to_timeout(deadline)
        completion = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            tools=tools if tools else None,
            timeout=timeout,
        )
        return _completion_to_response(completion)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(self, rendered: RenderedPrompt) -> list[Mapping[str, object]]:
        messages: list[Mapping[str, object]] = [
            {"role": self._config.system_role, "content": rendered.text}
        ]
        messages.extend(self._config.extra_messages)
        return messages


# =============================================================================
# Conversion helpers
# =============================================================================


def _tools_payload(
    tools: Sequence[Tool[Any, Any]],
) -> list[Mapping[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def _deadline_to_timeout(deadline: Deadline | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline.remaining().total_seconds()
    return remaining if remaining > 0 else 0.0


def _completion_to_response(completion: ChatCompletion) -> PromptResponse:
    if not completion.choices:
        return PromptResponse(text="", finish_reason="stop")
    choice = completion.choices[0]
    message = choice.message
    text = message.content or ""
    tool_calls = _convert_tool_calls(message.tool_calls)
    usage = _convert_usage(completion.usage)
    return PromptResponse(
        text=text,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=choice.finish_reason,
    )


def _convert_tool_calls(
    raw: Sequence[ChatToolCall] | None,
) -> tuple[ToolCall, ...]:
    if not raw:
        return ()
    converted: list[ToolCall] = []
    for call in raw:
        try:
            arguments = json.loads(call.function.arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {"_raw": call.function.arguments}
        converted.append(
            ToolCall(
                name=call.function.name,
                arguments=cast("Mapping[str, object]", arguments)
                if isinstance(arguments, dict)
                else {"_value": arguments},
                call_id=call.id,
            )
        )
    return tuple(converted)


def _convert_usage(usage: ChatUsage | None) -> Usage:
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )
