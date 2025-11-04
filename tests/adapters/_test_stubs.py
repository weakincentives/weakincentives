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

"""Shared adapter test stubs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from weakincentives.prompt import ToolContext, ToolResult


@dataclass
class GreetingParams:
    user: str


@dataclass(slots=True)
class DummyFunctionCall:
    name: str
    arguments: str | None


class DummyToolCall:
    def __init__(self, call_id: str, name: str, arguments: str | None) -> None:
        self.id = call_id
        self.function = DummyFunctionCall(name=name, arguments=arguments)

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class DummyMessage:
    def __init__(
        self,
        *,
        content: str | Sequence[object] | None,
        tool_calls: Sequence[DummyToolCall] | None = None,
        parsed: object | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tuple(tool_calls) if tool_calls is not None else None
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


class DummyChoice:
    def __init__(self, message: DummyMessage) -> None:
        self.message = message

    def model_dump(self) -> dict[str, Any]:
        return {"message": self.message.model_dump()}


class DummyResponse:
    def __init__(self, choices: Sequence[DummyChoice]) -> None:
        self.choices = list(choices)

    def model_dump(self) -> dict[str, Any]:
        return {"choices": [choice.model_dump() for choice in self.choices]}


class MappingResponse(dict[str, object]):
    def __init__(self, choices: Sequence[DummyChoice]) -> None:
        super().__init__({"meta": "value"})
        self.choices = list(choices)


class WeirdResponse:
    def __init__(self, choices: Sequence[DummyChoice]) -> None:
        self.choices = list(choices)

    def model_dump(self) -> list[object]:
        return ["unexpected"]


class SimpleResponse:
    def __init__(self, choices: Sequence[DummyChoice]) -> None:
        self.choices = list(choices)


ResponseType = DummyResponse | MappingResponse | WeirdResponse | SimpleResponse


class DummyCompletionsAPI:
    def __init__(self, responses: Sequence[ResponseType]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> ResponseType:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


@dataclass(slots=True)
class DummyChatAPI:
    completions: DummyCompletionsAPI


class DummyOpenAIClient:
    def __init__(self, responses: Sequence[ResponseType]) -> None:
        completions = DummyCompletionsAPI(responses)
        self.chat = DummyChatAPI(completions)
        self.completions = completions


@dataclass
class ToolParams:
    query: str


@dataclass
class ToolPayload:
    answer: str


@dataclass
class StructuredAnswer:
    answer: str


@dataclass
class OptionalParams:
    query: str = "default"


@dataclass
class OptionalPayload:
    value: str


def simple_handler(
    params: ToolParams, *, context: ToolContext
) -> ToolResult[ToolPayload]:
    del context
    return ToolResult(message="ok", value=ToolPayload(answer=params.query))


class RecordingCompletion:
    """Test double that mimics litellm.completion."""

    def __init__(self, responses: Sequence[ResponseType]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> ResponseType:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


@dataclass(slots=True)
class DummyAnthropicTextBlock:
    text: str
    type: str = "text"


@dataclass(slots=True)
class DummyAnthropicToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass(slots=True)
class DummyAnthropicMessage:
    content: list[object]

    def model_dump(self) -> dict[str, Any]:
        return {"content": [dict(_normalise_block(part)) for part in self.content]}


def _normalise_block(part: object) -> dict[str, Any]:
    if isinstance(part, Mapping):
        return {str(key): value for key, value in part.items()}
    block = {"type": getattr(part, "type", "text")}
    for attribute in ("text", "id", "name", "input", "tool_use_id", "content"):
        if hasattr(part, attribute):
            block[attribute] = getattr(part, attribute)
    return block


class DummyAnthropicMessagesAPI:
    def __init__(self, responses: Sequence[object]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


class DummyAnthropicClient:
    def __init__(self, responses: Sequence[object]) -> None:
        self.messages = DummyAnthropicMessagesAPI(responses)
