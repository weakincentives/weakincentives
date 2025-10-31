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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from weakincentives.prompts import ToolResult


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
        content: str | None,
        tool_calls: Sequence[DummyToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = list(tool_calls) if tool_calls is not None else None

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": self.content}
        if self.tool_calls is not None:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
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


class MappingResponse(dict):
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


def simple_handler(params: ToolParams) -> ToolResult[ToolPayload]:
    return ToolResult(message="ok", payload=ToolPayload(answer=params.query))


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
