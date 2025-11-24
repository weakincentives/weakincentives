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
from typing import Any, cast

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


@dataclass
class DummyContent:
    type: str = "output_text"
    text: str | None = None
    tool_calls: Sequence[DummyToolCall] | None = None
    json: object | None = None
    parsed: object | None = None

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            payload["text"] = self.text
        if self.tool_calls is not None:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.json is not None:
            payload["json"] = self.json
        if self.parsed is not None:
            payload["parsed"] = self.parsed
        return payload


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

    def to_content_parts(self) -> list[object]:
        if isinstance(self.content, Sequence) and not isinstance(
            self.content, (str, bytes, bytearray)
        ):
            return list(self.content)
        return [
            DummyContent(
                text=self.content if isinstance(self.content, str) else None,
                tool_calls=self.tool_calls,
                parsed=self.parsed,
            )
        ]

    def model_dump(self) -> dict[str, Any]:  # pragma: no cover - compatibility
        normalized: list[object] = []
        for part in self.to_content_parts():
            model_dump_fn = getattr(part, "model_dump", None)
            if callable(model_dump_fn):
                normalized.append(cast(Any, model_dump_fn)())
            else:
                normalized.append(part)
        return {"content": normalized}


class DummyChoice:
    def __init__(self, message: DummyMessage) -> None:
        self.message = message

    def model_dump(self) -> dict[str, Any]:
        return {"message": self.message.model_dump()}


@dataclass
class DummyResponseOutput:
    content: list[object]

    def model_dump(self) -> dict[str, Any]:
        normalized: list[object] = []
        for part in self.content:
            model_dump_fn = getattr(part, "model_dump", None)
            if callable(model_dump_fn):
                normalized.append(cast(Any, model_dump_fn)())
            else:
                normalized.append(part)
        return {"content": normalized}


class DummyResponse:
    def __init__(self, choices: Sequence[DummyChoice]) -> None:
        self.choices = list(choices)
        self.output = [
            DummyResponseOutput(choice.message.to_content_parts()) for choice in choices
        ]

    def model_dump(self) -> dict[str, Any]:
        return {"output": [output.model_dump() for output in self.output]}


class MappingResponse(dict[str, object]):
    def __init__(self, choices: Sequence[DummyChoice]) -> None:
        super().__init__({"meta": "value"})
        self.choices = list(choices)
        self.output = [
            DummyResponseOutput(choice.message.to_content_parts()) for choice in choices
        ]


class WeirdResponse:
    def __init__(self, choices: Sequence[DummyChoice]) -> None:
        self.choices = list(choices)
        self.output = [
            DummyResponseOutput(choice.message.to_content_parts()) for choice in choices
        ]

    def model_dump(self) -> list[object]:
        return ["unexpected"]


class SimpleResponse:
    def __init__(self, choices: Sequence[DummyChoice]) -> None:
        self.choices = list(choices)
        self.output = [
            DummyResponseOutput(choice.message.to_content_parts()) for choice in choices
        ]


ResponseType = DummyResponse | MappingResponse | WeirdResponse | SimpleResponse


class DummyResponsesAPI:
    def __init__(self, responses: Sequence[ResponseType]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> ResponseType:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


class DummyOpenAIClient:
    def __init__(self, responses: Sequence[ResponseType]) -> None:
        self.responses = DummyResponsesAPI(responses)


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
