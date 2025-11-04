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

from weakincentives.prompt import ToolResult


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


class DummyGeminiFunctionCall:
    def __init__(
        self,
        name: str,
        args: Mapping[str, object] | None = None,
        call_id: str | None = None,
    ) -> None:
        self.name = name
        self.args = args or {}
        self.id = call_id


class DummyGeminiPart:
    def __init__(
        self,
        *,
        text: str | None = None,
        function_call: DummyGeminiFunctionCall | None = None,
    ) -> None:
        self.text = text
        self.function_call = function_call

    def model_dump(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.text is not None:
            payload["text"] = self.text
        if self.function_call is not None:
            function_payload: dict[str, object] = {
                "name": self.function_call.name,
                "args": dict(self.function_call.args),
            }
            if self.function_call.id is not None:
                function_payload["id"] = self.function_call.id
            payload["functionCall"] = function_payload
        return payload


@dataclass(slots=True)
class DummyGeminiContent:
    parts: Sequence[DummyGeminiPart]


@dataclass(slots=True)
class DummyGeminiCandidate:
    content: DummyGeminiContent


class DummyGeminiResponse:
    def __init__(
        self,
        candidates: Sequence[DummyGeminiCandidate],
        *,
        parsed: object | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        self.candidates = list(candidates)
        self.parsed = parsed
        self._payload = dict(payload or {})

    def model_dump(self) -> dict[str, object]:
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [part.model_dump() for part in candidate.content.parts]
                    }
                }
                for candidate in self.candidates
            ],
            **self._payload,
        }


class DummyGeminiModels:
    def __init__(self, responses: Sequence[DummyGeminiResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> DummyGeminiResponse:
        self.requests.append(dict(kwargs))
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)


class DummyGeminiClient:
    def __init__(self, responses: Sequence[DummyGeminiResponse]) -> None:
        self.models = DummyGeminiModels(responses)


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
