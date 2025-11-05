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


def _maybe_model_dump(value: object) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: getattr(value, key) for key in vars(value) if not key.startswith("_")
        }
    raise TypeError("Unsupported value for model_dump stub")


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


def _function_call_from_dummy(
    tool_call: DummyToolCall,
) -> DummyResponseFunctionToolCall:
    return DummyResponseFunctionToolCall(
        call_id=tool_call.id,
        tool_id=tool_call.id,
        name=tool_call.function.name,
        arguments=tool_call.function.arguments,
    )


class DummyResponseOutputText:
    def __init__(self, text: str, *, parsed: object | None = None) -> None:
        self.type = "output_text"
        self.text = text
        self.annotations: list[object] = []
        self.parsed = parsed
        self.logprobs: list[object] | None = None

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "text": self.text,
            "annotations": [],
        }
        if self.parsed is not None:
            payload["parsed"] = self.parsed
        return payload


class DummyResponseOutputMessage:
    def __init__(
        self,
        content: Sequence[object],
        *,
        role: str = "assistant",
        status: str = "completed",
        message_id: str = "msg_1",
    ) -> None:
        self.type = "message"
        self.id = message_id
        self.role = role
        self.status = status
        self.content = list(content)

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "status": self.status,
            "type": self.type,
            "content": [
                _maybe_model_dump(item) if not isinstance(item, dict) else item
                for item in self.content
            ],
        }


class DummyResponseFunctionToolCall:
    def __init__(
        self,
        *,
        call_id: str,
        name: str,
        arguments: str | None,
        tool_id: str | None = None,
        status: str = "completed",
    ) -> None:
        self.type = "function_call"
        self.call_id = call_id
        self.id = tool_id
        self.name = name
        self.arguments = arguments or "{}"
        self.status = status

    def model_dump(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "call_id": self.call_id,
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
        }


ResponseOutputItem = DummyResponseOutputMessage | DummyResponseFunctionToolCall


class DummyResponse:
    def __init__(self, output: Sequence[ResponseOutputItem | DummyChoice]) -> None:
        if output and isinstance(output[0], DummyChoice):
            choice_items = cast(Sequence[DummyChoice], output)
            collected: list[ResponseOutputItem] = []
            for choice in choice_items:
                collected.extend(choice.output_items)
            self.output = collected
            self.choices = list(choice_items)
        else:
            self.output = [cast(ResponseOutputItem, item) for item in output]
            self.choices: list[DummyChoice] = []
        self.model = "dummy"
        self.id = "resp_1"
        self.object = "response"

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "output": [item.model_dump() for item in self.output],
        }


class DummyChoice:
    def __init__(self, message: DummyMessage) -> None:
        self.message = message
        self.output_items = message.to_output_items()

    def model_dump(self) -> dict[str, Any]:
        return {"message": self.message.model_dump()}


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

    def to_output_items(self) -> list[ResponseOutputItem]:
        if isinstance(self.content, Sequence) and not isinstance(
            self.content, (str, bytes, bytearray)
        ):
            content_items = list(self.content)
        else:
            text = "" if self.content is None else str(self.content)
            content_items = [DummyResponseOutputText(text, parsed=self.parsed)]
        items: list[ResponseOutputItem] = [DummyResponseOutputMessage(content_items)]
        if self.tool_calls:
            items.extend(_function_call_from_dummy(call) for call in self.tool_calls)
        return items

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": self.content}
        if self.tool_calls is not None:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.parsed is not None:
            payload["parsed"] = self.parsed
        return payload


class MappingResponse(dict[str, object]):
    def __init__(self, output: Sequence[ResponseOutputItem | DummyChoice]) -> None:
        super().__init__({"meta": "value"})
        if output and isinstance(output[0], DummyChoice):
            choice_items = cast(Sequence[DummyChoice], output)
            self.choices = list(choice_items)
            collected: list[ResponseOutputItem] = []
            for choice in choice_items:
                collected.extend(choice.output_items)
            self.output = collected
        else:
            self.output = [cast(ResponseOutputItem, item) for item in output]
            self.choices: list[DummyChoice] = []


class WeirdResponse:
    def __init__(self, output: Sequence[ResponseOutputItem | DummyChoice]) -> None:
        if output and isinstance(output[0], DummyChoice):
            choice_items = cast(Sequence[DummyChoice], output)
            self.choices = list(choice_items)
            collected: list[ResponseOutputItem] = []
            for choice in choice_items:
                collected.extend(choice.output_items)
            self.output = collected
        else:
            self.output = [cast(ResponseOutputItem, item) for item in output]
            self.choices: list[DummyChoice] = []

    def model_dump(self) -> list[object]:
        return ["unexpected"]


class SimpleResponse:
    def __init__(self, output: Sequence[ResponseOutputItem | DummyChoice]) -> None:
        if output and isinstance(output[0], DummyChoice):
            choice_items = cast(Sequence[DummyChoice], output)
            self.choices = list(choice_items)
            collected: list[ResponseOutputItem] = []
            for choice in choice_items:
                collected.extend(choice.output_items)
            self.output = collected
        else:
            self.output = [cast(ResponseOutputItem, item) for item in output]
            self.choices: list[DummyChoice] = []


ResponseType = DummyResponse | MappingResponse | WeirdResponse | SimpleResponse


class DummyResponsesAPI:
    def __init__(self, responses: Sequence[ResponseType]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def _record(self, payload: dict[str, object]) -> ResponseType:
        self.requests.append(payload)
        if not self._responses:
            raise AssertionError("No responses available")
        return self._responses.pop(0)

    def create(self, **kwargs: object) -> ResponseType:
        return self._record(dict(kwargs))

    def parse(self, **kwargs: object) -> ResponseType:
        payload = dict(kwargs)
        payload["__parse__"] = True
        return self._record(payload)


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
