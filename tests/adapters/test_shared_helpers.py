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

from collections.abc import Mapping
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from weakincentives.adapters import PromptEvaluationError, shared
from weakincentives.adapters.core import (
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.prompt import Prompt
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.runtime.events import EventBus, NullEventBus
from weakincentives.runtime.session import Session


def test_first_choice_returns_first_item() -> None:
    response = SimpleNamespace(choices=["first", "second"])

    assert shared.first_choice(response, prompt_name="example") == "first"


def test_first_choice_requires_sequence() -> None:
    response = SimpleNamespace(choices=None)

    with pytest.raises(PromptEvaluationError):
        shared.first_choice(response, prompt_name="example")


def test_parse_tool_arguments_rejects_non_string_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_loads(_: str) -> Mapping[Any, Any]:
        # Simulate a mapping that does not use string keys to exercise defensive branch.
        return {1: "value"}

    monkeypatch.setattr(shared.json, "loads", fake_loads)

    with pytest.raises(PromptEvaluationError) as err:
        shared.parse_tool_arguments(
            "{}",
            prompt_name="example",
            provider_payload=None,
        )

    message = str(err.value)
    assert "string keys" in message


def test_mapping_to_str_dict_rejects_non_string_keys() -> None:
    assert shared._mapping_to_str_dict({1: "value"}) is None


def test_run_conversation_requires_message_payload() -> None:
    rendered = RenderedPrompt(
        text="system",
        output_type=None,
        container=None,
        allow_extra_keys=None,
        deadline=None,
    )
    bus = NullEventBus()

    class DummyChoice:
        def __init__(self) -> None:
            self.message = None

    class DummyResponse:
        def __init__(self) -> None:
            self.choices = [DummyChoice()]

    def call_provider(
        messages: list[dict[str, Any]],
        tool_specs: list[Mapping[str, Any]],
        tool_choice: shared.ToolChoice | None,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        return DummyResponse()

    def select_choice(response: DummyResponse) -> shared.ProviderChoice:
        return response.choices[0]

    serialize_stub = cast(
        shared.ToolMessageSerializer,
        lambda _result, *, payload=None: "",
    )

    class DummyAdapter(ProviderAdapter[object]):
        def evaluate(
            self,
            prompt: Prompt[object],
            *params: SupportsDataclass,
            parse_output: bool = True,
            bus: EventBus,
            session: SessionProtocol,
            deadline: datetime | None = None,
        ) -> PromptResponse[object]:
            raise NotImplementedError

    adapter = DummyAdapter()
    prompt = Prompt(ns="tests", key="example")
    session = Session(bus=bus)

    with pytest.raises(PromptEvaluationError):
        shared.run_conversation(
            adapter_name="test",
            adapter=adapter,
            prompt=prompt,
            prompt_name="example",
            rendered=rendered,
            initial_messages=[{"role": "system", "content": rendered.text}],
            parse_output=False,
            bus=bus,
            session=session,
            tool_choice="auto",
            response_format=None,
            require_structured_output_text=False,
            call_provider=call_provider,
            select_choice=select_choice,
            serialize_tool_message_fn=serialize_stub,
        )
