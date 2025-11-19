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
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.helpers.adapters import TEST_ADAPTER_NAME
from tests.helpers.events import NullEventBus
from weakincentives.adapters import PromptEvaluationError, shared
from weakincentives.adapters.core import (
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.overrides import PromptOverridesStore
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.runtime.events import EventBus
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
    rendered = RenderedPrompt(text="system")
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
            deadline: Deadline | None = None,
            overrides_store: PromptOverridesStore | None = None,
            overrides_tag: str = "latest",
        ) -> PromptResponse[object]:
            raise NotImplementedError

    adapter = DummyAdapter()
    prompt = Prompt(ns="tests", key="example")
    session = Session(bus=bus)

    with pytest.raises(PromptEvaluationError):
        shared.run_conversation(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=adapter,
            prompt=prompt,
            prompt_name="example",
            rendered=rendered,
            render_inputs=(),
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


def test_run_conversation_retries_on_throttle() -> None:
    rendered = RenderedPrompt(text="system")
    bus = NullEventBus()
    throttle_policy = shared.ThrottlePolicy(
        base_delay_seconds=0.01,
        max_delay_seconds=0.02,
        max_attempts=3,
        max_total_seconds=1.0,
    )
    sleep_calls: list[float] = []

    class DummyChoice:
        def __init__(self) -> None:
            self.message = SimpleNamespace(content="ok", tool_calls=[])

    class DummyResponse:
        def __init__(self) -> None:
            self.choices = [DummyChoice()]

    attempts = 0

    def call_provider(
        messages: list[dict[str, Any]],
        tool_specs: list[Mapping[str, Any]],
        tool_choice: shared.ToolChoice | None,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise shared.ThrottleError(
                kind="rate_limit",
                retry_after=0.01,
                provider_payload={"status": 429},
                message="rate limited",
            )
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
            deadline: Deadline | None = None,
            overrides_store: PromptOverridesStore | None = None,
            overrides_tag: str = "latest",
        ) -> PromptResponse[object]:
            raise NotImplementedError

    adapter = DummyAdapter()
    prompt = Prompt(ns="tests", key="example")
    session = Session(bus=bus)

    result = shared.run_conversation(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=adapter,
        prompt=prompt,
        prompt_name="example",
        rendered=rendered,
        render_inputs=(),
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
        throttle_policy=throttle_policy,
        sleep_fn=sleep_calls.append,
    )

    assert attempts == 2
    assert sleep_calls and sleep_calls[0] <= throttle_policy.max_delay_seconds
    assert result.text == "ok"


def test_run_conversation_honors_throttle_budget() -> None:
    rendered = RenderedPrompt(text="system")
    bus = NullEventBus()
    throttle_policy = shared.ThrottlePolicy(
        base_delay_seconds=0.01,
        max_delay_seconds=0.01,
        max_attempts=1,
        max_total_seconds=0.01,
    )

    class DummyChoice:
        def __init__(self) -> None:
            self.message = SimpleNamespace(content="ok", tool_calls=[])

    class DummyResponse:
        def __init__(self) -> None:
            self.choices = [DummyChoice()]

    def call_provider(
        messages: list[dict[str, Any]],
        tool_specs: list[Mapping[str, Any]],
        tool_choice: shared.ToolChoice | None,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        raise shared.ThrottleError(
            kind="rate_limit",
            retry_after=0.0,
            provider_payload=None,
            message="rate limited",
        )

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
            deadline: Deadline | None = None,
            overrides_store: PromptOverridesStore | None = None,
            overrides_tag: str = "latest",
        ) -> PromptResponse[object]:
            raise NotImplementedError

    adapter = DummyAdapter()
    prompt = Prompt(ns="tests", key="example")
    session = Session(bus=bus)

    with pytest.raises(PromptEvaluationError) as err:
        shared.run_conversation(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=adapter,
            prompt=prompt,
            prompt_name="example",
            rendered=rendered,
            render_inputs=(),
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
            throttle_policy=throttle_policy,
            sleep_fn=lambda _: None,
        )

    assert "throttled" in str(err.value)
    payload = getattr(err.value, "provider_payload", {})
    assert payload.get("throttle", {}).get("kind") == "rate_limit"
