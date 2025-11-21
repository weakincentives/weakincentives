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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
from weakincentives.prompt._types import SupportsDataclass, SupportsToolResult
from weakincentives.prompt.overrides import PromptOverridesStore
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.structured_output import (
    ARRAY_WRAPPER_KEY,
    StructuredOutputConfig,
)
from weakincentives.prompt.tool import Tool
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import EventBus, HandlerFailure
from weakincentives.runtime.session import Session


@dataclass(slots=True)
class _ExampleOutput:
    value: str


@dataclass(slots=True)
class _ToolParams:
    value: str


class _StubAdapter(ProviderAdapter[object]):
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


def test_format_publish_failures_handles_empty_sequence() -> None:
    assert (
        shared.format_publish_failures(())
        == "Reducer errors prevented applying tool result."
    )


def test_format_publish_failures_includes_handler_errors() -> None:
    failure = HandlerFailure(handler=lambda _: None, error=ValueError("boom"))

    message = shared.format_publish_failures((failure,))

    assert "boom" in message


def test_format_publish_failures_uses_error_class_name_when_message_missing() -> None:
    failure = HandlerFailure(handler=lambda _: None, error=Exception())

    message = shared.format_publish_failures((failure,))

    assert "Exception" in message


def test_extract_payload_rejects_non_string_keys() -> None:
    class Response:
        def model_dump(self) -> Mapping[object, object]:  # pragma: no cover - stub
            return {1: "value"}

    assert shared.extract_payload(Response()) is None


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


def test_run_conversation_raises_when_deadline_expires_during_backoff() -> None:
    rendered = RenderedPrompt(text="system")
    bus = NullEventBus()
    throttle_policy = shared.ThrottlePolicy(
        base_delay_seconds=1.0,
        max_delay_seconds=1.0,
        max_attempts=2,
        max_total_seconds=5.0,
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
            kind="timeout",
            retry_after=0.0,
            provider_payload=None,
            message="timed out",
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

    class DummyDeadline:
        def __init__(self) -> None:
            self.expires_at = datetime.now(UTC) + timedelta(hours=1)

        def remaining(self) -> timedelta:
            return timedelta(milliseconds=50)

    adapter = DummyAdapter()
    prompt = Prompt(ns="tests", key="example")
    session = Session(bus=bus)
    deadline = DummyDeadline()

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
            deadline=deadline,  # type: ignore[arg-type]
        )

    assert "Deadline expired" in str(err.value)
    payload = getattr(err.value, "provider_payload", {})
    assert payload.get("throttle", {}).get("kind") == "timeout"


def test_parse_tool_arguments_returns_empty_when_missing() -> None:
    assert (
        shared.parse_tool_arguments(
            None,
            prompt_name="example",
            provider_payload=None,
        )
        == {}
    )


def test_parse_tool_arguments_raises_on_invalid_json() -> None:
    with pytest.raises(PromptEvaluationError):
        shared.parse_tool_arguments(
            "{invalid}",
            prompt_name="example",
            provider_payload=None,
        )


def test_parse_tool_arguments_requires_mapping_payload() -> None:
    with pytest.raises(PromptEvaluationError):
        shared.parse_tool_arguments(
            "[]",
            prompt_name="example",
            provider_payload=None,
        )


def test_execute_tool_call_requires_registered_tool() -> None:
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="missing", arguments="{}"),
        id="call",
    )
    empty_registry = cast(
        Mapping[str, Tool[SupportsDataclass, SupportsToolResult]],
        {},
    )

    with pytest.raises(PromptEvaluationError):
        shared.execute_tool_call(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=_StubAdapter(),
            prompt=Prompt(ns="tests", key="example"),
            rendered_prompt=None,
            tool_call=tool_call,
            tool_registry=empty_registry,
            bus=NullEventBus(),
            session=Session(bus=NullEventBus()),
            prompt_name="example",
            provider_payload=None,
            deadline=None,
            format_publish_failures=shared.format_publish_failures,
            parse_arguments=shared.parse_tool_arguments,
            logger_override=None,
        )


def test_execute_tool_call_requires_registered_handler() -> None:
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="no_handler", arguments="{}"),
        id="call",
    )
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclass, SupportsToolResult]],
        {
            "no_handler": Tool[_ToolParams, _ExampleOutput](
                name="no_handler",
                description="desc",
                handler=None,
            )
        },
    )

    with pytest.raises(PromptEvaluationError):
        shared.execute_tool_call(
            adapter_name=TEST_ADAPTER_NAME,
            adapter=_StubAdapter(),
            prompt=Prompt(ns="tests", key="example"),
            rendered_prompt=None,
            tool_call=tool_call,
            tool_registry=tool_registry,
            bus=NullEventBus(),
            session=Session(bus=NullEventBus()),
            prompt_name="example",
            provider_payload=None,
            deadline=None,
            format_publish_failures=shared.format_publish_failures,
            parse_arguments=shared.parse_tool_arguments,
            logger_override=None,
        )


def test_build_json_schema_response_format_wraps_array_payloads() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=_ExampleOutput,
            container="array",
            allow_extra_keys=False,
        ),
    )

    schema = shared.build_json_schema_response_format(rendered, "Example Prompt")

    assert schema is not None
    json_schema_wrapper = cast(dict[str, Any], schema["json_schema"])
    json_schema = cast(dict[str, Any], json_schema_wrapper["schema"])
    assert json_schema["required"] == [ARRAY_WRAPPER_KEY]
    assert cast(str, json_schema_wrapper["name"]) == "Example_Prompt_schema"


def test_build_json_schema_response_format_returns_none_without_structured_output() -> (
    None
):
    rendered = RenderedPrompt(text="system")

    assert shared.build_json_schema_response_format(rendered, "example") is None


def test_parse_schema_constrained_payload_requires_structured_output() -> None:
    rendered = RenderedPrompt(text="system")

    with pytest.raises(TypeError):
        shared.parse_schema_constrained_payload({}, rendered)


def test_message_text_content_concatenates_sequence_parts() -> None:
    parts = [
        {"type": "text", "text": "Hello"},
        {"type": "output_text", "text": " world"},
    ]

    assert shared.message_text_content(parts) == "Hello world"


def test_message_text_content_falls_back_to_str_conversion() -> None:
    assert shared.message_text_content(42) == "42"


def test_extract_parsed_content_reads_sequence_payloads() -> None:
    message = SimpleNamespace(
        content=[{"type": "output_json", "json": {"value": "ok"}}]
    )

    assert shared.extract_parsed_content(message) == {"value": "ok"}


def test_schema_name_strips_non_alphanumeric_characters() -> None:
    assert shared._schema_name("  Example Prompt!  ") == "Example_Prompt_schema"


def test_content_part_text_handles_mapping_payloads() -> None:
    part = {"type": "output_text", "text": "result"}

    assert shared._content_part_text(part) == "result"


def test_content_part_text_handles_object_payloads() -> None:
    part = SimpleNamespace(type="text", text="value")

    assert shared._content_part_text(part) == "value"


def test_content_part_text_returns_empty_for_missing_text() -> None:
    assert shared._content_part_text(None) == ""
    assert shared._content_part_text({"type": "text"}) == ""
    assert shared._content_part_text(SimpleNamespace(type="text", text=None)) == ""


def test_parsed_payload_from_part_handles_output_json() -> None:
    part = {"type": "output_json", "json": {"value": "ok"}}

    assert shared._parsed_payload_from_part(part) == {"value": "ok"}


def test_parsed_payload_from_part_handles_object_payload() -> None:
    part = SimpleNamespace(type="output_json", json={"value": "ok"})

    assert shared._parsed_payload_from_part(part) == {"value": "ok"}


def test_parsed_payload_from_part_returns_none_when_missing_json_marker() -> None:
    assert shared._parsed_payload_from_part({"type": "text"}) is None
    assert shared._parsed_payload_from_part(SimpleNamespace(type="text")) is None


def test_response_parser_raises_on_invalid_parsed_payload() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=_ExampleOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    parser = shared.ResponseParser[_ExampleOutput](
        prompt_name="example",
        rendered=rendered,
        parse_output=True,
        require_structured_output_text=False,
    )
    message = SimpleNamespace(parsed={"unexpected": "field"}, content=None)

    with pytest.raises(PromptEvaluationError):
        parser.parse(message, provider_payload=None)


def test_response_parser_requires_structured_output_text_when_configured() -> None:
    rendered = RenderedPrompt(
        text="system",
        structured_output=StructuredOutputConfig(
            dataclass_type=_ExampleOutput,
            container="object",
            allow_extra_keys=False,
        ),
    )
    parser = shared.ResponseParser[_ExampleOutput](
        prompt_name="example",
        rendered=rendered,
        parse_output=True,
        require_structured_output_text=True,
    )
    message = SimpleNamespace(parsed=None, content=None)

    with pytest.raises(PromptEvaluationError):
        parser.parse(message, provider_payload=None)


def test_handle_tool_calls_resets_next_tool_choice_after_function_lock() -> None:
    rendered = RenderedPrompt(text="system")
    bus = NullEventBus()

    runner = shared.ConversationRunner[object](
        adapter_name=TEST_ADAPTER_NAME,
        adapter=_StubAdapter(),
        prompt=Prompt(ns="tests", key="example"),
        prompt_name="example",
        rendered=rendered,
        render_inputs=(),
        initial_messages=[{"role": "system", "content": rendered.text}],
        parse_output=False,
        bus=bus,
        session=Session(bus=bus),
        tool_choice={"type": "function", "function": {"name": "forced"}},
        response_format=None,
        require_structured_output_text=False,
        call_provider=lambda *_args: None,
        select_choice=lambda response: response,
        serialize_tool_message_fn=lambda result, *, payload=None: {"payload": payload},
        format_publish_failures=shared.format_publish_failures,
        parse_arguments=shared.parse_tool_arguments,
        logger_override=None,
        deadline=None,
        throttle_policy=shared.DEFAULT_THROTTLE_POLICY,
    )
    runner._prepare_payload()
    runner._provider_payload = {}
    runner._messages = []

    class StubExecutor:
        def __init__(self) -> None:
            self.tool_message_records: list[
                tuple[ToolResult[SupportsDataclass], dict[str, Any]]
            ] = []
            self.tool_events: list[Any] = []

        def execute(
            self,
            tool_calls: Sequence[shared.ProviderToolCall],
            provider_payload: Mapping[str, Any] | None,
        ) -> tuple[list[dict[str, Any]], shared.ToolChoice]:
            return ([{"role": "tool", "content": "result"}], "auto")

    runner._tool_executor = cast(shared.ToolExecutor, StubExecutor())

    tool_call = SimpleNamespace(
        function=SimpleNamespace(name="tool", arguments="{}"),
        id="call",
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    runner._handle_tool_calls(message, message.tool_calls)

    assert runner._next_tool_choice == "auto"


def test_finalize_response_updates_last_tool_message_with_structured_output() -> None:
    rendered = RenderedPrompt(text="system")
    bus = NullEventBus()

    runner = shared.ConversationRunner[object](
        adapter_name=TEST_ADAPTER_NAME,
        adapter=_StubAdapter(),
        prompt=Prompt(ns="tests", key="example"),
        prompt_name="example",
        rendered=rendered,
        render_inputs=(),
        initial_messages=[{"role": "system", "content": rendered.text}],
        parse_output=True,
        bus=bus,
        session=Session(bus=bus),
        tool_choice="auto",
        response_format=None,
        require_structured_output_text=False,
        call_provider=lambda *_args: None,
        select_choice=lambda response: response,
        serialize_tool_message_fn=lambda result, *, payload=None: {"payload": payload},
        format_publish_failures=shared.format_publish_failures,
        parse_arguments=shared.parse_tool_arguments,
        logger_override=None,
        deadline=None,
        throttle_policy=shared.DEFAULT_THROTTLE_POLICY,
    )
    runner._prepare_payload()
    runner._provider_payload = {}
    runner._messages = []

    success_result = ToolResult(message="ok", value=None, success=True)
    runner._tool_executor = cast(
        shared.ToolExecutor,
        SimpleNamespace(
            tool_message_records=[(success_result, {"content": "tool"})],
            tool_events=[],
        ),
    )
    runner._response_parser = cast(
        shared.ResponseParser[object],
        SimpleNamespace(
            parse=lambda message, payload: (_ExampleOutput(value="ok"), None),
            should_parse_structured_output=False,
        ),
    )

    response = runner._finalize_response(SimpleNamespace(content="assistant"))

    last_message = runner._tool_executor.tool_message_records[-1][1]
    assert last_message["content"]["payload"].value == "ok"
    assert response.output == _ExampleOutput(value="ok")
