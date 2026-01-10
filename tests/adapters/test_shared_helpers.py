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
from tests.helpers.events import NullDispatcher
from weakincentives.adapters import PromptEvaluationError
from weakincentives.adapters._provider_protocols import ProviderChoice
from weakincentives.adapters.core import (
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.adapters.inner_loop import (
    InnerLoopConfig,
    InnerLoopInputs,
    run_inner_loop,
)
from weakincentives.adapters.response_parser import (
    ResponseParser,
    build_json_schema_response_format,
)
from weakincentives.adapters.tool_executor import ToolMessageSerializer
from weakincentives.adapters.utilities import (
    ToolChoice,
    extract_payload,
    first_choice,
    mapping_to_str_dict,
    parse_tool_arguments,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import Prompt, PromptTemplate
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.runtime.events import Dispatcher
from weakincentives.runtime.session import Session


def test_first_choice_returns_first_item() -> None:
    response = SimpleNamespace(choices=["first", "second"])

    assert first_choice(response, prompt_name="example") == "first"


def test_first_choice_requires_sequence() -> None:
    response = SimpleNamespace(choices=None)

    with pytest.raises(PromptEvaluationError):
        first_choice(response, prompt_name="example")


def test_parse_tool_arguments_rejects_non_string_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_loads(_: str) -> Mapping[Any, Any]:
        # Simulate a mapping that does not use string keys to exercise defensive branch.
        return {1: "value"}

    import weakincentives.adapters.tool_spec as tool_spec_module

    monkeypatch.setattr(tool_spec_module, "json", SimpleNamespace(loads=fake_loads))

    with pytest.raises(PromptEvaluationError) as err:
        parse_tool_arguments(
            "{}",
            prompt_name="example",
            provider_payload=None,
        )

    message = str(err.value)
    assert "string keys" in message


def test_mapping_to_str_dict_rejects_non_string_keys() -> None:
    assert mapping_to_str_dict({1: "value"}) is None


def test_run_inner_loop_requires_message_payload() -> None:
    rendered = RenderedPrompt(text="system")
    dispatcher = NullDispatcher()

    class DummyChoice:
        def __init__(self) -> None:
            self.message = None

    class DummyResponse:
        def __init__(self) -> None:
            self.choices = [DummyChoice()]

    def call_provider(
        messages: list[dict[str, Any]],
        tool_specs: list[Mapping[str, Any]],
        tool_choice: ToolChoice | None,
        response_format: Mapping[str, Any] | None,
    ) -> DummyResponse:
        return DummyResponse()

    def select_choice(response: DummyResponse) -> ProviderChoice:
        return response.choices[0]

    serialize_stub = cast(
        ToolMessageSerializer,
        lambda _result, *, payload=None: "",
    )

    class DummyAdapter(ProviderAdapter[object]):
        def evaluate(
            self,
            prompt: Prompt[object],
            *,
            dispatcher: Dispatcher,
            session: SessionProtocol,
            deadline: Deadline | None = None,
        ) -> PromptResponse[object]:
            raise NotImplementedError

    adapter = DummyAdapter()
    prompt = Prompt(PromptTemplate(ns="tests", key="example"))
    session = Session(dispatcher=dispatcher)

    config = InnerLoopConfig(
        session=session,
        tool_choice="auto",
        response_format=None,
        require_structured_output_text=False,
        call_provider=call_provider,
        select_choice=select_choice,
        serialize_tool_message_fn=serialize_stub,
    )

    inputs = InnerLoopInputs[object](
        adapter_name=TEST_ADAPTER_NAME,
        adapter=adapter,
        prompt=prompt,
        prompt_name="example",
        rendered=rendered,
        render_inputs=(),
        initial_messages=[{"role": "system", "content": rendered.text}],
    )

    with pytest.raises(PromptEvaluationError):
        run_inner_loop(inputs=inputs, config=config)


def test_extract_payload_handles_mapping_with_non_string_keys() -> None:
    """Test branch 389->391: mapping_payload is None due to non-string keys."""

    class ResponseWithBadKeys:
        def model_dump(self) -> dict[int, str]:
            # Mapping with non-string keys
            return {1: "value"}

    response = ResponseWithBadKeys()
    result = extract_payload(response)
    assert result is None


def test_build_json_schema_response_format_with_extra_keys_allowed() -> None:
    """Test branch 997->1002: allow_extra_keys is True."""
    from dataclasses import dataclass

    from weakincentives.prompt import Prompt, PromptTemplate

    @dataclass
    class TestOutput:
        value: str

    template = PromptTemplate[list[TestOutput]](
        ns="tests",
        key="test-extra-keys",
        name="test",
        sections=[],
        allow_extra_keys=True,
    )

    rendered = Prompt(template).bind().render()
    result = build_json_schema_response_format(rendered, "test")

    assert result is not None
    schema_payload = result["json_schema"]["schema"]  # type: ignore[index]
    # When allow_extra_keys is True, additionalProperties should not be set to False
    assert schema_payload.get("additionalProperties") != False  # noqa: E712


def test_response_parser_preserves_text_when_output_is_none() -> None:
    """Test branch 1455->1458: output is None, text_value is preserved."""
    from types import SimpleNamespace

    from weakincentives.prompt import Prompt, PromptTemplate

    # Create a plain text prompt (no structured output)
    template = PromptTemplate(
        ns="tests",
        key="test-plain",
        name="test",
        sections=[],
    )

    rendered = Prompt(template).bind().render()
    parser = ResponseParser(
        prompt_name="test",
        rendered=rendered,
        require_structured_output_text=False,
    )

    # Create a message with plain text content
    message = SimpleNamespace(content="Plain text response")
    output, text_value = parser.parse(message, provider_payload=None)

    # output should be None (no structured output)
    assert output is None
    # text_value should be preserved (not set to None)
    assert text_value == "Plain text response"


def test_response_parser_clears_text_when_output_is_not_none() -> None:
    """Test branch 1455->1456: when output is successfully parsed, text_value is set to None."""
    from dataclasses import dataclass
    from types import SimpleNamespace

    from weakincentives.prompt import Prompt, PromptTemplate

    @dataclass
    class TestOutput:
        message: str

    # Create a prompt with structured output
    template = PromptTemplate[TestOutput](
        ns="tests",
        key="test-structured",
        name="test",
        sections=[],
    )

    rendered = Prompt(template).bind().render()
    parser = ResponseParser(
        prompt_name="test",
        rendered=rendered,
        require_structured_output_text=False,
    )

    # Create a message with text content and a parsed attribute containing valid JSON
    # This triggers line 1423-1430 where extract_parsed_content returns non-None
    message = SimpleNamespace(
        content="Some text content",
        parsed={
            "message": "parsed output"
        },  # This will be parsed by parse_schema_constrained_payload
    )
    output, text_value = parser.parse(message, provider_payload=None)

    # output should be parsed (line 1425-1430 succeeds)
    assert output is not None
    assert output.message == "parsed output"  # type: ignore[attr-defined]
    # text_value should be set to None (line 1456, triggered by line 1455 when output is not None)
    assert text_value is None
