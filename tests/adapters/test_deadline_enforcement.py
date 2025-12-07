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

"""Tests covering deadline enforcement helpers in shared adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.helpers import FrozenUtcNow
from tests.helpers.adapters import TEST_ADAPTER_NAME
from weakincentives.adapters import shared
from weakincentives.adapters.core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.prompt._types import SupportsDataclassOrNone, SupportsToolResult
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.tool import Tool, ToolContext
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import InProcessEventBus, ToolInvoked
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.protocols import SessionProtocol


@dataclass(slots=True)
class BodyParams:
    content: str


@dataclass(slots=True)
class BodyResult:
    message: str


@dataclass(slots=True)
class EchoParams:
    content: str


@dataclass(slots=True)
class EchoResult:
    content: str


def _build_prompt() -> PromptTemplate[BodyResult]:
    section = MarkdownSection[BodyParams](
        title="Body",
        key="body",
        template="${content}",
    )
    return PromptTemplate[BodyResult](ns="tests", key="deadline", sections=(section,))


def _tool_context(
    *,
    prompt: Prompt[BodyResult],
    rendered: RenderedPrompt[BodyResult],
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
    bus: InProcessEventBus,
    session: SessionProtocol,
    prompt_name: str,
    provider_payload: dict[str, Any] | None = None,
    deadline: Deadline | None = None,
) -> shared.ToolExecutionContext:
    return shared.ToolExecutionContext(
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[BodyResult], object()),
        prompt=prompt,
        rendered_prompt=rendered,
        tool_registry=tool_registry,
        bus=bus,
        session=session,
        prompt_name=prompt_name,
        parse_arguments=shared.parse_tool_arguments,
        format_publish_failures=shared.format_publish_failures,
        deadline=deadline,
        provider_payload=provider_payload,
    )


def test_deadline_provider_payload_handles_none() -> None:
    assert shared.deadline_provider_payload(None) is None


def test_raise_tool_deadline_error() -> None:
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))
    with pytest.raises(PromptEvaluationError) as excinfo:
        shared._raise_tool_deadline_error(
            prompt_name="test", tool_name="tool", deadline=deadline
        )
    assert isinstance(excinfo.value, PromptEvaluationError)
    error = excinfo.value
    assert error.phase == PROMPT_EVALUATION_PHASE_TOOL
    assert error.provider_payload == {
        "deadline_expires_at": deadline.expires_at.isoformat()
    }


def test_inner_loop_raise_deadline_error() -> None:
    prompt = Prompt(_build_prompt()).bind(BodyParams(content="ready"))
    rendered = prompt.render()
    bus = InProcessEventBus()
    session: SessionProtocol = Session(bus=bus)
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))

    inputs = shared.InnerLoopInputs[BodyResult](
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[BodyResult], object()),
        prompt=prompt,
        prompt_name="deadline",
        rendered=rendered,
        render_inputs=prompt.params,
        initial_messages=[{"role": "system", "content": rendered.text}],
    )
    config = shared.InnerLoopConfig(
        bus=bus,
        session=session,
        tool_choice="auto",
        response_format=None,
        require_structured_output_text=False,
        call_provider=lambda *_args: SimpleNamespace(choices=[]),
        select_choice=lambda response: response.choices[0],
        serialize_tool_message_fn=lambda *_args, **_kwargs: {},
        deadline=deadline,
    )
    loop = shared.InnerLoop[BodyResult](inputs=inputs, config=config)
    with pytest.raises(PromptEvaluationError):
        loop._raise_deadline_error("expired", phase=PROMPT_EVALUATION_PHASE_REQUEST)


def test_inner_loop_detects_expired_deadline(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    prompt = Prompt(_build_prompt()).bind(BodyParams(content="ready"))
    rendered = prompt.render()
    bus = InProcessEventBus()
    session: SessionProtocol = Session(bus=bus)
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))

    inputs = shared.InnerLoopInputs[BodyResult](
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[BodyResult], object()),
        prompt=prompt,
        prompt_name="deadline",
        rendered=rendered,
        render_inputs=prompt.params,
        initial_messages=[{"role": "system", "content": rendered.text}],
    )
    config = shared.InnerLoopConfig(
        bus=bus,
        session=session,
        tool_choice="auto",
        response_format=None,
        require_structured_output_text=False,
        call_provider=lambda *_args: SimpleNamespace(choices=[]),
        select_choice=lambda response: response.choices[0],
        serialize_tool_message_fn=lambda *_args, **_kwargs: {},
        deadline=deadline,
    )
    loop = shared.InnerLoop[BodyResult](inputs=inputs, config=config)
    frozen_utcnow.advance(timedelta(seconds=10))
    with pytest.raises(PromptEvaluationError):
        loop._ensure_deadline_remaining(
            "expired", phase=PROMPT_EVALUATION_PHASE_REQUEST
        )


def test_execute_tool_call_raises_when_deadline_expired(
    frozen_utcnow: FrozenUtcNow,
) -> None:
    prompt = Prompt(_build_prompt()).bind(BodyParams(content="ready"))
    rendered = prompt.render()
    bus = InProcessEventBus()
    session: SessionProtocol = Session(bus=bus)

    def handler(params: EchoParams, *, context: ToolContext) -> ToolResult[EchoResult]:
        del context
        return ToolResult(message="", value=EchoResult(content=params.content))

    tool = Tool[EchoParams, EchoResult](
        name="echo",
        description="echo",
        handler=handler,
    )
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
        {tool.name: tool},
    )
    call = SimpleNamespace(
        id="call", function=SimpleNamespace(name="echo", arguments='{"content": "hi"}')
    )
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    frozen_utcnow.set(anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))
    frozen_utcnow.advance(timedelta(seconds=10))
    with pytest.raises(PromptEvaluationError) as excinfo:
        shared.execute_tool_call(
            context=_tool_context(
                prompt=prompt,
                rendered=rendered,
                tool_registry=tool_registry,
                bus=bus,
                session=session,
                prompt_name="deadline",
                deadline=deadline,
            ),
            tool_call=cast(shared.ProviderToolCall, call),
        )
    assert isinstance(excinfo.value, PromptEvaluationError)
    error = excinfo.value
    assert error.phase == PROMPT_EVALUATION_PHASE_TOOL


def test_execute_tool_call_publishes_invocation() -> None:
    prompt = Prompt(_build_prompt()).bind(BodyParams(content="ready"))
    rendered = prompt.render()
    bus = InProcessEventBus()
    session: SessionProtocol = Session(bus=bus)

    events: list[ToolInvoked] = []

    def record_event(event: object) -> None:
        assert isinstance(event, ToolInvoked)
        events.append(event)

    bus.subscribe(ToolInvoked, record_event)

    def handler(params: EchoParams, *, context: ToolContext) -> ToolResult[EchoResult]:
        del context
        return ToolResult(message="done", value=EchoResult(content=params.content))

    tool = Tool[EchoParams, EchoResult](
        name="echo",
        description="echo",
        handler=handler,
    )
    tool_registry = cast(
        Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
        {tool.name: tool},
    )
    call = SimpleNamespace(
        id="call", function=SimpleNamespace(name="echo", arguments='{"content": "hi"}')
    )

    invocation, result = shared.execute_tool_call(
        context=_tool_context(
            prompt=prompt,
            rendered=rendered,
            tool_registry=tool_registry,
            bus=bus,
            session=session,
            prompt_name="publish",
        ),
        tool_call=cast(shared.ProviderToolCall, call),
    )

    assert result.success is True
    assert invocation in events
    assert invocation.name == "echo"
    assert isinstance(invocation.params, EchoParams)
    assert invocation.params.content == "hi"


def test_run_inner_loop_replaces_rendered_deadline() -> None:
    prompt = Prompt(_build_prompt()).bind(BodyParams(content="ready"))
    rendered = prompt.render()
    bus = InProcessEventBus()
    session = Session(bus=bus)
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))

    def call_provider(
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        tool_choice: shared.ToolChoice,
        response_format: Mapping[str, Any] | None,
        /,
    ) -> object:
        del messages, tool_specs, tool_choice, response_format
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        parsed={"message": "done"},
                    )
                )
            ]
        )

    def select_choice(response: SimpleNamespace, /) -> shared.ProviderChoice:
        return cast(shared.ProviderChoice, response.choices[0])

    config = shared.InnerLoopConfig(
        bus=bus,
        session=session,
        tool_choice="auto",
        response_format=None,
        require_structured_output_text=False,
        call_provider=call_provider,
        select_choice=select_choice,
        serialize_tool_message_fn=lambda *_args, **_kwargs: {},
        format_publish_failures=shared.format_publish_failures,
        parse_arguments=shared.parse_tool_arguments,
        deadline=deadline,
    )

    inputs = shared.InnerLoopInputs[BodyResult](
        adapter_name=TEST_ADAPTER_NAME,
        adapter=cast(ProviderAdapter[BodyResult], object()),
        prompt=prompt,
        prompt_name="deadline",
        rendered=rendered,
        render_inputs=prompt.params,
        initial_messages=[{"role": "system", "content": rendered.text}],
    )

    result = shared.run_inner_loop(inputs=inputs, config=config)

    assert isinstance(result, PromptResponse)
