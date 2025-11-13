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

from weakincentives import deadlines
from weakincentives.adapters import shared
from weakincentives.adapters.core import (
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
)
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.prompt import RenderedPrompt
from weakincentives.prompt.tool import Tool, ToolContext
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import InProcessEventBus
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


def _build_prompt() -> Prompt[BodyResult]:
    section = MarkdownSection[BodyParams](
        title="Body",
        key="body",
        template="${content}",
    )
    return Prompt[BodyResult](ns="tests", key="deadline", sections=(section,))


def _build_rendered(prompt: Prompt[BodyResult]) -> RenderedPrompt[BodyResult]:
    return prompt.render(BodyParams(content="ready"))


def test_deadline_provider_payload_handles_none() -> None:
    assert shared.deadline_provider_payload(None) is None


def test_raise_tool_deadline_error() -> None:
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))
    with pytest.raises(PromptEvaluationError) as excinfo:
        shared._raise_tool_deadline_error(
            prompt_name="test", tool_name="tool", deadline=deadline
        )
    error = cast(PromptEvaluationError, excinfo.value)
    assert error.phase == "deadline"
    assert error.provider_payload == {
        "deadline_expires_at": deadline.expires_at.isoformat()
    }


def test_conversation_runner_raise_deadline_error() -> None:
    prompt = _build_prompt()
    rendered = _build_rendered(prompt)
    bus = InProcessEventBus()
    session: SessionProtocol = Session(bus=bus)
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))
    runner = shared.ConversationRunner[BodyResult](
        adapter_name="test",
        adapter=cast(ProviderAdapter[BodyResult], object()),
        prompt=prompt,
        prompt_name="deadline",
        rendered=rendered,
        render_inputs=(BodyParams(content="ready"),),
        initial_messages=[{"role": "system", "content": rendered.text}],
        parse_output=False,
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
    with pytest.raises(PromptEvaluationError):
        runner._raise_deadline_error("expired")


def test_conversation_runner_detects_expired_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _build_prompt()
    rendered = _build_rendered(prompt)
    bus = InProcessEventBus()
    session: SessionProtocol = Session(bus=bus)
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(deadlines, "_utcnow", lambda: anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))
    runner = shared.ConversationRunner[BodyResult](
        adapter_name="test",
        adapter=cast(ProviderAdapter[BodyResult], object()),
        prompt=prompt,
        prompt_name="deadline",
        rendered=rendered,
        render_inputs=(BodyParams(content="ready"),),
        initial_messages=[{"role": "system", "content": rendered.text}],
        parse_output=False,
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
    monkeypatch.setattr(
        deadlines,
        "_utcnow",
        lambda: anchor + timedelta(seconds=10),
    )
    with pytest.raises(PromptEvaluationError):
        runner._ensure_deadline_remaining("expired")


def test_execute_tool_call_raises_when_deadline_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _build_prompt()
    rendered = _build_rendered(prompt)
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
        Mapping[str, Tool[SupportsDataclass, SupportsDataclass]], {tool.name: tool}
    )
    call = SimpleNamespace(
        id="call", function=SimpleNamespace(name="echo", arguments='{"content": "hi"}')
    )
    anchor = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(deadlines, "_utcnow", lambda: anchor)
    deadline = Deadline(anchor + timedelta(seconds=5))
    monkeypatch.setattr(
        deadlines,
        "_utcnow",
        lambda: anchor + timedelta(seconds=10),
    )
    with pytest.raises(PromptEvaluationError) as excinfo:
        shared.execute_tool_call(
            adapter_name="test",
            adapter=cast(ProviderAdapter[BodyResult], object()),
            prompt=prompt,
            rendered_prompt=rendered,
            tool_call=cast(shared.ProviderToolCall, call),
            tool_registry=tool_registry,
            bus=bus,
            session=session,
            prompt_name="deadline",
            provider_payload=None,
            deadline=deadline,
            format_publish_failures=shared.format_publish_failures,
            parse_arguments=shared.parse_tool_arguments,
            logger_override=None,
        )
    error = cast(PromptEvaluationError, excinfo.value)
    assert error.phase == "deadline"


def test_run_conversation_replaces_rendered_deadline() -> None:
    prompt = _build_prompt()
    rendered = _build_rendered(prompt)
    bus = InProcessEventBus()
    session = Session(bus=bus)
    deadline = Deadline(datetime.now(UTC) + timedelta(seconds=5))

    def call_provider(
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        tool_choice: shared.ToolChoice | None,
        response_format: dict[str, Any] | None,
    ) -> object:
        del messages, tool_specs, tool_choice, response_format
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None)
                )
            ]
        )

    def select_choice(response: SimpleNamespace) -> shared.ProviderChoice:
        return cast(shared.ProviderChoice, response.choices[0])

    result = shared.run_conversation(
        adapter_name="test",
        adapter=cast(ProviderAdapter[BodyResult], object()),
        prompt=prompt,
        prompt_name="deadline",
        rendered=rendered,
        render_inputs=(BodyParams(content="ready"),),
        initial_messages=[{"role": "system", "content": rendered.text}],
        parse_output=False,
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
        logger_override=None,
        deadline=deadline,
    )

    assert isinstance(result, PromptResponse)
