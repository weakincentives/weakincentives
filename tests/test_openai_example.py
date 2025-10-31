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

"""Tests for the example code review agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol, cast

import pytest

from examples import common
from examples import openai as openai_example
from weakincentives.adapters.core import PromptResponse
from weakincentives.events import EventBus, ToolInvoked
from weakincentives.prompt import Prompt
from weakincentives.prompt.tool import ToolResult


class _CodeReviewModule(Protocol):
    CodeReviewSession: type[object]
    ReviewGuidance: type[object]
    ReviewResponse: type[object]
    ReviewTurnParams: type[object]
    ReadFileParams: type[object]
    ReadFileResult: type[object]
    build_code_review_prompt: Callable[[], Prompt[object]]
    build_tools: Callable[[], tuple[object, ...]]


code_review = cast(_CodeReviewModule, common)


def test_build_prompt_lists_code_review_tools() -> None:
    prompt = code_review.build_code_review_prompt()
    rendered = prompt.render(
        code_review.ReviewGuidance(),
        code_review.ReviewTurnParams(request="Review the latest change."),
    )

    tool_names = [tool.name for tool in rendered.tools]

    assert "code review assistant" in rendered.text.lower()
    assert tool_names == [
        "read_file",
        "list_changed_files",
        "show_git_diff",
        "show_git_history",
    ]
    assert rendered.output_type is code_review.ReviewResponse


def test_session_evaluate_routes_through_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool_result = ToolResult(
        message="Read lines 1 to 1 from README.md.",
        value=code_review.ReadFileResult(snippet="Example"),
    )
    tool_event = ToolInvoked(
        prompt_name="code_review_agent",
        adapter="openai",
        name="read_file",
        params=code_review.ReadFileParams(path="README.md"),
        result=tool_result,
        call_id="tool-call-1",
    )
    prompt_response = PromptResponse(
        prompt_name="code_review_agent",
        text=None,
        output=code_review.ReviewResponse(
            summary="Looks good.", issues=["None"], next_steps=["Ship it."]
        ),
        tool_results=(tool_event,),
        provider_payload={"raw": "payload"},
    )

    captured_model: list[str] = []
    captured_calls: list[tuple[Prompt[object], tuple[object, ...], bool, EventBus]] = []

    class StubAdapter:
        def __init__(self, *, model: str) -> None:
            captured_model.append(model)

        def evaluate(
            self,
            prompt: Prompt[object],
            *params: object,
            parse_output: bool = True,
            bus: EventBus,
        ) -> PromptResponse:
            captured_calls.append((prompt, params, parse_output, bus))
            bus.publish(tool_event)
            return prompt_response

    monkeypatch.setattr(openai_example, "OpenAIAdapter", StubAdapter)

    session = openai_example.CodeReviewSession(StubAdapter(model="gpt-mock"))
    result = session.evaluate("Please review the latest diff.")
    output_lines = capsys.readouterr().out.splitlines()

    assert json.loads(result) == {
        "summary": "Looks good.",
        "issues": ["None"],
        "next_steps": ["Ship it."],
    }
    assert captured_model[0] == "gpt-mock"
    assert len(captured_calls) == 1

    call_prompt, call_params, parse_output, bus = captured_calls[0]
    assert isinstance(call_prompt, Prompt)
    assert parse_output is True
    assert len(call_params) == 1
    assert isinstance(call_params[0], code_review.ReviewTurnParams)
    assert bus is not None

    assert output_lines[0].startswith("[tool] read_file called with")
    assert "payload" in " ".join(output_lines)
    assert "session recorded this call" in output_lines[-1]
