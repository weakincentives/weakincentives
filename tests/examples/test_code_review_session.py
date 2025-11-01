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

import json
from typing import cast

import pytest

from weakincentives.adapters.core import PromptResponse
from weakincentives.events import EventBus, PromptExecuted, ToolInvoked
from weakincentives.examples.code_review_prompt import ReviewResponse, ReviewTurnParams
from weakincentives.examples.code_review_session import CodeReviewSession
from weakincentives.examples.code_review_tools import (
    BranchListParams,
    BranchListResult,
)
from weakincentives.prompt import Prompt, SupportsDataclass
from weakincentives.prompt.tool import ToolResult
from weakincentives.session import PromptData, ToolData


class _StubAdapter:
    def __init__(self, response: PromptResponse[ReviewResponse]) -> None:
        self.response = response
        self.calls: list[
            tuple[
                Prompt[ReviewResponse],
                tuple[SupportsDataclass, ...],
                bool,
                EventBus,
            ]
        ] = []

    def evaluate(
        self,
        prompt: Prompt[ReviewResponse],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
    ) -> PromptResponse[ReviewResponse]:
        self.calls.append((prompt, params, parse_output, bus))
        return self.response


def test_code_review_session_evaluate_serializes_output() -> None:
    prompt_response = PromptResponse(
        prompt_name="code_review_agent",
        text=None,
        output=ReviewResponse(
            summary="Looks good.",
            issues=["Missing test coverage"],
            next_steps=["Add tests"],
        ),
        tool_results=(),
    )
    adapter = _StubAdapter(prompt_response)
    session = CodeReviewSession(adapter)

    output = session.evaluate("Review the PR")

    payload = json.loads(output)
    assert payload == {
        "summary": "Looks good.",
        "issues": ["Missing test coverage"],
        "next_steps": ["Add tests"],
    }
    ((prompt, params, parse_output, bus),) = adapter.calls
    assert prompt.name == "code_review_agent"
    assert isinstance(params[0], ReviewTurnParams)
    assert params[0].request == "Review the PR"
    assert parse_output is True
    assert bus is not None


def test_code_review_session_tool_history(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt_response = PromptResponse(
        prompt_name="code_review_agent",
        text="fallback",
        output=None,
        tool_results=(),
    )
    adapter = _StubAdapter(prompt_response)
    session = CodeReviewSession(adapter)

    assert session.render_tool_history() == "No tool calls recorded yet."

    result = cast(
        ToolResult[object],
        ToolResult(
            message="Listed branches.",
            value=BranchListResult(branches=["main"]),
        ),
    )
    event = ToolInvoked(
        prompt_name="code_review_agent",
        adapter="stub",
        name="show_git_branches",
        params=BranchListParams(),
        result=result,
        call_id="abc123",
    )

    session._session._dispatch_data_event(
        BranchListResult,
        ToolData(
            value=cast(SupportsDataclass, result.value),
            source=event,
        ),
    )

    session._bus.publish(event)
    captured = capsys.readouterr()
    assert "[tool] show_git_branches called with" in captured.out

    history = session.render_tool_history()
    assert "1. show_git_branches (code_review_agent)" in history
    assert "call_id: abc123" in history
    assert '"branches": ["main"]' in history


def test_code_review_session_returns_text_when_no_structured_output() -> None:
    prompt_response = PromptResponse(
        prompt_name="code_review_agent",
        text="Fallback text",
        output=None,
        tool_results=(),
    )
    adapter = _StubAdapter(prompt_response)
    session = CodeReviewSession(adapter)

    output = session.evaluate("Provide a summary")

    assert output == "Fallback text"
    assert adapter.calls  # ensure adapter was invoked


def test_code_review_session_returns_default_message() -> None:
    prompt_response = PromptResponse(
        prompt_name="code_review_agent",
        text="",
        output=None,
        tool_results=(),
    )
    session = CodeReviewSession(_StubAdapter(prompt_response))

    output = session.evaluate("Provide a summary")

    assert output == "(no response from assistant)"


def test_code_review_session_display_tool_event_prints_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_response = PromptResponse(
        prompt_name="code_review_agent",
        text=None,
        output=None,
        tool_results=(),
    )
    session = CodeReviewSession(_StubAdapter(prompt_response))

    result = cast(
        ToolResult[object],
        ToolResult(
            message="Listed branches.",
            value=BranchListResult(branches=["dev"]),
        ),
    )
    event = ToolInvoked(
        prompt_name="code_review_agent",
        adapter="stub",
        name="show_git_branches",
        params=BranchListParams(),
        result=result,
        call_id="xyz",
    )

    session._session._dispatch_data_event(
        BranchListResult,
        ToolData(
            value=cast(SupportsDataclass, result.value),
            source=event,
        ),
    )

    session._display_tool_event(event)

    output = capsys.readouterr().out
    assert "payload" in output
    assert "session recorded this call as #1" in output


def test_code_review_session_display_tool_event_ignores_other_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = CodeReviewSession(
        _StubAdapter(
            PromptResponse(
                prompt_name="code_review_agent",
                text=None,
                output=None,
                tool_results=(),
            )
        )
    )

    session._display_tool_event(object())

    assert capsys.readouterr().out == ""


def test_code_review_session_record_tool_call_ignores_non_tool_data() -> None:
    session = CodeReviewSession(
        _StubAdapter(
            PromptResponse(
                prompt_name="code_review_agent",
                text=None,
                output=None,
                tool_results=(),
            )
        )
    )

    prompt_response = PromptResponse(
        prompt_name="code_review_agent",
        text=None,
        output=ReviewResponse(summary="", issues=[], next_steps=[]),
        tool_results=(),
    )
    prompt_event = PromptExecuted(
        prompt_name="code_review_agent",
        adapter="stub",
        result=cast(PromptResponse[object], prompt_response),
    )

    result = session._record_tool_call(
        (),
        PromptData(
            value=ReviewResponse(summary="", issues=[], next_steps=[]),
            source=prompt_event,
        ),
    )

    assert result == ()
