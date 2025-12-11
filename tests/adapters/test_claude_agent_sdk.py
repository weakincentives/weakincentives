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

"""Tests for the Claude Agent SDK adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import CLINotFoundError
from claude_agent_sdk.types import ResultMessage

from weakincentives.adapters.claude_agent_sdk._errors import normalize_sdk_error
from weakincentives.adapters.claude_agent_sdk.adapter import ClaudeAgentSDKAdapter
from weakincentives.adapters.claude_agent_sdk.workspace import ClaudeAgentWorkspace
from weakincentives.budget import Budget
from weakincentives.dataclasses import FrozenDataclass
from weakincentives.deadlines import Deadline
from weakincentives.prompt._structured_output_config import StructuredOutputConfig
from weakincentives.prompt.prompt import Prompt
from weakincentives.prompt.rendering import RenderedPrompt
from weakincentives.runtime.session.protocols import SessionProtocol
from weakincentives.tools.errors import DeadlineExceededError


class _StubPrompt:
    def __init__(self, rendered: RenderedPrompt[object], name: str = "stub") -> None:
        self.rendered = rendered
        self.name = name

    def render(
        self, *, visibility_overrides: object | None = None
    ) -> RenderedPrompt[object]:
        return self.rendered


async def _stream_result(
    result_message: ResultMessage,
) -> AsyncIterator[ResultMessage]:
    await asyncio.sleep(0)
    yield result_message


def test_evaluate_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = RenderedPrompt(text="Do things")
    prompt = cast(Prompt[object], _StubPrompt(rendered))
    session = MagicMock(spec=SessionProtocol)

    result = ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session-1",
        total_cost_usd=None,
        usage={"input_tokens": 1, "output_tokens": 2},
        result="done",
        structured_output=None,
    )

    async def fake_query(*_: object, **__: object) -> AsyncIterator[ResultMessage]:
        await asyncio.sleep(0)
        yield result

    monkeypatch.setattr(
        "weakincentives.adapters.claude_agent_sdk.adapter.query", fake_query
    )

    adapter = ClaudeAgentSDKAdapter()
    response = adapter.evaluate(prompt, session=session)

    assert response.text == "done"
    assert response.output is None


def test_evaluate_parses_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    @FrozenDataclass()
    class Example:
        message: str

    rendered = RenderedPrompt(
        text="Respond with JSON",
        structured_output=StructuredOutputConfig(
            Example, container="object", allow_extra_keys=False
        ),
    )
    prompt = cast(Prompt[object], _StubPrompt(rendered, name="structured"))
    session = MagicMock(spec=SessionProtocol)

    result = ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session-2",
        total_cost_usd=None,
        usage=None,
        result=None,
        structured_output={"message": "ok"},
    )

    monkeypatch.setattr(
        "weakincentives.adapters.claude_agent_sdk.adapter.query",
        lambda *_, **__: _stream_result(result),
    )

    adapter = ClaudeAgentSDKAdapter()
    response = adapter.evaluate(prompt, session=session)

    assert response.output == Example(message="ok")


def test_workspace_stages_and_cleans(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("data")

    workspace = ClaudeAgentWorkspace(mounts=(str(source),))
    try:
        staged = workspace.root / source.name
        assert staged.read_text() == "data"
    finally:
        workspace.cleanup()
        assert not workspace.root.exists()


def test_normalize_sdk_error_handles_cli_missing() -> None:
    error = CLINotFoundError("missing")
    normalized = normalize_sdk_error(error, prompt_name="prompt")

    assert "Claude Code CLI" in normalized.message


def test_normalize_other_sdk_errors() -> None:
    from claude_agent_sdk import CLIConnectionError, CLIJSONDecodeError, ProcessError

    timeout_error = normalize_sdk_error(
        CLIConnectionError("timeout"), prompt_name="prompt"
    )
    assert "timeout" in timeout_error.message

    process_error = normalize_sdk_error(
        ProcessError(message="failed", exit_code=1, stderr="boom"),
        prompt_name="prompt",
    )
    assert "boom" in process_error.message

    parse_error = normalize_sdk_error(
        CLIJSONDecodeError("invalid", ValueError("bad")), prompt_name="prompt"
    )
    assert "Failed to parse" in parse_error.message

    fallback = normalize_sdk_error(Exception("other"), prompt_name="prompt")
    assert fallback.message == "other"


def test_budget_and_allowed_tools_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = RenderedPrompt(text="Tool usage")
    prompt = cast(Prompt[object], _StubPrompt(rendered))
    session = MagicMock(spec=SessionProtocol)

    result = ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session-3",
        total_cost_usd=None,
        usage=None,
        result="ok",
        structured_output=None,
    )

    monkeypatch.setattr(
        "weakincentives.adapters.claude_agent_sdk.adapter.query",
        lambda *_, **__: _stream_result(result),
    )

    adapter = ClaudeAgentSDKAdapter(
        allowed_tools=("write_file",), disallowed_tools=("bash", "write_file")
    )
    response = adapter.evaluate(
        prompt, session=session, budget=Budget(max_total_tokens=1)
    )

    assert response.text == "ok"


def test_deadline_violation_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDeadline:
        def remaining(self) -> timedelta:
            return timedelta(seconds=-1)

    rendered = RenderedPrompt(text="Deadline")
    prompt = cast(Prompt[object], _StubPrompt(rendered))
    session = MagicMock(spec=SessionProtocol)

    adapter = ClaudeAgentSDKAdapter()

    with pytest.raises(DeadlineExceededError):
        adapter.evaluate(
            prompt, session=session, deadline=cast(Deadline, FakeDeadline())
        )


def test_workspace_directory_mount(tmp_path: Path) -> None:
    nested = tmp_path / "dir"
    nested.mkdir()
    (nested / "file.txt").write_text("directory data")

    with ClaudeAgentWorkspace(mounts=(str(nested),)) as workspace:
        staged_dir = workspace.root / nested.name
        assert (staged_dir / "file.txt").read_text() == "directory data"


def test_last_result_message_returns_none() -> None:
    assert ClaudeAgentSDKAdapter._last_result_message([]) is None
