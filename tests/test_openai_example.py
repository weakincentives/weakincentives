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

"""Tests for the OpenAI example agent."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from importlib import util
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from weakincentives.adapters.core import PromptResponse, ToolCallRecord
from weakincentives.prompts import Prompt


def _load_openai_example() -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / "openai_example.py"
    spec = util.spec_from_file_location("openai_example", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load openai_example module.")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ExampleModule(Protocol):
    AgentGuidance: type[object]
    EchoToolParams: type[object]
    EchoToolResult: type[object]
    OpenAIAdapter: type[object]
    OpenAIReActSession: type[object]
    ToolResult: type[object]
    UserTurnParams: type[object]
    build_prompt: Callable[[], Prompt[object]]


example = cast(_ExampleModule, _load_openai_example())


def test_build_prompt_renders_tool_metadata() -> None:
    prompt = example.build_prompt()
    rendered = prompt.render(
        example.AgentGuidance(),
        example.UserTurnParams(content="Hello"),
    )

    tool_names = [tool.name for tool in rendered.tools]

    assert rendered.text.startswith("## Agent Guidance")
    assert tool_names == [
        "echo_text",
        "solve_math",
        "search_notes",
        "current_time",
    ]


def test_session_evaluate_routes_through_adapter(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool_result = example.ToolResult(
        message="Echoed text: HELLO",
        payload=example.EchoToolResult(text="HELLO"),
    )
    tool_record = ToolCallRecord(
        name="echo_text",
        params=example.EchoToolParams(text="hello"),
        result=tool_result,
        call_id="tool-call-1",
    )
    prompt_response = PromptResponse(
        prompt_name="echo_agent",
        text="All done.",
        output=None,
        tool_results=(tool_record,),
        provider_payload={"raw": "payload"},
    )

    captured_model: list[str] = []
    captured_kwargs: list[dict[str, object]] = []
    captured_calls: list[tuple[Prompt[object], tuple[object, ...], bool]] = []

    class StubAdapter:
        def __init__(self, *, model: str, **kwargs: object) -> None:
            captured_model.append(model)
            captured_kwargs.append(dict(kwargs))

        def evaluate(
            self, prompt: Prompt[object], *params: object, parse_output: bool = True
        ) -> PromptResponse:
            captured_calls.append((prompt, params, parse_output))
            return prompt_response

    monkeypatch.setattr(example, "OpenAIAdapter", StubAdapter)

    session = example.OpenAIReActSession(model="gpt-mock")
    result = session.evaluate("Use the echo tool.")
    output = capsys.readouterr().out.splitlines()

    assert result == "All done."
    assert captured_model[0] == "gpt-mock"
    assert captured_kwargs[0] == {}
    assert len(captured_calls) == 1

    call_prompt, call_params, parse_output = captured_calls[0]
    assert isinstance(call_prompt, Prompt)
    assert parse_output is True
    assert len(call_params) == 1
    assert isinstance(call_params[0], example.UserTurnParams)

    assert output[0].startswith("[tool 1] echo_text called with")
    assert "Echoed text: HELLO" in " ".join(output)
    assert any("payload" in line for line in output)


def test_session_evaluate_serializes_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_response = PromptResponse(
        prompt_name="echo_agent",
        text=None,
        output=example.EchoToolResult(text="structured"),
        tool_results=(),
        provider_payload=None,
    )

    class StubAdapter:
        def __init__(self, *, model: str) -> None:
            self.model = model

        def evaluate(
            self, prompt: Prompt[object], *params: object, parse_output: bool = True
        ) -> PromptResponse:
            return prompt_response

    monkeypatch.setattr(example, "OpenAIAdapter", StubAdapter)

    session = example.OpenAIReActSession(model="gpt-structured")
    result = session.evaluate("Return structured output.")
    serialized = json.loads(result)

    assert serialized == {"text": "structured"}
