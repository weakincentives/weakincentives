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

import importlib
import sys
import types
from dataclasses import dataclass
from importlib import import_module as std_import_module
from typing import Any, cast

import pytest

from weakincentives.adapters import PromptEvaluationError, PromptResponse
from weakincentives.prompt import MarkdownSection, Prompt, Tool

try:
    from tests.adapters._test_stubs import (
        DummyGeminiCandidate,
        DummyGeminiClient,
        DummyGeminiContent,
        DummyGeminiFunctionCall,
        DummyGeminiPart,
        DummyGeminiResponse,
        GreetingParams,
        ToolParams,
        ToolPayload,
        simple_handler,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for direct invocation
    from ._test_stubs import (  # type: ignore[no-redef]
        DummyGeminiCandidate,
        DummyGeminiClient,
        DummyGeminiContent,
        DummyGeminiFunctionCall,
        DummyGeminiPart,
        DummyGeminiResponse,
        GreetingParams,
        ToolParams,
        ToolPayload,
        simple_handler,
    )
from weakincentives.events import NullEventBus

MODULE_PATH = "weakincentives.adapters.google"
PROMPT_NS = "tests/adapters/google"


def _reload_module() -> types.ModuleType:
    return importlib.reload(std_import_module(MODULE_PATH))


def _build_prompt() -> Prompt[object]:
    return Prompt(
        ns=PROMPT_NS,
        key="gemini-greeting",
        name="greeting",
        sections=[
            MarkdownSection[GreetingParams](
                title="Greeting",
                key="greeting",
                template="Say hello to ${user}.",
            )
        ],
    )


def _build_tool() -> Tool[ToolParams, ToolPayload]:
    return Tool[ToolParams, ToolPayload](
        name="uppercase_text",
        description="Uppercase the provided text.",
        handler=simple_handler,
    )


def test_json_or_str_returns_original_payload() -> None:
    module = cast(Any, _reload_module())

    payload = {"value": 1}

    assert module._json_or_str(payload) is payload


def test_build_function_calling_config_returns_any_mode() -> None:
    module = cast(Any, _reload_module())

    config = module._build_function_calling_config(
        {"type": "function", "function": {"name": "demo"}}
    )

    assert config == {"mode": "ANY", "allowed_function_names": ["demo"]}


def test_build_function_calling_config_handles_null_choice() -> None:
    module = cast(Any, _reload_module())

    config = module._build_function_calling_config(None)

    assert config == {"mode": "NONE"}


def test_build_function_calling_config_handles_none_choice() -> None:
    module = cast(Any, _reload_module())

    config = module._build_function_calling_config({"type": "none"})

    assert config == {"mode": "NONE"}


def test_google_adapter_rejects_conflicting_client_arguments() -> None:
    module = cast(Any, _reload_module())
    client = DummyGeminiClient([])

    with pytest.raises(ValueError):
        module.GoogleGeminiAdapter(
            model="gemini-test",
            client=client,
            client_factory=lambda **_: client,
        )

    with pytest.raises(ValueError):
        module.GoogleGeminiAdapter(
            model="gemini-test",
            client=client,
            client_kwargs={"api_key": "secret"},
        )


def test_create_gemini_client_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    def fail_import(name: str, package: str | None = None) -> types.ModuleType:
        if name == "google.genai":
            raise ModuleNotFoundError("No module named 'google.genai'")
        return std_import_module(name, package)

    monkeypatch.setattr(module, "import_module", fail_import)

    with pytest.raises(RuntimeError) as err:
        module.create_gemini_client()

    message = str(err.value)
    assert "uv sync --extra google-genai" in message
    assert "pip install weakincentives[google-genai]" in message


def test_create_gemini_client_returns_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())

    class DummyClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    dummy_module = cast(Any, types.ModuleType("google.genai"))
    dummy_module.Client = DummyClient

    monkeypatch.setitem(sys.modules, "google.genai", dummy_module)

    client = module.create_gemini_client(api_key="secret")

    assert isinstance(client, DummyClient)
    assert client.kwargs == {"api_key": "secret"}


def test_google_adapter_constructs_client_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())
    prompt = _build_prompt()

    response = DummyGeminiResponse(
        [
            DummyGeminiCandidate(
                DummyGeminiContent([DummyGeminiPart(text="Hello, Dana!")])
            )
        ]
    )
    client = DummyGeminiClient([response])
    captured_kwargs: list[dict[str, object]] = []

    def fake_factory(**kwargs: object) -> DummyGeminiClient:
        captured_kwargs.append(dict(kwargs))
        return client

    monkeypatch.setattr(module, "create_gemini_client", fake_factory)

    adapter = module.GoogleGeminiAdapter(
        model="gemini-test",
        client_kwargs={"api_key": "secret-key"},
    )

    result = adapter.evaluate(
        prompt,
        GreetingParams(user="Dana"),
        bus=NullEventBus(),
    )

    assert result.text == "Hello, Dana!"
    assert captured_kwargs == [{"api_key": "secret-key"}]


def test_google_adapter_processes_tool_invocation() -> None:
    module = cast(Any, _reload_module())
    prompt = Prompt(
        ns=PROMPT_NS,
        key="gemini-tool",
        name="uppercase_workflow",
        sections=[
            MarkdownSection[ToolParams](
                title="Instruction",
                key="instruction",
                template=(
                    "You must call the `uppercase_text` tool exactly once using the "
                    'payload {"query": "${query}"}. After observing the tool response, '
                    "reply with the uppercase text."
                ),
                tools=(_build_tool(),),
            )
        ],
    )

    responses = [
        DummyGeminiResponse(
            [
                DummyGeminiCandidate(
                    DummyGeminiContent(
                        [
                            DummyGeminiPart(
                                function_call=DummyGeminiFunctionCall(
                                    "uppercase_text",
                                    {"query": "sam"},
                                    call_id="tool-1",
                                )
                            )
                        ]
                    )
                )
            ]
        ),
        DummyGeminiResponse(
            [DummyGeminiCandidate(DummyGeminiContent([DummyGeminiPart(text="SAM")]))]
        ),
    ]

    client = DummyGeminiClient(responses)
    adapter = module.GoogleGeminiAdapter(
        model="gemini-test",
        client=client,
        tool_choice={
            "type": "function",
            "function": {"name": "uppercase_text"},
        },
    )

    result = adapter.evaluate(
        prompt,
        ToolParams(query="sam"),
        bus=NullEventBus(),
    )

    assert result.text == "SAM"
    assert len(result.tool_results) == 1
    tool_event = result.tool_results[0]
    assert tool_event.name == "uppercase_text"

    first_request = cast(dict[str, object], client.models.requests[0])
    assert first_request["model"] == "gemini-test"
    first_contents = cast(list[dict[str, object]], first_request["contents"])
    first_parts = cast(list[dict[str, object]], first_contents[0]["parts"])
    first_text = cast(str, first_parts[0]["text"])
    assert first_text.startswith("#")
    config_payload = cast(dict[str, object], first_request["config"])
    tools_config = cast(list[dict[str, object]], config_payload["tools"])
    first_tool = tools_config[0]
    declarations = cast(list[dict[str, object]], first_tool["function_declarations"])
    declaration = declarations[0]
    assert declaration["name"] == "uppercase_text"
    tool_config = cast(dict[str, object], config_payload["tool_config"])
    assert tool_config["function_calling_config"] == {
        "mode": "ANY",
        "allowed_function_names": ["uppercase_text"],
    }

    second_request = cast(dict[str, object], client.models.requests[1])
    second_contents = cast(list[dict[str, object]], second_request["contents"])
    function_call_parts = cast(list[dict[str, object]], second_contents[1]["parts"])
    call_payload = cast(dict[str, object], function_call_parts[0]["functionCall"])
    assert call_payload["name"] == "uppercase_text"
    response_parts = cast(list[dict[str, object]], second_contents[2]["parts"])
    response_payload = cast(dict[str, object], response_parts[0]["functionResponse"])
    assert response_payload["name"] == "uppercase_text"


def test_google_adapter_parses_structured_output() -> None:
    module = cast(Any, _reload_module())

    @dataclass(slots=True)
    class ReviewParams:
        text: str

    @dataclass(slots=True)
    class ReviewAnalysis:
        summary: str
        sentiment: str

    prompt = Prompt[ReviewAnalysis](
        ns=PROMPT_NS,
        key="gemini-structured",
        name="review",
        sections=[
            MarkdownSection[ReviewParams](
                title="Review",
                key="review",
                template="Analyse: ${text}",
            )
        ],
    )

    response = DummyGeminiResponse(
        [
            DummyGeminiCandidate(
                DummyGeminiContent([DummyGeminiPart(text='{"summary": "Short"}')])
            )
        ],
        parsed={"summary": "Short", "sentiment": "positive"},
    )

    adapter = module.GoogleGeminiAdapter(
        model="gemini-test",
        client=DummyGeminiClient([response]),
    )

    result: PromptResponse[ReviewAnalysis] = adapter.evaluate(
        prompt,
        ReviewParams(text="A nice cafe."),
        bus=NullEventBus(),
    )

    assert result.output is not None
    assert result.output.summary == "Short"
    assert result.output.sentiment == "positive"


def test_google_adapter_raises_on_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = cast(Any, _reload_module())
    prompt = _build_prompt()

    class FailingClient(DummyGeminiClient):
        def __init__(self) -> None:
            self.models = types.SimpleNamespace()

            def generate_content(**kwargs: object) -> object:
                raise RuntimeError("boom")

            self.models.generate_content = generate_content

    adapter = module.GoogleGeminiAdapter(model="gemini-test", client=FailingClient())

    with pytest.raises(PromptEvaluationError):
        adapter.evaluate(
            prompt,
            GreetingParams(user="Dana"),
            bus=NullEventBus(),
        )
