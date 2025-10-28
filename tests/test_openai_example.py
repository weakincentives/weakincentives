"""Tests for the OpenAI example agent."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol, cast

import pytest


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
    BasicOpenAIAgent: type[Any]
    UserTurnParams: type[Any]
    create_openai_client: Callable[..., Any]


example = cast(_ExampleModule, _load_openai_example())


class FakeClient:
    """Stub OpenAI client that replays pre-defined responses."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs: object) -> SimpleNamespace:
        if not self._responses:
            raise AssertionError("No more fake responses queued.")
        snapshot: dict[str, object] = dict(kwargs)
        messages = snapshot.get("messages")
        if isinstance(messages, list):
            snapshot["messages"] = [dict(message) for message in messages]
        self.calls.append(snapshot)
        return self._responses.pop(0)


def _response_with_tool_call(*, arguments: dict[str, object]) -> SimpleNamespace:
    tool_call = SimpleNamespace(
        id="tool-call-1",
        function=SimpleNamespace(
            name="echo_text",
            arguments=json.dumps(arguments),
        ),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _final_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def test_agent_run_handles_multiple_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _response_with_tool_call(arguments={"text": "hello"}),
        _final_response("All done."),
    ]
    fake_client = FakeClient(responses)
    monkeypatch.setattr(example, "create_openai_client", lambda: fake_client)

    agent = example.BasicOpenAIAgent()
    expected_user_prompt = agent.user_prompt_template.render(
        example.UserTurnParams(content="Use the echo tool.")
    )
    result = agent.run("Use the echo tool.")

    assert result == "All done."
    assert len(fake_client.calls) == 2
    first_messages = fake_client.calls[0]["messages"]
    assert isinstance(first_messages, list)
    assert [message["role"] for message in first_messages] == ["system", "user"]
    assert first_messages[1]["content"] == expected_user_prompt
    second_messages = fake_client.calls[1]["messages"]
    assert isinstance(second_messages, list)
    last_message = second_messages[-1]
    assert last_message["role"] == "tool"
    assert last_message["tool_call_id"] == "tool-call-1"
    content = json.loads(last_message["content"])
    assert content == {
        "message": "Echoed text: HELLO",
        "payload": {"text": "HELLO"},
    }


def test_agent_run_respects_max_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _response_with_tool_call(arguments={"text": "oops"}),
    ] * 2
    fake_client = FakeClient(responses)
    monkeypatch.setattr(example, "create_openai_client", lambda: fake_client)

    agent = example.BasicOpenAIAgent()
    with pytest.raises(RuntimeError, match="maximum number of turns"):
        agent.run("Loop forever.", max_turns=1)


def test_agent_supports_multiple_user_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        _final_response("First answer."),
        _final_response("Second answer."),
    ]
    fake_client = FakeClient(responses)
    monkeypatch.setattr(example, "create_openai_client", lambda: fake_client)

    agent = example.BasicOpenAIAgent()

    first_user_prompt = agent.user_prompt_template.render(
        example.UserTurnParams(content="Hello")
    )
    first_reply = agent.send_user_message("Hello")
    second_user_prompt = agent.user_prompt_template.render(
        example.UserTurnParams(content="Thanks")
    )
    second_reply = agent.send_user_message("Thanks")

    assert first_reply == "First answer."
    assert second_reply == "Second answer."

    assert len(fake_client.calls) == 2
    first_messages = fake_client.calls[0]["messages"]
    second_messages = fake_client.calls[1]["messages"]

    assert [message["role"] for message in first_messages] == ["system", "user"]
    assert first_messages[1]["content"] == first_user_prompt

    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_messages[2]["content"] == "First answer."
    assert second_messages[3]["content"] == second_user_prompt
