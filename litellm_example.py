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

"""Minimal LiteLLM example demonstrating tool usage and structured output."""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from typing import Any

from weakincentives.adapters import LiteLLMAdapter
from weakincentives.events import EventBus, InProcessEventBus, ToolInvoked
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult
from weakincentives.serde import dump


@dataclass
class LookupParams:
    topic: str


@dataclass
class LookupResult:
    note: str


@dataclass
class AgentGuidance:
    objective: str = "Produce concise answers using local knowledge."


@dataclass
class UserTurnParams:
    question: str


@dataclass
class AgentReply:
    answer: str
    citations: list[str]


NOTES: dict[str, str] = {
    "pricing": "Pricing updates ship on the first Monday of each quarter.",
    "roadmap": "Roadmap themes: prompt safety, typed tools, audit workflows.",
    "releases": "Release cadence is monthly; hotfix windows stay open mid-cycle.",
}


def lookup_note(params: LookupParams) -> ToolResult[LookupResult]:
    note = NOTES.get(params.topic.lower())
    if note is None:
        message = "No note found. Try topics like 'pricing' or 'roadmap'."
        return ToolResult(message=message, payload=LookupResult(note=""))
    return ToolResult(
        message=f"Located note for {params.topic}.",
        payload=LookupResult(note=note),
    )


def build_prompt() -> Prompt[AgentReply]:
    note_tool = Tool[LookupParams, LookupResult](
        name="lookup_note",
        description="Retrieve a short note for well-known company topics.",
        handler=lookup_note,
    )
    guidance = TextSection[AgentGuidance](
        title="Assistant Guidance",
        body=textwrap.dedent(
            """
            You are a concise assistant that relies on the `lookup_note` tool for
            company knowledge. When you answer questions:
            - Think about whether a lookup is required.
            - Call the tool when you need a fresh citation.
            - Provide a JSON reply with your answer and cite any notes you used.
            """
        ).strip(),
        defaults=AgentGuidance(),
        tools=[note_tool],
    )
    user_turn = TextSection[UserTurnParams](
        title="User Question",
        body="The user asked: ${question}",
    )
    return Prompt[AgentReply](
        key="litellm-demo-session",
        name="litellm_demo",
        sections=[guidance, user_turn],
    )


class LiteLLMReActSession:
    """Interactive session powered by the LiteLLMAdapter."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        bus: EventBus | None = None,
        completion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._bus = bus or InProcessEventBus()
        self._bus.subscribe(ToolInvoked, self._display_tool_event)
        adapter_kwargs: dict[str, Any] = {"model": model}
        if completion_kwargs:
            adapter_kwargs["completion_kwargs"] = completion_kwargs
        self._adapter = LiteLLMAdapter(**adapter_kwargs)
        self._prompt = build_prompt()

    def _display_tool_event(self, event: ToolInvoked) -> None:
        serialized_params = dump(event.params, exclude_none=True)
        payload = dump(event.result.payload, exclude_none=True)
        print(
            f"[tool] {event.name} called with {serialized_params}\n"
            f"       → {event.result.message}"
        )
        if payload:
            print(f"       payload: {payload}")

    def evaluate(self, question: str) -> str:
        response = self._adapter.evaluate(
            self._prompt,
            UserTurnParams(question=question),
            bus=self._bus,
        )

        if response.output is not None:
            rendered_output = dump(response.output, exclude_none=True)
            return json.dumps(rendered_output)
        if response.text:
            return response.text
        return "(no response from assistant)"


if __name__ == "__main__":
    api_key = os.getenv("LITELLM_API_KEY")
    if api_key is None:
        raise SystemExit("Set LITELLM_API_KEY before running this example.")

    model = os.getenv("LITELLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LITELLM_BASE_URL")
    completion_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        completion_kwargs["api_base"] = base_url

    session = LiteLLMReActSession(model=model, completion_kwargs=completion_kwargs)
    print("Type 'exit' or 'quit' to stop the conversation.")
    while True:
        try:
            prompt = input("You: ").strip()
        except EOFError:  # pragma: no cover - interactive convenience
            break
        if not prompt:
            print("Goodbye.")
            break
        if prompt.lower() in {"exit", "quit"}:
            break
        answer = session.evaluate(prompt)
        print(f"Agent: {answer}")
