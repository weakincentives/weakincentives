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

"""Demonstration OpenAI agent with a lightweight ReAct loop and tool suite."""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from weakincentives.adapters import OpenAIAdapter
from weakincentives.events import EventBus, InProcessEventBus, ToolInvoked
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult
from weakincentives.serde import dump
from weakincentives.session import (
    DataEvent,
    Session,
    ToolData,
    select_all,
    select_latest,
)

try:  # pragma: no cover - optional dependency in stdlib for 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - defensive fallback
    ZoneInfo = None  # type: ignore[assignment]


@dataclass
class EchoToolParams:
    text: str


@dataclass
class EchoToolResult:
    text: str


@dataclass
class MathToolParams:
    expression: str


@dataclass
class MathToolResult:
    value: float


@dataclass
class SearchNotesToolParams:
    query: str


@dataclass
class SearchNotesToolResult:
    matches: list[str]


@dataclass
class CurrentTimeToolParams:
    timezone: str | None = None


@dataclass
class CurrentTimeToolResult:
    timestamp: str


@dataclass
class AgentGuidance:
    primary_tool: str = "echo_text"


@dataclass
class UserTurnParams:
    content: str


@dataclass(slots=True, frozen=True)
class ToolCallLog:
    """Recorded tool invocation captured by the session."""

    name: str
    prompt_name: str
    message: str
    payload: dict[str, Any]
    call_id: str | None


KNOWLEDGE_BASE: dict[str, str] = {
    "expense policy": textwrap.dedent(
        """
        Reimbursements are processed every Friday.
        Submit receipts through the finance portal before 5pm local time for same-week
        processing.
        """
    ).strip(),
    "travel guidelines": textwrap.dedent(
        """
        Book flights at least 14 days in advance when possible.
        Use the corporate travel dashboard to compare options and capture approval notes.
        """
    ).strip(),
    "product roadmap": textwrap.dedent(
        """
        We ship quarterly updates focused on agent safety, prompt tooling, and audit
        workflows.
        Review the planning doc to see the upcoming milestones.
        """
    ).strip(),
}


def echo_text_handler(params: EchoToolParams) -> ToolResult[EchoToolResult]:
    result = EchoToolResult(text=params.text.upper())
    return ToolResult(message=f"Echoed text: {result.text}", payload=result)


def _insecure_eval_math(expression: str) -> float:
    """Evaluate math expressions using plain eval for demo purposes."""

    return float(eval(expression))


def solve_math_handler(params: MathToolParams) -> ToolResult[MathToolResult]:
    try:
        value = _insecure_eval_math(params.expression)
    except Exception as error:
        message = f"Unable to evaluate expression: {error}"
        return ToolResult(message=message, payload=MathToolResult(value=float("nan")))
    return ToolResult(
        message=f"Computed result is {value}.",
        payload=MathToolResult(value=value),
    )


def search_notes_handler(
    params: SearchNotesToolParams,
) -> ToolResult[SearchNotesToolResult]:
    matches: list[str] = []
    lowered_query = params.query.lower()
    for title, content in KNOWLEDGE_BASE.items():
        if lowered_query in title or lowered_query in content.lower():
            matches.append(f"{title.title()}: {content}")
    if not matches:
        matches.append(
            "No matches found. Try related keywords like 'expense policy' or 'roadmap'."
        )
    return ToolResult(
        message="Knowledge search complete.",
        payload=SearchNotesToolResult(matches=matches),
    )


def current_time_handler(
    params: CurrentTimeToolParams,
) -> ToolResult[CurrentTimeToolResult]:
    timezone_name = params.timezone or "UTC"
    tz = None
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:  # pragma: no cover - timezone lookup may fail
            tz = ZoneInfo("UTC")
            timezone_name = "UTC"
    now = datetime.now(tz)
    formatted = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    message = f"Current time in {timezone_name} is {formatted}."
    return ToolResult(
        message=message,
        payload=CurrentTimeToolResult(timestamp=formatted),
    )


def _build_tools() -> tuple[Tool[Any, Any], ...]:
    echo_tool = Tool[EchoToolParams, EchoToolResult](
        name="echo_text",
        description="Return the provided text in uppercase characters.",
        handler=echo_text_handler,
    )
    math_tool = Tool[MathToolParams, MathToolResult](
        name="solve_math",
        description="Safely evaluate arithmetic expressions, supports sqrt/log/trig.",
        handler=solve_math_handler,
    )
    notes_tool = Tool[SearchNotesToolParams, SearchNotesToolResult](
        name="search_notes",
        description="Search lightweight company notes for relevant guidance.",
        handler=search_notes_handler,
    )
    time_tool = Tool[CurrentTimeToolParams, CurrentTimeToolResult](
        name="current_time",
        description="Return the current timestamp, optionally in a specific timezone.",
        handler=current_time_handler,
    )
    return (echo_tool, math_tool, notes_tool, time_tool)


def build_prompt() -> Prompt:
    tools = _build_tools()
    tool_overview = TextSection[AgentGuidance](
        title="Available Tools",
        body=textwrap.dedent(
            """
            You can interact with four tools during the conversation:
            - ${primary_tool}: uppercases any provided text.
            - solve_math: evaluate structured math expressions.
            - search_notes: surface stored company knowledge.
            - current_time: fetch the current timestamp in a requested timezone.
            """
        ).strip(),
        tools=tools,
        key="available-tools",
    )
    guidance_section = TextSection[AgentGuidance](
        title="Agent Guidance",
        body=(
            "You are a demo assistant that follows a Reason + Act pattern. When a user "
            "asks for help, think through the request, decide whether a tool is "
            "needed, call it, observe the result, and then reply with a concise "
            "answer grounded in those observations."
        ),
        defaults=AgentGuidance(),
        children=[tool_overview],
        key="agent-guidance",
    )
    user_turn_section = TextSection[UserTurnParams](
        title="User Turn",
        body=(
            "The user has provided a new instruction. Use it to decide whether to "
            "call tools or respond directly.\n\nInstruction:\n${content}"
        ),
        key="user-turn",
    )
    return Prompt(
        ns="examples/openai",
        key="example-echo-agent",
        name="echo_agent",
        sections=[guidance_section, user_turn_section],
    )


class OpenAIReActSession:
    """Interactive session powered by the OpenAIAdapter."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        bus: EventBus | None = None,
    ) -> None:
        self._bus = bus or InProcessEventBus()
        self._session = Session(bus=self._bus)
        self._register_tool_history()
        self._bus.subscribe(ToolInvoked, self._display_tool_event)
        self._adapter = OpenAIAdapter(model=model)
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
        latest = select_latest(self._session, ToolCallLog)
        if latest is not None:
            count = len(select_all(self._session, ToolCallLog))
            print(f"       (session recorded this call as #{count})")

    def evaluate(self, user_message: str) -> str:
        response = self._adapter.evaluate(
            self._prompt,
            UserTurnParams(content=user_message),
            bus=self._bus,
        )

        if response.output is not None:
            rendered_output = dump(response.output, exclude_none=True)
            return json.dumps(rendered_output)
        if response.text:
            return response.text
        return "(no response from assistant)"

    def render_tool_history(self) -> str:
        """Return a formatted history of tool calls captured by the session."""

        history = select_all(self._session, ToolCallLog)
        if not history:
            return "No tool calls recorded yet."

        lines: list[str] = []
        for index, record in enumerate(history, start=1):
            lines.append(
                f"{index}. {record.name} ({record.prompt_name}) → {record.message}"
            )
            if record.call_id:
                lines.append(f"   call_id: {record.call_id}")
            if record.payload:
                payload_dump = json.dumps(record.payload, ensure_ascii=False)
                lines.append(f"   payload: {payload_dump}")
        return "\n".join(lines)

    def _register_tool_history(self) -> None:
        """Register reducers that project tool calls into a shared log slice."""

        for result_type in (
            EchoToolResult,
            MathToolResult,
            SearchNotesToolResult,
            CurrentTimeToolResult,
        ):
            self._session.register_reducer(
                result_type,
                self._record_tool_call,
                slice_type=ToolCallLog,
            )

    def _record_tool_call(
        self,
        slice_values: tuple[ToolCallLog, ...],
        event: DataEvent,
    ) -> tuple[ToolCallLog, ...]:
        if not isinstance(event, ToolData):
            return slice_values

        payload = dump(event.value, exclude_none=True)
        record = ToolCallLog(
            name=event.source.name,
            prompt_name=event.source.prompt_name,
            message=event.source.result.message,
            payload=payload,
            call_id=event.source.call_id,
        )
        return slice_values + (record,)


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    session = OpenAIReActSession(model=model)
    print("Type 'exit' or 'quit' to stop the conversation.")
    print("Type 'history' to display the recorded tool call summary.")
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
        if prompt.lower() == "history":
            print(session.render_tool_history())
            continue
        answer = session.evaluate(prompt)
        print(f"Agent: {answer}")
