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

from weakincentives.adapters import OpenAIAdapter, create_openai_client
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult
from weakincentives.serde import dump, parse, schema

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


@dataclass
class ReasoningStep:
    thought: str
    action: str | None
    action_input: dict[str, Any] | None
    observation: str | None


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
    echo_tool, math_tool, notes_tool, time_tool = _build_tools()
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
        tools=[echo_tool, math_tool, notes_tool, time_tool],
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
    )
    user_turn_section = TextSection[UserTurnParams](
        title="User Turn",
        body=(
            "The user has provided a new instruction. Use it to decide whether to "
            "call tools or respond directly.\n\nInstruction:\n${content}"
        ),
    )
    return Prompt(name="echo_agent", sections=[guidance_section, user_turn_section])


def tool_to_openai_spec(tool: Tool[Any, Any]) -> dict[str, Any]:
    parameters_schema = schema(tool.params_type, extra="forbid")
    parameters_schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema,
        },
    }


def build_system_prompt() -> Prompt:
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
    )
    return Prompt(name="echo_agent", sections=[guidance_section])


def build_user_turn_prompt() -> Prompt:
    user_turn_section = TextSection[UserTurnParams](
        title="User Turn",
        body=(
            "The user has provided a new instruction. Use it to decide whether to "
            "call tools or respond directly.\n\nInstruction:\n${content}"
        ),
    )
    return Prompt(name="echo_user_turn", sections=[user_turn_section])


class BasicOpenAIAgent:
    """Multi-turn OpenAI agent that wires Prompt + tool definitions."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._guidance = AgentGuidance()
        self.system_prompt_template = build_system_prompt()
        self.user_prompt_template = build_user_turn_prompt()
        self.model = model
        rendered_system_prompt = self.system_prompt_template.render(self._guidance)
        self._system_prompt = rendered_system_prompt.text
        self._tools = rendered_system_prompt.tools
        self._tool_specs = [tool_to_openai_spec(tool) for tool in self._tools]
        self._tool_registry = {
            tool.name: tool for tool in self._tools if tool.handler is not None
        }
        self._client = create_openai_client()
        self._messages: list[dict[str, Any]] = []
        self._last_trace: list[ReasoningStep] = []
        self.reset()

    def run(self, user_message: str, *, max_turns: int = 8) -> str:
        self.reset()
        return self.send_user_message(user_message, max_turns=max_turns)

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._last_trace = []

    def send_user_message(self, user_message: str, *, max_turns: int = 8) -> str:
        self._last_trace = []
        user_content = self.user_prompt_template.render(
            UserTurnParams(content=user_message)
        ).text
        self._messages.append({"role": "user", "content": user_content})
        return self._advance_conversation(max_turns=max_turns)

    def _advance_conversation(self, max_turns: int) -> str:
        for _ in range(max_turns):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                tools=self._tool_specs,
                tool_choice="auto",
            )
            choice = response.choices[0]
            message = choice.message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                final_content = message.content or ""
                self._messages.append({"role": "assistant", "content": final_content})
                if final_content:
                    self._last_trace.append(
                        ReasoningStep(
                            thought=final_content,
                            action=None,
                            action_input=None,
                            observation=None,
                        )
                    )
                return final_content

            self._messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                arguments = self._parse_arguments(tool_call.function.arguments)
                tool_result, serialized_params = self._call_tool(
                    tool_call.function.name, arguments
                )
                tool_payload = dump(tool_result.payload, exclude_none=True)
                tool_content = {
                    "message": tool_result.message,
                    "payload": tool_payload,
                }
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_content),
                    }
                )
                self._last_trace.append(
                    ReasoningStep(
                        thought=message.content or "",
                        action=tool_call.function.name,
                        action_input=serialized_params,
                        observation=tool_result.message,
                    )
                )

        raise RuntimeError("Agent stopped after reaching the maximum number of turns.")

    @staticmethod
    def _parse_arguments(arguments_json: str | None) -> dict[str, Any]:
        if not arguments_json:
            return {}
        try:
            parsed = json.loads(arguments_json)
        except json.JSONDecodeError as error:  # pragma: no cover - defensive
            raise RuntimeError("Failed to decode tool call arguments.") from error
        if not isinstance(parsed, dict):
            raise RuntimeError("Tool call arguments must be a JSON object.")
        return parsed

    def _call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[ToolResult[Any], dict[str, Any]]:
        tool = self._tool_registry.get(name)
        if tool is None or tool.handler is None:
            raise RuntimeError(f"No handler registered for tool '{name}'.")
        params = parse(tool.params_type, arguments, extra="forbid")
        serialized_params = dump(params, exclude_none=True)
        return tool.handler(params), serialized_params

    @property
    def last_trace(self) -> list[ReasoningStep]:
        """Return a shallow copy of the most recent ReAct trace."""

        return list(self._last_trace)


class OpenAIReActSession:
    """Interactive session powered by the OpenAIAdapter."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        client = create_openai_client()
        self._adapter = OpenAIAdapter(client, model=model)
        self._prompt = build_prompt()

    def evaluate(self, user_message: str) -> str:
        response = self._adapter.evaluate(
            self._prompt,
            UserTurnParams(content=user_message),
        )

        for index, record in enumerate(response.tool_results, start=1):
            serialized_params = dump(record.params, exclude_none=True)
            payload = dump(record.result.payload, exclude_none=True)
            print(
                f"[tool {index}] {record.name} called with {serialized_params}\n"
                f"           → {record.result.message}"
            )
            if payload:
                print(f"           payload: {payload}")

        if response.output is not None:
            rendered_output = dump(response.output, exclude_none=True)
            return json.dumps(rendered_output)
        if response.text:
            return response.text
        return "(no response from assistant)"


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    session = OpenAIReActSession(model=model)
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
