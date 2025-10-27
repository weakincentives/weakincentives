"""Minimal OpenAI agent powered by the prompt toolkit and tools extensions."""

from __future__ import annotations

import json
import os
from dataclasses import MISSING, dataclass, fields
from typing import Any

from weakincentives.adapters import create_openai_client
from weakincentives.prompts import Prompt, TextSection, Tool, ToolResult


@dataclass
class EchoToolParams:
    text: str


@dataclass
class EchoToolResult:
    text: str


@dataclass
class AgentGuidance:
    primary_tool: str = "echo_text"


@dataclass
class ToolOverview:
    primary_tool: str = "echo_text"


def echo_text_handler(params: EchoToolParams) -> ToolResult[EchoToolResult]:
    result = EchoToolResult(text=params.text.upper())
    return ToolResult(message=f"Echoed text: {result.text}", payload=result)


def _schema_for_params(params_type: type[Any]) -> dict[str, Any]:
    """Naive JSON schema generator for the tool parameter dataclass."""

    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in fields(params_type):
        properties[field.name] = {"type": "string"}
        if field.default is MISSING and field.default_factory is MISSING:
            required.append(field.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def tool_to_openai_spec(tool: Tool[Any, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": _schema_for_params(tool.params_type),
        },
    }


def build_prompt() -> Prompt:
    echo_tool = Tool[EchoToolParams, EchoToolResult](
        name="echo_text",
        description="Return the provided text in uppercase characters.",
        handler=echo_text_handler,
    )
    tool_overview = TextSection[ToolOverview](
        title="Available Tools",
        body="Expose ${primary_tool} to turn arbitrary input into uppercase text.",
        tools=[echo_tool],
        defaults=ToolOverview(),
    )
    guidance_section = TextSection[AgentGuidance](
        title="Agent Guidance",
        body=(
            "You are a minimal demo agent. Call ${primary_tool} whenever the user "
            "wants text transformed."
        ),
        defaults=AgentGuidance(),
        children=[tool_overview],
    )
    return Prompt(name="echo_agent", sections=[guidance_section])


class BasicOpenAIAgent:
    """Single-turn OpenAI agent that wires Prompt + tool definitions."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._guidance = AgentGuidance()
        self._tool_overview = ToolOverview(primary_tool=self._guidance.primary_tool)
        self.prompt = build_prompt()
        self.model = model
        self._system_prompt = self.prompt.render(self._guidance, self._tool_overview)
        self._tools = self.prompt.tools(self._guidance, self._tool_overview)
        self._tool_specs = [tool_to_openai_spec(tool) for tool in self._tools]
        self._tool_registry = {
            tool.name: tool for tool in self._tools if tool.handler is not None
        }
        self._client = create_openai_client()

    def run(self, user_message: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message},
        ]
        first_response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self._tool_specs,
            tool_choice="auto",
        )
        choice = first_response.choices[0]
        message = choice.message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return message.content or ""

        tool_call = tool_calls[0]
        arguments = self._parse_arguments(tool_call.function.arguments)
        tool_result = self._call_tool(tool_call.function.name, arguments)

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": tool_calls,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result.message,
            }
        )

        follow_up = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return follow_up.choices[0].message.content or ""

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

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult[Any]:
        tool = self._tool_registry.get(name)
        if tool is None or tool.handler is None:
            raise RuntimeError(f"No handler registered for tool '{name}'.")
        params = tool.params_type(**arguments)
        return tool.handler(params)


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("Set OPENAI_API_KEY before running this example.")
    agent = BasicOpenAIAgent()
    answer = agent.run("Please shout 'hello world'.")
    print(answer)


if __name__ == "__main__":
    main()
