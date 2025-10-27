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
class UserTurnParams:
    content: str


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


def build_system_prompt() -> Prompt:
    echo_tool = Tool[EchoToolParams, EchoToolResult](
        name="echo_text",
        description="Return the provided text in uppercase characters.",
        handler=echo_text_handler,
    )
    tool_overview = TextSection[AgentGuidance](
        title="Available Tools",
        body="Expose ${primary_tool} to turn arbitrary input into uppercase text.",
        tools=[echo_tool],
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
        self._system_prompt = self.system_prompt_template.render(self._guidance)
        self._tools = self.system_prompt_template.tools(self._guidance)
        self._tool_specs = [tool_to_openai_spec(tool) for tool in self._tools]
        self._tool_registry = {
            tool.name: tool for tool in self._tools if tool.handler is not None
        }
        self._client = create_openai_client()
        self._messages: list[dict[str, Any]] = []
        self.reset()

    def run(self, user_message: str, *, max_turns: int = 8) -> str:
        self.reset()
        return self.send_user_message(user_message, max_turns=max_turns)

    def reset(self) -> None:
        self._messages = [{"role": "system", "content": self._system_prompt}]

    def send_user_message(self, user_message: str, *, max_turns: int = 8) -> str:
        user_content = self.user_prompt_template.render(
            UserTurnParams(content=user_message)
        )
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
                tool_result = self._call_tool(tool_call.function.name, arguments)
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result.message,
                    }
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
        answer = agent.send_user_message(prompt)
        print(f"Agent: {answer}")


if __name__ == "__main__":
    main()
