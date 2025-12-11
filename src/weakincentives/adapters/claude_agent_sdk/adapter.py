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

"""Claude Agent SDK adapter implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import timedelta
from typing import Any, TypeVar, cast, override

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import Message, ResultMessage

from ...budget import Budget, BudgetTracker
from ...deadlines import Deadline
from ...prompt import SectionVisibility
from ...prompt.errors import SectionPath
from ...prompt.prompt import Prompt
from ...prompt.rendering import RenderedPrompt
from ...runtime.logging import get_logger
from ...serde.parse import parse
from ...serde.schema import schema
from ..core import PromptResponse, ProviderAdapter, SessionProtocol
from ..shared import ThrottleError
from ._errors import normalize_sdk_error
from .config import ClaudeAgentSDKClientConfig, ClaudeAgentSDKModelConfig

OutputT = TypeVar("OutputT")

_logger = get_logger(__name__)


class ClaudeAgentSDKAdapter(ProviderAdapter[OutputT]):
    """Adapter using the claude-agent-sdk package."""

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        *,
        model: str = "claude-sonnet-4-5-20250929",
        client_config: ClaudeAgentSDKClientConfig | None = None,
        model_config: ClaudeAgentSDKModelConfig | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        disallowed_tools: tuple[str, ...] = (),
    ) -> None:
        self._model = model
        self._client_config = client_config or ClaudeAgentSDKClientConfig()
        self._model_config = model_config or ClaudeAgentSDKModelConfig(model=model)
        self._allowed_tools = allowed_tools
        self._disallowed_tools = disallowed_tools

        self._logger = _logger.bind(adapter="claude_agent_sdk")

    @override
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[OutputT]:
        if budget is not None and budget_tracker is None:
            budget_tracker = BudgetTracker(budget)

        return asyncio.run(
            self._evaluate_async(
                prompt,
                session=session,
                deadline=deadline,
                visibility_overrides=visibility_overrides,
                budget_tracker=budget_tracker,
            )
        )

    async def _evaluate_async(
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None,
        budget_tracker: BudgetTracker | None,
    ) -> PromptResponse[OutputT]:
        prompt_name = prompt.name or "<anonymous>"

        if deadline is not None and deadline.remaining() <= timedelta():
            from ...tools.errors import DeadlineExceededError

            raise DeadlineExceededError

        rendered = prompt.render(visibility_overrides=visibility_overrides)
        options = self._build_options(rendered)

        self._logger.info(
            "claude_agent_sdk.evaluate.start",
            extra={
                "event": "sdk.evaluate.start",
                "prompt_name": prompt.name,
                "model": self._model,
                "tool_count": len(rendered.tools),
                "has_structured_output": rendered.output_type is not None,
            },
        )

        try:
            messages = await self._run_query(rendered.text, options)
        except Exception as exc:  # pragma: no cover - passthrough
            normalized = normalize_sdk_error(exc, prompt_name=prompt_name)
            if isinstance(normalized, ThrottleError):
                raise normalized from exc
            raise normalized from exc

        result_message = self._last_result_message(messages)
        text = result_message.result if result_message else None
        output: OutputT | None = None

        if rendered.output_type and result_message is not None:
            structured = getattr(result_message, "structured_output", None)
            if structured is not None:
                output = cast(
                    OutputT, parse(rendered.output_type, structured, extra="ignore")
                )

        self._logger.info(
            "claude_agent_sdk.evaluate.complete",
            extra={
                "event": "sdk.evaluate.complete",
                "prompt_name": prompt_name,
                "duration_ms": result_message.duration_ms if result_message else None,
                "input_tokens": (result_message.usage or {}).get("input_tokens")
                if result_message
                else None,
                "output_tokens": (result_message.usage or {}).get("output_tokens")
                if result_message
                else None,
                "total_cost_usd": result_message.total_cost_usd
                if result_message
                else None,
                "session_id": result_message.session_id if result_message else None,
                "num_turns": result_message.num_turns if result_message else None,
            },
        )

        return PromptResponse(
            prompt_name=prompt_name,
            text=text,
            output=output,
        )

    @staticmethod
    async def _run_query(
        prompt_text: str, options: ClaudeAgentOptions
    ) -> list[Message]:
        return [message async for message in query(prompt=prompt_text, options=options)]

    def _build_options(self, rendered: RenderedPrompt[OutputT]) -> ClaudeAgentOptions:
        output_format = self._build_output_format(rendered.output_type)

        options = ClaudeAgentOptions(
            model=self._model_config.model,
            permission_mode=self._client_config.permission_mode,
            cwd=self._client_config.cwd,
            add_dirs=list(self._client_config.add_dirs),
            env=dict(self._client_config.env) if self._client_config.env else {},
            hooks=None,
            disallowed_tools=list(self._disallowed_tools),
            include_partial_messages=self._client_config.include_partial_messages,
            max_turns=self._client_config.max_turns,
            setting_sources=list(self._client_config.setting_sources)
            if self._client_config.setting_sources
            else None,
            output_format=output_format,
        )

        if self._allowed_tools is not None:
            options.disallowed_tools = [
                tool
                for tool in options.disallowed_tools
                if tool not in self._allowed_tools
            ]
        return options

    @staticmethod
    def _build_output_format(
        output_type: type[Any] | None,
    ) -> dict[str, Any] | None:
        if output_type is None:
            return None
        return {
            "type": "json_schema",
            "schema": schema(output_type),
        }

    @staticmethod
    def _last_result_message(messages: list[Message]) -> ResultMessage | None:
        for message in reversed(messages):
            if isinstance(message, ResultMessage):
                return message
        return None
