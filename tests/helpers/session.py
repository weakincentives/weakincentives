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

"""Shared fixtures for session tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from tests.helpers.adapters import GENERIC_ADAPTER_NAME
from weakincentives.adapters.core import PromptResponse
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)

DEFAULT_SESSION_ID = uuid4()


@dataclass(slots=True, frozen=True)
class ExampleParams:
    value: int


@dataclass(slots=True, frozen=True)
class ExamplePayload:
    value: int


@dataclass(slots=True, frozen=True)
class ExampleOutput:
    text: str


def make_tool_event(value: int) -> ToolInvoked:
    payload = ExamplePayload(value=value)
    tool_result = cast(
        ToolResult[object],
        ToolResult(message="ok", value=payload),
    )
    rendered_output = tool_result.render()
    return ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=value),
        success=tool_result.success,
        message=tool_result.message,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=rendered_output,
    )


def make_prompt_event(output: object) -> PromptExecuted:
    return PromptExecuted(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
    )


def make_prompt_rendered(
    *,
    rendered_prompt: str,
    session_id: UUID | None = None,
    params_value: int = 1,
) -> PromptRendered:
    return PromptRendered(
        prompt_ns="example",
        prompt_key="example",
        prompt_name="Example",
        adapter=GENERIC_ADAPTER_NAME,
        session_id=session_id,
        render_inputs=(ExampleParams(value=params_value),),
        rendered_prompt=rendered_prompt,
        created_at=datetime.now(UTC),
    )


__all__ = [
    "DEFAULT_SESSION_ID",
    "ExampleOutput",
    "ExampleParams",
    "ExamplePayload",
    "make_prompt_event",
    "make_prompt_rendered",
    "make_tool_event",
]
