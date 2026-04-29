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

"""Test adapter that replays scripted responses.

:class:`NoopAdapter` is a deterministic, side-effect-free
:class:`weakincentives.core.ProviderAdapter`. It returns whatever the
caller scripted, indexed by call number. Useful for driving
:class:`weakincentives.runtime.AgentLoop` in tests without a real provider.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import final

from weakincentives.core import (
    Deadline,
    PromptResponse,
    Session,
    Tool,
    ToolCall,
    Usage,
)
from weakincentives.core.prompt import RenderedPrompt

__all__ = [
    "NoopAdapter",
    "ScriptedResponse",
]


@final
@dataclass(frozen=True, slots=True)
class ScriptedResponse:
    """Single scripted reply for :class:`NoopAdapter`."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None

    def to_response(self) -> PromptResponse:
        return PromptResponse(
            text=self.text,
            tool_calls=self.tool_calls,
            usage=self.usage,
            finish_reason=self.finish_reason,
        )


@final
class NoopAdapter:
    """:class:`weakincentives.core.ProviderAdapter` that returns scripted replies.

    The adapter walks ``responses`` once, returning each in order. After
    the script is exhausted it returns an empty terminal response with
    ``finish_reason="stop"``.
    """

    def __init__(self, responses: Sequence[ScriptedResponse] = ()) -> None:
        super().__init__()
        self._responses = tuple(responses)
        self._lock = threading.Lock()
        self._index = 0
        self._calls = 0

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def evaluate(
        self,
        rendered: object,
        session: object,
        *,
        deadline: Deadline | None = None,
    ) -> PromptResponse:
        del deadline
        if not isinstance(rendered, RenderedPrompt):
            raise TypeError(
                f"NoopAdapter expected RenderedPrompt, got {type(rendered).__name__}"
            )
        if not isinstance(session, Session):
            raise TypeError(
                f"NoopAdapter expected Session, got {type(session).__name__}"
            )
        with self._lock:
            self._calls += 1
            if self._index >= len(self._responses):
                return PromptResponse(text="", finish_reason="stop")
            scripted = self._responses[self._index]
            self._index += 1
        _ = _validate_tool_names(rendered, scripted.tool_calls)
        return scripted.to_response()


def _validate_tool_names(
    rendered: RenderedPrompt, tool_calls: tuple[ToolCall, ...]
) -> tuple[Tool[object, object], ...]:
    available = {tool.name for tool in rendered.tools}
    for call in tool_calls:
        if call.name not in available:
            raise ValueError(f"NoopAdapter scripted unknown tool: {call.name!r}")
    return tuple(rendered.tools)
