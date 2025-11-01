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

"""Session orchestration for the code review examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..adapters import PromptResponse
from ..events import EventBus, InProcessEventBus, ToolInvoked
from ..prompt import Prompt, SupportsDataclass
from ..serde import dump
from ..session import (
    DataEvent,
    Session,
    ToolData,
    select_all,
    select_latest,
)
from .code_review_prompt import (
    ReviewResponse,
    ReviewTurnParams,
    build_code_review_prompt,
)
from .code_review_tools import (
    BranchListResult,
    GitLogResult,
    TagListResult,
    TimeQueryResult,
)


@dataclass(slots=True, frozen=True)
class ToolCallLog:
    """Recorded tool invocation captured by the session."""

    name: str
    prompt_name: str
    message: str
    value: dict[str, Any]
    call_id: str | None


class SupportsReviewEvaluate(Protocol):
    """Protocol describing the adapter interface consumed by the session."""

    def evaluate(
        self,
        prompt: Prompt[ReviewResponse],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
    ) -> PromptResponse[ReviewResponse]: ...


class CodeReviewSession:
    """Interactive session wrapper shared by example adapters."""

    def __init__(
        self,
        adapter: SupportsReviewEvaluate,
        *,
        bus: EventBus | None = None,
    ) -> None:
        self._adapter = adapter
        self._bus = bus or InProcessEventBus()
        self._session = Session(bus=self._bus)
        self._prompt = build_code_review_prompt(self._session)
        self._bus.subscribe(ToolInvoked, self._display_tool_event)
        self._register_tool_history()

    def evaluate(self, request: str) -> str:
        response = self._adapter.evaluate(
            self._prompt,
            ReviewTurnParams(request=request),
            bus=self._bus,
        )
        if response.output is not None:
            rendered_output = dump(response.output, exclude_none=True)
            return json.dumps(
                rendered_output,
                ensure_ascii=False,
                indent=2,
            )
        if response.text:
            return response.text  # pragma: no cover - convenience path for plain text
        return "(no response from assistant)"

    def render_tool_history(self) -> str:
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
            if record.value:
                payload_dump = json.dumps(record.value, ensure_ascii=False)
                lines.append(f"   payload: {payload_dump}")
        return "\n".join(lines)

    def _display_tool_event(self, event: object) -> None:
        if not isinstance(event, ToolInvoked):
            return

        serialized_params = dump(event.params, exclude_none=True)
        payload = dump(event.result.value, exclude_none=True)
        print(
            f"[tool] {event.name} called with {serialized_params}\n"
            f"       → {event.result.message}"
        )
        if payload:
            print(
                f"       payload: {payload}"
            )  # pragma: no cover - console output only
        latest = select_latest(self._session, ToolCallLog)
        if latest is not None:
            count = len(select_all(self._session, ToolCallLog))
            print(  # pragma: no cover - console output only
                f"       (session recorded this call as #{count})"
            )

    def _register_tool_history(self) -> None:
        for result_type in (
            GitLogResult,
            TimeQueryResult,
            BranchListResult,
            TagListResult,
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
            value=payload,
            call_id=event.source.call_id,
        )
        return slice_values + (record,)


__all__ = [
    "ToolCallLog",
    "SupportsReviewEvaluate",
    "CodeReviewSession",
]
