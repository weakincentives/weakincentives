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

"""Shared utilities for the code review example agents."""

from __future__ import annotations

import json
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from weakincentives import MarkdownSection, Prompt, Tool, ToolResult
from weakincentives.adapters.core import PromptResponse
from weakincentives.events import EventBus, InProcessEventBus, ToolInvoked
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.serde import dump
from weakincentives.session import (
    DataEvent,
    Session,
    ToolData,
    select_all,
    select_latest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_CHARS = 4000


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 20]
    return f"{truncated}\n... (truncated {len(text) - len(truncated)} characters)"


@dataclass
class ReadFileParams:
    """Parameters for the read_file tool."""

    path: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class ReadFileResult:
    snippet: str


@dataclass
class ListChangedFilesParams:
    include_untracked: bool = True


@dataclass
class ListChangedFilesResult:
    entries: list[str]


@dataclass
class GitDiffParams:
    path: str | None = None
    revision: str | None = None
    staged: bool = False


@dataclass
class GitDiffResult:
    diff: str


@dataclass
class GitHistoryParams:
    path: str | None = None
    limit: int = 5


@dataclass
class GitHistoryResult:
    entries: list[str]


@dataclass
class ReviewGuidance:
    focus: str = (
        "Identify potential issues, risks, and follow-up questions for the changes "
        "under review."
    )


@dataclass
class ReviewTurnParams:
    request: str


@dataclass
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]


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


def read_file_handler(params: ReadFileParams) -> ToolResult[ReadFileResult]:
    target_path = (REPO_ROOT / params.path).resolve()
    try:
        target_path.relative_to(REPO_ROOT)
    except ValueError:
        message = "Requested path is outside of the repository."
        return ToolResult(message=message, value=ReadFileResult(snippet=""))

    if not target_path.exists() or not target_path.is_file():
        message = f"File '{params.path}' was not found."
        return ToolResult(message=message, value=ReadFileResult(snippet=""))

    text = target_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return ToolResult(
            message=f"File '{params.path}' is empty.",
            value=ReadFileResult(snippet=""),
        )

    start_line = params.start_line or 1
    end_line = params.end_line or len(lines)
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line

    start_index = start_line - 1
    end_index = min(end_line, len(lines))
    snippet = "\n".join(lines[start_index:end_index])
    snippet = _truncate(snippet)
    message = f"Read lines {start_line} to {end_index} from {params.path}."
    return ToolResult(message=message, value=ReadFileResult(snippet=snippet))


def list_changed_files_handler(
    params: ListChangedFilesParams,
) -> ToolResult[ListChangedFilesResult]:
    args = ["git", "status", "--short"]
    if not params.include_untracked:
        args.append("--untracked-files=no")

    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = f"git status failed: {result.stderr.strip()}"
        return ToolResult(message=message, value=ListChangedFilesResult(entries=[]))

    entries = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not entries:
        message = "No tracked changes detected."
    else:
        message = f"Found {len(entries)} changed file(s)."
    return ToolResult(message=message, value=ListChangedFilesResult(entries=entries))


def git_diff_handler(params: GitDiffParams) -> ToolResult[GitDiffResult]:
    args = ["git", "diff", "--unified=3"]
    if params.staged:
        args.append("--staged")
    if params.revision:
        args.append(params.revision)
    if params.path:
        if params.revision:
            args.append("--")
        args.append(params.path)

    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = f"git diff failed: {result.stderr.strip()}"
        return ToolResult(message=message, value=GitDiffResult(diff=""))

    diff_output = result.stdout.strip()
    if not diff_output:
        message = "No diff output produced."
    else:
        message = "Generated git diff output."
    diff_output = _truncate(diff_output)
    return ToolResult(message=message, value=GitDiffResult(diff=diff_output))


def git_history_handler(params: GitHistoryParams) -> ToolResult[GitHistoryResult]:
    limit = max(1, params.limit)
    args = ["git", "log", "--oneline", f"-n{limit}"]
    if params.path:
        args.extend(["--", params.path])

    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = f"git log failed: {result.stderr.strip()}"
        return ToolResult(message=message, value=GitHistoryResult(entries=[]))

    entries = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not entries:
        message = "No git history entries returned."
    else:
        message = f"Returned {len(entries)} git history entr{'y' if len(entries) == 1 else 'ies'}."
    return ToolResult(message=message, value=GitHistoryResult(entries=entries))


def build_tools() -> tuple[Tool[Any, Any], ...]:
    read_file_tool = Tool[ReadFileParams, ReadFileResult](
        name="read_file",
        description="Read repository files with optional line ranges to inspect code.",
        handler=read_file_handler,
    )
    changed_files_tool = Tool[ListChangedFilesParams, ListChangedFilesResult](
        name="list_changed_files",
        description="List files with modifications according to git status.",
        handler=list_changed_files_handler,
    )
    diff_tool = Tool[GitDiffParams, GitDiffResult](
        name="show_git_diff",
        description="Show git diff output for the workspace or a specific path.",
        handler=git_diff_handler,
    )
    history_tool = Tool[GitHistoryParams, GitHistoryResult](
        name="show_git_history",
        description="Summarize recent git history entries, optionally for a path.",
        handler=git_history_handler,
    )
    return (read_file_tool, changed_files_tool, diff_tool, history_tool)


def build_code_review_prompt() -> Prompt[ReviewResponse]:
    tools = build_tools()
    guidance_section = MarkdownSection[ReviewGuidance](
        title="Code Review Brief",
        template=textwrap.dedent(
            """
            You are a code review assistant focused on the current repository.
            Use the available tools to inspect files, review the latest changes,
            and study recent history before forming conclusions.

            Always provide a JSON response with the following keys:
            - summary: Single paragraph capturing the overall state of the changes.
            - issues: List of concrete problems, risks, or questions to raise.
            - next_steps: List of actionable recommendations or follow-ups.
            """
        ).strip(),
        default_params=ReviewGuidance(),
        tools=tools,
        key="code-review-brief",
    )
    user_turn_section = MarkdownSection[ReviewTurnParams](
        title="Review Request",
        template="${request}",
        key="review-request",
    )
    return Prompt[ReviewResponse](
        ns="examples/code-review",
        key="code-review-session",
        name="code_review_agent",
        sections=[guidance_section, user_turn_section],
    )


class CodeReviewSession:
    """Interactive session wrapper shared by example adapters."""

    def __init__(
        self, adapter: SupportsReviewEvaluate, *, bus: EventBus | None = None
    ) -> None:
        self._adapter = adapter
        self._bus = bus or InProcessEventBus()
        self._session = Session(bus=self._bus)
        self._prompt = build_code_review_prompt()
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
            return json.dumps(rendered_output)
        if response.text:
            return response.text
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

    def _display_tool_event(self, event: ToolInvoked) -> None:
        serialized_params = dump(event.params, exclude_none=True)
        payload = dump(event.result.value, exclude_none=True)
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

    def _register_tool_history(self) -> None:
        for result_type in (
            ReadFileResult,
            ListChangedFilesResult,
            GitDiffResult,
            GitHistoryResult,
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
