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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from weakincentives import (
    MarkdownSection,
    Prompt,
    PromptResponse,
    SupportsDataclass,
    Tool,
    ToolResult,
)
from weakincentives.events import EventBus, InProcessEventBus, ToolInvoked
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
class GitLogParams:
    """Parameters for querying git log history."""

    revision_range: str | None = field(
        default=None,
        metadata={
            "description": (
                "Commit range spec passed to `git log` (for example 'main..HEAD')."
            )
        },
    )
    path: str | None = field(
        default=None,
        metadata={
            "description": "Restrict history to a specific file or directory.",
        },
    )
    max_count: int | None = field(
        default=20,
        metadata={"description": "Maximum number of commits to return."},
    )
    skip: int | None = field(
        default=None,
        metadata={
            "description": "Number of commits to skip from the top of the result.",
        },
    )
    author: str | None = field(
        default=None,
        metadata={
            "description": ("Filter commits to those authored by the provided string."),
        },
    )
    since: str | None = field(
        default=None,
        metadata={
            "description": (
                "Only include commits after this date/time (forwarded to git)."
            ),
        },
    )
    until: str | None = field(
        default=None,
        metadata={
            "description": (
                "Only include commits up to this date/time (forwarded to git)."
            ),
        },
    )
    grep: str | None = field(
        default=None,
        metadata={
            "description": (
                "Include commits whose messages match this regular expression."
            ),
        },
    )
    additional_args: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "description": "Extra raw arguments forwarded to `git log`.",
        },
    )


@dataclass
class GitLogResult:
    entries: list[str]


@dataclass
class TimeQueryParams:
    """Parameters for requesting the current time."""

    timezone: str | None = field(
        default=None,
        metadata={
            "description": (
                "IANA timezone identifier to convert the current time. Defaults to UTC."
            ),
        },
    )


@dataclass
class TimeQueryResult:
    iso_timestamp: str
    timezone: str
    source: str


@dataclass
class BranchListParams:
    """Parameters for listing git branches."""

    include_remote: bool = field(
        default=False,
        metadata={
            "description": "Include remote branches when set to true (uses --all).",
        },
    )
    pattern: str | None = field(
        default=None,
        metadata={
            "description": "Optional glob to filter branch names (passed to git).",
        },
    )
    contains: str | None = field(
        default=None,
        metadata={
            "description": "Only branches containing this commit (uses --contains).",
        },
    )


@dataclass
class BranchListResult:
    branches: list[str]


@dataclass
class TagListParams:
    """Parameters for listing git tags."""

    pattern: str | None = field(
        default=None,
        metadata={
            "description": "Optional glob to filter tags (passed to git).",
        },
    )
    sort: str | None = field(
        default=None,
        metadata={
            "description": "Sort directive forwarded to git tag --sort (for example '-version:refname').",
        },
    )
    contains: str | None = field(
        default=None,
        metadata={
            "description": "Only tags containing this commit (uses --contains).",
        },
    )


@dataclass
class TagListResult:
    tags: list[str]


@dataclass
class ReviewGuidance:
    focus: str = field(
        default=(
            "Identify potential issues, risks, and follow-up questions for the changes "
            "under review."
        ),
        metadata={
            "description": "Default framing instructions for the review assistant.",
        },
    )


@dataclass
class ReviewTurnParams:
    request: str = field(
        metadata={
            "description": "User-provided review task or question to address.",
        }
    )


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


def git_log_handler(params: GitLogParams) -> ToolResult[GitLogResult]:
    max_count = params.max_count
    args = ["git", "log", "--oneline"]
    if max_count is not None:
        max_count = max(1, max_count)
        args.append(f"-n{max_count}")
    if params.skip:
        args.extend(["--skip", str(max(params.skip, 0))])
    if params.author:
        args.extend(["--author", params.author])
    if params.since:
        args.extend(["--since", params.since])
    if params.until:
        args.extend(["--until", params.until])
    if params.grep:
        args.extend(["--grep", params.grep])
    if params.additional_args:
        args.extend(params.additional_args)
    if params.revision_range:
        args.append(params.revision_range)
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
        return ToolResult(message=message, value=GitLogResult(entries=[]))

    entries = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not entries:
        message = "No git log entries matched the query."
    else:
        joined_entries = "\n".join(entries)
        message = (
            f"Returned {len(entries)} git log entr{'y' if len(entries) == 1 else 'ies'}:\n"
            f"{_truncate(joined_entries)}"
        )
    return ToolResult(message=message, value=GitLogResult(entries=entries))


def current_time_handler(params: TimeQueryParams) -> ToolResult[TimeQueryResult]:
    requested_timezone = params.timezone or "UTC"
    timezone_name = requested_timezone
    try:
        tzinfo = ZoneInfo(requested_timezone)
        source = "zoneinfo"
    except ZoneInfoNotFoundError:
        tzinfo = ZoneInfo("UTC")
        timezone_name = "UTC"
        source = "fallback"

    now = datetime.now(tzinfo)
    iso_timestamp = now.isoformat()
    if source == "fallback" and requested_timezone != "UTC":
        message = (
            f"Timezone '{requested_timezone}' not found. "
            f"Using UTC instead.\nCurrent time (UTC): {iso_timestamp}"
        )
    else:
        message = f"Current time in {timezone_name}: {iso_timestamp}"

    return ToolResult(
        message=message,
        value=TimeQueryResult(
            iso_timestamp=iso_timestamp,
            timezone=timezone_name,
            source=source,
        ),
    )


def branch_list_handler(params: BranchListParams) -> ToolResult[BranchListResult]:
    args = ["git", "branch", "--list"]
    if params.include_remote:
        args.append("--all")
    if params.contains:
        args.extend(["--contains", params.contains])
    if params.pattern:
        args.append(params.pattern)

    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = f"git branch failed: {result.stderr.strip()}"
        return ToolResult(message=message, value=BranchListResult(branches=[]))

    branches = [line.strip().lstrip("* ") for line in result.stdout.splitlines()]
    branches = [branch for branch in branches if branch]
    if not branches:
        message = "No branches matched the query."
    else:
        message = (
            f"Returned {len(branches)} branch entr{'y' if len(branches) == 1 else 'ies'}:\n"
            f"{_truncate('\n'.join(branches))}"
        )
    return ToolResult(message=message, value=BranchListResult(branches=branches))


def tag_list_handler(params: TagListParams) -> ToolResult[TagListResult]:
    args = ["git", "tag", "--list"]
    if params.contains:
        args.extend(["--contains", params.contains])
    if params.sort:
        args.extend(["--sort", params.sort])
    if params.pattern:
        args.append(params.pattern)

    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = f"git tag failed: {result.stderr.strip()}"
        return ToolResult(message=message, value=TagListResult(tags=[]))

    tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not tags:
        message = "No tags matched the query."
    else:
        message = (
            f"Returned {len(tags)} tag entr{'y' if len(tags) == 1 else 'ies'}:\n"
            f"{_truncate('\n'.join(tags))}"
        )
    return ToolResult(message=message, value=TagListResult(tags=tags))


def build_tools() -> tuple[Tool[Any, Any], ...]:
    git_log_tool = Tool[GitLogParams, GitLogResult](
        name="show_git_log",
        description=(
            "Inspect repository history using git log filters such as revision "
            "ranges, authors, dates, grep patterns, and file paths."
        ),
        handler=git_log_handler,
    )
    current_time_tool = Tool[TimeQueryParams, TimeQueryResult](
        name="show_current_time",
        description="Fetch the current time in UTC or a provided timezone using zoneinfo.",
        handler=current_time_handler,
    )
    branch_list_tool = Tool[BranchListParams, BranchListResult](
        name="show_git_branches",
        description=(
            "List local or remote branches with optional glob filters and commit containment checks."
        ),
        handler=branch_list_handler,
    )
    tag_list_tool = Tool[TagListParams, TagListResult](
        name="show_git_tags",
        description=(
            "List repository tags with optional glob filters, sorting, and commit containment checks."
        ),
        handler=tag_list_handler,
    )
    return (
        git_log_tool,
        current_time_tool,
        branch_list_tool,
        tag_list_tool,
    )


def build_code_review_prompt() -> Prompt[ReviewResponse]:
    tools = build_tools()
    guidance_section = MarkdownSection[ReviewGuidance](
        title="Code Review Brief",
        template=textwrap.dedent(
            """
            You are a code review assistant working in this repository.
            Every response must stay anchored to the specific task described
            in the review request. If the task is unclear, ask for the missing
            details before proceeding.

            Use the available tools to stay grounded:
            - `show_git_log` retrieves commit history relevant to the task.
            - `show_git_branches` lists branches that match specified filters.
            - `show_git_tags` lists tags that match specified filters.
            - `show_current_time` reports the present time (default UTC or a
              requested timezone).
            If the task requires information beyond these capabilities, ask the
            user for clarification rather than guessing.

            Always provide a JSON response with the following keys:
            - summary: Single paragraph capturing the overall state of the changes.
            - issues: List of concrete problems, risks, or follow-up questions tied
              to the task.
            - next_steps: List of actionable recommendations or follow-ups that
              help complete the task or mitigate the issues.
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
            return json.dumps(
                rendered_output,
                ensure_ascii=False,
                indent=2,
            )
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
            print(f"       payload: {payload}")
        latest = select_latest(self._session, ToolCallLog)
        if latest is not None:
            count = len(select_all(self._session, ToolCallLog))
            print(f"       (session recorded this call as #{count})")

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
