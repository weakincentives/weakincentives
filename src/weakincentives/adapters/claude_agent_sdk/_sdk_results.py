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

"""Typed dataclasses for Claude Agent SDK native tool results.

SDK native tools (Read, Bash, Glob, Grep, Write, Edit) produce raw dict
responses. This module provides typed dataclasses that auto-parse these
outputs, enabling first-class access via session queries:

    files_read = session.query(ClaudeFileRead).all()
    bash_results = session.query(ClaudeBashResult).where(lambda r: r.exit_code != 0)

Each result type includes event_id and created_at fields, which are used
by the wink debug UI to classify these as event slices.
"""

from __future__ import annotations

import uuid
from dataclasses import field
from datetime import UTC, datetime
from typing import Any, Final

from ...dataclasses import FrozenDataclass

# Render output limits
_BASH_OUTPUT_PREVIEW_CHARS: Final[int] = 100
_GLOB_PREVIEW_COUNT: Final[int] = 5
_GREP_PREVIEW_COUNT: Final[int] = 3
_TRUNCATION_THRESHOLD: Final[int] = 50000

__all__ = [
    "ClaudeBashResult",
    "ClaudeEditResult",
    "ClaudeFileRead",
    "ClaudeGlobResult",
    "ClaudeGrepResult",
    "ClaudeToolResult",
    "ClaudeWriteResult",
    "parse_claude_tool_result",
]


@FrozenDataclass()
class ClaudeFileRead:
    """Result from the Claude Agent SDK Read tool.

    Captures file content read by the agent, enabling queries like:
        session.query(ClaudeFileRead).where(lambda r: "error" in r.path)
    """

    event_id: str = field(metadata={"description": "Unique identifier for this event."})
    path: str = field(metadata={"description": "Absolute file path that was read."})
    content: str = field(metadata={"description": "File content returned by the tool."})
    line_count: int = field(
        metadata={"description": "Number of lines in the returned content."}
    )
    created_at: datetime = field(
        metadata={"description": "Timestamp when the tool was executed."}
    )
    offset: int = field(
        default=0,
        metadata={"description": "Line offset if pagination was used."},
    )
    limit: int | None = field(
        default=None,
        metadata={"description": "Line limit if pagination was used."},
    )
    truncated: bool = field(
        default=False,
        metadata={"description": "Whether the content was truncated."},
    )

    def render(self) -> str:
        truncation = " (truncated)" if self.truncated else ""
        return f"Read {self.path}: {self.line_count} lines{truncation}"


@FrozenDataclass()
class ClaudeBashResult:
    """Result from the Claude Agent SDK Bash tool.

    Captures command execution results, enabling queries like:
        failed = session.query(ClaudeBashResult).where(lambda r: r.exit_code != 0)
    """

    event_id: str = field(metadata={"description": "Unique identifier for this event."})
    command: str = field(
        metadata={"description": "The bash command that was executed."}
    )
    stdout: str = field(metadata={"description": "Standard output from the command."})
    stderr: str = field(metadata={"description": "Standard error from the command."})
    exit_code: int = field(
        metadata={"description": "Exit code from the command (0 = success)."}
    )
    created_at: datetime = field(
        metadata={"description": "Timestamp when the tool was executed."}
    )
    interrupted: bool = field(
        default=False,
        metadata={"description": "Whether the command was interrupted/timed out."},
    )

    def render(self) -> str:
        status = "OK" if self.exit_code == 0 else f"exit {self.exit_code}"
        if len(self.stdout) > _BASH_OUTPUT_PREVIEW_CHARS:
            output_preview = self.stdout[:_BASH_OUTPUT_PREVIEW_CHARS] + "..."
        else:
            output_preview = self.stdout
        return f"$ {self.command} [{status}]\n{output_preview}"


@FrozenDataclass()
class ClaudeGlobResult:
    """Result from the Claude Agent SDK Glob tool.

    Captures file pattern matching results, enabling queries like:
        py_globs = session.query(ClaudeGlobResult).where(lambda r: "*.py" in r.pattern)
    """

    event_id: str = field(metadata={"description": "Unique identifier for this event."})
    pattern: str = field(metadata={"description": "The glob pattern that was matched."})
    path: str = field(metadata={"description": "Directory where glob was executed."})
    matches: tuple[str, ...] = field(
        metadata={"description": "File paths matching the pattern."}
    )
    match_count: int = field(metadata={"description": "Total number of matches found."})
    created_at: datetime = field(
        metadata={"description": "Timestamp when the tool was executed."}
    )

    def render(self) -> str:
        preview = ", ".join(self.matches[:_GLOB_PREVIEW_COUNT])
        remaining = self.match_count - _GLOB_PREVIEW_COUNT
        more = f" (+{remaining} more)" if remaining > 0 else ""
        return f"Glob {self.pattern}: {preview}{more}"


@FrozenDataclass()
class ClaudeGrepResult:
    """Result from the Claude Agent SDK Grep tool.

    Captures content search results, enabling queries like:
        todo_greps = session.query(ClaudeGrepResult).where(lambda r: "TODO" in r.pattern)
    """

    event_id: str = field(metadata={"description": "Unique identifier for this event."})
    pattern: str = field(
        metadata={"description": "The regex pattern that was searched."}
    )
    path: str = field(
        metadata={"description": "File or directory where grep was executed."}
    )
    matches: tuple[str, ...] = field(
        metadata={"description": "Lines containing matches."}
    )
    match_count: int = field(metadata={"description": "Total number of matches found."})
    created_at: datetime = field(
        metadata={"description": "Timestamp when the tool was executed."}
    )
    files_searched: int = field(
        default=0,
        metadata={"description": "Number of files that were searched."},
    )

    def render(self) -> str:
        preview = "\n".join(self.matches[:_GREP_PREVIEW_COUNT])
        remaining = self.match_count - _GREP_PREVIEW_COUNT
        more = f"\n(+{remaining} more)" if remaining > 0 else ""
        return f"Grep {self.pattern}:\n{preview}{more}"


@FrozenDataclass()
class ClaudeWriteResult:
    """Result from the Claude Agent SDK Write tool.

    Captures file write operations, enabling queries like:
        writes = session.query(ClaudeWriteResult).all()
    """

    event_id: str = field(metadata={"description": "Unique identifier for this event."})
    path: str = field(metadata={"description": "Absolute file path that was written."})
    bytes_written: int = field(
        metadata={"description": "Number of bytes written to the file."}
    )
    created_at: datetime = field(
        metadata={"description": "Timestamp when the tool was executed."}
    )
    created: bool = field(
        default=False,
        metadata={"description": "Whether the file was newly created."},
    )

    def render(self) -> str:
        action = "Created" if self.created else "Wrote"
        return f"{action} {self.path}: {self.bytes_written} bytes"


@FrozenDataclass()
class ClaudeEditResult:
    """Result from the Claude Agent SDK Edit tool.

    Captures file edit operations, enabling queries like:
        edits = session.query(ClaudeEditResult).where(lambda r: r.replacements > 0)
    """

    event_id: str = field(metadata={"description": "Unique identifier for this event."})
    path: str = field(metadata={"description": "Absolute file path that was edited."})
    old_string: str = field(metadata={"description": "The string that was replaced."})
    new_string: str = field(metadata={"description": "The replacement string."})
    replacements: int = field(metadata={"description": "Number of replacements made."})
    created_at: datetime = field(
        metadata={"description": "Timestamp when the tool was executed."}
    )

    def render(self) -> str:
        return f"Edited {self.path}: {self.replacements} replacement(s)"


# Union type for all Claude tool results
type ClaudeToolResult = (
    ClaudeFileRead
    | ClaudeBashResult
    | ClaudeGlobResult
    | ClaudeGrepResult
    | ClaudeWriteResult
    | ClaudeEditResult
)


def _parse_file_read(
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | str,
    event_id: str,
    created_at: datetime,
) -> ClaudeFileRead | None:
    """Parse Read tool response into ClaudeFileRead."""
    path = tool_input.get("file_path", "")
    if not path:
        return None

    content: str
    if isinstance(tool_response, str):
        content = tool_response
    elif isinstance(tool_response, dict):
        content = tool_response.get("stdout", "") or str(tool_response)
    else:
        content = str(tool_response) if tool_response else ""

    lines = content.splitlines()
    return ClaudeFileRead(
        event_id=event_id,
        path=path,
        content=content,
        line_count=len(lines),
        created_at=created_at,
        offset=tool_input.get("offset", 0),
        limit=tool_input.get("limit"),
        truncated=len(content) > _TRUNCATION_THRESHOLD,
    )


def _parse_bash_result(
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | str,
    event_id: str,
    created_at: datetime,
) -> ClaudeBashResult | None:
    """Parse Bash tool response into ClaudeBashResult."""
    command = tool_input.get("command", "")
    if not command:
        return None

    if isinstance(tool_response, str):
        stdout = tool_response
        stderr = ""
        exit_code = 0
        interrupted = False
    elif isinstance(tool_response, dict):
        stdout = tool_response.get("stdout", "")
        stderr = tool_response.get("stderr", "")
        # Exit code may be in various places depending on SDK version
        exit_code = tool_response.get("exitCode", tool_response.get("exit_code", 0))
        if exit_code is None:
            exit_code = 0
        interrupted = tool_response.get("interrupted", False)
    else:
        return None

    return ClaudeBashResult(
        event_id=event_id,
        command=command,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        created_at=created_at,
        interrupted=interrupted,
    )


def _parse_glob_result(
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | str,
    event_id: str,
    created_at: datetime,
) -> ClaudeGlobResult | None:
    """Parse Glob tool response into ClaudeGlobResult."""
    pattern = tool_input.get("pattern", "")
    path = tool_input.get("path", ".")
    if not pattern:
        return None

    matches: list[str] = []
    if isinstance(tool_response, str):
        # Response is typically newline-separated file paths
        matches = [line.strip() for line in tool_response.splitlines() if line.strip()]
    elif isinstance(tool_response, dict):
        stdout = tool_response.get("stdout", "")
        if stdout:
            matches = [line.strip() for line in stdout.splitlines() if line.strip()]

    return ClaudeGlobResult(
        event_id=event_id,
        pattern=pattern,
        path=path,
        matches=tuple(matches),
        match_count=len(matches),
        created_at=created_at,
    )


def _parse_grep_result(
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | str,
    event_id: str,
    created_at: datetime,
) -> ClaudeGrepResult | None:
    """Parse Grep tool response into ClaudeGrepResult."""
    pattern = tool_input.get("pattern", "")
    path = tool_input.get("path", ".")
    if not pattern:
        return None

    matches: list[str] = []
    if isinstance(tool_response, str):
        matches = [line.strip() for line in tool_response.splitlines() if line.strip()]
    elif isinstance(tool_response, dict):
        stdout = tool_response.get("stdout", "")
        if stdout:
            matches = [line.strip() for line in stdout.splitlines() if line.strip()]

    return ClaudeGrepResult(
        event_id=event_id,
        pattern=pattern,
        path=path,
        matches=tuple(matches),
        match_count=len(matches),
        created_at=created_at,
        files_searched=0,  # SDK doesn't provide this
    )


def _parse_write_result(
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | str,
    event_id: str,
    created_at: datetime,
) -> ClaudeWriteResult | None:
    """Parse Write tool response into ClaudeWriteResult."""
    path = tool_input.get("file_path", "")
    content = tool_input.get("content", "")
    if not path:
        return None

    return ClaudeWriteResult(
        event_id=event_id,
        path=path,
        bytes_written=len(content.encode("utf-8")) if content else 0,
        created_at=created_at,
        created=True,  # Write always creates
    )


def _parse_edit_result(
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | str,
    event_id: str,
    created_at: datetime,
) -> ClaudeEditResult | None:
    """Parse Edit tool response into ClaudeEditResult."""
    path = tool_input.get("file_path", "")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    if not path or not old_string:
        return None

    # We can't know the actual replacement count from the SDK response
    replacements = 1

    return ClaudeEditResult(
        event_id=event_id,
        path=path,
        old_string=old_string,
        new_string=new_string,
        replacements=replacements,
        created_at=created_at,
    )


# Registry mapping Claude tool names to their result types
_CLAUDE_TOOL_TYPES: dict[
    str,
    type[ClaudeFileRead]
    | type[ClaudeBashResult]
    | type[ClaudeGlobResult]
    | type[ClaudeGrepResult]
    | type[ClaudeWriteResult]
    | type[ClaudeEditResult],
] = {
    "Read": ClaudeFileRead,
    "Bash": ClaudeBashResult,
    "Glob": ClaudeGlobResult,
    "Grep": ClaudeGrepResult,
    "Write": ClaudeWriteResult,
    "Edit": ClaudeEditResult,
}


def parse_claude_tool_result(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | str,
    event_id: str | None = None,
    created_at: datetime | None = None,
) -> ClaudeToolResult | None:
    """Parse a Claude Agent SDK native tool result into a typed dataclass.

    Args:
        tool_name: Name of the SDK tool (e.g., "Read", "Bash", "Glob").
        tool_input: Input parameters passed to the tool.
        tool_response: Raw response from the tool.
        event_id: Unique identifier for this event. Defaults to a new UUID.
        created_at: Timestamp when the tool was executed. Defaults to now.

    Returns:
        Typed dataclass instance if parsing succeeds, None otherwise.
    """
    parsers: dict[str, Any] = {
        "Read": _parse_file_read,
        "Bash": _parse_bash_result,
        "Glob": _parse_glob_result,
        "Grep": _parse_grep_result,
        "Write": _parse_write_result,
        "Edit": _parse_edit_result,
    }

    parser = parsers.get(tool_name)
    if parser is None:
        return None

    eid = event_id if event_id is not None else str(uuid.uuid4())
    timestamp = created_at if created_at is not None else datetime.now(UTC)

    try:
        return parser(tool_input, tool_response, eid, timestamp)
    except Exception:
        # Silently ignore parsing errors to avoid breaking the hook
        return None
