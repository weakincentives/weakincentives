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

    files_read = session.query(SdkFileRead).all()
    bash_results = session.query(SdkBashResult).where(lambda r: r.exit_code != 0)
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Final

from ...dataclasses import FrozenDataclass

# Render output limits
_BASH_OUTPUT_PREVIEW_CHARS: Final[int] = 100
_GLOB_PREVIEW_COUNT: Final[int] = 5
_GREP_PREVIEW_COUNT: Final[int] = 3
_TRUNCATION_THRESHOLD: Final[int] = 50000

__all__ = [
    "SdkBashResult",
    "SdkEditResult",
    "SdkFileRead",
    "SdkGlobResult",
    "SdkGrepResult",
    "SdkToolResult",
    "SdkWriteResult",
    "parse_sdk_tool_result",
]


@FrozenDataclass()
class SdkFileRead:
    """Result from the SDK Read tool.

    Captures file content read by the agent, enabling queries like:
        session.query(SdkFileRead).where(lambda r: "error" in r.path)
    """

    path: str = field(metadata={"description": "Absolute file path that was read."})
    content: str = field(metadata={"description": "File content returned by the tool."})
    line_count: int = field(
        metadata={"description": "Number of lines in the returned content."}
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
class SdkBashResult:
    """Result from the SDK Bash tool.

    Captures command execution results, enabling queries like:
        failed = session.query(SdkBashResult).where(lambda r: r.exit_code != 0)
    """

    command: str = field(
        metadata={"description": "The bash command that was executed."}
    )
    stdout: str = field(metadata={"description": "Standard output from the command."})
    stderr: str = field(metadata={"description": "Standard error from the command."})
    exit_code: int = field(
        metadata={"description": "Exit code from the command (0 = success)."}
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
class SdkGlobResult:
    """Result from the SDK Glob tool.

    Captures file pattern matching results, enabling queries like:
        py_globs = session.query(SdkGlobResult).where(lambda r: "*.py" in r.pattern)
    """

    pattern: str = field(metadata={"description": "The glob pattern that was matched."})
    path: str = field(metadata={"description": "Directory where glob was executed."})
    matches: tuple[str, ...] = field(
        metadata={"description": "File paths matching the pattern."}
    )
    match_count: int = field(metadata={"description": "Total number of matches found."})

    def render(self) -> str:
        preview = ", ".join(self.matches[:_GLOB_PREVIEW_COUNT])
        remaining = self.match_count - _GLOB_PREVIEW_COUNT
        more = f" (+{remaining} more)" if remaining > 0 else ""
        return f"Glob {self.pattern}: {preview}{more}"


@FrozenDataclass()
class SdkGrepResult:
    """Result from the SDK Grep tool.

    Captures content search results, enabling queries like:
        todo_greps = session.query(SdkGrepResult).where(lambda r: "TODO" in r.pattern)
    """

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
class SdkWriteResult:
    """Result from the SDK Write tool.

    Captures file write operations, enabling queries like:
        writes = session.query(SdkWriteResult).all()
    """

    path: str = field(metadata={"description": "Absolute file path that was written."})
    bytes_written: int = field(
        metadata={"description": "Number of bytes written to the file."}
    )
    created: bool = field(
        default=False,
        metadata={"description": "Whether the file was newly created."},
    )

    def render(self) -> str:
        action = "Created" if self.created else "Wrote"
        return f"{action} {self.path}: {self.bytes_written} bytes"


@FrozenDataclass()
class SdkEditResult:
    """Result from the SDK Edit tool.

    Captures file edit operations, enabling queries like:
        edits = session.query(SdkEditResult).where(lambda r: r.replacements > 0)
    """

    path: str = field(metadata={"description": "Absolute file path that was edited."})
    old_string: str = field(metadata={"description": "The string that was replaced."})
    new_string: str = field(metadata={"description": "The replacement string."})
    replacements: int = field(metadata={"description": "Number of replacements made."})

    def render(self) -> str:
        return f"Edited {self.path}: {self.replacements} replacement(s)"


# Union type for all SDK tool results
type SdkToolResult = (
    SdkFileRead
    | SdkBashResult
    | SdkGlobResult
    | SdkGrepResult
    | SdkWriteResult
    | SdkEditResult
)


def _parse_file_read(
    tool_input: dict[str, Any], tool_response: dict[str, Any] | str
) -> SdkFileRead | None:
    """Parse Read tool response into SdkFileRead."""
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
    return SdkFileRead(
        path=path,
        content=content,
        line_count=len(lines),
        offset=tool_input.get("offset", 0),
        limit=tool_input.get("limit"),
        truncated=len(content) > _TRUNCATION_THRESHOLD,
    )


def _parse_bash_result(
    tool_input: dict[str, Any], tool_response: dict[str, Any] | str
) -> SdkBashResult | None:
    """Parse Bash tool response into SdkBashResult."""
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

    return SdkBashResult(
        command=command,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        interrupted=interrupted,
    )


def _parse_glob_result(
    tool_input: dict[str, Any], tool_response: dict[str, Any] | str
) -> SdkGlobResult | None:
    """Parse Glob tool response into SdkGlobResult."""
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

    return SdkGlobResult(
        pattern=pattern,
        path=path,
        matches=tuple(matches),
        match_count=len(matches),
    )


def _parse_grep_result(
    tool_input: dict[str, Any], tool_response: dict[str, Any] | str
) -> SdkGrepResult | None:
    """Parse Grep tool response into SdkGrepResult."""
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

    return SdkGrepResult(
        pattern=pattern,
        path=path,
        matches=tuple(matches),
        match_count=len(matches),
        files_searched=0,  # SDK doesn't provide this
    )


def _parse_write_result(
    tool_input: dict[str, Any], tool_response: dict[str, Any] | str
) -> SdkWriteResult | None:
    """Parse Write tool response into SdkWriteResult."""
    path = tool_input.get("file_path", "")
    content = tool_input.get("content", "")
    if not path:
        return None

    return SdkWriteResult(
        path=path,
        bytes_written=len(content.encode("utf-8")) if content else 0,
        created=True,  # Write always creates
    )


def _parse_edit_result(
    tool_input: dict[str, Any], tool_response: dict[str, Any] | str
) -> SdkEditResult | None:
    """Parse Edit tool response into SdkEditResult."""
    path = tool_input.get("file_path", "")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    if not path or not old_string:
        return None

    # We can't know the actual replacement count from the SDK response
    replacements = 1

    return SdkEditResult(
        path=path,
        old_string=old_string,
        new_string=new_string,
        replacements=replacements,
    )


# Registry mapping SDK tool names to their parsers
_SDK_TOOL_PARSERS: dict[
    str,
    type[SdkFileRead]
    | type[SdkBashResult]
    | type[SdkGlobResult]
    | type[SdkGrepResult]
    | type[SdkWriteResult]
    | type[SdkEditResult],
] = {
    "Read": SdkFileRead,
    "Bash": SdkBashResult,
    "Glob": SdkGlobResult,
    "Grep": SdkGrepResult,
    "Write": SdkWriteResult,
    "Edit": SdkEditResult,
}


def parse_sdk_tool_result(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | str,
) -> SdkToolResult | None:
    """Parse an SDK native tool result into a typed dataclass.

    Args:
        tool_name: Name of the SDK tool (e.g., "Read", "Bash", "Glob").
        tool_input: Input parameters passed to the tool.
        tool_response: Raw response from the tool.

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

    try:
        return parser(tool_input, tool_response)
    except Exception:
        # Silently ignore parsing errors to avoid breaking the hook
        return None
