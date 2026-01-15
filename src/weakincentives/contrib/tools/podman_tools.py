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

"""Podman tool definitions factory.

This module provides the factory function for creating the Podman sandbox
tool definitions including filesystem operations, shell execution, and
Python evaluation.

Example usage::

    from weakincentives.contrib.tools.podman_tools import build_podman_tools

    tools = build_podman_tools(
        fs_handlers=handlers,
        shell_suite=shell,
        eval_suite=eval_suite,
        accepts_overrides=False,
    )
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ...prompt.tool import Tool, ToolExample
from .asteval import EvalParams, EvalResult
from .podman_shell import PodmanShellParams, PodmanShellResult
from .vfs import (
    DeleteEntry,
    EditFileParams,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    ListDirectoryParams,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    VfsPath,
    WriteFile,
    WriteFileParams,
)

if TYPE_CHECKING:
    from .podman_eval import PodmanEvalSuite
    from .podman_shell import PodmanShellSuite
    from .vfs import FilesystemToolHandlers


def build_podman_tools(
    *,
    fs_handlers: FilesystemToolHandlers,
    shell_suite: PodmanShellSuite,
    eval_suite: PodmanEvalSuite,
    accepts_overrides: bool,
) -> tuple[
    Tool[ListDirectoryParams, tuple[FileInfo, ...]],
    Tool[ReadFileParams, ReadFileResult],
    Tool[WriteFileParams, WriteFile],
    Tool[EditFileParams, WriteFile],
    Tool[GlobParams, tuple[GlobMatch, ...]],
    Tool[GrepParams, tuple[GrepMatch, ...]],
    Tool[RemoveParams, DeleteEntry],
    Tool[PodmanShellParams, PodmanShellResult],
    Tool[EvalParams, EvalResult],
]:
    """Build the complete set of Podman sandbox tools.

    Creates Tool instances for filesystem operations (ls, read_file, write_file,
    edit_file, glob, grep, rm), shell execution, and Python evaluation.

    Args:
        fs_handlers: Filesystem tool handlers instance.
        shell_suite: Shell execution suite instance.
        eval_suite: Python evaluation suite instance.
        accepts_overrides: Whether tools accept prompt overrides.

    Returns:
        Tuple of all Podman sandbox tools.
    """
    return (
        Tool[ListDirectoryParams, tuple[FileInfo, ...]](
            name="ls",
            description="List directory entries under a relative path.",
            handler=fs_handlers.list_directory,
            examples=(
                ToolExample[ListDirectoryParams, tuple[FileInfo, ...]](
                    description="List the workspace root",
                    input=ListDirectoryParams(path="/workspace"),
                    output=(
                        FileInfo(
                            path=VfsPath(("workspace", "README.md")),
                            kind="file",
                            size_bytes=4_096,
                            version=1,
                            updated_at=datetime(2024, 1, 1, tzinfo=UTC),
                        ),
                        FileInfo(
                            path=VfsPath(("workspace", "src")),
                            kind="directory",
                            size_bytes=None,
                            version=None,
                            updated_at=None,
                        ),
                    ),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
        Tool[ReadFileParams, ReadFileResult](
            name="read_file",
            description="Read UTF-8 file contents with pagination support.",
            handler=fs_handlers.read_file,
            examples=(
                ToolExample[ReadFileParams, ReadFileResult](
                    description="Read the top of README.md",
                    input=ReadFileParams(
                        file_path="/workspace/README.md", offset=0, limit=3
                    ),
                    output=ReadFileResult(
                        path=VfsPath(("workspace", "README.md")),
                        content=(
                            "   1 | # weakincentives\n"
                            "   2 | Open source automation harness\n"
                            "   3 | for safe agents"
                        ),
                        offset=0,
                        limit=3,
                        total_lines=120,
                    ),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
        Tool[WriteFileParams, WriteFile](
            name="write_file",
            description="Create a new UTF-8 text file.",
            handler=fs_handlers.write_file,
            examples=(
                ToolExample[WriteFileParams, WriteFile](
                    description="Create a notes file in the container",
                    input=WriteFileParams(
                        file_path="/workspace/notes.txt",
                        content="Remember to run make check",
                    ),
                    output=WriteFile(
                        path=VfsPath(("workspace", "notes.txt")),
                        content="Remember to run make check",
                        mode="create",
                    ),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
        Tool[EditFileParams, WriteFile](
            name="edit_file",
            description="Replace occurrences of a string within a file.",
            handler=fs_handlers.edit_file,
            examples=(
                ToolExample[EditFileParams, WriteFile](
                    description="Update a TODO entry",
                    input=EditFileParams(
                        file_path="/workspace/notes.txt",
                        old_string="TODO: add tests",
                        new_string="TODO: add integration tests",
                        replace_all=False,
                    ),
                    output=WriteFile(
                        path=VfsPath(("workspace", "notes.txt")),
                        content="Completed: scaffold\nTODO: add integration tests",
                        mode="overwrite",
                    ),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
        Tool[GlobParams, tuple[GlobMatch, ...]](
            name="glob",
            description="Match files beneath a directory using shell patterns.",
            handler=fs_handlers.glob,
            examples=(
                ToolExample[GlobParams, tuple[GlobMatch, ...]](
                    description="Find Python files under src",
                    input=GlobParams(pattern="**/*.py", path="/workspace/src"),
                    output=(
                        GlobMatch(
                            path=VfsPath(("workspace", "src", "__init__.py")),
                            size_bytes=128,
                            version=1,
                            updated_at=datetime(2024, 1, 1, tzinfo=UTC),
                        ),
                        GlobMatch(
                            path=VfsPath(
                                (
                                    "workspace",
                                    "src",
                                    "weakincentives",
                                    "__init__.py",
                                )
                            ),
                            size_bytes=256,
                            version=2,
                            updated_at=datetime(2024, 1, 2, tzinfo=UTC),
                        ),
                    ),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
        Tool[GrepParams, tuple[GrepMatch, ...]](
            name="grep",
            description="Search files for a regular expression pattern.",
            handler=fs_handlers.grep,
            examples=(
                ToolExample[GrepParams, tuple[GrepMatch, ...]](
                    description="Search for TODO comments",
                    input=GrepParams(
                        pattern="TODO", path="/workspace/src", glob="**/*.py"
                    ),
                    output=(
                        GrepMatch(
                            path=VfsPath(
                                (
                                    "workspace",
                                    "src",
                                    "weakincentives",
                                    "tools",
                                    "podman.py",
                                )
                            ),
                            line_number=42,
                            line="# TODO: improve sandbox docs",
                        ),
                        GrepMatch(
                            path=VfsPath(
                                (
                                    "workspace",
                                    "src",
                                    "weakincentives",
                                    "runtime",
                                    "__init__.py",
                                )
                            ),
                            line_number=10,
                            line="TODO: replace placeholder logger",
                        ),
                    ),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
        Tool[RemoveParams, DeleteEntry](
            name="rm",
            description="Remove files or directories recursively.",
            handler=fs_handlers.remove,
            examples=(
                ToolExample[RemoveParams, DeleteEntry](
                    description="Delete a stale build artifact",
                    input=RemoveParams(path="/workspace/build/output"),
                    output=DeleteEntry(path=VfsPath(("workspace", "build", "output"))),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
        Tool[PodmanShellParams, PodmanShellResult](
            name="shell_execute",
            description="Run a short command inside the Podman workspace.",
            handler=shell_suite.run_shell,
            examples=(
                ToolExample[PodmanShellParams, PodmanShellResult](
                    description="Check the current working directory",
                    input=PodmanShellParams(command=("pwd",), cwd=None),
                    output=PodmanShellResult(
                        command=("pwd",),
                        cwd="/workspace",
                        exit_code=0,
                        stdout="/workspace",
                        stderr="",
                        duration_ms=12,
                        timed_out=False,
                    ),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
        Tool[EvalParams, EvalResult](
            name="evaluate_python",
            description=(
                "Run a short Python script via `python3 -c` inside the Podman workspace. "
                "Captures stdout/stderr and reports the exit code."
            ),
            handler=eval_suite.evaluate_python,
            examples=(
                ToolExample[EvalParams, EvalResult](
                    description="Run a small calculation",
                    input=EvalParams(code="print(3 * 7)"),
                    output=EvalResult(
                        value_repr=None,
                        stdout="21\n",
                        stderr="",
                        globals={},
                        reads=(),
                        writes=(),
                    ),
                ),
            ),
            accepts_overrides=accepts_overrides,
        ),
    )


__all__ = [
    "build_podman_tools",
]
