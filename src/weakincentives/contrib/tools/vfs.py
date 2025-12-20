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

"""Virtual filesystem tool suite.

This module provides VFS tools for LLM agents, building on top of the
Filesystem protocol. It defines:

- VfsPath: Structured path representation for tool serialization
- Tool result types: Rich types with VfsPath for LLM output
- Tool handlers: Convert Filesystem protocol results to tool outputs
- VfsToolsSection: Prompt section exposing VFS tools

The separation between filesystem.py (protocol with str paths) and this module
(tool types with VfsPath) keeps concerns clean:

- Filesystem protocol: Backend abstraction, simple str paths
- VFS tools: LLM-facing types with structured paths for serialization
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast, override

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError
from ...prompt.markdown import MarkdownSection
from ...prompt.tool import Tool, ToolContext, ToolExample, ToolResult
from ...runtime.session import Session
from ...types import SupportsDataclass, SupportsToolResult
from .filesystem import (
    READ_ENTIRE_FILE,
    Filesystem,
    InMemoryFilesystem,
)

_ASCII: Final[str] = "ascii"
_DEFAULT_ENCODING: Final[Literal["utf-8"]] = "utf-8"
_MAX_WRITE_LENGTH: Final[int] = 48_000
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_SEGMENT_LENGTH: Final[int] = 80
_MAX_READ_LIMIT: Final[int] = 2_000
_MAX_MOUNT_PREVIEW_ENTRIES: Final[int] = 20
_VFS_SECTION_TEMPLATE: Final[str] = (
    "The virtual filesystem starts empty unless host mounts are configured."
    " It is the only filesystem available and will contain files relevant to the task;"
    " use it as scratch space when necessary.\n"
    "1. Remember the snapshot begins empty aside from configured host mounts.\n"
    "2. Explore with `ls` or `glob` before reading or modifying files.\n"
    "3. Fetch file content with `read_file`; pagination keeps responses focused.\n"
    "4. Create files via `write_file` (create-only) and edit them with `edit_file`.\n"
    "5. Remove files or directories recursively with `rm` when they are no longer needed.\n"
    "6. Host mounts are fixed at session start; additional directories cannot be mounted later.\n"
    "7. Avoid mirroring large repositories or binary assets—only UTF-8 text up to 48k characters is accepted.\n"
    "8. Use `grep` to search for patterns across files when the workspace grows."
)

FileEncoding = Literal["utf-8"]
WriteMode = Literal["create", "overwrite", "append"]


# ---------------------------------------------------------------------------
# VfsPath - Structured path for tool serialization
# ---------------------------------------------------------------------------


@FrozenDataclass()
class VfsPath:
    """Relative POSIX-style path representation for tool serialization.

    This type is used in tool inputs/outputs to provide structured path
    representation that serializes cleanly to JSON for the LLM.
    """

    segments: tuple[str, ...] = field(
        metadata={
            "description": (
                "Ordered path segments. Values must be relative, ASCII-only, and "
                "free of '.' or '..'."
            )
        }
    )


def format_path(path: VfsPath) -> str:
    """Format a VfsPath as a string."""
    return "/".join(path.segments) or "/"


def format_timestamp(value: datetime | None) -> str:
    """Format a timestamp for display."""
    if value is None:
        return "-"
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return aware.isoformat()


def path_from_string(path_str: str) -> VfsPath:
    """Convert a string path to VfsPath."""
    if not path_str or path_str in {".", "/"}:
        return VfsPath(())
    segments = tuple(s for s in path_str.strip("/").split("/") if s and s != ".")
    return VfsPath(segments)


# ---------------------------------------------------------------------------
# Tool Result Types
# ---------------------------------------------------------------------------


@FrozenDataclass()
class FileInfo:
    """Metadata describing a directory entry (tool result type)."""

    path: VfsPath = field(
        metadata={"description": "Normalized VFS path referencing the directory entry."}
    )
    kind: Literal["file", "directory"] = field(
        metadata={
            "description": (
                "Entry type; directories surface nested paths while files carry metadata."
            )
        }
    )
    size_bytes: int | None = field(
        default=None,
        metadata={
            "description": (
                "On-disk size for files. Directories omit sizes to avoid redundant traversal."
            )
        },
    )
    version: int | None = field(
        default=None,
        metadata={
            "description": (
                "Monotonic file version propagated from the VFS snapshot. Directories omit versions."
            )
        },
    )
    updated_at: datetime | None = field(
        default=None,
        metadata={
            "description": (
                "Timestamp describing the most recent mutation for files; directories omit the value."
            )
        },
    )

    def render(self) -> str:
        path_label = format_path(self.path)
        if self.kind == "directory":
            directory_label = path_label if path_label == "/" else f"{path_label}/"
            return f"DIR  {directory_label}"
        size_label = "size ?"
        if self.size_bytes is not None:
            size_label = f"{self.size_bytes} B"
        version_label = "v?" if self.version is None else f"v{self.version}"
        updated_label = format_timestamp(self.updated_at)
        return (
            f"FILE {path_label} ("
            f"{size_label}, {version_label}, updated {updated_label})"
        )


@FrozenDataclass()
class GlobMatch:
    """Match returned by glob operations (tool result type)."""

    path: VfsPath = field(
        metadata={"description": "Path of the file or directory that matched the glob."}
    )
    size_bytes: int = field(
        metadata={
            "description": (
                "File size in bytes captured at snapshot time to help prioritize large assets."
            )
        }
    )
    version: int = field(
        metadata={
            "description": "Monotonic VFS version counter reflecting the latest write to the entry."
        }
    )
    updated_at: datetime = field(
        metadata={
            "description": "Timestamp from the snapshot identifying when the entry last changed."
        }
    )

    def render(self) -> str:
        path_label = format_path(self.path)
        timestamp = format_timestamp(self.updated_at)
        return (
            f"{path_label} - {self.size_bytes} B, v{self.version}, updated {timestamp}"
        )


@FrozenDataclass()
class GrepMatch:
    """Regex match returned by grep operations (tool result type)."""

    path: VfsPath = field(
        metadata={
            "description": "Path of the file containing the regex hit, normalized to the VFS."
        }
    )
    line_number: int = field(
        metadata={"description": "One-based line number where the regex matched."}
    )
    line: str = field(
        metadata={
            "description": "Full line content containing the match so callers can review context."
        }
    )

    def render(self) -> str:
        path_label = format_path(self.path)
        return f"{path_label}:{self.line_number}: {self.line}"


@FrozenDataclass()
class ReadFileResult:
    """Payload returned from read_file tool."""

    path: VfsPath = field(
        metadata={"description": "Path of the file that was read inside the VFS."}
    )
    content: str = field(
        metadata={
            "description": (
                "Formatted slice of the file contents with line numbers applied for clarity."
            )
        }
    )
    offset: int = field(
        metadata={
            "description": (
                "Zero-based line offset applied to the read window after normalization."
            )
        }
    )
    limit: int = field(
        metadata={
            "description": (
                "Maximum number of lines returned in this response after clamping to file length."
            )
        }
    )
    total_lines: int = field(
        metadata={
            "description": "Total line count of the file so callers can paginate follow-up reads."
        }
    )

    def render(self) -> str:
        path_label = format_path(self.path)
        if self.limit == 0:
            window = "no lines returned"
        else:
            end_line = self.offset + self.limit
            window = f"lines {self.offset + 1}-{end_line}"
        header = f"{path_label} - {window} of {self.total_lines}"
        body = self.content or "<empty>"
        return f"{header}\n\n{body}"


@FrozenDataclass()
class WriteFile:
    """Result of a file write operation (tool result type)."""

    path: VfsPath = field(
        metadata={"description": "Destination file path being written inside the VFS."}
    )
    content: str = field(
        metadata={
            "description": (
                "UTF-8 payload that will be persisted to the target file after validation."
            )
        }
    )
    mode: WriteMode = field(
        default="create",
        metadata={
            "description": (
                "Write strategy describing whether the file is newly created, overwritten, or appended."
            )
        },
    )
    encoding: FileEncoding = field(
        default="utf-8",
        metadata={
            "description": (
                "Codec used to encode the content when persisting it to the virtual filesystem."
            )
        },
    )

    def render(self) -> str:
        path_label = format_path(self.path)
        byte_count = len(self.content.encode(self.encoding))
        header = (
            f"{path_label} - mode {self.mode}, {byte_count} bytes ({self.encoding})"
        )
        body = self.content or "<empty>"
        return f"{header}\n\n{body}"


@FrozenDataclass()
class DeleteEntry:
    """Result of a file/directory deletion."""

    path: VfsPath = field(
        metadata={
            "description": "Path of the file or directory slated for removal from the VFS."
        }
    )

    def render(self) -> str:
        path_label = format_path(self.path)
        return f"Removed entries under {path_label}"


@FrozenDataclass()
class VfsFile:
    """Snapshot of a single file stored in the virtual filesystem."""

    path: VfsPath = field(
        metadata={"description": "Location of the file within the virtual filesystem."}
    )
    content: str = field(
        metadata={
            "description": (
                "UTF-8 text content of the file. Binary data is not supported."
            )
        }
    )
    encoding: FileEncoding = field(
        metadata={"description": "Name of the codec used to decode the file contents."}
    )
    size_bytes: int = field(
        metadata={"description": "Size of the encoded file on disk, in bytes."}
    )
    version: int = field(
        metadata={
            "description": (
                "Monotonic version counter that increments after each write."
            )
        }
    )
    created_at: datetime = field(
        metadata={
            "description": "Timestamp indicating when the file was first created."
        }
    )
    updated_at: datetime = field(
        metadata={"description": "Timestamp of the most recent write operation."}
    )


# ---------------------------------------------------------------------------
# Tool Parameter Types
# ---------------------------------------------------------------------------


@FrozenDataclass()
class ListDirectoryParams:
    path: str | None = field(
        default=None,
        metadata={
            "description": (
                "Directory path to list. Provide a relative VFS path or omit to list the root."
            )
        },
    )


@FrozenDataclass()
class ReadFileParams:
    file_path: str = field(
        metadata={
            "description": "Relative VFS path of the file to read (leading slashes are optional)."
        }
    )
    offset: int = field(
        default=0,
        metadata={
            "description": (
                "Zero-based line offset where reading should begin. Must be non-negative."
            )
        },
    )
    limit: int = field(
        default=_MAX_READ_LIMIT,
        metadata={
            "description": (
                "Maximum number of lines to return. Values are capped at 2,000 lines per request."
            )
        },
    )


@FrozenDataclass()
class WriteFileParams:
    file_path: str = field(
        metadata={
            "description": "Destination VFS path for the new file. Must not already exist."
        }
    )
    content: str = field(
        metadata={
            "description": (
                "UTF-8 text that will be written to the file. Content is limited to 48,000 characters."
            )
        }
    )


@FrozenDataclass()
class EditFileParams:
    file_path: str = field(
        metadata={
            "description": "Path to the file that should be edited inside the VFS."
        }
    )
    old_string: str = field(
        metadata={
            "description": (
                "Exact text to search for within the file. At least one occurrence must be present."
            )
        }
    )
    new_string: str = field(
        metadata={
            "description": "Replacement text that will substitute the matched content."
        }
    )
    replace_all: bool = field(
        default=False,
        metadata={
            "description": (
                "When true, replace every occurrence of `old_string`. Otherwise require a single match."
            )
        },
    )


@FrozenDataclass()
class GlobParams:
    pattern: str = field(
        metadata={
            "description": ("Shell-style pattern used to match files (e.g. `**/*.py`).")
        }
    )
    path: str = field(
        default="/",
        metadata={
            "description": (
                "Directory to treat as the search root. Defaults to the VFS root (`/`)."
            )
        },
    )


@FrozenDataclass()
class GrepParams:
    pattern: str = field(
        metadata={
            "description": "Regular expression pattern to search for in matching files."
        }
    )
    path: str | None = field(
        default=None,
        metadata={
            "description": (
                "Optional directory path that scopes the search. Defaults to the entire VFS snapshot."
            )
        },
    )
    glob: str | None = field(
        default=None,
        metadata={
            "description": (
                "Optional glob pattern that filters files before applying the regex search."
            )
        },
    )


@FrozenDataclass()
class RemoveParams:
    path: str = field(
        metadata={
            "description": "Relative VFS path targeting the file or directory that should be removed."
        }
    )


# ---------------------------------------------------------------------------
# Additional Result Types
# ---------------------------------------------------------------------------


@FrozenDataclass()
class ListDirectory:
    path: VfsPath | None = field(
        default=None,
        metadata={
            "description": (
                "Directory path to enumerate. When omitted the VFS root is listed."
            )
        },
    )


@FrozenDataclass()
class ListDirectoryResult:
    path: VfsPath = field(
        metadata={"description": "Directory that was listed after normalization."}
    )
    directories: tuple[str, ...] = field(
        metadata={
            "description": (
                "Immediate child directories contained within the listed path, sorted lexicographically."
            )
        }
    )
    files: tuple[str, ...] = field(
        metadata={
            "description": (
                "Immediate child files contained within the listed path, sorted lexicographically."
            )
        }
    )

    def render(self) -> str:
        path_label = "/".join(self.path.segments) or "/"
        lines = [f"Directory listing for {path_label}:"]
        lines.append("Directories:" if self.directories else "Directories: <none>")
        lines.extend(f"- {entry}" for entry in self.directories)
        lines.append("Files:" if self.files else "Files: <none>")
        lines.extend(f"- {entry}" for entry in self.files)
        return "\n".join(lines)


@FrozenDataclass()
class ReadFile:
    path: VfsPath = field(
        metadata={
            "description": (
                "Normalized path referencing the file read via :func:`read_file`."
            )
        }
    )


@FrozenDataclass()
class HostMount:
    host_path: str = field(
        metadata={
            "description": (
                "Relative path (within an allowed host root) that should be mirrored into the VFS snapshot."
            )
        }
    )
    mount_path: VfsPath | None = field(
        default=None,
        metadata={
            "description": (
                "Optional target path inside the VFS. Defaults to the host-relative path when omitted."
            )
        },
    )
    include_glob: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Whitelist of glob patterns applied to host files before mounting. Empty means include all files."
            )
        },
    )
    exclude_glob: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Blacklist of glob patterns that remove host files from the mount after inclusion filtering."
            )
        },
    )
    max_bytes: int | None = field(
        default=None,
        metadata={
            "description": (
                "Optional limit on the total number of bytes that may be imported from the host directory."
            )
        },
    )
    follow_symlinks: bool = field(
        default=False,
        metadata={
            "description": (
                "Whether to follow symbolic links when traversing the host directory tree during the mount."
            )
        },
    )


@FrozenDataclass()
class HostMountPreview:
    host_path: str = field(
        metadata={
            "description": (
                "User-specified relative path identifying the host directory or file to mount."
            )
        }
    )
    resolved_host: Path = field(
        metadata={
            "description": (
                "Absolute host filesystem path derived from the allowed mount roots."
            )
        }
    )
    mount_path: VfsPath = field(
        metadata={
            "description": "Destination path inside the VFS where the host content will appear."
        }
    )
    entries: tuple[str, ...] = field(
        metadata={
            "description": (
                "Sample of files or directories that will be imported for previewing the mount."
            )
        }
    )
    is_directory: bool = field(
        metadata={
            "description": (
                "Indicates whether the host path resolves to a directory (True) or a file (False)."
            )
        }
    )


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------


class FilesystemToolHandlers:
    """Reusable tool handlers that operate on the Filesystem protocol.

    This class provides handlers for common filesystem operations (ls, read,
    write, edit, glob, grep, rm) that work with any Filesystem implementation.

    Handlers get the filesystem from ToolContext.filesystem, allowing the same
    handlers to be shared across workspace implementations.

    Handlers convert Filesystem protocol results (str paths) to tool result
    types (VfsPath) for LLM serialization.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        """Initialize handlers.

        Args:
            clock: Optional callable returning current datetime. Defaults to UTC now.
        """
        super().__init__()
        self._clock = clock or _now

    @staticmethod
    def _get_filesystem(context: ToolContext) -> Filesystem:
        """Get the filesystem from context, raising if not available."""
        if context.filesystem is None:
            raise ToolValidationError("No filesystem available in this context.")
        return context.filesystem

    def list_directory(
        self, params: ListDirectoryParams, *, context: ToolContext
    ) -> ToolResult[tuple[FileInfo, ...]]:
        """List directory contents."""
        fs = self._get_filesystem(context)
        path = _normalize_string_path(
            params.path, allow_empty=True, field="path", mount_point=fs.mount_point
        )
        path_str = "/".join(path.segments) if path.segments else "."

        try:
            entries = fs.list(path_str)
        except FileNotFoundError:
            raise ToolValidationError(
                "Directory does not exist in the filesystem."
            ) from None
        except NotADirectoryError:
            raise ToolValidationError(
                "Cannot list a file path; provide a directory."
            ) from None

        result_entries: list[FileInfo] = []
        for entry in entries:
            entry_path = path_from_string(entry.path)
            if entry.is_file:
                stat = fs.stat(entry.path)
                result_entries.append(
                    FileInfo(
                        path=entry_path,
                        kind="file",
                        size_bytes=stat.size_bytes,
                        version=1,
                        updated_at=stat.modified_at,
                    )
                )
            else:
                result_entries.append(FileInfo(path=entry_path, kind="directory"))

        result_entries.sort(key=lambda entry: entry.path.segments)
        message = _format_directory_message(path, result_entries)
        return ToolResult(message=message, value=tuple(result_entries))

    def read_file(
        self, params: ReadFileParams, *, context: ToolContext
    ) -> ToolResult[ReadFileResult]:
        """Read file contents with pagination."""
        fs = self._get_filesystem(context)
        path = _normalize_string_path(
            params.file_path, field="file_path", mount_point=fs.mount_point
        )
        offset = _normalize_offset(params.offset)
        limit = _normalize_limit(params.limit)

        path_str = "/".join(path.segments)

        try:
            read_result = fs.read(path_str, offset=offset, limit=limit)
        except FileNotFoundError:
            raise ToolValidationError(
                "File does not exist in the filesystem."
            ) from None
        except ValueError as err:
            raise ToolValidationError(str(err)) from None

        lines = read_result.content.splitlines()
        numbered = [
            f"{index + 1:>4} | {line}"
            for index, line in enumerate(lines, start=read_result.offset)
        ]
        content = "\n".join(numbered)

        result = ReadFileResult(
            path=path,
            content=content,
            offset=read_result.offset,
            limit=read_result.limit,
            total_lines=read_result.total_lines,
        )
        message = _format_read_file_message_from_result(
            path, read_result.offset, read_result.offset + read_result.limit
        )
        return ToolResult(message=message, value=result)

    def write_file(
        self, params: WriteFileParams, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        """Create a new file (fails if file exists)."""
        fs = self._get_filesystem(context)
        path = _normalize_string_path(
            params.file_path, field="file_path", mount_point=fs.mount_point
        )
        content = _normalize_content(params.content)

        path_str = "/".join(path.segments)

        if fs.exists(path_str):
            raise ToolValidationError(
                "File already exists; use edit_file to modify existing content."
            )

        try:
            _ = fs.write(path_str, content, mode="create")
        except FileExistsError:  # pragma: no cover
            raise ToolValidationError(  # pragma: no cover
                "File already exists; use edit_file to modify existing content."
            ) from None

        normalized = WriteFile(path=path, content=content, mode="create")
        message = _format_write_file_message(path, content, mode="create")
        return ToolResult(message=message, value=normalized)

    def edit_file(
        self, params: EditFileParams, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        """Edit an existing file using string replacement."""
        fs = self._get_filesystem(context)
        path = _normalize_string_path(
            params.file_path, field="file_path", mount_point=fs.mount_point
        )
        path_str = "/".join(path.segments)

        try:
            # Read entire file to avoid truncation
            read_result = fs.read(path_str, limit=READ_ENTIRE_FILE)
        except FileNotFoundError:
            raise ToolValidationError(
                "File does not exist in the filesystem."
            ) from None
        except ValueError as err:
            raise ToolValidationError(str(err)) from None

        file_content = read_result.content

        old = params.old_string
        new = params.new_string
        if not old:
            raise ToolValidationError("old_string must not be empty.")
        if len(old) > _MAX_WRITE_LENGTH or len(new) > _MAX_WRITE_LENGTH:
            raise ToolValidationError(
                "Replacement strings must be 48,000 characters or fewer."
            )

        occurrences = file_content.count(old)
        if occurrences == 0:
            raise ToolValidationError("old_string not found in the target file.")
        if not params.replace_all and occurrences != 1:
            raise ToolValidationError(
                "old_string must match exactly once unless replace_all is true."
            )

        if params.replace_all:
            replacements = occurrences
            updated = file_content.replace(old, new)
        else:
            replacements = 1
            updated = file_content.replace(old, new, 1)

        normalized_content = _normalize_content(updated)
        _ = fs.write(path_str, normalized_content, mode="overwrite")

        normalized = WriteFile(
            path=path,
            content=normalized_content,
            mode="overwrite",
        )
        message = _format_edit_message(path, replacements)
        return ToolResult(message=message, value=normalized)

    def glob(
        self, params: GlobParams, *, context: ToolContext
    ) -> ToolResult[tuple[GlobMatch, ...]]:
        """Search for files matching a glob pattern."""
        fs = self._get_filesystem(context)
        base = _normalize_string_path(
            params.path, allow_empty=True, field="path", mount_point=fs.mount_point
        )
        pattern = params.pattern.strip()
        if not pattern:
            raise ToolValidationError("Pattern must not be empty.")
        _ensure_ascii(pattern, "pattern")

        base_str = "/".join(base.segments) if base.segments else "."

        glob_results = fs.glob(pattern, path=base_str)
        matches: list[GlobMatch] = []

        for match in glob_results:
            if match.is_file:
                stat = fs.stat(match.path)
                matches.append(
                    GlobMatch(
                        path=path_from_string(match.path),
                        size_bytes=stat.size_bytes,
                        version=1,
                        updated_at=stat.modified_at or self._clock(),
                    )
                )

        matches.sort(key=lambda match: match.path.segments)
        message = _format_glob_message(base, pattern, matches)
        return ToolResult(message=message, value=tuple(matches))

    def grep(
        self, params: GrepParams, *, context: ToolContext
    ) -> ToolResult[tuple[GrepMatch, ...]]:
        """Search for a pattern in file contents."""
        fs = self._get_filesystem(context)
        try:
            _ = re.compile(params.pattern)
        except re.error as error:
            return ToolResult(
                message=f"Invalid regular expression: {error}",
                value=None,
                success=False,
            )

        base_path: VfsPath | None = None
        if params.path is not None:
            base_path = _normalize_string_path(
                params.path, allow_empty=True, field="path", mount_point=fs.mount_point
            )
        glob_pattern = params.glob.strip() if params.glob is not None else None
        if glob_pattern:
            _ensure_ascii(glob_pattern, "glob")
        else:
            glob_pattern = None  # Treat empty/blank as no filter

        base_str = (
            "/".join(base_path.segments)
            if base_path is not None and base_path.segments
            else "."
        )

        grep_results = fs.grep(params.pattern, path=base_str, glob=glob_pattern)
        matches = [
            GrepMatch(
                path=path_from_string(result.path),
                line_number=result.line_number,
                line=result.line_content,
            )
            for result in grep_results
        ]
        matches.sort(key=lambda match: (match.path.segments, match.line_number))
        message = _format_grep_message(params.pattern, matches)
        return ToolResult(message=message, value=tuple(matches))

    def remove(
        self, params: RemoveParams, *, context: ToolContext
    ) -> ToolResult[DeleteEntry]:
        """Remove files or directories recursively."""
        fs = self._get_filesystem(context)
        path = _normalize_string_path(
            params.path, field="path", mount_point=fs.mount_point
        )
        path_str = "/".join(path.segments)

        if not fs.exists(path_str):
            raise ToolValidationError("No files matched the provided path.")

        # Count files being deleted for message
        deleted_count = 0
        stat = fs.stat(path_str)
        if stat.is_file:
            deleted_count = 1
        else:
            # Count files in directory
            glob_results = fs.glob("**/*", path=path_str)
            deleted_count = len([m for m in glob_results if m.is_file])
            if deleted_count == 0:
                deleted_count = 1  # At least the directory itself

        fs.delete(path_str, recursive=True)
        normalized = DeleteEntry(path=path)
        message = _format_delete_message_count(path, deleted_count)
        return ToolResult(message=message, value=normalized)


# ---------------------------------------------------------------------------
# VfsToolsSection
# ---------------------------------------------------------------------------


@FrozenDataclass()
class _VfsSectionParams:
    pass


@FrozenDataclass()
class VfsConfig:
    """Configuration for :class:`VfsToolsSection`.

    All constructor arguments for VfsToolsSection are consolidated here.
    This avoids accumulating long argument lists as the section evolves.

    Example::

        from weakincentives.contrib.tools import VfsConfig, VfsToolsSection

        config = VfsConfig(
            mounts=(HostMount(host_path="src"),),
            allowed_host_roots=("/home/user/project",),
        )
        section = VfsToolsSection(session=session, config=config)
    """

    mounts: Sequence[HostMount] = field(
        default=(),
        metadata={"description": "Host directories to mount into the VFS."},
    )
    allowed_host_roots: Sequence[os.PathLike[str] | str] = field(
        default=(),
        metadata={"description": "Allowed root paths for host mounts."},
    )
    accepts_overrides: bool = field(
        default=False,
        metadata={"description": "Whether the section accepts parameter overrides."},
    )


class VfsToolsSection(MarkdownSection[_VfsSectionParams]):
    """Prompt section exposing the virtual filesystem tool suite.

    Use :class:`VfsConfig` to consolidate configuration::

        config = VfsConfig(mounts=(HostMount(host_path="src"),))
        section = VfsToolsSection(session=session, config=config)

    Individual parameters are still accepted for backward compatibility,
    but config takes precedence when provided.
    """

    def __init__(
        self,
        *,
        session: Session,
        config: VfsConfig | None = None,
        mounts: Sequence[HostMount] = (),
        allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
        accepts_overrides: bool = False,
        _filesystem: InMemoryFilesystem | None = None,
        _mount_previews: tuple[HostMountPreview, ...] | None = None,
    ) -> None:
        # Resolve config - explicit config takes precedence
        if config is not None:
            resolved_mounts = tuple(config.mounts)
            resolved_roots = tuple(
                normalize_host_root(path) for path in config.allowed_host_roots
            )
            resolved_accepts_overrides = config.accepts_overrides
        else:
            resolved_mounts = tuple(mounts)
            resolved_roots = tuple(
                normalize_host_root(path) for path in allowed_host_roots
            )
            resolved_accepts_overrides = accepts_overrides

        self._allowed_roots = resolved_roots
        self._mounts = resolved_mounts

        # Use provided filesystem or create a new one
        if _filesystem is not None and _mount_previews is not None:
            # Cloning path - reuse existing state
            self._filesystem = _filesystem
            mount_previews = _mount_previews
        else:
            # Fresh initialization
            self._filesystem = InMemoryFilesystem()
            mount_previews = _materialize_host_mounts_to_filesystem(
                self._filesystem, self._mounts, self._allowed_roots
            )

        self._mount_previews = mount_previews
        self._session = session

        # Store config for cloning
        self._config = VfsConfig(
            mounts=self._mounts,
            allowed_host_roots=self._allowed_roots,
            accepts_overrides=resolved_accepts_overrides,
        )

        tools = _build_tools(accepts_overrides=resolved_accepts_overrides)
        super().__init__(
            title="Virtual Filesystem Tools",
            key="vfs.tools",
            template=_render_section_template(mount_previews),
            default_params=_VfsSectionParams(),
            tools=tools,
            accepts_overrides=resolved_accepts_overrides,
        )

    @property
    def session(self) -> Session:
        return self._session

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem managed by this section."""
        return self._filesystem

    @override
    def clone(self, **kwargs: object) -> VfsToolsSection:
        session_obj = kwargs.get("session")
        if not isinstance(session_obj, Session):
            msg = "session is required to clone VfsToolsSection."
            raise TypeError(msg)
        provided_bus = kwargs.get("bus")
        if provided_bus is not None and provided_bus is not session_obj.event_bus:
            msg = "Provided bus must match the target session's event bus."
            raise TypeError(msg)
        return VfsToolsSection(
            session=session_obj,
            config=self._config,
            _filesystem=self._filesystem,
            _mount_previews=self._mount_previews,
        )


# ---------------------------------------------------------------------------
# Tool Building
# ---------------------------------------------------------------------------


def _build_tools(
    *,
    accepts_overrides: bool,
) -> tuple[Tool[SupportsDataclass, SupportsToolResult], ...]:
    handlers = FilesystemToolHandlers()
    return cast(
        tuple[Tool[SupportsDataclass, SupportsToolResult], ...],
        (
            Tool[ListDirectoryParams, tuple[FileInfo, ...]](
                name="ls",
                description="List directory entries under a relative path.",
                handler=handlers.list_directory,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample(
                        description=(
                            "List the source directory to see top-level modules."
                        ),
                        input=ListDirectoryParams(path="src"),
                        output=(
                            FileInfo(
                                path=VfsPath(("src", "tests")),
                                kind="directory",
                            ),
                            FileInfo(
                                path=VfsPath(("src", "weakincentives")),
                                kind="directory",
                            ),
                        ),
                    ),
                ),
            ),
            Tool[ReadFileParams, ReadFileResult](
                name="read_file",
                description="Read UTF-8 file contents with pagination support.",
                handler=handlers.read_file,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample(
                        description="Read the repository README header.",
                        input=ReadFileParams(file_path="README.md", offset=0, limit=2),
                        output=ReadFileResult(
                            path=VfsPath(("README.md",)),
                            content=(
                                "   1 | # Weak Incentives\n"
                                "   2 | Open-source agent orchestration platform."
                            ),
                            offset=0,
                            limit=2,
                            total_lines=120,
                        ),
                    ),
                ),
            ),
            Tool[WriteFileParams, WriteFile](
                name="write_file",
                description="Create a new UTF-8 text file.",
                handler=handlers.write_file,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample(
                        description="Create a scratch note in the workspace.",
                        input=WriteFileParams(
                            file_path="notes/todo.txt",
                            content="- Outline VFS design decisions",
                        ),
                        output=WriteFile(
                            path=VfsPath(("notes", "todo.txt")),
                            content="- Outline VFS design decisions",
                            mode="create",
                        ),
                    ),
                ),
            ),
            Tool[EditFileParams, WriteFile](
                name="edit_file",
                description="Replace occurrences of a string within a file.",
                handler=handlers.edit_file,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample(
                        description="Update a configuration value in place.",
                        input=EditFileParams(
                            file_path="src/weakincentives/config.py",
                            old_string="DEBUG = True",
                            new_string="DEBUG = False",
                            replace_all=False,
                        ),
                        output=WriteFile(
                            path=VfsPath(("src", "weakincentives", "config.py")),
                            content='DEBUG = False\nLOG_LEVEL = "info"',
                            mode="overwrite",
                        ),
                    ),
                ),
            ),
            Tool[GlobParams, tuple[GlobMatch, ...]](
                name="glob",
                description="Match files beneath a directory using shell patterns.",
                handler=handlers.glob,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample(
                        description="Find Python tests under the tests directory.",
                        input=GlobParams(pattern="**/test_*.py", path="tests"),
                        output=(
                            GlobMatch(
                                path=VfsPath(("tests", "unit", "test_vfs.py")),
                                size_bytes=2048,
                                version=3,
                                updated_at=datetime(2024, 1, 5, tzinfo=UTC),
                            ),
                        ),
                    ),
                ),
            ),
            Tool[GrepParams, tuple[GrepMatch, ...]](
                name="grep",
                description="Search files for a regular expression pattern.",
                handler=handlers.grep,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample(
                        description="Search for log level constants in config files.",
                        input=GrepParams(
                            pattern=r"LOG_LEVEL", path="src/weakincentives"
                        ),
                        output=(
                            GrepMatch(
                                path=VfsPath(("src", "weakincentives", "config.py")),
                                line_number=5,
                                line='LOG_LEVEL = "info"',
                            ),
                        ),
                    ),
                ),
            ),
            Tool[RemoveParams, DeleteEntry](
                name="rm",
                description="Remove files or directories recursively.",
                handler=handlers.remove,
                accepts_overrides=accepts_overrides,
                examples=(
                    ToolExample(
                        description="Delete a temporary build directory.",
                        input=RemoveParams(path="tmp/build"),
                        output=DeleteEntry(path=VfsPath(("tmp", "build"))),
                    ),
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _normalize_content(content: str) -> str:
    if len(content) > _MAX_WRITE_LENGTH:
        raise ToolValidationError(
            "Content exceeds maximum length of 48,000 characters."
        )
    return content


def _normalize_offset(offset: int) -> int:
    if offset < 0:
        raise ToolValidationError("offset must be non-negative.")
    return offset


def _normalize_limit(limit: int) -> int:
    if limit <= 0:
        raise ToolValidationError("limit must be a positive integer.")
    return min(limit, _MAX_READ_LIMIT)


def _strip_mount_point(path: str, mount_point: str | None) -> str:
    """Strip mount point prefix from a path."""
    if mount_point is None:
        return path
    prefix = mount_point.lstrip("/")
    if path.startswith(prefix + "/"):
        return path[len(prefix) + 1 :]
    if path == prefix:
        return ""
    return path


def _normalize_string_path(
    raw: str | None,
    *,
    allow_empty: bool = False,
    field: str,
    mount_point: str | None = None,
) -> VfsPath:
    """Normalize a raw path string to a VfsPath.

    Args:
        raw: The raw path string from the tool parameters.
        allow_empty: Whether empty paths are allowed.
        field: Field name for error messages.
        mount_point: Optional virtual mount point prefix to strip. For example,
            if mount_point="/workspace", then "/workspace/sunfish" becomes "sunfish".
    """
    if raw is None:
        if not allow_empty:
            raise ToolValidationError(f"{field} is required.")
        return VfsPath(())

    stripped = raw.strip()
    if not stripped:
        if allow_empty:
            return VfsPath(())
        raise ToolValidationError(f"{field} must not be empty.")

    if stripped.startswith("/"):
        stripped = stripped.lstrip("/")

    stripped = _strip_mount_point(stripped, mount_point)

    segments = _normalize_segments(stripped.split("/"))
    if len(segments) > _MAX_PATH_DEPTH:
        raise ToolValidationError("Path depth exceeds the allowed limit (16 segments).")
    if not segments and not allow_empty:
        raise ToolValidationError(f"{field} must reference a file or directory.")
    return VfsPath(segments)


def _normalize_optional_path(path: VfsPath | None) -> VfsPath:
    if path is None:
        return VfsPath(())
    return _normalize_path(path)


def _normalize_path(path: VfsPath) -> VfsPath:
    segments = _normalize_segments(path.segments)
    if len(segments) > _MAX_PATH_DEPTH:
        raise ToolValidationError("Path depth exceeds the allowed limit (16 segments).")
    return VfsPath(segments)


def _normalize_segments(raw_segments: Sequence[str]) -> tuple[str, ...]:
    segments: list[str] = []
    for raw_segment in raw_segments:
        cleaned_segment = raw_segment.strip()
        if not cleaned_segment:
            continue
        if cleaned_segment.startswith("/"):
            raise ToolValidationError("Absolute paths are not allowed in the VFS.")
        for piece in cleaned_segment.split("/"):
            if not piece:
                continue
            if piece in {".", ".."}:
                raise ToolValidationError("Path segments may not include '.' or '..'.")
            _ensure_ascii(piece, "path segment")
            if len(piece) > _MAX_SEGMENT_LENGTH:
                raise ToolValidationError(
                    "Path segments must be 80 characters or fewer."
                )
            segments.append(piece)
    return tuple(segments)


def _ensure_ascii(value: str, field: str) -> None:
    try:
        _ = value.encode(_ASCII)
    except UnicodeEncodeError as error:  # pragma: no cover - defensive guard
        raise ToolValidationError(
            f"{field.capitalize()} must be ASCII text."
        ) from error


def _format_directory_message(path: VfsPath, entries: Sequence[FileInfo]) -> str:
    directory_count = sum(1 for entry in entries if entry.kind == "directory")
    file_count = sum(1 for entry in entries if entry.kind == "file")
    prefix = format_path(path)
    subdir_label = "subdir" if directory_count == 1 else "subdirs"
    file_label = "file" if file_count == 1 else "files"
    return (
        f"Listed directory {prefix} ("
        f"{directory_count} {subdir_label}, {file_count} {file_label})."
    )


def _format_read_file_message_from_result(path: VfsPath, start: int, end: int) -> str:
    path_label = format_path(path)
    if start == end:
        return f"Read file {path_label} (no lines returned)."
    return f"Read file {path_label} (lines {start + 1}-{end})."


def _format_write_file_message(path: VfsPath, content: str, mode: WriteMode) -> str:
    path_label = format_path(path)
    action = {
        "create": "Created",
        "overwrite": "Updated",
        "append": "Appended to",
    }[mode]
    size = len(content.encode(_DEFAULT_ENCODING))
    return f"{action} {path_label} ({size} bytes)."


def _format_edit_message(path: VfsPath, replacements: int) -> str:
    path_label = format_path(path)
    label = "occurrence" if replacements == 1 else "occurrences"
    return f"Replaced {replacements} {label} in {path_label}."


def _format_glob_message(
    base: VfsPath, pattern: str, matches: Sequence[GlobMatch]
) -> str:
    path_label = format_path(base)
    match_label = "match" if len(matches) == 1 else "matches"
    return f"Found {len(matches)} {match_label} under {path_label} for pattern '{pattern}'."


def _format_grep_message(pattern: str, matches: Sequence[GrepMatch]) -> str:
    match_label = "match" if len(matches) == 1 else "matches"
    return f"Found {len(matches)} {match_label} for pattern '{pattern}'."


def _format_delete_message_count(path: VfsPath, count: int) -> str:
    path_label = format_path(path)
    entry_label = "entry" if count == 1 else "entries"
    return f"Deleted {count} {entry_label} under {path_label}."


# ---------------------------------------------------------------------------
# Host Mount Functions
# ---------------------------------------------------------------------------


def normalize_host_root(path: os.PathLike[str] | str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ToolValidationError("Allowed host root does not exist.")
    return root


def _materialize_host_mounts_to_filesystem(
    fs: InMemoryFilesystem,
    mounts: Sequence[HostMount],
    allowed_roots: Sequence[Path],
) -> tuple[HostMountPreview, ...]:
    """Materialize host mounts directly into the InMemoryFilesystem."""
    if not mounts:
        return ()

    previews: list[HostMountPreview] = []
    for mount in mounts:
        preview = _load_mount_to_filesystem(fs, mount, allowed_roots)
        previews.append(preview)
    return tuple(previews)


def render_host_mounts_block(previews: Sequence[HostMountPreview]) -> str:
    if not previews:
        return ""

    lines: list[str] = ["Configured host mounts:"]
    for preview in previews:
        mount_label = format_path(preview.mount_path)
        resolved_label = str(preview.resolved_host)
        lines.append(
            f"- Host `{resolved_label}` (configured as `{preview.host_path}`) mounted at `{mount_label}`."
        )
        contents = _format_mount_entries(preview.entries)
        lines.append(f"  Contents: {contents}")
    return "\n".join(lines)


def _render_section_template(previews: Sequence[HostMountPreview]) -> str:
    block = render_host_mounts_block(previews)
    if not block:
        return _VFS_SECTION_TEMPLATE
    return f"{_VFS_SECTION_TEMPLATE}\n\n{block}"


def _format_mount_entries(entries: Sequence[str]) -> str:
    if not entries:
        return "<empty>"
    preview = entries[:_MAX_MOUNT_PREVIEW_ENTRIES]
    formatted = " ".join(f"`{entry}`" for entry in preview)
    remaining = len(entries) - len(preview)
    if remaining > 0:
        formatted += f" … (+{remaining} more)"
    return formatted


def _load_mount_to_filesystem(
    fs: InMemoryFilesystem,
    mount: HostMount,
    allowed_roots: Sequence[Path],
) -> HostMountPreview:
    """Load a single host mount into the filesystem."""
    host_path = mount.host_path.strip()
    if not host_path:
        raise ToolValidationError("Host mount path must not be empty.")
    _ensure_ascii(host_path, "host path")
    resolved_host = _resolve_mount_path(host_path, allowed_roots)
    include_patterns = _normalize_globs(mount.include_glob, "include_glob")
    exclude_patterns = _normalize_globs(mount.exclude_glob, "exclude_glob")
    mount_prefix = _normalize_optional_path(mount.mount_path)
    preview = HostMountPreview(
        host_path=host_path,
        resolved_host=resolved_host,
        mount_path=mount_prefix,
        entries=_list_mount_entries(resolved_host),
        is_directory=resolved_host.is_dir(),
    )

    context = _MountContext(
        resolved_host=resolved_host,
        mount_prefix=mount_prefix,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        timestamp=_now(),
        max_bytes=mount.max_bytes,
    )

    consumed_bytes = 0
    for path in _iter_mount_files(resolved_host, mount.follow_symlinks):
        consumed_bytes = _read_mount_entry_to_filesystem(
            fs=fs, path=path, context=context, consumed_bytes=consumed_bytes
        )

    return preview


@FrozenDataclass()
class _MountContext:
    resolved_host: Path = field(
        metadata={"description": "Absolute host path resolved against allowed roots."}
    )
    mount_prefix: VfsPath = field(
        metadata={
            "description": "Normalized VFS path where the host mount will appear."
        }
    )
    include_patterns: tuple[str, ...] = field(
        metadata={
            "description": "Glob patterns that must match mounted files when provided."
        }
    )
    exclude_patterns: tuple[str, ...] = field(
        metadata={
            "description": "Glob patterns that exclude files from the mounted set."
        }
    )
    timestamp: datetime = field(
        metadata={
            "description": "Timestamp applied to mounted files for created/updated metadata."
        }
    )
    max_bytes: int | None = field(
        metadata={
            "description": "Optional byte budget limiting total mounted file size."
        }
    )


def _read_mount_entry_to_filesystem(
    *,
    fs: InMemoryFilesystem,
    path: Path,
    context: _MountContext,
    consumed_bytes: int,
) -> int:
    """Read a mount entry and write it to the filesystem. Returns updated consumed_bytes."""
    relative = (
        Path(path.name)
        if context.resolved_host.is_file()
        else path.relative_to(context.resolved_host)
    )
    relative_posix = relative.as_posix()
    if context.include_patterns and not any(
        _match_glob(relative_posix, pattern) for pattern in context.include_patterns
    ):
        return consumed_bytes
    if any(
        _match_glob(relative_posix, pattern) for pattern in context.exclude_patterns
    ):
        return consumed_bytes

    try:
        content = path.read_text(encoding=_DEFAULT_ENCODING)
    except UnicodeDecodeError as error:  # pragma: no cover - defensive guard
        raise ToolValidationError("Mounted file must be valid UTF-8.") from error
    except OSError as error:
        raise ToolValidationError(f"Failed to read mounted file {path}.") from error
    size = len(content.encode(_DEFAULT_ENCODING))
    if context.max_bytes is not None and consumed_bytes + size > context.max_bytes:
        raise ToolValidationError("Host mount exceeded the configured byte budget.")
    consumed_bytes += size

    # Build the VFS path
    segments = context.mount_prefix.segments + relative.parts
    normalized_path = _normalize_path(VfsPath(segments))
    vfs_path_str = "/".join(normalized_path.segments)

    # Write to filesystem
    _ = fs.write(vfs_path_str, content, mode="overwrite")

    return consumed_bytes


def _list_mount_entries(root: Path) -> tuple[str, ...]:
    if root.is_file():
        return (root.name,)
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name.lower())
    except OSError as error:  # pragma: no cover - defensive guard
        raise ToolValidationError(f"Failed to inspect host mount {root}.") from error
    labels: list[str] = []
    for child in children:
        suffix = "/" if child.is_dir() else ""
        labels.append(f"{child.name}{suffix}")
    return tuple(labels)


def _resolve_mount_path(host_path: str, allowed_roots: Sequence[Path]) -> Path:
    if not allowed_roots:
        raise ToolValidationError("No allowed host roots configured for mounts.")
    for root in allowed_roots:
        candidate = (root / host_path).resolve()
        try:
            _ = candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.exists():
            return candidate
    raise ToolValidationError("Host path is outside the allowed roots or missing.")


def _normalize_globs(patterns: Sequence[str], field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for pattern in patterns:
        stripped = pattern.strip()
        if not stripped:
            continue
        _ensure_ascii(stripped, field)
        normalized.append(stripped)
    return tuple(normalized)


def _match_glob(path: str, pattern: str) -> bool:
    """Match a path against a glob pattern with proper ** support.

    The standard fnmatch module doesn't treat ** as "zero or more directories".
    This function handles that case: **/*.py should match both "foo.py" (zero
    directories) and "bar/baz.py" (one+ directories).
    """
    if "**/" not in pattern:
        return fnmatch.fnmatchcase(path, pattern)

    # Match the full pattern first (handles one+ directories case).
    if fnmatch.fnmatchcase(path, pattern):
        return True

    # Allow **/ to match zero directories by removing one or more occurrences.
    # This preserves any prefix before **/ (e.g., "src/**/test_*.py" -> "src/test_*.py").
    candidates = [pattern]
    seen: set[str] = {pattern}
    while candidates:
        current = candidates.pop()
        start = 0
        while True:
            index = current.find("**/", start)
            if index == -1:
                break
            variant = f"{current[:index]}{current[index + 3 :]}"
            if variant not in seen:
                if fnmatch.fnmatchcase(path, variant):
                    return True
                seen.add(variant)
                candidates.append(variant)
            start = index + 3

    return False


def _iter_mount_files(root: Path, follow_symlinks: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for current, _dirnames, filenames in root.walk(
        follow_symlinks=follow_symlinks,
    ):
        for name in filenames:
            yield current / name


def _now() -> datetime:
    return _truncate_to_milliseconds(datetime.now(UTC))


def _truncate_to_milliseconds(value: datetime) -> datetime:
    microsecond = value.microsecond - (value.microsecond % 1000)
    return value.replace(microsecond=microsecond, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------------

# Re-export for backward compatibility with podman.py
MAX_WRITE_LENGTH: Final[int] = _MAX_WRITE_LENGTH
normalize_string_path = _normalize_string_path
normalize_path = _normalize_path
normalize_content = _normalize_content
normalize_offset = _normalize_offset
normalize_limit = _normalize_limit
ensure_ascii = _ensure_ascii
format_directory_message = _format_directory_message
format_write_file_message = _format_write_file_message
format_edit_message = _format_edit_message
format_glob_message = _format_glob_message
format_grep_message = _format_grep_message


__all__ = [
    "MAX_WRITE_LENGTH",
    "DeleteEntry",
    "EditFileParams",
    "FileEncoding",
    "FileInfo",
    "FilesystemToolHandlers",
    "GlobMatch",
    "GlobParams",
    "GrepMatch",
    "GrepParams",
    "HostMount",
    "HostMountPreview",
    "ListDirectory",
    "ListDirectoryParams",
    "ListDirectoryResult",
    "ReadFile",
    "ReadFileParams",
    "ReadFileResult",
    "RemoveParams",
    "VfsConfig",
    "VfsFile",
    "VfsPath",
    "VfsToolsSection",
    "WriteFile",
    "WriteFileParams",
    "WriteMode",
    "ensure_ascii",
    "format_directory_message",
    "format_edit_message",
    "format_glob_message",
    "format_grep_message",
    "format_path",
    "format_timestamp",
    "format_write_file_message",
    "normalize_content",
    "normalize_limit",
    "normalize_offset",
    "normalize_path",
    "normalize_string_path",
    "path_from_string",
]
