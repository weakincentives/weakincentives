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

"""VFS data structures and path utilities.

This module provides the core data types for the virtual filesystem:

- VfsPath: Structured path representation for tool serialization
- Tool result types: FileInfo, GlobMatch, GrepMatch, ReadFileResult, etc.
- Tool parameter types: ListDirectoryParams, ReadFileParams, etc.
- Path utilities: format_path, path_from_string, normalization functions

These types are used by both the VFS tool handlers and the mount system.
"""

from __future__ import annotations

from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from ...dataclasses import FrozenDataclass
from ...errors import ToolValidationError

_ASCII: Final[str] = "ascii"
_DEFAULT_ENCODING: Final[Literal["utf-8"]] = "utf-8"
_MAX_WRITE_LENGTH: Final[int] = 48_000
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_SEGMENT_LENGTH: Final[int] = 80
_MAX_READ_LIMIT: Final[int] = 2_000

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
# Path Normalization Utilities
# ---------------------------------------------------------------------------


def normalize_content(content: str) -> str:
    """Normalize and validate content for writing."""
    if len(content) > _MAX_WRITE_LENGTH:
        raise ToolValidationError(
            "Content exceeds maximum length of 48,000 characters."
        )
    return content


def normalize_offset(offset: int) -> int:
    """Normalize and validate a read offset."""
    if offset < 0:
        raise ToolValidationError("offset must be non-negative.")
    return offset


def normalize_limit(limit: int) -> int:
    """Normalize and validate a read limit."""
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


def normalize_string_path(
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

    segments = normalize_segments(stripped.split("/"))
    if len(segments) > _MAX_PATH_DEPTH:
        raise ToolValidationError("Path depth exceeds the allowed limit (16 segments).")
    if not segments and not allow_empty:
        raise ToolValidationError(f"{field} must reference a file or directory.")
    return VfsPath(segments)


def normalize_optional_path(path: VfsPath | None) -> VfsPath:
    """Normalize an optional VfsPath, returning empty path if None."""
    if path is None:
        return VfsPath(())
    return normalize_path(path)


def normalize_path(path: VfsPath) -> VfsPath:
    """Normalize a VfsPath by validating and cleaning its segments."""
    segments = normalize_segments(path.segments)
    if len(segments) > _MAX_PATH_DEPTH:
        raise ToolValidationError("Path depth exceeds the allowed limit (16 segments).")
    return VfsPath(segments)


def normalize_segments(raw_segments: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize path segments by validating and cleaning them."""
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
            ensure_ascii(piece, "path segment")
            if len(piece) > _MAX_SEGMENT_LENGTH:
                raise ToolValidationError(
                    "Path segments must be 80 characters or fewer."
                )
            segments.append(piece)
    return tuple(segments)


def ensure_ascii(value: str, field: str) -> None:
    """Ensure a string contains only ASCII characters."""
    try:
        _ = value.encode(_ASCII)
    except UnicodeEncodeError as error:  # pragma: no cover - defensive guard
        raise ToolValidationError(
            f"{field.capitalize()} must be ASCII text."
        ) from error


# ---------------------------------------------------------------------------
# Constants re-export
# ---------------------------------------------------------------------------

MAX_WRITE_LENGTH: Final[int] = _MAX_WRITE_LENGTH
DEFAULT_ENCODING: Final[Literal["utf-8"]] = _DEFAULT_ENCODING


__all__ = [
    "DEFAULT_ENCODING",
    "MAX_WRITE_LENGTH",
    "DeleteEntry",
    "EditFileParams",
    "FileEncoding",
    "FileInfo",
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
    "VfsFile",
    "VfsPath",
    "WriteFile",
    "WriteFileParams",
    "WriteMode",
    "ensure_ascii",
    "format_path",
    "format_timestamp",
    "normalize_content",
    "normalize_limit",
    "normalize_offset",
    "normalize_optional_path",
    "normalize_path",
    "normalize_segments",
    "normalize_string_path",
    "path_from_string",
]
