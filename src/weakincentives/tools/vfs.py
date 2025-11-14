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

"""Virtual filesystem tool suite."""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast
from weakref import WeakSet

from ..prompt import SupportsDataclass
from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolResult
from ..runtime.session import (
    ReducerContextProtocol,
    ReducerEvent,
    Session,
    TypedReducer,
    replace_latest,
    select_latest,
)
from .errors import ToolValidationError

FileEncoding = Literal["utf-8"]
WriteMode = Literal["create", "overwrite", "append"]

_ASCII: Final[str] = "ascii"
_DEFAULT_ENCODING: Final[FileEncoding] = "utf-8"
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


@dataclass(slots=True, frozen=True)
class VfsPath:
    """Relative POSIX-style path representation."""

    segments: tuple[str, ...] = field(
        metadata={
            "description": (
                "Ordered path segments. Values must be relative, ASCII-only, and "
                "free of '.' or '..'."
            )
        }
    )


@dataclass(slots=True, frozen=True)
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


@dataclass(slots=True, frozen=True)
class VirtualFileSystem:
    """Immutable snapshot of the virtual filesystem state."""

    files: tuple[VfsFile, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Collection of tracked files. Each entry captures file metadata "
                "and contents."
            )
        },
    )


@dataclass(slots=True, frozen=True)
class FileInfo:
    """Metadata describing a directory entry."""

    path: VfsPath
    kind: Literal["file", "directory"]
    size_bytes: int | None = None
    version: int | None = None
    updated_at: datetime | None = None


@dataclass(slots=True, frozen=True)
class ReadFileResult:
    """Payload returned from :func:`read_file`."""

    path: VfsPath
    content: str
    offset: int
    limit: int
    total_lines: int


@dataclass(slots=True, frozen=True)
class GlobMatch:
    """Match returned by the :func:`glob` tool."""

    path: VfsPath
    size_bytes: int
    version: int
    updated_at: datetime


@dataclass(slots=True, frozen=True)
class GrepMatch:
    """Regex match returned by :func:`grep`."""

    path: VfsPath
    line_number: int
    line: str


@dataclass(slots=True, frozen=True)
class ListDirectoryParams:
    path: str | None = field(
        default=None,
        metadata={
            "description": (
                "Directory path to list. Provide a relative VFS path or omit to list the root."
            )
        },
    )


@dataclass(slots=True, frozen=True)
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


@dataclass(slots=True, frozen=True)
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


@dataclass(slots=True, frozen=True)
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


@dataclass(slots=True, frozen=True)
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


@dataclass(slots=True, frozen=True)
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


@dataclass(slots=True, frozen=True)
class RemoveParams:
    path: str = field(
        metadata={
            "description": "Relative VFS path targeting the file or directory that should be removed."
        }
    )


@dataclass(slots=True, frozen=True)
class ListDirectory:
    path: VfsPath | None = field(default=None)


@dataclass(slots=True, frozen=True)
class ListDirectoryResult:
    path: VfsPath
    directories: tuple[str, ...]
    files: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ReadFile:
    path: VfsPath


@dataclass(slots=True, frozen=True)
class WriteFile:
    path: VfsPath
    content: str
    mode: WriteMode = "create"
    encoding: FileEncoding = _DEFAULT_ENCODING


@dataclass(slots=True, frozen=True)
class DeleteEntry:
    path: VfsPath


@dataclass(slots=True, frozen=True)
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


@dataclass(slots=True, frozen=True)
class _HostMountPreview:
    host_path: str
    resolved_host: Path
    mount_path: VfsPath
    entries: tuple[str, ...]
    is_directory: bool


@dataclass(slots=True, frozen=True)
class _VfsSectionParams:
    pass


class VfsToolsSection(MarkdownSection[_VfsSectionParams]):
    """Prompt section exposing the virtual filesystem tool suite."""

    def __init__(
        self,
        *,
        mounts: Sequence[HostMount] = (),
        allowed_host_roots: Sequence[os.PathLike[str] | str] = (),
        accepts_overrides: bool = False,
    ) -> None:
        allowed_roots = tuple(_normalize_root(path) for path in allowed_host_roots)
        self._mount_snapshot, mount_previews = _materialize_mounts(
            mounts, allowed_roots
        )
        self._configured_sessions: WeakSet[Session] = WeakSet()

        tools = _build_tools(section=self, accepts_overrides=accepts_overrides)
        super().__init__(
            title="Virtual Filesystem Tools",
            key="vfs.tools",
            template=_render_section_template(mount_previews),
            default_params=_VfsSectionParams(),
            tools=tools,
            accepts_overrides=accepts_overrides,
        )

    def ensure_session(self, context: ToolContext) -> Session:
        session = context.session
        if not isinstance(session, Session):
            raise ToolValidationError(
                "VfsToolsSection requires ToolContext.session to be a Session instance."
            )
        if session not in self._configured_sessions:
            self._initialize_session(session)
            self._configured_sessions.add(session)
        return session

    def _initialize_session(self, session: Session) -> None:
        session.register_reducer(VirtualFileSystem, replace_latest)
        session.seed_slice(VirtualFileSystem, (self._mount_snapshot,))
        session.register_reducer(
            WriteFile,
            _make_write_reducer(),
            slice_type=VirtualFileSystem,
        )
        session.register_reducer(
            DeleteEntry,
            _make_delete_reducer(),
            slice_type=VirtualFileSystem,
        )

    def latest_snapshot(self, session: Session) -> VirtualFileSystem:
        snapshot = select_latest(session, VirtualFileSystem)
        return snapshot or VirtualFileSystem()


def _build_tools(
    *,
    section: VfsToolsSection,
    accepts_overrides: bool,
) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]:
    suite = _VfsToolSuite(section=section)
    return cast(
        tuple[Tool[SupportsDataclass, SupportsDataclass], ...],
        (
            Tool[ListDirectoryParams, tuple[FileInfo, ...]](
                name="ls",
                description="List directory entries under a relative path.",
                handler=suite.list_directory,
                accepts_overrides=accepts_overrides,
            ),
            Tool[ReadFileParams, ReadFileResult](
                name="read_file",
                description="Read UTF-8 file contents with pagination support.",
                handler=suite.read_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[WriteFileParams, WriteFile](
                name="write_file",
                description="Create a new UTF-8 text file.",
                handler=suite.write_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[EditFileParams, WriteFile](
                name="edit_file",
                description="Replace occurrences of a string within a file.",
                handler=suite.edit_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[GlobParams, tuple[GlobMatch, ...]](
                name="glob",
                description="Match files beneath a directory using shell patterns.",
                handler=suite.glob,
                accepts_overrides=accepts_overrides,
            ),
            Tool[GrepParams, tuple[GrepMatch, ...]](
                name="grep",
                description="Search files for a regular expression pattern.",
                handler=suite.grep,
                accepts_overrides=accepts_overrides,
            ),
            Tool[RemoveParams, DeleteEntry](
                name="rm",
                description="Remove files or directories recursively.",
                handler=suite.remove,
                accepts_overrides=accepts_overrides,
            ),
        ),
    )


class _VfsToolSuite:
    """Collection of VFS handlers bound to a section instance."""

    def __init__(self, *, section: VfsToolsSection) -> None:
        super().__init__()
        self._section = section

    def list_directory(
        self, params: ListDirectoryParams, *, context: ToolContext
    ) -> ToolResult[tuple[FileInfo, ...]]:
        session = self._section.ensure_session(context)
        path = _normalize_string_path(params.path, allow_empty=True, field="path")
        snapshot = self._section.latest_snapshot(session)
        if _find_file(snapshot.files, path) is not None:
            raise ToolValidationError("Cannot list a file path; provide a directory.")

        directories: set[tuple[str, ...]] = set()
        files: list[VfsFile] = []
        for file in snapshot.files:
            segments = file.path.segments
            if not _is_path_prefix(segments, path.segments):
                continue
            prefix_length = len(path.segments)
            next_segment = segments[prefix_length]
            subpath = (*path.segments, next_segment)
            if len(segments) == prefix_length + 1:
                files.append(file)
            else:
                directories.add(subpath)

        entries: list[FileInfo] = [
            FileInfo(
                path=file.path,
                kind="file",
                size_bytes=file.size_bytes,
                version=file.version,
                updated_at=file.updated_at,
            )
            for file in files
            if len(file.path.segments) == len(path.segments) + 1
        ]
        entries.extend(
            FileInfo(path=VfsPath(directory), kind="directory")
            for directory in directories
        )

        entries.sort(key=lambda entry: entry.path.segments)
        message = _format_directory_message(path, entries)
        return ToolResult(message=message, value=tuple(entries))

    def read_file(
        self, params: ReadFileParams, *, context: ToolContext
    ) -> ToolResult[ReadFileResult]:
        session = self._section.ensure_session(context)
        path = _normalize_string_path(params.file_path, field="file_path")
        offset = _normalize_offset(params.offset)
        limit = _normalize_limit(params.limit)

        snapshot = self._section.latest_snapshot(session)
        file = _require_file(snapshot.files, path)
        lines = file.content.splitlines()
        total_lines = len(lines)
        start = min(offset, total_lines)
        end = min(start + limit, total_lines)
        numbered = [
            f"{index + 1:>4} | {line}"
            for index, line in enumerate(lines[start:end], start=start)
        ]
        content = "\n".join(numbered)
        message = _format_read_file_message(file, start, end)
        result = ReadFileResult(
            path=file.path,
            content=content,
            offset=start,
            limit=limit,
            total_lines=total_lines,
        )
        return ToolResult(message=message, value=result)

    def write_file(
        self, params: WriteFileParams, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        session = self._section.ensure_session(context)
        path = _normalize_string_path(params.file_path, field="file_path")
        content = _normalize_content(params.content)

        snapshot = self._section.latest_snapshot(session)
        if _find_file(snapshot.files, path) is not None:
            raise ToolValidationError(
                "File already exists; use edit_file to modify existing content."
            )

        normalized = WriteFile(path=path, content=content, mode="create")
        message = _format_write_file_message(path, content, mode="create")
        return ToolResult(message=message, value=normalized)

    def edit_file(
        self, params: EditFileParams, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        session = self._section.ensure_session(context)
        path = _normalize_string_path(params.file_path, field="file_path")
        snapshot = self._section.latest_snapshot(session)
        file = _require_file(snapshot.files, path)

        old = params.old_string
        new = params.new_string
        if not old:
            raise ToolValidationError("old_string must not be empty.")
        if len(old) > _MAX_WRITE_LENGTH or len(new) > _MAX_WRITE_LENGTH:
            raise ToolValidationError(
                "Replacement strings must be 48,000 characters or fewer."
            )

        occurrences = file.content.count(old)
        if occurrences == 0:
            raise ToolValidationError("old_string not found in the target file.")
        if not params.replace_all and occurrences != 1:
            raise ToolValidationError(
                "old_string must match exactly once unless replace_all is true."
            )

        if params.replace_all:
            replacements = occurrences
            updated = file.content.replace(old, new)
        else:
            replacements = 1
            updated = file.content.replace(old, new, 1)

        normalized_content = _normalize_content(updated)
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
        session = self._section.ensure_session(context)
        base = _normalize_string_path(params.path, allow_empty=True, field="path")
        pattern = params.pattern.strip()
        if not pattern:
            raise ToolValidationError("Pattern must not be empty.")
        _ensure_ascii(pattern, "pattern")

        snapshot = self._section.latest_snapshot(session)
        matches: list[GlobMatch] = []
        for file in snapshot.files:
            if not _is_path_prefix(file.path.segments, base.segments):
                continue
            relative_segments = file.path.segments[len(base.segments) :]
            relative = "/".join(relative_segments)
            if fnmatch.fnmatchcase(relative, pattern):
                matches.append(
                    GlobMatch(
                        path=file.path,
                        size_bytes=file.size_bytes,
                        version=file.version,
                        updated_at=file.updated_at,
                    )
                )
        matches.sort(key=lambda match: match.path.segments)
        message = _format_glob_message(base, pattern, matches)
        return ToolResult(message=message, value=tuple(matches))

    def grep(
        self, params: GrepParams, *, context: ToolContext
    ) -> ToolResult[tuple[GrepMatch, ...]]:
        session = self._section.ensure_session(context)
        try:
            pattern = re.compile(params.pattern)
        except re.error as error:
            return ToolResult(
                message=f"Invalid regular expression: {error}",
                value=None,
                success=False,
            )

        base_path: VfsPath | None = None
        if params.path is not None:
            base_path = _normalize_string_path(
                params.path, allow_empty=True, field="path"
            )
        glob_pattern = params.glob.strip() if params.glob is not None else None
        if glob_pattern:
            _ensure_ascii(glob_pattern, "glob")

        snapshot = self._section.latest_snapshot(session)
        matches: list[GrepMatch] = []
        for file in snapshot.files:
            if base_path is not None and not _is_path_prefix(
                file.path.segments, base_path.segments
            ):
                continue
            if glob_pattern:
                relative = (
                    "/".join(file.path.segments[len(base_path.segments) :])
                    if base_path is not None
                    else "/".join(file.path.segments)
                )
                if not fnmatch.fnmatchcase(relative, glob_pattern):
                    continue
            for index, line in enumerate(file.content.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(
                        GrepMatch(
                            path=file.path,
                            line_number=index,
                            line=line,
                        )
                    )
        matches.sort(key=lambda match: (match.path.segments, match.line_number))
        message = _format_grep_message(params.pattern, matches)
        return ToolResult(message=message, value=tuple(matches))

    def remove(
        self, params: RemoveParams, *, context: ToolContext
    ) -> ToolResult[DeleteEntry]:
        session = self._section.ensure_session(context)
        path = _normalize_string_path(params.path, field="path")
        snapshot = self._section.latest_snapshot(session)
        matches = [
            file
            for file in snapshot.files
            if _is_path_prefix(file.path.segments, path.segments)
        ]
        if not matches:
            raise ToolValidationError("No files matched the provided path.")
        normalized = DeleteEntry(path=path)
        message = _format_delete_message(path, matches)
        return ToolResult(message=message, value=normalized)


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


def _normalize_string_path(
    raw: str | None, *, allow_empty: bool = False, field: str
) -> VfsPath:
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


def _find_file(files: Iterable[VfsFile], path: VfsPath) -> VfsFile | None:
    target = path.segments
    for file in files:
        if file.path.segments == target:
            return file
    return None


def _require_file(files: Iterable[VfsFile], path: VfsPath) -> VfsFile:
    file = _find_file(files, path)
    if file is None:
        raise ToolValidationError("File does not exist in the virtual filesystem.")
    return file


def _is_path_prefix(path: Sequence[str], prefix: Sequence[str]) -> bool:
    if len(path) < len(prefix):
        return False
    return all(path[index] == prefix[index] for index in range(len(prefix)))


def _format_directory_message(path: VfsPath, entries: Sequence[FileInfo]) -> str:
    directory_count = sum(1 for entry in entries if entry.kind == "directory")
    file_count = sum(1 for entry in entries if entry.kind == "file")
    prefix = _format_path(path)
    subdir_label = "subdir" if directory_count == 1 else "subdirs"
    file_label = "file" if file_count == 1 else "files"
    return (
        f"Listed directory {prefix} ("
        f"{directory_count} {subdir_label}, {file_count} {file_label})."
    )


def _format_read_file_message(file: VfsFile, start: int, end: int) -> str:
    path_label = _format_path(file.path)
    if start == end:
        return f"Read file {path_label} (no lines returned)."
    return f"Read file {path_label} (lines {start + 1}-{end})."


def _format_write_file_message(path: VfsPath, content: str, mode: WriteMode) -> str:
    path_label = _format_path(path)
    action = {
        "create": "Created",
        "overwrite": "Updated",
        "append": "Appended to",
    }[mode]
    size = len(content.encode(_DEFAULT_ENCODING))
    return f"{action} {path_label} ({size} bytes)."


def _format_edit_message(path: VfsPath, replacements: int) -> str:
    path_label = _format_path(path)
    label = "occurrence" if replacements == 1 else "occurrences"
    return f"Replaced {replacements} {label} in {path_label}."


def _format_glob_message(
    base: VfsPath, pattern: str, matches: Sequence[GlobMatch]
) -> str:
    path_label = _format_path(base)
    match_label = "match" if len(matches) == 1 else "matches"
    return f"Found {len(matches)} {match_label} under {path_label} for pattern '{pattern}'."


def _format_grep_message(pattern: str, matches: Sequence[GrepMatch]) -> str:
    match_label = "match" if len(matches) == 1 else "matches"
    return f"Found {len(matches)} {match_label} for pattern '{pattern}'."


def _format_delete_message(path: VfsPath, files: Sequence[VfsFile]) -> str:
    path_label = _format_path(path)
    entry_label = "entry" if len(files) == 1 else "entries"
    return f"Deleted {len(files)} {entry_label} under {path_label}."


def _format_path(path: VfsPath) -> str:
    return "/".join(path.segments) or "/"


def _normalize_root(path: os.PathLike[str] | str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ToolValidationError("Allowed host root does not exist.")
    return root


def _materialize_mounts(
    mounts: Sequence[HostMount], allowed_roots: Sequence[Path]
) -> tuple[VirtualFileSystem, tuple[_HostMountPreview, ...]]:
    if not mounts:
        return VirtualFileSystem(), ()

    aggregated: dict[tuple[str, ...], VfsFile] = {}
    previews: list[_HostMountPreview] = []
    for mount in mounts:
        loaded, preview = _load_mount(mount, allowed_roots)
        previews.append(preview)
        for file in loaded:
            aggregated[file.path.segments] = file
    files = tuple(sorted(aggregated.values(), key=lambda file: file.path.segments))
    return VirtualFileSystem(files=files), tuple(previews)


def _render_section_template(previews: Sequence[_HostMountPreview]) -> str:
    if not previews:
        return _VFS_SECTION_TEMPLATE

    lines: list[str] = [_VFS_SECTION_TEMPLATE, "", "Configured host mounts:"]
    for preview in previews:
        mount_label = _format_path(preview.mount_path)
        resolved_label = str(preview.resolved_host)
        lines.append(
            f"- Host `{resolved_label}` (configured as `{preview.host_path}`) mounted at `{mount_label}`."
        )
        if preview.is_directory:
            contents = _format_mount_entries(preview.entries)
            lines.append(f"  Contents: {contents}")
        else:
            lines.append(f"  File: `{preview.entries[0]}`")
    return "\n".join(lines)


def _format_mount_entries(entries: Sequence[str]) -> str:
    if not entries:
        return "<empty>"
    preview = entries[:_MAX_MOUNT_PREVIEW_ENTRIES]
    formatted = " ".join(f"`{entry}`" for entry in preview)
    remaining = len(entries) - len(preview)
    if remaining > 0:
        formatted += f" … (+{remaining} more)"
    return formatted


def _load_mount(
    mount: HostMount, allowed_roots: Sequence[Path]
) -> tuple[tuple[VfsFile, ...], _HostMountPreview]:
    host_path = mount.host_path.strip()
    if not host_path:
        raise ToolValidationError("Host mount path must not be empty.")
    _ensure_ascii(host_path, "host path")
    resolved_host = _resolve_mount_path(host_path, allowed_roots)
    include_patterns = _normalize_globs(mount.include_glob, "include_glob")
    exclude_patterns = _normalize_globs(mount.exclude_glob, "exclude_glob")
    mount_prefix = _normalize_optional_path(mount.mount_path)
    preview_entries = _list_mount_entries(resolved_host)
    preview = _HostMountPreview(
        host_path=host_path,
        resolved_host=resolved_host,
        mount_path=mount_prefix,
        entries=preview_entries,
        is_directory=resolved_host.is_dir(),
    )

    files: list[VfsFile] = []
    consumed_bytes = 0
    timestamp = _now()
    for path in _iter_mount_files(resolved_host, mount.follow_symlinks):
        relative = (
            Path(path.name)
            if resolved_host.is_file()
            else path.relative_to(resolved_host)
        )
        relative_posix = relative.as_posix()
        if include_patterns and not any(
            fnmatch.fnmatchcase(relative_posix, pattern) for pattern in include_patterns
        ):
            continue
        if any(
            fnmatch.fnmatchcase(relative_posix, pattern) for pattern in exclude_patterns
        ):
            continue

        try:
            content = path.read_text(encoding=_DEFAULT_ENCODING)
        except UnicodeDecodeError as error:  # pragma: no cover - defensive guard
            raise ToolValidationError("Mounted file must be valid UTF-8.") from error
        except OSError as error:
            raise ToolValidationError(f"Failed to read mounted file {path}.") from error
        size = len(content.encode(_DEFAULT_ENCODING))
        if mount.max_bytes is not None and consumed_bytes + size > mount.max_bytes:
            raise ToolValidationError("Host mount exceeded the configured byte budget.")
        consumed_bytes += size

        segments = mount_prefix.segments + relative.parts
        normalized_path = _normalize_path(VfsPath(segments))
        file = VfsFile(
            path=normalized_path,
            content=content,
            encoding=_DEFAULT_ENCODING,
            size_bytes=size,
            version=1,
            created_at=timestamp,
            updated_at=timestamp,
        )
        files.append(file)
    return tuple(files), preview


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


def _iter_mount_files(root: Path, follow_symlinks: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        current = Path(dirpath)
        for name in filenames:
            yield current / name


def _make_write_reducer() -> TypedReducer[VirtualFileSystem]:
    def reducer(
        slice_values: tuple[VirtualFileSystem, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[VirtualFileSystem, ...]:
        del context
        previous = slice_values[-1] if slice_values else VirtualFileSystem()
        params = cast(WriteFile, event.value)
        timestamp = _now()
        files = list(previous.files)
        existing_index = _index_of(files, params.path)
        existing = files[existing_index] if existing_index is not None else None
        if params.mode == "append" and existing is not None:
            content = existing.content + params.content
            created_at = existing.created_at
            version = existing.version + 1
        elif existing is not None:
            content = params.content
            created_at = existing.created_at
            version = existing.version + 1
        else:
            content = params.content
            created_at = timestamp
            version = 1
        size = len(content.encode(_DEFAULT_ENCODING))
        updated_file = VfsFile(
            path=params.path,
            content=content,
            encoding=_DEFAULT_ENCODING,
            size_bytes=size,
            version=version,
            created_at=_truncate_to_milliseconds(created_at),
            updated_at=_truncate_to_milliseconds(timestamp),
        )
        if existing_index is not None:
            del files[existing_index]
        files.append(updated_file)
        files.sort(key=lambda file: file.path.segments)
        snapshot = VirtualFileSystem(files=tuple(files))
        return (snapshot,)

    return reducer


def _make_delete_reducer() -> TypedReducer[VirtualFileSystem]:
    def reducer(
        slice_values: tuple[VirtualFileSystem, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[VirtualFileSystem, ...]:
        del context
        previous = slice_values[-1] if slice_values else VirtualFileSystem()
        params = cast(DeleteEntry, event.value)
        target = params.path.segments
        files = [
            file
            for file in previous.files
            if not _is_path_prefix(file.path.segments, target)
        ]
        files.sort(key=lambda file: file.path.segments)
        snapshot = VirtualFileSystem(files=tuple(files))
        return (snapshot,)

    return reducer


def _index_of(files: list[VfsFile], path: VfsPath) -> int | None:
    for index, file in enumerate(files):
        if file.path.segments == path.segments:
            return index
    return None


def _now() -> datetime:
    return _truncate_to_milliseconds(datetime.now(UTC))


def _truncate_to_milliseconds(value: datetime) -> datetime:
    microsecond = value.microsecond - (value.microsecond % 1000)
    return value.replace(microsecond=microsecond, tzinfo=UTC)


__all__ = [
    "DeleteEntry",
    "EditFileParams",
    "FileInfo",
    "GlobMatch",
    "GlobParams",
    "GrepMatch",
    "GrepParams",
    "HostMount",
    "ListDirectory",
    "ListDirectoryParams",
    "ListDirectoryResult",
    "ReadFile",
    "ReadFileParams",
    "ReadFileResult",
    "RemoveParams",
    "VfsFile",
    "VfsPath",
    "VfsToolsSection",
    "VirtualFileSystem",
    "WriteFile",
    "WriteFileParams",
]
