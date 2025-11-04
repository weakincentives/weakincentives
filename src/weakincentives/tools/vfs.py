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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast
from weakref import WeakSet

from ..prompt import SupportsDataclass
from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolResult
from ..session import ReducerEvent, Session, replace_latest, select_latest
from .errors import ToolValidationError

FileEncoding = str
WriteMode = Literal["create", "overwrite", "append"]

_ASCII: Final[str] = "ascii"
_DEFAULT_ENCODING: Final[FileEncoding] = "utf-8"
_MAX_WRITE_LENGTH: Final[int] = 48_000
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_SEGMENT_LENGTH: Final[int] = 80
_VFS_SECTION_TEMPLATE: Final[str] = (
    "The virtual filesystem starts empty unless host mounts are configured."
    " Use it to stage edits before applying them to the host workspace.\n"
    "1. Use `vfs_list_directory` to inspect directories before reading or writing"
    " specific files; keep listings focused to reduce output.\n"
    "2. Fetch file contents with `vfs_read_file` and work from the returned version"
    " to avoid conflicts.\n"
    "3. Create or update files with `vfs_write_file`; supply UTF-8 content up to"
    " 48k characters and prefer overwriting full files unless streaming append"
    " updates.\n"
    "4. Remove obsolete files or directories with `vfs_delete_entry` to keep the"
    " snapshot tidy.\n"
    "5. Host mounts are session-initialization only; agents cannot mount additional"
    " directories later.\n"
    "6. Avoid mirroring large repositories or binary assets—only UTF-8 text is"
    " accepted and host mounts remain constrained by their configuration."
)


@dataclass(slots=True, frozen=True)
class VfsPath:
    """Relative POSIX-style path representation."""

    segments: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class VfsFile:
    path: VfsPath
    content: str
    encoding: FileEncoding
    size_bytes: int
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True, frozen=True)
class VirtualFileSystem:
    files: tuple[VfsFile, ...] = field(default_factory=tuple)


@dataclass(slots=True, frozen=True)
class ListDirectory:
    path: VfsPath | None = None


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
    host_path: str
    mount_path: VfsPath | None = None
    include_glob: tuple[str, ...] = field(default_factory=tuple)
    exclude_glob: tuple[str, ...] = field(default_factory=tuple)
    max_bytes: int | None = None
    follow_symlinks: bool = False


@dataclass(slots=True, frozen=True)
class _VfsSectionParams:
    """Placeholder params container for the VFS tools section."""

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
        self._mount_snapshot = _materialize_mounts(mounts, allowed_roots)
        self._configured_sessions: WeakSet[Session] = WeakSet()

        tools = _build_tools(
            section=self,
            accepts_overrides=accepts_overrides,
        )
        super().__init__(
            title="Virtual Filesystem Tools",
            key="vfs.tools",
            template=_VFS_SECTION_TEMPLATE,
            default_params=_VfsSectionParams(),
            tools=tools,
            accepts_overrides=accepts_overrides,
        )

    def ensure_session(self, context: ToolContext) -> Session:
        session = context.session
        if not isinstance(session, Session):
            raise ToolValidationError(
                "VfsToolsSection requires ToolContext.session to be a Session instance.",
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
            Tool[ListDirectory, ListDirectoryResult](
                name="vfs_list_directory",
                description="Enumerate files and directories at a path.",
                handler=suite.list_directory,
                accepts_overrides=accepts_overrides,
            ),
            Tool[ReadFile, VfsFile](
                name="vfs_read_file",
                description="Read file contents and metadata.",
                handler=suite.read_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[WriteFile, WriteFile](
                name="vfs_write_file",
                description="Create or update a file in the virtual filesystem.",
                handler=suite.write_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[DeleteEntry, DeleteEntry](
                name="vfs_delete_entry",
                description="Delete a file or directory subtree.",
                handler=suite.delete_entry,
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
        self, params: ListDirectory, *, context: ToolContext
    ) -> ToolResult[ListDirectoryResult]:
        session = self._section.ensure_session(context)
        target = _normalize_optional_path(params.path)
        snapshot = self._section.latest_snapshot(session)
        if _has_file(snapshot.files, target):
            raise ToolValidationError("Cannot list a file path; provide a directory.")

        directory_names: set[str] = set()
        file_names: set[str] = set()
        prefix_length = len(target.segments)
        for file in snapshot.files:
            segments = file.path.segments
            if not _is_path_prefix(segments, target.segments):
                continue
            next_segment = segments[prefix_length]
            if len(segments) == prefix_length + 1:
                file_names.add(next_segment)
            else:
                directory_names.add(next_segment)

        directories = tuple(sorted(directory_names))
        files = tuple(sorted(file_names))
        normalized = ListDirectoryResult(
            path=target, directories=directories, files=files
        )
        message = _format_directory_message(target, directories, files)
        return ToolResult(message=message, value=normalized)

    def read_file(
        self, params: ReadFile, *, context: ToolContext
    ) -> ToolResult[VfsFile]:
        session = self._section.ensure_session(context)
        path = _normalize_required_path(params.path)
        snapshot = self._section.latest_snapshot(session)
        file = _find_file(snapshot.files, path)
        if file is None:
            raise ToolValidationError("File does not exist in the virtual filesystem.")
        message = _format_read_file_message(file)
        return ToolResult(message=message, value=file)

    def write_file(
        self, params: WriteFile, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        session = self._section.ensure_session(context)
        path = _normalize_required_path(params.path)
        if params.encoding != _DEFAULT_ENCODING:
            raise ToolValidationError("Only UTF-8 encoding is supported.")
        content = _normalize_content(params.content)
        mode = params.mode
        snapshot = self._section.latest_snapshot(session)
        existing = _find_file(snapshot.files, path)
        if mode == "create" and existing is not None:
            raise ToolValidationError("File already exists; use overwrite or append.")
        if mode in {"overwrite", "append"} and existing is None:
            raise ToolValidationError("File does not exist for the requested mode.")
        normalized = WriteFile(path=path, content=content, mode=mode)
        message = _format_write_file_message(path, content, mode)
        return ToolResult(message=message, value=normalized)

    def delete_entry(
        self, params: DeleteEntry, *, context: ToolContext
    ) -> ToolResult[DeleteEntry]:
        session = self._section.ensure_session(context)
        path = _normalize_path(params.path)
        snapshot = self._section.latest_snapshot(session)
        matches = tuple(
            file
            for file in snapshot.files
            if _is_path_prefix(file.path.segments, path.segments)
        )
        deleted_count = len(matches)
        if deleted_count == 0:
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


def _normalize_optional_path(path: VfsPath | None) -> VfsPath:
    if path is None:
        return VfsPath(())
    return _normalize_path(path)


def _normalize_required_path(path: VfsPath) -> VfsPath:
    normalized = _normalize_path(path)
    if not normalized.segments:
        raise ToolValidationError("Path must reference a file or directory.")
    return normalized


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


def _has_file(files: Iterable[VfsFile], path: VfsPath) -> bool:
    return _find_file(files, path) is not None


def _find_file(files: Iterable[VfsFile], path: VfsPath) -> VfsFile | None:
    target = path.segments
    for file in files:
        if file.path.segments == target:
            return file
    return None


def _is_path_prefix(path: Sequence[str], prefix: Sequence[str]) -> bool:
    if len(path) < len(prefix):
        return False
    return all(path[index] == prefix[index] for index in range(len(prefix)))


def _format_directory_message(
    path: VfsPath, directories: tuple[str, ...], files: tuple[str, ...]
) -> str:
    prefix = _format_path(path)
    subdir_label = "subdir" if len(directories) == 1 else "subdirs"
    file_label = "file" if len(files) == 1 else "files"
    return (
        f"Listed directory {prefix} "
        f"({len(directories)} {subdir_label}, {len(files)} {file_label})."
    )


def _format_read_file_message(file: VfsFile) -> str:
    path_label = _format_path(file.path)
    return f"Read file {path_label}."


def _format_write_file_message(path: VfsPath, _content: str, mode: WriteMode) -> str:
    path_label = _format_path(path)
    action = {
        "create": "create",
        "overwrite": "overwrite",
        "append": "append",
    }[mode]
    return f"Staged {action} for {path_label}."


def _format_delete_message(path: VfsPath, files: Sequence[VfsFile]) -> str:
    path_label = _format_path(path)
    entry_label = "entry" if len(files) == 1 else "entries"
    return f"Deleted {len(files)} {entry_label} under {path_label}."


def _format_path(path: VfsPath) -> str:
    return "/".join(path.segments) or "."


def _normalize_root(path: os.PathLike[str] | str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise ToolValidationError("Allowed host root does not exist.")
    return root


def _materialize_mounts(
    mounts: Sequence[HostMount], allowed_roots: Sequence[Path]
) -> VirtualFileSystem:
    if not mounts:
        return VirtualFileSystem()

    aggregated: dict[tuple[str, ...], VfsFile] = {}
    for mount in mounts:
        loaded = _load_mount(mount, allowed_roots)
        for file in loaded:
            aggregated[file.path.segments] = file
    files = tuple(sorted(aggregated.values(), key=lambda file: file.path.segments))
    return VirtualFileSystem(files=files)


def _load_mount(mount: HostMount, allowed_roots: Sequence[Path]) -> tuple[VfsFile, ...]:
    host_path = mount.host_path.strip()
    if not host_path:
        raise ToolValidationError("Host mount path must not be empty.")
    _ensure_ascii(host_path, "host path")
    resolved_host = _resolve_mount_path(host_path, allowed_roots)
    include_patterns = _normalize_globs(mount.include_glob, "include_glob")
    exclude_patterns = _normalize_globs(mount.exclude_glob, "exclude_glob")
    mount_prefix = _normalize_optional_path(mount.mount_path)

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
    return tuple(files)


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


def _make_write_reducer() -> Callable[
    [tuple[VirtualFileSystem, ...], ReducerEvent], tuple[VirtualFileSystem, ...]
]:
    def reducer(
        slice_values: tuple[VirtualFileSystem, ...], event: ReducerEvent
    ) -> tuple[VirtualFileSystem, ...]:
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


def _make_delete_reducer() -> Callable[
    [tuple[VirtualFileSystem, ...], ReducerEvent], tuple[VirtualFileSystem, ...]
]:
    def reducer(
        slice_values: tuple[VirtualFileSystem, ...], event: ReducerEvent
    ) -> tuple[VirtualFileSystem, ...]:
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
    "HostMount",
    "ListDirectory",
    "ListDirectoryResult",
    "ReadFile",
    "VfsFile",
    "VfsPath",
    "VfsToolsSection",
    "VirtualFileSystem",
    "WriteFile",
]
