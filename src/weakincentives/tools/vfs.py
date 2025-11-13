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
import tempfile
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

FileEncoding = Literal["utf-8", "binary"] | str | None
WriteMode = Literal["create", "overwrite", "append"]

_ASCII: Final[str] = "ascii"
_DEFAULT_ENCODING: Final[str] = "utf-8"
_BINARY_ENCODING: Final[str] = "binary"
_MAX_WRITE_LENGTH: Final[int] = 48_000
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_SEGMENT_LENGTH: Final[int] = 80
_TEMP_DIR_PREFIX: Final[str] = "weakincentives-vfs-"
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
    content: bytes = field(
        metadata={
            "description": (
                "Raw byte content of the file. Encoding hints describe how to"
                " interpret the payload when rendering to text."
            )
        }
    )
    encoding: FileEncoding = field(
        metadata={
            "description": (
                "Name of the codec used to decode the file contents. Use 'utf-8'"
                " for text, 'binary' or None for arbitrary bytes."
            )
        }
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

    root_path: str = field(
        metadata={
            "description": "Absolute path to the session's disk-backed workspace root."
        }
    )
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
class ListDirectory:
    """Parameters for enumerating filesystem entries."""

    path: VfsPath | None = field(
        default=None,
        metadata={
            "description": (
                "Directory to inspect. Omit to list the virtual filesystem root."
            )
        },
    )


@dataclass(slots=True, frozen=True)
class ListDirectoryResult:
    """Result payload returned by :func:`vfs_list_directory`."""

    path: VfsPath = field(metadata={"description": "Directory that was enumerated."})
    directories: tuple[str, ...] = field(
        metadata={"description": "Child directory names sorted alphabetically."}
    )
    files: tuple[str, ...] = field(
        metadata={"description": "Child file names sorted alphabetically."}
    )


@dataclass(slots=True, frozen=True)
class ReadFile:
    """Parameters for fetching the contents of a file."""

    path: VfsPath = field(
        metadata={
            "description": "Path to the file to read within the virtual filesystem."
        }
    )


@dataclass(slots=True, frozen=True)
class WriteFile:
    """Parameters for creating or updating a file."""

    path: VfsPath = field(
        metadata={
            "description": "Target path to create or modify within the virtual filesystem."
        }
    )
    content: bytes = field(
        metadata={
            "description": (
                "Raw byte payload to write. Content is limited to 48k bytes per call."
            )
        }
    )
    mode: WriteMode = field(
        default="create",
        metadata={
            "description": (
                "Write strategy: create new file, overwrite existing content, or "
                "append to the end."
            )
        },
    )
    encoding: FileEncoding = field(
        default=_DEFAULT_ENCODING,
        metadata={
            "description": (
                "Encoding hint for the provided content. Defaults to UTF-8 text; "
                "set to 'binary' or None for raw bytes."
            )
        },
    )


@dataclass(slots=True, frozen=True)
class DeleteEntry:
    """Parameters for removing a file or directory."""

    path: VfsPath = field(
        metadata={"description": "Path to delete from the virtual filesystem."}
    )


@dataclass(slots=True, frozen=True)
class HostMount:
    """Mount configuration used to seed the virtual filesystem."""

    host_path: str = field(
        metadata={
            "description": (
                "Absolute path on the host machine that should be exposed in the "
                "virtual filesystem."
            )
        }
    )
    mount_path: VfsPath | None = field(
        default=None,
        metadata={
            "description": (
                "Optional virtual filesystem destination. Defaults to mirroring "
                "the host directory name."
            )
        },
    )
    include_glob: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Glob patterns that must match for files to be mounted. Empty "
                "tuple includes everything."
            )
        },
    )
    exclude_glob: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "description": (
                "Glob patterns used to filter files from the mount. Applied after "
                "includes."
            )
        },
    )
    max_bytes: int | None = field(
        default=None,
        metadata={
            "description": (
                "Optional byte budget for the mounted tree. Files exceeding the "
                "limit are skipped."
            )
        },
    )
    follow_symlinks: bool = field(
        default=False,
        metadata={
            "description": (
                "Whether to resolve symbolic links inside the mounted directory."
            )
        },
    )


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
        self._mount_files = _materialize_mounts(mounts, allowed_roots)
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
        root = _create_session_root()
        _sync_mounts_to_disk(root, self._mount_files)
        initial_snapshot = VirtualFileSystem(
            root_path=str(root),
            files=tuple(self._mount_files),
        )
        session.register_reducer(VirtualFileSystem, replace_latest)
        session.seed_slice(VirtualFileSystem, (initial_snapshot,))
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
        if snapshot is None:
            raise ToolValidationError("Virtual filesystem not initialized for session.")
        return snapshot


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
                description=(
                    "Enumerate files and directories under a relative VFS path. "
                    "Omit the path to list the root."
                ),
                handler=suite.list_directory,
                accepts_overrides=accepts_overrides,
            ),
            Tool[ReadFile, VfsFile](
                name="vfs_read_file",
                description=(
                    "Read the latest version of a file and return its UTF-8 "
                    "contents with metadata."
                ),
                handler=suite.read_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[WriteFile, WriteFile](
                name="vfs_write_file",
                description=(
                    "Create or update a UTF-8 text file. Supply the target path, "
                    "content, and write mode."
                ),
                handler=suite.write_file,
                accepts_overrides=accepts_overrides,
            ),
            Tool[DeleteEntry, DeleteEntry](
                name="vfs_delete_entry",
                description=(
                    "Delete a file or directory tree from the virtual filesystem. "
                    "Directories are removed recursively."
                ),
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
        encoding = _normalize_encoding(params.encoding)
        payload = _normalize_content(params.content, encoding)
        mode = params.mode
        snapshot = self._section.latest_snapshot(session)
        existing = _find_file(snapshot.files, path)
        if mode == "create" and existing is not None:
            raise ToolValidationError("File already exists; use overwrite or append.")
        if mode in {"overwrite", "append"} and existing is None:
            raise ToolValidationError("File does not exist for the requested mode.")
        if mode == "append" and existing is not None:
            final_content = existing.content + payload
            prior_encoding = existing.encoding
            message_size = len(payload)
        else:
            final_content = payload
            prior_encoding = existing.encoding if existing is not None else None
            message_size = len(final_content)
        resolved_encoding = encoding if encoding is not None else prior_encoding
        root = Path(snapshot.root_path)
        _write_bytes_to_disk(root, path, final_content)
        normalized = WriteFile(
            path=path,
            content=final_content,
            mode=mode,
            encoding=resolved_encoding,
        )
        message = _format_write_file_message(
            path, message_size, mode, resolved_encoding
        )
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
        root = Path(snapshot.root_path)
        _delete_files_from_disk(root, matches)
        normalized = DeleteEntry(path=path)
        message = _format_delete_message(path, matches)
        return ToolResult(message=message, value=normalized)


def _normalize_encoding(encoding: FileEncoding) -> FileEncoding:
    if encoding is None:
        return None
    normalized = encoding.strip()
    if not normalized:
        return None
    _ensure_ascii(normalized, "encoding")
    lowered = normalized.lower()
    if lowered == _DEFAULT_ENCODING:
        return _DEFAULT_ENCODING
    if lowered == _BINARY_ENCODING:
        return _BINARY_ENCODING
    return lowered


def _normalize_content(content: bytes, encoding: FileEncoding) -> bytes:
    if encoding == _DEFAULT_ENCODING:
        try:
            decoded = content.decode(_DEFAULT_ENCODING)
        except UnicodeDecodeError as error:
            raise ToolValidationError(
                "Content is not valid UTF-8 for the requested encoding."
            ) from error
        if len(decoded) > _MAX_WRITE_LENGTH:
            raise ToolValidationError(
                "Content exceeds maximum length of 48,000 characters."
            )
        return content

    if len(content) > _MAX_WRITE_LENGTH:
        raise ToolValidationError("Content exceeds maximum length of 48,000 bytes.")
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
    encoding = file.encoding or _BINARY_ENCODING
    return (
        f"Read file {path_label} (size={file.size_bytes} bytes, encoding={encoding})."
    )


def _format_write_file_message(
    path: VfsPath, size_bytes: int, mode: WriteMode, encoding: FileEncoding
) -> str:
    path_label = _format_path(path)
    action = {
        "create": "create",
        "overwrite": "overwrite",
        "append": "append",
    }[mode]
    encoding_label = encoding or _BINARY_ENCODING
    if mode == "append":
        size_label = f"appended {size_bytes} bytes"
    else:
        size_label = f"{size_bytes} bytes"
    return (
        f"Staged {action} for {path_label} ({size_label}, encoding={encoding_label})."
    )


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
) -> tuple[VfsFile, ...]:
    if not mounts:
        return ()

    aggregated: dict[tuple[str, ...], VfsFile] = {}
    for mount in mounts:
        loaded = _load_mount(mount, allowed_roots)
        for file in loaded:
            aggregated[file.path.segments] = file
    return tuple(sorted(aggregated.values(), key=lambda file: file.path.segments))


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
            content = path.read_bytes()
        except OSError as error:
            raise ToolValidationError(f"Failed to read mounted file {path}.") from error
        try:
            _ = content.decode(_DEFAULT_ENCODING)
            encoding: FileEncoding = _DEFAULT_ENCODING
        except UnicodeDecodeError:
            encoding = _BINARY_ENCODING
        size = len(content)
        if mount.max_bytes is not None and consumed_bytes + size > mount.max_bytes:
            raise ToolValidationError("Host mount exceeded the configured byte budget.")
        consumed_bytes += size

        segments = mount_prefix.segments + relative.parts
        normalized_path = _normalize_path(VfsPath(segments))
        file = VfsFile(
            path=normalized_path,
            content=content,
            encoding=encoding,
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


def _create_session_root() -> Path:
    return Path(tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX))


def _sync_mounts_to_disk(root: Path, files: Sequence[VfsFile]) -> None:
    for file in files:
        _write_bytes_to_disk(root, file.path, file.content)


def _write_bytes_to_disk(root: Path, path: VfsPath, content: bytes) -> None:
    target = _resolve_disk_path(root, path)
    try:
        _ = target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_bytes(content)
    except OSError as error:
        raise ToolValidationError(f"Failed to write VFS file {target}.") from error


def _delete_files_from_disk(root: Path, files: Sequence[VfsFile]) -> None:
    for file in files:
        target = _resolve_disk_path(root, file.path)
        try:
            target.unlink()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ToolValidationError(f"Failed to delete VFS file {target}.") from error
        _prune_empty_parents(root, target.parent)


def _resolve_disk_path(root: Path, path: VfsPath) -> Path:
    return root.joinpath(*path.segments)


def _prune_empty_parents(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return
    parts = relative.parts
    for index in range(len(parts), 0, -1):
        current = root.joinpath(*parts[:index])
        if not current.exists():
            continue
        try:
            current.rmdir()
        except OSError:
            break


def _make_write_reducer() -> TypedReducer[VirtualFileSystem]:
    def reducer(
        slice_values: tuple[VirtualFileSystem, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[VirtualFileSystem, ...]:
        del context
        if not slice_values:
            raise RuntimeError("Virtual filesystem reducer invoked without state.")
        previous = slice_values[-1]
        params = cast(WriteFile, event.value)
        timestamp = _now()
        files = list(previous.files)
        existing_index = _index_of(files, params.path)
        existing = files[existing_index] if existing_index is not None else None
        if existing is not None:
            created_at = existing.created_at
            version = existing.version + 1
        else:
            created_at = timestamp
            version = 1
        content = params.content
        resolved_encoding = params.encoding
        if resolved_encoding is None and existing is not None:
            resolved_encoding = existing.encoding
        size = len(content)
        updated_file = VfsFile(
            path=params.path,
            content=content,
            encoding=resolved_encoding,
            size_bytes=size,
            version=version,
            created_at=_truncate_to_milliseconds(created_at),
            updated_at=_truncate_to_milliseconds(timestamp),
        )
        if existing_index is not None:
            del files[existing_index]
        files.append(updated_file)
        files.sort(key=lambda file: file.path.segments)
        snapshot = VirtualFileSystem(root_path=previous.root_path, files=tuple(files))
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
        if not slice_values:
            raise RuntimeError("Virtual filesystem reducer invoked without state.")
        previous = slice_values[-1]
        params = cast(DeleteEntry, event.value)
        target = params.path.segments
        files = [
            file
            for file in previous.files
            if not _is_path_prefix(file.path.segments, target)
        ]
        files.sort(key=lambda file: file.path.segments)
        snapshot = VirtualFileSystem(root_path=previous.root_path, files=tuple(files))
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
