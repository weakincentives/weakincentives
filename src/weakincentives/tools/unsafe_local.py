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

"""Unsafe local sandbox tool surface for containerized environments.

This module provides the same tool surface as :class:`PodmanSandboxSection`
but operates directly on the local filesystem using a temporary directory.
It is intended for environments where the agent already runs inside an
isolated container, eliminating the need for nested container orchestration.
"""

from __future__ import annotations

import fnmatch
import math
import os
import posixpath
import re
import shutil
import subprocess  # nosec: B404
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, override

from ..dataclasses import FrozenDataclass
from ..prompt.markdown import MarkdownSection
from ..prompt.tool import Tool, ToolContext, ToolExample, ToolResult
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session import Session, replace_latest
from . import vfs as vfs_module
from ._context import ensure_context_uses_session
from .asteval import (
    EvalParams,
    EvalResult,
    make_eval_result_reducer,
)
from .errors import ToolValidationError
from .vfs import (
    DeleteEntry,
    EditFileParams,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    HostMount,
    ListDirectoryParams,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    VfsPath,
    VirtualFileSystem,
    WriteFile,
    WriteFileParams,
)

_LOGGER: StructuredLogger = get_logger(
    __name__, context={"component": "tools.unsafe_local"}
)

_DEFAULT_WORKDIR: Final[str] = "/workspace"
_MAX_STDIO_CHARS: Final[int] = 32 * 1024
_MAX_COMMAND_LENGTH: Final[int] = 4_096
_MAX_ENV_LENGTH: Final[int] = 512
_MAX_ENV_VARS: Final[int] = 64
_MAX_TIMEOUT: Final[float] = 120.0
_MIN_TIMEOUT: Final[float] = 1.0
_DEFAULT_TIMEOUT: Final[float] = 30.0
_EVAL_TIMEOUT_SECONDS: Final[float] = 5.0
_MAX_PATH_DEPTH: Final[int] = 16
_MAX_PATH_SEGMENT: Final[int] = 80
_EVAL_MAX_STREAM_LENGTH: Final[int] = 4_096
_ASCII: Final[str] = "ascii"
_LOWEST_PRINTABLE_CODEPOINT: Final[int] = 32
_ALLOWED_CONTROL_CHARACTERS: Final[tuple[str, str]] = ("\n", "\t")
_CAPTURE_DISABLED: Final[str] = "capture disabled"
_WORKSPACE_ROOT_ENV: Final[str] = "WEAKINCENTIVES_WORKSPACE_ROOT"
_SHELL_TIMEOUT_ENV: Final[str] = "WEAKINCENTIVES_SHELL_TIMEOUT"
_MAX_MATCH_RESULTS: Final[int] = 2_000
_REMOVE_PATH_SCRIPT: Final[str] = """
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1])
if not target.exists():
    raise SystemExit(3)
if target.is_symlink():
    target.unlink()
elif target.is_dir():
    shutil.rmtree(target)
else:
    target.unlink()
"""
_UNSAFE_LOCAL_TEMPLATE: Final[str] = """\
You have access to a local workspace directory. The `ls`, `read_file`,
`write_file`, `glob`, `grep`, and `rm` tools operate on `/workspace` inside
the temporary directory. The `evaluate_python` tool executes Python via
`python3 -c` (<=5 seconds). `shell_execute` runs commands directly on the host
(<=120 seconds). No container isolation is applied--this sandbox relies on
external isolation. Do not assume files outside `/workspace` are accessible."""


@FrozenDataclass()
class _UnsafeLocalSectionParams:
    workspace_root: str = _DEFAULT_WORKDIR


@FrozenDataclass()
class _SubprocessOptions:
    """Internal options for subprocess execution."""

    command: Sequence[str]
    cwd: str | None = None
    environment: Mapping[str, str] | None = None
    stdin: str | None = None
    timeout: float | None = None
    capture_output: bool = True


@FrozenDataclass()
class UnsafeLocalSandboxConfig:
    """Configuration for :class:`UnsafeLocalSandboxSection`."""

    mounts: Sequence[HostMount] = ()
    allowed_host_roots: Sequence[os.PathLike[str] | str] = ()
    base_environment: Mapping[str, str] | None = None
    workspace_root: os.PathLike[str] | str | None = None
    clock: Callable[[], datetime] | None = None
    accepts_overrides: bool = False


@FrozenDataclass()
class LocalShellParams:
    """Parameter payload accepted by the ``shell_execute`` tool."""

    command: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=lambda: dict[str, str]())
    stdin: str | None = None
    timeout_seconds: float = _DEFAULT_TIMEOUT
    capture_output: bool = True


@FrozenDataclass()
class LocalShellResult:
    """Structured command summary returned by the ``shell_execute`` tool."""

    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool

    def render(self) -> str:
        command_str = " ".join(self.command)
        lines = [
            "Shell command result:",
            f"Command: {command_str}",
            f"CWD: {self.cwd}",
            f"Exit code: {self.exit_code}",
            f"Timed out: {self.timed_out}",
            f"Duration: {self.duration_ms} ms",
            "STDOUT:",
            self.stdout or "<empty>",
            "STDERR:",
            self.stderr or "<empty>",
        ]
        return "\n".join(lines)


@FrozenDataclass()
class _ShellOutcome:
    """Internal capture of subprocess execution result."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


@FrozenDataclass()
class UnsafeLocalWorkspace:
    """Active local workspace backing the session."""

    workspace_path: str
    workdir: str
    env: tuple[tuple[str, str], ...]
    started_at: datetime
    last_used_at: datetime


@FrozenDataclass()
class _ResolvedHostMount:
    source_label: str
    resolved_host: Path
    mount_path: VfsPath
    include_glob: tuple[str, ...]
    exclude_glob: tuple[str, ...]
    max_bytes: int | None
    follow_symlinks: bool
    preview: vfs_module.HostMountPreview


def _default_workspace_root() -> Path:
    override = os.environ.get(_WORKSPACE_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    return Path(tempfile.gettempdir())


def _resolve_local_host_mounts(
    mounts: Sequence[HostMount],
    allowed_roots: Sequence[Path],
) -> tuple[tuple[_ResolvedHostMount, ...], tuple[vfs_module.HostMountPreview, ...]]:
    if not mounts:
        return (), ()
    resolved: list[_ResolvedHostMount] = []
    previews: list[vfs_module.HostMountPreview] = []
    for mount in mounts:
        spec = _resolve_single_host_mount(mount, allowed_roots)
        resolved.append(spec)
        previews.append(spec.preview)
    return tuple(resolved), tuple(previews)


def _resolve_single_host_mount(
    mount: HostMount,
    allowed_roots: Sequence[Path],
) -> _ResolvedHostMount:
    host_path = mount.host_path.strip()
    if not host_path:
        raise ToolValidationError("Host mount path must not be empty.")
    vfs_module.ensure_ascii(host_path, "host path")
    resolved_host = _resolve_host_path(host_path, allowed_roots)
    include_glob = _normalize_mount_globs(mount.include_glob, "include_glob")
    exclude_glob = _normalize_mount_globs(mount.exclude_glob, "exclude_glob")
    mount_path = (
        vfs_module.normalize_path(mount.mount_path)
        if mount.mount_path is not None
        else VfsPath(())
    )
    preview_entries = _preview_mount_entries(resolved_host)
    preview = vfs_module.HostMountPreview(
        host_path=host_path,
        resolved_host=resolved_host,
        mount_path=mount_path,
        entries=preview_entries,
        is_directory=resolved_host.is_dir(),
    )
    return _ResolvedHostMount(
        source_label=host_path,
        resolved_host=resolved_host,
        mount_path=mount_path,
        include_glob=include_glob,
        exclude_glob=exclude_glob,
        max_bytes=mount.max_bytes,
        follow_symlinks=mount.follow_symlinks,
        preview=preview,
    )


def _resolve_host_path(host_path: str, allowed_roots: Sequence[Path]) -> Path:
    if not allowed_roots:
        raise ToolValidationError("No allowed host roots configured for mounts.")
    for root in allowed_roots:
        candidate = (root / host_path).expanduser().resolve()
        try:
            _ = candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.exists():
            return candidate
    raise ToolValidationError("Host path is outside the allowed roots or missing.")


def _normalize_mount_globs(patterns: Sequence[str], field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for pattern in patterns:
        stripped = pattern.strip()
        if not stripped:
            continue
        vfs_module.ensure_ascii(stripped, field)
        normalized.append(stripped)
    return tuple(normalized)


def _preview_mount_entries(root: Path) -> tuple[str, ...]:
    if root.is_file():
        return (root.name,)
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name.lower())
    except OSError as error:
        raise ToolValidationError(f"Failed to inspect host mount {root}.") from error
    labels: list[str] = []
    for child in children:
        suffix = "/" if child.is_dir() else ""
        labels.append(f"{child.name}{suffix}")
    return tuple(labels)


def _iter_host_mount_files(root: Path, follow_symlinks: bool) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for current, _dirnames, filenames in root.walk(
        follow_symlinks=follow_symlinks,
    ):
        for name in filenames:
            yield current / name


def _ensure_ascii(value: str, *, field: str) -> str:
    try:
        _ = value.encode(_ASCII)
    except UnicodeEncodeError as error:
        raise ToolValidationError(f"{field} must be ASCII.") from error
    return value


def _normalize_command(command: tuple[str, ...]) -> tuple[str, ...]:
    if not command:
        raise ToolValidationError("command must contain at least one entry.")
    total_length = 0
    normalized: list[str] = []
    for index, entry in enumerate(command):
        if not entry:
            raise ToolValidationError(f"command[{index}] must not be empty.")
        normalized_entry = _ensure_ascii(entry, field="command")
        total_length += len(normalized_entry)
        if total_length > _MAX_COMMAND_LENGTH:
            raise ToolValidationError("command is too long (limit 4,096 characters).")
        normalized.append(normalized_entry)
    return tuple(normalized)


def _normalize_env(env: Mapping[str, str]) -> dict[str, str]:
    if len(env) > _MAX_ENV_VARS:
        raise ToolValidationError("env contains too many entries (max 64).")
    normalized: dict[str, str] = {}
    for key, value in env.items():
        normalized_key = _ensure_ascii(key, field="env key").upper()
        if not normalized_key:
            raise ToolValidationError("env keys must not be empty.")
        if len(normalized_key) > _MAX_PATH_SEGMENT:
            raise ToolValidationError(
                f"env key {normalized_key!r} is longer than {_MAX_PATH_SEGMENT} characters."
            )
        normalized_value = _ensure_ascii(value, field="env value")
        if len(normalized_value) > _MAX_ENV_LENGTH:
            raise ToolValidationError(
                f"env value for {normalized_key!r} exceeds {_MAX_ENV_LENGTH} characters."
            )
        normalized[normalized_key] = normalized_value
    return normalized


def _normalize_timeout(timeout_seconds: float) -> float:
    if math.isnan(timeout_seconds):
        raise ToolValidationError("timeout_seconds must be a real number.")
    return max(_MIN_TIMEOUT, min(_MAX_TIMEOUT, timeout_seconds))


def _normalize_cwd(path: str | None, workspace_root: Path) -> str:
    if path is None or path == "":
        return str(workspace_root)
    stripped = path.strip()
    if stripped.startswith("/"):
        raise ToolValidationError("cwd must be relative to /workspace.")
    parts = [segment for segment in stripped.split("/") if segment]
    if len(parts) > _MAX_PATH_DEPTH:
        raise ToolValidationError("cwd exceeds maximum depth of 16 segments.")
    normalized_segments: list[str] = []
    for segment in parts:
        if segment in {".", ".."}:
            raise ToolValidationError("cwd must not contain '.' or '..' segments.")
        if len(segment) > _MAX_PATH_SEGMENT:
            raise ToolValidationError(
                f"cwd segment {segment!r} exceeds {_MAX_PATH_SEGMENT} characters."
            )
        normalized_segment = _ensure_ascii(segment, field="cwd")
        normalized_segments.append(normalized_segment)
    if not normalized_segments:  # pragma: no cover - defensive fallback
        return str(workspace_root)
    return str(workspace_root / Path(*normalized_segments))


def _truncate_stream(value: str) -> str:
    if len(value) <= _MAX_STDIO_CHARS:
        return value
    truncated = value[: _MAX_STDIO_CHARS - len("[truncated]")]
    return f"{truncated}[truncated]"


def _truncate_eval_stream(value: str) -> str:
    if len(value) <= _EVAL_MAX_STREAM_LENGTH:
        return value
    suffix = "..."
    keep = _EVAL_MAX_STREAM_LENGTH - len(suffix)
    return f"{value[:keep]}{suffix}"


def _normalize_local_eval_code(code: str) -> str:
    for char in code:
        code_point = ord(char)
        if (
            code_point < _LOWEST_PRINTABLE_CODEPOINT
            and char not in _ALLOWED_CONTROL_CHARACTERS
        ):
            raise ToolValidationError("Code contains unsupported control characters.")
    return code


class UnsafeLocalSandboxSection(MarkdownSection[_UnsafeLocalSectionParams]):
    """Prompt section exposing the unsafe local sandbox tool suite.

    This section provides the same tool surface as :class:`PodmanSandboxSection`
    but operates directly on the local filesystem using a temporary directory.
    """

    def __init__(
        self, *, session: Session, config: UnsafeLocalSandboxConfig | None = None
    ) -> None:
        config = config or UnsafeLocalSandboxConfig()
        self._session = session
        self._mounts = tuple(config.mounts)
        self._base_env = tuple(
            sorted((config.base_environment or {}).items(), key=lambda item: item[0])
        )
        self._workspace_root = (
            Path(config.workspace_root).expanduser()
            if config.workspace_root is not None
            else _default_workspace_root()
        )
        allowed_roots = tuple(
            vfs_module.normalize_host_root(path) for path in config.allowed_host_roots
        )
        self._allowed_roots = allowed_roots
        (
            self._resolved_mounts,
            self._mount_previews,
        ) = _resolve_local_host_mounts(self._mounts, self._allowed_roots)
        self._mount_snapshot = VirtualFileSystem()
        self._clock = config.clock or (lambda: datetime.now(UTC))
        self._workspace_handle: _WorkspaceHandle | None = None
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(
            self, UnsafeLocalSandboxSection._cleanup_from_finalizer, weakref.ref(self)
        )
        self._config = UnsafeLocalSandboxConfig(
            mounts=self._mounts,
            allowed_host_roots=self._allowed_roots,
            base_environment=dict(self._base_env),
            workspace_root=self._workspace_root,
            clock=self._clock,
            accepts_overrides=config.accepts_overrides,
        )

        session.mutate(UnsafeLocalWorkspace).register(
            UnsafeLocalWorkspace, replace_latest
        )
        self._initialize_vfs_state(session)
        session.mutate(VirtualFileSystem).register(
            EvalResult, make_eval_result_reducer()
        )

        self._vfs_suite = _LocalVfsSuite(section=self)
        self._shell_suite = _LocalShellSuite(section=self)
        self._eval_suite = _LocalEvalSuite(section=self)
        accepts_overrides = config.accepts_overrides
        tools = (
            Tool[ListDirectoryParams, tuple[FileInfo, ...]](
                name="ls",
                description="List directory entries under a relative path.",
                handler=self._vfs_suite.list_directory,
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
                handler=self._vfs_suite.read_file,
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
                handler=self._vfs_suite.write_file,
                examples=(
                    ToolExample[WriteFileParams, WriteFile](
                        description="Create a notes file in the workspace",
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
                handler=self._vfs_suite.edit_file,
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
                handler=self._vfs_suite.glob,
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
                handler=self._vfs_suite.grep,
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
                handler=self._vfs_suite.remove,
                examples=(
                    ToolExample[RemoveParams, DeleteEntry](
                        description="Delete a stale build artifact",
                        input=RemoveParams(path="/workspace/build/output"),
                        output=DeleteEntry(
                            path=VfsPath(("workspace", "build", "output"))
                        ),
                    ),
                ),
                accepts_overrides=accepts_overrides,
            ),
            Tool[LocalShellParams, LocalShellResult](
                name="shell_execute",
                description="Run a short command in the local workspace.",
                handler=self._shell_suite.run_shell,
                examples=(
                    ToolExample[LocalShellParams, LocalShellResult](
                        description="Check the current working directory",
                        input=LocalShellParams(command=("pwd",), cwd=None),
                        output=LocalShellResult(
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
                    "Run a short Python script via `python3 -c` in the local workspace. "
                    "Captures stdout/stderr and reports the exit code."
                ),
                handler=self._eval_suite.evaluate_python,
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
        template = _UNSAFE_LOCAL_TEMPLATE
        mounts_block = vfs_module.render_host_mounts_block(self._mount_previews)
        if mounts_block:
            template = f"{_UNSAFE_LOCAL_TEMPLATE}\n\n{mounts_block}"
        super().__init__(
            title="Unsafe Local Workspace",
            key="unsafe_local.shell",
            template=template,
            default_params=_UnsafeLocalSectionParams(workspace_root=_DEFAULT_WORKDIR),
            tools=tools,
            accepts_overrides=accepts_overrides,
        )

    @property
    def session(self) -> Session:
        return self._session

    @override
    def clone(self, **kwargs: object) -> UnsafeLocalSandboxSection:
        session = kwargs.get("session")
        if not isinstance(session, Session):
            msg = "session is required to clone UnsafeLocalSandboxSection."
            raise TypeError(msg)
        provided_bus = kwargs.get("bus")
        if provided_bus is not None and provided_bus is not session.event_bus:
            msg = "Provided bus must match the target session's event bus."
            raise TypeError(msg)
        return UnsafeLocalSandboxSection(session=session, config=self._config)

    def _initialize_vfs_state(self, session: Session) -> None:
        session.mutate(VirtualFileSystem).register(VirtualFileSystem, replace_latest)
        session.mutate(VirtualFileSystem).seed(self._mount_snapshot)
        session.mutate(VirtualFileSystem).register(
            WriteFile, vfs_module.make_write_reducer()
        )
        session.mutate(VirtualFileSystem).register(
            DeleteEntry, vfs_module.make_delete_reducer()
        )

    def latest_snapshot(self) -> VirtualFileSystem:
        snapshot = self._session.query(VirtualFileSystem).latest()
        return snapshot or self._mount_snapshot

    def _ensure_workspace(self) -> _WorkspaceHandle:
        with self._lock:
            if self._workspace_handle is not None:
                return self._workspace_handle
            handle = self._create_workspace()
            self._workspace_handle = handle
            self._session.mutate(UnsafeLocalWorkspace).seed(handle.descriptor)
            return handle

    def _create_workspace(self) -> _WorkspaceHandle:
        workspace_dir = self._workspace_root / f"wink-{self._session.session_id}"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self._hydrate_workspace_mounts(workspace_dir)
        _LOGGER.info(
            "Creating local workspace",
            event="unsafe_local.workspace.create",
            context={"workspace": str(workspace_dir)},
        )
        env = _normalize_env(dict(self._base_env))
        now = self._clock().astimezone(UTC)
        descriptor = UnsafeLocalWorkspace(
            workspace_path=str(workspace_dir),
            workdir=_DEFAULT_WORKDIR,
            env=tuple(sorted(env.items())),
            started_at=now,
            last_used_at=now,
        )
        return _WorkspaceHandle(descriptor=descriptor, workspace_path=workspace_dir)

    def _hydrate_workspace_mounts(self, workspace_dir: Path) -> None:
        if not self._resolved_mounts:
            return
        iterator = workspace_dir.iterdir()
        try:
            _ = next(iterator)
        except StopIteration:
            pass
        else:
            return  # pragma: no cover - skip if already hydrated
        for mount in self._resolved_mounts:
            self._copy_mount_into_workspace(workspace=workspace_dir, mount=mount)

    def _workspace_env(self) -> dict[str, str]:
        return (
            dict(self._workspace_handle.descriptor.env)
            if self._workspace_handle
            else dict(self._base_env)
        )

    @staticmethod
    def _copy_mount_into_workspace(
        *,
        workspace: Path,
        mount: _ResolvedHostMount,
    ) -> None:
        base_target = _host_path_for(workspace, mount.mount_path)
        consumed_bytes = 0
        source = mount.resolved_host
        for file_path in _iter_host_mount_files(source, mount.follow_symlinks):
            relative = (
                Path(file_path.name)
                if source.is_file()
                else file_path.relative_to(source)
            )
            relative_label = relative.as_posix()
            if mount.include_glob and not any(
                fnmatch.fnmatchcase(relative_label, pattern)
                for pattern in mount.include_glob
            ):
                continue
            if any(
                fnmatch.fnmatchcase(relative_label, pattern)
                for pattern in mount.exclude_glob
            ):
                continue
            target = base_target / relative
            _assert_within_workspace(workspace, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = file_path.stat().st_size
            except OSError as error:  # pragma: no cover - race condition
                raise ToolValidationError(
                    f"Failed to stat mounted file {file_path}."
                ) from error
            if mount.max_bytes is not None and consumed_bytes + size > mount.max_bytes:
                raise ToolValidationError(
                    "Host mount exceeded the configured byte budget."
                )
            consumed_bytes += size
            try:
                _ = shutil.copy2(file_path, target)
            except OSError as error:
                raise ToolValidationError(
                    "Failed to materialize host mounts inside the local workspace."
                ) from error

    def _touch_workspace(self) -> None:
        with self._lock:
            handle = self._workspace_handle
            if handle is None:
                return
            now = self._clock().astimezone(UTC)
            updated_descriptor = replace(handle.descriptor, last_used_at=now)
            self._workspace_handle = _WorkspaceHandle(
                descriptor=updated_descriptor,
                workspace_path=handle.workspace_path,
            )
            self._session.mutate(UnsafeLocalWorkspace).seed(updated_descriptor)

    def _teardown_workspace(self) -> None:
        with self._lock:
            handle = self._workspace_handle
            self._workspace_handle = None
        if handle is None:
            return
        with suppress(OSError):
            shutil.rmtree(handle.workspace_path)

    def ensure_workspace(self) -> _WorkspaceHandle:
        return self._ensure_workspace()

    def current_workspace(self) -> _WorkspaceHandle | None:
        """Return the current workspace handle, or None if not yet created."""
        return self._workspace_handle

    def workspace_environment(self) -> dict[str, str]:
        return self._workspace_env()

    def touch_workspace(self) -> None:
        self._touch_workspace()

    def run_subprocess(
        self,
        options: _SubprocessOptions,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a subprocess using the workspace environment."""
        handle = self.ensure_workspace()
        env = self.workspace_environment()
        if options.environment:
            env.update(options.environment)
        working_dir = options.cwd if options.cwd else str(handle.workspace_path)

        full_env = {**os.environ, **env}
        return subprocess.run(  # nosec: B603
            list(options.command),
            input=options.stdin,
            text=True,
            capture_output=options.capture_output,
            timeout=options.timeout,
            cwd=working_dir,
            env=full_env,
        )

    def run_python_script(
        self,
        *,
        script: str,
        args: Sequence[str],
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a Python script using python3 -c."""
        handle = self.ensure_workspace()
        return self.run_subprocess(
            _SubprocessOptions(
                command=["python3", "-c", script, *args],
                cwd=str(handle.workspace_path),
                timeout=timeout,
            )
        )

    def write_file_to_workspace(
        self,
        *,
        path: VfsPath,
        content: str,
        mode: str,
    ) -> None:
        handle = self.ensure_workspace()
        host_path = _host_path_for(handle.workspace_path, path)
        payload = content
        if mode == "append" and host_path.exists():  # pragma: no cover - append mode
            try:
                existing = host_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ToolValidationError("File is not valid UTF-8.") from error
            except OSError as error:
                raise ToolValidationError("Failed to read existing file.") from error
            payload = f"{existing}{content}"
        parent = host_path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        try:
            _ = host_path.write_text(payload, encoding="utf-8")
        except OSError as error:  # pragma: no cover - filesystem error
            raise ToolValidationError("Failed to write file.") from error

    def close(self) -> None:
        finalizer = self._finalizer
        if finalizer.alive:
            _ = finalizer()

    @staticmethod
    def _cleanup_from_finalizer(
        section_ref: weakref.ReferenceType[UnsafeLocalSandboxSection],
    ) -> None:
        section = section_ref()
        if section is not None:
            section._teardown_workspace()


@FrozenDataclass()
class _WorkspaceHandle:
    """Internal handle referencing the active workspace."""

    descriptor: UnsafeLocalWorkspace
    workspace_path: Path


def _host_path_for(root: Path, path: VfsPath) -> Path:
    host = root
    for segment in path.segments:
        host /= segment
    return host


def _assert_within_workspace(root: Path, candidate: Path) -> None:
    try:
        resolved = candidate.resolve()
    except FileNotFoundError:
        try:
            resolved = candidate.parent.resolve()
        except FileNotFoundError as error:
            raise ToolValidationError("Workspace path is unavailable.") from error
    try:
        _ = resolved.relative_to(root)
    except ValueError as error:
        raise ToolValidationError("Path escapes the workspace boundary.") from error


def _compose_child_path(base: VfsPath, name: str) -> VfsPath | None:
    candidate = VfsPath((*base.segments, name))
    try:
        return vfs_module.normalize_path(candidate)
    except ToolValidationError:  # pragma: no cover - invalid path component
        return None


def _compose_relative_path(base: VfsPath, relative: Path) -> VfsPath | None:
    segments = (*base.segments, *relative.parts)
    candidate = VfsPath(segments)
    try:
        return vfs_module.normalize_path(candidate)
    except ToolValidationError:  # pragma: no cover - invalid path component
        return None


def _iter_workspace_files(base: Path) -> Iterator[Path]:
    if not base.exists():  # pragma: no cover - deleted during iteration
        return
    for dirpath, _, filenames in os.walk(base, followlinks=False):
        current = Path(dirpath)
        for name in filenames:
            yield current / name


def _stat_file(path: Path) -> tuple[int, datetime]:
    try:
        stat_result = path.stat()
    except OSError as error:  # pragma: no cover - filesystem error
        raise ToolValidationError("Failed to stat workspace file.") from error
    size = stat_result.st_size
    updated_at = datetime.fromtimestamp(stat_result.st_mtime, tz=UTC)
    return size, updated_at


def _format_remove_message(path: VfsPath, count: int) -> str:
    path_label = "/".join(path.segments) or "/"
    label = "entry" if count == 1 else "entries"
    return f"Deleted {count} {label} under {path_label}."


def _format_read_message(path: VfsPath, start: int, end: int) -> str:
    path_label = "/".join(path.segments) or "/"
    if start == end:  # pragma: no cover - empty file
        return f"Read file {path_label} (no lines returned)."
    return f"Read file {path_label} (lines {start + 1}-{end})."


class _LocalVfsSuite:
    """Filesystem tool handlers bound to a :class:`UnsafeLocalSandboxSection`."""

    def __init__(self, *, section: UnsafeLocalSandboxSection) -> None:
        super().__init__()
        self._section = section

    def list_directory(
        self, params: ListDirectoryParams, *, context: ToolContext
    ) -> ToolResult[tuple[FileInfo, ...]]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(
            params.path, allow_empty=True, field="path"
        )
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.workspace_path, path)
        _assert_within_workspace(handle.workspace_path, host_path)
        if host_path.exists() and host_path.is_file():
            raise ToolValidationError("Cannot list a file path; provide a directory.")
        snapshot = self._section.latest_snapshot()
        entries = self._build_directory_entries(
            base=path,
            host_path=host_path,
            snapshot=snapshot,
            workspace_root=handle.workspace_path,
        )
        message = vfs_module.format_directory_message(path, entries)
        self._section.touch_workspace()
        return ToolResult(message=message, value=tuple(entries))

    def read_file(
        self, params: ReadFileParams, *, context: ToolContext
    ) -> ToolResult[ReadFileResult]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(params.file_path, field="file_path")
        offset = vfs_module.normalize_offset(params.offset)
        limit = vfs_module.normalize_limit(params.limit)
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.workspace_path, path)
        if not host_path.exists() or not host_path.is_file():
            raise ToolValidationError("File does not exist in the workspace.")
        _assert_within_workspace(handle.workspace_path, host_path)
        try:
            content = host_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolValidationError("File is not valid UTF-8.") from error
        except OSError as error:
            raise ToolValidationError("Failed to read file contents.") from error
        lines = content.splitlines()
        total_lines = len(lines)
        start = min(offset, total_lines)
        end = min(start + limit, total_lines)
        numbered = [
            f"{index + 1:>4} | {line}"
            for index, line in enumerate(lines[start:end], start=start)
        ]
        formatted = "\n".join(numbered)
        message = _format_read_message(path, start, end)
        self._section.touch_workspace()
        return ToolResult(
            message=message,
            value=ReadFileResult(
                path=path,
                content=formatted,
                offset=start,
                limit=end - start,
                total_lines=total_lines,
            ),
        )

    def write_file(
        self, params: WriteFileParams, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(params.file_path, field="file_path")
        content = vfs_module.normalize_content(params.content)
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.workspace_path, path)
        if host_path.exists():
            raise ToolValidationError(
                "File already exists; use edit_file to modify existing content."
            )
        _assert_within_workspace(handle.workspace_path, host_path)
        self._section.write_file_to_workspace(path=path, content=content, mode="create")
        self._section.touch_workspace()
        message = vfs_module.format_write_file_message(path, content, "create")
        return ToolResult(
            message=message,
            value=WriteFile(path=path, content=content, mode="create"),
        )

    def edit_file(
        self, params: EditFileParams, *, context: ToolContext
    ) -> ToolResult[WriteFile]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(params.file_path, field="file_path")
        if len(params.old_string) > vfs_module.MAX_WRITE_LENGTH:
            raise ToolValidationError("old_string exceeds the 48,000 character limit.")
        if len(params.new_string) > vfs_module.MAX_WRITE_LENGTH:
            raise ToolValidationError("new_string exceeds the 48,000 character limit.")
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.workspace_path, path)
        if not host_path.exists() or not host_path.is_file():
            raise ToolValidationError("File does not exist in the workspace.")
        _assert_within_workspace(handle.workspace_path, host_path)
        try:
            existing = host_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolValidationError("File is not valid UTF-8.") from error
        occurrences = existing.count(params.old_string)
        if occurrences == 0:
            raise ToolValidationError("old_string not found in the target file.")
        if not params.replace_all and occurrences != 1:
            raise ToolValidationError(
                "old_string must match exactly once unless replace_all is true."
            )
        if params.replace_all:
            replacements = occurrences
            updated = existing.replace(params.old_string, params.new_string)
        else:
            replacements = 1
            updated = existing.replace(params.old_string, params.new_string, 1)
        normalized = vfs_module.normalize_content(updated)
        self._section.write_file_to_workspace(
            path=path, content=normalized, mode="overwrite"
        )
        self._section.touch_workspace()
        message = vfs_module.format_edit_message(path, replacements)
        return ToolResult(
            message=message,
            value=WriteFile(path=path, content=normalized, mode="overwrite"),
        )

    def glob(
        self, params: GlobParams, *, context: ToolContext
    ) -> ToolResult[tuple[GlobMatch, ...]]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        base = vfs_module.normalize_string_path(
            params.path, allow_empty=True, field="path"
        )
        pattern = params.pattern.strip()
        if not pattern:
            raise ToolValidationError("Pattern must not be empty.")
        _ = vfs_module.ensure_ascii(pattern, "pattern")
        handle = self._section.ensure_workspace()
        host_base = _host_path_for(handle.workspace_path, base)
        _assert_within_workspace(handle.workspace_path, host_base)
        matches: list[GlobMatch] = []
        snapshot = self._section.latest_snapshot()
        for file_path in _iter_workspace_files(host_base):
            try:
                relative = file_path.relative_to(host_base)
            except ValueError:  # pragma: no cover - path outside base
                continue
            candidate_path = _compose_relative_path(base, relative)
            if candidate_path is None:  # pragma: no cover - invalid path component
                continue
            relative_label = relative.as_posix()
            if not fnmatch.fnmatchcase(relative_label, pattern):
                continue
            try:
                match = self._build_glob_match(
                    target=candidate_path,
                    host_path=file_path,
                    snapshot=snapshot,
                    workspace_root=handle.workspace_path,
                )
            except ToolValidationError:  # pragma: no cover - stat failure
                continue
            matches.append(match)
            if len(matches) >= _MAX_MATCH_RESULTS:  # pragma: no cover - limit
                break
        matches.sort(key=lambda match: match.path.segments)
        message = vfs_module.format_glob_message(base, pattern, matches)
        self._section.touch_workspace()
        return ToolResult(message=message, value=tuple(matches))

    def grep(
        self, params: GrepParams, *, context: ToolContext
    ) -> ToolResult[tuple[GrepMatch, ...]]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        compiled = self._compile_grep_pattern(params.pattern)
        if isinstance(compiled, ToolResult):
            return compiled
        base_path = self._normalize_grep_base_path(params.path)
        glob_pattern = self._normalize_glob_pattern(params.glob)
        handle = self._section.ensure_workspace()
        host_base = _host_path_for(handle.workspace_path, base_path or VfsPath(()))
        _assert_within_workspace(handle.workspace_path, host_base)
        matches = self._collect_grep_matches(
            pattern=compiled,
            host_base=host_base,
            workspace_root=handle.workspace_path,
            base_path=base_path or VfsPath(()),
            glob_pattern=glob_pattern,
        )
        message = self._format_grep_result(params.pattern, matches)
        self._section.touch_workspace()
        return ToolResult(message=message, value=matches)

    @staticmethod
    def _compile_grep_pattern(
        pattern: str,
    ) -> re.Pattern[str] | ToolResult[tuple[GrepMatch, ...]]:
        try:
            return re.compile(pattern)
        except re.error as error:
            return ToolResult(
                message=f"Invalid regular expression: {error}",
                value=None,
                success=False,
            )

    @staticmethod
    def _normalize_grep_base_path(raw_path: str | None) -> VfsPath | None:
        if raw_path is None:
            return None
        return vfs_module.normalize_string_path(
            raw_path, allow_empty=True, field="path"
        )

    @staticmethod
    def _normalize_glob_pattern(raw_glob: str | None) -> str | None:
        if raw_glob is None:
            return None
        glob_pattern = raw_glob.strip()
        if not glob_pattern:
            return None
        _ = vfs_module.ensure_ascii(glob_pattern, "glob")
        return glob_pattern

    @staticmethod
    def _iter_grep_targets(
        *,
        host_base: Path,
        workspace_root: Path,
        base_path: VfsPath,
        glob_pattern: str | None,
    ) -> Iterator[tuple[VfsPath, Path]]:
        for file_path in _iter_workspace_files(host_base):
            try:
                relative = file_path.relative_to(host_base)
            except ValueError:  # pragma: no cover - path outside base
                continue
            relative_label = relative.as_posix()
            if glob_pattern and not fnmatch.fnmatchcase(relative_label, glob_pattern):
                continue
            target_path = _compose_relative_path(base_path, relative)
            if target_path is None:  # pragma: no cover - invalid path component
                continue
            try:
                _assert_within_workspace(workspace_root, file_path)
            except ToolValidationError:  # pragma: no cover - symlink escape
                continue
            yield target_path, file_path

    @staticmethod
    def _read_candidate_content(file_path: Path) -> str | None:
        try:
            return file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # pragma: no cover - binary/error
            return None

    def _collect_grep_matches(
        self,
        *,
        pattern: re.Pattern[str],
        host_base: Path,
        workspace_root: Path,
        base_path: VfsPath,
        glob_pattern: str | None,
    ) -> tuple[GrepMatch, ...]:
        matches: list[GrepMatch] = []
        for target_path, host_path in self._iter_grep_targets(
            host_base=host_base,
            workspace_root=workspace_root,
            base_path=base_path,
            glob_pattern=glob_pattern,
        ):
            content = self._read_candidate_content(host_path)
            if content is None:
                continue
            for index, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(
                        GrepMatch(
                            path=target_path,
                            line_number=index,
                            line=line,
                        )
                    )
                    if len(matches) >= _MAX_MATCH_RESULTS:  # pragma: no cover
                        return tuple(matches)
        return tuple(matches)

    @staticmethod
    def _format_grep_result(pattern: str, matches: tuple[GrepMatch, ...]) -> str:
        return vfs_module.format_grep_message(pattern, matches)

    def remove(
        self, params: RemoveParams, *, context: ToolContext
    ) -> ToolResult[DeleteEntry]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        path = vfs_module.normalize_string_path(params.path, field="path")
        if not path.segments:  # pragma: no cover - root removal
            raise ToolValidationError("Cannot remove the workspace root.")
        handle = self._section.ensure_workspace()
        host_path = _host_path_for(handle.workspace_path, path)
        if not host_path.exists():
            raise ToolValidationError("No files matched the provided path.")
        _assert_within_workspace(handle.workspace_path, host_path)
        removed_entries = sum(1 for _ in _iter_workspace_files(host_path))
        removed_entries = 1 if host_path.is_file() else max(removed_entries, 1)
        args = (str(host_path),)
        try:
            completed = self._section.run_python_script(
                script=_REMOVE_PATH_SCRIPT,
                args=args,
            )
        except FileNotFoundError as error:  # pragma: no cover - no python
            raise ToolValidationError(
                "Python interpreter is required to execute filesystem commands."
            ) from error
        if completed.returncode != 0:  # pragma: no cover - removal failure
            message = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "Removal failed."
            )
            raise ToolValidationError(message)
        self._section.touch_workspace()
        message = _format_remove_message(path, removed_entries)
        return ToolResult(
            message=message,
            value=DeleteEntry(path=path),
        )

    def _build_directory_entries(
        self,
        *,
        base: VfsPath,
        host_path: Path,
        snapshot: VirtualFileSystem,
        workspace_root: Path,
    ) -> list[FileInfo]:
        entries: list[FileInfo] = []
        if not host_path.exists():
            return entries
        try:
            children = sorted(host_path.iterdir(), key=lambda child: child.name.lower())
        except OSError as error:  # pragma: no cover - permission error
            raise ToolValidationError(
                "Failed to inspect directory contents."
            ) from error
        for child in children:
            entry_path = _compose_child_path(base, child.name)
            if entry_path is None:  # pragma: no cover - invalid path component
                continue
            if child.is_dir() and not child.is_symlink():
                entries.append(
                    FileInfo(
                        path=entry_path,
                        kind="directory",
                        size_bytes=None,
                        version=None,
                        updated_at=None,
                    )
                )
                continue
            try:
                info = self._build_file_info(
                    path=entry_path,
                    host_file=child,
                    snapshot=snapshot,
                    workspace_root=workspace_root,
                )
            except ToolValidationError:  # pragma: no cover - stat failure
                continue
            entries.append(info)
        entries.sort(key=lambda entry: entry.path.segments)
        return entries[:_MAX_MATCH_RESULTS]

    @staticmethod
    def _build_file_info(
        *,
        path: VfsPath,
        host_file: Path,
        snapshot: VirtualFileSystem,
        workspace_root: Path,
    ) -> FileInfo:
        _assert_within_workspace(workspace_root, host_file)
        snapshot_entry = vfs_module.find_file(snapshot.files, path)
        size_bytes, updated_at = _stat_file(host_file)
        version = snapshot_entry.version if snapshot_entry else None
        updated = snapshot_entry.updated_at if snapshot_entry else updated_at
        return FileInfo(
            path=path,
            kind="file",
            size_bytes=size_bytes,
            version=version,
            updated_at=updated,
        )

    @staticmethod
    def _build_glob_match(
        *,
        target: VfsPath,
        host_path: Path,
        snapshot: VirtualFileSystem,
        workspace_root: Path,
    ) -> GlobMatch:
        _assert_within_workspace(workspace_root, host_path)
        snapshot_entry = vfs_module.find_file(snapshot.files, target)
        size_bytes, updated_at = _stat_file(host_path)
        if snapshot_entry is None:
            return GlobMatch(
                path=target,
                size_bytes=size_bytes,
                version=1,
                updated_at=updated_at,
            )
        return GlobMatch(  # pragma: no cover - existing snapshot
            path=target,
            size_bytes=size_bytes,
            version=snapshot_entry.version,
            updated_at=snapshot_entry.updated_at,
        )


class _LocalEvalSuite:
    def __init__(self, *, section: UnsafeLocalSandboxSection) -> None:
        super().__init__()
        self._section = section

    def evaluate_python(
        self, params: EvalParams, *, context: ToolContext
    ) -> ToolResult[EvalResult]:
        ensure_context_uses_session(context=context, session=self._section.session)
        del context
        self._ensure_passthrough_payload_is_empty(params)
        code = _normalize_local_eval_code(params.code)
        _ = self._section.ensure_workspace()
        try:
            completed = self._section.run_python_script(
                script=code,
                args=(),
                timeout=_EVAL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return self._timeout_result()
        except FileNotFoundError as error:  # pragma: no cover - no python
            raise ToolValidationError(
                "Python interpreter is required to execute evaluation commands."
            ) from error

        stdout = _truncate_eval_stream(str(completed.stdout or ""))
        stderr = _truncate_eval_stream(str(completed.stderr or ""))
        success = completed.returncode == 0
        if success:
            message = f"Evaluation succeeded (exit code {completed.returncode})."
        else:
            message = f"Evaluation failed (exit code {completed.returncode})."
        result = EvalResult(
            value_repr=None,
            stdout=stdout,
            stderr=stderr,
            globals={},
            reads=(),
            writes=(),
        )
        self._section.touch_workspace()
        return ToolResult(message=message, value=result, success=success)

    @staticmethod
    def _ensure_passthrough_payload_is_empty(params: EvalParams) -> None:
        if params.reads:
            raise ToolValidationError(
                "Local evaluate_python reads are not supported; access the workspace directly."
            )
        if params.writes:
            raise ToolValidationError(
                "Local evaluate_python writes are not supported; use the write_file tool."
            )
        if params.globals:
            raise ToolValidationError(
                "Local evaluate_python globals are not supported."
            )

    @staticmethod
    def _timeout_result() -> ToolResult[EvalResult]:
        result = EvalResult(
            value_repr=None,
            stdout="",
            stderr="Execution timed out.",
            globals={},
            reads=(),
            writes=(),
        )
        return ToolResult(message="Evaluation timed out.", value=result, success=False)


class _LocalShellSuite:
    """Handler collection bound to a :class:`UnsafeLocalSandboxSection`."""

    def __init__(self, *, section: UnsafeLocalSandboxSection) -> None:
        super().__init__()
        self._section = section

    def run_shell(
        self, params: LocalShellParams, *, context: ToolContext
    ) -> ToolResult[LocalShellResult]:
        ensure_context_uses_session(context=context, session=self._section.session)
        command = _normalize_command(params.command)
        handle = self._section.ensure_workspace()
        cwd = _normalize_cwd(params.cwd, handle.workspace_path)
        env_overrides = _normalize_env(params.env)
        timeout_seconds = _normalize_timeout(params.timeout_seconds)
        if params.stdin:
            _ = _ensure_ascii(params.stdin, field="stdin")

        return self._run_shell_via_subprocess(
            params=params,
            command=command,
            cwd=cwd,
            environment=env_overrides,
            timeout_seconds=timeout_seconds,
        )

    def _run_shell_via_subprocess(
        self,
        *,
        params: LocalShellParams,
        command: tuple[str, ...],
        cwd: str,
        environment: Mapping[str, str],
        timeout_seconds: float,
    ) -> ToolResult[LocalShellResult]:
        start = time.perf_counter()
        options = _SubprocessOptions(
            command=list(command),
            stdin=params.stdin if params.stdin else None,
            cwd=cwd,
            environment=environment,
            timeout=timeout_seconds,
            capture_output=params.capture_output,
        )
        outcome = self._execute_and_capture(options)
        duration_ms = int((time.perf_counter() - start) * 1_000)
        self._section.touch_workspace()
        result = self._build_shell_result(
            params=params,
            command=command,
            cwd=cwd,
            outcome=outcome,
            duration_ms=duration_ms,
        )
        message = f"`shell_execute` exited with {outcome.exit_code}."
        if outcome.timed_out:
            message = "`shell_execute` exceeded the configured timeout."
        return ToolResult(message=message, value=result)

    def _execute_and_capture(self, options: _SubprocessOptions) -> _ShellOutcome:
        """Execute subprocess and capture the outcome."""
        try:
            completed = self._section.run_subprocess(options)
            return _ShellOutcome(
                exit_code=completed.returncode,
                stdout=str(completed.stdout or ""),
                stderr=str(completed.stderr or ""),
                timed_out=False,
            )
        except subprocess.TimeoutExpired as error:
            return _ShellOutcome(
                exit_code=124,
                stdout=str(error.stdout or ""),
                stderr=str(error.stderr or ""),
                timed_out=True,
            )
        except FileNotFoundError as error:
            raise ToolValidationError(
                "Command not found or executable unavailable."
            ) from error

    def _build_shell_result(
        self,
        *,
        params: LocalShellParams,
        command: tuple[str, ...],
        cwd: str,
        outcome: _ShellOutcome,
        duration_ms: int,
    ) -> LocalShellResult:
        """Build the final shell result from outcome."""
        stdout_clean = outcome.stdout.rstrip()
        stderr_clean = outcome.stderr.rstrip()
        if not params.capture_output:
            stdout_final = _CAPTURE_DISABLED
            stderr_final = _CAPTURE_DISABLED
        else:
            stdout_final = _truncate_stream(stdout_clean)
            stderr_final = _truncate_stream(stderr_clean)

        cwd_display = self._format_cwd_display(cwd)
        return LocalShellResult(
            command=command,
            cwd=cwd_display,
            exit_code=outcome.exit_code,
            stdout=stdout_final,
            stderr=stderr_final,
            duration_ms=duration_ms,
            timed_out=outcome.timed_out,
        )

    def _format_cwd_display(self, cwd: str) -> str:
        """Format the cwd for display in results."""
        handle = self._section.current_workspace()
        if handle is None:
            return _DEFAULT_WORKDIR
        try:
            relative = Path(cwd).relative_to(handle.workspace_path)
            if str(relative) == ".":
                return _DEFAULT_WORKDIR
            return posixpath.join(_DEFAULT_WORKDIR, str(relative))
        except ValueError:
            return cwd


__all__ = [
    "LocalShellParams",
    "LocalShellResult",
    "UnsafeLocalSandboxConfig",
    "UnsafeLocalSandboxSection",
    "UnsafeLocalWorkspace",
]
