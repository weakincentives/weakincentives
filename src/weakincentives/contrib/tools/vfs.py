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

"""Virtual filesystem tool suite for LLM agents.

This module provides filesystem tools that enable LLM agents to interact with
files safely within a virtual filesystem. The VFS can optionally mount host
directories for controlled access to real files.

Key components:

- :class:`VfsToolsSection`: Prompt section exposing all VFS tools. Add this
  to your prompt template to give agents filesystem access.

- :class:`VfsConfig`: Configuration object for VfsToolsSection, consolidating
  mount points and security settings.

- :class:`FilesystemToolHandlers`: Standalone tool handlers for filesystem
  operations. Use when building custom tool configurations.

Available tools:
    - ``ls``: List directory contents
    - ``read_file``: Read file contents with pagination
    - ``write_file``: Create new files
    - ``edit_file``: Modify files via string replacement
    - ``glob``: Find files matching shell patterns
    - ``grep``: Search file contents with regex
    - ``rm``: Remove files or directories

Example::

    from weakincentives.contrib.tools import VfsConfig, VfsToolsSection, HostMount

    config = VfsConfig(
        mounts=(HostMount(host_path="/path/to/project/src"),),
        allowed_host_roots=("/path/to/project",),
    )
    section = VfsToolsSection(session=session, config=config)

For data types (VfsPath, FileInfo, etc.), see vfs_types.py.
For host mount logic, see vfs_mounts.py.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import field
from datetime import UTC, datetime
from typing import Final, Literal, cast, override

from weakincentives.filesystem import READ_ENTIRE_FILE, Filesystem

from ...dataclasses import FrozenDataclass
from ...resources import ResourceRegistry
from ...errors import ToolValidationError
from ...prompt.markdown import MarkdownSection
from ...prompt.policy import ReadBeforeWritePolicy
from ...prompt.tool import Tool, ToolContext, ToolExample, ToolResult
from ...runtime.session import Session
from ...types import SupportsDataclass, SupportsToolResult
from .filesystem_memory import InMemoryFilesystem
from .vfs_mounts import (
    MAX_MOUNT_PREVIEW_ENTRIES,
    MountContext,
    get_current_time,
    match_glob,
    materialize_host_mounts_to_filesystem,
    normalize_host_root,
    render_host_mounts_block,
    render_section_template,
    resolve_mount_path,
)
from .vfs_types import (
    MAX_WRITE_LENGTH,
    DeleteEntry,
    EditFileParams,
    FileEncoding,
    FileInfo,
    GlobMatch,
    GlobParams,
    GrepMatch,
    GrepParams,
    HostMount,
    HostMountPreview,
    ListDirectory,
    ListDirectoryParams,
    ListDirectoryResult,
    ReadFile,
    ReadFileParams,
    ReadFileResult,
    RemoveParams,
    VfsFile,
    VfsPath,
    WriteFile,
    WriteFileParams,
    WriteMode,
    ensure_ascii,
    format_path,
    format_timestamp,
    normalize_content,
    normalize_limit,
    normalize_offset,
    normalize_optional_path,
    normalize_path,
    normalize_segments,
    normalize_string_path,
    path_from_string,
)

_DEFAULT_ENCODING: Final[Literal["utf-8"]] = "utf-8"
_MAX_WRITE_LENGTH: Final[int] = MAX_WRITE_LENGTH
_MAX_READ_LIMIT: Final[int] = 2_000


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------


class FilesystemToolHandlers:
    """Reusable tool handlers for filesystem operations.

    Provides handlers for common filesystem operations that work with any
    implementation of the Filesystem protocol:

    - ``list_directory``: List directory contents (ls)
    - ``read_file``: Read file contents with pagination
    - ``write_file``: Create new files
    - ``edit_file``: Modify existing files via string replacement
    - ``glob``: Find files matching shell patterns
    - ``grep``: Search file contents with regex
    - ``remove``: Delete files or directories (rm)

    Handlers obtain the filesystem from ``ToolContext.filesystem``, enabling
    the same handlers to work across different filesystem implementations
    (in-memory, host-mounted, sandboxed containers, etc.).

    Example::

        handlers = FilesystemToolHandlers()
        # Use with Tool definitions:
        tool = Tool(
            name="ls",
            handler=handlers.list_directory,
            description="List directory contents",
        )

        # Or call directly in tests:
        result = handlers.list_directory(
            ListDirectoryParams(path="src"),
            context=context,
        )
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        mount_point: str | None = None,
    ) -> None:
        """Initialize handlers.

        Args:
            clock: Optional callable returning current datetime. Defaults to UTC now.
            mount_point: Optional virtual mount point prefix to strip from paths.
                For example, if mount_point="/workspace", then "/workspace/file.txt"
                becomes "file.txt".
        """
        super().__init__()
        self._clock = clock or get_current_time
        self._mount_point = mount_point

    @staticmethod
    def _get_filesystem(context: ToolContext) -> Filesystem:
        """Get the filesystem from context, raising if not available."""
        if context.filesystem is None:
            raise ToolValidationError("No filesystem available in this context.")
        return context.filesystem

    def list_directory(
        self, params: ListDirectoryParams, *, context: ToolContext
    ) -> ToolResult[tuple[FileInfo, ...]]:
        """List directory contents and return file metadata.

        Retrieves all entries (files and subdirectories) under the specified
        path. Results are sorted alphabetically by path segments.

        Args:
            params: Parameters containing the directory path to list.
            context: Tool execution context providing filesystem access.

        Returns:
            ToolResult containing a tuple of FileInfo objects, each describing
            a file (with size and modification time) or directory entry.

        Raises:
            ToolValidationError: If the directory does not exist or if the
                path points to a file instead of a directory.
        """
        fs = self._get_filesystem(context)
        path = normalize_string_path(
            params.path, allow_empty=True, field="path", mount_point=self._mount_point
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
        """Read file contents with line-based pagination.

        Reads UTF-8 file content and returns lines with line numbers prefixed
        in the format "   N | content". Supports offset/limit pagination for
        reading large files in chunks.

        Args:
            params: Parameters containing file_path, offset (starting line),
                and limit (max lines to read).
            context: Tool execution context providing filesystem access.

        Returns:
            ToolResult containing ReadFileResult with numbered content,
            pagination metadata (offset, limit, total_lines).

        Raises:
            ToolValidationError: If the file does not exist or parameters
                are invalid.
        """
        fs = self._get_filesystem(context)
        path = normalize_string_path(
            params.file_path, field="file_path", mount_point=self._mount_point
        )
        offset = normalize_offset(params.offset)
        limit = normalize_limit(params.limit)

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
        """Create a new file with the specified content.

        Creates a new UTF-8 text file at the given path. This operation
        fails if the file already exists - use edit_file to modify
        existing files.

        Args:
            params: Parameters containing file_path and content to write.
            context: Tool execution context providing filesystem access.

        Returns:
            ToolResult containing WriteFile with normalized path, content,
            and mode="create".

        Raises:
            ToolValidationError: If the file already exists or the content
                exceeds MAX_WRITE_LENGTH (48,000 characters).
        """
        fs = self._get_filesystem(context)
        path = normalize_string_path(
            params.file_path, field="file_path", mount_point=self._mount_point
        )
        content = normalize_content(params.content)

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
        """Edit an existing file using exact string replacement.

        Replaces occurrences of old_string with new_string in the file.
        By default, old_string must match exactly once; set replace_all=True
        to replace all occurrences.

        Args:
            params: Parameters containing file_path, old_string, new_string,
                and optional replace_all flag.
            context: Tool execution context providing filesystem access.

        Returns:
            ToolResult containing WriteFile with the updated content and
            mode="overwrite".

        Raises:
            ToolValidationError: If the file does not exist, old_string is
                empty, old_string is not found, old_string matches multiple
                times without replace_all=True, or strings exceed 48,000 chars.
        """
        fs = self._get_filesystem(context)
        path = normalize_string_path(
            params.file_path, field="file_path", mount_point=self._mount_point
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

        normalized_content = normalize_content(updated)
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
        """Search for files matching a shell glob pattern.

        Finds all files under the specified path that match the glob pattern.
        Supports standard patterns like ``*`` (any characters), ``**`` (recursive),
        and ``?`` (single character). Only returns files, not directories.

        Args:
            params: Parameters containing pattern (glob expression) and optional
                path (base directory, defaults to root).
            context: Tool execution context providing filesystem access.

        Returns:
            ToolResult containing a tuple of GlobMatch objects sorted by path,
            each with path, size_bytes, version, and updated_at metadata.

        Raises:
            ToolValidationError: If the pattern is empty or contains non-ASCII
                characters.
        """
        fs = self._get_filesystem(context)
        base = normalize_string_path(
            params.path, allow_empty=True, field="path", mount_point=self._mount_point
        )
        pattern = params.pattern.strip()
        if not pattern:
            raise ToolValidationError("Pattern must not be empty.")
        ensure_ascii(pattern, "pattern")

        base_str = "/".join(base.segments) if base.segments else "."

        glob_results = fs.glob(pattern, path=base_str)
        matches: list[GlobMatch] = []

        # fs.glob already filters to files only, no need to check is_file
        for match in glob_results:
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
        """Search for a regex pattern in file contents.

        Searches all files under the specified path for lines matching the
        regular expression pattern. Optionally filter files by glob pattern.

        Args:
            params: Parameters containing pattern (regex), optional path
                (base directory), and optional glob (file filter pattern).
            context: Tool execution context providing filesystem access.

        Returns:
            ToolResult containing a tuple of GrepMatch objects sorted by
            (path, line_number), each with the matching line content.

        Note:
            Returns an error result (success=False) if the regex pattern
            is invalid, rather than raising an exception.
        """
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
            base_path = normalize_string_path(
                params.path, allow_empty=True, field="path", mount_point=self._mount_point
            )
        glob_pattern = params.glob.strip() if params.glob is not None else None
        if glob_pattern:
            ensure_ascii(glob_pattern, "glob")
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
        """Remove a file or directory recursively.

        Deletes the specified path. If the path is a directory, all contents
        are deleted recursively. The root directory cannot be deleted.

        Args:
            params: Parameters containing the path to remove.
            context: Tool execution context providing filesystem access.

        Returns:
            ToolResult containing DeleteEntry with the deleted path.

        Raises:
            ToolValidationError: If the path does not exist, or if attempting
                to delete the root directory.
        """
        fs = self._get_filesystem(context)
        path = normalize_string_path(
            params.path, field="path", mount_point=self._mount_point
        )
        if not path.segments:
            raise ToolValidationError("Cannot delete root directory.")
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

    Consolidates all VfsToolsSection constructor arguments into a single
    immutable configuration object. This pattern avoids accumulating long
    argument lists as the section evolves and makes configuration reusable.

    Attributes:
        mounts: Host directories to mount into the virtual filesystem.
            Each HostMount specifies a host_path and optional virtual_path.
        allowed_host_roots: Security boundary - only paths under these roots
            can be mounted. Prevents agents from accessing arbitrary host files.
        accepts_overrides: If True, tools accept parameter overrides from
            the prompt. Default is False for safety.

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
    """Prompt section exposing virtual filesystem tools to LLM agents.

    Provides a complete set of filesystem tools (ls, read_file, write_file,
    edit_file, glob, grep, rm) that operate on an in-memory virtual filesystem.
    Host directories can be mounted into the VFS for controlled access to
    real files.

    The section includes:
    - An in-memory filesystem that can be pre-populated with host mounts
    - ReadBeforeWritePolicy enforcing safe file modification patterns
    - Resource registration for dependency injection of the filesystem

    Example::

        # Create section with host directory mounted
        config = VfsConfig(
            mounts=(HostMount(host_path="src"),),
            allowed_host_roots=("/home/user/project",),
        )
        section = VfsToolsSection(session=session, config=config)

        # Add to a prompt template
        prompt = PromptTemplate(
            ns="myapp",
            key="agent",
            sections=(section,),
        )

    See Also:
        - :class:`VfsConfig` for configuration options
        - :class:`FilesystemToolHandlers` for the underlying tool implementations
        - :class:`HostMount` for mounting host directories
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
        """Initialize a VFS tools section with filesystem access.

        Creates a prompt section that exposes virtual filesystem tools (ls,
        read_file, write_file, edit_file, glob, grep, rm) to LLM agents.

        Args:
            session: The session managing agent state and event dispatch.
            config: Optional VfsConfig consolidating all configuration options.
                Takes precedence over individual parameters when provided.
            mounts: Host directories to mount into the virtual filesystem.
                Ignored if config is provided.
            allowed_host_roots: Allowed root paths for host mounts (security
                boundary). Ignored if config is provided.
            accepts_overrides: Whether tools accept parameter overrides.
                Ignored if config is provided.

        Example::

            # Using VfsConfig (recommended)
            config = VfsConfig(
                mounts=(HostMount(host_path="src"),),
                allowed_host_roots=("/home/user/project",),
            )
            section = VfsToolsSection(session=session, config=config)

            # Using individual parameters
            section = VfsToolsSection(
                session=session,
                mounts=(HostMount(host_path="src"),),
                allowed_host_roots=("/home/user/project",),
            )
        """
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
            mount_previews = materialize_host_mounts_to_filesystem(
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

        # Default policy: must read file before overwriting
        default_policies = (ReadBeforeWritePolicy(),)

        super().__init__(
            title="Virtual Filesystem Tools",
            key="vfs.tools",
            template=render_section_template(mount_previews),
            default_params=_VfsSectionParams(),
            tools=tools,
            policies=default_policies,
            accepts_overrides=resolved_accepts_overrides,
        )

    @property
    def session(self) -> Session:
        """Return the session managing this section's state.

        The session provides event dispatch and state management for
        agent interactions with the virtual filesystem.
        """
        return self._session

    @property
    def filesystem(self) -> Filesystem:
        """Return the filesystem managed by this section.

        Provides direct access to the underlying in-memory filesystem for
        inspection or programmatic manipulation outside of tool calls.

        Returns:
            The Filesystem instance containing all mounted files and any
            files created/modified by agent tool calls.
        """
        return self._filesystem

    @override
    def resources(self) -> ResourceRegistry:
        """Return resources contributed by this section for dependency injection.

        Registers the section's filesystem as a resource, making it available
        to tools via ToolContext.filesystem. This enables tool handlers to
        access the VFS without explicit parameter passing.

        Returns:
            ResourceRegistry containing a binding for the Filesystem protocol
            to this section's in-memory filesystem instance.
        """
        return ResourceRegistry.build({Filesystem: self._filesystem})

    @override
    def clone(self, **kwargs: object) -> VfsToolsSection:
        """Create a copy of this section bound to a new session.

        Clones the section while preserving the underlying filesystem state.
        The new section shares the same filesystem instance, allowing
        continuity of file operations across session boundaries.

        Args:
            **kwargs: Must include 'session' (Session). Optionally may
                include 'dispatcher' which must match the session's dispatcher.

        Returns:
            A new VfsToolsSection bound to the provided session.

        Raises:
            TypeError: If session is not provided or if dispatcher doesn't
                match the session's dispatcher.
        """
        session_obj = kwargs.get("session")
        if not isinstance(session_obj, Session):
            msg = "session is required to clone VfsToolsSection."
            raise TypeError(msg)
        provided_dispatcher = kwargs.get("dispatcher")
        if (
            provided_dispatcher is not None
            and provided_dispatcher is not session_obj.dispatcher
        ):
            msg = "Provided dispatcher must match the target session's dispatcher."
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
# Public Exports
# ---------------------------------------------------------------------------

# Re-export from vfs_types for backward compatibility
format_directory_message = _format_directory_message
format_write_file_message = _format_write_file_message
format_edit_message = _format_edit_message
format_glob_message = _format_glob_message
format_grep_message = _format_grep_message

__all__ = [
    "MAX_MOUNT_PREVIEW_ENTRIES",
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
    "MountContext",
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
    "match_glob",
    "normalize_content",
    "normalize_limit",
    "normalize_offset",
    "normalize_optional_path",
    "normalize_path",
    "normalize_segments",
    "normalize_string_path",
    "path_from_string",
    "render_host_mounts_block",
    "resolve_mount_path",
]
