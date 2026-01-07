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

"""Host filesystem backend for workspace operations.

This module provides a filesystem implementation backed by a host directory,
with path restrictions to prevent escaping the sandbox root.

Example usage::

    from weakincentives.filesystem import HostFilesystem

    # Create a host filesystem with sandbox root
    fs = HostFilesystem(_root="/path/to/workspace")

    # Write and read files
    fs.write("src/main.py", "print('hello')")
    result = fs.read("src/main.py")
    assert result.content == "print('hello')"
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # nosec: B404
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from weakincentives.errors import SnapshotError, SnapshotRestoreError

from ._types import (
    DEFAULT_READ_LIMIT,
    MAX_GREP_MATCHES,
    MAX_WRITE_BYTES,
    MAX_WRITE_LENGTH,
    READ_ENTIRE_FILE,
    FileEntry,
    FileStat,
    FilesystemSnapshot,
    GlobMatch,
    GrepMatch,
    ReadBytesResult,
    ReadResult,
    WriteResult,
    glob_match,
    normalize_path,
    validate_path,
)

__all__ = ["HostFilesystem"]


# ---------------------------------------------------------------------------
# HostFilesystem Implementation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HostFilesystem:
    """Filesystem backed by a host directory with path restrictions.

    Provides sandboxed access to a root directory on the host filesystem.
    All paths are resolved relative to the root and validated to ensure
    they cannot escape the sandbox via symlinks or path traversal.
    Supports snapshot and restore operations via git commits.

    The git repository used for snapshots is stored outside the workspace
    root to prevent agents from accessing or modifying it. By default,
    a temporary directory is created using Python's ``tempfile.mkdtemp()``
    (e.g., ``/tmp/wink-git-abc12345``).
    """

    _root: str
    _read_only: bool = False
    _git_initialized: bool = False
    _git_dir: str | None = None

    @property
    def root(self) -> str:
        """Workspace root path."""
        return self._root

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled."""
        return self._read_only

    @staticmethod
    def _git_env() -> dict[str, str]:
        """Create a clean environment for git subprocesses.

        Removes GIT_* environment variables to prevent conflicts when running
        inside git hooks (e.g., pre-commit) where these variables are set by
        the parent git process.
        """
        return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}

    def _resolve_path(self, path: str) -> Path:
        """Resolve a relative path to an absolute path within root.

        Raises:
            PermissionError: If resolved path escapes root directory.
        """
        root_path = Path(self._root).resolve()

        # Handle empty/root path
        if not path or path in {".", "/"}:
            return root_path

        # Join path with root and resolve to handle .. segments
        candidate = (root_path / path).resolve()

        # Ensure the resolved path is within root
        try:
            _ = candidate.relative_to(root_path)
        except ValueError:
            msg = f"Path escapes root directory: {path}"
            raise PermissionError(msg) from None

        return candidate

    def _run_git(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        """Run git command with external git-dir and work-tree.

        All git commands use --git-dir and --work-tree to keep the git
        repository separate from the workspace files.

        Args:
            args: Git subcommand and arguments (e.g., ["add", "-A"]).
            check: Raise CalledProcessError on non-zero exit.
            text: Decode output as text.

        Returns:
            CompletedProcess with stdout/stderr captured.
        """
        cmd: list[str] = [
            "git",
            f"--git-dir={self._git_dir}",
            f"--work-tree={self._root}",
            *args,
        ]
        return subprocess.run(  # nosec B603 B607
            cmd,
            cwd=self._root,
            check=check,
            capture_output=True,
            text=text,
            env=self._git_env(),
        )

    def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> ReadResult:
        """Read file content with optional pagination."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        if resolved.is_dir():
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        try:
            with resolved.open(encoding=encoding) as f:
                lines = f.readlines()
        except UnicodeDecodeError as err:
            msg = (
                f"Cannot read '{path}' as text: file contains binary content that "
                f"cannot be decoded as {encoding}. Use read_bytes() for binary files."
            )
            raise ValueError(msg) from err

        total_lines = len(lines)
        # READ_ENTIRE_FILE (-1) reads all lines; None uses default window
        if limit == READ_ENTIRE_FILE:
            actual_limit = total_lines
        else:
            actual_limit = limit if limit is not None else DEFAULT_READ_LIMIT
        start = min(offset, total_lines)
        end = min(start + actual_limit, total_lines)
        selected_lines = lines[start:end]
        content = "".join(selected_lines)

        # Remove trailing newline for consistency
        if content.endswith("\n"):
            content = content[:-1]

        # Return path relative to root
        rel_path = normalize_path(path) or "/"

        return ReadResult(
            content=content,
            path=rel_path,
            total_lines=total_lines,
            offset=start,
            limit=end - start,
            truncated=end < total_lines,
        )

    def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> ReadBytesResult:
        """Read file content as raw bytes with optional pagination."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        if resolved.is_dir():
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg)

        file_size = resolved.stat().st_size

        with resolved.open("rb") as f:
            if offset > 0:
                _ = f.seek(offset)
            data = f.read(limit) if limit is not None else f.read()

        # Calculate actual positions after read
        actual_offset = min(offset, file_size)
        bytes_read = len(data)
        truncated = actual_offset + bytes_read < file_size

        rel_path = normalize_path(path) or "/"

        return ReadBytesResult(
            content=data,
            path=rel_path,
            size_bytes=file_size,
            offset=actual_offset,
            limit=bytes_read,
            truncated=truncated,
        )

    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        try:
            resolved = self._resolve_path(path)
            return resolved.exists()
        except PermissionError:
            return False

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        st = resolved.stat()
        is_dir = resolved.is_dir()
        rel_path = normalize_path(path) or "/"

        return FileStat(
            path=rel_path,
            is_file=not is_dir,
            is_directory=is_dir,
            size_bytes=st.st_size if not is_dir else 0,
            created_at=datetime.fromtimestamp(st.st_ctime, tz=UTC),
            modified_at=datetime.fromtimestamp(st.st_mtime, tz=UTC),
        )

    def list(self, path: str = ".") -> Sequence[FileEntry]:
        """List directory contents."""
        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        if not resolved.is_dir():
            msg = f"Not a directory: {path}"
            raise NotADirectoryError(msg)

        entries: list[FileEntry] = []
        base_path = normalize_path(path)

        for item in resolved.iterdir():
            is_dir = item.is_dir()
            rel_path = f"{base_path}/{item.name}" if base_path else item.name

            entries.append(
                FileEntry(
                    name=item.name,
                    path=rel_path,
                    is_file=not is_dir,
                    is_directory=is_dir,
                )
            )

        entries.sort(key=lambda e: e.name)
        return entries

    def glob(
        self,
        pattern: str,
        *,
        path: str = ".",
    ) -> Sequence[GlobMatch]:
        """Match files by glob pattern."""
        resolved_base = self._resolve_path(path)
        root_path = Path(self._root)

        if not resolved_base.exists():
            return []

        matches: list[GlobMatch] = []

        for file_path in resolved_base.rglob("*"):
            if not file_path.is_file():
                continue
            # Get path relative to root filesystem
            rel_to_fs_root = str(file_path.relative_to(root_path))
            # Get path relative to search base
            rel_to_base = str(file_path.relative_to(resolved_base))

            if glob_match(rel_to_base, pattern, ""):
                matches.append(GlobMatch(path=rel_to_fs_root, is_file=True))

        matches.sort(key=lambda m: m.path)
        return matches

    def grep(
        self,
        pattern: str,
        *,
        path: str = ".",
        glob: str | None = None,
        max_matches: int | None = None,
    ) -> Sequence[GrepMatch]:
        """Search file contents by regex."""
        resolved_base = self._resolve_path(path)
        root_path = Path(self._root)

        try:
            regex = re.compile(pattern)
        except re.error as err:
            msg = f"Invalid regex pattern: {err}"
            raise ValueError(msg) from err

        actual_max = max_matches if max_matches is not None else MAX_GREP_MATCHES
        matches: list[GrepMatch] = []

        if not resolved_base.exists():
            return []

        for file_path in sorted(resolved_base.rglob("*")):
            if not file_path.is_file():
                continue

            rel_to_fs_root = str(file_path.relative_to(root_path))
            rel_to_base = str(file_path.relative_to(resolved_base))

            # Apply glob filter if provided
            if glob is not None and not glob_match(rel_to_base, glob, ""):
                continue

            grep_matches = self._grep_file(
                file_path, regex, rel_to_fs_root, actual_max - len(matches)
            )
            matches.extend(grep_matches)
            if len(matches) >= actual_max:
                break

        return matches

    @staticmethod
    def _grep_file(
        file_path: Path, regex: re.Pattern[str], rel_path: str, max_remaining: int
    ) -> list[GrepMatch]:  # ty: ignore[invalid-type-form]  # ty bug: resolves list to class method
        """Search a single file for regex matches."""
        matches: list[GrepMatch] = []
        try:
            with file_path.open(encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.rstrip("\n")
                    match = regex.search(line)
                    if match:
                        matches.append(
                            GrepMatch(
                                path=rel_path,
                                line_number=line_num,
                                line_content=line,
                                match_start=match.start(),
                                match_end=match.end(),
                            )
                        )
                        if len(matches) >= max_remaining:
                            break
        except (OSError, UnicodeDecodeError):
            # Skip binary or inaccessible files
            pass
        return matches

    def write(
        self,
        path: str,
        content: str,
        *,
        mode: Literal["create", "overwrite", "append"] = "overwrite",
        create_parents: bool = True,
    ) -> WriteResult:
        """Write content to a file."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot write to root directory"
            raise ValueError(msg)

        validate_path(normalized)

        if len(content) > MAX_WRITE_LENGTH:
            msg = f"Content exceeds maximum length of {MAX_WRITE_LENGTH} characters."
            raise ValueError(msg)

        resolved = self._resolve_path(path)
        parent_dir = resolved.parent

        # Check parent directory
        if not parent_dir.exists():
            if not create_parents:
                raise FileNotFoundError(
                    f"Parent directory does not exist: {parent_dir}"
                )
            parent_dir.mkdir(parents=True, exist_ok=True)

        exists = resolved.exists()

        if mode == "create" and exists:
            raise FileExistsError(f"File already exists: {path}")

        file_mode = "a" if mode == "append" else "w"

        with resolved.open(file_mode, encoding="utf-8") as f:
            _ = f.write(content)

        # Calculate bytes written
        bytes_written = resolved.stat().st_size

        return WriteResult(
            path=normalized,
            bytes_written=bytes_written,
            mode=mode,
        )

    def write_bytes(
        self,
        path: str,
        content: bytes,
        *,
        mode: Literal["create", "overwrite", "append"] = "overwrite",
        create_parents: bool = True,
    ) -> WriteResult:
        """Write raw bytes to a file."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot write to root directory"
            raise ValueError(msg)

        validate_path(normalized)

        if len(content) > MAX_WRITE_BYTES:
            msg = f"Content exceeds maximum size of {MAX_WRITE_BYTES} bytes."
            raise ValueError(msg)

        resolved = self._resolve_path(path)
        parent_dir = resolved.parent

        # Check parent directory
        if not parent_dir.exists():
            if not create_parents:
                raise FileNotFoundError(
                    f"Parent directory does not exist: {parent_dir}"
                )
            parent_dir.mkdir(parents=True, exist_ok=True)

        exists = resolved.exists()

        if mode == "create" and exists:
            raise FileExistsError(f"File already exists: {path}")

        file_mode = "ab" if mode == "append" else "wb"

        with resolved.open(file_mode) as f:
            _ = f.write(content)

        # Calculate bytes written
        bytes_written = resolved.stat().st_size

        return WriteResult(
            path=normalized,
            bytes_written=bytes_written,
            mode=mode,
        )

    def delete(
        self,
        path: str,
        *,
        recursive: bool = False,
    ) -> None:
        """Delete a file or directory."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = normalize_path(path)
        if not normalized:
            msg = "Cannot delete root directory"
            raise PermissionError(msg)

        resolved = self._resolve_path(path)

        if not resolved.exists():
            raise FileNotFoundError(path)

        if resolved.is_file():
            resolved.unlink()
            return

        # It's a directory
        if any(resolved.iterdir()) and not recursive:
            msg = f"Directory not empty: {path}"
            raise IsADirectoryError(msg)

        if recursive:
            shutil.rmtree(resolved)
        else:
            resolved.rmdir()

    def mkdir(
        self,
        path: str,
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> None:
        """Create a directory."""
        if self._read_only:
            msg = "Filesystem is read-only"
            raise PermissionError(msg)

        normalized = normalize_path(path)
        if not normalized:
            return  # Root always exists

        validate_path(normalized)
        resolved = self._resolve_path(path)

        if resolved.exists():
            if resolved.is_file():
                raise FileExistsError(f"A file exists at path: {path}")
            if not exist_ok:
                raise FileExistsError(f"Directory already exists: {path}")
            return

        if parents:
            resolved.mkdir(parents=True, exist_ok=True)
        else:
            if not resolved.parent.exists():
                raise FileNotFoundError(
                    f"Parent directory does not exist: {resolved.parent}"
                )
            resolved.mkdir()

    # --- Snapshot Operations ---

    def _ensure_git(self) -> None:
        """Initialize git repository if needed for snapshot support.

        Creates the git repository in an external directory (outside the
        workspace root) to prevent agents from accessing git internals.
        """
        if self._git_initialized:
            return

        # Create external git directory if not specified
        needs_init = False
        if self._git_dir is None:
            self._git_dir = tempfile.mkdtemp(prefix="wink-git-")
            needs_init = True

        # Handle custom _git_dir that doesn't exist yet (mkdtemp already creates
        # the directory when using the default, so this only applies to custom paths)
        git_path = Path(self._git_dir)
        if not git_path.exists():
            git_path.mkdir(parents=True, exist_ok=True)
            needs_init = True

        if needs_init:
            env = self._git_env()
            # Initialize bare repository in external directory
            _ = subprocess.run(  # nosec B603 B607
                ["git", "init", "--bare"],
                cwd=self._git_dir,
                check=True,
                capture_output=True,
                env=env,
            )
            # Configure for snapshot use (local config only)
            # Use --git-dir since it's a bare repo
            _ = subprocess.run(  # nosec B603 B607
                [
                    "git",
                    "--git-dir",
                    self._git_dir,
                    "config",
                    "user.email",
                    "wink@localhost",
                ],
                check=True,
                capture_output=True,
                env=env,
            )
            _ = subprocess.run(  # nosec B603 B607
                [
                    "git",
                    "--git-dir",
                    self._git_dir,
                    "config",
                    "user.name",
                    "WINK Snapshots",
                ],
                check=True,
                capture_output=True,
                env=env,
            )
        self._git_initialized = True

    def snapshot(self, *, tag: str | None = None) -> FilesystemSnapshot:
        """Capture current filesystem state as a git commit.

        Uses git's content-addressed storage for copy-on-write semantics.
        Identical files share storage automatically between snapshots.

        Args:
            tag: Optional human-readable label for the snapshot.

        Returns:
            Immutable snapshot that can be stored in session state.
        """
        self._ensure_git()

        # Stage all changes (including new and deleted files)
        _ = self._run_git(["add", "-A"])

        # Commit (allow empty for idempotent snapshots)
        # Use --no-gpg-sign to avoid issues in environments with signing hooks
        message = tag or f"snapshot-{datetime.now(UTC).isoformat()}"
        commit_result = self._run_git(
            ["commit", "-m", message, "--allow-empty", "--no-gpg-sign"],
            check=False,
            text=True,
        )

        # If commit failed and we don't have any commits yet, we may need
        # to check for empty repo scenario
        if commit_result.returncode != 0:
            # Check if this is because there's nothing to commit
            # and no prior commits exist (empty repo)
            head_check = self._run_git(
                ["rev-parse", "--verify", "HEAD"],
                check=False,
            )
            if head_check.returncode != 0:
                # No HEAD commit exists - create an initial empty commit
                _ = self._run_git(
                    ["commit", "--allow-empty", "--no-gpg-sign", "-m", message],
                )
            else:
                # HEAD exists but commit failed - this is an error condition
                # (e.g., disk full, permission denied, corrupted git state)
                stderr = (
                    commit_result.stderr.strip()
                    if commit_result.stderr
                    else "unknown error"
                )
                raise SnapshotError(f"Failed to create snapshot commit: {stderr}")

        # Get commit hash (text=True guarantees str stdout)
        result = self._run_git(["rev-parse", "HEAD"], text=True)
        commit_ref = str(result.stdout).strip()

        return FilesystemSnapshot(
            snapshot_id=uuid4(),
            created_at=datetime.now(UTC),
            commit_ref=commit_ref,
            root_path=self._root,
            git_dir=self._git_dir,
            tag=tag,
        )

    def restore(self, snapshot: FilesystemSnapshot) -> None:
        """Restore filesystem to a previous git commit.

        Performs a hard reset to the snapshot's commit and removes any
        untracked files (including ignored files) for a strict rollback.

        Args:
            snapshot: The snapshot to restore.

        Raises:
            SnapshotRestoreError: If git reset fails (e.g., invalid commit).
        """
        # Use git_dir from snapshot if available and we don't have one yet
        if snapshot.git_dir is not None and self._git_dir is None:
            self._git_dir = snapshot.git_dir

        self._ensure_git()

        # Hard reset to the commit (restores tracked files)
        result = self._run_git(
            ["reset", "--hard", snapshot.commit_ref],
            check=False,
            text=True,
        )
        if result.returncode != 0:
            msg = f"Failed to restore snapshot: {result.stderr}"
            raise SnapshotRestoreError(msg)

        # Remove untracked files and directories for strict rollback
        # -x removes ignored files too (e.g., cache, logs) for full restore
        _ = self._run_git(["clean", "-xfd"], check=False)

    def cleanup(self) -> None:
        """Remove the external git directory.

        Call this when the filesystem is no longer needed to clean up
        the snapshot storage. Safe to call multiple times.
        """
        if self._git_dir is not None and Path(self._git_dir).exists():
            shutil.rmtree(self._git_dir)
            self._git_initialized = False

    @property
    def git_dir(self) -> str | None:
        """External git directory path, if initialized."""
        return self._git_dir
