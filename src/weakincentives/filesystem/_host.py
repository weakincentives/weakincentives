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

"""Host directory backend for the filesystem facade.

Implements :class:`~weakincentives.filesystem.FilesystemBackend` primitives
over a sandboxed host directory. The root is resolved once at construction
and every operation re-validates that its (symlink-resolved) target stays
inside the sandbox.

Snapshots follow the strict rollback contract documented in ``_git_ops``:
they capture every file (including ``.gitignore``-matched ones) and restore
removes everything that was not present at snapshot time.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..clock import SYSTEM_CLOCK
from ._backend import SnapshotRef
from ._git_ops import (
    cleanup_git_dir,
    create_snapshot,
    init_git_repo,
    restore_snapshot,
)
from ._types import (
    FileEntry,
    FileStat,
    GlobMatch,
    GrepMatch,
    WriteMode,
    glob_match,
)

__all__ = ["HostBackend"]


class HostBackend:
    """Filesystem primitives over a sandboxed host directory.

    All paths are facade-normalized relative strings (``""`` is the root).
    Each operation resolves its target and rejects paths that escape the
    root via symlinks with ``PermissionError``. Snapshot/restore is backed
    by an external git repository kept outside the workspace so agents
    cannot reach it.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        read_only: bool = False,
        git_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._root = Path(root).resolve()
        self._read_only = read_only
        self._git_dir = git_dir
        self._git_initialized = False

    @property
    def root(self) -> str:
        """Workspace root path (resolved)."""
        return str(self._root)

    @property
    def read_only(self) -> bool:
        """True if write operations are disabled (enforced by the facade)."""
        return self._read_only

    def _resolve(self, path: str) -> Path:
        """Resolve a normalized relative path inside the sandbox root.

        Raises:
            PermissionError: If the resolved path escapes the root (e.g.,
                through a symlink).
        """
        if not path:
            return self._root
        candidate = (self._root / path).resolve()
        try:
            _ = candidate.relative_to(self._root)
        except ValueError:
            msg = f"Path escapes root directory: {path}"
            raise PermissionError(msg) from None
        return candidate

    # --- Primitives ---------------------------------------------------------

    def stat(self, path: str) -> FileStat:
        """Get metadata for a path."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(path)
        st = resolved.stat()
        is_dir = resolved.is_dir()
        return FileStat(
            path=path or "/",
            is_file=not is_dir,
            is_directory=is_dir,
            size_bytes=0 if is_dir else st.st_size,
            created_at=datetime.fromtimestamp(st.st_ctime, tz=UTC),
            modified_at=datetime.fromtimestamp(st.st_mtime, tz=UTC),
        )

    def list(self, path: str) -> Sequence[FileEntry]:
        """List entries of an existing directory, sorted by name."""
        resolved = self._resolve(path)
        entries = [
            FileEntry(
                name=item.name,
                path=f"{path}/{item.name}" if path else item.name,
                is_file=not item.is_dir(),
                is_directory=item.is_dir(),
            )
            for item in resolved.iterdir()
        ]
        entries.sort(key=lambda e: e.name)
        return entries

    def glob(self, pattern: str, *, path: str) -> Sequence[GlobMatch]:
        """Match files under a base directory, sorted by path."""
        base = self._resolve(path)
        matches = [
            GlobMatch(path=str(item.relative_to(self._root)), is_file=True)
            for item in base.rglob("*")
            if item.is_file() and glob_match(str(item.relative_to(base)), pattern, "")
        ]
        matches.sort(key=lambda m: m.path)
        return matches

    def grep(
        self,
        pattern: str,
        *,
        path: str,
        glob: str | None,
        max_matches: int,
    ) -> Sequence[GrepMatch]:
        """Regex-search file contents under a base directory."""
        base = self._resolve(path)
        regex = re.compile(pattern)
        matches: list[GrepMatch] = []
        for item in sorted(base.rglob("*")):
            if not item.is_file():
                continue
            if glob is not None and not glob_match(
                str(item.relative_to(base)), glob, ""
            ):
                continue
            matches.extend(
                self._grep_file(
                    item,
                    regex,
                    str(item.relative_to(self._root)),
                    max_matches - len(matches),
                )
            )
            if len(matches) >= max_matches:
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

    def read_range(
        self,
        path: str,
        *,
        offset: int,
        length: int | None,
    ) -> bytes:
        """Read up to ``length`` bytes starting at ``offset``."""
        resolved = self._resolve(path)
        try:
            with resolved.open("rb") as f:
                if offset:
                    _ = f.seek(offset)
                return f.read(length) if length is not None else f.read()
        except FileNotFoundError:
            raise FileNotFoundError(path) from None
        except IsADirectoryError:
            msg = f"Is a directory: {path}"
            raise IsADirectoryError(msg) from None

    def write(self, path: str, data: bytes, *, mode: WriteMode) -> None:
        """Write bytes to a file, creating parent directories as needed."""
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if mode == "create":
            # "xb" atomically fails if the file exists (no TOCTOU race)
            try:
                handle = resolved.open("xb")
            except FileExistsError:
                raise FileExistsError(f"File already exists: {path}") from None
        else:
            handle = resolved.open("ab" if mode == "append" else "wb")
        with handle:
            _ = handle.write(data)

    def delete(self, path: str, *, recursive: bool) -> None:
        """Delete a file, or a directory when empty or ``recursive=True``."""
        resolved = self._resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(path)
        if resolved.is_file():
            resolved.unlink()
            return
        if recursive:
            shutil.rmtree(resolved)
            return
        if any(resolved.iterdir()):
            msg = f"Directory not empty: {path}"
            raise IsADirectoryError(msg)
        resolved.rmdir()

    def mkdir(self, path: str) -> None:
        """Create a directory and any missing parents (idempotent)."""
        self._resolve(path).mkdir(parents=True, exist_ok=True)

    # --- Snapshots ------------------------------------------------------------

    def _ensure_git(self) -> str:
        """Initialize the external git repository if needed."""
        if not self._git_initialized:
            self._git_dir = init_git_repo(self._git_dir)
            self._git_initialized = True
        git_dir = self._git_dir
        if git_dir is None:  # pragma: no cover - always set after init_git_repo
            msg = "git_dir must be set after init_git_repo"
            raise RuntimeError(msg)
        return git_dir

    def snapshot(self, *, tag: str | None = None) -> SnapshotRef:
        """Capture the workspace as a git commit (strict rollback contract).

        Every file is captured, including ``.gitignore``-matched ones, so
        :meth:`restore` returns the workspace to this exact state.
        """
        git_dir = self._ensure_git()
        commit_ref = create_snapshot(self.root, git_dir, tag=tag)
        return SnapshotRef(
            snapshot_id=uuid4(),
            created_at=SYSTEM_CLOCK.utcnow(),
            token=f"{commit_ref}:{git_dir}",
            tag=tag,
        )

    def restore(self, ref: SnapshotRef) -> None:
        """Restore the workspace to a snapshot's exact state.

        Files created after the snapshot are removed regardless of ignore
        rules; files present at snapshot time (ignored or not) come back.

        Raises:
            SnapshotRestoreError: The ref does not name a commit known to
                this backend's git repository.
        """
        commit_ref, _, ref_git_dir = ref.token.partition(":")
        # Adopt the ref's git dir for cross-instance restore
        if self._git_dir is None and ref_git_dir:
            self._git_dir = ref_git_dir
        git_dir = self._ensure_git()
        restore_snapshot(self.root, git_dir, commit_ref)

    def cleanup(self) -> None:
        """Remove the external git directory. Safe to call multiple times."""
        if cleanup_git_dir(self._git_dir):
            self._git_initialized = False

    @property
    def git_dir(self) -> str | None:
        """External git directory path, if initialized."""
        return self._git_dir
