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

"""Git operations for filesystem snapshots.

Provides functions for initializing a git repository, creating snapshots
via git commits, restoring to previous snapshots, and cleaning up.
These are used by :class:`~weakincentives.filesystem.HostFilesystem` to
implement :class:`~weakincentives.filesystem.SnapshotableFilesystem`.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec: B404
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from weakincentives.errors import SnapshotError, SnapshotRestoreError

from ._types import FilesystemSnapshot

__all__: list[str] = []


def git_env() -> dict[str, str]:
    """Create a clean environment for git subprocesses.

    Removes GIT_* environment variables to prevent conflicts when running
    inside git hooks (e.g., pre-commit) where these variables are set by
    the parent git process.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def run_git(
    args: Sequence[str],
    *,
    git_dir: str | None,
    root: str,
    check: bool = True,
    text: bool = False,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """Run git command with external git-dir and work-tree.

    All git commands use --git-dir and --work-tree to keep the git
    repository separate from the workspace files.

    Args:
        args: Git subcommand and arguments (e.g., ["add", "-A"]).
        git_dir: Path to the external git directory.
        root: Workspace root path.
        check: Raise CalledProcessError on non-zero exit.
        text: Decode output as text.

    Returns:
        CompletedProcess with stdout/stderr captured.
    """
    cmd: list[str] = [
        "git",
        f"--git-dir={git_dir}",
        f"--work-tree={root}",
        *args,
    ]
    return subprocess.run(  # nosec B603 B607
        cmd,
        cwd=root,
        check=check,
        capture_output=True,
        text=text,
        env=git_env(),
    )


def init_git_repo(root: str, git_dir: str | None) -> str:
    """Initialize git repository for snapshot support.

    Creates the git repository in an external directory (outside the
    workspace root) to prevent agents from accessing git internals.

    Args:
        root: Workspace root path (unused directly, reserved for future use).
        git_dir: Optional custom git directory path. If None, a temporary
            directory is created.

    Returns:
        The git directory path (newly created or existing).
    """
    _ = root  # Reserved for future use
    needs_init = False
    if git_dir is None:
        git_dir = tempfile.mkdtemp(prefix="wink-git-")
        needs_init = True

    # Handle custom git_dir that doesn't exist yet (mkdtemp already creates
    # the directory when using the default, so this only applies to custom paths)
    git_path = Path(git_dir)
    if not git_path.exists():
        git_path.mkdir(parents=True, exist_ok=True)
        needs_init = True

    if needs_init:
        env = git_env()
        # Initialize bare repository in external directory
        _ = subprocess.run(  # nosec B603 B607
            ["git", "init", "--bare"],
            cwd=git_dir,
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
                git_dir,
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
                git_dir,
                "config",
                "user.name",
                "WINK Snapshots",
            ],
            check=True,
            capture_output=True,
            env=env,
        )

    return git_dir


def create_snapshot(
    root: str, git_dir: str | None, *, tag: str | None = None
) -> FilesystemSnapshot:
    """Capture current filesystem state as a git commit.

    Uses git's content-addressed storage for copy-on-write semantics.
    Identical files share storage automatically between snapshots.

    Args:
        root: Workspace root path.
        git_dir: External git directory path.
        tag: Optional human-readable label for the snapshot.

    Returns:
        Immutable snapshot that can be stored in session state.
    """
    # Stage all changes (including new and deleted files)
    _ = run_git(["add", "-A"], git_dir=git_dir, root=root)

    # Commit (allow empty for idempotent snapshots)
    # Use --no-gpg-sign to avoid issues in environments with signing hooks
    message = tag or f"snapshot-{datetime.now(UTC).isoformat()}"
    commit_result = run_git(
        ["commit", "-m", message, "--allow-empty", "--no-gpg-sign"],
        git_dir=git_dir,
        root=root,
        check=False,
        text=True,
    )

    # If commit failed and we don't have any commits yet, we may need
    # to check for empty repo scenario
    if commit_result.returncode != 0:
        # Check if this is because there's nothing to commit
        # and no prior commits exist (empty repo)
        head_check = run_git(
            ["rev-parse", "--verify", "HEAD"],
            git_dir=git_dir,
            root=root,
            check=False,
        )
        if head_check.returncode != 0:
            # No HEAD commit exists - create an initial empty commit
            _ = run_git(
                ["commit", "--allow-empty", "--no-gpg-sign", "-m", message],
                git_dir=git_dir,
                root=root,
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
    result = run_git(["rev-parse", "HEAD"], git_dir=git_dir, root=root, text=True)
    commit_ref = str(result.stdout).strip()

    return FilesystemSnapshot(
        snapshot_id=uuid4(),
        created_at=datetime.now(UTC),
        commit_ref=commit_ref,
        root_path=root,
        git_dir=git_dir,
        tag=tag,
    )


def restore_snapshot(
    root: str, git_dir: str | None, snapshot: FilesystemSnapshot
) -> None:
    """Restore filesystem to a previous git commit.

    Performs a hard reset to the snapshot's commit and removes any
    untracked files (including ignored files) for a strict rollback.

    Args:
        root: Workspace root path.
        git_dir: External git directory path.
        snapshot: The snapshot to restore.

    Raises:
        SnapshotRestoreError: If git reset fails (e.g., invalid commit).
    """
    # Hard reset to the commit (restores tracked files)
    result = run_git(
        ["reset", "--hard", snapshot.commit_ref],
        git_dir=git_dir,
        root=root,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        msg = f"Failed to restore snapshot: {result.stderr}"
        raise SnapshotRestoreError(msg)

    # Remove untracked files and directories for strict rollback
    # -x removes ignored files too (e.g., cache, logs) for full restore
    _ = run_git(["clean", "-xfd"], git_dir=git_dir, root=root, check=False)


def cleanup_git_dir(git_dir: str | None) -> bool:
    """Remove the external git directory.

    Call this when the filesystem is no longer needed to clean up
    the snapshot storage. Safe to call multiple times or with None.

    Args:
        git_dir: External git directory path to remove.

    Returns:
        True if directory was removed, False otherwise.
    """
    if git_dir is not None and Path(git_dir).exists():
        shutil.rmtree(git_dir)
        return True
    return False
