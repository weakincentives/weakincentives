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
from pathlib import Path
from uuid import uuid4

from weakincentives.clock import SYSTEM_CLOCK
from weakincentives.errors import SnapshotError, SnapshotRestoreError

from ._types import FilesystemSnapshot, SnapshotHistoryEntry

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
    git_dir: str,
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


def init_git_repo(git_dir: str | None) -> str:
    """Initialize git repository for snapshot support.

    Creates the git repository in an external directory (outside the
    workspace root) to prevent agents from accessing git internals.

    Args:
        git_dir: Optional custom git directory path. If None, a temporary
            directory is created.

    Returns:
        The git directory path (newly created or existing).
    """
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
    root: str, git_dir: str, *, tag: str | None = None
) -> FilesystemSnapshot:
    """Capture current filesystem state as a git commit.

    Uses git's content-addressed storage for copy-on-write semantics.
    Identical files share storage automatically between snapshots.

    Args:
        root: Workspace root path.
        git_dir: External git directory path (must be initialized).
        tag: Optional human-readable label for the snapshot.

    Returns:
        Immutable snapshot that can be stored in session state.
    """
    # Stage all changes (including new and deleted files)
    _ = run_git(["add", "-A"], git_dir=git_dir, root=root)

    # Commit (allow empty for idempotent snapshots)
    # Use --no-gpg-sign to avoid issues in environments with signing hooks
    message = tag or f"snapshot-{SYSTEM_CLOCK.utcnow().isoformat()}"
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
        created_at=SYSTEM_CLOCK.utcnow(),
        commit_ref=commit_ref,
        root_path=root,
        git_dir=git_dir,
        tag=tag,
    )


def restore_snapshot(root: str, git_dir: str, snapshot: FilesystemSnapshot) -> None:
    """Restore filesystem to a previous git commit.

    Performs a hard reset to the snapshot's commit and removes any
    untracked files (including ignored files) for a strict rollback.

    Args:
        root: Workspace root path.
        git_dir: External git directory path (must be initialized).
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


def _parse_snapshot_tag(tag: str | None) -> tuple[str | None, str | None]:
    """Extract tool_name and tool_call_id from a snapshot tag.

    Tags follow the format ``"pre:{tool_name}:{tool_call_id}"``.

    Returns:
        Tuple of (tool_name, tool_call_id), both None if tag is unparseable.
    """
    if not tag or not tag.startswith("pre:"):
        return None, None
    parts = tag.split(":", 2)
    if len(parts) == 3:  # noqa: PLR2004
        return parts[1], parts[2]
    return None, None


def _parse_numstat_line(line: str) -> tuple[int, int, str]:
    """Parse a single line of ``git diff --numstat`` output.

    Returns:
        Tuple of (insertions, deletions, file_path).
    """
    parts = line.split("\t", 2)
    if len(parts) != 3:  # noqa: PLR2004
        return 0, 0, ""
    ins_str, del_str, path = parts
    ins = int(ins_str) if ins_str != "-" else 0
    dels = int(del_str) if del_str != "-" else 0
    return ins, dels, path


def _parse_reflog(lines: list[str]) -> list[tuple[str, str]]:
    """Parse reflog output into (commit_hash, action) tuples.

    Reflog format is 3 lines per entry: hash, action, selector.
    Returns events in chronological order (oldest first).
    """
    events: list[tuple[str, str]] = []
    i = 0
    while i + 2 < len(lines):
        commit_hash = lines[i].strip()
        action = lines[i + 1].strip()
        i += 3
        while i < len(lines) and not lines[i].strip():
            i += 1
        events.append((commit_hash, action))
    events.reverse()
    return events


def _detect_rollbacks(
    events: list[tuple[str, str]],
) -> tuple[list[str], set[str]]:
    """Walk reflog events to collect commits and detect rolled-back refs.

    Returns:
        Tuple of (commit_refs in order, rolled_back_refs set).
    """
    commit_refs: list[str] = []
    rolled_back_refs: set[str] = set()

    for commit_hash, action in events:
        if action.startswith("commit"):
            commit_refs.append(commit_hash)
        elif action.startswith("reset:") and commit_refs:
            try:
                target_idx = commit_refs.index(commit_hash)
            except ValueError:
                continue
            for ref in commit_refs[target_idx + 1 :]:
                rolled_back_refs.add(ref)

    return commit_refs, rolled_back_refs


def snapshot_history(root: str, git_dir: str) -> list[SnapshotHistoryEntry]:
    """Build an ordered list of snapshot history entries from git reflog.

    Uses ``git reflog`` to reconstruct the complete history including
    rolled-back commits. Commit entries become snapshot entries, and
    reset entries mark the preceding commits as rolled back.

    Args:
        root: Workspace root path.
        git_dir: External git directory path (must be initialized).

    Returns:
        List of history entries ordered oldest-first.
    """
    head_check = run_git(
        ["rev-parse", "--verify", "HEAD"],
        git_dir=git_dir,
        root=root,
        check=False,
        text=True,
    )
    if head_check.returncode != 0:
        return []

    reflog_result = run_git(
        ["reflog", "--format=%H%n%gs%n%gd"],
        git_dir=git_dir,
        root=root,
        check=False,
        text=True,
    )
    if reflog_result.returncode != 0:
        return []

    raw = str(reflog_result.stdout).strip()
    if not raw:
        return []

    events = _parse_reflog(raw.split("\n"))
    commit_refs, rolled_back_refs = _detect_rollbacks(events)

    seen: set[str] = set()
    entries: list[SnapshotHistoryEntry] = []
    for commit_ref in commit_refs:
        if commit_ref in seen:
            continue
        seen.add(commit_ref)
        entry = _build_entry(root, git_dir, commit_ref, commit_ref in rolled_back_refs)
        if entry is not None:
            entries.append(entry)

    return entries


def _commit_diffstat(
    root: str, git_dir: str, commit_ref: str, parent_ref: str | None
) -> tuple[list[str], int, int]:
    """Compute diffstat for a single commit.

    Returns:
        Tuple of (files_changed, total_insertions, total_deletions).
    """
    diff_args = ["diff", "--numstat", commit_ref + "^", commit_ref]
    if parent_ref is None:
        diff_args = ["diff", "--numstat", "--root", commit_ref]

    result = run_git(diff_args, git_dir=git_dir, root=root, check=False, text=True)
    files: list[str] = []
    ins_total = 0
    del_total = 0
    if result.returncode == 0:
        for line in str(result.stdout).strip().split("\n"):
            if not line.strip():
                continue
            ins, dels, path = _parse_numstat_line(line)
            if path:
                files.append(path)
            ins_total += ins
            del_total += dels
    return files, ins_total, del_total


def _build_entry(
    root: str, git_dir: str, commit_ref: str, rolled_back: bool
) -> SnapshotHistoryEntry | None:
    """Build a single history entry from a git commit."""
    show_result = run_git(
        ["log", "-1", "--format=%P%n%aI%n%s", commit_ref],
        git_dir=git_dir,
        root=root,
        check=False,
        text=True,
    )
    if show_result.returncode != 0:
        return None

    # Root commits have an empty %P (parent), so the first line is empty.
    # We must not strip() the full output before splitting, or we lose it.
    meta_lines = str(show_result.stdout).rstrip("\n").split("\n")
    if len(meta_lines) < 3:  # noqa: PLR2004
        return None

    parent_ref = meta_lines[0].strip() or None
    created_at = meta_lines[1].strip()
    tag = meta_lines[2].strip() or None

    files_changed, total_ins, total_dels = _commit_diffstat(
        root, git_dir, commit_ref, parent_ref
    )
    tool_name, tool_call_id = _parse_snapshot_tag(tag)

    return SnapshotHistoryEntry(
        commit_ref=commit_ref,
        created_at=created_at,
        tag=tag,
        parent_ref=parent_ref,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        files_changed=tuple(files_changed),
        insertions=total_ins,
        deletions=total_dels,
        rolled_back=rolled_back,
    )


def export_history_bundle(root: str, git_dir: str, target: Path) -> Path | None:
    """Create a portable git bundle file from the snapshot repository.

    Args:
        root: Workspace root path.
        git_dir: External git directory path (must be initialized).
        target: Directory where the bundle file will be written.

    Returns:
        Path to the created git bundle file, or None if no history exists.
    """
    # Check that HEAD exists
    head_check = run_git(
        ["rev-parse", "--verify", "HEAD"],
        git_dir=git_dir,
        root=root,
        check=False,
        text=True,
    )
    if head_check.returncode != 0:
        return None

    target.mkdir(parents=True, exist_ok=True)
    bundle_path = target / "history.bundle"

    result = run_git(
        ["bundle", "create", str(bundle_path), "--all"],
        git_dir=git_dir,
        root=root,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None

    return bundle_path
