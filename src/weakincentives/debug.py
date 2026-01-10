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

"""Debug helpers for persisting session snapshots."""

from __future__ import annotations

import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID, uuid4

from .dbc import dbc_enabled
from .filesystem import Filesystem
from .runtime.session import Session, iter_sessions_bottom_up

logger: logging.Logger = logging.getLogger(__name__)


def dump_session(root_session: Session, target: str | Path) -> Path | None:
    """Persist a session tree to a JSONL snapshot file.

    Each line in the resulting ``.jsonl`` file contains a serialized snapshot for
    a session in the tree rooted at ``root_session``. The snapshots are written
    from the root down to the leaves so that the primary session appears first.

    The provided ``target`` may be a directory or a file path. Regardless of
    input, the final snapshot file is named ``<root_session_id>.jsonl``.
    """

    with dbc_enabled(False):
        target_path = _resolve_target(Path(target), root_session)
        snapshots = _collect_snapshots(root_session)
        if not snapshots:
            logger.info(
                "Session snapshot dump skipped; no slices to persist.",
                extra={
                    "session_id": str(root_session.session_id),
                    "snapshot_path": str(target_path),
                },
            )
            return None

        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(snapshots) + "\n"
        _ = target_path.write_text(payload, encoding="utf-8")
        logger.info(
            "Session snapshots persisted.",
            extra={
                "session_id": str(root_session.session_id),
                "snapshot_path": str(target_path),
                "snapshot_count": len(snapshots),
            },
        )
        return target_path


def archive_filesystem(
    fs: Filesystem,
    target: str | Path,
    *,
    archive_id: UUID | None = None,
) -> Path | None:
    """Archive filesystem contents to a zip file.

    Creates a zip archive containing all files from the given filesystem.
    The archive is named ``<archive_id>.zip`` and written to the target
    directory.

    Args:
        fs: Filesystem instance to archive.
        target: Target directory for the archive file.
        archive_id: Unique identifier for the archive. If not provided,
            a new UUID is generated.

    Returns:
        Path to the created archive file, or None if the filesystem is empty.
    """
    with dbc_enabled(False):
        archive_id = archive_id or uuid4()
        target_path = Path(target).expanduser()
        if target_path.is_file():
            target_path = target_path.parent
        archive_path = target_path / f"{archive_id}.zip"

        # Collect all files recursively from the filesystem
        files = _collect_files_recursive(fs, ".")

        if not files:
            logger.info(
                "Filesystem archive skipped; no files to archive.",
                extra={
                    "archive_id": str(archive_id),
                    "archive_path": str(archive_path),
                },
            )
            return None

        target_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in files:
                try:
                    result = fs.read_bytes(file_path)
                    zf.writestr(file_path, result.content)
                except (PermissionError, ValueError) as exc:
                    logger.warning(
                        "Skipping file due to read error: %s",
                        file_path,
                        extra={"error": str(exc)},
                    )
                    continue

        logger.info(
            "Filesystem archived.",
            extra={
                "archive_id": str(archive_id),
                "archive_path": str(archive_path),
                "file_count": len(files),
            },
        )
        return archive_path


def _collect_files_recursive(fs: Filesystem, path: str) -> list[str]:
    """Recursively collect all file paths from the filesystem."""
    files: list[str] = []
    try:
        entries = fs.list(path)
    except (FileNotFoundError, NotADirectoryError):
        return files

    for entry in entries:
        if entry.is_file:
            files.append(entry.path)
        elif entry.is_directory:
            files.extend(_collect_files_recursive(fs, entry.path))
    return files


def _collect_snapshots(root_session: Session) -> list[str]:
    snapshots: list[str] = []
    with dbc_enabled(False):
        for session in _iter_sessions_top_down(root_session):
            snapshot = session.snapshot(include_all=True)
            if not snapshot.slices:
                logger.info(
                    "Session snapshot skipped; no slices to persist.",
                    extra={
                        "session_id": str(session.session_id),
                        "root_session_id": str(root_session.session_id),
                    },
                )
                continue
            snapshots.append(snapshot.to_json())
        return snapshots


def _resolve_target(target: Path, root_session: Session) -> Path:
    target = target.expanduser()
    root_name = f"{root_session.session_id}.jsonl"

    if target.is_dir():
        return target / root_name
    if target.suffix != ".jsonl":
        return target.with_name(root_name)
    if target.stem != str(root_session.session_id):
        return target.with_name(root_name)
    return target


def _iter_sessions_top_down(root_session: Session) -> Iterable[Session]:
    sessions = list(iter_sessions_bottom_up(root_session))
    sessions.reverse()
    return sessions
