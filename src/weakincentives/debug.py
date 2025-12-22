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

"""Debug helpers for persisting session and filesystem snapshots."""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .dbc import dbc_enabled
from .runtime.session import Session, iter_sessions_bottom_up

if TYPE_CHECKING:
    from .contrib.tools.filesystem import Filesystem

logger: logging.Logger = logging.getLogger(__name__)

# Metadata file stored inside the ZIP archive
_WINK_METADATA_FILENAME = "_wink_metadata.json"


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


def dump_filesystem_snapshot(
    filesystem: Filesystem,
    target: str | Path,
    *,
    session_id: str | None = None,
) -> Path | None:
    """Persist filesystem state to a ZIP archive.

    Creates a ``.fs.zip`` archive containing all files from the filesystem.
    The archive is named to pair with the session JSONL file when both
    exist in the same directory.

    Args:
        filesystem: The filesystem to export.
        target: Target path. If a directory, creates ``<session_id>.fs.zip``.
            If a ``.jsonl`` file path, creates a sibling ``.fs.zip`` with
            matching stem.
        session_id: Session identifier for metadata and filename generation.
            Required when target is a directory.

    Returns:
        Path to the created archive, or None if the filesystem is empty.

    Raises:
        ValueError: If target is a directory and session_id is not provided.
    """
    target_path = _resolve_filesystem_target(Path(target), session_id)

    # Collect all files using glob
    all_files = filesystem.glob("**/*")
    file_paths = [match.path for match in all_files if match.is_file]

    if not file_paths:
        logger.info(
            "Filesystem snapshot dump skipped; no files to persist.",
            extra={
                "session_id": session_id,
                "archive_path": str(target_path),
            },
        )
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    file_count = 0

    with zipfile.ZipFile(target_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(file_paths):
            try:
                # Read file content - use limit=-1 to read entire file
                result = filesystem.read(file_path, limit=-1)
                content = result.content
                content_bytes = content.encode("utf-8")
                total_bytes += len(content_bytes)
                file_count += 1
                zf.writestr(file_path, content_bytes)
            except (FileNotFoundError, IsADirectoryError, PermissionError) as exc:
                logger.warning(
                    "Failed to read file for archive",
                    extra={
                        "path": file_path,
                        "error": str(exc),
                    },
                )
                continue
            except (UnicodeDecodeError, UnicodeEncodeError):
                # File content couldn't be encoded to UTF-8
                logger.debug(
                    "Skipping file with encoding issues",
                    extra={"path": file_path},
                )
                continue

        # Write metadata file
        metadata = {
            "version": "1",
            "created_at": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "root_path": filesystem.root,
            "file_count": file_count,
            "total_bytes": total_bytes,
        }
        zf.writestr(_WINK_METADATA_FILENAME, json.dumps(metadata, indent=2))

    logger.info(
        "Filesystem archive created.",
        extra={
            "event": "debug.filesystem.archive_created",
            "path": str(target_path),
            "file_count": file_count,
            "total_bytes": total_bytes,
        },
    )

    return target_path


def dump_session_with_filesystem(
    root_session: Session,
    target: str | Path,
    *,
    filesystem: Filesystem | None = None,
) -> tuple[Path | None, Path | None]:
    """Persist session tree and associated filesystem as paired archives.

    This is the recommended function for exporting complete debug snapshots.
    Creates:

    - ``<session_id>.jsonl`` for session state
    - ``<session_id>.fs.zip`` for filesystem state (if filesystem provided)

    Args:
        root_session: Root session of the tree to export.
        target: Target directory for output files.
        filesystem: Optional filesystem to include. When None, only the
            session JSONL is created.

    Returns:
        Tuple of (session_path, filesystem_path). Either may be None if
        the respective content was empty or not provided.
    """
    session_path = dump_session(root_session, target)

    filesystem_path: Path | None = None
    if filesystem is not None:
        session_id = str(root_session.session_id)
        # If session was dumped, use its path as reference for the filesystem archive
        if session_path is not None:
            filesystem_path = dump_filesystem_snapshot(
                filesystem, session_path, session_id=session_id
            )
        else:
            # Session was empty but we still want to dump filesystem
            filesystem_path = dump_filesystem_snapshot(
                filesystem, target, session_id=session_id
            )

    return session_path, filesystem_path


def _resolve_filesystem_target(target: Path, session_id: str | None) -> Path:
    """Resolve the target path for a filesystem archive.

    Args:
        target: Target path (directory or .jsonl file path).
        session_id: Session identifier for filename generation.

    Returns:
        Resolved path for the .fs.zip archive.

    Raises:
        ValueError: If target is a directory and session_id is not provided.
    """
    target = target.expanduser()

    if target.is_dir():
        if session_id is None:
            msg = "session_id is required when target is a directory"
            raise ValueError(msg)
        return target / f"{session_id}.fs.zip"

    # If target is a .jsonl file, create sibling .fs.zip
    if target.suffix == ".jsonl":
        return target.with_suffix(".fs.zip")

    # Otherwise, assume it's a desired archive name
    if not target.name.endswith(".fs.zip"):
        return target.with_suffix(".fs.zip")

    return target
