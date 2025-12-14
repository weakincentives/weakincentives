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

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from .dbc import dbc_enabled
from .runtime.annotations import build_header
from .runtime.session import Session, Snapshot, iter_sessions_bottom_up
from .serde._utils import type_identifier

logger = logging.getLogger(__name__)


def dump_session(root_session: Session, target: str | Path) -> Path | None:
    """Persist a session tree to a JSONL snapshot file.

    The first line of the file is a header containing annotation metadata for
    all dataclass types present in the snapshots. Each subsequent line contains
    a serialized snapshot for a session in the tree rooted at ``root_session``.
    Snapshots are written from the root down to the leaves so that the primary
    session appears first.

    The provided ``target`` may be a directory or a file path. Regardless of
    input, the final snapshot file is named ``<root_session_id>.jsonl``.
    """

    with dbc_enabled(False):
        target_path = _resolve_target(Path(target), root_session)
        snapshots, type_ids = _collect_snapshots(root_session)
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

        # Build and serialize header with annotation metadata
        header = build_header(type_ids)
        header_json = json.dumps(header, sort_keys=True)

        # Write header followed by snapshot lines
        lines = [header_json, *snapshots]
        payload = "\n".join(lines) + "\n"
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


def _collect_snapshots(root_session: Session) -> tuple[list[str], set[str]]:
    """Collect snapshots and type identifiers from a session tree.

    Returns:
        A tuple of (snapshot JSON strings, set of type identifiers).
    """
    snapshots: list[str] = []
    type_ids: set[str] = set()
    with dbc_enabled(False):
        for session in _iter_sessions_top_down(root_session):
            snapshot = session.snapshot()
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
            type_ids.update(_collect_type_ids(snapshot))
        return snapshots, type_ids


def _collect_type_ids(snapshot: Snapshot) -> set[str]:
    """Extract all dataclass type identifiers from a snapshot."""
    ids: set[str] = set()
    for slice_type, values in snapshot.slices.items():
        ids.add(type_identifier(slice_type))
        for value in values:
            ids.add(type_identifier(type(value)))
    return ids


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
