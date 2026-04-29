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

"""Debug bundles: transcript + session snapshot + metadata in JSON.

A :class:`DebugBundle` packages everything needed to reconstruct an agent
run after the fact: the typed slice state captured by the spine's
:class:`Snapshot`, the recorded :class:`Transcript`, and a metadata dict
the caller controls. Bundles round-trip through JSON via the spine's
:class:`TypeRegistry`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast, final

from weakincentives.core import (
    Session,
    Snapshot,
    SnapshotError,
    TypeRegistry,
    capture,
)
from weakincentives.transcript import Transcript, TranscriptEntry

__all__ = [
    "DebugBundle",
    "from_session",
]

_SCHEMA_VERSION = "1"


def _empty_metadata() -> Mapping[str, object]:
    return {}


@final
@dataclass(frozen=True, slots=True)
class DebugBundle:
    """A serializable capture of a session run."""

    snapshot: Snapshot
    entries: tuple[TranscriptEntry, ...]
    created_at: datetime
    metadata: Mapping[str, object] = field(default_factory=_empty_metadata)

    def to_json(self, *, registry: TypeRegistry) -> str:
        payload: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
            "snapshot": json.loads(self.snapshot.to_json(registry=registry)),
            "entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "kind": entry.kind,
                    "payload": dict(entry.payload),
                }
                for entry in self.entries
            ],
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str, *, registry: TypeRegistry) -> DebugBundle:
        try:
            decoded: object = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SnapshotError("invalid debug bundle JSON") from error
        if not isinstance(decoded, dict):
            raise SnapshotError("debug bundle payload must be an object")
        payload = cast("dict[str, Any]", decoded)
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise SnapshotError(
                f"unsupported debug bundle schema: {payload.get('schema_version')!r}"
            )
        created_at = _parse_timestamp(payload.get("created_at"))
        metadata = _parse_metadata(payload.get("metadata"))
        snapshot = _parse_snapshot(payload.get("snapshot"), registry)
        entries = _parse_entries(payload.get("entries"))
        return cls(
            snapshot=snapshot,
            entries=entries,
            created_at=created_at,
            metadata=metadata,
        )


def from_session(
    session: Session,
    transcript: Transcript,
    *,
    metadata: Mapping[str, object] | None = None,
) -> DebugBundle:
    """Capture a :class:`DebugBundle` for ``session`` + ``transcript``."""
    return DebugBundle(
        snapshot=capture(session),
        entries=transcript.entries(),
        created_at=datetime.now(UTC),
        metadata=metadata if metadata is not None else _empty_metadata(),
    )


# =============================================================================
# Parsing helpers
# =============================================================================


def _parse_timestamp(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise SnapshotError("debug bundle created_at must be a string")
    try:
        return datetime.fromisoformat(raw)
    except ValueError as error:
        raise SnapshotError("invalid created_at timestamp") from error


def _parse_metadata(raw: object) -> Mapping[str, object]:
    if raw is None:
        return _empty_metadata()
    if not isinstance(raw, dict):
        raise SnapshotError("debug bundle metadata must be an object")
    raw_dict = cast("dict[str, object]", raw)
    return {str(key): value for key, value in raw_dict.items()}


def _parse_snapshot(raw: object, registry: TypeRegistry) -> Snapshot:
    if not isinstance(raw, dict):
        raise SnapshotError("debug bundle snapshot must be an object")
    return Snapshot.from_json(json.dumps(raw), registry=registry)


def _parse_entries(raw: object) -> tuple[TranscriptEntry, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SnapshotError("debug bundle entries must be a list")
    parsed: list[TranscriptEntry] = []
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            raise SnapshotError("transcript entry must be an object")
        entry = cast("dict[str, Any]", item)
        timestamp_raw = entry.get("timestamp")
        kind = entry.get("kind")
        payload = entry.get("payload", {})
        if not isinstance(timestamp_raw, str) or not isinstance(kind, str):
            raise SnapshotError("transcript entry malformed")
        if not isinstance(payload, dict):
            raise SnapshotError("transcript entry payload must be an object")
        parsed.append(
            TranscriptEntry(
                timestamp=datetime.fromisoformat(timestamp_raw),
                kind=kind,
                payload=cast("dict[str, object]", payload),
            )
        )
    return tuple(parsed)
