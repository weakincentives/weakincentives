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

"""Tests for ``weakincentives.debug``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytest

from weakincentives.core import Session, SnapshotError, TypeRegistry
from weakincentives.debug import DebugBundle, from_session
from weakincentives.transcript import Transcript, TranscriptEntry


@dataclass(frozen=True)
class Plan:
    steps: tuple[str, ...] = ()


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    registry.register(Plan)
    return registry


def test_from_session_captures_snapshot_and_transcript() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("a", "b")))
    transcript = Transcript()
    transcript.append(
        TranscriptEntry(
            timestamp=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
            kind="x",
            payload={"k": "v"},
        )
    )
    bundle = from_session(session, transcript, metadata={"label": "test"})
    assert bundle.metadata["label"] == "test"
    assert len(bundle.entries) == 1
    assert Plan in bundle.snapshot.slices


def test_round_trip_through_json() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("a",)))
    transcript = Transcript()
    transcript.append(
        TranscriptEntry(
            timestamp=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
            kind="kind",
            payload={"k": 1},
        )
    )
    bundle = from_session(session, transcript, metadata={"k": "v"})
    raw = bundle.to_json(registry=_registry())
    restored = DebugBundle.from_json(raw, registry=_registry())
    assert restored.metadata == {"k": "v"}
    assert restored.entries[0].kind == "kind"
    items = restored.snapshot.slices[Plan]
    assert isinstance(items[0], Plan)


def test_from_json_rejects_invalid_json() -> None:
    with pytest.raises(SnapshotError, match="invalid debug bundle"):
        DebugBundle.from_json("{", registry=_registry())


def test_from_json_rejects_non_object() -> None:
    with pytest.raises(SnapshotError, match="must be an object"):
        DebugBundle.from_json("[]", registry=_registry())


def test_from_json_rejects_unsupported_schema() -> None:
    raw = (
        '{"schema_version": "9", "created_at": "2025-01-01T00:00:00", '
        '"metadata": {}, "snapshot": {}, "entries": []}'
    )
    with pytest.raises(SnapshotError, match="unsupported debug bundle"):
        DebugBundle.from_json(raw, registry=_registry())


def test_from_json_rejects_missing_created_at() -> None:
    raw = '{"schema_version": "1", "metadata": {}, "snapshot": {}, "entries": []}'
    with pytest.raises(SnapshotError, match="created_at"):
        DebugBundle.from_json(raw, registry=_registry())


def test_from_json_rejects_invalid_created_at() -> None:
    raw = (
        '{"schema_version": "1", "created_at": "nope", "metadata": {}, '
        '"snapshot": {}, "entries": []}'
    )
    with pytest.raises(SnapshotError, match="created_at"):
        DebugBundle.from_json(raw, registry=_registry())


def test_from_json_rejects_non_object_metadata() -> None:
    raw = (
        '{"schema_version": "1", "created_at": "2025-01-01T00:00:00", '
        '"metadata": [], "snapshot": {}, "entries": []}'
    )
    with pytest.raises(SnapshotError, match="metadata"):
        DebugBundle.from_json(raw, registry=_registry())


def test_from_json_treats_missing_metadata_as_empty() -> None:
    bundle = from_session(Session(), Transcript())
    raw = bundle.to_json(registry=_registry())
    parsed_raw = raw.replace('"metadata": {}', '"metadata": null')
    restored = DebugBundle.from_json(parsed_raw, registry=_registry())
    assert restored.metadata == {}


def test_from_json_rejects_non_object_snapshot() -> None:
    raw = (
        '{"schema_version": "1", "created_at": "2025-01-01T00:00:00", '
        '"metadata": {}, "snapshot": [], "entries": []}'
    )
    with pytest.raises(SnapshotError, match="snapshot"):
        DebugBundle.from_json(raw, registry=_registry())


def test_from_json_rejects_non_list_entries() -> None:
    bundle = from_session(Session(), Transcript())
    raw = bundle.to_json(registry=_registry()).replace(
        '"entries": []', '"entries": "no"'
    )
    with pytest.raises(SnapshotError, match="entries"):
        DebugBundle.from_json(raw, registry=_registry())


def test_from_json_treats_missing_entries_as_empty() -> None:
    bundle = from_session(Session(), Transcript())
    raw = bundle.to_json(registry=_registry()).replace(
        '"entries": []', '"entries": null'
    )
    restored = DebugBundle.from_json(raw, registry=_registry())
    assert restored.entries == ()


def test_from_json_rejects_malformed_entry_object() -> None:
    bundle = from_session(Session(), Transcript())
    raw = bundle.to_json(registry=_registry()).replace(
        '"entries": []', '"entries": ["not an object"]'
    )
    with pytest.raises(SnapshotError, match="entry"):
        DebugBundle.from_json(raw, registry=_registry())


def test_from_json_rejects_malformed_entry_fields() -> None:
    bundle = from_session(Session(), Transcript())
    raw = bundle.to_json(registry=_registry()).replace(
        '"entries": []',
        '"entries": [{"timestamp": 1, "kind": "x", "payload": {}}]',
    )
    with pytest.raises(SnapshotError, match="malformed"):
        DebugBundle.from_json(raw, registry=_registry())


def test_from_json_rejects_non_object_entry_payload() -> None:
    bundle = from_session(Session(), Transcript())
    raw = bundle.to_json(registry=_registry()).replace(
        '"entries": []',
        (
            '"entries": [{"timestamp": "2025-01-01T00:00:00", '
            '"kind": "x", "payload": "no"}]'
        ),
    )
    with pytest.raises(SnapshotError, match="payload"):
        DebugBundle.from_json(raw, registry=_registry())
