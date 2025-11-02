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

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pytest

from weakincentives.session import snapshots
from weakincentives.session.snapshots import (
    SNAPSHOT_SCHEMA_VERSION,
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    _ensure_timezone,
    _infer_item_type,
    _resolve_type,
    normalize_snapshot_state,
)


@dataclass(slots=True, frozen=True)
class SnapshotItem:
    value: int


def make_snapshot_payload() -> dict[str, object]:
    """Return a JSON-ready payload for a minimal snapshot."""

    snapshot = Snapshot(
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        slices={SnapshotItem: (SnapshotItem(1),)},
    )
    return json.loads(snapshot.to_json())


def test_normalize_snapshot_state_validates_keys() -> None:
    with pytest.raises(ValueError):
        normalize_snapshot_state({"not a type": ()})  # type: ignore[arg-type]


def test_normalize_snapshot_state_rejects_nondataclass_values() -> None:
    with pytest.raises(ValueError):
        normalize_snapshot_state({SnapshotItem: ("value",)})  # type: ignore[arg-type]


def test_normalize_snapshot_state_reports_serialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(slots=True, frozen=True)
    class BadItem:
        value: int

    def failing_dump(value: object) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(snapshots, "dump", failing_dump)

    with pytest.raises(ValueError):
        normalize_snapshot_state({BadItem: (BadItem(1),)})


def test_resolve_type_validation_errors() -> None:
    with pytest.raises(SnapshotRestoreError):
        _resolve_type("invalid")

    with pytest.raises(SnapshotRestoreError):
        _resolve_type("weakincentives.session.snapshots:MissingType")

    with pytest.raises(SnapshotRestoreError):
        _resolve_type("weakincentives.session.snapshots:normalize_snapshot_state")


def test_ensure_timezone_adds_utc() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    assert _ensure_timezone(naive).tzinfo is UTC

    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert _ensure_timezone(aware) is aware


def test_infer_item_type_requires_uniform_dataclasses() -> None:
    @dataclass(slots=True, frozen=True)
    class AnotherItem:
        value: int

    with pytest.raises(SnapshotSerializationError):
        _infer_item_type(SnapshotItem, (SnapshotItem(1), AnotherItem(2)))

    assert _infer_item_type(SnapshotItem, (SnapshotItem(1),)) is SnapshotItem
    assert _infer_item_type(SnapshotItem, ()) is SnapshotItem


def test_snapshot_normalizes_state_and_is_hashable() -> None:
    snapshot = Snapshot(
        created_at=datetime(2024, 1, 1), slices={SnapshotItem: (SnapshotItem(1),)}
    )

    assert snapshot.created_at.tzinfo is UTC
    assert isinstance(snapshot.slices, Mapping)
    assert isinstance(snapshot.slices, types.MappingProxyType)
    assert hash(snapshot)


def test_snapshot_to_json_surfaces_serialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = Snapshot(
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        slices={SnapshotItem: (SnapshotItem(1),)},
    )

    def failing_dump(value: object) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(snapshots, "dump", failing_dump)

    with pytest.raises(SnapshotSerializationError):
        snapshot.to_json()


def test_snapshot_from_json_error_branches() -> None:
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json("{")

    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps([]))

    payload = make_snapshot_payload()
    payload["version"] = "2"
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))

    payload = make_snapshot_payload()
    payload["version"] = SNAPSHOT_SCHEMA_VERSION
    payload["created_at"] = 123
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))
    payload["created_at"] = "not-a-timestamp"
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))

    payload = make_snapshot_payload()
    payload["created_at"] = make_snapshot_payload()["created_at"]
    payload["slices"] = {}
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))
    payload["slices"] = [1]
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))

    payload = make_snapshot_payload()
    slices = cast(list[object], payload["slices"])
    bad_entry = cast(dict[str, object], slices[0])
    bad_entry["slice_type"] = 1
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))

    payload = make_snapshot_payload()
    slices = cast(list[object], payload["slices"])
    bad_entry = cast(dict[str, object], slices[0])
    bad_entry["slice_type"] = "builtins:int"
    bad_entry["item_type"] = "builtins:int"
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))

    payload = make_snapshot_payload()
    slices = cast(list[object], payload["slices"])
    bad_entry = cast(dict[str, object], slices[0])
    bad_entry["items"] = {"not": "a list"}
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))

    payload = make_snapshot_payload()
    slices = cast(list[object], payload["slices"])
    bad_entry = cast(dict[str, object], slices[0])
    bad_entry["items"] = [{}]
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))


def test_snapshot_from_json_success_sets_timezone() -> None:
    payload = make_snapshot_payload()
    json_payload = json.dumps(payload)

    restored = Snapshot.from_json(json_payload)

    assert restored.created_at.tzinfo is UTC
    assert restored.slices[SnapshotItem] == (SnapshotItem(1),)
