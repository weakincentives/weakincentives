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
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

from weakincentives.runtime.session import snapshots
from weakincentives.runtime.session.snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    SnapshotState,
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


def make_snapshot_payload_with(**mutations: object) -> str:
    """Return the snapshot payload with overrides serialized to JSON."""

    payload = deepcopy(make_snapshot_payload())
    payload.update(mutations)
    return json.dumps(payload)


def make_snapshot_payload_with_slice_mutation(**entry_overrides: object) -> str:
    payload = make_snapshot_payload()
    slices = cast(list[object], payload["slices"])
    base_entry = cast(dict[str, object], deepcopy(slices[0]))
    base_entry.update(entry_overrides)
    return make_snapshot_payload_with(slices=[base_entry])


def test_normalize_snapshot_state_validates_keys() -> None:
    with pytest.raises(ValueError):
        normalize_snapshot_state(cast(SnapshotState, {"not a type": ()}))


def test_normalize_snapshot_state_rejects_nondataclass_values() -> None:
    with pytest.raises(ValueError):
        normalize_snapshot_state(cast(SnapshotState, {SnapshotItem: ("value",)}))


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
        _resolve_type("weakincentives.runtime.session.snapshots:MissingType")

    with pytest.raises(SnapshotRestoreError):
        _resolve_type(
            "weakincentives.runtime.session.snapshots:normalize_snapshot_state"
        )


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


def test_snapshot_serializes_relationship_metadata() -> None:
    parent_id = uuid4()
    child_one = uuid4()
    child_two = uuid4()

    snapshot = Snapshot(
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        parent_id=parent_id,
        children_ids=(child_one, child_two),
        slices={SnapshotItem: (SnapshotItem(1),)},
    )

    payload = json.loads(snapshot.to_json())

    assert payload["parent_id"] == str(parent_id)
    assert payload["children_ids"] == [str(child_one), str(child_two)]

    restored = Snapshot.from_json(json.dumps(payload))

    assert restored.parent_id == parent_id
    assert restored.children_ids == (child_one, child_two)


def test_snapshot_rejects_non_string_tags() -> None:
    with pytest.raises(SnapshotSerializationError):
        Snapshot(
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            slices={SnapshotItem: (SnapshotItem(1),)},
            tags={"scope": "session", "invalid": 123},
        )


def test_snapshot_from_json_rejects_non_string_tags() -> None:
    payload = make_snapshot_payload()
    payload["tags"] = {"scope": 123}

    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(json.dumps(payload))


def test_snapshot_round_trip_injects_parent_tag() -> None:
    parent_id = uuid4()
    snapshot = Snapshot(
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        parent_id=parent_id,
        slices={SnapshotItem: (SnapshotItem(1),)},
    )

    restored = Snapshot.from_json(snapshot.to_json())

    assert restored.tags["parent_session_id"] == str(parent_id)


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


@pytest.mark.parametrize(
    "raw_payload",
    [
        "{",
        json.dumps([]),
        make_snapshot_payload_with(version="2"),
        make_snapshot_payload_with(version=2),
        make_snapshot_payload_with(created_at=123),
        make_snapshot_payload_with(created_at="not-a-timestamp"),
        make_snapshot_payload_with(parent_id=123),
        make_snapshot_payload_with(parent_id="not-a-uuid"),
        make_snapshot_payload_with(children_ids="not-a-list"),
        make_snapshot_payload_with(children_ids=[123]),
        make_snapshot_payload_with(children_ids=["not-a-uuid"]),
        make_snapshot_payload_with(slices={}),
        make_snapshot_payload_with(slices=[1]),
        make_snapshot_payload_with_slice_mutation(slice_type=1),
        make_snapshot_payload_with_slice_mutation(
            slice_type="builtins:int",
            item_type="builtins:int",
        ),
        make_snapshot_payload_with_slice_mutation(items={"not": "a list"}),
        make_snapshot_payload_with_slice_mutation(items=[{}]),
        make_snapshot_payload_with_slice_mutation(items=["not-a-mapping"]),
    ],
)
def test_snapshot_from_json_error_branches(raw_payload: str) -> None:
    with pytest.raises(SnapshotRestoreError):
        Snapshot.from_json(raw_payload)


def test_snapshot_from_json_success_sets_timezone() -> None:
    payload = make_snapshot_payload()
    json_payload = json.dumps(payload)

    restored = Snapshot.from_json(json_payload)

    assert restored.created_at.tzinfo is UTC
    assert restored.slices[SnapshotItem] == (SnapshotItem(1),)
    assert restored.parent_id is None
    assert restored.children_ids == ()
