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
from pathlib import Path
from typing import cast

import pytest

from weakincentives.runtime.events import InProcessEventBus
from weakincentives.runtime.session import (
    Session,
    replace_latest,
    select_latest,
    snapshots,
)
from weakincentives.runtime.session.snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    _ensure_timezone,
    _infer_item_type,
    _resolve_type,
    normalize_snapshot_state,
)
from weakincentives.tools.vfs import VfsFile, VfsPath, VirtualFileSystem


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
        normalize_snapshot_state({"not a type": ()})


def test_normalize_snapshot_state_rejects_nondataclass_values() -> None:
    with pytest.raises(ValueError):
        normalize_snapshot_state({SnapshotItem: ("value",)})


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


def test_snapshot_clones_and_restores_vfs(tmp_path: Path) -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    root = tmp_path / "vfs"
    nested = root / "docs"
    nested.mkdir(parents=True)
    original_file = nested / "note.txt"
    original_file.write_text("hello", encoding="utf-8")

    vfs_file = VfsFile(
        path=VfsPath(("docs", "note.txt")),
        encoding="utf-8",
        size_bytes=5,
        version=1,
        created_at=timestamp,
        updated_at=timestamp,
    )
    snapshot_state = VirtualFileSystem(root_path=str(root), files=(vfs_file,))
    session.register_reducer(VirtualFileSystem, replace_latest)
    session.seed_slice(VirtualFileSystem, (snapshot_state,))

    snapshot = session.snapshot()
    cloned_vfs = cast(VirtualFileSystem, snapshot.slices[VirtualFileSystem][0])
    cloned_root = Path(cloned_vfs.root_path)
    assert cloned_root != root
    assert (cloned_root / "docs" / "note.txt").read_text(encoding="utf-8") == "hello"

    original_file.write_text("changed", encoding="utf-8")
    assert (cloned_root / "docs" / "note.txt").read_text(encoding="utf-8") == "hello"

    session.rollback(snapshot)
    restored = select_latest(session, VirtualFileSystem)
    assert restored is not None
    restored_root = Path(restored.root_path)
    assert restored_root != cloned_root
    assert (restored_root / "docs" / "note.txt").read_text(encoding="utf-8") == "hello"


def test_snapshot_handles_missing_vfs_root(tmp_path: Path) -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)
    missing_root = tmp_path / "missing"
    session.register_reducer(VirtualFileSystem, replace_latest)
    session.seed_slice(
        VirtualFileSystem,
        (VirtualFileSystem(root_path=str(missing_root), files=()),),
    )

    snapshot = session.snapshot()
    cloned_vfs = cast(VirtualFileSystem, snapshot.slices[VirtualFileSystem][0])
    cloned_root = Path(cloned_vfs.root_path)
    assert cloned_root.exists()
    assert list(cloned_root.iterdir()) == []


@pytest.mark.parametrize(
    "raw_payload",
    [
        "{",
        json.dumps([]),
        make_snapshot_payload_with(version="2"),
        make_snapshot_payload_with(created_at=123),
        make_snapshot_payload_with(created_at="not-a-timestamp"),
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
