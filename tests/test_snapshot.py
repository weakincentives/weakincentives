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

"""Tests for snapshot capture/restore and JSON round-trip."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from weakincentives.core.errors import SnapshotError
from weakincentives.core.session import Session
from weakincentives.core.snapshot import (
    Snapshot,
    Snapshotable,
    TypeRegistry,
    capture,
    restore,
)


@dataclass(frozen=True)
class Inner:
    label: str
    value: int


@dataclass(frozen=True)
class Plan:
    steps: tuple[str, ...] = ()
    inner: Inner | None = None


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    registry.register(Plan)
    registry.register(Inner)
    return registry


def test_capture_and_restore_round_trip() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("a", "b")))
    snapshot = capture(session)
    session[Plan].clear()
    restore(session, snapshot)
    latest = session[Plan].latest()
    assert latest is not None
    assert latest.steps == ("a", "b")


def test_json_round_trip() -> None:
    session = Session()
    session[Plan].seed(Plan(steps=("a",), inner=Inner(label="x", value=7)))
    snapshot = capture(session)
    raw = snapshot.to_json(registry=_registry())
    parsed = Snapshot.from_json(raw, registry=_registry())
    assert Plan in parsed.slices
    items = parsed.slices[Plan]
    assert len(items) == 1
    plan = items[0]
    assert isinstance(plan, Plan)
    assert plan.steps == ("a",)
    assert plan.inner == Inner(label="x", value=7)


def test_register_rejects_non_dataclass() -> None:
    class NotADataclass:
        pass

    registry = TypeRegistry()
    with pytest.raises(SnapshotError, match="not a dataclass"):
        registry.register(NotADataclass)


def test_register_collision_raises() -> None:
    registry = TypeRegistry()
    registry.register(Plan)

    @dataclass(frozen=True)
    class _LocalCollision:
        pass

    other = _LocalCollision
    other.__module__ = Plan.__module__
    other.__qualname__ = Plan.__qualname__
    with pytest.raises(SnapshotError, match="identifier collision"):
        registry.register(other)


def test_register_same_class_is_idempotent() -> None:
    registry = TypeRegistry()
    registry.register(Plan)
    registry.register(Plan)
    assert registry.identifier_for(Plan).endswith("Plan")


def test_identifier_for_unregistered_raises() -> None:
    registry = TypeRegistry()
    with pytest.raises(SnapshotError, match="not registered"):
        registry.identifier_for(Plan)


def test_resolve_unknown_identifier_raises() -> None:
    registry = TypeRegistry()
    with pytest.raises(SnapshotError, match="unknown type identifier"):
        registry.resolve("nope")


def test_from_json_rejects_invalid_json() -> None:
    with pytest.raises(SnapshotError, match="invalid snapshot JSON"):
        Snapshot.from_json("{", registry=_registry())


def test_from_json_rejects_non_object_payload() -> None:
    with pytest.raises(SnapshotError, match="must be an object"):
        Snapshot.from_json("[]", registry=_registry())


def test_from_json_rejects_wrong_schema_version() -> None:
    raw = '{"schema_version": "999", "created_at": "x", "slices": []}'
    with pytest.raises(SnapshotError, match="unsupported snapshot schema"):
        Snapshot.from_json(raw, registry=_registry())


def test_from_json_rejects_invalid_timestamp() -> None:
    raw = '{"schema_version": "1", "created_at": "nope", "slices": []}'
    with pytest.raises(SnapshotError, match="invalid created_at"):
        Snapshot.from_json(raw, registry=_registry())


def test_encode_rejects_unsupported_value() -> None:
    @dataclass(frozen=True)
    class Holder:
        payload: object

    registry = TypeRegistry()
    registry.register(Holder)
    session = Session()
    session[Holder].seed(Holder(payload=object()))
    snapshot = capture(session)
    with pytest.raises(SnapshotError, match="unsupported value"):
        snapshot.to_json(registry=registry)


def test_decode_rejects_missing_type_tag() -> None:
    type_id = f"{Plan.__module__}.{Plan.__qualname__}"
    raw = (
        f'{{"schema_version": "1", "created_at": "2025-01-01T00:00:00", '
        f'"slices": [{{"type": "{type_id}", '
        f'"items": [{{"steps": []}}]}}]}}'
    )
    with pytest.raises(SnapshotError, match="missing __type__"):
        Snapshot.from_json(raw, registry=_registry())


def test_decode_rejects_unsupported_payload() -> None:
    type_id = f"{Plan.__module__}.{Plan.__qualname__}"
    raw = (
        f'{{"schema_version": "1", "created_at": "2025-01-01T00:00:00", '
        f'"slices": [{{"type": "{type_id}", "items": [42.5]}}]}}'
    )
    parsed = Snapshot.from_json(raw, registry=_registry())
    assert parsed.slices[Plan] == (42.5,)


def test_snapshotable_protocol_runtime_check() -> None:
    class StubResource:
        def snapshot(self) -> int:
            return 1

        def restore(self, state: int) -> None:
            del state

    assert isinstance(StubResource(), Snapshotable)


def test_decode_unsupported_payload_at_top_level() -> None:
    type_id = f"{Plan.__module__}.{Plan.__qualname__}"
    raw = (
        f'{{"schema_version": "1", "created_at": "2025-01-01T00:00:00", '
        f'"slices": [{{"type": "{type_id}", "items": [true]}}]}}'
    )
    parsed = Snapshot.from_json(raw, registry=_registry())
    assert parsed.slices[Plan] == (True,)
