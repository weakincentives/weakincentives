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

from dataclasses import dataclass

from weakincentives.runtime.events import EventBus
from weakincentives.runtime.session import Session
from weakincentives.runtime.session._types import ReducerContextProtocol
from weakincentives.runtime.session.reducers import (
    append_all,
    replace_latest_by,
    upsert_by,
)


@dataclass(slots=True)
class _Sample:
    key: str
    data: str


@dataclass(slots=True)
class _Context(ReducerContextProtocol):
    session: Session
    event_bus: EventBus


def test_replace_latest_by_replaces_matching_entry() -> None:
    reducer = replace_latest_by(lambda item: item.key)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    initial = (_Sample("a", "first"), _Sample("b", "second"))

    updated = reducer(
        initial,
        _Sample("a", "updated"),
        context=context,
    )

    assert len(updated) == 2
    assert updated[-1].key == "a"
    assert updated[-1].data == "updated"
    assert any(item.data == "second" for item in updated)


def test_append_all_always_appends() -> None:
    """append_all appends unconditionally (ledger semantics)."""
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    initial = (_Sample("a", "first"),)

    # Append same value - should still append
    updated = append_all(initial, _Sample("a", "first"), context=context)

    assert len(updated) == 2
    assert updated[0] == _Sample("a", "first")
    assert updated[1] == _Sample("a", "first")


def test_append_all_appends_to_empty_slice() -> None:
    """append_all works on empty slices."""
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)

    updated = append_all((), _Sample("a", "first"), context=context)

    assert len(updated) == 1
    assert updated[0] == _Sample("a", "first")


def test_replace_latest_by_handles_duplicate_keys() -> None:
    """replace_latest_by removes duplicates when replacing."""
    reducer = replace_latest_by(lambda item: item.key)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    # Multiple items with the same key "a"
    initial = (_Sample("a", "first"), _Sample("a", "duplicate"), _Sample("b", "other"))

    updated = reducer(
        initial,
        _Sample("a", "replaced"),
        context=context,
    )

    # Should replace first "a", skip duplicate "a", keep "b"
    assert len(updated) == 2
    assert any(item.key == "a" and item.data == "replaced" for item in updated)
    assert any(item.key == "b" for item in updated)


def test_upsert_by_replaces_first_matching_key() -> None:
    """upsert_by replaces first matching entry and removes duplicates."""
    reducer = upsert_by(lambda item: item.key)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    initial = (_Sample("a", "first"), _Sample("b", "second"))

    updated = reducer(
        initial,
        _Sample("a", "updated"),
        context=context,
    )

    assert len(updated) == 2
    assert updated[0] == _Sample("a", "updated")
    assert updated[1] == _Sample("b", "second")


def test_upsert_by_appends_when_no_match() -> None:
    """upsert_by appends when key not found."""
    reducer = upsert_by(lambda item: item.key)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    initial = (_Sample("a", "first"),)

    updated = reducer(
        initial,
        _Sample("b", "new"),
        context=context,
    )

    assert len(updated) == 2
    assert updated[0] == _Sample("a", "first")
    assert updated[1] == _Sample("b", "new")


def test_upsert_by_removes_duplicates() -> None:
    """upsert_by removes duplicate keys when upserting."""
    reducer = upsert_by(lambda item: item.key)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    # Multiple items with the same key "a"
    initial = (_Sample("a", "first"), _Sample("a", "duplicate"), _Sample("b", "other"))

    updated = reducer(
        initial,
        _Sample("a", "replaced"),
        context=context,
    )

    # Should upsert first "a", skip duplicate "a", keep "b"
    assert len(updated) == 2
    assert updated[0] == _Sample("a", "replaced")
    assert updated[1] == _Sample("b", "other")
