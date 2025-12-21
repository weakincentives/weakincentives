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


def test_upsert_by_replaces_first_duplicate_and_removes_others() -> None:
    """upsert_by replaces first matching key and discards subsequent duplicates."""
    reducer = upsert_by(lambda item: item.key)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    # Slice with duplicate keys - this covers the branch at line 61->64
    initial = (
        _Sample("a", "first"),
        _Sample("a", "duplicate"),  # duplicate key
        _Sample("b", "other"),
    )

    updated = reducer(
        initial,
        _Sample("a", "updated"),
        context=context,
    )

    # Should have 2 items: the updated "a" and "b"
    assert len(updated) == 2
    assert updated[0] == _Sample("a", "updated")
    assert updated[1] == _Sample("b", "other")


def test_upsert_by_appends_when_key_not_found() -> None:
    """upsert_by appends when key doesn't exist."""
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


@dataclass(slots=True)
class _ConcurrentValue:
    value: int


def test_session_retries_state_write_if_slice_changes_during_dispatch() -> None:
    """Session retries reducer loop when state mutates between lock acquisitions."""

    session = Session()
    mutated = False

    def reducer(
        previous: tuple[_ConcurrentValue, ...],
        event: _ConcurrentValue,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[_ConcurrentValue, ...]:
        nonlocal mutated
        if not mutated:
            mutated = True
            context.session._mutation_seed_slice(
                _ConcurrentValue, (_ConcurrentValue(event.value + 1),)
            )
        return (event,)

    session._mutation_register_reducer(_ConcurrentValue, reducer)

    session._mutation_dispatch_event(_ConcurrentValue, _ConcurrentValue(1))

    assert session._state[_ConcurrentValue] == (_ConcurrentValue(1),)
