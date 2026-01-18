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

from weakincentives.runtime.clock import FakeClock
from weakincentives.runtime.events import Dispatcher
from weakincentives.runtime.session import (
    Append,
    MemorySlice,
    Replace,
    Session,
)
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
    dispatcher: Dispatcher


def test_replace_latest_by_replaces_matching_entry(clock: FakeClock) -> None:
    reducer = replace_latest_by(lambda item: item.key)
    session = Session(clock=clock)
    context = _Context(session=session, dispatcher=session.dispatcher)
    initial = MemorySlice((_Sample("a", "first"), _Sample("b", "second")))

    op = reducer(
        initial.view(),
        _Sample("a", "updated"),
        context=context,
    )

    assert isinstance(op, Replace)
    updated = op.items
    assert len(updated) == 2
    assert updated[-1].key == "a"
    assert updated[-1].data == "updated"
    assert any(item.data == "second" for item in updated)


def test_append_all_always_appends(clock: FakeClock) -> None:
    """append_all appends unconditionally (ledger semantics)."""
    session = Session(clock=clock)
    context = _Context(session=session, dispatcher=session.dispatcher)
    initial = MemorySlice((_Sample("a", "first"),))

    # Append same value - should still append (returns Append operation)
    op = append_all(initial.view(), _Sample("a", "first"), context=context)

    assert isinstance(op, Append)
    assert op.item == _Sample("a", "first")


def test_append_all_appends_to_empty_slice(clock: FakeClock) -> None:
    """append_all works on empty slices."""
    session = Session(clock=clock)
    context = _Context(session=session, dispatcher=session.dispatcher)
    initial = MemorySlice[_Sample]()

    op = append_all(initial.view(), _Sample("a", "first"), context=context)

    assert isinstance(op, Append)
    assert op.item == _Sample("a", "first")


def test_upsert_by_replaces_first_duplicate_and_removes_others(
    clock: FakeClock,
) -> None:
    """upsert_by replaces first matching key and discards subsequent duplicates."""
    reducer = upsert_by(lambda item: item.key)
    session = Session(clock=clock)
    context = _Context(session=session, dispatcher=session.dispatcher)
    # Slice with duplicate keys - this covers the branch at line 61->64
    initial = MemorySlice(
        (
            _Sample("a", "first"),
            _Sample("a", "duplicate"),  # duplicate key
            _Sample("b", "other"),
        )
    )

    op = reducer(
        initial.view(),
        _Sample("a", "updated"),
        context=context,
    )

    assert isinstance(op, Replace)
    updated = op.items
    # Should have 2 items: the updated "a" and "b"
    assert len(updated) == 2
    assert updated[0] == _Sample("b", "other")
    assert updated[1] == _Sample("a", "updated")


def test_upsert_by_appends_when_key_not_found(clock: FakeClock) -> None:
    """upsert_by appends when key doesn't exist."""
    reducer = upsert_by(lambda item: item.key)
    session = Session(clock=clock)
    context = _Context(session=session, dispatcher=session.dispatcher)
    initial = MemorySlice((_Sample("a", "first"),))

    op = reducer(
        initial.view(),
        _Sample("b", "new"),
        context=context,
    )

    assert isinstance(op, Replace)
    updated = op.items
    assert len(updated) == 2
    assert updated[0] == _Sample("a", "first")
    assert updated[1] == _Sample("b", "new")
