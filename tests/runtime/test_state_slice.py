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

"""Tests for declarative state slice decorators."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pytest

from weakincentives.runtime.session import (
    get_state_slice_meta,
    is_state_slice,
    reducer,
    state_slice,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory


# ──────────────────────────────────────────────────────────────────────
# Test fixtures - Event types and state slices
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AddItem:
    """Event to add an item."""

    item: str


@dataclass(frozen=True)
class RemoveItem:
    """Event to remove an item."""

    item: str


@dataclass(frozen=True)
class ClearItems:
    """Event to clear all items."""


@dataclass(frozen=True)
class Increment:
    """Event to increment a counter."""

    amount: int = 1


@dataclass(frozen=True)
class Reset:
    """Event to reset counter to zero."""


@state_slice
@dataclass(frozen=True)
class ItemList:
    """Simple state slice with no initial factory."""

    items: tuple[str, ...]

    @reducer(on=AddItem)
    def add_item(self, event: AddItem) -> ItemList:
        return replace(self, items=(*self.items, event.item))

    @reducer(on=RemoveItem)
    def remove_item(self, event: RemoveItem) -> ItemList:
        return replace(self, items=tuple(i for i in self.items if i != event.item))

    @reducer(on=ClearItems)
    def clear(self, event: ClearItems) -> ItemList:
        del event
        return replace(self, items=())


def _counter_initial() -> Counter:
    return Counter()


@state_slice(initial=_counter_initial)
@dataclass(frozen=True)
class Counter:
    """State slice with initial factory."""

    count: int = 0

    @reducer(on=Increment)
    def increment(self, event: Increment) -> Counter:
        return replace(self, count=self.count + event.amount)

    @reducer(on=Reset)
    def reset(self, event: Reset) -> Counter:
        del event
        return Counter()


# ──────────────────────────────────────────────────────────────────────
# @state_slice decorator tests
# ──────────────────────────────────────────────────────────────────────


def test_state_slice_marks_class() -> None:
    """Verify @state_slice adds metadata to the class."""
    assert is_state_slice(ItemList)
    assert is_state_slice(Counter)


def test_state_slice_extracts_reducers() -> None:
    """Verify @state_slice extracts @reducer decorated methods."""
    meta = get_state_slice_meta(ItemList)

    assert meta is not None
    assert len(meta.reducers) == 3

    event_types = {r.event_type for r in meta.reducers}
    assert AddItem in event_types
    assert RemoveItem in event_types
    assert ClearItems in event_types


def test_state_slice_stores_initial_factory() -> None:
    """Verify @state_slice stores the initial factory."""
    item_meta = get_state_slice_meta(ItemList)
    counter_meta = get_state_slice_meta(Counter)

    assert item_meta is not None
    assert item_meta.initial_factory is None

    assert counter_meta is not None
    assert counter_meta.initial_factory is not None
    assert counter_meta.initial_factory() == Counter()


def test_state_slice_requires_dataclass() -> None:
    """Verify @state_slice rejects non-dataclass."""
    with pytest.raises(TypeError, match="requires a dataclass"):

        @state_slice
        class NotADataclass:
            pass


def test_state_slice_requires_frozen_dataclass() -> None:
    """Verify @state_slice rejects non-frozen dataclass."""
    with pytest.raises(TypeError, match="frozen dataclass"):

        @state_slice
        @dataclass
        class NotFrozen:
            value: int


def test_is_state_slice_returns_false_for_non_slices() -> None:
    """Verify is_state_slice returns False for regular classes."""
    assert not is_state_slice(str)
    assert not is_state_slice(int)
    assert not is_state_slice(AddItem)


def test_get_state_slice_meta_returns_none_for_non_slices() -> None:
    """Verify get_state_slice_meta returns None for regular classes."""
    assert get_state_slice_meta(str) is None
    assert get_state_slice_meta(AddItem) is None


# ──────────────────────────────────────────────────────────────────────
# session.install() tests
# ──────────────────────────────────────────────────────────────────────


def test_install_registers_reducers(session_factory: SessionFactory) -> None:
    """Verify install() registers all reducer methods."""
    session, _ = session_factory()
    session.install(ItemList)

    # Seed initial state
    session.mutate(ItemList).seed(ItemList(items=()))

    # Dispatch events - should work via installed reducers
    session.mutate(ItemList).dispatch(AddItem(item="first"))
    session.mutate(ItemList).dispatch(AddItem(item="second"))

    result = session[ItemList].latest()
    assert result is not None
    assert result.items == ("first", "second")


def test_install_rejects_non_state_slice(session_factory: SessionFactory) -> None:
    """Verify install() rejects classes without @state_slice."""
    session, _ = session_factory()

    with pytest.raises(TypeError, match="not a state slice"):
        session.install(AddItem)


def test_install_with_initial_factory(session_factory: SessionFactory) -> None:
    """Verify install() with initial factory handles empty slice."""
    session, _ = session_factory()
    session.install(Counter)

    # Dispatch without seeding - should use initial factory
    session.mutate(Counter).dispatch(Increment(amount=5))

    result = session[Counter].latest()
    assert result is not None
    assert result.count == 5


def test_install_without_initial_factory_ignores_empty(
    session_factory: SessionFactory,
) -> None:
    """Verify install() without initial factory ignores events on empty slice."""
    session, _ = session_factory()
    session.install(ItemList)

    # Dispatch without seeding - should be ignored (no initial factory)
    session.mutate(ItemList).dispatch(AddItem(item="ignored"))

    result = session[ItemList].latest()
    assert result is None


# ──────────────────────────────────────────────────────────────────────
# session[SliceType] indexing tests
# ──────────────────────────────────────────────────────────────────────


def test_getitem_returns_query_builder(session_factory: SessionFactory) -> None:
    """Verify session[SliceType] returns QueryBuilder."""
    from weakincentives.runtime.session import QueryBuilder

    session, _ = session_factory()

    builder = session[ItemList]

    assert isinstance(builder, QueryBuilder)


def test_getitem_query_latest(session_factory: SessionFactory) -> None:
    """Verify session[SliceType].latest() works."""
    session, _ = session_factory()
    session.install(ItemList)

    session.mutate(ItemList).seed(ItemList(items=("a", "b")))

    result = session[ItemList].latest()

    assert result == ItemList(items=("a", "b"))


def test_getitem_query_all(session_factory: SessionFactory) -> None:
    """Verify session[SliceType].all() works."""
    session, _ = session_factory()
    session.mutate(ItemList).seed([ItemList(items=("a",)), ItemList(items=("b",))])

    result = session[ItemList].all()

    assert result == (ItemList(items=("a",)), ItemList(items=("b",)))


def test_getitem_query_where(session_factory: SessionFactory) -> None:
    """Verify session[SliceType].where() works."""
    session, _ = session_factory()
    session.mutate(ItemList).seed(
        [
            ItemList(items=()),
            ItemList(items=("a",)),
            ItemList(items=("a", "b")),
        ]
    )

    result = session[ItemList].where(lambda x: len(x.items) > 0)

    assert len(result) == 2


# ──────────────────────────────────────────────────────────────────────
# @reducer decorator tests
# ──────────────────────────────────────────────────────────────────────


def test_reducer_method_transforms_state(session_factory: SessionFactory) -> None:
    """Verify reducer method correctly transforms state."""
    session, _ = session_factory()
    session.install(ItemList)

    session.mutate(ItemList).seed(ItemList(items=("a", "b", "c")))
    session.mutate(ItemList).dispatch(RemoveItem(item="b"))

    result = session[ItemList].latest()
    assert result is not None
    assert result.items == ("a", "c")


def test_reducer_method_can_clear_state(session_factory: SessionFactory) -> None:
    """Verify reducer method can clear state."""
    session, _ = session_factory()
    session.install(ItemList)

    session.mutate(ItemList).seed(ItemList(items=("a", "b")))
    session.mutate(ItemList).dispatch(ClearItems())

    result = session[ItemList].latest()
    assert result is not None
    assert result.items == ()


def test_reducer_method_receives_event(session_factory: SessionFactory) -> None:
    """Verify reducer method receives event data."""
    session, _ = session_factory()
    session.install(Counter)

    session.mutate(Counter).dispatch(Increment(amount=10))
    session.mutate(Counter).dispatch(Increment(amount=5))

    result = session[Counter].latest()
    assert result is not None
    assert result.count == 15


def test_reducer_method_reset(session_factory: SessionFactory) -> None:
    """Verify reducer method can reset state."""
    session, _ = session_factory()
    session.install(Counter)

    session.mutate(Counter).dispatch(Increment(amount=100))
    session.mutate(Counter).dispatch(Reset())

    result = session[Counter].latest()
    assert result is not None
    assert result.count == 0


# ──────────────────────────────────────────────────────────────────────
# Edge cases and error handling
# ──────────────────────────────────────────────────────────────────────


def test_multiple_reducers_for_same_event(session_factory: SessionFactory) -> None:
    """Test that multiple slices can have reducers for the same event."""

    @dataclass(frozen=True)
    class SharedEvent:
        value: int

    @state_slice(initial=lambda: SliceA())
    @dataclass(frozen=True)
    class SliceA:
        value: int = 0

        @reducer(on=SharedEvent)
        def handle(self, event: SharedEvent) -> SliceA:
            return replace(self, value=event.value * 2)

    @state_slice(initial=lambda: SliceB())
    @dataclass(frozen=True)
    class SliceB:
        value: int = 0

        @reducer(on=SharedEvent)
        def handle(self, event: SharedEvent) -> SliceB:
            return replace(self, value=event.value * 3)

    session, _ = session_factory()
    session.install(SliceA)
    session.install(SliceB)

    # Dispatch to SliceA - but since both slices have reducers for
    # SharedEvent, both will be invoked (registered to same event type)
    session.mutate(SliceA).dispatch(SharedEvent(value=10))

    a_result = session[SliceA].latest()
    b_result = session[SliceB].latest()

    assert a_result is not None
    assert a_result.value == 20

    # Both slices have reducers for SharedEvent, so SliceB is also updated
    assert b_result is not None
    assert b_result.value == 30


def test_chained_dispatches(session_factory: SessionFactory) -> None:
    """Test multiple dispatches in sequence."""
    session, _ = session_factory()
    session.install(Counter)

    for i in range(1, 6):
        session.mutate(Counter).dispatch(Increment(amount=i))

    result = session[Counter].latest()
    assert result is not None
    assert result.count == 15  # 1+2+3+4+5


def test_state_slice_with_callable_syntax(session_factory: SessionFactory) -> None:
    """Test @state_slice(...) callable syntax."""
    # Counter uses the callable syntax with initial parameter
    meta = get_state_slice_meta(Counter)
    assert meta is not None
    assert meta.initial_factory is not None


def test_state_slice_without_callable_syntax(session_factory: SessionFactory) -> None:
    """Test @state_slice decorator syntax."""
    # ItemList uses the bare decorator syntax
    meta = get_state_slice_meta(ItemList)
    assert meta is not None
    assert meta.initial_factory is None


def test_reducer_qualname_set_correctly() -> None:
    """Verify generated reducers have meaningful qualnames."""
    from weakincentives.runtime.session.state_slice import _create_reducer_for_method

    generated_reducer = _create_reducer_for_method(ItemList, "add_item")

    # Access __qualname__ via getattr since it's set dynamically
    qualname = getattr(generated_reducer, "__qualname__", "")
    assert "ItemList" in qualname
    assert "add_item" in qualname


def test_install_state_slice_rejects_non_state_slice() -> None:
    """Verify install_state_slice raises TypeError for non-state-slice classes."""
    from weakincentives.runtime.session import Session
    from weakincentives.runtime.session.state_slice import install_state_slice

    @dataclass(frozen=True)
    class NotASlice:
        value: int

    session = Session()

    with pytest.raises(TypeError, match="not a state slice"):
        install_state_slice(session, NotASlice)


def test_extract_reducer_handles_none_attributes() -> None:
    """Verify _extract_reducer_metadata handles classes with None attributes."""
    from weakincentives.runtime.session.state_slice import _extract_reducer_metadata

    # Create a class with a class attribute set to None
    @dataclass(frozen=True)
    class SliceWithNoneAttr:
        value: int = 0

    # Add a class attribute that is None
    SliceWithNoneAttr.none_attr = None  # type: ignore[attr-defined]

    # Should not raise, should just skip the None attribute
    reducers = _extract_reducer_metadata(SliceWithNoneAttr)
    assert reducers == ()
