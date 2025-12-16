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

"""Tests for declarative @reducer decorator and session.install()."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pytest

from weakincentives.runtime.session import reducer

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


@dataclass(frozen=True)
class Counter:
    """State slice for testing with initial factory."""

    count: int = 0

    @reducer(on=Increment)
    def increment(self, event: Increment) -> Counter:
        return replace(self, count=self.count + event.amount)

    @reducer(on=Reset)
    def reset(self, event: Reset) -> Counter:
        del event
        return Counter()


# ──────────────────────────────────────────────────────────────────────
# session.install() tests
# ──────────────────────────────────────────────────────────────────────


def test_install_registers_reducers(session_factory: SessionFactory) -> None:
    """Verify install() registers all reducer methods."""
    session, _ = session_factory()
    session.install(ItemList)

    # Seed initial state
    session[ItemList].seed(ItemList(items=()))

    # Apply events - should work via installed reducers
    session[ItemList].apply(AddItem(item="first"))
    session[ItemList].apply(AddItem(item="second"))

    result = session[ItemList].latest()
    assert result is not None
    assert result.items == ("first", "second")


def test_install_rejects_non_dataclass(session_factory: SessionFactory) -> None:
    """Verify install() rejects classes that are not dataclasses."""
    session, _ = session_factory()

    class NotADataclass:
        pass

    with pytest.raises(TypeError, match="must be a dataclass"):
        session.install(NotADataclass)


def test_install_rejects_class_without_reducers(
    session_factory: SessionFactory,
) -> None:
    """Verify install() rejects classes without @reducer methods."""
    session, _ = session_factory()

    with pytest.raises(ValueError, match="no @reducer decorated methods"):
        session.install(AddItem)


def test_install_rejects_non_frozen_dataclass(
    session_factory: SessionFactory,
) -> None:
    """Verify install() rejects non-frozen dataclasses."""
    session, _ = session_factory()

    @dataclass  # Not frozen
    class MutableSlice:
        value: int

        @reducer(on=AddItem)
        def add(self, event: AddItem) -> MutableSlice:
            return MutableSlice(value=len(event.item))

    with pytest.raises(TypeError, match="must be a frozen dataclass"):
        session.install(MutableSlice)


def test_install_with_initial_factory(session_factory: SessionFactory) -> None:
    """Verify install() with initial factory handles empty slice."""
    session, _ = session_factory()
    session.install(Counter, initial=Counter)

    # Apply without seeding - should use initial factory
    session[Counter].apply(Increment(amount=5))

    result = session[Counter].latest()
    assert result is not None
    assert result.count == 5


def test_install_without_initial_factory_ignores_empty(
    session_factory: SessionFactory,
) -> None:
    """Verify install() without initial factory ignores events on empty slice."""
    session, _ = session_factory()
    session.install(ItemList)

    # Apply without seeding - should be ignored (no initial factory)
    session[ItemList].apply(AddItem(item="ignored"))

    result = session[ItemList].latest()
    assert result is None


# ──────────────────────────────────────────────────────────────────────
# session[SliceType] indexing tests
# ──────────────────────────────────────────────────────────────────────


def test_getitem_returns_slice_accessor(session_factory: SessionFactory) -> None:
    """Verify session[SliceType] returns SliceAccessor."""
    from weakincentives.runtime.session import SliceAccessor

    session, _ = session_factory()

    accessor = session[ItemList]

    assert isinstance(accessor, SliceAccessor)


def test_getitem_query_latest(session_factory: SessionFactory) -> None:
    """Verify session[SliceType].latest() works."""
    session, _ = session_factory()
    session.install(ItemList)

    session[ItemList].seed(ItemList(items=("a", "b")))

    result = session[ItemList].latest()

    assert result == ItemList(items=("a", "b"))


def test_getitem_query_all(session_factory: SessionFactory) -> None:
    """Verify session[SliceType].all() works."""
    session, _ = session_factory()
    session[ItemList].seed([ItemList(items=("a",)), ItemList(items=("b",))])

    result = session[ItemList].all()

    assert result == (ItemList(items=("a",)), ItemList(items=("b",)))


def test_getitem_query_where(session_factory: SessionFactory) -> None:
    """Verify session[SliceType].where() works."""
    session, _ = session_factory()
    session[ItemList].seed(
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

    session[ItemList].seed(ItemList(items=("a", "b", "c")))
    session[ItemList].apply(RemoveItem(item="b"))

    result = session[ItemList].latest()
    assert result is not None
    assert result.items == ("a", "c")


def test_reducer_method_can_clear_state(session_factory: SessionFactory) -> None:
    """Verify reducer method can clear state."""
    session, _ = session_factory()
    session.install(ItemList)

    session[ItemList].seed(ItemList(items=("a", "b")))
    session[ItemList].apply(ClearItems())

    result = session[ItemList].latest()
    assert result is not None
    assert result.items == ()


def test_reducer_method_receives_event(session_factory: SessionFactory) -> None:
    """Verify reducer method receives event data."""
    session, _ = session_factory()
    session.install(Counter, initial=Counter)

    session[Counter].apply(Increment(amount=10))
    session[Counter].apply(Increment(amount=5))

    result = session[Counter].latest()
    assert result is not None
    assert result.count == 15


def test_reducer_method_reset(session_factory: SessionFactory) -> None:
    """Verify reducer method can reset state."""
    session, _ = session_factory()
    session.install(Counter, initial=Counter)

    session[Counter].apply(Increment(amount=100))
    session[Counter].apply(Reset())

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

    @dataclass(frozen=True)
    class SliceA:
        value: int = 0

        @reducer(on=SharedEvent)
        def handle(self, event: SharedEvent) -> SliceA:
            return replace(self, value=event.value * 2)

    @dataclass(frozen=True)
    class SliceB:
        value: int = 0

        @reducer(on=SharedEvent)
        def handle(self, event: SharedEvent) -> SliceB:
            return replace(self, value=event.value * 3)

    session, _ = session_factory()
    session.install(SliceA, initial=SliceA)
    session.install(SliceB, initial=SliceB)

    # Broadcast applies to all reducers for SharedEvent across all slices
    session.apply(SharedEvent(value=10))

    a_result = session[SliceA].latest()
    b_result = session[SliceB].latest()

    assert a_result is not None
    assert a_result.value == 20

    # Both slices have reducers for SharedEvent, so both are updated
    assert b_result is not None
    assert b_result.value == 30


def test_chained_applies(session_factory: SessionFactory) -> None:
    """Test multiple applies in sequence."""
    session, _ = session_factory()
    session.install(Counter, initial=Counter)

    for i in range(1, 6):
        session[Counter].apply(Increment(amount=i))

    result = session[Counter].latest()
    assert result is not None
    assert result.count == 15  # 1+2+3+4+5


def test_reducer_qualname_set_correctly() -> None:
    """Verify generated reducers have meaningful qualnames."""
    from weakincentives.runtime.session.state_slice import _create_reducer_for_method

    generated_reducer = _create_reducer_for_method(ItemList, "add_item")

    # Access __qualname__ via getattr since it's set dynamically
    qualname = getattr(generated_reducer, "__qualname__", "")
    assert "ItemList" in qualname
    assert "add_item" in qualname


def test_install_state_slice_rejects_class_without_reducers() -> None:
    """Verify install_state_slice raises ValueError for classes without @reducer."""
    from weakincentives.runtime.session import Session
    from weakincentives.runtime.session.state_slice import install_state_slice

    @dataclass(frozen=True)
    class NoReducers:
        value: int

    session = Session()

    with pytest.raises(ValueError, match="no @reducer decorated methods"):
        install_state_slice(session, NoReducers)


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


def test_install_rejects_duplicate_event_handlers(
    session_factory: SessionFactory,
) -> None:
    """Verify install() rejects classes with multiple handlers for same event."""

    @dataclass(frozen=True)
    class DuplicateEvent:
        value: int

    @dataclass(frozen=True)
    class DuplicateHandlers:
        value: int = 0

        @reducer(on=DuplicateEvent)
        def handle_first(self, event: DuplicateEvent) -> DuplicateHandlers:
            return replace(self, value=event.value)

        @reducer(on=DuplicateEvent)
        def handle_second(self, event: DuplicateEvent) -> DuplicateHandlers:
            return replace(self, value=event.value * 2)

    session, _ = session_factory()

    with pytest.raises(ValueError, match="multiple @reducer methods"):
        session.install(DuplicateHandlers)


def test_reducer_validates_return_type(
    session_factory: SessionFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify reducer wrapper validates return type at runtime."""
    import logging

    @dataclass(frozen=True)
    class BadEvent:
        pass

    @dataclass(frozen=True)
    class WrongReturn:
        value: int = 0

        @reducer(on=BadEvent)
        def handle(self, event: BadEvent) -> str:
            del event
            return "wrong type"  # Returns str, not WrongReturn

    session, _ = session_factory()
    session.install(WrongReturn, initial=WrongReturn)

    # Session logs reducer failures instead of raising
    with caplog.at_level(logging.ERROR):
        session[WrongReturn].apply(BadEvent())

    # Verify the TypeError was logged
    assert "must return WrongReturn, got str" in caplog.text
    assert "Reducer application failed" in caplog.text
