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

"""Property-based tests for Session state management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from hypothesis import given, settings, strategies as st

from weakincentives.runtime.events import Dispatcher
from weakincentives.runtime.session import Session
from weakincentives.runtime.session._types import ReducerContextProtocol
from weakincentives.runtime.session.reducers import (
    append_all,
    replace_latest,
    replace_latest_by,
    upsert_by,
)
from weakincentives.runtime.session.slice_policy import SlicePolicy


# ============================================================================
# Test Fixtures: Dataclass Definitions
# ============================================================================


@dataclass(frozen=True, slots=True)
class Counter:
    """Simple counter for slice testing."""

    value: int


@dataclass(frozen=True, slots=True)
class KeyedItem:
    """Item with a key for upsert testing."""

    key: str
    data: str


@dataclass(slots=True)
class MockContext(ReducerContextProtocol):
    """Minimal context implementation for reducer tests."""

    session: Session
    dispatcher: Dispatcher


def make_context() -> MockContext:
    session = Session()
    return MockContext(session=session, dispatcher=session.dispatcher)


# ============================================================================
# Hypothesis Strategies
# ============================================================================

_counter_values = st.integers(min_value=-1000, max_value=1000)

_key_text = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=10,
)

_data_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=0,
    max_size=20,
)


def counter_strategy() -> st.SearchStrategy[Counter]:
    return st.builds(Counter, value=_counter_values)


def keyed_item_strategy() -> st.SearchStrategy[KeyedItem]:
    return st.builds(KeyedItem, key=_key_text, data=_data_text)


def counter_list_strategy(
    min_size: int = 0, max_size: int = 10
) -> st.SearchStrategy[list[Counter]]:
    return st.lists(counter_strategy(), min_size=min_size, max_size=max_size)


def keyed_item_list_strategy(
    min_size: int = 0, max_size: int = 10
) -> st.SearchStrategy[list[KeyedItem]]:
    return st.lists(keyed_item_strategy(), min_size=min_size, max_size=max_size)


# ============================================================================
# Property Tests: Reducer append_all
# ============================================================================


@given(counter_list_strategy(), counter_strategy())
@settings(max_examples=100)
def test_append_all_increases_length_by_one(
    initial: list[Counter], new_item: Counter
) -> None:
    """append_all always adds exactly one item."""
    context = make_context()
    result = append_all(tuple(initial), new_item, context=context)
    assert len(result) == len(initial) + 1


@given(counter_list_strategy(), counter_strategy())
@settings(max_examples=100)
def test_append_all_preserves_existing_items(
    initial: list[Counter], new_item: Counter
) -> None:
    """append_all preserves all existing items in order."""
    context = make_context()
    result = append_all(tuple(initial), new_item, context=context)
    assert result[:-1] == tuple(initial)


@given(counter_list_strategy(), counter_strategy())
@settings(max_examples=100)
def test_append_all_new_item_at_end(
    initial: list[Counter], new_item: Counter
) -> None:
    """append_all places new item at the end."""
    context = make_context()
    result = append_all(tuple(initial), new_item, context=context)
    assert result[-1] == new_item


@given(counter_strategy())
@settings(max_examples=50)
def test_append_all_idempotent_on_empty(new_item: Counter) -> None:
    """append_all on empty tuple produces single-item tuple."""
    context = make_context()
    result = append_all((), new_item, context=context)
    assert result == (new_item,)


# ============================================================================
# Property Tests: Reducer replace_latest
# ============================================================================


@given(counter_list_strategy(), counter_strategy())
@settings(max_examples=100)
def test_replace_latest_produces_single_item(
    initial: list[Counter], new_item: Counter
) -> None:
    """replace_latest always produces a single-item tuple."""
    context = make_context()
    result = replace_latest(tuple(initial), new_item, context=context)
    assert len(result) == 1
    assert result[0] == new_item


@given(counter_strategy())
@settings(max_examples=50)
def test_replace_latest_on_empty(new_item: Counter) -> None:
    """replace_latest on empty tuple produces single-item tuple."""
    context = make_context()
    result = replace_latest((), new_item, context=context)
    assert result == (new_item,)


# ============================================================================
# Property Tests: Reducer upsert_by
# ============================================================================


@given(keyed_item_list_strategy(), keyed_item_strategy())
@settings(max_examples=100)
def test_upsert_by_contains_new_key(
    initial: list[KeyedItem], new_item: KeyedItem
) -> None:
    """upsert_by always includes the new item's key in result."""
    reducer = upsert_by(lambda item: item.key)
    context = make_context()
    result = reducer(tuple(initial), new_item, context=context)
    keys_in_result = {item.key for item in result}
    assert new_item.key in keys_in_result


@given(keyed_item_list_strategy(), keyed_item_strategy())
@settings(max_examples=100)
def test_upsert_by_deduplicates_new_key(
    initial: list[KeyedItem], new_item: KeyedItem
) -> None:
    """upsert_by ensures the new item's key appears exactly once."""
    reducer = upsert_by(lambda item: item.key)
    context = make_context()
    result = reducer(tuple(initial), new_item, context=context)
    # The new item's key should appear exactly once
    count = sum(1 for item in result if item.key == new_item.key)
    assert count == 1


@given(keyed_item_list_strategy(min_size=1), st.integers(min_value=0, max_value=9))
@settings(max_examples=100)
def test_upsert_by_update_existing(
    initial: list[KeyedItem], index_mod: int
) -> None:
    """upsert_by updates existing item when key matches."""
    # Pick an existing key to update
    existing = initial[index_mod % len(initial)]
    new_item = KeyedItem(key=existing.key, data="updated")

    reducer = upsert_by(lambda item: item.key)
    context = make_context()
    result = reducer(tuple(initial), new_item, context=context)

    # Find the item with the matching key
    matching = [item for item in result if item.key == existing.key]
    assert len(matching) == 1
    assert matching[0].data == "updated"


@given(keyed_item_list_strategy())
@settings(max_examples=50)
def test_upsert_by_append_new_key(initial: list[KeyedItem]) -> None:
    """upsert_by appends when key doesn't exist."""
    new_item = KeyedItem(key="__unique_test_key__", data="new")
    reducer = upsert_by(lambda item: item.key)
    context = make_context()
    result = reducer(tuple(initial), new_item, context=context)

    # New item should be at the end
    assert result[-1] == new_item
    assert len(result) == len(initial) + 1


# ============================================================================
# Property Tests: Reducer replace_latest_by
# ============================================================================


@given(keyed_item_list_strategy(), keyed_item_strategy())
@settings(max_examples=100)
def test_replace_latest_by_contains_new_key(
    initial: list[KeyedItem], new_item: KeyedItem
) -> None:
    """replace_latest_by always includes the new item's key in result."""
    reducer = replace_latest_by(lambda item: item.key)
    context = make_context()
    result = reducer(tuple(initial), new_item, context=context)
    keys_in_result = {item.key for item in result}
    assert new_item.key in keys_in_result


@given(keyed_item_list_strategy(), keyed_item_strategy())
@settings(max_examples=100)
def test_replace_latest_by_deduplicates_new_key(
    initial: list[KeyedItem], new_item: KeyedItem
) -> None:
    """replace_latest_by ensures the new item's key appears exactly once."""
    reducer = replace_latest_by(lambda item: item.key)
    context = make_context()
    result = reducer(tuple(initial), new_item, context=context)
    # The new item's key should appear exactly once (the new item itself)
    count = sum(1 for item in result if item.key == new_item.key)
    assert count == 1


@given(keyed_item_list_strategy(), keyed_item_strategy())
@settings(max_examples=100)
def test_replace_latest_by_new_item_at_end(
    initial: list[KeyedItem], new_item: KeyedItem
) -> None:
    """replace_latest_by places new item at the end."""
    reducer = replace_latest_by(lambda item: item.key)
    context = make_context()
    result = reducer(tuple(initial), new_item, context=context)
    assert result[-1] == new_item


# ============================================================================
# Property Tests: Session Snapshot/Restore
# ============================================================================


@given(counter_list_strategy(min_size=1, max_size=5))
@settings(max_examples=50)
def test_snapshot_restore_preserves_state(counters: list[Counter]) -> None:
    """Snapshot followed by restore preserves session state."""
    session = Session()

    # Seed the slice with counters
    session[Counter].seed(counters)

    # Take snapshot
    snapshot = session.snapshot()

    # Modify state
    session[Counter].clear()
    assert session[Counter].all() == ()

    # Restore and verify
    session.restore(snapshot)
    restored = session[Counter].all()
    assert restored == tuple(counters)


@given(counter_list_strategy(min_size=1, max_size=5))
@settings(max_examples=50)
def test_double_snapshot_restore(counters: list[Counter]) -> None:
    """Double snapshot/restore cycle preserves state."""
    session = Session()
    session[Counter].seed(counters)

    # First snapshot/restore
    snapshot1 = session.snapshot()
    session[Counter].clear()
    session.restore(snapshot1)

    # Second snapshot/restore
    snapshot2 = session.snapshot()
    session[Counter].clear()
    session.restore(snapshot2)

    assert session[Counter].all() == tuple(counters)


@given(counter_list_strategy(max_size=5), counter_list_strategy(max_size=5))
@settings(max_examples=50)
def test_restore_replaces_state(
    initial: list[Counter], replacement: list[Counter]
) -> None:
    """Restore replaces current state with snapshot state."""
    session = Session()
    session[Counter].seed(initial)

    # Take snapshot
    snapshot = session.snapshot()

    # Replace with different data
    session[Counter].seed(replacement)

    # Restore
    session.restore(snapshot)

    # Should have original data
    assert session[Counter].all() == tuple(initial)


# ============================================================================
# Property Tests: Session Slice Operations
# ============================================================================


@given(counter_list_strategy())
@settings(max_examples=100)
def test_seed_then_all_returns_same(counters: list[Counter]) -> None:
    """Seeding a slice and reading it back returns the same values."""
    session = Session()
    session[Counter].seed(counters)
    assert session[Counter].all() == tuple(counters)


@given(counter_list_strategy(min_size=1), st.integers(min_value=0, max_value=10))
@settings(max_examples=100)
def test_latest_returns_last_item(counters: list[Counter], idx: int) -> None:
    """latest() returns the last item in the slice."""
    session = Session()
    session[Counter].seed(counters)
    assert session[Counter].latest() == counters[-1]


@given(counter_list_strategy())
@settings(max_examples=50)
def test_clear_empties_slice(counters: list[Counter]) -> None:
    """clear() empties the slice."""
    session = Session()
    session[Counter].seed(counters)
    session[Counter].clear()
    assert session[Counter].all() == ()


@given(counter_list_strategy(min_size=1))
@settings(max_examples=50)
def test_where_filters_correctly(counters: list[Counter]) -> None:
    """where() filters items by predicate."""
    session = Session()
    session[Counter].seed(counters)

    # Find the median value to use as threshold
    sorted_values = sorted(c.value for c in counters)
    threshold = sorted_values[len(sorted_values) // 2]

    # Filter using where
    filtered = session[Counter].where(lambda c: c.value >= threshold)

    # Verify all items match predicate
    for c in filtered:
        assert c.value >= threshold

    # Verify we got the right count
    expected_count = sum(1 for c in counters if c.value >= threshold)
    assert len(filtered) == expected_count


# ============================================================================
# Property Tests: Session Reducer Registration
# ============================================================================


@given(counter_list_strategy(max_size=5))
@settings(max_examples=50)
def test_dispatch_with_custom_reducer(counters: list[Counter]) -> None:
    """Custom reducers are invoked on dispatch."""
    session = Session()

    # Register a custom reducer that doubles the value
    def double_reducer(
        slice_values: tuple[Counter, ...],
        event: Counter,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[Counter, ...]:
        doubled = Counter(value=event.value * 2)
        return (*slice_values, doubled)

    session[Counter].register(Counter, double_reducer)

    # Dispatch counters
    for counter in counters:
        session.dispatch(counter)

    # Verify all values were doubled
    result = session[Counter].all()
    assert len(result) == len(counters)
    for i, counter in enumerate(counters):
        assert result[i].value == counter.value * 2


# ============================================================================
# Property Tests: Session Clone
# ============================================================================


@given(counter_list_strategy(min_size=1, max_size=5))
@settings(max_examples=50)
def test_clone_preserves_state(counters: list[Counter]) -> None:
    """Cloned session has same state as original."""
    session = Session()
    session[Counter].seed(counters)

    cloned = session.clone(bus=session.dispatcher)

    assert cloned[Counter].all() == session[Counter].all()


@given(counter_list_strategy(min_size=1), counter_strategy())
@settings(max_examples=50)
def test_clone_is_independent(initial: list[Counter], new_item: Counter) -> None:
    """Modifications to clone don't affect original."""
    session = Session()
    session[Counter].seed(initial)

    cloned = session.clone(bus=session.dispatcher)
    cloned[Counter].seed([new_item])

    # Original should be unchanged
    assert session[Counter].all() == tuple(initial)
    # Clone should have new data
    assert cloned[Counter].all() == (new_item,)


# ============================================================================
# Property Tests: Session Reset
# ============================================================================


@given(counter_list_strategy(min_size=1))
@settings(max_examples=50)
def test_reset_clears_all_slices(counters: list[Counter]) -> None:
    """reset() clears all session state."""
    session = Session()
    session[Counter].seed(counters)

    session.reset()

    assert session[Counter].all() == ()


# ============================================================================
# Property Tests: Multiple Slice Types
# ============================================================================


@given(counter_list_strategy(max_size=5), keyed_item_list_strategy(max_size=5))
@settings(max_examples=50)
def test_multiple_slice_types_independent(
    counters: list[Counter], items: list[KeyedItem]
) -> None:
    """Different slice types are independent."""
    session = Session()

    session[Counter].seed(counters)
    session[KeyedItem].seed(items)

    assert session[Counter].all() == tuple(counters)
    assert session[KeyedItem].all() == tuple(items)

    # Clear one slice doesn't affect the other
    session[Counter].clear()
    assert session[Counter].all() == ()
    assert session[KeyedItem].all() == tuple(items)
