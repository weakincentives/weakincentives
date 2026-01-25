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

"""Built-in reducers for Session state slices.

This module provides common reducer functions and reducer factories for managing
session state. Reducers are pure functions that receive a SliceView of the current
state and an event, then return a SliceOp describing how to mutate the slice.

The reducers in this module cover common patterns:

- **append_all**: Ledger-style recording where every event is kept
- **replace_latest**: Keep only the most recent value (singleton state)
- **upsert_by**: Update or insert items based on a derived key
- **replace_latest_by**: Keep only the most recent item per key (keyed singletons)

Example usage with Session.register::

    from weakincentives.runtime.session import Session
    from weakincentives.runtime.session.reducers import append_all, upsert_by

    session = Session()

    # Register a slice that keeps all events (ledger)
    session.register(EventLog, reducer=append_all)

    # Register a slice that upserts by id field
    session.register(User, reducer=upsert_by(lambda u: u.id))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from ...dbc import pure
from ...types.dataclass import SupportsDataclass
from ._types import (
    ReducerContextProtocol,
    ReducerEvent,
    TypedReducer,
)
from .slices import Append, Replace, SliceView


@pure
def append_all[T: SupportsDataclass](
    view: SliceView[T],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> Append[T]:
    """Append every dispatched event to the slice (ledger semantics).

    This reducer records every event without deduplication, making it ideal
    for event streams, audit logs, and ledgers where history matters.

    Performance: O(1) for all backends. Never reads existing state, so
    file-backed slices only append without loading the file.

    Args:
        view: Lazy view of current slice state (not accessed by this reducer).
        event: The event being dispatched, cast to the slice's type T.
        context: Reducer context providing access to the session.

    Returns:
        An Append operation containing the event.

    Example::

        @dataclass(frozen=True, slots=True)
        class AuditEntry:
            action: str
            timestamp: datetime

        session.register(AuditEntry, reducer=append_all)
        session.dispatch(AuditEntry("login", now()))
        session.dispatch(AuditEntry("logout", now()))
        # Slice now contains both entries in order
    """
    del context, view
    return Append(cast(T, event))


@pure
def replace_latest[T: SupportsDataclass](
    view: SliceView[T],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> Replace[T]:
    """Keep only the most recent value, discarding all previous state.

    This reducer maintains a singleton slice containing only the latest
    dispatched event. Use it for current state that supersedes all previous
    values, such as configuration, status, or mode settings.

    Performance: O(1) for memory slices; O(n) write for file-backed slices
    since Replace rewrites the entire file. Never reads existing state.

    Args:
        view: Lazy view of current slice state (not accessed by this reducer).
        event: The event being dispatched, cast to the slice's type T.
        context: Reducer context providing access to the session.

    Returns:
        A Replace operation with a single-element tuple containing the event.

    Example::

        @dataclass(frozen=True, slots=True)
        class AppConfig:
            theme: str
            language: str

        session.register(AppConfig, reducer=replace_latest)
        session.dispatch(AppConfig("dark", "en"))
        session.dispatch(AppConfig("light", "fr"))
        # session[AppConfig].latest() returns AppConfig("light", "fr")
    """
    del context, view
    return Replace((cast(T, event),))


def upsert_by[T: SupportsDataclass, K](
    key_fn: Callable[[T], K],
) -> TypedReducer[T]:
    """Create a reducer that updates existing items or inserts new ones by key.

    This factory returns a reducer that maintains a collection of items keyed
    by a derived value. When an event is dispatched, items with matching keys
    are replaced; otherwise the event is appended. Order is preserved for
    non-matching items, with the new/updated item added at the end.

    Use this for entity collections where each entity has a unique identifier
    (e.g., users by ID, products by SKU, sessions by token).

    Performance: O(n) for all backends. Must iterate the entire slice to find
    and filter out items with matching keys.

    Args:
        key_fn: Function that extracts the unique key from an item. The key
            must be hashable and comparable with ``!=``.

    Returns:
        A TypedReducer that performs upsert operations based on the key.

    Example::

        @dataclass(frozen=True, slots=True)
        class User:
            id: str
            name: str
            email: str

        session.register(User, reducer=upsert_by(lambda u: u.id))

        session.dispatch(User("u1", "Alice", "alice@example.com"))
        session.dispatch(User("u2", "Bob", "bob@example.com"))
        # Slice contains both users

        session.dispatch(User("u1", "Alice Smith", "alice.smith@example.com"))
        # User u1 is updated, u2 unchanged; slice still has 2 items
    """

    @pure
    def reducer(
        view: SliceView[T],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Replace[T]:
        del context
        value = cast(T, event)
        event_key = key_fn(value)
        items = [item for item in view if key_fn(item) != event_key]
        items.append(value)
        return Replace(tuple(items))

    return reducer


def replace_latest_by[T: SupportsDataclass, K](
    key_fn: Callable[[T], K],
) -> TypedReducer[T]:
    """Create a reducer that keeps only the most recent item for each key.

    This factory returns a reducer that maintains at most one item per unique
    key. When an event is dispatched, any existing item with the same key is
    removed and the new event is appended. This is equivalent to ``upsert_by``
    but emphasizes the "latest wins" semantics.

    Use this for per-key singleton state, such as the most recent status per
    component, latest metric per sensor, or current settings per user.

    Performance: O(n) for all backends. Must iterate the entire slice to
    filter out items with matching keys.

    Args:
        key_fn: Function that extracts the unique key from an item. The key
            must be hashable and comparable with ``!=``.

    Returns:
        A TypedReducer that replaces items by key, keeping only the latest.

    Note:
        This function is functionally equivalent to ``upsert_by`` with the
        same key function. The distinction is semantic: use ``upsert_by``
        when thinking about entity updates, and ``replace_latest_by`` when
        thinking about "latest value per category" patterns.

    Example::

        @dataclass(frozen=True, slots=True)
        class SensorReading:
            sensor_id: str
            value: float
            timestamp: datetime

        session.register(
            SensorReading,
            reducer=replace_latest_by(lambda r: r.sensor_id)
        )

        session.dispatch(SensorReading("temp_1", 22.5, now()))
        session.dispatch(SensorReading("temp_2", 19.0, now()))
        session.dispatch(SensorReading("temp_1", 23.1, now()))
        # Slice contains 2 items: latest reading for temp_1 and temp_2
    """

    @pure
    def reducer(
        view: SliceView[T],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Replace[T]:
        del context
        value = cast(T, event)
        key = key_fn(value)
        filtered = tuple(item for item in view if key_fn(item) != key)
        return Replace((*filtered, value))

    return reducer


__all__ = [
    "append_all",
    "replace_latest",
    "replace_latest_by",
    "upsert_by",
]
