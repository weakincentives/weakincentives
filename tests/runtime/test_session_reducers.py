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
from typing import cast

from weakincentives.runtime.events import EventBus
from weakincentives.runtime.session import Session
from weakincentives.runtime.session._types import (
    ReducerContextProtocol,
    ReducerEvent,
    SimpleReducer,
)
from weakincentives.runtime.session.reducers import as_typed_reducer, replace_latest_by


@dataclass(slots=True)
class _Sample:
    key: str
    value: str


@dataclass(slots=True)
class _ReducerEvent:
    value: _Sample


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
        _ReducerEvent(value=_Sample("a", "updated")),
        context=context,
    )

    assert len(updated) == 2
    assert updated[-1].key == "a"
    assert updated[-1].value == "updated"
    assert any(item.value == "second" for item in updated)


def test_as_typed_reducer_wraps_simple_reducer() -> None:
    """Verify as_typed_reducer converts a SimpleReducer to TypedReducer."""

    def simple_reducer(
        slice_values: tuple[_Sample, ...],
        event: ReducerEvent,
    ) -> tuple[_Sample, ...]:
        value = cast(_Sample, event)
        return (*slice_values, value)

    typed_reducer = as_typed_reducer(simple_reducer)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)
    initial = (_Sample("a", "first"),)
    new_item = _Sample("b", "second")

    result = typed_reducer(initial, new_item, context=context)

    assert len(result) == 2
    assert result[0] == initial[0]
    assert result[1] == new_item


def test_as_typed_reducer_ignores_context() -> None:
    """Verify the context is not passed to the underlying simple reducer."""
    context_was_used = False

    def simple_reducer(
        slice_values: tuple[_Sample, ...],
        event: ReducerEvent,
    ) -> tuple[_Sample, ...]:
        # This should never see the context
        return slice_values

    typed_reducer = as_typed_reducer(simple_reducer)
    session = Session()
    context = _Context(session=session, event_bus=session.event_bus)

    # The simple reducer should be called without error
    result = typed_reducer((), _Sample("a", "first"), context=context)
    assert result == ()
    assert not context_was_used


def test_simple_reducer_type_annotation() -> None:
    """Verify SimpleReducer type alias works correctly."""

    def my_reducer(
        slice_values: tuple[_Sample, ...],
        event: ReducerEvent,
    ) -> tuple[_Sample, ...]:
        value = cast(_Sample, event)
        return (*slice_values, value)

    # Type checking: this should satisfy SimpleReducer[_Sample]
    reducer: SimpleReducer[_Sample] = my_reducer
    result = reducer((), _Sample("key", "value"))
    assert len(result) == 1
