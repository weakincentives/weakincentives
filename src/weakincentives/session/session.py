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

"""Session state container synchronized with the event bus."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, is_dataclass
from datetime import UTC, datetime
from typing import Any, cast

from ..events import EventBus, NullEventBus, PromptExecuted, ToolInvoked
from ..prompt._types import SupportsDataclass
from .snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    normalize_snapshot_state,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ToolData:
    """Wrapper containing tool payloads and their originating event."""

    value: SupportsDataclass | None
    source: ToolInvoked


_TOOL_DATA_TYPE: type[SupportsDataclass] = cast(type[SupportsDataclass], ToolData)


def _append_tool_data(
    slice_values: tuple[ToolData, ...], event: DataEvent
) -> tuple[ToolData, ...]:
    if not isinstance(event, ToolData):
        return slice_values
    return (*slice_values, event)


@dataclass(slots=True, frozen=True)
class PromptData[T: SupportsDataclass]:
    """Wrapper containing prompt outputs and their originating event."""

    value: T
    source: PromptExecuted


type DataEvent = ToolData | PromptData[SupportsDataclass]

type TypedReducer[S] = Callable[[tuple[S, ...], DataEvent], tuple[S, ...]]


@dataclass(slots=True)
class _ReducerRegistration:
    reducer: TypedReducer[Any]
    slice_type: type[Any]


class Session:
    """Collect dataclass payloads from prompt executions and tool invocations."""

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        session_id: str | None = None,
        created_at: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.created_at = created_at
        self._bus = bus or NullEventBus()
        self._reducers: dict[type[SupportsDataclass], list[_ReducerRegistration]] = {}
        self._state: dict[type[Any], tuple[Any, ...]] = {}

        if bus is not None:
            bus.subscribe(ToolInvoked, self._on_tool_invoked)
            bus.subscribe(PromptExecuted, self._on_prompt_executed)

    def register_reducer[S](
        self,
        data_type: type[SupportsDataclass],
        reducer: TypedReducer[S],
        *,
        slice_type: type[S] | None = None,
    ) -> None:
        """Register a reducer for the provided data type."""

        target_slice_type: type[Any] = data_type if slice_type is None else slice_type
        registration = _ReducerRegistration(
            reducer=cast(TypedReducer[Any], reducer),
            slice_type=target_slice_type,
        )
        bucket = self._reducers.setdefault(data_type, [])
        bucket.append(registration)
        self._state.setdefault(target_slice_type, ())

    def select_all[S](self, slice_type: type[S]) -> tuple[S, ...]:
        """Return the tuple slice maintained for the provided type."""

        return cast(tuple[S, ...], self._state.get(slice_type, ()))

    def seed_slice[S](self, slice_type: type[S], values: Iterable[S]) -> None:
        """Initialize or replace the stored tuple for the provided type."""

        self._state[slice_type] = tuple(values)

    def snapshot(self) -> Snapshot:
        """Capture an immutable snapshot of the current session state."""

        try:
            normalized: Mapping[type[Any], tuple[Any, ...]] = normalize_snapshot_state(
                cast(Mapping[object, tuple[object, ...]], self._state)
            )
        except ValueError as error:
            msg = "Unable to serialize session slices"
            raise SnapshotSerializationError(msg) from error

        created_at = datetime.now(UTC)
        return Snapshot(created_at=created_at, slices=normalized)

    def rollback(self, snapshot: Snapshot) -> None:
        """Restore session slices from the provided snapshot."""

        registered_slices = self._registered_slice_types()
        missing = [
            slice_type
            for slice_type in snapshot.slices
            if slice_type not in registered_slices
        ]
        if missing:
            missing_names = ", ".join(sorted(cls.__qualname__ for cls in missing))
            msg = f"Slice types not registered: {missing_names}"
            raise SnapshotRestoreError(msg)

        new_state: dict[type[Any], tuple[Any, ...]] = dict(self._state)
        for slice_type in registered_slices:
            new_state[slice_type] = snapshot.slices.get(slice_type, ())

        self._state = new_state

    def _registered_slice_types(self) -> set[type[Any]]:
        types: set[type[Any]] = set(self._state)
        for registrations in self._reducers.values():
            for registration in registrations:
                types.add(registration.slice_type)
        return types

    def _on_tool_invoked(self, event: object) -> None:
        if isinstance(event, ToolInvoked):
            self._handle_tool_invoked(event)

    def _on_prompt_executed(self, event: object) -> None:
        if isinstance(event, PromptExecuted):
            self._handle_prompt_executed(event)

    def _handle_tool_invoked(self, event: ToolInvoked) -> None:
        payload = event.result.value
        dataclass_payload: SupportsDataclass | None = None
        if _is_dataclass_instance(payload):
            dataclass_payload = cast(SupportsDataclass, payload)

        data = ToolData(value=dataclass_payload, source=event)
        self._dispatch_data_event(_TOOL_DATA_TYPE, data)

        if dataclass_payload is not None:
            self._dispatch_data_event(type(dataclass_payload), data)

    def _handle_prompt_executed(self, event: PromptExecuted) -> None:
        output = event.result.output
        if _is_dataclass_instance(output):
            dataclass_output = cast(SupportsDataclass, output)
            data = PromptData(value=dataclass_output, source=event)
            self._dispatch_data_event(type(dataclass_output), data)
            return
        if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
            for item in cast(Iterable[object], output):
                if _is_dataclass_instance(item):
                    dataclass_item = cast(SupportsDataclass, item)
                    data = PromptData(value=dataclass_item, source=event)
                    self._dispatch_data_event(type(dataclass_item), data)

    def _dispatch_data_event(
        self, data_type: type[SupportsDataclass], event: DataEvent
    ) -> None:
        registrations = self._reducers.get(data_type)
        if not registrations:
            if data_type is _TOOL_DATA_TYPE:
                registrations = [
                    _ReducerRegistration(
                        reducer=cast(TypedReducer[Any], _append_tool_data),
                        slice_type=ToolData,
                    )
                ]
            else:
                from .reducers import append

                registrations = [
                    _ReducerRegistration(
                        reducer=cast(TypedReducer[Any], append),
                        slice_type=data_type,
                    )
                ]

        for registration in registrations:
            slice_type = registration.slice_type
            previous = self._state.get(slice_type, ())
            try:
                result = registration.reducer(previous, event)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Reducer %r failed for data type %s",
                    registration.reducer,
                    data_type,
                )
                continue
            normalized = tuple(result)
            self._state[slice_type] = normalized


def _is_dataclass_instance(value: object) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


__all__ = [
    "Session",
    "ToolData",
    "PromptData",
    "DataEvent",
    "TypedReducer",
]
