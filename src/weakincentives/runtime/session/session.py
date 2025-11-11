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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import RLock
from typing import Any, cast, override

from ...prompt._types import SupportsDataclass
from ..events import (
    EventBus,
    PromptExecuted,
    PromptExecutionFailed,
    PromptRendered,
    ToolInvoked,
)
from ..logging import StructuredLogger, get_logger
from ._types import ReducerContextProtocol, ReducerEvent, TypedReducer
from .dataclasses import is_dataclass_instance
from .protocols import SessionProtocol, SnapshotProtocol
from .reducers import append
from .snapshots import (
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    normalize_snapshot_state,
)

logger: StructuredLogger = get_logger(__name__, context={"component": "session"})


type DataEvent = PromptExecuted | PromptExecutionFailed | PromptRendered | ToolInvoked


_PROMPT_RENDERED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptRendered
)
_TOOL_INVOKED_TYPE: type[SupportsDataclass] = cast(type[SupportsDataclass], ToolInvoked)
_PROMPT_EXECUTED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptExecuted
)
_PROMPT_FAILED_TYPE: type[SupportsDataclass] = cast(
    type[SupportsDataclass], PromptExecutionFailed
)


def _append_event(
    slice_values: tuple[SupportsDataclass, ...],
    event: ReducerEvent,
    *,
    context: ReducerContextProtocol,
) -> tuple[SupportsDataclass, ...]:
    del context
    appended = cast(SupportsDataclass, event)
    return (*slice_values, appended)


@dataclass(slots=True)
class _ReducerRegistration:
    reducer: TypedReducer[Any]
    slice_type: type[Any]


class Session(SessionProtocol):
    """Collect dataclass payloads from prompt executions and tool invocations."""

    def __init__(
        self,
        *,
        bus: EventBus,
        session_id: str | None = None,
        created_at: str | None = None,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.created_at = created_at
        self._bus: EventBus = bus
        self._reducers: dict[type[SupportsDataclass], list[_ReducerRegistration]] = {}
        self._state: dict[type[Any], tuple[Any, ...]] = {}
        self._lock = RLock()
        self._subscriptions_attached = False
        self._attach_to_bus(bus)

    def clone(
        self,
        *,
        bus: EventBus,
        session_id: str | None = None,
        created_at: str | None = None,
    ) -> Session:
        """Return a new session that mirrors the current state and reducers."""

        with self._lock:
            reducer_snapshot = [
                (data_type, tuple(registrations))
                for data_type, registrations in self._reducers.items()
            ]
            state_snapshot = dict(self._state)

        clone = Session(
            bus=bus,
            session_id=session_id if session_id is not None else self.session_id,
            created_at=created_at if created_at is not None else self.created_at,
        )

        for data_type, registrations in reducer_snapshot:
            for registration in registrations:
                clone.register_reducer(
                    data_type,
                    registration.reducer,
                    slice_type=registration.slice_type,
                )

        with clone._lock:
            clone._state = state_snapshot

        return clone

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
        with self._lock:
            bucket = self._reducers.setdefault(data_type, [])
            bucket.append(registration)
            _ = self._state.setdefault(target_slice_type, ())

    def select_all[S](self, slice_type: type[S]) -> tuple[S, ...]:
        """Return the tuple slice maintained for the provided type."""

        with self._lock:
            return cast(tuple[S, ...], self._state.get(slice_type, ()))

    def seed_slice[S](self, slice_type: type[S], values: Iterable[S]) -> None:
        """Initialize or replace the stored tuple for the provided type."""

        with self._lock:
            self._state[slice_type] = tuple(values)

    @override
    def snapshot(self) -> SnapshotProtocol:
        """Capture an immutable snapshot of the current session state."""

        with self._lock:
            state_snapshot = dict(self._state)
        for ephemeral in (
            _TOOL_INVOKED_TYPE,
            _PROMPT_EXECUTED_TYPE,
            _PROMPT_RENDERED_TYPE,
        ):
            _ = state_snapshot.pop(ephemeral, None)
        try:
            normalized: Mapping[type[Any], tuple[Any, ...]] = normalize_snapshot_state(
                cast(Mapping[object, tuple[object, ...]], state_snapshot)
            )
        except ValueError as error:
            msg = "Unable to serialize session slices"
            raise SnapshotSerializationError(msg) from error

        created_at = datetime.now(UTC)
        return Snapshot(created_at=created_at, slices=normalized)

    @override
    def rollback(self, snapshot: SnapshotProtocol) -> None:
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

        with self._lock:
            new_state: dict[type[Any], tuple[Any, ...]] = dict(self._state)
            for slice_type in registered_slices:
                new_state[slice_type] = snapshot.slices.get(slice_type, ())

            self._state = new_state

    def _registered_slice_types(self) -> set[type[Any]]:
        with self._lock:
            types: set[type[Any]] = set(self._state)
            for registrations in self._reducers.values():
                for registration in registrations:
                    types.add(registration.slice_type)
            return types

    def _on_tool_invoked(self, event: object) -> None:
        tool_event = cast(ToolInvoked, event)
        self._handle_tool_invoked(tool_event)

    def _on_prompt_executed(self, event: object) -> None:
        prompt_event = cast(PromptExecuted, event)
        self._handle_prompt_executed(prompt_event)

    def _on_prompt_execution_failed(self, event: object) -> None:
        failed_event = cast(PromptExecutionFailed, event)
        self._handle_prompt_failed(failed_event)

    def _on_prompt_rendered(self, event: object) -> None:
        start_event = cast(PromptRendered, event)
        self._handle_prompt_rendered(start_event)

    def _handle_tool_invoked(self, event: ToolInvoked) -> None:
        normalized_event = event
        payload = event.value if event.value is not None else event.result.value
        if event.value is None and is_dataclass_instance(payload):
            normalized_event = replace(event, value=payload)

        self._dispatch_data_event(
            _TOOL_INVOKED_TYPE,
            cast(ReducerEvent, normalized_event),
        )

        if normalized_event.value is not None:
            self._dispatch_data_event(
                type(normalized_event.value),
                cast(ReducerEvent, normalized_event),
            )

    def _handle_prompt_executed(self, event: PromptExecuted) -> None:
        normalized_event = event
        output = event.result.output
        if event.value is None and is_dataclass_instance(output):
            normalized_event = replace(event, value=output)

        self._dispatch_data_event(
            _PROMPT_EXECUTED_TYPE,
            cast(ReducerEvent, normalized_event),
        )

        if normalized_event.value is not None:
            self._dispatch_data_event(
                type(normalized_event.value),
                cast(ReducerEvent, normalized_event),
            )
            return

        if isinstance(output, Iterable) and not isinstance(output, (str, bytes)):
            for item in cast(Iterable[object], output):
                if is_dataclass_instance(item):
                    enriched_event = replace(normalized_event, value=item)
                    self._dispatch_data_event(
                        type(item),
                        cast(ReducerEvent, enriched_event),
                    )

    def _handle_prompt_rendered(self, event: PromptRendered) -> None:
        self._dispatch_data_event(
            _PROMPT_RENDERED_TYPE,
            cast(ReducerEvent, event),
        )

    def _handle_prompt_failed(self, event: PromptExecutionFailed) -> None:
        self._dispatch_data_event(
            _PROMPT_FAILED_TYPE,
            cast(ReducerEvent, event),
        )

    def _dispatch_data_event(
        self, data_type: type[SupportsDataclass], event: ReducerEvent
    ) -> None:
        from .reducer_context import build_reducer_context

        with self._lock:
            registrations = list(self._reducers.get(data_type, ()))
            if not registrations:
                default_reducer: TypedReducer[Any]
                if data_type in {_TOOL_INVOKED_TYPE, _PROMPT_EXECUTED_TYPE}:
                    default_reducer = cast(TypedReducer[Any], _append_event)
                else:
                    default_reducer = cast(TypedReducer[Any], append)
                registrations = [
                    _ReducerRegistration(
                        reducer=default_reducer,
                        slice_type=data_type,
                    )
                ]
            event_bus = self._bus

        context = build_reducer_context(session=self, event_bus=event_bus)

        for registration in registrations:
            slice_type = registration.slice_type
            while True:
                with self._lock:
                    previous = self._state.get(slice_type, ())
                try:
                    result = registration.reducer(previous, event, context=context)
                except Exception:  # log and continue
                    reducer_name = getattr(
                        registration.reducer, "__qualname__", repr(registration.reducer)
                    )
                    logger.exception(
                        "Reducer application failed.",
                        event="session_reducer_failed",
                        context={
                            "reducer": reducer_name,
                            "data_type": data_type.__qualname__,
                            "slice_type": slice_type.__qualname__,
                        },
                    )
                    break
                normalized = tuple(result)
                with self._lock:
                    current = self._state.get(slice_type, ())
                    if current is previous or current == normalized:
                        self._state[slice_type] = normalized
                        break

    def _attach_to_bus(self, bus: EventBus) -> None:
        with self._lock:
            if self._subscriptions_attached and self._bus is bus:
                return
            self._bus = bus
            self._subscriptions_attached = True
            bus.subscribe(ToolInvoked, self._on_tool_invoked)
            bus.subscribe(PromptExecuted, self._on_prompt_executed)
            bus.subscribe(PromptExecutionFailed, self._on_prompt_execution_failed)
            bus.subscribe(PromptRendered, self._on_prompt_rendered)


__all__ = [
    "DataEvent",
    "Session",
    "TypedReducer",
]
