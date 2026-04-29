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

"""Event-sourced sessions with pure reducers.

The session is the spine's state container. All mutations route through
:meth:`Session.dispatch`, which routes events to typed reducers; reducers
return a :class:`SliceOp` describing how to update the typed slice tuple.
The session owns an ``RLock`` and is safe for concurrent use.
"""

# ``SliceAccessor`` and snapshot helpers reach into ``Session``'s underscore
# attributes by design; they are co-designed within the spine package.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, cast, final

from .errors import SessionError

if TYPE_CHECKING:
    from .protocols import EventListener

__all__ = [
    "Append",
    "PromptExecuted",
    "PromptRendered",
    "Replace",
    "Session",
    "SliceAccessor",
    "SliceOp",
    "ToolInvoked",
    "TypedReducer",
    "reducer",
]


@final
@dataclass(frozen=True, slots=True)
class Replace[SliceT]:
    """Reducer return value: replace the slice contents."""

    items: tuple[SliceT, ...]


@final
@dataclass(frozen=True, slots=True)
class Append[SliceT]:
    """Reducer return value: append items to the slice."""

    items: tuple[SliceT, ...]


SliceOp = Replace[Any] | Append[Any]
TypedReducer = Callable[[tuple[Any, ...], Any], SliceOp]

_REDUCER_ATTR = "__wink_reducer_event__"


def reducer(*, on: type[object]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method on a state slice dataclass as a reducer.

    The decorated method must accept ``(self, event)`` and return a
    :class:`SliceOp`. Register it with :meth:`Session.install`.
    """

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        setattr(method, _REDUCER_ATTR, on)
        return method

    return decorate


# =============================================================================
# Spine-emitted observability events
# =============================================================================


@final
@dataclass(frozen=True, slots=True)
class PromptRendered:
    """Emitted after :meth:`Prompt.render` produces a :class:`RenderedPrompt`."""

    ns: str
    key: str
    tool_count: int


@final
@dataclass(frozen=True, slots=True)
class ToolInvoked:
    """Emitted after a tool handler returns (success or error)."""

    name: str
    success: bool
    message: str


@final
@dataclass(frozen=True, slots=True)
class PromptExecuted:
    """Emitted after a higher-level evaluation completes."""

    ns: str
    key: str
    tool_calls: int


# =============================================================================
# Slice accessor
# =============================================================================


class SliceAccessor[SliceT]:
    """Fluent atomic API exposed by ``session[SliceType]``."""

    def __init__(self, session: Session, slice_type: type[SliceT]) -> None:
        super().__init__()
        self._session = session
        self._slice_type = slice_type

    def all(self) -> tuple[SliceT, ...]:
        return cast(
            "tuple[SliceT, ...]",
            self._session._snapshot_slice(self._slice_type),
        )

    def latest(self) -> SliceT | None:
        items = self.all()
        return items[-1] if items else None

    def where(self, predicate: Callable[[SliceT], bool]) -> tuple[SliceT, ...]:
        return tuple(item for item in self.all() if predicate(item))

    def seed(self, value: SliceT) -> None:
        self._session._set_slice(self._slice_type, (value,))

    def append(self, value: SliceT) -> None:
        self._session._append_to_slice(self._slice_type, value)

    def clear(self) -> None:
        self._session._set_slice(self._slice_type, ())


# =============================================================================
# Session
# =============================================================================


@dataclass(frozen=True, slots=True)
class _Registration:
    reducer_fn: TypedReducer
    slice_type: type[object]


class Session:
    """Thread-safe event-sourced state container.

    All public methods acquire an internal ``RLock`` so concurrent use from
    callback threads is safe. Reducers run inside the lock; keep them
    lightweight and side-effect free.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._slices: dict[type[object], tuple[object, ...]] = {}
        self._reducers: dict[type[object], list[_Registration]] = {}
        self._initials: dict[type[object], Callable[[], object]] = {}
        self._listeners: list[EventListener] = []

    def __getitem__[SliceT](self, slice_type: type[SliceT]) -> SliceAccessor[SliceT]:
        return SliceAccessor(self, slice_type)

    def subscribe(self, listener: EventListener) -> None:
        """Register an :class:`EventListener` for spine-emitted events."""
        with self._lock:
            self._listeners.append(listener)

    def publish(self, event: object) -> None:
        """Notify subscribed listeners of an event.

        Listeners are called synchronously; exceptions in one listener are
        re-raised to the caller (so debug bundles or eval traces can fail
        loudly during development).
        """
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener.on_event(event)

    def register[EventT](
        self,
        event_type: type[EventT],
        reducer_fn: TypedReducer,
        *,
        slice_type: type[object],
    ) -> None:
        """Register a free-function reducer for ``event_type``."""
        with self._lock:
            self._reducers.setdefault(event_type, []).append(
                _Registration(reducer_fn=reducer_fn, slice_type=slice_type)
            )

    def install[SliceT](
        self,
        slice_type: type[SliceT],
        *,
        initial: Callable[[], SliceT] | None = None,
    ) -> None:
        """Register every ``@reducer`` method on a state slice dataclass."""
        _require_frozen_dataclass(slice_type)
        with self._lock:
            if initial is not None:
                self._initials[slice_type] = cast("Callable[[], object]", initial)
        registrations = _collect_reducer_methods(slice_type)
        if not registrations:
            raise SessionError(f"{slice_type.__name__} declares no @reducer methods")
        for event_type, method_name in registrations:
            self.register(
                event_type,
                _make_method_reducer(slice_type, method_name),
                slice_type=slice_type,
            )

    def dispatch(self, event: object) -> None:
        """Route ``event`` to every reducer registered for ``type(event)``."""
        with self._lock:
            registrations = list(self._reducers.get(type(event), []))
            for registration in registrations:
                self._apply_registration(registration, event)

    def _apply_registration(self, registration: _Registration, event: object) -> None:
        current = self._slices.get(registration.slice_type, ())
        if not current and registration.slice_type in self._initials:
            current = (self._initials[registration.slice_type](),)
        op = registration.reducer_fn(current, event)
        if isinstance(op, Replace):
            self._slices[registration.slice_type] = tuple(op.items)
        else:
            self._slices[registration.slice_type] = (*current, *op.items)

    def reset(self) -> None:
        """Clear every slice without altering registrations."""
        with self._lock:
            self._slices.clear()

    # ------------------------------------------------------------------
    # SliceAccessor support
    # ------------------------------------------------------------------

    def _snapshot_slice(self, slice_type: type[object]) -> tuple[object, ...]:
        with self._lock:
            return self._slices.get(slice_type, ())

    def _set_slice(self, slice_type: type[object], items: tuple[object, ...]) -> None:
        with self._lock:
            if items:
                self._slices[slice_type] = items
            else:
                _ = self._slices.pop(slice_type, None)

    def _append_to_slice(self, slice_type: type[object], value: object) -> None:
        with self._lock:
            current = self._slices.get(slice_type, ())
            self._slices[slice_type] = (*current, value)

    # ------------------------------------------------------------------
    # Snapshot integration
    # ------------------------------------------------------------------

    def _slice_state(self) -> dict[type[object], tuple[object, ...]]:
        with self._lock:
            return dict(self._slices)

    def _restore_slice_state(
        self, slices: dict[type[object], tuple[object, ...]]
    ) -> None:
        with self._lock:
            self._slices = dict(slices)


# =============================================================================
# Reducer-method machinery
# =============================================================================


def _require_frozen_dataclass(slice_type: type[object]) -> None:
    if not is_dataclass(slice_type):
        raise SessionError(f"{slice_type.__name__} must be a frozen dataclass")
    params = getattr(slice_type, "__dataclass_params__", None)
    if params is None or not getattr(params, "frozen", False):
        raise SessionError(f"{slice_type.__name__} must be a frozen dataclass")


def _collect_reducer_methods(
    slice_type: type[object],
) -> tuple[tuple[type[object], str], ...]:
    seen: dict[type[object], str] = {}
    for name in dir(slice_type):
        if name.startswith("_"):
            continue
        event_type = getattr(getattr(slice_type, name, None), _REDUCER_ATTR, None)
        if event_type is None:
            continue
        if event_type in seen:
            msg = (
                f"{slice_type.__name__} has duplicate reducers for "
                + event_type.__name__
            )
            raise SessionError(msg)
        seen[event_type] = name
    return tuple(seen.items())


def _make_method_reducer(slice_type: type[object], method_name: str) -> TypedReducer:
    def method_reducer(current: tuple[Any, ...], event: object) -> SliceOp:
        if not current:
            raise SessionError(
                f"no state for {slice_type.__name__}; pass initial= to install()"
            )
        instance = current[-1]
        method = getattr(instance, method_name)
        return cast(SliceOp, method(event))

    method_reducer.__name__ = f"{slice_type.__name__}.{method_name}"
    method_reducer.__qualname__ = method_reducer.__name__
    return method_reducer
