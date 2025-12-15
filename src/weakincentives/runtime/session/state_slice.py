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

# pyright: reportImportCycles=false

"""Declarative state slice decorators for self-describing session state.

This module provides decorator-based tools for defining state slices with
their reducers co-located as methods on the dataclass. This eliminates
the need for manual reducer registration.

Example::

    from dataclasses import replace
    from weakincentives.runtime.session import state_slice, reducer

    @state_slice
    @dataclass(frozen=True)
    class AgentPlan:
        steps: tuple[str, ...]
        current_step: int = 0

        @reducer(on=AddStep)
        def add_step(self, event: AddStep) -> "AgentPlan":
            return replace(self, steps=self.steps + (event.step,))

        @reducer(on=CompleteStep)
        def complete(self, event: CompleteStep) -> "AgentPlan":
            return replace(self, current_step=self.current_step + 1)

    # Register once:
    session.install(AgentPlan)

    # Query naturally:
    session[AgentPlan].latest()

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, cast

from ...dbc import pure
from ...prompt._types import SupportsDataclass
from ._types import ReducerContextProtocol, ReducerEvent, TypedReducer

if TYPE_CHECKING:
    from .session import Session

# Marker attribute for state slice classes
_STATE_SLICE_MARKER = "__wink_state_slice__"

# Attribute storing reducer registrations on a method
_REDUCER_META = "__wink_reducer_meta__"


@dataclass(slots=True, frozen=True)
class ReducerMeta:
    """Metadata for a reducer method."""

    event_type: type[SupportsDataclass]
    method_name: str


@dataclass(slots=True, frozen=True)
class StateSliceMeta:
    """Metadata stored on a state slice class."""

    reducers: tuple[ReducerMeta, ...]
    initial_factory: Callable[[], SupportsDataclass] | None = None


def reducer[E: SupportsDataclass](
    on: type[E],
) -> Callable[[Callable[[Any, E], Any]], Callable[[Any, E], Any]]:
    """Decorator to mark a method as a reducer for a specific event type.

    The decorated method receives `self` (the current slice value) and the
    event, returning a new instance of the slice type.

    Args:
        on: The event type this reducer handles.

    Returns:
        A decorator that marks the method with reducer metadata.

    Example::

        @reducer(on=AddStep)
        def add_step(self, event: AddStep) -> "AgentPlan":
            return replace(self, steps=self.steps + (event.step,))

    """

    def decorator(method: Callable[[Any, E], Any]) -> Callable[[Any, E], Any]:
        # Store event type on method for later extraction
        setattr(method, _REDUCER_META, on)
        return method

    return decorator


def _extract_reducer_metadata[T: SupportsDataclass](
    cls: type[T],
) -> tuple[ReducerMeta, ...]:
    """Extract reducer metadata from a decorated class."""
    reducers: list[ReducerMeta] = []

    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name, None)
        if attr is None:
            continue
        event_type = getattr(attr, _REDUCER_META, None)
        if event_type is not None:
            reducers.append(ReducerMeta(event_type=event_type, method_name=name))

    return tuple(reducers)


def state_slice[T: SupportsDataclass](
    cls: type[T] | None = None,
    *,
    initial: Callable[[], T] | None = None,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Decorator to mark a frozen dataclass as a declarative state slice.

    The decorator scans the class for methods decorated with ``@reducer``
    and stores metadata for later auto-registration with a session.

    The decorated class must be a frozen dataclass.

    Args:
        cls: The frozen dataclass to mark as a state slice.
        initial: Optional factory function to create the initial state when
            the slice is empty. If provided, reducers can handle events even
            when no state exists yet.

    Returns:
        The same class with state slice metadata attached.

    Raises:
        TypeError: If the class is not a frozen dataclass.

    Example::

        @state_slice
        @dataclass(frozen=True)
        class AgentPlan:
            steps: tuple[str, ...]
            current_step: int = 0

            @reducer(on=AddStep)
            def add_step(self, event: AddStep) -> "AgentPlan":
                return replace(self, steps=self.steps + (event.step,))

        # With initial factory for auto-initialization:
        @state_slice(initial=lambda: Counters())
        @dataclass(frozen=True)
        class Counters:
            count: int = 0

            @reducer(on=Increment)
            def increment(self, event: Increment) -> "Counters":
                return replace(self, count=self.count + event.amount)

    """

    def decorator(target_cls: type[T]) -> type[T]:
        if not is_dataclass(target_cls):
            msg = f"@state_slice requires a dataclass, got {target_cls.__name__}"
            raise TypeError(msg)

        # Check if frozen by looking at __dataclass_fields__
        # A frozen dataclass will have its instances be unhashable if mutable
        # types are used, but we can check the frozen parameter through
        # __dataclass_params__
        params = getattr(target_cls, "__dataclass_params__", None)
        if params is not None and not getattr(params, "frozen", False):
            msg = (
                f"@state_slice requires a frozen dataclass, "
                f"{target_cls.__name__} is not frozen"
            )
            raise TypeError(msg)

        # Extract reducer metadata from methods
        reducer_meta = _extract_reducer_metadata(target_cls)

        # Store metadata on the class
        meta = StateSliceMeta(
            reducers=reducer_meta,
            initial_factory=cast(
                Callable[[], SupportsDataclass] | None,
                initial,
            ),
        )
        setattr(target_cls, _STATE_SLICE_MARKER, meta)

        return target_cls

    # Handle both @state_slice and @state_slice(...) syntax
    if cls is not None:
        return decorator(cls)
    return decorator


def is_state_slice(cls: type[object]) -> bool:
    """Check if a class is decorated with @state_slice.

    Args:
        cls: The class to check.

    Returns:
        True if the class is a state slice, False otherwise.
    """
    return hasattr(cls, _STATE_SLICE_MARKER)


def get_state_slice_meta(cls: type[object]) -> StateSliceMeta | None:
    """Get the state slice metadata from a decorated class.

    Args:
        cls: The class to get metadata from.

    Returns:
        The state slice metadata, or None if not a state slice.
    """
    return getattr(cls, _STATE_SLICE_MARKER, None)


def _create_reducer_for_method[S: SupportsDataclass, E: SupportsDataclass](
    slice_type: type[S],
    method_name: str,
    initial_factory: Callable[[], S] | None = None,
) -> TypedReducer[S]:
    """Create a reducer function that wraps a method on the slice class.

    The generated reducer:
    1. Gets the latest slice value (or creates a default via initial_factory)
    2. Calls the method with the event
    3. Returns the result as a singleton tuple

    Args:
        slice_type: The state slice class.
        method_name: The name of the method to wrap.
        initial_factory: Optional factory to create initial state if empty.

    Returns:
        A reducer function compatible with the session reducer protocol.
    """

    @pure
    def method_reducer(
        slice_values: tuple[S, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[S, ...]:
        del context  # Unused, method receives only self and event

        # Get the current state or create initial if factory provided
        if slice_values:
            current = slice_values[-1]
        elif initial_factory is not None:
            current = initial_factory()
        else:
            # No current state and no factory - cannot invoke method
            # Return empty to signal no state change
            return slice_values

        # Get the method from the instance
        method = getattr(current, method_name)

        # Call the method with the event
        result = method(cast(E, event))

        # Return as singleton tuple (replace_latest semantics)
        return (result,)

    # Set qualname for better error messages
    method_reducer.__qualname__ = f"{slice_type.__name__}.{method_name}"
    method_reducer.__name__ = method_name

    return method_reducer


def install_state_slice[T: SupportsDataclass](
    session: Session,
    slice_type: type[T],
    *,
    initial: Callable[[], T] | None = None,
) -> None:
    """Install a declarative state slice into a session.

    This registers all reducers defined on the slice class with the session.
    It should be called once per session to enable the slice.

    The slice class can optionally be decorated with ``@state_slice`` for
    additional configuration (like initial factory), but this is not required.
    Methods decorated with ``@reducer`` will be discovered automatically.

    Args:
        session: The session to install the slice into.
        slice_type: The state slice class (a frozen dataclass with @reducer methods).
        initial: Optional factory function to create initial state when empty.
            Overrides any initial factory from @state_slice decorator.

    Raises:
        TypeError: If the class is not a frozen dataclass.
        ValueError: If no @reducer methods are found.

    Example::

        session.install(AgentPlan)
        # or equivalently:
        install_state_slice(session, AgentPlan)

        # With initial factory:
        session.install(AgentPlan, initial=lambda: AgentPlan(steps=()))

    """
    # Validate it's a dataclass
    if not is_dataclass(slice_type):
        msg = f"{slice_type.__name__} must be a dataclass"
        raise TypeError(msg)

    # Check if frozen
    params = getattr(slice_type, "__dataclass_params__", None)
    if params is not None and not getattr(params, "frozen", False):
        msg = f"{slice_type.__name__} must be a frozen dataclass"
        raise TypeError(msg)

    # Get metadata from @state_slice if present, otherwise scan directly
    meta = get_state_slice_meta(slice_type)
    if meta is not None:
        reducer_metas = meta.reducers
        initial_factory = initial or cast(
            Callable[[], T] | None,
            meta.initial_factory,
        )
    else:
        # Scan class directly for @reducer decorated methods
        reducer_metas = _extract_reducer_metadata(slice_type)
        initial_factory = initial

    if not reducer_metas:
        msg = f"{slice_type.__name__} has no @reducer decorated methods"
        raise ValueError(msg)

    # Register each reducer method
    for reducer_meta in reducer_metas:
        method_reducer = _create_reducer_for_method(
            slice_type,
            reducer_meta.method_name,
            initial_factory,
        )
        session.mutation_register_reducer(
            reducer_meta.event_type,
            method_reducer,
            slice_type=slice_type,
        )


__all__ = [
    "ReducerMeta",
    "StateSliceMeta",
    "get_state_slice_meta",
    "install_state_slice",
    "is_state_slice",
    "reducer",
    "state_slice",
]
