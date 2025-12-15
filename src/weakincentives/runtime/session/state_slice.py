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

"""Declarative reducer decorator for self-describing session state.

This module provides a decorator-based approach for defining reducers
co-located as methods on state slice dataclasses. This eliminates
the need for separate reducer functions and manual registration.

Example::

    from dataclasses import dataclass, replace
    from weakincentives.runtime.session import reducer

    @dataclass(frozen=True)
    class AgentPlan:
        steps: tuple[str, ...]
        current_step: int = 0

        @reducer(on=AddStep)
        def add_step(self, event: AddStep) -> "AgentPlan":
            return replace(self, steps=(*self.steps, event.step))

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

# Attribute storing reducer event type on a method
_REDUCER_META = "__wink_reducer_meta__"


@dataclass(slots=True, frozen=True)
class ReducerMeta:
    """Metadata for a reducer method."""

    event_type: type[SupportsDataclass]
    method_name: str


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
            return replace(self, steps=(*self.steps, event.step))

    """

    def decorator(method: Callable[[Any, E], Any]) -> Callable[[Any, E], Any]:
        setattr(method, _REDUCER_META, on)
        return method

    return decorator


def _extract_reducer_metadata[T: SupportsDataclass](
    cls: type[T],
) -> tuple[ReducerMeta, ...]:
    """Extract reducer metadata from methods decorated with @reducer."""
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
            return slice_values

        # Get the method from the instance and call with event
        method = getattr(current, method_name)
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
    """Install a state slice into a session by registering its @reducer methods.

    Scans the class for methods decorated with ``@reducer`` and registers
    each as a reducer for the corresponding event type.

    Args:
        session: The session to install the slice into.
        slice_type: A frozen dataclass with ``@reducer`` decorated methods.
        initial: Optional factory to create initial state when empty.
            When provided, reducers can handle events even when no state
            exists yet.

    Raises:
        TypeError: If the class is not a frozen dataclass.
        ValueError: If no @reducer methods are found.

    Example::

        @dataclass(frozen=True)
        class AgentPlan:
            steps: tuple[str, ...]

            @reducer(on=AddStep)
            def add_step(self, event: AddStep) -> AgentPlan:
                return replace(self, steps=(*self.steps, event.step))

        session.install(AgentPlan)
        # Or with initial factory:
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

    # Scan for @reducer decorated methods
    reducer_metas = _extract_reducer_metadata(slice_type)
    if not reducer_metas:
        msg = f"{slice_type.__name__} has no @reducer decorated methods"
        raise ValueError(msg)

    # Register each reducer method
    for reducer_meta in reducer_metas:
        method_reducer = _create_reducer_for_method(
            slice_type,
            reducer_meta.method_name,
            initial,
        )
        session.mutation_register_reducer(
            reducer_meta.event_type,
            method_reducer,
            slice_type=slice_type,
        )


__all__ = [
    "ReducerMeta",
    "install_state_slice",
    "reducer",
]
