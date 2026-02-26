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
# pyright: reportPrivateUsage=false

"""Declarative reducer decorator for self-describing session state.

This module provides a decorator-based approach for defining reducers
co-located as methods on state slice dataclasses. This eliminates
the need for separate reducer functions and manual registration.

Declarative reducers return SliceOp, maintaining consistency with the
functional reducer API:

Example::

    from dataclasses import dataclass, replace
    from weakincentives.runtime.session import reducer
    from weakincentives.runtime.session.slices import Replace

    @dataclass(frozen=True)
    class AgentPlan:
        steps: tuple[str, ...]
        current_step: int = 0

        @reducer(on=AddStep)
        def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
            new_plan = replace(self, steps=(*self.steps, event.step))
            return Replace((new_plan,))

        @reducer(on=CompleteStep)
        def complete(self, event: CompleteStep) -> Replace["AgentPlan"]:
            new_plan = replace(self, current_step=self.current_step + 1)
            return Replace((new_plan,))

    # Register once:
    session.install(AgentPlan)

    # Query naturally:
    session[AgentPlan].latest()

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, cast

from ...dataclasses import FrozenDataclassMixin
from ...types.dataclass import SupportsDataclass
from ._types import ReducerContextProtocol, ReducerEvent, TypedReducer
from .slices import Replace, SliceOp, SliceView

if TYPE_CHECKING:
    from .session import Session

# Attribute storing reducer event type on a method
_REDUCER_META = "__wink_reducer_meta__"


@dataclass(slots=True, frozen=True)
class ReducerMeta(FrozenDataclassMixin):
    """Metadata for a reducer method."""

    event_type: type[SupportsDataclass]
    method_name: str


def reducer[E: SupportsDataclass](
    on: type[E],
) -> Callable[[Callable[[Any, E], Any]], Callable[[Any, E], Any]]:
    """Decorator to mark a method as a reducer for a specific event type.

    The decorated method receives ``self`` (the current slice value) and the
    event, returning a SliceOp describing the mutation. Each event type may
    only have one reducer method per slice class.

    Args:
        on: The event type this reducer handles. Must be a frozen dataclass.

    Returns:
        A decorator that marks the method with reducer metadata.

    Note:
        The method signature must be ``(self, event: E) -> SliceOp[Self]`` where:

        - ``self`` is the current state (used to compute the new state)
        - ``event`` is the dispatched event of type ``E``
        - Return type must be a SliceOp containing the new state

    Example::

        @reducer(on=AddStep)
        def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
            new_plan = replace(self, steps=(*self.steps, event.step))
            return Replace((new_plan,))

    """

    def decorator(method: Callable[[Any, E], Any]) -> Callable[[Any, E], Any]:
        setattr(method, _REDUCER_META, on)
        return method

    return decorator


def _extract_reducer_metadata[T: SupportsDataclass](
    cls: type[T],
) -> tuple[ReducerMeta, ...]:
    """Extract reducer metadata from methods decorated with @reducer.

    Raises:
        ValueError: If multiple methods handle the same event type.
    """
    reducers: list[ReducerMeta] = []
    seen_events: dict[type[SupportsDataclass], str] = {}

    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name, None)
        if attr is None:
            continue
        event_type = getattr(attr, _REDUCER_META, None)
        if event_type is not None:
            # Check for duplicate event handlers
            if event_type in seen_events:
                msg = (
                    f"{cls.__name__} has multiple @reducer methods for "
                    f"{event_type.__name__}: {seen_events[event_type]} and {name}"
                )
                raise ValueError(msg)
            seen_events[event_type] = name
            reducers.append(ReducerMeta(event_type=event_type, method_name=name))

    return tuple(reducers)


def _create_reducer_for_method[S: SupportsDataclass, E: SupportsDataclass](
    slice_type: type[S],
    method_name: str,
    initial_factory: Callable[[], S] | None = None,
) -> TypedReducer[S]:
    """Create a reducer function that wraps a method on the slice class.

    The generated reducer:
    1. Gets the latest slice value from view (or creates a default via initial_factory)
    2. Calls the method with the event
    3. Returns the SliceOp from the method directly

    Args:
        slice_type: The state slice class.
        method_name: The name of the method to wrap.
        initial_factory: Optional factory to create initial state if empty.

    Returns:
        A reducer function compatible with the session reducer protocol.
    """

    def method_reducer(
        view: SliceView[S],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> SliceOp[S]:
        del context  # Unused, method receives only self and event

        # Get the current state or create initial if factory provided
        current = view.latest()
        if current is None:
            if initial_factory is not None:
                current = initial_factory()
            else:
                # No current state and no factory - return empty replace
                return Replace(())

        # Get the method from the instance and call with event
        method = getattr(current, method_name)
        result = method(cast(E, event))

        # The method should return a SliceOp directly
        return cast(SliceOp[S], result)

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
        ValueError: If no @reducer methods are found, or if multiple methods
            handle the same event type.

    Example::

        @dataclass(frozen=True)
        class AgentPlan:
            steps: tuple[str, ...]

            @reducer(on=AddStep)
            def add_step(self, event: AddStep) -> Replace[AgentPlan]:
                new_plan = replace(self, steps=(*self.steps, event.step))
                return Replace((new_plan,))

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
        session._mutation_register_reducer(
            reducer_meta.event_type,
            method_reducer,
            slice_type=slice_type,
        )


__all__ = [
    "ReducerMeta",
    "install_state_slice",
    "reducer",
]
