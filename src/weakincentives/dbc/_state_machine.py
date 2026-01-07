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

"""State machine transition enforcement decorators.

See specs/STATE_MACHINES.md for the full specification.
"""

# pyright: reportImportCycles=false

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from ._errors import InvalidStateError

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
StateT = TypeVar("StateT", bound=Enum)


@dataclass(frozen=True, slots=True)
class TransitionSpec:
    """Specification of a single transition."""

    from_states: frozenset[Enum]
    to_state: Enum | None
    method_name: str


@dataclass(frozen=True, slots=True)
class StateMachineSpec:
    """Extracted state machine specification."""

    cls: type[Any]
    state_var: str
    states: type[Enum]
    initial: Enum
    transitions: tuple[TransitionSpec, ...] = field(default_factory=tuple)

    def to_mermaid(self) -> str:
        """Export as Mermaid state diagram."""
        lines = ["stateDiagram-v2", f"    [*] --> {self.initial.name}"]
        for t in self.transitions:
            if t.to_state is not None:
                lines.extend(
                    f"    {from_state.name} --> {t.to_state.name}: {t.method_name}()"
                    for from_state in t.from_states
                )
        return "\n".join(lines)


# Sentinel for "any state" in @enters
_ANY_STATE: frozenset[Enum] = frozenset()

# Module-level flag import (avoid circular import)
_dbc_active: Callable[[], bool] | None = None


def _get_dbc_active() -> bool:
    """Lazily import and call dbc_active()."""
    global _dbc_active
    if _dbc_active is None:
        from . import dbc_active

        _dbc_active = dbc_active
    return _dbc_active()


def state_machine(
    *,
    state_var: str,
    states: type[StateT],
    initial: StateT,
) -> Callable[[type[T]], type[T]]:
    """Class decorator that enables state machine enforcement.

    Args:
        state_var: Name of the instance attribute holding current state.
        states: Enum class defining valid states.
        initial: Initial state, set after __init__ completes.

    Example::

        class LoopState(Enum):
            IDLE = auto()
            RUNNING = auto()
            STOPPED = auto()

        @state_machine(state_var="_state", states=LoopState, initial=LoopState.IDLE)
        class MainLoop:
            ...
    """
    if initial not in states:
        msg = f"Initial state {initial} not in states enum {states.__name__}"
        raise ValueError(msg)

    def decorator(cls: type[T]) -> type[T]:
        # Collect transition specs from decorated methods
        transitions: list[TransitionSpec] = []
        for name in dir(cls):
            attr = getattr(cls, name, None)
            if attr is not None and hasattr(attr, "__transition_spec__"):
                transitions.append(attr.__transition_spec__)

        # Store spec on class
        spec = StateMachineSpec(
            cls=cls,
            state_var=state_var,
            states=states,
            initial=initial,
            transitions=tuple(transitions),
        )
        cls.__state_machine_spec__ = spec  # type: ignore[attr-defined]

        # Wrap __init__ to set initial state
        original_init = cls.__init__

        @wraps(original_init)
        def init_wrapper(self: T, *args: object, **kwargs: object) -> None:
            original_init(self, *args, **kwargs)
            object.__setattr__(self, state_var, initial)

        type.__setattr__(cls, "__init__", init_wrapper)

        return cls

    return decorator


def transition(
    *,
    from_: StateT | tuple[StateT, ...],
    to: StateT,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Method decorator declaring a state transition.

    Args:
        from_: Valid source state(s) for this transition.
        to: Target state after method completes successfully.

    Example::

        @transition(from_=LoopState.IDLE, to=LoopState.RUNNING)
        def run(self) -> None:
            ...
    """
    from_states = frozenset((from_,) if isinstance(from_, Enum) else from_)

    def decorator(method: Callable[P, R]) -> Callable[P, R]:
        method_name = getattr(method, "__name__", repr(method))

        @wraps(method)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            if _get_dbc_active():
                cls = type(self)
                spec = cast(StateMachineSpec, cls.__state_machine_spec__)  # type: ignore[attr-defined]
                current = getattr(self, spec.state_var)
                if current not in from_states:
                    raise InvalidStateError(
                        cls, method_name, current, tuple(from_states)
                    )

            result = method(*args, **kwargs)

            if _get_dbc_active():
                cls = type(self)
                spec = cast(StateMachineSpec, cls.__state_machine_spec__)  # type: ignore[attr-defined]
                object.__setattr__(self, spec.state_var, to)

            return result

        # Store spec for extraction
        wrapper.__transition_spec__ = TransitionSpec(  # type: ignore[attr-defined]
            from_states=from_states,
            to_state=to,
            method_name=method_name,
        )
        return wrapper

    return decorator


def in_state(
    *valid_states: Enum,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Method decorator requiring specific state(s) without transition.

    Args:
        *valid_states: One or more valid states for this method.

    Example::

        @in_state(LoopState.RUNNING)
        def execute(self, request: Request) -> Response:
            ...
    """
    if not valid_states:
        msg = "@in_state requires at least one state"
        raise ValueError(msg)

    states_set = frozenset(valid_states)

    def decorator(method: Callable[P, R]) -> Callable[P, R]:
        method_name = getattr(method, "__name__", repr(method))

        @wraps(method)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self = args[0]
            if _get_dbc_active():
                cls = type(self)
                spec = cast(StateMachineSpec, cls.__state_machine_spec__)  # type: ignore[attr-defined]
                current = getattr(self, spec.state_var)
                if current not in states_set:
                    raise InvalidStateError(
                        cls, method_name, current, tuple(states_set)
                    )

            return method(*args, **kwargs)

        # Store spec for extraction
        wrapper.__transition_spec__ = TransitionSpec(  # type: ignore[attr-defined]
            from_states=states_set,
            to_state=None,
            method_name=method_name,
        )
        return wrapper

    return decorator


def enters(
    target_state: Enum,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Method decorator for transitions from any state.

    Shorthand for @transition(from_=<all states>, to=target_state).

    Args:
        target_state: Target state after method completes.

    Example::

        @enters(ContextState.CLOSED)
        def close(self) -> None:
            ...
    """

    def decorator(method: Callable[P, R]) -> Callable[P, R]:
        method_name = getattr(method, "__name__", repr(method))

        @wraps(method)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            result = method(*args, **kwargs)

            if _get_dbc_active():
                self = args[0]
                cls = type(self)
                spec = cast(StateMachineSpec, cls.__state_machine_spec__)  # type: ignore[attr-defined]
                object.__setattr__(self, spec.state_var, target_state)

            return result

        # Store spec for extraction (empty from_states = any state)
        wrapper.__transition_spec__ = TransitionSpec(  # type: ignore[attr-defined]
            from_states=_ANY_STATE,
            to_state=target_state,
            method_name=method_name,
        )
        return wrapper

    return decorator


def extract_state_machine(cls: type[Any]) -> StateMachineSpec:
    """Extract state machine specification from a decorated class.

    Args:
        cls: Class decorated with @state_machine.

    Returns:
        The StateMachineSpec for the class.

    Raises:
        AttributeError: If class is not decorated with @state_machine.

    Example::

        sm = extract_state_machine(MainLoop)
        print(sm.to_mermaid())
    """
    return cls.__state_machine_spec__


def iter_state_machines() -> Iterator[StateMachineSpec]:
    """Iterate over all registered state machine specs.

    Note: This only finds specs in classes that have been imported.
    """
    # This would require a registry - skip for now
    raise NotImplementedError("Use extract_state_machine(cls) instead")


__all__ = [
    "StateMachineSpec",
    "TransitionSpec",
    "enters",
    "extract_state_machine",
    "in_state",
    "state_machine",
    "transition",
]
