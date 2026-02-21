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

"""Session dispatch helpers for slice operations.

This module provides helper functions for applying slice operations and
dispatching events to registered reducers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, assert_never, cast

from ...types.dataclass import SupportsDataclass
from ..logging import StructuredLogger, get_logger
from ._slice_types import SessionSliceType
from ._types import ReducerEvent, TypedReducer
from .reducer_registry import ReducerRegistration
from .reducers import append_all
from .slice_mutations import ClearSlice, InitializeSlice
from .slices import Append, Clear, Extend, Replace, Slice, SliceOp

if TYPE_CHECKING:
    from .session import Session

logger: StructuredLogger = get_logger(__name__, context={"component": "session"})


def apply_slice_op[S: SupportsDataclass](
    op: SliceOp[S],
    slice_instance: Slice[S],
) -> None:
    """Apply slice operation using optimal method.

    Args:
        op: The slice operation to apply (Append, Extend, Replace, or Clear).
        slice_instance: The slice instance to mutate.
    """
    match op:
        case Append(item=item):
            slice_instance.append(item)
        case Extend(items=items):
            slice_instance.extend(items)
        case Replace(items=items):
            slice_instance.replace(items)
        case Clear(predicate=pred):
            slice_instance.clear(pred)
        case _ as unreachable:  # pragma: no cover - exhaustiveness sentinel
            assert_never(unreachable)  # pyright: ignore[reportUnreachable]


def handle_system_mutation_event(
    session: Session,
    event: ReducerEvent,
) -> bool:
    """Handle system mutation events (InitializeSlice, ClearSlice).

    These events bypass normal reducer dispatch and directly mutate state,
    ensuring consistent behavior regardless of registered reducers.

    Returns:
        True if the event was a system mutation event and was handled,
        False otherwise.
    """
    if isinstance(event, InitializeSlice):
        # Use cast to work around generic type parameter inference
        init_event = cast("InitializeSlice[Any]", event)
        slice_type: SessionSliceType = init_event.slice_type
        values = init_event.values
        logger.debug(
            "session.initialize_slice",
            event="session.initialize_slice",
            context={
                "session_id": str(session.session_id),
                "slice_type": slice_type.__qualname__,
                "value_count": len(values),
            },
        )
        with session.locked():
            slice_instance = session._store.get_or_create(slice_type)  # pyright: ignore[reportPrivateUsage]
            slice_instance.replace(values)
        return True

    if isinstance(event, ClearSlice):
        # Use cast to work around generic type parameter inference
        clear_event = cast("ClearSlice[Any]", event)
        slice_type = clear_event.slice_type
        predicate: Callable[[Any], bool] | None = clear_event.predicate
        logger.debug(
            "session.clear_slice",
            event="session.clear_slice",
            context={
                "session_id": str(session.session_id),
                "slice_type": slice_type.__qualname__,
                "has_predicate": predicate is not None,
            },
        )
        with session.locked():
            slice_instance = session._store.get_or_create(slice_type)  # pyright: ignore[reportPrivateUsage]
            slice_instance.clear(predicate)
        return True

    return False


def dispatch_data_event(
    session: Session,
    data_type: SessionSliceType,
    event: ReducerEvent,
) -> None:
    """Dispatch a data event to registered reducers."""
    # Handle system mutation events specially
    if handle_system_mutation_event(session, event):
        return

    from .reducer_context import build_reducer_context
    from .session_view import SessionView

    with session.locked():
        registrations = list(session._registry.get_registrations(data_type))  # pyright: ignore[reportPrivateUsage]

        if not registrations:
            # Default: ledger semantics (always append)
            registrations = [
                ReducerRegistration(
                    reducer=cast(TypedReducer[Any], append_all),
                    slice_type=data_type,
                )
            ]

    logger.debug(
        "session.dispatch_data_event",
        event="session.dispatch_data_event",
        context={
            "session_id": str(session.session_id),
            "data_type": data_type.__qualname__,
            "reducer_count": len(registrations),
        },
    )

    view = SessionView(session)
    context = build_reducer_context(session=view)

    for registration in registrations:
        slice_type = registration.slice_type
        reducer_name = getattr(
            registration.reducer, "__qualname__", repr(registration.reducer)
        )
        with session.locked():
            slice_instance = session._store.get_or_create(slice_type)  # pyright: ignore[reportPrivateUsage]
            slice_view = slice_instance.view()
        try:
            op = registration.reducer(slice_view, event, context=context)
            op_type = type(op).__name__
            logger.debug(
                "session.reducer_applied",
                event="session.reducer_applied",
                context={
                    "session_id": str(session.session_id),
                    "reducer": reducer_name,
                    "slice_type": slice_type.__qualname__,
                    "operation": op_type,
                },
            )
            # Apply the slice operation
            with session.locked():
                apply_slice_op(op, slice_instance)
        except Exception:  # log and continue
            logger.exception(
                "Reducer application failed.",
                event="session_reducer_failed",
                context={
                    "reducer": reducer_name,
                    "data_type": data_type.__qualname__,
                    "slice_type": slice_type.__qualname__,
                },
            )
            continue


__all__ = [
    "apply_slice_op",
    "dispatch_data_event",
    "handle_system_mutation_event",
]
