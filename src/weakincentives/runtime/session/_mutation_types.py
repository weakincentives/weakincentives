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

"""Type definitions for mutation builders and reducers.

This module defines foundational types for the session system to avoid
import cycles. Types defined here can be imported by mutation.py,
_types.py, and other modules without creating circular dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from ...prompt._types import SupportsDataclass
from ..events._types import EventBus

if TYPE_CHECKING:
    from .snapshots import Snapshot


# ---------------------------------------------------------------------------
# Reducer-related types (moved from _types.py to break import cycle)
# ---------------------------------------------------------------------------


@runtime_checkable
class ReducerEventWithValue(Protocol):
    """Structural type satisfied by events exposing payloads via ``value``."""

    @property
    def value(self) -> SupportsDataclass | None: ...


ReducerEvent = SupportsDataclass | ReducerEventWithValue


S = TypeVar("S", bound=SupportsDataclass)


class ReducerSessionProtocol(Protocol):
    """Minimal session protocol for reducer context.

    This protocol captures the subset of :class:`SessionProtocol` that
    reducers may access, deliberately omitting mutation methods to avoid
    import cycles with :mod:`mutation`.
    """

    def snapshot(self) -> Snapshot: ...

    @property
    def event_bus(self) -> EventBus: ...

    def select_all(
        self, slice_type: type[SupportsDataclass]
    ) -> tuple[SupportsDataclass, ...]: ...

    @property
    def tags(self) -> Mapping[str, str]: ...


class ReducerContextProtocol(Protocol):
    """Protocol implemented by reducer context objects."""

    session: ReducerSessionProtocol


class TypedReducer(Protocol[S]):
    """Protocol for reducer callables maintained by :class:`Session`."""

    def __call__(
        self,
        slice_values: tuple[S, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[S, ...]: ...


# ---------------------------------------------------------------------------
# Mutation provider protocol
# ---------------------------------------------------------------------------


class MutationProvider(Protocol):
    """Protocol for objects that can provide mutation operations.

    This protocol is implemented by :class:`Session` and defines the
    internal interface used by :class:`MutationBuilder`.
    """

    def mutation_seed_slice[T: SupportsDataclass](
        self, slice_type: type[T], values: Iterable[T]
    ) -> None: ...

    def mutation_clear_slice[T: SupportsDataclass](
        self,
        slice_type: type[T],
        predicate: Callable[[T], bool] | None = None,
    ) -> None: ...

    def mutation_reset(self) -> None: ...

    def mutation_rollback(self, snapshot: Snapshot) -> None: ...

    def mutation_register_reducer[T: SupportsDataclass](
        self,
        data_type: type[SupportsDataclass],
        reducer: TypedReducer[T],
        *,
        slice_type: type[T] | None = None,
    ) -> None: ...

    def mutation_dispatch_event(
        self, slice_type: type[SupportsDataclass], event: SupportsDataclass
    ) -> None: ...


__all__ = [
    "MutationProvider",
    "ReducerContextProtocol",
    "ReducerEvent",
    "ReducerEventWithValue",
    "ReducerSessionProtocol",
    "TypedReducer",
]
