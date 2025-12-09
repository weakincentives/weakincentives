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

"""Protocols describing Session behavior exposed to other modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, Self, overload

from ...prompt._types import SupportsDataclass
from ..events._types import EventBus
from ._observer_types import SliceObserver, Subscription
from .snapshots import Snapshot

if TYPE_CHECKING:
    from .mutation import GlobalMutationBuilder, MutationBuilder
    from .query import QueryBuilder

type SnapshotProtocol = Snapshot


class SessionProtocol(Protocol):
    """Structural protocol implemented by session state containers."""

    def snapshot(self) -> SnapshotProtocol: ...

    @property
    def event_bus(self) -> EventBus: ...

    def select_all(
        self, slice_type: type[SupportsDataclass]
    ) -> tuple[SupportsDataclass, ...]: ...

    def query[T: SupportsDataclass](self, slice_type: type[T]) -> QueryBuilder[T]: ...

    @overload
    def mutate[T: SupportsDataclass](
        self, slice_type: type[T]
    ) -> MutationBuilder[T]: ...

    @overload
    def mutate(self) -> GlobalMutationBuilder: ...

    def mutate[T: SupportsDataclass](
        self, slice_type: type[T] | None = None
    ) -> MutationBuilder[T] | GlobalMutationBuilder: ...

    def observe[T: SupportsDataclass](
        self,
        slice_type: type[T],
        observer: SliceObserver[T],
    ) -> Subscription: ...

    @property
    def parent(self) -> Self | None: ...

    @property
    def children(self) -> tuple[Self, ...]: ...

    @property
    def tags(self) -> Mapping[str, str]: ...


__all__ = ["SessionProtocol", "SnapshotProtocol"]
