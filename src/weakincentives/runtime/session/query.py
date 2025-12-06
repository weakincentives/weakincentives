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

"""Fluent query builder for Session slices."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ...dbc import pure
from ...prompt._types import SupportsDataclass


class _SelectAllProvider(Protocol):
    """Protocol for objects that can provide slice data."""

    def select_all[S: SupportsDataclass](
        self, slice_type: type[S]
    ) -> tuple[S, ...]: ...


class QueryBuilder[T: SupportsDataclass]:
    """Fluent interface for querying session slices.

    Usage::

        session.query(Plan).latest()
        session.query(Plan).all()
        session.query(Plan).where(lambda p: p.active)

    """

    __slots__ = ("_provider", "_slice_type")

    def __init__(self, provider: _SelectAllProvider, slice_type: type[T]) -> None:
        super().__init__()
        self._provider = provider
        self._slice_type = slice_type

    @pure
    def all(self) -> tuple[T, ...]:
        """Return the entire slice for the provided type."""
        return self._provider.select_all(self._slice_type)

    @pure
    def latest(self) -> T | None:
        """Return the most recent item in the slice, if any."""
        values = self._provider.select_all(self._slice_type)
        if not values:
            return None
        return values[-1]

    @pure
    def where(self, predicate: Callable[[T], bool]) -> tuple[T, ...]:
        """Return items that satisfy the predicate."""
        return tuple(
            value
            for value in self._provider.select_all(self._slice_type)
            if predicate(value)
        )


__all__ = ["QueryBuilder"]
