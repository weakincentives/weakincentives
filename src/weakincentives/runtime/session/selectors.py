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

"""Helpers for querying Session slices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...dbc import pure
from ...prompt._types import SupportsDataclass

if TYPE_CHECKING:
    from collections.abc import Callable

    from .session import Session


@pure
def select_all[T: SupportsDataclass](
    session: Session, slice_type: type[T]
) -> tuple[T, ...]:
    """Return the entire slice for the provided type."""

    return session.select_all(slice_type)


@pure
def select_latest[T: SupportsDataclass](
    session: Session, slice_type: type[T]
) -> T | None:
    """Return the most recent item in the slice, if any."""

    values = session.select_all(slice_type)
    if not values:
        return None
    return values[-1]


@pure
def select_where[T: SupportsDataclass](
    session: Session,
    slice_type: type[T],
    predicate: Callable[[T], bool],
) -> tuple[T, ...]:
    """Return items that satisfy the predicate."""

    return tuple(value for value in session.select_all(slice_type) if predicate(value))


__all__ = ["select_all", "select_latest", "select_where"]
