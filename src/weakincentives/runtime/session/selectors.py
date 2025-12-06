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

"""Helpers for querying Session slices.

These standalone functions are convenience wrappers around the fluent
``session.query(T)`` API. They are maintained for backward compatibility
and for cases where a functional style is preferred.
"""

from __future__ import annotations

from collections.abc import Callable

from ...dbc import pure
from ...prompt._types import SupportsDataclass
from .session import Session


@pure
def select_all[T: SupportsDataclass](
    session: Session, slice_type: type[T]
) -> tuple[T, ...]:
    """Return the entire slice for the provided type.

    Equivalent to ``session.query(slice_type).all()``.
    """
    return session.query(slice_type).all()


@pure
def select_latest[T: SupportsDataclass](
    session: Session, slice_type: type[T]
) -> T | None:
    """Return the most recent item in the slice, if any.

    Equivalent to ``session.query(slice_type).latest()``.
    """
    return session.query(slice_type).latest()


@pure
def select_where[T: SupportsDataclass](
    session: Session,
    slice_type: type[T],
    predicate: Callable[[T], bool],
) -> tuple[T, ...]:
    """Return items that satisfy the predicate.

    Equivalent to ``session.query(slice_type).where(predicate)``.
    """
    return session.query(slice_type).where(predicate)


__all__ = ["select_all", "select_latest", "select_where"]
