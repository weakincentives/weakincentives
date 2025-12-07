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

"""Reducer context threaded through session reducer invocations."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from ...dataclasses import FrozenDataclass
from ._types import ReducerContextProtocol
from .protocols import SessionProtocol

if TYPE_CHECKING:
    from ..events import EventBus


@FrozenDataclass()
class ReducerContext(ReducerContextProtocol):
    """Immutable bundle of runtime services shared with reducers."""

    session: SessionProtocol

    @property
    @override
    def event_bus(self) -> EventBus:
        """Return the event bus from the session."""
        return self.session.event_bus


def build_reducer_context(*, session: SessionProtocol) -> ReducerContext:
    """Return a :class:`ReducerContext` for the provided session."""

    return ReducerContext(session=session)


__all__ = ["ReducerContext", "build_reducer_context"]
