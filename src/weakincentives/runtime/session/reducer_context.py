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

from ...dataclasses import FrozenDataclass
from ._protocols import ReducerContextProtocol, SessionViewProtocol


@FrozenDataclass()
class ReducerContext(ReducerContextProtocol):
    """Immutable bundle of runtime services shared with reducers.

    Provides read-only access to session state via :class:`SessionViewProtocol`.
    """

    session: SessionViewProtocol


def build_reducer_context(*, session: SessionViewProtocol) -> ReducerContext:
    """Return a :class:`ReducerContext` for the provided session view.

    Args:
        session: A SessionViewProtocol providing read-only session access.

    Returns:
        A ReducerContext wrapping the provided session view.

    """
    return ReducerContext(session=session)


__all__ = ["ReducerContext", "build_reducer_context"]
