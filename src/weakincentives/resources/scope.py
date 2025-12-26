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

"""Resource lifetime scopes."""

from __future__ import annotations

from enum import Enum


class Scope(Enum):
    """Determines instance lifetime and caching behavior.

    Scopes control when resources are constructed and how long they live:

    - ``SINGLETON``: One instance per session, created on first access.
    - ``TOOL_CALL``: Fresh instance per tool invocation, disposed after.
    - ``PROTOTYPE``: Fresh instance on every access, never cached.

    Example::

        from weakincentives.resources import Binding, Scope

        # Connection pool lives for entire session
        pool_binding = Binding(ConnectionPool, make_pool, scope=Scope.SINGLETON)

        # Request tracer is fresh per tool call
        tracer_binding = Binding(Tracer, make_tracer, scope=Scope.TOOL_CALL)

        # Builder is fresh every time
        builder_binding = Binding(QueryBuilder, make_builder, scope=Scope.PROTOTYPE)
    """

    SINGLETON = "singleton"
    """One instance per session. Created on first access, reused thereafter."""

    TOOL_CALL = "tool_call"
    """Fresh instance per tool invocation. Disposed after tool completes."""

    PROTOTYPE = "prototype"
    """Fresh instance on every access. Never cached."""


__all__ = ["Scope"]
