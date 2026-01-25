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

"""Resource lifetime scopes for dependency injection.

This module defines the ``Scope`` enum, which controls how resource instances
are created and cached within the dependency injection container. Choosing the
correct scope is critical for resource management, thread safety, and memory
efficiency.
"""

from __future__ import annotations

from enum import Enum


class Scope(Enum):
    """Determines instance lifetime and caching behavior for resources.

    Scopes control when resources are constructed and how long they live.
    Choose a scope based on the resource's characteristics:

    - ``SINGLETON``: Use for expensive-to-create, stateless, or shared resources
      (e.g., connection pools, configuration objects, caches).
    - ``TOOL_CALL``: Use for resources that need isolation per operation
      (e.g., request tracers, transaction contexts, temporary scratch space).
    - ``PROTOTYPE``: Use for lightweight, stateful builders or objects that
      must not be shared (e.g., query builders, mutable accumulators).

    Resources implementing ``Closeable`` are automatically disposed when their
    scope ends. Singletons are closed at session end; tool-call-scoped resources
    are closed after each tool invocation.

    Example::

        from weakincentives.resources import Binding, Scope

        # Connection pool lives for entire session
        pool_binding = Binding(ConnectionPool, make_pool, scope=Scope.SINGLETON)

        # Request tracer is fresh per tool call
        tracer_binding = Binding(Tracer, make_tracer, scope=Scope.TOOL_CALL)

        # Builder is fresh every time
        builder_binding = Binding(QueryBuilder, make_builder, scope=Scope.PROTOTYPE)

    Note:
        The default scope for bindings is ``SINGLETON``. Explicitly specify
        the scope when the default is not appropriate.
    """

    SINGLETON = "singleton"
    """One instance per session, created on first access and reused thereafter.

    Best for: Connection pools, caches, configuration objects, and any resource
    that is expensive to create or should be shared across all tool calls.
    If the resource implements ``Closeable``, it is closed when the session ends.
    """

    TOOL_CALL = "tool_call"
    """Fresh instance per tool invocation, disposed after the tool completes.

    Best for: Request-scoped state, transaction contexts, per-operation tracers,
    and resources that require isolation between tool calls. If the resource
    implements ``Closeable``, it is closed immediately after the tool returns.
    """

    PROTOTYPE = "prototype"
    """Fresh instance on every access, never cached or reused.

    Best for: Mutable builders, accumulators, and lightweight objects that must
    not be shared. The container does not track prototype instances, so they
    are not automatically closed (rely on garbage collection or manual cleanup).
    """


__all__ = ["Scope"]
