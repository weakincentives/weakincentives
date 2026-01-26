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

"""Dependency injection with scoped lifecycle management.

This package provides a lightweight dependency injection system for agent
resources with automatic lifecycle management. Resources are registered via
bindings, resolved lazily on first access, and automatically cleaned up when
their scope ends.

Overview
--------

The resources package consists of four main components:

- **Binding**: Describes how to construct a resource and its lifetime scope.
- **Scope**: Enum defining resource lifetimes (SINGLETON, TOOL_CALL, PROTOTYPE).
- **ResourceRegistry**: Immutable container of bindings that creates contexts.
- **ScopedResourceContext**: Runtime context that resolves and caches resources.

Quick Start
-----------

Basic usage with pre-constructed instances::

    from weakincentives.resources import ResourceRegistry

    # Register pre-constructed instances
    config = Config.from_env()
    filesystem = InMemoryFilesystem()

    registry = ResourceRegistry.build({
        Config: config,
        Filesystem: filesystem,
    })

    # Resolve resources within a context
    with registry.open() as ctx:
        cfg = ctx.get(Config)
        fs = ctx.get(Filesystem)
    # Resources cleaned up automatically

Lazy construction with dependencies::

    from weakincentives.resources import Binding, ResourceRegistry, Scope

    registry = ResourceRegistry.of(
        Binding(Config, lambda r: Config.from_env()),
        Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
        Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
    )

    with registry.open() as ctx:
        # Config constructed first, then HTTPClient (which depends on it)
        http = ctx.get(HTTPClient)

        # Tracer is fresh per tool scope
        with ctx.tool_scope() as resolver:
            tracer = resolver.get(Tracer)
        # Tracer.close() called here

Binding Configuration
---------------------

A ``Binding`` associates a protocol type with a provider function and scope.
The provider receives a ``ResourceResolver`` for obtaining dependencies::

    from weakincentives.resources import Binding, Scope

    # Basic binding with dependencies
    http_binding = Binding(
        protocol=HTTPClient,
        provider=lambda r: HTTPClient(
            base_url=r.get(Config).api_url,
            timeout=r.get(Config).timeout,
        ),
        scope=Scope.SINGLETON,  # Default
    )

    # Pre-constructed instance (shorthand)
    config_binding = Binding.instance(Config, config)

    # Eager instantiation (constructed when context starts)
    pool_binding = Binding(
        protocol=ConnectionPool,
        provider=lambda r: ConnectionPool(r.get(Config).db_url),
        eager=True,  # Fail fast on startup if construction fails
    )

Scope Lifetimes
---------------

Three scopes control when resources are constructed and how long they live:

**SINGLETON** (default):
    One instance per session. Created on first access, reused for all
    subsequent requests. Disposed when the context closes.

    Use for: database pools, HTTP clients, shared configuration.

    ::

        Binding(ConnectionPool, make_pool, scope=Scope.SINGLETON)

**TOOL_CALL**:
    Fresh instance per tool invocation. Created when first accessed within
    a ``tool_scope()`` context. Disposed when the tool scope exits.

    Use for: request tracers, per-operation state, audit loggers.

    ::

        Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL)

        with ctx.tool_scope() as resolver:
            tracer1 = resolver.get(Tracer)  # Fresh instance

        with ctx.tool_scope() as resolver:
            tracer2 = resolver.get(Tracer)  # Different instance

        assert tracer1 is not tracer2

**PROTOTYPE**:
    Fresh instance on every access. Never cached. Not tracked for cleanup.

    Use for: builders, transient objects, stateless factories.

    ::

        Binding(QueryBuilder, lambda r: QueryBuilder(), scope=Scope.PROTOTYPE)

        builder1 = ctx.get(QueryBuilder)
        builder2 = ctx.get(QueryBuilder)
        assert builder1 is not builder2

Registry Creation
-----------------

Two factory methods create registries:

**ResourceRegistry.of()** - Preferred for explicit bindings::

    registry = ResourceRegistry.of(
        Binding(Config, lambda r: Config.from_env()),
        Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
        Binding.instance(Filesystem, filesystem),
    )

**ResourceRegistry.build()** - Convenient for instance mapping::

    registry = ResourceRegistry.build({
        Config: config,              # Pre-constructed instance
        Filesystem: filesystem,      # Wrapped in Binding.instance()
        HTTPClient: Binding(HTTPClient, lambda r: HTTPClient(...)),
    })

Registries can be merged (other takes precedence on conflicts)::

    base = ResourceRegistry.build({Filesystem: default_fs})
    override = ResourceRegistry.build({Filesystem: custom_fs})

    # Strict mode (default) raises on conflicts
    merged = base.merge(override, strict=False)  # Uses custom_fs

Lifecycle Protocols
-------------------

Resources can implement protocols for lifecycle hooks:

**Closeable**:
    Resources with ``close()`` are cleaned up when their scope ends::

        class ConnectionPool(Closeable):
            def close(self) -> None:
                for conn in self._connections:
                    conn.close()

**PostConstruct**:
    Called after construction, before caching. Use for initialization that
    requires the instance to exist first (e.g., validation, callbacks)::

        class DatabaseClient(PostConstruct):
            def post_construct(self) -> None:
                self._pool = create_pool(self._config.url)
                self._pool.verify_connection()  # Fail fast if unreachable

**Snapshotable**:
    Resources can capture and restore state for transactional operations::

        class InMemoryFilesystem(Snapshotable[FileSystemSnapshot]):
            def snapshot(self, *, tag: str | None = None) -> FileSystemSnapshot:
                return FileSystemSnapshot(files=dict(self._files))

            def restore(self, snapshot: FileSystemSnapshot) -> None:
                self._files = dict(snapshot.files)

Error Handling
--------------

The package defines a hierarchy of errors for resolution failures:

- **ResourceError**: Base class for all resource errors.
- **UnboundResourceError**: No binding exists for the requested protocol.
- **CircularDependencyError**: Dependency cycle detected (A -> B -> A).
- **DuplicateBindingError**: Same protocol bound multiple times.
- **ProviderError**: Provider function raised an exception.

Example error handling::

    from weakincentives.resources import (
        ResourceRegistry,
        UnboundResourceError,
        CircularDependencyError,
    )

    registry = ResourceRegistry.build({Config: config})

    with registry.open() as ctx:
        try:
            service = ctx.get(UnregisteredService)
        except UnboundResourceError as e:
            print(f"Missing binding: {e.protocol.__name__}")

Resolution Context
------------------

The ``ScopedResourceContext`` provides the runtime resolution interface:

- ``get(protocol)`` - Resolve and return the resource (raises if unbound).
- ``get_optional(protocol)`` - Return None if unbound.
- ``tool_scope()`` - Context manager for TOOL_CALL scoped resources.

Resolution order:

1. Check scope-specific cache (singleton or tool-call)
2. Detect circular dependencies
3. Invoke provider function
4. Call ``post_construct()`` if implemented
5. Cache per scope (except PROTOTYPE)

Resources are closed in reverse instantiation order when the context exits,
ensuring proper cleanup of dependent resources.

Module Structure
----------------

- ``binding.py``: Binding and Provider types
- ``context.py``: ScopedResourceContext implementation
- ``errors.py``: Error hierarchy (ResourceError and subclasses)
- ``protocols.py``: Closeable, PostConstruct, Snapshotable, ResourceResolver
- ``registry.py``: ResourceRegistry configuration container
- ``scope.py``: Scope enum (SINGLETON, TOOL_CALL, PROTOTYPE)
"""

from __future__ import annotations

from .binding import Binding, Provider
from .context import ScopedResourceContext
from .errors import (
    CircularDependencyError,
    DuplicateBindingError,
    ProviderError,
    ResourceError,
    UnboundResourceError,
)
from .protocols import Closeable, PostConstruct, ResourceResolver, Snapshotable
from .registry import ResourceRegistry
from .scope import Scope

__all__ = [
    "Binding",
    "CircularDependencyError",
    "Closeable",
    "DuplicateBindingError",
    "PostConstruct",
    "Provider",
    "ProviderError",
    "ResourceError",
    "ResourceRegistry",
    "ResourceResolver",
    "Scope",
    "ScopedResourceContext",
    "Snapshotable",
    "UnboundResourceError",
]
