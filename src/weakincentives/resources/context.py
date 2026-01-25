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

# pyright: reportImportCycles=false

"""Scoped resource resolution context.

This module provides ``ScopedResourceContext``, the runtime component responsible
for resolving resources from a ``ResourceRegistry``. It manages instance caching
per scope, lifecycle hooks (``PostConstruct``, ``Closeable``), and cleanup.

Typical usage is via ``registry.open()`` which returns a context manager that
handles ``start()`` and ``close()`` automatically.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from ..runtime.logging import StructuredLogger, get_logger
from .binding import Binding
from .errors import CircularDependencyError, ProviderError, UnboundResourceError
from .protocols import Closeable, PostConstruct, ResourceResolver
from .scope import Scope

if TYPE_CHECKING:
    from .registry import ResourceRegistry

logger: StructuredLogger = get_logger(__name__, context={"component": "resources"})


@dataclass(slots=True)
class ScopedResourceContext:
    """Scoped resolution context with lifecycle management.

    Manages resource construction, caching per scope, and cleanup. Implements
    the ``ResourceResolver`` protocol, allowing it to be passed to providers
    that need to resolve their own dependencies.

    Scope Behavior:
        - ``SINGLETON``: One instance per context, cached for lifetime.
        - ``TOOL_CALL``: Fresh instance per ``tool_scope()`` block, disposed on exit.
        - ``PROTOTYPE``: New instance on every ``get()`` call, never cached.

    Typical Usage:
        Use ``registry.open()`` rather than constructing directly. The context
        manager handles ``start()`` and ``close()`` automatically.

    Example::

        # Use registry.open() context manager (handles start/close automatically)
        with registry.open() as ctx:
            # Singleton resources are cached
            http1 = ctx.get(HTTPClient)
            http2 = ctx.get(HTTPClient)
            assert http1 is http2

            # Tool-call scoped resources are fresh per scope
            with ctx.tool_scope() as resolver:
                tracer1 = resolver.get(Tracer)

            with ctx.tool_scope() as resolver:
                tracer2 = resolver.get(Tracer)

            assert tracer1 is not tracer2
        # Cleanup all instantiated resources on exit
    """

    registry: ResourceRegistry
    """Immutable registry configuration."""

    singleton_cache: dict[type[object], object] = field(
        default_factory=lambda: dict[type[object], object]()
    )
    """Cache for SINGLETON-scoped resources."""

    _tool_call_cache: dict[type[object], object] = field(
        default_factory=lambda: dict[type[object], object]()
    )
    """Cache for TOOL_CALL-scoped resources (cleared per scope)."""

    _resolving: set[type[object]] = field(default_factory=lambda: set[type[object]]())
    """Tracks in-flight resolutions for cycle detection."""

    _instantiation_order: list[tuple[Scope, type[object]]] = field(
        default_factory=lambda: list[tuple[Scope, type[object]]]()
    )
    """Tracks instantiation order for cleanup."""

    def get[T](self, protocol: type[T]) -> T:
        """Resolve and return a resource instance for the given protocol.

        Looks up the binding registered for ``protocol``, constructs the
        instance if not cached, and returns it. Caching behavior depends
        on the binding's scope.

        Args:
            protocol: The protocol (typically a ``typing.Protocol`` or ABC)
                identifying the resource to resolve.

        Returns:
            The resolved resource instance, cast to type ``T``.

        Raises:
            UnboundResourceError: No binding registered for ``protocol``.
            CircularDependencyError: A dependency cycle was detected during
                resolution (e.g., A depends on B, B depends on A).
            ProviderError: The provider factory raised an exception during
                construction or ``post_construct()`` failed.

        Resolution Order:
            1. Check scope-specific cache (singleton or tool-call).
            2. If not cached, invoke the provider and cache per scope.

        Example::

            http_client = ctx.get(HTTPClient)  # Returns cached or new instance
        """
        binding = self.registry.binding_for(protocol)
        if binding is None:
            raise UnboundResourceError(protocol)

        # Check caches
        cache = self._cache_for_scope(binding.scope)
        if cache is not None and protocol in cache:
            return cast(T, cache[protocol])

        # Cycle detection
        if protocol in self._resolving:
            cycle = (*self._resolving, protocol)
            raise CircularDependencyError(cycle)

        # Invoke provider
        self._resolving.add(protocol)
        try:
            constructed = self._construct(binding)
        finally:
            self._resolving.discard(protocol)

        # Cache per scope
        if cache is not None:
            cache[protocol] = constructed
            self._instantiation_order.append((binding.scope, protocol))

        return constructed

    def get_optional[T](self, protocol: type[T]) -> T | None:
        """Resolve a resource if bound, otherwise return None.

        Unlike ``get()``, this method does not raise ``UnboundResourceError``
        when no binding exists. Other resolution errors (circular dependencies,
        provider failures) are still raised.

        Args:
            protocol: The protocol identifying the resource to resolve.

        Returns:
            The resolved resource instance, or ``None`` if no binding exists.

        Raises:
            CircularDependencyError: A dependency cycle was detected.
            ProviderError: The provider factory raised an exception.

        Example::

            # Conditionally use a resource if available
            tracer = ctx.get_optional(Tracer)
            if tracer is not None:
                tracer.log("operation started")
        """
        if protocol not in self.registry:
            return None
        return self.get(protocol)

    def _construct[T](self, binding: Binding[T]) -> T:
        """Construct instance via provider, handling lifecycle hooks."""
        logger.debug(
            "resource.construct.start",
            event="resource.construct.start",
            context={
                "protocol": binding.protocol.__qualname__,
                "scope": binding.scope.name,
            },
        )
        try:
            instance = binding.provider(self)
        except (CircularDependencyError, UnboundResourceError, ProviderError):
            # Re-raise resolution errors without wrapping
            raise
        except Exception as e:
            raise ProviderError(binding.protocol, e) from e

        # Post-construct hook
        if isinstance(instance, PostConstruct):
            try:
                instance.post_construct()
            except Exception as e:
                # Cleanup if post_construct fails
                if isinstance(instance, Closeable):
                    try:
                        instance.close()
                    except Exception:
                        logger.warning(
                            "Error closing resource after post_construct failure",
                            event="resource.post_construct_cleanup_error",
                            context={"protocol": binding.protocol.__qualname__},
                        )
                raise ProviderError(binding.protocol, e) from e

        logger.debug(
            "resource.construct.complete",
            event="resource.construct.complete",
            context={
                "protocol": binding.protocol.__qualname__,
                "scope": binding.scope.name,
                "instance_type": type(instance).__qualname__,
            },
        )
        return instance

    def _cache_for_scope(self, scope: Scope) -> dict[type[object], object] | None:
        """Return appropriate cache for scope, or None for PROTOTYPE."""
        if scope == Scope.SINGLETON:
            return self.singleton_cache
        if scope == Scope.TOOL_CALL:
            return self._tool_call_cache
        return None  # PROTOTYPE or unknown scope

    def start(self) -> None:
        """Initialize context and instantiate eager singletons.

        Eagerly resolves all bindings marked with ``eager=True``, causing
        their providers to be invoked immediately. This is useful for:

        - Validating configuration at startup (fail fast).
        - Pre-warming expensive resources like connection pools.
        - Ensuring critical services are available before handling requests.

        This method is called automatically by ``registry.open()``. Only call
        directly if managing the context lifecycle manually.

        Raises:
            UnboundResourceError: An eager binding has an unbound dependency.
            CircularDependencyError: Circular dependency among eager bindings.
            ProviderError: A provider raised an exception during construction.
        """
        for binding in self.registry.eager_bindings():
            _ = self.get(binding.protocol)

    def close(self) -> None:
        """Dispose all instantiated resources implementing ``Closeable``.

        Iterates through all cached resources in reverse instantiation order
        and calls ``close()`` on each that implements the ``Closeable``
        protocol. This ensures proper cleanup of file handles, connections,
        and other system resources.

        Behavior:
            - Resources are closed in reverse instantiation order (LIFO).
            - Only ``SINGLETON`` and ``TOOL_CALL`` scoped resources are tracked.
            - ``PROTOTYPE`` resources are not cached and must be closed manually.
            - Errors during individual ``close()`` calls are logged but do not
              prevent other resources from being closed.
            - All caches are cleared after closing.

        This method is called automatically by ``registry.open()``. Only call
        directly if managing the context lifecycle manually.

        Note:
            After calling ``close()``, the context should not be reused.
            Create a new context via ``registry.open()`` if needed.
        """
        logger.debug(
            "resource.context.close.start",
            event="resource.context.close.start",
            context={"resource_count": len(self._instantiation_order)},
        )
        for scope, protocol in reversed(self._instantiation_order):
            # Only SINGLETON and TOOL_CALL are ever in _instantiation_order
            cache = (
                self.singleton_cache
                if scope == Scope.SINGLETON
                else self._tool_call_cache
            )
            instance = cache.get(protocol)
            if instance is not None and isinstance(instance, Closeable):
                try:
                    logger.debug(
                        "resource.close",
                        event="resource.close",
                        context={
                            "protocol": protocol.__qualname__,
                            "scope": scope.name,
                        },
                    )
                    instance.close()
                except Exception:
                    logger.warning(
                        "Error closing resource",
                        event="resource.close_error",
                        context={"protocol": protocol.__qualname__},
                    )
        self._instantiation_order.clear()
        self.singleton_cache.clear()
        self._tool_call_cache.clear()
        logger.debug(
            "resource.context.close.complete",
            event="resource.context.close.complete",
        )

    @contextmanager
    def tool_scope(self) -> Iterator[ResourceResolver]:
        """Enter a tool-call scope for isolated resource resolution.

        Creates an isolated scope where ``TOOL_CALL``-scoped resources are
        freshly instantiated and automatically disposed when the scope exits.
        This is typically used to provide per-request or per-operation isolation.

        Yields:
            A ``ResourceResolver`` (this context) for resolving resources
            within the tool scope.

        Behavior:
            - ``SINGLETON`` resources remain shared across all scopes.
            - ``TOOL_CALL`` resources get fresh instances within this scope.
            - ``PROTOTYPE`` resources are always fresh (unaffected by scoping).
            - On exit, all ``TOOL_CALL`` resources created in this scope that
              implement ``Closeable`` are closed in reverse instantiation order.
            - Nested ``tool_scope()`` calls are supported; each gets its own
              isolated ``TOOL_CALL`` cache.

        Example::

            with ctx.tool_scope() as resolver:
                tracer = resolver.get(Tracer)  # Fresh instance
                db_conn = resolver.get(DBConnection)  # Fresh instance
                # ... perform tool operation ...
            # tracer.close() and db_conn.close() called automatically

            # Outside the scope, a new tool_scope gets fresh instances
            with ctx.tool_scope() as resolver:
                tracer2 = resolver.get(Tracer)  # Different instance
        """
        logger.debug(
            "resource.tool_scope.enter",
            event="resource.tool_scope.enter",
        )
        # Save and clear tool-call cache
        previous_cache = self._tool_call_cache
        previous_order = [
            (s, p) for s, p in self._instantiation_order if s == Scope.TOOL_CALL
        ]
        self._tool_call_cache = {}
        # Remove tool-call entries from instantiation order
        self._instantiation_order = [
            (s, p) for s, p in self._instantiation_order if s != Scope.TOOL_CALL
        ]

        try:
            yield self
        finally:
            # Close tool-call scoped resources in reverse order
            tool_call_order = [
                (s, p) for s, p in self._instantiation_order if s == Scope.TOOL_CALL
            ]
            for _, protocol in reversed(tool_call_order):
                instance = self._tool_call_cache.get(protocol)
                if instance is not None and isinstance(instance, Closeable):
                    try:
                        instance.close()
                    except Exception:
                        logger.warning(
                            "Error closing tool-scoped resource",
                            event="resource.tool_scope_close_error",
                            context={"protocol": protocol.__qualname__},
                        )

            # Restore previous state
            self._tool_call_cache = previous_cache
            # Restore tool-call entries to instantiation order
            self._instantiation_order = [
                (s, p) for s, p in self._instantiation_order if s != Scope.TOOL_CALL
            ]
            self._instantiation_order.extend(previous_order)
            logger.debug(
                "resource.tool_scope.exit",
                event="resource.tool_scope.exit",
                context={"closed_count": len(tool_call_order)},
            )


__all__ = ["ScopedResourceContext"]
