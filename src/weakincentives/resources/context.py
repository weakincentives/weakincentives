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

"""Scoped resource resolution context."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, cast

from ..dbc import enters, in_state, state_machine, transition
from ..runtime.logging import StructuredLogger, get_logger
from .binding import Binding
from .errors import CircularDependencyError, ProviderError, UnboundResourceError
from .protocols import Closeable, PostConstruct, ResourceResolver
from .scope import Scope

if TYPE_CHECKING:
    from .registry import ResourceRegistry

logger: StructuredLogger = get_logger(__name__, context={"component": "resources"})


class ContextState(Enum):
    """Lifecycle states for ScopedResourceContext."""

    CREATED = auto()
    """Initial state after construction."""

    STARTED = auto()
    """Context has been started; resources can be resolved."""

    CLOSED = auto()
    """Context has been closed; no further operations allowed."""


@state_machine(state_var="_state", states=ContextState, initial=ContextState.CREATED)
@dataclass(slots=True)
class ScopedResourceContext:
    """Scoped resolution context with lifecycle management.

    Manages resource construction, caching per scope, and cleanup.
    Use ``tool_scope()`` to enter per-tool-call scopes.

    Lifecycle states (enforced via @state_machine when DbC is active):
        - CREATED: Initial state after construction
        - STARTED: After start() called; resources can be resolved
        - CLOSED: After close() called; no further operations allowed

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

    _state: ContextState = field(default=ContextState.CREATED, init=False, repr=False)
    """Current lifecycle state (managed by @state_machine decorator)."""

    @in_state(ContextState.STARTED)
    def get[T](self, protocol: type[T]) -> T:
        """Resolve and return resource for protocol.

        Resolution order:
        1. Scope-specific cache (singleton or tool-call)
        2. Invoke provider and cache per scope

        Raises:
            UnboundResourceError: No binding exists.
            CircularDependencyError: Dependency cycle detected.
            ProviderError: Provider raised an exception.
            InvalidStateError: Context not in STARTED state (DbC mode).
        """
        return self._get_internal(protocol)

    @in_state(ContextState.STARTED)
    def get_optional[T](self, protocol: type[T]) -> T | None:
        """Resolve if bound, return None otherwise.

        Raises:
            InvalidStateError: Context not in STARTED state (DbC mode).
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

    @transition(from_=ContextState.CREATED, to=ContextState.STARTED)
    def start(self) -> None:
        """Initialize context and instantiate eager singletons.

        Call before first use to fail fast on configuration errors.

        Raises:
            InvalidStateError: Context not in CREATED state (DbC mode).
        """
        for binding in self.registry.eager_bindings():
            _ = self._get_internal(binding.protocol)

    def _get_internal[T](self, protocol: type[T]) -> T:
        """Internal get that bypasses state check (for use during start)."""
        binding = self.registry.binding_for(protocol)
        if binding is None:
            raise UnboundResourceError(protocol)

        cache = self._cache_for_scope(binding.scope)
        if cache is not None and protocol in cache:
            return cast(T, cache[protocol])

        if protocol in self._resolving:
            cycle = (*self._resolving, protocol)
            raise CircularDependencyError(cycle)

        self._resolving.add(protocol)
        try:
            constructed = self._construct(binding)
        finally:
            self._resolving.discard(protocol)

        if cache is not None:
            cache[protocol] = constructed
            self._instantiation_order.append((binding.scope, protocol))

        return constructed

    @enters(ContextState.CLOSED)
    def close(self) -> None:
        """Dispose all instantiated resources implementing Closeable.

        Resources are closed in reverse instantiation order.
        Only SINGLETON and TOOL_CALL scoped resources are tracked and closed;
        PROTOTYPE resources are not cached and thus not tracked.

        Can be called from any state (idempotent). After close(), no further
        operations are allowed.
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

    @in_state(ContextState.STARTED)
    @contextmanager
    def tool_scope(self) -> Iterator[ResourceResolver]:
        """Enter a tool-call scope.

        Resources with TOOL_CALL scope are fresh within this context
        and disposed on exit.

        Example::

            with ctx.tool_scope() as resolver:
                tracer = resolver.get(Tracer)  # Fresh instance
                # ... use tracer ...
            # tracer.close() called automatically

        Raises:
            InvalidStateError: Context not in STARTED state (DbC mode).
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


__all__ = ["ContextState", "ScopedResourceContext"]
