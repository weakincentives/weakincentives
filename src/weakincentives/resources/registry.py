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

"""Immutable resource registry configuration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from .binding import Binding
from .errors import DuplicateBindingError

if TYPE_CHECKING:
    from .context import ScopedResourceContext


@dataclass(slots=True, frozen=True)
class ResourceRegistry:
    """Immutable configuration of resource bindings.

    Resources are registered via ``of()`` (for explicit bindings) or
    ``build()`` (for a type-to-instance mapping).

    Example::

        from weakincentives.resources import Binding, ResourceRegistry, Scope

        # Pre-constructed instances (common case)
        registry = ResourceRegistry.build({
            Filesystem: InMemoryFilesystem(),
            BudgetTracker: tracker,
        })

        # Lazy construction with dependencies (use .of() with explicit bindings)
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: Config.from_env()),
            Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
        )

        # Tool-call scoped resources (fresh per tool invocation)
        registry = ResourceRegistry.of(
            Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
        )

        # Access via context manager
        with registry.open() as ctx:
            http = ctx.get(HTTPClient)
    """

    _bindings: Mapping[type[object], Binding[object]] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    """Provider bindings for lazy construction."""

    # === Factory Methods ===

    @staticmethod
    def of(*bindings: Binding[object]) -> ResourceRegistry:
        """Create a registry from explicit bindings.

        Each binding specifies its own protocol, so the caller doesn't repeat
        the type in a mapping key. This is the preferred API when constructing
        bindings with custom scopes or dependencies.

        Args:
            *bindings: Binding objects with explicit protocols.

        Returns:
            New registry with the provided bindings.

        Raises:
            DuplicateBindingError: Same protocol bound twice.

        Example::

            from weakincentives.resources import Binding, ResourceRegistry, Scope

            registry = ResourceRegistry.of(
                Binding(Config, lambda r: Config.from_env()),
                Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
                Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
            )

            # Pre-constructed instances use Binding.instance()
            registry = ResourceRegistry.of(
                Binding.instance(Config, config),
                Binding.instance(Filesystem, filesystem),
            )
        """
        mapping: dict[type[object], Binding[object]] = {}
        for binding in bindings:
            if binding.protocol in mapping:
                raise DuplicateBindingError(binding.protocol)
            mapping[binding.protocol] = binding
        return ResourceRegistry(_bindings=MappingProxyType(mapping))

    @staticmethod
    def build(
        mapping: Mapping[type[object], object | Binding[object]],
    ) -> ResourceRegistry:
        """Create a registry from a type-to-resource mapping.

        Values can be either:
        - Pre-constructed instances (wrapped automatically)
        - Binding objects for lazy construction or custom scopes

        Args:
            mapping: Type-to-resource mapping. None values are filtered out.

        Returns:
            New registry with bindings for each entry.

        Raises:
            DuplicateBindingError: Same protocol bound twice.

        Example::

            # Simple instances
            registry = ResourceRegistry.build({
                Filesystem: fs,
                BudgetTracker: tracker,
            })

            # Mixed instances and bindings (protocol required in Binding)
            registry = ResourceRegistry.build({
                Config: config,  # Instance
                HTTPClient: Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
                Tracer: Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
            })

            # For bindings, prefer ResourceRegistry.of() to avoid repeating types
            registry = ResourceRegistry.of(
                Binding.instance(Config, config),
                Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
                Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
            )
        """
        entries: dict[type[object], Binding[object]] = {}
        for protocol, value in mapping.items():
            if value is None:
                continue
            if protocol in entries:
                raise DuplicateBindingError(protocol)
            if isinstance(value, Binding):
                entries[protocol] = value
            else:
                entries[protocol] = Binding.instance(protocol, value)
        return ResourceRegistry(_bindings=MappingProxyType(entries))

    # === Query Methods ===

    def __contains__(self, protocol: type[object]) -> bool:
        """Check if protocol has a binding."""
        return protocol in self._bindings

    def __len__(self) -> int:
        """Return number of bindings."""
        return len(self._bindings)

    def __iter__(self) -> Iterator[type[object]]:
        """Iterate over all bound protocol types."""
        yield from self._bindings

    def binding_for[T](self, protocol: type[T]) -> Binding[T] | None:
        """Return the binding for a protocol, or None if unbound."""
        binding = self._bindings.get(protocol)
        return cast(Binding[T], binding) if binding else None

    # === Composition ===

    def merge(
        self, other: ResourceRegistry, *, strict: bool = True
    ) -> ResourceRegistry:
        """Merge registries; other takes precedence on conflicts.

        Args:
            other: Registry to merge with.
            strict: If True (default), raise DuplicateBindingError on conflicts.
                    If False, other's bindings silently override.

        Returns:
            New registry with merged bindings.

        Raises:
            DuplicateBindingError: If strict=True (default) and registries
                share protocols.

        Example::

            base = ResourceRegistry.build({Filesystem: default_fs})
            override = ResourceRegistry.build({Filesystem: custom_fs})

            # Default strict=True detects conflicts
            base.merge(override)  # Raises DuplicateBindingError

            # Use strict=False for intentional overrides
            merged = base.merge(override, strict=False)  # Uses custom_fs
        """
        if strict:
            conflicts = self.conflicts(other)
            if conflicts:
                # Report first conflict for clarity
                raise DuplicateBindingError(next(iter(conflicts)))

        merged_bindings = dict(self._bindings)
        merged_bindings.update(other._bindings)
        return ResourceRegistry(_bindings=MappingProxyType(merged_bindings))

    def conflicts(self, other: ResourceRegistry) -> frozenset[type[object]]:
        """Return protocols bound in both registries.

        Useful for debugging merge conflicts without enabling strict mode.

        Example::

            conflicts = base.conflicts(override)
            if conflicts:
                print(f"Warning: {len(conflicts)} bindings will be overridden")
        """
        return frozenset(self._bindings.keys() & other._bindings.keys())

    # === Provider-Based Resolution ===

    def eager_bindings(self) -> Sequence[Binding[object]]:
        """Return all bindings marked as eager."""
        return [b for b in self._bindings.values() if b.eager]

    def _create_context(
        self,
        *,
        singleton_cache: dict[type[object], object] | None = None,
    ) -> ScopedResourceContext:
        """Create a scoped resolution context (internal).

        Args:
            singleton_cache: Shared cache for SINGLETON scope.

        Returns:
            Context that supports lazy resolution with scope awareness.
        """
        from .context import ScopedResourceContext

        return ScopedResourceContext(
            registry=self,
            singleton_cache=singleton_cache if singleton_cache is not None else {},
        )

    @contextmanager
    def open(self) -> Iterator[ScopedResourceContext]:
        """Context manager for resource lifecycle.

        Creates a context, starts it, and ensures cleanup on exit.

        Example::

            registry = ResourceRegistry.build({Config: config})

            with registry.open() as ctx:
                service = ctx.get(MyService)
            # Resources cleaned up automatically

        Yields:
            Started ScopedResourceContext for resolving resources.
        """
        ctx = self._create_context()
        ctx.start()
        try:
            yield ctx
        finally:
            ctx.close()


__all__ = ["ResourceRegistry"]
