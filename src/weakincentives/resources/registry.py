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
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from .binding import Binding
from .errors import DuplicateBindingError

if TYPE_CHECKING:
    from ..runtime.snapshotable import Snapshotable
    from .context import ScopedResourceContext


@dataclass(slots=True, frozen=True)
class ResourceRegistry:
    """Immutable configuration of resource bindings.

    All resources are registered as bindings. Pre-constructed instances
    are wrapped using ``Binding.instance()`` internally.

    Example::

        from weakincentives.resources import Binding, ResourceRegistry

        # Provider-based bindings (lazy construction)
        registry = ResourceRegistry.of(
            Binding(Config, lambda r: Config.from_env()),
            Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
        )

        # Pre-constructed instances (via Binding.instance)
        registry = ResourceRegistry.of(
            Binding.instance(Filesystem, InMemoryFilesystem()),
            Binding.instance(BudgetTracker, tracker),
        )

        # Or use the convenience method
        registry = ResourceRegistry.build({
            Filesystem: InMemoryFilesystem(),
            BudgetTracker: tracker,
        })

        # Access via context
        ctx = registry.create_context()
        http = ctx.get(HTTPClient)
    """

    _bindings: Mapping[type[object], Binding[object]] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    """Provider bindings for lazy construction."""

    # === Factory Methods ===

    @staticmethod
    def of(*bindings: Binding[object]) -> ResourceRegistry:
        """Construct a registry from provider bindings.

        This is the canonical way to create a registry. Use ``Binding.instance()``
        to wrap pre-constructed instances.

        Args:
            *bindings: Variable number of Binding instances.

        Returns:
            New registry containing all bindings.

        Raises:
            DuplicateBindingError: Same protocol bound twice.

        Example::

            registry = ResourceRegistry.of(
                Binding(Config, make_config),
                Binding.instance(Filesystem, fs),
            )
        """
        entries: dict[type[object], Binding[object]] = {}
        for binding in bindings:
            if binding.protocol in entries:
                raise DuplicateBindingError(binding.protocol)
            entries[binding.protocol] = binding
        return ResourceRegistry(_bindings=MappingProxyType(entries))

    @staticmethod
    def build(mapping: Mapping[type[object], object]) -> ResourceRegistry:
        """Convenience method to create a registry from pre-constructed instances.

        Equivalent to calling ``of()`` with ``Binding.instance()`` for each entry.

        Example::

            # These are equivalent:
            registry = ResourceRegistry.build({
                Filesystem: fs,
                BudgetTracker: tracker,
            })

            registry = ResourceRegistry.of(
                Binding.instance(Filesystem, fs),
                Binding.instance(BudgetTracker, tracker),
            )

        Args:
            mapping: Type-to-instance mapping. None values are filtered out.

        Returns:
            New registry with bindings for each instance.
        """
        bindings = [Binding.instance(k, v) for k, v in mapping.items() if v is not None]
        return ResourceRegistry.of(*bindings)

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

    def get[T](self, protocol: type[T], default: T | None = None) -> T | None:
        """Return the resource for the given protocol, or default if absent.

        For pre-constructed instances (via ``Binding.instance()``), returns
        the instance directly. For provider-based bindings, resolves via
        a temporary context.

        Args:
            protocol: The protocol type to look up.
            default: Value to return if not found.

        Returns:
            The resource instance if found, otherwise default.
        """
        binding = self._bindings.get(protocol)
        if binding is None:
            return default
        # For instance bindings, return directly
        if binding.preconstructed is not None:
            return cast(T, binding.preconstructed)
        # For provider bindings, resolve via context
        ctx = self.create_context()
        return ctx.get(protocol)

    def snapshotable_resources(self) -> Mapping[type[object], Snapshotable]:
        """Return all pre-constructed instances that implement Snapshotable.

        Only bindings created via ``Binding.instance()`` are checked.
        Provider-based bindings are not introspected.

        Returns:
            Mapping from resource type to snapshotable resource instance.
        """
        from ..runtime.snapshotable import Snapshotable

        result: dict[type[object], Snapshotable] = {}
        for protocol, binding in self._bindings.items():
            instance = binding.preconstructed
            if instance is not None and isinstance(instance, Snapshotable):
                result[protocol] = instance
        return result

    # === Composition ===

    def merge(self, other: ResourceRegistry) -> ResourceRegistry:
        """Merge registries; other takes precedence on conflicts.

        Example::

            base = ResourceRegistry.build({Filesystem: default_fs})
            override = ResourceRegistry.build({Filesystem: custom_fs})
            merged = base.merge(override)  # Uses custom_fs
        """
        merged_bindings = dict(self._bindings)
        merged_bindings.update(other._bindings)
        return ResourceRegistry(_bindings=MappingProxyType(merged_bindings))

    # === Provider-Based Resolution ===

    def eager_bindings(self) -> Sequence[Binding[object]]:
        """Return all bindings marked as eager."""
        return [b for b in self._bindings.values() if b.eager]

    def create_context(
        self,
        *,
        singleton_cache: dict[type[object], object] | None = None,
    ) -> ScopedResourceContext:
        """Create a scoped resolution context.

        The context resolves bindings lazily with caching per scope.

        Args:
            singleton_cache: Shared cache for SINGLETON scope. If None,
                creates a new cache (typical for session start).

        Returns:
            Context that supports lazy resolution with scope awareness.

        Example::

            ctx = registry.create_context()
            ctx.start()  # Instantiate eager singletons
            try:
                service = ctx.get(MyService)
            finally:
                ctx.close()  # Cleanup all resources
        """
        from .context import ScopedResourceContext

        return ScopedResourceContext(
            registry=self,
            singleton_cache=singleton_cache if singleton_cache is not None else {},
        )


__all__ = ["ResourceRegistry"]
