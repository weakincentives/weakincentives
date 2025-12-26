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
    """Immutable configuration of resource bindings and instances.

    ResourceRegistry supports two patterns:

    1. **Provider-based** (lazy construction with scopes)::

        registry = ResourceRegistry.of(
            Binding(Config, lambda r: Config.from_env()),
            Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
        )
        ctx = registry.create_context()
        http = ctx.get(HTTPClient)

    2. **Instance-based** (pre-constructed resources)::

        registry = ResourceRegistry.build({
            Filesystem: InMemoryFilesystem(),
            BudgetTracker: tracker,
        })
        fs = registry.get(Filesystem)

    Both patterns can be combined via ``merge()``.
    """

    _bindings: Mapping[type[object], Binding[object]] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    """Provider bindings for lazy construction."""

    _instances: Mapping[type[object], object] = field(
        default_factory=lambda: MappingProxyType({}),
    )
    """Pre-constructed instances for direct access."""

    # === Factory Methods ===

    @staticmethod
    def of(*bindings: Binding[object]) -> ResourceRegistry:
        """Construct a registry from provider bindings.

        Args:
            *bindings: Variable number of Binding instances.

        Returns:
            New registry containing all bindings.

        Raises:
            DuplicateBindingError: Same protocol bound twice.

        Example::

            registry = ResourceRegistry.of(
                Binding(Config, make_config),
                Binding(Database, make_database),
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
        """Construct a registry from pre-constructed instances.

        Use protocol types as keys to enable protocol-based lookup::

            registry = ResourceRegistry.build({
                Filesystem: InMemoryFilesystem(),
                HTTPClient: MyHTTPClient(),
            })
            fs = registry.get(Filesystem)

        Args:
            mapping: Type-to-instance mapping. None values are filtered out.

        Returns:
            New registry with direct instance access.
        """
        filtered = {k: v for k, v in mapping.items() if v is not None}
        return ResourceRegistry(_instances=MappingProxyType(filtered))

    # === Query Methods ===

    def __contains__(self, protocol: type[object]) -> bool:
        """Check if protocol has a binding or instance."""
        return protocol in self._bindings or protocol in self._instances

    def __len__(self) -> int:
        """Return number of bindings plus instances."""
        return len(self._bindings) + len(self._instances)

    def __iter__(self) -> Iterator[type[object]]:
        """Iterate over all bound protocol types."""
        seen: set[type[object]] = set()
        for protocol in self._bindings:
            seen.add(protocol)
            yield protocol
        for protocol in self._instances:
            if protocol not in seen:
                yield protocol

    def get[T](self, protocol: type[T], default: T | None = None) -> T | None:
        """Return pre-constructed instance, or default if absent.

        This method only returns from ``_instances`` (pre-constructed).
        For provider-based bindings, use ``create_context().get()``.

        Args:
            protocol: The protocol type to look up.
            default: Value to return if not found.

        Returns:
            The instance if found, otherwise default.
        """
        value = self._instances.get(protocol)
        if value is None:
            return default
        return cast(T, value)

    def binding_for[T](self, protocol: type[T]) -> Binding[T] | None:
        """Return the binding for a protocol, or None if unbound."""
        binding = self._bindings.get(protocol)
        return cast(Binding[T], binding) if binding else None

    def snapshotable_resources(self) -> Mapping[type[object], Snapshotable]:
        """Return all instances that implement Snapshotable.

        Returns:
            Mapping from resource type to snapshotable resource instance.
            Only pre-constructed instances that implement the Snapshotable
            protocol are included.
        """
        from ..runtime.snapshotable import Snapshotable

        return {k: v for k, v in self._instances.items() if isinstance(v, Snapshotable)}

    # === Composition ===

    def merge(self, other: ResourceRegistry) -> ResourceRegistry:
        """Merge registries; other takes precedence on conflicts.

        Both bindings and instances are merged, with ``other`` winning
        on conflicts within each category.

        Example::

            base = ResourceRegistry.build({Filesystem: default_fs})
            override = ResourceRegistry.build({Filesystem: custom_fs})
            merged = base.merge(override)  # Uses custom_fs
        """
        merged_bindings = dict(self._bindings)
        merged_bindings.update(other._bindings)
        merged_instances = dict(self._instances)
        merged_instances.update(other._instances)
        return ResourceRegistry(
            _bindings=MappingProxyType(merged_bindings),
            _instances=MappingProxyType(merged_instances),
        )

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

        The context resolves provider bindings lazily with caching.
        Pre-constructed instances from ``_instances`` are available
        directly without provider invocation.

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
