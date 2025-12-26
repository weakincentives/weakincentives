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
    from .context import ScopedResourceContext


@dataclass(slots=True, frozen=True)
class ResourceRegistry:
    """Immutable configuration of resource bindings.

    ResourceRegistry holds the blueprint for constructing resources but does
    not perform resolution itself. Use ``create_context()`` to obtain a
    ``ScopedResourceContext`` that performs lazy resolution with caching.

    Example::

        from weakincentives.resources import Binding, ResourceRegistry, Scope

        registry = ResourceRegistry.of(
            Binding(Config, lambda r: Config.from_env()),
            Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
            Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
        )

        # Create resolution context
        ctx = registry.create_context()

        # Resolve resources lazily
        http = ctx.get(HTTPClient)  # Constructs Config first, then HTTPClient
    """

    _bindings: Mapping[type[object], Binding[object]] = field(
        default_factory=lambda: MappingProxyType({}),
    )

    @staticmethod
    def of(*bindings: Binding[object]) -> ResourceRegistry:
        """Construct a registry from bindings.

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

    def __contains__(self, protocol: type[object]) -> bool:
        """Check if protocol has a binding."""
        return protocol in self._bindings

    def __len__(self) -> int:
        """Return number of bindings."""
        return len(self._bindings)

    def __iter__(self) -> Iterator[type[object]]:
        """Iterate over bound protocol types."""
        return iter(self._bindings)

    def binding_for[T](self, protocol: type[T]) -> Binding[T] | None:
        """Return the binding for a protocol, or None if unbound."""
        binding = self._bindings.get(protocol)
        return cast(Binding[T], binding) if binding else None

    def merge(self, other: ResourceRegistry) -> ResourceRegistry:
        """Merge registries; other takes precedence on conflicts.

        Example::

            base = ResourceRegistry.of(Binding(Config, default_config))
            override = ResourceRegistry.of(Binding(Config, test_config))
            merged = base.merge(override)  # Uses test_config
        """
        merged = dict(self._bindings)
        merged.update(other._bindings)
        return ResourceRegistry(_bindings=MappingProxyType(merged))

    def eager_bindings(self) -> Sequence[Binding[object]]:
        """Return all bindings marked as eager."""
        return [b for b in self._bindings.values() if b.eager]

    def create_context(
        self,
        *,
        singleton_cache: dict[type[object], object] | None = None,
    ) -> ScopedResourceContext:
        """Create a scoped resolution context.

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
