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

"""Resource module protocol and builder for accumulating bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .binding import Binding, Provider
from .errors import DuplicateBindingError
from .scope import Scope

if TYPE_CHECKING:
    from .registry import ResourceRegistry


@runtime_checkable
class ResourceModule(Protocol):
    """A unit of resource configuration that contributes bindings.

    Modules group related bindings into reusable, composable units.
    Any object with a ``configure`` method satisfying this protocol
    can serve as a module.

    Example::

        class WorkspaceModule:
            def __init__(self, filesystem: Filesystem) -> None:
                self._filesystem = filesystem

            def configure(self, builder: RegistryBuilder) -> None:
                builder.bind_instance(Filesystem, self._filesystem)

        class ObservabilityModule:
            def configure(self, builder: RegistryBuilder) -> None:
                builder.bind(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL)
                builder.bind(MetricsCollector, lambda r: MetricsCollector())

        # Compose modules
        registry = RegistryBuilder.from_modules(
            WorkspaceModule(fs),
            ObservabilityModule(),
        )
    """

    def configure(self, builder: RegistryBuilder) -> None:
        """Contribute bindings to the builder.

        Called once during registry construction. Implementations should
        use the builder's ``bind``, ``bind_instance``, and ``install``
        methods to register resources.

        Args:
            builder: The builder to contribute bindings to.
        """
        ...


class RegistryBuilder:
    """Accumulates bindings from modules and builds an immutable registry.

    The builder collects bindings contributed by modules and produces
    a ``ResourceRegistry``. Later bindings for the same protocol override
    earlier ones (last-write-wins), unless ``strict`` mode is used.

    Example::

        builder = RegistryBuilder()
        builder.bind_instance(Config, config)
        builder.bind(HTTPClient, lambda r: HTTPClient(r.get(Config).url))
        builder.install(ObservabilityModule())
        registry = builder.build()

    Or use the convenience factory::

        registry = RegistryBuilder.from_modules(
            WorkspaceModule(fs),
            ObservabilityModule(),
        )
    """

    __slots__ = ("_bindings", "_strict")

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, strict: bool = False
    ) -> None:
        """Initialize builder.

        Args:
            strict: If True, raise ``DuplicateBindingError`` when the same
                protocol is bound more than once. Default is False
                (last-write-wins).
        """
        self._bindings: dict[type[object], Binding[object]] = {}
        self._strict = strict

    def bind[T](
        self,
        protocol: type[T],
        provider: Provider[T],
        *,
        scope: Scope = Scope.SINGLETON,
        eager: bool = False,
    ) -> None:
        """Register a factory binding.

        Args:
            protocol: The protocol type this binding satisfies.
            provider: Factory function that constructs the instance.
            scope: Lifetime of constructed instances (default: SINGLETON).
            eager: If True, instantiate during context startup.

        Raises:
            DuplicateBindingError: If strict mode and protocol already bound.
        """
        self._add(Binding(protocol, provider, scope=scope, eager=eager))

    def bind_instance[T](self, protocol: type[T], instance: T) -> None:
        """Register a pre-constructed instance as an eager singleton.

        Args:
            protocol: The protocol type this binding satisfies.
            instance: The pre-constructed instance to return.

        Raises:
            DuplicateBindingError: If strict mode and protocol already bound.
        """
        self._add(Binding.instance(protocol, instance))

    def install(self, module: ResourceModule) -> None:
        """Install a module, incorporating its bindings into this builder.

        The module's ``configure()`` method is called with this builder.

        Args:
            module: The module to install.
        """
        module.configure(self)

    def build(self) -> ResourceRegistry:
        """Build an immutable registry from accumulated bindings.

        Returns:
            New ``ResourceRegistry`` containing all registered bindings.
        """
        from .registry import ResourceRegistry

        return ResourceRegistry.of(*self._bindings.values())

    @staticmethod
    def from_modules(*modules: ResourceModule) -> ResourceRegistry:
        """Create a registry from one or more modules.

        Convenience factory that creates a builder, installs all modules,
        and returns the built registry.

        Args:
            *modules: Modules to install.

        Returns:
            New ``ResourceRegistry`` with bindings from all modules.
        """
        builder = RegistryBuilder()
        for module in modules:
            builder.install(module)
        return builder.build()

    def _add(self, binding: Binding[object]) -> None:
        """Add a binding, enforcing strict mode if enabled."""
        if self._strict and binding.protocol in self._bindings:
            raise DuplicateBindingError(binding.protocol)
        self._bindings[binding.protocol] = binding


__all__ = ["RegistryBuilder", "ResourceModule"]
