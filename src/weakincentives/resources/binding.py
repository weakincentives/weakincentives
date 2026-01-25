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

"""Resource binding configuration for dependency injection.

This module provides the ``Binding`` class, which associates a protocol type
with a factory function (provider) and a lifecycle scope. Bindings are
registered with a ``ResourceRegistry`` and resolved through a
``ResourceContext``.

Typical usage::

    from weakincentives.resources import Binding, ResourceRegistry, Scope

    registry = ResourceRegistry.of(
        Binding(Config, lambda r: Config.from_env(), scope=Scope.SINGLETON),
        Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
    )

    with registry.context() as ctx:
        client = ctx.get(HTTPClient)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .protocols import ResourceResolver
from .scope import Scope

type Provider[T] = Callable[[ResourceResolver], T]
"""Factory function that constructs a resource given a resolver for dependencies.

A provider receives a ``ResourceResolver`` that can be used to obtain other
dependencies via ``resolver.get(ProtocolType)``. Providers are called lazily
on first access to the resource.

Example::

    def create_service(resolver: ResourceResolver) -> MyService:
        config = resolver.get(Config)
        db = resolver.get(Database)
        return MyService(config, db)

    binding = Binding(MyService, create_service)
"""


@dataclass(slots=True, frozen=True)
class Binding[T]:
    """Describes how to construct a resource and manage its lifetime.

    A binding associates a protocol type with a provider function and scope.
    The provider is called lazily on first access, with a resolver for
    obtaining dependencies.

    Scopes:
        - ``Scope.SINGLETON``: One instance per context, cached for reuse.
        - ``Scope.TOOL_CALL``: New instance per tool call, disposed after.
        - ``Scope.PROTOTYPE``: New instance on every ``get()`` call.

    Example::

        from weakincentives.resources import Binding, Scope

        # SINGLETON: shared instance, created once
        config_binding = Binding(
            protocol=Config,
            provider=lambda r: Config.from_env(),
            scope=Scope.SINGLETON,
        )

        # Service depending on Config
        http_binding = Binding(
            protocol=HTTPClient,
            provider=lambda r: HTTPClient(
                base_url=r.get(Config).api_url,
                timeout=r.get(Config).timeout,
            ),
            scope=Scope.SINGLETON,
        )

        # TOOL_CALL: fresh instance for each tool invocation
        temp_storage = Binding(
            protocol=TempStorage,
            provider=lambda r: TempStorage(),
            scope=Scope.TOOL_CALL,
        )

    Note:
        Use ``Binding.instance()`` for pre-constructed instances that don't
        need the provider pattern. Use ``eager=True`` when the resource must
        be initialized at context startup rather than on first access.

    Args:
        protocol: The protocol type this binding satisfies.
        provider: Factory function that constructs the instance.
        scope: Lifetime of constructed instances (default: SINGLETON).
        eager: If True, instantiate during context startup (SINGLETON only).
        provided: Pre-constructed instance, accessible without context manager.
    """

    protocol: type[T]
    """The protocol type this binding satisfies.

    This is the type used when calling ``resolver.get(protocol)`` to retrieve
    the resource. Typically a Protocol class or abstract base class.
    """

    provider: Provider[T]
    """Factory function that constructs the resource instance.

    Receives a ``ResourceResolver`` for obtaining dependencies. Called lazily
    on first access unless ``eager=True``.
    """

    scope: Scope = Scope.SINGLETON
    """Lifetime scope controlling instance caching and disposal.

    - SINGLETON: Cached for entire context lifetime (default).
    - TOOL_CALL: Cached per tool invocation, disposed after.
    - PROTOTYPE: No caching, new instance on every access.
    """

    eager: bool = False
    """If True, instantiate immediately when the context starts.

    Only valid for SINGLETON scope. Use for resources that must be initialized
    at startup or that perform validation during construction.
    """

    provided: T | None = None
    """Pre-constructed instance for direct access without context.

    Set by ``Binding.instance()``. Allows accessing the bound value without
    entering the resource context manager, useful for externally-managed
    instances or test fixtures.
    """

    @staticmethod
    def instance[U](protocol: type[U], value: U) -> Binding[U]:
        """Create a binding for a pre-constructed instance.

        The instance is wrapped in a provider and marked as eager, so it's
        resolved immediately when the context starts. This ensures the
        instance is available in the singleton cache for introspection.

        The ``provided`` field stores the instance directly, allowing access
        without entering the resource context manager. Use this for externally-
        managed instances that don't need lifecycle management.

        Example::

            from weakincentives.resources import Binding, ResourceRegistry

            # Pre-constructed instance
            config = Config.from_env()
            filesystem = InMemoryFilesystem()

            # Register using Binding.instance()
            registry = ResourceRegistry.of(
                Binding.instance(Config, config),
                Binding.instance(Filesystem, filesystem),
                Binding(Service, lambda r: Service(r.get(Config))),
            )

            # Access provided instances without context
            binding = registry.binding_for(Filesystem)
            fs = binding.provided  # Direct access, no context needed

        Args:
            protocol: The protocol type this binding satisfies.
            value: The pre-constructed instance to return.

        Returns:
            An eager SINGLETON binding that returns the instance.
        """
        return Binding(
            protocol, lambda _: value, scope=Scope.SINGLETON, eager=True, provided=value
        )


__all__ = ["Binding", "Provider"]
