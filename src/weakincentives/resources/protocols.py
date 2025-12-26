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

"""Protocols for resource resolution and lifecycle management."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ResourceResolver(Protocol):
    """Protocol for resolving dependencies during construction.

    Passed to provider functions to enable dependency injection::

        def make_service(resolver: ResourceResolver) -> MyService:
            config = resolver.get(Config)
            http = resolver.get(HTTPClient)
            return MyService(config=config, http=http)
    """

    def get[T](self, protocol: type[T]) -> T:
        """Return the resource for the given protocol.

        Raises:
            UnboundResourceError: No binding exists for the protocol.
            CircularDependencyError: Dependency cycle detected.
        """
        ...

    def get_optional[T](self, protocol: type[T]) -> T | None:
        """Return the resource if bound, None otherwise."""
        ...


@runtime_checkable
class Closeable(Protocol):
    """Protocol for resources requiring cleanup.

    Resources implementing this protocol have ``close()`` called when
    their scope ends (tool-call completion or session shutdown).

    Example::

        class ConnectionPool(Closeable):
            def close(self) -> None:
                for conn in self._connections:
                    conn.close()
                self._connections.clear()
    """

    def close(self) -> None:
        """Release resources. Called when scope ends."""
        ...


@runtime_checkable
class PostConstruct(Protocol):
    """Protocol for post-construction initialization.

    Called after the provider returns and before the instance is cached.
    Use for initialization that requires the instance to exist first
    (e.g., registering callbacks, validating configuration).

    Failures in ``post_construct()`` prevent caching and propagate as
    ``ProviderError``.

    Example::

        class DatabaseClient(PostConstruct):
            def __init__(self, config: Config) -> None:
                self._config = config
                self._pool: Pool | None = None

            def post_construct(self) -> None:
                self._pool = create_pool(self._config.connection_string)
                self._pool.verify_connection()  # Fail fast if unreachable
    """

    def post_construct(self) -> None:
        """Called after construction, before caching.

        Failures here prevent the resource from being cached and are
        wrapped in ``ProviderError``.
        """
        ...


__all__ = [
    "Closeable",
    "PostConstruct",
    "ResourceResolver",
]
