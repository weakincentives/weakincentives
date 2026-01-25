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

"""Protocols for resource resolution and lifecycle management.

This module defines the core protocols that resources can implement to
participate in dependency injection and lifecycle management:

- :class:`ResourceResolver`: Resolves dependencies during construction
- :class:`Closeable`: Cleanup when scope ends (tool-call or session)
- :class:`PostConstruct`: Initialization after construction but before caching
- :class:`Snapshotable`: Capture and restore state for transactional rollback

These protocols are runtime-checkable, so you can use ``isinstance()`` to
detect whether a resource supports a particular capability.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ResourceResolver(Protocol):
    """Protocol for resolving dependencies during construction.

    A resolver is passed to provider functions (factories) to enable
    dependency injection. Use it to obtain other resources that your
    resource depends on.

    Example::

        def make_service(resolver: ResourceResolver) -> MyService:
            config = resolver.get(Config)
            http = resolver.get(HTTPClient)
            return MyService(config=config, http=http)

    The resolver handles scope management automatically - singleton
    dependencies are cached, tool-call scoped dependencies are created
    fresh per tool invocation, and prototype dependencies are never cached.
    """

    def get[T](self, protocol: type[T]) -> T:
        """Resolve and return the resource bound to the given protocol.

        Args:
            protocol: The protocol type (class) to resolve. This should be
                the same type used when creating the binding.

        Returns:
            The resource instance bound to the protocol. For singletons,
            returns the cached instance. For other scopes, may create a
            new instance.

        Raises:
            UnboundResourceError: No binding exists for the protocol.
            CircularDependencyError: Dependency cycle detected during
                resolution (A depends on B which depends on A).
            ProviderError: The provider function raised an exception.
        """
        ...

    def get_optional[T](self, protocol: type[T]) -> T | None:
        """Resolve a resource if bound, returning None if unbound.

        Use this method when a dependency is optional and you want to
        gracefully handle its absence rather than raising an error.

        Args:
            protocol: The protocol type (class) to resolve.

        Returns:
            The resource instance if bound, or None if no binding exists.

        Raises:
            CircularDependencyError: Dependency cycle detected.
            ProviderError: The provider function raised an exception.

        Example::

            def make_service(resolver: ResourceResolver) -> MyService:
                # Use default logger if none bound
                logger = resolver.get_optional(Logger) or default_logger
                return MyService(logger=logger)
        """
        ...


@runtime_checkable
class Closeable(Protocol):
    """Protocol for resources requiring cleanup when their scope ends.

    Implement this protocol when your resource holds external resources
    (connections, file handles, threads, etc.) that must be released.
    The resource container automatically calls ``close()`` when:

    - A tool-call scoped resource's tool invocation completes
    - A singleton resource's session shuts down
    - Prototype resources are closed immediately after use

    Close is called in reverse construction order, so dependencies are
    closed after the resources that depend on them.

    Example::

        class ConnectionPool(Closeable):
            def __init__(self, config: Config) -> None:
                self._connections: list[Connection] = []
                self._config = config

            def close(self) -> None:
                for conn in self._connections:
                    conn.close()
                self._connections.clear()

    Note:
        Exceptions raised in ``close()`` are logged but do not prevent
        other resources from being closed. Always aim for ``close()``
        to be idempotent (safe to call multiple times).
    """

    def close(self) -> None:
        """Release held resources and perform cleanup.

        Called automatically by the resource container when the resource's
        scope ends. Implementations should:

        - Release external resources (connections, handles, threads)
        - Be idempotent (safe to call multiple times)
        - Not raise exceptions if possible (log instead)

        This method should complete promptly; avoid blocking operations
        that could delay shutdown.
        """
        ...


@runtime_checkable
class PostConstruct(Protocol):
    """Protocol for initialization that runs after construction.

    Implement this protocol when your resource needs additional setup
    after the constructor returns but before it becomes available for
    injection. Common use cases:

    - Validating configuration and failing fast if invalid
    - Establishing connections or verifying connectivity
    - Registering callbacks that reference ``self``
    - Loading initial state from external sources

    The ``post_construct()`` method is called after the provider returns
    and before the instance is cached (for singleton/tool-call scopes).
    If it raises an exception, the resource is not cached and the error
    propagates as ``ProviderError``.

    Example::

        class DatabaseClient(PostConstruct):
            def __init__(self, config: Config) -> None:
                self._config = config
                self._pool: Pool | None = None

            def post_construct(self) -> None:
                # Fail fast if database is unreachable
                self._pool = create_pool(self._config.connection_string)
                self._pool.verify_connection()

    Note:
        Unlike ``__init__``, this method runs after the object is fully
        constructed, so you can safely reference ``self`` and pass it
        to other objects.
    """

    def post_construct(self) -> None:
        """Perform post-construction initialization.

        Called automatically after the provider function returns and before
        the resource is cached or returned to the caller. Use this for:

        - Validation that should fail fast
        - Connection establishment
        - Self-registration with other services

        Raises:
            Any exception raised here prevents caching and is wrapped
            in ``ProviderError`` before propagating to the caller.
        """
        ...


@runtime_checkable
class Snapshotable[SnapshotT](Protocol):
    """Protocol for resources that support transactional rollback.

    Implement this protocol when your resource maintains mutable state
    that should be rolled back if a tool invocation fails. The resource
    container automatically:

    1. Calls ``snapshot()`` before each tool invocation
    2. Calls ``restore()`` with that snapshot if the tool fails
    3. Discards the snapshot if the tool succeeds

    This enables atomic tool execution - either the tool succeeds and
    all state changes persist, or it fails and state is fully restored.

    The type parameter ``SnapshotT`` is the type of snapshot your resource
    produces. Use a frozen dataclass or other immutable type to ensure
    snapshots cannot be accidentally modified.

    Example::

        @dataclass(frozen=True)
        class FileSystemSnapshot:
            files: Mapping[str, bytes]

        class InMemoryFilesystem(Snapshotable[FileSystemSnapshot]):
            def __init__(self) -> None:
                self._files: dict[str, bytes] = {}

            def snapshot(self, *, tag: str | None = None) -> FileSystemSnapshot:
                # Return immutable copy of current state
                return FileSystemSnapshot(files=dict(self._files))

            def restore(self, snapshot: FileSystemSnapshot) -> None:
                # Replace current state with snapshot
                self._files = dict(snapshot.files)

    Note:
        Snapshots should be deep copies. Shallow copies may allow
        modifications to the live resource to corrupt the snapshot.
    """

    def snapshot(self, *, tag: str | None = None) -> SnapshotT:
        """Capture the current state as an immutable snapshot.

        Called before tool invocations to enable rollback on failure.
        The returned snapshot must be immutable and independent of the
        resource's live state (i.e., a deep copy).

        Args:
            tag: Optional label for debugging and logging. The resource
                container may pass descriptive tags like "pre-tool:write_file"
                to help identify snapshots in logs.

        Returns:
            An immutable snapshot of the current state. This object will
            be passed to ``restore()`` if rollback is needed.
        """
        ...

    def restore(self, snapshot: SnapshotT) -> None:
        """Restore the resource to a previously captured state.

        Called when a tool invocation fails to roll back state changes.
        After this method returns, the resource should be indistinguishable
        from its state when the snapshot was taken.

        Args:
            snapshot: A snapshot previously returned by ``snapshot()``.
                The resource container guarantees this is a valid snapshot
                from this resource instance.

        Note:
            This method should not raise exceptions. If restoration fails,
            the resource may be left in an inconsistent state.
        """
        ...


__all__ = [
    "Closeable",
    "PostConstruct",
    "ResourceResolver",
    "Snapshotable",
]
