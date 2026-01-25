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

"""Resource resolution error hierarchy.

This module defines exceptions raised during dependency injection and resource
resolution. All errors inherit from :class:`ResourceError`, which itself
inherits from both :class:`~weakincentives.errors.WinkError` and
:class:`RuntimeError`.

Error Hierarchy
---------------
::

    WinkError
    └── ResourceError (RuntimeError)
        ├── UnboundResourceError (LookupError)
        ├── CircularDependencyError
        ├── DuplicateBindingError (ValueError)
        └── ProviderError

Example
-------
Catching resource errors during resolution::

    from weakincentives.resources import ResourceRegistry, ResourceError

    try:
        value = registry.get(SomeProtocol)
    except ResourceError as e:
        logger.error(f"Failed to resolve resource: {e}")
"""

from __future__ import annotations

from weakincentives.errors import WinkError


class ResourceError(WinkError, RuntimeError):
    """Base class for all resource resolution errors.

    This exception serves as the root of the resource error hierarchy. Catch
    this type to handle any error that occurs during resource resolution,
    binding, or provider execution.

    Inherits from both :class:`~weakincentives.errors.WinkError` (for
    framework-wide error handling) and :class:`RuntimeError` (for standard
    exception semantics).

    Example
    -------
    ::

        try:
            resource = resolver.get(MyProtocol)
        except ResourceError as e:
            # Handles UnboundResourceError, CircularDependencyError, etc.
            handle_resolution_failure(e)
    """


class UnboundResourceError(ResourceError, LookupError):
    """No binding exists for the requested protocol.

    Raised when attempting to resolve a resource that has no registered
    binding or direct instance in the resource registry.

    Parameters
    ----------
    protocol
        The protocol type that was requested but has no binding.

    Attributes
    ----------
    protocol : type[object]
        The unbound protocol type that triggered this error.

    Example
    -------
    This error commonly occurs when a required binding is missing::

        from weakincentives.resources import ResourceRegistry

        registry = ResourceRegistry.build([])  # Empty registry
        try:
            registry.get(DatabaseConnection)  # Not bound
        except UnboundResourceError as e:
            print(f"Missing binding for: {e.protocol.__name__}")
            # Fix: Add binding before building registry

    To prevent this error, ensure all required protocols are bound::

        registry = ResourceRegistry.build([
            Binding(DatabaseConnection, lambda r: create_db_connection()),
        ])
    """

    def __init__(self, protocol: type[object]) -> None:
        self.protocol = protocol
        super().__init__(f"No binding for {protocol.__name__}")


class CircularDependencyError(ResourceError):
    """Circular dependency detected during resource resolution.

    Raised when resource A depends on B, which depends on A (directly or
    transitively). This indicates a design flaw in the dependency graph
    that must be resolved by restructuring dependencies.

    Parameters
    ----------
    cycle
        Tuple of protocol types forming the dependency cycle, where the
        last element depends back on the first.

    Attributes
    ----------
    cycle : tuple[type[object], ...]
        The sequence of types forming the circular dependency. For example,
        ``(A, B, C, A)`` means A depends on B, B depends on C, and C depends
        on A.

    Example
    -------
    A circular dependency occurs when resources depend on each other::

        # This creates a cycle: ServiceA -> ServiceB -> ServiceA
        bindings = [
            Binding(ServiceA, lambda r: ServiceAImpl(r.get(ServiceB))),
            Binding(ServiceB, lambda r: ServiceBImpl(r.get(ServiceA))),
        ]

        registry = ResourceRegistry.build(bindings)
        try:
            registry.get(ServiceA)
        except CircularDependencyError as e:
            print(f"Cycle detected: {' -> '.join(t.__name__ for t in e.cycle)}")

    To fix circular dependencies:

    1. Extract shared logic into a third service that both depend on
    2. Use lazy initialization or setter injection
    3. Restructure to eliminate the mutual dependency
    """

    def __init__(self, cycle: tuple[type[object], ...]) -> None:
        self.cycle = cycle
        path = " -> ".join(t.__name__ for t in cycle)
        super().__init__(f"Circular dependency: {path}")


class DuplicateBindingError(ResourceError, ValueError):
    """Same protocol was bound multiple times in the registry.

    Raised when :meth:`ResourceRegistry.build()` receives multiple bindings
    for the same protocol type. Each protocol may only have one binding;
    duplicates indicate a configuration error.

    Parameters
    ----------
    protocol
        The protocol type that was bound more than once.

    Attributes
    ----------
    protocol : type[object]
        The protocol type that has duplicate bindings.

    Example
    -------
    This error occurs when the same protocol appears in multiple bindings::

        bindings = [
            Binding(Logger, lambda r: FileLogger()),
            Binding(Logger, lambda r: ConsoleLogger()),  # Duplicate!
        ]

        try:
            registry = ResourceRegistry.build(bindings)
        except DuplicateBindingError as e:
            print(f"Duplicate binding for: {e.protocol.__name__}")

    To fix, ensure each protocol has exactly one binding. If you need
    multiple implementations, consider:

    1. Using distinct protocol types for each implementation
    2. Binding a composite that delegates to multiple implementations
    3. Using a factory that selects the appropriate implementation
    """

    def __init__(self, protocol: type[object]) -> None:
        self.protocol = protocol
        super().__init__(f"Duplicate binding for {protocol.__name__}")


class ProviderError(ResourceError):
    """Provider function raised an exception during resource construction.

    Wraps exceptions from provider functions to provide context about which
    resource failed to construct. The original exception is preserved in the
    ``cause`` attribute for inspection or re-raising.

    Parameters
    ----------
    protocol
        The protocol type whose provider failed.
    cause
        The exception raised by the provider function.

    Attributes
    ----------
    protocol : type[object]
        The protocol type that failed to construct.
    cause : BaseException
        The original exception raised by the provider function. Inspect this
        to understand the root cause of the failure.

    Example
    -------
    Provider errors wrap underlying failures::

        def create_database(resolver):
            # This might raise ConnectionError, ValueError, etc.
            return Database(host="localhost", port=5432)

        bindings = [Binding(Database, create_database)]
        registry = ResourceRegistry.build(bindings)

        try:
            db = registry.get(Database)
        except ProviderError as e:
            print(f"Failed to create {e.protocol.__name__}")
            print(f"Cause: {e.cause}")
            # Access the original exception for detailed handling
            if isinstance(e.cause, ConnectionError):
                retry_with_backoff()

    To debug provider errors, examine the ``cause`` attribute to understand
    what went wrong in the provider function.
    """

    def __init__(self, protocol: type[object], cause: BaseException) -> None:
        self.protocol = protocol
        self.cause = cause
        super().__init__(
            f"Provider for {protocol.__name__} raised {type(cause).__name__}: {cause}"
        )


__all__ = [
    "CircularDependencyError",
    "DuplicateBindingError",
    "ProviderError",
    "ResourceError",
    "UnboundResourceError",
]
