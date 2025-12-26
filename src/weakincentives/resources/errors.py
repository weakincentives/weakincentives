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

"""Resource resolution error hierarchy."""

from __future__ import annotations

from weakincentives.errors import WinkError


class ResourceError(WinkError, RuntimeError):
    """Base class for resource resolution errors."""


class UnboundResourceError(ResourceError, LookupError):
    """No binding exists for the requested protocol.

    Raised when attempting to resolve a resource that has no registered
    binding or direct instance.
    """

    def __init__(self, protocol: type[object]) -> None:
        self.protocol = protocol
        super().__init__(f"No binding for {protocol.__name__}")


class CircularDependencyError(ResourceError):
    """Circular dependency detected during resolution.

    Raised when resource A depends on B, which depends on A (directly or
    transitively). The ``cycle`` attribute contains the types involved.
    """

    def __init__(self, cycle: tuple[type[object], ...]) -> None:
        self.cycle = cycle
        path = " -> ".join(t.__name__ for t in cycle)
        super().__init__(f"Circular dependency: {path}")


class DuplicateBindingError(ResourceError, ValueError):
    """Same protocol bound multiple times.

    Raised when ``ResourceRegistry.build()`` receives multiple bindings
    for the same protocol type.
    """

    def __init__(self, protocol: type[object]) -> None:
        self.protocol = protocol
        super().__init__(f"Duplicate binding for {protocol.__name__}")


class ProviderError(ResourceError):
    """Provider raised an exception during construction.

    Wraps exceptions from provider functions to provide context about
    which resource failed to construct.
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
