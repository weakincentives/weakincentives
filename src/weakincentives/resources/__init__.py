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

"""Resource injection with scoped lifecycles.

This module provides dependency injection for agent resources with
scope-aware lifecycle management.

Quick Start::

    from weakincentives.resources import Binding, ResourceRegistry, Scope

    # Define how to construct resources
    registry = ResourceRegistry.of(
        Binding(Config, lambda r: Config.from_env()),
        Binding(HTTPClient, lambda r: HTTPClient(r.get(Config).url)),
        Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
    )

    # Create resolution context
    ctx = registry.create_context()
    ctx.start()

    try:
        # Resolve resources (lazily constructed)
        http = ctx.get(HTTPClient)

        # Tool-call scoped resources are fresh per scope
        with ctx.tool_scope() as resolver:
            tracer = resolver.get(Tracer)
            # ... execute tool ...
        # Tracer disposed here
    finally:
        ctx.close()

Scopes
------

- ``SINGLETON``: One instance per session (default)
- ``TOOL_CALL``: Fresh instance per tool invocation
- ``PROTOTYPE``: Fresh instance every access

Lifecycle Protocols
-------------------

- ``Closeable``: Resources with ``close()`` are cleaned up when scope ends
- ``PostConstruct``: Resources with ``post_construct()`` are initialized after construction
"""

from __future__ import annotations

from .binding import Binding, Provider
from .context import ScopedResourceContext
from .errors import (
    CircularDependencyError,
    DuplicateBindingError,
    ProviderError,
    ResourceError,
    UnboundResourceError,
)
from .protocols import Closeable, PostConstruct, ResourceResolver
from .registry import ResourceRegistry
from .scope import Scope

__all__ = [
    "Binding",
    "CircularDependencyError",
    "Closeable",
    "DuplicateBindingError",
    "PostConstruct",
    "Provider",
    "ProviderError",
    "ResourceError",
    "ResourceRegistry",
    "ResourceResolver",
    "Scope",
    "ScopedResourceContext",
    "UnboundResourceError",
]
