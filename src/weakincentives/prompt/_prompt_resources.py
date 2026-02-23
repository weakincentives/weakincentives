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

"""PromptResources: accessor proxy for active resource context.

This module is intentionally separate from prompt.py to avoid import cycles.
Tool modules need access to PromptResources for type annotations, but prompt.py
imports from tool-related modules (via section.py/registry.py), creating a cycle.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol

from ..resources.protocols import ResourceResolver

if TYPE_CHECKING:
    from ..resources.context import ScopedResourceContext


class _PromptForResources(Protocol):
    """Minimal protocol describing what PromptResources needs from Prompt.

    This avoids importing Prompt directly which would create import cycles.
    """

    _resource_context: ScopedResourceContext | None


class PromptResources:
    """Resource accessor: proxy to the active resource context.

    Provides access to resources within an active ``resource_scope()``::

        with prompt.resource_scope():
            service = prompt.resources.get(MyService)

    Lifecycle management (construction and cleanup) is handled by
    ``Prompt.resource_scope()``, not by this class.

    Note:
        This class intentionally accesses Prompt internals (_resource_context)
        as it proxies the prompt's active resource context.
    """

    __slots__ = ("_prompt",)

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        prompt: _PromptForResources,
    ) -> None:
        self._prompt = prompt

    @property
    def context(self) -> ScopedResourceContext:
        """Return active resource context, or raise if not in scope.

        This provides access to the underlying ScopedResourceContext for
        framework internals that need direct context access (e.g., transactions).

        Raises:
            RuntimeError: If called outside resource_scope().
        """
        # Access prompt internals - PromptResources proxies prompt's context
        ctx = self._prompt._resource_context  # pyright: ignore[reportPrivateUsage]
        if ctx is None:
            msg = (
                "Resources accessed outside scope. "
                "Use 'with prompt.resource_scope():' to enter the resource lifecycle."
            )
            raise RuntimeError(msg)
        return ctx

    def get[T](self, protocol: type[T]) -> T:
        """Resolve and return resource for protocol.

        Requires being inside ``with prompt.resource_scope():``.

        Raises:
            RuntimeError: If called outside resource scope.
            UnboundResourceError: No binding exists.
            CircularDependencyError: Dependency cycle detected.
            ProviderError: Provider raised an exception.
        """
        return self.context.get(protocol)

    def get_optional[T](self, protocol: type[T]) -> T | None:
        """Resolve if bound, return None otherwise.

        Requires being inside ``with prompt.resource_scope():``.

        Raises:
            RuntimeError: If called outside resource scope.
        """
        return self.context.get_optional(protocol)

    @contextmanager
    def tool_scope(self) -> Iterator[ResourceResolver]:
        """Enter a tool-call scope.

        Resources with TOOL_CALL scope are fresh within this context
        and disposed on exit.

        Raises:
            RuntimeError: If called outside resource scope.

        Example::

            with prompt.resource_scope():
                with prompt.resources.tool_scope() as resolver:
                    tracer = resolver.get(Tracer)  # Fresh instance
                # tracer.close() called automatically
        """
        with self.context.tool_scope() as resolver:
            yield resolver


__all__ = ["PromptResources"]
