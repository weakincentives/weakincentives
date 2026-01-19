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

"""PromptResources: context manager + proxy for prompt resource lifecycle.

This module is intentionally separate from prompt.py to avoid import cycles.
Tool modules need access to PromptResources for type annotations, but prompt.py
imports from tool-related modules (via section.py/registry.py), creating a cycle.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self

from ..resources import ResourceRegistry
from ..resources.protocols import ResourceResolver

if TYPE_CHECKING:
    from ..resources.context import ScopedResourceContext


class _PromptForResources(Protocol):
    """Minimal protocol describing what PromptResources needs from Prompt.

    This avoids importing Prompt directly which would create import cycles.
    """

    _resource_context: ScopedResourceContext | None

    def _collected_resources(self) -> ResourceRegistry: ...


class PromptResources:
    """Resource accessor: context manager for lifecycle + proxy to active context.

    This class provides dual functionality:

    1. **Context manager**: Use ``with prompt.resources:`` to manage resource
       lifecycle (construction and cleanup).

    2. **Resource proxy**: Access resources via ``prompt.resources.get(Protocol)``
       within the context.

    Example::

        prompt = Prompt(template).bind(resources={Config: config})

        # As context manager
        with prompt.resources as ctx:
            service = ctx.get(MyService)
            # or directly via the proxy
            service = prompt.resources.get(MyService)

        # Outside context, accessing resources fails fast
        prompt.resources.get(MyService)  # Raises RuntimeError

    Note:
        This class intentionally accesses Prompt internals (_resource_context,
        _collected_resources) as it manages the prompt's resource lifecycle.
    """

    __slots__ = ("_prompt",)

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        prompt: _PromptForResources,
    ) -> None:
        self._prompt = prompt

    def __enter__(self) -> Self:
        """Enter resource context; initialize resources."""
        # Access prompt internals - PromptResources manages prompt's lifecycle
        ctx = self._prompt._resource_context  # pyright: ignore[reportPrivateUsage]
        if ctx is not None:
            raise RuntimeError("Resource context already entered")

        registry = self._prompt._collected_resources()  # pyright: ignore[reportPrivateUsage]
        new_ctx = registry._create_context()  # pyright: ignore[reportPrivateUsage]
        new_ctx.start()
        self._prompt._resource_context = new_ctx  # pyright: ignore[reportPrivateUsage]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit resource context; cleanup resources."""
        # Access prompt internals - PromptResources manages prompt's lifecycle
        ctx = self._prompt._resource_context  # pyright: ignore[reportPrivateUsage]
        if ctx is not None:
            ctx.close()
            self._prompt._resource_context = None  # pyright: ignore[reportPrivateUsage]

    @property
    def context(self) -> ScopedResourceContext:
        """Return active resource context, or raise if not in context.

        This provides access to the underlying ScopedResourceContext for
        framework internals that need direct context access (e.g., transactions).

        Raises:
            RuntimeError: If called outside resource context.
        """
        # Access prompt internals - PromptResources manages prompt's lifecycle
        ctx = self._prompt._resource_context  # pyright: ignore[reportPrivateUsage]
        if ctx is None:
            msg = (
                "Resources accessed outside context. "
                "Use 'with prompt.resources:' to enter the resource lifecycle."
            )
            raise RuntimeError(msg)
        return ctx

    def get[T](self, protocol: type[T]) -> T:
        """Resolve and return resource for protocol.

        For pre-provided instances (created via ``Binding.instance()`` or
        ``resources={Protocol: instance}``), returns the instance directly
        without requiring the resource context.

        For factory-constructed resources, requires being inside
        ``with prompt.resources:``.

        Raises:
            RuntimeError: If called outside resource context for non-provided resources.
            UnboundResourceError: No binding exists.
            CircularDependencyError: Dependency cycle detected.
            ProviderError: Provider raised an exception.
        """
        # Check for pre-provided instance first (no context needed)
        registry = self._prompt._collected_resources()  # pyright: ignore[reportPrivateUsage]
        binding = registry.binding_for(protocol)
        if binding is not None and binding.provided is not None:
            return binding.provided
        # Fall back to context-based resolution
        return self.context.get(protocol)

    def get_optional[T](self, protocol: type[T]) -> T | None:
        """Resolve if bound, return None otherwise.

        For pre-provided instances (created via ``Binding.instance()`` or
        ``resources={Protocol: instance}``), returns the instance directly
        without requiring the resource context.

        For factory-constructed resources, requires being inside
        ``with prompt.resources:``.

        Raises:
            RuntimeError: If called outside resource context for non-provided resources.
        """
        # Check for pre-provided instance first (no context needed)
        registry = self._prompt._collected_resources()  # pyright: ignore[reportPrivateUsage]
        binding = registry.binding_for(protocol)
        if binding is None:
            return None  # No binding exists, return None without context
        if binding.provided is not None:
            return binding.provided
        # Fall back to context-based resolution
        return self.context.get_optional(protocol)

    @contextmanager
    def tool_scope(self) -> Iterator[ResourceResolver]:
        """Enter a tool-call scope.

        Resources with TOOL_CALL scope are fresh within this context
        and disposed on exit.

        Raises:
            RuntimeError: If called outside resource context.

        Example::

            with prompt.resources:
                with prompt.resources.tool_scope() as resolver:
                    tracer = resolver.get(Tracer)  # Fresh instance
                # tracer.close() called automatically
        """
        with self.context.tool_scope() as resolver:
            yield resolver


__all__ = ["PromptResources"]
