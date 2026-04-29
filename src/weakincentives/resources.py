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

"""Scoped dependency injection container.

Bindings declare how a protocol is resolved and at what lifetime. The
spine's :class:`weakincentives.core.ResourceProvider` is satisfied by
:class:`ScopedResourceContext`, which a tool transaction enters before
running a handler and exits afterwards. Singleton resources are reused
across calls; tool-call resources are rebuilt per call; prototype
resources are rebuilt every time they are requested.
"""

# `ScopedResourceContext` reaches into peer attributes by design.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import enum
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast, final

from weakincentives.core import Snapshotable

__all__ = [
    "Binding",
    "Closeable",
    "PostConstruct",
    "ResourceRegistry",
    "ResourceScope",
    "ScopedResourceContext",
]


class ResourceScope(enum.Enum):
    """Lifetime of a resource produced by a :class:`Binding`."""

    SINGLETON = "singleton"
    TOOL_CALL = "tool_call"
    PROTOTYPE = "prototype"


class Closeable:
    """Optional protocol for resources that need a teardown hook."""

    def close(self) -> None:  # pragma: no cover - protocol stub
        raise NotImplementedError


class PostConstruct:
    """Optional protocol for resources that need post-construction setup."""

    def post_construct(self) -> None:  # pragma: no cover - protocol stub
        raise NotImplementedError


@final
@dataclass(frozen=True, slots=True)
class Binding[T]:
    """A registration that maps a protocol to a factory."""

    protocol: type[T]
    factory: Callable[[ScopedResourceContext], T]
    scope: ResourceScope = ResourceScope.SINGLETON


class ResourceRegistry:
    """Collects :class:`Binding` declarations and produces contexts."""

    def __init__(self) -> None:
        super().__init__()
        self._bindings: dict[type[object], Binding[object]] = {}

    def bind[T](self, binding: Binding[T]) -> None:
        self._bindings[binding.protocol] = cast("Binding[object]", binding)

    def has(self, protocol: type[object]) -> bool:
        return protocol in self._bindings

    @contextmanager
    def context(self) -> Iterator[ScopedResourceContext]:
        ctx = ScopedResourceContext(bindings=dict(self._bindings))
        try:
            yield ctx
        finally:
            ctx._close()


@final
class ScopedResourceContext:
    """Resolves bindings and caches per-scope instances.

    Singleton instances live for the lifetime of the context. Tool-call
    instances are reset between :meth:`enter_tool_call` and
    :meth:`exit_tool_call`. Prototype instances are produced fresh each
    time :meth:`get` is called.
    """

    def __init__(self, *, bindings: dict[type[object], Binding[object]]) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._bindings = bindings
        self._singletons: dict[type[object], object] = {}
        self._tool_call: dict[type[object], object] | None = None

    def get[T](self, protocol_type: type[T]) -> T:
        try:
            binding = self._bindings[protocol_type]
        except KeyError as error:
            raise LookupError(f"no binding for {protocol_type.__qualname__}") from error
        with self._lock:
            return self._resolve(cast("Binding[T]", binding))

    def _resolve[T](self, binding: Binding[T]) -> T:
        if binding.scope is ResourceScope.SINGLETON:
            existing = self._singletons.get(binding.protocol)
            if existing is None:
                produced = binding.factory(self)
                _maybe_post_construct(produced)
                self._singletons[binding.protocol] = produced
                return produced
            return cast("T", existing)
        if binding.scope is ResourceScope.TOOL_CALL:
            cache = self._tool_call
            if cache is None:
                raise LookupError("tool-call resources require enter_tool_call()")
            existing = cache.get(binding.protocol)
            if existing is None:
                produced = binding.factory(self)
                _maybe_post_construct(produced)
                cache[binding.protocol] = produced
                return produced
            return cast("T", existing)
        produced = binding.factory(self)
        _maybe_post_construct(produced)
        return produced

    @contextmanager
    def tool_call_scope(self) -> Iterator[None]:
        with self._lock:
            if self._tool_call is not None:
                raise LookupError("tool-call scope already active")
            self._tool_call = {}
        try:
            yield
        finally:
            with self._lock:
                cache = self._tool_call or {}
                self._tool_call = None
            for instance in cache.values():
                _maybe_close(instance)

    @property
    def singletons(self) -> dict[type[object], object]:
        with self._lock:
            return dict(self._singletons)

    def snapshotable_resources(self) -> tuple[Snapshotable[Any], ...]:
        """Return every cached singleton that satisfies ``Snapshotable``."""
        with self._lock:
            return tuple(
                cast("Snapshotable[Any]", value)
                for value in self._singletons.values()
                if isinstance(value, Snapshotable)
            )

    def _close(self) -> None:
        with self._lock:
            singletons = list(self._singletons.values())
            self._singletons.clear()
        for instance in singletons:
            _maybe_close(instance)


def _maybe_post_construct(instance: object) -> None:
    hook = getattr(instance, "post_construct", None)
    if callable(hook):
        _ = hook()


def _maybe_close(instance: object) -> None:
    hook = getattr(instance, "close", None)
    if callable(hook):
        _ = hook()
