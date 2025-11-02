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

"""Registry for prompt factories keyed by namespace and prompt key."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from ..session import Session
else:  # pragma: no cover - runtime does not require the concrete type
    Session = Any
from .prompt import Prompt


class PromptFactory(Protocol):
    """Callable that builds a prompt bound to the provided session."""

    def __call__(self, *, session: Session) -> Prompt[Any]: ...


_registry: dict[tuple[str, str], PromptFactory] = {}


def register(ns: str, key: str, factory: PromptFactory) -> None:
    """Register a prompt factory for the provided namespace/key pair."""

    lookup = (ns, key)
    if lookup in _registry:
        msg = f"Prompt already registered for namespace={ns!r} key={key!r}"
        raise ValueError(msg)
    _registry[lookup] = factory


def resolve(ns: str, key: str) -> PromptFactory | None:
    """Return the prompt factory for ``(ns, key)`` if registered."""

    return _registry.get((ns, key))


def unregister(ns: str, key: str) -> None:
    """Remove a registered prompt factory if present."""

    _registry.pop((ns, key), None)


def clear() -> None:
    """Clear all registered prompt factories (intended for tests)."""

    _registry.clear()


__all__ = ["PromptFactory", "register", "resolve", "unregister", "clear"]
