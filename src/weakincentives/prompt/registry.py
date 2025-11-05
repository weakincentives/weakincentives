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

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from .versioning import PromptLike


class PromptRegistry:
    """In-memory registry that maps namespace/key tuples to prompts."""

    def __init__(self) -> None:
        super().__init__()
        self._prompts: dict[tuple[str, str], PromptLike] = {}
        self._lock = RLock()

    def register(self, prompt: PromptLike) -> None:
        """Register ``prompt`` for lookup via ``(prompt.ns, prompt.key)``."""

        key = (prompt.ns, prompt.key)
        with self._lock:
            self._prompts[key] = prompt

    def unregister(self, ns: str, key: str) -> None:
        """Remove a prompt from the registry if present."""

        lookup = (ns, key)
        with self._lock:
            _ = self._prompts.pop(lookup, None)

    def get(self, ns: str, key: str) -> PromptLike | None:
        """Return the prompt registered under ``(ns, key)`` when present."""

        lookup = (ns, key)
        with self._lock:
            return self._prompts.get(lookup)

    def items(self) -> Iterable[PromptLike]:
        """Iterate over the registered prompts."""

        with self._lock:
            return tuple(self._prompts.values())

    def clear(self) -> None:
        """Remove every registered prompt."""

        with self._lock:
            self._prompts.clear()


_REGISTRY = PromptRegistry()


def register_prompt(prompt: PromptLike) -> None:
    """Register ``prompt`` with the global :class:`PromptRegistry`."""

    _REGISTRY.register(prompt)


def unregister_prompt(ns: str, key: str) -> None:
    """Unregister the prompt identified by ``ns`` and ``key``."""

    _REGISTRY.unregister(ns, key)


def get_prompt(ns: str, key: str) -> PromptLike | None:
    """Fetch the registered prompt identified by ``ns`` and ``key``."""

    return _REGISTRY.get(ns, key)


def iter_prompts() -> Iterable[PromptLike]:
    """Iterate over the prompts stored in the global registry."""

    return _REGISTRY.items()


def clear_registry() -> None:
    """Remove all prompts from the global registry."""

    _REGISTRY.clear()


__all__ = [
    "PromptRegistry",
    "clear_registry",
    "get_prompt",
    "iter_prompts",
    "register_prompt",
    "unregister_prompt",
]
