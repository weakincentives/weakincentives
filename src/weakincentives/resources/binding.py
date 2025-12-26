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

"""Resource binding configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .protocols import ResourceResolver
from .scope import Scope

T = TypeVar("T")

Provider = Callable[[ResourceResolver], T]
"""Factory function that constructs a resource given a resolver for dependencies."""


@dataclass(slots=True, frozen=True)
class Binding[T]:
    """Describes how to construct a resource and its lifetime.

    A binding associates a protocol type with a provider function and scope.
    The provider is called lazily on first access, with a resolver for
    obtaining dependencies.

    Example::

        from weakincentives.resources import Binding, Scope

        config_binding = Binding(
            protocol=Config,
            provider=lambda r: Config.from_env(),
            scope=Scope.SINGLETON,
        )

        http_binding = Binding(
            protocol=HTTPClient,
            provider=lambda r: HTTPClient(
                base_url=r.get(Config).api_url,
                timeout=r.get(Config).timeout,
            ),
            scope=Scope.SINGLETON,
        )

    Args:
        protocol: The protocol type this binding satisfies.
        provider: Factory function that constructs the instance.
        scope: Lifetime of constructed instances (default: SINGLETON).
        eager: If True, instantiate during context startup (SINGLETON only).
    """

    protocol: type[T]
    """The protocol type this binding satisfies."""

    provider: Provider[T]
    """Factory function that constructs the instance."""

    scope: Scope = Scope.SINGLETON
    """Lifetime of constructed instances."""

    eager: bool = False
    """If True, instantiate during context startup (SINGLETON only)."""


__all__ = ["Binding", "Provider"]
