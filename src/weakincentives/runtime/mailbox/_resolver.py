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

"""Mailbox resolver infrastructure for service discovery.

MailboxResolver provides dynamic reply routing via string identifiers,
enabling Message.reply() to resolve destination mailboxes internally.

See ``specs/MAILBOX.md`` for the complete specification.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...dataclasses import FrozenDataclassMixin
from ._types import MailboxError

if TYPE_CHECKING:
    from ._types import Mailbox


class MailboxResolutionError(MailboxError):
    """Cannot resolve mailbox identifier.

    Raised when:
    - Identifier not in registry and no factory configured
    - Factory cannot create mailbox for identifier
    """

    def __init__(self, identifier: str) -> None:
        super().__init__(f"Cannot resolve mailbox identifier: {identifier}")
        self.identifier = identifier


@runtime_checkable
class MailboxResolver[R](Protocol):
    """Resolves string identifiers to Mailbox instances.

    Type parameter R matches the reply type from Mailbox[T, R].
    """

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        """Resolve an identifier to a mailbox instance.

        Args:
            identifier: The string identifier for the mailbox.

        Returns:
            The resolved mailbox instance.

        Raises:
            MailboxResolutionError: Cannot resolve the identifier.
        """
        ...

    def resolve_optional(self, identifier: str) -> Mailbox[R, None] | None:
        """Resolve if possible, return None otherwise.

        Args:
            identifier: The string identifier for the mailbox.

        Returns:
            The resolved mailbox instance, or None if not found.
        """
        ...


@runtime_checkable
class MailboxFactory[R](Protocol):
    """Creates mailbox instances from string identifiers.

    Factories do not cache - that's the resolver's responsibility.
    """

    def create(self, identifier: str) -> Mailbox[R, None]:
        """Create a new mailbox for the given identifier.

        Args:
            identifier: The string identifier for the mailbox.

        Returns:
            A new mailbox instance.
        """
        ...


@dataclass(slots=True, frozen=True)
class RegistryResolver[R](FrozenDataclassMixin):
    """Simple resolver backed by a static registry.

    Resolution looks up the identifier in the registry mapping.
    No factory fallback - only pre-registered mailboxes are resolvable.

    Example::

        registry = {"responses": InMemoryMailbox()}
        resolver = RegistryResolver(registry)
        mailbox = resolver.resolve("responses")
    """

    registry: Mapping[str, Mailbox[R, None]]
    """Mapping of identifiers to mailbox instances."""

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        """Resolve an identifier to a mailbox instance.

        Args:
            identifier: The string identifier for the mailbox.

        Returns:
            The resolved mailbox instance.

        Raises:
            MailboxResolutionError: Identifier not in registry.
        """
        if identifier not in self.registry:
            raise MailboxResolutionError(identifier)
        return self.registry[identifier]

    def resolve_optional(self, identifier: str) -> Mailbox[R, None] | None:
        """Resolve if possible, return None otherwise.

        Args:
            identifier: The string identifier for the mailbox.

        Returns:
            The resolved mailbox instance, or None if not found.
        """
        return self.registry.get(identifier)


@dataclass(slots=True, frozen=True)
class CompositeResolver[R](FrozenDataclassMixin):
    """Combines a registry with a factory for dynamic resolution.

    Resolution order:
    1. Check registry for pre-registered mailbox
    2. Fall back to factory for dynamic creation

    Example::

        resolver = CompositeResolver(
            registry={"known": existing_mailbox},
            factory=RedisMailboxFactory(client=redis),
        )
        # Pre-registered
        resolver.resolve("known")  # Returns existing_mailbox
        # Dynamic creation
        resolver.resolve("dynamic")  # Creates via factory
    """

    registry: Mapping[str, Mailbox[R, None]]
    """Mapping of identifiers to pre-registered mailbox instances."""

    factory: MailboxFactory[R] | None = None
    """Optional factory for dynamic mailbox creation."""

    def resolve(self, identifier: str) -> Mailbox[R, None]:
        """Resolve an identifier to a mailbox instance.

        Args:
            identifier: The string identifier for the mailbox.

        Returns:
            The resolved mailbox instance.

        Raises:
            MailboxResolutionError: Identifier not in registry and no factory.
        """
        if identifier in self.registry:
            return self.registry[identifier]
        if self.factory is None:
            raise MailboxResolutionError(identifier)
        return self.factory.create(identifier)

    def resolve_optional(self, identifier: str) -> Mailbox[R, None] | None:
        """Resolve if possible, return None otherwise.

        Args:
            identifier: The string identifier for the mailbox.

        Returns:
            The resolved mailbox instance, or None if not resolvable.
        """
        if identifier in self.registry:
            return self.registry[identifier]
        if self.factory is None:
            return None
        try:
            return self.factory.create(identifier)
        except MailboxResolutionError:
            return None


__all__ = [
    "CompositeResolver",
    "MailboxFactory",
    "MailboxResolutionError",
    "MailboxResolver",
    "RegistryResolver",
]
