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

"""Tests for NullMailbox, CollectingMailbox, resolvers, and factory."""

from __future__ import annotations

import pytest

from tests.runtime.conftest import SampleEvent
from weakincentives.runtime.mailbox import (
    CollectingMailbox,
    CompositeResolver,
    InMemoryMailbox,
    InMemoryMailboxFactory,
    Mailbox,
    MailboxFactory,
    MailboxResolutionError,
    NullMailbox,
    RegistryResolver,
)

# =============================================================================
# NullMailbox Tests
# =============================================================================


class TestNullMailbox:
    """Tests for NullMailbox implementation."""

    def test_send_returns_id(self) -> None:
        """send() returns a message ID even though message is dropped."""
        mailbox: NullMailbox[str, None] = NullMailbox()
        msg_id = mailbox.send("hello")
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_receive_always_empty(self) -> None:
        """receive() always returns empty list."""
        mailbox: NullMailbox[str, None] = NullMailbox()
        mailbox.send("hello")
        mailbox.send("world")
        assert mailbox.receive(max_messages=10) == []

    def test_purge_returns_zero(self) -> None:
        """purge() returns zero (nothing to purge)."""
        mailbox: NullMailbox[str, None] = NullMailbox()
        mailbox.send("hello")
        assert mailbox.purge() == 0

    def test_approximate_count_zero(self) -> None:
        """approximate_count() always returns zero."""
        mailbox: NullMailbox[str, None] = NullMailbox()
        mailbox.send("hello")
        assert mailbox.approximate_count() == 0


# =============================================================================
# CollectingMailbox Tests
# =============================================================================


class TestCollectingMailbox:
    """Tests for CollectingMailbox implementation."""

    def test_send_collects_messages(self) -> None:
        """send() stores messages in sent list."""
        mailbox: CollectingMailbox[SampleEvent, None] = CollectingMailbox()
        event1 = SampleEvent(data="first")
        event2 = SampleEvent(data="second")
        mailbox.send(event1)
        mailbox.send(event2)

        assert len(mailbox.sent) == 2
        assert mailbox.sent[0] == event1
        assert mailbox.sent[1] == event2

    def test_receive_always_empty(self) -> None:
        """receive() returns empty (collecting only)."""
        mailbox: CollectingMailbox[str, None] = CollectingMailbox()
        mailbox.send("hello")
        assert mailbox.receive(max_messages=10) == []

    def test_purge_clears_collected(self) -> None:
        """purge() clears all collected messages."""
        mailbox: CollectingMailbox[str, None] = CollectingMailbox()
        mailbox.send("hello")
        mailbox.send("world")

        count = mailbox.purge()
        assert count == 2
        assert mailbox.sent == []

    def test_approximate_count(self) -> None:
        """approximate_count() returns number of collected messages."""
        mailbox: CollectingMailbox[str, None] = CollectingMailbox()
        assert mailbox.approximate_count() == 0
        mailbox.send("hello")
        assert mailbox.approximate_count() == 1


# =============================================================================
# Resolver Tests
# =============================================================================


def test_registry_resolver_resolve() -> None:
    """RegistryResolver.resolve() returns registered mailbox."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = RegistryResolver[str]({"test": mailbox})
        resolved = resolver.resolve("test")
        assert resolved is mailbox
    finally:
        mailbox.close()


def test_registry_resolver_resolve_optional() -> None:
    """RegistryResolver.resolve_optional() returns None for unknown."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = RegistryResolver[str]({"test": mailbox})
        assert resolver.resolve_optional("unknown") is None
        assert resolver.resolve_optional("test") is mailbox
    finally:
        mailbox.close()


def test_registry_resolver_resolve_raises_on_unknown() -> None:
    """RegistryResolver.resolve() raises MailboxResolutionError for unknown identifier."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = RegistryResolver[str]({"test": mailbox})
        with pytest.raises(MailboxResolutionError) as exc_info:
            resolver.resolve("unknown")
        assert exc_info.value.identifier == "unknown"
        assert "unknown" in str(exc_info.value)
    finally:
        mailbox.close()


def test_composite_resolver_resolve_from_registry() -> None:
    """CompositeResolver.resolve() returns mailbox from registry."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = CompositeResolver[str](registry={"test": mailbox}, factory=None)
        resolved = resolver.resolve("test")
        assert resolved is mailbox
    finally:
        mailbox.close()


def test_composite_resolver_resolve_raises_without_factory() -> None:
    """CompositeResolver.resolve() raises when identifier not in registry and no factory."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = CompositeResolver[str](registry={"test": mailbox}, factory=None)
        with pytest.raises(MailboxResolutionError) as exc_info:
            resolver.resolve("unknown")
        assert exc_info.value.identifier == "unknown"
    finally:
        mailbox.close()


def test_composite_resolver_resolve_uses_factory() -> None:
    """CompositeResolver.resolve() uses factory for unknown identifiers."""
    created_mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="created")

    class _TestFactory(MailboxFactory[str]):
        def create(self, identifier: str) -> Mailbox[str, None]:
            return created_mailbox

    try:
        resolver = CompositeResolver[str](registry={}, factory=_TestFactory())
        resolved = resolver.resolve("dynamic")
        assert resolved is created_mailbox
    finally:
        created_mailbox.close()


def test_composite_resolver_resolve_optional_from_registry() -> None:
    """CompositeResolver.resolve_optional() returns mailbox from registry."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = CompositeResolver[str](registry={"test": mailbox}, factory=None)
        resolved = resolver.resolve_optional("test")
        assert resolved is mailbox
    finally:
        mailbox.close()


def test_composite_resolver_resolve_optional_none_without_factory() -> None:
    """CompositeResolver.resolve_optional() returns None when no factory and not in registry."""
    mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
    try:
        resolver = CompositeResolver[str](registry={"test": mailbox}, factory=None)
        assert resolver.resolve_optional("unknown") is None
    finally:
        mailbox.close()


def test_composite_resolver_resolve_optional_uses_factory() -> None:
    """CompositeResolver.resolve_optional() uses factory for unknown identifiers."""
    created_mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="created")

    class _TestFactory(MailboxFactory[str]):
        def create(self, identifier: str) -> Mailbox[str, None]:
            return created_mailbox

    try:
        resolver = CompositeResolver[str](registry={}, factory=_TestFactory())
        resolved = resolver.resolve_optional("dynamic")
        assert resolved is created_mailbox
    finally:
        created_mailbox.close()


def test_composite_resolver_resolve_optional_catches_resolution_error() -> None:
    """CompositeResolver.resolve_optional() returns None when factory raises MailboxResolutionError."""

    class _FailingFactory(MailboxFactory[str]):
        def create(self, identifier: str) -> Mailbox[str, None]:
            raise MailboxResolutionError(identifier)

    resolver = CompositeResolver[str](registry={}, factory=_FailingFactory())
    assert resolver.resolve_optional("dynamic") is None


# =============================================================================
# InMemoryMailboxFactory Tests
# =============================================================================


class TestInMemoryMailboxFactory:
    """Tests for InMemoryMailboxFactory."""

    def test_factory_creates_mailbox(self) -> None:
        """Factory creates an InMemoryMailbox instance."""
        factory: InMemoryMailboxFactory[str] = InMemoryMailboxFactory()
        mailbox = factory.create("test-queue")
        try:
            assert mailbox.name == "test-queue"
            # Verify it's functional
            mailbox.send("hello")
            msgs = mailbox.receive()
            assert len(msgs) == 1
            assert msgs[0].body == "hello"
            msgs[0].acknowledge()
        finally:
            mailbox.close()

    def test_factory_caches_mailbox_with_shared_registry(self) -> None:
        """Factory caches mailbox when shared registry is provided."""
        registry: dict[str, Mailbox[str, None]] = {}
        factory: InMemoryMailboxFactory[str] = InMemoryMailboxFactory(registry=registry)

        mailbox1 = factory.create("test-queue")
        mailbox2 = factory.create("test-queue")

        try:
            # Same mailbox instance is returned
            assert mailbox1 is mailbox2
            assert "test-queue" in registry
            assert registry["test-queue"] is mailbox1
        finally:
            mailbox1.close()
