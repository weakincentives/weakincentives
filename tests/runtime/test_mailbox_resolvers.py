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

"""Tests for mailbox resolvers, parameter validation, and factory."""

from __future__ import annotations

import pytest

from weakincentives.runtime.mailbox import (
    CompositeResolver,
    FakeMailbox,
    InMemoryMailbox,
    InMemoryMailboxFactory,
    InvalidParameterError,
    Mailbox,
    MailboxFactory,
    MailboxResolutionError,
    RegistryResolver,
)

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
# Parameter Validation Tests
# =============================================================================


class TestParameterValidation:
    """Tests for timeout parameter validation."""

    def test_receive_negative_visibility_timeout_raises(self) -> None:
        """receive() raises InvalidParameterError for negative visibility_timeout."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            with pytest.raises(
                InvalidParameterError, match="visibility_timeout must be non-negative"
            ):
                mailbox.receive(visibility_timeout=-1)
        finally:
            mailbox.close()

    def test_receive_excessive_visibility_timeout_raises(self) -> None:
        """receive() raises InvalidParameterError for visibility_timeout > 43200."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            with pytest.raises(InvalidParameterError, match="must be at most 43200"):
                mailbox.receive(visibility_timeout=43201)
        finally:
            mailbox.close()

    def test_receive_negative_wait_time_raises(self) -> None:
        """receive() raises InvalidParameterError for negative wait_time_seconds."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            with pytest.raises(
                InvalidParameterError, match="wait_time_seconds must be non-negative"
            ):
                mailbox.receive(wait_time_seconds=-1)
        finally:
            mailbox.close()

    def test_receive_zero_visibility_timeout_allowed(self) -> None:
        """receive() allows visibility_timeout=0."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(visibility_timeout=0)
            assert len(messages) == 1
        finally:
            mailbox.close()

    def test_receive_max_visibility_timeout_allowed(self) -> None:
        """receive() allows visibility_timeout=43200 (max value)."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive(visibility_timeout=43200)
            assert len(messages) == 1
            messages[0].acknowledge()
        finally:
            mailbox.close()

    def test_nack_negative_visibility_timeout_raises(self) -> None:
        """nack() raises InvalidParameterError for negative visibility_timeout."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive()
            with pytest.raises(
                InvalidParameterError, match="visibility_timeout must be non-negative"
            ):
                messages[0].nack(visibility_timeout=-1)
        finally:
            mailbox.close()

    def test_nack_excessive_visibility_timeout_raises(self) -> None:
        """nack() raises InvalidParameterError for visibility_timeout > 43200."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive()
            with pytest.raises(InvalidParameterError, match="must be at most 43200"):
                messages[0].nack(visibility_timeout=43201)
        finally:
            mailbox.close()

    def test_extend_visibility_negative_timeout_raises(self) -> None:
        """extend_visibility() raises InvalidParameterError for negative timeout."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive()
            with pytest.raises(
                InvalidParameterError, match="timeout must be non-negative"
            ):
                messages[0].extend_visibility(-1)
        finally:
            mailbox.close()

    def test_extend_visibility_excessive_timeout_raises(self) -> None:
        """extend_visibility() raises InvalidParameterError for timeout > 43200."""
        mailbox: InMemoryMailbox[str, None] = InMemoryMailbox(name="test")
        try:
            mailbox.send("hello")
            messages = mailbox.receive()
            with pytest.raises(InvalidParameterError, match="must be at most 43200"):
                messages[0].extend_visibility(43201)
        finally:
            mailbox.close()

    def test_fake_mailbox_receive_validates_parameters(self) -> None:
        """FakeMailbox.receive() also validates parameters."""
        mailbox: FakeMailbox[str, None] = FakeMailbox()
        mailbox.send("hello")

        with pytest.raises(InvalidParameterError):
            mailbox.receive(visibility_timeout=-1)

        with pytest.raises(InvalidParameterError):
            mailbox.receive(wait_time_seconds=-1)


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
