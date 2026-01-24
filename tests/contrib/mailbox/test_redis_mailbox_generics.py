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

"""Tests for RedisMailbox generic type inference.

These tests verify that RedisMailbox can infer the body type from the
generic type parameter when using subscripted instantiation syntax:
    ``RedisMailbox[MyType, None](name="q", client=c)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from redis import Redis


pytestmark = [pytest.mark.redis_standalone]


@dataclass(slots=True, frozen=True)
class _TestEvent:
    """Sample event type for testing generic type inference."""

    message: str
    value: int = 0


@dataclass(slots=True, frozen=True)
class _GenericContainer[T]:
    """Generic dataclass for testing nested generic type inference."""

    data: T
    label: str = "default"


class TestRedisMailboxGenericTypeInference:
    """Tests for automatic body type inference from generic parameters."""

    def test_subscripted_instantiation_infers_dataclass_type(
        self, redis_client: Redis[bytes]
    ) -> None:
        """Subscripted instantiation ``RedisMailbox[T, R](...)`` infers body type."""
        from weakincentives.contrib.mailbox import RedisMailbox

        # Use subscripted syntax - Python sets __orig_class__ on the instance
        mailbox = RedisMailbox[_TestEvent, None](
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )
        try:
            # Send and receive without explicit body_type
            event = _TestEvent(message="hello", value=42)
            mailbox.send(event)

            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].body == event
            assert isinstance(messages[0].body, _TestEvent)
            messages[0].acknowledge()
        finally:
            mailbox.close()
            mailbox.purge()

    def test_subscripted_instantiation_infers_primitive_type(
        self, redis_client: Redis[bytes]
    ) -> None:
        """Subscripted instantiation works with primitive types."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox = RedisMailbox[str, None](
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )
        try:
            mailbox.send("hello world")

            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].body == "hello world"
            assert isinstance(messages[0].body, str)
            messages[0].acknowledge()
        finally:
            mailbox.close()
            mailbox.purge()

    def test_subscripted_instantiation_infers_int_type(
        self, redis_client: Redis[bytes]
    ) -> None:
        """Subscripted instantiation works with int type."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox = RedisMailbox[int, None](
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )
        try:
            mailbox.send(42)

            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].body == 42
            assert isinstance(messages[0].body, int)
            messages[0].acknowledge()
        finally:
            mailbox.close()
            mailbox.purge()

    def test_subscripted_instantiation_infers_generic_alias(
        self, redis_client: Redis[bytes]
    ) -> None:
        """Subscripted instantiation works with generic alias types."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox = RedisMailbox[_GenericContainer[str], None](
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )
        try:
            container = _GenericContainer(data="nested", label="test")
            mailbox.send(container)

            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            assert messages[0].body == container
            assert isinstance(messages[0].body, _GenericContainer)
            assert messages[0].body.data == "nested"
            messages[0].acknowledge()
        finally:
            mailbox.close()
            mailbox.purge()

    def test_without_subscript_returns_raw_json(
        self, redis_client: Redis[bytes]
    ) -> None:
        """Without subscripted syntax, raw JSON data is returned."""
        from weakincentives.contrib.mailbox import RedisMailbox

        # Plain instantiation without subscript
        mailbox = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )
        try:
            # Send a dict (which serializes as JSON)
            mailbox.send({"key": "value", "num": 123})

            messages = mailbox.receive(max_messages=1)
            assert len(messages) == 1
            # Without type info, we get raw JSON data
            assert messages[0].body == {"key": "value", "num": 123}
            messages[0].acknowledge()
        finally:
            mailbox.close()
            mailbox.purge()

    def test_resolved_body_type_is_cached(self, redis_client: Redis[bytes]) -> None:
        """Resolved body type is cached after first access."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mailbox = RedisMailbox[_TestEvent, None](
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            reaper_interval=0.1,
        )
        try:
            # First call resolves and caches
            resolved1 = mailbox._get_body_type()
            # Second call returns cached value
            resolved2 = mailbox._get_body_type()

            assert resolved1 is _TestEvent
            assert resolved2 is _TestEvent
            # Should be the same object (cached)
            assert resolved1 is resolved2
        finally:
            mailbox.close()
            mailbox.purge()


class TestRedisMailboxFactoryGenerics:
    """Tests for RedisMailboxFactory with generic types."""

    def test_factory_propagates_type_from_subscript(
        self, redis_client: Redis[bytes]
    ) -> None:
        """Factory propagates body type from its own subscripted instantiation."""
        from weakincentives.contrib.mailbox import RedisMailboxFactory

        # Use subscripted syntax to specify the type
        factory = RedisMailboxFactory[_TestEvent](client=redis_client)

        created = factory.create("test-reply")
        assert created.name == "test-reply"
        # The factory should have resolved its type and passed it to the created mailbox
        assert factory._get_body_type() is _TestEvent
