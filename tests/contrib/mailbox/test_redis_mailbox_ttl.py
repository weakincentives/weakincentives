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

"""Tests for RedisMailbox TTL functionality.

TTL is applied to all Redis keys to prevent orphaned data from accumulating.
Keys are refreshed on each operation, so active queues stay alive indefinitely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from weakincentives.runtime.clock import FakeClock

if TYPE_CHECKING:
    from redis import Redis

pytestmark = pytest.mark.redis_standalone


class TestTTLConstant:
    """Tests for DEFAULT_TTL_SECONDS constant."""

    def test_default_ttl_seconds_exported(self) -> None:
        """DEFAULT_TTL_SECONDS should be exported from the module."""
        from weakincentives.contrib.mailbox import DEFAULT_TTL_SECONDS

        assert DEFAULT_TTL_SECONDS == 259200  # 3 days in seconds

    def test_default_ttl_equals_three_days(self) -> None:
        """DEFAULT_TTL_SECONDS should equal 3 days in seconds."""
        from weakincentives.contrib.mailbox import DEFAULT_TTL_SECONDS

        three_days = 3 * 24 * 60 * 60
        assert three_days == DEFAULT_TTL_SECONDS


class TestTTLConfiguration:
    """Tests for TTL configuration on RedisMailbox and RedisMailboxFactory."""

    @pytest.fixture
    def mock_redis_client(self) -> MagicMock:
        """Create a mock Redis client."""
        client = MagicMock(spec=["register_script", "ping"])
        client.register_script.return_value = MagicMock()
        return client

    def test_mailbox_uses_default_ttl(
        self, mock_redis_client: MagicMock, clock: FakeClock
    ) -> None:
        """RedisMailbox should use DEFAULT_TTL_SECONDS by default."""
        from weakincentives.contrib.mailbox import (
            DEFAULT_TTL_SECONDS,
            RedisMailbox,
        )

        mb: RedisMailbox[str, None] = RedisMailbox(
            name="test",
            client=mock_redis_client,
            clock=clock,
            _send_only=True,
        )

        assert mb.default_ttl == DEFAULT_TTL_SECONDS
        mb.close()

    def test_mailbox_custom_ttl(
        self, mock_redis_client: MagicMock, clock: FakeClock
    ) -> None:
        """RedisMailbox should accept custom TTL."""
        from weakincentives.contrib.mailbox import RedisMailbox

        custom_ttl = 86400  # 1 day
        mb: RedisMailbox[str, None] = RedisMailbox(
            name="test",
            client=mock_redis_client,
            default_ttl=custom_ttl,
            clock=clock,
            _send_only=True,
        )

        assert mb.default_ttl == custom_ttl
        mb.close()

    def test_mailbox_ttl_zero_disables(
        self, mock_redis_client: MagicMock, clock: FakeClock
    ) -> None:
        """Setting TTL to 0 should disable expiration."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mb: RedisMailbox[str, None] = RedisMailbox(
            name="test",
            client=mock_redis_client,
            default_ttl=0,
            clock=clock,
            _send_only=True,
        )

        assert mb.default_ttl == 0
        mb.close()

    def test_factory_uses_default_ttl(
        self, mock_redis_client: MagicMock, clock: FakeClock
    ) -> None:
        """RedisMailboxFactory should use DEFAULT_TTL_SECONDS by default."""
        from weakincentives.contrib.mailbox import (
            DEFAULT_TTL_SECONDS,
            RedisMailboxFactory,
        )

        factory: RedisMailboxFactory[str] = RedisMailboxFactory(
            client=mock_redis_client,
            clock=clock,
        )

        assert factory.default_ttl == DEFAULT_TTL_SECONDS

    def test_factory_custom_ttl(
        self, mock_redis_client: MagicMock, clock: FakeClock
    ) -> None:
        """RedisMailboxFactory should accept custom TTL."""
        from weakincentives.contrib.mailbox import RedisMailboxFactory

        custom_ttl = 86400
        factory: RedisMailboxFactory[str] = RedisMailboxFactory(
            client=mock_redis_client,
            default_ttl=custom_ttl,
            clock=clock,
        )

        assert factory.default_ttl == custom_ttl

    def test_factory_propagates_ttl_to_mailbox(
        self, mock_redis_client: MagicMock, clock: FakeClock
    ) -> None:
        """Factory.create() should propagate TTL to created mailbox."""
        from weakincentives.contrib.mailbox import RedisMailboxFactory

        custom_ttl = 86400
        factory: RedisMailboxFactory[str] = RedisMailboxFactory(
            client=mock_redis_client,
            default_ttl=custom_ttl,
            clock=clock,
        )

        mailbox = factory.create("reply-queue")

        assert mailbox.default_ttl == custom_ttl
        mailbox.close()


class TestTTLOnOperations:
    """Tests verifying TTL is passed to Lua scripts in each operation."""

    def test_send_passes_ttl(
        self, redis_client: Redis[bytes], clock: FakeClock
    ) -> None:
        """send() should pass TTL to the Lua script."""
        from weakincentives.contrib.mailbox import RedisMailbox

        custom_ttl = 3600  # 1 hour
        mb: RedisMailbox[str, None] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            default_ttl=custom_ttl,
            clock=clock,
        )

        try:
            msg_id = mb.send("hello")
            assert msg_id is not None

            # Verify TTL was set on keys
            ttl_pending = redis_client.ttl(mb._keys.pending)
            ttl_data = redis_client.ttl(mb._keys.data)
            ttl_meta = redis_client.ttl(mb._keys.meta)

            # TTL should be positive and close to custom_ttl
            # (allow some tolerance for execution time)
            assert 3500 < ttl_pending <= custom_ttl
            assert 3500 < ttl_data <= custom_ttl
            assert 3500 < ttl_meta <= custom_ttl
        finally:
            mb.close()
            mb.purge()

    def test_receive_refreshes_ttl(
        self, redis_client: Redis[bytes], clock: FakeClock
    ) -> None:
        """receive() should refresh TTL on all keys."""
        from weakincentives.contrib.mailbox import RedisMailbox

        custom_ttl = 3600
        mb: RedisMailbox[str, None] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            default_ttl=custom_ttl,
            clock=clock,
        )

        try:
            mb.send("hello")
            msgs = mb.receive(visibility_timeout=30)
            assert len(msgs) == 1

            # Verify TTL was refreshed on keys that exist
            ttl_invisible = redis_client.ttl(mb._keys.invisible)
            ttl_data = redis_client.ttl(mb._keys.data)

            # TTL should be positive (keys exist and have TTL)
            # invisible should have the message, data should have payload
            assert ttl_invisible > 3500
            assert 3500 < ttl_data <= custom_ttl

            msgs[0].acknowledge()
        finally:
            mb.close()
            mb.purge()

    def test_nack_refreshes_data_ttl(
        self, redis_client: Redis[bytes], clock: FakeClock
    ) -> None:
        """nack() should refresh TTL including the data key."""
        from weakincentives.contrib.mailbox import RedisMailbox

        custom_ttl = 3600
        mb: RedisMailbox[str, None] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            default_ttl=custom_ttl,
            clock=clock,
        )

        try:
            mb.send("hello")
            msgs = mb.receive(visibility_timeout=30)
            assert len(msgs) == 1

            # Nack the message
            msgs[0].nack(visibility_timeout=0)

            # Verify TTL on data key
            ttl_data = redis_client.ttl(mb._keys.data)
            assert 3500 < ttl_data <= custom_ttl
        finally:
            mb.close()
            mb.purge()

    def test_extend_refreshes_data_ttl(
        self, redis_client: Redis[bytes], clock: FakeClock
    ) -> None:
        """extend() should refresh TTL including the data key."""
        from weakincentives.contrib.mailbox import RedisMailbox

        custom_ttl = 3600
        mb: RedisMailbox[str, None] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            default_ttl=custom_ttl,
            clock=clock,
        )

        try:
            mb.send("hello")
            msgs = mb.receive(visibility_timeout=30)
            assert len(msgs) == 1

            # Extend visibility
            msgs[0].extend_visibility(60)

            # Verify TTL on data key
            ttl_data = redis_client.ttl(mb._keys.data)
            assert 3500 < ttl_data <= custom_ttl

            msgs[0].acknowledge()
        finally:
            mb.close()
            mb.purge()

    def test_ttl_zero_no_expiration(
        self, redis_client: Redis[bytes], clock: FakeClock
    ) -> None:
        """TTL=0 should not set any expiration on keys."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mb: RedisMailbox[str, None] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            default_ttl=0,  # Disable TTL
            clock=clock,
        )

        try:
            mb.send("hello")

            # Verify no TTL is set (returns -1 for no expiration)
            ttl_pending = redis_client.ttl(mb._keys.pending)
            ttl_data = redis_client.ttl(mb._keys.data)

            assert ttl_pending == -1  # No expiration
            assert ttl_data == -1
        finally:
            mb.close()
            mb.purge()


class TestReaperTTLRefresh:
    """Tests for TTL refresh by the background reaper."""

    def test_reaper_refreshes_ttl_with_no_expired_messages(
        self, redis_client: Redis[bytes], clock: FakeClock
    ) -> None:
        """Reaper should refresh TTL even when no messages expire.

        This is critical to prevent data loss for queues with long visibility
        timeouts where the reaper runs but finds no expired messages.
        """
        import time

        from weakincentives.contrib.mailbox import RedisMailbox

        custom_ttl = 3600
        mb: RedisMailbox[str, None] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            default_ttl=custom_ttl,
            reaper_interval=0.1,  # Fast reaper for testing
            clock=clock,
        )

        try:
            # Send a message and receive it with long visibility timeout
            mb.send("hello")
            msgs = mb.receive(visibility_timeout=300)  # 5 minutes
            assert len(msgs) == 1

            # Get initial TTL on data key
            initial_ttl = redis_client.ttl(mb._keys.data)
            assert initial_ttl > 0

            # Manually set a shorter TTL to test refresh
            redis_client.expire(mb._keys.data, 100)
            short_ttl = redis_client.ttl(mb._keys.data)
            assert short_ttl <= 100

            # Wait for reaper to run (should refresh TTL even with no expired msgs)
            time.sleep(0.3)

            # TTL should be refreshed back to near custom_ttl
            refreshed_ttl = redis_client.ttl(mb._keys.data)
            assert refreshed_ttl > 3500  # Should be close to 3600 again

            msgs[0].acknowledge()
        finally:
            mb.close()
            mb.purge()

    def test_reaper_refreshes_all_keys(
        self, redis_client: Redis[bytes], clock: FakeClock
    ) -> None:
        """Reaper should refresh TTL on all four Redis keys."""
        import time

        from weakincentives.contrib.mailbox import RedisMailbox

        custom_ttl = 3600
        mb: RedisMailbox[str, None] = RedisMailbox(
            name=f"test-{uuid4().hex[:8]}",
            client=redis_client,
            default_ttl=custom_ttl,
            reaper_interval=0.1,
            clock=clock,
        )

        try:
            # Send and receive to create all keys
            mb.send("hello")
            msgs = mb.receive(visibility_timeout=300)
            assert len(msgs) == 1

            # Set shorter TTLs on all keys
            redis_client.expire(mb._keys.pending, 100)
            redis_client.expire(mb._keys.invisible, 100)
            redis_client.expire(mb._keys.data, 100)
            redis_client.expire(mb._keys.meta, 100)

            # Wait for reaper
            time.sleep(0.3)

            # All keys should have TTL refreshed
            # (pending may be empty, but others should have TTL)
            ttl_invisible = redis_client.ttl(mb._keys.invisible)
            ttl_data = redis_client.ttl(mb._keys.data)
            ttl_meta = redis_client.ttl(mb._keys.meta)

            assert ttl_invisible > 3500
            assert ttl_data > 3500
            assert ttl_meta > 3500

            msgs[0].acknowledge()
        finally:
            mb.close()
            mb.purge()
