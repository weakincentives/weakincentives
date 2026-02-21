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

"""Tests for send-only RedisMailbox behavior.

Send-only mailboxes are created by RedisMailboxFactory for reply routing.
They should not start reaper threads or auto-create reply resolvers to
prevent resource leaks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from redis import Redis


pytestmark = pytest.mark.redis_standalone


class TestSendOnlyMailbox:
    """Tests for _send_only flag behavior."""

    @pytest.fixture
    def mock_redis_client(self) -> MagicMock:
        """Create a mock Redis client that passes script registration."""
        client = MagicMock(spec=["register_script", "ping"])
        # register_script returns a callable that we won't use
        client.register_script.return_value = MagicMock()
        return client

    def test_send_only_mailbox_does_not_start_reaper_thread(
        self, mock_redis_client: MagicMock
    ) -> None:
        """Send-only mailboxes should not start a reaper thread."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mb: RedisMailbox[str, None] = RedisMailbox(
            name="test-send-only",
            client=mock_redis_client,
            _send_only=True,
        )

        # Reaper thread should not be started
        assert mb._reaper_thread is None

        # Clean up (should be a no-op for send-only)
        mb.close()

    def test_send_only_mailbox_does_not_auto_create_resolver(
        self, mock_redis_client: MagicMock
    ) -> None:
        """Send-only mailboxes should not auto-create a reply resolver."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mb: RedisMailbox[str, None] = RedisMailbox(
            name="test-send-only",
            client=mock_redis_client,
            _send_only=True,
        )

        # Reply resolver should remain None
        assert mb.reply_resolver is None

        mb.close()

    def test_regular_mailbox_starts_reaper_thread(
        self, mock_redis_client: MagicMock
    ) -> None:
        """Regular mailboxes should start a reaper thread."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mb: RedisMailbox[str, None] = RedisMailbox(
            name="test-regular",
            client=mock_redis_client,
            _send_only=False,
        )

        try:
            # Reaper thread should be started
            assert mb._reaper_thread is not None
            assert mb._reaper_thread.is_alive()
        finally:
            mb.close()

    def test_regular_mailbox_auto_creates_resolver(
        self, mock_redis_client: MagicMock
    ) -> None:
        """Regular mailboxes should auto-create a reply resolver."""
        from weakincentives.contrib.mailbox import RedisMailbox

        mb: RedisMailbox[str, None] = RedisMailbox(
            name="test-regular",
            client=mock_redis_client,
            _send_only=False,
        )

        try:
            # Reply resolver should be auto-created
            assert mb.reply_resolver is not None
        finally:
            mb.close()


class TestRedisMailboxFactory:
    """Tests for RedisMailboxFactory send-only behavior."""

    @pytest.fixture
    def mock_redis_client(self) -> MagicMock:
        """Create a mock Redis client that passes script registration."""
        client = MagicMock(spec=["register_script", "ping"])
        client.register_script.return_value = MagicMock()
        return client

    def test_factory_creates_send_only_mailboxes(
        self, mock_redis_client: MagicMock
    ) -> None:
        """Factory.create() should produce send-only mailboxes."""
        from weakincentives.contrib.mailbox import RedisMailboxFactory

        factory: RedisMailboxFactory[str] = RedisMailboxFactory(
            client=mock_redis_client,
        )

        mailbox = factory.create("reply-queue")

        # Factory-created mailboxes are send-only
        assert mailbox._send_only is True
        assert mailbox._reaper_thread is None
        assert mailbox.reply_resolver is None

        mailbox.close()

    def test_factory_mailbox_can_send(self, redis_client: Redis[bytes]) -> None:
        """Factory-created send-only mailboxes can still send messages."""
        from weakincentives.contrib.mailbox import RedisMailbox, RedisMailboxFactory

        factory: RedisMailboxFactory[str] = RedisMailboxFactory(
            client=redis_client,
        )

        send_only = factory.create("reply-queue")
        receiver = RedisMailbox[str, None](
            name="reply-queue",
            client=redis_client,
        )

        try:
            # Send via send-only mailbox
            msg_id = send_only.send("hello")
            assert msg_id is not None

            # Receive on regular mailbox
            msgs = receiver.receive(wait_time_seconds=1)
            assert len(msgs) == 1
            assert msgs[0].body == "hello"
            msgs[0].acknowledge()
        finally:
            send_only.close()
            receiver.close()
            receiver.purge()
