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

"""Redis mailbox test fixtures."""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from redis import Redis

    from weakincentives.contrib.mailbox import RedisMailbox


@pytest.fixture
def redis_client() -> Generator[Redis[bytes], None, None]:
    """Fresh Redis connection for each test."""
    try:
        from redis import Redis
    except ImportError:
        pytest.skip("redis package not installed")

    client: Redis[bytes] = Redis(host="localhost", port=6379, db=15)
    try:
        client.ping()
    except Exception:
        pytest.skip("Redis not available at localhost:6379")

    yield client
    client.flushdb()
    client.close()


@pytest.fixture
def mailbox(
    redis_client: Redis[bytes],
) -> Generator[RedisMailbox[Any, Any], None, None]:
    """Fresh RedisMailbox for each test."""
    from weakincentives.contrib.mailbox import RedisMailbox

    mb: RedisMailbox[Any, Any] = RedisMailbox(
        name=f"test-{uuid4().hex[:8]}",
        client=redis_client,
        reaper_interval=0.1,  # Fast reaper for testing
    )
    yield mb
    mb.close()
    mb.purge()
