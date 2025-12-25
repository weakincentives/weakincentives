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

"""Redis and SQS mailbox test fixtures."""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from mypy_boto3_sqs import SQSClient
    from redis import Redis

    from weakincentives.contrib.mailbox import RedisMailbox, SQSMailbox


# LocalStack default endpoint
LOCALSTACK_ENDPOINT = "http://localhost:4566"


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
) -> Generator[RedisMailbox[Any], None, None]:
    """Fresh RedisMailbox for each test."""
    from weakincentives.contrib.mailbox import RedisMailbox

    mb: RedisMailbox[Any] = RedisMailbox(
        name=f"test-{uuid4().hex[:8]}",
        client=redis_client,
        reaper_interval=0.1,  # Fast reaper for testing
    )
    yield mb
    mb.close()
    mb.purge()


# =============================================================================
# SQS / LocalStack fixtures
# =============================================================================


@pytest.fixture
def sqs_client() -> Generator[SQSClient, None, None]:
    """Create an SQS client connected to LocalStack."""
    try:
        import boto3
    except ImportError:
        pytest.skip("boto3 package not installed")

    client: SQSClient = boto3.client(
        "sqs",
        endpoint_url=LOCALSTACK_ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    # Verify LocalStack is running
    try:
        client.list_queues()
    except Exception:
        pytest.skip("LocalStack not available at localhost:4566")

    yield client


@pytest.fixture
def sqs_queue_url(sqs_client: SQSClient) -> Generator[str, None, None]:
    """Create a fresh SQS queue for each test."""
    queue_name = f"test-queue-{uuid4().hex[:8]}"

    response = sqs_client.create_queue(QueueName=queue_name)
    queue_url = response["QueueUrl"]

    yield queue_url

    # Cleanup - delete the queue
    with contextlib.suppress(Exception):
        sqs_client.delete_queue(QueueUrl=queue_url)


@pytest.fixture
def sqs_mailbox(
    sqs_client: SQSClient,
    sqs_queue_url: str,
) -> Generator[SQSMailbox[Any], None, None]:
    """Fresh SQSMailbox for each test."""
    from weakincentives.contrib.mailbox import SQSMailbox

    mb: SQSMailbox[Any] = SQSMailbox(
        queue_url=sqs_queue_url,
        client=sqs_client,
    )
    yield mb
    mb.close()
