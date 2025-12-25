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

"""Utilities for SQS integration tests using LocalStack.

This module provides context managers for running LocalStack and creating
SQS queues for integration tests.

Example::

    from sqs_utils import localstack_sqs, is_localstack_available

    if is_localstack_available():
        with localstack_sqs() as (client, queue_url):
            client.send_message(QueueUrl=queue_url, MessageBody="hello")
            response = client.receive_message(QueueUrl=queue_url)
            print(response["Messages"][0]["Body"])
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from mypy_boto3_sqs import SQSClient

# Default LocalStack endpoint
LOCALSTACK_ENDPOINT = "http://localhost:4566"


def _find_container_runtime() -> str | None:
    """Find available container runtime (podman preferred)."""
    for runtime in ("podman", "docker"):
        if shutil.which(runtime) is not None:
            return runtime
    return None


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def _wait_for_localstack(endpoint: str, timeout: float = 60.0) -> bool:
    """Wait for LocalStack to become available."""
    try:
        import boto3
    except ImportError:
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client = boto3.client(
                "sqs",
                endpoint_url=endpoint,
                region_name="us-east-1",
                aws_access_key_id="test",
                aws_secret_access_key="test",
            )
            client.list_queues()
        except Exception:
            time.sleep(0.5)
        else:
            return True
    return False


def _start_localstack_container(
    runtime: str,
    port: int,
    image: str = "localstack/localstack:latest",
) -> str:
    """Start a LocalStack container and return its ID."""
    cmd = [
        runtime,
        "run",
        "-d",
        "--rm",
        "-p",
        f"{port}:4566",
        "-e",
        "SERVICES=sqs",
        "-e",
        "DEBUG=0",
        image,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _stop_container(runtime: str, container_id: str) -> None:
    """Stop and remove a container."""
    subprocess.run(
        [runtime, "stop", container_id],
        capture_output=True,
        check=False,
    )


@contextmanager
def localstack_sqs(
    port: int | None = None,
    image: str = "localstack/localstack:latest",
) -> Iterator[tuple[SQSClient, str]]:
    """Context manager that starts LocalStack and creates an SQS queue.

    Args:
        port: Port to expose LocalStack on. If None, finds a free port.
        image: Docker image to use for LocalStack.

    Yields:
        A tuple of (SQS client, queue URL).

    Raises:
        RuntimeError: If no container runtime is available or LocalStack fails to start.
    """
    import boto3

    runtime = _find_container_runtime()
    if runtime is None:
        raise RuntimeError("No container runtime found (docker or podman required)")

    if port is None:
        port = _find_free_port()

    endpoint = f"http://localhost:{port}"
    container_id = _start_localstack_container(runtime, port, image)

    try:
        if not _wait_for_localstack(endpoint):
            raise RuntimeError(f"LocalStack failed to start on port {port}")

        client: SQSClient = boto3.client(
            "sqs",
            endpoint_url=endpoint,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )

        # Create a test queue
        queue_name = f"test-queue-{uuid4().hex[:8]}"
        response = client.create_queue(QueueName=queue_name)
        queue_url = response["QueueUrl"]

        try:
            yield client, queue_url
        finally:
            with contextlib.suppress(Exception):
                client.delete_queue(QueueUrl=queue_url)
    finally:
        _stop_container(runtime, container_id)


def is_localstack_available() -> bool:
    """Check if LocalStack integration tests can run.

    Returns True if:
    - boto3 is installed
    - A container runtime (docker/podman) is available
    """
    try:
        import boto3  # noqa: F401

        return _find_container_runtime() is not None
    except ImportError:
        return False


def skip_if_no_localstack() -> str:
    """Return skip reason if LocalStack is not available, empty string otherwise."""
    try:
        import boto3  # noqa: F401
    except ImportError:
        return "boto3 is not installed"

    if _find_container_runtime() is None:
        return "No container runtime (docker/podman) available"

    return ""


def is_localstack_running(endpoint: str = LOCALSTACK_ENDPOINT) -> bool:
    """Check if LocalStack is already running at the given endpoint."""
    try:
        import boto3

        client = boto3.client(
            "sqs",
            endpoint_url=endpoint,
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        client.list_queues()
    except Exception:
        return False
    else:
        return True


# Environment variable to control SQS tests
SQS_TESTS_ENABLED = os.environ.get("SQS_TESTS", "1") == "1"


__all__ = [
    "LOCALSTACK_ENDPOINT",
    "SQS_TESTS_ENABLED",
    "is_localstack_available",
    "is_localstack_running",
    "localstack_sqs",
    "skip_if_no_localstack",
]
