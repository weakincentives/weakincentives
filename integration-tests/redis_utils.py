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

"""Utilities for starting and stopping Redis instances for integration tests.

This module provides context managers for running Redis in standalone mode
and Redis Cluster mode using Docker/Podman containers.

Example::

    from redis_utils import redis_standalone, redis_cluster

    # Standalone Redis
    with redis_standalone() as client:
        client.set("key", "value")
        assert client.get("key") == b"value"

    # Redis Cluster
    with redis_cluster() as client:
        client.set("key", "value")
        assert client.get("key") == b"value"
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis
    from redis.cluster import RedisCluster


@dataclass(frozen=True, slots=True)
class RedisContainer:
    """Information about a running Redis container."""

    container_id: str
    host: str
    port: int
    container_runtime: str  # "docker" or "podman"


def _find_container_runtime() -> str | None:
    """Find available container runtime (podman preferred)."""
    # Prefer podman as it doesn't require a daemon
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


def _wait_for_redis(host: str, port: int, timeout: float = 30.0) -> bool:
    """Wait for Redis to become available."""
    try:
        from redis import Redis as RedisClient
    except ImportError:
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client = RedisClient(host=host, port=port, socket_timeout=1.0)
            if client.ping():
                client.close()
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def _wait_for_cluster_ready(host: str, port: int, timeout: float = 60.0) -> bool:
    """Wait for Redis Cluster to become ready."""
    try:
        from redis import Redis as RedisClient
    except ImportError:
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client = RedisClient(host=host, port=port, socket_timeout=1.0)
            info = client.cluster("info")
            if isinstance(info, dict) and info.get("cluster_state") == "ok":
                client.close()
                return True
            client.close()
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _start_redis_container(
    runtime: str,
    port: int,
    image: str = "redis:7-alpine",
    extra_args: list[str] | None = None,
) -> str:
    """Start a Redis container and return its ID."""
    cmd = [
        runtime,
        "run",
        "-d",
        "--rm",
        "-p",
        f"{port}:6379",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(image)

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
def redis_standalone(
    port: int | None = None,
    image: str = "redis:7-alpine",
) -> Iterator[Redis[bytes]]:
    """Context manager that starts a standalone Redis instance.

    Args:
        port: Port to expose Redis on. If None, finds a free port.
        image: Docker image to use for Redis.

    Yields:
        A connected Redis client.

    Raises:
        RuntimeError: If no container runtime is available or Redis fails to start.
    """
    from redis import Redis

    runtime = _find_container_runtime()
    if runtime is None:
        raise RuntimeError("No container runtime found (docker or podman required)")

    if port is None:
        port = _find_free_port()

    container_id = _start_redis_container(runtime, port, image)
    try:
        if not _wait_for_redis("localhost", port):
            raise RuntimeError(f"Redis failed to start on port {port}")

        client: Redis[bytes] = Redis(host="localhost", port=port)
        try:
            yield client
        finally:
            client.close()
    finally:
        _stop_container(runtime, container_id)


def _find_free_port_range(count: int) -> int:
    """Find a range of free consecutive ports."""
    base = _find_free_port()
    while True:
        all_available = True
        for offset in range(count):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("", base + offset))
                except OSError:
                    all_available = False
                    break
        if all_available:
            return base
        base += count


def _build_cluster_command(
    runtime: str,
    base_port: int,
    node_count: int,
    replicas: int,
    image: str,
) -> list[str]:
    """Build the docker/podman command for starting a Redis Cluster."""
    port_mappings = []
    for i in range(node_count):
        port_mappings.extend(["-p", f"{base_port + i}:{7000 + i}"])

    env_vars = [
        "-e",
        "IP=0.0.0.0",
        "-e",
        "INITIAL_PORT=7000",
        "-e",
        f"MASTERS={node_count // (replicas + 1)}",
        "-e",
        f"SLAVES_PER_MASTER={replicas}",
    ]

    return [runtime, "run", "-d", "--rm", *port_mappings, *env_vars, image]


@contextmanager
def redis_cluster(
    base_port: int | None = None,
    node_count: int = 6,
    replicas: int = 1,
    image: str = "grokzen/redis-cluster:7.0.10",
) -> Iterator[RedisCluster[bytes]]:
    """Context manager that starts a Redis Cluster.

    This uses a pre-built Redis Cluster image that handles cluster setup.
    The cluster runs with 3 masters and 3 replicas by default (6 nodes).

    Args:
        base_port: Base port for cluster nodes. If None, finds a free range.
        node_count: Number of cluster nodes (default 6).
        replicas: Number of replicas per master (default 1).
        image: Docker image to use for Redis Cluster.

    Yields:
        A connected RedisCluster client.

    Raises:
        RuntimeError: If no container runtime is available or cluster fails to start.
    """
    from redis.cluster import RedisCluster

    runtime = _find_container_runtime()
    if runtime is None:
        raise RuntimeError("No container runtime found (docker or podman required)")

    if base_port is None:
        base_port = _find_free_port_range(node_count)

    cmd = _build_cluster_command(runtime, base_port, node_count, replicas, image)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    container_id = result.stdout.strip()

    try:
        if not _wait_for_cluster_ready("localhost", base_port, timeout=90.0):
            raise RuntimeError(
                f"Redis Cluster failed to start on ports {base_port}-{base_port + node_count - 1}"
            )

        startup_nodes = [{"host": "localhost", "port": base_port}]
        client: RedisCluster[bytes] = RedisCluster(
            startup_nodes=startup_nodes,
            decode_responses=False,
        )
        try:
            yield client
        finally:
            client.close()
    finally:
        _stop_container(runtime, container_id)


def is_redis_available() -> bool:
    """Check if Redis integration tests can run.

    Returns True if:
    - redis-py is installed
    - A container runtime (docker/podman) is available
    """
    try:
        import redis  # noqa: F401

        return _find_container_runtime() is not None
    except ImportError:
        return False


def skip_if_no_redis() -> str:
    """Return skip reason if Redis is not available, empty string otherwise."""
    try:
        import redis  # noqa: F401
    except ImportError:
        return "redis-py is not installed"

    if _find_container_runtime() is None:
        return "No container runtime (docker/podman) available"

    return ""


# Environment variable to control cluster tests (they're slower)
REDIS_CLUSTER_TESTS_ENABLED = os.environ.get("REDIS_CLUSTER_TESTS", "0") == "1"


__all__ = [
    "REDIS_CLUSTER_TESTS_ENABLED",
    "RedisContainer",
    "is_redis_available",
    "redis_cluster",
    "redis_standalone",
    "skip_if_no_redis",
]
