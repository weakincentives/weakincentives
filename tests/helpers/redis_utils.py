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

"""Redis test utilities for starting/stopping Redis standalone and cluster.

This module provides utilities to manage Redis instances for integration testing.
It supports both standalone Redis and Redis Cluster configurations.

Example usage:

    # Standalone Redis
    with redis_standalone() as client:
        client.set("key", "value")
        assert client.get("key") == b"value"

    # Redis Cluster
    with redis_cluster() as client:
        client.set("key", "value")
        assert client.get("key") == b"value"

    # Manual lifecycle management
    server = RedisStandalone(port=6380)
    server.start()
    try:
        client = server.client()
        # ... use client ...
    finally:
        server.stop()
"""

from __future__ import annotations

import atexit
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis
    from redis.cluster import RedisCluster


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for a port to become available."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _redis_server_available() -> bool:
    """Check if redis-server is available on the system."""
    return shutil.which("redis-server") is not None


def _redis_cli_available() -> bool:
    """Check if redis-cli is available on the system."""
    return shutil.which("redis-cli") is not None


@dataclass
class RedisStandalone:
    """Manages a standalone Redis server instance for testing.

    Attributes:
        port: Port to run Redis on. If 0, a free port is chosen.
        host: Host to bind to. Defaults to 127.0.0.1.
        data_dir: Directory for Redis data. If None, uses a temp directory.
    """

    port: int = 0
    host: str = "127.0.0.1"
    data_dir: str | None = None

    _process: subprocess.Popen[bytes] | None = field(
        default=None, init=False, repr=False
    )
    _temp_dir: str | None = field(default=None, init=False, repr=False)
    _actual_port: int = field(default=0, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def start(self, timeout: float = 10.0) -> None:
        """Start the Redis server.

        Args:
            timeout: Maximum seconds to wait for Redis to start.

        Raises:
            RuntimeError: If Redis server cannot be started.
            FileNotFoundError: If redis-server is not installed.
        """
        if self._started:
            return

        if not _redis_server_available():
            raise FileNotFoundError(
                "redis-server not found. Install Redis to run integration tests."
            )

        # Choose port if not specified
        self._actual_port = self.port if self.port > 0 else _find_free_port()

        # Create temp directory for data if needed
        if self.data_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="redis-test-")
            data_dir = Path(self._temp_dir)
        else:
            data_dir = Path(self.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

        # Write minimal config
        config_path = data_dir / "redis.conf"
        with config_path.open("w") as f:
            f.write(f"port {self._actual_port}\n")
            f.write(f"bind {self.host}\n")
            f.write(f"dir {data_dir}\n")
            f.write("appendonly no\n")
            f.write('save ""\n')  # Disable RDB snapshots
            f.write("loglevel warning\n")
            f.write("daemonize no\n")

        # Start Redis
        self._process = subprocess.Popen(
            ["redis-server", str(config_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Register cleanup
        atexit.register(self.stop)

        # Wait for Redis to be ready
        if not _wait_for_port(self.host, self._actual_port, timeout):
            self.stop()
            raise RuntimeError(
                f"Redis failed to start on {self.host}:{self._actual_port}"
            )

        self._started = True

    def stop(self) -> None:
        """Stop the Redis server and clean up resources."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        self._started = False

    def client(self) -> Redis[bytes]:
        """Get a Redis client connected to this server.

        Returns:
            A Redis client instance.

        Raises:
            RuntimeError: If the server is not running.
        """
        if not self._started:
            raise RuntimeError("Redis server not started. Call start() first.")

        from redis import Redis

        return Redis(host=self.host, port=self._actual_port, decode_responses=False)

    @property
    def url(self) -> str:
        """Get the Redis URL for this server."""
        return f"redis://{self.host}:{self._actual_port}"

    @property
    def actual_port(self) -> int:
        """Get the actual port Redis is running on."""
        return self._actual_port


@dataclass
class RedisClusterNode:
    """Represents a single node in a Redis Cluster."""

    port: int
    host: str = "127.0.0.1"
    cluster_port: int = 0

    _process: subprocess.Popen[bytes] | None = field(
        default=None, init=False, repr=False
    )
    _data_dir: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cluster_port == 0:
            self.cluster_port = self.port + 10000


@dataclass
class RedisClusterManager:
    """Manages a Redis Cluster for testing.

    Creates a minimal 3-node cluster suitable for testing.
    Each node is both a master (no replicas in test mode).

    Attributes:
        base_port: Starting port for cluster nodes. If 0, free ports are chosen.
        host: Host to bind all nodes to. Defaults to 127.0.0.1.
        num_nodes: Number of nodes in the cluster (minimum 3).
    """

    base_port: int = 0
    host: str = "127.0.0.1"
    num_nodes: int = 3

    _nodes: list[RedisClusterNode] = field(default_factory=list, init=False, repr=False)
    _temp_dir: str | None = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def start(self, timeout: float = 30.0) -> None:
        """Start the Redis Cluster.

        Args:
            timeout: Maximum seconds to wait for cluster to be ready.

        Raises:
            RuntimeError: If cluster cannot be started.
            FileNotFoundError: If redis-server or redis-cli is not installed.
        """
        if self._started:
            return

        if not _redis_server_available():
            raise FileNotFoundError(
                "redis-server not found. Install Redis to run integration tests."
            )
        if not _redis_cli_available():
            raise FileNotFoundError(
                "redis-cli not found. Install Redis to run integration tests."
            )

        # Create temp directory
        self._temp_dir = tempfile.mkdtemp(prefix="redis-cluster-test-")

        # Determine ports
        if self.base_port > 0:
            ports = [self.base_port + i for i in range(self.num_nodes)]
        else:
            ports = [_find_free_port() for _ in range(self.num_nodes)]

        # Create and start nodes
        for i, port in enumerate(ports):
            node = RedisClusterNode(port=port, host=self.host)
            node_dir = Path(self._temp_dir) / f"node-{i}"
            node_dir.mkdir(parents=True, exist_ok=True)
            node._data_dir = str(node_dir)

            # Write cluster config
            config_path = node_dir / "redis.conf"
            with config_path.open("w") as f:
                f.write(f"port {port}\n")
                f.write(f"bind {self.host}\n")
                f.write(f"dir {node_dir}\n")
                f.write("cluster-enabled yes\n")
                f.write(f"cluster-config-file nodes-{port}.conf\n")
                f.write("cluster-node-timeout 5000\n")
                f.write("appendonly no\n")
                f.write('save ""\n')
                f.write("loglevel warning\n")
                f.write("daemonize no\n")
                # Use the port+10000 convention for cluster bus
                f.write(f"cluster-port {node.cluster_port}\n")

            # Start node
            node._process = subprocess.Popen(
                ["redis-server", str(config_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._nodes.append(node)

        # Register cleanup
        atexit.register(self.stop)

        # Wait for all nodes to be ready
        for node in self._nodes:
            if not _wait_for_port(node.host, node.port, timeout / 2):
                self.stop()
                raise RuntimeError(f"Redis node failed to start on port {node.port}")

        # Create the cluster
        self._create_cluster(timeout)
        self._started = True

    def _create_cluster(self, timeout: float) -> None:
        """Initialize the cluster using redis-cli --cluster create."""
        node_addrs = [f"{n.host}:{n.port}" for n in self._nodes]

        # Use redis-cli to create cluster
        cmd = [
            "redis-cli",
            "--cluster",
            "create",
            *node_addrs,
            "--cluster-replicas",
            "0",  # No replicas for testing
            "--cluster-yes",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            self.stop()
            raise RuntimeError(f"Failed to create cluster: {result.stderr}")

        # Wait for cluster to be ready
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._cluster_ready():
                return
            time.sleep(0.5)

        self.stop()
        raise RuntimeError("Cluster failed to reach ready state")

    def _cluster_ready(self) -> bool:
        """Check if all cluster slots are assigned and nodes are connected."""
        try:
            from redis import Redis

            client = Redis(host=self._nodes[0].host, port=self._nodes[0].port)
            info = client.cluster_info()
            client.close()
            # Check that cluster is in "ok" state
            return info.get("cluster_state") == "ok"
        except Exception:
            return False

    def stop(self) -> None:
        """Stop all cluster nodes and clean up resources."""
        for node in self._nodes:
            if node._process is not None:
                node._process.terminate()
                try:
                    node._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    node._process.kill()
                    node._process.wait()
                node._process = None

        self._nodes.clear()

        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

        self._started = False

    def client(self) -> RedisCluster[bytes]:
        """Get a Redis Cluster client connected to this cluster.

        Returns:
            A RedisCluster client instance.

        Raises:
            RuntimeError: If the cluster is not running.
        """
        if not self._started:
            raise RuntimeError("Redis cluster not started. Call start() first.")

        from redis.cluster import RedisCluster

        startup_nodes = [{"host": node.host, "port": node.port} for node in self._nodes]
        return RedisCluster(
            startup_nodes=startup_nodes,
            decode_responses=False,
        )

    @property
    def nodes(self) -> list[tuple[str, int]]:
        """Get list of (host, port) tuples for all nodes."""
        return [(n.host, n.port) for n in self._nodes]


@contextmanager
def redis_standalone(
    port: int = 0,
    host: str = "127.0.0.1",
) -> Iterator[Redis[bytes]]:
    """Context manager that provides a temporary Redis standalone instance.

    Args:
        port: Port to run Redis on. If 0, a free port is chosen.
        host: Host to bind to. Defaults to 127.0.0.1.

    Yields:
        A Redis client connected to the temporary server.

    Raises:
        FileNotFoundError: If redis-server is not installed.

    Example:
        with redis_standalone() as client:
            client.set("key", "value")
            assert client.get("key") == b"value"
    """
    server = RedisStandalone(port=port, host=host)
    server.start()
    try:
        client = server.client()
        try:
            yield client
        finally:
            client.close()
    finally:
        server.stop()


@contextmanager
def redis_cluster(
    base_port: int = 0,
    host: str = "127.0.0.1",
    num_nodes: int = 3,
) -> Iterator[RedisCluster[bytes]]:
    """Context manager that provides a temporary Redis Cluster.

    Args:
        base_port: Starting port for cluster nodes. If 0, free ports are chosen.
        host: Host to bind all nodes to. Defaults to 127.0.0.1.
        num_nodes: Number of nodes in the cluster (minimum 3).

    Yields:
        A RedisCluster client connected to the temporary cluster.

    Raises:
        FileNotFoundError: If redis-server or redis-cli is not installed.

    Example:
        with redis_cluster() as client:
            client.set("key", "value")
            assert client.get("key") == b"value"
    """
    manager = RedisClusterManager(base_port=base_port, host=host, num_nodes=num_nodes)
    manager.start()
    try:
        client = manager.client()
        try:
            yield client
        finally:
            client.close()
    finally:
        manager.stop()


def skip_if_no_redis() -> bool:
    """Check if Redis is available for testing.

    Returns:
        True if redis-server is available, False otherwise.

    Use with pytest.mark.skipif:
        @pytest.mark.skipif(not skip_if_no_redis(), reason="Redis not installed")
    """
    return _redis_server_available()


def skip_if_no_redis_cluster() -> bool:
    """Check if Redis Cluster testing is available.

    Returns:
        True if both redis-server and redis-cli are available, False otherwise.
    """
    return _redis_server_available() and _redis_cli_available()


__all__ = [
    "RedisClusterManager",
    "RedisStandalone",
    "redis_cluster",
    "redis_standalone",
    "skip_if_no_redis",
    "skip_if_no_redis_cluster",
]
