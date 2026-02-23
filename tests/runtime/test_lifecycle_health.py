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

"""Tests for HealthServer and LoopGroup health/watchdog integration."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Self

import pytest

from weakincentives.runtime import (
    LoopGroup,
    ShutdownCoordinator,
    wait_until,
)

if TYPE_CHECKING:
    from weakincentives.runtime.watchdog import Heartbeat


class _MockRunnable:
    """Mock implementation of Runnable for testing LoopGroup."""

    def __init__(self, *, run_delay: float = 0.0) -> None:
        self._run_delay = run_delay
        self._shutdown_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self.run_called = False
        self.shutdown_called = False

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        del max_iterations, visibility_timeout, wait_time_seconds
        with self._lock:
            self._running = True
        self.run_called = True

        # Simulate work until shutdown
        while not self._shutdown_event.is_set():
            time.sleep(0.01)
            if self._run_delay > 0:
                time.sleep(self._run_delay)
                break

        with self._lock:
            self._running = False

    def shutdown(self, *, timeout: float = 30.0) -> bool:
        self.shutdown_called = True
        self._shutdown_event.set()
        return wait_until(lambda: not self.running, timeout=timeout)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        _ = self.shutdown()


@pytest.fixture
def reset_coordinator() -> None:
    """Reset ShutdownCoordinator singleton before and after each test."""
    ShutdownCoordinator.reset()
    yield
    ShutdownCoordinator.reset()


# =============================================================================
# HealthServer Tests
# =============================================================================


def test_health_server_liveness_endpoint() -> None:
    """HealthServer returns 200 for /health/live."""
    import json
    import urllib.request

    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0)  # OS-assigned port
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        server.stop()


def test_health_server_readiness_endpoint_healthy() -> None:
    """HealthServer returns 200 for /health/ready when check passes."""
    import json
    import urllib.request

    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0, readiness_check=lambda: True)
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        server.stop()


def test_health_server_readiness_endpoint_unhealthy() -> None:
    """HealthServer returns 503 for /health/ready when check fails."""
    import urllib.error
    import urllib.request

    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0, readiness_check=lambda: False)
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
            msg = "Expected HTTPError with 503 status"
            raise AssertionError(msg)
        except urllib.error.HTTPError as e:
            assert e.code == 503
    finally:
        server.stop()


def test_health_server_404_for_unknown_path() -> None:
    """HealthServer returns 404 for unknown paths."""
    import urllib.error
    import urllib.request

    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0)
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/unknown")
            msg = "Expected HTTPError with 404 status"
            raise AssertionError(msg)
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()


def test_health_server_address_none_when_not_started() -> None:
    """HealthServer.address is None before start."""
    from weakincentives.runtime import HealthServer

    server = HealthServer(port=0)
    assert server.address is None


def test_health_server_start_idempotent() -> None:
    """HealthServer.start() is idempotent."""
    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0)
    server.start()

    try:
        addr1 = server.address
        server.start()  # Should be no-op
        addr2 = server.address
        assert addr1 == addr2
    finally:
        server.stop()


def test_health_server_stop_idempotent() -> None:
    """HealthServer.stop() is idempotent."""
    from weakincentives.runtime import HealthServer

    server = HealthServer(host="127.0.0.1", port=0)
    server.start()
    server.stop()
    server.stop()  # Should be no-op, no error
    assert server.address is None


def test_health_server_dynamic_readiness_check() -> None:
    """HealthServer readiness check is evaluated dynamically."""
    import json
    import urllib.error
    import urllib.request

    from weakincentives.runtime import HealthServer

    state = {"ready": False}
    server = HealthServer(
        host="127.0.0.1", port=0, readiness_check=lambda: state["ready"]
    )
    server.start()

    try:
        _, port = server.address  # type: ignore[misc]

        # Initially not ready
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
            msg = "Expected HTTPError with 503 status"
            raise AssertionError(msg)
        except urllib.error.HTTPError as e:
            assert e.code == 503

        # Now ready
        state["ready"] = True
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        server.stop()


# =============================================================================
# LoopGroup Health Server Integration Tests
# =============================================================================


def test_loop_group_starts_health_server(reset_coordinator: None) -> None:
    """LoopGroup starts health server when health_port is set."""
    import json
    import urllib.request

    _ = reset_coordinator
    loops = [_MockRunnable(run_delay=0.2)]
    group = LoopGroup(loops=loops, health_port=0, health_host="127.0.0.1")

    # Run in background thread
    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        # Wait for server to start
        time.sleep(0.05)

        # Health server should be running
        assert group._health_server is not None
        _, port = group._health_server.address  # type: ignore[misc]

        # Liveness should work
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)


def test_loop_group_readiness_reflects_loop_state(reset_coordinator: None) -> None:
    """LoopGroup readiness endpoint reflects loop running state."""
    import json
    import urllib.request

    _ = reset_coordinator
    loops = [_MockRunnable()]
    group = LoopGroup(loops=loops, health_port=0, health_host="127.0.0.1")

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        # Wait for server and loops to start
        time.sleep(0.1)

        assert group._health_server is not None
        _, port = group._health_server.address  # type: ignore[misc]

        # Readiness should be healthy when loops are running
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data == {"status": "healthy"}
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)


def test_loop_group_no_health_server_without_port(reset_coordinator: None) -> None:
    """LoopGroup does not start health server when health_port is None."""
    _ = reset_coordinator
    loops = [_MockRunnable(run_delay=0.05)]
    group = LoopGroup(loops=loops)  # No health_port

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        time.sleep(0.02)
        assert group._health_server is None
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)


def test_loop_group_health_server_stops_on_shutdown(reset_coordinator: None) -> None:
    """LoopGroup stops health server on shutdown."""
    _ = reset_coordinator
    loops = [_MockRunnable()]
    group = LoopGroup(loops=loops, health_port=0, health_host="127.0.0.1")

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        time.sleep(0.05)
        assert group._health_server is not None
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)

    # After shutdown, health server should be stopped
    assert group._health_server is None


# =============================================================================
# LoopGroup Watchdog Integration Tests
# =============================================================================


class _MockRunnableWithHeartbeat(_MockRunnable):
    """Mock implementation of Runnable with heartbeat for testing watchdog."""

    def __init__(self, *, run_delay: float = 0.0) -> None:
        super().__init__(run_delay=run_delay)
        from weakincentives.runtime.watchdog import Heartbeat as HeartbeatCls

        self._heartbeat: Heartbeat = HeartbeatCls()
        self.name = "test-loop"

    @property
    def heartbeat(self) -> Heartbeat:
        return self._heartbeat

    def run(
        self,
        *,
        max_iterations: int | None = None,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> None:
        # Beat the heartbeat before calling parent run
        self._heartbeat.beat()
        super().run(
            max_iterations=max_iterations,
            visibility_timeout=visibility_timeout,
            wait_time_seconds=wait_time_seconds,
        )


def test_loop_group_starts_watchdog_with_heartbeats(reset_coordinator: None) -> None:
    """LoopGroup starts watchdog when loops have heartbeat properties."""
    _ = reset_coordinator
    loops = [_MockRunnableWithHeartbeat(run_delay=0.05)]
    group = LoopGroup(
        loops=loops,
        watchdog_threshold=60.0,
        watchdog_interval=1.0,
    )

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        time.sleep(0.02)
        # Watchdog should be started
        assert group._watchdog is not None
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)

    # Watchdog should be stopped after shutdown
    assert group._watchdog is None


def test_loop_group_watchdog_disabled_when_threshold_none(
    reset_coordinator: None,
) -> None:
    """LoopGroup does not start watchdog when watchdog_threshold is None."""
    _ = reset_coordinator
    loops = [_MockRunnableWithHeartbeat(run_delay=0.05)]
    group = LoopGroup(
        loops=loops,
        watchdog_threshold=None,  # Disable watchdog
    )

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    try:
        time.sleep(0.02)
        # Watchdog should not be started
        assert group._watchdog is None
    finally:
        group.shutdown(timeout=1.0)
        thread.join(timeout=2.0)


def test_loop_group_build_readiness_check_with_heartbeats(
    reset_coordinator: None,
) -> None:
    """LoopGroup._build_readiness_check incorporates heartbeat freshness."""
    _ = reset_coordinator

    loop = _MockRunnableWithHeartbeat()
    group = LoopGroup(
        loops=[loop],
        watchdog_threshold=0.1,  # Short threshold for testing
    )

    # Build the readiness check manually
    heartbeats = [loop._heartbeat]
    check = group._build_readiness_check(heartbeats)

    # Initially loop is not running, so check should fail
    assert check() is False

    # Simulate loop running
    loop._running = True
    loop._heartbeat.beat()

    # Now should be healthy
    assert check() is True

    # Wait for heartbeat to go stale
    time.sleep(0.15)

    # Now should be unhealthy due to stale heartbeat
    assert check() is False

    # Beat again to restore
    loop._heartbeat.beat()
    assert check() is True


def test_loop_group_build_readiness_check_without_threshold(
    reset_coordinator: None,
) -> None:
    """LoopGroup._build_readiness_check works when watchdog_threshold is None."""
    _ = reset_coordinator

    loop = _MockRunnableWithHeartbeat()
    group = LoopGroup(
        loops=[loop],
        watchdog_threshold=None,  # No threshold
    )

    # Build the readiness check manually with heartbeats
    heartbeats = [loop._heartbeat]
    check = group._build_readiness_check(heartbeats)

    # Simulate loop running
    loop._running = True

    # Should be healthy regardless of heartbeat age when threshold is None
    time.sleep(0.05)  # Let heartbeat go a bit stale
    assert check() is True

    # Even with very stale heartbeat, still healthy (no threshold check)
    time.sleep(0.1)
    assert check() is True
