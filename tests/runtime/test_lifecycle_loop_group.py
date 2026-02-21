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

"""Tests for LoopGroup, including health server and watchdog integration."""

from __future__ import annotations

import threading
import time

from tests.runtime.conftest import (
    LifecycleMockRunnable,
    LifecycleMockRunnableWithHeartbeat,
)
from weakincentives.runtime import (
    LoopGroup,
    ShutdownCoordinator,
)

# =============================================================================
# LoopGroup Tests
# =============================================================================


def test_loop_group_runs_all_loops(reset_coordinator: None) -> None:
    """LoopGroup.run() starts all loops."""
    _ = reset_coordinator
    loops = [LifecycleMockRunnable(run_delay=0.1) for _ in range(3)]
    group = LoopGroup(loops=loops)

    # Run in background thread
    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    # Wait a bit for loops to start
    time.sleep(0.05)

    # Trigger shutdown
    group.shutdown(timeout=1.0)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    for loop in loops:
        assert loop.run_called


def test_loop_group_shutdown_stops_all_loops(reset_coordinator: None) -> None:
    """LoopGroup.shutdown() stops all loops."""
    _ = reset_coordinator
    loops = [LifecycleMockRunnable() for _ in range(3)]
    group = LoopGroup(loops=loops)

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    time.sleep(0.05)
    result = group.shutdown(timeout=1.0)
    thread.join(timeout=2.0)

    assert result is True
    for loop in loops:
        assert loop.shutdown_called
        assert not loop.running


def test_loop_group_context_manager(reset_coordinator: None) -> None:
    """LoopGroup supports context manager protocol."""
    _ = reset_coordinator
    loops = [LifecycleMockRunnable() for _ in range(2)]

    with LoopGroup(loops=loops) as group:
        thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
        thread.start()
        time.sleep(0.05)

    # Context exit should trigger shutdown
    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_loop_group_with_signals(reset_coordinator: None) -> None:
    """LoopGroup integrates with ShutdownCoordinator (main thread)."""
    _ = reset_coordinator

    # Pre-install coordinator to avoid signal handler issues
    coordinator = ShutdownCoordinator.install()

    loops = [LifecycleMockRunnable(run_delay=0.05) for _ in range(2)]
    group = LoopGroup(loops=loops)

    # Run in background thread with install_signals=True
    # The coordinator is already installed, so it will just register
    thread = threading.Thread(target=group.run, kwargs={"install_signals": True})
    thread.start()

    time.sleep(0.02)

    # Trigger via coordinator
    coordinator.trigger()

    thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_loop_group_trigger_shutdown_returns_result(reset_coordinator: None) -> None:
    """LoopGroup._trigger_shutdown returns shutdown result."""
    _ = reset_coordinator
    loops = [LifecycleMockRunnable(run_delay=0.05) for _ in range(2)]
    group = LoopGroup(loops=loops)

    thread = threading.Thread(target=group.run, kwargs={"install_signals": False})
    thread.start()

    time.sleep(0.02)

    # Call _trigger_shutdown directly
    result = group._trigger_shutdown()

    thread.join(timeout=2.0)

    assert result is True
    assert not thread.is_alive()


# =============================================================================
# LoopGroup Health Server Integration Tests
# =============================================================================


def test_loop_group_starts_health_server(reset_coordinator: None) -> None:
    """LoopGroup starts health server when health_port is set."""
    import json
    import urllib.request

    _ = reset_coordinator
    loops = [LifecycleMockRunnable(run_delay=0.2)]
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
    loops = [LifecycleMockRunnable()]
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
    loops = [LifecycleMockRunnable(run_delay=0.05)]
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
    loops = [LifecycleMockRunnable()]
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


def test_loop_group_starts_watchdog_with_heartbeats(reset_coordinator: None) -> None:
    """LoopGroup starts watchdog when loops have heartbeat properties."""
    _ = reset_coordinator
    loops = [LifecycleMockRunnableWithHeartbeat(run_delay=0.05)]
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
    loops = [LifecycleMockRunnableWithHeartbeat(run_delay=0.05)]
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

    loop = LifecycleMockRunnableWithHeartbeat()
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

    loop = LifecycleMockRunnableWithHeartbeat()
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
