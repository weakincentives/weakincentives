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

"""Tests for watchdog and health monitoring primitives."""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from weakincentives.runtime.clock import FakeClock
from weakincentives.runtime.watchdog import HealthServer, Heartbeat, Watchdog

# =============================================================================
# Heartbeat Tests
# =============================================================================


def test_heartbeat_initial_elapsed_is_small(clock: FakeClock) -> None:
    """Heartbeat.elapsed() is near zero immediately after creation."""
    hb = Heartbeat(clock=clock)
    assert hb.elapsed() < 0.1


@pytest.mark.allow_system_clock
def test_heartbeat_elapsed_increases_over_time(clock: FakeClock) -> None:
    """Heartbeat.elapsed() increases as time passes."""
    hb = Heartbeat()
    time.sleep(0.1)
    elapsed = hb.elapsed()
    assert 0.1 <= elapsed < 0.3


@pytest.mark.allow_system_clock
def test_heartbeat_beat_resets_elapsed(clock: FakeClock) -> None:
    """Heartbeat.beat() resets elapsed time to near zero."""
    hb = Heartbeat()
    time.sleep(0.1)
    assert hb.elapsed() >= 0.1

    hb.beat()
    assert hb.elapsed() < 0.05


def test_heartbeat_is_thread_safe(clock: FakeClock) -> None:
    """Heartbeat can be used from multiple threads safely."""
    hb = Heartbeat(clock=clock)
    errors: list[Exception] = []

    def beat_repeatedly() -> None:
        try:
            for _ in range(100):
                hb.beat()
                _ = hb.elapsed()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=beat_repeatedly) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# =============================================================================
# Watchdog Tests
# =============================================================================


def test_watchdog_check_heartbeats_empty_when_fresh(clock: FakeClock) -> None:
    """Watchdog._check_heartbeats returns empty list when all heartbeats are fresh."""
    hb = Heartbeat(clock=clock)
    watchdog = Watchdog([hb], stall_threshold=1.0, check_interval=0.1)

    stalled = watchdog._check_heartbeats()
    assert stalled == []


@pytest.mark.allow_system_clock
def test_watchdog_check_heartbeats_detects_stall(clock: FakeClock) -> None:
    """Watchdog._check_heartbeats detects stalled heartbeats."""
    hb = Heartbeat()
    watchdog = Watchdog([hb], stall_threshold=0.1, check_interval=0.05)

    # Wait for stall
    time.sleep(0.15)

    stalled = watchdog._check_heartbeats()
    assert len(stalled) == 1
    assert stalled[0][0] == "loop-0"
    assert stalled[0][1] > 0.1


@pytest.mark.allow_system_clock
def test_watchdog_check_heartbeats_clears_after_beat(clock: FakeClock) -> None:
    """Watchdog._check_heartbeats clears stall after beat."""
    hb = Heartbeat()
    watchdog = Watchdog([hb], stall_threshold=0.1, check_interval=0.05)

    time.sleep(0.15)
    stalled = watchdog._check_heartbeats()
    assert len(stalled) == 1

    hb.beat()
    stalled = watchdog._check_heartbeats()
    assert stalled == []


@pytest.mark.allow_system_clock
def test_watchdog_custom_loop_names(clock: FakeClock) -> None:
    """Watchdog uses custom loop names in stall reports."""
    hb = Heartbeat()
    watchdog = Watchdog(
        [hb],
        stall_threshold=0.1,
        check_interval=0.05,
        loop_names=["my-worker"],
    )

    time.sleep(0.15)
    stalled = watchdog._check_heartbeats()
    assert len(stalled) == 1
    assert stalled[0][0] == "my-worker"


@pytest.mark.allow_system_clock
def test_watchdog_multiple_heartbeats() -> None:
    """Watchdog monitors multiple heartbeats independently."""
    hb1 = Heartbeat()
    hb2 = Heartbeat()
    watchdog = Watchdog(
        [hb1, hb2],
        stall_threshold=0.1,
        check_interval=0.05,
        loop_names=["worker-1", "worker-2"],
    )

    # Wait for both to stall
    time.sleep(0.15)

    stalled = watchdog._check_heartbeats()
    assert len(stalled) == 2

    # Beat only first one
    hb1.beat()

    stalled = watchdog._check_heartbeats()
    assert len(stalled) == 1
    assert stalled[0][0] == "worker-2"


def test_watchdog_start_stop(clock: FakeClock) -> None:
    """Watchdog can be started and stopped."""
    hb = Heartbeat(clock=clock)
    watchdog = Watchdog([hb], stall_threshold=10.0, check_interval=0.01)

    watchdog.start()
    assert watchdog._thread is not None
    assert watchdog._thread.is_alive()

    watchdog.stop()
    assert watchdog._thread is None


def test_watchdog_start_is_idempotent(clock: FakeClock) -> None:
    """Multiple calls to Watchdog.start() have no effect."""
    hb = Heartbeat(clock=clock)
    watchdog = Watchdog([hb], stall_threshold=10.0, check_interval=0.01)

    watchdog.start()
    thread1 = watchdog._thread

    watchdog.start()
    thread2 = watchdog._thread

    assert thread1 is thread2

    watchdog.stop()


def test_watchdog_stop_without_start(clock: FakeClock) -> None:
    """Watchdog.stop() is safe when never started."""
    hb = Heartbeat(clock=clock)
    watchdog = Watchdog([hb], stall_threshold=10.0, check_interval=0.01)

    # Should not raise when stop is called without start
    watchdog.stop()
    assert watchdog._thread is None


def test_watchdog_stall_threshold_property(clock: FakeClock) -> None:
    """Watchdog.stall_threshold property returns configured value."""
    hb = Heartbeat(clock=clock)
    watchdog = Watchdog([hb], stall_threshold=123.0)
    assert watchdog.stall_threshold == 123.0


@pytest.mark.allow_system_clock
def test_watchdog_terminate_calls_sigkill() -> None:
    """Watchdog._terminate calls os.kill with SIGKILL."""
    hb = Heartbeat()
    watchdog = Watchdog([hb], stall_threshold=0.1)

    with patch("os.kill") as mock_kill:
        with patch("os.getpid", return_value=12345):
            watchdog._terminate([("test-loop", 1.0)])
            mock_kill.assert_called_once()
            call_args = mock_kill.call_args[0]
            assert call_args[0] == 12345
            # SIGKILL = 9
            assert call_args[1] == 9


def test_watchdog_terminate_logs_diagnostics(clock: FakeClock) -> None:
    """Watchdog._terminate logs critical diagnostics before termination."""
    hb = Heartbeat(clock=clock)
    watchdog = Watchdog([hb], stall_threshold=60.0)

    with patch("os.kill"):
        with patch("weakincentives.runtime.watchdog.logger.critical") as mock_log:
            watchdog._terminate([("my-loop", 65.2)])

            # Should log about the stalled loop
            assert mock_log.call_count >= 2


# =============================================================================
# HealthServer Tests
# =============================================================================


def test_health_server_liveness_returns_200() -> None:
    """HealthServer /health/live returns 200."""
    server = HealthServer(port=0)  # OS assigns port
    server.start()

    try:
        assert server.address is not None
        _, port = server.address
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live")
        assert resp.status == 200
        data = resp.read()
        assert b"healthy" in data
    finally:
        server.stop()


def test_health_server_readiness_returns_200_when_ready() -> None:
    """HealthServer /health/ready returns 200 when ready."""
    server = HealthServer(port=0, readiness_check=lambda: True)
    server.start()

    try:
        assert server.address is not None
        _, port = server.address
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
        assert resp.status == 200
        data = resp.read()
        assert b"healthy" in data
    finally:
        server.stop()


def test_health_server_readiness_returns_503_when_not_ready() -> None:
    """HealthServer /health/ready returns 503 when not ready."""
    server = HealthServer(port=0, readiness_check=lambda: False)
    server.start()

    try:
        assert server.address is not None
        _, port = server.address
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
        assert exc_info.value.code == 503
    finally:
        server.stop()


def test_health_server_readiness_uses_callback(clock: FakeClock) -> None:
    """HealthServer readiness check is called for each request."""
    call_count = 0

    def check() -> bool:
        nonlocal call_count
        call_count += 1
        return True

    server = HealthServer(port=0, readiness_check=check)
    server.start()

    try:
        assert server.address is not None
        _, port = server.address

        urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")

        assert call_count == 2
    finally:
        server.stop()


def test_health_server_404_for_unknown_path() -> None:
    """HealthServer returns 404 for unknown paths."""
    server = HealthServer(port=0)
    server.start()

    try:
        assert server.address is not None
        _, port = server.address
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/unknown")
        assert exc_info.value.code == 404
    finally:
        server.stop()


def test_health_server_start_is_idempotent(clock: FakeClock) -> None:
    """Multiple calls to HealthServer.start() have no effect."""
    server = HealthServer(port=0)
    server.start()

    try:
        addr1 = server.address
        server.start()
        addr2 = server.address

        assert addr1 == addr2
    finally:
        server.stop()


def test_health_server_stop_clears_address(clock: FakeClock) -> None:
    """HealthServer.stop() clears the address property."""
    server = HealthServer(port=0)
    server.start()
    assert server.address is not None

    server.stop()
    assert server.address is None


def test_health_server_address_is_none_before_start(clock: FakeClock) -> None:
    """HealthServer.address is None before start."""
    server = HealthServer(port=0)
    assert server.address is None


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.allow_system_clock
def test_watchdog_runs_check_loop(clock: FakeClock) -> None:
    """Watchdog periodically checks heartbeats."""
    hb = Heartbeat()
    check_count = 0

    class CountingWatchdog(Watchdog):
        def _check_heartbeats(self) -> list[tuple[str, float]]:
            nonlocal check_count
            check_count += 1
            return super()._check_heartbeats()

    watchdog = CountingWatchdog([hb], stall_threshold=10.0, check_interval=0.05)
    watchdog.start()

    time.sleep(0.2)
    watchdog.stop()

    # Should have checked multiple times
    assert check_count >= 2


@pytest.mark.allow_system_clock
def test_health_server_readiness_with_heartbeat(clock: FakeClock) -> None:
    """HealthServer readiness check can incorporate heartbeat freshness."""
    hb = Heartbeat()
    threshold = 0.1

    def readiness_check() -> bool:
        return hb.elapsed() < threshold

    server = HealthServer(port=0, readiness_check=readiness_check)
    server.start()

    try:
        assert server.address is not None
        _, port = server.address

        # Initially ready
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
        assert resp.status == 200

        # Wait for heartbeat to go stale
        time.sleep(0.15)

        # Now not ready
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
        assert exc_info.value.code == 503

        # Beat to restore
        hb.beat()

        # Ready again
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health/ready")
        assert resp.status == 200
    finally:
        server.stop()
