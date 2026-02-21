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

"""Tests for HealthServer."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from weakincentives.runtime import HealthServer

# =============================================================================
# HealthServer Tests
# =============================================================================


def test_health_server_liveness_endpoint() -> None:
    """HealthServer returns 200 for /health/live."""
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
    server = HealthServer(port=0)
    assert server.address is None


def test_health_server_start_idempotent() -> None:
    """HealthServer.start() is idempotent."""
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
    server = HealthServer(host="127.0.0.1", port=0)
    server.start()
    server.stop()
    server.stop()  # Should be no-op, no error
    assert server.address is None


def test_health_server_dynamic_readiness_check() -> None:
    """HealthServer readiness check is evaluated dynamically."""
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
