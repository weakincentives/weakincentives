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

"""Integration tests for StatsD sink with real UDP server."""

from __future__ import annotations

import socket
import threading
import time
from typing import TYPE_CHECKING

import pytest

from weakincentives.metrics import (
    AdapterCallParams,
    InMemoryMetricsCollector,
    StatsdSink,
)

if TYPE_CHECKING:
    from collections.abc import Generator


class UDPStatsdServer:
    """Simple UDP server that captures StatsD metrics for testing."""

    def __init__(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))  # Bind to random available port
        self._socket.settimeout(0.1)  # Non-blocking reads
        self._port = self._socket.getsockname()[1]
        self._running = False
        self._packets: list[bytes] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Return the port the server is listening on."""
        return self._port

    @property
    def packets(self) -> list[bytes]:
        """Return a copy of received packets."""
        with self._lock:
            return list(self._packets)

    def start(self) -> None:
        """Start the server in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._socket.close()

    def _run(self) -> None:
        """Receive loop."""
        while self._running:
            try:
                data, _ = self._socket.recvfrom(4096)
                with self._lock:
                    self._packets.append(data)
            except TimeoutError:
                continue
            except OSError:
                break

    def clear(self) -> None:
        """Clear all received packets."""
        with self._lock:
            self._packets.clear()

    def wait_for_packets(self, count: int, timeout: float = 1.0) -> bool:
        """Wait until at least count packets are received or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._packets) >= count:
                    return True
            time.sleep(0.01)
        return False


@pytest.fixture
def statsd_server() -> Generator[UDPStatsdServer]:
    """Fixture that provides a running UDP server."""
    server = UDPStatsdServer()
    server.start()
    yield server
    server.stop()


class TestStatsdIntegration:
    """Integration tests with real UDP server."""

    def test_counter_published_to_real_server(
        self, statsd_server: UDPStatsdServer
    ) -> None:
        """Counter metrics should be received by real UDP server."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="test",
        )

        sink.emit_counter("requests", 5, {"env": "test"})

        assert statsd_server.wait_for_packets(1)
        packets = statsd_server.packets
        assert len(packets) == 1
        assert packets[0] == b"test.requests:5|c|#env:test"

    def test_histogram_published_to_real_server(
        self, statsd_server: UDPStatsdServer
    ) -> None:
        """Histogram metrics should be received by real UDP server."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="test",
        )

        sink.emit_histogram("latency_ms", 150, {"tool": "read_file"})

        assert statsd_server.wait_for_packets(1)
        packets = statsd_server.packets
        assert len(packets) == 1
        assert packets[0] == b"test.latency_ms:150|ms|#tool:read_file"

    def test_gauge_published_to_real_server(
        self, statsd_server: UDPStatsdServer
    ) -> None:
        """Gauge metrics should be received by real UDP server."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="test",
        )

        sink.emit_gauge("queue_depth", 42, {"queue": "requests"})

        assert statsd_server.wait_for_packets(1)
        packets = statsd_server.packets
        assert len(packets) == 1
        assert packets[0] == b"test.queue_depth:42|g|#queue:requests"

    def test_collector_emits_tool_metrics_to_real_server(
        self, statsd_server: UDPStatsdServer
    ) -> None:
        """Collector should emit tool metrics to real UDP server."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="wink",
        )
        collector = InMemoryMetricsCollector(worker_id="worker-1", sinks=[sink])

        collector.record_tool_call("read_file", latency_ms=25, success=True)

        # Should receive latency histogram + calls counter (success doesn't emit extra)
        assert statsd_server.wait_for_packets(2)
        packets = statsd_server.packets

        # Verify packets contain expected metrics
        packet_strs = [p.decode("utf-8") for p in packets]

        # Find latency metric
        latency_packets = [p for p in packet_strs if "tool.latency_ms" in p]
        assert len(latency_packets) == 1
        assert "25|ms" in latency_packets[0]
        assert "tool:read_file" in latency_packets[0]

        # Find calls counter
        calls_packets = [p for p in packet_strs if "tool.calls" in p]
        assert len(calls_packets) == 1
        assert ":1|c" in calls_packets[0]

    def test_collector_emits_adapter_metrics_to_real_server(
        self, statsd_server: UDPStatsdServer
    ) -> None:
        """Collector should emit adapter metrics to real UDP server."""
        from weakincentives.runtime.events import TokenUsage

        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="wink",
        )
        collector = InMemoryMetricsCollector(worker_id="worker-1", sinks=[sink])

        collector.record_adapter_call(
            "openai",
            AdapterCallParams(
                render_ms=10,
                call_ms=500,
                parse_ms=5,
                tool_ms=100,
                usage=TokenUsage(input_tokens=1000, output_tokens=200),
                experiment="baseline",
            ),
        )

        # Should receive multiple metrics
        assert statsd_server.wait_for_packets(7, timeout=2.0)
        packets = statsd_server.packets
        packet_strs = [p.decode("utf-8") for p in packets]

        # Verify call latency histogram
        latency_packets = [p for p in packet_strs if "adapter.call_latency_ms" in p]
        assert len(latency_packets) == 1
        assert "500|ms" in latency_packets[0]
        assert "adapter:openai" in latency_packets[0]
        assert "experiment:baseline" in latency_packets[0]

        # Verify requests counter
        requests_packets = [p for p in packet_strs if "adapter.requests" in p]
        assert len(requests_packets) == 1
        assert ":1|c" in requests_packets[0]

        # Verify input tokens counter
        input_packets = [p for p in packet_strs if "adapter.input_tokens" in p]
        assert len(input_packets) == 1
        assert ":1000|c" in input_packets[0]

        # Verify output tokens counter
        output_packets = [p for p in packet_strs if "adapter.output_tokens" in p]
        assert len(output_packets) == 1
        assert ":200|c" in output_packets[0]

    def test_collector_emits_mailbox_metrics_to_real_server(
        self, statsd_server: UDPStatsdServer
    ) -> None:
        """Collector should emit mailbox metrics to real UDP server."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="wink",
        )
        collector = InMemoryMetricsCollector(worker_id="worker-1", sinks=[sink])

        # Record message received
        collector.record_message_received("requests", delivery_count=1, age_ms=100)

        # Wait for packets
        assert statsd_server.wait_for_packets(2)
        packets = statsd_server.packets
        packet_strs = [p.decode("utf-8") for p in packets]

        # Verify lag histogram
        lag_packets = [p for p in packet_strs if "mailbox.lag_ms" in p]
        assert len(lag_packets) == 1
        assert "100|ms" in lag_packets[0]
        assert "queue:requests" in lag_packets[0]

        # Verify received counter
        received_packets = [p for p in packet_strs if "mailbox.received" in p]
        assert len(received_packets) == 1
        assert ":1|c" in received_packets[0]

        # Test queue depth gauge
        statsd_server.clear()
        collector.record_queue_depth("requests", depth=42)

        assert statsd_server.wait_for_packets(1)
        depth_packets = [
            p.decode("utf-8") for p in statsd_server.packets if b"mailbox.depth" in p
        ]
        assert len(depth_packets) == 1
        assert "42|g" in depth_packets[0]

    def test_sample_rate_affects_metric_format(
        self, statsd_server: UDPStatsdServer
    ) -> None:
        """Sample rate should be included in metric format."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="test",
            sample_rate=0.5,
        )

        sink.emit_counter("requests", 1, {})

        assert statsd_server.wait_for_packets(1)
        packets = statsd_server.packets
        assert len(packets) == 1
        assert b"|@0.5" in packets[0]

    def test_multiple_tags_sorted_correctly(
        self, statsd_server: UDPStatsdServer
    ) -> None:
        """Multiple tags should be sorted alphabetically."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="test",
        )

        sink.emit_counter("requests", 1, {"zebra": "z", "apple": "a", "mango": "m"})

        assert statsd_server.wait_for_packets(1)
        packets = statsd_server.packets
        assert len(packets) == 1
        # Tags should be sorted alphabetically
        assert b"|#apple:a,mango:m,zebra:z" in packets[0]

    def test_rapid_metrics_emission(self, statsd_server: UDPStatsdServer) -> None:
        """Rapid metric emission should work correctly."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="test",
        )
        collector = InMemoryMetricsCollector(sinks=[sink])

        # Emit many metrics rapidly
        for i in range(100):
            collector.record_tool_call(f"tool_{i % 10}", latency_ms=i, success=True)

        # Wait for all packets (2 per call: latency + calls counter)
        assert statsd_server.wait_for_packets(200, timeout=5.0)
        assert len(statsd_server.packets) >= 200

    def test_close_sink_cleanup(self, statsd_server: UDPStatsdServer) -> None:
        """Closing sink should clean up resources properly."""
        sink = StatsdSink(
            host="127.0.0.1",
            port=statsd_server.port,
            prefix="test",
        )

        # Emit to create socket
        sink.emit_counter("test", 1, {})
        assert statsd_server.wait_for_packets(1)

        # Close should work cleanly
        sink.close()

        # Emitting after close should still work (creates new socket)
        sink.emit_counter("test2", 2, {})
        assert statsd_server.wait_for_packets(2)
