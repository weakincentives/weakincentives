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

"""Tests for metrics sinks."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from weakincentives.metrics import DebugSink, StatsdSink


class TestStatsdSink:
    """Tests for StatsdSink."""

    def test_statsd_sink_init(self) -> None:
        """StatsdSink should initialize with defaults."""
        sink = StatsdSink()
        assert sink._host == "localhost"
        assert sink._port == 8125
        assert sink._prefix == "wink"
        assert sink._sample_rate == 1.0

    def test_statsd_sink_custom_config(self) -> None:
        """StatsdSink should accept custom config."""
        sink = StatsdSink(
            host="statsd.example.com",
            port=9125,
            prefix="myapp",
            sample_rate=0.5,
        )
        assert sink._host == "statsd.example.com"
        assert sink._port == 9125
        assert sink._prefix == "myapp"
        assert sink._sample_rate == 0.5

    def test_format_tags_empty(self) -> None:
        """Empty tags should return empty string."""
        sink = StatsdSink()
        assert sink._format_tags({}) == ""

    def test_format_tags(self) -> None:
        """Tags should be formatted in DogStatsD format."""
        sink = StatsdSink()
        tags = {"env": "prod", "service": "api"}
        result = sink._format_tags(tags)
        # Tags should be sorted and comma-separated
        assert result == "|#env:prod,service:api"

    @patch("socket.socket")
    def test_emit_counter(self, mock_socket_class: MagicMock) -> None:
        """emit_counter should send StatsD counter format."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sink = StatsdSink(prefix="test")
        sink.emit_counter("requests", 5, {"env": "prod"})

        mock_socket.sendto.assert_called_once()
        data, addr = mock_socket.sendto.call_args[0]
        assert data == b"test.requests:5|c|#env:prod"
        assert addr == ("localhost", 8125)

    @patch("socket.socket")
    def test_emit_histogram(self, mock_socket_class: MagicMock) -> None:
        """emit_histogram should send StatsD timing format."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sink = StatsdSink(prefix="test")
        sink.emit_histogram("latency_ms", 100, {"tool": "read"})

        mock_socket.sendto.assert_called_once()
        data, _ = mock_socket.sendto.call_args[0]
        assert data == b"test.latency_ms:100|ms|#tool:read"

    @patch("socket.socket")
    def test_emit_gauge(self, mock_socket_class: MagicMock) -> None:
        """emit_gauge should send StatsD gauge format."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sink = StatsdSink(prefix="test")
        sink.emit_gauge("queue_depth", 42, {"queue": "requests"})

        mock_socket.sendto.assert_called_once()
        data, _ = mock_socket.sendto.call_args[0]
        assert data == b"test.queue_depth:42|g|#queue:requests"

    @patch("socket.socket")
    def test_emit_counter_with_sample_rate(self, mock_socket_class: MagicMock) -> None:
        """emit_counter should include sample rate when < 1.0."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sink = StatsdSink(prefix="test", sample_rate=0.5)
        sink.emit_counter("requests", 1, {})

        data, _ = mock_socket.sendto.call_args[0]
        assert b"|@0.5" in data

    @patch("socket.socket")
    def test_emit_histogram_with_sample_rate(
        self, mock_socket_class: MagicMock
    ) -> None:
        """emit_histogram should include sample rate when < 1.0."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sink = StatsdSink(prefix="test", sample_rate=0.5)
        sink.emit_histogram("latency", 100, {})

        data, _ = mock_socket.sendto.call_args[0]
        assert b"|@0.5" in data

    @patch("socket.socket")
    def test_emit_handles_network_error(self, mock_socket_class: MagicMock) -> None:
        """emit should handle network errors gracefully."""
        mock_socket = MagicMock()
        mock_socket.sendto.side_effect = OSError("Network unreachable")
        mock_socket_class.return_value = mock_socket

        sink = StatsdSink()
        # Should not raise
        sink.emit_counter("test", 1, {})

    def test_close(self) -> None:
        """close() should close the socket."""
        sink = StatsdSink()
        # Force socket creation
        _ = sink._get_socket()
        assert sink._socket is not None

        sink.close()
        assert sink._socket is None

    def test_close_without_socket(self) -> None:
        """close() should be safe to call without socket."""
        sink = StatsdSink()
        assert sink._socket is None
        sink.close()  # Should not raise
        assert sink._socket is None

    def test_flush_noop(self) -> None:
        """flush() should be a no-op for UDP."""
        sink = StatsdSink()
        sink.flush()  # Should not raise


class TestDebugSink:
    """Tests for DebugSink."""

    def test_debug_sink_default_logger(self) -> None:
        """DebugSink should use module logger by default."""
        sink = DebugSink()
        assert sink._logger is not None

    def test_debug_sink_custom_logger(self) -> None:
        """DebugSink should accept custom logger."""
        custom_logger = logging.getLogger("custom")
        sink = DebugSink(log=custom_logger)
        assert sink._logger is custom_logger

    def test_emit_counter_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """emit_counter should log at debug level."""
        logger = logging.getLogger("test_debug_sink")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)

        with caplog.at_level(logging.DEBUG, logger="test_debug_sink"):
            sink.emit_counter("requests", 5, {"env": "prod"})

        assert "Counter: requests=5" in caplog.text

    def test_emit_histogram_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """emit_histogram should log at debug level."""
        logger = logging.getLogger("test_debug_sink_hist")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)

        with caplog.at_level(logging.DEBUG, logger="test_debug_sink_hist"):
            sink.emit_histogram("latency_ms", 100, {"tool": "read"})

        assert "Histogram: latency_ms=100" in caplog.text

    def test_emit_gauge_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """emit_gauge should log at debug level."""
        logger = logging.getLogger("test_debug_sink_gauge")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)

        with caplog.at_level(logging.DEBUG, logger="test_debug_sink_gauge"):
            sink.emit_gauge("queue_depth", 42, {"queue": "requests"})

        assert "Gauge: queue_depth=42" in caplog.text

    def test_flush_noop(self) -> None:
        """flush() should be a no-op."""
        sink = DebugSink()
        sink.flush()  # Should not raise


class TestSinkIntegration:
    """Integration tests for sinks with collector."""

    @patch("socket.socket")
    def test_collector_with_statsd_sink(self, mock_socket_class: MagicMock) -> None:
        """Collector should emit to sinks."""
        from weakincentives.metrics import InMemoryMetricsCollector

        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket

        sink = StatsdSink(prefix="wink")
        collector = InMemoryMetricsCollector(worker_id="worker-1", sinks=[sink])

        collector.record_tool_call("read_file", latency_ms=25, success=True)

        # Should have emitted to sink
        assert mock_socket.sendto.call_count >= 2  # latency + calls counter

    def test_collector_with_debug_sink(self, caplog: pytest.LogCaptureFixture) -> None:
        """Collector should emit to debug sink."""
        from weakincentives.metrics import InMemoryMetricsCollector

        logger = logging.getLogger("test_collector_debug")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)
        collector = InMemoryMetricsCollector(sinks=[sink])

        with caplog.at_level(logging.DEBUG, logger="test_collector_debug"):
            collector.record_tool_call("read_file", latency_ms=25, success=True)

        assert "tool.latency_ms" in caplog.text

    def test_adapter_call_emits_to_sink(self, caplog: pytest.LogCaptureFixture) -> None:
        """Adapter calls should emit to sinks."""
        from weakincentives.metrics import AdapterCallParams, InMemoryMetricsCollector
        from weakincentives.runtime.events import TokenUsage

        logger = logging.getLogger("test_adapter_sink")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)
        collector = InMemoryMetricsCollector(worker_id="w1", sinks=[sink])

        with caplog.at_level(logging.DEBUG, logger="test_adapter_sink"):
            collector.record_adapter_call(
                "openai",
                AdapterCallParams(
                    render_ms=10,
                    call_ms=500,
                    parse_ms=5,
                    tool_ms=100,
                    usage=TokenUsage(input_tokens=100, output_tokens=50),
                    experiment="exp-1",
                ),
            )

        assert "adapter.call_latency_ms" in caplog.text
        assert "adapter.input_tokens" in caplog.text
        assert "adapter.output_tokens" in caplog.text

    def test_adapter_partial_usage_emits_to_sink(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Adapter calls with partial usage should emit only set tokens."""
        from weakincentives.metrics import AdapterCallParams, InMemoryMetricsCollector
        from weakincentives.runtime.events import TokenUsage

        logger = logging.getLogger("test_adapter_partial_sink")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)
        collector = InMemoryMetricsCollector(sinks=[sink])

        # Test with only input_tokens
        with caplog.at_level(logging.DEBUG, logger="test_adapter_partial_sink"):
            collector.record_adapter_call(
                "openai",
                AdapterCallParams(
                    render_ms=10,
                    call_ms=500,
                    parse_ms=5,
                    tool_ms=100,
                    usage=TokenUsage(input_tokens=100, output_tokens=None),
                ),
            )
        assert "adapter.input_tokens" in caplog.text

        # Test with only output_tokens
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="test_adapter_partial_sink"):
            collector.record_adapter_call(
                "openai",
                AdapterCallParams(
                    render_ms=10,
                    call_ms=500,
                    parse_ms=5,
                    tool_ms=100,
                    usage=TokenUsage(input_tokens=None, output_tokens=50),
                ),
            )
        assert "adapter.output_tokens" in caplog.text

    def test_adapter_error_emits_to_sink(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Adapter errors should emit to sinks."""
        from weakincentives.metrics import AdapterCallParams, InMemoryMetricsCollector

        logger = logging.getLogger("test_adapter_error_sink")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)
        collector = InMemoryMetricsCollector(sinks=[sink])

        with caplog.at_level(logging.DEBUG, logger="test_adapter_error_sink"):
            collector.record_adapter_call(
                "openai",
                AdapterCallParams(
                    render_ms=10,
                    call_ms=500,
                    parse_ms=5,
                    tool_ms=100,
                    error="rate_limit",
                    throttled=True,
                ),
            )

        assert "adapter.errors" in caplog.text
        assert "adapter.throttles" in caplog.text

    def test_tool_failure_emits_to_sink(self, caplog: pytest.LogCaptureFixture) -> None:
        """Tool failures should emit to sinks."""
        from weakincentives.metrics import InMemoryMetricsCollector

        logger = logging.getLogger("test_tool_failure_sink")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)
        collector = InMemoryMetricsCollector(sinks=[sink])

        with caplog.at_level(logging.DEBUG, logger="test_tool_failure_sink"):
            collector.record_tool_call("read_file", latency_ms=25, success=False)

        assert "tool.failures" in caplog.text

    def test_mailbox_operations_emit_to_sink(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Mailbox operations should emit to sinks."""
        from weakincentives.metrics import InMemoryMetricsCollector

        logger = logging.getLogger("test_mailbox_sink")
        logger.setLevel(logging.DEBUG)
        sink = DebugSink(log=logger)
        collector = InMemoryMetricsCollector(sinks=[sink])

        with caplog.at_level(logging.DEBUG, logger="test_mailbox_sink"):
            # First call creates new mailbox metrics
            collector.record_message_received("requests", delivery_count=1, age_ms=100)
            # Second call updates existing mailbox metrics
            collector.record_message_received("requests", delivery_count=1, age_ms=50)
            collector.record_message_ack("requests")
            collector.record_message_nack("requests")
            collector.record_message_dead_lettered("requests")
            collector.record_queue_depth("requests", depth=42)

        assert "mailbox.lag_ms" in caplog.text
        assert "mailbox.received" in caplog.text
        assert "mailbox.acked" in caplog.text
        assert "mailbox.nacked" in caplog.text
        assert "mailbox.dead_lettered" in caplog.text
        assert "mailbox.depth" in caplog.text
