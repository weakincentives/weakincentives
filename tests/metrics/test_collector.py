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

"""Tests for MetricsCollector and InMemoryMetricsCollector."""

from __future__ import annotations

import threading
from datetime import UTC

from weakincentives.metrics import (
    AdapterCallParams,
    InMemoryMetricsCollector,
    MetricsSnapshot,
)
from weakincentives.runtime.events import TokenUsage


class TestInMemoryMetricsCollector:
    """Tests for InMemoryMetricsCollector."""

    def test_collector_empty_snapshot(self) -> None:
        """Empty collector should return empty snapshot."""
        collector = InMemoryMetricsCollector()
        snapshot = collector.snapshot()

        assert isinstance(snapshot, MetricsSnapshot)
        assert len(snapshot.adapters) == 0
        assert len(snapshot.tools) == 0
        assert len(snapshot.mailboxes) == 0

    def test_collector_worker_id(self) -> None:
        """Collector should include worker_id in snapshot."""
        collector = InMemoryMetricsCollector(worker_id="worker-1")
        snapshot = collector.snapshot()
        assert snapshot.worker_id == "worker-1"

    def test_record_adapter_call(self) -> None:
        """record_adapter_call should track adapter metrics."""
        collector = InMemoryMetricsCollector()
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(render_ms=10, call_ms=500, parse_ms=5, tool_ms=100),
        )

        snapshot = collector.snapshot()
        assert len(snapshot.adapters) == 1

        metrics = snapshot.adapters[0]
        assert metrics.adapter == "openai"
        assert metrics.render_latency.total_count == 1
        assert metrics.call_latency.total_count == 1
        assert metrics.total_latency.total_count == 1
        assert metrics.request_count.value == 1

    def test_record_adapter_call_with_usage(self) -> None:
        """record_adapter_call should track token usage."""
        collector = InMemoryMetricsCollector()
        usage = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=20)
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(
                render_ms=10, call_ms=500, parse_ms=5, tool_ms=100, usage=usage
            ),
        )

        snapshot = collector.snapshot()
        metrics = snapshot.adapters[0]
        assert metrics.input_tokens.value == 100
        assert metrics.output_tokens.value == 50
        assert metrics.cached_tokens.value == 20

    def test_record_adapter_call_with_partial_usage(self) -> None:
        """record_adapter_call should handle partial token usage."""
        collector = InMemoryMetricsCollector()
        # Only input_tokens set
        usage1 = TokenUsage(input_tokens=100, output_tokens=None, cached_tokens=None)
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(
                render_ms=10, call_ms=500, parse_ms=5, tool_ms=100, usage=usage1
            ),
        )

        # Only output_tokens set
        usage2 = TokenUsage(input_tokens=None, output_tokens=50, cached_tokens=None)
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(
                render_ms=10, call_ms=500, parse_ms=5, tool_ms=100, usage=usage2
            ),
        )

        snapshot = collector.snapshot()
        metrics = snapshot.adapters[0]
        assert metrics.input_tokens.value == 100
        assert metrics.output_tokens.value == 50

    def test_record_adapter_call_with_error(self) -> None:
        """record_adapter_call should track errors."""
        collector = InMemoryMetricsCollector()
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(
                render_ms=10,
                call_ms=500,
                parse_ms=5,
                tool_ms=100,
                error="Connection timeout",
            ),
        )

        snapshot = collector.snapshot()
        metrics = snapshot.adapters[0]
        assert metrics.error_count.value == 1

    def test_record_adapter_call_throttled(self) -> None:
        """record_adapter_call should track throttling."""
        collector = InMemoryMetricsCollector()
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(
                render_ms=10, call_ms=500, parse_ms=5, tool_ms=100, throttled=True
            ),
        )

        snapshot = collector.snapshot()
        metrics = snapshot.adapters[0]
        assert metrics.throttle_count.value == 1

    def test_record_adapter_call_with_experiment(self) -> None:
        """record_adapter_call should track by experiment."""
        collector = InMemoryMetricsCollector()

        # Record for baseline
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(
                render_ms=10,
                call_ms=500,
                parse_ms=5,
                tool_ms=100,
                experiment="baseline",
            ),
        )

        # Record for variant
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(
                render_ms=10,
                call_ms=600,
                parse_ms=5,
                tool_ms=100,
                experiment="v2-concise",
            ),
        )

        snapshot = collector.snapshot()
        assert len(snapshot.adapters) == 2

        by_experiment = {m.experiment: m for m in snapshot.adapters}
        assert by_experiment["baseline"].request_count.value == 1
        assert by_experiment["v2-concise"].request_count.value == 1

    def test_record_adapter_call_accumulates(self) -> None:
        """Multiple calls should accumulate."""
        collector = InMemoryMetricsCollector()
        for _ in range(5):
            collector.record_adapter_call(
                "openai",
                AdapterCallParams(render_ms=10, call_ms=500, parse_ms=5, tool_ms=100),
            )

        snapshot = collector.snapshot()
        metrics = snapshot.adapters[0]
        assert metrics.request_count.value == 5
        assert metrics.call_latency.total_count == 5

    def test_record_tool_call_success(self) -> None:
        """record_tool_call should track successful calls."""
        collector = InMemoryMetricsCollector()
        collector.record_tool_call(
            "read_file",
            latency_ms=25,
            success=True,
        )

        snapshot = collector.snapshot()
        assert len(snapshot.tools) == 1

        metrics = snapshot.tools[0]
        assert metrics.tool_name == "read_file"
        assert metrics.call_count.value == 1
        assert metrics.success_count.value == 1
        assert metrics.failure_count.value == 0

    def test_record_tool_call_failure(self) -> None:
        """record_tool_call should track failures."""
        collector = InMemoryMetricsCollector()
        collector.record_tool_call(
            "execute",
            latency_ms=100,
            success=False,
            error_code="TIMEOUT",
        )

        snapshot = collector.snapshot()
        metrics = snapshot.tools[0]
        assert metrics.failure_count.value == 1
        assert ("TIMEOUT", 1) in metrics.error_codes

    def test_record_tool_call_multiple_tools(self) -> None:
        """Collector should track multiple tools separately."""
        collector = InMemoryMetricsCollector()
        collector.record_tool_call("read_file", latency_ms=10, success=True)
        collector.record_tool_call("write_file", latency_ms=20, success=True)
        collector.record_tool_call("read_file", latency_ms=15, success=True)

        snapshot = collector.snapshot()
        assert len(snapshot.tools) == 2

        by_name = {m.tool_name: m for m in snapshot.tools}
        assert by_name["read_file"].call_count.value == 2
        assert by_name["write_file"].call_count.value == 1

    def test_record_message_received(self) -> None:
        """record_message_received should track queue metrics."""
        collector = InMemoryMetricsCollector()
        collector.record_message_received(
            "requests",
            delivery_count=1,
            age_ms=500,
        )

        snapshot = collector.snapshot()
        assert len(snapshot.mailboxes) == 1

        metrics = snapshot.mailboxes[0]
        assert metrics.queue_name == "requests"
        assert metrics.messages_received.value == 1
        assert metrics.queue_lag.total_count == 1
        assert metrics.delivery_count_dist[0] == 1

    def test_record_message_received_retry(self) -> None:
        """Retries should be tracked."""
        collector = InMemoryMetricsCollector()
        collector.record_message_received(
            "requests",
            delivery_count=3,  # This is a retry
            age_ms=1000,
        )

        snapshot = collector.snapshot()
        metrics = snapshot.mailboxes[0]
        assert metrics.total_retries.value == 1
        assert metrics.delivery_count_dist[2] == 1

    def test_record_message_ack(self) -> None:
        """record_message_ack should track acks."""
        collector = InMemoryMetricsCollector()
        collector.record_message_ack("requests")
        collector.record_message_ack("requests")

        snapshot = collector.snapshot()
        metrics = snapshot.mailboxes[0]
        assert metrics.messages_acked.value == 2

    def test_record_message_nack(self) -> None:
        """record_message_nack should track nacks."""
        collector = InMemoryMetricsCollector()
        collector.record_message_nack("requests")

        snapshot = collector.snapshot()
        metrics = snapshot.mailboxes[0]
        assert metrics.messages_nacked.value == 1

    def test_record_message_dead_lettered(self) -> None:
        """record_message_dead_lettered should track DLQ."""
        collector = InMemoryMetricsCollector()
        collector.record_message_dead_lettered("requests")

        snapshot = collector.snapshot()
        metrics = snapshot.mailboxes[0]
        assert metrics.messages_dead_lettered.value == 1

    def test_record_queue_depth(self) -> None:
        """record_queue_depth should update depth."""
        collector = InMemoryMetricsCollector()
        collector.record_queue_depth("requests", 42)

        snapshot = collector.snapshot()
        metrics = snapshot.mailboxes[0]
        assert metrics.queue_depth == 42

    def test_reset(self) -> None:
        """reset() should clear all metrics."""
        collector = InMemoryMetricsCollector()
        collector.record_adapter_call(
            "openai",
            AdapterCallParams(render_ms=10, call_ms=500, parse_ms=5, tool_ms=100),
        )
        collector.record_tool_call("test", latency_ms=10, success=True)
        collector.record_message_ack("requests")

        collector.reset()
        snapshot = collector.snapshot()

        assert len(snapshot.adapters) == 0
        assert len(snapshot.tools) == 0
        assert len(snapshot.mailboxes) == 0

    def test_snapshot_timestamp(self) -> None:
        """Snapshot should have captured_at timestamp."""
        collector = InMemoryMetricsCollector()
        snapshot = collector.snapshot()
        assert snapshot.captured_at is not None
        assert snapshot.captured_at.tzinfo == UTC

    def test_thread_safety(self) -> None:
        """Collector should be thread-safe."""
        collector = InMemoryMetricsCollector()
        num_threads = 10
        calls_per_thread = 100

        def record_calls(thread_id: int) -> None:
            for i in range(calls_per_thread):
                collector.record_adapter_call(
                    "openai",
                    AdapterCallParams(
                        render_ms=i, call_ms=i * 2, parse_ms=i, tool_ms=i
                    ),
                )
                collector.record_tool_call(
                    f"tool-{thread_id}",
                    latency_ms=i,
                    success=True,
                )

        threads = [
            threading.Thread(target=record_calls, args=(tid,))
            for tid in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snapshot = collector.snapshot()
        # All adapter calls should be recorded
        assert (
            snapshot.adapters[0].request_count.value == num_threads * calls_per_thread
        )
        # Each thread created its own tool
        assert len(snapshot.tools) == num_threads


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot."""

    def test_empty_snapshot(self) -> None:
        """empty() should create an empty snapshot."""
        snapshot = MetricsSnapshot.empty(worker_id="test")
        assert len(snapshot.adapters) == 0
        assert len(snapshot.tools) == 0
        assert len(snapshot.mailboxes) == 0
        assert snapshot.worker_id == "test"
        assert snapshot.captured_at is not None
