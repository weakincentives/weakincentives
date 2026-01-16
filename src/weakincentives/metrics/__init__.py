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

"""In-memory metrics collection for observability.

This module provides metrics primitives and collectors for tracking latency,
throughput, error rates, and queue health across adapters, tools, and
mailbox operations.

Quick Start::

    from weakincentives.metrics import (
        InMemoryMetricsCollector,
        MetricsSnapshot,
        StatsdSink,
    )

    # Create collector with optional StatsD export
    collector = InMemoryMetricsCollector(
        worker_id="worker-1",
        sinks=[StatsdSink(host="localhost", port=8125)],
    )

    # Record adapter call metrics
    from weakincentives.metrics import AdapterCallParams

    collector.record_adapter_call(
        "openai",
        AdapterCallParams(
            render_ms=10,
            call_ms=500,
            parse_ms=5,
            tool_ms=100,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        ),
    )

    # Record tool call metrics
    collector.record_tool_call(
        "read_file",
        latency_ms=25,
        success=True,
    )

    # Get a snapshot of all metrics
    snapshot = collector.snapshot()

Resource Binding::

    from weakincentives.resources import Binding, Scope

    metrics_binding = Binding(
        MetricsCollector,
        lambda r: InMemoryMetricsCollector(worker_id="worker-1"),
        scope=Scope.SINGLETON,
    )

Metric Types
------------

- :class:`Counter`: Monotonically increasing counter
- :class:`Histogram`: Distribution with exponential bucket boundaries
- :class:`Gauge`: Point-in-time value that can increase or decrease

Core Metric Aggregates
----------------------

- :class:`AdapterMetrics`: Latencies, token usage, errors for LLM providers
- :class:`ToolMetrics`: Latencies, success/failure rates for tool execution
- :class:`MailboxMetrics`: Queue health, delivery attempts, DLQ counts

Sinks
-----

- :class:`StatsdSink`: Export to StatsD/DataDog/Telegraf
- :class:`DebugSink`: Log metrics for development/testing
"""

from __future__ import annotations

from ._collector import AdapterCallParams, InMemoryMetricsCollector, MetricsCollector
from ._debug import archive_metrics, dump_metrics
from ._primitives import Counter, Gauge, Histogram
from ._sinks import DebugSink, MetricsSink, StatsdSink
from ._snapshot import MetricsSnapshot
from ._types import AdapterMetrics, MailboxMetrics, ToolMetrics

__all__ = [
    "AdapterCallParams",
    "AdapterMetrics",
    "Counter",
    "DebugSink",
    "Gauge",
    "Histogram",
    "InMemoryMetricsCollector",
    "MailboxMetrics",
    "MetricsCollector",
    "MetricsSink",
    "MetricsSnapshot",
    "StatsdSink",
    "ToolMetrics",
    "archive_metrics",
    "dump_metrics",
]
