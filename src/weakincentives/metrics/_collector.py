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

"""Metrics collector protocol and in-memory implementation."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from ..types import AdapterName
from ._snapshot import MetricsSnapshot
from ._types import AdapterMetrics, MailboxMetrics, ToolMetrics

if TYPE_CHECKING:
    from ..runtime.events import TokenUsage
    from ..runtime.run_context import RunContext
    from ._sinks import MetricsSink


@dataclass(slots=True, frozen=True)
class AdapterCallParams:
    """Parameters for recording an adapter call."""

    render_ms: int
    call_ms: int
    parse_ms: int
    tool_ms: int
    usage: TokenUsage | None = None
    error: str | None = None
    throttled: bool = False
    run_context: RunContext | None = None
    experiment: str | None = None


class MetricsCollector(Protocol):
    """Protocol for metrics collection and aggregation.

    Defines the interface for recording metrics from adapters, tools,
    and mailbox operations.
    """

    def record_adapter_call(
        self,
        adapter: AdapterName,
        params: AdapterCallParams,
    ) -> None:
        """Record metrics for an adapter call.

        Args:
            adapter: Provider identifier.
            params: Call parameters including latencies and usage.
        """
        ...

    def record_tool_call(
        self,
        tool_name: str,
        *,
        latency_ms: int,
        success: bool,
        error_code: str | None = None,
    ) -> None:
        """Record metrics for a tool call.

        Args:
            tool_name: Tool identifier.
            latency_ms: Execution time in milliseconds.
            success: Whether the call succeeded.
            error_code: Error code if the call failed.
        """
        ...

    def record_message_received(
        self, queue_name: str, *, delivery_count: int, age_ms: int
    ) -> None:
        """Record a message received from a queue.

        Args:
            queue_name: Queue identifier.
            delivery_count: Delivery attempt number (1-based).
            age_ms: Message age at receive time in milliseconds.
        """
        ...

    def record_message_ack(self, queue_name: str) -> None:
        """Record a message acknowledgment.

        Args:
            queue_name: Queue identifier.
        """
        ...

    def record_message_nack(self, queue_name: str) -> None:
        """Record a message negative acknowledgment.

        Args:
            queue_name: Queue identifier.
        """
        ...

    def record_message_dead_lettered(self, queue_name: str) -> None:
        """Record a message sent to dead letter queue.

        Args:
            queue_name: Queue identifier.
        """
        ...

    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """Record current queue depth.

        Args:
            queue_name: Queue identifier.
            depth: Current number of messages in queue.
        """
        ...

    def snapshot(self) -> MetricsSnapshot:
        """Capture a point-in-time snapshot of all metrics.

        Returns:
            MetricsSnapshot containing all collected metrics.
        """
        ...

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        ...


class InMemoryMetricsCollector:
    """Thread-safe in-memory metrics collector.

    Stores metrics in memory with optional export to external sinks.
    Designed to be bound as a singleton via ResourceRegistry.

    Args:
        worker_id: Optional worker identifier for tagging.
        sinks: Optional sequence of MetricsSink instances for export.
    """

    def __init__(
        self,
        worker_id: str | None = None,
        sinks: Sequence[MetricsSink] = (),
    ) -> None:
        super().__init__()
        self._worker_id = worker_id
        self._sinks = list(sinks)
        self._lock = threading.Lock()

        # Internal state - keyed by (adapter, experiment) for adapters
        self._adapters: dict[tuple[str, str | None], AdapterMetrics] = {}
        self._tools: dict[str, ToolMetrics] = {}
        self._mailboxes: dict[str, MailboxMetrics] = {}

    def record_adapter_call(
        self,
        adapter: AdapterName,
        params: AdapterCallParams,
    ) -> None:
        """Record metrics for an adapter call."""
        total_ms = params.render_ms + params.call_ms + params.parse_ms + params.tool_ms
        key = (adapter, params.experiment)

        with self._lock:
            metrics = self._get_or_create_adapter_metrics(adapter, params.experiment)
            metrics = self._update_adapter_latencies(metrics, params, total_ms)
            metrics = self._update_adapter_tokens(metrics, params.usage)
            metrics = self._update_adapter_errors(
                metrics, params.error, params.throttled
            )
            self._adapters[key] = metrics

        self._emit_adapter_metrics_to_sinks(adapter, params, total_ms)

    def _get_or_create_adapter_metrics(
        self, adapter: AdapterName, experiment: str | None
    ) -> AdapterMetrics:
        """Get existing or create new adapter metrics."""
        key = (adapter, experiment)
        metrics = self._adapters.get(key)
        if metrics is None:
            return AdapterMetrics.create(adapter, experiment=experiment)
        return metrics

    @staticmethod
    def _update_adapter_latencies(
        metrics: AdapterMetrics, params: AdapterCallParams, total_ms: int
    ) -> AdapterMetrics:
        """Update adapter latency histograms."""
        return replace(
            metrics,
            render_latency=metrics.render_latency.observe(params.render_ms),
            call_latency=metrics.call_latency.observe(params.call_ms),
            parse_latency=metrics.parse_latency.observe(params.parse_ms),
            tool_latency=metrics.tool_latency.observe(params.tool_ms),
            total_latency=metrics.total_latency.observe(total_ms),
            request_count=metrics.request_count.inc(),
        )

    @staticmethod
    def _update_adapter_tokens(
        metrics: AdapterMetrics, usage: TokenUsage | None
    ) -> AdapterMetrics:
        """Update adapter token counters."""
        if usage is None:
            return metrics
        if usage.input_tokens is not None:
            metrics = replace(
                metrics, input_tokens=metrics.input_tokens.inc(usage.input_tokens)
            )
        if usage.output_tokens is not None:
            metrics = replace(
                metrics, output_tokens=metrics.output_tokens.inc(usage.output_tokens)
            )
        if usage.cached_tokens is not None:
            metrics = replace(
                metrics, cached_tokens=metrics.cached_tokens.inc(usage.cached_tokens)
            )
        return metrics

    @staticmethod
    def _update_adapter_errors(
        metrics: AdapterMetrics, error: str | None, throttled: bool
    ) -> AdapterMetrics:
        """Update adapter error and throttle counters."""
        if error is not None:
            metrics = replace(metrics, error_count=metrics.error_count.inc())
        if throttled:
            metrics = replace(metrics, throttle_count=metrics.throttle_count.inc())
        return metrics

    def _emit_adapter_metrics_to_sinks(
        self,
        adapter: AdapterName,
        params: AdapterCallParams,
        total_ms: int,
    ) -> None:
        """Emit adapter metrics to configured sinks."""
        tags = {"adapter": adapter, "worker": self._worker_id or "unknown"}
        if params.experiment:
            tags["experiment"] = params.experiment

        for sink in self._sinks:
            sink.emit_histogram("adapter.call_latency_ms", params.call_ms, tags)
            sink.emit_histogram("adapter.render_latency_ms", params.render_ms, tags)
            sink.emit_histogram("adapter.parse_latency_ms", params.parse_ms, tags)
            sink.emit_histogram("adapter.tool_latency_ms", params.tool_ms, tags)
            sink.emit_histogram("adapter.total_latency_ms", total_ms, tags)
            sink.emit_counter("adapter.requests", 1, tags)
            self._emit_adapter_usage_to_sink(sink, params.usage, tags)
            self._emit_adapter_errors_to_sink(
                sink, params.error, params.throttled, tags
            )

    @staticmethod
    def _emit_adapter_usage_to_sink(
        sink: MetricsSink,
        usage: TokenUsage | None,
        tags: dict[str, str],
    ) -> None:
        """Emit token usage to a single sink."""
        if usage is None:
            return
        if usage.input_tokens is not None:
            sink.emit_counter("adapter.input_tokens", usage.input_tokens, tags)
        if usage.output_tokens is not None:
            sink.emit_counter("adapter.output_tokens", usage.output_tokens, tags)

    @staticmethod
    def _emit_adapter_errors_to_sink(
        sink: MetricsSink,
        error: str | None,
        throttled: bool,
        tags: dict[str, str],
    ) -> None:
        """Emit error/throttle counts to a single sink."""
        if error is not None:
            sink.emit_counter("adapter.errors", 1, tags)
        if throttled:
            sink.emit_counter("adapter.throttles", 1, tags)

    def record_tool_call(
        self,
        tool_name: str,
        *,
        latency_ms: int,
        success: bool,
        error_code: str | None = None,
    ) -> None:
        """Record metrics for a tool call."""
        with self._lock:
            metrics = self._tools.get(tool_name)
            if metrics is None:
                metrics = ToolMetrics.create(tool_name)

            metrics = replace(
                metrics,
                latency=metrics.latency.observe(latency_ms),
                call_count=metrics.call_count.inc(),
            )

            if success:
                metrics = replace(metrics, success_count=metrics.success_count.inc())
            else:
                metrics = replace(metrics, failure_count=metrics.failure_count.inc())
                if error_code is not None:
                    metrics = metrics.with_error_code(error_code)

            self._tools[tool_name] = metrics

        # Emit to sinks
        tags = {"tool": tool_name, "worker": self._worker_id or "unknown"}
        for sink in self._sinks:
            sink.emit_histogram("tool.latency_ms", latency_ms, tags)
            sink.emit_counter("tool.calls", 1, tags)
            if not success:
                sink.emit_counter("tool.failures", 1, tags)

    def record_message_received(
        self, queue_name: str, *, delivery_count: int, age_ms: int
    ) -> None:
        """Record a message received from a queue."""
        with self._lock:
            metrics = self._mailboxes.get(queue_name)
            if metrics is None:
                metrics = MailboxMetrics.create(queue_name)

            metrics = replace(
                metrics,
                queue_lag=metrics.queue_lag.observe(age_ms),
                messages_received=metrics.messages_received.inc(),
            )
            metrics = metrics.with_delivery(delivery_count)

            # Track retries (delivery_count > 1 means retry)
            if delivery_count > 1:
                metrics = replace(metrics, total_retries=metrics.total_retries.inc())

            self._mailboxes[queue_name] = metrics

        # Emit to sinks
        tags = {"queue": queue_name, "worker": self._worker_id or "unknown"}
        for sink in self._sinks:
            sink.emit_histogram("mailbox.lag_ms", age_ms, tags)
            sink.emit_counter("mailbox.received", 1, tags)

    def record_message_ack(self, queue_name: str) -> None:
        """Record a message acknowledgment."""
        with self._lock:
            metrics = self._mailboxes.get(queue_name)
            if metrics is None:
                metrics = MailboxMetrics.create(queue_name)

            metrics = replace(metrics, messages_acked=metrics.messages_acked.inc())
            self._mailboxes[queue_name] = metrics

        # Emit to sinks
        tags = {"queue": queue_name, "worker": self._worker_id or "unknown"}
        for sink in self._sinks:
            sink.emit_counter("mailbox.acked", 1, tags)

    def record_message_nack(self, queue_name: str) -> None:
        """Record a message negative acknowledgment."""
        with self._lock:
            metrics = self._mailboxes.get(queue_name)
            if metrics is None:
                metrics = MailboxMetrics.create(queue_name)

            metrics = replace(metrics, messages_nacked=metrics.messages_nacked.inc())
            self._mailboxes[queue_name] = metrics

        # Emit to sinks
        tags = {"queue": queue_name, "worker": self._worker_id or "unknown"}
        for sink in self._sinks:
            sink.emit_counter("mailbox.nacked", 1, tags)

    def record_message_dead_lettered(self, queue_name: str) -> None:
        """Record a message sent to dead letter queue."""
        with self._lock:
            metrics = self._mailboxes.get(queue_name)
            if metrics is None:
                metrics = MailboxMetrics.create(queue_name)

            metrics = replace(
                metrics, messages_dead_lettered=metrics.messages_dead_lettered.inc()
            )
            self._mailboxes[queue_name] = metrics

        # Emit to sinks
        tags = {"queue": queue_name, "worker": self._worker_id or "unknown"}
        for sink in self._sinks:
            sink.emit_counter("mailbox.dead_lettered", 1, tags)

    def record_queue_depth(self, queue_name: str, depth: int) -> None:
        """Record current queue depth."""
        with self._lock:
            metrics = self._mailboxes.get(queue_name)
            if metrics is None:
                metrics = MailboxMetrics.create(queue_name)

            metrics = replace(metrics, queue_depth=depth)
            self._mailboxes[queue_name] = metrics

        # Emit to sinks
        tags = {"queue": queue_name, "worker": self._worker_id or "unknown"}
        for sink in self._sinks:
            sink.emit_gauge("mailbox.depth", depth, tags)

    def snapshot(self) -> MetricsSnapshot:
        """Capture a point-in-time snapshot of all metrics."""
        with self._lock:
            return MetricsSnapshot(
                adapters=tuple(self._adapters.values()),
                tools=tuple(self._tools.values()),
                mailboxes=tuple(self._mailboxes.values()),
                captured_at=datetime.now(UTC),
                worker_id=self._worker_id,
            )

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self._adapters.clear()
            self._tools.clear()
            self._mailboxes.clear()


__all__ = [
    "AdapterCallParams",
    "InMemoryMetricsCollector",
    "MetricsCollector",
]
