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

"""Core metric types for adapter, tool, and mailbox tracking."""

from __future__ import annotations

from dataclasses import replace

from ..dataclasses import FrozenDataclass
from ..types import AdapterName
from ._primitives import Counter, Histogram


@FrozenDataclass()
class AdapterMetrics:
    """Metrics for adapter (LLM provider) interactions.

    Tracks latencies across evaluation phases, token usage, and error counts.

    Attributes:
        adapter: Provider identifier (e.g., "openai", "litellm").
        render_latency: Prompt rendering time distribution.
        call_latency: LLM API call time distribution.
        parse_latency: Response parsing time distribution.
        tool_latency: Tool execution time distribution.
        total_latency: End-to-end request time distribution.
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens consumed.
        cached_tokens: Tokens served from cache.
        request_count: Total requests made.
        error_count: Failed requests.
        throttle_count: Rate-limited requests.
        experiment: Optional experiment tag for A/B testing.
    """

    adapter: AdapterName
    render_latency: Histogram
    call_latency: Histogram
    parse_latency: Histogram
    tool_latency: Histogram
    total_latency: Histogram
    input_tokens: Counter
    output_tokens: Counter
    cached_tokens: Counter
    request_count: Counter
    error_count: Counter
    throttle_count: Counter
    experiment: str | None = None

    @classmethod
    def create(
        cls,
        adapter: AdapterName,
        *,
        experiment: str | None = None,
    ) -> AdapterMetrics:
        """Create a new AdapterMetrics instance with initialized histograms.

        Args:
            adapter: Provider identifier.
            experiment: Optional experiment tag.

        Returns:
            New AdapterMetrics with zero counts.
        """
        labels = (("adapter", adapter),)
        if experiment:
            labels = (*labels, ("experiment", experiment))

        return cls(
            adapter=adapter,
            render_latency=Histogram(name="adapter.render_latency_ms", labels=labels),
            call_latency=Histogram(name="adapter.call_latency_ms", labels=labels),
            parse_latency=Histogram(name="adapter.parse_latency_ms", labels=labels),
            tool_latency=Histogram(name="adapter.tool_latency_ms", labels=labels),
            total_latency=Histogram(name="adapter.total_latency_ms", labels=labels),
            input_tokens=Counter(name="adapter.input_tokens", labels=labels),
            output_tokens=Counter(name="adapter.output_tokens", labels=labels),
            cached_tokens=Counter(name="adapter.cached_tokens", labels=labels),
            request_count=Counter(name="adapter.requests", labels=labels),
            error_count=Counter(name="adapter.errors", labels=labels),
            throttle_count=Counter(name="adapter.throttles", labels=labels),
            experiment=experiment,
        )


@FrozenDataclass()
class ToolMetrics:
    """Metrics for tool execution tracking.

    Attributes:
        tool_name: Tool identifier.
        latency: Execution time distribution.
        call_count: Total invocations.
        success_count: Successful calls.
        failure_count: Failed calls.
        error_codes: Error code breakdown as (code, count) pairs.
    """

    tool_name: str
    latency: Histogram
    call_count: Counter
    success_count: Counter
    failure_count: Counter
    error_codes: tuple[tuple[str, int], ...] = ()

    @classmethod
    def create(cls, tool_name: str) -> ToolMetrics:
        """Create a new ToolMetrics instance with initialized counters.

        Args:
            tool_name: Tool identifier.

        Returns:
            New ToolMetrics with zero counts.
        """
        labels = (("tool", tool_name),)
        return cls(
            tool_name=tool_name,
            latency=Histogram(name="tool.latency_ms", labels=labels),
            call_count=Counter(name="tool.calls", labels=labels),
            success_count=Counter(name="tool.successes", labels=labels),
            failure_count=Counter(name="tool.failures", labels=labels),
        )

    def failure_rate(self) -> float | None:
        """Calculate the failure rate.

        Returns:
            Ratio of failures to total calls, or None if no calls.
        """
        total = self.call_count.value
        if total == 0:
            return None
        return self.failure_count.value / total

    def with_error_code(self, error_code: str) -> ToolMetrics:
        """Return a new ToolMetrics with an incremented error code count.

        Args:
            error_code: The error code to increment.

        Returns:
            New ToolMetrics with updated error code counts.
        """
        new_codes = dict(self.error_codes)
        new_codes[error_code] = new_codes.get(error_code, 0) + 1
        return replace(self, error_codes=tuple(sorted(new_codes.items())))


# Distribution of delivery attempts (indices 0-9 for attempts 1-10, index 10 for 10+)
_DELIVERY_DIST_SIZE: int = 11


@FrozenDataclass()
class MailboxMetrics:
    """Metrics for mailbox/queue health tracking.

    Attributes:
        queue_name: Queue identifier.
        queue_lag: Age of messages at receive time distribution.
        delivery_count_dist: Distribution of delivery attempts (1-10+).
        messages_received: Total messages received.
        messages_acked: Successfully processed messages.
        messages_nacked: Messages returned to queue.
        messages_expired: Messages exceeding TTL.
        messages_dead_lettered: Messages sent to DLQ.
        total_retries: Sum of retry attempts.
        queue_depth: Current queue size (point-in-time).
        oldest_message_age_ms: Age of oldest message (point-in-time).
    """

    queue_name: str
    queue_lag: Histogram
    delivery_count_dist: tuple[int, ...]
    messages_received: Counter
    messages_acked: Counter
    messages_nacked: Counter
    messages_expired: Counter
    messages_dead_lettered: Counter
    total_retries: Counter
    queue_depth: int = 0
    oldest_message_age_ms: int | None = None

    @classmethod
    def create(cls, queue_name: str) -> MailboxMetrics:
        """Create a new MailboxMetrics instance with initialized counters.

        Args:
            queue_name: Queue identifier.

        Returns:
            New MailboxMetrics with zero counts.
        """
        labels = (("queue", queue_name),)
        return cls(
            queue_name=queue_name,
            queue_lag=Histogram(name="mailbox.lag_ms", labels=labels),
            delivery_count_dist=tuple(0 for _ in range(_DELIVERY_DIST_SIZE)),
            messages_received=Counter(name="mailbox.received", labels=labels),
            messages_acked=Counter(name="mailbox.acked", labels=labels),
            messages_nacked=Counter(name="mailbox.nacked", labels=labels),
            messages_expired=Counter(name="mailbox.expired", labels=labels),
            messages_dead_lettered=Counter(name="mailbox.dead_lettered", labels=labels),
            total_retries=Counter(name="mailbox.retries", labels=labels),
        )

    def with_delivery(self, delivery_count: int) -> MailboxMetrics:
        """Return a new MailboxMetrics with updated delivery distribution.

        Args:
            delivery_count: The delivery attempt number (1-based).

        Returns:
            New MailboxMetrics with updated distribution.
        """
        # Map delivery count to distribution index (0-9 for 1-10, 10 for 10+)
        idx = min(delivery_count - 1, _DELIVERY_DIST_SIZE - 1)
        idx = max(0, idx)  # Ensure non-negative
        new_dist = list(self.delivery_count_dist)
        new_dist[idx] += 1
        return replace(self, delivery_count_dist=tuple(new_dist))


__all__ = [
    "AdapterMetrics",
    "MailboxMetrics",
    "ToolMetrics",
]
