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

"""Tests for core metric types."""

from __future__ import annotations

import pytest

from weakincentives.metrics import AdapterMetrics, MailboxMetrics, ToolMetrics


class TestAdapterMetrics:
    """Tests for AdapterMetrics type."""

    def test_create_adapter_metrics(self) -> None:
        """create() should initialize all metrics."""
        metrics = AdapterMetrics.create("openai")
        assert metrics.adapter == "openai"
        assert metrics.render_latency.total_count == 0
        assert metrics.call_latency.total_count == 0
        assert metrics.input_tokens.value == 0
        assert metrics.request_count.value == 0
        assert metrics.error_count.value == 0
        assert metrics.experiment is None

    def test_create_with_experiment(self) -> None:
        """create() should include experiment tag in labels."""
        metrics = AdapterMetrics.create("openai", experiment="v2-test")
        assert metrics.experiment == "v2-test"
        assert ("experiment", "v2-test") in metrics.render_latency.labels

    def test_adapter_labels(self) -> None:
        """Metrics should have adapter label."""
        metrics = AdapterMetrics.create("litellm")
        assert ("adapter", "litellm") in metrics.render_latency.labels
        assert ("adapter", "litellm") in metrics.input_tokens.labels

    def test_adapter_metrics_immutable(self) -> None:
        """AdapterMetrics should be immutable."""
        metrics = AdapterMetrics.create("openai")
        with pytest.raises(AttributeError):
            metrics.adapter = "other"  # type: ignore[misc]

    def test_adapter_metrics_update(self) -> None:
        """update() should create new instance with changes."""
        metrics = AdapterMetrics.create("openai")
        new_latency = metrics.call_latency.observe(100)
        updated = metrics.update(call_latency=new_latency)
        assert updated.call_latency.total_count == 1
        assert metrics.call_latency.total_count == 0


class TestToolMetrics:
    """Tests for ToolMetrics type."""

    def test_create_tool_metrics(self) -> None:
        """create() should initialize all metrics."""
        metrics = ToolMetrics.create("read_file")
        assert metrics.tool_name == "read_file"
        assert metrics.latency.total_count == 0
        assert metrics.call_count.value == 0
        assert metrics.success_count.value == 0
        assert metrics.failure_count.value == 0
        assert metrics.error_codes == ()

    def test_tool_labels(self) -> None:
        """Metrics should have tool label."""
        metrics = ToolMetrics.create("execute_cmd")
        assert ("tool", "execute_cmd") in metrics.latency.labels

    def test_failure_rate_no_calls(self) -> None:
        """failure_rate() should return None with no calls."""
        metrics = ToolMetrics.create("test")
        assert metrics.failure_rate() is None

    def test_failure_rate_with_calls(self) -> None:
        """failure_rate() should return correct ratio."""
        metrics = ToolMetrics.create("test")
        metrics = metrics.update(
            call_count=metrics.call_count.inc(10),
            failure_count=metrics.failure_count.inc(3),
        )
        assert metrics.failure_rate() == 0.3

    def test_failure_rate_all_success(self) -> None:
        """failure_rate() should return 0 when all succeed."""
        metrics = ToolMetrics.create("test")
        metrics = metrics.update(call_count=metrics.call_count.inc(5))
        assert metrics.failure_rate() == 0.0

    def test_with_error_code(self) -> None:
        """with_error_code() should track error codes."""
        metrics = ToolMetrics.create("test")
        metrics = metrics.with_error_code("TIMEOUT")
        metrics = metrics.with_error_code("TIMEOUT")
        metrics = metrics.with_error_code("PERMISSION_DENIED")

        error_dict = dict(metrics.error_codes)
        assert error_dict["TIMEOUT"] == 2
        assert error_dict["PERMISSION_DENIED"] == 1

    def test_error_codes_sorted(self) -> None:
        """error_codes should be sorted by key."""
        metrics = ToolMetrics.create("test")
        metrics = metrics.with_error_code("ZEBRA")
        metrics = metrics.with_error_code("ALPHA")
        metrics = metrics.with_error_code("BETA")

        keys = [k for k, _ in metrics.error_codes]
        assert keys == ["ALPHA", "BETA", "ZEBRA"]

    def test_tool_metrics_immutable(self) -> None:
        """ToolMetrics should be immutable."""
        metrics = ToolMetrics.create("test")
        with pytest.raises(AttributeError):
            metrics.tool_name = "other"  # type: ignore[misc]


class TestMailboxMetrics:
    """Tests for MailboxMetrics type."""

    def test_create_mailbox_metrics(self) -> None:
        """create() should initialize all metrics."""
        metrics = MailboxMetrics.create("requests")
        assert metrics.queue_name == "requests"
        assert metrics.queue_lag.total_count == 0
        assert len(metrics.delivery_count_dist) == 11
        assert metrics.messages_received.value == 0
        assert metrics.messages_acked.value == 0
        assert metrics.messages_dead_lettered.value == 0
        assert metrics.queue_depth == 0
        assert metrics.oldest_message_age_ms is None

    def test_mailbox_labels(self) -> None:
        """Metrics should have queue label."""
        metrics = MailboxMetrics.create("replies")
        assert ("queue", "replies") in metrics.queue_lag.labels

    def test_with_delivery_first_attempt(self) -> None:
        """with_delivery(1) should increment first bucket."""
        metrics = MailboxMetrics.create("test")
        metrics = metrics.with_delivery(1)
        assert metrics.delivery_count_dist[0] == 1
        assert sum(metrics.delivery_count_dist) == 1

    def test_with_delivery_multiple_attempts(self) -> None:
        """with_delivery should track retry distribution."""
        metrics = MailboxMetrics.create("test")
        metrics = metrics.with_delivery(1)
        metrics = metrics.with_delivery(2)
        metrics = metrics.with_delivery(3)
        metrics = metrics.with_delivery(10)
        metrics = metrics.with_delivery(15)  # Should go to 10+ bucket

        assert metrics.delivery_count_dist[0] == 1  # 1 attempt
        assert metrics.delivery_count_dist[1] == 1  # 2 attempts
        assert metrics.delivery_count_dist[2] == 1  # 3 attempts
        assert metrics.delivery_count_dist[9] == 1  # 10 attempts
        assert metrics.delivery_count_dist[10] == 1  # 10+ attempts

    def test_with_delivery_zero_handled(self) -> None:
        """with_delivery(0) should not crash."""
        metrics = MailboxMetrics.create("test")
        metrics = metrics.with_delivery(0)
        # Should go to first bucket due to max(0, idx)
        assert metrics.delivery_count_dist[0] == 1

    def test_queue_depth_update(self) -> None:
        """queue_depth should be updatable."""
        metrics = MailboxMetrics.create("test")
        metrics = metrics.update(queue_depth=42)
        assert metrics.queue_depth == 42

    def test_oldest_message_age_update(self) -> None:
        """oldest_message_age_ms should be updatable."""
        metrics = MailboxMetrics.create("test")
        metrics = metrics.update(oldest_message_age_ms=5000)
        assert metrics.oldest_message_age_ms == 5000

    def test_mailbox_metrics_immutable(self) -> None:
        """MailboxMetrics should be immutable."""
        metrics = MailboxMetrics.create("test")
        with pytest.raises(AttributeError):
            metrics.queue_name = "other"  # type: ignore[misc]
