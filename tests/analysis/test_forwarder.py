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

"""Tests for AnalysisForwarder."""

from __future__ import annotations

import random
from pathlib import Path
from uuid import uuid4

from weakincentives.analysis import (
    AnalysisBudget,
    AnalysisForwarder,
    AnalysisForwarderConfig,
    AnalysisRequest,
    CompletionNotification,
)
from weakincentives.clock import FakeClock
from weakincentives.runtime.mailbox import InMemoryMailbox


def _make_notification(
    *,
    success: bool = True,
    source: str = "agent_loop",
) -> CompletionNotification:
    """Create a CompletionNotification for testing."""
    return CompletionNotification(
        source=source,  # type: ignore[arg-type]
        bundle_path=Path(f"/tmp/bundle-{uuid4()}.zip"),
        request_id=uuid4(),
        success=success,
    )


class TestAnalysisForwarder:
    """Tests for AnalysisForwarder message processing."""

    def test_forwards_failure_when_always_forward_failures(self) -> None:
        """Failures are always forwarded when always_forward_failures=True."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        # Use rng that always returns 1.0 (never passes sample threshold)
        rng = random.Random(42)
        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(
                objective="test",
                sample_rate=0.0,  # Never sample successes
                always_forward_failures=True,
            ),
            rng=rng,
        )

        # Send a failure notification
        notifications.send(_make_notification(success=False))

        forwarder.run(max_iterations=1, wait_time_seconds=0)

        # Should have forwarded the failure
        assert forwarder.requests_sent == 1
        msgs = analysis_requests.receive(max_messages=10, wait_time_seconds=0)
        assert len(msgs) == 1
        assert msgs[0].body.objective == "test"
        assert msgs[0].body.source == "agent_loop"

    def test_samples_successes(self) -> None:
        """Successful notifications are sampled at the configured rate."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        # Use a deterministic rng
        rng = random.Random(0)
        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(
                objective="test",
                sample_rate=0.5,
                always_forward_failures=False,
            ),
            rng=rng,
        )

        # Send 20 success notifications
        for _ in range(20):
            notifications.send(_make_notification(success=True))

        forwarder.run(max_iterations=20, wait_time_seconds=0)

        # With sample_rate=0.5, roughly half should be forwarded
        # With seed=0, the exact count is deterministic
        assert 0 < forwarder.requests_sent < 20

    def test_zero_sample_rate_skips_successes(self) -> None:
        """With sample_rate=0, no successes are forwarded."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(
                objective="test",
                sample_rate=0.0,
                always_forward_failures=False,
            ),
        )

        notifications.send(_make_notification(success=True))
        forwarder.run(max_iterations=1, wait_time_seconds=0)

        assert forwarder.requests_sent == 0

    def test_budget_limits_forwarding(self) -> None:
        """Forwarder stops forwarding when budget is exhausted."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(
                objective="test",
                sample_rate=1.0,  # Forward everything
                budget=AnalysisBudget(max_requests=2),
            ),
        )

        # Send 5 success notifications
        for _ in range(5):
            notifications.send(_make_notification(success=True))

        forwarder.run(max_iterations=5, wait_time_seconds=0)

        # Only 2 should have been forwarded (budget limit)
        assert forwarder.requests_sent == 2

    def test_budget_resets_after_interval(self) -> None:
        """Budget resets after the configured interval."""
        clock = FakeClock()
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(
                objective="test",
                sample_rate=1.0,
                budget=AnalysisBudget(max_requests=1),
            ),
            clock=clock,
        )

        # First notification - should be forwarded
        notifications.send(_make_notification(success=True))
        forwarder.run(max_iterations=1, wait_time_seconds=0)
        assert forwarder.requests_sent == 1

        # Second notification - should be blocked (budget exhausted)
        notifications.send(_make_notification(success=True))
        forwarder.run(max_iterations=1, wait_time_seconds=0)
        assert forwarder.requests_sent == 1

        # Advance past budget reset interval
        clock.advance(3601)  # 1 hour + 1 second

        # Third notification - should be forwarded (budget reset)
        notifications.send(_make_notification(success=True))
        forwarder.run(max_iterations=1, wait_time_seconds=0)
        assert forwarder.requests_sent == 1  # Counter was reset

    def test_budget_blocks_failures_too(self) -> None:
        """Budget limits apply even to failures."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(
                objective="test",
                sample_rate=1.0,
                always_forward_failures=True,
                budget=AnalysisBudget(max_requests=1),
            ),
        )

        # First failure - forwarded
        notifications.send(_make_notification(success=False))
        forwarder.run(max_iterations=1, wait_time_seconds=0)
        assert forwarder.requests_sent == 1

        # Second failure - blocked by budget
        notifications.send(_make_notification(success=False))
        forwarder.run(max_iterations=1, wait_time_seconds=0)
        assert forwarder.requests_sent == 1

    def test_shutdown(self) -> None:
        """Forwarder supports graceful shutdown."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(objective="test"),
        )

        assert forwarder.running is False

    def test_context_manager(self) -> None:
        """Forwarder works as a context manager."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        with AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(objective="test"),
        ) as forwarder:
            assert isinstance(forwarder, AnalysisForwarder)

    def test_preserves_notification_source(self) -> None:
        """AnalysisRequest source matches the notification source."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(
                objective="test",
                sample_rate=1.0,
            ),
        )

        notifications.send(_make_notification(success=True, source="eval_loop"))
        forwarder.run(max_iterations=1, wait_time_seconds=0)

        msgs = analysis_requests.receive(max_messages=10, wait_time_seconds=0)
        assert len(msgs) == 1
        assert msgs[0].body.source == "eval_loop"

    def test_preserves_bundle_path(self) -> None:
        """AnalysisRequest bundles contain the notification's bundle path."""
        notifications: InMemoryMailbox[CompletionNotification, None] = InMemoryMailbox(
            name="notifications"
        )
        analysis_requests: InMemoryMailbox[AnalysisRequest, None] = InMemoryMailbox(
            name="analysis-requests"
        )

        forwarder = AnalysisForwarder(
            notifications=notifications,
            analysis_requests=analysis_requests,
            config=AnalysisForwarderConfig(
                objective="test",
                sample_rate=1.0,
            ),
        )

        expected_path = Path("/tmp/specific-bundle.zip")
        notification = CompletionNotification(
            source="agent_loop",
            bundle_path=expected_path,
            request_id=uuid4(),
            success=True,
        )
        notifications.send(notification)
        forwarder.run(max_iterations=1, wait_time_seconds=0)

        msgs = analysis_requests.receive(max_messages=10, wait_time_seconds=0)
        assert len(msgs) == 1
        assert msgs[0].body.bundles == (expected_path,)
