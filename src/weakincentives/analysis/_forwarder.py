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

"""AnalysisForwarder: bridges completion notifications to analysis requests.

Consumes CompletionNotification messages, applies sampling and budget
policies, and forwards selected notifications as AnalysisRequest messages
to the AnalysisLoop's mailbox.
"""

from __future__ import annotations

import contextlib
import random
from typing import override

from ..clock import SYSTEM_CLOCK, MonotonicClock
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.mailbox import Mailbox, Message, ReceiptHandleExpiredError
from ..runtime.mailbox_worker import MailboxWorker
from ._types import (
    AnalysisForwarderConfig,
    AnalysisRequest,
    CompletionNotification,
)

_logger: StructuredLogger = get_logger(
    __name__, context={"component": "analysis.forwarder"}
)


class AnalysisForwarder(MailboxWorker[CompletionNotification, None]):
    """Consumes completion notifications and forwards selected ones for analysis.

    Bridges the gap between loop execution and analysis. Applies sampling
    to control volume and budget limits to control cost.

    The forwarder:
    - Samples notifications at the configured rate
    - Always forwards failures (if configured)
    - Stops forwarding when budget is exhausted
    - Resets budget after the configured interval

    Example::

        forwarder = AnalysisForwarder(
            notifications=notifications_mailbox,
            analysis_requests=analysis_requests_mailbox,
            config=AnalysisForwarderConfig(
                objective="Identify patterns in failing samples",
                sample_rate=0.1,
            ),
        )
        forwarder.run(max_iterations=100)
    """

    _analysis_requests: Mailbox[AnalysisRequest, None]
    _config: AnalysisForwarderConfig
    _requests_sent: int
    _budget_reset_at: float

    def __init__(
        self,
        *,
        notifications: Mailbox[CompletionNotification, None],
        analysis_requests: Mailbox[AnalysisRequest, None],
        config: AnalysisForwarderConfig,
        clock: MonotonicClock = SYSTEM_CLOCK,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize the AnalysisForwarder.

        Args:
            notifications: Mailbox to receive CompletionNotification messages from.
            analysis_requests: Mailbox to send AnalysisRequest messages to.
            config: Forwarder configuration (objective, sampling, budget).
            clock: Clock for budget reset timing.
            rng: Random number generator for sampling. If None, uses a new
                Random instance. Inject for deterministic testing.
        """
        super().__init__(requests=notifications)
        self._analysis_requests = analysis_requests
        self._config = config
        self._clock = clock
        self._rng = rng if rng is not None else random.Random()  # nosec B311 - sampling, not security
        self._requests_sent = 0
        self._budget_reset_at = (
            clock.monotonic() + config.budget.reset_interval.total_seconds()
        )

    @override
    def _process_message(self, msg: Message[CompletionNotification, None]) -> None:
        """Process a completion notification.

        Decides whether to forward based on sampling rate, failure policy,
        and budget. If forwarded, creates an AnalysisRequest and sends it
        to the analysis_requests mailbox.
        """
        notification = msg.body

        should_forward = self._should_forward(notification)

        if should_forward:
            request = AnalysisRequest(
                objective=self._config.objective,
                bundles=(notification.bundle_path,),
                source=notification.source,
            )
            _ = self._analysis_requests.send(request)
            self._requests_sent += 1
            _logger.debug(
                "Forwarded notification for analysis.",
                event="analysis.forwarder.forwarded",
                context={
                    "request_id": str(notification.request_id),
                    "source": notification.source,
                    "success": notification.success,
                },
            )

        # Always acknowledge - notifications are fire-and-forget
        with contextlib.suppress(ReceiptHandleExpiredError):
            msg.acknowledge()

    def _should_forward(self, notification: CompletionNotification) -> bool:
        """Decide whether a notification should be forwarded for analysis.

        Checks budget limits, failure policy, and sampling rate in order.

        Args:
            notification: The completion notification to evaluate.

        Returns:
            True if the notification should be forwarded.
        """
        # Reset budget if interval has elapsed
        now = self._clock.monotonic()
        if now >= self._budget_reset_at:
            self._requests_sent = 0
            self._budget_reset_at = (
                now + self._config.budget.reset_interval.total_seconds()
            )

        # Check budget
        if self._requests_sent >= self._config.budget.max_requests:
            return False

        # Always forward failures if configured
        if not notification.success and self._config.always_forward_failures:
            return True

        # Sample based on configured rate
        return self._rng.random() < self._config.sample_rate

    @property
    def requests_sent(self) -> int:
        """Number of analysis requests sent in the current budget window."""
        return self._requests_sent


__all__ = [
    "AnalysisForwarder",
]
