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

"""Metrics snapshot for point-in-time capture."""

from __future__ import annotations

from datetime import datetime

from ..dataclasses import FrozenDataclass
from ._types import AdapterMetrics, MailboxMetrics, ToolMetrics


@FrozenDataclass()
class MetricsSnapshot:
    """Point-in-time capture of all collected metrics.

    Attributes:
        adapters: Adapter metrics by (adapter, experiment) key.
        tools: Tool metrics by tool name.
        mailboxes: Mailbox metrics by queue name.
        captured_at: Timestamp when snapshot was taken.
        worker_id: Worker identifier, if available.
    """

    adapters: tuple[AdapterMetrics, ...]
    tools: tuple[ToolMetrics, ...]
    mailboxes: tuple[MailboxMetrics, ...]
    captured_at: datetime
    worker_id: str | None = None

    @classmethod
    def empty(cls, *, worker_id: str | None = None) -> MetricsSnapshot:
        """Create an empty snapshot.

        Args:
            worker_id: Optional worker identifier.

        Returns:
            Empty MetricsSnapshot with current timestamp.
        """
        from datetime import UTC

        return cls(
            adapters=(),
            tools=(),
            mailboxes=(),
            captured_at=datetime.now(UTC),
            worker_id=worker_id,
        )


__all__ = ["MetricsSnapshot"]
