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

"""Protocol-level tests for ACP adapter - drain quiet period."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

from weakincentives.adapters.acp._execution import drain_quiet_period
from weakincentives.adapters.acp.config import ACPClientConfig


class TestDrainQuietPeriod:
    """Tests for drain_quiet_period behaviour."""

    def test_exits_immediately_when_no_updates(self) -> None:
        """Drain returns immediately when no updates have been received."""
        from weakincentives.adapters.acp.client import ACPClient
        from weakincentives.clock import SYSTEM_CLOCK

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        assert client.last_update_time is None

        # Should return immediately — not block for 5 s.
        asyncio.run(
            drain_quiet_period(
                client,
                deadline=None,
                quiet_period_ms=5000,
                clock=SYSTEM_CLOCK,
                async_sleeper=SYSTEM_CLOCK,
            )
        )

    def test_respects_max_drain_cap(self) -> None:
        """Drain exits within the max cap when no deadline is set."""
        from weakincentives.adapters.acp import _execution as execution_mod
        from weakincentives.adapters.acp.client import ACPClient
        from weakincentives.clock import SYSTEM_CLOCK

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        client._last_update_time = time.monotonic()

        original_max_drain = execution_mod._MAX_DRAIN_S
        execution_mod._MAX_DRAIN_S = 0.05  # 50 ms cap for test speed

        try:
            start = time.monotonic()
            asyncio.run(
                drain_quiet_period(
                    client,
                    deadline=None,
                    quiet_period_ms=60_000,
                    clock=SYSTEM_CLOCK,
                    async_sleeper=SYSTEM_CLOCK,
                )
            )
            elapsed = time.monotonic() - start

            # Should finish well under 1 s (capped at ~50 ms, not 60 s).
            assert elapsed < 1.0
        finally:
            execution_mod._MAX_DRAIN_S = original_max_drain

    def test_drain_snapshot_consistency(self) -> None:
        """Drain uses a snapshot of last_update_time per iteration."""
        from weakincentives.adapters.acp import _execution as execution_mod
        from weakincentives.adapters.acp.client import ACPClient
        from weakincentives.clock import SYSTEM_CLOCK

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        # Set last_update_time to "just now" so drain waits for quiet period.
        client._last_update_time = time.monotonic()

        original_max_drain = execution_mod._MAX_DRAIN_S
        execution_mod._MAX_DRAIN_S = 1.0

        try:
            start = time.monotonic()
            asyncio.run(
                drain_quiet_period(
                    client,
                    deadline=None,
                    quiet_period_ms=100,
                    clock=SYSTEM_CLOCK,
                    async_sleeper=SYSTEM_CLOCK,
                )
            )
            elapsed = time.monotonic() - start

            # Should terminate after ~100 ms quiet period, not hang.
            assert elapsed < 1.0
            # Should have waited at least the quiet period.
            assert elapsed >= 0.05
        finally:
            execution_mod._MAX_DRAIN_S = original_max_drain

    def test_drain_exits_when_snapshot_becomes_none(self) -> None:
        """Drain exits if last_update_time becomes None mid-loop."""
        from weakincentives.adapters.acp import _execution as execution_mod
        from weakincentives.adapters.acp.client import ACPClient
        from weakincentives.clock import SYSTEM_CLOCK

        client = ACPClient(ACPClientConfig(), workspace_root="/tmp")
        client._last_update_time = time.monotonic()

        original_max_drain = execution_mod._MAX_DRAIN_S
        execution_mod._MAX_DRAIN_S = 5.0

        original_sleep = asyncio.sleep

        # Create a custom async sleeper that clears last_update_time mid-loop.
        mock_sleeper = MagicMock()

        async def _clear_and_sleep(s: float) -> None:
            client._last_update_time = None
            await original_sleep(min(s, 0.01))

        mock_sleeper.async_sleep = _clear_and_sleep

        try:
            start = time.monotonic()
            asyncio.run(
                drain_quiet_period(
                    client,
                    deadline=None,
                    quiet_period_ms=60_000,
                    clock=SYSTEM_CLOCK,
                    async_sleeper=mock_sleeper,
                )
            )
            elapsed = time.monotonic() - start

            # Should exit quickly, not wait the full 60 s quiet period.
            assert elapsed < 1.0
        finally:
            execution_mod._MAX_DRAIN_S = original_max_drain
