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

"""Hook context classes for Claude Agent SDK state synchronization.

Provides :class:`HookStats`, :class:`HookConstraints`, and :class:`HookContext`
which together manage the mutable state shared across hook callbacks during a
single SDK execution run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...budget import BudgetTracker
from ...deadlines import Deadline
from ...prompt.protocols import PromptProtocol
from ...runtime.run_context import RunContext
from ...runtime.session.protocols import SessionProtocol
from ...runtime.transactions import PendingToolTracker
from ...runtime.watchdog import Heartbeat
from ._bridge import MCPToolExecutionState

if TYPE_CHECKING:
    from ...prompt.prompt import PromptResources

__all__ = [
    "HookConstraints",
    "HookContext",
    "HookStats",
]


@dataclass(slots=True)
class HookStats:
    """Cumulative statistics tracked during hook execution.

    These metrics provide visibility into the execution flow for debugging.
    """

    tool_count: int = 0
    """Number of tools invoked during this execution."""

    turn_count: int = 0
    """Number of user prompt submissions (turns) during this execution."""

    subagent_count: int = 0
    """Number of subagents spawned during this execution."""

    in_subagent: bool = False
    """True when currently executing within a sub-agent context."""

    compact_count: int = 0
    """Number of context compaction events during this execution."""

    total_input_tokens: int = 0
    """Cumulative input tokens from all messages."""

    total_output_tokens: int = 0
    """Cumulative output tokens from all messages."""

    total_thinking_tokens: int = 0
    """Cumulative thinking tokens from extended thinking."""

    hook_errors: int = 0
    """Number of hook execution errors encountered."""


@dataclass(slots=True)
class HookConstraints:
    """Constraint configuration for hook execution.

    Groups optional deadline, budget, heartbeat, run context, and MCP tool state
    together to simplify HookContext construction.
    """

    deadline: Deadline | None = None
    """Optional deadline for constraint checking."""

    budget_tracker: BudgetTracker | None = None
    """Optional budget tracker for token limits."""

    heartbeat: Heartbeat | None = None
    """Optional heartbeat for liveness monitoring."""

    run_context: RunContext | None = None
    """Optional run context for tracing."""

    mcp_tool_state: MCPToolExecutionState | None = None
    """Shared state for passing tool_use_id from hooks to MCP bridge."""


class HookContext:
    """Context passed to hook callbacks for state access.

    Provides unified access to session, prompt resources, and tool transaction
    tracking for hook-based execution management.

    Note
    ----
    This is WINK's HookContext, distinct from the SDK's ``HookContext`` TypedDict
    which only contains a ``signal`` field for future abort support. WINK's
    HookContext provides richer functionality for session state management.
    """

    def __init__(
        self,
        *,
        session: SessionProtocol,
        prompt: PromptProtocol[object],
        adapter_name: str,
        prompt_name: str,
        constraints: HookConstraints | None = None,
    ) -> None:
        self._session = session
        self._prompt = prompt
        self.adapter_name = adapter_name
        self.prompt_name = prompt_name
        # Unpack constraints or use defaults
        self.deadline = constraints.deadline if constraints else None
        self.budget_tracker = constraints.budget_tracker if constraints else None
        self.heartbeat = constraints.heartbeat if constraints else None
        self.run_context = constraints.run_context if constraints else None
        self.mcp_tool_state = constraints.mcp_tool_state if constraints else None
        self.stop_reason: str | None = None
        self._tool_count = 0
        self._tool_tracker: PendingToolTracker | None = None
        self.stats: HookStats = HookStats()
        self._start_time = time.monotonic()

    def beat(self) -> None:
        """Record a heartbeat to prove processing is active.

        Hooks should call this during native tool execution to extend
        the message visibility timeout. This is a no-op if heartbeat
        is not configured.
        """
        if self.heartbeat is not None:
            self.heartbeat.beat()

    @property
    def session(self) -> SessionProtocol:
        """Get session."""
        return self._session

    @property
    def resources(
        self,
    ) -> PromptResources:  # pragma: no cover - tested via integration
        """Get resources from prompt."""
        return self._prompt.resources

    @property
    def _tracker(self) -> PendingToolTracker:
        """Get or create tool tracker (lazy initialization)."""
        if self._tool_tracker is None:
            self._tool_tracker = PendingToolTracker(
                session=self._session,
                resources=self._prompt.resources.context,
            )
        return self._tool_tracker

    def begin_tool_execution(self, tool_use_id: str, tool_name: str) -> None:
        """Take snapshot before native tool execution."""
        self._tracker.begin_tool_execution(tool_use_id, tool_name)

    def end_tool_execution(self, tool_use_id: str, *, success: bool) -> bool:
        """Complete tool execution, restoring on failure."""
        return self._tracker.end_tool_execution(tool_use_id, success=success)

    def abort_tool_execution(
        self, tool_use_id: str
    ) -> bool:  # pragma: no cover - tested via integration
        """Abort tool execution and restore state."""
        return self._tracker.abort_tool_execution(tool_use_id)

    @property
    def elapsed_ms(self) -> int:
        """Return elapsed time in milliseconds since context creation."""
        return int((time.monotonic() - self._start_time) * 1000)
