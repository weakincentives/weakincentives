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

"""Data types for MainLoop request and response handling.

This module provides the data classes used by MainLoop for request/response
handling, including MainLoopRequest, MainLoopResult, and MainLoopConfig.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from ..budget import Budget
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from .lease_extender import LeaseExtenderConfig
from .run_context import RunContext

if TYPE_CHECKING:
    from ..evals._experiment import Experiment


@dataclass(frozen=True, slots=True)
class MainLoopResult[OutputT]:
    """Response from MainLoop execution.

    Consolidates success and failure into a single type. Check ``success``
    property to determine outcome.
    """

    request_id: UUID
    """Correlates with MainLoopRequest.request_id."""

    output: OutputT | None = None
    """Present on success. The parsed output from the prompt response."""

    error: str | None = None
    """Error message on failure."""

    session_id: UUID | None = None
    """Session that processed the request (if available)."""

    run_context: RunContext | None = None
    """Execution context with correlation identifiers and metadata."""

    completed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when processing completed."""

    @property
    def success(self) -> bool:
        """Return True if this result represents successful completion."""
        return self.error is None


@FrozenDataclass()
class MainLoopConfig:
    """Configuration for MainLoop execution defaults.

    Request-level ``budget``, ``deadline``, and ``resources`` override these defaults.

    The ``lease_extender`` field controls automatic message visibility extension
    during processing. When enabled, heartbeats from tool execution extend the
    message lease, preventing timeout during long-running requests.
    """

    deadline: Deadline | None = None
    budget: Budget | None = None
    resources: Mapping[type[object], object] | None = None
    lease_extender: LeaseExtenderConfig | None = None


@FrozenDataclass()
class MainLoopRequest[UserRequestT]:
    """Request for MainLoop execution with optional constraints.

    The ``budget``, ``deadline``, and ``resources`` fields override config defaults.
    The ``experiment`` field specifies a configuration variant for A/B testing.
    """

    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: Mapping[type[object], object] | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_context: RunContext | None = None
    """Optional execution context. If not provided, MainLoop creates one."""
    experiment: Experiment | None = None
    """Optional experiment for A/B testing. When provided, prepare() receives it."""


__all__ = [
    "MainLoopConfig",
    "MainLoopRequest",
    "MainLoopResult",
]
