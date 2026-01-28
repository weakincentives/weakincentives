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

"""Data types for AgentLoop request and response handling.

This module provides the data classes used by AgentLoop for request/response
handling, including AgentLoopRequest, AgentLoopResult, and AgentLoopConfig.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from ..budget import Budget
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..experiment import Experiment
from .lease_extender import LeaseExtenderConfig
from .run_context import RunContext

if TYPE_CHECKING:
    from ..adapters.core import PromptResponse
    from ..debug.bundle import BundleConfig, BundleWriter
    from .session import Session


@dataclass(frozen=True, slots=True)
class AgentLoopResult[OutputT]:
    """Response from AgentLoop execution.

    Consolidates success and failure into a single type. Check ``success``
    property to determine outcome.
    """

    request_id: UUID
    """Correlates with AgentLoopRequest.request_id."""

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

    bundle_path: Path | None = None
    """Path to debug bundle if bundling was enabled."""

    @property
    def success(self) -> bool:
        """Return True if this result represents successful completion."""
        return self.error is None


class BundleContext[OutputT]:
    """Context for bundled execution with metadata injection.

    Provides access to execution results and allows adding eval-specific
    metadata before the bundle is finalized.

    This class is yielded by AgentLoop.execute_with_bundle() and should not
    be instantiated directly.

    Attributes:
        response: The prompt response from execution.
        session: The session used for execution.
        latency_ms: Execution latency in milliseconds.
        bundle_path: Path to bundle (available after context manager exits).

    Example::

        with loop.execute_with_bundle(request, bundle_target=dir) as ctx:
            score = compute_score(ctx.response.output)
            ctx.write_metadata("eval", {
                "sample_id": "sample-1",
                "score": {"value": score.value, "passed": score.passed},
            })
        bundle_path = ctx.bundle_path  # Available after 'with' exits
    """

    __slots__ = ("_writer", "latency_ms", "response", "session")

    def __init__(
        self,
        writer: BundleWriter,
        response: PromptResponse[OutputT],
        session: Session,
        latency_ms: int,
    ) -> None:
        """Initialize bundle context."""
        super().__init__()
        self._writer = writer
        self.response = response
        self.session = session
        self.latency_ms = latency_ms

    @property
    def bundle_path(self) -> Path | None:
        """Path to the created bundle. Available after context manager exits."""
        return self._writer.path

    def write_metadata(self, name: str, data: Mapping[str, Any]) -> None:
        """Add arbitrary metadata to the bundle.

        Call this before the context manager exits to include domain-specific
        metadata in the bundle. The data will be written to {name}.json.

        Args:
            name: The metadata type name (e.g., "eval", "metrics").
            data: Dictionary of metadata to write.
        """
        self._writer.write_metadata(name, data)


@FrozenDataclass()
class AgentLoopConfig:
    """Configuration for AgentLoop execution defaults.

    Request-level ``budget``, ``deadline``, and ``resources`` override these defaults.

    The ``lease_extender`` field controls automatic message visibility extension
    during processing. When enabled, heartbeats from tool execution extend the
    message lease, preventing timeout during long-running requests.

    The ``debug_bundle`` field enables debug bundle creation for each execution.
    When set, AgentLoop automatically creates a bundle capturing request/response,
    session state, logs, and other artifacts.
    """

    deadline: Deadline | None = None
    budget: Budget | None = None
    resources: Mapping[type[object], object] | None = None
    lease_extender: LeaseExtenderConfig | None = None
    debug_bundle: BundleConfig | None = None


@FrozenDataclass()
class AgentLoopRequest[UserRequestT]:
    """Request for AgentLoop execution with optional constraints.

    The ``budget``, ``deadline``, and ``resources`` fields override config defaults.
    The ``experiment`` field specifies a configuration variant for A/B testing.
    The ``debug_bundle`` field overrides ``AgentLoopConfig.debug_bundle`` for this request.
    """

    request: UserRequestT
    budget: Budget | None = None
    deadline: Deadline | None = None
    resources: Mapping[type[object], object] | None = None
    request_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    run_context: RunContext | None = None
    """Optional execution context. If not provided, AgentLoop creates one."""
    experiment: Experiment | None = None
    """Optional experiment for A/B testing. When provided, prepare() receives it."""
    debug_bundle: BundleConfig | None = None
    """Optional per-request debug bundle config. Overrides AgentLoopConfig.debug_bundle."""


# ---------------------------------------------------------------------------
# Session State Slice Types for Request/Response Capture
# ---------------------------------------------------------------------------
# These types are written to the session during agent loop execution,
# allowing inspection of request/response data in the debug UI sessions view.


@FrozenDataclass()
class LoopRequestState[UserRequestT]:
    """Session state capturing the incoming request after prepare().

    This slice is published to the session immediately after prepare() is called,
    allowing inspection of the request that initiated the agent loop execution.
    """

    request: AgentLoopRequest[UserRequestT]
    """The incoming AgentLoopRequest object."""

    event_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this state entry."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when this state was captured."""


@FrozenDataclass()
class LoopRawResponse[OutputT]:
    """Session state capturing the response before finalize() transforms it.

    This slice is published to the session after adapter evaluation completes
    but before finalize() is called, capturing the raw adapter response.
    """

    prompt_name: str
    """Name of the prompt that was evaluated."""

    text: str | None
    """Raw text response from the model."""

    output: OutputT | None
    """Parsed structured output before finalize() transformation."""

    event_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this state entry."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when this state was captured."""


@FrozenDataclass()
class LoopFinalResponse[OutputT]:
    """Session state capturing the response after finalize() transforms it.

    This slice is published to the session after finalize() completes,
    capturing the final output that will be returned to the caller.
    """

    prompt_name: str
    """Name of the prompt that was evaluated."""

    text: str | None
    """Raw text response from the model."""

    output: OutputT | None
    """Final output after finalize() transformation."""

    event_id: UUID = field(default_factory=uuid4)
    """Unique identifier for this state entry."""

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """Timestamp when this state was captured."""


__all__ = [
    "AgentLoopConfig",
    "AgentLoopRequest",
    "AgentLoopResult",
    "BundleContext",
    "LoopFinalResponse",
    "LoopRawResponse",
    "LoopRequestState",
]
