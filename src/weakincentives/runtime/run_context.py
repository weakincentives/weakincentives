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

"""Immutable execution context for request processing."""

from __future__ import annotations

from dataclasses import field
from uuid import UUID, uuid4

from ..dataclasses import FrozenDataclass


@FrozenDataclass()
class RunContext:
    """Immutable context capturing execution metadata for a single request run.

    RunContext provides correlation identifiers and execution metadata
    that flows through the system from AgentLoop to tool handlers and
    telemetry events. This enables distributed tracing and debugging.

    Attributes:
        run_id: Unique identifier for this execution run. Generated fresh
            for each run, distinct from request_id which may be retried.
        request_id: Correlates with AgentLoopRequest.request_id. Stable
            across retries of the same logical request.
        session_id: UUID of the session processing this request, if available.
        attempt: Delivery count from mailbox (1 = first attempt). Maps to
            Message.delivery_count for retry tracking.
        worker_id: Identifier for the worker processing this request.
            Useful for debugging which worker handled a request.
        trace_id: Optional distributed trace identifier (e.g., OpenTelemetry trace ID).
        span_id: Optional span identifier within the trace.

    Example:
        Bind RunContext fields to a structured logger::

            from weakincentives.runtime import RunContext, get_logger

            run_ctx = RunContext(worker_id="worker-1")
            log = get_logger(__name__)
            bound_log = log.bind(**run_ctx.to_log_context())
    """

    run_id: UUID = field(default_factory=uuid4)
    request_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    attempt: int = 1
    worker_id: str = ""
    trace_id: str | None = None
    span_id: str | None = None

    def to_log_context(self) -> dict[str, str | int | None]:
        """Return fields as a dict suitable for StructuredLogger.bind.

        UUIDs are converted to strings for JSON serialization.
        Only non-None optional fields are included.

        Returns:
            Dict with run_id, request_id, session_id, attempt, worker_id,
            and optionally trace_id and span_id.
        """
        ctx: dict[str, str | int | None] = {
            "run_id": str(self.run_id),
            "request_id": str(self.request_id),
            "attempt": self.attempt,
            "worker_id": self.worker_id,
        }
        if self.session_id is not None:
            ctx["session_id"] = str(self.session_id)
        if self.trace_id is not None:
            ctx["trace_id"] = self.trace_id
        if self.span_id is not None:
            ctx["span_id"] = self.span_id
        return ctx


__all__ = ["RunContext"]
