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

"""Execution metadata: trace ids, request ids, attempt counters.

A :class:`RunContext` is a small immutable bag of correlation ids that
flows alongside a request. Tools, listeners, and adapters can accept
the same value for free — it's just a frozen dataclass — and emit
structured logs / transcript entries that line up across the stack.
"""

from __future__ import annotations

import contextvars
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import final

from weakincentives.clock import SYSTEM_CLOCK, WallClock

__all__ = [
    "RunContext",
    "current_run_context",
    "fresh_run_context",
    "use_run_context",
]


def _now(clock: WallClock) -> datetime:
    return clock.utcnow()


def _empty_attributes() -> Mapping[str, str]:
    return {}


@final
@dataclass(frozen=True, slots=True)
class RunContext:
    """A correlation envelope for one logical request.

    ``trace_id`` is the long-lived identifier shared across every
    component handling the same request. ``request_id`` differentiates
    sub-requests (for example, individual agent iterations) inside that
    trace. ``attempt`` counts retries.
    """

    trace_id: str
    request_id: str
    started_at: datetime
    attempt: int = 1
    parent_request_id: str | None = None
    attributes: Mapping[str, str] = field(default_factory=_empty_attributes)

    @classmethod
    def create(
        cls,
        *,
        trace_id: str | None = None,
        request_id: str | None = None,
        attributes: Mapping[str, str] | None = None,
        clock: WallClock = SYSTEM_CLOCK,
    ) -> RunContext:
        """Build a fresh :class:`RunContext` with generated ids."""
        return cls(
            trace_id=trace_id or _new_id(),
            request_id=request_id or _new_id(),
            started_at=_now(clock),
            attempt=1,
            parent_request_id=None,
            attributes=dict(attributes) if attributes is not None else {},
        )

    def child(
        self,
        *,
        request_id: str | None = None,
        attributes: Mapping[str, str] | None = None,
        clock: WallClock = SYSTEM_CLOCK,
    ) -> RunContext:
        """Return a child context that shares this trace."""
        merged: dict[str, str] = dict(self.attributes)
        if attributes is not None:
            merged.update(attributes)
        return RunContext(
            trace_id=self.trace_id,
            request_id=request_id or _new_id(),
            started_at=_now(clock),
            attempt=1,
            parent_request_id=self.request_id,
            attributes=merged,
        )

    def retry(self) -> RunContext:
        """Return a copy with ``attempt`` incremented by one."""
        return replace(self, attempt=self.attempt + 1)


def _new_id() -> str:
    return uuid.uuid4().hex


# =============================================================================
# Implicit context propagation
# =============================================================================


_CURRENT: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "weakincentives_run_context",
    default=None,
)
_LOCK = threading.Lock()


def current_run_context() -> RunContext | None:
    """Return the active :class:`RunContext`, if any."""
    return _CURRENT.get()


def fresh_run_context(
    *,
    trace_id: str | None = None,
    request_id: str | None = None,
    attributes: Mapping[str, str] | None = None,
    clock: WallClock = SYSTEM_CLOCK,
) -> RunContext:
    """Convenience wrapper around :meth:`RunContext.create`."""
    with _LOCK:
        return RunContext.create(
            trace_id=trace_id,
            request_id=request_id,
            attributes=attributes,
            clock=clock,
        )


@contextmanager
def use_run_context(context: RunContext) -> Iterator[RunContext]:
    """Bind ``context`` as the implicit run context inside the block."""
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)
