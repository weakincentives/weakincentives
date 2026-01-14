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

"""Tests for RunContext execution context."""

from __future__ import annotations

from uuid import UUID

import pytest

from weakincentives.runtime.run_context import RunContext


def test_run_context_default_values() -> None:
    """RunContext has sensible defaults."""
    ctx = RunContext()
    assert isinstance(ctx.run_id, UUID)
    assert isinstance(ctx.request_id, UUID)
    assert ctx.session_id is None
    assert ctx.attempt == 1
    assert ctx.worker_id == ""
    assert ctx.trace_id is None
    assert ctx.span_id is None


def test_run_context_custom_values() -> None:
    """RunContext accepts custom values."""
    run_id = UUID("11111111-1111-1111-1111-111111111111")
    request_id = UUID("22222222-2222-2222-2222-222222222222")
    session_id = UUID("33333333-3333-3333-3333-333333333333")
    ctx = RunContext(
        run_id=run_id,
        request_id=request_id,
        session_id=session_id,
        attempt=3,
        worker_id="worker-42",
        trace_id="trace-abc",
        span_id="span-xyz",
    )
    assert ctx.run_id == run_id
    assert ctx.request_id == request_id
    assert ctx.session_id == session_id
    assert ctx.attempt == 3
    assert ctx.worker_id == "worker-42"
    assert ctx.trace_id == "trace-abc"
    assert ctx.span_id == "span-xyz"


def test_run_context_is_frozen() -> None:
    """RunContext is immutable."""
    ctx = RunContext()
    with pytest.raises(AttributeError):
        ctx.attempt = 5  # type: ignore[misc]


def test_to_log_context_basic() -> None:
    """to_log_context returns dict with core fields."""
    run_id = UUID("11111111-1111-1111-1111-111111111111")
    request_id = UUID("22222222-2222-2222-2222-222222222222")
    ctx = RunContext(
        run_id=run_id,
        request_id=request_id,
        attempt=2,
        worker_id="worker-1",
    )
    log_ctx = ctx.to_log_context()
    assert log_ctx["run_id"] == str(run_id)
    assert log_ctx["request_id"] == str(request_id)
    assert log_ctx["attempt"] == 2
    assert log_ctx["worker_id"] == "worker-1"
    assert "session_id" not in log_ctx
    assert "trace_id" not in log_ctx
    assert "span_id" not in log_ctx


def test_to_log_context_with_session_id() -> None:
    """to_log_context includes session_id when set."""
    session_id = UUID("33333333-3333-3333-3333-333333333333")
    ctx = RunContext(session_id=session_id)
    log_ctx = ctx.to_log_context()
    assert log_ctx["session_id"] == str(session_id)


def test_to_log_context_with_trace_id() -> None:
    """to_log_context includes trace_id when set."""
    ctx = RunContext(trace_id="trace-abc")
    log_ctx = ctx.to_log_context()
    assert log_ctx["trace_id"] == "trace-abc"


def test_to_log_context_with_span_id() -> None:
    """to_log_context includes span_id when set."""
    ctx = RunContext(span_id="span-xyz")
    log_ctx = ctx.to_log_context()
    assert log_ctx["span_id"] == "span-xyz"


def test_to_log_context_all_fields() -> None:
    """to_log_context includes all fields when set."""
    run_id = UUID("11111111-1111-1111-1111-111111111111")
    request_id = UUID("22222222-2222-2222-2222-222222222222")
    session_id = UUID("33333333-3333-3333-3333-333333333333")
    ctx = RunContext(
        run_id=run_id,
        request_id=request_id,
        session_id=session_id,
        attempt=5,
        worker_id="worker-42",
        trace_id="trace-abc",
        span_id="span-xyz",
    )
    log_ctx = ctx.to_log_context()
    assert log_ctx == {
        "run_id": str(run_id),
        "request_id": str(request_id),
        "session_id": str(session_id),
        "attempt": 5,
        "worker_id": "worker-42",
        "trace_id": "trace-abc",
        "span_id": "span-xyz",
    }
