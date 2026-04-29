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

"""Tests for ``weakincentives.runcontext``."""

from __future__ import annotations

from weakincentives.clock import FakeClock
from weakincentives.runcontext import (
    RunContext,
    current_run_context,
    fresh_run_context,
    use_run_context,
)


def test_create_generates_ids() -> None:
    ctx = RunContext.create(clock=FakeClock())
    assert len(ctx.trace_id) == 32
    assert len(ctx.request_id) == 32
    assert ctx.attempt == 1
    assert ctx.parent_request_id is None


def test_run_context_default_attributes_is_empty_mapping() -> None:
    from datetime import datetime as _dt

    ctx = RunContext(
        trace_id="t",
        request_id="r",
        started_at=_dt.fromisoformat("2025-01-01T00:00:00+00:00"),
    )
    assert ctx.attributes == {}


def test_create_accepts_explicit_values() -> None:
    ctx = RunContext.create(
        trace_id="t",
        request_id="r",
        attributes={"env": "prod"},
        clock=FakeClock(),
    )
    assert ctx.trace_id == "t"
    assert ctx.request_id == "r"
    assert ctx.attributes == {"env": "prod"}


def test_child_inherits_trace_and_attributes() -> None:
    parent = RunContext.create(attributes={"env": "prod"}, clock=FakeClock())
    child = parent.child(attributes={"step": "render"}, clock=FakeClock())
    assert child.trace_id == parent.trace_id
    assert child.parent_request_id == parent.request_id
    assert child.attributes == {"env": "prod", "step": "render"}


def test_child_generates_request_id_by_default() -> None:
    parent = RunContext.create(clock=FakeClock())
    child = parent.child()
    assert child.request_id != parent.request_id


def test_child_accepts_explicit_request_id() -> None:
    parent = RunContext.create(clock=FakeClock())
    child = parent.child(request_id="explicit")
    assert child.request_id == "explicit"


def test_retry_increments_attempt() -> None:
    ctx = RunContext.create(clock=FakeClock())
    retried = ctx.retry()
    assert retried.attempt == 2
    assert retried.trace_id == ctx.trace_id
    assert retried.request_id == ctx.request_id


def test_use_run_context_binds_and_unbinds() -> None:
    assert current_run_context() is None
    ctx = fresh_run_context(clock=FakeClock())
    with use_run_context(ctx) as bound:
        assert bound is ctx
        assert current_run_context() is ctx
    assert current_run_context() is None


def test_nested_use_run_context_restores_outer() -> None:
    outer = fresh_run_context(clock=FakeClock())
    inner = fresh_run_context(clock=FakeClock())
    with use_run_context(outer):
        with use_run_context(inner):
            assert current_run_context() is inner
        assert current_run_context() is outer
    assert current_run_context() is None
