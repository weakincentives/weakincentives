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

"""Transactional tool execution.

A tool transaction snapshots the session and any registered
:class:`Snapshotable` resources before the handler runs, restores them on
any failure (raised exception or ``ToolResult.error``), and emits a
:class:`ToolInvoked` event for every invocation.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast, final

from .errors import ToolError, TransactionError, WinkError
from .prompt import Prompt
from .session import (
    PromptRendered,
    Session,
    ToolInvoked,
)
from .snapshot import Snapshot, Snapshotable, capture, restore
from .tool import ToolContext, ToolPolicy, ToolResult

__all__ = [
    "PendingToolTracker",
    "execute_tool",
    "tool_transaction",
]


@final
@dataclass(frozen=True, slots=True)
class _CompositeSnapshot:
    session: Snapshot
    resources: tuple[tuple[Snapshotable[Any], object], ...]


def _capture_composite(
    session: Session,
    snapshotables: Sequence[Snapshotable[Any]],
) -> _CompositeSnapshot:
    return _CompositeSnapshot(
        session=capture(session),
        resources=tuple((s, s.snapshot()) for s in snapshotables),
    )


def _restore_composite(session: Session, composite: _CompositeSnapshot) -> None:
    try:
        restore(session, composite.session)
        for snapshotable, state in composite.resources:
            snapshotable.restore(state)
    except Exception as error:
        raise TransactionError(f"failed to roll back transaction: {error}") from error


@contextmanager
def tool_transaction(
    session: Session,
    *,
    snapshotables: Sequence[Snapshotable[Any]] = (),
) -> Iterator[_CompositeSnapshot]:
    """Snapshot before the block; restore everything on any exception."""
    composite = _capture_composite(session, snapshotables)
    try:
        yield composite
    except BaseException:
        _restore_composite(session, composite)
        raise


def execute_tool(
    prompt: Prompt,
    session: Session,
    name: str,
    params: object,
    *,
    snapshotables: Sequence[Snapshotable[Any]] = (),
    policies: Sequence[ToolPolicy] = (),
) -> ToolResult[Any]:
    """Run a tool by name, rolling back state on any failure.

    Failure is either a raised exception or a ``ToolResult`` with
    ``success=False``. :class:`WinkError` instances re-raise after rollback;
    other exceptions are wrapped in :class:`ToolResult.error` so adapters can
    forward them to the model.
    """
    rendered = prompt.render()
    matches = [tool for tool in rendered.tools if tool.name == name]
    if not matches:
        raise ToolError(f"no tool named {name!r}")
    tool = matches[0]
    context = ToolContext(session=session, prompt=prompt, rendered=rendered)
    session.publish(
        PromptRendered(ns=prompt.ns, key=prompt.key, tool_count=len(rendered.tools))
    )
    composite = _capture_composite(session, snapshotables)
    for policy in policies:
        policy.before_invoke(tool, context)
    try:
        result = tool.handler(params, context)
    except WinkError:
        _restore_composite(session, composite)
        raise
    except Exception as error:
        _restore_composite(session, composite)
        wrapped = cast(
            "ToolResult[Any]",
            ToolResult.error(f"{type(error).__name__}: {error}"),
        )
        session.publish(ToolInvoked(name=name, success=False, message=wrapped.message))
        return wrapped
    if not result.success:
        _restore_composite(session, composite)
    for policy in policies:
        policy.on_result(tool, result)
    session.publish(
        ToolInvoked(name=name, success=result.success, message=result.message)
    )
    return result


# =============================================================================
# Pending-tool tracker for hook-based adapters
# =============================================================================


@dataclass(slots=True)
class _PendingExecution:
    tool_name: str
    snapshot: _CompositeSnapshot


@dataclass(slots=True)
class PendingToolTracker:
    """Track in-flight tool executions across pre/post hook boundaries.

    Adapters that drive native tool calls via callbacks (Claude Agent SDK,
    ACP, …) call :meth:`begin` before the tool runs and :meth:`end` after.
    State is rolled back automatically when ``success=False`` or
    :meth:`abort` is called.
    """

    session: Session
    snapshotables: tuple[Snapshotable[Any], ...] = ()
    _pending: dict[str, _PendingExecution] = field(
        default_factory=lambda: {},
        repr=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def begin(self, call_id: str, tool_name: str) -> None:
        composite = _capture_composite(self.session, self.snapshotables)
        with self._lock:
            self._pending[call_id] = _PendingExecution(
                tool_name=tool_name, snapshot=composite
            )

    def end(self, call_id: str, *, success: bool) -> bool:
        with self._lock:
            pending = self._pending.pop(call_id, None)
        if pending is None:
            return False
        if not success:
            _restore_composite(self.session, pending.snapshot)
            return True
        return False

    def abort(self, call_id: str) -> bool:
        with self._lock:
            pending = self._pending.pop(call_id, None)
        if pending is None:
            return False
        _restore_composite(self.session, pending.snapshot)
        return True
