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

"""Built-in :class:`ToolPolicy` implementations.

The spine ships zero concrete policies (``core.ToolPolicy`` is just a
protocol). The policies here are the common ones every project tends to
reach for. They are lightweight by design: each one tracks the bare
minimum of state needed to make its decision and operates only on
information :meth:`before_invoke` / :meth:`on_result` already receive.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, final

from weakincentives.core import (
    PolicyDeniedError,
    Tool,
    ToolContext,
    ToolResult,
)

__all__ = [
    "MaxInvocationsPolicy",
    "OnceOnlyPolicy",
    "ReadBeforeWritePolicy",
    "SequentialDependencyPolicy",
]


@final
@dataclass(slots=True)
class SequentialDependencyPolicy:
    """Require ``predecessor`` to have run successfully before ``successor``.

    Tools other than ``successor`` are unaffected. Only successful
    invocations of ``predecessor`` (i.e. ``ToolResult.success``) count.
    """

    predecessor: str
    successor: str
    _seen: bool = field(default=False, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def before_invoke(self, tool: Tool[Any, Any], context: ToolContext) -> None:
        del context
        if tool.name != self.successor:
            return
        with self._lock:
            allowed = self._seen
        if not allowed:
            raise PolicyDeniedError(
                f"{self.successor!r} requires successful {self.predecessor!r} first"
            )

    def on_result(self, tool: Tool[Any, Any], result: ToolResult[Any]) -> None:
        if not result.success or tool.name != self.predecessor:
            return
        with self._lock:
            self._seen = True


@final
@dataclass(slots=True)
class ReadBeforeWritePolicy:
    """Demand a successful read before any write.

    Tracks whether ``read_tool`` has produced a successful result; once
    it has, ``write_tool`` is allowed any number of times. The intent is
    to nudge agents toward "look before you leap" without trying to
    match individual paths — finer-grained guardrails belong in the
    handler itself, where the params are already strongly typed.
    """

    read_tool: str
    write_tool: str
    _read_succeeded: bool = field(default=False, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def before_invoke(self, tool: Tool[Any, Any], context: ToolContext) -> None:
        del context
        if tool.name != self.write_tool:
            return
        with self._lock:
            allowed = self._read_succeeded
        if not allowed:
            raise PolicyDeniedError(
                f"{self.write_tool!r} must follow successful {self.read_tool!r}"
            )

    def on_result(self, tool: Tool[Any, Any], result: ToolResult[Any]) -> None:
        if not result.success or tool.name != self.read_tool:
            return
        with self._lock:
            self._read_succeeded = True


@dataclass(slots=True)
class MaxInvocationsPolicy:
    """Cap how many times a tool (or any of a set) can be called.

    The cap is shared across ``targets`` so the policy is also a natural
    fit for "at most N file writes regardless of which writer".
    """

    max_calls: int
    targets: tuple[str, ...]
    _count: int = field(default=0, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def for_tool(cls, tool_name: str, *, max_calls: int) -> MaxInvocationsPolicy:
        return cls(max_calls=max_calls, targets=(tool_name,))

    @classmethod
    def for_tools(
        cls, tool_names: Iterable[str], *, max_calls: int
    ) -> MaxInvocationsPolicy:
        return cls(max_calls=max_calls, targets=tuple(tool_names))

    def before_invoke(self, tool: Tool[Any, Any], context: ToolContext) -> None:
        del context
        if tool.name not in self.targets:
            return
        with self._lock:
            if self._count >= self.max_calls:
                raise PolicyDeniedError(
                    f"{tool.name!r} reached max_calls={self.max_calls}"
                )
            self._count += 1

    def on_result(self, tool: Tool[Any, Any], result: ToolResult[Any]) -> None:
        del tool, result


@final
class OnceOnlyPolicy(MaxInvocationsPolicy):
    """A :class:`MaxInvocationsPolicy` with ``max_calls=1`` across targets."""

    def __init__(self, *targets: str) -> None:
        super().__init__(max_calls=1, targets=tuple(targets))
