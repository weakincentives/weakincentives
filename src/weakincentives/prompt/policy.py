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

"""Tool policy types for enforcing sequential dependencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ..dataclasses import FrozenDataclass
from ..types.dataclass import SupportsDataclass

if TYPE_CHECKING:
    from .tool import Tool, ToolContext
    from .tool_result import ToolResult


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    """Result of a policy check."""

    allowed: bool
    reason: str | None = None

    @classmethod
    def allow(cls) -> PolicyDecision:
        """Permit the tool call."""
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str) -> PolicyDecision:
        """Block the tool call with an explanation."""
        return cls(allowed=False, reason=reason)


@FrozenDataclass()
class PolicyState:
    """Tracks which tools/keys have been invoked for policy enforcement.

    This dataclass is stored in a session slice to track policy state
    across tool invocations. It supports both unconditional tool tracking
    and parameter-keyed tracking.
    """

    policy_name: str
    invoked_tools: frozenset[str] = frozenset()
    invoked_keys: frozenset[tuple[str, str]] = frozenset()  # (tool_name, key) pairs


class ToolPolicy(Protocol):
    """Protocol for sequential dependency constraints on tool invocations.

    Policies declare that tool B requires tool A to have been called first,
    either unconditionally or keyed by a parameter value. Policies are
    declared on prompts/sections; their state is tracked in the session.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this policy."""
        ...

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Evaluate whether the tool call should proceed.

        Called before handler execution. Return PolicyDecision.allow() to
        permit execution, or PolicyDecision.deny(reason) to block it.
        """
        ...

    def on_result(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        """Update session state after successful execution.

        Called after the handler returns a successful result. Use this to
        record the invocation in the session's PolicyState slice.
        """
        ...


@dataclass(frozen=True)
class SequentialDependencyPolicy:
    """Enforce unconditional tool invocation order.

    Tracks which tools have been successfully invoked and blocks tools
    whose prerequisites have not been satisfied.

    Example::

        policy = SequentialDependencyPolicy(
            dependencies={
                "deploy": frozenset({"test", "build"}),
                "build": frozenset({"lint"}),
            }
        )
        # Required order: lint → build, then test, then deploy
    """

    dependencies: Mapping[str, frozenset[str]]  # tool -> required predecessors

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "sequential_dependency"

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Check if all prerequisite tools have been invoked."""
        required = self.dependencies.get(tool.name, frozenset())
        if not required:
            return PolicyDecision.allow()

        state = context.session[PolicyState].latest()
        invoked: frozenset[str] = state.invoked_tools if state else frozenset()
        missing = required - invoked

        if missing:
            return PolicyDecision.deny(
                f"Tool '{tool.name}' requires: {', '.join(sorted(missing))}"
            )
        return PolicyDecision.allow()

    def on_result(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        """Record successful tool invocation in session state."""
        if not result.success:
            return

        state = context.session[PolicyState].latest()
        if state is None:
            state = PolicyState(policy_name=self.name)

        new_state = PolicyState(
            policy_name=self.name,
            invoked_tools=state.invoked_tools | {tool.name},
            invoked_keys=state.invoked_keys,
        )
        context.session[PolicyState].seed(new_state)


def _extract_path(params: SupportsDataclass | None) -> str | None:
    """Extract path from filesystem tool parameters."""
    if params is None:
        return None
    for field_name in ("path", "file_path"):
        if hasattr(params, field_name):
            value = getattr(params, field_name)
            if isinstance(value, str):
                return value
    return None


@dataclass(frozen=True)
class ReadBeforeWritePolicy:
    """Enforce read-before-write semantics on filesystem tools.

    A file must be read before it can be overwritten or edited. However,
    creating new files (paths that don't exist) is always allowed.

    Example::

        policy = ReadBeforeWritePolicy()

        # write_file(path="new.txt")      → OK (file doesn't exist)
        # write_file(path="config.yaml")  → DENIED (exists, not read)
        # read_file(path="config.yaml")   → OK (records path)
        # write_file(path="config.yaml")  → OK (was read)
    """

    read_tools: frozenset[str] = field(default_factory=lambda: frozenset({"read_file"}))
    write_tools: frozenset[str] = field(
        default_factory=lambda: frozenset({"write_file", "edit_file"})
    )

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "read_before_write"

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Check if file was read before write attempt."""
        if tool.name not in self.write_tools:
            return PolicyDecision.allow()

        path = _extract_path(params)
        if path is None:
            return PolicyDecision.allow()

        # No filesystem available: allow (other safety checks apply)
        fs = context.filesystem
        if fs is None:
            return PolicyDecision.allow()

        # New file: allow creation without reading
        if not fs.exists(path):
            return PolicyDecision.allow()

        # Existing file: check if it was read
        state = context.session[PolicyState].latest()
        invoked_keys: frozenset[tuple[str, str]] = (
            state.invoked_keys if state else frozenset()
        )
        read_paths: set[str] = {k for t, k in invoked_keys if t in self.read_tools}

        if path not in read_paths:
            return PolicyDecision.deny(
                f"File '{path}' must be read before overwriting."
            )
        return PolicyDecision.allow()

    def on_result(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        """Record read operation in session state."""
        if not result.success:
            return
        if tool.name not in self.read_tools:
            return

        path = _extract_path(params)
        if path is None:
            return

        state = context.session[PolicyState].latest()
        if state is None:
            state = PolicyState(policy_name=self.name)

        new_state = PolicyState(
            policy_name=self.name,
            invoked_tools=state.invoked_tools,
            invoked_keys=state.invoked_keys | {(tool.name, path)},
        )
        context.session[PolicyState].seed(new_state)
