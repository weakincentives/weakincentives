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

"""Tool policy types for enforcing constraints on tool invocations.

Policies are declarative constraints that gate tool calls. Each policy
evaluates independently; all must allow for a call to proceed (fail-closed).

Built-in policies use isolated session state: each policy type stores its
own state in a dedicated session slice, preventing cross-policy interference.

Combinators (:class:`AllOfPolicy`, :class:`AnyOfPolicy`) compose multiple
policies with AND/OR semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ..dataclasses import FrozenDataclass
from ..filesystem import strip_mount_point
from ..types.dataclass import SupportsDataclass

if TYPE_CHECKING:
    from .tool import Tool, ToolContext
    from .tool_result import ToolResult


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    """Result of a policy check.

    Contains the allow/deny verdict, an optional human-readable reason,
    and optional remediation suggestions the agent can act on.
    """

    allowed: bool
    reason: str | None = None
    suggestions: tuple[str, ...] = ()

    @classmethod
    def allow(cls) -> PolicyDecision:
        """Permit the tool call."""
        return cls(allowed=True)

    @classmethod
    def deny(
        cls,
        reason: str,
        *,
        suggestions: tuple[str, ...] = (),
    ) -> PolicyDecision:
        """Block the tool call with an explanation and optional suggestions."""
        return cls(allowed=False, reason=reason, suggestions=suggestions)


# ---------------------------------------------------------------------------
# Per-policy state types (isolated session slices)
# ---------------------------------------------------------------------------


@FrozenDataclass()
class SequentialDependencyState:
    """Session slice for :class:`SequentialDependencyPolicy`.

    Tracks which tools have been successfully invoked so that dependency
    checks can determine whether prerequisites are satisfied.
    """

    invoked_tools: frozenset[str] = frozenset()


@FrozenDataclass()
class ReadBeforeWriteState:
    """Session slice for :class:`ReadBeforeWritePolicy`.

    Tracks which (tool, path) pairs have been recorded as reads so that
    write attempts can verify prior read access.
    """

    invoked_keys: frozenset[tuple[str, str]] = frozenset()


@FrozenDataclass()
class PolicyState:
    """Generic policy state for custom policy implementations.

    Built-in policies use their own dedicated state types
    (:class:`SequentialDependencyState`, :class:`ReadBeforeWriteState`)
    for isolation. This type is provided as a convenience for custom
    policies that need simple tool/key tracking without defining their
    own state dataclass.
    """

    policy_name: str
    invoked_tools: frozenset[str] = frozenset()
    invoked_keys: frozenset[tuple[str, str]] = frozenset()  # (tool_name, key) pairs


# ---------------------------------------------------------------------------
# ToolPolicy protocol
# ---------------------------------------------------------------------------


class ToolPolicy(Protocol):
    """Protocol for declarative constraints on tool invocations.

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
        record the invocation in the session's state slice.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequentialDependencyPolicy:
    """Enforce unconditional tool invocation order.

    Tracks which tools have been successfully invoked and blocks tools
    whose prerequisites have not been satisfied. State is stored in a
    dedicated :class:`SequentialDependencyState` session slice.

    Example::

        policy = SequentialDependencyPolicy(
            dependencies={
                "deploy": frozenset({"test", "build"}),
                "build": frozenset({"lint"}),
            }
        )
        # Required order: lint -> build, then test, then deploy
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

        state = context.session[SequentialDependencyState].latest()
        invoked: frozenset[str] = state.invoked_tools if state else frozenset()
        missing = required - invoked

        if missing:
            sorted_missing = sorted(missing)
            return PolicyDecision.deny(
                f"Tool '{tool.name}' requires: {', '.join(sorted_missing)}",
                suggestions=tuple(f"Call '{t}' first." for t in sorted_missing),
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

        state = context.session[SequentialDependencyState].latest()
        current_tools: frozenset[str] = state.invoked_tools if state else frozenset()

        new_state = SequentialDependencyState(
            invoked_tools=current_tools | {tool.name},
        )
        context.session[SequentialDependencyState].seed(new_state)


# ---------------------------------------------------------------------------
# Path helpers (shared by ReadBeforeWritePolicy)
# ---------------------------------------------------------------------------


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


def _normalize_path(path: str, mount_point: str | None) -> str:
    """Normalize a path by stripping leading slashes and mount point prefix.

    This mirrors the normalization done by FilesystemToolHandlers so that
    policy checks use the same path form that handlers will use.

    Delegates to the shared :func:`strip_mount_point` implementation.
    """
    # Strip leading slashes (absolute -> relative)
    normalized = path.lstrip("/")
    # Strip mount point prefix using shared implementation
    return strip_mount_point(normalized, mount_point)


@dataclass(frozen=True)
class ReadBeforeWritePolicy:
    """Enforce read-before-write semantics on filesystem tools.

    A file must be read before it can be overwritten or edited. However,
    creating new files (paths that don't exist) is always allowed. State
    is stored in a dedicated :class:`ReadBeforeWriteState` session slice.

    Example::

        policy = ReadBeforeWritePolicy()

        # write_file(path="new.txt")      -> OK (file doesn't exist)
        # write_file(path="config.yaml")  -> DENIED (exists, not read)
        # read_file(path="config.yaml")   -> OK (records path)
        # write_file(path="config.yaml")  -> OK (was read)

    For tools that use mount point prefixes (e.g., Podman with /workspace),
    specify mount_point to normalize paths before existence checks::

        policy = ReadBeforeWritePolicy(mount_point="/workspace")

        # write_file(path="/workspace/config.yaml") checks "config.yaml"
    """

    read_tools: frozenset[str] = field(default_factory=lambda: frozenset({"read_file"}))
    write_tools: frozenset[str] = field(
        default_factory=lambda: frozenset({"write_file", "edit_file"})
    )
    mount_point: str | None = None

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

        raw_path = _extract_path(params)
        if raw_path is None:
            return PolicyDecision.allow()

        # Normalize path to match what handlers will use
        path = _normalize_path(raw_path, self.mount_point)

        # No filesystem available: allow (other safety checks apply)
        fs = context.filesystem
        if fs is None:
            return PolicyDecision.allow()

        # New file: allow creation without reading
        if not fs.exists(path):
            return PolicyDecision.allow()

        # Existing file: check if it was read
        state = context.session[ReadBeforeWriteState].latest()
        invoked_keys: frozenset[tuple[str, str]] = (
            state.invoked_keys if state else frozenset()
        )
        read_paths: set[str] = {k for t, k in invoked_keys if t in self.read_tools}

        if path not in read_paths:
            return PolicyDecision.deny(
                f"File '{raw_path}' must be read before overwriting.",
                suggestions=(f"Read '{raw_path}' first with a read tool.",),
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

        raw_path = _extract_path(params)
        if raw_path is None:
            return

        # Normalize path to match what check() will use
        path = _normalize_path(raw_path, self.mount_point)

        state = context.session[ReadBeforeWriteState].latest()
        current_keys: frozenset[tuple[str, str]] = (
            state.invoked_keys if state else frozenset()
        )

        new_state = ReadBeforeWriteState(
            invoked_keys=current_keys | {(tool.name, path)},
        )
        context.session[ReadBeforeWriteState].seed(new_state)


# ---------------------------------------------------------------------------
# Policy combinators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllOfPolicy:
    """Composite policy requiring all child policies to allow.

    Evaluates child policies in order. The first denial short-circuits
    and is returned immediately. If all allow, the call proceeds.

    Delegates ``on_result`` to all children so each can update its state.

    Example::

        policy = AllOfPolicy(policies=(
            ReadBeforeWritePolicy(),
            SequentialDependencyPolicy(dependencies={...}),
        ))
    """

    policies: Sequence[ToolPolicy]

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "all_of"

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Check all child policies; first denial wins."""
        for policy in self.policies:
            decision = policy.check(tool, params, context=context)
            if not decision.allowed:
                return decision
        return PolicyDecision.allow()

    def on_result(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        """Notify all child policies of the result."""
        for policy in self.policies:
            policy.on_result(tool, params, result, context=context)


@dataclass(frozen=True)
class AnyOfPolicy:
    """Composite policy requiring at least one child policy to allow.

    Evaluates child policies in order. The first allow short-circuits
    and the call proceeds. If all deny, a combined denial is returned.

    Delegates ``on_result`` to all children so each can update its state.

    Example::

        policy = AnyOfPolicy(policies=(
            policy_a,
            policy_b,
        ))
        # Call proceeds if either policy_a OR policy_b allows
    """

    policies: Sequence[ToolPolicy]

    @property
    def name(self) -> str:
        """Return the policy name."""
        return "any_of"

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Check child policies; first allow wins."""
        denials: list[PolicyDecision] = []
        for policy in self.policies:
            decision = policy.check(tool, params, context=context)
            if decision.allowed:
                return decision
            denials.append(decision)

        # All denied: combine reasons and suggestions
        reasons = [d.reason for d in denials if d.reason]
        all_suggestions: list[str] = []
        for d in denials:
            all_suggestions.extend(d.suggestions)

        return PolicyDecision.deny(
            "; ".join(reasons) if reasons else "All policies denied.",
            suggestions=tuple(all_suggestions),
        )

    def on_result(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        """Notify all child policies of the result."""
        for policy in self.policies:
            policy.on_result(tool, params, result, context=context)
