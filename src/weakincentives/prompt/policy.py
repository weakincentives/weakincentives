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
from ..filesystem._path import strip_mount_point
from ..types.dataclass import SupportsDataclass

if TYPE_CHECKING:
    from .tool import Tool, ToolContext
    from .tool_result import ToolResult


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    """Result of a policy check indicating whether a tool call should proceed.

    Use the factory methods :meth:`allow` and :meth:`deny` to create instances
    rather than constructing directly.

    Attributes:
        allowed: True if the tool call should proceed, False to block it.
        reason: Human-readable explanation when denied; None when allowed.

    Example::

        def check(self, tool, params, *, context) -> PolicyDecision:
            if tool.name in self.blocked_tools:
                return PolicyDecision.deny(f"Tool '{tool.name}' is blocked")
            return PolicyDecision.allow()
    """

    allowed: bool
    reason: str | None = None

    @classmethod
    def allow(cls) -> PolicyDecision:
        """Create a decision permitting the tool call to proceed.

        Returns:
            A PolicyDecision with allowed=True.
        """
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str) -> PolicyDecision:
        """Create a decision blocking the tool call.

        Args:
            reason: Human-readable explanation of why the call is denied.
                This message is returned to the model to inform retry logic.

        Returns:
            A PolicyDecision with allowed=False and the given reason.
        """
        return cls(allowed=False, reason=reason)


@FrozenDataclass()
class PolicyState:
    """Tracks which tools and keyed operations have been invoked for policy enforcement.

    This dataclass is stored in a session slice to maintain policy state
    across tool invocations. Policies update this state via
    :meth:`ToolPolicy.on_result` after successful tool executions.

    Two tracking modes are supported:

    - **Unconditional**: Track tool names in ``invoked_tools`` (e.g., "build was run")
    - **Keyed**: Track (tool, key) pairs in ``invoked_keys`` (e.g., "read_file on config.yaml")

    Attributes:
        policy_name: Identifier of the policy that owns this state.
        invoked_tools: Set of tool names that have been successfully invoked.
        invoked_keys: Set of (tool_name, key) pairs for parameter-keyed tracking,
            such as file paths for read-before-write enforcement.

    Example::

        # Access current state from a tool context
        state = context.session[PolicyState].latest()
        if state and "build" in state.invoked_tools:
            # build has already been run
            ...
    """

    policy_name: str
    invoked_tools: frozenset[str] = frozenset()
    invoked_keys: frozenset[tuple[str, str]] = frozenset()  # (tool_name, key) pairs


class ToolPolicy(Protocol):
    """Protocol for sequential dependency constraints on tool invocations.

    Implement this protocol to enforce ordering constraints on tool calls.
    Common patterns include:

    - **Unconditional dependencies**: Tool B requires tool A to run first
      (e.g., deploy requires build)
    - **Keyed dependencies**: Tool B on key X requires tool A on key X first
      (e.g., write_file on "config.yaml" requires read_file on "config.yaml")

    Policies are attached to prompts or sections and evaluated by the tool
    execution pipeline. State is tracked in the session via :class:`PolicyState`.

    Example implementation::

        @dataclass(frozen=True)
        class MyPolicy:
            @property
            def name(self) -> str:
                return "my_policy"

            def check(self, tool, params, *, context) -> PolicyDecision:
                # Check prerequisites and return allow/deny
                ...

            def on_result(self, tool, params, result, *, context) -> None:
                # Update PolicyState after successful execution
                ...
    """

    @property
    def name(self) -> str:
        """Unique identifier for this policy instance.

        Used to namespace :class:`PolicyState` entries when multiple policies
        are active. Should be a stable string like "read_before_write".
        """
        ...

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Evaluate whether the tool call should proceed.

        Called before handler execution. Inspect the tool, parameters, and
        session state to determine if prerequisites are satisfied.

        Args:
            tool: The tool being invoked.
            params: Parsed parameters dataclass, or None if parsing failed.
            context: Execution context with access to session and resources.

        Returns:
            :meth:`PolicyDecision.allow` to permit execution, or
            :meth:`PolicyDecision.deny` with a reason to block it.
            Denied calls return the reason as an error to the model.
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
        """Update session state after tool execution completes.

        Called after the handler returns, regardless of success or failure.
        Typically used to record successful invocations in :class:`PolicyState`
        so subsequent :meth:`check` calls can see the updated state.

        Args:
            tool: The tool that was invoked.
            params: Parsed parameters dataclass, or None if parsing failed.
            result: The result returned by the tool handler.
            context: Execution context with access to session and resources.

        Note:
            Only update state on ``result.success`` to avoid recording failed
            attempts as satisfied prerequisites.
        """
        ...


@dataclass(frozen=True)
class SequentialDependencyPolicy:
    """Enforce unconditional tool invocation order via a dependency graph.

    Tracks which tools have been successfully invoked and blocks tools
    whose prerequisites have not been satisfied. Dependencies are expressed
    as a mapping from tool name to the set of tools that must run first.

    Attributes:
        dependencies: Mapping from tool name to frozenset of required
            predecessor tool names. Tools not in this mapping have no
            prerequisites and are always allowed.

    Example::

        policy = SequentialDependencyPolicy(
            dependencies={
                "deploy": frozenset({"test", "build"}),
                "build": frozenset({"lint"}),
            }
        )
        # Required order: lint → build, then test, then deploy
        # (lint and test can run in any order relative to each other)

    Note:
        This policy tracks tool names only, not parameter values. For
        parameter-keyed dependencies (e.g., read file X before writing X),
        use :class:`ReadBeforeWritePolicy` or implement a custom policy.
    """

    dependencies: Mapping[str, frozenset[str]]  # tool -> required predecessors

    @property
    def name(self) -> str:
        """Return the policy identifier: ``"sequential_dependency"``."""
        return "sequential_dependency"

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Check if all prerequisite tools have been invoked.

        Args:
            tool: The tool being invoked.
            params: Parsed parameters (unused by this policy).
            context: Execution context for accessing session state.

        Returns:
            :meth:`PolicyDecision.allow` if the tool has no dependencies or
            all dependencies have been satisfied, otherwise
            :meth:`PolicyDecision.deny` listing the missing prerequisites.
        """
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
        """Record successful tool invocation in session state.

        Adds the tool name to ``PolicyState.invoked_tools`` so subsequent
        calls to :meth:`check` will see it as a satisfied dependency.
        Only records on success; failed invocations are not counted.

        Args:
            tool: The tool that was invoked.
            params: Parsed parameters (unused by this policy).
            result: The result from the tool handler.
            context: Execution context for updating session state.
        """
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


def _normalize_path(path: str, mount_point: str | None) -> str:
    """Normalize a path by stripping leading slashes and mount point prefix.

    This mirrors the normalization done by FilesystemToolHandlers so that
    policy checks use the same path form that handlers will use.

    Delegates to the shared :func:`strip_mount_point` implementation.
    """
    # Strip leading slashes (absolute → relative)
    normalized = path.lstrip("/")
    # Strip mount point prefix using shared implementation
    return strip_mount_point(normalized, mount_point)


@dataclass(frozen=True)
class ReadBeforeWritePolicy:
    """Enforce read-before-write semantics on filesystem tools.

    Prevents accidental overwrites by requiring a file to be read before it
    can be modified. Creating new files (paths that don't exist) is always
    allowed without a prior read.

    This policy tracks file paths in ``PolicyState.invoked_keys`` as
    ``(tool_name, normalized_path)`` tuples to enforce per-file constraints.

    Attributes:
        read_tools: Tool names that satisfy the "read" requirement.
            Defaults to ``{"read_file"}``.
        write_tools: Tool names that require a prior read for existing files.
            Defaults to ``{"write_file", "edit_file"}``.
        mount_point: Optional path prefix to strip when normalizing paths.
            Use this when tools operate with mount prefixes (e.g., Podman
            containers using ``/workspace`` as a mount point).

    Example::

        policy = ReadBeforeWritePolicy()

        # write_file(path="new.txt")      -> OK (file doesn't exist)
        # write_file(path="config.yaml")  -> DENIED (exists, not read)
        # read_file(path="config.yaml")   -> OK (records path)
        # write_file(path="config.yaml")  -> OK (was read)

    With mount point normalization::

        policy = ReadBeforeWritePolicy(mount_point="/workspace")

        # read_file(path="/workspace/config.yaml")  -> records "config.yaml"
        # write_file(path="/workspace/config.yaml") -> OK (normalized path matches)
    """

    read_tools: frozenset[str] = field(default_factory=lambda: frozenset({"read_file"}))
    write_tools: frozenset[str] = field(
        default_factory=lambda: frozenset({"write_file", "edit_file"})
    )
    mount_point: str | None = None

    @property
    def name(self) -> str:
        """Return the policy identifier: ``"read_before_write"``."""
        return "read_before_write"

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Check if an existing file was read before a write attempt.

        Applies only to tools in ``write_tools``. Non-write tools and writes
        to non-existent files (new file creation) are always allowed.

        Args:
            tool: The tool being invoked.
            params: Parsed parameters; must have a ``path`` or ``file_path``
                attribute for write tools.
            context: Execution context with filesystem and session access.

        Returns:
            :meth:`PolicyDecision.allow` if the tool is not a write tool,
            the target file doesn't exist, no filesystem is available, or
            the file was previously read. Otherwise :meth:`PolicyDecision.deny`
            with a message instructing the model to read the file first.
        """
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
        state = context.session[PolicyState].latest()
        invoked_keys: frozenset[tuple[str, str]] = (
            state.invoked_keys if state else frozenset()
        )
        read_paths: set[str] = {k for t, k in invoked_keys if t in self.read_tools}

        if path not in read_paths:
            return PolicyDecision.deny(
                f"File '{raw_path}' must be read before overwriting."
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
        """Record a successful read operation in session state.

        Adds the normalized file path to ``PolicyState.invoked_keys`` so
        subsequent :meth:`check` calls will allow writes to that path.
        Only records read tools on success.

        Args:
            tool: The tool that was invoked.
            params: Parsed parameters with ``path`` or ``file_path`` attribute.
            result: The result from the tool handler.
            context: Execution context for updating session state.
        """
        if not result.success:
            return
        if tool.name not in self.read_tools:
            return

        raw_path = _extract_path(params)
        if raw_path is None:
            return

        # Normalize path to match what check() will use
        path = _normalize_path(raw_path, self.mount_point)

        state = context.session[PolicyState].latest()
        if state is None:
            state = PolicyState(policy_name=self.name)

        new_state = PolicyState(
            policy_name=self.name,
            invoked_tools=state.invoked_tools,
            invoked_keys=state.invoked_keys | {(tool.name, path)},
        )
        context.session[PolicyState].seed(new_state)
