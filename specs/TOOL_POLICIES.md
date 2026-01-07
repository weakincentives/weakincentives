# Tool Policy Specification

## Purpose

Tool policies provide a mechanism to enforce stateful constraints on tool
invocations. The core abstraction is sequential dependencies: requiring certain
tools to be invoked before others. This can be unconditional ("run tests before
deploy") or parameter-keyed ("read file X before writing file X"). Policies are
bound to a session and track invocation history to make allow/deny decisions.

## Guiding Principles

- **Stateful by design**: Policies maintain state across tool invocations to
  enforce temporal constraints (e.g., "must read before write").
- **Composable**: Multiple policies can govern the same tool; all must allow
  for execution to proceed.
- **Fail-closed**: When a policy denies a tool call, the tool returns an error
  result without executing the handler.
- **Session-scoped**: Policy state is tied to the session lifecycle and can be
  snapshotted/restored alongside session state.
- **Declarative bindings**: Policies declare which tools they govern via name
  patterns or explicit tool references.

```mermaid
flowchart TB
    subgraph Invocation["Tool Invocation with Policies"]
        LLM["LLM Response"] --> Parse["Parse tool_call"]
        Parse --> Validate["Validate params<br/>(serde.parse)"]
        Validate --> PolicyCheck{"Check<br/>Policies"}
        PolicyCheck -->|All Allow| BuildCtx["Build ToolContext"]
        PolicyCheck -->|Any Deny| PolicyError["ToolResult(success=False)"]
        BuildCtx --> Handler["Execute handler"]
        Handler --> PostPolicy["Post-execution<br/>policy hooks"]
    end

    subgraph PolicyState["Policy State Management"]
        Session["Session"] --> PolicyRegistry["PolicyRegistry"]
        PolicyRegistry --> Policy1["Policy 1"]
        PolicyRegistry --> Policy2["Policy 2"]
        Policy1 --> StateSlice["Policy State<br/>(session slice)"]
        Policy2 --> StateSlice
    end
```

## Core Types

### ToolPolicy Protocol

```python
class ToolPolicy(Protocol):
    """Stateful constraint on tool invocations."""

    @property
    def name(self) -> str:
        """Unique identifier for this policy."""
        ...

    @property
    def tools(self) -> frozenset[str]:
        """Tool names this policy governs (empty = all tools)."""
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
        """Hook called after successful tool execution.

        Use this to update policy state based on the result. Not called
        when the policy denies execution or the handler raises.
        """
        ...
```

### PolicyDecision

```python
@dataclass(slots=True, frozen=True)
class PolicyDecision:
    """Result of a policy check."""

    allowed: bool
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> PolicyDecision:
        """Permit the tool call."""
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str, **metadata: Any) -> PolicyDecision:
        """Block the tool call with an explanation."""
        return cls(allowed=False, reason=reason, metadata=metadata)
```

### PolicyRegistry

```python
@dataclass
class PolicyRegistry:
    """Container for policies bound to a session."""

    session: SessionProtocol
    _policies: list[ToolPolicy] = field(default_factory=list)

    def register(self, policy: ToolPolicy) -> None:
        """Add a policy to the registry."""
        ...

    def unregister(self, policy_name: str) -> bool:
        """Remove a policy by name. Returns True if found."""
        ...

    def policies_for(self, tool_name: str) -> tuple[ToolPolicy, ...]:
        """Return all policies that govern the given tool."""
        ...

    def check_all(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        """Check all applicable policies. Returns first denial or allow."""
        ...

    def notify_result(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        """Notify all applicable policies of execution result."""
        ...
```

## Sequential Dependency Model

The core abstraction is **sequential dependencies**: tool B requires tool A to
have been successfully invoked first. This comes in two forms:

1. **Unconditional**: "tool A must be called before tool B" (e.g., `test` before
   `deploy`)
2. **Parameter-keyed**: "tool A with key X must be called before tool B with
   key X" (e.g., `read_file(path=X)` before `write_file(path=X)`)

```mermaid
flowchart LR
    subgraph Unconditional["Unconditional Dependency"]
        test["test"] --> deploy["deploy"]
    end

    subgraph Keyed["Parameter-Keyed Dependency"]
        read1["read_file(path=X)"] --> write1["write_file(path=X)"]
        read2["read_file(path=Y)"] --> write2["write_file(path=Y)"]
    end
```

## Built-in Policies

### SequentialDependencyPolicy

The foundation policy that enforces unconditional tool ordering:

```python
@dataclass
class SequentialDependencyPolicy:
    """Enforce tool invocation order.

    Tracks which tools have been successfully invoked and blocks tools
    whose prerequisites have not been satisfied.
    """

    dependencies: Mapping[str, frozenset[str]]  # tool -> required predecessors
    _invoked: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return "sequential_dependency"

    @property
    def tools(self) -> frozenset[str]:
        all_tools: set[str] = set(self.dependencies.keys())
        for prereqs in self.dependencies.values():
            all_tools.update(prereqs)
        return frozenset(all_tools)

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        required = self.dependencies.get(tool.name, frozenset())
        missing = required - self._invoked
        if missing:
            return PolicyDecision.deny(
                f"Tool '{tool.name}' requires prior invocation of: "
                f"{', '.join(sorted(missing))}",
                tool=tool.name,
                missing=tuple(sorted(missing)),
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
        if result.success:
            self._invoked.add(tool.name)
```

**Example usage:**

```python
# Deploy requires both test and build to have succeeded
policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
        "build": frozenset({"lint"}),
    }
)

# Invocation order must be: lint → build → test → deploy
# (test and build can be in either order after lint)
```

### KeyedDependencyPolicy

Extends sequential dependencies with parameter-based tracking. A tool with
key X only requires the prerequisite with the same key X:

```python
KeyExtractor = Callable[[SupportsDataclass | None], str | None]

@dataclass
class KeyedDependencyPolicy:
    """Enforce tool ordering keyed by a parameter value.

    Unlike SequentialDependencyPolicy which tracks tool names globally,
    this policy tracks (tool_name, key) pairs. A dependent tool is only
    blocked if its specific key hasn't been enabled by a prerequisite.
    """

    # tool_name -> (prerequisite_tools, key_extractor)
    dependencies: Mapping[str, tuple[frozenset[str], KeyExtractor]]
    # (tool_name, key) pairs that have been satisfied
    _satisfied: set[tuple[str, str]] = field(default_factory=set)

    @property
    def name(self) -> str:
        return "keyed_dependency"

    @property
    def tools(self) -> frozenset[str]:
        all_tools: set[str] = set(self.dependencies.keys())
        for prereqs, _ in self.dependencies.values():
            all_tools.update(prereqs)
        return frozenset(all_tools)

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        dep = self.dependencies.get(tool.name)
        if dep is None:
            return PolicyDecision.allow()

        prereqs, key_fn = dep
        key = key_fn(params)
        if key is None:
            return PolicyDecision.allow()  # No key = no constraint

        # Check if any prerequisite has been satisfied for this key
        for prereq in prereqs:
            if (prereq, key) in self._satisfied:
                return PolicyDecision.allow()

        return PolicyDecision.deny(
            f"Tool '{tool.name}' with key '{key}' requires prior invocation "
            f"of one of: {', '.join(sorted(prereqs))} with the same key.",
            tool=tool.name,
            key=key,
            prerequisites=tuple(sorted(prereqs)),
        )

    def on_result(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        if not result.success:
            return

        # Check if this tool is a prerequisite for anything
        for dep_tool, (prereqs, key_fn) in self.dependencies.items():
            if tool.name in prereqs:
                key = key_fn(params)
                if key is not None:
                    self._satisfied.add((tool.name, key))
```

### ReadBeforeWritePolicy

A specialized `KeyedDependencyPolicy` for filesystem tools, using the file
path as the key:

```python
def _extract_path(params: SupportsDataclass | None) -> str | None:
    """Extract path from common filesystem tool parameter patterns."""
    if params is None:
        return None
    # Try common field names
    for field in ("path", "file_path", "filepath"):
        if hasattr(params, field):
            return getattr(params, field)
    return None


@dataclass
class ReadBeforeWritePolicy:
    """Enforce read-before-write semantics on filesystem tools.

    Built on KeyedDependencyPolicy with path as the key. A file must be
    read before it can be written or edited.
    """

    read_tools: frozenset[str] = frozenset({"read_file", "vfs_read_file"})
    write_tools: frozenset[str] = frozenset({
        "write_file", "edit_file", "vfs_write_file", "vfs_edit_file"
    })
    _read_paths: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return "read_before_write"

    @property
    def tools(self) -> frozenset[str]:
        return self.read_tools | self.write_tools

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        if tool.name not in self.write_tools:
            return PolicyDecision.allow()

        path = _extract_path(params)
        if path is None:
            return PolicyDecision.allow()

        if path not in self._read_paths:
            return PolicyDecision.deny(
                f"File '{path}' must be read before writing. "
                f"Use one of: {', '.join(sorted(self.read_tools))} first.",
                path=path,
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
        if tool.name in self.read_tools and result.success:
            path = _extract_path(params)
            if path is not None:
                self._read_paths.add(path)
```

**Example usage:**

```python
policy = ReadBeforeWritePolicy()

# This sequence is allowed:
# read_file(path="config.yaml")  → OK, records path
# write_file(path="config.yaml") → OK, path was read
# write_file(path="other.yaml")  → DENIED, other.yaml not read

# Custom tool names:
policy = ReadBeforeWritePolicy(
    read_tools=frozenset({"fetch_document"}),
    write_tools=frozenset({"update_document", "delete_document"}),
)
```

### Relationship Between Policies

```
SequentialDependencyPolicy (unconditional)
    │
    └── KeyedDependencyPolicy (parameter-keyed)
            │
            └── ReadBeforeWritePolicy (path-keyed filesystem specialization)
```

`ReadBeforeWritePolicy` can be seen as syntactic sugar over
`KeyedDependencyPolicy`:

```python
# These are equivalent:
rbw = ReadBeforeWritePolicy()

keyed = KeyedDependencyPolicy(
    dependencies={
        "write_file": (frozenset({"read_file"}), _extract_path),
        "edit_file": (frozenset({"read_file"}), _extract_path),
    }
)
```

## Session Integration

### Binding Policies to Sessions

Policies are registered via the session's policy registry:

```python
from weakincentives.runtime import Session
from weakincentives.policies import (
    PolicyRegistry,
    ReadBeforeWritePolicy,
    SequentialDependencyPolicy,
)

session = Session(bus=bus)

# Create and register policies
policies = PolicyRegistry(session=session)
policies.register(ReadBeforeWritePolicy())
policies.register(SequentialDependencyPolicy(
    dependencies={"deploy": frozenset({"test", "build"})}
))

# Bind to tool execution context
response = adapter.evaluate(
    prompt,
    params,
    session=session,
    policy_registry=policies,
)
```

### State Persistence

Policy state can be persisted via a dedicated session slice:

```python
@FrozenDataclass()
class PolicyState:
    """Snapshot of policy state for persistence."""

    policy_name: str
    state: Mapping[str, Any]
    created_at: datetime

# Policies can store state in session
session[PolicyState].seed(PolicyState(
    policy_name="read_before_write",
    state={"read_paths": list(policy._read_paths)},
    created_at=datetime.now(UTC),
))
```

### Snapshot and Restore

The `PolicyRegistry` integrates with session snapshots:

```python
# Capture
snapshot = session.snapshot()
policy_snapshot = policies.snapshot()

# Restore
session.restore(snapshot)
policies.restore(policy_snapshot)
```

## Execution Flow Integration

### Tool Executor Changes

The tool executor checks policies before handler invocation:

```python
@contextmanager
def tool_execution(
    call: ToolCall,
    *,
    execution_context: ToolExecutionContext,
    policy_registry: PolicyRegistry | None = None,
) -> Iterator[ToolExecutionOutcome]:
    tool = resolve_tool(call, execution_context)
    params = parse_params(call, tool)

    # Policy check (new step)
    if policy_registry is not None:
        decision = policy_registry.check_all(
            tool,
            params,
            context=build_context(execution_context),
        )
        if not decision.allowed:
            yield ToolExecutionOutcome(
                tool=tool,
                params=params,
                result=ToolResult.error(decision.reason or "Policy denied"),
                call_id=call.id,
                log=log,
                snapshot=snapshot,
            )
            return

    # Execute handler
    result = execute_handler(tool, params, context)

    # Notify policies of result (new step)
    if policy_registry is not None:
        policy_registry.notify_result(tool, params, result, context=context)

    yield ToolExecutionOutcome(...)
```

### ToolContext Extension

Policies can access the policy registry via an extended context:

```python
@dataclass(slots=True, frozen=True)
class ToolContext:
    # ... existing fields ...
    policy_registry: PolicyRegistry | None = None
```

This allows handlers to query policy state if needed (e.g., checking which
files have been read).

## Custom Policies

### Implementing a Custom Policy

Example: A policy requiring code review approval before merge, keyed by PR:

```python
from dataclasses import dataclass, field
from weakincentives.policies import ToolPolicy, PolicyDecision

@dataclass
class ReviewBeforeMergePolicy:
    """Require review approval before merging a PR.

    Tracks which PRs have been approved via the review tool and blocks
    merge attempts for PRs without approval.
    """

    review_tool: str = "approve_pr"
    merge_tool: str = "merge_pr"
    _approved_prs: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return "review_before_merge"

    @property
    def tools(self) -> frozenset[str]:
        return frozenset({self.review_tool, self.merge_tool})

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        if tool.name != self.merge_tool:
            return PolicyDecision.allow()

        pr_id = getattr(params, "pr_id", None) if params else None
        if pr_id is None:
            return PolicyDecision.allow()

        if pr_id not in self._approved_prs:
            return PolicyDecision.deny(
                f"PR '{pr_id}' must be approved before merging. "
                f"Use {self.review_tool} first.",
                pr_id=pr_id,
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
        if tool.name == self.review_tool and result.success:
            pr_id = getattr(params, "pr_id", None) if params else None
            if pr_id is not None:
                self._approved_prs.add(pr_id)
```

### Pattern-Based Tool Matching

Policies can use glob patterns for tool matching:

```python
@dataclass
class PatternPolicy:
    """Base class for policies with pattern-based tool matching."""

    patterns: tuple[str, ...]  # e.g., ("vfs_*", "podman_*")

    @property
    def tools(self) -> frozenset[str]:
        # Empty set signals dynamic matching
        return frozenset()

    def matches(self, tool_name: str) -> bool:
        return any(fnmatch(tool_name, p) for p in self.patterns)
```

The registry calls `matches()` when `tools` is empty:

```python
def policies_for(self, tool_name: str) -> tuple[ToolPolicy, ...]:
    result = []
    for policy in self._policies:
        if not policy.tools:
            # Dynamic matching via pattern
            if hasattr(policy, "matches") and policy.matches(tool_name):
                result.append(policy)
        elif tool_name in policy.tools:
            result.append(policy)
    return tuple(result)
```

## Composition

### Policy Evaluation Order

Policies are evaluated in registration order. The first denial stops evaluation:

```python
def check_all(
    self,
    tool: Tool[Any, Any],
    params: SupportsDataclass | None,
    *,
    context: ToolContext,
) -> PolicyDecision:
    for policy in self.policies_for(tool.name):
        decision = policy.check(tool, params, context=context)
        if not decision.allowed:
            return decision
    return PolicyDecision.allow()
```

### Combining Policies

```python
# Multiple policies on overlapping tool sets
policies.register(ReadBeforeWritePolicy())
policies.register(SequentialDependencyPolicy(
    dependencies={"deploy": frozenset({"test", "build"})}
))
policies.register(KeyedDependencyPolicy(
    dependencies={
        "commit": (frozenset({"lint", "test"}), lambda p: getattr(p, "repo", None)),
    }
))

# All applicable policies must allow for execution to proceed
# Evaluation order follows registration order
```

### Conditional Policies

Policies can be conditionally active based on session state:

```python
@dataclass
class ConditionalPolicy:
    """Wrapper that activates a policy based on session state."""

    inner: ToolPolicy
    predicate: Callable[[SessionProtocol], bool]

    @property
    def name(self) -> str:
        return f"conditional:{self.inner.name}"

    @property
    def tools(self) -> frozenset[str]:
        return self.inner.tools

    def check(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        *,
        context: ToolContext,
    ) -> PolicyDecision:
        if not self.predicate(context.session):
            return PolicyDecision.allow()  # Policy inactive
        return self.inner.check(tool, params, context=context)

    def on_result(
        self,
        tool: Tool[Any, Any],
        params: SupportsDataclass | None,
        result: ToolResult[Any],
        *,
        context: ToolContext,
    ) -> None:
        if self.predicate(context.session):
            self.inner.on_result(tool, params, result, context=context)


# Example: Only enforce read-before-write in "strict" mode
strict_policy = ConditionalPolicy(
    inner=ReadBeforeWritePolicy(),
    predicate=lambda s: s[Config].latest().strict_mode,
)
```

## Telemetry

### PolicyDenied Event

When a policy denies execution, emit a telemetry event:

```python
@FrozenDataclass()
class PolicyDenied:
    """Event emitted when a policy blocks tool execution."""

    policy_name: str
    tool_name: str
    reason: str
    metadata: Mapping[str, Any]
    session_id: UUID | None
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)
```

### Logging

Policy checks are logged at DEBUG level; denials at WARNING:

```python
logger.debug("Policy check", policy=policy.name, tool=tool.name)
logger.warning(
    "Policy denied tool execution",
    policy=policy.name,
    tool=tool.name,
    reason=decision.reason,
)
```

## Design Considerations

### Why Session-Scoped State?

Policies track state across tool invocations within a single agent run. This
aligns with:

- **Session semantics**: One session = one coherent task execution
- **Snapshot/restore**: Policy state can be persisted alongside session state
- **Isolation**: Different sessions have independent policy state

### Why Not Decorators?

While decorators could wrap individual handlers, the policy approach provides:

- **Cross-tool coordination**: Policies can track relationships between tools
- **Centralized management**: All policies visible in one registry
- **Dynamic activation**: Policies can be enabled/disabled at runtime
- **Composability**: Multiple policies can govern the same tool

### Thread Safety

Policy state is mutated only from the tool execution thread. For concurrent
tool execution (not currently supported), policies would need internal locking
or immutable state patterns.

## Limitations

- **Synchronous only**: Policies run on the tool execution thread
- **No async support**: `check()` and `on_result()` are synchronous
- **Session-scoped**: No cross-session policy state (use external storage)
- **No rollback hooks**: Policies are not notified when session state rolls back
- **Manual registration**: Policies must be explicitly registered per session

## Future Extensions

### Policy DSL

A declarative DSL for common dependency patterns:

```python
policy = Policy.define(
    name="safe_deployment",
    rules=[
        Rule.require_before("write_file", reads="read_file", key="path"),
        Rule.require_before("deploy", requires=["test", "build"]),
        Rule.require_before("merge", requires="review", key="pr_id"),
    ],
)
```

### Session-Aware Policies

Policies that query session slices for stateful decisions:

```python
@dataclass
class ApprovalRequiredPolicy:
    """Require approval event in session before destructive tools."""

    def check(self, tool, params, *, context):
        approval = context.session[Approval].latest()
        if approval is None or not approval.granted:
            return PolicyDecision.deny("Approval required")
        return PolicyDecision.allow()
```

### Policy Middleware

Stack-based policy composition with middleware semantics:

```python
@dataclass
class LoggingMiddleware:
    """Wrap policies with logging."""

    def wrap(self, policy: ToolPolicy) -> ToolPolicy:
        # Return wrapped policy with before/after logging
        ...
```
