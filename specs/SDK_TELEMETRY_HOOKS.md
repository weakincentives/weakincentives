# Claude Agent SDK Telemetry Hooks Plan

> **Status**: Planning
> **Scope**: Hooks for robust telemetry and observability

## Overview

This document maps out Claude Agent SDK hooks that should be implemented and
recorded in the weakincentives Session for comprehensive telemetry and
observability. The goal is to capture all meaningful lifecycle events for
monitoring, debugging, tracing, and cost attribution.

## Current State

### Implemented Hooks

| Hook | Event Published | Purpose |
|------|-----------------|---------|
| `PreToolUse` | (none - control only) | Deadline/budget enforcement |
| `PostToolUse` | `ToolInvoked` | Tool execution recording |
| `UserPromptSubmit` | (none - placeholder) | No-op currently |
| `Stop` | (none - state only) | Records stop reason |

### Existing Events

| Event | When Published | Data Captured |
|-------|----------------|---------------|
| `PromptRendered` | After render, before provider call | Rendered text, tools, inputs |
| `PromptExecuted` | After SDK completion | Output, usage, duration |
| `ToolInvoked` | After each tool execution | Name, params, result, call_id |

## Recommended Hook Additions

### Priority 1: Critical for Observability

#### 1. SessionStart Hook → `SessionStarted` Event

**Why**: Establishes the telemetry context for an entire agent run. Required for
correlating all subsequent events.

```python
@FrozenDataclass()
class SessionStarted:
    """Agent session initialization event."""

    sdk_session_id: str              # SDK's internal session identifier
    prompt_name: str                  # Associated prompt
    adapter: AdapterName
    source: Literal["startup", "resume", "clear", "compact"]
    cwd: str                          # Working directory
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)

    # Telemetry correlation
    parent_session_id: UUID | None = None  # For subagent hierarchy
    trace_id: str | None = None            # OpenTelemetry trace ID
```

**Hook implementation**:

```python
async def session_start_hook(input_data, tool_use_id, sdk_context):
    event = SessionStarted(
        sdk_session_id=input_data.get("session_id", ""),
        prompt_name=hook_context.prompt_name,
        adapter=hook_context.adapter_name,
        source=input_data.get("source", "startup"),
        cwd=input_data.get("cwd", ""),
        created_at=datetime.now(UTC),
    )
    hook_context.session.event_bus.publish(event)
    return {}
```

#### 2. SessionEnd Hook → `SessionEnded` Event

**Why**: Marks session completion with final metrics. Essential for calculating
total duration, final token consumption, and success/failure status.

```python
@FrozenDataclass()
class SessionEnded:
    """Agent session completion event."""

    sdk_session_id: str
    prompt_name: str
    adapter: AdapterName
    duration_ms: int                  # Total wall-clock time
    tool_count: int                   # Total tools invoked
    final_usage: TokenUsage | None    # Cumulative token usage
    stop_reason: str                  # How session ended
    success: bool                     # Did session complete normally?
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)

    # Error tracking
    error_type: str | None = None     # Exception class name if failed
    error_message: str | None = None
```

#### 3. PostToolUseFailure Hook → `ToolFailed` Event

**Why**: Separates tool failures from successes for error rate tracking,
alerting, and debugging. The current `ToolInvoked` event conflates success
and failure.

```python
@FrozenDataclass()
class ToolFailed:
    """Event for tool execution failures."""

    prompt_name: str
    adapter: AdapterName
    name: str                         # Tool name
    params: Any                       # Tool input
    call_id: str | None
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)

    # Error details
    error_type: str                   # Exception class name
    error_message: str                # Error description
    stderr: str                       # Captured stderr
    interrupted: bool                 # Was execution interrupted?

    # Context
    tool_number: int                  # Nth tool in session
    elapsed_ms: int | None = None     # Time spent before failure
```

**Hook implementation**:

```python
async def post_tool_use_failure_hook(input_data, tool_use_id, sdk_context):
    event = ToolFailed(
        prompt_name=hook_context.prompt_name,
        adapter=hook_context.adapter_name,
        name=input_data.get("tool_name", ""),
        params=input_data.get("tool_input", {}),
        call_id=tool_use_id,
        created_at=datetime.now(UTC),
        error_type=input_data.get("error_type", "ToolExecutionError"),
        error_message=input_data.get("error_message", ""),
        stderr=input_data.get("tool_response", {}).get("stderr", ""),
        interrupted=input_data.get("tool_response", {}).get("interrupted", False),
        tool_number=hook_context._tool_count,
    )
    hook_context.session.event_bus.publish(event)
    return {}
```

### Priority 2: Important for Production Systems

#### 4. SubagentStart Hook → `SubagentSpawned` Event

**Why**: Tracks nested agent execution for multi-agent observability. Critical
for understanding parallel execution, resource allocation, and cost attribution
across agent hierarchies.

```python
@FrozenDataclass()
class SubagentSpawned:
    """Event when a subagent (Task tool) is launched."""

    parent_session_id: str            # Parent SDK session
    subagent_session_id: str          # New subagent's session
    subagent_type: str                # Agent type (e.g., "Explore", "Plan")
    prompt_name: str
    adapter: AdapterName
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)

    # Context
    task_description: str | None = None  # Brief description of task
    model: str | None = None             # Model used by subagent
```

#### 5. SubagentStop Hook → `SubagentCompleted` Event

**Why**: Records subagent completion with results summary. Required for
aggregating metrics across agent trees.

```python
@FrozenDataclass()
class SubagentCompleted:
    """Event when a subagent finishes execution."""

    parent_session_id: str
    subagent_session_id: str
    subagent_type: str
    prompt_name: str
    adapter: AdapterName
    duration_ms: int
    tool_count: int
    usage: TokenUsage | None
    success: bool
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)

    # Result summary
    stop_reason: str
    output_summary: str | None = None  # Truncated result for logging
```

#### 6. PreCompact Hook → `ContextCompacting` Event

**Why**: Monitors context window management. Compaction events indicate the
agent is hitting context limits, which affects response quality and cost.

```python
@FrozenDataclass()
class ContextCompacting:
    """Event before context compaction occurs."""

    sdk_session_id: str
    prompt_name: str
    adapter: AdapterName
    trigger_type: Literal["automatic", "manual"]
    current_token_count: int
    estimated_compaction_tokens: int
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)
```

### Priority 3: Enhanced Observability

#### 7. PermissionRequest Hook → `PermissionRequested` Event

**Why**: Tracks permission decisions for audit trails and understanding agent
behavior. Important for security monitoring in non-bypass modes.

```python
@FrozenDataclass()
class PermissionRequested:
    """Event when permission is requested for a tool."""

    sdk_session_id: str
    prompt_name: str
    adapter: AdapterName
    tool_name: str
    tool_input: dict[str, Any]
    permission_suggestion: str        # SDK's suggested decision
    decision: Literal["allow", "deny", "ask"]
    decision_reason: str | None
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)
```

#### 8. Notification Hook → `AgentNotification` Event

**Why**: Captures system messages and notifications from the agent. Useful for
debugging and understanding agent state.

```python
@FrozenDataclass()
class AgentNotification:
    """System notification from the agent."""

    sdk_session_id: str
    prompt_name: str
    adapter: AdapterName
    notification_type: str
    message: str
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)
```

### Priority 4: Enhanced Tool Telemetry

#### 9. Enhance PreToolUse Hook → `ToolStarted` Event

**Why**: Currently PreToolUse only enforces constraints. Publishing an event
enables latency calculation and tracking in-flight tools.

```python
@FrozenDataclass()
class ToolStarted:
    """Event when a tool begins execution."""

    prompt_name: str
    adapter: AdapterName
    name: str
    params: Any
    call_id: str | None
    tool_number: int                  # Nth tool in session
    created_at: datetime
    event_id: UUID = field(default_factory=uuid4)

    # Resource context
    tokens_consumed_so_far: int | None = None
    deadline_remaining_ms: int | None = None
```

Combined with `ToolInvoked` (success) or `ToolFailed` (failure), this enables:
- Per-tool latency calculation
- In-flight tool tracking
- Resource consumption at tool boundaries

## Event Hierarchy for Tracing

```
SessionStarted
├── ToolStarted → ToolInvoked/ToolFailed
├── ToolStarted → ToolInvoked/ToolFailed
├── SubagentSpawned
│   ├── SessionStarted (subagent)
│   ├── ToolStarted → ToolInvoked
│   └── SessionEnded (subagent)
├── SubagentCompleted
├── ContextCompacting
├── ToolStarted → ToolInvoked
└── SessionEnded
```

## Implementation Considerations

### HookContext Enhancements

```python
class HookContext:
    """Enhanced context for hook callbacks."""

    session: SessionProtocol
    adapter_name: str
    prompt_name: str
    deadline: Deadline | None = None
    budget_tracker: BudgetTracker | None = None

    # Existing
    stop_reason: str | None = None
    _tool_count: int = 0

    # New: timing and correlation
    _session_start_time: datetime | None = None
    _sdk_session_id: str | None = None
    _trace_id: str | None = None
    _current_tool_start: datetime | None = None

    # New: subagent tracking
    _subagent_sessions: dict[str, datetime] = field(default_factory=dict)
```

### Structured Logging Integration

Each hook should emit structured logs with consistent fields:

```python
logger.info(
    "claude_agent_sdk.session.started",
    event="session.started",
    context={
        "sdk_session_id": session_id,
        "prompt_name": prompt_name,
        "source": source,
    },
)
```

### OpenTelemetry Compatibility

Events should support OpenTelemetry span attributes:

```python
@FrozenDataclass()
class SessionStarted:
    # ... fields ...

    def to_otel_attributes(self) -> dict[str, str | int | bool]:
        """Convert to OpenTelemetry span attributes."""
        return {
            "agent.session.id": self.sdk_session_id,
            "agent.prompt.name": self.prompt_name,
            "agent.adapter": self.adapter,
            "agent.session.source": self.source,
        }
```

## Metrics Derived from Events

| Metric | Events Used | Description |
|--------|-------------|-------------|
| `agent_session_duration_seconds` | SessionStarted, SessionEnded | Total session wall time |
| `agent_tool_duration_seconds` | ToolStarted, ToolInvoked/ToolFailed | Per-tool latency |
| `agent_tool_error_rate` | ToolInvoked, ToolFailed | Tool failure percentage |
| `agent_tokens_consumed_total` | SessionEnded | Cumulative token usage |
| `agent_subagent_count` | SubagentSpawned | Subagents per session |
| `agent_context_compactions` | ContextCompacting | Compaction frequency |

## Event Storage Recommendations

For production observability, events should be:

1. **Published to EventBus** - Real-time in-process handling
2. **Stored in Session slices** - For query/snapshot capabilities
3. **Exported to telemetry backend** - Via EventBus subscriber that forwards to:
   - OpenTelemetry collector
   - Prometheus metrics
   - Log aggregator (structured JSON)

Example telemetry subscriber:

```python
def create_telemetry_subscriber(
    otel_tracer: Tracer | None = None,
    metrics: PrometheusMetrics | None = None,
) -> Callable[[object], None]:
    """Create an event subscriber that exports to telemetry backends."""

    def handler(event: object) -> None:
        if isinstance(event, SessionStarted):
            if otel_tracer:
                # Start trace span
                ...
            if metrics:
                metrics.sessions_started.inc()
        elif isinstance(event, ToolInvoked):
            if metrics:
                metrics.tool_invocations.labels(tool=event.name).inc()
        # ... etc

    return handler
```

## Implementation Order

1. **Phase 1**: Core lifecycle events
   - `SessionStarted` (SessionStart hook)
   - `SessionEnded` (SessionEnd hook)
   - `ToolFailed` (PostToolUseFailure hook)

2. **Phase 2**: Multi-agent support
   - `SubagentSpawned` (SubagentStart hook)
   - `SubagentCompleted` (SubagentStop hook)

3. **Phase 3**: Enhanced observability
   - `ToolStarted` (enhance PreToolUse hook)
   - `ContextCompacting` (PreCompact hook)

4. **Phase 4**: Audit and debugging
   - `PermissionRequested` (PermissionRequest hook)
   - `AgentNotification` (Notification hook)

## File Changes Required

```
src/weakincentives/
├── adapters/claude_agent_sdk/
│   ├── _hooks.py            # Add new hook creators
│   └── adapter.py           # Wire up new hooks
└── runtime/events/
    └── _types.py            # Add new event dataclasses
```

## Testing Strategy

Each new hook/event requires:

1. **Unit tests** - Hook creation and event publishing
2. **Integration tests** - End-to-end with real SDK (requires CLI)
3. **Telemetry tests** - Verify subscriber receives events correctly

Example test pattern:

```python
def test_session_start_hook_publishes_event(session: Session) -> None:
    """SessionStart hook publishes SessionStarted event."""
    context = HookContext(session=session, adapter_name="test", prompt_name="test")
    hook = create_session_start_hook(context)

    events: list[SessionStarted] = []
    session.event_bus.subscribe(SessionStarted, events.append)

    asyncio.run(hook({"session_id": "abc", "source": "startup"}, None, None))

    assert len(events) == 1
    assert events[0].sdk_session_id == "abc"
    assert events[0].source == "startup"
```

## Summary

| Hook | Event | Priority | Purpose |
|------|-------|----------|---------|
| SessionStart | `SessionStarted` | P1 | Session lifecycle start |
| SessionEnd | `SessionEnded` | P1 | Session lifecycle end |
| PostToolUseFailure | `ToolFailed` | P1 | Error tracking |
| SubagentStart | `SubagentSpawned` | P2 | Multi-agent tracking |
| SubagentStop | `SubagentCompleted` | P2 | Multi-agent tracking |
| PreCompact | `ContextCompacting` | P2 | Context management |
| PreToolUse (enhance) | `ToolStarted` | P3 | Latency measurement |
| PermissionRequest | `PermissionRequested` | P4 | Audit trail |
| Notification | `AgentNotification` | P4 | Debugging |
