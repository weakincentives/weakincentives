# OpenCode Adapter Overview

## Purpose

This document provides an overview of OpenCode integration options for WINK
and guides the choice between two implementation approaches. For detailed
specifications, see the linked documents.

**Implementation:** `src/weakincentives/adapters/opencode/`

## Background

OpenCode is an open-source coding agent that provides two integration surfaces:

1. **HTTP API** - REST endpoints + SSE event streaming via `opencode serve`
2. **ACP (Agent Client Protocol)** - JSON-RPC over stdio via `opencode acp`

Both approaches enable WINK to "rent" OpenCode's execution harness while
maintaining control over orchestration, tools, and state management.

## Integration Approaches

### Approach 1: HTTP API

Spawn `opencode serve` as a subprocess, communicate via HTTP REST and SSE.

**Spec:** `specs/OPENCODE_HTTP_API.md`

```
┌────────────────────┐         ┌────────────────────┐
│ WINK Adapter       │──HTTP──▶│ OpenCode Server    │
│ ├─ ServerManager   │         │ ├─ Sessions        │
│ ├─ SessionClient   │◀──SSE───│ ├─ Tools           │
│ └─ MCP Server      │         │ └─ MCP Integration │
└────────────────────┘         └────────────────────┘
```

### Approach 2: ACP (Recommended)

Spawn `opencode acp` as a subprocess, communicate via JSON-RPC over stdio.

**Spec:** `specs/OPENCODE_ACP.md`

```
┌────────────────────┐         ┌────────────────────┐
│ WINK Adapter       │         │ OpenCode Process   │
│ ├─ ProcessManager  │──stdio──│ ├─ ACP Handler     │
│ └─ AcpClient       │◀─JSON───│ └─ Session/Tools   │
└────────────────────┘  RPC    └────────────────────┘
```

## Approach Comparison

### Architectural Alignment

| Aspect | HTTP API | ACP | Claude Agent SDK |
|--------|----------|-----|------------------|
| Process Model | HTTP server | Subprocess | Subprocess |
| Communication | HTTP + SSE | JSON-RPC stdio | SDK methods |
| Event Delivery | SSE subscription | Notifications | Hook callbacks |
| Tool Bridging | External MCP server | Native capability | SDK MCP helpers |

**Verdict:** ACP's subprocess model closely mirrors Claude Agent SDK.

### Implementation Effort

| Component | HTTP API | ACP |
|-----------|----------|-----|
| Process/Server Management | ~200 LOC | ~100 LOC |
| Client (HTTP/JSON-RPC) | ~400 LOC | ~300 LOC |
| Event Handling | ~250 LOC (SSE) | ~200 LOC |
| Tool Bridging | ~400 LOC (MCP) | ~150 LOC |
| Main Adapter | ~500 LOC | ~400 LOC |
| Tests | ~800 LOC | ~400 LOC |
| **Total** | **~2550 LOC** | **~1550 LOC** |

**Verdict:** ACP requires ~40% less code.

### Dependencies

| Aspect | HTTP API | ACP |
|--------|----------|-----|
| External | httpx, sse library | None (stdlib only) |
| OpenCode SDK | Required | Optional |
| MCP Server | Must implement | Not needed |

**Verdict:** ACP has no external dependencies.

### Risk Assessment

| Risk | HTTP API | ACP |
|------|----------|-----|
| Network reliability | High (SSE drops) | Low (stdio) |
| Server management | High (zombies, crashes) | Low |
| Protocol complexity | Medium | Low |
| OpenCode support | Mature | Newer |

**Verdict:** ACP has lower implementation risk.

### Feature Parity

| Feature | HTTP API | ACP |
|---------|----------|-----|
| Tool execution | Yes (MCP) | Yes (native) |
| Streaming updates | Yes (SSE) | Yes (notifications) |
| Structured output | Tool-based | Tool-based |
| Permissions | API responses | TBD |
| Multi-session | Yes | One per process |

**Verdict:** Feature parity expected; ACP permissions need verification.

## Recommendation

**ACP is the recommended approach** for the following reasons:

1. **Architectural Match**: Process model mirrors Claude Agent SDK
2. **Implementation Simplicity**: ~40% less code to write and maintain
3. **Fewer Dependencies**: Standard library only
4. **Lower Risk**: Stdio communication is simpler than HTTP+SSE
5. **Native Tool Support**: No external MCP server required

### When to Choose HTTP API

Consider HTTP API if:
- Multi-session server pooling is required
- Web/IDE client integration is needed
- OpenCode's ACP implementation lacks required features

### Implementation Priority

1. **Phase 1**: Implement `OpenCodeAcpAdapter` (recommended)
2. **Phase 2**: Implement `OpenCodeHttpAdapter` if multi-session needed

## Common Configuration

Both approaches share configuration patterns:

### OpenCodeModelConfig

```python
@FrozenDataclass()
class OpenCodeModelConfig:
    model: str = "anthropic/claude-sonnet-4-20250514"
    provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
```

### OpenCodeIsolationConfig

```python
@FrozenDataclass()
class OpenCodeIsolationConfig:
    config_content: str | None = None  # Inline JSON config
    config_dir: Path | None = None
    env: Mapping[str, str] | None = None
    include_host_env: bool = False
    cache_dir: Path | None = None
```

### Workspace Isolation

Both approaches support hermetic workspace isolation:

1. Create temp directory with mounted files
2. Inject config via `OPENCODE_CONFIG_CONTENT`
3. Isolate cache via `OPENCODE_CACHE_DIR`
4. Clean up on completion

## Shared Patterns

### Tool Bridging

Both approaches reuse WINK's transactional tool execution:

```python
with tool_transaction(session, resources, tag=f"tool:{name}"):
    result = tool.handler(params, context=context)
```

### State Synchronization

Both map OpenCode events to WINK patterns:
- Tool start → Create snapshot
- Tool success → Dispatch `ToolInvoked`, clear snapshot
- Tool failure → Restore snapshot, dispatch error event

### Structured Output

Both use tool-based output contract:

```python
@FrozenDataclass()
class WinkEmitParams:
    output: dict[str, Any]

# Tool validates against output schema and stores in session
```

## Requirements

- Python: `pip install 'weakincentives[opencode]'`
- OpenCode CLI: `npm install -g opencode` or `bun add -g opencode`

## Related Specifications

| Spec | Description |
|------|-------------|
| `OPENCODE_ACP.md` | ACP adapter (recommended) |
| `OPENCODE_HTTP_API.md` | HTTP API adapter |
| `ADAPTERS.md` | Provider adapter protocol |
| `CLAUDE_AGENT_SDK.md` | Reference implementation |
| `TOOLS.md` | Tool bridging patterns |

## External References

- [OpenCode Documentation](https://opencode.ai/docs/)
- [Agent Client Protocol](https://agentclientprotocol.com/)
- [OpenCode GitHub](https://github.com/anomalyco/opencode)
