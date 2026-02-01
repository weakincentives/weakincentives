# MCP Server Registration Design

> **Status**: Draft design document for discussion

## Problem Statement

The Claude Agent SDK adapter currently registers a single in-process MCP server
("wink") containing bridged weakincentives tools. Users may want to:

1. Register additional MCP servers (filesystem, git, databases, etc.)
2. Compose external MCP tools with weakincentives tools and policies
3. Maintain observability over all tool invocations

This document explores design options for MCP server registration.

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Agent SDK                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    mcp_servers                             │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  "wink" (in-process)                                │  │  │
│  │  │  - BridgedTool(read_file) ─► weakincentives Tool    │  │  │
│  │  │  - BridgedTool(write_file) ─► weakincentives Tool   │  │  │
│  │  │  - BridgedTool(search) ─► weakincentives Tool       │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

Key characteristics of bridged tools:
- **Transactional**: Snapshot/restore on failure
- **Policy-governed**: Sequential dependencies enforced
- **Observable**: `ToolInvoked` events dispatched
- **Resource-aware**: Access session and prompt resources

## Design Question

Should MCP servers become a core library concept (as "bundles of tools"), or
remain a Claude Agent SDK adapter concern?

## Option A: Adapter-Level Configuration

Keep MCP as an adapter-specific concept. Add configuration to register
additional MCP servers alongside the "wink" server.

### Configuration

```python
@dataclass(slots=True, frozen=True)
class MCPServerConfig:
    """Configuration for an external MCP server."""
    name: str  # Server key in mcp_servers dict
    command: str  # e.g., "npx", "uvx", path to binary
    args: tuple[str, ...] = ()  # e.g., ("-y", "@anthropic/mcp-server-git")
    env: Mapping[str, str] | None = None
    cwd: str | None = None


@dataclass(slots=True, frozen=True)
class ClaudeAgentSDKClientConfig:
    # ... existing fields ...
    mcp_servers: tuple[MCPServerConfig, ...] = ()  # NEW
```

### Usage

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        mcp_servers=(
            MCPServerConfig(
                name="git",
                command="npx",
                args=("-y", "@anthropic/mcp-server-git"),
            ),
            MCPServerConfig(
                name="filesystem",
                command="npx",
                args=("-y", "@anthropic/mcp-server-filesystem", "/path/to/allowed"),
            ),
        ),
    ),
)
```

### Architecture After

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Agent SDK                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    mcp_servers                             │  │
│  │  ┌──────────────────────┐  ┌──────────────────────┐       │  │
│  │  │  "wink" (in-process) │  │  "git" (stdio)       │       │  │
│  │  │  - bridged tools     │  │  - git_status        │       │  │
│  │  │  - policies apply    │  │  - git_diff          │       │  │
│  │  │  - events dispatched │  │  - git_commit        │       │  │
│  │  └──────────────────────┘  └──────────────────────┘       │  │
│  │  ┌──────────────────────┐                                  │  │
│  │  │  "filesystem" (stdio)│                                  │  │
│  │  │  - read_file         │                                  │  │
│  │  │  - write_file        │                                  │  │
│  │  └──────────────────────┘                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Pros
- Simple configuration model
- Maintains provider-agnostic core
- Direct mapping to SDK capabilities
- External servers run as-is (no wrapping needed)

### Cons
- External MCP tools bypass weakincentives policies
- No `ToolInvoked` events for external tools (reduced observability)
- No transactional semantics for external tools
- Name collisions possible between servers

---

## Option B: External Tool Bundles as Core Concept

Introduce a provider-agnostic "external tool bundle" abstraction in core.
MCP servers would be one implementation (for Claude Agent SDK).

### Core Abstraction

```python
# In weakincentives.prompt.external

@dataclass(slots=True, frozen=True)
class ExternalToolBundle:
    """A bundle of tools provided by an external system."""
    name: str
    tool_specs: tuple[ExternalToolSpec, ...]  # Read-only specs

@dataclass(slots=True, frozen=True)
class ExternalToolSpec:
    """Specification for an external tool (no handler)."""
    name: str
    description: str
    input_schema: dict[str, Any]


class ExternalToolProvider(Protocol):
    """Provider-specific external tool integration."""

    def get_bundles(self) -> tuple[ExternalToolBundle, ...]:
        """Return available tool bundles."""
        ...

    def to_provider_config(self, bundle: ExternalToolBundle) -> object:
        """Convert bundle to provider-specific configuration."""
        ...
```

### MCP Implementation

```python
# In weakincentives.adapters.claude_agent_sdk

@dataclass(slots=True, frozen=True)
class MCPToolBundle(ExternalToolBundle):
    """MCP server as an external tool bundle."""
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None

    @classmethod
    def from_npm(cls, name: str, package: str, args: tuple[str, ...] = ()) -> Self:
        """Create from an npm package."""
        return cls(
            name=name,
            tool_specs=(),  # Discovered at runtime
            command="npx",
            args=("-y", package, *args),
        )
```

### Pros
- Unified abstraction across providers
- Could support other external tool systems in future
- Clean separation of concerns

### Cons
- Over-engineering if MCP is the only external tool system
- Tool specs often unknown until runtime (MCP discovery)
- Still doesn't solve policy/observability gaps

---

## Option C: MCP Proxy for Full Integration

Create a proxy layer that intercepts external MCP tool calls, enabling:
- Policy checks before invocation
- `ToolInvoked` event dispatch
- Optional transactional semantics (where possible)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude Agent SDK                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    mcp_servers                             │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  "wink" (in-process)                                │  │  │
│  │  │  - bridged weakincentives tools                     │  │  │
│  │  │  - proxied external MCP tools ──┐                   │  │  │
│  │  └──────────────────────────────────│──────────────────┘  │  │
│  └─────────────────────────────────────│─────────────────────┘  │
│                                        │                         │
│  ┌─────────────────────────────────────▼─────────────────────┐  │
│  │              MCP Proxy Layer (in-process)                  │  │
│  │  - Policy checks                                          │  │
│  │  - Event dispatch                                          │  │
│  │  - Routes to external MCP servers                          │  │
│  └─────────────────────────────────────┬─────────────────────┘  │
│                                        │                         │
│  ┌─────────────────────────────────────▼─────────────────────┐  │
│  │              External MCP Servers (stdio)                  │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │  │
│  │  │   git    │  │filesystem│  │ database │                 │  │
│  │  └──────────┘  └──────────┘  └──────────┘                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Sketch

```python
@dataclass(slots=True, frozen=True)
class ProxiedMCPServer:
    """External MCP server with proxy integration."""
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None

    # Policy integration
    policies: tuple[ToolPolicy, ...] = ()

    # Telemetry
    dispatch_events: bool = True


class MCPProxy:
    """Proxies calls to external MCP servers with weakincentives integration."""

    def __init__(
        self,
        servers: tuple[ProxiedMCPServer, ...],
        session: SessionProtocol,
    ):
        self._servers = {s.name: s for s in servers}
        self._session = session
        self._clients: dict[str, MCPClient] = {}

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Call external tool with policy checks and event dispatch."""
        server = self._servers[server_name]

        # Policy checks (if any policies configured)
        for policy in server.policies:
            # Would need tool spec for full check
            decision = policy.check_by_name(tool_name, args)
            if not decision.allowed:
                return {"error": decision.reason}

        # Dispatch pre-invocation event
        start = datetime.now(UTC)

        # Call external server
        client = await self._get_or_create_client(server)
        result = await client.call_tool(tool_name, args)

        # Dispatch ToolInvoked event
        if server.dispatch_events:
            self._session.dispatcher.dispatch(
                ToolInvoked(
                    name=f"{server_name}:{tool_name}",
                    params=args,
                    # ... result mapping ...
                )
            )

        return result
```

### Pros
- Full observability over all tool calls
- Policy integration possible
- Single MCP server from SDK perspective (simpler)
- Could add caching, rate limiting, etc.

### Cons
- Significant implementation complexity
- Requires running MCP clients in-process
- May introduce latency
- External server lifecycle management

---

## Option D: Hybrid Approach (Recommended)

Combine Option A's simplicity with optional observability hooks.

### Design

1. **Simple passthrough by default**: External MCP servers registered directly
   with the SDK, no interception.

2. **Optional hook integration**: SDK hooks (`pre_tool_use`, `post_tool_use`)
   already fire for ALL tool calls including external MCP tools. Enhance hook
   handlers to dispatch `ToolInvoked` events for external tools.

3. **Policy-governed tools stay in wink**: If a tool needs policy enforcement,
   wrap it as a weakincentives `Tool` and bridge it via "wink" server.

### Configuration

```python
@dataclass(slots=True, frozen=True)
class MCPServerConfig:
    """Configuration for an external MCP server."""
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None

    # Observability (uses SDK hooks, no interception)
    track_invocations: bool = True  # Dispatch ToolInvoked events


@dataclass(slots=True, frozen=True)
class ClaudeAgentSDKClientConfig:
    # ... existing fields ...
    mcp_servers: tuple[MCPServerConfig, ...] = ()
```

### Hook Enhancement

The SDK already fires `pre_tool_use` and `post_tool_use` hooks for all tools.
Enhance the existing hook handlers to:

1. Detect tools from external MCP servers (not in bridged tools registry)
2. Dispatch `ToolInvoked` events with `adapter="mcp:{server_name}"`
3. Track duration between pre/post hooks

```python
# In _hooks.py, enhance post_tool_use_hook

def create_post_tool_use_hook(...):
    bridged_names = {bt.name for bt in bridged_tools}

    def hook(tool_use: ToolUse, tool_result: ToolResult) -> ToolResult:
        tool_name = tool_use.name

        if tool_name not in bridged_names:
            # External MCP tool - dispatch event for observability
            server_name = _infer_server_from_tool(tool_name)  # SDK provides this
            session.dispatcher.dispatch(
                ToolInvoked(
                    name=tool_name,
                    adapter=f"mcp:{server_name}",
                    params=tool_use.input,
                    result=_convert_mcp_result(tool_result),
                    # ...
                )
            )

        return tool_result  # Unchanged
```

### Pros
- Simple configuration (Option A)
- Observability via existing hooks (no interception overhead)
- No policy gaps for external tools (explicit trade-off)
- Clear separation: policy-governed tools go in wink, external tools run as-is
- Minimal implementation effort

### Cons
- Policies don't apply to external MCP tools
- Must explicitly wrap tools needing policy enforcement

---

## Recommendation

**Option D (Hybrid)** provides the best balance:

1. **Immediate value**: Simple configuration to add MCP servers
2. **Observability**: Hook-based event dispatch for telemetry
3. **Clear boundaries**: Policy-governed tools stay in weakincentives

### Migration Path

Users can start with external MCP servers and selectively wrap tools as
weakincentives `Tool` objects if they need:
- Policy enforcement
- Transactional semantics
- Custom handlers

### Non-Goals (For Now)

- Automatic tool wrapping from MCP specs
- MCP server discovery/introspection
- Cross-server policy enforcement

---

## Should MCP Servers Become a Core Concept?

**Recommendation: No**, at least not yet.

Reasons:
1. **MCP is Claude-specific**: Other providers (OpenAI, LiteLLM) don't use MCP
2. **Core should stay provider-agnostic**: The abstraction adds complexity without
   clear benefit for non-Claude adapters
3. **External tools are inherently different**: They lack handlers, can't
   participate in transactions, and come from untrusted sources

If external tool bundles become common across providers, we can revisit with:
- A minimal `ExternalToolBundle` protocol in core
- Provider-specific implementations (MCP for Claude, plugins for OpenAI, etc.)

---

## Implementation Plan

### Phase 1: Basic MCP Server Registration

1. Add `MCPServerConfig` dataclass to `config.py`
2. Add `mcp_servers` field to `ClaudeAgentSDKClientConfig`
3. Modify `_run_sdk_query()` to register additional MCP servers
4. Test with common MCP servers (git, filesystem)

### Phase 2: Observability via Hooks

1. Enhance `post_tool_use_hook` to detect external MCP tools
2. Dispatch `ToolInvoked` events with `adapter="mcp:{server}"
3. Add `track_invocations` option to `MCPServerConfig`

### Phase 3: Documentation

1. Update CLAUDE_AGENT_SDK.md spec
2. Add examples in guides/
3. Document policy limitations for external tools

---

## Open Questions

1. **Tool name collisions**: What if external MCP server has tool named same as
   a bridged tool? SDK behavior? Our handling?

2. **Server lifecycle**: Should we manage MCP server processes, or let SDK
   handle entirely?

3. **Credential passing**: How do external MCP servers get credentials? Via
   `env` field? Inherited from isolation config?

4. **Skills integration**: Should MCP servers be mountable as "skills" alongside
   the existing skill system?
