# MCP Section Registration Specification

> **Status**: Draft RFC for section-level MCP server registration

## Summary

This specification proposes adding MCP server declarations to sections via an
`mcps` parameter, mirroring how `tools` are currently registered. MCP servers
would follow the same progressive disclosure, enabled predicates, and visibility
rules as tools—aggregated at render time and injected into the Claude Agent SDK
adapter.

## Motivation

The current design (see `MCP_SERVER_REGISTRATION.md`) proposes adapter-level
MCP configuration. While simple, this approach misses key benefits:

1. **Co-location**: MCP servers belong with the sections that use them
2. **Progressive disclosure**: Hide MCP capabilities until relevant
3. **Conditional inclusion**: Enable MCP servers based on section params/state
4. **Composition**: Combine prompt fragments with their MCP dependencies

By treating MCP servers like tools, we get these benefits for free.

## Design Principles

- **Section-first**: MCP servers declared on sections, not adapters
- **Progressive disclosure**: SUMMARY visibility hides MCP servers
- **Enabled predicates**: Section `enabled` controls MCP inclusion
- **Aggregation at render**: `RenderedPrompt.mcps` contains active servers
- **Graceful degradation**: Non-Claude adapters ignore MCP configs
- **No tool wrapping**: MCP tools remain external (not bridged)

## Data Model

### MCPServer

Core dataclass representing an MCP server configuration.

```python
# At src/weakincentives/prompt/mcp.py

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal

@dataclass(slots=True, frozen=True)
class MCPServer:
    """Configuration for an MCP server."""

    name: str
    """Unique identifier for this MCP server.

    Must match pattern: ^[a-z0-9][a-z0-9_-]{0,63}$
    Used as key in SDK's mcp_servers dict.
    """

    transport: Literal["stdio", "sse"]
    """Transport protocol for MCP communication."""

    command: str | None = None
    """Command to start the MCP server (stdio transport).

    Examples: "npx", "uvx", "/usr/local/bin/mcp-server"
    """

    args: tuple[str, ...] = ()
    """Arguments passed to the command.

    Examples: ("-y", "@anthropic/mcp-server-git")
    """

    url: str | None = None
    """Server URL (sse transport only).

    Example: "http://localhost:3000/sse"
    """

    env: Mapping[str, str] | None = None
    """Environment variables for the MCP server process.

    Merged with isolation config env if present.
    """

    cwd: str | None = None
    """Working directory for the MCP server process."""

    description: str | None = None
    """Human-readable description for documentation."""

    def __post_init__(self) -> None:
        """Validate transport-specific fields."""
        if self.transport == "stdio" and self.command is None:
            raise ValueError("stdio transport requires 'command'")
        if self.transport == "sse" and self.url is None:
            raise ValueError("sse transport requires 'url'")
        _validate_mcp_name(self.name)
```

### Name Validation

```python
import re

_MCP_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

def _validate_mcp_name(name: str) -> None:
    """Validate MCP server name matches allowed pattern."""
    if not _MCP_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid MCP server name '{name}'. "
            f"Must match pattern: {_MCP_NAME_PATTERN.pattern}"
        )
```

### Factory Methods

```python
@dataclass(slots=True, frozen=True)
class MCPServer:
    # ... fields ...

    @classmethod
    def stdio(
        cls,
        name: str,
        command: str,
        args: tuple[str, ...] = (),
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        description: str | None = None,
    ) -> MCPServer:
        """Create a stdio-based MCP server configuration."""
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            description=description,
        )

    @classmethod
    def sse(
        cls,
        name: str,
        url: str,
        *,
        description: str | None = None,
    ) -> MCPServer:
        """Create an SSE-based MCP server configuration."""
        return cls(
            name=name,
            transport="sse",
            url=url,
            description=description,
        )

    @classmethod
    def npm(
        cls,
        name: str,
        package: str,
        args: tuple[str, ...] = (),
        *,
        env: Mapping[str, str] | None = None,
        description: str | None = None,
    ) -> MCPServer:
        """Create MCP server from npm package via npx.

        Example:
            MCPServer.npm("git", "@anthropic/mcp-server-git")
            MCPServer.npm("fs", "@anthropic/mcp-server-filesystem", ("/allowed/path",))
        """
        return cls(
            name=name,
            transport="stdio",
            command="npx",
            args=("-y", package, *args),
            env=env,
            description=description or f"MCP server: {package}",
        )

    @classmethod
    def uvx(
        cls,
        name: str,
        package: str,
        args: tuple[str, ...] = (),
        *,
        env: Mapping[str, str] | None = None,
        description: str | None = None,
    ) -> MCPServer:
        """Create MCP server from Python package via uvx.

        Example:
            MCPServer.uvx("sqlite", "mcp-server-sqlite", ("--db", "/path/to/db.sqlite"))
        """
        return cls(
            name=name,
            transport="stdio",
            command="uvx",
            args=(package, *args),
            env=env,
            description=description or f"MCP server: {package}",
        )
```

## Section Integration

### Section Constructor

Add `mcps` parameter to `Section.__init__()`:

```python
# In src/weakincentives/prompt/section.py

class Section(GenericParamsSpecializer[SectionParamsT], ABC):

    def __init__(
        self,
        *,
        title: str,
        key: str,
        default_params: SectionParamsT | None = None,
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: EnabledPredicate | None = None,
        tools: Sequence[object] | None = None,
        mcps: Sequence[MCPServer] | None = None,  # NEW
        policies: Sequence[ToolPolicy] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        # ... existing initialization ...
        self._mcps = self._normalize_mcps(mcps)

    def mcps(self) -> tuple[MCPServer, ...]:
        """Return the MCP servers declared by this section."""
        return self._mcps

    @staticmethod
    def _normalize_mcps(
        mcps: Sequence[MCPServer] | None,
    ) -> tuple[MCPServer, ...]:
        if not mcps:
            return ()

        from .mcp import MCPServer

        normalized: list[MCPServer] = []
        for mcp in mcps:
            if not isinstance(mcp, MCPServer):
                raise TypeError("Section mcps must be MCPServer instances.")
            normalized.append(mcp)
        return tuple(normalized)
```

### MarkdownSection

Propagate to `MarkdownSection.__init__()` and `clone()`:

```python
class MarkdownSection(Section[MarkdownParamsT]):

    def __init__(
        self,
        *,
        title: str,
        template: str,
        key: str,
        default_params: MarkdownParamsT | None = None,
        children: Sequence[Section[SupportsDataclass]] | None = None,
        enabled: Callable[[SupportsDataclass], bool] | None = None,
        tools: Sequence[object] | None = None,
        mcps: Sequence[MCPServer] | None = None,  # NEW
        policies: Sequence[ToolPolicy] | None = None,
        accepts_overrides: bool = True,
        summary: str | None = None,
        visibility: VisibilitySelector = SectionVisibility.FULL,
    ) -> None:
        self.template = template
        super().__init__(
            title=title,
            key=key,
            default_params=default_params,
            children=children,
            enabled=enabled,
            tools=tools,
            mcps=mcps,  # NEW
            policies=policies,
            accepts_overrides=accepts_overrides,
            summary=summary,
            visibility=visibility,
        )

    def clone(self, **kwargs: object) -> Self:
        # ... existing clone logic ...
        clone = cls(
            # ... existing fields ...
            mcps=self.mcps(),  # NEW
        )
        return cast(Self, clone)
```

## Rendering

### RenderedPrompt

Add `mcps` property to `RenderedPrompt`:

```python
# In src/weakincentives/prompt/rendering.py

@FrozenDataclass()
class RenderedPrompt[OutputT_co]:
    """Rendered prompt text paired with structured output metadata."""

    text: str
    structured_output: StructuredOutputConfig[SupportsDataclass] | None = None
    deadline: Deadline | None = None
    descriptor: PromptDescriptor | None = None
    _tools: tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...] = field(
        default_factory=tuple
    )
    _mcps: tuple[MCPServer, ...] = field(default_factory=tuple)  # NEW
    _tool_param_descriptions: Mapping[str, Mapping[str, str]] = field(
        default=_EMPTY_TOOL_PARAM_DESCRIPTIONS
    )

    @property
    def mcps(self) -> tuple[MCPServer, ...]:
        """MCP servers contributed by enabled sections in traversal order."""
        return self._mcps
```

### PromptRenderer

Collect MCP servers alongside tools:

```python
class PromptRenderer[OutputT]:

    def render(
        self,
        param_lookup: Mapping[type[SupportsDataclass], SupportsDataclass],
        # ... other params ...
    ) -> RenderedPrompt[OutputT]:
        collected_tools: list[Tool[...]] = []
        collected_mcps: list[MCPServer] = []  # NEW
        seen_mcp_names: set[str] = set()      # NEW - for deduplication

        for node, section_params in self._iter_enabled_sections(...):
            # ... existing visibility logic ...

            # Don't collect when rendering with SUMMARY visibility
            if effective_visibility != SectionVisibility.SUMMARY:
                self._collect_section_tools(...)
                self._collect_section_mcps(       # NEW
                    node.section,
                    collected_mcps,
                    seen_mcp_names,
                )

            # ...

        return RenderedPrompt[OutputT](
            text=text,
            # ... existing fields ...
            _tools=tuple(collected_tools),
            _mcps=tuple(collected_mcps),  # NEW
        )

    def _collect_section_mcps(
        self,
        section: Section[SupportsDataclass],
        collected_mcps: list[MCPServer],
        seen_names: set[str],
    ) -> None:
        """Collect MCP servers from a section, deduplicating by name."""
        section_mcps = section.mcps()
        if not section_mcps:
            return

        for mcp in section_mcps:
            if mcp.name in seen_names:
                # Duplicate - log warning but don't fail
                logger.warning(
                    "prompt.render.duplicate_mcp",
                    event="prompt.render.duplicate_mcp",
                    context={"mcp_name": mcp.name},
                )
                continue
            seen_names.add(mcp.name)
            collected_mcps.append(mcp)
```

### Deduplication Strategy

When the same MCP server name appears in multiple sections:
- **First wins**: The first occurrence in traversal order is used
- **Warning logged**: Subsequent duplicates trigger a warning
- **No error**: Allows composition where multiple sections may declare the same dependency

Alternative: Raise `PromptValidationError` for duplicates (stricter).

## Adapter Integration

### Claude Agent SDK Adapter

Modify `_run_sdk_query()` to register MCP servers from rendered prompt:

```python
# In src/weakincentives/adapters/claude_agent_sdk/adapter.py

async def _run_sdk_query(
    self,
    rendered: RenderedPromptProtocol[OutputT],
    session: SessionProtocol,
    # ...
) -> str:
    # ... existing bridged tools setup ...

    # Build mcp_servers dict
    mcp_servers: dict[str, object] = {}

    # Add bridged weakincentives tools as "wink" server
    if bridged_tools:
        mcp_server_config = create_mcp_server(bridged_tools)
        mcp_servers["wink"] = mcp_server_config

    # Add external MCP servers from rendered prompt
    for mcp in rendered.mcps:
        if mcp.name in mcp_servers:
            logger.warning(
                "claude_agent_sdk.mcp_name_collision",
                event="mcp_name_collision",
                context={"name": mcp.name, "collides_with": "wink"},
            )
            continue
        mcp_servers[mcp.name] = self._mcp_to_sdk_config(mcp)

    options_kwargs["mcp_servers"] = mcp_servers

    # ...

def _mcp_to_sdk_config(self, mcp: MCPServer) -> object:
    """Convert MCPServer to SDK-specific configuration."""
    from claude_agent_sdk import MCPServerConfig  # or appropriate import

    if mcp.transport == "stdio":
        return MCPServerConfig(
            type="stdio",
            command=mcp.command,
            args=list(mcp.args),
            env=dict(mcp.env) if mcp.env else None,
            cwd=mcp.cwd,
        )
    elif mcp.transport == "sse":
        return MCPServerConfig(
            type="sse",
            url=mcp.url,
        )
    else:
        assert_never(mcp.transport)
```

### Other Adapters

For OpenAI and LiteLLM adapters, MCP servers are ignored:

```python
# In adapters that don't support MCP

def evaluate(self, prompt: PromptProtocol[OutputT], ...) -> PromptResponse[OutputT]:
    rendered = prompt.render(...)

    # MCP servers ignored - these adapters don't support MCP
    if rendered.mcps:
        logger.debug(
            "adapter.mcp_servers_ignored",
            event="mcp_servers_ignored",
            context={
                "adapter": self.__class__.__name__,
                "mcp_count": len(rendered.mcps),
            },
        )

    # ... continue with tools only ...
```

## Usage Examples

### Basic Usage

```python
from weakincentives.prompt import MarkdownSection, MCPServer

git_section = MarkdownSection(
    title="Git Operations",
    key="git",
    template="""
    You have access to git operations via the git MCP server.
    Use these tools to inspect and modify the repository.
    """,
    mcps=[
        MCPServer.npm("git", "@anthropic/mcp-server-git"),
    ],
)
```

### Conditional MCP Servers

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class DatabaseParams:
    db_path: str
    enable_write: bool = False

database_section = MarkdownSection[DatabaseParams](
    title="Database Access",
    key="database",
    template="""
    Database available at: $db_path
    Write access: $enable_write
    """,
    enabled=lambda params: params.db_path is not None,
    mcps=[
        MCPServer.uvx(
            "sqlite",
            "mcp-server-sqlite",
            ("--db", "${db_path}"),  # Placeholder - see note below
        ),
    ],
)
```

**Note**: MCP server args don't support template substitution in this design.
Consider either:
1. Factory functions that construct `MCPServer` with resolved values
2. A `params_factory` callback that receives section params

### Progressive Disclosure

```python
advanced_section = MarkdownSection(
    title="Advanced Tools",
    key="advanced",
    template="""
    Advanced filesystem and database tools are available.
    Use these for complex operations.
    """,
    summary="Advanced tools available. Expand for details.",
    visibility=SectionVisibility.SUMMARY,
    mcps=[
        MCPServer.npm("fs", "@anthropic/mcp-server-filesystem", ("/data",)),
        MCPServer.uvx("sqlite", "mcp-server-sqlite"),
    ],
)
```

When rendered with SUMMARY visibility, MCP servers are NOT registered.
After `open_sections` expands the section, MCP servers become active.

### Composition

```python
from weakincentives.prompt import PromptTemplate

# Reusable section with MCP dependency
def git_section(repo_path: str) -> MarkdownSection:
    return MarkdownSection(
        title="Git",
        key="git",
        template=f"Working with repository at {repo_path}",
        mcps=[MCPServer.npm("git", "@anthropic/mcp-server-git")],
    )

# Compose into different prompts
code_review_prompt = PromptTemplate(
    ns="reviews",
    key="code-review",
    sections=[
        git_section("/path/to/repo"),
        # ... other sections ...
    ],
)

release_prompt = PromptTemplate(
    ns="releases",
    key="prepare-release",
    sections=[
        git_section("/path/to/repo"),
        changelog_section(),  # May also declare MCP dependencies
    ],
)
```

## Observability

### Event Dispatch

The Claude Agent SDK adapter already fires hooks for all tool calls, including
external MCP tools. Enhance hook handlers to dispatch events:

```python
# In _hooks.py

def create_post_tool_use_hook(
    bridged_tools: tuple[BridgedTool, ...],
    session: SessionProtocol,
    ...
) -> PostToolUseHook:
    bridged_names = {bt.name for bt in bridged_tools}

    def hook(tool_use: ToolUse, tool_result: ToolResult) -> ToolResult:
        if tool_use.name not in bridged_names:
            # External MCP tool - dispatch event
            session.dispatcher.dispatch(
                ToolInvoked(
                    name=tool_use.name,
                    adapter="mcp",  # Or infer server name
                    params=tool_use.input,
                    result=_convert_mcp_result(tool_result),
                    ...
                )
            )
        return tool_result

    return hook
```

### Logging

```python
logger.info(
    "claude_agent_sdk.mcp_servers_registered",
    event="mcp_servers_registered",
    context={
        "wink_tool_count": len(bridged_tools),
        "external_mcp_count": len(rendered.mcps),
        "external_mcp_names": [m.name for m in rendered.mcps],
    },
)
```

## Validation

### At Section Creation

- MCP server name matches pattern
- Transport-specific fields present (command for stdio, url for sse)
- No duplicate names within same section

### At Render Time

- Warn on duplicate MCP names across sections (first wins)
- Log when adapter ignores MCP servers

### At Adapter Time

- Warn if MCP name collides with "wink" (reserved for bridged tools)

## Limitations

1. **No parameter substitution**: MCP server args are static, not templated
2. **No policy enforcement**: External MCP tools bypass weakincentives policies
3. **No transactional semantics**: External tools don't participate in snapshots
4. **Claude-only**: MCP servers ignored by OpenAI/LiteLLM adapters
5. **No tool discovery**: MCP tool schemas not available at render time

## Future Considerations

### Dynamic MCP Args

Support parameter substitution in MCP server configuration:

```python
# Hypothetical future API
mcps=[
    MCPServer.stdio(
        "sqlite",
        command="uvx",
        args_factory=lambda params: ("mcp-server-sqlite", "--db", params.db_path),
    ),
]
```

### MCP Tool Introspection

Query MCP servers at render time to discover available tools:

```python
# Hypothetical future API
rendered = prompt.render(params, discover_mcp_tools=True)
# rendered.mcp_tools contains tool specs from all MCP servers
```

### Policy Integration

Wrap MCP tools for policy enforcement (significant complexity):

```python
# Hypothetical future API
mcps=[
    MCPServer.npm("git", "@anthropic/mcp-server-git").with_policies(
        SequentialDependencyPolicy(dependencies={"git_commit": {"git_diff"}}),
    ),
]
```

## Migration

Existing adapter-level MCP configuration (if implemented) can coexist:

1. Adapter-level configs registered first
2. Section-level configs added during render
3. Name collisions logged as warnings

This allows gradual migration from adapter-level to section-level declaration.

## Related Specifications

- `MCP_SERVER_REGISTRATION.md` - Adapter-level MCP configuration (alternative)
- `PROMPTS.md` - Prompt system and sections
- `TOOLS.md` - Tool registration pattern
- `CLAUDE_AGENT_SDK.md` - Claude Agent SDK adapter

## Open Questions

1. **Reserved names**: Should "wink" be the only reserved name, or should we
   reserve a pattern (e.g., `wink-*`) for future internal use?

2. **Parameter substitution**: Should we support `$field` placeholders in MCP
   args, or require factory functions for dynamic configuration?

3. **Strict vs. lenient duplicates**: Warn-and-skip (proposed) vs. error on
   duplicate MCP names?

4. **Isolation inheritance**: Should MCP servers inherit `env` from
   `IsolationConfig`, or require explicit configuration?

5. **Lifecycle management**: Should the adapter manage MCP server process
   lifecycle, or delegate entirely to the SDK?
