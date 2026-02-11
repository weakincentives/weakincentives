# Claude Agent SDK Integration

*Canonical spec: [specs/CLAUDE_AGENT_SDK.md](../specs/CLAUDE_AGENT_SDK.md)*

The Claude Agent SDK adapter is WINK's primary integration for production
applications. Instead of WINK executing tools itself, the adapter delegates to
Claude Code's native tool execution—giving you Claude's battle-tested tooling
(Read, Write, Bash, Glob, Grep) with WINK's prompt composition, session
management, and orchestration.

## Why Claude Agent SDK for Production

**Native tooling quality**: Claude Code's tools are optimized for real-world
software engineering. File operations handle edge cases (encoding, permissions,
large files) that custom implementations often miss.

**Sandboxing included**: The adapter runs in hermetic isolation by default,
preventing access to host configuration, credentials, and session state. This
is critical for multi-tenant deployments.

**MCP bridging**: Your WINK tools are automatically exposed to Claude Code as
MCP tools. You keep side effects and validation in Python while Claude uses
tools natively.

**Workspace management**: `WorkspaceSection` provides structured
access to host files with security boundaries, size limits, and exclude
patterns.

## Requirements

- **Python package**: `pip install 'weakincentives[claude-agent-sdk]'`
- **Claude Code CLI**: `npm install -g @anthropic-ai/claude-code`
- **Linux sandboxing**: bubblewrap (`bwrap`) available on PATH

The adapter requires claude-agent-sdk version 0.1.15 or later.

## WorkspaceSection

`WorkspaceSection` is the core abstraction for giving Claude Code
access to host files. It creates an isolated workspace directory with mounted
content and enforces security boundaries.

```python nocheck
from weakincentives.prompt import WorkspaceSection, HostMount

workspace = WorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/abs/path/to/repo",
            mount_path="repo",
            exclude_glob=(".git/*", "*.pyc", "__pycache__/*"),
            max_bytes=5_000_000,
        ),
    ),
    allowed_host_roots=("/abs/path/to",),
)
```

### HostMount Configuration

| Field | Type | Description |
|-------|------|-------------|
| `host_path` | `str` | Absolute path to host directory |
| `mount_path` | `str` | Relative path in workspace |
| `include_glob` | `tuple[str, ...]` | Patterns to include (default: all) |
| `exclude_glob` | `tuple[str, ...]` | Patterns to exclude |
| `max_bytes` | `int` | Maximum total size to mount |

### Security: allowed_host_roots

The `allowed_host_roots` parameter is a critical security boundary. It
restricts which host paths can be mounted:

```python nocheck
# Only allows mounting from /home/app/repos
workspace = WorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="/home/app/repos/myproject", mount_path="code"),),
    allowed_host_roots=("/home/app/repos",),
)

# This would fail validation:
# HostMount(host_path="/etc/passwd", mount_path="secrets")
```

Without `allowed_host_roots`, any path could be mounted—dangerous in
multi-tenant environments where mount paths might come from user input.

### Workspace Lifecycle

The workspace section manages a temporary directory:

```python nocheck
with workspace:
    # workspace.temp_dir exists and contains mounted files
    adapter = ClaudeAgentSDKAdapter(
        client_config=ClaudeAgentSDKClientConfig(cwd=str(workspace.temp_dir)),
    )
    response = adapter.evaluate(prompt, session=session)
# temp_dir cleaned up automatically
```

The section implements the context manager protocol. When used with
`Prompt.resources`, cleanup is automatic.

## Adapter Configuration

### ClaudeAgentSDKClientConfig

Controls how the SDK subprocess operates:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)

config = ClaudeAgentSDKClientConfig(
    permission_mode="bypassPermissions",
    cwd="/path/to/workspace",
    max_turns=10,
    max_budget_usd=1.0,
    suppress_stderr=True,
    stop_on_structured_output=True,
)

adapter = ClaudeAgentSDKAdapter(client_config=config)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `permission_mode` | `PermissionMode` | `"bypassPermissions"` | Permission handling |
| `cwd` | `str \| None` | `None` | Working directory |
| `max_turns` | `int \| None` | `None` | Maximum conversation turns |
| `max_budget_usd` | `float \| None` | `None` | Maximum budget in USD |
| `suppress_stderr` | `bool` | `True` | Suppress stderr output |
| `stop_on_structured_output` | `bool` | `True` | Stop after structured output |
| `task_completion_checker` | `TaskCompletionChecker \| None` | `None` | Task completion verification |
| `isolation` | `IsolationConfig \| None` | `None` | Isolation configuration |
| `betas` | `tuple[str, ...] \| None` | `None` | Beta features to enable |

### ClaudeAgentSDKModelConfig

Controls model selection and parameters:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKModelConfig

model_config = ClaudeAgentSDKModelConfig(
    model="claude-opus-4-6",
    reasoning="max",  # Unconstrained adaptive reasoning (Opus 4.6 only)
)

adapter = ClaudeAgentSDKAdapter(model_config=model_config)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | `"claude-opus-4-6"` | Claude model identifier |
| `reasoning` | `ReasoningEffort \| None` | `"high"` | Adaptive reasoning effort level (`"low"`, `"medium"`, `"high"`, `"max"`, or `None`) |

**Note**: Parameters like `seed`, `stop`, `presence_penalty`, `frequency_penalty`
are not supported and raise `ValueError` if set.

## Isolation and Security

The adapter **always runs in hermetic isolation by default**. This prevents the
SDK from accessing the host's `~/.claude` configuration, credentials, and
session state.

### IsolationConfig

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

isolation = IsolationConfig(
    network_policy=NetworkPolicy.no_network(),
    sandbox=SandboxConfig(enabled=True),
    api_key="sk-ant-...",
    include_host_env=False,
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(isolation=isolation),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `network_policy` | `NetworkPolicy \| None` | `None` | Tool network restrictions |
| `sandbox` | `SandboxConfig \| None` | `None` | Sandbox configuration |
| `env` | `Mapping[str, str] \| None` | `None` | Environment variables |
| `api_key` | `str \| None` | `None` | Explicit API key (disables Bedrock) |
| `aws_config_path` | `Path \| str \| None` | `None` | AWS config path for Docker |
| `include_host_env` | `bool` | `False` | Copy non-sensitive vars |
| `skills` | `SkillConfig \| None` | `None` | Skills to mount |

**Authentication modes:**

1. **Inherit host auth** (default): When `api_key` is `None`, inherits auth from
   the host environment. Works with both Anthropic API (`ANTHROPIC_API_KEY`) and
   AWS Bedrock (`CLAUDE_CODE_USE_BEDROCK=1` + AWS credentials).

1. **Explicit API key**: When `api_key` is set, uses that key with Anthropic API
   and disables Bedrock.

**Factory methods** (recommended for fail-fast validation):

```python
# Inherit host auth - fails fast if no auth configured
isolation = IsolationConfig.inherit_host_auth()

# Explicit API key - fails fast if key is empty
isolation = IsolationConfig.with_api_key("sk-ant-...")

# Require Bedrock - fails fast if not configured
isolation = IsolationConfig.for_bedrock(aws_config_path="/mnt/aws")
```

**Docker usage with Bedrock:**

When running in Docker, mount your AWS config directory and specify the path:

```bash
# Run container with AWS config mounted
docker run -v ~/.aws:/mnt/aws:ro \
    -e CLAUDE_CODE_USE_BEDROCK=1 \
    -e AWS_REGION=us-east-1 \
    my-agent-image
```

```python
# In your agent code
isolation = IsolationConfig.for_bedrock(aws_config_path="/mnt/aws")
```

### NetworkPolicy

Controls which network resources tools can access. This restricts tools only,
not the API connection.

```python nocheck
# Block all tool network access (recommended for code review)
policy = NetworkPolicy.no_network()

# Allow specific domains for documentation lookup
policy = NetworkPolicy.with_domains("docs.python.org", "pypi.org")
```

For production code review, `no_network()` is strongly recommended—it prevents
the model from exfiltrating code or fetching malicious payloads.

### SandboxConfig

Fine-grained control over the sandbox:

```python nocheck
sandbox = SandboxConfig(
    enabled=True,
    writable_paths=("/tmp/workspace",),
    readable_paths=("/opt/references",),
    excluded_commands=("curl", "wget"),
    allow_unsandboxed_commands=False,
    bash_auto_allow=True,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable sandboxing |
| `writable_paths` | `tuple[str, ...]` | `()` | Paths where writing is allowed |
| `readable_paths` | `tuple[str, ...]` | `()` | Additional readable paths |
| `excluded_commands` | `tuple[str, ...]` | `()` | Commands to block |
| `allow_unsandboxed_commands` | `bool` | `False` | Allow unsandboxed execution |
| `bash_auto_allow` | `bool` | `True` | Auto-allow bash commands |

### How Isolation Works

When `isolation` is set:

1. Creates ephemeral `HOME` directory with `$HOME/.claude/settings.json`
1. Passes `env` and `setting_sources=["user"]` to SDK
1. Copies `~/.aws` to ephemeral home for Bedrock credential access
1. Host environment variables are excluded unless `include_host_env=True`

**Auth inheritance** (when no explicit `api_key`):

Auth vars are read from two sources (in priority order):

1. Shell environment variables (highest priority)
1. Host `~/.claude/settings.json` env section (fallback)

This ensures that if `claude` works on the host, WINK agents will too.

Vars passed through for Bedrock: `AWS_PROFILE`, `AWS_REGION`, `AWS_DEFAULT_REGION`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_ROLE_ARN`,
`AWS_WEB_IDENTITY_TOKEN_FILE`, `CLAUDE_CODE_USE_BEDROCK`.

### Model ID Helpers

Model IDs differ between Anthropic API and Bedrock. Use these helpers for
unified model selection:

```python
from weakincentives.adapters.claude_agent_sdk import (
    get_default_model,      # Returns Opus 4.6 in correct format
    to_bedrock_model_id,    # "claude-opus-4-5-20251101" -> "us.anthropic.claude-opus-4-5-20251101-v1:0"
    to_anthropic_model_name, # Reverse conversion
)

# Automatic: returns correct format based on CLAUDE_CODE_USE_BEDROCK
model = get_default_model()

adapter = ClaudeAgentSDKAdapter(model=model)
```

## Tool Bridging via MCP

WINK tools attached to prompt sections are automatically exposed to Claude Code
as MCP tools under the server key `"wink"`. This lets you keep side effects and
validation in Python while Claude uses the tools natively.

```python nocheck
from dataclasses import dataclass
from weakincentives.prompt import MarkdownSection, Tool, ToolContext, ToolResult

@dataclass(frozen=True)
class SearchParams:
    query: str

@dataclass(frozen=True)
class SearchResult:
    snippets: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(f"- {s}" for s in self.snippets)


def search_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[SearchResult]:
    # Your search logic here
    results = perform_search(params.query)
    return ToolResult.ok(SearchResult(snippets=tuple(results)))


search_tool = Tool[SearchParams, SearchResult](
    name="search_index",
    description="Search the internal documentation index",
    handler=search_handler,
)

# Attach to a section - Claude Code sees it as an MCP tool
section = MarkdownSection(
    title="Task",
    key="task",
    template="Use the search_index tool to find relevant documentation.",
    tools=(search_tool,),
)
```

When Claude Code calls the `search_index` tool, WINK:

1. Receives the call via MCP
1. Deserializes parameters to `SearchParams`
1. Calls your handler with the `ToolContext`
1. Serializes the result back to Claude Code
1. Publishes a `ToolInvoked` event

### Tool Result Rendering

The `render()` method on your result dataclass controls what Claude sees:

```python nocheck
@dataclass(frozen=True)
class FileAnalysis:
    issues: list[str]
    suggestions: list[str]

    def render(self) -> str:
        parts = []
        if self.issues:
            parts.append("Issues found:\n" + "\n".join(f"- {i}" for i in self.issues))
        if self.suggestions:
            parts.append("Suggestions:\n" + "\n".join(f"- {s}" for s in self.suggestions))
        return "\n\n".join(parts) if parts else "No issues found."
```

## Skill Mounting

Skills are domain-specific instructions that enhance Claude Code's behavior.
They follow the [Agent Skills specification](https://agentskills.io).

```python nocheck
from pathlib import Path
from weakincentives.adapters.claude_agent_sdk import IsolationConfig
from weakincentives.skills import SkillConfig, SkillMount

isolation = IsolationConfig(
    skills=SkillConfig(
        skills=(
            SkillMount(source=Path("skills/code-review.md")),
            SkillMount(source=Path("skills/testing"), enabled=False),
        ),
    ),
)
```

### Skill Structure

A skill is a directory containing a `SKILL.md` file:

```
skills/code-review/
├── SKILL.md         # Required: instructions with optional YAML frontmatter
├── scripts/         # Optional: helper scripts
└── references/      # Optional: reference documents
```

### SKILL.md Format

```markdown
---
name: code-review
description: Thorough code review for Python projects
---

# Code Review Skill

When reviewing code, follow these steps:
1. Check for security vulnerabilities
2. Verify error handling
3. Ensure tests cover new functionality
4. Look for performance issues
```

### Validation

Skills are validated at mount time:

- Directory exists and contains SKILL.md
- YAML frontmatter is valid (if present)
- Name follows naming rules (lowercase, hyphens, 1-64 chars)
- Total size is under 10 MiB

See [specs/SKILLS.md](../specs/SKILLS.md) for complete validation rules.

## Events and Observability

All adapter operations publish events to the session's dispatcher:

| Event | When | Key Fields |
|-------|------|------------|
| `PromptRendered` | After prompt render, before API call | `rendered_prompt`, `adapter` |
| `ToolInvoked` | Each tool call (native + bridged) | `name`, `params`, `result`, `usage` |
| `PromptExecuted` | After evaluation completes | `result`, `usage` (TokenUsage) |

Native Claude Code tools (Read, Write, Bash) are tracked via SDK hooks and also
publish `ToolInvoked` events. This gives you complete visibility into what the
agent is doing.

```python nocheck
from weakincentives.runtime import InProcessDispatcher, ToolInvoked

def log_tool_calls(event: ToolInvoked) -> None:
    print(f"Tool: {event.name}, Params: {event.params}")

dispatcher = InProcessDispatcher()
dispatcher.subscribe(ToolInvoked, log_tool_calls)
```

## Production Patterns

### Secure Code Review

The most common production pattern—reviewing code without network access:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)
from weakincentives.prompt import WorkspaceSection, HostMount

# Create isolated workspace
workspace = WorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/repos/project",
            mount_path="code",
            exclude_glob=(".git/*", "*.pyc", "__pycache__/*", "node_modules/*"),
            max_bytes=10_000_000,
        ),
    ),
    allowed_host_roots=("/repos",),
)

# Configure adapter with full isolation
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        cwd=str(workspace.temp_dir),
        max_turns=20,
        max_budget_usd=2.0,
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.no_network(),
            sandbox=SandboxConfig(
                enabled=True,
                readable_paths=(str(workspace.temp_dir),),
            ),
        ),
    ),
)
```

### Documentation Lookup with Domain Allowlist

When the agent needs to fetch external documentation:

```python nocheck
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.with_domains(
                "docs.python.org",
                "typing.readthedocs.io",
                "peps.python.org",
            ),
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)
```

### Adaptive Reasoning for Complex Analysis

Enable maximum reasoning effort for thorough code analysis:

```python nocheck
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKModelConfig

adapter = ClaudeAgentSDKAdapter(
    model_config=ClaudeAgentSDKModelConfig(
        model="claude-opus-4-6",
        reasoning="max",  # Unconstrained thinking depth (Opus 4.6 only)
    ),
    client_config=ClaudeAgentSDKClientConfig(
        max_turns=30,
        max_budget_usd=5.0,
    ),
)
```

### Budget and Turn Limits

Always set limits in production to prevent runaway costs:

```python nocheck
config = ClaudeAgentSDKClientConfig(
    max_turns=25,           # Prevent infinite loops
    max_budget_usd=3.0,     # Hard cost cap
    stop_on_structured_output=True,  # Stop when task complete
)
```

## Comparison with Codex App Server Adapter

| Feature | Claude Agent SDK | Codex App Server |
|---------|-----------------|------------------|
| Tool execution | Native (Claude Code) | Native (Codex) |
| Sandboxing | Built-in | Built-in |
| File operations | Battle-tested | Built-in |
| MCP bridging | Yes | No |
| Skills | Yes | No |
| Network isolation | Built-in | Built-in |

For production workloads involving code, the Claude Agent SDK adapter provides
the most complete feature set with MCP bridging and skill mounting.

## Next Steps

- [Code Reviewer Example](code-review-agent.md): Complete working example using
  Claude Agent SDK
- [Orchestration](orchestration.md): AgentLoop integration
- [specs/CLAUDE_AGENT_SDK.md](../specs/CLAUDE_AGENT_SDK.md): Complete
  configuration reference
