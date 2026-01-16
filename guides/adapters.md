# Adapters

*Canonical spec: [specs/ADAPTERS.md](../specs/ADAPTERS.md)*

Adapters bridge a prompt to a provider and enforce consistent semantics:

- Render prompt markdown
- Expose tools to the model
- Execute tool calls synchronously
- Parse structured output when declared

## ProviderAdapter.evaluate

All adapters implement the same interface:

```python
response = adapter.evaluate(
    prompt,
    session=session,
    deadline=...,        # optional
    budget=...,          # optional
    budget_tracker=...,  # optional
    resources=...,       # optional ResourceRegistry
)
```

It returns `PromptResponse[OutputT]`:

- `prompt_name`: string
- `text`: raw assistant text
- `output`: parsed structured output (or `None`)

The adapter handles all the provider-specific details: API formatting, tool
schema translation, response parsing. Your code just calls `evaluate()` and gets
back typed results.

## OpenAIAdapter

**Install:** `pip install "weakincentives[openai]"`

```python
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters import OpenAIClientConfig, OpenAIModelConfig

adapter = OpenAIAdapter(
    model="gpt-4.1-mini",
    client_config=OpenAIClientConfig(),
    model_config=OpenAIModelConfig(max_tokens=800),
)
response = adapter.evaluate(prompt, session=session)
```

**Key configs:**

- `OpenAIClientConfig(api_key=..., base_url=..., timeout=..., max_retries=...)`
- `OpenAIModelConfig(temperature=..., max_tokens=..., top_p=..., ...)`

The adapter uses OpenAI's native JSON schema response format for structured
output. It handles tool calls synchronously, executing each tool and feeding
results back to the model.

## LiteLLMAdapter

**Install:** `pip install "weakincentives[litellm]"`

LiteLLM provides a unified interface to many providers. Use this when you want
to switch between providers without changing code.

```python
from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters import LiteLLMClientConfig, LiteLLMModelConfig

adapter = LiteLLMAdapter(
    model="openai/gpt-4.1-mini",
    completion_config=LiteLLMClientConfig(),
    model_config=LiteLLMModelConfig(max_tokens=800),
)
```

## Claude Agent SDK Adapter

**Install:** `pip install "weakincentives[claude-agent-sdk]"`

The Claude Agent SDK adapter is fundamentally different from OpenAI/LiteLLM.
Instead of WINK executing tools itself, it delegates to Claude Code's native
tool execution. This gives you Claude's native tooling (Read, Write, Bash, Glob,
Grep) with WINK's prompt composition and session management.

### Requirements

- **Python package:** `pip install 'weakincentives[claude-agent-sdk]'`
- **Claude Code CLI:** `npm install -g @anthropic-ai/claude-code`
- **Linux sandboxing:** bubblewrap (`bwrap`) available on PATH

### Basic Usage

```python
from dataclasses import dataclass

from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import InProcessDispatcher, Session


@dataclass(frozen=True)
class Hello:
    message: str


session = Session(dispatcher=InProcessDispatcher())

template = PromptTemplate[Hello](
    ns="demo",
    key="hello",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Say hello. Return JSON with a single field: message.",
        ),
    ],
)

response = ClaudeAgentSDKAdapter().evaluate(Prompt(template), session=session)
print(response.output)  # Hello(message="...")
```

### Client Configuration

`ClaudeAgentSDKClientConfig` controls how the SDK subprocess operates:

```python
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

### Workspace Management

`ClaudeAgentWorkspaceSection` creates an isolated workspace with host files
mounted in:

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentWorkspaceSection,
    HostMount,
)

workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/abs/path/to/repo",
            mount_path="repo",
            exclude_glob=(".git/*", "*.pyc"),
            max_bytes=5_000_000,
        ),
    ),
    allowed_host_roots=("/abs/path/to",),
)
```

The `allowed_host_roots` parameter restricts which host paths can be mounted—a
security boundary that prevents accidental exposure.

### Isolation Configuration

The adapter **always runs in hermetic isolation by default**. This prevents the
SDK from accessing the host's `~/.claude` configuration, credentials, and
session state.

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

isolation = IsolationConfig(
    network_policy=NetworkPolicy.no_network(),  # Block tool network access
    sandbox=SandboxConfig(enabled=True),
    api_key="sk-ant-...",
    include_host_env=False,
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(isolation=isolation),
)
```

**NetworkPolicy** controls which network resources tools can access:

```python
# Block all tool network access
policy = NetworkPolicy.no_network()

# Allow specific domains
policy = NetworkPolicy.with_domains("docs.python.org", "pypi.org")
```

### Tool Bridging via MCP

WINK tools attached to prompt sections are automatically exposed to Claude Code
as MCP tools under the server key `"wink"`. This lets you keep side effects and
validation in Python while Claude uses the tools natively.

```python
@dataclass(frozen=True)
class SearchResult:
    matches: int

    def render(self) -> str:
        return f"Found {self.matches} matches"


def mcp_search(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    return ToolResult(message="ok", value=SearchResult(matches=3))


mcp_search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search the internal index",
    handler=mcp_search,
)

# Attach to a section - Claude Code will see it as an MCP tool
section = MarkdownSection(
    title="Task",
    key="task",
    template="Use the search tool.",
    tools=(mcp_search_tool,),
)
```

### Skill Mounting

Mount skills into the hermetic environment for domain-specific instructions:

```python
from pathlib import Path
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

**What is a skill?** A skill is a directory containing a `SKILL.md` file with
domain-specific instructions for Claude Code. Skills follow the
[Agent Skills specification](https://agentskills.io).

**Skill structure:**

```
skills/code-review/
├── SKILL.md         # Required: instructions with optional YAML frontmatter
├── scripts/         # Optional: helper scripts
└── references/      # Optional: reference documents
```

**SKILL.md format:**

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
```

Skills are validated at mount time. Validation checks:

- Directory exists and contains SKILL.md
- YAML frontmatter is valid (if present)
- Name follows naming rules (lowercase, hyphens, 1-64 chars)
- Total size is under 10 MiB

See [specs/SKILLS.md](../specs/SKILLS.md) for full validation rules.

### Adapter Events

All adapters publish events to the session's dispatcher:

| Event | When | Fields |
| --- | --- | --- |
| `PromptRendered` | After prompt render, before API call | `rendered_prompt`, `adapter` |
| `ToolInvoked` | Each tool call (native + bridged) | `tool_name`, `params`, `result`, `duration_ms` |
| `PromptExecuted` | After evaluation completes | `result`, `usage` (TokenUsage) |

For Claude Agent SDK, native tools (Read, Write, Bash) are tracked via SDK hooks
and also publish `ToolInvoked` events.

## Throttling

Adapters support throttle policies for rate limiting:

```python
from weakincentives.adapters import new_throttle_policy

policy = new_throttle_policy(requests_per_minute=60)
adapter = OpenAIAdapter(model="gpt-4o", throttle_policy=policy)
```

**Full throttle configuration:**

```python
from weakincentives.adapters import ThrottlePolicy

policy = ThrottlePolicy(
    requests_per_minute=60,
    max_attempts=5,           # Total attempts before giving up
    base_delay=1.0,           # Initial backoff delay (seconds)
    max_delay=60.0,           # Maximum backoff delay
    max_total_delay=300.0,    # Total time budget for retries
    jitter=0.1,               # Randomization factor (0-1)
)
```

**How throttling works:**

1. When a rate limit is hit, the adapter backs off exponentially
2. Jitter prevents thundering herd when multiple workers retry
3. If `Retry-After` header is present, it's respected
4. `ThrottleError` is raised if all attempts fail

## Next Steps

- [Orchestration](orchestration.md): Use MainLoop for request handling
- [Evaluation](evaluation.md): Test agents with datasets
- [Claude Agent SDK Spec](../specs/CLAUDE_AGENT_SDK.md): Full configuration
  reference
