# Weak Incentives (Is All You Need)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/weakincentives/weakincentives)

WINK is the agent-definition layer for building unattended/background agents.
You define the prompt, tools, policies, and feedback that stay stable while
runtimes change. The planning loop, sandboxing, retries, and orchestration live
in the execution harness—often a vendor runtime. WINK keeps your agent
definition portable across all supported runtimes:

- **Claude Agent SDK** — Claude Code's runtime with native file/shell tools and MCP bridging
- **Codex** — OpenAI's Codex runtime via the App Server protocol (stdio JSON-RPC)
- **OpenCode** — via the vendor-neutral Agent Client Protocol (ACP), which also supports Gemini CLI and future ACP agents

> **New to WINK?** If you're like, "I don't want to read, I just want to build!"—point
> your favorite agent at a clone of https://github.com/weakincentives/starter and prompt
> your way into a great unattended agent for your use case. You can prompt better if you
> read some [guides](guides/README.md).

## Definition vs. Harness

A high-quality unattended agent has two parts:

**Agent definition (you own):**

- Prompt structure (context engineering)
- Tools + typed I/O contracts (the side-effect boundary)
- Policies (gates on tool use and state transitions)
- Feedback ("done" criteria, drift detection)

**Execution harness (runtime-owned):**

- Planning/act loop and tool-call sequencing
- Sandboxing/permissions (filesystem/shell/network)
- Retries/backoff, throttling, and lifecycle management
- Scheduling, budgets/deadlines, crash recovery
- Multi-agent orchestration (when used)

The harness will keep changing—and increasingly comes from vendor runtimes—but
your agent definition should not. WINK makes the definition a first-class
artifact you can version, review, test, and port across runtimes via adapters.

## The Prompt is the Agent

Most agent frameworks treat prompts as an afterthought—templates glued to
separately registered tool lists. WINK inverts this: **the prompt _is_ the
agent**. You define an agent as a single hierarchical document where each
section bundles its own instructions and tools together.

```
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)
├── WorkspaceDigestSection     ← auto-generated codebase summary
├── MarkdownSection (reference docs, progressive disclosure)
├── WorkspaceSection            ← contributes file tools via SDK
│   └── (nested workspace docs)
└── MarkdownSection (user request)
```

Each section can render instructions, contribute tools, nest child sections, and
enable or disable itself based on runtime state. When a section disables, its
entire subtree—tools included—vanishes from the prompt.

The result: **the prompt fully determines what the agent can think and do**.
There's no separate tool registry to synchronize, no routing layer to maintain,
no configuration that can drift from documentation. You define the agent's
capabilities once, in one place, and the definition ports across runtimes.

**Why this matters:**

1. **Co-location.** Instructions and tools live together. The section that
   explains filesystem navigation is the same section that provides the
   `read_file` tool. Documentation can't drift from implementation.

1. **Progressive disclosure.** Nest child sections to reveal advanced
   capabilities only when relevant. The LLM sees numbered, hierarchical headings
   that mirror your code structure.

1. **Dynamic scoping.** Each section has an `enabled` predicate. Disable a
   section and its entire subtree—tools included—disappears from the prompt.
   Swap sections based on runtime conditions; the prompt adapts automatically.

1. **Typed all the way down.** Sections are parameterized with dataclasses.
   Placeholders are validated at construction time. Tools declare typed params
   and results. The framework catches mismatches before the request reaches
   an LLM.

## Key Capabilities

### Prompts

- **Typed sections.** Build prompts from composable `Section` objects that
  bundle instructions and tools together.
- **Hash-based overrides.** Prompt descriptors carry content hashes so overrides
  apply only to the intended version. Teams iterate on prompts via
  version-controlled JSON without risking stale edits.
  See [Prompts](specs/PROMPTS.md).

### Tools

- **Transactional execution.** Tool calls are atomic transactions. When a tool
  fails, WINK automatically rolls back session state and filesystem changes to
  their pre-call state. Failed tools don't leave traces in mutable state.
- **Workspace integration.** Mount host directories into agent workspaces with
  configurable include/exclude patterns. The Claude Agent SDK adapter provides
  sandboxed file tools with automatic cleanup.
  See [Workspace](specs/WORKSPACE.md).

### Policies

- **Invariants over workflows.** Gate tool calls with explicit policies instead
  of brittle orchestration graphs. Encode constraints like "don't write before
  you've read" or "don't call tool B until tool A ran."
  See [Policies Over Workflows](specs/POLICIES_OVER_WORKFLOWS.md).

### Feedback

- **Completion resistance.** Encode "done means X" checks that run during
  execution to catch drift and premature termination.
  See [Guardrails](specs/GUARDRAILS.md) (design spec).

### State and Adapters

- **Event-driven state.** Every state change flows through pure reducers that
  process published events. State is immutable and inspectable—you can snapshot
  at any point. See [Session State](specs/SESSIONS.md).
- **Harness-swappable adapters.** Keep the agent definition stable while
  switching runtimes. WINK integrates with agentic harnesses—Claude Agent SDK,
  Codex App Server, and ACP-compatible agents (e.g., OpenCode)—that provide
  native tools and sandboxing while WINK supplies the definition. See
  [Adapters](specs/ADAPTERS.md).

## Getting Started

**Requirements:** Python 3.12+, [`uv`](https://github.com/astral-sh/uv)

```bash
uv add weakincentives
# optional extras
uv add "weakincentives[claude-agent-sdk]" # Claude Agent SDK adapter
uv add "weakincentives[acp]"              # ACP adapter (OpenCode, etc.)
uv add "weakincentives[wink]"             # debug UI
```

### Debug UI

```bash
uv run --extra wink wink debug snapshots/session.jsonl --port 8000
```

![Debug UI](debug_ui.png)

## Tutorial: Code Review Agent

Build a code review assistant with structured output and observable state.

### 1. Define structured output

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]
```

### 2. Compose the prompt

```python nocheck
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.contrib.tools import WorkspaceDigestSection

template = PromptTemplate[ReviewResponse](
    ns="examples/code-review",
    key="code-review-session",
    name="code_review_agent",
    sections=(
        MarkdownSection(title="Guide", key="guide", template="Review code."),
        WorkspaceDigestSection(session=session),       # cached codebase summary
        MarkdownSection(title="Request", key="request", template="${request}"),
    ),
)

prompt = Prompt(template).bind(ReviewTurnParams(request="Review main.py"))
```

**Note:** Filesystem and planning tools are provided by the execution harness
(e.g., Claude Agent SDK) rather than defined in the prompt. This keeps agent
definitions portable across runtimes.

### 3. Run and get typed results

```python nocheck
from dataclasses import dataclass
from typing import Any
from weakincentives.runtime import AgentLoop, Session
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.prompt import Prompt, PromptTemplate

# Type stubs for example (defined in your application)
@dataclass(frozen=True)
class ReviewTurnParams:
    request: str

@dataclass(frozen=True)
class ReviewResponse:
    summary: str

def build_task_prompt(*, session: Session) -> PromptTemplate[ReviewResponse]:
    ...

class ReviewLoop(AgentLoop[ReviewTurnParams, ReviewResponse]):
    def __init__(self, adapter: Any, dispatcher: Any) -> None:
        super().__init__(adapter=adapter, dispatcher=dispatcher)
        self._session = Session(dispatcher=dispatcher)
        self._template = build_task_prompt(session=self._session)

    def prepare(self, request: ReviewTurnParams) -> tuple[Prompt[ReviewResponse], Session]:
        return Prompt(self._template).bind(request), self._session

dispatcher = InProcessDispatcher()
loop = ReviewLoop(ClaudeAgentSDKAdapter(), dispatcher)
response, _ = loop.execute(ReviewTurnParams(request="Find bugs in main.py"))
if response.output is not None:
    review: ReviewResponse = response.output  # typed, validated
```

### 4. Iterate prompts without code changes

```python nocheck
from weakincentives.prompt.overrides import LocalPromptOverridesStore

prompt = Prompt(
    template,
    overrides_store=LocalPromptOverridesStore(),
    overrides_tag="assertive-feedback",
).bind(ReviewTurnParams(request="..."))
```

Overrides live in `.weakincentives/prompts/overrides/` and match by namespace,
key, and tag.

## Execution Harnesses

WINK integrates with agentic execution harnesses—vendor runtimes that own the
planning loop, sandboxing, and tool execution. Your agent definition stays the
same; the adapter bridges it to the harness. Three harnesses are supported today.

### Claude Agent SDK

The Claude Agent SDK harness delegates execution to Claude Code's runtime.
Claude drives the agent loop and provides native file/shell tools; WINK
supplies the portable agent definition and bridges custom tools via MCP.

- **Native tools**: Claude Code's built-in Read, Write, Edit, Glob, Grep, Bash
- **Hermetic isolation**: Ephemeral home directory prevents access to host config
- **Network policy**: Restrict tool network access to specific domains
- **MCP bridging**: Custom WINK tools bridged to Claude via MCP server
- **Sandbox**: OS-level sandboxing (bubblewrap on Linux, seatbelt on macOS)

```bash
uv add "weakincentives[claude-agent-sdk]"  # requires claude-code-sdk
```

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)
from weakincentives.prompt import WorkspaceSection, HostMount

workspace = WorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="src", mount_path="src"),),
    allowed_host_roots=("/path/to/project",),
)

adapter = ClaudeAgentSDKAdapter(
    model="claude-opus-4-6",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        cwd=str(workspace.temp_dir),
        isolation=IsolationConfig(
            network_policy=NetworkPolicy(allowed_domains=("docs.python.org",)),
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)

response = adapter.evaluate(prompt, session=session)
workspace.cleanup()
```

See [Claude Agent SDK Adapter](specs/CLAUDE_AGENT_SDK.md) for full configuration.

### Codex App Server

The Codex App Server harness delegates execution to OpenAI's Codex runtime via
its app-server protocol (stdio JSON-RPC). Codex drives the agent loop and
provides native command execution, file editing, and web search; WINK bridges
custom tools as Codex dynamic tools.

- **Native tools**: Codex's built-in command execution, file changes, web search
- **Dynamic tools**: WINK tools bridged as Codex dynamic tools over stdio
- **Structured output**: Native `outputSchema` support via OpenAI strict mode
- **No extra Python deps**: Only requires the `codex` CLI on PATH

```bash
uv add weakincentives  # no extra needed; just have `codex` CLI on PATH
```

```python
from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter,
    CodexAppServerClientConfig,
    CodexAppServerModelConfig,
)
from weakincentives.prompt import WorkspaceSection, HostMount

workspace = WorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="src", mount_path="src"),),
    allowed_host_roots=("/path/to/project",),
)

adapter = CodexAppServerAdapter(
    model_config=CodexAppServerModelConfig(model="gpt-5.3-codex"),
    client_config=CodexAppServerClientConfig(
        cwd=str(workspace.temp_dir),
        approval_policy="never",
    ),
)

response = adapter.evaluate(prompt, session=session)
workspace.cleanup()
```

See [Codex App Server Adapter](specs/CODEX_APP_SERVER.md) for full configuration.

### ACP (Agent Client Protocol)

The ACP harness delegates execution to any ACP-compatible agent binary (e.g.,
OpenCode) via JSON-RPC over stdio. The generic `ACPAdapter` handles the full
protocol flow (initialize, new_session, prompt dispatch, drain); the
`OpenCodeACPAdapter` subclass adds model validation, empty-response detection,
and OpenCode-specific quirks.

- **Native tools**: Agent's built-in command execution, file changes, web search
- **MCP bridging**: WINK tools bridged to the agent via an in-process MCP HTTP server
- **Vendor-neutral**: Same protocol works across OpenCode, Gemini CLI, and future ACP agents
- **Transcript emission**: Canonical transcript entries via `ACPTranscriptBridge`

```bash
uv add "weakincentives[acp]"  # requires agent-client-protocol, mcp, uvicorn
```

```python
from weakincentives.adapters.opencode_acp import (
    OpenCodeACPAdapter,
    OpenCodeACPAdapterConfig,
    OpenCodeACPClientConfig,
)
from weakincentives.prompt import WorkspaceSection, HostMount

workspace = WorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="src", mount_path="src"),),
    allowed_host_roots=("/path/to/project",),
)

adapter = OpenCodeACPAdapter(
    adapter_config=OpenCodeACPAdapterConfig(),
    client_config=OpenCodeACPClientConfig(
        cwd=str(workspace.temp_dir),
    ),
)

response = adapter.evaluate(prompt, session=session)
workspace.cleanup()
```

See [ACP Adapter](specs/ACP_ADAPTER.md) and
[OpenCode Adapter](specs/OPENCODE_ADAPTER.md) for full configuration.

## Development

```bash
uv sync && ./install-hooks.sh
```

Key targets:

- `make format` / `make lint` / `make typecheck`
- `make test` (100% coverage enforced)
- `make check` (all of the above plus Bandit, Deptry, pip-audit)

**Quality gates:**

- Pyright strict mode enforced
- Design-by-contract decorators (`@require`, `@ensure`, `@invariant`)
- 100% test coverage required
- Security scanning on every build

```bash
export OPENAI_API_KEY="sk-..."
make integration-tests
```

## Documentation

- `AGENTS.md` — contributor workflow
- `llms.md` — agent-friendly API overview (also the PyPI README)
- `specs/` — design documents
- `ROADMAP.md` — upcoming features

## License

Apache 2.0 • Status: Alpha (APIs may change)
