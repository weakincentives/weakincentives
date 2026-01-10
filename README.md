# Weak Incentives (Is All You Need)

WINK is the agent-definition layer for building unattended/background agents.
You define the prompt, tools, policies, and feedback that stay stable while
runtimes change. The planning loop, sandboxing, retries, and orchestration live
in the execution harness—often a vendor runtime. WINK keeps your agent
definition portable.

> **New to WINK?** Read the [WINK Guide](WINK_GUIDE.md) for a comprehensive
> introduction—philosophy, quickstart, and practical patterns for building agents.

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
├── PlanningToolsSection       ← contributes planning_* tools
│   └── (nested planning docs)
├── VfsToolsSection            ← contributes ls/read_file/write_file/...
│   └── (nested filesystem docs)
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
   Swap in a `PodmanSandboxSection` instead of `VfsToolsSection` when a shell
   is available; the prompt adapts automatically.

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
  See [Prompt Optimization](specs/PROMPT_OPTIMIZATION.md).

### Tools

- **Transactional execution.** Tool calls are atomic transactions. When a tool
  fails, WINK automatically rolls back session state and filesystem changes to
  their pre-call state. Failed tools don't leave traces in mutable state.
- **Sandboxed virtual filesystem.** Agents get an in-memory VFS tracked as
  session state. Mount host directories read-only when needed; the sandbox
  prevents accidental writes to the host.
  See [Workspace Tools](specs/WORKSPACE.md).

### Policies

- **Invariants over workflows.** Gate tool calls with explicit policies instead
  of brittle orchestration graphs. Encode constraints like "don't write before
  you've read" or "don't call tool B until tool A ran."
  See [Policies Over Workflows](specs/POLICIES_OVER_WORKFLOWS.md).

### Feedback

- **Completion resistance.** Encode "done means X" checks that run during
  execution to catch drift and premature termination.
  See [Task Completion Checking](specs/TASK_COMPLETION.md) and
  [Trajectory Observers](specs/TRAJECTORY_OBSERVERS.md) (design spec).

### State and Adapters

- **Event-driven state.** Every state change flows through pure reducers that
  process published events. State is immutable and inspectable—you can snapshot
  at any point. See [Session State](specs/SESSIONS.md).
- **Harness-swappable adapters.** Keep the agent definition stable while
  switching runtimes (OpenAI, LiteLLM, Claude Agent SDK). The Claude Agent SDK
  adapter is an example of "renting the harness": native tools + OS-level
  sandboxing, while WINK supplies the definition.
  See [Adapters](specs/ADAPTERS.md).

## Getting Started

**Requirements:** Python 3.12+, [`uv`](https://github.com/astral-sh/uv)

```bash
uv add weakincentives
# optional extras
uv add "weakincentives[openai]"           # OpenAI adapter
uv add "weakincentives[litellm]"          # LiteLLM adapter
uv add "weakincentives[claude-agent-sdk]" # Claude Agent SDK adapter
uv add "weakincentives[podman]"           # Podman sandbox
uv add "weakincentives[wink]"             # debug UI
```

### Debug UI

```bash
uv run --extra wink wink debug snapshots/session.jsonl --port 8000
```

![Debug UI](debug_ui.png)

## Tutorial: Code Review Agent

Build a code review assistant with structured output, sandboxed file access,
and observable state. Full source: [`code_reviewer_example.py`](code_reviewer_example.py)

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

```python
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.contrib.tools import PlanningToolsSection, VfsToolsSection, WorkspaceDigestSection

template = PromptTemplate[ReviewResponse](
    ns="examples/code-review",
    key="code-review-session",
    name="code_review_agent",
    sections=(
        MarkdownSection(...),                          # guidance
        WorkspaceDigestSection(session=session),       # auto-generated summary
        PlanningToolsSection(session=session),         # planning tools
        VfsToolsSection(session=session, mounts=...),  # sandboxed files
        MarkdownSection[ReviewTurnParams](...),        # user input
    ),
)

prompt = Prompt(template).bind(ReviewTurnParams(request="Review main.py"))
```

### 3. Mount files safely

```python
from weakincentives.contrib.tools import HostMount, VfsPath, VfsToolsSection

mounts = (
    HostMount(
        host_path="repo",
        mount_path=VfsPath(("repo",)),
        include_glob=("*.py", "*.md", "*.toml"),
        exclude_glob=("**/*.pickle",),
        max_bytes=600_000,
    ),
)
vfs_section = VfsToolsSection(session=session, mounts=mounts, allowed_host_roots=(SAFE_ROOT,))
```

### 4. Run and get typed results

```python
from weakincentives.runtime import MainLoop, Session
from weakincentives.runtime.events import InProcessDispatcher
from weakincentives.adapters.openai import OpenAIAdapter

class ReviewLoop(MainLoop[ReviewTurnParams, ReviewResponse]):
    def __init__(self, adapter, bus):
        super().__init__(adapter=adapter, bus=bus)
        self._session = Session(bus=bus)
        self._template = build_task_prompt(session=self._session)

    def prepare(self, request: ReviewTurnParams) -> tuple[Prompt[ReviewResponse], Session]:
        return Prompt(self._template).bind(request), self._session

bus = InProcessDispatcher()
loop = ReviewLoop(OpenAIAdapter(model="gpt-4o"), bus)
response, _ = loop.execute(ReviewTurnParams(request="Find bugs in main.py"))
review: ReviewResponse = response.output  # typed, validated
```

### 5. Inspect state

```python
from weakincentives.contrib.tools.planning import Plan

plan = session[Plan].latest()
if plan:
    for step in plan.steps:
        print(f"[{step.status}] {step.title}")
```

### 6. Iterate prompts without code changes

```python
from weakincentives.prompt.overrides import LocalPromptOverridesStore

prompt = Prompt(
    template,
    overrides_store=LocalPromptOverridesStore(),
    overrides_tag="assertive-feedback",
).bind(ReviewTurnParams(request="..."))
```

Overrides live in `.weakincentives/prompts/overrides/` and match by namespace,
key, and tag.

## Renting the Harness: Claude Agent SDK

This is the "rent the harness" path: Claude's runtime drives the agent loop and
native tools; WINK provides the portable agent definition and bridges custom
tools where needed.

```bash
python code_reviewer_example.py --claude-agent
```

Key differences:

- **Native tools**: Uses Claude Code's built-in tools instead of VFS
- **Hermetic isolation**: Ephemeral home directory prevents access to host config
- **Network policy**: Restricted to specific documentation domains
- **MCP bridging**: Custom WINK tools bridged via MCP
- **Sandbox**: OS-level sandboxing (bubblewrap on Linux, seatbelt on macOS)

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(HostMount(host_path="src", mount_path="src"),),
    allowed_host_roots=("/path/to/project",),
)

adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
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
