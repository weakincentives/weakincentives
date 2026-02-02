---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }
  h1 {
    color: #2563eb;
  }
  h2 {
    color: #1e40af;
  }
  code {
    background-color: #f1f5f9;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
---

# WINK
## Weak Incentives (Is All You Need)

The agent-definition layer for building unattended/background agents

---

# The Problem with Agent Frameworks

Most agent frameworks scatter concerns across multiple places:

- Prompt text in one place
- Tool definitions in another
- Schema expectations in yet another
- Memory management somewhere else

**Result:** Teams add layers (prompt registries, tool catalogs, schema validators) that drift apart over time.

---

# What is WINK?

**WINK is the agent-definition layer** for building unattended/background agents.

You define:
- **Prompt structure** (context engineering)
- **Tools** + typed I/O contracts
- **Policies** (gates on tool use)
- **Feedback** ("done" criteria)

These stay stable while runtimes change.

---

# What "Weak Incentives" Means

From mechanism design: a system with the right incentives is one where participants naturally gravitate toward intended behavior.

**Applied to agents:** Shape the prompt, tools, and context so the model's easiest path is also the correct one.

---

# Encouraging Correct Behavior Through Structure

- **Clear instructions co-located with tools** make the right action obvious
- **Typed contracts** guide the model toward valid outputs
- **Progressive disclosure** keeps the model focused on what matters now
- **Explicit state** gives the model context for good decisions

---

# Definition vs. Harness

<div class="columns">

**Agent Definition (you own)**
- Prompt structure
- Tools + typed I/O contracts
- Policies (gates on tool use)
- Feedback ("done" criteria)

**Execution Harness (runtime-owned)**
- Planning/act loop
- Sandboxing/permissions
- Retries/backoff, throttling
- Scheduling, budgets, crash recovery

</div>

---

# Why Separate Definition from Harness?

The harness keeps changing—increasingly from vendor runtimes—but your agent definition should not.

**WINK makes the definition a first-class artifact** you can:
- Version
- Review
- Test
- Port across runtimes via adapters

---

# The Prompt is the Agent

**WINK inverts the typical approach:** the prompt *is* the agent.

You define an agent as a **single hierarchical document** where each section bundles its own instructions and tools together.

---

# Prompt Structure Example

```
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)
├── WorkspaceDigestSection     ← auto-generated codebase summary
├── MarkdownSection (reference docs)
├── PlanningToolsSection       ← contributes planning_* tools
│   └── (nested planning docs)
├── VfsToolsSection            ← contributes ls/read_file/write_file
│   └── (nested filesystem docs)
└── MarkdownSection (user request)
```

---

# Why Co-location Matters

1. **Instructions and tools live together** - The section that explains filesystem navigation provides the `read_file` tool
2. **Documentation can't drift from implementation**
3. **Progressive disclosure** - Reveal advanced capabilities only when relevant
4. **Dynamic scoping** - Disable a section and its entire subtree disappears

---

# Core Abstractions

| Abstraction | Purpose |
|-------------|---------|
| **PromptTemplate** | Immutable blueprint with sections |
| **Prompt** | Runtime binding of template + parameters |
| **Session** | Event-driven state with pure reducers |
| **Tools** | Side effects boundary with typed contracts |
| **Adapters** | Provider-agnostic execution |

---

# PromptTemplate

An **immutable object graph** (a tree of sections).

Each section can:
- Render markdown instructions
- Declare typed placeholders
- Register tools
- Render as a summary to save tokens

---

# Sessions: Event-Driven State

- Every state change flows through **pure reducers** that process published events
- State is **immutable and inspectable**
- You can **snapshot at any point**
- Query with `session[Type].latest()`, `.all()`, `.where(predicate)`

---

# Tools: The Side Effects Boundary

**Side effects live in tool handlers.** Everything else is deterministic and pure.

- **Transactional execution** - Tool calls are atomic
- **Automatic rollback** - Failed tools don't leave partial state
- **Typed contracts** - Params, results, all validated at construction

---

# Tool Policies Over Workflows

| Aspect | Workflow | Policy |
|--------|----------|--------|
| Specifies | Steps to execute | Constraints to satisfy |
| On unexpected | Fails or branches | Agent reasons |
| Composability | Sequential coupling | Independent conjunction |
| Agent role | Executor | Reasoner |

---

# Provider Adapters

Same agent definition works across providers:

| Adapter | Use Case |
|---------|----------|
| **OpenAI** | Full tool execution in WINK |
| **LiteLLM** | Unified interface to multiple providers |
| **Claude Agent SDK** | Production with native tools + sandboxing |

---

# Claude Agent SDK: "Rent the Harness"

Claude's runtime drives the agent loop and native tools; WINK provides the portable agent definition.

- **Native tools** - Uses Claude Code's built-in tools
- **Hermetic isolation** - Ephemeral home directory
- **OS-level sandboxing** - bubblewrap on Linux, seatbelt on macOS
- **MCP bridging** - Custom WINK tools bridged via MCP

---

# Technical Strategy

**Don't compete at the model layer.** Models and agent frameworks will commoditize quickly.

**Differentiate with your system of record.** Long-term advantage comes from owning authoritative data, definitions, permissions—not the reasoning loop.

**Build evaluation as your control plane.** Make model and runtime upgrades safe via scenario tests.

---

# The Shift: Orchestration Shrinks

**Models are absorbing more of the reasoning loop.** What required explicit multi-step orchestration yesterday often works in a single prompt today.

**The durable part of agent systems:**
- Tools define what the agent can do
- Retrieval determines available information
- Context engineering shapes how models reason

---

# Code Example: Define Structured Output

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]
```

---

# Code Example: Compose the Prompt

```python
template = PromptTemplate[ReviewResponse](
    ns="examples/code-review",
    key="code-review-session",
    name="code_review_agent",
    sections=(
        MarkdownSection(title="Guide", key="guide",
                        template="Review code."),
        WorkspaceDigestSection(session=session),
        PlanningToolsSection(session=session),
        VfsToolsSection(session=session, mounts=mounts),
        MarkdownSection(title="Request", key="request",
                        template="${request}"),
    ),
)
```

---

# Code Example: Run and Get Typed Results

```python
loop = ReviewLoop(OpenAIAdapter(model="gpt-4o"), dispatcher)
response, _ = loop.execute(
    ReviewTurnParams(request="Find bugs in main.py")
)

if response.output is not None:
    review: ReviewResponse = response.output  # typed, validated
```

---

# Code Example: Inspect State

```python
from weakincentives.contrib.tools.planning import Plan

plan = session[Plan].latest()
if plan:
    for step in plan.steps:
        print(f"[{step.status}] {step.title}")
```

---

# Key Capabilities Summary

| Capability | Benefit |
|------------|---------|
| **Typed sections** | Composable prompt objects |
| **Hash-based overrides** | Safe prompt iteration |
| **Transactional tools** | Automatic rollback on failure |
| **Virtual filesystem** | Sandboxed file access |
| **Pure reducers** | Deterministic state management |
| **Adapter abstraction** | Provider-agnostic code |

---

# What WINK Is

- A Python library for building prompts-as-agents
- A small runtime for state (`Session`) and orchestration (`AgentLoop`)
- Adapters that execute tools and parse outputs consistently
- Contributed tool suites for background agents

---

# What WINK Is Not

- Not a distributed workflow engine
- Not a framework that "owns" your architecture
- Not a multi-agent coordination system
- Not an async-first streaming framework

**WINK plays well with others** - use it for the pieces that benefit from determinism.

---

# Getting Started

```bash
# Install
uv add weakincentives

# Optional extras
uv add "weakincentives[openai]"           # OpenAI adapter
uv add "weakincentives[litellm]"          # LiteLLM adapter
uv add "weakincentives[claude-agent-sdk]" # Claude Agent SDK
uv add "weakincentives[podman]"           # Podman sandbox
```

---

# Learning Path

| Step | Guide |
|------|-------|
| 1 | **Quickstart** - Get something running |
| 2 | **Philosophy** - Understand *why* WINK works this way |
| 3 | **Prompts** - Learn the core abstraction |
| 4 | **Tools** - Add capabilities to your agent |
| 5 | **Sessions** - Manage state correctly |
| 6 | **Adapters** - Connect to your preferred provider |

---

# Key Takeaways

1. **The prompt is the agent** - single hierarchical document
2. **Definition vs harness** - own what matters, rent the rest
3. **Policies over workflows** - encode constraints, not steps
4. **Typed contracts everywhere** - catch errors early
5. **Provider-agnostic** - your definition is portable

---

# Resources

- **Guides:** `guides/README.md`
- **Specs:** `specs/` directory
- **API Reference:** `llms.md`
- **Starter:** https://github.com/weakincentives/starter

---

# Thank You

## WINK: Weak Incentives (Is All You Need)

*Structure your prompts, tools, and context so the model's easiest path is also the correct one.*

**Status:** Alpha | **License:** Apache 2.0
