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
  blockquote {
    border-left: 4px solid #2563eb;
    padding-left: 1rem;
    font-style: italic;
  }
  table {
    font-size: 0.9em;
  }
---

# WINK
## The Agent Definition Layer

Portable agents across sophisticated execution harnesses

---

# The Landscape

Three production-grade agentic harnesses:

| Harness | Provider | Protocol |
|---------|----------|----------|
| **Claude Code** | Anthropic | Claude Agent SDK (Python) |
| **OpenAI Codex** | OpenAI | Codex App Server (stdio JSON-RPC) |
| **OpenCode** | opencode.ai | ACP (stdio JSON-RPC 2.0) |

Each provides planning loops, native tools, sandboxing, and orchestration.

**Your agent definition shouldn't be locked to any of them.**

---

# What These Harnesses Provide

All three are **full agentic runtimes**:

- Planning and reasoning loops
- Native file and shell tools
- Sandboxing and permission management
- Crash recovery and retries
- Token budgets and deadlines

This is sophisticated machinery. You don't want to rebuild it.

---

# The Problem

You build an agent on Claude Code. It works.

Now you need to:
- Run on Codex for a different customer
- Try OpenCode for its model flexibility
- Support multiple for redundancy

**How much do you rewrite?**

---

# The Agent Definition Layer

WINK separates what you own from what the harness provides:

<div class="columns">

**Definition (yours)**
- Prompt structure
- Custom tools
- Policies & feedback
- Completion criteria

**Harness (theirs)**
- Planning loop
- Native tools
- Sandboxing
- Orchestration

</div>

Your definition is **portable**. The harness is **rented**.

---

# Three Harnesses, One Definition

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter, ClaudeAgentSDKClientConfig,
)
from weakincentives.adapters.codex_app_server import (
    CodexAppServerAdapter, CodexAppServerModelConfig,
    CodexAppServerClientConfig,
)
from weakincentives.adapters.opencode_acp import (
    OpenCodeACPAdapter, OpenCodeACPAdapterConfig,
    OpenCodeACPClientConfig,
)
```

Same prompt, same tools, same policies — pick your adapter.

---

# Adapter Instantiation

```python
# Claude Code
claude = ClaudeAgentSDKAdapter(
    model="claude-opus-4-6",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
    ),
)

# OpenAI Codex
codex = CodexAppServerAdapter(
    model_config=CodexAppServerModelConfig(model="gpt-5.3-codex"),
    client_config=CodexAppServerClientConfig(sandbox_policy=WorkspaceWritePolicy()),
)

# OpenCode (via ACP)
opencode = OpenCodeACPAdapter(
    adapter_config=OpenCodeACPAdapterConfig(model_id="openai/gpt-5.1-codex"),
    client_config=OpenCodeACPClientConfig(permission_mode="auto"),
)
```

---

# Same Evaluation Call

```python
# Any adapter — same prompt, same session
response = adapter.evaluate(prompt, session=session)
```

Your definition doesn't change. Only the adapter does.

---

# Claude Code: What It Provides

| Capability | Implementation |
|------------|----------------|
| Native tools | Read, Write, Edit, Glob, Grep, Bash |
| Sandboxing | bubblewrap (Linux), seatbelt (macOS) |
| Isolation | Ephemeral home directory |
| Custom tools | MCP bridging (in-process) |
| Permissions | bypassPermissions, acceptEdits, etc. |

WINK bridges your tools via an in-process MCP server.

---

# Codex: What It Provides

| Capability | Implementation |
|------------|----------------|
| Native tools | Command execution, file changes, web search |
| Sandboxing | read-only, workspace-write, full-access |
| Custom tools | Dynamic tools over stdio |
| Structured output | Native `outputSchema` |
| Approvals | Configurable per action type |

WINK bridges your tools via Codex dynamic tools.

---

# OpenCode (ACP): What It Provides

| Capability | Implementation |
|------------|----------------|
| Native tools | Command execution, file edits, web search |
| Protocol | ACP v1 (vendor-neutral JSON-RPC 2.0) |
| Model access | OpenAI, OpenCode Zen, and more |
| Custom tools | MCP passthrough (HTTP, in-process) |
| Modes | build (execute), plan (read-only) |

WINK bridges your tools via an MCP HTTP server.

---

# Why ACP Matters

The Agent Client Protocol is a **vendor-neutral standard**:

- Same protocol implemented by multiple agents
- Not tied to any single provider's runtime
- Sessions, model discovery, and MCP passthrough built-in

WINK's `ACPAdapter` is a reusable base — `OpenCodeACPAdapter` is just the first concrete implementation. Future ACP agents (Gemini CLI, others) plug in via the same base.

---

# Tool Bridging: Three Paths, One Definition

```python
def search_handler(
    params: SearchParams, *, context: ToolContext
) -> ToolResult[SearchResult]:
    results = engine.query(params.query)
    return ToolResult.ok(SearchResult(matches=results))

search = Tool[SearchParams, SearchResult](
    name="search_docs",
    description="Search documentation for relevant content",
    handler=search_handler,
)
```

---

# Automatic Bridging Per Harness

```
┌───────────────────────────────────────────────────┐
│          Tool[SearchParams, SearchResult]          │
└──────────────────┬────────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
┌────────────┐ ┌────────┐ ┌───────────┐
│ Claude Code│ │ Codex  │ │ OpenCode  │
│(MCP in-proc)│ │(Dynamic│ │(MCP HTTP) │
│            │ │ Tool)  │ │           │
└────────────┘ └────────┘ └───────────┘
```

Same tool definition. Three bridging mechanisms. Automatic.

---

# The Three Control Layers

Your definition includes control at three strengths:

| Layer | Enforcement | Purpose |
|-------|-------------|---------|
| **Hard guardrails** | Fail-closed block | Gate tool calls |
| **Soft feedback** | Advisory guidance | Nudge behavior |
| **Completion check** | Block termination | Verify goals |

These work identically across all three harnesses.

---

# Hard Guardrails: Tool Policies

Policies **gate tool invocations**. If denied, the tool doesn't run.

```python
from weakincentives.prompt import ReadBeforeWritePolicy, SequentialDependencyPolicy

ReadBeforeWritePolicy()
# -> Blocks write_file if the file exists but wasn't read

SequentialDependencyPolicy(
    dependencies={"deploy": frozenset({"test", "build"})}
)
# -> Blocks deploy until test and build have succeeded
```

Same policy definition on all three harnesses.

---

# Policy Characteristics

- **Fail-closed** — when uncertain, deny
- **Composable** — multiple policies combine; all must allow
- **Observable** — denials logged with reason
- **Educational** — denial message helps agent self-correct

The agent sees: *"Policy denied: file exists but was not read first."*

---

# Soft Feedback: Guidance Without Blocking

Feedback providers **observe and advise**. The agent decides how to respond.

```python
from weakincentives.prompt import (
    FeedbackProviderConfig, FeedbackTrigger, DeadlineFeedback,
)

FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),
)
```

Every 30 seconds: *"2 minutes remaining. Consider wrapping up."*

---

# Completion Checking: "Done" Means Done

Agents sometimes declare victory prematurely.

**Task completion checkers** verify goals before allowing termination.

```python
from weakincentives.adapters.claude_agent_sdk import (
    PlanBasedChecker, CompositeChecker,
)

PlanBasedChecker(plan_type=MyPlan)
# -> Blocks termination if plan steps remain incomplete

CompositeChecker(
    checkers=(PlanBasedChecker(plan_type=MyPlan), FileExistsChecker(...)),
    all_must_pass=True,
)
```

---

# Prompt as Agent

The prompt is a **typed, hierarchical document** where sections bundle instructions and tools together.

```
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)
├── WorkspaceSection            <- host file mounts (any adapter)
├── CustomToolsSection          <- your domain tools + policies
└── MarkdownSection (request)
```

Sections travel with the prompt. Switch harnesses, keep everything.

---

# Event-Driven State

Every mutation flows through pure reducers on frozen dataclasses:

```python
@dataclass(frozen=True)
class ReadFiles:
    paths: tuple[str, ...] = ()

    @reducer(on=FileRead)
    def track_read(self, event: FileRead) -> ReadFiles:
        return replace(self, paths=(*self.paths, event.path))
```

- State is **immutable and inspectable**
- Policies and feedback query this state
- Same state model across all harnesses

---

# Testing Without a Harness

Because the definition is portable and deterministic:

```python
def test_policy_blocks_unread_write():
    session = Session(dispatcher=InProcessDispatcher())
    policy = ReadBeforeWritePolicy()

    decision = policy.check(
        tool_name="write_file",
        params={"path": "config.yaml"},
        context=policy_context,
    )

    assert not decision.allowed
    assert "not read" in decision.reason
```

Test your definition. No harness mocking required.

---

# When to Use Each Harness

| Scenario | Recommendation |
|----------|----------------|
| Claude models, OS-level sandboxing | Claude Code |
| OpenAI models, native structured output | Codex |
| Multi-provider model access | OpenCode (ACP) |
| Vendor-neutral protocol | OpenCode (ACP) |
| Redundancy / failover | Mix adapters |

Your definition works on any of them.

---

# What You Get

- **Portability** — same definition on Claude Code, Codex, and OpenCode
- **Hard guardrails** — fail-closed policies across all harnesses
- **Soft feedback** — advisory guidance across all harnesses
- **Completion checking** — goal verification across all harnesses
- **Custom tools** — automatic bridging per harness
- **Testability** — validate definition without any harness

---

# What WINK Is

- A Python library for **portable agent definitions**
- Adapters for **Claude Code**, **Codex**, and **OpenCode (ACP)**
- **Three-tier control**: policies, feedback, completion checking
- A philosophy: **own the definition, rent the harness**

---

# What WINK Is Not

- Not a planning loop (the harness provides that)
- Not a sandboxing system (the harness provides that)
- Not trying to replace Claude Code, Codex, or OpenCode

WINK is the **definition layer** that makes your agent portable across these harnesses.

---

# The Bet

Agentic harnesses will keep improving. Planning loops will get smarter. Sandboxing will get tighter. Native tools will expand. New harnesses will appear.

**You don't want to own that machinery.**

You want to own the definition: your prompts, your tools, your policies, your completion criteria. WINK lets you own what matters and rent the rest.

---

# Key Takeaways

1. **Three harnesses today** — Claude Code, Codex, OpenCode (ACP)
2. **Agent definitions should be portable** across all of them
3. **Hard guardrails** (policies) fail-closed on every harness
4. **Soft feedback** nudges on every harness
5. **Custom tools** bridge automatically per harness
6. **Own the definition, rent the harness**

---

# WINK
## Weak Incentives (Is All You Need)

*Portable agents across sophisticated execution harnesses*

**github.com/weakincentives** | Apache 2.0 | Alpha
