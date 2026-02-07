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

Two production-grade agentic harnesses exist today:

| Harness | Provider | Runtime |
|---------|----------|---------|
| **Claude Code** | Anthropic | Claude Agent SDK |
| **OpenAI Codex** | OpenAI | Codex App Server |

Both provide planning loops, native tools, sandboxing, and orchestration.

**Your agent definition shouldn't be locked to either.**

---

# What These Harnesses Provide

Both Claude Code and Codex are **full agentic runtimes**:

- Planning and reasoning loops
- Native file and shell tools
- OS-level or policy-based sandboxing
- Approval flows and permissions
- Crash recovery and retries
- Token budgets and deadlines

This is sophisticated machinery. You don't want to rebuild it.

---

# The Problem

You build an agent on Claude Code. It works.

Now you need to:
- Run on Codex for a different customer
- Support both for redundancy
- Migrate when pricing changes

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

# Two Harnesses, One Definition

```python
# Run on Claude Code
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    sandbox=SandboxConfig(enabled=True),
)

# Run on OpenAI Codex
adapter = CodexAppServerAdapter(
    model="gpt-5.3-codex",
    sandbox_mode="workspace-write",
)

# Same prompt, same tools, same policies
response = adapter.evaluate(prompt, session=session)
```

---

# Claude Code: What It Provides

| Capability | Implementation |
|------------|----------------|
| Native tools | File read/write, shell, network |
| Sandboxing | bubblewrap (Linux), seatbelt (macOS) |
| Isolation | Ephemeral home directory |
| Custom tools | MCP bridging |
| Permissions | Configurable approval modes |

WINK bridges your tools via MCP. Claude handles everything else.

---

# Codex: What It Provides

| Capability | Implementation |
|------------|----------------|
| Native tools | Command execution, file changes, web search |
| Sandboxing | read-only, workspace-write, full-access policies |
| Threads | Persistent conversation state with fork/resume |
| Custom tools | Dynamic tools (in-process) |
| Approvals | Configurable policy per action type |

WINK bridges your tools via dynamic tools. Codex handles everything else.

---

# What Lives in Your Definition?

| Component | Purpose | Portable? |
|-----------|---------|-----------|
| **Prompt structure** | Context engineering | Yes |
| **Custom tools** | Domain-specific capabilities | Yes |
| **Hard guardrails** | Policies that gate tool calls | Yes |
| **Soft feedback** | Guidance that nudges behavior | Yes |
| **Completion criteria** | "Done" verification | Yes |

All of this travels with your agent. None of it is harness-specific.

---

# The Three Control Layers

Your definition includes control at three strengths:

| Layer | Enforcement | Purpose |
|-------|-------------|---------|
| **Hard guardrails** | Fail-closed block | Gate tool calls |
| **Soft feedback** | Advisory guidance | Nudge behavior |
| **Completion check** | Block termination | Verify goals |

These work identically on both harnesses.

---

# Hard Guardrails: Tool Policies

Policies **gate tool invocations**. If denied, the tool doesn't run.

```python
ReadBeforeWritePolicy()
# → Blocks write_file if the file exists but wasn't read

SequentialDependencyPolicy(
    dependencies={"deploy": frozenset({"test", "build"})}
)
# → Blocks deploy until test and build have succeeded
```

Works on Claude Code. Works on Codex. Same policy definition.

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
FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),
)
```

Every 30 seconds: *"2 minutes remaining. Consider wrapping up."*

Same feedback, both harnesses.

---

# Completion Checking: "Done" Means Done

Agents sometimes declare victory prematurely.

**Task completion checkers** verify goals before allowing termination.

```python
PlanBasedChecker(plan_type=Plan)
# → Blocks termination if plan steps remain incomplete

CompositeChecker(
    checkers=(PlanBasedChecker(), FileExistsChecker(("output.txt",))),
    all_must_pass=True,
)
```

---

# Custom Tools: Bridged Automatically

Define tools once. WINK bridges them to each harness:

```python
@tool(name="search_docs", description="Search documentation")
def search_docs(params: SearchParams, *, context: ToolContext) -> ToolResult:
    results = my_search_engine.query(params.query)
    return ToolResult.ok(results)
```

- **Claude Code**: Bridged via MCP server
- **Codex**: Bridged via dynamic tools

Same tool definition. Automatic bridging.

---

# Tool Bridging Architecture

```
┌─────────────────────────────────────────────┐
│            Your Tool Definition              │
│  @tool(name="search_docs", ...)             │
└─────────────────┬───────────────────────────┘
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
┌──────────────┐     ┌──────────────┐
│  Claude Code │     │    Codex     │
│  (MCP Bridge)│     │(Dynamic Tool)│
└──────────────┘     └──────────────┘
```

---

# Prompt as Agent

The prompt is a **typed, hierarchical document** where sections bundle instructions and tools together.

```
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)
├── WorkspaceDigestSection     ← auto-generated codebase summary
├── CustomToolsSection         ← your domain tools + policies
└── MarkdownSection (request)
```

Sections travel with the prompt. Switch harnesses, keep everything.

---

# Co-location Enables Portability

A section bundles:
- Instructions (markdown)
- Tools (your custom capabilities)
- Policies (constraints on those tools)
- Documentation (for the model)

All travel together. Nothing to synchronize across harness boundaries.

---

# Event-Driven State

Every mutation flows through pure reducers:

```python
@reducer(on=FileRead)
def track_read(state: ReadFiles, event: FileRead) -> SliceOp[ReadFiles]:
    return Append(ReadFiles(path=event.path))
```

- State is **immutable and inspectable**
- Policies and feedback query this state
- Same state model on both harnesses

---

# Testing Without a Harness

Because the definition is portable and deterministic:

```python
def test_policy_blocks_unread_write():
    session = Session()
    policy = ReadBeforeWritePolicy()

    decision = policy.check("write_file", {"path": "config.yaml"},
                            context=ctx)

    assert not decision.allowed
    assert "not read" in decision.reason
```

Test your definition. No harness mocking required.

---

# When to Use Each Harness

| Scenario | Recommendation |
|----------|----------------|
| Claude-native features needed | Claude Code |
| OpenAI models required | Codex |
| Maximum sandboxing | Claude Code (OS-level) |
| Thread persistence | Codex |
| Redundancy/failover | Both |

Your definition works on either. Choose based on requirements.

---

# What You Get

- **Portability** — same definition on Claude Code and Codex
- **Hard guardrails** — fail-closed policies on both harnesses
- **Soft feedback** — advisory guidance on both harnesses
- **Completion checking** — goal verification on both harnesses
- **Custom tools** — automatic bridging to each harness
- **Testability** — validate definition without harness

---

# What WINK Is

- A Python library for **portable agent definitions**
- Adapters for **Claude Code** and **OpenAI Codex**
- **Three-tier control**: policies, feedback, completion checking
- A philosophy: **own the definition, rent the harness**

---

# What WINK Is Not

- Not a planning loop (the harness provides that)
- Not a sandboxing system (the harness provides that)
- Not trying to replace Claude Code or Codex

WINK is the **definition layer** that makes your agent portable across these sophisticated harnesses.

---

# The Bet

Agentic harnesses will keep improving. Planning loops will get smarter. Sandboxing will get tighter. Native tools will expand.

**You don't want to own that machinery.**

You want to own the definition: your prompts, your tools, your policies, your completion criteria.

WINK lets you own what matters and rent the rest.

---

# Key Takeaways

1. **Two production harnesses exist**: Claude Code and Codex
2. **Agent definitions should be portable** across both
3. **Hard guardrails** (policies) fail-closed on both harnesses
4. **Soft feedback** nudges on both harnesses
5. **Custom tools** bridge automatically to each harness
6. **Own the definition, rent the harness**

---

# WINK
## Weak Incentives (Is All You Need)

*Portable agents across sophisticated execution harnesses*

**github.com/weakincentives** | Apache 2.0 | Alpha
