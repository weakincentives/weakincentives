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
---

# WINK
## The Agent Definition Layer

Portable agents with soft feedback and hard guardrails

---

# The Problem

You build an agent. It works with one provider's runtime.

Then:
- You want to switch providers
- The runtime upgrades its planning loop
- You move from dev to production sandboxing

**How much do you rewrite?**

---

# The Insight

High-quality agents have two distinct parts:

1. **Definition** — prompt, tools, policies, feedback
2. **Harness** — planning loop, sandboxing, retries, scheduling

These change at different rates, for different reasons, by different teams.

---

# The Agent Definition Layer

WINK makes your agent definition a **portable artifact**:

- Same definition runs on OpenAI, Claude Agent SDK, or other harnesses
- Version it, review it, test it independently of runtime
- Swap harnesses without touching agent logic

**Own the definition. Rent the harness.**

---

# What Lives in the Definition?

| Component | Purpose |
|-----------|---------|
| **Prompt structure** | Context engineering, co-located tools |
| **Tool contracts** | Typed inputs/outputs, side effect boundary |
| **Hard guardrails** | Policies that gate tool calls |
| **Soft feedback** | Guidance that nudges behavior |
| **Completion criteria** | "Done" means what you say it means |

---

# What Lives in the Harness?

| Component | Purpose |
|-----------|---------|
| Planning loop | Decides what to do next |
| Sandboxing | OS-level isolation |
| Retries & throttling | Error recovery |
| Budgets & deadlines | Resource limits |
| Native tools | File, shell, network |

You don't own this. You rent it.

---

# Portability in Practice

```python
# Development: run everything in WINK
adapter = OpenAIAdapter(model="gpt-4o")

# Production: rent Claude's native harness
adapter = ClaudeAgentSDKAdapter(
    model="claude-opus-4-6",
    sandbox=SandboxConfig(enabled=True),
)

# Same prompt, same tools, same policies
response = adapter.evaluate(prompt, session=session)
```

---

# The Three Control Layers

Agents need control at different strengths:

| Layer | Enforcement | When it fires |
|-------|-------------|---------------|
| **Hard guardrails** | Fail-closed block | Before tool executes |
| **Soft feedback** | Advisory guidance | After tool executes |
| **Completion check** | Block termination | When agent says "done" |

---

# Hard Guardrails: Tool Policies

Policies **gate tool invocations**. If a policy denies, the tool doesn't run.

```python
@policy
def must_read_before_write(ctx: PolicyContext) -> bool:
    """Don't overwrite a file you haven't read."""
    if ctx.tool_name == "write_file":
        path = ctx.params["path"]
        if ctx.filesystem.exists(path):
            return ctx.session[ReadFiles].contains(path)
    return True
```

---

# Policy Characteristics

- **Fail-closed** — when uncertain, deny
- **Composable** — multiple policies combine; all must allow
- **Observable** — denials logged with reason
- **Agent learns** — denial message helps self-correction

The agent sees: *"Policy denied write_file: file exists but was not read first."*

---

# Built-in Policies

**ReadBeforeWritePolicy** — existing files must be read before overwritten

**SequentialDependencyPolicy** — tool B requires tool A first

```python
SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
        "build": frozenset({"lint"}),
    }
)
```

---

# Why Policies Over Workflows?

| Aspect | Workflow | Policy |
|--------|----------|--------|
| Specifies | Steps to execute | Constraints to satisfy |
| On unexpected | Fails or branches | Agent reasons |
| Composability | Sequential coupling | Independent conjunction |
| Agent role | Executor | Reasoner |

Policies say *what* must hold. The agent finds *how*.

---

# Soft Feedback: Guidance Without Blocking

Feedback providers **observe and advise**. They don't block—the agent decides.

```python
FeedbackProviderConfig(
    provider=DeadlineFeedback(warning_threshold_seconds=120),
    trigger=FeedbackTrigger(every_n_seconds=30),
)
```

Every 30 seconds: *"2 minutes remaining. Consider wrapping up."*

---

# Feedback vs Policies

| Aspect | Feedback | Policy |
|--------|----------|--------|
| Enforcement | Advisory | Mandatory |
| Timing | After tool completes | Before tool executes |
| Response | Agent interprets | Hard block |
| Purpose | Course correction | Invariant protection |

Use feedback for soft nudges. Use policies for hard limits.

---

# Feedback Triggers

| Trigger | Use case |
|---------|----------|
| `every_n_calls` | After N tool invocations |
| `every_n_seconds` | Periodic time-based |
| `on_file_created` | When specific file appears |

```python
FeedbackTrigger(
    on_file_created=FileCreatedTrigger(filename="AGENTS.md"),
)
# → "AGENTS.md detected. Follow conventions defined within."
```

---

# Custom Feedback Provider

```python
@dataclass(frozen=True)
class ToolUsageMonitor:
    max_calls: int = 20

    def provide(self, *, context: FeedbackContext) -> Feedback:
        count = context.tool_call_count
        if count > self.max_calls:
            return Feedback(
                summary=f"{count} tool calls without progress marker.",
                suggestions=("Review your approach.",),
                severity="caution",
            )
        return Feedback(summary="On track.", severity="info")
```

---

# Completion Checking: "Done" Means Done

Agents sometimes declare victory prematurely.

**Task completion checkers** verify goals before allowing termination.

```python
adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        task_completion_checker=PlanBasedChecker(plan_type=Plan),
    ),
)
```

---

# How Completion Checking Works

1. Agent signals completion
2. Checker inspects session state
3. If incomplete: inject feedback, request more turns
4. If complete: allow termination

```python
class PlanBasedChecker:
    def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
        plan = context.session[Plan].latest()
        incomplete = [s for s in plan.steps if s.status != "done"]
        if incomplete:
            return TaskCompletionResult.incomplete(
                f"Incomplete: {', '.join(s.title for s in incomplete[:3])}"
            )
        return TaskCompletionResult.ok()
```

---

# The Three Layers Together

```
┌─────────────────────────────────────────────┐
│              Agent Definition               │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │ Completion Checking                  │   │
│  │ "Block termination until done"       │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │ Soft Feedback                        │   │
│  │ "Nudge toward better behavior"       │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │ Hard Guardrails                      │   │
│  │ "Block unsafe operations"            │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

# Prompt as Agent

The prompt is a **typed, hierarchical document** where sections bundle instructions and tools together.

```
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)
├── PlanningToolsSection       ← contributes tools + policies
├── VfsToolsSection            ← contributes tools + policies
└── MarkdownSection (request)
```

Disable a section → its tools and policies vanish.

---

# Co-location Enables Portability

The section that explains filesystem operations:
- Provides the `read_file` tool
- Declares `ReadBeforeWritePolicy`
- Includes usage documentation

All travel together. Switch harnesses, keep the section.

---

# Event-Driven State

Every mutation flows through pure reducers:

```python
@reducer(on=FileRead)
def track_read(state: ReadFiles, event: FileRead) -> SliceOp[ReadFiles]:
    return Append(ReadFiles(path=event.path))
```

- State is **immutable and inspectable**
- Snapshot at any point, restore later
- Policies and feedback query session state

---

# Testing the Definition

Because the definition is portable and deterministic:

```python
def test_policy_blocks_unread_write():
    session = Session()
    policy = ReadBeforeWritePolicy()

    # File exists but wasn't read
    decision = policy.check("write_file", {"path": "config.yaml"},
                            context=ctx)

    assert not decision.allowed
    assert "not read" in decision.reason
```

No mocking the runtime. Test the definition directly.

---

# Adapter Integration

| Adapter | Harness | Your definition |
|---------|---------|-----------------|
| Claude Agent SDK | Claude's native loop, sandboxing | Portable |
| OpenAI | WINK-managed loop | Portable |
| LiteLLM | WINK loop, multiple providers | Portable |

Custom tools bridge via MCP where needed.

---

# What You Get

- **Portability** — same definition, different runtimes
- **Hard guardrails** — fail-closed policies protect invariants
- **Soft feedback** — advisory guidance for course correction
- **Completion checking** — agents finish what they start
- **Testability** — validate the definition independently

---

# What WINK Is

- A Python library for **portable agent definitions**
- **Three-tier control**: policies, feedback, completion checking
- **Adapters** that run definitions on different harnesses
- A philosophy: **own the definition, rent the harness**

---

# What WINK Is Not

- Not a workflow engine
- Not a multi-agent coordinator
- Not trying to own the planning loop

WINK is the **definition layer** that makes your agent portable.

---

# Key Takeaways

1. **Agent definitions should be portable** across harnesses
2. **Hard guardrails** (policies) fail-closed to protect invariants
3. **Soft feedback** nudges without blocking
4. **Completion checking** ensures agents finish the job
5. **The prompt is the agent** — co-located tools, policies, docs

---

# WINK
## Weak Incentives (Is All You Need)

*Portable agents with soft feedback and hard guardrails*

**github.com/weakincentives** | Apache 2.0 | Alpha
