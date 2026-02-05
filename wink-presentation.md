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

Separate what you own from what the runtime provides

---

# The Core Insight

High-quality unattended agents have two distinct parts:

1. **What makes your agent yours** — prompt structure, tools, policies, feedback
2. **Generic execution machinery** — planning loops, sandboxing, retries, scheduling

These change at different rates and for different reasons.

---

# The Problem Today

Most frameworks conflate definition and execution:

- Your prompt logic is tangled with a specific runtime's orchestration
- Switching providers means rewriting agent code
- Testing requires mocking the entire execution stack
- No clear boundary for what you version vs. what you rent

---

# The Agent Definition Layer

<div class="columns">

**Definition (you own)**
- Prompt structure
- Tools + typed contracts
- Policies (constraints)
- Feedback ("done" criteria)

**Harness (runtime-owned)**
- Planning/act loop
- Sandboxing
- Retries, throttling
- Budgets, crash recovery

</div>

WINK makes the definition a **first-class artifact** you can version, review, test, and port.

---

# Why This Split Matters

The harness keeps changing—increasingly provided by vendor runtimes.

Your agent definition should **not** change when:
- You switch from OpenAI to Claude
- The provider upgrades their planning loop
- You move from development to production sandboxing

**Own the definition. Rent the harness.**

---

# Novel Concept #1: The Prompt IS the Agent

Most frameworks: prompts are strings, tools are registered separately, schema lives elsewhere.

**WINK inverts this:** The prompt is a typed, hierarchical document where sections bundle instructions and tools together.

```
PromptTemplate[ReviewResponse]
├── MarkdownSection (guidance)
├── PlanningToolsSection       ← contributes planning tools
│   └── (nested docs)
├── VfsToolsSection            ← contributes file tools
│   └── (nested docs)
└── MarkdownSection (request)
```

---

# Co-location: Why It Matters

The section that explains "here's how to search files" **is the same section** that provides the `grep` tool.

- Documentation and capability live together
- They **cannot drift**
- Disable a section → its tools vanish from the prompt
- No separate tool registry to synchronize

---

# Dynamic Scoping

Each section has an `enabled` predicate:

```python
VfsToolsSection(
    enabled=lambda ctx: ctx.session[Workspace].latest() is not None
)
```

When disabled, the **entire subtree disappears**—instructions and tools.

Swap `VfsToolsSection` for `PodmanSandboxSection` when a shell is available; the prompt adapts automatically.

---

# Novel Concept #2: Policies Over Workflows

| Aspect | Workflow | Policy |
|--------|----------|--------|
| Specifies | Steps to execute | Constraints to satisfy |
| On unexpected | Fails or branches | Agent reasons |
| Composability | Sequential coupling | Independent conjunction |
| Agent role | Executor | Reasoner |

---

# Workflows Encode "How"

A workflow is a predetermined sequence:

```
1. Read the file
2. Analyze the code
3. Write the review
```

When the agent encounters something unexpected, it fails or needs explicit branching logic.

---

# Policies Encode "What"

A policy declares constraints the agent must satisfy:

```python
@policy
def must_read_before_write(ctx: PolicyContext) -> bool:
    """Don't write to a file you haven't read."""
    if ctx.tool_name == "write_file":
        return ctx.session[ReadFiles].contains(ctx.params["path"])
    return True
```

The agent remains free to find **any valid path** that satisfies constraints.

---

# Policy Characteristics

- **Declarative** — state what must hold, not how to achieve it
- **Composable** — policies combine via conjunction (all must pass)
- **Fail-closed** — unclear situations block rather than proceed
- **Observable** — violations are logged with context

---

# Novel Concept #3: Hash-Based Safe Iteration

Prompt overrides carry **content hashes**:

```json
{
  "namespace": "code-review",
  "key": "guide",
  "hash": "a1b2c3d4",
  "content": "Be more assertive about security issues."
}
```

When you change the section in code, **existing overrides stop applying** until explicitly updated.

No silent drift between "tested" and "running".

---

# Novel Concept #4: Transactional Tools

Tool calls are **atomic transactions**.

When a tool fails:
1. Session state rolls back to pre-call state
2. Filesystem changes revert
3. Error result returned to LLM

**Failed tools don't leave partial state.** This enables aggressive retry and recovery.

---

# Novel Concept #5: Event-Driven State

Every mutation flows through pure reducers:

```python
@reducer(on=FileRead)
def track_read(state: ReadFiles, event: FileRead) -> SliceOp[ReadFiles]:
    return Append(ReadFiles(path=event.path, content=event.content))
```

- State is **immutable and inspectable**
- Snapshot at any point, restore later
- Query with `session[Type].latest()`, `.all()`, `.where(predicate)`

---

# Deterministic Inspection

Because prompts are typed objects and state flows through reducers:

- Render the exact prompt that will be sent
- Diff prompts between versions
- Write tests that assert on prompt content
- Replay sessions for debugging

**Same inputs → same outputs** (except tool side effects)

---

# The Bet: Orchestration Shrinks

Early agent frameworks invested heavily in workflow logic: routers, planners, branching graphs.

**Models are absorbing the reasoning loop.** What required explicit multi-step orchestration yesterday often works in a single prompt today.

---

# Where Durable Value Lives

- **Tools** define what the agent can do
- **Retrieval** determines available information
- **Context engineering** shapes how models reason

The model is a commodity. Your domain knowledge—encoded in tools and context—is not.

---

# Context Engineering

A genuinely new discipline:

- What's relevant now vs. what to summarize
- How to structure information so models reason over it well
- When to expand detail vs. preserve tokens

No clean precedent from traditional engineering. Builders who master it early win.

---

# Progressive Disclosure

Sections can render as **summaries by default**, expanding on demand:

```python
DocsSection(
    content=large_reference_docs,
    summary="Reference documentation available. Call expand_docs to see.",
    expanded=lambda ctx: ctx.session[DocsExpanded].latest() is not None
)
```

The model requests what it needs. Token counts stay manageable.

---

# Adapters: Same Definition, Different Runtimes

```python
# Development: full control
adapter = OpenAIAdapter(model="gpt-4o")

# Production: rent Claude's harness
adapter = ClaudeAgentSDKAdapter(
    model="claude-sonnet-4-5-20250929",
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        sandbox=SandboxConfig(enabled=True),
    ),
)
```

Your prompt, tools, and policies stay **identical**.

---

# "Rent the Harness" with Claude Agent SDK

Claude's runtime handles:
- Planning and tool-call sequencing
- OS-level sandboxing (bubblewrap/seatbelt)
- Native file and shell tools

WINK provides:
- The portable agent definition
- Custom tools bridged via MCP
- Session state and policies

---

# What WINK Is

- A Python library for building **prompts-as-agents**
- A runtime for **event-driven state** and **typed tool contracts**
- **Adapters** that let definitions port across providers
- A philosophy: **own the definition, rent the harness**

---

# What WINK Is Not

- Not a distributed workflow engine
- Not a multi-agent coordination framework
- Not trying to own your architecture

WINK is a library for the pieces that benefit from **determinism and portability**.

---

# Key Takeaways

1. **Agent Definition Layer** — separate what you own from runtime machinery
2. **The Prompt IS the Agent** — co-located instructions and tools
3. **Policies Over Workflows** — declare constraints, not steps
4. **Hash-Based Overrides** — no silent drift
5. **Transactional Tools** — atomic execution with rollback
6. **Event-Driven State** — deterministic, inspectable, snapshotable

---

# WINK
## Weak Incentives (Is All You Need)

*Structure prompts, tools, and context so the model's easiest path is the correct one.*

**Status:** Alpha | **License:** Apache 2.0 | **github.com/weakincentives**
