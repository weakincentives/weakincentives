# The Prompt Is the Agent

The reasoning loop is moving model-side.

Two years ago, building an AI agent meant orchestrating chains of prompts,
managing state machines, routing between specialized sub-agents, and writing
elaborate retry logic. Today, that scaffolding is becoming obsolete. Models are
absorbing it.

This shift changes what it means to build agents. The job isn't orchestration
anymore—it's context engineering.

Weak Incentives (WINK) is a small framework for building agents as a single,
structured prompt document—where instructions, tools, and progressive disclosure
live together—and for running that agent with event-sourced, replayable state.
It's designed for this new reality.

## The Great Migration

Consider how agent architectures have evolved:

**2023**: Multi-prompt chains with explicit routing, separate tool registries,
state machines controlling flow, nested sub-agents for specialized tasks.

**2025**: The default architecture is getting flatter. A single prompt with
good context, clear instructions, and the right tools—plus a smaller layer of
boring infrastructure for constraints, budgets, and auditability.

Complex nested workflows may still work today, but they age poorly. Every
clever routing decision you encode is a bet against model improvement. Every
sub-agent you create is a boundary the model will eventually internalize.
Orchestration becomes infrastructure, not strategy.

What remains after this migration? Four things:

1. **Tools** — external capabilities the model can invoke
2. **Retrieval** — finding and surfacing relevant context
3. **Context engineering** — the genuinely new discipline
4. **Infrastructure** — budgets, deadlines, sandboxing, typed I/O

Tools and retrieval draw on familiar software skills. APIs, databases, search
indexes—we know how to build these. Infrastructure is boring by design. But
context engineering doesn't have clean precedents. It's the craft of deciding
what's relevant now, what to summarize versus preserve, how to structure
information so models reason over it well.

## The Prompt as Agent

Most frameworks treat prompts as an afterthought—string templates glued to
separately registered tool lists. This creates drift. The documentation says
one thing, the tool registry does another, and the routing logic adds yet more
behavior.

What if we inverted this? What if the prompt *was* the agent?

By "prompt" I don't mean a single blob of text. I mean a structured document
with typed sections, visibility rules, and a tool interface—a hierarchical
spec that fully describes the agent's capabilities.

```python
from dataclasses import dataclass
from weakincentives import MarkdownSection, Prompt
from weakincentives.tools.planning import PlanningToolsSection
from weakincentives.tools.vfs import VfsToolsSection

@dataclass(slots=True, frozen=True)
class ReviewResponse:
    summary: str
    issues: list[str]
    next_steps: list[str]

review_agent = Prompt[ReviewResponse](
    ns="code-review",
    key="reviewer",
    name="code_review_agent",
    sections=(
        MarkdownSection(
            title="Code Review Brief",
            template="""
            You are a code review assistant exploring the mounted workspace.
            Use planning tools to track multi-step investigations.
            Use filesystem tools to read and navigate code.
            """,
            key="brief",
        ),
        PlanningToolsSection(session=session),  # Contributes plan_* tools
        VfsToolsSection(session=session, mounts=mounts),  # Contributes vfs_* tools
        MarkdownSection[UserRequest](
            title="Request",
            template="${request}",
            key="request",
        ),
    ),
)
```

This isn't a prompt template plus separately registered tools. It's a single
hierarchical document where each section bundles its own instructions and
capabilities. The `PlanningToolsSection` contributes planning tools *and*
documentation for using them. The `VfsToolsSection` contributes filesystem
tools *and* their usage instructions.

The prompt is the explicit interface: what tools exist, how they're described,
and what context is available. The runtime still enforces hard boundaries—but
the prompt is the spec.

## Co-location Prevents Drift

Here's the problem with separate registries:

```python
# Elsewhere in your codebase...
tool_registry.register(vfs_read_file)
tool_registry.register(vfs_write_file)
tool_registry.register(vfs_list_files)

# In your prompt template (another file)...
SYSTEM_PROMPT = """
You can read files using the read_file tool.
"""
```

The tool is named `vfs_read_file` but the prompt says `read_file`. This happens
constantly. Documentation drifts from implementation. The model gets confused.
You debug for hours.

Co-location fixes this:

```python
VfsToolsSection(
    session=session,
    mounts=(
        HostMount(
            host_path="repo",
            mount_path=VfsPath(("repo",)),
        ),
    ),
)
```

The section that explains filesystem navigation is the same section that
provides the tools. They're defined together, versioned together, deployed
together. Documentation can't drift from implementation because they're the
same object.

Co-location helps, but drift can still happen—tool behavior changes while docs
remain accurate-but-incomplete, or models hallucinate tool names under pressure.
WINK validates tool mentions against the live schema at render time and
auto-generates parameter documentation from type annotations. When the model
calls a tool that doesn't exist, the error message includes the actual tool
names available.

## Dynamic Scoping

Agents need different capabilities in different contexts. Traditional
approaches scatter this across configuration files, environment checks, and
conditional logic.

Sections solve this declaratively:

```python
from weakincentives.prompt import SectionVisibility

reference_docs = MarkdownSection[ReferenceParams](
    title="Reference Documentation",
    template="""
    ## Architecture Overview
    Core components are organized into discrete packages.
    Dependencies flow inward toward the domain layer.

    ## Code Conventions
    Follow PEP 8 style guidelines.
    Use type annotations for all public functions.
    """,
    summary="Documentation available. Request expansion if needed.",
    key="reference-docs",
    visibility=SectionVisibility.SUMMARY,
)
```

This section starts summarized—the model sees only "Documentation available."
When detailed docs are needed, the model can call `open_sections` to expand it.
The full content appears; the prompt adapts.

Who decides to expand? The model does—but expanding costs tokens and budget.
The runtime can enforce maximum expansions per turn and total context limits.
Section summaries are designed to be decision-useful: enough information to
know *whether* to expand, not enough to skip it entirely. This is the "weak
incentives" philosophy in action: make the efficient path the easy path.

Disable a section and its entire subtree—tools included—vanishes:

```python
debug_section = MarkdownSection[DebugParams](
    title="Debug Tools",
    key="debug",
    template="...",
    tools=(shell_execute, memory_dump),
    enabled=lambda params: params.debug_mode,
)
```

When `debug_mode` is False, the debug tools don't exist. No conditional logic
in the routing layer. No "if debug_mode then allow these tools" checks. The
prompt structure itself encodes the capability boundary.

## Observable State

Agent state shouldn't live in scattered dictionaries and instance variables.
It should be inspectable, replayable, and serializable.

WINK sessions use Redux-style state management:

```python
from weakincentives.runtime import Session, InProcessEventBus

bus = InProcessEventBus()
session = Session(bus=bus)

# Query state
plan = session.query(Plan).latest()
for step in plan.steps:
    print(f"[{step.status}] {step.title}")

# Mutate through events
session.mutate(Plan).dispatch(UpdateStep(step_id=1, status="complete"))

# Snapshot for persistence
snapshot = session.snapshot()
json_str = snapshot.to_json()
```

Every state change flows through pure reducers processing published events.
Tool calls become events. Prompt evaluations become events. Internal decisions
become events. The full history is a replayable ledger—not scattered mutations
in free-form dicts.

State drives prompt behavior. Section enablement is a pure function of session
state—when the plan changes, sections that depend on plan state automatically
adjust. Tool handlers read from and write to the session, and those writes
flow through reducers that other parts of the system can observe:

```python
# A tool call produces an event
@dataclass(frozen=True)
class FileRead:
    path: str
    content: str

# The session captures it
session.mutate(FileRead).append(FileRead(path="main.py", content="..."))

# Other sections can react
enabled=lambda params: session.query(FileRead).count() > 0
```

This isn't just architectural purity. It's practical:

- **Debugging**: Replay the event stream to reproduce any state
- **Testing**: Seed state, run a turn, assert on the result
- **Auditing**: Every decision has a trace

## The Real Work

If the reasoning loop moves model-side, what's left for agent builders?

Not orchestration. Not routing. Not the scaffolding that frameworks have
focused on for years.

What's left is **context engineering**: the discipline of preparing information
so models can reason over it effectively. This includes:

**Relevance filtering** — What matters right now? What can be summarized? What
should be omitted entirely?

**Structure** — How should information be organized? Hierarchical sections?
Flat lists? Progressive disclosure?

**Grounding** — How do you connect abstract instructions to concrete
capabilities? How do you prevent hallucinated tool names?

**Adaptation** — How does the context change as the task evolves? What triggers
expansion or contraction of available information?

Consider the code reviewer example:

```python
workspace_section = VfsToolsSection(
    session=session,
    mounts=(
        HostMount(
            host_path="repo",
            mount_path=VfsPath(("repo",)),
            include_glob=("*.py", "*.md", "*.yaml"),
            exclude_glob=("**/*.pickle", "**/*.png"),
            max_bytes=600_000,
        ),
    ),
)
```

This isn't just mounting a filesystem. It's a context engineering decision:
include source code and docs, exclude binaries and images, cap the total
bytes. The agent gets what it needs without drowning in irrelevant data.

## The Shift in Practice

Here's what building agents looks like when you embrace this shift:

**Before**: Design state machines. Write routing logic. Create sub-agents for
specialized tasks. Debug the orchestration.

**After**: Design the prompt. Define sections with their tools. Set visibility
rules. Let the model orchestrate itself.

The code review agent is a single prompt:

```
Prompt[ReviewResponse]
├── MarkdownSection (brief)
├── WorkspaceDigestSection (context)
├── MarkdownSection (reference docs, summarized)
├── PlanningToolsSection (plan_*, reflect tools)
├── VfsToolsSection (vfs_* tools)
└── MarkdownSection (user request)
```

That's the entire agent. No separate tool registry. No routing layer. No
sub-agents. The prompt determines capabilities. The model determines execution.

## Some Orchestration Stays

Not everything should move model-side. Some scaffolding remains for good
reasons:

- **Auditability** — Events for every state change
- **Cost control** — Budgets and deadlines
- **Hard constraints** — Sandboxed execution, filtered file access
- **Determinism** — Typed inputs and outputs

These are the concerns that frameworks should handle. Not clever routing.
Not elaborate chains. The mechanical infrastructure that needs to exist but
shouldn't require creative solutions for every project.

## Why "Weak Incentives"?

The name captures a design philosophy: make the safe, boring path the easiest
path.

"Strong incentives" push models toward brittle hacks—hallucinated tool names
to escape constraints, over-expansion to gather context, elaborate reasoning
chains to work around missing information. These emerge when the system design
fights the model's defaults.

"Weak incentives" align the system with good behavior. Capabilities are
explicit in the prompt, so there's nothing to hallucinate. State is observable,
so there's no need for elaborate tracking hacks. Context is staged through
progressive disclosure, so expansion has a natural cost. The runtime enforces
hard boundaries, so the model doesn't need to self-police.

We don't assume perfect behavior—we make failure cheap, debuggable, and
contained. When a tool call fails, the error is typed. When the budget runs
out, the session snapshots cleanly. When context overflows, sections
summarize gracefully.

## The Boring Parts

Production agents need more than elegant abstractions. Here's how WINK handles
the unglamorous requirements:

**Security.** Progressive disclosure and tool gating help contain prompt
injection—untrusted text lands in leaf sections with limited tool access, not
in sections that control expansion or planning. The VFS sandbox filters file
access by glob patterns and size limits. Podman integration provides full
process isolation when available. But defense in depth matters: validate
inputs, audit tool calls, treat the model as untrusted.

**Testing.** Prompts are deterministic given their inputs—snapshot the rendered
text and diff it. Sessions serialize to JSON for replay tests. Tool handlers
have typed signatures that support contract testing. The event stream is a
complete trace: assert on the sequence, not just the final output.

**Versioning.** Prompts carry content hashes. Override files reference specific
versions, so a prompt edit doesn't silently invalidate your tuned variants.
Session snapshots include schema versions for forward compatibility.

**Failure handling.** Budgets enforce token and time limits at checkpoints
(before provider calls, before tool execution, after responses). Deadlines
propagate through the tool context. Retry policies live in the adapter, not
scattered across tool handlers. When limits hit, the session state is
consistent and serializable.

## What This Means for You

If you're building agents today, consider:

1. **Simplify your orchestration**. Complex routing ages poorly. A single
   well-structured prompt often beats elaborate multi-agent systems.

2. **Co-locate tools and instructions**. They should be defined together,
   tested together, deployed together.

3. **Invest in context engineering**. What information does the model need?
   How should it be structured? When should sections expand or contract?

4. **Make state observable**. Events, snapshots, replay. Debug by
   understanding what happened, not by adding more logging.

5. **Trust model improvement**. Today's clever scaffolding is tomorrow's
   technical debt. Build for the trajectory, not the current capability.

The reasoning loop is moving model-side. The question isn't whether to
adapt—it's how quickly you can shift your effort from orchestration to
context engineering.

The prompt isn't a template for an agent. The prompt *is* the agent.
