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

**2025**: A single prompt with good context, clear instructions, and the right
tools. The model handles the rest.

Complex nested workflows may still work today, but they won't age well. Every
clever routing decision you encode is a bet against model improvement. Every
sub-agent you create is a boundary the model will eventually internalize.

What remains after this migration? Three things:

1. **Tools** — external capabilities the model can invoke
2. **Retrieval** — finding and surfacing relevant context
3. **Context engineering** — the genuinely new discipline

Tools and retrieval draw on familiar software skills. APIs, databases, search
indexes—we know how to build these. Context engineering doesn't have clean
precedents. It's the craft of deciding what's relevant now, what to summarize
versus preserve, how to structure information so models reason over it well.

## The Prompt as Agent

Most frameworks treat prompts as an afterthought—string templates glued to
separately registered tool lists. This creates drift. The documentation says
one thing, the tool registry does another, and the routing logic adds yet more
behavior.

What if we inverted this? What if the prompt *was* the agent?

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

The prompt fully determines what the agent can think and do.

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
