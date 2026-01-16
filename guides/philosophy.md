# Philosophy

This guide explains the thinking behind WINK. Understanding these ideas will
help you make better design decisions when building agents.

## What "Weak Incentives" Means

The name comes from mechanism design: a system with the right incentives is one
where participants naturally gravitate toward intended behavior. Applied to
agents, this means shaping the prompt, tools, and context so the model's easiest
path is also the correct one.

This isn't about constraining the model or managing downside risk. It's about
*encouraging* correct behavior through structure:

- **Clear instructions co-located with tools** make the right action obvious
- **Typed contracts** guide the model toward valid outputs
- **Progressive disclosure** keeps the model focused on what matters now
- **Explicit state** gives the model the context it needs to make good decisions

The optimization process strengthens these incentives. When you refine a prompt
override or add a tool example, you're making the correct path even more
natural. Over iterations, the system becomes increasingly well-tuned—not through
constraints, but through clarity.

## Technical Strategy

WINK is built around a specific bet about where durable value lives in agent
systems:

**Don't compete at the model layer.** Models and agent frameworks will
commoditize quickly. Treating them as swappable dependencies is the winning
posture. WINK's adapters exist precisely so you can switch providers without
rewriting your agent.

**Differentiate with your system of record.** Long-term advantage comes from
owning authoritative data, definitions, permissions, and business context—not
from the reasoning loop. The model is a commodity; your domain knowledge isn't.

**Keep product semantics out of prompts.** Encode domain meaning and safety in
stable tools and structured context, not provider-specific prompt glue. When
your business logic lives in typed tool handlers and well-defined state, it
survives model upgrades.

**Use provider runtimes; own the tools.** Let vendors handle planning,
orchestration, and retries. Invest in high-quality tools that expose your
system-of-record capabilities. The Claude Agent SDK adapter is an example: it
delegates execution to Claude Code's native runtime while WINK owns the tool
definitions and session state.

**Build evaluation as your control plane.** Make model and runtime upgrades safe
via scenario tests and structured output validation. When you can verify
behavior programmatically, you can improve without rewrites.

## The Shift: Orchestration Shrinks, Context Engineering Grows

Many early agent frameworks assumed the hard part would be workflow logic:
routers, planners, branching graphs, and elaborate loops. These frameworks spent
their complexity budget on orchestration—deciding which prompts to run when,
routing between specialized agents, managing elaborate state machines.

WINK makes a different bet.

**Models are steadily absorbing more of the reasoning loop.** What required
explicit multi-step orchestration yesterday often works in a single prompt
today. The frontier models are increasingly capable of planning, reasoning, and
self-correction within a single context window. Elaborate routing graphs often
just get in the way.

**The durable part of agent systems is tools, retrieval, and context
engineering.** Tools define what the agent can do. Retrieval determines what
information is available. Context engineering—deciding what to include, what to
summarize, how to structure information so the model reasons well—is where the
real leverage lives.

Context engineering is the tricky part. It's a genuinely new discipline: what's
relevant now, what to summarize versus preserve, how to shape information so
models reason over it well. No clean precedent from traditional engineering.
Builders who master it early win.

## The Core Bet: Prompts as First-Class, Typed Programs

Most systems treat prompts as strings and hope conventions keep everything
aligned:

- Prompt text in one place
- Tool definitions in another
- Schema expectations in another
- Memory in another

Teams add layers: prompt registries, tool catalogs, schema validators. Each
layer is separately maintained. They drift. When something breaks, you're
hunting across files to understand what was actually sent to the model.

WINK inverts this:

**A `PromptTemplate` is an immutable object graph (a tree of sections).** Each
section can render markdown instructions, declare typed placeholders, register
tools, and optionally render as a summary to save tokens.

The section that explains "here's how to search files" is the same section that
provides the `grep` tool. Documentation and capability live together. They can't
drift.

**A `Prompt` binds runtime configuration:** parameter dataclasses that fill
template placeholders, an overrides store for safe iteration, and optionally a
session for dynamic visibility and scoping.

**A `ProviderAdapter` evaluates the prompt:** renders markdown from the section
tree, executes tool calls synchronously, and returns text and/or parsed
structured output.

**A `Session` captures everything as an event-driven, reducer-managed state
log.** Every prompt render, every tool invocation, every state change is
recorded. You can query the session, snapshot it, restore it.

In other words: **your agent is a typed prompt + tools + state.**

Two "novel" properties fall out of this structure:

1. **Deterministic inspection**: render, snapshot, and diff prompts. The same
   inputs produce the same outputs. You can write tests that assert on exact
   prompt text.

2. **Safe iteration**: apply prompt tweaks via overrides that are validated
   against hashes. When you change a section in code, existing overrides stop
   applying until you explicitly update them. No silent drift.

## Concretely, WINK Pushes You Toward:

**Explicit side effects.** Side effects live in tool handlers. Everything
else—prompt rendering, state transitions, reducers—is deterministic and pure.
When something goes wrong, you know exactly where to look: the tool handler that
executed.

**Typed contracts everywhere.** Params, tool calls, tool results, structured
outputs, session state—all typed with dataclasses. Type mismatches surface at
construction time, not at runtime when the model is mid-response. Pyright strict
mode is enforced; the type checker is your first line of defense.

**Inspectability.** If a run went wrong, you can inspect exactly what was
rendered and what tools ran. Sessions record every event as an immutable ledger.
Snapshots let you capture state at any point and restore it later.

**Controlled context growth.** Progressive disclosure lets you default to
summaries and expand on demand. Instead of stuffing everything into the prompt
upfront, you let the model request what it needs. This keeps token counts
manageable and models focused.

**Safe iteration.** Hash-validated prompt overrides prevent accidental drift
between "tested" and "running". When you override a section's text, the system
validates that you're overriding the version you think you're overriding.

## What WINK Is (and Is Not)

**WINK is:**

- A Python library (`weakincentives`) for building prompts-as-agents
- A small runtime for state (`Session`) and orchestration (`MainLoop`)
- Adapters (`OpenAI`, `LiteLLM`, `Claude Agent SDK`) that execute tools and
  parse outputs consistently
- Contributed tool suites for background agents (planning, virtual FS,
  sandboxes, workspace digests)

**WINK is not:**

- A distributed workflow engine—if you need to coordinate across machines or
  manage long-running jobs, use something built for that
- A framework that tries to "own" your application architecture—WINK is a
  library; use it for the pieces that benefit from determinism
- A multi-agent coordination system—WINK focuses on single-agent patterns done
  well; multi-agent is possible but not the primary design target
- An async-first streaming framework—today the adapter contract is synchronous;
  streaming may come later

If you need a graph engine or multi-agent coordination, you can still use WINK
for the pieces that benefit from determinism (prompt design, tool contracts,
state snapshots) and let something else coordinate the rest. WINK plays well
with others.

## Next Steps

- [Quickstart](quickstart.md): Get a working agent running
- [Prompts](prompts.md): Learn how prompt composition works in detail
