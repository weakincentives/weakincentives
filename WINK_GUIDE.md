# Weak Incentives (WINK)

A practical guide to building deterministic, typed, safe background agents.

**What you'll learn to build:**

By the end of this guide, you'll know how to build agents that:

- Browse codebases, answer questions, and propose changes—safely sandboxed
- Use structured output to return typed, validated responses
- Maintain explicit, inspectable state across turns
- Manage token costs with progressive disclosure (show summaries, expand on
  demand)
- Iterate on prompts quickly using version-controlled overrides

The running example is a code review agent. It's a practical pattern: the agent
reads files, makes a plan, and returns structured feedback. The same patterns
apply to research agents, Q&A bots, automation assistants, and more.

---

This guide is written for engineers who want to:

- Build agents that can run unattended without turning into "a pile of prompt
  glue".
- Treat prompts as real software artifacts: testable, inspectable, and
  versionable.
- Keep tool use and side effects explicit, gated, and auditable.
- Iterate on prompts quickly without compromising correctness.

**If you only read one thing**: in WINK, the prompt is the agent.

**Status**: Alpha. Expect some APIs to evolve as the library matures.

---

## Coming from LangGraph or LangChain?

If you've built agents with LangGraph, LangChain, or similar frameworks, here's
a quick orientation.

**Different philosophy, different primitives.**

LangGraph centers on **graphs**: nodes are functions, edges are transitions,
state flows through the graph. You model agent behavior as explicit control
flow. LangChain centers on **chains**: composable sequences of calls to LLMs,
tools, and retrievers.

WINK centers on **the prompt itself**. There's no graph. There's no chain. The
prompt—a tree of typed sections—*is* your agent. The model decides what to do
next based on what's in the prompt. Tools, instructions, and state all live in
that tree.

**Concept mapping:**

| LangGraph / LangChain | WINK equivalent |
|----------------------|-----------------|
| Graph / Chain | `PromptTemplate` (tree of sections) |
| Node / Tool | `Tool` + handler function |
| State / Memory | `Session` (typed slices + reducers) |
| Router / Conditional edge | `enabled()` predicate on sections |
| Checkpointing | `session.snapshot()` / `session.restore()` |
| LangSmith tracing | Session events + debug UI |

**What's familiar:**

- Tools are functions with typed params and results. You'll recognize this.
- State management exists. Sessions use an event-driven pattern: state is
  immutable, and changes flow through pure functions called "reducers."
- Provider abstraction exists. Adapters swap between OpenAI, LiteLLM, Claude.

**What's different:**

- **No explicit routing.** You don't define edges. The model reads the prompt
  and decides which tools to call. Sections can be conditionally enabled, but
  there's no "if tool X returns Y, go to node Z."

- **Prompt and tools are co-located.** In LangChain, you define tools in one
  place and prompts in another. In WINK, the section that explains "use this
  tool for searching" is the same section that registers the tool. They can't
  drift apart.

- **Deterministic by default.** Prompt rendering is pure. State transitions
  flow through reducers. Side effects are confined to tool handlers. You can
  snapshot the entire state at any point and restore it later.

- **No async (yet).** Adapters are synchronous. This simplifies debugging at
  the cost of throughput. Async may come later.

**When to use WINK instead of LangGraph:**

- You want the prompt to be the source of truth, not a graph definition.
- You're building single-agent workflows where the model handles most routing.
- You value determinism, testability, and auditability over flexibility.
- You're tired of prompt text and tool definitions drifting apart.

**When to stick with LangGraph:**

- You need explicit multi-step workflows with complex branching logic.
- You're building multi-agent systems with explicit handoffs.
- You need async streaming throughout.

You can also use both: WINK for prompt/tool/state management, LangGraph for
higher-level orchestration.

---

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

**Build evaluation as your control plane.** Make model and runtime upgrades
safe via scenario tests and structured output validation. When you can verify
behavior programmatically, you can improve without rewrites.

The future points toward SDKs shaped like the Claude Agent SDK: sophisticated
sandboxing, native tool integration, seamless handoff between local and hosted
execution. Models will increasingly come with their own tool runtimes, deeply
integrated into training. WINK's job is to give you stable primitives—prompts,
tools, state—that work across that evolving landscape.

---

## Table of Contents

0. [Coming from LangGraph or LangChain?](#coming-from-langgraph-or-langchain)
0. [Technical Strategy](#technical-strategy)
1. [Philosophy](#1-philosophy)
   1. [What "weak incentives" means](#11-what-weak-incentives-means)
   2. [The shift: orchestration shrinks, context engineering grows](#12-the-shift-orchestration-shrinks-context-engineering-grows)
   3. [The core bet: prompts as first-class, typed programs](#13-the-core-bet-prompts-as-first-class-typed-programs)
   4. [What WINK is (and is not)](#14-what-wink-is-and-is-not)
2. [Quickstart](#2-quickstart)
   1. [Install](#21-install)
   2. [End-to-end: a tiny structured agent](#22-end-to-end-a-tiny-structured-agent)
   3. [Add a tool](#23-add-a-tool)
   4. [Your first complete agent](#24-your-first-complete-agent-copy-paste-ready)
3. [Prompts](#3-prompts)
   1. [PromptTemplate](#31-prompttemplate)
   2. [Prompt](#32-prompt)
   3. [Sections](#33-sections)
   4. [MarkdownSection](#34-markdownsection)
   5. [Structured output](#35-structured-output)
   6. [Dynamic scoping with enabled()](#36-dynamic-scoping-with-enabled)
   7. [Session-bound sections and cloning](#37-session-bound-sections-and-cloning)
   8. [Few-shot traces with TaskExamplesSection](#38-few-shot-traces-with-taskexamplessection)
4. [Tools](#4-tools)
   1. [Tool contracts](#41-tool-contracts)
   2. [ToolContext, resources, and Filesystem](#42-toolcontext-resources-and-filesystem)
   3. [ToolResult semantics](#43-toolresult-semantics)
   4. [Tool examples](#44-tool-examples)
   5. [Tool suites as sections](#45-tool-suites-as-sections)
5. [Sessions](#5-sessions)
   1. [Session as deterministic memory](#51-session-as-deterministic-memory)
   2. [Queries](#52-queries)
   3. [Reducers](#53-reducers)
   4. [Declarative reducers with @reducer](#54-declarative-reducers-with-reducer)
   5. [Snapshots and restore](#55-snapshots-and-restore)
   6. [SlicePolicy: state vs logs](#56-slicepolicy-state-vs-logs)
6. [Adapters](#6-adapters)
   1. [ProviderAdapter.evaluate](#61-provideradapterevaluate)
   2. [OpenAIAdapter](#62-openaiadapter)
   3. [LiteLLMAdapter](#63-litellmadapter)
   4. [Claude Agent SDK adapter](#64-claude-agent-sdk-adapter)
7. [Orchestration with MainLoop](#7-orchestration-with-mainloop)
   1. [The minimal MainLoop](#71-the-minimal-mainloop)
   2. [Deadlines and budgets](#72-deadlines-and-budgets)
8. [Progressive disclosure](#8-progressive-disclosure)
   1. [SectionVisibility: FULL vs SUMMARY](#81-sectionvisibility-full-vs-summary)
   2. [open_sections and read_section](#82-open_sections-and-read_section)
   3. [Visibility overrides in session state](#83-visibility-overrides-in-session-state)
9. [Prompt overrides and optimization](#9-prompt-overrides-and-optimization)
   1. [Hash-based safety: override only what you intended](#91-hash-based-safety-override-only-what-you-intended)
   2. [LocalPromptOverridesStore](#92-localpromptoverridesstore)
   3. [Override file format](#93-override-file-format)
   4. [A practical override workflow](#94-a-practical-override-workflow)
10. [Workspace tools](#10-workspace-tools)
    1. [PlanningToolsSection](#101-planningtoolssection)
    2. [VfsToolsSection](#102-vfstoolssection)
    3. [WorkspaceDigestSection](#103-workspacedigestsection)
    4. [AstevalSection](#104-astevalsection)
    5. [PodmanSandboxSection](#105-podmansandboxsection)
    6. [Wiring a workspace into a prompt](#106-wiring-a-workspace-into-a-prompt)
11. [Debugging and observability](#11-debugging-and-observability)
    1. [Structured logging](#111-structured-logging)
    2. [Session events](#112-session-events)
    3. [Dumping snapshots to JSONL](#113-dumping-snapshots-to-jsonl)
    4. [The debug UI](#114-the-debug-ui)
12. [Testing and reliability](#12-testing-and-reliability)
13. [Recipes](#13-recipes)
    1. [A code-review agent](#131-a-code-review-agent)
    2. [A repo Q&A agent](#132-a-repo-qa-agent)
    3. [A "safe patch" agent](#133-a-safe-patch-agent)
    4. [A research agent with progressive disclosure](#134-a-research-agent-with-progressive-disclosure)
14. [Troubleshooting](#14-troubleshooting)
15. [API reference](#15-api-reference)
    1. [Top-level exports](#151-top-level-exports)
    2. [weakincentives.prompt](#152-weakincentivesprompt)
    3. [weakincentives.runtime](#153-weakincentivesruntime)
    4. [weakincentives.adapters](#154-weakincentivesadapters)
    5. [weakincentives.contrib.tools](#155-weakincentivescontribtools)
    6. [weakincentives.optimizers](#156-weakincentivesoptimizers)
    7. [weakincentives.serde](#157-weakincentivesserde)
    8. [CLI](#158-cli)

---

## 1. Philosophy

### 1.1 What "weak incentives" means

"Weak incentives" is an engineering stance:

> Build agent systems where well-constructed prompts and tools create weak
> incentives for the model to do the right thing and stay on task.

The name comes from mechanism design: a system with the right incentives is one
where participants naturally gravitate toward intended behavior. Applied to
agents, this means shaping the prompt, tools, and context so the model's
easiest path is also the correct one.

This isn't about constraining the model or managing downside risk. It's about
*encouraging* correct behavior through structure:

- **Clear instructions co-located with tools** make the right action obvious
- **Typed contracts** guide the model toward valid outputs
- **Progressive disclosure** keeps the model focused on what matters now
- **Explicit state** gives the model the context it needs to make good decisions

The optimization process strengthens these incentives. When you refine a prompt
override or add a tool example, you're making the correct path even more
natural. Over iterations, the system becomes increasingly well-tuned—not
through constraints, but through clarity.

Concretely, WINK pushes you toward:

**Explicit side effects**

Side effects live in tool handlers. Everything else—prompt rendering, state
transitions, reducers—is deterministic and pure. When something goes wrong, you
know exactly where to look: the tool handler that executed.

**Typed contracts everywhere**

Params, tool calls, tool results, structured outputs, session state—all typed
with dataclasses. Type mismatches surface at construction time, not at runtime
when the model is mid-response. Pyright strict mode is enforced; the type
checker is your first line of defense.

**Inspectability**

If a run went wrong, you can inspect exactly what was rendered and what tools
ran. Sessions record every event as an immutable ledger. Snapshots let you
capture state at any point and restore it later.

**Controlled context growth**

Progressive disclosure lets you default to summaries and expand on demand.
Instead of stuffing everything into the prompt upfront, you let the model
request what it needs. This keeps token counts manageable and models focused.

**Safe iteration**

Hash-validated prompt overrides prevent accidental drift between "tested" and
"running". When you override a section's text, the system validates that you're
overriding the version you think you're overriding.

The goal isn't to constrain the model—it's to give it the best possible
starting point. When prompts are clear, tools are well-documented, and state is
explicit, the model has strong signals about what to do. When something goes
wrong, you can see exactly what happened and refine the incentives for next
time.

### 1.2 The shift: orchestration shrinks, context engineering grows

Many early "agent frameworks" assumed the hard part would be workflow logic:
routers, planners, branching graphs, and elaborate loops. These frameworks
spent their complexity budget on orchestration—deciding which prompts to run
when, routing between specialized agents, managing elaborate state machines.

WINK makes a different bet:

**Models are steadily absorbing more of the reasoning loop.**

What required explicit multi-step orchestration yesterday often works in a
single prompt today. The frontier models are increasingly capable of planning,
reasoning, and self-correction within a single context window. Elaborate
routing graphs often just get in the way.

**The durable part of agent systems is tools, retrieval, and context
engineering.**

Tools define what the agent can do. Retrieval determines what information is
available. Context engineering—deciding what to include, what to summarize, how
to structure information so the model reasons well—is where the real leverage
lives.

Context engineering is the tricky part. It's a genuinely new discipline: what's
relevant now, what to summarize versus preserve, how to shape information so
models reason over it well. No clean precedent from traditional engineering.
Builders who master it early win.

WINK's core abstractions exist to make that discipline real:

- Prompts are structured, typed objects that you can inspect and test
- Tools are explicit contracts that surface what the model can do
- State is inspectable so you can debug failures
- Safety is enforced at tool boundaries where side effects happen

If you want the formal version of these behaviors, skim the specs:
[specs/PROMPTS.md](specs/PROMPTS.md), [specs/TOOLS.md](specs/TOOLS.md), [specs/SESSIONS.md](specs/SESSIONS.md), [specs/MAIN_LOOP.md](specs/MAIN_LOOP.md).

### 1.3 The core bet: prompts as first-class, typed programs

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

**A `PromptTemplate` is an immutable object graph (a tree of sections).**

Each section can:

- Render markdown instructions
- Declare placeholders backed by typed dataclasses
- Register tools next to the instructions that describe how to use them
- Optionally render as a summary to save tokens

The section that explains "here's how to search files" is the same section that
provides the `grep` tool. Documentation and capability live together. They
can't drift.

**A `Prompt` binds runtime configuration:**

- Parameter dataclasses that fill template placeholders
- Prompt overrides store and tag for safe iteration
- Optionally a session for dynamic visibility and scoping

**A `ProviderAdapter` evaluates the prompt:**

- Renders markdown from the section tree
- Executes tool calls synchronously
- Returns text and/or parsed structured output

**A `Session` captures everything as an event-driven, reducer-managed state
log.**

Every prompt render, every tool invocation, every state change is recorded. You
can query the session, snapshot it, restore it.

In other words: **your agent is a typed prompt + tools + state.**

Two "novel" properties fall out of this structure:

1. **Deterministic inspection**: render, snapshot, and diff prompts. The same
   inputs produce the same outputs. You can write tests that assert on exact
   prompt text.

2. **Safe iteration**: apply prompt tweaks via overrides that are validated
   against hashes. When you change a section in code, existing overrides stop
   applying until you explicitly update them. No silent drift.

### 1.4 What WINK is (and is not)

**WINK is:**

- A Python library (`weakincentives`) for building prompts-as-agents.
- A small runtime for state (`Session`) and orchestration (`MainLoop`).
- Adapters (`OpenAI`, `LiteLLM`, `Claude Agent SDK`) that execute tools and
  parse outputs consistently.
- Contributed tool suites for background agents (planning, virtual FS,
  sandboxes, workspace digests).

**WINK is not:**

- A distributed workflow engine. If you need to coordinate across machines or
  manage long-running jobs, use something built for that.
- A framework that tries to "own" your application architecture. WINK is a
  library. Use it for the pieces that benefit from determinism.
- A multi-agent coordination system. WINK focuses on single-agent patterns done
  well. Multi-agent is possible but not the primary design target.
- An async-first streaming framework. Today the adapter contract is
  synchronous. Streaming may come later.

If you need a graph engine or multi-agent coordination, you can still use WINK
for the pieces that benefit from determinism (prompt design, tool contracts,
state snapshots) and let something else coordinate the rest. WINK plays well
with others.

---

## 2. Quickstart

### 2.1 Install

The core package has no mandatory third-party dependencies. This is
intentional: you should be able to use WINK's prompt and session primitives
without pulling in OpenAI or any other provider SDK.

```bash
pip install weakincentives
```

Provider and tool extras (pick what you need):

```bash
pip install "weakincentives[openai]"           # OpenAI adapter
pip install "weakincentives[litellm]"          # LiteLLM adapter
pip install "weakincentives[claude-agent-sdk]" # Claude Agent SDK adapter

pip install "weakincentives[asteval]"          # Safe Python expression eval tool
pip install "weakincentives[podman]"           # Podman sandbox tools
pip install "weakincentives[wink]"             # Debug UI (wink CLI)
```

**Python requirement**: 3.12+. We use modern Python features liberally and
don't maintain compatibility with older versions.

### 2.2 End-to-end: a tiny structured agent

This example is intentionally small but complete:

- Typed params (`SummarizeRequest`)
- Structured output (`Summary`)
- Deterministic prompt structure (`PromptTemplate`)
- A session to record telemetry

```python
from dataclasses import dataclass

from weakincentives.prompt import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import InProcessEventBus, Session

@dataclass(slots=True, frozen=True)
class SummarizeRequest:
    text: str

@dataclass(slots=True, frozen=True)
class Summary:
    title: str
    bullets: tuple[str, ...]

template = PromptTemplate[Summary](
    ns="docs",
    key="summarize",
    name="Doc summarizer",
    sections=(
        MarkdownSection(
            title="Task",
            key="task",
            template=(
                "Summarize the input.\n\n"
                "Return JSON with:\n"
                "- title: short title\n"
                "- bullets: 3-7 bullet points\n"
            ),
        ),
        MarkdownSection(
            title="Input",
            key="input",
            template="${text}",
        ),
    ),
)

prompt = Prompt(template).bind(SummarizeRequest(
    text="WINK is a Python library for building agents. It treats prompts as "
         "typed programs. Tools are explicit. State is inspectable."
))

bus = InProcessEventBus()
session = Session(bus=bus)

# To actually run this, you need an adapter and API key:
#
# from weakincentives.adapters.openai import OpenAIAdapter
# adapter = OpenAIAdapter(model="gpt-4o-mini")
# response = adapter.evaluate(prompt, session=session)
# print(response.output)
# # -> Summary(title='WINK Overview', bullets=('Python library...', ...))
```

**Notes:**

- Binding is by dataclass type. `bind(SummarizeRequest(...))` sets `${text}`.
- `PromptTemplate[Summary]` declares structured output. Adapters parse the
  model's JSON response into your dataclass automatically.
- The `slots=True, frozen=True` pattern is used everywhere. Immutable
  dataclasses prevent accidental mutation and work well with the session's
  event-driven model.

### 2.3 Add a tool

Tools are registered by sections. Handlers receive immutable `ToolContext`
(including session + resources) and return a `ToolResult`.

Here's a toy tool that returns the current time. In a real agent this would be
a filesystem read, an API call, or some other operation with side effects.

```python
from dataclasses import dataclass
from datetime import UTC, datetime

from weakincentives.prompt import Tool, ToolContext, ToolResult, MarkdownSection, ToolExample

@dataclass(slots=True, frozen=True)
class NowParams:
    tz: str = "UTC"

@dataclass(slots=True, frozen=True)
class NowResult:
    iso: str

    def render(self) -> str:
        return self.iso

def now_handler(params: NowParams, *, context: ToolContext) -> ToolResult[NowResult]:
    del context  # not used here
    if params.tz != "UTC":
        return ToolResult(
            message="Only UTC supported in this demo.",
            value=None,
            success=False,
        )
    return ToolResult(
        message="Current time.",
        value=NowResult(iso=datetime.now(UTC).isoformat()),
    )

now_tool = Tool[NowParams, NowResult](
    name="now",
    description="Return the current time (UTC).",
    handler=now_handler,
    examples=(
        ToolExample(
            description="Get UTC time",
            input=NowParams(tz="UTC"),
            output=NowResult(iso="2025-01-01T00:00:00+00:00"),
        ),
    ),
)

tools_section = MarkdownSection(
    title="Tools",
    key="tools",
    template="You may call tools when needed.",
    tools=(now_tool,),
)
```

Once this section is included in your template, adapters will advertise `now`
to the model and execute calls synchronously.

The pattern to notice: **tools and their documentation live together**. The
section says "You may call tools when needed" and provides the `now` tool. The
model sees both in the same context.

### 2.4 Your first complete agent (copy-paste ready)

Here's a minimal but complete agent you can run. It answers questions about a
topic using a search tool. Copy this into a file and run it.

```python
"""Minimal WINK agent: a topic Q&A bot with a mock search tool."""
import os
from dataclasses import dataclass
from weakincentives.prompt import (
    Prompt, PromptTemplate, MarkdownSection, Tool, ToolContext, ToolResult
)
from weakincentives.runtime import Session
from weakincentives.adapters.openai import OpenAIAdapter

# 1. Define structured output
@dataclass(slots=True, frozen=True)
class Answer:
    summary: str
    sources: tuple[str, ...]

# 2. Define a tool
@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str

@dataclass(slots=True, frozen=True)
class SearchResult:
    snippets: tuple[str, ...]
    def render(self) -> str:
        return "\n".join(f"- {s}" for s in self.snippets)

def search_handler(params: SearchParams, *, context: ToolContext) -> ToolResult[SearchResult]:
    # In a real agent, this would call a search API
    del context
    return ToolResult(
        message=f"Found results for: {params.query}",
        value=SearchResult(snippets=(
            f"Result 1 about {params.query}",
            f"Result 2 about {params.query}",
        )),
    )

search_tool = Tool[SearchParams, SearchResult](
    name="search",
    description="Search for information about a topic.",
    handler=search_handler,
)

# 3. Define params
@dataclass(slots=True, frozen=True)
class QuestionParams:
    question: str

# 4. Build the prompt template
template = PromptTemplate[Answer](
    ns="demo",
    key="qa-agent",
    sections=(
        MarkdownSection(
            title="Instructions",
            key="instructions",
            template=(
                "You are a helpful research assistant.\n\n"
                "Use the search tool to find information, then answer the "
                "question with a summary and list of sources."
            ),
            tools=(search_tool,),
        ),
        MarkdownSection(
            title="Question",
            key="question",
            template="${question}",
        ),
    ),
)

# 5. Run the agent
def main():
    session = Session()
    adapter = OpenAIAdapter(model="gpt-4o-mini")

    prompt = Prompt(template).bind(QuestionParams(
        question="What is the capital of France?"
    ))

    response = adapter.evaluate(prompt, session=session)

    print(f"Summary: {response.output.summary}")
    print(f"Sources: {response.output.sources}")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example")
    else:
        main()
```

**What's happening:**

1. `PromptTemplate[Answer]` declares that the model must return JSON matching
   the `Answer` dataclass
2. The `search` tool is registered on the Instructions section—the model sees
   the tool alongside the instructions for using it
3. `adapter.evaluate()` sends the prompt to OpenAI, executes any tool calls,
   and parses the structured response
4. You get back `response.output` as a typed `Answer` instance

This is the core loop. Everything else in WINK builds on this: sessions for
state, sections for organization, progressive disclosure for token management.

---

## 3. Prompts

*Canonical spec: [specs/PROMPTS.md](specs/PROMPTS.md)*

The prompt system is the heart of WINK. It is intentionally **not** a generic
templating engine. The design goal is predictability:

- Prompt text is deterministic
- Placeholder names are validated at construction time
- Composition is explicit (no magic string concatenation)
- Overrides are safe (hash-validated)

### 3.1 PromptTemplate

`PromptTemplate[OutputT]` is the immutable definition of an agent prompt.

**Key properties:**

- `ns` (namespace) and `key` uniquely identify a prompt family
- `name` is for human readability and logging
- `sections` is a tree (technically a forest) of `Section` objects
- Optional structured output is declared by the type parameter `OutputT`

```python
from dataclasses import dataclass
from weakincentives.prompt import PromptTemplate, MarkdownSection

@dataclass(slots=True, frozen=True)
class Params:
    question: str

template = PromptTemplate(
    ns="support",
    key="faq",
    sections=(
        MarkdownSection(
            title="Instruction",
            key="instruction",
            template="Answer clearly and briefly.",
        ),
        MarkdownSection(
            title="Question",
            key="question",
            template="Question: ${question}",
        ),
    ),
)
```

**Validation rules** (you'll hit these early, and that's the point):

- Section keys must be stable identifiers (lowercase alphanumerics + `-`)
- Placeholders must match fields on a dataclass bound at render time
- Binding is by dataclass type: you can't bind two instances of the same
  dataclass type

These constraints exist to catch errors early. A typo in a placeholder name
fails at construction, not when the model is mid-response.

### 3.2 Prompt

`Prompt[OutputT]` is the runtime binding:

```python
Prompt(template, overrides_store=..., overrides_tag=...)
prompt.bind(Params(...), OtherParams(...))
prompt.render(session=...)
```

Rendering returns a `RenderedPrompt[OutputT]` which contains:

- `text`: final markdown
- `tools`: tools contributed by enabled sections (in traversal order)
- `structured_output`: schema config when declared
- `descriptor`: a hash-based descriptor used by the overrides system

```python
prompt = Prompt(template).bind(Params(question="What is WINK?"))
rendered = prompt.render(session=session)
print(rendered.text)
print([t.name for t in rendered.tools])
```

The distinction between `PromptTemplate` and `Prompt` matters: templates are
reusable and immutable; prompts are bound to specific parameters and can carry
override configuration.

### 3.3 Sections

A `Section[ParamsT]` is a node in the prompt tree.

Every section has:

- `title`: used to render markdown headings
- `key`: stable ID within the prompt tree
- Optional `children`: subsections
- Optional `tools`: tool contracts to expose
- Optional `enabled`: a predicate that can disable the section at render time
- Optional `summary` + `visibility`: for progressive disclosure
- `accepts_overrides`: whether the override system may replace its body

A section must implement `render_body(...)`. Many sections use `MarkdownSection`,
but contributed tool suites are also sections (planning, VFS, sandboxes, etc.).

### 3.4 MarkdownSection

`MarkdownSection` is the workhorse: it renders a `string.Template` with
`${name}` placeholders.

```python
from dataclasses import dataclass
from weakincentives.prompt import MarkdownSection

@dataclass(slots=True, frozen=True)
class User:
    name: str
    plan: str

profile = MarkdownSection(
    title="User",
    key="user",
    template="Name: ${name}\nPlan: ${plan}",
)
```

**Why `string.Template`?** It's deliberately simple:

- No expressions
- No loops
- No conditionals

Complex formatting belongs in your Python code (where it can be tested). If you
find yourself wanting loops in your template, that's a sign you should compute
the string in Python and pass it as a param.

### 3.5 Structured output

Structured output is declared by parameterizing the prompt template:

- `PromptTemplate[OutputDataclass]` → output is a dataclass instance
- `PromptTemplate[list[OutputDataclass]]` → output is a list of dataclass
  instances

Adapters will:

1. Instruct the model to return JSON for the schema
2. Parse the response into your dataclass type
3. Return it as `PromptResponse.output`

If you need to parse output yourself (or you're using a custom adapter), use
`parse_structured_output(...)`:

```python
from weakincentives.prompt import parse_structured_output

rendered = prompt.render(session=session)
output = parse_structured_output(response_text, rendered)
```

When `OutputT` is a list, the parser accepts either:

- A JSON array, or
- An object wrapper of the form `{"items": [...]}` (ARRAY_WRAPPER_KEY is
  `"items"`)

### 3.6 Dynamic scoping with enabled()

Sections can be turned on/off at render time using `enabled`.

Supported signatures:

- `() -> bool`
- `(*, session) -> bool`
- `(params) -> bool`
- `(params, *, session) -> bool`

This is one of WINK's most powerful context tools: you can build a large prompt
template, then render only what's relevant.

**Example**: include "deep debugging instructions" only when a session flag is
set.

```python
from dataclasses import dataclass
from weakincentives.prompt import MarkdownSection
from weakincentives.runtime import Session

@dataclass(slots=True, frozen=True)
class DebugFlag:
    enabled: bool

def debug_enabled(flag: DebugFlag, *, session: Session) -> bool:
    del session
    return flag.enabled

debug_section = MarkdownSection(
    title="Debug",
    key="debug",
    template="If something fails, include stack traces and hypotheses.",
    enabled=debug_enabled,
)
```

Disabled sections don't just hide their text—their tools also disappear from
the prompt. This lets you build a comprehensive template and enable only the
capabilities relevant to the current context.

### 3.7 Session-bound sections and cloning

Some sections are **pure**: they depend only on params and render the same text
every time. You can safely store those in a module-level `PromptTemplate`.

Other sections are **session-bound**: they capture runtime resources (a
session, filesystem, sandbox connection, etc.). Examples:

- `PlanningToolsSection(session=...)`
- `VfsToolsSection(session=...)`
- `WorkspaceDigestSection(session=...)`
- `PodmanSandboxSection(session=...)`

For those, prefer one of these patterns:

**Pattern A: build the template per session**

```python
def build_prompt_template(*, session: Session) -> PromptTemplate[OutputT]:
    return PromptTemplate(
        ns="...",
        key="...",
        sections=(
            MarkdownSection(...),
            PlanningToolsSection(session=session),
            VfsToolsSection(session=session, ...),
        ),
    )
```

**Pattern B: keep a mostly-static template, and clone the session-bound
sections**

Sections support `clone(**overrides)` to create a new instance with updated
fields. This lets you reuse "static" pieces but still pass a fresh session each
run.

This sounds minor, but it prevents a common bug: accidentally sharing a tool
section (and its internal state) across multiple sessions. Each session should
get its own tool sections.

### 3.8 Few-shot traces with TaskExamplesSection

WINK also supports few-shot examples as first-class sections via
`TaskExamplesSection`.

**Why this matters**: examples are often more effective than "more
instructions", and keeping them as typed objects makes them easier to maintain
and override.

A `TaskExample` can include:

- Input params (dataclasses)
- Expected output (structured)
- Optional tool call traces

This is especially useful for tools: you can show correct tool usage once, and
many models generalize better from examples than from abstract instructions.

See `weakincentives.prompt.task_examples` for details.

---

## 4. Tools

*Canonical spec: [specs/TOOLS.md](specs/TOOLS.md)*

The tool system is designed around one hard rule:

> **Tool handlers are the only place where side effects happen.**

Everything else (prompt rendering, state transitions, reducers) is meant to be
pure and deterministic. This constraint has a purpose: when something goes
wrong, you know exactly where to look.

### 4.1 Tool contracts

A tool is defined by:

- `name`: `^[a-z0-9_-]{1,64}$`
- `description`: short model-facing string
- `params_type`: a dataclass type (or `None`)
- `result_type`: a dataclass type (or `None`)
- `handler(params, *, context) -> ToolResult[result_type]`

**Skeleton:**

```python
from dataclasses import dataclass
from weakincentives.prompt import Tool, ToolContext, ToolResult

@dataclass(slots=True, frozen=True)
class MyParams:
    query: str

@dataclass(slots=True, frozen=True)
class MyResult:
    answer: str

    def render(self) -> str:
        return self.answer

def handler(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
    # Do work here (read files, call an API, etc.)
    return ToolResult(message="ok", value=MyResult(answer="42"))

tool = Tool[MyParams, MyResult](
    name="my_tool",
    description="Do a thing.",
    handler=handler,
)
```

The type parameters matter. `Tool[MyParams, MyResult]` tells WINK how to
serialize parameters for the model and how to parse results. Type mismatches
are caught at construction time.

### 4.2 ToolContext, resources, and Filesystem

`ToolContext` provides access to execution-time state:

- `context.prompt`: the `Prompt` being executed
- `context.rendered_prompt`: the `RenderedPrompt` (when available)
- `context.adapter`: the adapter executing tools
- `context.session`: the current session
- `context.deadline`: optional wall-clock deadline
- `context.budget_tracker`: optional token budget tracker
- `context.resources`: a typed `ResourceRegistry`

**A key idea**: you can pass your own resources (HTTP clients, DB handles,
tracers) without adding new fields to the core dataclass.

```python
from weakincentives.prompt import ResourceRegistry

registry = ResourceRegistry.build({
    MyHttpClient: MyHttpClient(...),
})
```

For workspace agents, the most common resource is a `Filesystem` implementation.
Many contributed tool suites install one automatically (VFS, Podman).

### 4.3 ToolResult semantics

Tool handlers return `ToolResult`:

```python
ToolResult(
    message="Human-readable status",
    value=...,                        # dataclass | mapping | sequence | str | None
    success=True,                     # if False, adapters treat as tool failure
    exclude_value_from_context=False, # hide large payloads from model context
)
```

**Key behaviors:**

- If `value` is a dataclass and implements `render() -> str`, adapters use that
  as the textual tool output. This lets you control exactly what the model
  sees.
- If `render()` is missing, WINK logs a warning and serializes the dataclass to
  JSON. This works but is less controlled.
- Exceptions raised by handlers are caught and converted into tool failures
  (with some safety exceptions; see [specs/TOOLS.md](specs/TOOLS.md)).

The `exclude_value_from_context=True` flag is useful for tools that return
large payloads (like file contents). The model sees a summary message, but the
full value is recorded in the session for debugging.

### 4.4 Tool examples

Tools can provide `ToolExample` entries for better model performance and better
debugging.

```python
from weakincentives.prompt import ToolExample

tool = Tool[NowParams, NowResult](
    name="now",
    description="Return UTC time.",
    handler=now_handler,
    examples=(
        ToolExample(
            description="Get UTC time",
            input=NowParams(tz="UTC"),
            output=NowResult(iso="2025-01-01T00:00:00+00:00"),
        ),
    ),
)
```

If you've ever seen a tool-capable model "almost" do the right call—wrong
parameter name, wrong format, etc.—examples tend to pay for themselves. One
good example often beats three paragraphs of instructions.

### 4.5 Tool suites as sections

In WINK, "a tool suite" is usually a section:

- It adds instructions explaining when to use the tools
- It attaches tool contracts
- It often owns some session slice(s)

That co-location is intentional: **tools without guidance are unreliable, and
guidance without tools is toothless**.

Examples in `weakincentives.contrib.tools`:

- Planning tools (`PlanningToolsSection`)
- VFS tools (`VfsToolsSection`)
- Sandbox tools (`PodmanSandboxSection`, `AstevalSection`)

Each section bundles the instructions ("here's how to use these tools") with
the tools themselves. The model sees them together.

---

## 5. Sessions

*Canonical spec: [specs/SESSIONS.md](specs/SESSIONS.md)*

A `Session` is WINK's answer to "agent memory", with a constraint:

> **Memory must be deterministic and inspectable.**

Instead of "a magic dict" you mutate, sessions store typed slices managed by
pure reducers. Every mutation flows through a reducer, and every change is
recorded as an event.

### 5.1 Session as deterministic memory

A session is a container keyed by dataclass type:

- Each type has a **slice**: `tuple[T, ...]`
- **Reducers** update slices in response to events
- The session subscribes to the event bus and records prompt/tool telemetry

Mental model: **"events in, new immutable slices out"**.

The session never mutates in place. Reducers return new tuples. This makes
snapshots trivial (just serialize the current tuples) and restoration
straightforward.

### 5.2 Queries

Use the slice accessor:

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class Fact:
    key: str
    value: str

facts: tuple[Fact, ...] = session[Fact].all()
latest_fact: Fact | None = session[Fact].latest()
selected: tuple[Fact, ...] = session[Fact].where(lambda f: f.key == "repo_root")
```

The slice accessor `session[T]` gives you a `QueryBuilder` with fluent methods.
Queries are read-only; they never mutate the session.

### 5.3 Reducers

A **reducer** is a pure function that takes the current state and an event,
and returns the new state. The name comes from functional programming (and was
popularized by Redux in frontend development), but the concept is simple:

```
new_state = reducer(current_state, event)
```

Reducers never mutate state directly. They always return a new value. This
makes state changes predictable: given the same inputs, you always get the same
output. It also makes debugging easier—you can log every event and trace the
exact sequence that led to any state.

In WINK, reducers have this signature: `(slice_values, event) -> new_slice_values`.

WINK ships helper reducers:

- `append_all`: append the event to the slice
- `replace_latest`: replace the most recent value
- `replace_latest_by`: replace by key
- `upsert_by`: insert or update by key

**Example**: keep only the latest plan:

```python
from dataclasses import dataclass
from weakincentives.runtime import replace_latest

@dataclass(slots=True, frozen=True)
class Plan:
    steps: tuple[str, ...]

session[Plan].register(Plan, replace_latest)
session.broadcast(Plan(steps=("step 1",)))
session.broadcast(Plan(steps=("step 2",)))
assert session[Plan].all() == (Plan(steps=("step 2",)),)
```

### 5.4 Declarative reducers with @reducer

For complex slices, attach reducers as methods using `@reducer`:

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session.state_slice import reducer

@dataclass(slots=True, frozen=True)
class AddStep:
    step: str

@dataclass(slots=True, frozen=True)
class AgentPlan:
    steps: tuple[str, ...]

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> "AgentPlan":
        return replace(self, steps=(*self.steps, event.step))

session.install(AgentPlan, initial=lambda: AgentPlan(steps=()))
session.broadcast(AddStep(step="read README"))
session.broadcast(AddStep(step="run tests"))
```

This pattern keeps reducer logic close to the data it operates on. The
`@reducer` decorator is just metadata; the actual reducer registration happens
in `session.install()`.

### 5.5 Snapshots and restore

Sessions can be snapshotted and restored:

```python
snapshot = session.snapshot()
# ... do work ...
session.restore(snapshot)
```

**Typical use cases:**

- Store a JSONL flight recorder for debugging
- Implement "rollback" on risky operations
- Attach snapshots to bug reports for reproduction

Snapshots serialize to JSON. You can persist them to disk and reload them
later. This is how the debug UI works: it reads snapshot files and displays the
session state at each point.

### 5.6 SlicePolicy: state vs logs

Not all slices should roll back the same way.

WINK distinguishes between:

- `SlicePolicy.STATE`: working state that should be restored on rollback
- `SlicePolicy.LOG`: append-only history that should be preserved

By default, `session.snapshot()` captures only `STATE` slices.

If you want everything (including logs), use:

```python
snapshot = session.snapshot(include_all=True)
```

This distinction matters for debugging. You often want to preserve the full
event log even when rolling back working state.

---

## 6. Adapters

*Canonical spec: [specs/ADAPTERS.md](specs/ADAPTERS.md)*

Adapters bridge a prompt to a provider and enforce consistent semantics:

- Render prompt markdown
- Expose tools
- Execute tool calls synchronously
- Parse structured output when declared

### 6.1 ProviderAdapter.evaluate

All adapters implement:

```python
response = adapter.evaluate(
    prompt,
    session=session,
    deadline=...,        # optional
    budget=...,          # optional
    budget_tracker=...,  # optional
)
```

It returns `PromptResponse[OutputT]`:

- `prompt_name`: string
- `text`: raw assistant text
- `output`: parsed structured output (or `None`)

The adapter handles all the provider-specific details: API formatting, tool
schema translation, response parsing. Your code just calls `evaluate()` and
gets back typed results.

### 6.2 OpenAIAdapter

**Install:** `pip install "weakincentives[openai]"`

**Key configs:**

- `OpenAIClientConfig(api_key=..., base_url=..., timeout=..., max_retries=...)`
- `OpenAIModelConfig(temperature=..., max_tokens=..., top_p=..., ...)`

**Example:**

```python
from weakincentives.adapters.openai import OpenAIAdapter
from weakincentives.adapters import OpenAIClientConfig, OpenAIModelConfig

adapter = OpenAIAdapter(
    model="gpt-4.1-mini",
    client_config=OpenAIClientConfig(),
    model_config=OpenAIModelConfig(max_tokens=800),
)
response = adapter.evaluate(prompt, session=session)
```

The adapter uses OpenAI's native JSON schema response format for structured
output. It handles tool calls synchronously, executing each tool and feeding
results back to the model.

### 6.3 LiteLLMAdapter

**Install:** `pip install "weakincentives[litellm]"`

LiteLLM provides a unified interface to many providers. Use this when you want
to switch between providers without changing code.

```python
from weakincentives.adapters.litellm import LiteLLMAdapter
from weakincentives.adapters import LiteLLMClientConfig, LiteLLMModelConfig

adapter = LiteLLMAdapter(
    model="openai/gpt-4.1-mini",
    client_config=LiteLLMClientConfig(),
    model_config=LiteLLMModelConfig(max_tokens=800),
)
```

### 6.4 Claude Agent SDK adapter

**Install:** `pip install "weakincentives[claude-agent-sdk]"`

The Claude Agent SDK adapter provides Claude's full agentic capabilities. It
runs Claude Code as a subprocess with native tools (Read, Write, Bash, Glob,
Grep) rather than emulating them.

This adapter is different from OpenAI/LiteLLM: instead of WINK executing tools
itself, it delegates to Claude Code's tool execution. This gives you Claude's
native tooling with WINK's prompt composition and session management.

See [specs/CLAUDE_AGENT_SDK.md](specs/CLAUDE_AGENT_SDK.md) for full configuration reference and isolation
guarantees.

---

## 7. Orchestration with MainLoop

*Canonical spec: [specs/MAIN_LOOP.md](specs/MAIN_LOOP.md)*

`MainLoop` exists for one reason:

> Make progressive disclosure and budgets/deadlines easy to handle correctly.

You could write the loop yourself. MainLoop just does it in a tested,
consistent way.

### 7.1 The minimal MainLoop

You implement:

- `create_prompt(request) -> Prompt[OutputT]`
- `create_session() -> Session`

Then call `loop.execute(request)`.

```python
from weakincentives.runtime import MainLoop, Session
from weakincentives.prompt import Prompt

class MyLoop(MainLoop[RequestType, OutputType]):
    def create_prompt(self, request: RequestType) -> Prompt[OutputType]:
        return Prompt(self._template).bind(request)

    def create_session(self) -> Session:
        return Session(bus=self._bus)
```

`MainLoop` also catches `VisibilityExpansionRequired` and retries
automatically. When the model calls `open_sections`, MainLoop applies the
visibility overrides and re-evaluates the prompt. You don't have to handle this
yourself.

### 7.2 Deadlines and budgets

`Deadline` is a wall-clock deadline. `Budget` can include token limits and/or a
deadline. `BudgetTracker` accumulates usage across retries.

**Typical pattern:**

```python
from datetime import timedelta
from weakincentives import Deadline, Budget

deadline = Deadline.from_timeout(timedelta(seconds=30))
budget = Budget(max_total_tokens=20_000)

response, session = loop.execute(request, deadline=deadline, budget=budget)
```

Deadlines prevent runaway agents. Budgets prevent runaway costs. Both are
enforced at the adapter level, so they work consistently across providers.

---

## 8. Progressive disclosure

*Canonical spec: [specs/PROMPTS.md](specs/PROMPTS.md) (Progressive Disclosure section)*

Long prompts are expensive. Progressive disclosure is WINK's first-class
solution:

- Sections can render as `SUMMARY` by default
- The model can request expansion via `open_sections`

This keeps initial prompts lean while giving the model access to details when
needed.

### 8.1 SectionVisibility: FULL vs SUMMARY

A section can have:

- `template`: full content
- `summary`: short content
- `visibility`: constant or callable

If visibility is `SUMMARY` and `summary` is present, WINK renders the summary
instead of the full template.

```python
from weakincentives.prompt import MarkdownSection, SectionVisibility

section = MarkdownSection(
    title="Reference",
    key="reference",
    template="Very long reference documentation...",
    summary="Reference documentation is available.",
    visibility=SectionVisibility.SUMMARY,
)
```

### 8.2 open_sections and read_section

When summarized sections exist, WINK injects builtin tools:

- `open_sections(section_keys, reason)` → raises `VisibilityExpansionRequired`
- `read_section(section_key)` → returns full rendered markdown for that section

`MainLoop` handles `open_sections` automatically by setting overrides and
retrying. The model asks to expand a section, MainLoop applies the expansion,
and evaluation continues with the full content visible.

`read_section` is different: it returns the content without changing
visibility. The section remains summarized in subsequent turns. Use this for
reference material that the model only needs temporarily.

### 8.3 Visibility overrides in session state

Visibility overrides live in the `VisibilityOverrides` session slice and are
applied at render time.

```python
from weakincentives.prompt import VisibilityOverrides, SetVisibilityOverride, SectionVisibility

session[VisibilityOverrides].apply(
    SetVisibilityOverride(path=("reference",), visibility=SectionVisibility.FULL)
)
```

This is what happens under the hood when the model calls `open_sections`.
MainLoop applies the override and re-renders the prompt.

---

## 9. Prompt overrides and optimization

*Canonical spec: [specs/PROMPT_OPTIMIZATION.md](specs/PROMPT_OPTIMIZATION.md)*

Overrides are how WINK supports fast iteration without code edits:

- Keep prompt templates stable in code
- Store patch files on disk
- Validate patches with hashes

This separation matters. Your templates are code: tested, reviewed, versioned.
Overrides are configuration: easy to tweak without a deploy.

### 9.1 Hash-based safety: override only what you intended

Overrides are validated against a `PromptDescriptor`:

- Each overridable section has a `content_hash`
- Each overridable tool has a `contract_hash`

If hashes don't match, WINK refuses to apply the override. This prevents a
common failure mode: you edit a section in code, but an old override still
applies, and you're running something different than you tested.

### 9.2 LocalPromptOverridesStore

The default store is `LocalPromptOverridesStore`, which writes JSON files
under:

```
.weakincentives/prompts/overrides/{ns}/{prompt_key}/{tag}.json
```

Wire it like:

```python
from weakincentives.prompt.overrides import LocalPromptOverridesStore
from weakincentives.prompt import Prompt

store = LocalPromptOverridesStore()
prompt = Prompt(template, overrides_store=store, overrides_tag="stable")
```

### 9.3 Override file format

The override JSON format is intentionally simple (and human editable):

```json
{
  "version": 1,
  "ns": "demo",
  "prompt_key": "welcome",
  "tag": "stable",
  "sections": {
    "system": {
      "expected_hash": "...",
      "body": "You are an assistant."
    }
  },
  "tools": {
    "search": {
      "expected_contract_hash": "...",
      "description": "Search the index.",
      "param_descriptions": {"query": "Keywords"}
    }
  }
}
```

**Notes:**

- `sections` keys are section paths encoded as strings. Single-level keys look
  like `"system"`, nested keys use dot notation.
- Tool overrides can patch the tool description and per-field param
  descriptions.
- Hashes prevent applying old overrides to changed prompts.

### 9.4 A practical override workflow

A workflow that works well in teams:

1. **Seed** override files from the current prompt
   (`store.seed(prompt, tag="v1")`)
2. **Run** your agent and collect failures / quality notes
3. **Edit** override sections directly (or generate them with an optimizer)
4. **Re-run** tests/evals
5. **Commit** override files alongside code

For "hardening", disable overrides on sensitive sections/tools with
`accepts_overrides=False`. This prevents accidental changes to
security-critical text.

---

## 10. Workspace tools

*Canonical spec: [specs/WORKSPACE.md](specs/WORKSPACE.md)*

WINK includes several tool suites aimed at background agents that need to
inspect and manipulate a repository safely. They live in
`weakincentives.contrib.tools`.

### 10.1 PlanningToolsSection

**Tools:**

- `planning_setup_plan`
- `planning_add_step`
- `planning_update_step`
- `planning_read_plan`

The plan is stored in session state and updated via reducers. Each step has an
ID, title, details, and status.

Use it when you want the model to externalize its plan without inventing its
own format. Many models plan better when they have explicit tools for planning.

### 10.2 VfsToolsSection

A copy-on-write virtual filesystem with tools:

- `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `rm`

You can mount host directories into the VFS snapshot via `HostMount`. The VFS
copies files into memory; writes go to the copy, not the host. This is the
default "repo agent" workspace because it avoids accidental host writes.

```python
from weakincentives.contrib.tools import VfsToolsSection, VfsConfig, HostMount

vfs = VfsToolsSection(
    session=session,
    config=VfsConfig(
        mounts=(
            HostMount(
                host_path="src",
                mount_path=None,  # mount at /src
                include_glob=("*.py",),
                exclude_glob=("__pycache__/*",),
            ),
        ),
        allowed_host_roots=(".",),
    ),
)
```

### 10.3 WorkspaceDigestSection

Renders a cached repo digest stored in session state. The digest is a
structured summary of the repository: file tree, key files, detected patterns.

It works well with progressive disclosure: default to `SUMMARY`, expand on
demand. The model gets an overview without the full file contents.

### 10.4 AstevalSection

Exposes `evaluate_python` (safe-ish expression evaluation) with captured
stdout/stderr.

`asteval` restricts what Python code can do: no imports, no file access, no
network. Useful for small transformations (string formatting, arithmetic)
without granting shell access.

**Install:** `pip install "weakincentives[asteval]"`

### 10.5 PodmanSandboxSection

Runs shell commands and Python evaluation inside a Podman container.

Use it when you need strict isolation and reproducible execution (tests,
linters). The container provides a clean environment; writes don't affect the
host.

**Install:** `pip install "weakincentives[podman]"`

### 10.6 Wiring a workspace into a prompt

A practical pattern (also used by `code_reviewer_example.py` in this repo):

```python
from weakincentives.contrib.tools import (
    PlanningToolsSection,
    PlanningStrategy,
    VfsToolsSection,
    VfsConfig,
    HostMount,
    WorkspaceDigestSection,
)
from weakincentives.prompt import PromptTemplate, MarkdownSection
from weakincentives.runtime import Session

def build_repo_agent_template(*, session: Session):
    mounts = (
        HostMount(host_path="src"),
        HostMount(host_path="README.md"),
    )
    vfs = VfsToolsSection(
        session=session,
        config=VfsConfig(
            mounts=mounts,
            allowed_host_roots=(".",),
        ),
        accepts_overrides=True,
    )

    return PromptTemplate(
        ns="examples",
        key="repo-agent",
        sections=(
            MarkdownSection(
                title="Task",
                key="task",
                template="Answer questions about the repo.",
            ),
            WorkspaceDigestSection(session=session),
            PlanningToolsSection(
                session=session,
                strategy=PlanningStrategy.REACT,
            ),
            vfs,
        ),
    )
```

**The important idea**: the workspace sections are built with the active
session. Each run gets its own session with its own tool sections.

---

## 11. Debugging and observability

### 11.1 Structured logging

```python
from weakincentives.runtime import configure_logging, get_logger

configure_logging(level="INFO", json_mode=True)
logger = get_logger(__name__)
logger.info("hello", event="demo.hello", context={"foo": "bar"})
```

Logs include structured `event` and `context` fields for downstream routing
and analysis. JSON mode makes logs machine-parseable.

### 11.2 Session events

Sessions subscribe to the event bus and capture telemetry events like:

- `PromptRendered`: emitted when a prompt is rendered
- `ToolInvoked`: emitted when a tool is called (includes params, result, timing)
- `PromptExecuted`: emitted when a prompt evaluation completes (includes token
  usage)
- `TokenUsage`: token counts from provider responses

You can use these for your own tracing pipeline. Subscribe to the bus and route
events wherever you need them.

### 11.3 Dumping snapshots to JSONL

Use `weakincentives.debug.dump_session(...)` to persist a session tree:

```python
from weakincentives.debug import dump_session

path = dump_session(session, target="snapshots/")  # writes <session_id>.jsonl
```

Each line is one serialized session snapshot (root → leaves). The JSONL format
is stable and human-readable.

### 11.4 The debug UI

**Install:** `pip install "weakincentives[wink]"`

**Run:**

```bash
wink debug snapshots/<session_id>.jsonl
```

This starts a local server that renders the prompt/tool timeline for
inspection. You can see exactly what was sent to the model, what tools were
called, and how state evolved.

---

## 12. Testing and reliability

WINK is designed so that most of your "agent logic" is testable without a
model.

**Practical approach:**

1. **Prompt rendering tests**: render prompts with fixed params and assert
   exact markdown (snapshot tests). These run fast and catch template regressions.

2. **Tool handler tests**: call handlers directly with fake `ToolContext` +
   resources. No model needed. Test the business logic in isolation.

3. **Reducer tests**: test state transitions as pure functions. Given this
   slice and this event, expect this new slice.

4. **Integration tests**: run `adapter.evaluate` behind a flag (and record
   sessions). These are slow and cost money, so run them selectively.

**A prompt snapshot test:**

```python
def test_prompt_renders_stably():
    rendered = prompt.bind(Params(question="x")).render(session=session)
    assert "Question: x" in rendered.text
```

The test doesn't call a model. It just verifies that the prompt renders as
expected. When prompts are deterministic, you can test them like regular code.

---

## 13. Recipes

These are intentionally opinionated. They reflect the "weak incentives" style:
reduce surprise, keep state explicit, and make side effects auditable.

### 13.1 A code-review agent

See `code_reviewer_example.py` in this repo for a full, runnable example that
demonstrates:

- Workspace digest + progressive disclosure
- Planning tools
- VFS or Podman sandbox
- Prompt overrides and optimizer hooks
- Structured output review responses

This is the canonical "put it all together" example. Read it after you
understand the individual pieces.

### 13.2 A repo Q&A agent

**Goal**: answer questions about a codebase quickly.

**Pattern**:

- Show workspace digest summary by default
- Allow the model to expand it via `read_section`
- Allow VFS `grep`/`glob`/`read_file` for verification

The model sees a summary, asks questions, digs into details as needed. Token
usage stays low for simple questions.

### 13.3 A "safe patch" agent

**Goal**: generate a patch but avoid uncontrolled writes.

**Pattern**:

- Use VFS tools for edits (writes go to the virtual copy, not the host)
- Require the model to output a diff as structured output
- Optionally run tests in Podman before proposing the patch

The model can experiment freely in the VFS. Only the final diff matters.
Humans review the diff before applying it to the real repo.

### 13.4 A research agent with progressive disclosure

**Goal**: answer deep questions without stuffing a giant blob into the prompt.

**Pattern**:

- Store sources as summarized sections
- Let the model open only what it needs
- Keep an audit trail via session snapshots

Progressive disclosure shines here. The model starts with summaries, expands
relevant sources, and cites its sources. The session log shows exactly which
sources it used.

---

## 14. Troubleshooting

Common issues you'll hit when getting started:

### "PromptValidationError: placeholder not found"

Your template uses `${foo}` but no bound dataclass has a `foo` field.

**Fix**: Check that your dataclass field names match placeholder names exactly.
Placeholders are case-sensitive.

```python
# Wrong: placeholder is ${query}, field is question
@dataclass
class Params:
    question: str  # Should be 'query'

# Right
@dataclass
class Params:
    query: str
```

### "Tool handler returned None"

Tool handlers must return `ToolResult`, not `None`.

**Fix**: Always return a `ToolResult`, even for failures:

```python
def handler(params, *, context):
    if something_wrong:
        return ToolResult(message="Failed", value=None, success=False)
    return ToolResult(message="OK", value=result)
```

### "OutputParseError: missing required field"

The model's JSON response doesn't match your output dataclass.

**Fix**: Check that your dataclass fields match what the model returns. Add
clear instructions in your prompt about the expected JSON structure. Use
`allow_extra_keys=True` on the template if you want to ignore extra fields.

### Model doesn't call tools

The model sees tools but chooses not to use them.

**Fixes**:

1. Make instructions clearer: "Use the search tool to find information before
   answering"
2. Add tool examples to show correct usage
3. Check that the tool description accurately describes what it does

### "DeadlineExceededError"

The agent ran past its deadline.

**Fixes**:

1. Increase the deadline
2. Reduce prompt size (use progressive disclosure)
3. Check for tool handlers that hang or take too long

### Session state not persisting

State changes aren't visible in subsequent queries.

**Fix**: Make sure you're using the same session instance, and that you've
registered reducers for your event types:

```python
session[Plan].register(AddStep, my_reducer)
session.broadcast(AddStep(step="do thing"))
```

### Debugging prompts

To see exactly what's being sent to the model:

```python
rendered = prompt.render(session=session)
print(rendered.text)  # Full prompt markdown
print([t.name for t in rendered.tools])  # Tool names
```

For full session inspection, use the debug UI:

```bash
pip install "weakincentives[wink]"
wink debug snapshots/session.jsonl
```

---

## 15. API reference

This is a curated reference of the APIs you'll touch most often. For complete
details, read module docstrings and the specs.

### 15.1 Top-level exports

Import from `weakincentives` when you want the "90% API":

**Budgets/time:**

- `Deadline`, `DeadlineExceededError`
- `Budget`, `BudgetTracker`, `BudgetExceededError`

**Prompt primitives:**

- `PromptTemplate`, `Prompt`, `RenderedPrompt`
- `Section`, `MarkdownSection`
- `Tool`, `ToolContext`, `ToolResult`
- `SectionVisibility`
- `parse_structured_output`, `OutputParseError`

**Runtime primitives:**

- `Session`, `InProcessEventBus`
- `MainLoop` and loop events (`MainLoopRequest`, `MainLoopCompleted`,
  `MainLoopFailed`)
- Reducer helpers (`append_all`, `replace_latest`, `upsert_by`, ...)
- Logging helpers (`configure_logging`, `get_logger`)

**Errors:**

- `WinkError`, `ToolValidationError`, snapshot/restore errors

### 15.2 weakincentives.prompt

```python
PromptTemplate[OutputT](ns, key, name=None, sections=..., allow_extra_keys=False)
Prompt(template, overrides_store=None, overrides_tag="latest")
    .bind(*params)
    .render(session=None)
    .find_section(SectionType)

MarkdownSection(title, key, template, summary=None, visibility=..., tools=...)
Tool(name, description, handler, examples=...)
ToolResult(message, value, success=True, exclude_value_from_context=False)
```

**Progressive disclosure:**

- `VisibilityExpansionRequired`
- `VisibilityOverrides`, `SetVisibilityOverride`, ...

### 15.3 weakincentives.runtime

```python
Session(bus, tags=None, parent=None)
session[Type].all() / latest() / where()
session.broadcast(event)
session.snapshot(include_all=False)
session.restore(snapshot, preserve_logs=True)
```

**Reducers:**

- `append_all`, `replace_latest`, `replace_latest_by`, `upsert_by`

**Event bus:**

- `InProcessEventBus`
- Telemetry events (`PromptRendered`, `ToolInvoked`, `PromptExecuted`,
  `TokenUsage`)

### 15.4 weakincentives.adapters

```python
ProviderAdapter.evaluate(prompt, session=..., deadline=..., budget=...)
PromptResponse(prompt_name, text, output)
PromptEvaluationError
```

**Configs:**

- `OpenAIClientConfig`, `OpenAIModelConfig`
- `LiteLLMClientConfig`, `LiteLLMModelConfig`

**Throttling:**

- `ThrottlePolicy`, `new_throttle_policy`, `ThrottleError`

### 15.5 weakincentives.contrib.tools

**Planning:**

- `PlanningToolsSection(session, strategy=..., accepts_overrides=False)`

**Workspace:**

- `VfsToolsSection(session, config=VfsConfig(...), accepts_overrides=False)`
- `HostMount(host_path, mount_path=None, include_glob=(), exclude_glob=())`
- `WorkspaceDigestSection(session, title="Workspace Digest",
  key="workspace-digest")`

**Sandboxes:**

- `AstevalSection(session, accepts_overrides=False)`
- `PodmanSandboxSection(session, config=PodmanSandboxConfig(...))` (extra)

### 15.6 weakincentives.optimizers

- `PromptOptimizer` protocol and `BasePromptOptimizer`
- `OptimizationContext`, `OptimizationResult`

**Contrib:**

- `WorkspaceDigestOptimizer`

### 15.7 weakincentives.serde

```python
dump(obj) -> dict
parse(DataclassType, mapping) -> DataclassType
schema(DataclassType) -> dict
clone(obj) -> dataclass copy
```

### 15.8 CLI

Installed via the `wink` extra:

```bash
wink debug <snapshot_path> [--host ...] [--port ...]
```

---

## Where to go deeper

- **Prompts**: [specs/PROMPTS.md](specs/PROMPTS.md)
- **Tools**: [specs/TOOLS.md](specs/TOOLS.md)
- **Sessions**: [specs/SESSIONS.md](specs/SESSIONS.md)
- **MainLoop**: [specs/MAIN_LOOP.md](specs/MAIN_LOOP.md)
- **Workspace**: [specs/WORKSPACE.md](specs/WORKSPACE.md)
- **Overrides & optimization**: [specs/PROMPT_OPTIMIZATION.md](specs/PROMPT_OPTIMIZATION.md)
- **Code review example**: [guides/code-review-agent.md](guides/code-review-agent.md)
- **Contributor guide**: [AGENTS.md](AGENTS.md)
