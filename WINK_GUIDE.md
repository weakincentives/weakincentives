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

______________________________________________________________________

This guide is written for engineers who want to:

- Build agents that can run unattended without turning into "a pile of prompt
  glue".
- Treat prompts as real software artifacts: testable, inspectable, and
  versionable.
- Keep tool use and side effects explicit, gated, and auditable.
- Iterate on prompts quickly without compromising correctness.

**If you only read one thing**: in WINK, the prompt is the agent.

**Status**: Alpha. Expect some APIs to evolve as the library matures.

______________________________________________________________________

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

The future points toward SDKs shaped like the Claude Agent SDK: sophisticated
sandboxing, native tool integration, seamless handoff between local and hosted
execution. Models will increasingly come with their own tool runtimes, deeply
integrated into training. WINK's job is to give you stable primitives—prompts,
tools, state—that work across that evolving landscape.

______________________________________________________________________

## Table of Contents

1. [Technical Strategy](#technical-strategy)
1. [Philosophy](#1-philosophy)
   1. [What "weak incentives" means](#11-what-weak-incentives-means)
   1. [The shift: orchestration shrinks, context engineering grows](#12-the-shift-orchestration-shrinks-context-engineering-grows)
   1. [The core bet: prompts as first-class, typed programs](#13-the-core-bet-prompts-as-first-class-typed-programs)
   1. [What WINK is (and is not)](#14-what-wink-is-and-is-not)
1. [Quickstart](#2-quickstart)
   1. [Install](#21-install)
   1. [End-to-end: a tiny structured agent](#22-end-to-end-a-tiny-structured-agent)
   1. [Add a tool](#23-add-a-tool)
   1. [Your first complete agent](#24-your-first-complete-agent-copy-paste-ready)
1. [Prompts](#3-prompts)
   1. [PromptTemplate](#31-prompttemplate)
   1. [Prompt](#32-prompt)
   1. [Sections](#33-sections)
   1. [MarkdownSection](#34-markdownsection)
   1. [Structured output](#35-structured-output)
   1. [Dynamic scoping with enabled()](#36-dynamic-scoping-with-enabled)
   1. [Session-bound sections and cloning](#37-session-bound-sections-and-cloning)
   1. [Few-shot traces with TaskExamplesSection](#38-few-shot-traces-with-taskexamplessection)
1. [Tools](#4-tools)
   1. [Tool contracts](#41-tool-contracts)
   1. [ToolContext, resources, and Filesystem](#42-toolcontext-resources-and-filesystem)
   1. [ToolResult semantics](#43-toolresult-semantics)
   1. [Tool examples](#44-tool-examples)
   1. [Tool suites as sections](#45-tool-suites-as-sections)
   1. [Transactional tool execution](#46-transactional-tool-execution)
   1. [Tool policies](#47-tool-policies)
1. [Sessions](#5-sessions)
   1. [Session as deterministic memory](#51-session-as-deterministic-memory)
   1. [Queries](#52-queries)
   1. [Reducers](#53-reducers)
   1. [Declarative reducers with @reducer](#54-declarative-reducers-with-reducer)
   1. [Snapshots and restore](#55-snapshots-and-restore)
   1. [SlicePolicy: state vs logs](#56-slicepolicy-state-vs-logs)
1. [Adapters](#6-adapters)
   1. [ProviderAdapter.evaluate](#61-provideradapterevaluate)
   1. [OpenAIAdapter](#62-openaiadapter)
   1. [LiteLLMAdapter](#63-litellmadapter)
   1. [Claude Agent SDK adapter](#64-claude-agent-sdk-adapter)
      1. [Requirements](#641-requirements)
      1. [Basic usage](#642-basic-usage)
      1. [Client and model configuration](#643-client-and-model-configuration)
      1. [Workspace management](#644-workspace-management)
      1. [Isolation configuration](#645-isolation-configuration)
      1. [Tool bridging via MCP](#646-tool-bridging-via-mcp)
      1. [Events](#647-events)
      1. [Complete example: secure code review](#648-complete-example-secure-code-review)
      1. [Docs assistant with domain allowlist](#649-docs-assistant-with-domain-allowlist)
      1. [Operational notes](#6410-operational-notes)
1. [Orchestration with MainLoop](#7-orchestration-with-mainloop)
   1. [The minimal MainLoop](#71-the-minimal-mainloop)
   1. [Deadlines and budgets](#72-deadlines-and-budgets)
1. [Evaluation with EvalLoop](#8-evaluation-with-evalloop)
   1. [The composition philosophy](#81-the-composition-philosophy)
   1. [Core types](#82-core-types)
   1. [LLM-as-judge](#83-llm-as-judge)
   1. [Session evaluators](#84-session-evaluators)
   1. [Running evaluations](#85-running-evaluations)
   1. [Production deployment pattern](#86-production-deployment-pattern)
   1. [Reply-to routing](#87-reply-to-routing)
1. [Lifecycle Management](#9-lifecycle-management)
   1. [LoopGroup: running multiple loops](#91-loopgroup-running-multiple-loops)
   1. [ShutdownCoordinator: manual signal handling](#92-shutdowncoordinator-manual-signal-handling)
   1. [The Runnable protocol](#93-the-runnable-protocol)
   1. [Health and watchdog configuration](#94-health-and-watchdog-configuration)
1. [Progressive disclosure](#10-progressive-disclosure)
   1. [SectionVisibility: FULL vs SUMMARY](#101-sectionvisibility-full-vs-summary)
   1. [open_sections and read_section](#102-open_sections-and-read_section)
   1. [Visibility overrides in session state](#103-visibility-overrides-in-session-state)
1. [Prompt overrides and optimization](#11-prompt-overrides-and-optimization)
   1. [Hash-based safety: override only what you intended](#111-hash-based-safety-override-only-what-you-intended)
   1. [LocalPromptOverridesStore](#112-localpromptoverridesstore)
   1. [Override file format](#113-override-file-format)
   1. [A practical override workflow](#114-a-practical-override-workflow)
1. [Workspace tools](#12-workspace-tools)
   1. [PlanningToolsSection](#121-planningtoolssection)
   1. [VfsToolsSection](#122-vfstoolssection)
   1. [WorkspaceDigestSection](#123-workspacedigestsection)
   1. [AstevalSection](#124-astevalsection)
   1. [PodmanSandboxSection](#125-podmansandboxsection)
   1. [Wiring a workspace into a prompt](#126-wiring-a-workspace-into-a-prompt)
1. [Debugging and observability](#13-debugging-and-observability)
   1. [Structured logging](#131-structured-logging)
   1. [Session events](#132-session-events)
   1. [Dumping snapshots to JSONL](#133-dumping-snapshots-to-jsonl)
   1. [The debug UI](#134-the-debug-ui)
1. [Testing and reliability](#14-testing-and-reliability)
1. [Approach to code quality](#15-approach-to-code-quality)
   1. [Strict type checking](#151-strict-type-checking)
   1. [Design-by-contract](#152-design-by-contract)
   1. [Coverage requirements](#153-coverage-requirements)
   1. [Security scanning](#154-security-scanning)
   1. [Quality gates in practice](#155-quality-gates-in-practice)
1. [Recipes](#16-recipes)
   1. [A code-review agent](#161-a-code-review-agent)
   1. [A repo Q&A agent](#162-a-repo-qa-agent)
   1. [A "safe patch" agent](#163-a-safe-patch-agent)
   1. [A research agent with progressive disclosure](#164-a-research-agent-with-progressive-disclosure)
1. [Troubleshooting](#17-troubleshooting)
1. [API reference](#18-api-reference)
   1. [Top-level exports](#181-top-level-exports)
   1. [weakincentives.prompt](#182-weakincentivesprompt)
   1. [weakincentives.runtime](#183-weakincentivesruntime)
   1. [weakincentives.adapters](#184-weakincentivesadapters)
   1. [weakincentives.contrib.tools](#185-weakincentivescontribtools)
   1. [weakincentives.optimizers](#186-weakincentivesoptimizers)
   1. [weakincentives.serde](#187-weakincentivesserde)
   1. [weakincentives.evals](#188-weakincentivesevals)
   1. [weakincentives.skills](#189-weakincentivesskills)
   1. [weakincentives.filesystem](#1810-weakincentivesfilesystem)
   1. [CLI](#1811-cli)
1. [Appendix A: Coming from LangGraph or LangChain?](#appendix-a-coming-from-langgraph-or-langchain)
1. [Appendix B: Coming from DSPy?](#appendix-b-coming-from-dspy)
1. [Appendix C: Formal Verification with TLA+](#appendix-c-formal-verification-with-tla)

______________________________________________________________________

## 1. Philosophy

### 1.1 What "weak incentives" means

"Weak incentives" is an engineering stance:

> Build agent systems where well-constructed prompts and tools create weak
> incentives for the model to do the right thing and stay on task.

The name comes from mechanism design: a system with the right incentives is one
where participants naturally gravitate toward intended behavior. Applied to
agents, this means shaping the prompt, tools, and context so the model's easiest
path is also the correct one.

This isn't about constraining the model or managing downside risk. It's about
_encouraging_ correct behavior through structure:

- **Clear instructions co-located with tools** make the right action obvious
- **Typed contracts** guide the model toward valid outputs
- **Progressive disclosure** keeps the model focused on what matters now
- **Explicit state** gives the model the context it needs to make good decisions

The optimization process strengthens these incentives. When you refine a prompt
override or add a tool example, you're making the correct path even more
natural. Over iterations, the system becomes increasingly well-tuned—not through
constraints, but through clarity.

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

The goal isn't to constrain the model—it's to give it the best possible starting
point. When prompts are clear, tools are well-documented, and state is explicit,
the model has strong signals about what to do. When something goes wrong, you
can see exactly what happened and refine the incentives for next time.

### 1.2 The shift: orchestration shrinks, context engineering grows

Many early "agent frameworks" assumed the hard part would be workflow logic:
routers, planners, branching graphs, and elaborate loops. These frameworks spent
their complexity budget on orchestration—deciding which prompts to run when,
routing between specialized agents, managing elaborate state machines.

WINK makes a different bet:

**Models are steadily absorbing more of the reasoning loop.**

What required explicit multi-step orchestration yesterday often works in a
single prompt today. The frontier models are increasingly capable of planning,
reasoning, and self-correction within a single context window. Elaborate routing
graphs often just get in the way.

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
[specs/PROMPTS.md](specs/PROMPTS.md), [specs/TOOLS.md](specs/TOOLS.md),
[specs/SESSIONS.md](specs/SESSIONS.md),
[specs/MAIN_LOOP.md](specs/MAIN_LOOP.md).

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
provides the `grep` tool. Documentation and capability live together. They can't
drift.

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

1. **Safe iteration**: apply prompt tweaks via overrides that are validated
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
- An async-first streaming framework. Today the adapter contract is synchronous.
  Streaming may come later.

If you need a graph engine or multi-agent coordination, you can still use WINK
for the pieces that benefit from determinism (prompt design, tool contracts,
state snapshots) and let something else coordinate the rest. WINK plays well
with others.

______________________________________________________________________

## 2. Quickstart

### 2.1 Install

The core package has no mandatory third-party dependencies. This is intentional:
you should be able to use WINK's prompt and session primitives without pulling
in OpenAI or any other provider SDK.

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

**Python requirement**: 3.12+. We use modern Python features liberally and don't
maintain compatibility with older versions.

### 2.2 End-to-end: a tiny structured agent

This example is intentionally small but complete:

- Typed params (`SummarizeRequest`)
- Structured output (`Summary`)
- Deterministic prompt structure (`PromptTemplate`)
- A session to record telemetry

```python
from dataclasses import dataclass

from weakincentives.prompt import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import InProcessDispatcher, Session

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

dispatcher = InProcessDispatcher()
session = Session(dispatcher=dispatcher)

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

Here's a toy tool that returns the current time. In a real agent this would be a
filesystem read, an API call, or some other operation with side effects.

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
        return ToolResult.error("Only UTC supported in this demo.")
    return ToolResult.ok(
        NowResult(iso=datetime.now(UTC).isoformat()),
        message="Current time.",
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

Once this section is included in your template, adapters will advertise `now` to
the model and execute calls synchronously.

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
class SearchToolParams:
    query: str


@dataclass(slots=True, frozen=True)
class SearchToolOutput:
    snippets: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(f"- {s}" for s in self.snippets)


def search_handler(
    params: SearchToolParams, *, context: ToolContext
) -> ToolResult[SearchToolOutput]:
    # In a real agent, this would call a search API
    del context
    return ToolResult.ok(
        SearchToolOutput(
            snippets=(
                f"Result 1 about {params.query}",
                f"Result 2 about {params.query}",
            )
        ),
        message=f"Found results for: {params.query}",
    )


search_tool = Tool[SearchToolParams, SearchToolOutput](
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

    if response.output is not None:
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
1. The `search` tool is registered on the Instructions section—the model sees
   the tool alongside the instructions for using it
1. `adapter.evaluate()` sends the prompt to OpenAI, executes any tool calls, and
   parses the structured response
1. You get back `response.output` as a typed `Answer` instance

This is the core loop. Everything else in WINK builds on this: sessions for
state, sections for organization, progressive disclosure for token management.

______________________________________________________________________

## 3. Prompts

_Canonical spec: [specs/PROMPTS.md](specs/PROMPTS.md)_

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

A section must implement `render_body(...)`. Many sections use
`MarkdownSection`, but contributed tool suites are also sections (planning, VFS,
sandboxes, etc.).

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
1. Parse the response into your dataclass type
1. Return it as `PromptResponse.output`

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


@dataclass(slots=True, frozen=True)
class DebugFlag:
    enabled: bool


debug_section = MarkdownSection[DebugFlag](
    title="Debug",
    key="debug",
    template="If something fails, include stack traces and hypotheses.",
    default_params=DebugFlag(enabled=False),
)
```

Disabled sections don't just hide their text—their tools also disappear from the
prompt. This lets you build a comprehensive template and enable only the
capabilities relevant to the current context.

### 3.7 Session-bound sections and cloning

Some sections are **pure**: they depend only on params and render the same text
every time. You can safely store those in a module-level `PromptTemplate`.

Other sections are **session-bound**: they capture runtime resources (a session,
filesystem, sandbox connection, etc.). Examples:

- `PlanningToolsSection(session=...)`
- `VfsToolsSection(session=...)`
- `WorkspaceDigestSection(session=...)`
- `PodmanSandboxSection(session=...)`

For those, prefer one of these patterns:

**Pattern A: build the template per session**

```python
from typing import Any
from weakincentives.contrib.tools import PlanningToolsSection, VfsToolsSection
from weakincentives.prompt import PromptTemplate, MarkdownSection
from weakincentives.runtime import Session


def build_prompt_template(*, session: Session) -> PromptTemplate[Any]:
    return PromptTemplate(
        ns="example",
        key="session-bound",
        sections=(
            MarkdownSection(title="Instructions", key="instructions", template="Follow the plan."),
            PlanningToolsSection(session=session),
            VfsToolsSection(session=session),
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

______________________________________________________________________

## 4. Tools

_Canonical spec: [specs/TOOLS.md](specs/TOOLS.md)_

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
    return ToolResult.ok(MyResult(answer="42"), message="ok")

tool = Tool[MyParams, MyResult](
    name="my_tool",
    description="Do a thing.",
    handler=handler,
)
```

The type parameters matter. `Tool[MyParams, MyResult]` tells WINK how to
serialize parameters for the model and how to parse results. Type mismatches are
caught at construction time.

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
from weakincentives.resources import Binding, ResourceRegistry

registry = ResourceRegistry.of(
    Binding.instance(MyHttpClient, MyHttpClient(...)),
)
```

**Binding resources to prompts:**

Resources are bound to `Prompt` via the `bind()` method. Pass a mapping of
protocol types to instances or Binding objects:

```python
from typing import Any

from weakincentives.resources import Binding, Scope
from weakincentives.prompt import Prompt


# Example resource types (define your own in your application)
class Config:
    def __init__(self) -> None:
        self.url: str = "https://api.example.com"


class HTTPClient:
    def __init__(self, url: str = "") -> None:
        self.url = url


class Tracer:
    pass


# Type stubs (defined in your application)
template: Any = ...  # type: ignore[assignment]
params: Any = ...  # type: ignore[assignment]
adapter: Any = ...  # type: ignore[assignment]
session: Any = ...  # type: ignore[assignment]

# Simple case: pre-constructed instances (pass a dict)
http_client = HTTPClient(url="https://api.example.com")
prompt = Prompt(template).bind(params, resources={HTTPClient: http_client})

# Advanced: lazy construction with dependencies and scopes
# Pass Binding objects in the mapping for custom providers/scopes
prompt = Prompt(template).bind(
    params,
    resources={
        Config: Binding(Config, lambda r: Config()),
        HTTPClient: Binding(HTTPClient, lambda r: HTTPClient(url=r.get(Config).url)),  # type: ignore[attr-defined]
        Tracer: Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
    },
)

# Use prompt.resources context manager for lifecycle management
with prompt.resources:
    response = adapter.evaluate(prompt, session=session)
    # Resources are accessible within this block
```

**Using MainLoop (recommended):**

`MainLoop.execute()` handles resource binding and lifecycle automatically:

```python
from typing import Any

from weakincentives.resources import Binding
from weakincentives.runtime import MainLoopConfig


# Example resource types (from previous example)
class Config:
    def __init__(self) -> None:
        self.url: str = "https://api.example.com"


class HTTPClient:
    def __init__(self, url: str = "") -> None:
        self.url = url


class Tracer:
    pass


# Type stubs (defined in your application)
MyLoop: Any = ...  # type: ignore[assignment]
adapter: Any = ...  # type: ignore[assignment]
dispatcher: Any = ...  # type: ignore[assignment]
request: Any = ...  # type: ignore[assignment]
tracer: Any = ...  # type: ignore[assignment]

# Configure resources at the loop level (pass a mapping, not a ResourceRegistry)
config = MainLoopConfig(
    resources={
        Config: Binding(Config, lambda r: Config()),
        HTTPClient: Binding(HTTPClient, lambda r: HTTPClient(url=r.get(Config).url)),  # type: ignore[attr-defined]
    }
)
loop = MyLoop(adapter=adapter, dispatcher=dispatcher, config=config)

# Resources are bound to prompt automatically
response, session = loop.execute(request)

# Or pass per-request resources (also a mapping)
response, session = loop.execute(request, resources={Tracer: tracer})
```

**Scopes control instance lifetime:**

- `Scope.SINGLETON`: One instance per session (default)
- `Scope.TOOL_CALL`: Fresh instance per tool invocation
- `Scope.PROTOTYPE`: Fresh instance on every access

**Dependency injection with providers:**

Bindings support lazy construction with dependency resolution. The provider
function receives a resolver that can look up other resources:

```python
from weakincentives.resources import Binding, ResourceRegistry, Scope


# Example resource types (from previous example)
class Config:
    def __init__(self) -> None:
        self.url: str = "https://api.example.com"


class HTTPClient:
    def __init__(self, url: str = "") -> None:
        self.url = url


class Tracer:
    pass


resources = ResourceRegistry.of(
    # Config is constructed first (no dependencies)
    Binding(Config, lambda r: Config()),
    # HTTPClient depends on Config
    Binding(HTTPClient, lambda r: HTTPClient(url=r.get(Config).url)),  # type: ignore[attr-defined]
    # Tracer is fresh per tool call
    Binding(Tracer, lambda r: Tracer(), scope=Scope.TOOL_CALL),
)

# Create resolution context with lifecycle management
# open() handles start() and close() automatically
with resources.open() as ctx:
    # Lazy resolution with dependency graph walking
    http = ctx.get(HTTPClient)  # Also resolves Config

    # Tool-call scoped resources
    with ctx.tool_scope() as tool_resolver:
        tracer = tool_resolver.get(Tracer)  # Fresh instance
# Resources cleaned up automatically on exit
```

**Lifecycle protocols:**

Resources can implement lifecycle protocols for automatic management:

- `Closeable`: Resources with a `close()` method are closed when the context
  ends
- `PostConstruct`: Resources with a `post_construct()` method are initialized
  after construction

**Key behaviors:**

- `ResourceRegistry.merge()` combines registries with the second taking
  precedence on conflicts
- `ResourceRegistry.conflicts()` returns protocols bound in both registries
- Circular dependencies raise `CircularDependencyError`
- User-provided resources override workspace defaults (e.g., custom filesystem)
- Tool handlers access resources via `context.resources.get(ResourceType)`
- Resources are bound to prompts via `prompt.bind(resources=...)` or via
  `MainLoop.execute(resources=...)`

For workspace agents, the most common resource is a `Filesystem` implementation.
Many contributed tool suites install one automatically (VFS, Podman).

### 4.3 ToolResult semantics

Tool handlers return `ToolResult`. Use the convenience constructors:

```python
# Success with typed value (most common)
ToolResult.ok(MyResult(...), message="Done")

# Failure with no value
ToolResult.error("Something went wrong")

# Full form (when exclude_value_from_context is needed)
ToolResult(
    message="Human-readable status",
    value=...,                        # dataclass | mapping | sequence | str | None
    success=True,                     # if False, adapters treat as tool failure
    exclude_value_from_context=False, # hide large payloads from model context
)
```

**Key behaviors:**

- If `value` is a dataclass and implements `render() -> str`, adapters use that
  as the textual tool output. This lets you control exactly what the model sees.
- If `render()` is missing, WINK logs a warning and serializes the dataclass to
  JSON. This works but is less controlled.
- Exceptions raised by handlers are caught and converted into tool failures
  (with some safety exceptions; see [specs/TOOLS.md](specs/TOOLS.md)).

The `exclude_value_from_context=True` flag is useful for tools that return large
payloads (like file contents). The model sees a summary message, but the full
value is recorded in the session for debugging.

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
parameter name, wrong format, etc.—examples tend to pay for themselves. One good
example often beats three paragraphs of instructions.

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

Each section bundles the instructions ("here's how to use these tools") with the
tools themselves. The model sees them together.

### 4.6 Transactional tool execution

_Canonical spec: [specs/SESSIONS.md](specs/SESSIONS.md)_

One of the hardest problems in building agents is handling partial failures.
When a tool call fails halfway through, you're left with corrupted session
state, inconsistent filesystem changes, or both. Debugging becomes a nightmare
of "what state was the agent actually in when this happened?"

WINK solves this with **transactional tool execution**. Every tool call is
wrapped in a transaction:

1. **Snapshot**: Before the tool runs, WINK captures the current state of the
   session and any snapshotable resources (like the filesystem)
1. **Execute**: The tool handler runs
1. **Commit or rollback**: If the tool succeeds, changes are kept. If it fails,
   WINK automatically restores the snapshot

This happens by default—you don't need to opt in or write rollback logic. Failed
tools simply don't leave traces in mutable state.

**What gets rolled back:**

- Session slices marked as `STATE` (working state like plans, visibility
  overrides, workspace digests)
- Filesystem changes (both in-memory VFS and disk-backed via git commits)

**What's preserved:**

- Session slices marked as `LOG` (historical records like `ToolInvoked`,
  `PromptRendered`)
- External side effects (API calls, network requests)

**Example scenario:**

```python
# Tool handler that might fail
def risky_handler(params, *, context):
    fs = context.resources.get(Filesystem)
    session = context.session

    # Update session state
    session.dispatch(UpdatePlan(status="in-progress"))

    # Write to filesystem
    fs.write("output.txt", "partial results")

    # Oops, something goes wrong
    if params.force:
        raise ValueError("Simulated failure")

    return ToolResult(message="done", value=None)
```

If this tool raises an exception:

- The session state rolls back (plan status reverts)
- The filesystem write is undone
- The model sees a clean failure message
- No debugging required to figure out "what state are we actually in"

**Why this matters:**

- **Simpler error handling**: You don't need defensive rollback code in every
  tool handler
- **Consistent state**: Failed operations never leave the agent in an
  inconsistent state
- **Easier debugging**: When something fails, you know exactly what state you're
  in—the state from before the failed call
- **Adapter parity**: OpenAI, LiteLLM, and Claude Agent SDK adapters all use the
  same transaction semantics

This is especially valuable for agents that modify files, update plans, or
maintain complex working state. A failed `write_file` or `update_plan` doesn't
corrupt your agent's world model.

**Manual transaction control:**

For advanced use cases, you can use the transaction API directly:

```python
from typing import Any

from weakincentives.runtime import (
    CompositeSnapshot,
    create_snapshot,
    restore_snapshot,
    tool_transaction,
    PendingToolTracker,
)


# Type stubs (defined in your application)
session: Any = ...  # type: ignore[assignment]
resources: Any = ...  # type: ignore[assignment]


def risky_operation() -> bool:
    return True


def do_work() -> None:
    pass


# Option 1: Context manager (auto-rollback on exception)
with tool_transaction(session, resources, tag="my_operation") as snapshot:
    # Do work that might fail
    success = risky_operation()
    if not success:
        restore_snapshot(session, resources, snapshot)  # Manual rollback

# Option 2: Manual snapshot/restore
snapshot = create_snapshot(session, resources, tag="checkpoint")
try:
    do_work()
except Exception:
    restore_snapshot(session, resources, snapshot)
    raise

# Option 3: Hook-based tracking (for Claude Agent SDK)
tracker = PendingToolTracker(session=session, resources=resources)
tracker.begin_tool_execution(tool_use_id="abc", tool_name="write_file")
# ... native tool executes ...
tracker.end_tool_execution(tool_use_id="abc", success=False)  # Auto-rollback
```

### 4.7 Tool policies

_Canonical spec: [specs/TOOLS.md](specs/TOOLS.md)_

Tool policies provide declarative constraints that govern when tools can be
invoked. Rather than embedding validation logic in each tool handler, policies
express cross-cutting concerns as composable constraints.

**Why policies exist:**

Without constraints, models can call tools in problematic orders—deploying code
that was never tested, overwriting files they haven't read, or skipping required
validation steps. Policies catch these issues before the tool executes.

**Built-in policies:**

```python
from weakincentives.prompt import (
    SequentialDependencyPolicy,
    ReadBeforeWritePolicy,
)

# Require 'test' and 'build' before 'deploy'
deploy_policy = SequentialDependencyPolicy(
    dependencies={
        "deploy": frozenset({"test", "build"}),
    }
)

# Require reading a file before overwriting it (new files allowed)
read_first = ReadBeforeWritePolicy()
```

**Attaching policies to tools:**

Policies are attached at the section level:

```python
from weakincentives.prompt import MarkdownSection, Tool

section = MarkdownSection(
    title="Deployment",
    key="deployment",
    template="Deploy the application after testing.",
    tools=(deploy_tool, test_tool, build_tool),
    policies=(deploy_policy,),
)
```

**Enforcement:**

When a tool call violates a policy, WINK returns a `ToolResult.error()` without
executing the handler. The error message explains which policy was violated and
what the model should do instead.

**Default policies on contrib sections:**

`VfsToolsSection` and `PodmanSandboxSection` apply `ReadBeforeWritePolicy` by
default. This prevents accidental overwrites without reading the existing
content first.

```python
from weakincentives.contrib.tools import VfsToolsSection

# ReadBeforeWritePolicy is applied automatically
vfs = VfsToolsSection(session=session, config=vfs_config)
```

**Key behaviors:**

- Policies are evaluated in order; first violation stops execution
- `SequentialDependencyPolicy` checks session `ToolInvoked` events for required
  tool completions
- `ReadBeforeWritePolicy` tracks `read_file` calls and allows writes only to
  paths that were read (or new files)
- Custom policies can implement the `ToolPolicy` protocol

______________________________________________________________________

## 5. Sessions

_Canonical spec: [specs/SESSIONS.md](specs/SESSIONS.md)_

A `Session` is WINK's answer to "agent memory", with a constraint:

> **Memory must be deterministic and inspectable.**

Instead of "a magic dict" you mutate, sessions store typed slices managed by
pure reducers. Every mutation flows through a reducer, and every change is
recorded as an event.

### 5.1 Session as deterministic memory

A session is a container keyed by dataclass type:

- Each type has a **slice**: `tuple[T, ...]`
- **Reducers** update slices in response to events
- The session subscribes to the event dispatcher and records prompt/tool telemetry

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

A **reducer** is a pure function that takes the current state and an event, and
returns the new state. The name comes from functional programming (and was
popularized by Redux in frontend development), but the concept is simple:

```
new_state = reducer(current_state, event)
```

Reducers never mutate state directly. They always return a new value. This makes
state changes predictable: given the same inputs, you always get the same
output. It also makes debugging easier—you can log every event and trace the
exact sequence that led to any state.

In WINK, reducers receive a `SliceView[S]` (read-only access to current values)
and return a `SliceOp[S]` (describing the mutation to apply):

```python
def my_reducer(state: SliceView[Plan], event: AddStep) -> Append[Plan]:
    return Append(Plan(steps=(event.step,)))
```

**SliceOp variants:**

- `Append[T]`: Add a single value to the slice
- `Extend[T]`: Add multiple values to the slice
- `Replace[T]`: Replace all values in the slice
- `Clear`: Remove all values from the slice

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
session.dispatch(Plan(steps=("step 1",)))
session.dispatch(Plan(steps=("step 2",)))
assert session[Plan].all() == (Plan(steps=("step 2",)),)
```

### 5.4 Declarative reducers with @reducer

For complex slices, attach reducers as methods using `@reducer`. Methods must
return `Replace[T]` wrapping the new value:

```python
from dataclasses import dataclass, replace
from weakincentives.runtime.session import reducer, Replace

@dataclass(slots=True, frozen=True)
class AddStep:
    step: str

@dataclass(slots=True, frozen=True)
class AgentPlan:
    steps: tuple[str, ...]

    @reducer(on=AddStep)
    def add_step(self, event: AddStep) -> Replace["AgentPlan"]:
        return Replace((replace(self, steps=(*self.steps, event.step)),))

session.install(AgentPlan, initial=lambda: AgentPlan(steps=()))
session.dispatch(AddStep(step="read README"))
session.dispatch(AddStep(step="run tests"))
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

Snapshots serialize to JSON. You can persist them to disk and reload them later.
This is how the debug UI works: it reads snapshot files and displays the session
state at each point.

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

______________________________________________________________________

## 6. Adapters

_Canonical spec: [specs/ADAPTERS.md](specs/ADAPTERS.md)_

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
    resources=...,       # optional ResourceRegistry
)
```

It returns `PromptResponse[OutputT]`:

- `prompt_name`: string
- `text`: raw assistant text
- `output`: parsed structured output (or `None`)

The adapter handles all the provider-specific details: API formatting, tool
schema translation, response parsing. Your code just calls `evaluate()` and gets
back typed results.

The optional `resources` parameter lets you inject custom resources (HTTP
clients, databases, external services) that tool handlers can access via
`context.resources.get(ResourceType)`. Injected resources are merged with any
workspace-provided resources, with user-provided resources taking precedence.

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
    completion_config=LiteLLMClientConfig(),
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

See [specs/CLAUDE_AGENT_SDK.md](specs/CLAUDE_AGENT_SDK.md) for full
configuration reference and isolation guarantees.

#### 6.4.1 Requirements

- **Python package:** `pip install 'weakincentives[claude-agent-sdk]'`
- **Claude Code CLI:** `npm install -g @anthropic-ai/claude-code`
- **Linux sandboxing:** bubblewrap (`bwrap`) available on PATH (for sandbox
  enforcement)

#### 6.4.2 Basic usage

The simplest usage requires minimal configuration:

```python
from dataclasses import dataclass

from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import InProcessDispatcher, Session


@dataclass(frozen=True)
class Hello:
    message: str


session = Session(dispatcher=InProcessDispatcher())

template = PromptTemplate[Hello](
    ns="demo",
    key="hello",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Say hello. Return JSON with a single field: message.",
        ),
    ],
)

response = ClaudeAgentSDKAdapter().evaluate(Prompt(template), session=session)
print(response.output)  # Hello(message="...")
```

#### 6.4.3 Client and model configuration

**ClaudeAgentSDKClientConfig** controls how the SDK subprocess operates:

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
)

config = ClaudeAgentSDKClientConfig(
    permission_mode="bypassPermissions",  # "default", "acceptEdits", "plan", "bypassPermissions"
    cwd="/path/to/workspace",             # Working directory for SDK
    max_turns=10,                         # Limit conversation turns
    max_budget_usd=1.0,                   # Budget cap in USD
    suppress_stderr=True,                 # Hide CLI noise
    stop_on_structured_output=True,       # Stop after structured output
    betas=("feature-x",),                 # Enable beta features
)

adapter = ClaudeAgentSDKAdapter(client_config=config)
```

| Field | Default | Description |
| -------------------------- | --------------------- | --------------------------------------------------- |
| `permission_mode` | `"bypassPermissions"` | Tool permission handling mode |
| `cwd` | `None` | Working directory (None = current directory) |
| `max_turns` | `None` | Max conversation turns (None = unlimited) |
| `max_budget_usd` | `None` | Budget cap in USD (None = unlimited) |
| `suppress_stderr` | `True` | Hide stderr from Claude Code CLI |
| `stop_on_structured_output`| `True` | Stop immediately after structured output |
| `isolation` | `None` | Custom isolation config (isolation is always enabled) |
| `betas` | `None` | Beta feature identifiers to enable |

**ClaudeAgentSDKModelConfig** controls model-specific parameters:

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKModelConfig,
)

model_config = ClaudeAgentSDKModelConfig(
    model="claude-sonnet-4-5-20250929",
    max_thinking_tokens=4096,  # Extended thinking mode
)

adapter = ClaudeAgentSDKAdapter(model_config=model_config)
```

| Field | Default | Description |
| -------------------- | --------------------------- | -------------------------------------------------- |
| `model` | `"claude-sonnet-4-5-20250929"` | Claude model identifier |
| `max_thinking_tokens`| `None` | Tokens for extended thinking (None = disabled) |

Note: The SDK does not support `seed`, `stop`, `presence_penalty`, or
`frequency_penalty`. Providing these raises `ValueError`.

#### 6.4.4 Workspace management

`ClaudeAgentWorkspaceSection` creates an isolated workspace with host files
mounted in. Claude Code's native tools (Read, Write, Edit, Glob, Grep, Bash)
operate on this workspace.

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentWorkspaceSection,
    HostMount,
)
from weakincentives.runtime import InProcessDispatcher, Session

session = Session(dispatcher=InProcessDispatcher())

workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/abs/path/to/repo",
            mount_path="repo",                           # Appears as "repo/" in workspace
            exclude_glob=(".git/*", "*.pyc", "__pycache__/*"),
            max_bytes=5_000_000,                         # 5MB limit
        ),
    ),
    allowed_host_roots=("/abs/path/to",),  # Security boundary
)

# workspace.temp_dir is the path to pass as cwd
# workspace.cleanup() removes the temp directory when done
```

**HostMount** attributes:

| Field | Default | Description |
| --------------- | ------- | ------------------------------------------------- |
| `host_path` | required| Absolute or relative path to host file/directory |
| `mount_path` | `None` | Relative path in temp dir (default: basename) |
| `include_glob` | `()` | Patterns to include (empty = all) |
| `exclude_glob` | `()` | Patterns to exclude |
| `max_bytes` | `None` | Maximum bytes to copy (None = unlimited) |
| `follow_symlinks`| `False` | Whether to follow symlinks when copying |

**Security:** The `allowed_host_roots` parameter restricts which host paths can
be mounted. Paths outside these roots raise `WorkspaceSecurityError`. Exceeding
`max_bytes` raises `WorkspaceBudgetExceededError`.

The workspace section automatically contributes a `Filesystem` resource that
tools can use. Claude Code's native tools operate directly on the temp
directory.

#### 6.4.5 Isolation configuration

The adapter **always runs in hermetic isolation by default**. This prevents the
SDK from accessing the host's `~/.claude` configuration, credentials, and
session state—ensuring reproducible behavior regardless of the host environment.

The adapter automatically:

1. Creates an ephemeral `HOME` directory containing `.claude/settings.json`
1. Disables alternative providers (AWS Bedrock, etc.) to ensure Anthropic API usage
1. Passes the environment to the SDK subprocess
1. Cleans up the ephemeral directory after execution

Use `IsolationConfig` to customize isolation behavior (network policy, sandbox
settings, environment variables):

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

# Customize isolation with restricted network and explicit API key
isolation = IsolationConfig(
    network_policy=NetworkPolicy.no_network(),
    sandbox=SandboxConfig(enabled=True),
    api_key="sk-ant-...",          # Or uses ANTHROPIC_API_KEY from env
    include_host_env=False,        # Don't inherit host env vars
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(isolation=isolation),
)
```

**IsolationConfig** attributes:

| Field | Default | Description |
| ----------------- | ------- | ---------------------------------------------------- |
| `network_policy` | `None` | Network access constraints (None = no network) |
| `sandbox` | `None` | Sandbox configuration (None = secure defaults) |
| `env` | `None` | Additional env vars for SDK subprocess |
| `api_key` | `None` | API key (None = uses ANTHROPIC_API_KEY from env) |
| `include_host_env`| `False` | Inherit non-sensitive host env vars |
| `skills` | `None` | Skills to mount in hermetic environment |

##### NetworkPolicy

Controls which network resources tools can access. This affects tools making
outbound connections (curl, wget, etc.) but **not** the Claude API connection.

```python
from weakincentives.adapters.claude_agent_sdk import NetworkPolicy

# Block all tool network access
policy = NetworkPolicy.no_network()

# Allow specific domains
policy = NetworkPolicy.with_domains("docs.python.org", "pypi.org")

# Unrestricted (not recommended for production)
policy = NetworkPolicy(allowed_domains=("*",))
```

##### SandboxConfig

Provides programmatic control over OS-level sandboxing (bubblewrap on Linux,
seatbelt on macOS).

```python
from weakincentives.adapters.claude_agent_sdk import SandboxConfig

sandbox = SandboxConfig(
    enabled=True,                          # Enable OS-level sandboxing
    writable_paths=("/tmp/output",),       # Additional writable paths
    readable_paths=("/etc/ssl/certs",),    # Additional readable paths
    excluded_commands=("docker",),         # Commands that bypass sandbox
    allow_unsandboxed_commands=False,      # Require sandbox for all commands
    bash_auto_allow=True,                  # Auto-approve bash in sandbox mode
)
```

| Field | Default | Description |
| ------------------------- | ------- | ------------------------------------------------- |
| `enabled` | `True` | Enable OS-level sandboxing |
| `writable_paths` | `()` | Paths the SDK can write beyond workspace |
| `readable_paths` | `()` | Additional readable paths beyond workspace |
| `excluded_commands` | `()` | Commands that bypass sandbox (use sparingly) |
| `allow_unsandboxed_commands`| `False`| Allow commands outside sandbox |
| `bash_auto_allow` | `True` | Auto-approve Bash when sandboxed |

##### SkillConfig

Mount skills into the hermetic environment. Skills are markdown files
(`SKILL.md`) or directories containing a `SKILL.md` that provide domain-specific
instructions to Claude Code. Skills are copied to `~/.claude/skills/` in the
ephemeral home before Claude Code starts.

```python
from pathlib import Path

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
)
from weakincentives.skills import SkillConfig, SkillMount

# Mount a single skill from a file
isolation = IsolationConfig(
    skills=SkillConfig(
        skills=(
            SkillMount(source=Path("skills/code-review.md")),
        ),
    ),
)

# Mount multiple skills from directories
isolation = IsolationConfig(
    skills=SkillConfig(
        skills=(
            SkillMount(source=Path("demo-skills/code-review")),
            SkillMount(source=Path("demo-skills/ascii-art")),
            # Override the skill name
            SkillMount(source=Path("my-skill.md"), name="custom-name"),
            # Disable a skill without removing from config
            SkillMount(source=Path("experimental"), enabled=False),
        ),
        validate_on_mount=True,  # Default: validate SKILL.md presence
    ),
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(isolation=isolation),
)
```

**SkillMount** attributes:

| Field | Default | Description |
| --------- | ------- | --------------------------------------------------------- |
| `source` | — | Path to skill file (`.md`) or directory with `SKILL.md` |
| `name` | `None` | Override skill name (default: derived from source path) |
| `enabled` | `True` | Whether to mount this skill |

**SkillConfig** attributes:

| Field | Default | Description |
| ------------------- | ------- | ---------------------------------------------- |
| `skills` | `()` | Tuple of `SkillMount` instances |
| `validate_on_mount` | `True` | Validate skill structure before copying |

**Skill structure:**

A skill can be either:

1. **A markdown file** (e.g., `code-review.md`): Copied as `SKILL.md` into a
   directory named after the file (without extension)
1. **A directory** (e.g., `code-review/`): Must contain a `SKILL.md` file. The
   entire directory is copied, preserving structure.

**Validation** (when `validate_on_mount=True`):

- Directory skills must contain `SKILL.md`
- File skills must have `.md` extension
- Individual files limited to 1 MiB
- Total skill size limited to 10 MiB

**Auto-discovering skills:**

```python
from pathlib import Path

from weakincentives.adapters.claude_agent_sdk import IsolationConfig
from weakincentives.skills import SkillConfig, SkillMount

SKILLS_ROOT = Path("demo-skills")

# Find all valid skill directories
skill_mounts = tuple(
    SkillMount(source=skill_dir)
    for skill_dir in SKILLS_ROOT.iterdir()
    if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists()
)

isolation = IsolationConfig(
    skills=SkillConfig(skills=skill_mounts),
)
```

**Creating a skill:**

Create a `SKILL.md` file with instructions for Claude Code:

```markdown
# Code Review Skill

You are a thorough code reviewer. When reviewing code:

## Review Checklist

- [ ] Check for security vulnerabilities
- [ ] Verify error handling covers edge cases
- [ ] Ensure tests cover new functionality

## Output Format

1. **Summary**: One-paragraph overview
2. **Issues**: Problems found (severity: high/medium/low)
3. **Suggestions**: Non-blocking improvements
```

See `demo-skills/` in the repository for example skills. For more information
about Claude Code skills, see [What are Skills?](https://agentskills.io/what-are-skills).

#### 6.4.6 Tool bridging via MCP

WINK tools attached to prompt sections are automatically exposed to Claude Code
as MCP tools under the server key `"wink"`. This lets you keep side effects and
validation in Python while Claude uses the tools natively.

```python
from dataclasses import dataclass

from weakincentives.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from weakincentives.prompt import (
    MarkdownSection,
    Prompt,
    PromptTemplate,
    Tool,
    ToolContext,
    ToolResult,
)
from weakincentives.runtime import InProcessDispatcher, Session


@dataclass(frozen=True)
class MCPSearchParams:
    query: str


@dataclass(frozen=True)
class MCPSearchResult:
    matches: int

    def render(self) -> str:
        return f"Found {self.matches} matches"


def mcp_search(params: MCPSearchParams, *, context: ToolContext) -> ToolResult[MCPSearchResult]:
    # Your search logic here
    return ToolResult(message="ok", value=MCPSearchResult(matches=3))


mcp_search_tool = Tool[MCPSearchParams, MCPSearchResult](
    name="search",
    description="Search the internal index",
    handler=mcp_search,
)

session = Session(dispatcher=InProcessDispatcher())

template = PromptTemplate[None](
    ns="demo",
    key="mcp-tool",
    sections=[
        MarkdownSection(
            title="Task",
            key="task",
            template="Use the search tool for query: weakincentives.",
            tools=(mcp_search_tool,),  # Tool attached here
        ),
    ],
)

response = ClaudeAgentSDKAdapter().evaluate(Prompt(template), session=session)
```

Each bridged tool call:

- Publishes a `ToolInvoked` event to the session
- Executes within a transaction (state rolls back on failure)
- Has access to the full `ToolContext` including session and resources

#### 6.4.7 Events

The adapter publishes these events to the session's dispatcher:

| Event | When | Fields |
| --------------- | --------------------------------- | ------------------------------------- |
| `PromptRendered`| After prompt render, before SDK | `rendered_prompt`, `adapter`, etc. |
| `ToolInvoked` | Each native + bridged tool call | `tool_name`, `params`, `result`, etc. |
| `PromptExecuted`| After completion | `result`, `usage` (TokenUsage) |

Native Claude Code tools (Read, Write, Bash, etc.) are tracked via SDK hooks
and also publish `ToolInvoked` events.

#### 6.4.8 Complete example: secure code review

This example combines workspace management, isolation, and structured output
for a secure code review agent:

```python
from dataclasses import dataclass

from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    ClaudeAgentWorkspaceSection,
    HostMount,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import InProcessDispatcher, Session


@dataclass(frozen=True)
class Review:
    summary: str
    findings: list[str]


session = Session(dispatcher=InProcessDispatcher())

# Create workspace with mounted repository
workspace = ClaudeAgentWorkspaceSection(
    session=session,
    mounts=(
        HostMount(
            host_path="/abs/path/to/repo",
            mount_path="repo",
            exclude_glob=(".git/*", "*.pyc", "__pycache__/*"),
            max_bytes=5_000_000,
        ),
    ),
    allowed_host_roots=("/abs/path/to",),
)

try:
    # Configure adapter with isolation (no tool network access)
    adapter = ClaudeAgentSDKAdapter(
        client_config=ClaudeAgentSDKClientConfig(
            permission_mode="bypassPermissions",
            cwd=str(workspace.temp_dir),
            isolation=IsolationConfig(
                network_policy=NetworkPolicy.no_network(),
                sandbox=SandboxConfig(
                    enabled=True,
                    readable_paths=(str(workspace.temp_dir),),
                ),
            ),
        ),
    )

    template = PromptTemplate[Review](
        ns="review",
        key="security",
        sections=[
            MarkdownSection(
                title="Task",
                key="task",
                template=(
                    "Review the code in repo/ for security issues. "
                    "Return JSON with: summary (string), findings (list of strings)."
                ),
            ),
            workspace,
        ],
    )

    response = adapter.evaluate(Prompt(template), session=session)
    print(response.output)  # Review(summary="...", findings=["...", ...])
finally:
    workspace.cleanup()
```

#### 6.4.9 Docs assistant with domain allowlist

For agents that need controlled network access:

```python
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    ClaudeAgentSDKClientConfig,
    IsolationConfig,
    NetworkPolicy,
    SandboxConfig,
)

adapter = ClaudeAgentSDKAdapter(
    client_config=ClaudeAgentSDKClientConfig(
        permission_mode="bypassPermissions",
        isolation=IsolationConfig(
            network_policy=NetworkPolicy.with_domains(
                "docs.python.org",
                "pypi.org",
            ),
            sandbox=SandboxConfig(enabled=True),
        ),
    ),
)
```

Tools can now access `docs.python.org` and `pypi.org` but no other domains.

#### 6.4.10 Operational notes

- **Token tracking:** Pass a `BudgetTracker` to `evaluate(..., budget_tracker=...)`
  to track usage across multiple evaluations.
- **Structured output:** `stop_on_structured_output=True` (default) stops the
  agent immediately after the StructuredOutput tool runs, ensuring clean turn
  termination.
- **Windows:** Sandbox settings may not be enforced; HOME redirection still
  applies.
- **Extended thinking:** Set `max_thinking_tokens` in `ClaudeAgentSDKModelConfig`
  to enable Claude's extended thinking mode (requires minimum ~1024 tokens).

______________________________________________________________________

## 7. Orchestration with MainLoop

_Canonical spec: [specs/MAIN_LOOP.md](specs/MAIN_LOOP.md)_

`MainLoop` exists for one reason:

> Make progressive disclosure and budgets/deadlines easy to handle correctly.

You could write the loop yourself. MainLoop just does it in a tested, consistent
way.

### 7.1 The minimal MainLoop

You implement:

- `prepare(request) -> tuple[Prompt[OutputT], Session]`

Then call `loop.execute(request)`.

```python
from weakincentives.runtime import MainLoop, MainLoopConfig, Session
from weakincentives.prompt import Prompt

class MyLoop(MainLoop[RequestType, OutputType]):
    def prepare(self, request: RequestType) -> tuple[Prompt[OutputType], Session]:
        prompt = Prompt(self._template).bind(request)
        session = Session(tags={"loop": "my-loop"})
        return prompt, session
```

`MainLoop` also catches `VisibilityExpansionRequired` and retries automatically.
When the model calls `open_sections`, MainLoop applies the visibility overrides
and re-evaluates the prompt. You don't have to handle this yourself.

**Configuring MainLoop with resources:**

You can inject custom resources at the loop level via `MainLoopConfig`:

```python
from typing import Any

from weakincentives.resources import Binding, ResourceRegistry
from weakincentives.runtime import MainLoopConfig


# Example resource types (from previous example)
class Config:
    def __init__(self) -> None:
        self.url: str = "https://api.example.com"


class HTTPClient:
    def __init__(self, url: str = "") -> None:
        self.url = url


# Type stubs (defined in your application)
MyLoop: Any = ...  # type: ignore[assignment]
adapter: Any = ...  # type: ignore[assignment]
dispatcher: Any = ...  # type: ignore[assignment]
request: Any = ...  # type: ignore[assignment]
http_client: Any = ...  # type: ignore[assignment]

# Simple case: pre-constructed instances
resources = ResourceRegistry.of(Binding.instance(HTTPClient, http_client))

# Or with lazy construction and scopes
resources = ResourceRegistry.of(
    Binding(Config, lambda r: Config()),
    Binding(HTTPClient, lambda r: HTTPClient(url=r.get(Config).url)),  # type: ignore[attr-defined]
)

config = MainLoopConfig(resources=resources)
loop = MyLoop(adapter=adapter, dispatcher=dispatcher, config=config)
response, session = loop.execute(request)
```

Resources configured this way are available to all tool handlers during
execution. You can also pass resources directly to
`loop.execute(request, resources=...)` for per-request overrides.

### 7.2 Deadlines and budgets

`Deadline` is a wall-clock deadline. `Budget` can include token limits and/or a
deadline. `BudgetTracker` accumulates usage across retries.

**Typical pattern:**

```python
from datetime import datetime, timedelta, UTC
from typing import Any

from weakincentives import Deadline, Budget


# Type stubs (defined in your application)
loop: Any = ...  # type: ignore[assignment]
request: Any = ...  # type: ignore[assignment]

deadline = Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=30))
budget = Budget(max_total_tokens=20_000)

response, session = loop.execute(request, deadline=deadline, budget=budget)
```

Deadlines prevent runaway agents. Budgets prevent runaway costs. Both are
enforced at the adapter level, so they work consistently across providers.

______________________________________________________________________

## 8. Evaluation with EvalLoop

_Canonical spec: [specs/EVALS.md](specs/EVALS.md)_

Evaluation is built on the same composition pattern as everything else in WINK:
**EvalLoop wraps MainLoop**. Rather than a separate evaluation framework, evals
are just another way to drive your existing MainLoop with datasets and scoring.

This means a worker in production can run both your regular agent logic
(`MainLoop`) and your evaluation suite (`EvalLoop`) side by side—same prompt
templates, same tools, same adapters. Canary deployments become natural: run
evals against your production configuration before rolling out changes.

### 8.1 The composition philosophy

```
┌─────────────┐     ┌──────────┐     ┌───────────┐     ┌────────────┐
│   Dataset   │────▶│ EvalLoop │────▶│ MainLoop  │────▶│  Adapter   │
│  (samples)  │     │.execute()│     │ .execute()│     │            │
└─────────────┘     └────┬─────┘     └───────────┘     └────────────┘
                         │
                         ▼
                   ┌───────────┐     ┌────────────┐
                   │ Evaluator │────▶│ EvalReport │
                   │ (scoring) │     │ (metrics)  │
                   └───────────┘     └────────────┘
```

`EvalLoop` orchestrates evaluation: for each sample, it executes through the
provided `MainLoop`, scores the output with an evaluator function, and
aggregates results into a report. You already have a `MainLoop`—evals just add
datasets and scoring.

### 8.2 Core types

**Sample and Dataset:**

```python
from weakincentives.evals import Dataset, Sample

# A sample pairs input with expected output
sample = Sample(
    id="math-1",
    input="What is 2 + 2?",
    expected="4",
)

# Datasets are immutable collections of samples
dataset = Dataset(samples=(sample,))

# Or load from JSONL
dataset = Dataset.load(Path("qa.jsonl"), str, str)
```

**Score and Evaluator:**

Evaluators are pure functions—no side effects, no state:

```python
from weakincentives.evals import Score, Evaluator

def my_evaluator(output: str, expected: str) -> Score:
    passed = expected.lower() in output.lower()
    return Score(
        value=1.0 if passed else 0.0,
        passed=passed,
        reason="Found expected answer" if passed else "Missing expected answer",
    )
```

**Built-in evaluators:**

```python
from typing import cast
from weakincentives.evals import exact_match, contains, all_of, any_of, Score, Evaluator

# Strict equality
score = exact_match("hello", "hello")  # passed=True

# Substring presence
score = contains("The answer is 42.", "42")  # passed=True


# Define custom evaluators for combining
def my_custom_check(output: object, expected: object) -> Score:
    return Score(value=1.0, passed=True)


def fuzzy_match(output: object, expected: object) -> Score:
    return Score(value=0.9, passed=True)


# Combine evaluators (cast needed due to strict typing)
evaluator = all_of(cast(Evaluator, contains), my_custom_check)  # All must pass
evaluator = any_of(cast(Evaluator, exact_match), fuzzy_match)  # At least one must pass
```

### 8.3 LLM-as-judge

For subjective criteria, use an LLM to score outputs. The judge selects from a
fixed set of rating labels that map to values:

```python
from typing import cast
from weakincentives.evals import llm_judge, all_of, Evaluator, contains
from weakincentives.adapters.openai import OpenAIAdapter

# Use a smaller model for judging
judge_adapter = OpenAIAdapter(model="gpt-4o-mini")

evaluator = all_of(
    cast(Evaluator, contains),  # Cast for strict type compatibility
    llm_judge(judge_adapter, "Response is helpful"),  # type: ignore[arg-type]
    llm_judge(judge_adapter, "No hallucinated info"),  # type: ignore[arg-type]
)
```

The `llm_judge` factory creates an evaluator that prompts the model to rate
outputs as "excellent", "good", "fair", "poor", or "wrong"—each mapping to a
numeric value.

### 8.4 Session evaluators

Sometimes you need to evaluate not just _what_ the agent produced, but _how_ it
got there. Session evaluators receive a read-only `SessionView` and can assert
on tool usage patterns, token budgets, and custom state invariants.

**Built-in session evaluators:**

```python
from weakincentives.evals import (
    tool_called,
    tool_not_called,
    tool_call_count,
    all_tools_succeeded,
    token_usage_under,
    slice_contains,
    all_of,
    adapt,
)

# Combine output evaluation with behavioral assertions
evaluator = all_of(
    exact_match,                           # Output must match expected
    tool_called("search"),                 # Agent must have used search
    tool_not_called("fallback"),           # Should not have used fallback
    all_tools_succeeded(),                 # No tool failures
    token_usage_under(max_tokens=5000),    # Stay within budget
)
```

**Available session evaluators:**

- `tool_called(name)` - Assert a specific tool was invoked
- `tool_not_called(name)` - Assert a tool was never invoked
- `tool_call_count(name, min_count, max_count)` - Assert call count within
  bounds
- `all_tools_succeeded()` - Assert no tool failures occurred
- `token_usage_under(max_tokens)` - Assert total token usage under limit
- `slice_contains(T, predicate)` - Assert session slice contains matching value

**Converting standard evaluators:**

Standard evaluators (that only see output and expected) can be converted to
session-aware evaluators using `adapt()`:

```python
from weakincentives.evals import adapt, exact_match, all_of, tool_called

# adapt() wraps a standard evaluator to ignore the session
evaluator = all_of(
    adapt(exact_match),    # Now works with session evaluators
    tool_called("search"),
)
```

**Custom session evaluators:**

```python
from weakincentives.evals import Score, SessionEvaluator
from weakincentives.runtime.session import SessionView
from weakincentives.runtime import ToolInvoked

def custom_session_check(
    output: str,
    expected: str,
    session: SessionView,
) -> Score:
    # Check how many tools were called
    tool_events = session[ToolInvoked].all()
    if len(tool_events) > 10:
        return Score(value=0.0, passed=False, reason="Too many tool calls")
    return Score(value=1.0, passed=True)
```

### 8.5 Running evaluations

**EvalLoop wraps your MainLoop:**

```python
from typing import Any, cast
from weakincentives.evals import EvalLoop, EvalRequest, EvalResult, exact_match, Evaluator
from weakincentives.runtime import InMemoryMailbox, MainLoop

# Your existing MainLoop (defined elsewhere in your application)
main_loop: MainLoop[Any, str] = ...  # type: ignore[assignment]

# Create mailbox for evaluation requests (uses reply_to for result routing)
eval_requests: InMemoryMailbox[EvalRequest[str, str], EvalResult] = InMemoryMailbox(
    name="eval-requests"
)

# Create EvalLoop wrapping your MainLoop
eval_loop = EvalLoop(
    loop=main_loop,
    evaluator=cast(Evaluator, exact_match),  # Cast for strict type compatibility
    requests=eval_requests,  # type: ignore[arg-type]
)
```

**Submit samples and collect results:**

```python
from typing import Any
from weakincentives.evals import submit_dataset, collect_results, Dataset, EvalResult
from weakincentives.runtime.mailbox import InMemoryMailbox, RegistryResolver

# Dataset defined in your application
eval_dataset: Dataset[str, str] = ...  # type: ignore[assignment]

# Create mailboxes
eval_requests_mailbox: InMemoryMailbox[Any, Any] = InMemoryMailbox(name="eval-requests")
eval_results_mailbox: InMemoryMailbox[EvalResult, None] = InMemoryMailbox(
    name="eval-results"
)

# Submit all samples to the requests mailbox with reply_to mailbox reference
submit_dataset(eval_dataset, eval_requests_mailbox, reply_to=eval_results_mailbox)  # type: ignore[arg-type]

# Run the evaluation worker
eval_loop.run(max_iterations=1)

# Collect results into a report
report = collect_results(eval_results_mailbox, expected_count=len(eval_dataset))

# Inspect the report
print(f"Pass rate: {report.pass_rate:.1%}")
print(f"Mean score: {report.mean_score:.2f}")
print(f"Mean latency: {report.mean_latency_ms:.0f}ms")

# Review failures
for failed in report.failed_samples():
    print(f"Failed: {failed.sample_id} - {failed.score.reason}")
```

### 8.6 Production deployment pattern

In production, run both `MainLoop` and `EvalLoop` workers from the same process
or container. This ensures your evaluation suite runs against the exact same
configuration as your production agent:

```python
from threading import Thread
from typing import Any, cast
from weakincentives.contrib.mailbox import RedisMailbox
from weakincentives.evals import EvalLoop, exact_match, Evaluator
from weakincentives.runtime import MainLoop

# Type stubs for external dependencies (defined in your application)
prod_redis_client: Any = ...  # type: ignore[assignment]
prod_main_loop: MainLoop[Any, str] = ...  # type: ignore[assignment]

# Production mailboxes (Redis-backed for durability)
prod_requests: RedisMailbox[Any, Any] = RedisMailbox(
    name="agent-requests", client=prod_redis_client
)
prod_results: RedisMailbox[Any, None] = RedisMailbox(
    name="agent-results", client=prod_redis_client
)

prod_eval_requests: RedisMailbox[Any, Any] = RedisMailbox(
    name="eval-requests", client=prod_redis_client
)


# Production worker
def run_production() -> None:
    while True:
        for msg in prod_requests.receive():
            response, _session = prod_main_loop.execute(msg.body)
            prod_results.send(response)
            msg.acknowledge()


# Eval worker (wraps the same MainLoop, uses reply_to for result routing)
prod_eval_loop = EvalLoop(
    loop=prod_main_loop,
    evaluator=cast(Evaluator, exact_match),  # Cast for strict type compatibility
    requests=prod_eval_requests,
)

# Run both in parallel
Thread(target=run_production, daemon=True).start()
prod_eval_loop.run()  # Blocks, processing eval requests
```

**Canary deployment:**

Before rolling out prompt or configuration changes, submit your eval dataset to
the new worker and verify the pass rate meets your threshold:

```python
# Submit eval dataset to canary worker
submit_dataset(regression_dataset, canary_eval_requests)

# Collect and check results
report = collect_results(canary_eval_results, expected_count=len(regression_dataset))

if report.pass_rate < 0.95:
    raise RollbackError(f"Canary failed: {report.pass_rate:.1%} pass rate")
```

This is the control plane for safe model upgrades: verify behavior
programmatically before promoting changes.

### 8.7 Reply-to routing

When workers need to send results to dynamic destinations (not a fixed result
mailbox), use the `reply_to` pattern. The worker derives the response
destination from the incoming message:

```python
from dataclasses import dataclass
from typing import Any
from weakincentives.runtime.mailbox import InMemoryMailbox, RegistryResolver

@dataclass(frozen=True)
class AnalysisRequest:
    query: str

def process(body: Any) -> Any:
    ...  # type: ignore[empty-body]

# Setup: resolver maps identifiers to mailboxes
# Create request and response mailboxes
requests: InMemoryMailbox[Any, Any] = InMemoryMailbox(name="requests")
client_responses: InMemoryMailbox[Any, None] = InMemoryMailbox(name="client-123")

# Client sends request with reply_to mailbox reference
requests.send(
    body=AnalysisRequest(query="Find all bugs"),  # type: ignore[arg-type]
    reply_to=client_responses,  # Mailbox to send the result to
)

# Worker processes and replies
for msg in requests.receive():
    result = process(msg.body)
    msg.reply(result)       # Sends directly to client_responses mailbox
    msg.acknowledge()
```

**Eval run collection** is a natural fit for this pattern. All samples specify
the same `reply_to`, and results collect into one mailbox regardless of which
worker processes each sample:

```python
from typing import Any
from uuid import uuid4
from weakincentives.contrib.mailbox import RedisMailbox, RedisMailboxFactory
from weakincentives.evals import EvalRequest, Sample
from weakincentives.runtime.mailbox import CompositeResolver

# External dependencies (defined in your application)
redis_client: Any = ...  # type: ignore[assignment]
eval_samples: list[Sample[str, str]] = []

# Factory creates mailboxes on demand for reconstructing from names
factory: RedisMailboxFactory[Any] = RedisMailboxFactory(client=redis_client)
redis_resolver: CompositeResolver[Any] = CompositeResolver(registry={}, factory=factory)

redis_requests: RedisMailbox[Any, Any] = RedisMailbox(
    name="eval-requests", client=redis_client, reply_resolver=redis_resolver
)

# Create run-specific results mailbox
run_id = f"eval-run-{uuid4()}"
results_mailbox: RedisMailbox[Any, None] = RedisMailbox(
    name=run_id, client=redis_client
)

# Submit all samples with mailbox reference (name serialized to Redis)
for sample in eval_samples:
    redis_requests.send(
        body=EvalRequest(sample=sample),  # type: ignore[arg-type]
        reply_to=results_mailbox,  # All results go to same mailbox
    )

# Collect results from the results mailbox
collected: list[Any] = []
while len(collected) < len(eval_samples):
    for msg in results_mailbox.receive(wait_time_seconds=5):
        collected.append(msg.body)
        msg.acknowledge()
```

**Multiple replies** are also supported. Workers can send progress updates
before the final result:

```python
for msg in requests.receive():
    msg.reply(Progress(step=1, status="Analyzing..."))
    msg.reply(Progress(step=2, status="Generating fix..."))
    msg.reply(Complete(result=fix))
    msg.acknowledge()
```

See `specs/MAILBOX.md` for the full resolver protocol and advanced
patterns like multi-tenant isolation.

______________________________________________________________________

## 9. Lifecycle Management

_Canonical spec: [specs/HEALTH.md](specs/HEALTH.md)_

When running agents in production—especially in containerized environments like
Kubernetes—you need coordinated shutdown, health monitoring, and watchdog
protection. WINK provides lifecycle primitives that integrate with MainLoop and
EvalLoop.

### 9.1 LoopGroup: running multiple loops

`LoopGroup` runs multiple loops in separate threads with coordinated shutdown
and optional health endpoints:

```python
from weakincentives.runtime import LoopGroup

# Run MainLoop and EvalLoop together
group = LoopGroup(loops=[main_loop, eval_loop])
group.run()  # Blocks until SIGTERM/SIGINT
```

For Kubernetes deployments, enable health endpoints and watchdog monitoring:

```python
group = LoopGroup(
    loops=[main_loop],
    health_port=8080,           # Exposes /health/live and /health/ready
    watchdog_threshold=720.0,   # Terminate if worker stalls for 12 minutes
)
group.run()
```

**Key features:**

- **Health endpoints**: `/health/live` (liveness) and `/health/ready`
  (readiness) for Kubernetes probes
- **Watchdog monitoring**: Detects stuck workers and terminates the process via
  SIGKILL when heartbeats stall
- **Coordinated shutdown**: SIGTERM/SIGINT triggers graceful shutdown of all
  loops

### 9.2 ShutdownCoordinator: manual signal handling

For finer control, use `ShutdownCoordinator` directly:

```python
from weakincentives.runtime import ShutdownCoordinator

coordinator = ShutdownCoordinator.install()
coordinator.register(loop.shutdown)
loop.run()
```

The coordinator installs signal handlers for SIGTERM and SIGINT. When a signal
arrives, all registered callbacks are invoked in registration order.

### 9.3 The Runnable protocol

Both `MainLoop` and `EvalLoop` implement the `Runnable` protocol:

```python
from typing import Protocol
from weakincentives.runtime import Heartbeat


class Runnable(Protocol):
    """Protocol for loops managed by LoopGroup."""

    def run(self, *, max_iterations: int | None = None) -> None: ...
    def shutdown(self, *, timeout: float = 30.0) -> bool: ...

    @property
    def running(self) -> bool: ...

    @property
    def heartbeat(self) -> Heartbeat | None: ...
```

This enables `LoopGroup` to manage any compliant loop implementation.

### 9.4 Health and watchdog configuration

The watchdog monitors heartbeats from loops and terminates the process if any
loop stalls beyond the threshold. This prevents "stuck worker" scenarios where a
loop hangs indefinitely without processing requests.

```python
group = LoopGroup(
    loops=[main_loop, eval_loop],
    health_port=8080,           # Health endpoint port
    health_host="0.0.0.0",      # Bind to all interfaces
    watchdog_threshold=720.0,   # 12 minutes (calibrated for 10-min prompts)
    watchdog_interval=60.0,     # Check every minute
)
```

**Timeout calibration:**

- `watchdog_threshold` should exceed your maximum expected prompt evaluation
  time
- `visibility_timeout` (in `run()`) should exceed `watchdog_threshold` to
  prevent message redelivery during long evaluations

______________________________________________________________________

## 10. Progressive disclosure

_Canonical spec: [specs/PROMPTS.md](specs/PROMPTS.md) (Progressive Disclosure
section)_

Long prompts are expensive. Progressive disclosure is WINK's first-class
solution:

- Sections can render as `SUMMARY` by default
- The model can request expansion via `open_sections`

This keeps initial prompts lean while giving the model access to details when
needed.

### 10.1 SectionVisibility: FULL vs SUMMARY

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

### 10.2 open_sections and read_section

When summarized sections exist, WINK injects builtin tools:

- `open_sections(section_keys, reason)` → raises `VisibilityExpansionRequired`
- `read_section(section_key)` → returns full rendered markdown for that section

`MainLoop` handles `open_sections` automatically by setting overrides and
retrying. The model asks to expand a section, MainLoop applies the expansion,
and evaluation continues with the full content visible.

`read_section` is different: it returns the content without changing visibility.
The section remains summarized in subsequent turns. Use this for reference
material that the model only needs temporarily.

### 10.3 Visibility overrides in session state

Visibility overrides live in the `VisibilityOverrides` session slice and are
applied at render time.

```python
from weakincentives.prompt import SectionVisibility
from weakincentives.runtime.session import SetVisibilityOverride, VisibilityOverrides

session.dispatch(
    SetVisibilityOverride(path=("reference",), visibility=SectionVisibility.FULL)
)
```

This is what happens under the hood when the model calls `open_sections`.
MainLoop applies the override and re-renders the prompt.

______________________________________________________________________

## 11. Prompt overrides and optimization

_Canonical spec: [specs/PROMPTS.md](specs/PROMPTS.md)_

Overrides are how WINK supports fast iteration without code edits:

- Keep prompt templates stable in code
- Store patch files on disk
- Validate patches with hashes

This separation matters. Your templates are code: tested, reviewed, versioned.
Overrides are configuration: easy to tweak without a deploy.

### 11.1 Hash-based safety: override only what you intended

Overrides are validated against a `PromptDescriptor`:

- Each overridable section has a `content_hash`
- Each overridable tool has a `contract_hash`

If hashes don't match, WINK refuses to apply the override. This prevents a
common failure mode: you edit a section in code, but an old override still
applies, and you're running something different than you tested.

### 11.2 LocalPromptOverridesStore

The default store is `LocalPromptOverridesStore`, which writes JSON files under:

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

### 11.3 Override file format

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
      "param_descriptions": { "query": "Keywords" }
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

### 11.4 A practical override workflow

A workflow that works well in teams:

1. **Seed** override files from the current prompt
   (`store.seed(prompt, tag="v1")`)
1. **Run** your agent and collect failures / quality notes
1. **Edit** override sections directly (or generate them with an optimizer)
1. **Re-run** tests/evals
1. **Commit** override files alongside code

For "hardening", disable overrides on sensitive sections/tools with
`accepts_overrides=False`. This prevents accidental changes to security-critical
text.

______________________________________________________________________

## 12. Workspace tools

_Canonical spec: [specs/WORKSPACE.md](specs/WORKSPACE.md)_

WINK includes several tool suites aimed at background agents that need to
inspect and manipulate a repository safely. They live in
`weakincentives.contrib.tools`.

### 12.1 PlanningToolsSection

**Tools:**

- `planning_setup_plan`
- `planning_add_step`
- `planning_update_step`
- `planning_read_plan`

The plan is stored in session state and updated via reducers. Each step has an
ID, title, details, and status.

Use it when you want the model to externalize its plan without inventing its own
format. Many models plan better when they have explicit tools for planning.

### 12.2 VfsToolsSection

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

### 12.3 WorkspaceDigestSection

Renders a cached repo digest stored in session state. The digest is a structured
summary of the repository: file tree, key files, detected patterns.

It works well with progressive disclosure: default to `SUMMARY`, expand on
demand. The model gets an overview without the full file contents.

### 12.4 AstevalSection

Exposes `evaluate_python` (safe-ish expression evaluation) with captured
stdout/stderr.

`asteval` restricts what Python code can do: no imports, no file access, no
network. Useful for small transformations (string formatting, arithmetic)
without granting shell access.

**Install:** `pip install "weakincentives[asteval]"`

### 12.5 PodmanSandboxSection

Runs shell commands and Python evaluation inside a Podman container.

Use it when you need strict isolation and reproducible execution (tests,
linters). The container provides a clean environment; writes don't affect the
host.

**Install:** `pip install "weakincentives[podman]"`

### 12.6 Wiring a workspace into a prompt

A practical pattern (also used by `code_reviewer_example.py` in this repo):

```python
from typing import Any
from weakincentives.contrib.tools import (
    PlanningToolsSection,
    PlanningStrategy,
    VfsToolsSection,
    VfsConfig,
    WorkspaceDigestSection,
)
from weakincentives.contrib.tools.vfs_types import HostMount as VfsHostMount
from weakincentives.prompt import PromptTemplate, MarkdownSection
from weakincentives.runtime import Session


def build_repo_agent_template(*, session: Session) -> PromptTemplate[Any]:
    vfs_mounts: tuple[VfsHostMount, ...] = (
        VfsHostMount(host_path="src"),
        VfsHostMount(host_path="README.md"),
    )
    vfs = VfsToolsSection(
        session=session,
        config=VfsConfig(
            mounts=vfs_mounts,
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

______________________________________________________________________

## 13. Debugging and observability

### 13.1 Structured logging

```python
from weakincentives.runtime import configure_logging, get_logger

configure_logging(level="INFO", json_mode=True)
logger = get_logger(__name__)
logger.info("hello", event="demo.hello", context={"foo": "bar"})
```

Logs include structured `event` and `context` fields for downstream routing and
analysis. JSON mode makes logs machine-parseable.

### 13.2 Session events

Sessions subscribe to the event dispatcher and capture telemetry events like:

- `PromptRendered`: emitted when a prompt is rendered
- `ToolInvoked`: emitted when a tool is called (includes params, result, timing)
- `PromptExecuted`: emitted when a prompt evaluation completes (includes token
  usage)
- `TokenUsage`: token counts from provider responses

You can use these for your own tracing pipeline. Subscribe to the dispatcher and route
events wherever you need them.

### 13.3 Dumping snapshots to JSONL

Use `weakincentives.debug.dump_session(...)` to persist a session tree:

```python
from weakincentives.debug import dump_session

path = dump_session(session, target="snapshots/")  # writes <session_id>.jsonl
```

Each line is one serialized session snapshot (root → leaves). The JSONL format
is stable and human-readable.

### 13.4 The debug UI

**Install:** `pip install "weakincentives[wink]"`

**Run:**

```bash
wink debug snapshots/<session_id>.jsonl
```

This starts a local server that renders the prompt/tool timeline for inspection.
You can see exactly what was sent to the model, what tools were called, and how
state evolved.

______________________________________________________________________

## 14. Testing and reliability

WINK is designed so that most of your "agent logic" is testable without a model.

**Practical approach:**

1. **Prompt rendering tests**: render prompts with fixed params and assert exact
   markdown (snapshot tests). These run fast and catch template regressions.

1. **Tool handler tests**: call handlers directly with fake `ToolContext` +
   resources. No model needed. Test the dispatcheriness logic in isolation.

1. **Reducer tests**: test state transitions as pure functions. Given this slice
   and this event, expect this new slice.

1. **Integration tests**: run `adapter.evaluate` behind a flag (and record
   sessions). These are slow and cost money, so run them selectively.

**A prompt snapshot test:**

```python
from dataclasses import dataclass
from typing import Any
from weakincentives.prompt import Prompt, PromptTemplate, MarkdownSection
from weakincentives.runtime import Session


@dataclass(frozen=True)
class TestParams:
    question: str


template = PromptTemplate[Any](
    ns="test",
    key="snapshot",
    sections=(MarkdownSection(title="Q", key="q", template="Question: ${question}"),),
)
prompt = Prompt(template)
session = Session()


def test_prompt_renders_stably() -> None:
    rendered = prompt.bind(TestParams(question="x")).render(session=session)
    assert "Question: x" in rendered.text
```

The test doesn't call a model. It just verifies that the prompt renders as
expected. When prompts are deterministic, you can test them like regular code.

______________________________________________________________________

## 15. Approach to code quality

WINK applies strict quality gates that go beyond typical Python projects. These
gates exist because agent code has unusual failure modes: type mismatches
surface mid-conversation, subtle bugs can cause cascading failures across tool
calls, and security vulnerabilities in tool handlers can have serious
consequences.

The gates aren't bureaucracy—they're aligned with the "weak incentives"
philosophy. Just as we design prompts to make correct model behavior natural, we
design the codebase to make correct code natural. Strict types catch errors at
construction time. Contracts document and enforce invariants. Coverage ensures
tests exercise all code paths.

### 15.1 Strict type checking

WINK enforces pyright strict mode. Type annotations are the source of truth:

```python
# Pyright catches this at edit time, not runtime
def handler(params: MyParams, *, context: ToolContext) -> ToolResult[MyResult]:
    return ToolResult(message="ok", value=None)  # Error: expected MyResult, got None
```

**Why this matters for agents:**

- Tool params and results are serialized/deserialized automatically. Type
  mismatches that would cause runtime failures are caught at construction.
- Session slices are keyed by type. A typo in a type annotation silently creates
  a separate slice.
- Adapters use type information to generate JSON schemas. Wrong types mean wrong
  schemas sent to the model.

**Practical implications:**

- Every function has type annotations
- Use `slots=True, frozen=True` dataclasses for immutable data
- Avoid `Any` except where truly necessary
- Run `make typecheck` (or your IDE's type checker) frequently

### 15.2 Design-by-contract

Public APIs use decorators from `weakincentives.dbc`:

```python
from weakincentives.dbc import require, ensure, invariant, pure

@require(lambda x: x > 0)  # x must be positive
@ensure(lambda result: result >= 0)  # result must be non-negative
def compute(x: int) -> int:
    ...

@pure  # Marks function as having no side effects
def render_template(template: str, params: dict[str, object]) -> str:
    ...
```

**What the decorators do:**

- `@require`: precondition checked on entry
- `@ensure`: postcondition checked on exit
- `@invariant`: class invariant checked after each method
- `@pure`: documents (and can verify) side-effect-free functions

**Why this matters for agents:**

- Contracts document expectations that types can't express ("non-empty list",
  "valid path", "positive budget")
- Violations fail fast with clear messages, not silently corrupted state
- The `@pure` marker identifies deterministic functions—important for
  understanding what can be snapshotted/replayed

**When to use contracts:**

- Public API boundaries
- Tool handlers (validate params beyond type checking)
- Reducers (invariants on state transitions)
- Anywhere a comment would say "assumes X" or "requires Y"

Read `specs/DBC.md` before modifying DbC-decorated modules.

For safety-critical state machines (like the mailbox algorithms), WINK also
supports **formal verification** with embedded TLA+ specifications. See
[Appendix C: Formal Verification](#appendix-c-formal-verification-with-tla) for
details.

### 15.3 Coverage requirements

WINK requires 100% line coverage for `src/weakincentives/`. This is enforced by
pytest-cov in CI and blocks merges if coverage drops below 100%.

**Running the tests:**

```bash
make test           # Coverage-gated unit tests
```

### 15.4 Security scanning

Agent code often handles untrusted input (user requests, model outputs) and
performs privileged operations (file access, command execution). Security
scanning is not optional.

**Tools:**

- **Bandit**: static analysis for common Python security issues
- **Deptry**: finds unused, missing, or misplaced dependencies
- **pip-audit**: checks dependencies for known vulnerabilities

**These run automatically in CI.** You can also run them locally:

```bash
make bandit      # Security analysis
make deptry      # Dependency hygiene
make pip-audit   # Vulnerability scan
```

**Security considerations for tool handlers:**

- Never pass unsanitized model output to shell commands
- Validate file paths against allowed roots
- Use VFS or sandboxes for file operations when possible
- Avoid pickle, eval, or exec on untrusted data

### 15.5 Quality gates in practice

All gates are combined in `make check`:

```bash
make check  # Runs: format, lint, typecheck, test, bandit, deptry, pip-audit
```

**Before every commit:**

1. Run `make check`
1. Fix any failures
1. Commit only when clean

**Pre-commit hooks enforce this.** After running `./install-hooks.sh`, commits
are blocked unless `make check` passes.

**Why the gates are strict:**

Agent systems have compounding failure modes. A type mismatch in a tool param
causes a serialization error, which causes a tool failure, which causes the
model to retry with bad assumptions, which causes a cascade of wasted tokens and
incorrect behavior. Catching errors early—at the type level, at the contract
level, at the test level—prevents these cascades.

The 100% coverage requirement isn't about the number. It's about the habit:
every line of code should have a reason to exist, and that reason should be
testable. If a line can't be tested, it probably shouldn't exist.

______________________________________________________________________

## 16. Recipes

These are intentionally opinionated. They reflect the "weak incentives" style:
reduce surprise, keep state explicit, and make side effects auditable.

### 16.1 A code-review agent

See `code_reviewer_example.py` in this repo for a full, runnable example that
demonstrates:

- Workspace digest + progressive disclosure
- Planning tools
- VFS or Podman sandbox
- Prompt overrides and optimizer hooks
- Structured output review responses

This is the canonical "put it all together" example. Read it after you
understand the individual pieces.

### 16.2 A repo Q&A agent

**Goal**: answer questions about a codebase quickly.

**Pattern**:

- Show workspace digest summary by default
- Allow the model to expand it via `read_section`
- Allow VFS `grep`/`glob`/`read_file` for verification

The model sees a summary, asks questions, digs into details as needed. Token
usage stays low for simple questions.

### 16.3 A "safe patch" agent

**Goal**: generate a patch but avoid uncontrolled writes.

**Pattern**:

- Use VFS tools for edits (writes go to the virtual copy, not the host)
- Require the model to output a diff as structured output
- Optionally run tests in Podman before proposing the patch

The model can experiment freely in the VFS. Only the final diff matters. Humans
review the diff before applying it to the real repo.

### 16.4 A research agent with progressive disclosure

**Goal**: answer deep questions without stuffing a giant blob into the prompt.

**Pattern**:

- Store sources as summarized sections
- Let the model open only what it needs
- Keep an audit trail via session snapshots

Progressive disclosure shines here. The model starts with summaries, expands
relevant sources, and cites its sources. The session log shows exactly which
sources it used.

______________________________________________________________________

## 17. Troubleshooting

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
        return ToolResult.error("Failed")
    return ToolResult.ok(result, message="OK")
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
1. Add tool examples to show correct usage
1. Check that the tool description accurately describes what it does

### "DeadlineExceededError"

The agent ran past its deadline.

**Fixes**:

1. Increase the deadline
1. Reduce prompt size (use progressive disclosure)
1. Check for tool handlers that hang or take too long

### Session state not persisting

State changes aren't visible in subsequent queries.

**Fix**: Make sure you're using the same session instance, and that you've
registered reducers for your event types:

```python
session[Plan].register(AddStep, my_reducer)
session.dispatch(AddStep(step="do thing"))
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

______________________________________________________________________

## 18. API reference

This is a curated reference of the APIs you'll touch most often. For complete
details, read module docstrings and the specs.

### 18.1 Top-level exports

Import from `weakincentives` when you want the "90% API":

**Budgets/time:**

- `Deadline`, `DeadlineExceededError`
- `Budget`, `BudgetTracker`, `BudgetExceededError`

**Prompt primitives:**

- `PromptTemplate`, `Prompt`, `RenderedPrompt`
- `Section`, `MarkdownSection`
- `Tool`, `ToolContext`, `ToolResult`, `ResourceRegistry`
- `SectionVisibility`
- `parse_structured_output`, `OutputParseError`

**Runtime primitives:**

- `Session`, `InProcessDispatcher`
- `MainLoop`, `MainLoopConfig` and loop events (`MainLoopRequest`,
  `MainLoopCompleted`, `MainLoopFailed`)
- Reducer helpers (`append_all`, `replace_latest`, `upsert_by`, ...)
- Logging helpers (`configure_logging`, `get_logger`)

**Errors:**

- `WinkError`, `ToolValidationError`, snapshot/restore errors

### 18.2 weakincentives.prompt

```python
PromptTemplate[OutputT](ns, key, name=None, sections=..., allow_extra_keys=False)
Prompt(template, overrides_store=None, overrides_tag="latest")
    .bind(*params)
    .render(session=None)
    .find_section(SectionType)

MarkdownSection(title, key, template, summary=None, visibility=..., tools=..., policies=...)
Tool(name, description, handler, examples=...)
ToolResult.ok(value, message="OK")      # success case
ToolResult.error(message)               # failure case
```

**Tool policies:**

- `ToolPolicy`: Protocol for tool invocation constraints
- `SequentialDependencyPolicy(dependencies)`: Enforce tool ordering
- `ReadBeforeWritePolicy()`: Prevent overwrites without reading first

**Progressive disclosure:**

- `VisibilityExpansionRequired`

### 18.3 weakincentives.runtime

```python
Session(dispatcher, tags=None, parent=None)
SessionView(session)                    # Read-only wrapper for reducer contexts
session[Type].all() / latest() / where()
session.dispatch(event)                 # All mutations go through dispatch

# Convenience methods (dispatch events internally)
session[Type].seed(value)               # → InitializeSlice
session[Type].clear()                   # → ClearSlice

# Visibility overrides (in runtime.session)
VisibilityOverrides, SetVisibilityOverride, ClearVisibilityOverride
session.snapshot(include_all=False)
session.restore(snapshot, preserve_logs=True)

# MainLoop configuration
MainLoopConfig(deadline=..., budget=..., resources=...)
MainLoop.execute(request, deadline=..., budget=..., resources=...)
```

**Slice storage (in runtime.session):**

- `SliceView[T]`: Read-only protocol for reducer input
- `Slice[T]`: Mutable protocol for storage operations
- `SliceOp`: Algebraic type (`Append | Extend | Replace | Clear`)
- `InitializeSlice[T]`, `ClearSlice[T]`: System events for slice mutations
- `MemorySlice` / `JsonlSlice`: In-memory and JSONL backends

**Reducers:**

- `append_all`, `replace_latest`, `replace_latest_by`, `upsert_by`
- Reducers receive `SliceView[S]` and return `SliceOp[S]`

**Event dispatcher:**

- `InProcessDispatcher`
- Telemetry events (`PromptRendered`, `ToolInvoked`, `PromptExecuted`,
  `TokenUsage`)

**Lifecycle management:**

- `Runnable`: Protocol for loops with graceful shutdown (`run()`, `shutdown()`,
  `running`, `heartbeat`)
- `ShutdownCoordinator.install()`: Singleton for SIGTERM/SIGINT handling
- `LoopGroup(loops, health_port=..., watchdog_threshold=...)`: Run multiple
  loops with coordinated shutdown, health endpoints, and watchdog monitoring
- `Heartbeat`: Thread-safe timestamp tracker for worker liveness
- `Watchdog`: Daemon thread that monitors heartbeats and terminates on stall
- `HealthServer`: Minimal HTTP server for `/health/live` and `/health/ready`
- `wait_until(predicate, timeout=...)`: Poll predicate with timeout

### 18.4 weakincentives.adapters

```python
ProviderAdapter.evaluate(prompt, session=..., deadline=..., budget=..., budget_tracker=...)
PromptResponse(prompt_name, text, output)
PromptEvaluationError
```

**Configs:**

- `OpenAIClientConfig`, `OpenAIModelConfig`
- `LiteLLMClientConfig`, `LiteLLMModelConfig`

**Resources (weakincentives.resources):**

- `Binding[T](protocol, provider, scope=Scope.SINGLETON, eager=False)`
- `Binding.instance(protocol, value)` - bind pre-constructed instance
- `Scope` enum: `SINGLETON`, `TOOL_CALL`, `PROTOTYPE`
- `ResourceRegistry.of(*bindings)` - build registry from bindings
- `ResourceRegistry.merge(base, override)` - combine registries (override wins)
- `ResourceRegistry.conflicts(other)` - return protocols bound in both
- `ResourceRegistry.open()` - context manager for resource lifecycle
- `ScopedResourceContext` - resolution context with lifecycle management
- `Closeable`, `PostConstruct` - lifecycle protocols
- `CircularDependencyError`, `DuplicateBindingError`, `ProviderError`,
  `UnboundResourceError` - dependency injection errors

**Throttling:**

- `ThrottlePolicy`, `new_throttle_policy`, `ThrottleError`

### 18.5 weakincentives.contrib.tools

**Planning:**

- `PlanningToolsSection(session, strategy=..., accepts_overrides=False)`

**Workspace:**

- `VfsToolsSection(session, config=VfsConfig(...), accepts_overrides=False)`
- `HostMount(host_path, mount_path=None, include_glob=(), exclude_glob=())`
- `WorkspaceDigestSection(session, title="Workspace Digest", key="workspace-digest")`

**Sandboxes:**

- `AstevalSection(session, accepts_overrides=False)`
- `PodmanSandboxSection(session, config=PodmanSandboxConfig(...))` (extra)

### 18.6 weakincentives.optimizers

- `PromptOptimizer` protocol and `BasePromptOptimizer`
- `OptimizationContext`, `OptimizationResult`

**Contrib:**

- `WorkspaceDigestOptimizer`

### 18.7 weakincentives.serde

Dataclass serialization utilities (no Pydantic required):

```python
from weakincentives.serde import dump, parse, schema, clone

# Serialize a dataclass to a JSON-compatible dict
data = dump(my_dataclass)

# Parse a dict back into a dataclass (with validation)
obj = parse(MyDataclass, {"field": "value"})

# Generate JSON schema for a dataclass
json_schema = schema(MyDataclass)

# Deep copy a frozen dataclass
copy = clone(my_dataclass)
```

**Key behaviors:**

- `parse()` validates required fields and rejects unknown keys by default
- Nested dataclasses are recursively parsed
- `tuple`, `frozenset`, and other immutable collections are handled
- `schema()` produces OpenAI-compatible JSON schemas for structured output

### 18.8 weakincentives.evals

**Core types:**

```python
Sample[InputT, ExpectedT](id, input, expected)
Dataset[InputT, ExpectedT](samples)
Dataset.load(path, input_type, expected_type)

Score(value, passed, reason="")
EvalResult(sample_id, score, latency_ms, error=None)
EvalReport(results)
    .pass_rate / .mean_score / .mean_latency_ms / .failed_samples()
```

**Evaluators:**

```python
exact_match(output, expected) -> Score
contains(output, expected) -> Score
all_of(*evaluators) -> Evaluator
any_of(*evaluators) -> Evaluator
llm_judge(adapter, criterion) -> Evaluator
adapt(evaluator) -> SessionEvaluator  # Convert standard to session-aware
```

**Session evaluators:**

```text
tool_called(name) -> SessionEvaluator
tool_not_called(name) -> SessionEvaluator
tool_call_count(name, min_count, max_count) -> SessionEvaluator
all_tools_succeeded() -> SessionEvaluator
token_usage_under(max_tokens) -> SessionEvaluator
slice_contains(T, predicate) -> SessionEvaluator
```

**Loop and helpers:**

```text
EvalLoop(loop, evaluator, requests)
    .run(max_iterations=None)

submit_dataset(dataset, requests)
collect_results(results, expected_count, timeout_seconds=300)
```

### 18.9 weakincentives.skills

Agent Skills specification support (following https://agentskills.io):

```python
from weakincentives.skills import (
    Skill,
    SkillMount,
    SkillConfig,
    validate_skill,
    validate_skill_name,
    resolve_skill_name,
    MAX_SKILL_FILE_BYTES,      # 1 MiB
    MAX_SKILL_TOTAL_BYTES,     # 10 MiB
)

# Mount skills for Claude Agent SDK isolation
config = SkillConfig(
    skills=(
        SkillMount(source=Path("./skills/code-review")),
        SkillMount(source=Path("./skills/testing"), enabled=False),
    ),
    validate_on_mount=True,  # Default: validate before copying
)

# Validation functions
validate_skill(Path("./skills/my-skill"))  # Raises SkillValidationError
name = resolve_skill_name(mount)           # Derive name from path
```

**Errors:**

- `SkillError` (base), `SkillValidationError`, `SkillNotFoundError`,
  `SkillMountError`

### 18.10 weakincentives.filesystem

Filesystem protocol and implementations:

```python
from weakincentives.filesystem import (
    Filesystem,                # Protocol for file operations
    SnapshotableFilesystem,    # Extended protocol with snapshot/restore
    HostFilesystem,            # Host filesystem with git-based snapshots
)
# InMemoryFilesystem is in contrib:
# from weakincentives.contrib.tools import InMemoryFilesystem

# Binary operations (new in v0.19.0)
content = fs.read_bytes("image.png", offset=0, limit=1024)
fs.write_bytes("output.bin", b"\x00\x01\x02", mode="overwrite")

# Text operations
result = fs.read("config.json")
fs.write("output.txt", "content", mode="overwrite")

# Search operations
matches = fs.glob("**/*.py")
grep_result = fs.grep("TODO", path="src/", glob="*.py")
```

**Key behaviors:**

- `read_bytes()` supports offset and limit for partial reads
- `write_bytes()` supports "overwrite" and "append" modes
- `read()` raises `ValueError` with actionable message for binary content
- `grep()` silently skips non-UTF-8 files
- UTF-8 paths are now allowed (ASCII-only restriction removed)

### 18.11 CLI

Installed via the `wink` extra:

```bash
pip install "weakincentives[wink]"
```

**Commands:**

```bash
# Start the debug UI server
wink debug <snapshot_path> [options]

# Access bundled documentation
wink docs --guide       # Print WINK_GUIDE.md
wink docs --reference   # Print llms.md (API reference)
wink docs --specs       # Print all spec files concatenated
wink docs --changelog   # Print CHANGELOG.md
```

**Debug options:**

| Option | Default | Description | | ------------------- | ----------- |
-------------------------------------- | | `--host` | `127.0.0.1` | Host
interface to bind | | `--port` | `8000` | Port to bind | | `--open-browser` |
`true` | Open browser automatically | | `--no-open-browser` | - | Disable
auto-open | | `--log-level` | `INFO` | Log verbosity (DEBUG, INFO, etc.) | |
`--json-logs` | `true` | Emit structured JSON logs | | `--no-json-logs` | - |
Emit plain text logs |

**Exit codes:**

- `0`: Success
- `2`: Invalid input (missing file, parse error)
- `3`: Server failed to start

______________________________________________________________________

## Where to go deeper

- **Prompts**: [specs/PROMPTS.md](specs/PROMPTS.md)
- **Tools & Policies**: [specs/TOOLS.md](specs/TOOLS.md)
- **Sessions**: [specs/SESSIONS.md](specs/SESSIONS.md)
- **MainLoop**: [specs/MAIN_LOOP.md](specs/MAIN_LOOP.md)
- **Evals**: [specs/EVALS.md](specs/EVALS.md)
- **Health & Lifecycle**: [specs/HEALTH.md](specs/HEALTH.md)
- **Resources**: [specs/RESOURCE_REGISTRY.md](specs/RESOURCE_REGISTRY.md)
- **Skills**: [specs/SKILLS.md](specs/SKILLS.md)
- **Filesystem**: [specs/FILESYSTEM.md](specs/FILESYSTEM.md)
- **Workspace**: [specs/WORKSPACE.md](specs/WORKSPACE.md)
- **Prompts & overrides**: [specs/PROMPTS.md](specs/PROMPTS.md)
- **DbC & exhaustiveness**: [specs/DBC.md](specs/DBC.md)
- **Formal verification**: [specs/FORMAL_VERIFICATION.md](specs/FORMAL_VERIFICATION.md)
  (embedding TLA+ specs in Python)
- **Code review example**:
  [guides/code-review-agent.md](guides/code-review-agent.md)
- **Contributor guide**: [AGENTS.md](AGENTS.md)

______________________________________________________________________

## Appendix A: Coming from LangGraph or LangChain?

If you've built agents with LangGraph, LangChain, or similar frameworks, here's
a quick orientation.

**Different philosophy, different primitives.**

LangGraph centers on **graphs**: nodes are functions, edges are transitions,
state flows through the graph. You model agent behavior as explicit control
flow. LangChain centers on **chains**: composable sequences of calls to LLMs,
tools, and retrievers.

WINK centers on **the prompt itself**. There's no graph. There's no chain. The
prompt—a tree of typed sections—_is_ your agent. The model decides what to do
next based on what's in the prompt. Tools, instructions, and state all live in
that tree.

**Concept mapping:**

- **Graph / Chain** → `PromptTemplate` (tree of sections)
- **Node / Tool** → `Tool` + handler function
- **State / Memory** → `Session` (typed slices + reducers)
- **Router / Conditional edge** → `enabled()` predicate on sections
- **Checkpointing** → `session.snapshot()` / `session.restore()`
- **LangSmith tracing** → Session events + debug UI

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

- **Deterministic by default.** Prompt rendering is pure. State transitions flow
  through reducers. Side effects are confined to tool handlers. You can snapshot
  the entire state at any point and restore it later.

- **No async (yet).** Adapters are synchronous. This simplifies debugging at the
  cost of throughput. Async may come later.

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

______________________________________________________________________

## Appendix B: Coming from DSPy?

If you've built LLM programs with DSPy, here's how WINK compares.

**Different bets on where value lives.**

DSPy centers on **automatic optimization**: you declare input/output signatures,
compose modules, and let optimizers compile better prompts and few-shot
examples. The framework treats prompts as implementation details that should be
generated, not written.

WINK centers on **explicit, inspectable prompts**: you write prompts as typed
section trees, control exactly what the model sees, and iterate via
version-controlled overrides. The framework treats prompts as first-class
artifacts that should be readable, testable, and auditable.

Both approaches have merit. DSPy shines when you have good metrics and want to
automate prompt tuning. WINK shines when you need to understand exactly what's
being sent to the model and why.

**Concept mapping:**

- **Signature** → Structured output dataclass + `PromptTemplate`
- **Module** (`Predict`, `ChainOfThought`) → `Section` (instructions + tools)
- **Program** (composed modules) → `PromptTemplate` (tree of sections)
- **Optimizer / Teleprompter** → Prompt overrides + manual iteration
- **Compilation** → No equivalent (prompts are explicit)
- **`dspy.ReAct`** → `PlanningToolsSection` + tool sections
- **Metric** → Evaluation framework (see `specs/EVALS.md`)
- **Trace** → Session events + debug UI

**What's familiar:**

- **Typed inputs and outputs.** DSPy signatures declare input/output fields;
  WINK uses frozen dataclasses for the same purpose. Both catch type mismatches
  early.

- **Composition.** DSPy composes modules into programs; WINK composes sections
  into prompt templates. Both encourage modular, reusable components.

- **Tool use.** DSPy modules like `ReAct` handle tool calling; WINK sections
  register tools alongside their instructions.

**What's different:**

- **Prompts are visible.** In DSPy, prompts are generated artifacts—you don't
  typically read or edit them directly. In WINK, `prompt.render()` returns the
  exact markdown sent to the model. You can inspect, test, and version it.

- **No automatic optimization.** DSPy's optimizers (BootstrapFewShot, MIPROv2,
  etc.) generate prompts automatically. WINK uses hash-validated overrides for
  manual iteration. You can build optimization workflows on top, but the
  framework doesn't assume you want automated prompt generation.

- **State is explicit.** DSPy traces execution but doesn't expose a structured
  state model. WINK sessions are typed, reducer-managed state containers. Every
  state change is an event you can query, snapshot, and restore.

- **Tools and instructions are co-located.** In DSPy, tool definitions are
  separate from module logic. In WINK, the section that explains "use this tool
  for X" is the same section that registers the tool. They can't drift apart.

- **Deterministic by default.** WINK prompt rendering is pure—same inputs
  produce same outputs. You can write tests that assert on exact prompt text.
  DSPy's compiled prompts depend on optimizer state and training data.

**When to use WINK instead of DSPy:**

- You need to inspect and understand exactly what prompts are being sent
- You're building systems where auditability matters (compliance, debugging)
- You want to iterate on prompts manually with version control
- You value determinism and testability over automatic optimization
- You're building tool-heavy agents where prompt/tool co-location helps

**When to stick with DSPy:**

- You have good metrics and want automated prompt optimization
- You're doing research where prompt generation is part of the experiment
- You want to bootstrap few-shot examples automatically
- You prefer declaring intent (signatures) over writing prompts

**Migration path:**

If you're moving from DSPy to WINK:

1. **Convert signatures to dataclasses.** A DSPy signature like
   `"question -> answer"` becomes input and output dataclasses:

   ```python
   # DSPy
   class QA(dspy.Signature):
       question = dspy.InputField()
       answer = dspy.OutputField()

   # WINK
   @dataclass(slots=True, frozen=True)
   class QuestionParams:
       question: str

   @dataclass(slots=True, frozen=True)
   class Answer:
       answer: str
   ```

1. **Convert modules to sections.** A DSPy module becomes a WINK section that
   renders instructions and optionally registers tools:

   ```python
   # DSPy
   qa = dspy.ChainOfThought(QA)

   # WINK
   qa_section = MarkdownSection(
       title="Question Answering",
       key="qa",
       template="Think step by step, then answer the question.\n\nQuestion: ${question}",
   )
   ```

1. **Convert programs to templates.** Composed DSPy modules become a
   `PromptTemplate` with nested sections:

   ```python
   template = PromptTemplate[Answer](
       ns="qa",
       key="chain-of-thought",
       sections=(qa_section,),
   )
   ```

1. **Replace optimizers with overrides.** Instead of compiled prompts, use
   WINK's override system to iterate on prompt text:

   ```python
   prompt = Prompt(template, overrides_store=store, overrides_tag="v2")
   ```

1. **Add tools explicitly.** DSPy's `ReAct` handles tool use implicitly; WINK
   requires explicit tool registration on sections. This is more verbose but
   makes tool availability obvious from the prompt structure.

The key mindset shift: DSPy optimizes prompts for you; WINK gives you tools to
write and iterate on prompts yourself. If you've been frustrated by not knowing
what DSPy is actually sending to the model, WINK's explicit approach may feel
liberating. If you've relied heavily on DSPy's optimizers, you'll need to build
or adopt optimization workflows separately.

______________________________________________________________________

## Appendix C: Formal Verification with TLA+

WINK supports embedding TLA+ formal specifications directly in Python code
using the `@formal_spec` decorator. This approach prevents specification drift
by keeping specs co-located with implementation.

### Why formal verification?

For correctness-critical code like distributed message queues, testing alone
isn't sufficient. The `RedisMailbox` implementation, for example, must maintain
invariants like "each message exists in exactly one place" across all possible
interleavings of concurrent operations.

TLA+ model checking exhaustively explores these interleavings, catching subtle
bugs that randomized testing might miss.

### Quick example

```python
from weakincentives.formal import formal_spec, StateVar, Action, Invariant

@formal_spec(
    module="Counter",
    state_vars=[
        StateVar("count", "Nat", "Current count value"),
    ],
    actions=[
        Action(
            name="Increment",
            preconditions=("count < MaxValue",),
            updates={"count": "count + 1"},
        ),
        Action(
            name="Decrement",
            preconditions=("count > 0",),
            updates={"count": "count - 1"},
        ),
    ],
    invariants=[
        Invariant("INV-1", "NonNegative", "count >= 0", "Count never goes negative"),
        Invariant("INV-2", "BelowMax", "count <= MaxValue", "Count never exceeds max"),
    ],
    constants={"MaxValue": 10},
    constraint="count <= 5",  # Limit state space exploration
)
class Counter:
    """Simple counter with formal spec."""

    def __init__(self):
        self.count = 0

    def increment(self):
        if self.count < 10:
            self.count += 1

    def decrement(self):
        if self.count > 0:
            self.count -= 1
```

### Running verification

```python
# formal-tests/test_counter.py
from pathlib import Path
from typing import Any
from weakincentives.formal.testing import extract_and_verify

# Counter class defined above with @formal_spec decorator
Counter: Any = ...  # type: ignore[assignment]


def test_counter_spec(tmp_path: Path) -> None:
    """Extract and verify Counter TLA+ specification."""
    spec, tla_file, cfg_file, result = extract_and_verify(
        Counter,
        output_dir=tmp_path,
        model_check_enabled=True,
        tlc_config={"workers": "auto", "cleanup": True},
    )

    assert spec.module == "Counter"
    if result is not None:
        assert result.passed
        assert result.states_generated > 0
```

Run with:

```bash
make verify-formal  # Runs TLC model checker
```

### Key concepts

**State variables** declare the TLA+ state space:

```python
StateVar("queue", "Seq(Message)", "Pending messages")
StateVar("inFlight", "[1..NumConsumers -> Seq(Message)]", "In-flight per consumer")
```

**Actions** define state transitions with preconditions and updates:

```python
Action(
    name="Receive",
    parameters=(ActionParameter("consumer", "1..NumConsumers"),),
    preconditions=("queue /= <<>>",),
    updates={
        "inFlight": "Append(inFlight[consumer], Head(queue))",
        "queue": "Tail(queue)",
    },
)
```

**Invariants** define safety properties that must always hold:

```python
Invariant("INV-1", "MessageExclusivity", "MessageInExactlyOnePlace(msg)")
Invariant("INV-2", "NoLostMessages", "CountMessages() = InitialMessageCount")
```

### State space management

The challenge with model checking is state space explosion. Strategies:

1. **Small constants**: Use `MaxMessages: 2` not `100`
1. **Tight constraints**: Add `constraint="now <= 2"` to bound exploration
1. **Narrow domains**: Use `"0..2"` not `"0..100"` for parameters

The RedisMailbox spec, for example, explores ~500K states in 60 seconds with
carefully chosen bounds.

### When to use formal verification

Use `@formal_spec` for:

- Distributed algorithms (message queues, consensus)
- State machines with complex invariants
- Concurrent data structures
- Any code where "it works in testing" isn't enough

Don't use it for:

- Simple CRUD operations
- Stateless transformations
- Code where types + tests provide sufficient confidence

### Testing utilities

```python
from weakincentives.formal.testing import (
    extract_spec,      # Extract FormalSpec from decorated class
    write_spec,        # Write .tla and .cfg files
    model_check,       # Run TLC model checker
    extract_and_verify # Combined extraction + verification
)
```

### Installation

TLC must be installed for model checking:

```bash
# macOS
brew install tlaplus

# Linux
wget https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
```

See `specs/FORMAL_VERIFICATION.md` for complete API documentation and advanced
topics like modeling time, helper operators, and CI integration.
