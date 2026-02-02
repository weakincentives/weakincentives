---
title: "WINK: Weak Incentives"
subtitle: "A Python Library for Building Agents"
author: "Weak Incentives Project"
date: "2026"
theme: moon
highlight-style: monokai
transition: slide
---

# What is WINK?

## The Name

**Weak Incentives** comes from mechanism design:

> A system with the right incentives is one where participants naturally
> gravitate toward intended behavior.

Applied to agents: shape the prompt, tools, and context so the model's
**easiest path is also the correct one**.

## Not About Constraints

This isn't about constraining the model or managing downside risk.

It's about **encouraging** correct behavior through structure:

- Clear instructions co-located with tools
- Typed contracts guide valid outputs
- Progressive disclosure keeps focus
- Explicit state for good decisions

---

# Core Philosophy

## The Prompt is the Agent

Prompts are hierarchical documents where sections bundle instructions and tools
together.

No separate tool registry; capabilities live in the prompt definition.

## Event-Driven State

All mutations flow through pure reducers processing typed events.

State is immutable and inspectable via snapshots.

## Provider-Agnostic

Same agent definition works across:

- OpenAI
- LiteLLM
- Claude Agent SDK

---

# Technical Strategy

## Don't Compete at the Model Layer

Models and agent frameworks will commoditize quickly.

Treating them as **swappable dependencies** is the winning posture.

WINK's adapters exist so you can switch providers without rewriting your agent.

## Differentiate with Your System of Record

Long-term advantage comes from owning:

- Authoritative data
- Definitions
- Permissions
- Business context

The model is a commodity; **your domain knowledge isn't**.

---

# The Shift

## Orchestration Shrinks

Many early agent frameworks assumed the hard part would be workflow logic:

- Routers
- Planners
- Branching graphs
- Elaborate loops

## Context Engineering Grows

Models are steadily absorbing the reasoning loop.

The durable part of agent systems is:

- **Tools**: what the agent can do
- **Retrieval**: what information is available
- **Context engineering**: what to include, what to summarize

---

# Prompts as First-Class Programs

## The Problem with String Prompts

Most systems treat prompts as strings:

- Prompt text in one place
- Tool definitions in another
- Schema expectations in another
- Memory in another

Teams add layers. They drift. Things break.

## WINK's Solution

A `PromptTemplate` is an immutable object graph (a tree of sections).

Each section can:

- Render markdown instructions
- Declare typed placeholders
- Register tools
- Render as a summary to save tokens

---

# Core Abstractions

## PromptTemplate

```python
@dataclass(slots=True, frozen=True)
class MyPrompt(PromptTemplate[str]):
    ns = "demo"
    key = "my-prompt"
    sections = (intro, tools, guidelines)
```

## Session

Every prompt render, tool invocation, and state change is recorded.

You can query, snapshot, and restore sessions.

## ProviderAdapter

Renders markdown, executes tool calls, returns parsed output.

---

# Tools

## Typed Tool Contracts

```python
@dataclass(slots=True, frozen=True)
class SearchParams:
    query: str
    max_results: int = 10

def search_handler(
    params: SearchParams,
    *,
    context: ToolContext
) -> ToolResult[list[str]]:
    results = do_search(params.query, params.max_results)
    return ToolResult.ok(results)
```

## Tool Policies

Enforce constraints like "must call X before Y".

Tools are **transactional**: failures roll back automatically.

---

# Sessions & Reducers

## Pure Reducers

```python
@reducer(on=MessageReceived)
def on_message(
    state: ConversationState,
    event: MessageReceived
) -> SliceOp[Message]:
    return Append(event.message)
```

Reducers return operations (Append, Replace, Clear) - never mutate directly.

## Event-Driven

All mutations via `session.dispatch(event)`.

Access state: `session[T].latest()`, `.all()`, `.where(predicate)`

---

# Progressive Disclosure

## The Problem

Large prompts waste tokens on irrelevant context.

## The Solution

Sections can start **collapsed** and expand on demand:

```python
rules_section = MarkdownSection(
    title="Game Rules",
    key="rules",
    template=FULL_RULES,
    disclosure=collapsed(
        summary="Rules available on request."
    ),
)
```

The model can request what it needs, keeping context focused.

---

# Production Patterns

## Skills: Domain Knowledge

Skills are markdown files injected into agent context:

```
skills/secret-trivia/
  secret-number.md   # Contains: 42
  secret-word.md     # Contains: banana
  secret-color.md    # Contains: purple
```

## Feedback: Soft Guidance

Observes agent behavior and provides course correction without aborting.

## Evaluators

Score agent outputs for automated testing.

---

# Code Quality

## Strict Type Checking

- Pyright strict mode enforced
- Type mismatches surface at construction time
- Annotations are source of truth

## Design-by-Contract

```python
@require(lambda x: x > 0, "x must be positive")
@ensure(lambda result: result >= 0, "result non-negative")
def sqrt(x: float) -> float:
    ...
```

## 100% Test Coverage

Required for all code in `src/weakincentives/`.

---

# Getting Started

## Install

```bash
pip install weakincentives
```

With adapters:

```bash
pip install "weakincentives[openai]"
pip install "weakincentives[litellm]"
pip install "weakincentives[claude-agent-sdk]"
```

## Starter Project

```bash
git clone https://github.com/weakincentives/starter.git
cd starter
make install && make agent
```

---

# What WINK Is

- A Python library for building prompts-as-agents
- A small runtime for state and orchestration
- Adapters for OpenAI, LiteLLM, Claude Agent SDK
- Contributed tool suites for background agents

# What WINK Is Not

- A distributed workflow engine
- A framework that owns your architecture
- A multi-agent coordination system
- An async-first streaming framework

---

# Learn More

## Guides

- [Quickstart](guides/quickstart.md)
- [Philosophy](guides/philosophy.md)
- [Prompts](guides/prompts.md)
- [Tools](guides/tools.md)
- [Sessions](guides/sessions.md)

## Resources

- GitHub: github.com/weakincentives/weakincentives
- Specs: `specs/` directory for precise guarantees

---

# Questions?

**WINK: The prompt is the agent.**

Build typed, testable, provider-agnostic agents with weak incentives.
