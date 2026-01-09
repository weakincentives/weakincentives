# Weak Incentives (WINK)

A practical guide to building deterministic, typed, safe background agents.

**What you'll learn to build:**

By the end of this guide, you'll know how to build agents that:

- Browse codebases, answer questions, and propose changes—safely sandboxed
- Use structured output to return typed, validated responses
- Maintain explicit, inspectable state across turns
- Manage token costs with progressive disclosure (show summaries, expand on demand)
- Iterate on prompts quickly using version-controlled overrides

The running example is a code review agent. It's a practical pattern: the agent reads files, makes a plan, and returns structured feedback. The same patterns apply to research agents, Q&A bots, automation assistants, and more.

---

This guide is written for engineers who want to:

- Build agents that can run unattended without turning into "a pile of prompt glue".
- Treat prompts as real software artifacts: testable, inspectable, and versionable.
- Keep tool use and side effects explicit, gated, and auditable.
- Iterate on prompts quickly without compromising correctness.

**If you only read one thing**: in WINK, the prompt is the agent.

**Status**: Alpha. Expect some APIs to evolve as the library matures.

---

## Technical Strategy

WINK is built around a specific bet about where durable value lives in agent systems:

**Don't compete at the model layer.** Models and agent frameworks will commoditize quickly. Treating them as swappable dependencies is the winning posture. WINK's adapters exist precisely so you can switch providers without rewriting your agent.

**Differentiate with your system of record.** Long-term advantage comes from owning authoritative data, definitions, permissions, and business context—not from the reasoning loop. The model is a commodity; your domain knowledge isn't.

**Keep product semantics out of prompts.** Encode domain meaning and safety in stable tools and structured context, not provider-specific prompt glue. When your business logic lives in typed tool handlers and well-defined state, it survives model upgrades.

**Use provider runtimes; own the tools.** Let vendors handle planning, orchestration, and retries. Invest in high-quality tools that expose your system-of-record capabilities. The Claude Agent SDK adapter is an example: it delegates execution to Claude Code's native runtime while WINK owns the tool definitions and session state.

**Build evaluation as your control plane.** Make model and runtime upgrades safe via scenario tests and structured output validation. When you can verify behavior programmatically, you can improve without rewrites.

The future points toward SDKs shaped like the Claude Agent SDK: sophisticated sandboxing, native tool integration, seamless handoff between local and hosted execution. Models will increasingly come with their own tool runtimes, deeply integrated into training. WINK's job is to give you stable primitives—prompts, tools, state—that work across that evolving landscape.

---

## Table of Contents

1. [Technical Strategy](#technical-strategy)
1. [Philosophy](#1-philosophy)
1. [Quickstart](#2-quickstart)
1. [Prompts](#3-prompts)
1. [Tools](#4-tools)
1. [Sessions](#5-sessions)
1. [Adapters](#6-adapters)
1. [Orchestration with MainLoop](#7-orchestration-with-mainloop)
1. [Evaluation with EvalLoop](#8-evaluation-with-evalloop)
1. [Lifecycle Management](#9-lifecycle-management)
1. [Progressive disclosure](#10-progressive-disclosure)
1. [Prompt overrides and optimization](#11-prompt-overrides-and-optimization)
1. [Workspace tools](#12-workspace-tools)
1. [Debugging and observability](#13-debugging-and-observability)
1. [Testing and reliability](#14-testing-and-reliability)
1. [Approach to code quality](#15-approach-to-code-quality)
1. [Recipes](#16-recipes)
1. [Troubleshooting](#17-troubleshooting)
1. [API reference](#18-api-reference)
1. [Appendix A: Coming from LangGraph or LangChain?](#appendix-a-coming-from-langgraph-or-langchain)
1. [Appendix B: Coming from DSPy?](#appendix-b-coming-from-dspy)
1. [Appendix C: Formal Verification with TLA+](#appendix-c-formal-verification-with-tla)

---

## 1. Philosophy

WINK's design philosophy centers on "weak incentives"—building agent systems where well-constructed prompts and tools create incentives for the model to do the right thing and stay on task. This isn't about constraining the model; it's about encouraging correct behavior through structure.

The name comes from mechanism design: a system with the right incentives is one where participants naturally gravitate toward intended behavior. Applied to agents, this means shaping the prompt, tools, and context so the model's easiest path is also the correct one. Clear instructions co-located with tools make the right action obvious. Typed contracts guide the model toward valid outputs. Progressive disclosure keeps the model focused on what matters now.

WINK pushes you toward explicit side effects (only in tool handlers), typed contracts everywhere (dataclasses), inspectability (session event ledgers), controlled context growth (progressive disclosure), and safe iteration (hash-validated overrides). The core bet is that models are steadily absorbing more of the reasoning loop—what required elaborate multi-step orchestration yesterday often works in a single prompt today. The durable part of agent systems is tools, retrieval, and context engineering.

**📖 Detailed Coverage**: See [Chapter 1: Philosophy](book/01-philosophy.md) for comprehensive discussion including:
- The meaning of "weak incentives" in practice
- The shift from orchestration to context engineering
- Prompts as first-class, typed programs
- What WINK is (and is not)

---

## 2. Quickstart

WINK is a Python library with no mandatory third-party dependencies. Install the core package, then add provider adapters and tool extras as needed. The minimal working example demonstrates typed params, structured output, deterministic prompt structure, and session-based telemetry—all in under 30 lines.

The quickstart introduces three key patterns: structured agents (with typed input/output), tool registration (handlers that return `ToolResult`), and complete agent workflows (combining prompts, tools, adapters, and sessions). Each example is intentionally small but demonstrates core WINK principles: immutable dataclasses, explicit type declarations, and deterministic composition.

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
            template="Summarize the input.\n\nReturn JSON with:\n- title: short title\n- bullets: 3-7 bullet points",
        ),
        MarkdownSection(
            title="Input",
            key="input",
            template="${text}",
        ),
    ),
)

prompt = Prompt(template).bind(SummarizeRequest(
    text="WINK is a Python library for building agents. It treats prompts as typed programs."
))

bus = InProcessDispatcher()
session = Session(bus=bus)

# To actually run: adapter.evaluate(prompt, session=session)
```

**📖 Detailed Coverage**: See [Chapter 2: Quickstart](book/02-quickstart.md) for comprehensive examples including:
- Installation instructions and extras
- End-to-end tiny structured agent
- Adding tools to prompts
- Your first complete agent (copy-paste ready)

---

## 3. Prompts

The prompt system is the heart of WINK. Prompts are structured, typed, immutable objects—not strings. A `PromptTemplate[OutputT]` defines the agent's structure: sections, tools, placeholders, and optional structured output. A `Prompt[OutputT]` binds runtime parameters and renders to final markdown plus tools.

This design ensures predictability: prompt text is deterministic, placeholder names are validated at construction time, composition is explicit, and overrides are hash-validated. Sections form a tree structure where each node can contribute markdown, tools, and children. Tools are registered alongside the instructions that describe them—documentation and capability live together and can't drift.

Key abstractions include `MarkdownSection` (parameterized markdown with template placeholders), structured output (declared via type parameter), dynamic scoping (sections can be enabled/disabled at render time), and few-shot examples (via `TaskExamplesSection`). The distinction between template and prompt matters: templates are reusable definitions; prompts are bound to specific parameters and carry override configuration.

**📖 Detailed Coverage**: See [Chapter 3: Prompts](book/03-prompts.md) for comprehensive documentation including:
- PromptTemplate and Prompt types
- Section tree composition
- MarkdownSection and parameterization
- Structured output patterns
- Dynamic scoping with enabled()
- Session-bound sections and cloning
- Few-shot traces with TaskExamplesSection

---

## 4. Tools

The tool system enforces one hard rule: tool handlers are the only place where side effects happen. Everything else (prompt rendering, state transitions, reducers) is pure and deterministic. When something goes wrong, you know exactly where to look.

A tool is defined by its name, description, typed params, typed result, and handler function. Handlers receive `ToolContext` (providing session, resources, adapter, deadline, budget) and return `ToolResult[T]` (success or error). The `ToolResult` type distinguishes between success (`ToolResult.ok`) and error (`ToolResult.error`), with optional continuation signals and structured error information.

Tools can access resources through the prompt's resource registry—HTTP clients, database handles, filesystems—without adding fields to core dataclasses. WINK provides tool policies for enforcing constraints like sequential dependencies, read-before-write patterns, and keyed resource access. Transactional tool execution ensures consistent state updates even when tools fail partway through.

**📖 Detailed Coverage**: See these chapters for complete documentation:
- [Chapter 4: Tools](book/04-tools.md) - Core tool concepts, contracts, context, and examples
- [Chapter 4.5: Tool Policies](book/04.5-tool-policies.md) - Custom policy development and enforcement
- [Chapter 4.6: Task Monitoring](book/04.6-task-monitoring.md) - Trajectory observers and progress assessment

---

## 5. Sessions

A `Session` is WINK's event-driven memory system. Every interaction—prompt renders, tool calls, state mutations—is recorded as an immutable event in a ledger. The session provides querying, snapshotting, and time-travel debugging. State is organized into typed slices (like Redux slices), each managed by reducers that transform events into state operations.

Sessions support queries (`session[Plan].latest()`, `session[Plan].all()`, `session[Plan].where(predicate)`), reducers (pure functions that return `SliceOp` instructions like `Append` or `Replace`), and snapshots (capture state at any point and restore later). The declarative `@reducer` decorator lets dataclasses register reducers directly on the class.

SlicePolicy determines retention behavior: `RETAIN` keeps all values (state history), `RETAIN_LATEST` keeps only the most recent value (typical state management), and `RETAIN_NONE` keeps nothing (logging only). This separation of state and logs lets you choose the right durability for each slice type.

**📖 Detailed Coverage**: See [Chapter 5: Sessions](book/05-sessions.md) for comprehensive documentation including:
- Session as deterministic memory
- Query patterns and predicates
- Reducer patterns and SliceOp
- Declarative reducers with @reducer
- Snapshots and restore
- SlicePolicy: state vs logs

---

## 6. Adapters

Provider adapters implement the `ProviderAdapter` protocol: they take a `Prompt`, execute tools, and return typed results. Adapters abstract away provider-specific details—OpenAI vs Anthropic vs LiteLLM—so your agent code stays portable. All adapters support structured output (via function calling or response schemas), tool execution, throttling, and token tracking.

WINK ships with three adapters: `OpenAIAdapter` (for OpenAI and compatible APIs), `LiteLLMAdapter` (for 100+ providers via LiteLLM), and `ClaudeAgentSdkAdapter` (for Claude Code's native execution runtime with sophisticated sandboxing). The Claude Agent SDK adapter bridges WINK tools to MCP tools, supports workspace management, and delegates planning/orchestration to Claude's runtime while WINK owns tool definitions and session state.

```python
from weakincentives.adapters.openai import OpenAIAdapter

adapter = OpenAIAdapter(model="gpt-4o")
response = adapter.evaluate(prompt, session=session)
output = response.output  # Typed result
```

**📖 Detailed Coverage**: See [Chapter 6: Adapters](book/06-adapters.md) for comprehensive documentation including:
- ProviderAdapter protocol and evaluate()
- OpenAIAdapter configuration and features
- LiteLLMAdapter setup and usage
- Claude Agent SDK adapter deep dive
- Tool bridging via MCP
- Workspace and isolation configuration
- Complete secure code review example

---

## 7. Orchestration with MainLoop

`MainLoop` orchestrates the agent lifecycle: pull messages from a mailbox, evaluate the prompt, dispatch state transitions, and repeat. It handles deadlines (wall-clock limits), budgets (token limits), and graceful shutdown. The loop is intentionally simple—most orchestration complexity lives in the prompt and tools, not the loop itself.

The minimal MainLoop requires a prompt, adapter, bus, mailbox, and session. Optional deadline and budget configuration prevents runaway executions. When the agent completes (or exhausts its budget/deadline), the loop stops. For distributed orchestration, WINK supports mailbox-based message routing, reply-to patterns, and multi-loop coordination.

**📖 Detailed Coverage**: See these chapters for complete documentation:
- [Chapter 7: Main Loop](book/07-main-loop.md) - Core orchestration patterns and deadlines/budgets
- [Chapter 7.5: Distributed Orchestration](book/07.5-distributed-orchestration.md) - Mailbox systems and multi-agent coordination

---

## 8. Evaluation with EvalLoop

`EvalLoop` runs parallel evaluation alongside your main agent loop. Evaluators observe agent behavior, check outputs against golden data, compute metrics, and inject feedback. WINK's evaluation framework supports dataset-based evals (compare output to expected), LLM-as-judge (another model grades the agent), and session evaluators (inspect session state and events).

The composition philosophy: evaluators are independent observers that can run in parallel with production traffic. EvalLoop pulls from the same mailbox as MainLoop, evaluates using registered evaluators, and records results to eval slices. For production deployment, run MainLoop and EvalLoop side-by-side with reply-to routing—MainLoop handles tasks, EvalLoop grades them, results flow to downstream consumers.

**📖 Detailed Coverage**: See [Chapter 8: Evaluation](book/08-evaluation.md) for comprehensive documentation including:
- Composition philosophy
- Core types (Dataset, Evaluator, EvalLoop)
- LLM-as-judge patterns
- Session evaluators and state inspection
- Running evaluations
- Production deployment patterns
- Reply-to routing

---

## 9. Lifecycle Management

`LoopGroup` coordinates multiple loops (MainLoop, EvalLoop, custom loops) with graceful shutdown, health checks, and watchdog monitoring. It catches SIGTERM/SIGINT, triggers shutdown coordinators, waits for clean termination, and exposes Kubernetes-compatible health endpoints (`/health/live`, `/health/ready`).

The `Runnable` protocol defines the lifecycle contract: `run()` blocks until stopped, `shutdown()` triggers graceful stop. Health configuration includes readiness checks (is the loop healthy?) and watchdog thresholds (terminate stuck workers after N seconds). For manual control, use `ShutdownCoordinator` to manage SIGTERM/SIGINT handling yourself.

**📖 Detailed Coverage**: See [Chapter 9: Lifecycle Management](book/09-lifecycle.md) for comprehensive documentation including:
- LoopGroup: running multiple loops
- ShutdownCoordinator: manual signal handling
- The Runnable protocol
- Health and watchdog configuration

---

## 10. Progressive disclosure

Progressive disclosure lets sections render as summaries by default and expand on demand via tools. This keeps token counts manageable and models focused. Sections declare `SectionVisibility` (FULL or SUMMARY) and provide summary text. The agent receives `open_sections` and `read_section` tools to expand specific sections when needed.

Visibility overrides live in session state—when the agent calls `open_sections`, the session records which sections should render fully on subsequent turns. This approach works well for large codebases, documentation, or any context that's too big to include upfront. The model sees high-level structure, requests details as needed, and stays focused on relevant information.

**📖 Detailed Coverage**: See [Chapter 10: Progressive Disclosure](book/10-progressive-disclosure.md) for comprehensive documentation including:
- SectionVisibility: FULL vs SUMMARY
- open_sections and read_section tools
- Visibility overrides in session state

---

## 11. Prompt overrides and optimization

Prompt overrides let you iterate on prompt text without modifying code. The override system is hash-validated: when you override a section, WINK checks that you're overriding the exact version you tested. If the underlying section changes in code, the override stops applying (no silent drift).

`LocalPromptOverridesStore` loads overrides from JSONL files. Each override specifies namespace, key, tag, section path, original hash, and replacement text. The practical workflow: run with `overrides_tag="baseline"`, identify a section to improve, add an override with a new tag, test with the new tag, verify improvement, commit the override file. The optimizer framework supports automated prompt refinement via feedback loops.

**📖 Detailed Coverage**: See [Chapter 11: Prompt Optimization](book/11-prompt-optimization.md) for comprehensive documentation including:
- Hash-based safety and override validation
- LocalPromptOverridesStore
- Override file format
- Practical override workflow
- Optimizer framework and patterns

---

## 12. Workspace tools

WINK's `contrib.tools` package provides batteries for workspace-oriented agents: planning tools, virtual filesystem (VFS), workspace digests, safe expression evaluation (asteval), and Podman sandboxes. These tools are designed for background agents that read codebases, answer questions, propose changes, or run tests—all safely sandboxed.

`PlanningToolsSection` provides plan/update/complete_task/finish_plan tools for structured task execution. `VfsToolsSection` exposes a virtual filesystem with read/write/search operations. `WorkspaceDigestSection` renders a compressed view of a codebase (directory tree + file summaries). `AstevalSection` provides safe Python expression evaluation. `PodmanSandboxSection` runs shell commands in isolated Podman containers.

```python
from weakincentives.contrib.tools import PlanningToolsSection, VfsToolsSection

sections = [
    PlanningToolsSection(),
    VfsToolsSection(),
]
```

**📖 Detailed Coverage**: See [Chapter 12: Workspace Tools](book/12-workspace-tools.md) for comprehensive documentation including:
- PlanningToolsSection and structured task execution
- VfsToolsSection and virtual filesystem operations
- WorkspaceDigestSection and codebase compression
- AstevalSection and safe Python evaluation
- PodmanSandboxSection and isolated execution
- Wiring workspace tools into prompts

---

## 13. Debugging and observability

WINK provides structured logging, session event inspection, JSONL snapshot dumps, and a debug web UI. Structured logs emit JSON with structured data for aggregation. Session events form an immutable ledger—inspect what was rendered, which tools ran, what state changed. Snapshots dump session state to JSONL for offline analysis.

The debug UI (`wink debug`) launches a web server that visualizes session state, events, and prompt renders. You can explore slices, inspect tool calls, diff prompt versions, and time-travel through session history. The UI is built for post-mortem debugging: load a snapshot, explore what happened, identify where the agent went off track.

**📖 Detailed Coverage**: See [Chapter 13: Debugging and Observability](book/13-debugging.md) for comprehensive documentation including:
- Structured logging patterns
- Session event inspection
- Dumping snapshots to JSONL
- The debug UI and features

---

## 14. Testing and reliability

WINK agents are designed to be testable: prompts render deterministically, tools are pure functions with explicit side effects, sessions can be snapshotted and restored. Test prompts by asserting on rendered text and tools. Test tools by mocking ToolContext and asserting on ToolResult. Test agents end-to-end using evaluation datasets.

The testing philosophy: prefer deterministic tests over flaky integration tests. Mock LLM responses for unit tests, use evaluation datasets for integration tests, run scenario tests in CI. WINK's test harness provides fixtures, fault injection, fuzzing, and coverage measurement. The library enforces 100% test coverage as a quality gate.

**📖 Detailed Coverage**: See [Chapter 14: Testing and Reliability](book/14-testing.md) for comprehensive documentation including:
- Testing prompts and tools
- Test harnesses and fixtures
- Fault injection and fuzzing
- Evaluation-based integration tests
- Coverage requirements

---

## 15. Approach to code quality

WINK takes code quality seriously: strict type checking (Pyright strict mode), design-by-contract (DbC decorators for preconditions/postconditions), 100% test coverage, security scanning (Bandit, pip-audit), and automated quality gates. These aren't suggestions—they're enforced in CI and pre-commit hooks.

Strict type checking catches errors at construction time, not runtime. Design-by-contract makes invariants explicit and machine-checkable. Coverage requirements ensure every code path is tested. Security scanning catches common vulnerabilities and outdated dependencies. The quality gates run on every commit via `make check`—if it doesn't pass, it doesn't merge.

**📖 Detailed Coverage**: See [Chapter 15: Code Quality](book/15-code-quality.md) for comprehensive documentation including:
- Strict type checking with Pyright
- Design-by-contract patterns
- Coverage requirements and enforcement
- Security scanning tools
- Quality gates in practice

---

## 16. Recipes

Practical patterns for common agent types: code review agents (read files, make plan, return feedback), repo Q&A agents (answer questions about a codebase), safe patch agents (propose changes without writing to disk), and research agents (use progressive disclosure to manage large contexts). Each recipe demonstrates core WINK patterns in a realistic scenario.

The code review agent is the running example: it uses VFS to read files, planning tools to structure work, structured output to return typed feedback, and prompt overrides to iterate on quality. The same patterns apply broadly—swap the tools and structured output to build different agent types.

**📖 Detailed Coverage**: See [Chapter 16: Recipes](book/16-recipes.md) for comprehensive examples including:
- Code review agent (complete implementation)
- Repo Q&A agent pattern
- Safe patch agent pattern
- Research agent with progressive disclosure

---

## 17. Troubleshooting

Common issues and solutions: placeholder validation errors (ensure dataclass fields match template placeholders), type mismatches (check dataclass types and slots=True/frozen=True), override hash mismatches (section changed in code, update override hash), tool execution failures (check ToolContext resources), session query errors (ensure slice is initialized), and adapter configuration issues (verify API keys and model names).

The troubleshooting guide covers error messages, common pitfalls, debugging strategies, and performance optimization. When in doubt, enable structured logging, inspect the session event ledger, and use the debug UI to visualize what happened.

**📖 Detailed Coverage**: See [Chapter 17: Troubleshooting](book/17-troubleshooting.md) for comprehensive guidance including:
- Common error messages and solutions
- Debugging strategies
- Performance optimization
- Known issues and workarounds

---

## 18. API reference

Complete API reference covering all public exports: top-level exports from `weakincentives`, submodules (`prompt`, `runtime`, `adapters`, `contrib.tools`, `optimizers`, `serde`, `evals`, `skills`, `filesystem`), and CLI commands. The reference documents types, functions, classes, protocols, and their signatures.

This is the exhaustive reference—for learning patterns, see the earlier chapters. For quick lookups (what parameters does OpenAIAdapter accept?), see this section.

**📖 Detailed Coverage**: See [Chapter 18: API Reference](book/18-api-reference.md) for complete documentation including:
- Top-level exports
- weakincentives.prompt API
- weakincentives.runtime API
- weakincentives.adapters API
- weakincentives.contrib.tools API
- weakincentives.optimizers API
- weakincentives.serde API
- weakincentives.evals API
- weakincentives.skills API
- weakincentives.filesystem API
- CLI command reference

---

## Appendix A: Coming from LangGraph or LangChain?

If you're coming from LangGraph or LangChain, the main conceptual shifts are: prompts are typed objects (not strings or templates), tools are registered with sections (not in separate catalogs), state is event-driven (not a graph), and orchestration is minimal (most logic lives in the prompt and tools). WINK doesn't have graph nodes, edges, or routing—the model handles reasoning, you provide tools and context.

**📖 Detailed Coverage**: See [Appendix A: Coming from LangGraph or LangChain](book/appendix-a-from-langgraph.md) for comprehensive comparison including:
- Conceptual mapping
- Migration patterns
- Feature comparison
- When to use which framework

---

## Appendix B: Coming from DSPy?

If you're coming from DSPy, the main conceptual shifts are: prompts are explicitly structured (not inferred), optimization is deliberate (not automatic), and modules are prompts (not Python classes). WINK shares DSPy's emphasis on typed signatures but takes a more explicit approach to prompt engineering and iteration.

**📖 Detailed Coverage**: See [Appendix B: Coming from DSPy](book/appendix-b-from-dspy.md) for comprehensive comparison including:
- Conceptual mapping
- Optimization approaches
- Feature comparison
- When to use which framework

---

## Appendix C: Formal Verification with TLA+

WINK supports embedding TLA+ formal specifications in Python code via the `@formal_spec` decorator. This lets you write invariants and temporal properties as TLA+ specs, run TLC model checking in CI, and prove properties about your agent's behavior. The formal verification system is designed for correctness-critical components like distributed mailbox semantics and session consistency.

**📖 Detailed Coverage**: See [Appendix C: Formal Verification with TLA+](book/appendix-c-formal-verification.md) for comprehensive documentation including:
- TLA+ specification embedding
- @formal_spec decorator usage
- TLC model checking integration
- Verification examples
- CI integration patterns
