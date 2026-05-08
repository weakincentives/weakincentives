# Adapters

An **adapter** is the bridge between WINK's portable agent definition and a
specific execution harness. It is the only place runtime-specific concerns
appear. Above the adapter, the same prompt, tools, policies, and feedback
run unchanged. Below the adapter, each runtime gets its native
representation.

This is the layer that makes "own the definition, rent the harness"
operational.

______________________________________________________________________

## What an adapter does

An adapter takes a prompt, a session, and runtime constraints (deadline,
budget, heartbeat), and orchestrates an evaluation against a specific
harness. Its job, in order:

- **Render.** Walk the prompt's section tree and produce the rendered
  text plus the bridged tool list.
- **Format.** Convert the rendered prompt and tool list into whatever
  shape the harness expects — Claude Agent SDK options, Codex
  app-server requests, ACP JSON-RPC messages, etc.
- **Execute.** Hand control to the harness. The harness drives the
  planning loop, calls back when a tool is invoked, and streams events
  during execution.
- **Bridge tools.** When the harness invokes a tool, route it through
  the transactional bridge so policies, snapshots, and feedback
  providers run uniformly.
- **Inject feedback.** Collect feedback messages from triggered providers
  and deliver them to the harness's appropriate channel.
- **Verify completion.** Run the prompt's completion checker at
  termination; signal continuation if the checker reports incomplete.
- **Parse.** Convert the harness's final response into a typed result
  (or a structured-output record when one is declared).
- **Emit.** Publish lifecycle events through the session dispatcher;
  emit transcript entries through the unified emitter; record budgets
  consumed.

The boundary above an adapter is uniform. The boundary below it is
runtime-specific.

______________________________________________________________________

## What stays the same

The prompt is a closed object graph: rendering it, walking it, and
listing its tools is the same work regardless of harness. Sections,
tools, policies, feedback, completion checkers, resources, structured
output types — all of these are properties of the prompt, not of the
adapter. They render and resolve identically across runtimes.

This means swapping adapters is a configuration change, not a code
change. The same agent runs against multiple runtimes by binding the
prompt to a different adapter at the call site.

______________________________________________________________________

## What changes per harness

Several real things differ between runtimes; adapters absorb them.

- **Tool bridging mechanism.** Claude Agent SDK uses an in-process MCP
  server. Codex App Server uses dynamic tools over stdio. ACP-compatible
  agents (OpenCode, Gemini CLI) use an MCP HTTP endpoint. Same
  `Tool` definition, three different transport stories.
- **Sandbox model.** Claude uses bubblewrap on Linux and seatbelt on
  macOS. Codex uses workspace-write or read-only profiles. ACP runtimes
  use their own permission models. Configuration is exposed by the
  adapter, not the prompt.
- **Native tools.** Each runtime ships its own file, shell, and search
  tools. They do not appear in the WINK prompt definition; they are
  contributed by the harness. The prompt only declares the WINK-native
  tools the agent should also have.
- **Approval and permission modes.** Prompting modes ("ask", "auto",
  "bypass") are runtime-specific. Adapters expose them as config.
- **Structured output mechanism.** Some runtimes have native JSON Schema
  enforcement; others rely on prompt-level instruction. The adapter
  knows which.
- **Transcript shape.** Each runtime emits its own event stream; the
  adapter normalizes it into the unified transcript schema.

______________________________________________________________________

## The harness contract

WINK only integrates with **agentic harnesses** — runtimes that own:

- A planning / act loop.
- Native tools and sandboxing.
- Tool-call orchestration with retries and timeouts.
- Crash recovery and lifecycle.

This is a deliberate scope. A bare model API (e.g., calling an OpenAI or
Anthropic completion endpoint directly) is not an execution harness;
WINK does not target it. The reasoning: a bare API leaves planning,
tool orchestration, and recovery as the framework's responsibility, and
WINK's bet is precisely that those responsibilities should be rented
from a vendor runtime, not implemented in-house.

______________________________________________________________________

## Throttling, retries, and budgets

Adapters absorb operational concerns that are common to all harnesses
but expressed differently in each.

- **Throttling.** Backoff on rate-limit signals, with caller-specified
  retry budgets and provider-suggested delays.
- **Retries.** Network errors, server errors, and transient timeouts
  retry within the deadline.
- **Budget tracking.** Token usage and time elapsed are recorded after
  every response and after every tool call. Budget checks happen at
  defined checkpoints — before the next call, before the next tool —
  and raise a typed exception when exhausted.
- **Deadlines.** Wall-clock deadlines propagate from the call site
  through the adapter and into tool contexts. Tools that run
  long-but-bounded operations can extend their lease via heartbeat.

The definition declares the *constraints* (a budget, a deadline). The
adapter enforces them.

______________________________________________________________________

## Adapter Compatibility Kit

Behavioral parity across adapters is non-trivial: each runtime's quirks
can leak through. The Adapter Compatibility Kit (ACK) is a unified suite
of integration tests that any new adapter must pass — covering prompt
evaluation, tool bridging, event emission, transcript logging, error
handling, and transactional semantics.

A passing ACK run *certifies* that the adapter implements the WINK
contract correctly. The kit is the guard against silent divergence: when
an adapter starts behaving subtly differently from the others, ACK
catches it.

______________________________________________________________________

## Telemetry through the dispatcher

Every adapter publishes a uniform set of events through the session
dispatcher:

- **PromptRendered** — text and metadata at render time.
- **RenderedTools** — the tool schemas, correlated with the render.
- **ToolInvoked** — for each tool call, with parameters and result.
- **PromptExecuted** — at the end of evaluation.

These events are how observability works. Sessions subscribe; debug
bundles capture; transcripts derive. An adapter that does not emit them
is invisible to the rest of the framework.

______________________________________________________________________

## Implementing a new adapter

The shape is fixed: a class implementing the adapter protocol. The
required work is mechanical:

- Define a client config and a model config.
- Render the prompt once at evaluation start.
- Bridge tools through the shared transactional wrapper.
- Wrap any SDK-specific failures as a typed evaluation error.
- Dispatch the four lifecycle events at the right moments.
- Emit transcript entries through the shared emitter.
- Pass ACK.

The non-mechanical part — the only adapter-specific work — is the
translation between the runtime's protocol and WINK's contracts. Everything
else is infrastructure that already exists.

______________________________________________________________________

## Anti-patterns

- **Adapter-specific prompt branching.** If a prompt has to carry "if
  on Codex, do X; if on Claude, do Y," the wrong layer is making the
  decision. Lift the difference into the adapter.
- **Definition logic that only works on one harness.** The Adapter
  Compatibility Kit will surface this; treat ACK failures as design
  problems, not test problems.
- **Adapters that own state.** State belongs in the session. Adapter
  instance variables that accumulate across calls are usually a sign of
  state that wants to be a slice.
- **Reinventing throttling per adapter.** The shared throttle policy is
  there because every adapter needs the same semantics. Extend it
  rather than rolling your own.

______________________________________________________________________

## Pointers

- [DEFINITION-VS-HARNESS](01-DEFINITION-VS-HARNESS.md) — the boundary
  the adapter implements.
- [TOOLS](04-TOOLS.md) — what gets bridged through the adapter.
- [STATE](05-STATE.md) — events the adapter publishes to the
  dispatcher.
- [OBSERVABILITY](14-OBSERVABILITY.md) — the unified transcript and
  debug bundle the adapter feeds.
- [PRINCIPLES](PRINCIPLES.md) §14 — the same definition runs on every
  harness.
