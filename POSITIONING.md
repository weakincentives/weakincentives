# WINK: The Agent-Definition Layer for Unattended Agents

WINK is the agent-definition layer: prompts, tools, policies, and feedback that
stay stable while runtimes change.

## Unattended Agents Have Two Parts

### 1. The Agent Definition (what you own and iterate)

- **Prompt**: a structured decision procedure, not a loose string.
- **Tools**: the capability surface; the only sanctioned place where side
  effects occur.
- **Policies**: enforceable invariants that constrain tool use and state
  transitions.
- **Feedback**: "are we actually done?" checks that prevent premature success
  and steer recovery when the agent drifts.

### 2. The Execution Harness (what the runtime provides)

- The planning/act loop that drives tool calls across turns.
- Sandboxing and permissions for side-effectful work (filesystem, shell,
  network).
- Tool-call orchestration (invocation lifecycle, retries/backoff, throttling).
- Scheduling, crash recovery, and operational guardrails (deadlines/budgets).
- Multi-agent orchestration (when you run in a multi-agent harness).

## WINK's Thesis

The harness will keep changing (and increasingly comes from vendor runtimes),
but your agent definition should not. WINK makes the definition a first-class
artifact you can version, review, test, and port across runtimes via adapters.

## What WINK Gives You Today

- A prompt system built from typed Sections (instructions + tools + progressive
  disclosure in one tree).
- Typed tools with a controlled execution boundary; transactional behavior so
  failures don't leak partial state.
- Session/state primitives designed for inspectability and regression testing.
- Policy primitives for gating tool calls (e.g., sequencing dependencies,
  read-before-write).
- Feedback primitives in the harness layer where needed (e.g., task-completion
  checking for Claude Agent SDK), with specs that generalize this pattern
  further.

## How This Changes How You Build

Instead of spending your complexity budget on bespoke orchestration graphs, you
invest in durable definition assets:

- a prompt that reliably shapes reasoning,
- a tool surface that is narrow and intentional,
- policies that encode "never do X before Y" constraints,
- and feedback that encodes "done means Z".

Then you can swap harnesses—local loop vs provider runtime—without rewriting
the core agent logic.

## One Sentence You Can Reuse

> "You write the agent definition (prompt, tools, policies, feedback); the
> runtime owns the harness (planning loop, sandboxing, orchestration). WINK
> keeps the definition portable while runtimes evolve."
