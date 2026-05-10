# Definition vs. Harness

The single most important distinction in WINK is between the **agent
definition** (what you own) and the **execution harness** (what you rent).
Every other concept in `core/` is downstream of this split.

______________________________________________________________________

## The two halves

A working unattended agent has two layers.

**Agent definition.** Everything that describes *what the agent is for*:

- The prompt — the structured decision procedure the model reads.
- The tools — the typed capability surface the model can invoke.
- The policies — the invariants the agent must respect.
- The feedback providers — the in-flight nudges that keep it on track.
- The completion criteria — the conditions that constitute "done."

This layer is *yours*. You version it, review it, test it, and carry it
across runtimes.

**Execution harness.** Everything that describes *how the agent runs*:

- The planning / act loop that drives turns.
- Sandboxing and permissions for side-effectful work.
- Tool-call orchestration: invocation lifecycle, retries, throttling.
- Scheduling, deadlines, budgets, crash recovery.
- Native, runtime-provided tools (file IO, shell, web search).
- Multi-agent orchestration.

This layer is *rented*. It comes from a vendor runtime — Claude Agent SDK,
Codex, OpenCode (via ACP), Gemini (via ACP). It will keep changing, in ways
you do not control, on a schedule you do not set.

______________________________________________________________________

## Why the split matters

If you mix them, every harness change becomes a rewrite. Pin the planning
loop to a vendor SDK and a vendor model upgrade can move your agent's
behavior. Embed approval flows in your prompt and you cannot port to a
runtime with stricter sandboxing. Couple your retry logic to a specific
provider's error types and an SDK refactor breaks your tests.

If you keep them separate, the harness is a substitutable detail. The same
prompt, the same tool definitions, the same policies, and the same
completion checks run on any harness via an **adapter** that bridges them.

The split also clarifies *where to invest*. Your competitive advantage is
not in re-implementing planning loops; vendors will outrun you. Your
advantage is in the definition: the precise instructions, the right tools,
the policies that encode your operational reality, the completion criteria
that match your business.

______________________________________________________________________

## What goes where

When in doubt, ask: *is this property a fact about the agent's goal, or a
fact about how this runtime executes?*

| Property | Goes where |
|---|---|
| "Read the source before suggesting a patch." | Definition (a policy) |
| "Don't stop until `report.md` exists." | Definition (a completion check) |
| "Re-render the prompt when the section asks to expand." | Definition (visibility) |
| "Retry on HTTP 429 with exponential backoff." | Harness |
| "Run the tool in a Linux sandbox with no network." | Harness |
| "Reissue the request if the SDK times out." | Harness |
| "Use the model named `claude-opus-4-7`." | Harness (configuration) |
| "Emit a structured event for every tool call." | Harness (with definition contract) |

______________________________________________________________________

## Consequences

Several other principles fall out of this split.

- **Portability is the test.** If a feature only works on one harness, it is
  in the wrong layer. Adapter Compatibility Kits exist to *prove* parity.
- **Constraints travel with the prompt.** Policies, feedback providers, and
  completion checkers attach to the prompt definition itself, not to the
  adapter. They survive the move from one runtime to another.
- **The harness is allowed to be opinionated.** Adapters can use whichever
  protocol the runtime provides — MCP for Claude, dynamic tools for Codex,
  ACP for OpenCode. The definition does not change.
- **The harness is typically remote.** In production, the harness runs in
  a separate sandbox — a different process, often a different host —
  reachable only through a protocol. The orchestrator and the sandbox do
  not share memory, file descriptors, or filesystem paths. The
  definition layer is designed for this from the start; local
  in-process execution is a special case for development.
  (See [Remote Execution](18-REMOTE-EXECUTION.md).)
- **The harness owns operational guarantees.** Throttling, deadlines,
  crash recovery, and budgets are enforced by the harness layer using
  signals you supply at the boundary; the definition declares limits but
  does not implement enforcement.

______________________________________________________________________

## Anti-patterns

- **Defining the planning loop yourself.** The harness is built for this and
  improves continuously. Owning it puts you on the wrong treadmill.
- **Embedding harness configuration in prompts.** Approval prompts,
  retries, sandboxing modes — all of these are harness-shaped. They do not
  belong inside the definition.
- **Letting prompt text drift between harnesses.** If you have a different
  prompt per runtime, you no longer have a portable definition.
- **Using the harness for goal verification.** Whether the agent is
  *actually done* is a goal-shaped question. It belongs in the definition,
  via a completion checker.

______________________________________________________________________

## Pointers

- [PROMPT-IS-THE-AGENT](02-PROMPT-IS-THE-AGENT.md) — what the definition
  itself looks like.
- [ADAPTERS](13-ADAPTERS.md) — how the definition reaches a harness.
- [POLICIES](06-POLICIES.md), [FEEDBACK](07-FEEDBACK.md),
  [COMPLETION-CHECKING](08-COMPLETION-CHECKING.md) — the
  constraint mechanisms that ride with the definition.
