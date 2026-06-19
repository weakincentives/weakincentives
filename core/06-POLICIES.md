# Policies

A **policy** is a hard guardrail. It gates tool invocations: when a tool
call would violate a constraint, the policy denies it and returns an error
result instead of executing. Policies are how WINK enforces invariants
without prescribing workflows.

______________________________________________________________________

## What a policy is, and isn't

A policy expresses *what must hold*: "the file must be read before it can
be overwritten", "tests must pass before deployment runs", "the agent
must not exceed N tool calls per minute". It does not express *how* the
agent should accomplish anything.

A policy is **not**:

- A pre-step in a workflow (that would couple sequencing into the
  framework).
- A property of a single tool (that would make it impossible to gate the
  same constraint across multiple tools).
- A part of the harness (it is a property of the agent's goal and travels
  with the prompt).
- A retry mechanism (it denies; it does not retry).

______________________________________________________________________

## Why policies, not workflows

A workflow encodes a sequence: read → parse → identify → patch → test.
When something unexpected happens — a file is missing, a parse fails, a
test flakes — the workflow has three options: fail, skip with bad state,
or branch. Each branch added to handle a new edge case grows the workflow
into a brittle decision tree the original author can't keep in their head.

A policy encodes a constraint: "you must have read the file before you
overwrite it." The agent now has *every* path that satisfies the
constraint available to it. If reading fails, the agent reasons about why
and tries something else. The framework does not need to anticipate every
recovery path; the policy denies actions that would violate the
invariant, and the model handles the rest.

| Workflow | Policy |
|---|---|
| Steps to execute | Constraints to satisfy |
| Fails or branches on the unexpected | Lets the agent reason |
| Sequentially coupled | Independently composable |
| Agent is the executor | Agent is the reasoner |

This is the central design move: trade prescriptive sequences for
declarative invariants, and let the agent's reasoning fill in the rest.

______________________________________________________________________

## Anatomy of a policy

A policy is a small object with two methods:

- A `check` that runs *before* the tool handler, given the tool name, its
  parameters, and the current context. It returns a *decision*: allow or
  deny, with a reason on denial.
- An optional `on_result` that runs *after* the tool handler succeeds.
  This is where the policy updates its tracking state — recording that a
  precondition has been met.

Policies maintain their state in session slices, like any other consumer
of the session. This means policy state is snapshotted with the session,
visible to debug bundles, and inspectable via the same query API as any
other slice.

______________________________________________________________________

## Fail-closed by default

When a policy can't decide whether to allow a call — missing context,
contradictory state, an unrecognized parameter — it denies. The denial
message goes back to the model as the tool result. The agent reads it and
reasons about how to proceed.

This is a deliberate posture. The alternative, fail-open, leads to
silent invariant violations that the agent never knows about. Fail-closed
makes constraints visible: the agent learns that something it tried to do
isn't allowed and adjusts.

The denial message is part of the policy's contract. A vague message ("not
allowed") teaches the agent nothing. A specific message ("file
`config.yaml` exists but was not read first") tells it exactly what to do
next.

______________________________________________________________________

## Composition

Multiple policies can govern the same tool. They compose by conjunction:
*every* policy must allow for the call to proceed. Each policy is
evaluable in isolation; composition is independent.

This is what makes policies scalable. Adding a new constraint never means
unwinding existing logic; it means adding another small policy that
expresses the new invariant. Removing a constraint never risks breaking
others.

______________________________________________________________________

## Where policies live

Policies attach to the prompt definition — either at the template level
(governing every tool the prompt provides) or scoped to a specific
section. They are not adapter configuration. They travel with the prompt
across harnesses.

This is consistent with the broader principle that constraints belong to
the *agent's goal*, not to the *runtime that executes it*. A "read before
write" rule is a property of how the agent should behave, regardless of
which model, sandbox, or runtime is in use.

______________________________________________________________________

## Two examples worth understanding

**Sequential dependency.** Some tools require others to have run first
unconditionally: "deploy" requires "test" and "build" to have succeeded.
A policy expresses this as a mapping from gated tool to its required
predecessors. It records each successful invocation in session state and
denies the gated tool until its dependencies are met.

**Read before write.** Some constraints are parameter-keyed: a write to
file `X` requires a prior read of file `X`, but a write to file `Y` is
unconstrained until `Y` exists. A policy tracks read paths in session
state and denies writes to paths that exist but haven't been read.

The same shape supports many real constraints: rate limits, idempotency
keys, prerequisite checks, and so on. The policy is small; the constraint
is precise.

______________________________________________________________________

## Anti-patterns

- **Workflow in policy clothing.** A "policy" that denies anything except
  a single allowed sequence is a workflow with extra steps. If the
  constraint reduces to "do these things in this order," the agent has no
  reasoning room.
- **Over-constraining.** A set of policies that leaves only one valid
  path defeats the purpose. Policies should constrain — not legislate —
  the agent's behavior.
- **Stateful misuse.** Policies maintain *tracking* state (what has been
  done), not *orchestration* state (what should happen next). Conflating
  these two pushes the policy toward being a workflow.
- **Silent denial.** A policy that denies without explanation forces the
  agent to guess. Always include a reason.
- **Policy logic in handlers.** If "always read before writing" lives in
  every write handler, the constraint is a code-duplication problem
  instead of a definition-level concern.

______________________________________________________________________

## When workflows are appropriate

Policies are not always the right answer. Use a workflow when:

- The sequence is genuinely invariant, e.g., a protocol handshake.
- Failure is preferable to adaptation, e.g., a critical migration.
- The agent lacks the reasoning capacity to recover.
- A human is in the loop and providing oversight at every step.

For unattended agents driven by capable LLMs, policies are almost always
the better fit.

______________________________________________________________________

## Pointers

- [PRINCIPLES](PRINCIPLES.md) §3 — *Policies, not workflows.*
- [FEEDBACK](07-FEEDBACK.md) — soft guidance during execution
  (advisory, not gating).
- [COMPLETION-CHECKING](08-COMPLETION-CHECKING.md) — the third tier of
  control: verifying that the agent's stop condition is met.
- [STATE](05-STATE.md) — where policies persist their tracking state.
- [TOOLS](04-TOOLS.md) — what policies gate.
