# AgentLoop

The **AgentLoop** is the user-facing orchestration shell — the thing you
subclass to build a runner. It owns the lifecycle of one evaluation:
build a prompt, execute it through an adapter, hand back a typed result,
clean up. Where adapters bridge the prompt to a harness, AgentLoop is
the outer loop *above* the adapter that decides what to send and what
to do with the response.

This is the unit that long-running agents run as. Multiple AgentLoops
can run in one process, each subscribed to its own request stream.

______________________________________________________________________

## Where AgentLoop sits

A clean mental model:

```
        Request → AgentLoop.prepare() → (Prompt, Session)
                       │
                       ▼
                   Adapter ─────► Harness (planning loop, sandbox)
                       │
                       ▼
        Response ← AgentLoop.finalize() ← parsed output
```

The harness owns its planning loop. The adapter bridges prompt and
harness. AgentLoop is the level *above* — it decides which prompt to
build for which request, what to do with the parsed output, and when to
release resources.

______________________________________________________________________

## What you implement

AgentLoop is generic over a request type and an output type. A subclass
implements two factory methods:

- **`prepare(request) → (Prompt, Session)`.** Build the prompt and the
  session that will record the run. This is where you bind parameters,
  inject resources, and register reducers.
- **`finalize(prompt, session, output)`** *(optional)*. Post-process
  the parsed output before it is returned. Useful when the model
  produces a partial structured value and the loop fills in derived
  fields from session state.

Everything else — visibility-expansion retries, resource lifecycle,
debug bundling, error wrapping — is handled by the framework.

______________________________________________________________________

## Two execution modes

The same loop runs in two modes.

**Direct.** Call `execute(request)` and get back a typed result. This
is what you use for tests, evaluations, one-shot scripts, and
synchronous request handling.

**Mailbox-driven.** Call `run()` and the loop subscribes to a message
queue, processes requests as they arrive, and replies via the queue's
reply channel. This is the long-running mode for distributed
deployments.

The choice is operational. The agent definition does not change.

______________________________________________________________________

## Requests are content-addressed

Every request carries an identifier the orchestrator owns. The loop
treats that identifier as the durable handle for the unit of work.

- **Same identifier + same content = same work.** A retry — whether
  from the orchestrator's own retry logic, from message-queue
  redelivery, or from an explicit reattach after a process restart —
  produces the same response. The sandbox does not re-execute the
  work; it returns the in-flight or completed result.
- **Same identifier + different content = explicit conflict.** A
  request that reuses an identifier but changes any contract field
  (input, deadline, declared tools, output type) is rejected. The
  orchestrator must use a fresh identifier or accept the original
  intent.

The contract hash covers everything that could change behavior. This
is what makes mailbox-driven mode safe — duplicate deliveries are
detected and treated as the same work, not as parallel attempts.

(See [Durable Work](19-DURABLE-WORK.md) for the cross-boundary
semantics.)

______________________________________________________________________

## Reattaching to in-flight work

A loop that crashes or restarts mid-evaluation can resume by sending
the same request identifier on a new connection. The sandbox returns
the current state — in-flight, completed, or failed — and the new
loop continues from there. Long-running evaluations are not bound to
a single loop instance.

This is what makes the loop robust against orchestrator-side
failures: a deployment that rolls workers does not abandon work; it
hands work off. A session is the durable unit; loop instances are
disposable.

______________________________________________________________________

## Compute and work are different lifecycles

The loop coordinates work; the sandbox runs compute. These are not
the same lifecycle.

- **Stopping a sandbox does not abandon sessions.** Work is preserved;
  files in the workspace remain; the loop can reattach to in-flight
  evaluations after a restart.
- **Ending an evaluation does not stop compute.** A finished
  evaluation just means the session is no longer hosting active work.
  The sandbox continues serving other sessions.
- **Connection drops are neither.** The loop reconnects; the sandbox
  preserves work for a bounded grace window.

The loop's job is to push work forward. Lifecycle decisions about
compute and connections happen at a different layer.

______________________________________________________________________

## What the loop owns

For each invocation:

- **Resolution of constraints.** Budgets, deadlines, and resources can
  come from configuration defaults *or* from the request itself.
  Request-level overrides win. A fresh budget tracker is created per
  execution.
- **Prompt resource lifecycle.** The prompt's resource context opens
  before evaluation and closes after — even if visibility expansion
  forced a retry, even if the adapter raised. Closeable resources see
  a clean shutdown every time.
- **Visibility-expansion retries.** When the adapter raises a
  visibility-expansion signal, the loop applies the override to the
  session and retries. The retry count is bounded — a pathological
  prompt cannot loop forever.
- **Result envelope.** Adapter responses are wrapped in a typed result
  carrying the output, the session ID, the run context, the completion
  time, and (optionally) a debug bundle path.
- **Error wrapping.** Exceptions from the adapter become typed errors
  on the result envelope. The loop never raises across its `execute()`
  boundary in the success path; failures surface as a result with an
  error field set.

______________________________________________________________________

## Debug bundling

If the loop is configured with a bundle target, every execution
produces a self-contained zip archive: request, response, session
before/after, logs, transcript, configuration, environment, metrics.
The bundle is finalized only after the prompt's resources are
released, so any artifacts those resources produced are captured.

(See [Observability](14-OBSERVABILITY.md) for the bundle layout.)

______________________________________________________________________

## Composition

AgentLoop is the substrate other loops build on:

- **EvalLoop** wraps AgentLoop to run a dataset of samples and score
  the outputs. (See [Evaluation](16-EVAL-LOOP.md).)
- A custom analysis loop can reuse the same shell to run scheduled or
  triggered evaluations.

The principle: anything that produces a `(Prompt, Session)` per
incoming request can be a loop. The framework provides the lifecycle.

______________________________________________________________________

## Why this layer exists

Without AgentLoop, every consumer would re-implement the same glue:
bind the prompt, open the resource context, call the adapter, catch
the visibility-expansion signal, retry, parse the output, close
resources, package the bundle. Every consumer would also re-implement
the mailbox-vs-direct split.

By making this shell standard, three things become true:

- **Definitions are uniformly executed.** The same lifecycle runs in
  tests, evaluations, and production. Nothing is "production-only."
- **Long-running agents are first-class.** Switching between direct
  execution and a queue-driven worker is a configuration change, not
  a rewrite.
- **Cross-cutting features attach in one place.** Debug bundles,
  budget tracking, deadline enforcement, run-context propagation —
  the loop is where they hook in once and apply to every consumer.

______________________________________________________________________

## What AgentLoop is not

- **Not the planning loop.** The planning loop — when to call which
  tool, when to think, when to stop — is the harness's job. AgentLoop
  drives one prompt evaluation at a time.
- **Not a multi-agent orchestrator.** Coordinating multiple agents is
  a higher-level concern. Each agent runs in its own loop;
  coordination happens above.
- **Not a workflow engine.** The loop executes one request at a time
  through one prompt. Branching, gating, or parallelism across
  prompts is a higher layer.
- **Not a transaction boundary.** Tool calls inside an evaluation are
  transactional (see [Transactions](11-TRANSACTIONS.md)); the loop as
  a whole is not.

______________________________________________________________________

## Anti-patterns

- **Treating disconnect as cancel.** A loop that aborts work whenever a
  connection drops creates duplicate execution and lost progress.
  Reattach using the request identifier; the sandbox preserves work
  for a bounded grace window.
- **Sharing session state across unrelated requests.** A session is
  durable, but it is durable *for one logical thread of work*. A loop
  that piggybacks on someone else's session leaks state across users.
  When work is unrelated, give it a fresh session identifier; when
  work is a continuation, reattach to the existing one deliberately.
- **Mutating the cached template in `prepare()`.** The template is
  often cached on the loop instance for reuse. Mutating it changes
  the prompt for later requests. Bind parameters to a fresh `Prompt`
  each time.
- **Catching the visibility-expansion signal in user code.** The
  framework catches it. User code that intercepts it breaks the
  re-render contract.
- **Threading manual cleanup.** The prompt context handles resource
  cleanup. Adding `try/finally` blocks around `execute()` for
  resource release is a sign that resources weren't bound properly.

______________________________________________________________________

## Pointers

- [PROMPT-IS-THE-AGENT](02-PROMPT-IS-THE-AGENT.md) — what `prepare()`
  returns.
- [STATE](05-STATE.md) — the session AgentLoop owns.
- [RESOURCES](09-RESOURCES.md) — the lifecycle the loop manages.
- [PROGRESSIVE-DISCLOSURE](10-PROGRESSIVE-DISCLOSURE.md) — the
  visibility expansion the loop retries.
- [ADAPTERS](13-ADAPTERS.md) — the harness boundary the loop calls.
- [OBSERVABILITY](14-OBSERVABILITY.md) — the bundles the loop
  produces.
- [EVAL-LOOP](16-EVAL-LOOP.md) — how AgentLoop composes into testing.
