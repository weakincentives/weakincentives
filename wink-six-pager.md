# WINK: The Agent-Definition Layer

A six-pager on what we are building, why now, and the world we are building for.

______________________________________________________________________

## 1. Thesis

Every production agent has two halves: the **definition** (what the agent *is*
— its prompt, its tools, its policies, its sense of "done") and the **harness**
(what the runtime *does* — the planning loop, sandboxing, retries, scheduling,
recovery). For a brief moment in the history of this technology, both halves
were built by the same team in the same repository. That moment is ending.

The harness is becoming a vendor product. Anthropic ships one. OpenAI ships
one. OpenCode and Gemini ship one. Each is a sophisticated, sandboxed,
crash-resilient agent runtime with its own native file tools, its own planning
strategy, its own permission model, its own protocol for communicating with the
outside world. None of them are the same. All of them are improving faster than
any individual team can keep up with. And — this is the part most agent
frameworks have not yet absorbed — the filesystem they operate on is no longer
the developer's laptop. It is increasingly a remote, ephemeral, sandboxed
workspace that the harness provisions on the agent's behalf.

WINK exists because the *definition* should not be locked to any single
harness, and should not be entangled with how the harness is implemented. WINK
is the agent-definition layer: a portable, typed, hierarchical artifact that
fully specifies *what the agent is*, while adapters carry it to whichever
harness wins on any given day, in any given environment, for any given task.

The bet is simple: **the harness is the depreciating asset; the definition is
the durable one**. Teams that conflate them will rewrite their agents every
time a vendor ships a better runtime. Teams that separate them will ship
durable agent IP that survives the churn.

______________________________________________________________________

## 2. Tenets (in priority order)

1. **The prompt *is* the agent.** Instructions and tools live together in one
   hierarchical document. There is no separate registry to synchronize, no
   routing layer to maintain, no configuration that can silently drift from
   documentation. If a capability is not in the prompt tree, the agent does not
   have it.

1. **Own the definition; rent the harness.** Planning loops, sandboxes,
   retries, deadlines, and crash recovery are commodities — and vendors are
   competing to commoditize them faster than you can build them. Your agent's
   reasoning structure, tool surface, and completion criteria are your IP.
   Spend complexity there.

1. **Policies over workflows.** Encode the invariants the agent must satisfy,
   not the steps it must follow. A workflow tells the agent *how*; it
   fractures the moment reality deviates from the plan. A policy tells the
   agent *what must hold*; it preserves the agent's ability to reason its way
   through the unexpected.

1. **Transactional side effects.** Every tool call is an atomic transaction.
   Session state and filesystem state snapshot before the call and roll back
   on failure. Failed tools leave no partial trace. This is the property that
   makes retries safe and recovery tractable.

1. **State is an event ledger.** Every mutation flows through pure reducers on
   typed events. State is immutable, inspectable, and replayable. Snapshots
   are first-class. There is no hidden mutable state to debug at 3am.

1. **Fail closed; surface reasoning.** When a policy is uncertain, deny. When
   completion is uncertain, block stop. Always emit a reason the agent can
   read and the operator can audit. Silent failure is the enemy of
   trustworthy autonomy.

______________________________________________________________________

## 3. The Situation

For the first generation of agent frameworks, the runtime was a thin shell
around a chat completion endpoint. You wrote a loop. You wrote the tool
plumbing. You wrote the retry logic. You wrote the sandbox. The framework's
contribution was the prompt template and the JSON schema for tool calls. The
agent and the runtime were the same artifact because the runtime was barely
anything at all.

That world is gone.

In its place, a small number of providers are now shipping full agentic
runtimes — *harnesses* that own the entire execution loop: planning, tool
sequencing, sandboxing at the operating-system level, network policy, deadline
enforcement, lease extension on long jobs, crash recovery, and increasingly
multi-agent orchestration. Anthropic's Claude Agent SDK, OpenAI's Codex App
Server, the vendor-neutral Agent Client Protocol implemented by OpenCode and
Gemini — these are not chat wrappers. They are agent platforms. They compete
on sandbox fidelity, on tool-call latency, on observability, on recovery
semantics, on the breadth of native capabilities they provide out of the box.
They will continue to compete, and the winners will keep changing.

A second shift is happening alongside the first: **the filesystem the agent
operates on is becoming remote**. Today most agents still run against a local
working tree on a developer machine. Tomorrow they will run unattended against
ephemeral, sandboxed, provider-provisioned workspaces — sometimes co-located
with the model, sometimes in dedicated containers, sometimes in customer VPCs,
almost never on the same machine as the operator. The harness will own the
provisioning, the lifecycle, and the access controls of that workspace. The
agent definition will need to be indifferent to where the filesystem actually
lives. A `read_file` that worked locally must work identically against a
container, a remote sandbox, or a virtualized workspace, with the same
preconditions, the same rollback semantics, and the same observable behavior.

Teams building unattended agents today are caught between these two shifts.
They have legacy agent code that assumes a local Python loop, a local
filesystem, and a single provider. They want to take advantage of the new
harnesses — for the sandboxing, for the recovery, for the native tools — but
porting their agent means rewriting everything. They want to run against
remote filesystems — for the security, for the scale, for the
reproducibility — but their tools assume `open()` works. Every harness
upgrade, every move to a remote workspace, every new vendor capability
becomes a multi-week porting exercise.

This is the kind of friction that ossifies an ecosystem. Without a
definition layer, agent teams will pick a harness and stay there — not
because it is the best harness, but because they cannot afford to leave.

______________________________________________________________________

## 4. The Problem We Are Solving

There is no industry-standard artifact that represents "the agent" — only
"the agent running on a specific harness." When you describe your agent to
another engineer, you describe a tangle: a prompt living in one file, a tool
registry assembled by another, a planning loop in a third, a set of sandbox
configurations in a fourth, a set of ad-hoc post-hoc checks in a fifth, and
implicit assumptions about the filesystem strewn across all of them. This
tangle is not portable, not reviewable as a single unit, not testable in
isolation, and not robust to harness churn.

The problems this creates are concrete:

- **Lock-in by accident.** Teams adopt a harness for one good reason
  (sandboxing, a specific model) and discover six months later that their
  "agent" is unportable. Switching costs grow with every iteration.
- **Workflow brittleness.** Faced with the unexpected, agents whose behavior
  is encoded as procedural workflows either fail outright or sprout decision
  trees that nobody can audit. The reasoning capability that motivated using
  an LLM in the first place is suppressed by the orchestration code wrapped
  around it.
- **Premature termination.** Unattended agents declare victory while real
  work remains undone. Without an explicit, definition-level notion of "done
  means X," there is no mechanism to catch this — and no portable mechanism
  to enforce it across harnesses.
- **Untestable definitions.** Agent behavior cannot be unit-tested because
  there is no agent artifact to test — only the running system. Regression
  testing requires standing up the full harness, which is slow, expensive,
  and provider-coupled.
- **Filesystem assumptions.** Tools written for a local filesystem must be
  rewritten for a remote one. Path handling, transactional semantics, and
  permission models diverge. The agent that worked on a laptop fails
  silently in a sandboxed workspace.
- **Operational opacity.** Without a unified state model, debugging an
  unattended run requires reconstructing what happened from logs that were
  never designed to be reconstructed. Why the agent did what it did is
  unanswerable.

The root cause beneath all of these is the same: **there is no first-class
definition layer that survives independently of the harness**. Everything
people call "the agent" is actually "the agent and its runtime, fused."

______________________________________________________________________

## 5. The Bet: WINK

WINK proposes that the agent definition is a first-class, portable artifact
with four parts — and that those four parts, and only those four parts,
constitute the *agent*:

**The prompt** is a typed, hierarchical document. Sections nest. Each
section bundles its own instructions, its own tools, its own preconditions
for being enabled, and its own children. The prompt is rendered into
deterministic markdown with deterministic heading levels, and the set of
tools available to the agent is exactly the set of tools attached to the
enabled sections. There is no second registry; there is no possibility of
documentation drift. A section that is disabled removes its tools from the
agent's surface as cleanly as it removes its instructions from the rendered
text. This is what we mean by "the prompt is the agent": the document fully
determines what the agent can think and do.

**The tools** are the agent's side-effect boundary. They are typed at both
input and output. They run as transactions: snapshot, execute, commit or
roll back. They access the world through narrow protocols — a `Filesystem`
protocol that abstracts over local, in-memory, sandboxed, and remote
backends; a clock protocol that is injectable for testing; a resource
registry that resolves dependencies with explicit lifetimes. The tools do
not know whether the filesystem they are talking to is on this machine or
on another one. They do not know whether the harness running them is
Anthropic's or OpenAI's or someone else's. This indifference is the
property that makes the definition portable.

**The policies** are declarative invariants that gate tool use. "A file
must be read before it can be overwritten." "Deployment cannot occur before
tests have passed." "A long operation must heartbeat to extend its lease."
Policies are checked synchronously, fail closed, surface their reasoning to
the agent in natural language, and compose by conjunction. They do not
script the agent's behavior; they constrain it. The agent remains free to
find any path that satisfies the invariants. When a policy denies a call,
the agent receives an explanation it can reason about, not an opaque error.

**The feedback** is the definition's answer to "are we actually done?" Two
mechanisms cooperate. *Feedback providers* observe the trajectory and inject
advisory guidance into the context — gentle, non-blocking, time- or
event-triggered. *Task completion checkers* block termination outright when
the definition's own success criteria are not satisfied. Both are declared
on the prompt, both travel with the agent, both work identically across
every harness WINK supports. This is how the definition encodes "done means
X" rather than relying on the agent's self-assessment.

These four parts — prompt, tools, policies, feedback — are the agent. They
are version-controlled, reviewable in a single change, testable without any
harness present, and portable across harnesses by means of *adapters*.

An adapter is the thinnest possible translation layer. It takes the
rendered prompt and bridges the tools to whatever protocol the chosen
harness speaks: in-process MCP for Claude's SDK, dynamic tool registration
for Codex, HTTP MCP for ACP-compatible runtimes. It surfaces the same
events and the same transcripts regardless of the underlying mechanism.
A compatibility kit — the Adapter Compatibility Kit, ACK — exists precisely
so that any new adapter can be certified against a shared behavioral
contract: same tool semantics, same transactional rollback, same policy
enforcement, same completion checking. The definition does not care which
adapter is running. The adapter does not get to redefine what the agent is.

The animating assumption is that **providers will continue to develop into
full agent-harness vendors**, and that those harnesses will increasingly
operate on **remote, provider-provisioned filesystems** that the developer
never touches directly. WINK is designed for that world. The Filesystem
protocol is bytes-first and streaming-first, with backend implementations
for in-memory, host-mounted, and containerized workspaces — and the same
protocol extends naturally to remote backends without any change to the
tools that consume it. The transactional model already assumes that
snapshot and restore are operations on an abstract workspace, not on a
local directory. The adapter layer already assumes that the harness owns
process lifecycle, that the workspace is something the harness provisions,
and that the agent definition must travel light enough to be shipped to
wherever the harness wants to run it.

In short: WINK is not a framework that wraps an LLM. It is the layer that
makes "the agent" a thing you can carry from one runtime to another, intact.

______________________________________________________________________

## 6. What Success Looks Like

A team adopts WINK and writes their first unattended agent. The prompt is a
single hierarchical document. The tools are typed and transactional. The
policies are three lines each and read like English. The completion check
encodes the deliverable. The team unit-tests the policies and the completion
check without instantiating any harness at all — pure functions on pure
state.

They run it the first time on a local Claude Agent SDK harness. It works.
They run the same definition on Codex the next week to compare model
behavior. The agent definition does not change; only the adapter
instantiation does. They run it a month later on a remote workspace
provisioned by a new provider whose SDK did not exist when they wrote the
agent. Same definition. New adapter. Same observable behavior.

A new harness emerges that is meaningfully better at sandboxing, or at
multi-agent orchestration, or at remote-workspace provisioning. An adapter
is written. It passes the compatibility kit. Every existing WINK agent —
across every team in the organization — gains the new capability at the
cost of a configuration change. Nobody rewrites a prompt. Nobody rewrites
a tool. The harness improvement compounds across the entire fleet.

A junior engineer joins the team and reads exactly one document — the
prompt — to understand what the agent is. They see the policies in the
same file. They see the completion criteria in the same file. They see
the tool descriptions in the same file. They do not have to chase
references across five repositories. They can submit a meaningful change
in their first week.

An auditor asks "what is this agent allowed to do, and what is it
required to do?" The answer is a single file. Not a runtime. Not a
deployment. Not a log. A file.

That is the success state. It is not a small one. It implies that the
center of gravity of agent engineering has moved from runtime plumbing to
definition design — that the durable craft is no longer "writing a planning
loop" but "writing a prompt tree, a tool surface, a policy set, and a
completion criterion that together specify the agent you actually want."
WINK exists to make that craft possible.

______________________________________________________________________

## 7. Frequently Asked Questions

**Is WINK trying to replace the agent harnesses?**

No. The harnesses are what makes the modern agent ecosystem possible. They
solve hard, important problems — sandboxing at the OS level, sub-second
tool dispatch, crash recovery, multi-agent orchestration — that no
individual team should be reinventing. WINK depends on them. WINK only
insists that the definition of the agent not be fused to any one of them.

**Why not just standardize on one harness and accept the lock-in?**

Because the harness market is in its earliest, most volatile phase. The
ranking among harnesses on sandbox fidelity, recovery semantics, native
tool surface, and supported models changes quarterly. Models migrate
across harnesses. Customers have preferences about which harnesses they
will permit in their environments. Locking the agent definition to the
current best harness is locking it to a snapshot of a moving frontier.

**Why "policies over workflows"? Are workflows wrong?**

Workflows are right when the sequence is genuinely invariant and the cost
of failure is lower than the cost of adaptation — protocol handshakes,
deterministic pipelines, regulated procedures. For unattended LLM agents,
that is rarely the case. The whole point of using an LLM is reasoning
under uncertainty. Workflows suppress that capability; policies preserve
it. WINK supports both, but defaults to policies because they compose
better, fail more informatively, and let the agent recover from situations
the author did not anticipate.

**What does "the prompt is the agent" actually buy us?**

Three things. First, co-location: the instructions for using a tool and
the tool itself live in the same section, so they cannot drift. Second,
dynamic scoping: disabling a section removes both its instructions and its
tools atomically, which makes it safe to compose agents from optional
capabilities. Third, reviewability: the rendered prompt is the complete
specification of what the agent will see and do, so a code review of the
prompt is a code review of the agent.

**How does this hold up when the filesystem is remote?**

It is designed for that case. Tools talk to a `Filesystem` protocol, not
to `open()`. The protocol is bytes-first and streaming-first so that
remote backends can implement it without buffering whole files. The
transactional model snapshots and restores through the same protocol,
which extends to remote backends as naturally as it does to local ones.
The harness owns the workspace lifecycle; the definition is indifferent
to where the workspace actually lives. None of this requires a tool author
to think about remoteness; that concern is absorbed by the backend.

**Where does WINK end and the harness begin?**

The harness owns: the planning loop, the model call, sandboxing,
process-level isolation, retries and backoff against rate limits, scheduling,
deadline enforcement at the OS level, multi-agent orchestration, and
workspace provisioning. WINK owns: the prompt structure, the tool
signatures and handlers, the policies, the feedback and completion
checking, the event/reducer state model, the transactional rollback
semantics over session and workspace, and the typed structured-output
contract. The adapter is the seam: it carries the WINK-owned definition
across to the harness-owned execution and translates events back.

**What happens when a new harness ships?**

An adapter is written against its protocol. The Adapter Compatibility Kit
certifies that the adapter respects the WINK behavioral contract: tools
are transactional, policies are enforced, completion is checked, events
are emitted, transcripts are structurally equivalent across adapters.
Once it passes the kit, every existing definition becomes runnable on the
new harness. Definitions do not change.

**Is this overengineered for current agent workloads?**

For chat-shaped assistants that run in the foreground for thirty seconds,
yes — the existing harnesses are more than sufficient. The bet WINK is
making is about *unattended* agents: agents that run for hours,
unobserved, against remote workspaces, with consequences for getting it
wrong. For that class of workload — which is the class that is growing
fastest — the definition/harness split is not overengineering. It is the
minimum structure required to keep the system auditable and portable.

**What is the worst case for adopters?**

The harness market consolidates onto one winner, lock-in becomes
acceptable, and the portability story is uninteresting. Even in that
scenario, adopters retain the other benefits of the definition layer: a
single reviewable artifact for the agent, declarative policies, explicit
completion criteria, transactional tool semantics, and a testable
definition that does not require the harness to validate. The portability
is the headline, but it is not the only value.

**What is the worst case for us?**

That we underinvest in the adapter compatibility contract and adapters
drift apart, so "same definition, different harness" becomes "almost the
same definition, mostly the same harness." This is the failure mode we
guard against most carefully. ACK is not optional; it is the load-bearing
artifact that keeps the abstraction honest.

______________________________________________________________________

## 8. Appendix: The Cleavage Diagram

```
┌────────────────────────────────────────────────────────────────┐
│  DEFINITION (WINK owns, you author, ships with your repo)       │
│                                                                 │
│   • Prompt          — hierarchical sections, typed              │
│   • Tools           — typed I/O, transactional, FS-protocol     │
│   • Policies        — fail-closed invariants                    │
│   • Feedback        — guidance providers + completion checking  │
│   • State model     — event ledger, pure reducers, snapshots    │
│                                                                 │
│   Portable. Reviewable. Testable without a harness.             │
└──────────────────────────────┬─────────────────────────────────┘
                               │  Adapter (thin, certified by ACK)
                               ▼
┌────────────────────────────────────────────────────────────────┐
│  HARNESS (vendor owns, you rent, may live on a remote machine)  │
│                                                                 │
│   • Planning loop, tool sequencing, model call                  │
│   • OS-level sandbox, network policy, permissions               │
│   • Workspace provisioning (local, container, remote)           │
│   • Deadlines, retries, lease extension, crash recovery         │
│   • Multi-agent orchestration                                   │
│                                                                 │
│   Improving fast. Differentiates between vendors. Swappable.    │
└────────────────────────────────────────────────────────────────┘
```

The line between the two boxes is the only architectural commitment that
matters. Everything else follows.
