# Core

This folder is the conceptual foundation of Weak Incentives (WINK). It is a
*library without code*: the durable ideas, the shape of the abstractions, and
the way they fit together — without API surfaces, file paths, or
implementation detail. If you point an agent (or a new contributor) at this
folder, it should walk away with an accurate, opinionated mental model of
what WINK is for and how it thinks.

For implementation depth — class signatures, behaviors, errors — see the
`specs/` folder. `core/` answers *why* and *what*; `specs/` answers *how*.

______________________________________________________________________

## The thesis in one paragraph

An unattended agent has two parts: the **definition** (prompt, tools,
policies, feedback) and the **execution harness** (planning loop, sandboxing,
retries, scheduling). The harness is rented from a vendor runtime and will
keep changing. The definition is what you actually own. WINK is the layer
that makes the definition a first-class artifact: portable, typed, testable,
and stable across harness changes. Inside that layer, three commitments do
most of the work — *the prompt is the agent*, *state is event-driven and
immutable*, and *constraints are policies, not workflows*.

______________________________________________________________________

## How to read this folder

Read top-down for the mental model, or jump to the concept you need.

```
PRINCIPLES.md                    The rules every other doc follows.

01  DEFINITION-VS-HARNESS.md     What you own vs. what you rent.
02  PROMPT-IS-THE-AGENT.md       Prompts as typed hierarchical documents.
03  SECTIONS.md                  How prompts are composed.
04  TOOLS.md                     The capability surface.
05  STATE.md                     Event-driven state, slices, reducers.
06  POLICIES.md                  Hard guardrails (fail-closed gates).
07  FEEDBACK.md                  Soft guidance during execution.
08  COMPLETION-CHECKING.md       "Done means done" verification.
09  RESOURCES.md                 Dependency injection with scoped lifetimes.
10  PROGRESSIVE-DISCLOSURE.md    Token-efficient context expansion.
11  TRANSACTIONS.md              Atomic tool execution with rollback.
12  TYPED-CONTRACTS.md           Dataclasses everywhere; strict types.
13  ADAPTERS.md                  The harness boundary.
14  OBSERVABILITY.md             Snapshots, transcripts, debug bundles.

GLOSSARY.md                      One-line definitions for fast lookup.
```

______________________________________________________________________

## How the concepts connect

```
                       Definition vs. Harness  (01)
                                │
                                ▼
                    Prompt-is-the-Agent  (02)
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
         Sections (03)    Three-tier control     Resources (09)
              │           ┌────┴────┬──────┐       │
              ▼           ▼         ▼      ▼       ▼
           Tools (04)  Policies  Feedback Completion
              │         (06)      (07)    (08)
              │           │         │      │
              └────┬──────┴────┬────┴──────┘
                   ▼           ▼
            Transactions  State (events,
               (11)        slices,        ◄──── Progressive
                │          reducers)             Disclosure (10)
                │            (05)
                ▼            │
          Typed Contracts ◄──┘
              (12)
                │
                ▼
            Adapters (13)  ──►  Observability (14)
```

Read the diagram top-down: every concept below the line "Prompt-is-the-Agent"
is something you declare *on the prompt definition*. The bottom edge —
adapters and observability — is how the definition reaches a real harness
and how you see what happened.

______________________________________________________________________

## What `core/` is not

- **Not an API reference.** No class names, signatures, or import paths.
  Map back to `specs/` and the source tree when you need that.
- **Not a tutorial.** It does not walk you through building an agent. See
  `guides/` and `README.md` at the repo root.
- **Not a changelog.** Concepts described here are the durable shape of WINK,
  not the latest delta.
- **Not exhaustive.** Adapter quirks, debug bundle formats, and other
  implementation-bound details intentionally live in `specs/`.

______________________________________________________________________

## Working rule

If a change to the codebase makes a `core/` doc inaccurate, the codebase is
probably moving away from a foundational idea. Pause and decide whether the
core idea is shifting (update `core/`) or whether the change is local and
should be reframed (keep `core/` as the anchor). The `core/` folder is
deliberately small so it can stay accurate.
