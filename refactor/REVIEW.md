# REVIEW — Library Review & Milestone-Generation Workflow

> A **repeatable process** for auditing the whole library *and* this refactor
> plan, then emitting prioritized, well-formed refactoring work that drives toward
> best-in-class. Findings land in [BACKLOG.md](BACKLOG.md)'s hardening track and
> graduate to a numbered release milestone (see [GOAL.md](GOAL.md)) when one grows
> past a single PR.
>
> Same ground rules as [GOAL.md](GOAL.md): think from first principles, prefer
> deletion/simplification, **backwards compatibility is a non-goal**, and every
> milestone is independently shippable and lands green (`make check`, 100%
> coverage).

______________________________________________________________________

## When to run

Before starting a refactor wave, after large merges, and on a periodic cadence.
The workflow is idempotent: re-running refreshes the backlog and metrics, it does
not duplicate accepted milestones.

## Inputs & outputs

| | |
|---|---|
| **Inputs** | `src/weakincentives/`, `tests/`, `specs/`, `refactor/GOAL.md` + `M*.md`, `toolchain/checkers/`, the metrics from Phase 0 |
| **Outputs** | A findings log; hardening-track items in [BACKLOG.md](BACKLOG.md); amendments folded into the milestone files |

## Roles

- **Orchestrator** — runs Phase 0/3/4/5/6, dispatches reviewers, owns the backlog.
- **Reviewer agents** — one per package cluster (Phase 1/2), read-only,
  **evidence-based** (every claim cites `file:line` or a metric). Fan out in
  parallel; each returns a package one-pager, not a file dump.

______________________________________________________________________

## Phase 0 — Calibrate & baseline

1. Read `GOAL.md` (the five cores + tenets), the milestone files (`M*.md`),
   `CLAUDE.md`, and the specs touching the area
   under review. Record the **target architecture** and the **quality bar**
   (strict typing, 100% coverage, 10 s/test, 4-layer boundaries, DbC on public
   APIs, serde over Pydantic, injected clocks, no monkeypatching).
1. Capture metrics (reproducible commands):

```bash
# LOC by package
for d in src/weakincentives/*/; do n=$(find "$d" -name '*.py' | xargs wc -l | tail -1 | awk '{print $1}'); echo "$n ${d}"; done | sort -rn
# Largest modules (cohesion pressure)
find src/weakincentives -name '*.py' | xargs wc -l | sort -rn | head -20
# Type-escape hatches, coverage exclusions, security waivers, dynamic access
for p in 'cast(' ': Any' 'Any]' '# pyright: ignore' '# ty: ignore' '# pragma: no cover' '# nosec' 'getattr(' 'TODO'; do
  echo "$(grep -rn -- "$p" src/weakincentives --include='*.py' | wc -l)  $p"; done
# Debt ledgers & layering
cat toolchain/checkers/code_length_baseline.txt
```

Snapshot the numbers (dated) at the bottom of this file and in `BACKLOG.md`.

## Phase 1 — Map (fan-out)

Walk packages in layer order (per `toolchain/checkers/architecture.py::_PACKAGE_LAYER`:
foundation → core → adapters → high-level). Dispatch one reviewer per cluster
(e.g. `filesystem`+`serde`+`dataclasses`; `runtime`; `prompt`; `adapters`;
`cli`+`debug`; `contrib`+`evals`). Each returns a **one-pager**:

- Responsibilities and whether they are cohesive (one reason to change?).
- Public surface (`__all__`) — is it minimal and intentional?
- Dependencies in/out — any cross-layer or circular imports?
- Module/function sizes vs the code-length budget.
- The top 3 smells with evidence.

## Phase 2 — Assess against the rubric

For each package, evaluate every dimension below with the listed **detection
tactic**. A dimension is a finding only with concrete evidence.

| # | Dimension | What "best in class" looks like | Detection tactic |
|---|-----------|--------------------------------|------------------|
| 1 | **Cohesion / SRP** | One reason to change per module/class | Largest-file list; god-objects mixing concerns |
| 2 | **Coupling / layering** | Acyclic, foundation→high-level only | `architecture.py`; `private_imports.py`; import graph |
| 3 | **API surface** | Narrow protocols; no redundant convenience | Oversized `Protocol`s; duplicated methods across backends |
| 4 | **Type safety** | Annotations are truth; few escapes | `cast(` / `: Any` / `# pyright: ignore` counts by package |
| 5 | **Duplication / DRY** | Shared logic factored to `_shared`/facade | Parallel impls across adapters; copy-paste blocks |
| 6 | **Error handling** | Fail-closed; typed; never swallowed | Bare `except`; ignored returns; inconsistent error types |
| 7 | **Concurrency** | Documented thread-safety; no hidden state | Module globals; locks; `THREAD_SAFETY.md` drift |
| 8 | **Performance** | No needless IO/allocation on hot paths | Per-call snapshots; client-side search; repeated parsing |
| 9 | **Security** | Justified subprocess/path/auth | `# nosec` count; `subprocess`; path traversal; token handling |
| 10 | **Tests** | Fakes over mocks; fast; property/fuzz where apt | Slow tests; mock-heavy suites; coverage *shape* (not just %) |
| 11 | **Docs / specs** | Spec ↔ code in sync | Spec claims vs implementation; `docs.py` link/codeblock checks |
| 12 | **Dead code** | No unused exports/params/shims | `deptry`; unused `__all__`; unreferenced helpers |
| 13 | **Consistency** | Follows CLAUDE.md patterns uniformly | `FrozenDataclass`, `@require/@ensure`, `serde`, injected `clock` |

## Phase 3 — Record findings

Each finding is a row in the working log:

```
F-NNN | dimension | evidence(file:line / metric) | root cause | impact | proposed change | blast radius | risk | confidence
```

Reject anything without evidence or a root cause. Prefer findings whose fix
*removes* code.

## Phase 4 — Form milestones

Cluster related findings into milestone proposals using the **template** below
(consistent with the milestone files, plus priority metadata). De-duplicate against the
existing milestones and `GOAL.md`; if a finding strengthens an existing milestone,
amend it rather than adding one.

## Phase 5 — Prioritize & sequence

Score each proposal and sort:

```
priority = (impact × reach) / (cost × risk)
```

- **impact** quality/maintenance gain; **reach** how much of the codebase it
  touches positively; **cost** effort; **risk** chance of regression.
- Group into waves. Within a wave, order by dependency. **Every** milestone must
  remain independently shippable, respect the 4-layer boundaries, keep tests
  under 10 s, and hold 100% coverage.

## Phase 6 — Record & iterate

Append findings to `BACKLOG.md`'s hardening track; graduate an item to a numbered
release milestone when it grows past a single PR; mark landed work; refresh the
metrics snapshot. Re-run on cadence.

______________________________________________________________________

## Anti-pattern catalog (this codebase)

Hunt these specifically — each has burned us before or is visible in the metrics:

- **Fat protocol reimplemented per backend** (the `Filesystem` story M1 fixes) —
  look for the same shape elsewhere.
- **God-object sections/adapters** mixing intent, materialization, rendering, and
  lifecycle (`WorkspaceSection`).
- **Per-adapter bespoke logic** that belongs in `adapters/_shared` (cwd
  resolution, guardrails, transcript, async bridging).
- **Type-escape clusters** — `cast(`/`# pyright: ignore` hot spots standing in for
  a missing generic or protocol.
- **Coverage masks** — `# pragma: no cover` hiding a real untested branch.
- **Dynamic access** — `getattr`/`hasattr` where a `Protocol` or union would type.
- **Spec/code drift** — a spec asserting behavior the code no longer has.
- **Cross-layer imports** outside `TYPE_CHECKING`.

______________________________________________________________________

## Milestone proposal template

```markdown
# M<N> — <Title>

**Depends on:** … **Unlocks:** …
**Dimensions:** <rubric #s>  **Priority:** <score>  **Size:** S/M/L

## Objective         (1–2 sentences)
## Why               (first-principles rationale; the smell + its root cause)
## Scope             (New / Rewrite / Move / Delete — file-grounded)
## New shape         (key type/code sketch)
## Risks & mitigations
## Exit criteria      (verifiable; ends with “make check green”)
```

## Reviewer guardrails

- Evidence or it didn't happen — cite `file:line` or a metric.
- Diagnose root cause before proposing; no speculative rewrites.
- Prefer the change that deletes the most code for the least risk.
- Respect `MODULE_BOUNDARIES.md`; never propose a cross-layer shortcut.
- Keep each proposal green-able in isolation; no "big bang" milestones.

______________________________________________________________________

## Baseline metrics — snapshot (2026-06-03)

| Metric | Value |
|--------|-------|
| `src/` LOC | ~62,300 |
| `tests/` LOC | ~92,700 |
| Largest packages | `adapters` 14.5k · `runtime` 12.6k · `prompt` 10.9k |
| Largest modules | several ~700 LOC (`codex…/_protocol.py` 741, `runtime/agent_loop.py` 719, `prompt/tool.py` 700) |
| `cast(` | 316 |
| `Any]` / `: Any` | 354 / 79 |
| `# pyright: ignore` / `# ty: ignore` | 82 / 14 |
| `# pragma: no cover` | 137 |
| `# nosec` | 36 |
| `getattr(` | 138 |
| Type-escapes by package | `adapters` 107 · `prompt` 103 · `runtime` 78 · `serde` 45 |
| Code-length debt ledger | 1 grandfathered entry (`codex_app_server/_protocol.py`) |

Refresh with the Phase 0 commands on each run. Numbers above are post-rebase onto
`main` (#1161–#1165); the first run's quick wins landed upstream, so the counts
barely moved — the residual escapes are the classifier/protocol-shaped ones the
H1 ratchet targets.
