# BACKLOG — Proposed Refactoring Milestones

> Living output of [REVIEW.md](REVIEW.md). Items below are **candidates** (`C#`)
> surfaced by the Phase 0 metrics pass; they need Phase 1–2 confirmation (root
> cause + evidence per package) before promotion to `M9.md`+. Ground rules from
> [GOAL.md](GOAL.md) apply: first principles, simplify, no back-compat, each
> milestone green.
>
> `M1`–`M8` (the sandbox arc) are **not** repeated here.

## Candidate backlog (first metrics pass — unconfirmed)

| ID | Candidate | Dim. | Evidence (Phase 0) | Sev | Size |
|----|-----------|------|--------------------|-----|------|
| C1 | **Type-safety hardening** — drive down `cast`/`ignore` via generics & protocols | 4 | 320 `cast(`, 82 `# pyright: ignore`; concentrated in `adapters`(109)/`prompt`(107)/`runtime`(78)/`serde`(44) | High | L |
| C2 | **Adapters decomposition** — extract more into `_shared`; shrink per-adapter modules | 1,5 | `adapters` = 14.5k LOC (largest pkg); 4× `adapter.py` ~700 LOC; parallel `_guardrails`/`_transcript`/`_async` per adapter | High | L |
| C3 | **serde simplification & typing** — narrow surface, kill casts | 3,4 | 44 type-escapes in 2.3k LOC; `serde` is foundation, so debt propagates upward | Med | M |
| C4 | **runtime cohesion** — split `agent_loop.py` (719) / `session.py` (651) by responsibility | 1 | two of the largest modules; `runtime` 12.6k LOC | Med | M |
| C5 | **Coverage-exclusion audit** — justify or remove `# pragma: no cover` | 10 | 137 occurrences; risk of masked untested branches | Med | S |
| C6 | **Security pass** — review every `# nosec` + subprocess/path site | 9 | 36 `# nosec`; subprocess in `_git_ops`, debug, formal, (future) `Shell` | Med | S |
| C7 | **De-dynamic** — replace `getattr`/`hasattr` duck-typing with protocols/unions | 4,13 | 138 `getattr`, 18 `hasattr` | Low | M |
| C8 | **Code-length burn-down** — shrink modules near the ~700-LOC ceiling; empty the baseline ledger | 1 | 15 modules ≥650 LOC; 1 grandfathered entry | Low | M |
| C9 | **Spec ↔ code reconciliation** — verify each `specs/*.md` matches reality | 11 | ~40 specs; drift accrues as code moves (e.g. this refactor) | Med | M |
| C10 | **Test-suite shape** — fakes over mocks, prune slow tests, add property/fuzz where apt | 10 | tests 92.7k vs src 62.3k LOC (1.5×); audit composition, not just % | Med | M |

Severity = quality/maintenance risk if left. Size = rough effort (S/M/L).

______________________________________________________________________

## Worked example (demonstrates the template)

The strongest metric-backed candidate, scoped as a real proposal. Promote to
`M9.md` once Phase 1–2 confirm the per-package root causes.

```markdown
# M9 — Type-Safety Hardening

**Depends on:** none (can interleave with the sandbox arc).
**Unlocks:** safer downstream refactors. **Dimensions:** 4 (type safety).
**Priority:** high (reach: foundation→adapters). **Size:** L (phase per package).

## Objective
Eliminate the bulk of `cast(...)` and `# pyright: ignore` by giving the underlying
types the expressiveness they were standing in for — generics, protocols, and
discriminated unions — so annotations become the source of truth again.

## Why
320 `cast(` and 82 `# pyright: ignore` are concentrated in `serde` (44, the
foundation — debt here propagates everywhere), `adapters` (109), `prompt` (107),
and `runtime` (78). Each escape is a place the type system stopped helping;
clusters usually mark one missing abstraction, not 300 unrelated holes.

## Scope
- **serde first** (foundation): replace casts in parse/dump with typed dispatch;
  this likely removes casts in callers for free.
- Then `prompt`, `runtime`, `adapters`: convert per hot spot; delete the escape
  with the fix, never relocate it.
- Add a checker (or extend `toolchain/checkers`) that **ratchets** the per-package
  `cast`/`ignore` count downward (a baseline ledger like `code_length_baseline`).

## New shape
Generic `serde.parse[T]`/`dump` returning precise types; `@overload`s replacing
`cast` at call sites; protocols where `getattr` duck-typing exists.

## Risks & mitigations
- *Churn.* Phase strictly by package; one green PR each.
- *Hidden behavior in casts.* Port the exact tests around each removed cast.

## Exit criteria
- Per-package `cast`/`ignore` counts drop to the ratchet target; no new escapes.
- `make check` green (ty + pyright strict) at each phase.
```

______________________________________________________________________

## Status log

| Date | Action |
|------|--------|
| 2026-06-03 | Backlog seeded from first metrics pass (C1–C10). Full Phase 1–2 per-package review pending. |
