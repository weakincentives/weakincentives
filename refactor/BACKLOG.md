# BACKLOG — Proposed Refactoring Milestones

> Output of [REVIEW.md](REVIEW.md). Ground rules from [GOAL.md](GOAL.md): first
> principles, simplify, no back-compat, each milestone independently shippable and
> green. The sandbox arc `M1`–`M8` is not repeated here.

## Review run — 2026-06-03 (refreshed after rebasing onto `main`)

Phase 1–2 was executed by a **7-agent reviewer team** (one per package cluster),
read-only and evidence-based, over all of `src/weakincentives/`. ~56 findings.

**Refresh:** since the run, `main` merged PRs #1161–#1165. Three of them implement
the entire **Wave 1**, which is now retired (see *Landed* below). Re-measured
metrics: **316** `cast(`, **82** `# pyright: ignore`, **14** `# ty: ignore`
(barely moved — #1162 did construction-validation/frozen-dataclass hardening, not
the classifier/protocol work M16 targets). M12–M18 were re-verified present on the
rebased tree and remain open.

## Wave plan

| ID | Milestone | Wave | Dims | Size | Status |
|----|-----------|------|------|------|--------|
| M9 | Dead-code & backcompat-shim sweep | 1 | 12,13 | S | **landed** (#1163) |
| M10 | Typed boundaries: `session_id` + de-dynamic | 1 | 4,13,3 | S–M | **landed** (#1164) |
| M11 | Spec ↔ code reconciliation | 1 | 11 | S | **landed** (#1165) |
| [M12](M12.md) | Unify adapter turn-loop + guardrails in `_shared` | 2 | 5,1 | L | proposed |
| [M13](M13.md) | Shared adapter primitives (home/MCP/bridge/errors) | 2 | 5,6,13 | M–L | proposed |
| [M14](M14.md) | De-leak runtime decomposition | 2 | 1,2,4 | M | proposed |
| [M15](M15.md) | CLI/debug bundle & query consolidation | 2 | 1,5,8 | L | proposed |
| [M16](M16.md) | Type-escape reduction (cast/ignore burn-down) | 3 | 4 | L | proposed |
| [M17](M17.md) | Prompt selection/render DRY | 3 | 5,6 | S–M | proposed |
| [M18](M18.md) | Error-handling & concurrency + coverage-mask audit | 3 | 6,7,9,10 | M | proposed |

Wave 2/3 milestone bodies are promoted to files (`M12.md`–`M18.md`).

## Landed on `main` (Wave 1)

- **M9 — dead-code & shim sweep** (#1163): deleted `Tool.wrap`,
  `safe_hook_wrapper`, `ACPSessionState` + accumulator, `session/dataclasses.py`,
  `runtime/snapshotable.py` (dup `Snapshotable`), and the `acp`/`codex` `_async.py`
  stubs. Verified gone on the rebased tree.
- **M10 — typed boundaries** (#1164): `SessionProtocol.session_id` now exists
  (`runtime/session/protocols.py:70`); **0** `getattr(session, "session_id")` sites
  remain; validators/mailbox/digests de-dynamicized.
- **M11 — spec reconciliation** (#1165): serde `__type__`, prompt `HIDDEN`, and ACP
  session-reuse docs reconciled. (Residual: the *bundled* docs copies under
  `src/weakincentives/docs/` still mention `ACPSessionState`; they regenerate from
  source — verify the next `wink docs` build.)

## Sandbox abstraction update — reconfigurable egress

New requirement folded into the sandbox arc (not a standalone milestone): the
`Sandbox` gains **egress configuration** backed by a **proxy sidecar that can be
reconfigured at any time**. `SandboxSpec.egress: EgressPolicy` seeds a default-deny
allowlist; `Sandbox.configure_egress(policy)` hot-reloads the sidecar (control
plane — runtime/policy-driven, not model-facing). Threaded through [GOAL.md](GOAL.md)
(concept + sketch + *Egress Control*), [M3](M3.md) (`EgressPolicy`,
`configure_egress`), [M6](M6.md) (transport op), and [M7](M7.md) (the container
egress chokepoint). Replaces the earlier vague `network: NetworkPolicy | None`
field on `SandboxSpec`.

______________________________________________________________________

## Amendments to M1–M8 (do not create new milestones)

Findings that strengthen the existing sandbox arc — fold in when those land:

- **M1** ← filesystem path-validation is write-only (`_host.py` validates on write
  but not read; `_resolve_path` is the real guard); `_resolve_path` re-resolves
  root every call (`_host.py:116`); `FilesystemSnapshot` hard-codes git internals
  (`commit_ref`/`git_dir`) → use the planned opaque `SnapshotRef`.
- **M3/M18** ← `git clean -xfd` on restore (`_git_ops.py:255`) deletes *ignored*
  files; document `HostFilesystem` single-instance threading.
- **M4** ← collapse the three `Filesystem | None` context shortcuts
  (`ToolContext`/`FeedbackContext`/`TaskCompletionContext`); the M4 text should
  name the latter two twins. Drop the `singleton_cache` snapshot scan
  (`transactions.py`) and the `_instantiation_order` rebuild in
  `resources/context.py:257-291`.
- **M5** ← `WorkspaceDigestSection` (`contrib/tools/digests.py`) is the same
  god-object smell as `WorkspaceSection` (with a `summary` setter that swallows the
  parent write); fold into the auto-preview/selector mechanism.

______________________________________________________________________

## Status log

| Date | Action |
|------|--------|
| 2026-06-03 | REVIEW.md Phase 1–6 by a 7-agent team. ~56 findings → M9–M18; Wave 1 promoted to files; M12–M18 scoped; M1–M8 amendments recorded. |
| 2026-06-03 | Rebased onto `main` (#1161–#1165). Wave 1 (M9–M11) landed upstream — files retired, status set to *landed*. Metrics re-measured. Folded the reconfigurable-egress requirement into M3/M6/M7 + GOAL. Promoted M12–M18 to files. |
