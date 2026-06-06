# BACKLOG — Continuous Hardening & Horizon

> The **release** milestones (R1–R3, `M1`–`M16`) live in the [GOAL.md](GOAL.md)
> roadmap. This file is the **hardening track** that runs alongside them — the
> review-driven cleanups governed by [REVIEW.md](REVIEW.md) — plus amendments to
> R1, what already landed on `main`, and the post-release horizon. Ground rules
> from GOAL: first principles, simplify, no back-compat, each item green.

## Review run — 2026-06-03 (refreshed after rebasing onto `main`)

Phase 1–2 was a **7-agent reviewer team** over all of `src/weakincentives/`, ~56
evidence-cited findings. Post-rebase metrics: **316** `cast(`, **82**
`# pyright: ignore`, **14** `# ty: ignore` (adapters 107 · prompt 103 · runtime 78
· serde 45). Wave 1 shipped upstream (below). The remaining cleanups don't each
warrant a release slot but materially raise quality, so they ride the hardening
track and feed specific releases.

## Hardening track (continuous; feeds releases)

| ID | Item | Feeds | Size | Evidence |
|----|------|-------|------|----------|
| H1 | Adapter orchestration consolidation | R1, M12 | L | triplicated turn-loop (`acp/_prompt_loop.py`, `codex…/_protocol.py`, `claude…/_sdk_execution.py`); `acp/_guardrails.py` ≡ `codex…/_guardrails.py` (~110 LOC); **three** ephemeral homes (`_ephemeral_home.py` in claude/codex/opencode); bridge logs hardcode `claude_agent_sdk.bridge.*` |
| H2 | Type-escape burn-down (ratchet) | R2 | L | 316 `cast(` + 96 ignores; serde `schema.py`↔`_coercers.py` taxonomy dup; event-type cast blocks; `MappingProxyType` cluster in `snapshots.py` (the generic-specialization unification itself is M9) |
| H3 | Prompt selection/render DRY | R1 | S–M | `_enabled_predicate.py` ≡ `_visibility.py` normalizer (~120 LOC); `Section.render` vs `MarkdownSection.render_override` duplicate assembly |
| H4 | Error-handling & concurrency + coverage masks | R3/M14 | M | `RedisMailbox._lock` decorative + reaper swallows all; `codex…/adapter.py:268 except BaseException`; coverage-masked token-aggregation/feedback paths; unescaped SQL identifiers (`_query_tables.py:114-133`) |
| H5 | Runtime de-leak & encapsulation | R2 | M | `session.py:602-644` backcompat block + 6 file-level `reportPrivateUsage` waivers; `_agent_loop_bundle.py:57 _LoopLike` mirror protocol; give `SliceStore`/`ReducerRegistry` real internal APIs — clean boundaries help modular, testable definitions |

**H1** is a co-requisite of R1 (the sandbox arc edits adapters) and the foundation
of the M11 conformance kit — do it early. **H2** is an ongoing ratchet (add a
per-package `cast`/`ignore` baseline like `code_length_baseline.txt`) and lifts the
quality of the M9 serialization work. Each H-item is independently shippable and
green; promote to a numbered milestone only if it grows past a single PR.

## Amendments to R1 (M1–M8) — do not create new milestones

Findings that strengthen the sandbox arc; fold in when those land:

- **M1** ← filesystem path-validation is write-only (`_resolve_path` is the real
  guard); `_resolve_path` re-resolves root every call (`_host.py:116`);
  `FilesystemSnapshot` hard-codes git internals → opaque `SnapshotRef`.
- **M3** ← `git clean -xfd` on restore (`_git_ops.py:255`) deletes *ignored* files;
  document `HostFilesystem` single-instance threading.
- **M4** ← collapse the three `Filesystem | None` shortcuts
  (`ToolContext`/`FeedbackContext`/`TaskCompletionContext`); drop the
  `singleton_cache` snapshot scan and the `_instantiation_order` rebuild
  (`resources/context.py:257-291`).
- **M5** ← `WorkspaceDigestSection` (`contrib/tools/digests.py`) is the same
  god-object smell as `WorkspaceSection`; fold into the auto-preview mechanism.

## Landed on `main`

The first review run's Wave 1 shipped upstream (#1161–#1165): dead-code &
backcompat-shim sweep (`Tool.wrap`, `safe_hook_wrapper`, `ACPSessionState`,
`_async.py` stubs, `snapshotable.py`, `session/dataclasses.py`); type-safe
`session_id` + validator/mailbox/digests de-dynamicization; and spec/doc
reconciliation. Verified gone/resolved on the rebased tree.

## Horizon (past R3)

- **Sub-agent orchestration as a definition primitive** — an `AgentDefinition`
  (M9) may declare sub-agents; the harness fans them out. Builds on M9 + M14.
- **Multi-harness routing / fallback** — pick (or fail over) the best harness per
  task using `HarnessCapabilities` (M10) and eval scores (M15).

## Status log

| Date | Action |
|------|--------|
| 2026-06-03 | REVIEW.md run by a 7-agent team; ~56 findings. |
| 2026-06-03 | Rebased onto `main`; Wave 1 (#1163–#1165) landed upstream; metrics re-measured; reconfigurable-egress folded into R1. |
| 2026-06-03 | Re-derived the plan from first principles around the agent-definition-over-harnesses mission: R1 (One Environment, M1–M8), R2 (M9–M12), R3 (Trustworthy Runs, M13–M16); cleanups moved to this hardening track (H1–H4); `*Spec` taxonomy adopted in GOAL. |
| 2026-06-03 | Corrected R2: definition is **versioned, modular code** (not serialized data) and portability is **contract-level, not output-level**. R2 rebuilt as Modular / Testable-in-isolation / Capability-negotiation / Realize-ACK (M9–M12); runtime de-leak moved here as H5; M13 replay reframed (diff expects differences). |
