# BACKLOG — Continuous Hardening & Horizon

> The release milestones (R1–R3, [M1](M1.md)–[M9](M9.md)) live in the
> [GOAL.md](GOAL.md) roadmap. This file is the **hardening track** that runs
> alongside them — review-driven cleanups governed by [REVIEW.md](REVIEW.md) —
> plus what already landed on `main` and the post-R3 horizon. Ground rules from
> GOAL apply: first principles, simplify, no back-compat, each item green.

## Review run — 2026-06-03 (refreshed after rebasing onto `main`)

Phase 1–2 was a **7-agent reviewer team** over all of `src/weakincentives/`, ~56
evidence-cited findings. Post-rebase metrics: **316** `cast(`, **82**
`# pyright: ignore`, **14** `# ty: ignore` (adapters 107 · prompt 103 · runtime 78
· serde 45). Findings were folded directly into the milestones where they
belong — the filesystem/sandbox findings into M1–M3, the adapter duplication into
M6 stage 1, the bundle/query duplication into M7 stage 1 — leaving this track for
what cuts across releases.

## Review run — 2026-06-10

A second pass: **6 reviewer agents** (one per package cluster), explicitly
de-duplicated against M1–M9 and H1–H4; 29 findings survived spot-check
verification (claims that failed — e.g. "structured_output exports private
helpers", "`ReadOnlySliceAccessor` is dead" — were dropped). Dispositions:
seam/evidence/definition findings folded into [M5](M5.md)/[M6](M6.md)/[M7](M7.md)
stage 1; prompt-cohesion and runtime-encapsulation findings extend H2/H4 below;
the rest land as H5–H7. Reviewed-and-rejected near-misses, recorded so the next
run skips them: `clock.py`'s narrow protocols and injected `clock`/`sleeper`
params are repo law (CLAUDE.md), not debt; `TranscriptCollector` (639 LOC, one
caller) is cohesive and adapter-specific.

## Hardening track (continuous; feeds releases)

| ID | Item | Feeds | Size | Evidence |
|----|------|-------|------|----------|
| H1 | Type-escape burn-down (ratchet) | all cores | L | 316 `cast(` + 96 ignores. serde: `schema.py` ↔ `_coercers.py` duplicate the type taxonomy (parallel dispatch tables; `_NOT_HANDLED` sentinel) — unify behind one `TypeKind` classifier. runtime: event types not satisfying `SupportsDataclass` force duplicated cast blocks (`_session_helpers.py:66-75` ≡ `session_telemetry.py:40-49`); ~18 `MappingProxyType` casts in `snapshots.py` → one `frozen_map` helper. adapters: parse JSON-RPC frames once at ingress into the existing TypedDicts (`codex…/_protocol.py:407-677`). Add a per-package `cast`/`ignore` baseline checker (mirroring `code_length_baseline.txt`) so counts only fall. (The prompt-layer specialization cluster is [M5](M5.md) stage 2.) |
| H2 | Prompt selection/render DRY + layer cohesion | M5 | M | `_enabled_predicate.py` ≡ `_visibility.py` — the same 2×2 normalizer twice (~120 LOC; `*_requires_positional_argument` byte-identical) → one generic `normalize_selector[R]`. `Section.render` (`section.py:231`) vs `MarkdownSection.render_override` (`markdown.py:109`) duplicate heading+body+tools assembly → extract `Section._assemble`; replace the defensive `getattr(…, "accepts_overrides", …)` (`rendering.py:211`) with a plain read. Added 2026-06-10: `_validators.py` mixes five validation concerns in 527 LOC and reaches into registry privates (`:174-175,196,214`) — split by concern behind an internal registry validation API; `FeedbackContext` re-filters session slices on every query (`feedback.py:200-301`; `tool_calls_since_last_feedback_for_provider` scans the same slice twice per call) — compute per-provider indexes once per context; render-error enrichment drops nested section context (`rendering.py:486-497`) — chain section paths on re-raise |
| H3 | Error-handling & concurrency + coverage masks | M7/M8 | M | `RedisMailbox._lock` decorative (guards only `close()`; `_closed` read unlocked at `_redis.py:442-464`) + reaper wraps `_reap_expired` in `suppress(Exception)` (`:312`) hiding connection failures; `codex…/adapter.py:268 except BaseException` → `try/finally`; coverage masks over real paths (bundle token aggregation `_agent_loop_bundle.py:130-143` incl. a swallowing `except … pass`; hook feedback `_hook_tools.py:325`; `_docs.py` no-branch pragmas; `_validators` type-mismatch branches) — test them, drop the pragmas |
| H4 | Runtime de-leak & encapsulation | State core / M9 | M | `session.py:602-644` "Backward Compatibility Attributes" forwarding into `SliceStore`/`ReducerRegistry` privates + 6 file-level `reportPrivateUsage` waivers; `_agent_loop_bundle.py:57 _LoopLike` mirror-protocol over 7 private `AgentLoop` members + `self`-cast; `_reply_and_ack` mislabeled a "test shim" while production. Give the subsystems real internal APIs; delete the shims and waivers. Added 2026-06-10: snapshot/restore is fragmented across three modules with separate locking paths (`session.py:367-395,484-510` → `session_snapshotter.py:161-200`, while `session_cloning.py:35-100` re-implements overlapping helpers) — consolidate into `SessionSnapshotter`, one locking discipline; `ControlDispatcher`/`TelemetryDispatcher` are documentation-only aliases of one protocol (`events/__init__.py:104-107`) — make them distinct protocols or unify on one `Dispatcher` |
| H5 | Fail-closed session dispatch | State core / M8 | S–M | Two fail-open paths in `runtime/session/session_dispatch.py` contradict the fail-closed tenet. (1) A raising reducer is caught, logged, and **skipped** (`:178-188`, `except Exception` → `logger.exception` → `continue`) after earlier reducers' ops already committed — an event can half-apply with no rollback and no caller-visible error. (2) Dispatching an event type with no registered reducer silently falls back to `append_all` ledger semantics (`:132-139`) — a typo'd event type "works" forever, creating a slice nobody asked for. Decide the contract: collect reducer errors into a typed `DispatchError` (or a dead-letter slice) and make ledger semantics an explicit registration, documented on `SessionProtocol.dispatch` |
| H6 | Evals simplification | M9 | M | The package is over-written relative to its job. Ten near-identical fetch-filter-score evaluator closures (`_session_evaluators.py:44-251`); `all_of`/`any_of` differ only in the aggregator, `mean` vs `max` (`_evaluators.py:184-273`) → parameterized evaluator/combinator factories (~260 → ~80 LOC). `EvalLoop` fragments one flow over 7 one-caller private methods with bundle/non-bundle near-duplication (`_loop.py:149-500`; `_evaluate_sample_with_bundle` ≡ `…_without_bundle`; `_compute_score` returns different shapes per path) → ~3 methods, one scoring path. `is_session_aware` + 2 helpers hand-roll ~50 LOC of annotation introspection (`_evaluators.py:72-164`) → `typing.get_type_hints` + small fallback. `_coerce` re-implements `serde.parse` dispatch for one call site (`_types.py:93-117,177-180`) → delete. `_helpers.py:25-131` is 5–10-line mailbox-loop forwarders with docstrings longer than the code → inline into `EvalLoop` or demote to documented examples |
| H7 | Foundation tidy | substrate | S–M | `formal/` (~1.0k LOC + TLA+ tooling) sits in the foundation layer with exactly one consumer (`contrib/mailbox/{_redis_spec,_redis}.py`) → move to `contrib/` or an optional extra; foundation stays substrate-only. Dead `types/` exports: `ParseableDataclassT`, `JSONArrayT`, `JSONObjectT` have zero importers outside `types/` (verified; `SupportsDataclassOrNone` **is** widely used — keep) → delete. `Experiment` ships two behaviorally identical sentinels (`BASELINE` ≡ `CONTROL`, `experiment.py`) → keep `BASELINE`, document it as the control. `ResourceRegistry.of()` vs `.build()` are two construction APIs for one job (`resources/registry.py:71-150`) → pick one entrypoint, delete the other. `dbc/__init__.py` (545 LOC: predicates, decorator wrapping, invariant metaprogramming, `ContextVar` state) is cohesive enough to defer — split into `_predicates`/`_decorators`/`_invariants` only when next touched |

Each H-item is independently shippable and green; promote one to a numbered
milestone only if it outgrows a single PR.

## Landed on `main`

The first review run's quick wins shipped upstream (#1161–#1165): dead-code &
backcompat-shim sweep (`Tool.wrap`, `safe_hook_wrapper`, `ACPSessionState`,
`_async.py` stubs, duplicate `Snapshotable`, `session/dataclasses.py`); type-safe
`session_id` on `SessionProtocol` + validator/mailbox/digests de-dynamicization;
spec/doc reconciliation (serde `__type__`, prompt `HIDDEN`, ACP session reuse).
Verified resolved on the rebased tree.

## Horizon (post-R3)

- **Sub-agent orchestration as a definition primitive** — definitions declare
  sub-agents; the harness fans out natively where capable (declared via
  `AdapterCapabilities`), with the policy engine (M8) governing the tree. Builds
  on M5 + M6 + M8.
- **Multi-harness routing / fallback** — pick or fail over the harness per task
  using `AdapterCapabilities` (M6) and eval scores (M9). Routing, not identical
  execution.
- **Hosted sandboxes** — when providers ship managed environments, they implement
  `SandboxTransport` (M4) and WINK deletes its own provisioning for that path:
  the design-to-shrink strategy in action.

## Status log

| Date | Action |
|------|--------|
| 2026-06-03 | REVIEW.md run by a 7-agent team; ~56 findings. |
| 2026-06-03 | Rebased onto `main`; quick wins (#1163–#1165) landed upstream; metrics re-measured; reconfigurable egress folded into the sandbox arc. |
| 2026-06-03 | Plan re-derived around the agent-definition mission; `*Spec` taxonomy adopted; R2 corrected (definition = modular code; portability = contract-level). |
| 2026-06-03 | Naming: dropped the separate `*Spec` suffix — `*Config` is the single declarative-input idiom; `SandboxSpec` → `SandboxConfig`; the SDK's isolation `SandboxConfig` is renamed `IsolationConfig` and folded into provider config at M3. |
| 2026-06-03 | **Core-strengthening rewrite:** plan reorganized around five cores (Definition · State · Environment · Contract · Evidence) with aging tenets and non-goals; 16 milestones consolidated to 9 (stages preserved); adapter consolidation absorbed into M6, review amendments inlined into M1–M3; hardening track renumbered H1–H4. |
| 2026-06-10 | Second review run (6 agents, de-duplicated against the plan); 29 verified findings folded in: M5/M6/M7 stage-1 amendments, H2/H4 extended, H5–H7 added; near-misses recorded. |
