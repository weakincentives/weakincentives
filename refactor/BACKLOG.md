# BACKLOG — Proposed Refactoring Milestones

> Output of [REVIEW.md](REVIEW.md). Ground rules from [GOAL.md](GOAL.md): first
> principles, simplify, no back-compat, each milestone independently shippable and
> green. The sandbox arc `M1`–`M8` is not repeated here.

## Review run — 2026-06-03

Phase 1–2 executed by a **7-agent reviewer team** (one per package cluster),
read-only and evidence-based, covering all of `src/weakincentives/` (~62k LOC).
~56 findings. Recurring cross-cutting themes drove the milestone clustering below:

- **Dead code / backcompat shims** kept alive by their own tests (→ M9).
- **Dynamic-access escapes** — `getattr`/`hasattr` for statically-known members;
  the `session_id` foot-gun alone is ~20 sites (→ M10).
- **Spec/doc drift** — documented features with no implementation (→ M11).
- **Triplicated adapter orchestration** — the turn/continuation loop and
  guardrails copied across acp/codex/sdk (→ M12, M13).
- **Leaky decomposition** — helpers reaching back into owner privates via shims
  and private-mirror protocols (→ M14).
- **CLI/bundle duplication** — ~600 LOC of hand-written SQL builders + triple env
  schema (→ M15).
- **Type-escape clusters** — `cast`/`# pyright: ignore` standing in for a missing
  classifier/protocol (→ M16).

## Wave plan

| ID | Milestone | Wave | Dims | Size | Status |
|----|-----------|------|------|------|--------|
| [M9](M9.md) | Dead-code & backcompat-shim sweep | 1 | 12,13 | S | **promoted** |
| [M10](M10.md) | Typed boundaries: `session_id` + de-dynamic | 1 | 4,13,3 | S–M | **promoted** |
| [M11](M11.md) | Spec ↔ code reconciliation | 1 | 11 | S | **promoted** |
| M12 | Unify adapter turn-loop + guardrails in `_shared` | 2 | 5,1 | L | proposed |
| M13 | Shared adapter primitives (home/MCP/bridge/errors) | 2 | 5,6,13 | M–L | proposed |
| M14 | De-leak runtime decomposition | 2 | 1,2,4 | M | proposed |
| M15 | CLI/debug bundle & query consolidation | 2 | 1,5,8 | L | proposed |
| M16 | Type-escape reduction (cast/ignore burn-down) | 3 | 4 | L | proposed |
| M17 | Prompt selection/render DRY | 3 | 5,6 | S–M | proposed |
| M18 | Error-handling & concurrency + coverage-mask audit | 3 | 6,7,9,10 | M | proposed |

**Wave 1** (M9–M11) is mostly deletions and docs — safe, immediate, promoted to
files. **Wave 2** is structural consolidation. **Wave 3** is the broad,
phased-per-package hardening.

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
  (`transactions.py:378,419`) and the `_instantiation_order` rebuild in
  `resources/context.py:257-291`.
- **M5** ← `WorkspaceDigestSection` (`contrib/tools/digests.py`) is the same
  god-object smell as `WorkspaceSection` (with a `summary` setter that swallows the
  parent write, `:184-188`); fold into the auto-preview/selector mechanism.

______________________________________________________________________

## Wave 2 — structural consolidation

### M12 — Unify adapter turn-loop + guardrails in `_shared`

**Dims 5,1 · L · med risk.** The per-turn "send → collect text/usage → check
completion → re-prompt (≤10 rounds) → visibility check" state machine is
reimplemented three times: `acp/_prompt_loop.py:157-246`,
`codex_app_server/_protocol.py:180-247`, `claude_agent_sdk/_sdk_execution.py:140-220`.
And `acp/_guardrails.py:59-130` ≡ `codex_app_server/_guardrails.py:61-132` (~110
near-identical LOC; differ only in a feedback `content` key and a log prefix).
**Change:** add `_shared/_turn_loop.py::run_continuation_loop(*, send_turn, …)`
taking each adapter's one-turn closure; move `check_task_completion`/`append_feedback`
into `_shared/_guardrails.py` (parameterize `content_type`). **Exit:** one loop +
one guardrail impl; ~60 LOC deleted per adapter + ~110 from guardrails; green.

### M13 — Shared adapter primitives

**Dims 5,6,13 · M–L · med risk.**

- **Ephemeral home** — `claude_agent_sdk/_ephemeral_home.py:89-142` and
  `codex_app_server/_ephemeral_home.py:43-94` are byte-identical (`_copy_skill` +
  mount/lifecycle, ~150 LOC). Extract `_shared/_ephemeral_home.py` base
  (`skills_subdir=`); Claude adds settings.json/AWS, Codex becomes a ~30-line
  subclass.
- **One MCP builder** — `_shared/_bridge.py:614` and `acp/_mcp_http.py:86` both
  register `BridgedTool`s via `asyncio.to_thread`; consolidate in `_shared`.
- **De-brand the bridge** — `_shared/_bridge.py` logs `"claude_agent_sdk.bridge.*"`
  and defaults `adapter_name="claude_agent_sdk"` even for acp/codex (mislabels
  telemetry). Neutralize to `"bridge.*"`; make `adapter_name` required.
- **Error normalization** — Codex has none (`adapter.py:351-366` flattens all to
  `phase="request"`; duplicated `except CodexClientError` at `_protocol.py:133,200`).
  Add a `_shared` normalization facade; extract the `[-8192:]` stderr-tail constant.
- **Empty-response hook** — `opencode_acp`/`gemini_acp` differ only by a label and
  hardcode `prompt_name=""`; thread `prompt_name`; one shared `_noop`.

**Exit:** ~300 LOC deleted; telemetry correctly labeled; green.

### M14 — De-leak runtime decomposition

**Dims 1,2,4 · M · med risk.** `session.py:594-642` is a "backcompat" block
forwarding into `SliceStore`/`ReducerRegistry` privates with
`# pyright: ignore[reportPrivateUsage]`; 8 session modules carry file-level
`reportPrivateUsage` waivers; `_agent_loop_bundle.py:57-107` declares a `_LoopLike`
Protocol mirroring 7 private `AgentLoop` members (needs a `self`-cast at
`agent_loop.py:653`). **Change:** give `SliceStore`/`ReducerRegistry` the small
public APIs these callers need; delete the backcompat block + suppressions; make
the bundle helper take public collaborators (delete `_LoopLike`); fix/remove the
`_reply_and_ack` "test shim" (it is production, `agent_loop.py:700`). **Exit:** no
private-mirror protocols; suppressions gone; green.

### M15 — CLI/debug bundle & query consolidation

**Dims 1,5,8 · L · med risk.**

- **Query builders** — `cli/_query_builders.py` + `_query_environment.py`: ~10
  hand-written `CREATE TABLE` + flatten-insert clones (~600 LOC). Replace with a
  table-spec descriptor + one `_build_kv_table`.
- **Triple env schema** — env shape is typed in `debug/environment.py:105-218`, the
  SQLite DDL, and `cli/_bundle_store.py:536-629`. Derive columns + UI projection
  from the dataclasses via `serde`.
- **Zip opened per access** — `debug/_bundle_reader.py` opens the zip 5× (and per
  checksummed file) with a dead `_zip_file` field. Open once.
- **Swallow-all writes** — `BundleWriter` has 11 copy-pasted `except Exception`
  blocks; a "successful" bundle can silently lack Required artifacts. One
  `_safe_write` that records failures into the manifest.
- **Misc** — collapse `_docs.py` doc iterators (`:251-281`); split the `BundleStore`
  god-object and inject the bundle-lister (removes a late-import-to-satisfy-a-
  monkeypatch, which violates `CLAUDE.md`); fix SQL identifier quoting at
  `_query_tables.py:114-133`.

**Exit:** ~700 LOC deleted; green.

______________________________________________________________________

## Wave 3 — broad hardening (phased)

### M16 — Type-escape reduction (cast/ignore burn-down)

**Dims 4 · L · phased.** 320 `cast(` + 82 `# pyright: ignore`, concentrated in
serde/runtime/prompt/adapters. Targets, each deleting escapes with the fix:

- **serde** — one `_typeclass.py` `TypeKind` classifier unifying the duplicated
  taxonomy in `schema.py` and `_coercers.py`; replace the `_NOT_HANDLED` sentinel
  with a typed `CoerceResult`.
- **runtime** — make the event dataclasses satisfy `SupportsDataclass` (delete the
  duplicated `cast` blocks in `_session_helpers.py`/`session_telemetry.py`); a
  `frozen_map` helper for the 18 `MappingProxyType` casts in `snapshots.py`.
- **prompt** — one `GenericArgCapture` for `Tool`/`PromptTemplate`/`Section` (three
  parallel specialization mechanisms today); make `_prompt_resources` contract
  members public (drops 10 `reportPrivateUsage`).
- **adapters** — parse JSON-RPC frames once at ingress into the existing TypedDicts
  (`codex_app_server/_protocol.py`); tagged-union notifications.
- **ratchet** — add a `toolchain` checker with a per-package `cast`/`ignore`
  baseline (mirroring `code_length_baseline.txt`) so the count only falls.

**Exit:** per-package counts hit the ratchet; ty + pyright strict green at each phase.

### M17 — Prompt selection/render DRY

**Dims 5,6 · S–M · low risk.** `_enabled_predicate.py` and `_visibility.py` are the
same 2×2 normalizer twice (~120 LOC; `*_requires_positional_argument` byte-identical)
→ one `normalize_selector[R]`. `Section.render` and `MarkdownSection.render_override`
duplicate heading+body+tools assembly → extract `Section._assemble`; replace the
defensive `getattr(node.section,"accepts_overrides",…)` (`rendering.py:211`) with a
plain read. **Exit:** dedup; green.

### M18 — Error-handling & concurrency + coverage-mask audit

**Dims 6,7,9,10 · M · low–med.**

- **RedisMailbox** — `_lock` is decorative (guards only `close()`; `_closed` read
  unlocked elsewhere); the reaper wraps `_reap_expired` in
  `suppress(Exception)`, hiding connection failures indefinitely. Delete the lock
  or use it consistently; narrow + log the reaper suppression.
- **Over-broad catches** — `codex_app_server/adapter.py:268` `except BaseException`
  → `try/finally`/`ExitStack`.
- **Coverage masks** — add tests and drop the pragmas hiding real paths: the bundle
  token-aggregation loop (`_agent_loop_bundle.py:130-143`, also a swallowing
  `except … pass`), the hook feedback path (`_hook_tools.py:325`), the `_docs.py`
  `# pragma: no branch`, and the `_validators` type-mismatch branches.

**Exit:** every `# nosec`/`# pragma: no cover` is justified or gone; green.

______________________________________________________________________

## Status log

| Date | Action |
|------|--------|
| 2026-06-03 | REVIEW.md Phase 1–6 executed by a 7-agent team. ~56 findings → M9–M18. Wave 1 (M9–M11) promoted to milestone files; M12–M18 scoped here; M1–M8 amendments recorded. |
