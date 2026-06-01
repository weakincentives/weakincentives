# Sandbox & Remote Execution Refactoring Plan

## Goal

Re-model *"the environment where tool calls take effect"* as a small, cohesive
set of types, so that — eventually — a remote sandbox can be the target for every
tool call. This document is a **step-by-step refactoring plan**: each step is an
independently shippable, PR-sized change that keeps `make check` green. Steps 1–8
preserve current behavior exactly; remote execution is the payoff of Steps 9–11,
not a prerequisite.

**Status:** Refactoring plan. **Related:** `FILESYSTEM.md`, `WORKSPACE.md`,
`ADAPTERS.md`, `RESOURCE_REGISTRY.md`, `TOOLS.md`, `MODULE_BOUNDARIES.md`.

______________________________________________________________________

## Why

Two facts about today's code motivate the refactor:

1. **Construction lives in the wrong place.** `WorkspaceSection`
   (`prompt/workspace.py`) builds a concrete `HostFilesystem` and binds it as the
   `Filesystem` resource, while each adapter *independently* picks the harness
   `cwd` (e.g. `acp/adapter.py::_resolve_cwd`). They coincide only by
   convention.
1. **That coupling already breaks remotely.** With
   `CodexAppServerClientConfig.remote_url` (or ACP external `cwd`), native tools
   mutate a *remote* disk while `context.filesystem` still reads a *local* temp
   dir. There is also **no command channel** in the `Filesystem` protocol, so
   "all tool calls" — which includes shell — cannot target a remote environment.

The fix is to give this problem the right concepts and move materialization to
the component that owns execution (the adapter), one safe step at a time.

______________________________________________________________________

## Concepts

The vocabulary the steps introduce — one responsibility, one lifecycle owner each:

| Concept | Kind | Responsibility | Owner |
|---------|------|----------------|-------|
| `SandboxSpec` | immutable value | Declares the desired environment: mounts, network (ingress/egress) policy, env, setup, read-only | Definition (section) |
| `SandboxProvider` | factory | `open(spec) -> Sandbox`; may pool connections | Adapter |
| `Sandbox` | **aggregate root** | The isolated place where effects land; vends facets; `Closeable` | Resource registry (SINGLETON) |
| `Filesystem` | facet | File view of the sandbox (existing protocol, **unchanged**) | The `Sandbox` |
| `Shell` | facet | Command execution inside the sandbox | The `Sandbox` |
| `SandboxTransport` | infrastructure | Wire RPC for a remote sandbox | The (remote) `Sandbox` |

```
  SandboxSpec ──► SandboxProvider.open() ──► Sandbox ──┬── filesystem: Filesystem  (Snapshotable facet)
   (intent)          (factory)           (aggregate)  └── shell:      Shell       (exec facet)
                                              │ remote impls over
                                              ▼
                                        SandboxTransport  (ws/JSON-RPC, podman exec, …)
```

**Relationships.** A `Sandbox` **HAS-A** `Filesystem` and a `Shell` (composition,
correct direction — a filesystem never "holds" a sandbox). It is realized **from**
a `SandboxSpec` **by** a `SandboxProvider`. A remote sandbox's facets share the
one `SandboxTransport` it owns. The **aggregate root is the only `Closeable`** —
facets never self-close, and whoever opens a `Sandbox` closes it (`open … try … finally: close`).

______________________________________________________________________

## Plan Invariants

- Every step is PR-sized and leaves `make check` green.
- **Steps 1–8 are behavior-preserving** — the local path stays byte-for-byte
  identical. **Steps 9–11 are additive and opt-in.**
- New protocols mirror the existing `filesystem/_protocol.py` + `_types.py`
  style; new packages respect `MODULE_BOUNDARIES.md` (foundation → core).

______________________________________________________________________

## Part 1 — Behavior-preserving refactor (local only)

### Step 1 — Add the command facet

- **Goal:** give "run a command" a home.
- **Changes:** new `src/weakincentives/sandbox/` package with `Shell` and
  `CommandResult` protocols (mirroring filesystem style) and a `LocalShell` that
  runs subprocesses under a root dir.
- **Behavior:** none — no callers.
- **Verify:** unit tests for `LocalShell` (exit code, stdout/stderr capture,
  `cwd`, timeout); `make check`.

### Step 2 — Add the aggregate

- **Goal:** introduce the root that vends facets.
- **Changes:** `Sandbox` protocol (`id`, `root`, `filesystem`, `shell`, `close`)
  - `LocalSandbox` composing a `Filesystem` and an optional `Shell`.
- **Behavior:** none.
- **Verify:** construct a `LocalSandbox` over a temp `HostFilesystem`; assert the
  facets resolve and `close()` is idempotent.

### Step 3 — Add intent + factory

- **Goal:** name the request and the thing that fulfills it.
- **Changes:** `SandboxSpec` (reusing the existing `HostMount`); `SandboxProvider`
  protocol; `LocalSandboxProvider.open(spec)` that performs today's
  `workspace._create_workspace` (temp dir + mount copy) and returns a
  `LocalSandbox` over a `HostFilesystem`.
- **Behavior:** none — not yet wired into sections or adapters.
- **Verify:** the provider materializes mounts identically to today; exercise it
  with the existing workspace-mount test cases.

### Step 4 — Route `WorkspaceSection` through the provider

- **Goal:** one materialization path; the section gains intent.
- **Changes:** `WorkspaceSection` builds its filesystem via
  `LocalSandboxProvider().open(spec)` instead of `HostFilesystem(...)` directly,
  and `resources()` additionally contributes the `SandboxSpec`. It still
  contributes the `Filesystem` exactly as today.
- **Behavior:** identical — same temp dir, mounts, and bound `Filesystem`.
- **Verify:** existing workspace + filesystem suites pass unchanged.

### Step 5 — Add the runtime-resource overlay

- **Goal:** a seam for the adapter to inject materialized facets at highest
  precedence.
- **Changes:** `Prompt._set_runtime_resources(registry)`; `_collected_resources()`
  merges it last (above `bind(resources=)`). Unset by default.
- **Behavior:** none unless set.
- **Verify:** unit test that an overlay supersedes a section binding; `make check`.

### Step 6 — One adapter owns the sandbox

- **Goal:** prove the inversion end-to-end on a single adapter.
- **Changes:** add `ProviderAdapter.sandbox_provider` (default
  `LocalSandboxProvider`). In the chosen adapter's `evaluate()`: read the
  `SandboxSpec`; `sandbox = self.sandbox_provider.open(spec)`; set the overlay
  `{Sandbox, Filesystem: sandbox.filesystem, Shell: sandbox.shell}`; point the
  harness `cwd` at `sandbox.root`; `finally: sandbox.close()`. Add a
  `context.shell` shortcut to `ToolContext`, parallel to `context.filesystem`.
- **Behavior:** identical for that adapter — the default provider yields the same
  local filesystem; the overlay only changes *who binds it*.
- **Verify:** that adapter's integration tests; assert
  `context.filesystem.root == sandbox.root ==` harness `cwd`.

### Step 7 — Section contributes intent only

- **Goal:** remove the now-duplicate materialization; the adapter becomes the
  sole materializer on the adapter path.
- **Changes:** `WorkspaceSection.__init__` stops eagerly creating a temp dir;
  `resources()` contributes the `SandboxSpec` plus a *lazily*-created default
  `LocalSandbox` for adapter-less/test use.
- **Behavior:** adapter path unchanged; adapter-less path uses the lazy default.
- **Verify:** update workspace tests to resolve via the provider/default; `make check`.

### Step 8 — Replicate across the remaining adapters

- **Goal:** finish the inversion.
- **Changes:** apply Step 6 to `acp`, `codex_app_server`, and
  `claude_agent_sdk`, replacing each bespoke `cwd` resolution with
  `sandbox.root`.
- **Behavior:** identical, per adapter.
- **Verify:** each adapter's suite; `make check`.

> **Checkpoint.** A clean `Sandbox` aggregate exists; the adapter is the factory
> (via its provider); local behavior is unchanged; `Shell` is available but unused
> by default. Nothing remote yet.

______________________________________________________________________

## Part 2 — Enable remote (additive, opt-in)

### Step 9 — Remote facets over a transport

- **Goal:** a `Sandbox` whose facets live elsewhere.
- **Changes:** `SandboxTransport` (narrow RPC:
  `stat`/`list`/`read_range`/`write_range`/`glob`/`grep`/`delete`/`mkdir`/`snapshot`/`restore`/`exec`);
  `RemoteFilesystem` (`Filesystem` + `SnapshotableFilesystem`, with **server-side
  `glob`/`grep`**, ranged streaming, and transport-fault → protocol-exception
  mapping); `RemoteShell`; `RemoteSandbox` composing them.
- **Behavior:** none until a provider returns one.
- **Verify:** run the existing `FilesystemValidationSuite` against
  `RemoteFilesystem` over an **in-process loopback transport** + `FakeClock`
  (full contract, no network, no 10 s-timeout risk); `Shell` contract tests.

### Step 10 — A remote provider + one transport (Codex reference topology)

- **Goal:** the first real remote sandbox.
- **Changes:** `RemoteSandboxProvider` + one concrete transport (`podman exec`
  or WebSocket/JSON-RPC). For Codex `remote_url`, the provider connects a
  `SandboxTransport` to a **co-resident FS/exec endpoint in the same container**
  that runs `codex app-server`, and sets `sandbox.root` to Codex's
  `thread/start` `cwd`.
- **Reference topology:** two channels, one container — Codex JSON-RPC (threads,
  native tools) and `SandboxTransport` (WINK facets) — so the disk Codex's native
  tools mutate **is** the disk WINK tools see. The container owns isolation and
  ingress/egress; WINK adds no second sandbox. (The Codex protocol exposes no
  general file/exec access to the client, which is *why* the second channel
  exists.)
- **Behavior:** opt-in via adapter config; the default stays local.
- **Verify:** integration test against a throwaway container asserting a WINK file
  write and a Codex native edit observe one filesystem.

### Step 11 — (Optional) Funnel native tools through the sandbox

- **Goal:** make "all tool calls" total and transactional on any adapter.
- **Changes:** disable harness-native FS/shell tools; expose WINK
  read/write/edit/grep/glob/bash routed through the `Sandbox` facets.
- **Behavior:** opt-in; every tool call then passes through
  `BridgedTool`/`tool_transaction`.
- **Verify:** with native tools off, a `Shell` command and a file write both land
  in the sandbox and roll back together on failure.

______________________________________________________________________

## What Does NOT Change

| Surface | Change |
|---------|--------|
| `Filesystem` / `SnapshotableFilesystem` protocol | **None** — it becomes a sandbox facet; remote is a new backend |
| `ToolContext.filesystem`, tool handler signature, `ToolResult` | **None** (add a `context.shell` shortcut) |
| `tool_transaction`, `CompositeSnapshot`, reducers, session | **None** — the `Filesystem` facet stays `Snapshotable` |
| Resource scopes / lifecycle protocols | **None** (add one runtime-overlay merge) |
| Every existing `context.filesystem.*` tool | **None** — works against any sandbox unmodified |

______________________________________________________________________

## Execution-Boundary Note (read before Step 11)

WINK-bridged tools execute **in the orchestrator process** (e.g. Codex invokes
them via a JSON-RPC dynamic-tool callback; the adapter runs the handler in a
worker thread — see `_shared/_bridge.py`). So `context.filesystem`/`context.shell`
reach the sandbox, but a handler's *own* subprocesses and **network egress** run
on the orchestrator host, outside the sandbox's ingress/egress controls. To place
tool side effects under a container's controls, route them through `Shell`
(Step 11) rather than spawning processes or opening sockets inside a handler.

______________________________________________________________________

## Open Questions

1. **Root noun** — `Sandbox` (emphasizes isolation; the term in use) vs.
   `Environment` (admits non-isolated locals more naturally)?
1. **Command facet name** — `Shell` (cohesive pair with `Filesystem`) vs.
   `CommandRunner` (explicit `argv` semantics, no `/bin/sh` implication)?
1. **Default placement (Step 11)** — funnel as default, co-located as opt-in?
1. **Transport of record (Step 10)** — `podman exec`, WebSocket/JSON-RPC, gRPC,
   SSH, or a managed dev-environment API?
1. **Remote snapshot strategy (Step 9)** — remote git (reuses `_git_ops`),
   overlay/btrfs, or container commit?
1. **Package home** — a new `sandbox/` package (recommended) vs. extending
   `filesystem/`.
