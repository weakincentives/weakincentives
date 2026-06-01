# Remote Filesystem & Sandbox Execution Specification

## Purpose

Enable a **remote filesystem inside a sandbox** to act as the single target
environment for **every tool call**—both WINK-bridged tools and harness-native
tools (Read, Write, Edit, Bash, Glob, Grep)—while keeping tool handlers and
prompt definitions unchanged.

The refactor is deliberately **minimal**: the `Filesystem` protocol is already
the seam. The change is *who constructs the backend and when*. We move
filesystem materialization out of the prompt definition and into the
`ProviderAdapter`, so the same component that owns the harness's execution
environment also owns the `Filesystem` that tools resolve. Location (local vs.
remote) then becomes invisible to everything above the adapter.

**Status:** Design proposal. **Related:** `FILESYSTEM.md`, `WORKSPACE.md`,
`ADAPTERS.md`, `RESOURCE_REGISTRY.md`, `TOOLS.md`, `MODULE_BOUNDARIES.md`.

______________________________________________________________________

## Problem Statement

WINK runs two classes of tool today, against two different filesystems:

| Tool class | Runs in | Filesystem it touches |
|------------|---------|-----------------------|
| **WINK-bridged** (`context.filesystem.*`) | Orchestrator process | The `Filesystem` resource (a `HostFilesystem` over a temp dir) |
| **Harness-native** (Read/Write/Bash/…) | The harness (Claude Code CLI, Codex, ACP agent) | The harness's own `cwd` on local disk |

These coincide **only because** `WorkspaceSection` builds a `HostFilesystem`
rooted at a temp dir and the adapter happens to point the harness `cwd` at the
same temp dir. The coupling is incidental, and it **already breaks** in the
remote transport modes the codebase ships:

- `CodexAppServerClientConfig.remote_url` connects to an external Codex
  app-server; `cwd` is "a **remote path** … which may not exist on the
  orchestrator host" (`codex_app_server/config.py`).
- ACP external WebSocket mode requires "an explicit `cwd` that exists on the
  remote server" (`acp/config.py`).

In those modes the harness-native tools mutate a **remote** disk while
`context.filesystem` still reads/writes a **local** temp dir. The two diverge
silently. There is also **no exec channel** in the `Filesystem` protocol, so
"all tool calls" (which includes shell commands) cannot be expressed against a
remote target at all today.

**Goal:** make the remote sandbox the authoritative environment for *all* tool
calls, with the smallest possible change to the framework's stable surface.

______________________________________________________________________

## Core Insight: The Adapter Is the Filesystem Factory

> *Where bytes physically live is an execution concern, not a definition
> concern.*

This follows directly from the repository's **Definition vs. Harness** split
(`CLAUDE.md`): the definition owns Prompt/Tools/Policies/Feedback; the harness
owns sandboxing, isolation, retries, and crash recovery. The filesystem
**backend**—local temp dir vs. remote sandbox—is squarely a harness concern.

Today the definition (`WorkspaceSection`) constructs a concrete
`HostFilesystem` and the adapter independently chooses a `cwd`. We invert that:

1. The **definition declares intent** (a `WorkspaceSpec`: mounts, allowed
   roots, read-only, env, setup) — pure data, no concrete backend.
1. The **adapter materializes** that intent into a `Workspace` handle and is the
   **factory for the `Filesystem`** (and the exec channel) that everything else
   resolves.

Because one component now owns **both** the harness's execution environment
**and** the `Filesystem` tools resolve, the two are **structurally incapable of
diverging**. Tools call `context.filesystem.*` and never learn whether the
backend is local or remote.

______________________________________________________________________

## Architecture

```
        DEFINITION (portable, location-agnostic)
        ┌────────────────────────────────────────────┐
        │ WorkspaceSection ──► WorkspaceSpec (intent) │
        │   mounts, allowed_roots, read_only, env     │
        └───────────────────────┬────────────────────┘
                                 │ render
        HARNESS / ADAPTER (owns execution)
        ┌───────────────────────▼────────────────────┐
        │ ProviderAdapter.open_workspace(spec)        │
        │   local  → HostFilesystem (today's path)    │
        │   remote → RemoteFilesystem + Sandbox.exec  │
        └───────────────────────┬────────────────────┘
                                 │ yields one Workspace
            ┌────────────────────┼─────────────────────┐
            ▼                    ▼                     ▼
   bind {Filesystem,     point harness cwd /    teardown on close
   Sandbox} into the     exec at the SAME       (Closeable)
   resource overlay      Workspace
            │
            ▼
   context.filesystem.*  ◄── tools, unchanged, location-transparent
```

`Workspace` is a runtime handle the adapter returns:

| Member | Type | Description |
|--------|------|-------------|
| `filesystem` | `Filesystem` | Backend bound as the `Filesystem` resource |
| `exec(...)` | `ExecResult` | Run a command **in the sandbox** (see Sandbox) |
| `root` | `str` | Absolute path of the workspace inside its environment |
| `close()` | `None` | Release/teardown (`Closeable`) |

______________________________________________________________________

## Design

### 1. `WorkspaceSpec` — definition-owned intent

Extract the *data* currently embedded in `WorkspaceSection` into a frozen,
serializable spec. The section keeps a **default local `Filesystem`** so
adapter-less and in-memory test usage is unaffected; it additionally contributes
the spec for adapters that want to materialize their own backend.

```python
@FrozenDataclass()
class WorkspaceSpec:
    mounts: tuple[HostMount, ...] = ()
    allowed_host_roots: tuple[str, ...] = ()
    read_only: bool = False
    env: Mapping[str, str] | None = None
    setup: tuple[str, ...] = ()          # commands run once after provisioning
    fingerprint: str = ""                 # reuse detection (existing helper)
```

`WorkspaceSection.resources()` contributes both `WorkspaceSpec` and a default
`Filesystem` (the local `HostFilesystem`, exactly as today). When an adapter
provides a `Workspace`, its `Filesystem` **supersedes** the section default via
the overlay below — so existing behavior is the *default*, not a special case.

### 2. `ProviderAdapter` as factory

Add one method to the adapter protocol (`adapters/core.py`) with a default
implementation that reproduces today's behavior:

```python
class ProviderAdapter(ABC):
    def open_workspace(
        self,
        spec: WorkspaceSpec,
        *,
        session: SessionProtocol,
        run_context: RunContext | None = None,
        deadline: Deadline | None = None,
    ) -> Workspace:
        """Materialize `spec` into a Workspace. Default: local HostFilesystem."""
        ...
```

`evaluate()` gains a thin, well-contained sequence (no change to the planning
loop or tool dispatch):

1. Locate `WorkspaceSpec` from the rendered prompt (skip if none → today's path).
1. `workspace = self.open_workspace(spec, ...)`.
1. Overlay `{Filesystem: workspace.filesystem, Sandbox: workspace}` as the
   highest-precedence resources (see §8).
1. Point the harness `cwd`/exec at `workspace.root` **on the same environment**.
1. Run; on exit, `workspace.close()`.

Local-only adapters (or prompts without a workspace) inherit the default and are
untouched. Remote-capable adapters override `open_workspace` to provision/locate
a sandbox and return a `RemoteFilesystem` bound to it.

### 3. `RemoteFilesystem` backend

A new backend in `weakincentives/filesystem/` implementing **`Filesystem` and
`SnapshotableFilesystem`** over a narrow transport. It is a peer of
`HostFilesystem`/`InMemoryFilesystem`; the protocol does not change, so every
existing `context.filesystem.*` tool works against it unmodified.

Implementation requirements:

- **Server-side search.** `glob`, `grep`, and `list` MUST execute *inside the
  sandbox* and return only matches. Client-side `rglob("*")` + per-file reads
  over RPC (as `HostFilesystem` does locally) would be catastrophic over a
  network. This is the single most important performance rule.
- **Streaming via ranges.** `open_read` → ranged reads (`offset`,`size`);
  `open_write` → chunked append/PUT; `seek`/`position`/`size` map to range
  metadata. Honor the existing 64 KB default chunk and 32 MB convenience caps.
- **Error mapping.** Translate transport faults into the protocol's exact
  exception contract (`FileNotFoundError`, `IsADirectoryError`,
  `PermissionError`, `FileExistsError`, `ValueError`); map *connectivity* faults
  to `RuntimeError` ("backend unavailable") per `FILESYSTEM.md`.
- **Path confinement enforced server-side.** Root confinement and the
  depth-16 / segment-80 limits still apply, but the authoritative boundary is
  now the sandbox.

### 4. `FilesystemTransport` — the wire seam

A minimal, synchronous RPC surface the `RemoteFilesystem` depends on. Keeping it
narrow lets the transport-of-record (WebSocket, gRPC, SSH, `podman exec`, a
cloud dev-environment API) live in `contrib/` without leaking heavy deps into
the foundation layer (`MODULE_BOUNDARIES.md`).

| Method | Purpose |
|--------|---------|
| `stat`, `list`, `exists` | Metadata |
| `read_range(path, offset, size)` | Streaming/convenience reads |
| `write_range(path, offset, bytes, mode)` | Streaming/convenience writes |
| `glob`, `grep` | **Server-side** search |
| `delete`, `mkdir` | Mutations |
| `snapshot(tag)`, `restore(ref)` | Rollback (§7) |
| `exec(argv, cwd, env, stdin, timeout)` | Command execution (§5) |

Transport auth reuses the bearer-token pattern already used by the ACP MCP HTTP
bridge (`acp/_mcp_http.py`). Connections are pooled and held for the session
(SINGLETON scope, §8); blocking calls are offloaded with `asyncio.to_thread`,
matching `BridgedTool`'s existing async/sync bridge.

### 5. `Sandbox` — the exec channel ("all tool calls" includes shell)

The `Filesystem` protocol has no command execution. To make shell tool calls
target the sandbox we add a sibling protocol, bound as its own resource:

```python
@runtime_checkable
class Sandbox(Protocol):
    @property
    def filesystem(self) -> Filesystem: ...
    def exec(
        self, argv: Sequence[str], *, cwd: str = ".",
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None, timeout_s: float | None = None,
    ) -> ExecResult: ...        # exit_code, stdout/stderr (streamable), truncated
```

`ExecResult` mirrors the result-dataclass style of `filesystem/_types.py`. A
WINK-bridged `bash`/`run` tool resolves `context.resources.get(Sandbox)` and
routes through `exec`, so even shell commands hit the remote environment.

### 6. Making *every* tool call target the sandbox — harness placement

§2–§5 make **WINK-bridged** tools location-transparent. Harness-native tools
need one of two placements; the adapter chooses per its capabilities:

| Mode | How native tools reach the sandbox | Transactional rollback covers | Best for |
|------|-----------------------------------|-------------------------------|----------|
| **A. Co-located harness** | Adapter runs/points the harness *inside* the sandbox; its native `cwd`/exec **is** the sandbox. WINK `Filesystem` = `RemoteFilesystem` to the same sandbox. | WINK-bridged tools only | Harnesses with remote/external transport (Codex `remote_url`, ACP external WS) |
| **B. Orchestrator-mediated funnel** | Disable native FS/shell tools (`disallowed_tools` / tool policies); expose WINK tools (read/write/edit/grep/glob/bash) routed through `Workspace.filesystem` + `Sandbox.exec`. | **Every** tool call (all pass through `BridgedTool`/`tool_transaction`) | Any adapter; harnesses without remote execution (e.g. Claude Agent SDK) |

**Recommendation.** Mode **B** is the portable default: it satisfies "all tool
calls target the sandbox" with one mechanism across *every* adapter, and it
makes the transactional guarantee **total** (native mutations are otherwise
outside WINK's rollback). Mode **A** is the optimization where a harness already
executes remotely and native tooling is desirable; it requires only that the
adapter make the harness `cwd` and the `RemoteFilesystem` agree on **one**
sandbox identity (the `Workspace` handle they both read).

### 7. Snapshots & transactions over the transport

`tool_transaction` snapshots the session and all **SINGLETON `Snapshotable`**
resources before each WINK tool call and restores on failure
(`runtime/transactions.py`). Since `RemoteFilesystem` implements
`SnapshotableFilesystem`, **the transactional machinery is unchanged**.

Two notes drive correctness and cost:

- **Remote execution.** `snapshot`/`restore` run *in the sandbox* (remote git,
  or an overlay/btrfs/container-commit strategy — see Open Questions). The
  existing `FilesystemSnapshot` fields (`commit_ref`, `root_path`, `git_dir`)
  carry a remote reference.
- **Cost control.** A remote commit *per tool call* is expensive. Snapshots
  SHOULD be **lazy / copy-on-write**: capture a cheap marker on entry and only
  materialize before the first mutation. Expose an opt-out for read-heavy
  sections. (This optimization is valid for the local backend too.)

### 8. Resource overlay — the only core DI change

Resources today merge lowest→highest: template → sections (depth-first) →
`bind(resources=)` (`RESOURCE_REGISTRY.md`). Add **one** higher layer: a
runtime overlay the adapter sets before entering `with prompt.resources:`.

```python
prompt._set_runtime_resources(ResourceRegistry.build({
    Filesystem: workspace.filesystem,
    Sandbox: workspace,
}))
```

`_collected_resources()` merges this overlay last so the adapter-materialized
backend wins over the section default. `Workspace` is SINGLETON and `Closeable`
(teardown on context exit), optionally `PostConstruct` to establish the
connection. This is the **only** change to the DI core; everything downstream
(`ToolContext.filesystem`, `tool_transaction`, reducers) is untouched.

______________________________________________________________________

## Reference Topology: Codex in a Remote Container (WebSocket / JSON-RPC)

This is the concrete target the design must satisfy and the reference
implementation for **Mode A**. A `codex app-server` runs **inside a remote
container**; the WINK adapter drives it over a WebSocket JSON-RPC channel
(`CodexAppServerClientConfig.transport="websocket"`, `remote_url=...`, already
in the codebase). The container is the **external sandbox** for Codex *and* its
native tools (commands, file changes); it owns isolation and ingress/egress.

**Two channels, one container:**

| Channel | Carries | Terminates at |
|---------|---------|---------------|
| Codex JSON-RPC (existing) | thread/turn lifecycle, native tool calls, dynamic-tool callbacks | `codex app-server` in the container |
| `FilesystemTransport` (new, §4) | WINK `context.filesystem.*` + `Sandbox.exec` | A small WINK FS/exec endpoint in the **same** container, rooted at the same path |

Because both channels terminate in the same container, the filesystem Codex's
native tools mutate **is** the filesystem WINK-bridged tools see.
`open_workspace` returns a `Workspace` whose `root` is the container path passed
as Codex's `cwd` in `thread/start`, and whose `filesystem` is a
`RemoteFilesystem` over the container's `FilesystemTransport` rooted at that same
path. **One handle, both consumers** — this is the agreement point (§6, Mode A)
that makes divergence impossible.

**Why a second channel.** The Codex app-server protocol orchestrates threads and
tools; it does **not** expose general-purpose file read/write or `exec` for the
client. WINK-bridged tools therefore need their own authenticated endpoint into
the container — a lightweight FS/exec daemon, or `podman exec` / `kubectl exec`,
or the container-runtime API. The transport-agnostic `FilesystemTransport` seam
absorbs this, so **no change to the Codex protocol is required**.

**Execution boundary (critical).** WINK-bridged tools execute **in the
orchestrator process** (Codex invokes them via a JSON-RPC dynamic-tool callback;
the adapter runs the handler in a worker thread — confirmed in
`_shared/_bridge.py`). Therefore:

- File operations via `context.filesystem.*` reach the **container** through
  `RemoteFilesystem`. ✓
- Arbitrary handler logic, subprocesses, and **network egress** from a
  WINK-bridged handler run on the **orchestrator host**, *outside* the
  container's namespace — so they are **not** governed by its ingress/egress
  controls.

To place tool *side effects* (shell, processes, network) under the container's
isolation, route them through `Sandbox.exec` so the command lands **inside** the
container — i.e. prefer **Mode B funnel** semantics for anything beyond file I/O.
Codex's own native tools already run inside the container and are governed by it,
so the native-tool half of "all tool calls" is satisfied for free.

**Division of responsibility.** The container owns isolation and ingress/egress;
WINK owns the agent definition plus the FS/exec bridge. WINK imposes **no second
sandbox** here (contrast the Claude SDK adapter's bubblewrap/seatbelt
isolation) — it trusts the container as the perimeter. Both channels authenticate
with the bearer-token pattern Codex WS and the ACP MCP HTTP bridge already use.

**Snapshots.** `RemoteFilesystem.snapshot/restore` run *inside* the container
(remote git or a container-level snapshot). `tool_transaction` wraps WINK-bridged
calls; Codex's native edits are governed by Codex/the container, consistent with
Mode A (§6, §7).

______________________________________________________________________

## What Does NOT Change (Minimality Proof)

| Surface | Change |
|---------|--------|
| `Filesystem` / `SnapshotableFilesystem` protocol | **None** — remote is a new backend |
| `ToolContext`, `context.filesystem` | **None** |
| Tool handler signature, `ToolResult` | **None** |
| `tool_transaction`, `CompositeSnapshot`, reducers, session | **None** |
| Resource scopes / lifecycle protocols | **None** (add a runtime overlay merge) |
| Every existing `context.filesystem.*` tool | **None** — works against the sandbox unmodified |

New, additive code: `WorkspaceSpec`, `ProviderAdapter.open_workspace` (+default),
`Workspace`/`Sandbox`/`ExecResult`, `RemoteFilesystem`, `FilesystemTransport`,
one concrete transport, and a one-line runtime-overlay merge.

______________________________________________________________________

## Cross-Cutting Concerns

**Security.** The sandbox boundary becomes the real isolation perimeter, owning
ingress/egress controls; WINK does not impose a second sandbox over it. The
orchestrator authenticates to the transport (bearer token); path confinement and
limits are enforced server-side; `read_only` and `allowed_host_roots` carry over.
Note the execution-boundary caveat in *Reference Topology*: in-process
WINK-bridged handlers reach the sandbox **filesystem**, but their CPU/network
side effects run on the orchestrator host unless funneled through `Sandbox.exec`.

**Failure & throttling.** Transient connectivity faults are retryable
(`ThrottleError`-style backoff per `ADAPTERS.md`); semantic FS errors
(`FileNotFoundError`, `PermissionError`) are **not** retried. Persistent
unavailability surfaces as `RuntimeError` and rolls back the tool.

**Performance.** Synchronous protocol over a network ⇒ minimize round-trips:
server-side `glob`/`grep`, batched metadata, tuned chunk size, pooled
connections (SINGLETON), and `asyncio.to_thread` offload so adapter event loops
never block.

**Module boundaries.** `RemoteFilesystem`, `Sandbox`, `ExecResult`,
`FilesystemTransport` are foundation-layer protocols in `filesystem/`. Concrete
transports (WebSocket/gRPC/SSH/`podman exec`/cloud API) live in `contrib/` to
keep heavy dependencies out of core.

______________________________________________________________________

## Phasing

| Phase | Deliverable | Behavior change |
|-------|-------------|-----------------|
| 0 | `WorkspaceSpec` + `ProviderAdapter.open_workspace` default + runtime overlay | None (pure refactor; local path preserved) |
| 1 | `RemoteFilesystem` + `FilesystemTransport` + one transport (loopback/`podman exec`) | Opt-in remote backend for WINK-bridged tools |
| 2 | `Sandbox.exec` + Mode B funnel tools | "All tool calls" target the sandbox on every adapter |
| 3 | Wire Codex `remote_url` / ACP external to `RemoteFilesystem` (Mode A — *Reference Topology*) | Native tools + WINK tools share one remote container |

______________________________________________________________________

## Open Questions

1. **Default placement** — ship Mode B (funnel) as the default, with Mode A as
   an adapter opt-in? (Recommended.)
1. **Transport of record** — WebSocket (reuses Codex/ACP patterns), gRPC, SSH,
   `podman exec`, or a managed dev-environment API? For the Codex remote-container
   topology this is a **co-resident FS/exec daemon** in the same container, since
   the Codex protocol exposes no general file/exec access to the client.
1. **Remote snapshot strategy** — remote git (reuses `_git_ops` semantics),
   overlay/btrfs, or container commit?
1. **Section default `Filesystem`** — keep it for adapter-less/test ergonomics
   (recommended) or require a `Workspace` for all workspace-bearing prompts?

______________________________________________________________________

## Testing

- Reuse `FilesystemValidationSuite` (`tests/filesystem/`) against
  `RemoteFilesystem` via an **in-process loopback transport** + `FakeClock`, so
  the whole protocol contract is exercised without a network or 10 s-timeout
  risk.
- Contract tests asserting transport-fault → protocol-exception mapping.
- A Mode-B integration test proving that with native tools disabled, a shell
  tool call and a file write both land in the sandbox and roll back together on
  failure.
